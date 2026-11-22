"""The GGUF weight formats BoltBeam reads, and a pure-Python dequantize-and-dot reference for any of them: the
correctness check every GEMV is held to before it is timed, on any GPU (collectors/boltbeam_gemv.py, engine_kernels,
the searches). The identities and block layouts come from their owners (profile/gguf, profile/decode_roles); the
reference writes the block layouts out once more, in Python, as ggml-common.h defines them."""
from __future__ import annotations

import functools
import pathlib
import re
import struct
from typing import Any, Callable

from boltbeam.math.numerics import TOLERANCE  # noqa: F401  (re-read here: the reference's tolerance)
from boltbeam import formats

# Formats this reference dequantizes: the rows of data/formats.json that name one of its functions (dequant); ids,
# layouts and metadata bytes (scales, extra bits: the rest of a block is its codes) are the rows'.
GGML_TYPES = {q: formats.row(q)["ggml"] for q in formats.field("dequant")}
BLOCK_ELEMS, BLOCK_BYTES = ({q: formats.row(q)["block"][i] for q in GGML_TYPES} for i in (0, 1))
METADATA_BYTES = formats.field("metadata_bytes")
KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)  # ggml-common.h kvalues_iq4nl
CHECK_ROWS = 9


# --- the CPU reference: pure Python, the GGUF block layouts written out once more ---------------------------

def _half(raw:bytes, at:int) -> float:
  return struct.unpack_from("<e", raw, at)[0]


def _scale_min_k4(j:int, q:bytes) -> tuple[int, int]:
  if j < 4:
    return q[j] & 63, q[j + 4] & 63
  return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant_q4_k_block(blk:bytes) -> list[float]:
  d, dmin = _half(blk, 0), _half(blk, 2)
  scales, qs = blk[4:16], blk[16:144]
  out = [0.0] * 256
  for j in range(4):
    s0, m0 = _scale_min_k4(2 * j, scales)
    s1, m1 = _scale_min_k4(2 * j + 1, scales)
    for l in range(32):
      q = qs[32 * j + l]
      out[64 * j + l] = d * s0 * (q & 0xF) - dmin * m0
      out[64 * j + 32 + l] = d * s1 * (q >> 4) - dmin * m1
  return out


def dequant_q6_k_block(blk:bytes) -> list[float]:
  d = _half(blk, 208)
  out = [0.0] * 256
  for n in range(2):
    ql, qh = blk[64 * n:64 * n + 64], blk[128 + 32 * n:160 + 32 * n]
    sc = struct.unpack_from("<8b", blk, 192 + 8 * n)
    for l in range(32):
      i, h = l // 16, qh[l]
      out[128 * n + l] = d * sc[i] * (((ql[l] & 0xF) | ((h & 3) << 4)) - 32)
      out[128 * n + l + 32] = d * sc[i + 2] * (((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32)
      out[128 * n + l + 64] = d * sc[i + 4] * (((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32)
      out[128 * n + l + 96] = d * sc[i + 6] * (((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32)
  return out


def dequant_q8_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q8_0: half d, 32 int8 (ggml-common.h)
  return [d * q for q in struct.unpack_from("<32b", blk, 2)]


def dequant_q4_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q4_0: half d, 16 qs bytes; byte j holds element j (low nibble) and j+16 (high)
  qs = blk[2:18]
  return [d * ((qs[j] & 0xF) - 8) for j in range(16)] + [d * ((qs[j] >> 4) - 8) for j in range(16)]


def dequant_q5_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q5_0: half d, uint32 qh (bit j = fifth bit of element j), 16 qs bytes as Q4_0
  qh = struct.unpack_from("<I", blk, 2)[0]
  qs = blk[6:22]
  lo = [d * (((qs[j] & 0xF) | ((qh >> j) & 1) << 4) - 16) for j in range(16)]
  hi = [d * (((qs[j] >> 4) | ((qh >> (j + 16)) & 1) << 4) - 16) for j in range(16)]
  return lo + hi


def dequant_q5_k_block(blk:bytes) -> list[float]:
  d, dmin = _half(blk, 0), _half(blk, 2)  # block_q5_K: d, dmin, scales[12], qh[32], qs[128] (ggml-common.h)
  sc12, qh, qs = blk[4:16], blk[16:48], blk[48:176]
  out = [0.0] * 256
  for i in range(4):          # four 32-byte rows of qs, two groups of 32 each
    for hi in (0, 1):
      j = 2 * i + hi
      sc, m = _scale_min_k4(j, sc12)
      for l in range(32):
        q = ((qs[32 * i + l] >> (4 * hi)) & 0xF) | (((qh[l] >> j) & 1) << 4)
        out[64 * i + 32 * hi + l] = d * sc * q - dmin * m
  return out


def dequant_q3_k_block(blk:bytes) -> list[float]:
  hm, q, d = blk[0:32], blk[32:96], _half(blk, 108)  # block_q3_K: hmask[32], qs[64], scales[12], d
  a0, a1, tmp = struct.unpack_from("<3I", blk, 96)  # the 16 6-bit scales, unpacked as dequantize_row_q3_K does
  k1, k2 = 0x03030303, 0x0f0f0f0f
  scales = struct.unpack("<16b", struct.pack("<4I", (a0 & k2) | ((tmp & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4),
                                             ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)))
  out, s, m = [], 0, 1
  for n in range(2):  # two halves of 128; the high-bit mask m runs on across both
    for shift in (0, 2, 4, 6):
      for off in (0, 16):
        dl, s = d * (scales[s] - 32), s + 1
        out += [dl * (((q[32 * n + off + l] >> shift) & 3) - (0 if hm[off + l] & m else 4)) for l in range(16)]
      m <<= 1
  return out


def dequant_iq4_nl_block(blk:bytes) -> list[float]:
  d, qs = _half(blk, 0), blk[2:18]  # block_iq4_nl: d, 16 qs bytes, low nibbles first, through kvalues_iq4nl
  return [d * KVALUES_IQ4NL[b & 0xF] for b in qs] + [d * KVALUES_IQ4NL[b >> 4] for b in qs]


def dequant_iq4_xs_block(blk:bytes) -> list[float]:
  d, sh, sl = _half(blk, 0), struct.unpack_from("<H", blk, 2)[0], blk[4:8]  # block_iq4_xs: d, scales_h, scales_l[4], qs[128]
  out = []
  for ib in range(8):
    dl = d * ((((sl[ib // 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4)) - 32)
    qs = blk[8 + 16 * ib:24 + 16 * ib]
    out += [dl * KVALUES_IQ4NL[b & 0xF] for b in qs] + [dl * KVALUES_IQ4NL[b >> 4] for b in qs]
  return out


# --- the fields an integer-dot kernel reads (MMVQ: collectors/region_reference.mmvq), vectorized over blocks -------------
# blocks: uint8 [..., block bytes] -> {"d" (and "dmin"): the block's half scales as f32 [...], "q": its integer values
# [..., elements] in element order, "sc" (and "m"): each element's integer group scale [..., elements] (absent: none)}

def _np_half(blocks, at:int):
  import numpy as np
  return np.ascontiguousarray(blocks[..., at:at + 2]).view("<f2")[..., 0].astype(np.float32)


def fields_q8_0(blocks) -> dict[str, Any]:
  import numpy as np
  return {"d": _np_half(blocks, 0), "q": np.ascontiguousarray(blocks[..., 2:34]).view(np.int8).astype(np.int64)}


def fields_iq4_nl(blocks) -> dict[str, Any]:
  import numpy as np
  qs, kv = blocks[..., 2:18], np.array(KVALUES_IQ4NL, np.int64)
  return {"d": _np_half(blocks, 0), "q": np.concatenate([kv[qs & 0xF], kv[qs >> 4]], axis=-1)}


def fields_iq4_xs(blocks) -> dict[str, Any]:
  import numpy as np
  sh = blocks[..., 2].astype(np.int64) | (blocks[..., 3].astype(np.int64) << 8)
  sub = np.stack([((blocks[..., 4 + ib // 2].astype(np.int64) >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4) for ib in range(8)], axis=-1) - 32
  qs, kv = blocks[..., 8:136].reshape(*blocks.shape[:-1], 8, 16), np.array(KVALUES_IQ4NL, np.int64)
  return {"d": _np_half(blocks, 0), "q": np.concatenate([kv[qs & 0xF], kv[qs >> 4]], axis=-1).reshape(*blocks.shape[:-1], 256), "sc": np.repeat(sub, 32, axis=-1)}


def fields_q4_k(blocks) -> dict[str, Any]:
  import numpy as np
  sm = np.array([[_scale_min_k4(j, bytes(b[4:16])) for j in range(8)] for b in blocks.reshape(-1, blocks.shape[-1])], np.int64).reshape(*blocks.shape[:-1], 8, 2)
  qs = blocks[..., 16:144].reshape(*blocks.shape[:-1], 4, 32).astype(np.int64)   # pair j: low nibbles sub-block 2j, high 2j + 1
  q = np.stack([qs & 0xF, qs >> 4], axis=-2).reshape(*blocks.shape[:-1], 256)
  return {"d": _np_half(blocks, 0), "dmin": _np_half(blocks, 2), "q": q, "sc": np.repeat(sm[..., 0], 32, axis=-1), "m": np.repeat(sm[..., 1], 32, axis=-1)}


def fields_q6_k(blocks) -> dict[str, Any]:
  import numpy as np
  lead = blocks.shape[:-1]
  ql = blocks[..., 0:128].reshape(*lead, 2, 2, 32).astype(np.int64)   # half h, ql's second 32 bytes, l
  qh = blocks[..., 128:192].reshape(*lead, 2, 32).astype(np.int64)
  q = np.empty((*lead, 2, 4, 32), np.int64)                            # element 128 h + 32 quarter + l
  for quarter in range(4):
    q[..., quarter, :] = ((ql[..., quarter % 2, :] >> 4 * (quarter // 2)) & 0xF) | (((qh >> 2 * quarter) & 3) << 4)
  sc = np.ascontiguousarray(blocks[..., 192:208]).view(np.int8).astype(np.int64)
  return {"d": _np_half(blocks, 208), "q": q.reshape(*lead, 256) - 32, "sc": np.repeat(sc, 16, axis=-1)}


# The codebook formats' tables in ggml-common.h: name -> entry count. Read from the engine source, never copied here.
GGML_TABLES = {"iq3s_grid": "512", "iq2xxs_grid": "256", "iq2xs_grid": "512", "ksigns_iq2xs": "128", "iq1s_grid": "NGRID_IQ1S"}
GGML_TABLE_SIZES = {"NGRID_IQ1S": 2048}
IQ1_DELTA = 0.125  # ggml-common.h IQ1S_DELTA (IQ1_M's dequantize_row_iq1_m adds it): a constant, not a table


@functools.lru_cache(maxsize=None)
def ggml_table(name:str) -> tuple[int, ...]:
  """A codebook format's table (GGML_TABLES) as unsigned ints, read from ggml-common.h beside the engine source the
  kernel is built from (engine_source.mmvq_source_dir): the format's own definition, not a table BoltBeam copies."""
  from boltbeam.collectors.engine_source import NoEngineSource, mmvq_source_dir
  size = GGML_TABLES[name]
  text = (pathlib.Path(mmvq_source_dir()["dir"]).parent / "ggml-common.h").read_text()
  m = re.search(r"GGML_TABLE_BEGIN\(\w+,\s*" + name + r",\s*" + size + r"\)(.*?)GGML_TABLE_END", text, re.S)
  table = tuple(int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", m.group(1))) if m else ()
  if len(table) != GGML_TABLE_SIZES.get(size, int(size) if size.isdigit() else -1):
    raise NoEngineSource(f"ggml-common.h has no {size}-entry {name}")
  return table


def _grid_bytes(entry:int, signed:bool = False) -> list[int]:
  """A grid entry's 8 (uint64) or 4 (uint32) bytes, low byte first, as ggml reads them through a uint8_t/int8_t pointer."""
  return [((entry >> 8 * j) & 0xFF) - (256 if signed and (entry >> 8 * j) & 0x80 else 0) for j in range(8)]


def dequant_iq3_s_block(blk:bytes) -> list[float]:
  d, qs, qh, signs, sc = _half(blk, 0), blk[2:66], blk[66:74], blk[74:106], blk[106:110]  # block_iq3_s
  grid, out = ggml_table("iq3s_grid"), []
  for k in range(8):  # 32 weights each: scale nibble k % 2 of byte k // 2, qh[k]'s bits as each pair's ninth index bit
    db = d * (1 + 2 * ((sc[k // 2] >> 4 * (k % 2)) & 0xF))
    for l in range(4):
      g1, g2 = grid[qs[8 * k + 2 * l] | ((qh[k] << (8 - 2 * l)) & 256)], grid[qs[8 * k + 2 * l + 1] | ((qh[k] << (7 - 2 * l)) & 256)]
      s = signs[4 * k + l]
      out += [db * ((g >> 8 * j) & 0xFF) * (-1 if s & (1 << (j + 4 * h)) else 1) for h, g in ((0, g1), (1, g2)) for j in range(4)]
  return out


def _signed8(d:float, entry:int, signs:int) -> list[float]:
  """IQ2's 8 weights of one grid entry: d x each byte, negated where the sign byte's bit j is set (kmask_iq2xs = 1 << j)."""
  return [d * g * (-1.0 if signs >> j & 1 else 1.0) for j, g in enumerate(_grid_bytes(entry))]


def dequant_iq2_xxs_block(blk:bytes) -> list[float]:
  d, qs = _half(blk, 0), struct.unpack_from("<32H", blk, 2)  # block_iq2_xxs: d, qs[32] as 8 pairs of uint32 per 32 weights
  grid, ksigns, out = ggml_table("iq2xxs_grid"), ggml_table("ksigns_iq2xs"), []
  for ib in range(8):  # aux32[0]: four grid indices (bytes); aux32[1]: four 7-bit sign indices, the scale in its top 4 bits
    a0, a1 = qs[4 * ib] | qs[4 * ib + 1] << 16, qs[4 * ib + 2] | qs[4 * ib + 3] << 16
    db = d * (0.5 + (a1 >> 28)) * 0.25
    for l in range(4):
      out += _signed8(db, grid[a0 >> 8 * l & 0xFF], ksigns[a1 >> 7 * l & 127])
  return out


def dequant_iq2_xs_block(blk:bytes) -> list[float]:
  d, qs, sc = _half(blk, 0), struct.unpack_from("<32H", blk, 2), blk[66:74]  # block_iq2_xs: d, qs[32], scales[8]
  grid, ksigns, out = ggml_table("iq2xs_grid"), ggml_table("ksigns_iq2xs"), []
  for ib in range(8):  # each uint16: 9-bit grid index, 7-bit sign index; the scale nibble per 16 weights
    for l in range(4):
      q = qs[4 * ib + l]
      out += _signed8(d * (0.5 + (sc[ib] >> 4 * (l // 2) & 0xF)) * 0.25, grid[q & 511], ksigns[q >> 9])
  return out


def dequant_iq1_m_block(blk:bytes) -> list[float]:
  qs, qh, sc = blk[0:32], blk[32:48], struct.unpack_from("<4H", blk, 48)  # block_iq1_m: qs[32], qh[16], scales[8]
  d = struct.unpack("<e", struct.pack("<H", (sc[0] >> 12) | (sc[1] >> 8 & 0xF0) | (sc[2] >> 4 & 0xF00) | (sc[3] & 0xF000)))[0]
  grid, out = ggml_table("iq1s_grid"), []
  for ib in range(8):  # 32 weights: 4 groups of 8, an 11-bit grid index each (qs byte + 3 qh bits), a delta sign bit each
    h0, h1 = qh[2 * ib], qh[2 * ib + 1]
    idx = (qs[4 * ib] | (h0 << 8 & 0x700), qs[4 * ib + 1] | (h0 << 4 & 0x700), qs[4 * ib + 2] | (h1 << 8 & 0x700), qs[4 * ib + 3] | (h1 << 4 & 0x700))
    delta = [-IQ1_DELTA if h & bit else IQ1_DELTA for h in (h0, h1) for bit in (0x08, 0x80)]
    for l in range(4):
      dl = d * (2 * (sc[ib // 2] >> (6 * (ib % 2) + 3 * (l // 2)) & 7) + 1)
      out += [dl * (g + delta[l]) for g in _grid_bytes(grid[idx[l]], signed=True)]
  return out


def code_bytes(quant:str) -> int:
  """The bytes of a block that hold its codes, after its scales and extra bits."""
  return BLOCK_BYTES[quant] - METADATA_BYTES[quant]


DEQUANT: dict[str, Callable[[bytes], list[float]]] = {q: formats.dequant(q) for q in GGML_TYPES}


def reference_row(weights:bytes, quant:str, row:int, cols:int, x:list[float]) -> float:
  if quant not in DEQUANT: raise ValueError(formats.refusal("reference", quant))  # in the row's words (data/formats.json)
  size, elems = BLOCK_BYTES[quant], BLOCK_ELEMS[quant]
  nb = cols // elems
  base = row * nb * size
  acc = 0.0
  for b in range(nb):
    vals = DEQUANT[quant](weights[base + b * size: base + (b + 1) * size])
    xb = x[b * elems:(b + 1) * elems]
    acc += sum(v * xv for v, xv in zip(vals, xb))
  return acc


def check_rows(rows:int) -> list[int]:
  return sorted({min(rows - 1, (rows - 1) * i // (CHECK_ROWS - 1)) for i in range(CHECK_ROWS)})


def tensor_for(quant:str, rows:int, cols:int, tensors:list[tuple[str, tuple[int, ...], int, int]],
               tensor_name:str | None = None, n_expert:int | None = None) -> tuple[str, int]:
  """The model tensor of this quant and shape (the named one when it is among them; an MoE role's experts stacked, at
  its first expert): its name and data offset. Shared by every collector that binds real weight bytes (the probes
  here, collectors/engine_kernels.py)."""
  want = formats.require(quant, "the model's weight formats")["ggml"]
  matches = [(name, off) for name, dims, typ, off in tensors if typ == want and tuple(dims) == (cols, rows, *([int(n_expert)] if n_expert else []))]
  if not matches:
    raise ValueError(f"no {quant} tensor of shape {rows}x{cols} in the model")
  named = [m for m in matches if m[0] == tensor_name]
  return (named or matches)[0]
