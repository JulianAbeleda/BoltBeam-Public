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
from boltbeam.profile.decode_roles import GGML_BLOCK_LAYOUTS
from boltbeam.profile.gguf import GGML_TYPE_IDS

# Formats supported by this reference; the identities and layouts come from their owners.
GGML_TYPES = {q: GGML_TYPE_IDS[q] for q in ("Q4_K", "Q6_K", "Q8_0", "Q5_0", "Q4_0", "Q5_K", "Q3_K", "IQ4_NL", "IQ4_XS", "IQ3_S")}
BLOCK_BYTES = {q: GGML_BLOCK_LAYOUTS[t][1] for q, t in GGML_TYPES.items()}
BLOCK_ELEMS = {q: GGML_BLOCK_LAYOUTS[t][0] for q, t in GGML_TYPES.items()}
METADATA_BYTES = {"Q4_K": 16, "Q6_K": 18, "Q8_0": 2, "Q5_0": 6, "Q4_0": 2, "Q5_K": 48,  # Q4_K: d, dmin, 12 scale bytes. Q6_K: 16 int8 scales, d.
                  "Q3_K": 46, "IQ4_NL": 2, "IQ4_XS": 8, "IQ3_S": 46}
# Q8_0, Q4_0: d. Q5_0: d and the 4 bytes of fifth bits. Q5_K: d, dmin, 12 scale bytes, 32 bytes of fifth bits. Q3_K: 32 bytes
# of high bits, 12 scale bytes, d. IQ4_NL: d. IQ4_XS: d, 2 + 4 scale bytes. IQ3_S: d, 8 bytes of ninth bits, 32 of signs, 4
# of scales. The rest of a block is its codes (code_bytes).
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


@functools.lru_cache(maxsize=None)
def iq3s_grid() -> tuple[int, ...]:
  """IQ3_S's 512-entry codebook, read from ggml-common.h beside the engine source the kernel is built from
  (engine_kernels.mmvq_source_dir): the format's own definition, not a table BoltBeam copies."""
  from boltbeam.collectors.engine_kernels import NoEngineSource, mmvq_source_dir
  text = (pathlib.Path(mmvq_source_dir()["dir"]).parent / "ggml-common.h").read_text()
  m = re.search(r"GGML_TABLE_BEGIN\(uint32_t,\s*iq3s_grid,\s*512\)(.*?)GGML_TABLE_END", text, re.S)
  grid = tuple(int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(1))) if m else ()
  if len(grid) != 512:
    raise NoEngineSource("ggml-common.h has no 512-entry iq3s_grid")
  return grid


def dequant_iq3_s_block(blk:bytes) -> list[float]:
  d, qs, qh, signs, sc = _half(blk, 0), blk[2:66], blk[66:74], blk[74:106], blk[106:110]  # block_iq3_s
  grid, out = iq3s_grid(), []
  for k in range(8):  # 32 weights each: scale nibble k % 2 of byte k // 2, qh[k]'s bits as each pair's ninth index bit
    db = d * (1 + 2 * ((sc[k // 2] >> 4 * (k % 2)) & 0xF))
    for l in range(4):
      g1, g2 = grid[qs[8 * k + 2 * l] | ((qh[k] << (8 - 2 * l)) & 256)], grid[qs[8 * k + 2 * l + 1] | ((qh[k] << (7 - 2 * l)) & 256)]
      s = signs[4 * k + l]
      out += [db * ((g >> 8 * j) & 0xFF) * (-1 if s & (1 << (j + 4 * h)) else 1) for h, g in ((0, g1), (1, g2)) for j in range(4)]
  return out


def code_bytes(quant:str) -> int:
  """The bytes of a block that hold its codes, after its scales and extra bits."""
  return BLOCK_BYTES[quant] - METADATA_BYTES[quant]


DEQUANT: dict[str, Callable[[bytes], list[float]]] = {"Q4_K": dequant_q4_k_block, "Q6_K": dequant_q6_k_block, "Q8_0": dequant_q8_0_block,
                                                      "Q5_0": dequant_q5_0_block, "Q4_0": dequant_q4_0_block, "Q5_K": dequant_q5_k_block,
                                                      "Q3_K": dequant_q3_k_block, "IQ4_NL": dequant_iq4_nl_block,
                                                      "IQ4_XS": dequant_iq4_xs_block, "IQ3_S": dequant_iq3_s_block}


def reference_row(weights:bytes, quant:str, row:int, cols:int, x:list[float]) -> float:
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
  want = GGML_TYPES.get(quant)
  matches = [(name, off) for name, dims, typ, off in tensors if typ == want and tuple(dims) == (cols, rows, *([int(n_expert)] if n_expert else []))]
  if not matches:
    raise ValueError(f"no {quant} tensor of shape {rows}x{cols} in the model")
  named = [m for m in matches if m[0] == tensor_name]
  return (named or matches)[0]


def _tensor_for(probe:dict[str, Any], tensors:list[tuple[str, tuple[int, ...], int, int]]) -> tuple[str, int]:
  rows, cols = (int(v) for v in probe["shape"])
  return tensor_for(probe["quant"], rows, cols, tensors, probe.get("tensor_name"), probe.get("n_expert"))
