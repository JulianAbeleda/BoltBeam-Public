#!/usr/bin/env python3
"""Model-driven decode kernel role attribution helpers.

The older weight-path attribution tools classified kernels with Qwen3-8B constants
like 4096/12288/151936. This module builds the same facts from a GGUF tensor
table, then classifies kernel names by matching their dimensions against the
profile. It is intentionally tinygrad-free so it can run before loading a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import collections, pathlib, struct
from typing import Any

GGML_TYPE_NAMES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
  12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S",
  23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
  39: "MXFP4", 41: "Q1_0",
}

# Effective bits per weight from tinygrad.llm.gguf._GGML_QUANT block sizes.
GGML_BITS_PER_WEIGHT = {
  0: 32.0, 1: 16.0, 2: 4.5, 3: 5.0, 6: 5.5, 7: 6.0, 8: 8.5,
  12: 4.5, 13: 5.5, 14: 6.5625, 18: 3.0625, 21: 3.4375, 22: 2.5625,
  23: 4.25, 24: 8.0, 25: 16.0, 26: 32.0, 27: 64.0, 28: 64.0, 30: 16.0,
  39: 4.25, 41: 1.125,
}

# GGML's on-disk blocks.  Bits-per-weight is useful for rough display, but a
# roofline byte denominator must use these physical block sizes instead: a
# partial block cannot be silently rounded into an "exact" byte count.
GGML_BLOCK_LAYOUTS: dict[int, tuple[int, int]] = {
  0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
  8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
  18: (256, 98), 21: (256, 110), 22: (256, 82), 23: (256, 136),
  24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
  30: (1, 2), 39: (32, 17), 41: (32, 4),
}



@dataclass(frozen=True)
class WeightRole:
  role: str
  tensor_name: str
  rows: int
  cols: int
  ggml_type: int
  quant: str
  count: int


@dataclass(frozen=True)
class DecodeRoleProfile:
  model_id: str
  model_path: str
  arch: str | None
  hidden: int | None
  ffn: int | None
  vocab: int | None
  layers: int | None
  weights: tuple[WeightRole, ...]

  def to_json(self) -> dict[str, Any]:
    return {**asdict(self), "weights": [asdict(w) for w in self.weights]}


class _GGUFReader:
  def __init__(self, path:pathlib.Path):
    self.f = path.open("rb")

  def read(self, n:int) -> bytes:
    data = self.f.read(n)
    if len(data) != n: raise EOFError("truncated GGUF")
    return data

  def u32(self) -> int: return struct.unpack("<I", self.read(4))[0]
  def i32(self) -> int: return struct.unpack("<i", self.read(4))[0]
  def u64(self) -> int: return struct.unpack("<Q", self.read(8))[0]
  def i64(self) -> int: return struct.unpack("<q", self.read(8))[0]
  def f32(self) -> float: return struct.unpack("<f", self.read(4))[0]
  def f64(self) -> float: return struct.unpack("<d", self.read(8))[0]
  def string(self) -> str: return self.read(self.u64()).decode("utf-8")

  def value(self, typ:int) -> Any:
    if typ == 0: return self.read(1)
    if typ == 1: return struct.unpack("<b", self.read(1))[0]
    if typ == 2: return struct.unpack("<H", self.read(2))[0]
    if typ == 3: return struct.unpack("<h", self.read(2))[0]
    if typ == 4: return self.u32()
    if typ == 5: return self.i32()
    if typ == 6: return self.f32()
    if typ == 7: return struct.unpack("<?", self.read(1))[0]
    if typ == 8: return self.string()
    if typ == 9:
      elem_typ, n = self.i32(), self.u64()
      # Token arrays are large but still tiny next to model load; reading them
      # keeps parsing simple and gives exact vocab when needed.
      return [self.value(elem_typ) for _ in range(n)]
    if typ == 10: return self.u64()
    if typ == 11: return self.i64()
    if typ == 12: return self.f64()
    raise ValueError(f"unsupported GGUF metadata type {typ}")


def read_gguf_metadata(path:str | pathlib.Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int, int]]]:
  p = pathlib.Path(path).expanduser()
  r = _GGUFReader(p)
  try:
    if r.read(4) != b"GGUF": raise ValueError(f"{p} is not a GGUF file")
    version = r.i32()
    if version not in (2, 3): raise ValueError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = r.i64(), r.i64()
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
      key, typ = r.string(), r.i32()
      kv[key] = r.value(typ)
    infos = []
    for _ in range(n_tensors):
      name = r.string()
      dims = tuple(r.u64() for _ in range(r.u32()))
      infos.append((name, dims, r.i32(), r.u64()))
    return kv, infos
  finally:
    r.f.close()


def _role_from_tensor_name(name:str) -> str:
  if name == "output.weight": return "lm_head"
  if "ffn_gate" in name or "ffn_up" in name: return "ffn_gate_up"
  if "ffn_down" in name: return "ffn_down"
  if "attn_output" in name or "attn_q.weight" in name: return "attn_qo"
  if "attn_k.weight" in name or "attn_v.weight" in name: return "attn_kv"
  if "token_embd" in name: return "embedding"
  return "other"


def profile_from_gguf(path:str | pathlib.Path, model_id:str | None=None) -> DecodeRoleProfile:
  p = pathlib.Path(path).expanduser()
  kv, infos = read_gguf_metadata(p)
  arch = kv.get("general.architecture")
  hidden = kv.get(f"{arch}.embedding_length") if arch else None
  ffn = kv.get(f"{arch}.feed_forward_length") if arch else None
  layers = kv.get(f"{arch}.block_count") if arch else None
  vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else None

  grouped: dict[tuple[str, int, int, int], list[str]] = collections.defaultdict(list)
  for name, dims, typ, _off in infos:
    if not name.endswith(".weight") or len(dims) != 2: continue
    role = _role_from_tensor_name(name)
    if role == "embedding": continue
    rows, cols = tuple(reversed(dims))
    grouped[(role, rows, cols, typ)].append(name)
    if role == "lm_head": vocab = vocab or rows

  weights = tuple(WeightRole(role=role, tensor_name=names[0], rows=rows, cols=cols, ggml_type=typ,
                             quant=GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"), count=len(names))
                  for (role, rows, cols, typ), names in sorted(grouped.items()))
  return DecodeRoleProfile(model_id=model_id or p.stem, model_path=str(p), arch=arch, hidden=hidden, ffn=ffn,
                           vocab=vocab, layers=layers, weights=weights)




def _physical_bytes(numel:int, ggml_type:int, *, tensor_name:str) -> int:
  """Return exact GGUF payload bytes, rejecting layouts we cannot prove."""
  try: block_elements, block_bytes = GGML_BLOCK_LAYOUTS[ggml_type]
  except KeyError as exc:
    raise ValueError(f"{tensor_name}: GGML type {ggml_type} has no physical block layout") from exc
  if numel % block_elements:
    raise ValueError(f"{tensor_name}: {numel} elements are not divisible by GGML block size {block_elements}")
  return numel // block_elements * block_bytes


def decode_roofline_inventory_from_gguf(path:str | pathlib.Path, model_id:str | None = None) -> dict[str, Any]:
  """Static, lower-bound decode work derived directly from one GGUF tensor table.

  Rank-two ``*.weight`` tensors are counted once as matrix-vector products and
  rank-one weights (normally norms) are counted once as elementwise reads.
  ``token_embd.weight`` is special: decode reads one selected row, not the full
  vocabulary table. This deliberately does not claim activation, KV, allocator,
  cache, or reload traffic; those are provider-measured additions to the shared
  roofline report.
  """
  p = pathlib.Path(path).expanduser()
  _kv, infos = read_gguf_metadata(p)
  rows: list[dict[str, Any]] = []
  for name, dims, typ, _offset in infos:
    if not name.endswith(".weight") or len(dims) not in (1, 2):
      continue
    role = _role_from_tensor_name(name)
    if len(dims) == 1:
      packed_bytes = _physical_bytes(int(dims[0]), typ, tensor_name=name)
      flops, execution, shape, role = 0, "one_elementwise_weight_read", [int(dims[0])], "normalization_elementwise"
    else:
      cols, out_rows = (int(dims[0]), int(dims[1]))
      shape = [out_rows, cols]
      if role == "embedding":
        packed_bytes = _physical_bytes(cols, typ, tensor_name=name)
        flops, execution = 0, "one_selected_embedding_row"
      else:
        packed_bytes = _physical_bytes(cols * out_rows, typ, tensor_name=name)
        flops, execution = 2 * cols * out_rows, "one_dense_matrix_vector"
    rows.append({"tensor_name": name, "role": role, "shape": shape,
                 "ggml_type": typ, "quant": GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"),
                 "packed_weight_bytes": packed_bytes, "semantic_flops": flops,
                 "execution": execution})
  if not rows:
    raise ValueError(f"{p}: no rank-two *.weight tensors found for decode inventory")
  total_bytes = sum(r["packed_weight_bytes"] for r in rows)
  total_flops = sum(r["semantic_flops"] for r in rows)
  by_role: dict[str, dict[str, int]] = {}
  for row in rows:
    item = by_role.setdefault(row["role"], {"packed_weight_bytes": 0, "semantic_flops": 0, "tensor_count": 0})
    item["packed_weight_bytes"] += row["packed_weight_bytes"]
    item["semantic_flops"] += row["semantic_flops"]
    item["tensor_count"] += 1
  for item in by_role.values():
    item["weight_byte_pct"] = 100.0 * item["packed_weight_bytes"] / total_bytes
    item["flop_pct"] = 100.0 * item["semantic_flops"] / total_flops if total_flops else 0.0
  return {
    "schema": "boltbeam.decode_roofline_inventory.v1", "model_id": model_id or p.stem,
    "model_path": str(p), "packed_weight_bytes_per_token": total_bytes,
    "dense_matrix_flops_per_token": total_flops,
    "arithmetic_intensity_flop_per_byte": total_flops / total_bytes if total_bytes else None,
    "role_contributions": dict(sorted(by_role.items())), "tensors": rows,
    "provenance": {"source": "GGUF tensor table", "weight_execution": "every rank-two matrix weight and active rank-one elementwise weight once; token_embd is one row",
                   "flop_definition": "two operations per dense matrix MAC"},
    "caveats": ["packed weight bytes are a lower bound, not measured DRAM traffic",
                "activation, KV-cache, intermediates, reload, allocator, and cache effects are unmeasured",
                "semantic FLOPs cover dense matrix MACs only; attention, nonlinear, and elementwise work are excluded"],
  }
