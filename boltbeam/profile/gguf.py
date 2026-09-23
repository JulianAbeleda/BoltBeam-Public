from __future__ import annotations

import collections
import math
import pathlib
import struct
from typing import Any

from boltbeam.profile.ir import ModelProfile, TensorRole
from boltbeam.profile.roles import classify_tensor_role, role_from_tensor_name
from boltbeam.profile.architecture import classify_architecture
from boltbeam.vocab import is_ssm_role

GGML_TYPE_NAMES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
  12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S",
  23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
  39: "MXFP4", 41: "Q1_0",
}


class GGUFReader:
  def __init__(self, path:pathlib.Path):
    self.path = path
    self.f = path.open("rb")

  def close(self) -> None:
    self.f.close()

  def __enter__(self) -> "GGUFReader": return self
  def __exit__(self, *exc) -> None: self.close()

  def read(self, n:int) -> bytes:
    data = self.f.read(n)
    if len(data) != n: raise EOFError(f"truncated GGUF while reading {self.path}")
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
      return [self.value(elem_typ) for _ in range(n)]
    if typ == 10: return self.u64()
    if typ == 11: return self.i64()
    if typ == 12: return self.f64()
    raise ValueError(f"unsupported GGUF metadata type {typ}")


def read_gguf_layout(path:str | pathlib.Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int, int]], int]:
  """Read GGUF metadata plus the absolute start of its aligned tensor-data region."""
  p = pathlib.Path(path).expanduser()
  with GGUFReader(p) as r:
    if r.read(4) != b"GGUF": raise ValueError(f"{p} is not a GGUF file")
    version = r.i32()
    if version not in (2, 3): raise ValueError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = r.i64(), r.i64()
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
      key, typ = r.string(), r.i32()
      kv[key] = r.value(typ)
    tensors = []
    for _ in range(n_tensors):
      name = r.string()
      dims = tuple(r.u64() for _ in range(r.u32()))
      tensors.append((name, dims, r.i32(), r.u64()))
    alignment = int(kv.get("general.alignment", 32))
    data_start = (r.f.tell() + alignment - 1) // alignment * alignment
  return kv, tensors, data_start


def read_gguf(path:str | pathlib.Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int, int]]]:
  kv, tensors, _ = read_gguf_layout(path)
  return kv, tensors


def profile_from_gguf(path:str | pathlib.Path, model_id:str | None=None) -> ModelProfile:
  p = pathlib.Path(path).expanduser()
  kv, tensors = read_gguf(p)
  arch = kv.get("general.architecture")
  hidden = kv.get(f"{arch}.embedding_length") if arch else None
  ffn = kv.get(f"{arch}.feed_forward_length") if arch else None
  layers = kv.get(f"{arch}.block_count") if arch else None
  head_count = kv.get(f"{arch}.attention.head_count") if arch else None
  head_count_kv = kv.get(f"{arch}.attention.head_count_kv") if arch else None
  head_dim = kv.get(f"{arch}.attention.key_length") if arch else None
  if head_dim is None and hidden and head_count:
    head_dim = hidden // head_count
  vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else None

  grouped: dict[tuple[str, int, int, int, int], list[str]] = collections.defaultdict(list)
  for name, dims, typ, _off in tensors:
    role = role_from_tensor_name(name)
    if not name.endswith(".weight") and not is_ssm_role(role): continue
    if role in ("embedding", "norm", "other"): continue
    if len(dims) == 2:
      rows, cols, n_expert = *tuple(reversed(dims)), 0
    elif len(dims) == 3 and not is_ssm_role(role):
      # MoE expert stack, GGUF ne = [in, out, n_expert]; per-expert GEMV shape is (out, in) (audit A5)
      in_, out_, n_expert = dims
      rows, cols = out_, in_
    elif is_ssm_role(role):
      # SSM/DeltaNet tensors are not necessarily 2D GEMV weights. Preserve them as profile roles so the
      # search can emit SSM-specific families instead of dropping them or pretending they are MoE expert stacks.
      rows, cols, n_expert = int(dims[-1]), int(math.prod(dims[:-1]) if len(dims) > 1 else 1), 0
    else:
      continue
    grouped[(role, rows, cols, typ, n_expert)].append(name)
    if role == "lm_head": vocab = vocab or rows

  roles = tuple(TensorRole(role=role, tensor_name=names[0], rows=rows, cols=cols,
                           ggml_type=typ, quant=GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"), count=len(names),
                           role_class=classify_tensor_role(names[0]).value, n_expert=n_expert)
                for (role, rows, cols, typ, n_expert), names in sorted(grouped.items()))
  architecture_class, arch_signals = classify_architecture(kv, tensors, arch)
  metadata = {
    "gguf_architecture": arch,
    "tensor_count": len(tensors),
    "quant_types": sorted({GGML_TYPE_NAMES.get(typ, f"GGML_{typ}") for _, _, typ, _ in tensors}),
    "architecture_signals": arch_signals,
    "attention": {k: v for k, v in {
      "head_count": head_count,
      "head_count_kv": head_count_kv,
      "head_dim": head_dim,
    }.items() if v is not None},
  }
  return ModelProfile(model_id=model_id or p.stem, source=str(p), architecture=arch, hidden_size=hidden,
                      ffn_size=ffn, vocab_size=vocab, layer_count=layers, roles=roles, metadata=metadata,
                      architecture_class=architecture_class)
