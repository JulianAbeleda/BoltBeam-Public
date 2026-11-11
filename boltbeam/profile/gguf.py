from __future__ import annotations

import collections
import math
import pathlib
import re
import struct
from typing import Any

from boltbeam.profile.ir import ModelProfile, TensorRole, model_name
from boltbeam.profile.roles import classify_tensor_role, role_from_tensor_name
from boltbeam.profile.architecture import classify_architecture, recurrent_kind
from boltbeam.vocab import is_ssm_role

GGML_TYPE_NAMES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
  18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
  23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
  34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 41: "Q1_0", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0",
}
# ids only an engine fork defines: where their names and block layouts (decode_roles.GGML_BLOCK_LAYOUTS) were read
_PRISM = {"engine_family": "llama.cpp", "fork": "prismml",  # data/engine_families.json names its repo
          "commit": "6447906bf96c1a023980d3453fb59daa6af3bb23", "files": "ggml/include/ggml.h, ggml/src/ggml-common.h"}
GGML_FORK_TYPES = {142: _PRISM, 143: _PRISM}  # a profile holding one names its source (metadata quant_sources)


GGML_TYPE_IDS = {name: value for value, name in GGML_TYPE_NAMES.items()}

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


def read_gguf_split(path:str | pathlib.Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int, int]], int]:
  """A model llama.cpp's gguf-split wrote as N files (<stem>-00001-of-0000N.gguf ...): the first shard's metadata,
  every shard's tensors and the data bytes of all of them. split.count says how many; the first shard alone holds the
  metadata and few or no tensors (Qwen3.8-Flash-Next UD-Q4_K_XL: 4 shards, 0 roles from the first, 2026-11-08).
  An unsplit file is itself. Offsets stay per shard: read weight bytes with read_tensor."""
  p = pathlib.Path(path).expanduser()
  kv, tensors, start = read_gguf_layout(p)
  # a shard with no tensors has no data region: its file can end before the aligned header end (-6 B on Flash-Next's first)
  region = lambda f, ts, at: f.stat().st_size - at if ts else 0
  data, n = region(p, tensors, start), int(kv.get("split.count") or 1)
  if n <= 1:
    return kv, tensors, data
  tensors = list(tensors)
  for shard in _shards(p, n)[1:]:
    if not shard.is_file():
      raise ValueError(f"split GGUF: {shard.name} is missing beside {p.name}")
    _, more, shard_start = read_gguf_layout(shard)
    tensors += more
    data += region(shard, more, shard_start)
  if kv.get("split.tensors.count") not in (None, len(tensors)):
    raise ValueError(f"split GGUF: its {n} files hold {len(tensors)} tensors, the first declares {kv['split.tensors.count']}")
  return kv, tensors, data


def _shards(p:pathlib.Path, n:int) -> list[pathlib.Path]:
  """The files of a model split in n (n=1: the file itself), from its first."""
  if n <= 1:
    return [p]
  m = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.gguf", p.name)
  if not m or int(m.group(2)) != 1 or int(m.group(3)) != n:
    raise ValueError(f"{p.name}: a model split in {n} files is read from its first, <stem>-00001-of-{n:05d}.gguf")
  return [p.with_name(f"{m.group(1)}-{i:0{len(m.group(2))}d}-of-{m.group(3)}.gguf") for i in range(1, n + 1)]


_LOCATIONS: dict[tuple[str, int, int], dict[str, tuple[pathlib.Path, int]]] = {}


def tensor_locations(path:str | pathlib.Path) -> dict[str, tuple[pathlib.Path, int]]:
  """Each tensor's file and absolute byte position, across every shard of a split model (read once per file state)."""
  p = pathlib.Path(path).expanduser().resolve()
  st = p.stat()
  key = (str(p), st.st_size, st.st_mtime_ns)
  if key not in _LOCATIONS:
    kv, _, _ = read_gguf_layout(p)
    found: dict[str, tuple[pathlib.Path, int]] = {}
    for shard in _shards(p, int(kv.get("split.count") or 1)):
      _, tensors, start = read_gguf_layout(shard)
      found.update({name: (shard, start + off) for name, _, _, off in tensors})
    _LOCATIONS[key] = found
  return _LOCATIONS[key]


def read_tensor(path:str | pathlib.Path, name:str, size:int) -> bytes:
  """A tensor's first `size` bytes from whichever file of the model holds it."""
  shard, at = tensor_locations(path)[name]
  with open(shard, "rb") as f:
    f.seek(at)
    got = f.read(size)
  if len(got) != size:
    raise ValueError(f"{shard.name}: {name} has {len(got)} of its {size} bytes")
  return got


RECURRENT_KEYS = ("conv_kernel", "state_size", "group_count", "inner_size")  # GGUF ssm.* keys that size the state




def attention_layers(tensors, layers:int | None) -> int | None:
  """The layers that keep a KV cache: those holding a key projection (`blk.N.attn_k`). A hybrid model attends in some
  layers only (the rest are recurrent); None when every layer does or no layer names one (block_count stands)."""
  blocks = {name.split(".")[1] for name, *_ in tensors if name.startswith("blk.") and ".attn_k." in name}
  return len(blocks) if blocks and layers and len(blocks) < layers else None


def declared_context(kv:dict[str, Any], arch:str | None) -> dict[str, Any] | None:
  """The context the model declares: `<arch>.context_length`, extended by a rope scaling it declares (factor x the
  original context). {"length", "source", "scaling"}; None when the file declares none."""
  if not arch or not kv.get(f"{arch}.context_length"):
    return None
  length, source = int(kv[f"{arch}.context_length"]), f"{arch}.context_length"
  scaling = {k.rsplit(".", 1)[1]: v for k, v in kv.items() if k.startswith(f"{arch}.rope.scaling.")}
  factor, original = scaling.get("factor"), scaling.get("original_context_length")
  if scaling.get("type") not in (None, "none") and factor and original and float(factor) * int(original) > length:
    length = int(float(factor) * int(original))
    source = f"{arch}.rope.scaling.factor x {arch}.rope.scaling.original_context_length"
  return {"length": length, "source": source, "scaling": scaling or None}


def profile_from_gguf(path:str | pathlib.Path, model_id:str | None=None) -> ModelProfile:
  from boltbeam.profile import weight_ledger
  p = pathlib.Path(path).expanduser()
  kv, all_tensors, data_bytes = read_gguf_split(p)
  arch = kv.get("general.architecture")
  # a multi-token-prediction head's blocks are no layer the engine runs: out of the layers and roles, excluded in the ledger
  drafts = weight_ledger.draft_blocks(kv, arch)
  tensors = [t for t in all_tensors if not t[0].startswith(drafts)]
  hidden = kv.get(f"{arch}.embedding_length") if arch else None
  ffn = kv.get(f"{arch}.feed_forward_length") if arch else None
  layers = kv.get(f"{arch}.block_count") - len(drafts) if arch and kv.get(f"{arch}.block_count") else None
  head_count = kv.get(f"{arch}.attention.head_count") if arch else None
  head_count_kv = kv.get(f"{arch}.attention.head_count_kv") if arch else None
  head_dim = kv.get(f"{arch}.attention.key_length") if arch else None
  if head_dim is None and hidden and head_count:
    head_dim = hidden // head_count
  value_dim = kv.get(f"{arch}.attention.value_length") if arch else None  # a latent attention's V head is narrower than its K
  vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else None

  scans = {n.split(".")[1] for n, *_ in tensors if n.startswith("blk.") and ".ssm_conv1d." in n}
  recurrent = len(scans)
  # a hybrid's row: its tensors per layer kind and fine class (a mixed quant per layer runs each format in some layers)
  kinds_of = lambda names: {k: c for k in ("recurrent", "attention") if (c := dict(collections.Counter(  # noqa: E731
    classify_tensor_role(n, shapes[n]).value for n in names if n.startswith("blk.") and (n.split(".")[1] in scans) == (k == "recurrent"))))} \
    or None if 0 < recurrent < (layers or 0) else None
  grouped: dict[tuple[str, int, int, int, int], list[str]] = collections.defaultdict(list)
  shapes = {name: tuple(dims) for name, dims, _typ, _off in tensors}
  for name, dims, typ, _off in tensors:
    role = role_from_tensor_name(name, tuple(dims))
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

  # a role held in several formats runs each in some layers only: each row's tensors per block, so a reader can tell
  # the layers that hold the same tensors apart from the rest (pairing.signatures)
  mixed = len({(r, n, c, e) for r, n, c, _, e in grouped}) < len(grouped)
  per_layer = lambda names: {b: dict(collections.Counter(classify_tensor_role(n, shapes[n]).value for n in names  # noqa: E731
                                                     if n.startswith(f"blk.{b}.")))
                             for b in sorted({int(n.split(".")[1]) for n in names if n.startswith("blk.")})} or None
  roles = tuple(TensorRole(role=role, tensor_name=names[0], rows=rows, cols=cols,
                           ggml_type=typ, quant=GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"), count=len(names),
                           role_class=classify_tensor_role(names[0], shapes[names[0]]).value, n_expert=n_expert,
                           experts_read=weight_ledger.experts_read(kv, arch, n_expert) if n_expert else 0,
                           last_layer=sum(n.startswith(f"blk.{layers - 1}.") for n in names) if layers else None,
                           layer_kinds=kinds_of(names), per_layer=per_layer(names) if mixed else None)
                for (role, rows, cols, typ, n_expert), names in sorted(grouped.items()))
  # every tensor counted in the limit or excluded with a reason; refused here when the two do not add up
  ledger = weight_ledger.build(kv, all_tensors, arch=arch, data_bytes=data_bytes,
                               alignment=int(kv.get("general.alignment", 32)))
  weight_ledger.check(ledger, roles)
  architecture_class, arch_signals = classify_architecture(kv, tensors, arch)
  metadata = {
    "gguf_architecture": arch,
    "tensor_count": len(all_tensors),
    "quant_types": sorted({GGML_TYPE_NAMES.get(typ, f"GGML_{typ}") for _, _, typ, _ in all_tensors}),
    **({"quant_sources": {GGML_TYPE_NAMES[t]: GGML_FORK_TYPES[t] for t in sorted(forks)}}
       if (forks := {typ for *_, typ, _ in all_tensors} & GGML_FORK_TYPES.keys()) else {}),
    "architecture_signals": arch_signals,
    "weights": ledger,
    "attention": {k: v for k, v in {
      "head_count": head_count,
      "head_count_kv": head_count_kv,
      "head_dim": head_dim,
      "value_dim": value_dim if value_dim != head_dim else None,
      "layer_count": attention_layers(tensors, layers),
    }.items() if v is not None},
    "context": declared_context(kv, arch),
    # a recurrent block's fixed state (Mamba2, Gated DeltaNet): its shape and the layers that hold it
    **({"recurrent": {"layer_count": recurrent, **{k: kv[f"{arch}.ssm.{k}"] for k in RECURRENT_KEYS if f"{arch}.ssm.{k}" in kv},
                      **({"kind": kind} if (kind := recurrent_kind(n for n, *_ in tensors)) else {})}}
       if recurrent else {}),
  }
  return ModelProfile(model_id=model_id or model_name(p), source=str(p), architecture=arch, hidden_size=hidden,
                      ffn_size=ffn, vocab_size=vocab, layer_count=layers, roles=roles, metadata=metadata,
                      architecture_class=architecture_class)
