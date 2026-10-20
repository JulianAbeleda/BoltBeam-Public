from __future__ import annotations

import json
import math
import pathlib
import struct
import fnmatch
import re
from dataclasses import dataclass
from typing import Any

from boltbeam.profile.architecture import classify_architecture
from boltbeam.profile.gguf import GGML_TYPE_NAMES, profile_from_gguf
from boltbeam.profile.ir import ModelProfile, TensorRole
from boltbeam.profile.roles import classify_tensor_role, role_from_tensor_name
from boltbeam.profile.weight_ledger import experts_read
from boltbeam.vocab import RoleClass, is_ssm_role

_DTYPE_TO_QUANT = {
  "F64": "F64", "F32": "F32", "F16": "F16", "BF16": "BF16",
  "I64": "I64", "I32": "I32", "I16": "I16", "I8": "I8",
  "U8": "U8", "BOOL": "BOOL",
}

_ONNX_DTYPE = {
  1: "F32", 2: "U8", 3: "I8", 6: "I32", 7: "I64", 10: "F16", 16: "BF16",
}


@dataclass(frozen=True)
class TensorSpec:
  name: str
  shape: tuple[int, ...]
  quant: str
  ggml_type: int = -1
  weight_bytes: int | None = None


def detect_model_format(path:str | pathlib.Path) -> str:
  p = pathlib.Path(path).expanduser()
  if p.is_file():
    suf = p.suffix.lower()
    if suf == ".gguf": return "gguf"
    if suf == ".safetensors": return "safetensors"
    if suf == ".onnx": return "onnx"
    if suf == ".npz": return "mlx"
  if p.is_dir():
    cfg = _read_json(p / "config.json")
    qcfg = _quant_config(p, cfg)
    qmethod = str(qcfg.get("quant_method", "")).lower()
    if qmethod == "awq": return "awq"
    if qmethod == "gptq": return "gptq"
    if _looks_mlx(p, cfg): return "mlx"
    if list(p.glob("*.safetensors")): return "safetensors"
    if list(p.glob("*.onnx")): return "onnx"
  raise ValueError(f"unsupported model artifact format for {p}")


def profile_from_model(path:str | pathlib.Path, model_id:str | None = None) -> ModelProfile:
  p = pathlib.Path(path).expanduser()
  fmt = detect_model_format(p)
  if fmt == "gguf":
    prof = profile_from_gguf(p, model_id=model_id)
    prof.metadata["format_family"] = "gguf"
    return prof
  if fmt in {"safetensors", "awq", "gptq"}:
    return _profile_from_hf_tensors(p, fmt, model_id)
  if fmt == "onnx":
    return _profile_from_onnx(p, model_id)
  if fmt == "mlx":
    return _profile_from_mlx(p, model_id)
  raise ValueError(f"unsupported model artifact format {fmt!r}")


def _read_json(path:pathlib.Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  return json.loads(path.read_text())


def _quant_config(path:pathlib.Path, cfg:dict[str, Any]) -> dict[str, Any]:
  if isinstance(cfg.get("quantization_config"), dict):
    return dict(cfg["quantization_config"])
  for name in ("quantize_config.json", "quant_config.json"):
    q = _read_json(path / name)
    if q:
      return q
  # ModelOpt also ships its declaration separately from transformers' config.json.
  modelopt = _read_json(path / "hf_quant_config.json").get("quantization") or {}
  if modelopt.get("quant_algo") == "NVFP4":
    return {"quant_method": "modelopt", "packed_weights_only": True, "ignore": modelopt.get("exclude_modules", []),
            "config_groups": {"nvfp4": {"targets": ["Linear"], "weights": {
              "type": "float", "num_bits": 4, "group_size": modelopt.get("group_size")}}}}
  return {}


def _looks_mlx(path:pathlib.Path, cfg:dict[str, Any]) -> bool:
  return bool(cfg.get("mlx") or cfg.get("mlx_lm") or (path / "weights.npz").exists()
              or (path / "model.npz").exists() or "mlx" in path.name.lower())


def _safetensors_files(path:pathlib.Path) -> list[pathlib.Path]:
  if path.is_file():
    return [path]
  return sorted(path.glob("*.safetensors"))


def _read_safetensors_header(path:pathlib.Path) -> dict[str, Any]:
  with path.open("rb") as f:
    raw = f.read(8)
    if len(raw) != 8:
      raise ValueError(f"{path} is too short to be a safetensors file")
    n = struct.unpack("<Q", raw)[0]
    return json.loads(f.read(n).decode("utf-8"))


# a group-quantized checkpoint stores each weight as a packed-integer tensor that stands for the weight, plus parts
# (scales, zero points) whose bytes are in the quant label's bits. Two exports are read:
#   compressed-tensors pack-quantized: <name>.weight_packed, int32 [out, in x bits / 32]
#   AMD Quark (real_quantized):        <name>.weight,        int32 [in, out x bits / 32]  (transposed)
_PACKED = ".weight_packed"
_PACKED_PARTS = (".weight_scale", ".weight_zero_point", ".weight_shape", ".weight_g_idx", ".weight_global_scale")
_FLOAT_QUANT_PARTS = (".weight_scale_2", ".input_scale", ".input_scale_2", ".output_scale")


@dataclass(frozen=True)
class PackedFormat:
  """How one checkpoint stores a group-quantized weight: its label (INT4_G128_ZP: 4-bit integers in groups of 128,
  each group with a 16-bit scale and a 4-bit zero point), the bits per integer, the packed tensor's suffix, and
  whether the packed tensor is [in, out] rather than [out, in]."""
  label: str
  bits: int
  suffix: str
  transposed: bool
# modules a text prompt's prefill does not run: a draft head for speculative decoding, a vision encoder
_NOT_IN_TEXT_PREFILL = ("mtp", "visual", "vision_tower", "vision_model")


def _group_quant(qcfg:dict[str, Any], names:set[str] | frozenset[str] = frozenset()) -> PackedFormat | None:
  """A group-quantized checkpoint's weight format, read from its quantization config, or None for any other.
  compressed-tensors says in its config whether zero points exist (asymmetric). A Quark export can store zero points
  for a symmetric scheme, and those stored bytes are read every token, so for Quark the zero point is in the label
  when the file holds weight_zero_point tensors (`names`), whatever the config's flag says."""
  groups = [g.get("weights") or {} for g in (qcfg.get("config_groups") or {}).values() if isinstance(g, dict)]
  w = groups[0] if len(groups) == 1 else {}
  if qcfg.get("format") == "pack-quantized" and w.get("type") == "int" and w.get("num_bits") and w.get("group_size"):
    return PackedFormat(f"INT{w['num_bits']}_G{w['group_size']}" + ("" if w.get("symmetric") else "_ZP"),
                        int(w["num_bits"]), _PACKED, transposed=False)
  q = (qcfg.get("global_quant_config") or {}).get("weight") or {}
  export = qcfg.get("export") or {}
  bits = {"int4": 4, "uint4": 4, "int8": 8, "uint8": 8}.get(str(q.get("dtype")))
  if (str(qcfg.get("quant_method")).lower() == "quark" and export.get("weight_format") == "real_quantized"
      and q.get("qscheme") == "per_group" and bits and q.get("group_size")):
    zp = any(n.endswith(".weight_zero_point") for n in names)
    return PackedFormat(f"INT{bits}_G{q['group_size']}" + ("_ZP" if zp else ""), bits, ".weight", transposed=True)
  return None


def _safetensors_specs(path:pathlib.Path, default_quant:str | None = None,
                       qcfg:dict[str, Any] | None = None) -> list[TensorSpec]:
  specs: list[TensorSpec] = []
  header = {name: meta for f in _safetensors_files(path) for name, meta in _read_safetensors_header(f).items()}
  group = _group_quant(qcfg or {}, set(header))
  for name, meta in header.items():
    if name == "__metadata__" or not isinstance(meta, dict):
      continue
    if set(name.split(".")) & set(_NOT_IN_TEXT_PREFILL) or name.endswith(_PACKED_PARTS + _FLOAT_QUANT_PARTS):
      continue
    dtype = str(meta.get("dtype", "F16")).upper()
    shape = tuple(int(x) for x in meta.get("shape", ()))
    stored = None
    scheme = _float_quant_scheme(name, qcfg or {})
    if (qcfg or {}).get("packed_weights_only") and dtype != "U8":
      scheme = None  # legacy ModelOpt leaves routers and excluded matrices in their header's float dtype
    if scheme and name.endswith((".weight", _PACKED)) and len(shape) == 2:
      module = name[:-len(_PACKED)] if name.endswith(_PACKED) else name[:-len(".weight")]
      bits = int(scheme.get("num_bits") or 0)
      if bits == 4:
        if dtype != "U8" or scheme.get("group_size") != 16:
          raise ValueError(f"{name}: unsupported NVFP4 packing {dtype}, group {scheme.get('group_size')}")
        shape = (shape[0], shape[1] * 2)
        quant = "NVFP4_G16"
        scale = header.get(module + ".weight_scale") or {}
        if scale and (scale.get("dtype") not in ("F8_E4M3", "F8_E4M3FN") or
                      tuple(scale.get("shape", [])) != (shape[0], math.ceil(shape[1] / 16))):
          raise ValueError(f"{name}: NVFP4 scale packing disagrees with the declared matrix")
      elif bits == 8 and dtype in ("F8_E4M3", "F8_E4M3FN"):
        quant = "FP8_E4M3"
      else:
        raise ValueError(f"{name}: declared float quantization {bits} bits disagrees with {dtype}")
      parts = [meta, *[header[k] for k in (module + ".weight_scale", module + ".weight_scale_2", module + ".weight_global_scale") if k in header]]
      if module + ".weight_scale" not in header:
        raise ValueError(f"{name}: quantized weight has no scale tensor")
      stored = sum(int(p["data_offsets"][1]) - int(p["data_offsets"][0]) for p in parts)
      specs.append(TensorSpec(name=module + ".weight", shape=shape, quant=quant, weight_bytes=stored))
      continue
    if group and name.endswith(group.suffix) and len(shape) == 2 and (group.suffix == _PACKED or dtype in ("I32", "U32")):
      # int32 words of `bits`-bit integers along the second axis; the weight is [out, in] either way. Quark's suffix
      # is plain .weight, so there the int32 dtype is what marks a packed tensor
      words = shape[1] * 32 // group.bits
      name = name[:-len(group.suffix)] + ".weight"
      shape, quant = ((words, shape[0]) if group.transposed else (shape[0], words)), group.label
    else:
      if dtype == "U8" and name.endswith(".weight") and len(shape) == 2:
        raise ValueError(f"{name}: packed U8 weight has no supported declared quantization scheme")
      quant = default_quant or _DTYPE_TO_QUANT.get(dtype, dtype)
    specs.append(TensorSpec(name=name, shape=shape, quant=quant))
  return specs


def _float_quant_scheme(name:str, cfg:dict[str, Any]) -> dict[str, Any] | None:
  """Resolve a declared float weight scheme per module; a mixed checkpoint is never one global dtype."""
  module = name.rsplit(".weight", 1)[0]
  matches = lambda pattern: bool(re.fullmatch(pattern[3:], module)) if pattern.startswith("re:") else fnmatch.fnmatchcase(module, pattern)
  if any(matches(pattern) for pattern in cfg.get("ignore", [])):
    return None
  fallback = None
  for group in (cfg.get("config_groups") or {}).values():
    weights = group.get("weights") or {}
    if weights.get("type") != "float":
      continue
    targets = group.get("targets") or []
    if any(matches(t) for t in targets if t != "Linear"):
      return weights
    if "Linear" in targets:
      fallback = weights
  return fallback


def _config_for(path:pathlib.Path) -> dict[str, Any]:
  if path.is_dir():
    return _read_json(path / "config.json")
  return _read_json(path.parent / "config.json")


def _profile_from_hf_tensors(path:pathlib.Path, fmt:str, model_id:str | None) -> ModelProfile:
  cfg = _config_for(path)
  qcfg = _quant_config(path if path.is_dir() else path.parent, cfg)
  qmethod = str(qcfg.get("quant_method", fmt if fmt in {"awq", "gptq"} else "")).upper()
  bits = qcfg.get("bits") or qcfg.get("w_bit")
  default_quant = f"{qmethod}_INT{bits}" if qmethod in {"AWQ", "GPTQ"} and bits else None
  specs = _safetensors_specs(path, default_quant, qcfg)
  return _profile_from_specs(path, fmt, cfg, specs, model_id, extra={"quantization_config": qcfg})


def _profile_from_mlx(path:pathlib.Path, model_id:str | None) -> ModelProfile:
  cfg = _config_for(path)
  specs: list[TensorSpec] = []
  st = _safetensors_files(path)
  if st:
    specs = _safetensors_specs(path)
  else:
    npz = path if path.is_file() else next((x for x in (path / "weights.npz", path / "model.npz") if x.exists()), None)
    if npz is not None:
      try:
        import numpy as np  # type: ignore
      except ImportError as exc:
        raise ValueError("MLX .npz profiling requires numpy; use safetensors export or install numpy") from exc
      with np.load(npz) as data:
        specs = [TensorSpec(name=k, shape=tuple(int(x) for x in data[k].shape),
                            quant=_DTYPE_TO_QUANT.get(str(data[k].dtype).upper(), str(data[k].dtype).upper()))
                 for k in data.files]
  return _profile_from_specs(path, "mlx", cfg, specs, model_id)


def _profile_from_onnx(path:pathlib.Path, model_id:str | None) -> ModelProfile:
  p = path if path.is_file() else next(iter(sorted(path.glob("*.onnx"))))
  cfg = _config_for(path if path.is_dir() else path.parent)
  specs = _onnx_specs(p)
  return _profile_from_specs(p, "onnx", cfg, specs, model_id)


def _shape_for_role(spec:TensorSpec, role:str) -> tuple[int, int, int]:
  dims = tuple(int(x) for x in spec.shape)
  if len(dims) == 2:
    return dims[0], dims[1], 0
  if len(dims) == 3 and _is_packed_quant_tensor(spec.name):
    return dims[0], int(math.prod(dims[1:])), 0
  if len(dims) == 3 and not is_ssm_role(role):
    n_expert, rows, cols = dims
    return rows, cols, n_expert
  if is_ssm_role(role):  # rows first, as the GGUF reader orients its (reversed) dims
    return int(dims[0]), int(math.prod(dims[1:]) if len(dims) > 1 else 1), 0
  return 0, 0, 0


def _is_packed_quant_tensor(name:str) -> bool:
  n = name.lower()
  return any(tok in n for tok in ("qweight", "qzeros", "matmulnbits", "scales", "g_idx"))


def _expert_stacks(specs:list[TensorSpec], cfg:dict[str, Any]) -> list[TensorSpec]:
  """Represent individually stored routed experts as one stack per layer and projection.

  Validate the declared expert count and uniform format before applying top-k reads. Stored bytes
  are per matrix, matching TensorRole.weight_bytes, not the bytes of all inactive experts.
  """
  groups:dict[str, list[tuple[int, TensorSpec]]] = {}
  out = []
  for spec in specs:
    match = re.search(r"\.experts\.(\d+)\.", spec.name)
    if match and len(spec.shape) == 2 and spec.name.endswith(".weight"):
      name = spec.name[:match.start()] + ".experts." + spec.name[match.end():]
      groups.setdefault(name, []).append((int(match[1]), spec))
    else:
      out.append(spec)
  declared = cfg.get("num_experts", cfg.get("n_routed_experts"))
  for name, entries in groups.items():
    n = int(declared or len(entries))
    if sorted(i for i, _ in entries) != list(range(n)):
      raise ValueError(f"{name}: expert tensors do not cover the declared {n} experts")
    first = entries[0][1]
    if any((s.shape, s.quant, s.weight_bytes) != (first.shape, first.quant, first.weight_bytes) for _, s in entries):
      raise ValueError(f"{name}: mixed expert shapes or formats require separate routing evidence")
    out.append(TensorSpec(name, (n, *first.shape), first.quant, first.ggml_type, first.weight_bytes))
  return out


def _profile_from_specs(path:pathlib.Path, fmt:str, cfg:dict[str, Any], specs:list[TensorSpec],
                        model_id:str | None, extra:dict[str, Any] | None = None) -> ModelProfile:
  cfg = {**cfg, **(cfg.get("text_config") or {})}  # a multimodal checkpoint keeps the language model's shape there
  specs = _expert_stacks(specs, cfg)
  grouped: dict[tuple[str, int, int, str, int], list[str]] = {}
  physical: dict[tuple[str, int, int, str, int], list[int]] = {}
  classes: dict[tuple[str, int, int, str, int], str] = {}
  tensors_for_arch: list[tuple[str, tuple[int, ...], int, int]] = []
  for spec in specs:
    rc = classify_tensor_role(spec.name, tuple(spec.shape))
    role = role_from_tensor_name(spec.name, tuple(spec.shape))
    if role in {"embedding", "norm", "other"}:
      continue
    rows, cols, n_expert = _shape_for_role(spec, role)
    if rows <= 0 or cols <= 0:
      continue
    # one row per (role, shape, quant), as the GGUF reader groups (k and v together); its class is the first tensor's
    key = (role, rows, cols, spec.quant, n_expert)
    grouped.setdefault(key, []).append(spec.name)
    if spec.weight_bytes is not None:
      physical.setdefault(key, []).append(spec.weight_bytes)
    classes.setdefault(key, rc.value)
    tensors_for_arch.append((spec.name, spec.shape, spec.ggml_type, 0))

  arch = cfg.get("model_type") or (cfg.get("architectures") or [None])[0]
  kv = _kv_from_config(cfg)
  roles = tuple(TensorRole(role=role, tensor_name=names[0], rows=rows, cols=cols, quant=quant,
                           ggml_type=-1, count=len(names), role_class=classes[(role, rows, cols, quant, n_expert)],
                           n_expert=n_expert,
                           experts_read=experts_read(kv, kv["general.architecture"], n_expert) if n_expert else 0,
                           weight_bytes=sum(physical[(role, rows, cols, quant, n_expert)]) / len(names)
                           if (role, rows, cols, quant, n_expert) in physical else None)
                for (role, rows, cols, quant, n_expert), names in sorted(grouped.items()))
  cls, signals = classify_architecture(kv, tensors_for_arch, str(arch) if arch else None)
  meta = {"format_family": fmt, "architecture_signals": signals, "tensor_count": len(specs),
          "quant_types": sorted({s.quant for s in specs}), **_attention_from_config(cfg, specs),
          **_recurrent_from_config(cfg, specs)}
  if extra: meta.update(extra)
  return ModelProfile(model_id=model_id or path.stem, source=str(path), architecture=str(arch) if arch else None,
                      hidden_size=cfg.get("hidden_size"), ffn_size=cfg.get("intermediate_size"),
                      vocab_size=cfg.get("vocab_size"), layer_count=cfg.get("num_hidden_layers"),
                      roles=roles, metadata=meta, architecture_class=cls)


def _attention_from_config(cfg:dict[str, Any], specs:list[TensorSpec]) -> dict[str, Any]:
  """The attention shape and declared context from config.json, as the GGUF reader records them from its keys: heads,
  kv heads, head dim (head_dim, else hidden / heads), the layers holding a q projection, the context length."""
  heads, hidden = cfg.get("num_attention_heads"), cfg.get("hidden_size")
  dim = cfg.get("head_dim") or (hidden // heads if hidden and heads else None)
  layers = len({s.name for s in specs if classify_tensor_role(s.name, tuple(s.shape)) is RoleClass.ATTENTION_Q})
  att = {k: v for k, v in {"head_count": heads, "head_count_kv": cfg.get("num_key_value_heads") or heads,
                           "head_dim": dim, "layer_count": layers or cfg.get("num_hidden_layers")}.items() if v}
  ctx = cfg.get("max_position_embeddings")
  return {**({"attention": att} if heads else {}),
          **({"context": {"length": int(ctx), "source": "config.json max_position_embeddings", "scaling": None}} if ctx else {})}


# config.json keys of a gated delta net block -> the GGUF reader's recurrent keys (prefill_work.recurrent)
_RECURRENT_CONFIG = {"conv_kernel": ("linear_conv_kernel_dim",), "state_size": ("linear_key_head_dim",),
                     "group_count": ("linear_num_key_heads",),
                     "inner_size": ("linear_num_value_heads", "linear_value_head_dim")}


def _recurrent_from_config(cfg:dict[str, Any], specs:list[TensorSpec]) -> dict[str, Any]:
  """A recurrent block's state shape from config.json, keyed as the GGUF reader keys it, and its layers: those
  holding a convolution tap, as the GGUF reader counts them. Empty for a model without one."""
  layers = len({s.name.split(".layers.")[1].split(".")[0] for s in specs
                if ".layers." in s.name and classify_tensor_role(s.name, tuple(s.shape)) is RoleClass.SSM_CONV})
  shape = {k: math.prod(int(cfg[x]) for x in keys) for k, keys in _RECURRENT_CONFIG.items() if all(cfg.get(x) for x in keys)}
  mamba = {"conv_kernel": "conv_kernel", "state_size": "ssm_state_size", "group_count": "n_groups"}
  if all(cfg.get(k) for k in (*mamba.values(), "mamba_num_heads", "mamba_head_dim")):
    shape = {k: int(cfg[v]) for k, v in mamba.items()}
    shape["inner_size"] = int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"])
    shape["kind"] = "mamba2"
    shape["head_count"] = int(cfg["mamba_num_heads"])
    if cfg.get("chunk_size"):
      shape["chunk_size"] = int(cfg["chunk_size"])
  return {"recurrent": {"layer_count": layers, **shape}} if layers and all(k in shape for k in _RECURRENT_CONFIG) else {}


def _kv_from_config(cfg:dict[str, Any]) -> dict[str, Any]:
  arch = cfg.get("model_type") or (cfg.get("architectures") or ["hf"])[0]
  kv = {"general.architecture": arch}
  for src, dst in (("hidden_size", "embedding_length"), ("intermediate_size", "feed_forward_length"),
                   ("num_hidden_layers", "block_count")):
    if src in cfg:
      kv[f"{arch}.{dst}"] = cfg[src]
  if "num_experts" in cfg:
    kv[f"{arch}.expert_count"] = cfg["num_experts"]
  elif "n_routed_experts" in cfg:
    kv[f"{arch}.expert_count"] = cfg["n_routed_experts"]
  if "num_experts_per_tok" in cfg:
    kv[f"{arch}.expert_used_count"] = cfg["num_experts_per_tok"]
  if isinstance(cfg.get("ssm"), dict):
    for k, v in cfg["ssm"].items():
      kv[f"{arch}.ssm.{k}"] = v
  return kv


def _read_varint(data:bytes, pos:int) -> tuple[int, int]:
  shift = 0
  out = 0
  while True:
    b = data[pos]
    pos += 1
    out |= (b & 0x7F) << shift
    if not (b & 0x80):
      return out, pos
    shift += 7


def _proto_fields(data:bytes) -> list[tuple[int, int, int | bytes]]:
  pos = 0
  fields: list[tuple[int, int, int | bytes]] = []
  while pos < len(data):
    key, pos = _read_varint(data, pos)
    field, wire = key >> 3, key & 7
    if wire == 0:
      val, pos = _read_varint(data, pos)
      fields.append((field, wire, val))
    elif wire == 2:
      n, pos = _read_varint(data, pos)
      fields.append((field, wire, data[pos:pos+n]))
      pos += n
    elif wire == 5:
      fields.append((field, wire, data[pos:pos+4]))
      pos += 4
    elif wire == 1:
      fields.append((field, wire, data[pos:pos+8]))
      pos += 8
    else:
      raise ValueError(f"unsupported protobuf wire type {wire}")
  return fields


def _onnx_specs(path:pathlib.Path) -> list[TensorSpec]:
  model = path.read_bytes()
  graph = next((v for f, _w, v in _proto_fields(model) if f == 7 and isinstance(v, bytes)), None)
  if graph is None:
    return []
  specs: list[TensorSpec] = []
  for field, _wire, val in _proto_fields(graph):
    if field != 5 or not isinstance(val, bytes):  # GraphProto.initializer
      continue
    dims: list[int] = []
    data_type = 1
    name = ""
    for tf, _tw, tv in _proto_fields(val):
      if tf == 1 and isinstance(tv, int):
        dims.append(tv)
      elif tf == 2 and isinstance(tv, int):
        data_type = tv
      elif tf == 8 and isinstance(tv, bytes):
        name = tv.decode("utf-8")
    if name:
      specs.append(TensorSpec(name=name, shape=tuple(dims), quant=_ONNX_DTYPE.get(data_type, f"ONNX_{data_type}")))
  return specs
