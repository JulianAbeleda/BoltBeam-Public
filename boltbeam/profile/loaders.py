from __future__ import annotations

import json
import math
import pathlib
import struct
from dataclasses import dataclass
from typing import Any

from boltbeam.profile.architecture import classify_architecture
from boltbeam.profile.gguf import GGML_TYPE_NAMES, profile_from_gguf
from boltbeam.profile.ir import ModelProfile, TensorRole
from boltbeam.profile.roles import classify_tensor_role, role_from_tensor_name
from boltbeam.vocab import is_ssm_role

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


def _safetensors_specs(path:pathlib.Path, default_quant:str | None = None) -> list[TensorSpec]:
  specs: list[TensorSpec] = []
  for f in _safetensors_files(path):
    header = _read_safetensors_header(f)
    for name, meta in header.items():
      if name == "__metadata__" or not isinstance(meta, dict):
        continue
      dtype = str(meta.get("dtype", "F16")).upper()
      quant = default_quant or _DTYPE_TO_QUANT.get(dtype, dtype)
      specs.append(TensorSpec(name=name, shape=tuple(int(x) for x in meta.get("shape", ())),
                              quant=quant))
  return specs


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
  specs = _safetensors_specs(path, default_quant)
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
  if is_ssm_role(role):
    return int(dims[-1]), int(math.prod(dims[:-1]) if len(dims) > 1 else 1), 0
  return 0, 0, 0


def _is_packed_quant_tensor(name:str) -> bool:
  n = name.lower()
  return any(tok in n for tok in ("qweight", "qzeros", "matmulnbits", "scales", "g_idx"))


def _profile_from_specs(path:pathlib.Path, fmt:str, cfg:dict[str, Any], specs:list[TensorSpec],
                        model_id:str | None, extra:dict[str, Any] | None = None) -> ModelProfile:
  grouped: dict[tuple[str, int, int, str, int, str], list[str]] = {}
  tensors_for_arch: list[tuple[str, tuple[int, ...], int, int]] = []
  for spec in specs:
    rc = classify_tensor_role(spec.name)
    role = role_from_tensor_name(spec.name)
    if role in {"embedding", "norm", "other"}:
      continue
    rows, cols, n_expert = _shape_for_role(spec, role)
    if rows <= 0 or cols <= 0:
      continue
    key = (role, rows, cols, spec.quant, n_expert, rc.value)
    grouped.setdefault(key, []).append(spec.name)
    tensors_for_arch.append((spec.name, spec.shape, spec.ggml_type, 0))

  roles = tuple(TensorRole(role=role, tensor_name=names[0], rows=rows, cols=cols, quant=quant,
                           ggml_type=-1, count=len(names), role_class=role_class, n_expert=n_expert)
                for (role, rows, cols, quant, n_expert, role_class), names in sorted(grouped.items()))
  arch = cfg.get("model_type") or (cfg.get("architectures") or [None])[0]
  cls, signals = classify_architecture(_kv_from_config(cfg), tensors_for_arch, str(arch) if arch else None)
  meta = {"format_family": fmt, "architecture_signals": signals, "tensor_count": len(specs),
          "quant_types": sorted({s.quant for s in specs})}
  if extra: meta.update(extra)
  return ModelProfile(model_id=model_id or path.stem, source=str(path), architecture=str(arch) if arch else None,
                      hidden_size=cfg.get("hidden_size"), ffn_size=cfg.get("intermediate_size"),
                      vocab_size=cfg.get("vocab_size"), layer_count=cfg.get("num_hidden_layers"),
                      roles=roles, metadata=meta, architecture_class=cls)


def _kv_from_config(cfg:dict[str, Any]) -> dict[str, Any]:
  arch = cfg.get("model_type") or (cfg.get("architectures") or ["hf"])[0]
  kv = {"general.architecture": arch}
  for src, dst in (("hidden_size", "embedding_length"), ("intermediate_size", "feed_forward_length"),
                   ("num_hidden_layers", "block_count")):
    if src in cfg:
      kv[f"{arch}.{dst}"] = cfg[src]
  if "num_experts" in cfg:
    kv[f"{arch}.expert_count"] = cfg["num_experts"]
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
