"""Canonical, backend-neutral identity for one generated kernel and a route of kernels."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib, json, math, re
from typing import Any

KERNEL_CANDIDATE_SCHEMA = "boltbeam.kernel_candidate.v1"
KERNEL_ROUTE_SCHEMA = "boltbeam.kernel_route.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_hash(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _strict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
  if not isinstance(value, dict): raise ValueError(f"{label} must be an object")
  missing, unknown = keys-set(value), set(value)-keys
  if missing: raise ValueError(f"{label} missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"{label} has unknown fields {sorted(unknown)}")
  return value


def _text(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")
  return value


def _positive(value: Any, label: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise ValueError(f"{label} must be a positive int")
  return value


def _json_copy(value: Any, label: str) -> Any:
  try: return json.loads(json.dumps(value, allow_nan=False))
  except (TypeError, ValueError) as exc: raise ValueError(f"{label} must be finite JSON data") from exc


def _validate_target(target: Any) -> dict[str, Any]:
  target = _strict(target, {"target_id", "backend", "arch", "subgroup_size", "resolved_target_hash"}, "target")
  for key in ("target_id", "backend", "arch"): _text(target[key], f"target.{key}")
  _positive(target["subgroup_size"], "target.subgroup_size")
  if not isinstance(target["resolved_target_hash"], str) or not _SHA256.fullmatch(target["resolved_target_hash"]):
    raise ValueError("target.resolved_target_hash must be a lowercase SHA-256 digest")
  return target


def _validate_launch(launch: Any) -> dict[str, Any]:
  launch = _strict(launch, {"grid", "block", "dynamic_shared_memory_bytes"}, "launch")
  for name in ("grid", "block"):
    if not isinstance(launch[name], list) or len(launch[name]) != 3: raise ValueError(f"launch.{name} must contain three dimensions")
    for i, value in enumerate(launch[name]): _positive(value, f"launch.{name}[{i}]")
  value = launch["dynamic_shared_memory_bytes"]
  if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError("launch.dynamic_shared_memory_bytes must be a non-negative int")
  return launch


def _validate_abi(abi: Any) -> list[dict[str, Any]]:
  if not isinstance(abi, list) or not abi: raise ValueError("abi must be a non-empty list")
  names: set[str] = set()
  for i, arg in enumerate(abi):
    arg = _strict(arg, {"name", "access", "dtype", "layout"}, f"abi[{i}]")
    name = _text(arg["name"], f"abi[{i}].name")
    if name in names: raise ValueError(f"duplicate ABI name {name!r}")
    names.add(name)
    if arg["access"] not in ("read", "write", "read_write"): raise ValueError(f"abi[{i}].access is invalid")
    _text(arg["dtype"], f"abi[{i}].dtype"); _text(arg["layout"], f"abi[{i}].layout")
  return abi


@dataclass(frozen=True)
class KernelCandidate:
  payload: dict[str, Any]
  _canonical: str = field(init=False, repr=False, compare=False)

  def __post_init__(self) -> None:
    payload = _json_copy(self.payload, "kernel_candidate")
    _strict(payload, {"schema_version", "kernel_id", "family", "phase", "role", "operation", "shape", "abi",
                      "parameters", "launch", "resources", "target", "correctness", "provenance"}, "kernel_candidate")
    if payload["schema_version"] != KERNEL_CANDIDATE_SCHEMA: raise ValueError("unsupported kernel candidate schema")
    for key in ("kernel_id", "family", "phase", "role", "operation"): _text(payload[key], key)
    shape = payload["shape"]
    if not isinstance(shape, dict) or not shape: raise ValueError("shape must be a non-empty object")
    for key, value in shape.items(): _text(key, "shape key"); _positive(value, f"shape.{key}")
    _validate_abi(payload["abi"])
    if not isinstance(payload["parameters"], dict): raise ValueError("parameters must be an object")
    _validate_launch(payload["launch"])
    resources = _strict(payload["resources"], {"workspace_bytes", "static_shared_memory_bytes"}, "resources")
    for key, value in resources.items():
      if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError(f"resources.{key} must be a non-negative int")
    _validate_target(payload["target"])
    correctness = _strict(payload["correctness"], {"oracle", "atol", "rtol", "bit_exact"}, "correctness")
    _text(correctness["oracle"], "correctness.oracle")
    for key in ("atol", "rtol"):
      if isinstance(correctness[key], bool) or not isinstance(correctness[key], (int, float)) or not math.isfinite(correctness[key]) or correctness[key] < 0:
        raise ValueError(f"correctness.{key} must be finite and non-negative")
    if not isinstance(correctness["bit_exact"], bool): raise ValueError("correctness.bit_exact must be a bool")
    provenance = _strict(payload["provenance"], {"generator_id", "generator_revision", "schema_revision"}, "provenance")
    for key in provenance: _text(provenance[key], f"provenance.{key}")
    if provenance["schema_revision"] != KERNEL_CANDIDATE_SCHEMA: raise ValueError("provenance.schema_revision mismatch")
    object.__setattr__(self, "payload", payload)
    object.__setattr__(self, "_canonical", canonical_json(payload))

  @property
  def candidate_hash(self) -> str: return hashlib.sha256(self._canonical.encode("ascii")).hexdigest()
  def canonical_json(self) -> str: return self._canonical
  def to_dict(self) -> dict[str, Any]: return json.loads(self._canonical)


@dataclass(frozen=True)
class KernelRoute:
  payload: dict[str, Any]
  _canonical: str = field(init=False, repr=False, compare=False)

  def __post_init__(self) -> None:
    payload = _json_copy(self.payload, "kernel_route")
    _strict(payload, {"schema_version", "route_id", "phase", "role", "components", "outputs", "provenance"}, "kernel_route")
    if payload["schema_version"] != KERNEL_ROUTE_SCHEMA: raise ValueError("unsupported kernel route schema")
    for key in ("route_id", "phase", "role"): _text(payload[key], key)
    components = payload["components"]
    if not isinstance(components, list) or not components: raise ValueError("components must be a non-empty list")
    seen: set[str] = set()
    for i, component in enumerate(components):
      component = _strict(component, {"name", "candidate_hash", "candidate", "depends_on"}, f"components[{i}]")
      name = _text(component["name"], f"components[{i}].name")
      if name in seen: raise ValueError(f"duplicate route component {name!r}")
      candidate = KernelCandidate(component["candidate"])
      if component["candidate_hash"] != candidate.candidate_hash: raise ValueError(f"component {name!r} candidate hash mismatch")
      if not isinstance(component["depends_on"], list) or any(x not in seen for x in component["depends_on"]):
        raise ValueError(f"component {name!r} has a missing or forward dependency")
      seen.add(name)
    if not isinstance(payload["outputs"], list) or not payload["outputs"] or any(x not in seen for x in payload["outputs"]):
      raise ValueError("outputs must name route components")
    provenance = _strict(payload["provenance"], {"generator_id", "generator_revision", "schema_revision"}, "provenance")
    if provenance["schema_revision"] != KERNEL_ROUTE_SCHEMA: raise ValueError("route provenance.schema_revision mismatch")
    for key in provenance: _text(provenance[key], f"provenance.{key}")
    object.__setattr__(self, "payload", payload); object.__setattr__(self, "_canonical", canonical_json(payload))

  @property
  def route_hash(self) -> str: return hashlib.sha256(self._canonical.encode("ascii")).hexdigest()
  def canonical_json(self) -> str: return self._canonical
  def to_dict(self) -> dict[str, Any]: return json.loads(self._canonical)


__all__ = ["KERNEL_CANDIDATE_SCHEMA", "KERNEL_ROUTE_SCHEMA", "KernelCandidate", "KernelRoute", "canonical_hash", "canonical_json"]
