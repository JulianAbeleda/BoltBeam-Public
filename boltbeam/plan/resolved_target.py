"""Canonical join of registry target identity and stable provider facts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from boltbeam.core.canonical import canonical_json as _canonical, sha256_hex as _sha256_hex
from boltbeam.target.targets import TARGETS

SCHEMA = "boltbeam.resolved_target.v1"
_VOLATILE_KEYS = frozenset({"current_allocated_bytes", "current_allocated_size", "timestamp", "captured_at"})


def _stable(value:Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _stable(child) for key, child in value.items()
            if key not in _VOLATILE_KEYS and not str(key).lower().endswith(("_path", "_root", "_dir"))}
  if isinstance(value, (list, tuple)): return [_stable(child) for child in value]
  return value


def resolved_target_document(target_id:str, observed_facts:Mapping[str, Any]) -> dict[str, Any]:
  """Return the deterministic target document used by candidates and providers."""
  if target_id not in TARGETS: raise ValueError(f"unknown resolved target {target_id!r}")
  if not isinstance(observed_facts, Mapping): raise ValueError("observed target facts must be an object")
  try: observed = json.loads(_canonical(_stable(observed_facts)))
  except (TypeError, ValueError) as exc: raise ValueError(f"observed target facts must be JSON data: {exc}") from exc
  target = TARGETS[target_id]
  # Target descriptors contain tuple-valued capability lists in Python. The
  # resolved document is a wire artifact, so normalize both halves to JSON
  # containers before validation or hashing.
  registry = json.loads(_canonical(_stable(target.to_json())))
  return {"schema": SCHEMA, "target_id": target_id, "registry_target": registry, "observed_facts": observed}


def resolved_target_hash(document:Mapping[str, Any]) -> str:
  validate_resolved_target_document(document)
  return _sha256_hex(_canonical(document).encode("ascii"))


def validate_resolved_target_document(document:Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(document, Mapping) or set(document) != {"schema", "target_id", "registry_target", "observed_facts"}:
    raise ValueError("resolved target document has invalid fields")
  if document.get("schema") != SCHEMA: raise ValueError(f"resolved target schema must be {SCHEMA}")
  target_id = document.get("target_id")
  if not isinstance(target_id, str) or target_id not in TARGETS: raise ValueError("resolved target_id is not registered")
  expected = resolved_target_document(target_id, document["observed_facts"])
  if dict(document) != expected: raise ValueError("resolved target registry descriptor is stale or non-canonical")
  return expected


def candidate_target(document:Mapping[str, Any]) -> dict[str, Any]:
  """The one v2 candidate target block that a resolved target document implies."""
  resolved = validate_resolved_target_document(document)
  digest = _sha256_hex(_canonical(resolved).encode("ascii"))
  observed, registry = resolved["observed_facts"], resolved["registry_target"]
  target = {
    "target_id": resolved["target_id"],
    "backend": str(observed.get("backend", registry["backend"])).upper(),
    "arch": observed.get("architecture", observed.get("arch")),
    "subgroup_size": observed.get("subgroup_size", registry.get("subgroup_size", registry["wave_size"])),
    "resolved_target_hash": digest,
  }
  if not target["arch"]: raise ValueError("resolved target observed facts require architecture")
  return target


def validate_candidate_target(target:Mapping[str, Any], document:Mapping[str, Any]) -> None:
  """Bind one v2 candidate target to the exact deterministic target document."""
  if dict(target) != candidate_target(document): raise ValueError("candidate target does not match resolved target document")


__all__ = ["SCHEMA", "candidate_target", "resolved_target_document", "resolved_target_hash", "validate_candidate_target",
           "validate_resolved_target_document"]
