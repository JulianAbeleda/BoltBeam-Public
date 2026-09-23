"""Canonical exact semantic identity shared by MR7 ranking and MR8 search handoff."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boltbeam.core.canonical import canonical_json, sha256_json as content_sha256


SEMANTIC_IDENTITY_FIELDS = (
  "phase", "tensor_name", "module_path", "role", "logical_m", "logical_n", "logical_k",
  "source_quant_storage", "source_layout", "module_representation", "input_dtype", "output_dtype",
  "accumulator_dtype",
)
SEMANTIC_WORKLOAD_SCHEMA = "tinygrad.semantic_provider_workload.v1"
_INTEGER_FIELDS = ("logical_m", "logical_n", "logical_k")
_STRING_FIELDS = tuple(field for field in SEMANTIC_IDENTITY_FIELDS if field not in _INTEGER_FIELDS)


SEMANTIC_IDENTITY_FIELDS_SHA256 = content_sha256(list(SEMANTIC_IDENTITY_FIELDS))


def require_matching_identity_fields(declared_fields: Any, declared_sha256: Any) -> None:
  """Validate that provider's declared fields match this repo's vocabulary."""
  if declared_fields is None or declared_sha256 is None:
    raise ValueError("provider must declare identity_fields and identity_fields_sha256")
  if not isinstance(declared_fields, (list, tuple)):
    raise ValueError("provider identity_fields must be a sequence")
  if tuple(declared_fields) != SEMANTIC_IDENTITY_FIELDS:
    raise ValueError("provider identity_fields do not match expected field vocabulary")
  if not isinstance(declared_sha256, str) or len(declared_sha256) != 64 or \
      any(char not in "0123456789abcdef" for char in declared_sha256):
    raise ValueError("provider identity_fields_sha256 must be a valid 64-char lowercase hex string")
  if declared_sha256 != SEMANTIC_IDENTITY_FIELDS_SHA256:
    raise ValueError("provider identity_fields_sha256 does not match computed hash of declared fields")


def normalize_semantic_identity(identity: Any) -> dict[str, Any]:
  """Return only exact identity fields, rejecting partial or ambiguous joins."""
  if not isinstance(identity, Mapping) or identity.get("metadata_status", "semantic") != "semantic":
    raise ValueError("semantic identity is unavailable")
  if any(field not in identity for field in SEMANTIC_IDENTITY_FIELDS):
    raise ValueError("semantic identity is incomplete")
  if any(not isinstance(identity[field], int) or isinstance(identity[field], bool) or identity[field] < 1
         for field in _INTEGER_FIELDS):
    raise ValueError("semantic identity has invalid MNK")
  if any(not isinstance(identity[field], str) or not identity[field] for field in _STRING_FIELDS):
    raise ValueError("semantic identity has ambiguous strings")
  return {field:identity[field] for field in SEMANTIC_IDENTITY_FIELDS}


def semantic_identity_sha256(identity: Any) -> str:
  return content_sha256(normalize_semantic_identity(identity))


def validate_exact_semantic_workload(workload: Any) -> dict[str, Any]:
  """Validate that an exported workload is an exact, internally consistent identity binding."""
  if not isinstance(workload, Mapping) or workload.get("schema") != SEMANTIC_WORKLOAD_SCHEMA:
    raise ValueError("exact semantic workload schema is unavailable")
  identity = normalize_semantic_identity(workload.get("semantic_identity"))
  expected_shape = {name:identity[f"logical_{name}"] for name in ("m", "n", "k")}
  if workload.get("operation") != "matmul" or workload.get("shape") != expected_shape:
    raise ValueError("semantic workload shape disagrees with its exact identity")
  operands = workload.get("operands")
  expected_operands = (("a", "dtype", identity["input_dtype"]),
                       ("b", "quantization", identity["source_quant_storage"]),
                       ("b", "layout", identity["source_layout"]),
                       ("c", "dtype", identity["output_dtype"]))
  if not isinstance(operands, Mapping) or any(not isinstance(operands.get(name), Mapping) or
      operands[name].get(field) != expected for name,field,expected in expected_operands):
    raise ValueError("semantic workload operands disagree with its exact identity")
  if workload.get("fixture_shape_substitution") != "forbidden":
    raise ValueError("exact semantic workload forbids fixture shape substitution")
  model_hash, target = workload.get("model_hash"), workload.get("target")
  if not isinstance(model_hash, str) or len(model_hash) != 64 or any(char not in "0123456789abcdef" for char in model_hash):
    raise ValueError("exact semantic workload requires a model SHA-256")
  if not isinstance(target, Mapping) or not isinstance(target.get("backend"), str) or not target.get("backend") or \
      not isinstance(target.get("target_id"), str) or not target.get("target_id") or \
      not isinstance(target.get("resolved_target_hash"), str) or len(target["resolved_target_hash"]) != 64:
    raise ValueError("exact semantic workload requires a resolved target")
  tolerance = workload.get("tolerance")
  if not isinstance(tolerance, Mapping) or any(not isinstance(tolerance.get(name), (int, float)) or
      isinstance(tolerance.get(name), bool) or tolerance[name] < 0 for name in ("atol", "rtol")):
    raise ValueError("exact semantic workload requires non-negative tolerances")
  return identity


__all__ = ["SEMANTIC_IDENTITY_FIELDS", "SEMANTIC_IDENTITY_FIELDS_SHA256", "SEMANTIC_WORKLOAD_SCHEMA",
           "canonical_json", "content_sha256", "normalize_semantic_identity", "require_matching_identity_fields",
           "semantic_identity_sha256", "validate_exact_semantic_workload"]
