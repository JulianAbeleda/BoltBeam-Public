"""Canonical CPU-only validation for BoltBeam observation envelopes.

This module owns identity and quality rules shared by tinygrad producers and
BoltBeam consumers.  It never imports tinygrad or touches a device.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from boltbeam.core.canonical import canonical_json, prefixed, sha256_hex

PROTOCOL_SCHEMA = "boltbeam.observation.v1"
OBSERVATION_KINDS = frozenset({"timing", "profile_events", "pmc", "static_resource", "launch", "correctness"})
EVIDENCE_QUALITIES = frozenset({"measured", "derived", "proxy", "estimated"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("protocol_schema", "producer_id", "producer_version", "source_commit", "source_tree_sha256",
             "run_id", "attempt_id", "model_digest", "workload_digest", "schedule_digest", "launch_digest",
             "target_id", "driver_id", "runtime_id", "clock_domain", "observation_kind", "capture_pass")


def canonical_digest(value: Any) -> str:
  return prefixed(sha256_hex(canonical_json(value).encode("utf-8")))


def binary_digest(value: bytes) -> str:
  if not isinstance(value, bytes):
    raise TypeError("binary_digest expects bytes")
  return prefixed(sha256_hex(value))


def observation_id(envelope: Mapping[str, Any]) -> str:
  """Derive the stable row identity from explicit dispatch/program fields."""
  fields = ("protocol_schema", "run_id", "attempt_id", "observation_kind", "capture_pass", "launch_digest",
            "dispatch_id", "program_id", "queue_id", "stream_id", "graph_id", "launch_order")
  return canonical_digest({field: envelope.get(field) for field in fields})


def validate_observation(envelope: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
  """Return structured validation errors; never infer missing identity."""
  errors: list[dict[str, Any]] = []
  if not isinstance(envelope, Mapping):
    return ({"field": "envelope", "reason": "expected mapping"},)
  if envelope.get("protocol_schema") != PROTOCOL_SCHEMA:
    errors.append({"field": "protocol_schema", "reason": f"expected {PROTOCOL_SCHEMA}"})
  for field in _REQUIRED:
    if field not in envelope:
      errors.append({"field": field, "reason": "required"})
  if envelope.get("source_commit") is None or envelope.get("source_tree_sha256") is None:
    errors.append({"field": "source_identity", "reason": "tinygrad-derived observations require source identity"})
  if envelope.get("observation_kind") not in OBSERVATION_KINDS:
    errors.append({"field": "observation_kind", "reason": "unknown observation kind"})
  if envelope.get("capture_pass") is not None and (
      not isinstance(envelope["capture_pass"], int) or isinstance(envelope["capture_pass"], bool) or envelope["capture_pass"] < 0):
    errors.append({"field": "capture_pass", "reason": "expected non-negative integer"})
  for field in ("model_digest", "workload_digest", "schedule_digest", "launch_digest"):
    value = envelope.get(field)
    if not isinstance(value, str) or not value.startswith("sha256:") or _SHA256.fullmatch(value[7:]) is None:
      errors.append({"field": field, "reason": "expected sha256:<64 lowercase hex>"})
  quality = envelope.get("evidence_quality")
  if quality is not None:
    values = quality.values() if isinstance(quality, Mapping) else (quality,)
    for value in values:
      if value not in EVIDENCE_QUALITIES:
        errors.append({"field": "evidence_quality", "reason": "unknown quality"})
  return tuple(errors)


__all__ = ["EVIDENCE_QUALITIES", "OBSERVATION_KINDS", "PROTOCOL_SCHEMA", "binary_digest", "canonical_digest",
           "canonical_json", "observation_id", "validate_observation"]
