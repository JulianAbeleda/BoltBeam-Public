"""Dependency-free semantic candidate helpers.

This module is a pure-data port of tinygrad's qk semantic-candidate helpers.
It intentionally does not import BoltBeam domain modules so callers can reuse
the logic in tests, reporting, or candidate assembly without pulling in CLI or
boundary dependencies.
"""
from __future__ import annotations

import re
from typing import Any


def slug(value: str) -> str:
  return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def current_runtime(row: dict[str, Any]) -> dict[str, Any]:
  lowering = row["current_lowering"]
  return {
    "winner": lowering.get("winner"),
    "family": lowering.get("family"),
    "parts": int(lowering.get("parts") or 0),
    "opts": list(lowering.get("opts") or []),
    "reduction": lowering.get("reduction"),
    "requires": list(lowering.get("requires") or []),
  }


def runtime_storage_bytes(policy: dict[str, Any]) -> int:
  total = 0
  for entry in policy.get("entries", []):
    if entry.get("winner") != "fused_graph":
      total += int((entry.get("storage") or {}).get("persistent_bytes") or 0)
  return total


def no_extra_storage_effect(note: str) -> dict[str, Any]:
  return {
    "persistent_bytes_delta": 0,
    "shared_bytes_delta": 0,
    "nonpersistent_bytes_delta": 0,
    "metadata_sidecar_bytes": 0,
    "storage_note": note,
  }


def correctness_provenance(*, full_decode_supported: bool) -> dict[str, Any]:
  return {
    "reference_unpacked": "covered_by_qk_layout_reference_tests",
    "amd_gemv": "required_by_amd_microbench_gate",
    "full_decode_ab": "required_for_full_decode_promotion" if full_decode_supported
                      else "not_applicable_to_this_candidate",
    "note": "CPU/Mac tests prove packed-layout reference semantics; AMD microbench proves GEMV kernel numerics.",
  }


def is_raw_accept_status(status: str | None) -> bool:
  return status in ("accept", "raw_accept")


__all__ = [
  "correctness_provenance",
  "current_runtime",
  "is_raw_accept_status",
  "no_extra_storage_effect",
  "runtime_storage_bytes",
  "slug",
]
