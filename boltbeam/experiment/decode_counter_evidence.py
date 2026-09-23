"""Fail-closed, CPU-only admission for decode PMC and HSACO evidence."""
from __future__ import annotations

from typing import Any, Mapping

PREFLIGHT_SCHEMA = "boltbeam.profiler_preflight.v1"
PMC_SCHEMA = "boltbeam.raw_pmc_capture.v1"
GRAPH_SCHEMA = "boltbeam.graph_dispatch_attribution.v1"
HSACO_SCHEMA = "boltbeam.hsaco_resource_audit.v1"


def admit_decode_counter_evidence(*, candidate_id: str, preflight: Mapping[str, Any], raw_pmc: Mapping[str, Any],
                                 graph: Mapping[str, Any], hsaco: Mapping[str, Any]) -> dict[str, Any]:
  """Atomically admit only validated, candidate-scoped decode counter evidence."""
  if not isinstance(candidate_id, str) or not candidate_id:
    raise ValueError("candidate_id must be non-empty")
  _schema(preflight, PREFLIGHT_SCHEMA)
  if preflight.get("status") != "passed":
    return _result("blocked", "profiler_preflight_failed")
  _schema(raw_pmc, PMC_SCHEMA)
  _schema(graph, GRAPH_SCHEMA)
  _schema(hsaco, HSACO_SCHEMA)
  for artifact in (raw_pmc, graph, hsaco):
    if artifact.get("candidate_id") != candidate_id:
      return _result("blocked", "candidate_identity_mismatch")
  if raw_pmc.get("status") != "validated" or not raw_pmc.get("positive_control_passed"):
    return _result("blocked", "raw_pmc_not_validated")
  rows = raw_pmc.get("rows")
  if not isinstance(rows, list) or not rows or any(not _raw_row(row) for row in rows):
    return _result("inconclusive", "raw_pmc_incomplete_or_counterless")
  dispatches = graph.get("dispatches")
  if graph.get("status") != "validated" or not isinstance(dispatches, list) or not dispatches:
    return _result("inconclusive", "graph_attribution_incomplete")
  raw_by_dispatch = {row["dispatch_id"]: row for row in rows}
  seen: set[str] = set()
  hot_gemm = False
  for row in dispatches:
    if not _graph_row(row) or row["dispatch_id"] in seen:
      return _result("inconclusive", "graph_attribution_non_atomic")
    seen.add(row["dispatch_id"])
    pmc = raw_by_dispatch.get(row["dispatch_id"])
    if pmc is None or not _same_identity(row, pmc):
      return _result("blocked", "stale_or_missing_dispatch_counter")
    hot_gemm |= row["kind"] == "gemm" and bool(row.get("hot"))
  if not hot_gemm:
    return _result("inconclusive", "hot_gemm_counter_missing")
  resources = hsaco.get("resources")
  if hsaco.get("status") != "validated" or not isinstance(resources, list) or not resources:
    return _result("inconclusive", "hsaco_resource_audit_incomplete")
  audit = {(row.get("binary_sha256"), row.get("kernel")): row for row in resources if _resource_row(row)}
  for row in dispatches:
    if (row["binary_sha256"], row["kernel"]) not in audit:
      return _result("inconclusive", "hsaco_resource_missing_for_dispatch")
  return {"status": "accepted", "candidate_id": candidate_id, "raw_counter_rows": len(rows),
          "attributed_dispatches": len(dispatches), "counter_quality": "measured",
          "resource_quality": "static_hsaco", "derived_occupancy": None}


def _schema(value: Mapping[str, Any], expected: str) -> None:
  if not isinstance(value, Mapping) or value.get("schema") != expected:
    raise ValueError(f"expected {expected}")


def _raw_row(row: Any) -> bool:
  return isinstance(row, Mapping) and _identity(row) and isinstance(row.get("raw_words"), list) and bool(row["raw_words"]) and \
    isinstance(row.get("counters"), Mapping) and bool(row["counters"]) and row.get("counter_quality") == "measured"


def _graph_row(row: Any) -> bool:
  return isinstance(row, Mapping) and _identity(row) and isinstance(row.get("kernel"), str) and bool(row["kernel"]) and \
    row.get("kind") in {"gemm", "other"} and row.get("completion_before_read") is True and row.get("read_before_copyout") is True


def _resource_row(row: Any) -> bool:
  return isinstance(row, Mapping) and isinstance(row.get("binary_sha256"), str) and len(row["binary_sha256"]) == 64 and \
    isinstance(row.get("kernel"), str) and all(isinstance(row.get(k), int) and row[k] >= 0 for k in
                                                ("vgpr", "sgpr", "lds_bytes", "vgpr_spills", "sgpr_spills", "wave_size"))


def _identity(row: Mapping[str, Any]) -> bool:
  return all(isinstance(row.get(k), str) and row[k] for k in ("dispatch_id", "source_sha256", "binary_sha256"))


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
  return all(left[key] == right[key] for key in ("source_sha256", "binary_sha256"))


def _result(status: str, reason: str) -> dict[str, Any]:
  return {"status": status, "reason": reason, "counter_quality": "unavailable", "derived_occupancy": None}
