"""Pure resource snapshot joins for MMQ epoch reports.

The input is a structural ``tinygrad.kernel_resource_trace.v1``-style mapping.
This module intentionally does not import tinygrad; it only validates and joins
resource evidence that has already been serialized.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from boltbeam.search.epochs.epoch_model import EpochReport


SCHEMA = "tinygrad.kernel_resource_trace.v1"
RESOURCE_FIELDS = frozenset(
  (
    "vgpr",
    "sgpr",
    "lds_bytes",
    "scratch_bytes",
    "workgroup_threads",
    "workgroup",
    "grid",
    "global_size",
    "local_size",
    "occupancy",
  )
)
_NONNEGATIVE_INT_FIELDS = frozenset(("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup_threads"))
_NONNEGATIVE_NUMBER_FIELDS = frozenset(("occupancy",))
_DIM_FIELDS = frozenset(("workgroup", "grid", "global_size", "local_size"))


@dataclass(frozen=True)
class ResourceJoinResult:
  candidate_id: str | None
  resources: Mapping[str, Any] | None
  missing: tuple[str, ...] = ()
  invalid: tuple[Mapping[str, Any], ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()
  report: EpochReport | None = None

  @property
  def ok(self) -> bool:
    return not self.missing and not self.invalid and not self.blockers and self.resources is not None


def validate_resource_snapshot(
  trace: Mapping[str, Any],
  candidate_id: str | None = None,
) -> ResourceJoinResult:
  """Validate and extract candidate-level resources from a resource trace."""
  invalid: list[dict[str, Any]] = []
  missing: list[str] = []
  blockers: list[dict[str, Any]] = []

  if not isinstance(trace, Mapping):
    return ResourceJoinResult(
      candidate_id=candidate_id,
      resources=None,
      invalid=({"field": "trace", "reason": "expected mapping"},),
      blockers=(_blocker("invalid_resource_trace", "resource trace must be a mapping"),),
    )

  schema = trace.get("schema")
  if schema != SCHEMA:
    invalid.append({"field": "schema", "value": schema, "reason": f"expected {SCHEMA}"})

  trace_candidate_id = _string_or_none(trace.get("candidate_id"))
  if candidate_id is not None and trace_candidate_id is not None and trace_candidate_id != candidate_id:
    blockers.append(
      _blocker(
        "candidate_id_mismatch",
        "resource trace candidate_id does not match join target",
        expected=candidate_id,
        actual=trace_candidate_id,
      )
    )

  joined_candidate_id = candidate_id or trace_candidate_id
  resources = _candidate_resources(trace, joined_candidate_id)
  if resources is None:
    missing.append("resources")
    blockers.append(_blocker("resources", "resource trace has no candidate resources"))
  else:
    invalid.extend(_validate_resources(resources))

  if invalid and not blockers:
    blockers.append(_blocker("invalid_resources", "resource trace contains invalid resource fields"))

  return ResourceJoinResult(
    candidate_id=joined_candidate_id,
    resources=copy.deepcopy(resources) if resources is not None else None,
    missing=tuple(missing),
    invalid=tuple(invalid),
    blockers=tuple(blockers),
  )


def join_resource_snapshot(
  trace: Mapping[str, Any],
  target: EpochReport | str,
) -> ResourceJoinResult:
  """Join a resource trace to an ``EpochReport`` or validate it for a candidate id.

  Valid joins augment ``EpochReport.workload["resources"]``. Missing or invalid
  resources leave workload unchanged and append structured blockers to the report.
  """
  if isinstance(target, EpochReport):
    result = validate_resource_snapshot(trace, target.candidate_id)
    report = _augment_report(target, result)
    return replace(result, report=report)
  return validate_resource_snapshot(trace, str(target))


def _candidate_resources(trace: Mapping[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
  top_resources = trace.get("resources")
  if isinstance(top_resources, Mapping):
    return _copy_resource_fields(top_resources)

  rows = trace.get("rows")
  if not isinstance(rows, list | tuple):
    return None

  for row in rows:
    if not isinstance(row, Mapping):
      continue
    row_candidate = _string_or_none(row.get("candidate_id")) or _string_or_none(trace.get("candidate_id"))
    if candidate_id is not None and row_candidate is not None and row_candidate != candidate_id:
      continue
    resources = row.get("resources")
    if isinstance(resources, Mapping):
      copied = _copy_resource_fields(resources)
      if copied:
        return copied
  return None


def _copy_resource_fields(resources: Mapping[str, Any]) -> dict[str, Any]:
  return {key: copy.deepcopy(resources[key]) for key in RESOURCE_FIELDS if key in resources}


def _validate_resources(resources: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
  invalid: list[dict[str, Any]] = []
  for field in _NONNEGATIVE_INT_FIELDS:
    if field not in resources:
      continue
    value = resources[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      invalid.append({"field": f"resources.{field}", "value": value, "reason": "expected non-negative int"})

  for field in _NONNEGATIVE_NUMBER_FIELDS:
    if field not in resources:
      continue
    value = resources[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
      invalid.append({"field": f"resources.{field}", "value": value, "reason": "expected non-negative number"})

  for field in _DIM_FIELDS:
    if field not in resources:
      continue
    value = resources[field]
    if not _is_dim3(value):
      invalid.append({"field": f"resources.{field}", "value": value, "reason": "expected three non-negative ints"})
  return tuple(invalid)


def _is_dim3(value: Any) -> bool:
  if not isinstance(value, list | tuple) or len(value) != 3:
    return False
  return all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)


def _augment_report(report: EpochReport, result: ResourceJoinResult) -> EpochReport:
  if result.ok:
    workload = dict(report.workload)
    workload["resources"] = copy.deepcopy(result.resources)
    workload.setdefault("resource_trace_schema", SCHEMA)
    return replace(report, workload=workload)

  blockers = tuple(dict(blocker, source="resource_snapshot") for blocker in result.blockers)
  return replace(report, blockers=(*report.blockers, *blockers))


def _blocker(missing: str, reason: str, **extra: Any) -> dict[str, Any]:
  blocker = {"missing": missing, "reason": reason}
  blocker.update(extra)
  return blocker


def _string_or_none(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value)
  return text if text else None


__all__ = ["ResourceJoinResult", "join_resource_snapshot", "validate_resource_snapshot"]
