"""Pure timing-result joins for MMQ epoch reports.

The input is a structural ``tinygrad.mmq_timing_result.v1``-style mapping.
This module intentionally does not import tinygrad; it only validates and joins
timing evidence that has already been serialized.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from boltbeam.search.epochs.epoch_model import EpochReport


SCHEMA = "tinygrad.mmq_timing_result.v1"
TIMING_FIELDS = ("shape", "comparator_id", "timing_status", "timings_ms", "speedup_vs_comparator")
REQUIRED_TIMING_FIELDS = ("shape", "comparator_id", "timing_status")
TIMING_STATUSES = frozenset(("measured", "blocked", "not_run", "oracle_only"))
SHAPE_FIELDS = frozenset(("M", "N", "K"))


@dataclass(frozen=True)
class TimingJoinResult:
  candidate_id: str | None
  timing: Mapping[str, Any] | None
  missing: tuple[str, ...] = ()
  invalid: tuple[Mapping[str, Any], ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()
  report: EpochReport | None = None

  @property
  def ok(self) -> bool:
    return not self.missing and not self.invalid and not self.blockers and self.timing is not None


def validate_timing_result(
  result: Mapping[str, Any],
  candidate_id: str | None = None,
) -> TimingJoinResult:
  """Validate and extract candidate-level timing from a timing result."""
  invalid: list[dict[str, Any]] = []
  missing: list[str] = []
  blockers: list[dict[str, Any]] = []

  if not isinstance(result, Mapping):
    return TimingJoinResult(
      candidate_id=candidate_id,
      timing=None,
      invalid=({"field": "result", "reason": "expected mapping"},),
      blockers=(_blocker("invalid_timing_result", "timing result must be a mapping"),),
    )

  schema = result.get("schema")
  if schema != SCHEMA:
    invalid.append({"field": "schema", "value": schema, "reason": f"expected {SCHEMA}"})

  result_candidate_id = _string_or_none(result.get("candidate_id"))
  if candidate_id is not None and result_candidate_id is not None and result_candidate_id != candidate_id:
    blockers.append(
      _blocker(
        "candidate_id_mismatch",
        "timing result candidate_id does not match join target",
        expected=candidate_id,
        actual=result_candidate_id,
      )
    )

  joined_candidate_id = candidate_id or result_candidate_id
  timing = _candidate_timing(result, joined_candidate_id)
  if timing is None:
    missing.append("timing")
    blockers.append(_blocker("timing", "timing result has no candidate timing"))
  else:
    timing_missing, timing_invalid = _validate_timing(timing)
    missing.extend(timing_missing)
    invalid.extend(timing_invalid)

  if (missing or invalid) and not blockers:
    blockers.append(_blocker("invalid_timing", "timing result contains missing or invalid timing fields"))

  return TimingJoinResult(
    candidate_id=joined_candidate_id,
    timing=copy.deepcopy(timing) if timing is not None else None,
    missing=tuple(missing),
    invalid=tuple(invalid),
    blockers=tuple(blockers),
  )


def join_timing_result(
  result: Mapping[str, Any],
  target: EpochReport | str,
) -> TimingJoinResult:
  """Join a timing result to an ``EpochReport`` or validate it for a candidate id.

  Valid joins augment ``EpochReport.workload["timing"]``. Missing or invalid
  timing leaves workload unchanged and appends structured blockers to the report.
  """
  if isinstance(target, EpochReport):
    joined = validate_timing_result(result, target.candidate_id)
    report = _augment_report(target, joined)
    return replace(joined, report=report)
  return validate_timing_result(result, str(target))


def _candidate_timing(result: Mapping[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
  if any(field in result for field in TIMING_FIELDS):
    return _copy_timing_fields(result)

  rows = result.get("rows")
  if not isinstance(rows, list | tuple):
    return None

  for row in rows:
    if not isinstance(row, Mapping):
      continue
    explicit_row_candidate = _string_or_none(row.get("candidate_id"))
    row_candidate = explicit_row_candidate or _string_or_none(result.get("candidate_id"))
    if candidate_id is not None and row_candidate is not None and row_candidate != candidate_id:
      continue
    copied = _copy_timing_fields(row)
    if copied or explicit_row_candidate is not None:
      return copied
  return None


def _copy_timing_fields(timing: Mapping[str, Any]) -> dict[str, Any]:
  return {key: copy.deepcopy(timing[key]) for key in TIMING_FIELDS if key in timing}


def _validate_timing(timing: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
  missing = tuple(field for field in REQUIRED_TIMING_FIELDS if _is_absent(timing.get(field)))
  invalid: list[dict[str, Any]] = []

  timings_ms = timing.get("timings_ms")
  shape = timing.get("shape")
  if shape is not None:
    invalid.extend(_validate_shape(shape))

  status = timing.get("timing_status")
  if status is not None and status not in TIMING_STATUSES:
    invalid.append({"field": "timing_status", "value": status, "reason": f"expected one of {sorted(TIMING_STATUSES)}"})

  if timings_ms is not None:
    if not isinstance(timings_ms, Mapping):
      invalid.append({"field": "timings_ms", "value": timings_ms, "reason": "expected mapping"})
    else:
      for key, value in timings_ms.items():
        if not _is_nonnegative_number(value):
          invalid.append(
            {"field": f"timings_ms.{key}", "value": value, "reason": "expected non-negative number"}
          )

  if "speedup_vs_comparator" in timing:
    value = timing["speedup_vs_comparator"]
    if not _is_nonnegative_number(value):
      invalid.append(
        {"field": "speedup_vs_comparator", "value": value, "reason": "expected non-negative number"}
      )

  return missing, tuple(invalid)


def _validate_shape(shape: Any) -> tuple[dict[str, Any], ...]:
  invalid: list[dict[str, Any]] = []
  if not isinstance(shape, Mapping):
    return ({"field": "shape", "value": shape, "reason": "expected mapping"},)
  unknown = set(shape) - SHAPE_FIELDS
  if unknown:
    invalid.append({"field": "shape", "value": sorted(unknown), "reason": "unknown fields"})
  missing = SHAPE_FIELDS - set(shape)
  if missing:
    invalid.append({"field": "shape", "value": sorted(missing), "reason": "missing required fields"})
  for field in SHAPE_FIELDS:
    if field not in shape:
      continue
    value = shape[field]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      invalid.append({"field": f"shape.{field}", "value": value, "reason": "expected positive int"})
  return tuple(invalid)


def _augment_report(report: EpochReport, result: TimingJoinResult) -> EpochReport:
  if result.ok:
    workload = dict(report.workload)
    workload["timing"] = copy.deepcopy(result.timing)
    workload.setdefault("timing_result_schema", SCHEMA)
    return replace(report, workload=workload)

  blockers = tuple(dict(blocker, source="timing_result") for blocker in result.blockers)
  return replace(report, blockers=(*report.blockers, *blockers))


def _blocker(missing: str, reason: str, **extra: Any) -> dict[str, Any]:
  blocker = {"missing": missing, "reason": reason}
  blocker.update(extra)
  return blocker


def _is_absent(value: Any) -> bool:
  return value is None or value == "" or value == [] or value == {}


def _is_nonnegative_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _string_or_none(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value)
  return text if text else None


__all__ = ["TimingJoinResult", "join_timing_result", "validate_timing_result"]
