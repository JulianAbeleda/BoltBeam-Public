"""Fail-closed admission of tinygrad KFD launch sidecars.

The producer owns observation: it writes JSON beside a tinygrad run.  This
module neither imports tinygrad nor opens KFD, so it is safe to use in CPU-only
analysis and cannot alter a parent runtime process.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from boltbeam.search.epochs.epoch_model import EpochReport
from boltbeam.search.joins.resource_join import ResourceJoinResult, validate_resource_snapshot
from boltbeam.search.joins.timing_join import TimingJoinResult, validate_timing_result


SCHEMA = "tinygrad.kfd_launch_sidecar.v1"
_HASH = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED = ("program_id", "source_sha256", "binary_sha256", "grid", "workgroup", "submit_ns", "complete_ns")


@dataclass(frozen=True)
class KfdObservationJoinResult:
  """Result of admitting one sidecar and its candidate-scoped evidence."""

  candidate_id: str | None
  records: tuple[Mapping[str, Any], ...] = ()
  missing: tuple[str, ...] = ()
  invalid: tuple[Mapping[str, Any], ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()
  timing: TimingJoinResult | None = None
  resources: ResourceJoinResult | None = None
  report: EpochReport | None = None

  @property
  def ok(self) -> bool:
    return (
      bool(self.records)
      and not self.missing
      and not self.invalid
      and not self.blockers
      and self.timing is not None
      and self.timing.ok
      and self.resources is not None
      and self.resources.ok
    )


def validate_kfd_launch_sidecar(
  sidecar: Mapping[str, Any], candidate_id: str | None = None,
) -> KfdObservationJoinResult:
  """Validate serialized direct-KFD observations without touching a runtime."""
  if not isinstance(sidecar, Mapping):
    return KfdObservationJoinResult(
      candidate_id=candidate_id,
      invalid=({"field": "sidecar", "reason": "expected mapping"},),
      blockers=(_blocker("invalid_kfd_sidecar", "KFD sidecar must be a mapping"),),
    )

  invalid: list[dict[str, Any]] = []
  missing: list[str] = []
  blockers: list[dict[str, Any]] = []
  declared_candidate = _string_or_none(sidecar.get("candidate_id"))
  if sidecar.get("schema") != SCHEMA:
    invalid.append({"field": "schema", "value": sidecar.get("schema"), "reason": f"expected {SCHEMA}"})
  if declared_candidate is None:
    missing.append("candidate_id")
  if candidate_id is not None and declared_candidate is not None and candidate_id != declared_candidate:
    blockers.append(_blocker("candidate_id_mismatch", "KFD sidecar candidate_id does not match join target",
                             expected=candidate_id, actual=declared_candidate))

  rows = sidecar.get("records")
  if not isinstance(rows, list | tuple) or not rows:
    missing.append("records")
  else:
    for index, record in enumerate(rows):
      if not isinstance(record, Mapping):
        invalid.append({"field": f"records[{index}]", "reason": "expected mapping"})
        continue
      row_candidate = _string_or_none(record.get("candidate_id"))
      if row_candidate is not None and declared_candidate is not None and row_candidate != declared_candidate:
        invalid.append({"field": f"records[{index}].candidate_id", "value": row_candidate,
                        "reason": "does not match sidecar candidate_id"})
      invalid.extend(_validate_record(record, index))

  if (missing or invalid) and not blockers:
    blockers.append(_blocker("invalid_kfd_sidecar", "KFD sidecar contains missing or invalid launch records"))
  joined_candidate = candidate_id or declared_candidate
  records = () if missing or invalid or blockers else tuple(copy.deepcopy(dict(row)) for row in rows)
  return KfdObservationJoinResult(joined_candidate, records, tuple(missing), tuple(invalid), tuple(blockers))


def join_kfd_launch_sidecar(
  sidecar: Mapping[str, Any], timing_result: Mapping[str, Any], resource_snapshot: Mapping[str, Any],
  target: EpochReport | str,
) -> KfdObservationJoinResult:
  """Atomically join direct-KFD observations with copied timing/resource evidence.

  No partial report augmentation occurs: a bad sidecar, timing result, or
  resource snapshot leaves the target workload untouched and records blockers.
  """
  target_id = target.candidate_id if isinstance(target, EpochReport) else str(target)
  observation = validate_kfd_launch_sidecar(sidecar, target_id)
  timing = validate_timing_result(timing_result, target_id)
  resources = validate_resource_snapshot(resource_snapshot, target_id)
  blockers = [*observation.blockers]
  blockers.extend(dict(blocker, source="timing_result") for blocker in timing.blockers)
  blockers.extend(dict(blocker, source="resource_snapshot") for blocker in resources.blockers)
  result = replace(observation, blockers=tuple(blockers), timing=timing, resources=resources)

  if isinstance(target, EpochReport):
    if result.ok:
      workload = dict(target.workload)
      workload["timing"] = copy.deepcopy(timing.timing)
      workload["resources"] = copy.deepcopy(resources.resources)
      workload["kfd_launch_sidecar"] = {"schema": SCHEMA, "records": copy.deepcopy(list(result.records))}
      workload.setdefault("timing_result_schema", "tinygrad.mmq_timing_result.v1")
      workload.setdefault("resource_trace_schema", "tinygrad.kernel_resource_trace.v1")
      report = replace(target, workload=workload)
    else:
      report = replace(target, blockers=(*target.blockers, *result.blockers))
    result = replace(result, report=report)
  return result


def _validate_record(record: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
  invalid: list[dict[str, Any]] = []
  prefix = f"records[{index}]"
  for field in _REQUIRED:
    if record.get(field) is None or record.get(field) == "":
      invalid.append({"field": f"{prefix}.{field}", "reason": "required"})
  program_id = record.get("program_id")
  if program_id is not None and (not isinstance(program_id, (str, int)) or isinstance(program_id, bool) or str(program_id) == ""):
    invalid.append({"field": f"{prefix}.program_id", "value": program_id, "reason": "expected non-empty string or int"})
  for field in ("source_sha256", "binary_sha256"):
    value = record.get(field)
    if value is not None and (not isinstance(value, str) or not _HASH.fullmatch(value)):
      invalid.append({"field": f"{prefix}.{field}", "value": value, "reason": "expected 64-character SHA-256 hex"})
  for field in ("grid", "workgroup"):
    if record.get(field) is not None and not _is_dim3(record[field]):
      invalid.append({"field": f"{prefix}.{field}", "value": record[field], "reason": "expected three positive ints"})
  submit, complete = record.get("submit_ns"), record.get("complete_ns")
  if submit is not None and not _is_nonnegative_number(submit):
    invalid.append({"field": f"{prefix}.submit_ns", "value": submit, "reason": "expected finite non-negative number"})
  if complete is not None and not _is_nonnegative_number(complete):
    invalid.append({"field": f"{prefix}.complete_ns", "value": complete, "reason": "expected finite non-negative number"})
  if _is_nonnegative_number(submit) and _is_nonnegative_number(complete) and complete < submit:
    invalid.append({"field": f"{prefix}.complete_ns", "value": complete, "reason": "must not precede submit_ns"})
  if "counters" in record:
    counters = record["counters"]
    if not isinstance(counters, Mapping):
      invalid.append({"field": f"{prefix}.counters", "value": counters, "reason": "expected mapping"})
    else:
      for name, value in counters.items():
        if not isinstance(name, str) or not name or not _is_nonnegative_number(value):
          invalid.append({"field": f"{prefix}.counters.{name}", "value": value,
                          "reason": "expected named finite non-negative number"})
  return invalid


def _is_dim3(value: Any) -> bool:
  return isinstance(value, list | tuple) and len(value) == 3 and all(
    isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value
  )


def _is_nonnegative_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _string_or_none(value: Any) -> str | None:
  if value is None:
    return None
  value = str(value)
  return value if value else None


def _blocker(missing: str, reason: str, **extra: Any) -> dict[str, Any]:
  blocker = {"missing": missing, "reason": reason}
  blocker.update(extra)
  return blocker


__all__ = ["KfdObservationJoinResult", "SCHEMA", "join_kfd_launch_sidecar", "validate_kfd_launch_sidecar"]
