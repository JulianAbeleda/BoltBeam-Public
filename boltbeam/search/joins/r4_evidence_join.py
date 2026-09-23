"""Pure R4 evidence joins for MMQ epoch reports.

The input is a structural ``tinygrad.mmq_owner_coverage.v1`` or
``tinygrad.mmq_staging_evidence.v1``-style mapping. This module intentionally
does not import tinygrad; it only validates and joins evidence that has already
been serialized.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from boltbeam.search.epochs.epoch_model import BLOCKED, FAIL, PASS, EpochReport


OWNER_COVERAGE_SCHEMA = "tinygrad.mmq_owner_coverage.v1"
STAGING_EVIDENCE_SCHEMA = "tinygrad.mmq_staging_evidence.v1"
SCHEMAS = frozenset((OWNER_COVERAGE_SCHEMA, STAGING_EVIDENCE_SCHEMA))
STATUSES = frozenset((PASS, FAIL, BLOCKED))
SHAPE_FIELDS = frozenset(("M", "N", "K"))
EVIDENCE_FIELDS = (
  "schema",
  "evidence_kind",
  "candidate_id",
  "backend",
  "shape",
  "production_dispatch_changed",
  "status",
  "exact_blocker",
  "research_only",
  "oracle_source",
  "oracle_backend",
  "expected_stores",
  "observed_stores",
  "duplicate_store_summary",
  "missing_store_summary",
  "q4k_tile_staging",
  "q8_1_ds4_staging",
  "sum_slot_map",
  "owner_coverage",
  "staging",
  "sum_slots",
  "metadata",
  "notes",
)
TOP_LEVEL_EVIDENCE_FIELDS = frozenset(EVIDENCE_FIELDS) - frozenset(("schema", "candidate_id", "evidence_kind"))


@dataclass(frozen=True)
class R4EvidenceJoinResult:
  candidate_id: str | None
  evidence_key: str | None
  evidence: Mapping[str, Any] | None
  missing: tuple[str, ...] = ()
  invalid: tuple[Mapping[str, Any], ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()
  report: EpochReport | None = None

  @property
  def ok(self) -> bool:
    return not self.missing and not self.invalid and not self.blockers and self.evidence is not None


def validate_r4_evidence(
  artifact: Mapping[str, Any],
  candidate_id: str | None = None,
) -> R4EvidenceJoinResult:
  """Validate and extract candidate-level R4 evidence."""
  invalid: list[dict[str, Any]] = []
  missing: list[str] = []
  blockers: list[dict[str, Any]] = []

  if not isinstance(artifact, Mapping):
    return R4EvidenceJoinResult(
      candidate_id=candidate_id,
      evidence_key=None,
      evidence=None,
      invalid=({"field": "artifact", "reason": "expected mapping"},),
      blockers=(_blocker("invalid_r4_evidence", "R4 evidence artifact must be a mapping"),),
    )

  artifact_candidate_id = _string_or_none(artifact.get("candidate_id"))
  if candidate_id is not None and artifact_candidate_id is not None and artifact_candidate_id != candidate_id:
    blockers.append(
      _blocker(
        "candidate_id_mismatch",
        "R4 evidence candidate_id does not match join target",
        expected=candidate_id,
        actual=artifact_candidate_id,
      )
    )

  joined_candidate_id = candidate_id or artifact_candidate_id
  evidence = _candidate_evidence(artifact, joined_candidate_id)
  if evidence is None:
    missing.append("evidence")
    blockers.append(_blocker("evidence", "R4 evidence artifact has no candidate evidence"))
  else:
    evidence_missing, evidence_invalid = _validate_evidence(evidence)
    missing.extend(evidence_missing)
    invalid.extend(evidence_invalid)

  if (missing or invalid) and not blockers:
    blockers.append(_blocker("invalid_r4_evidence", "R4 evidence contains missing or invalid fields"))

  evidence_key = _evidence_key(evidence) if evidence is not None else None
  return R4EvidenceJoinResult(
    candidate_id=joined_candidate_id,
    evidence_key=evidence_key,
    evidence=copy.deepcopy(evidence) if evidence is not None else None,
    missing=tuple(missing),
    invalid=tuple(invalid),
    blockers=tuple(blockers),
  )


def join_r4_evidence(
  artifact: Mapping[str, Any],
  target: EpochReport | str,
) -> R4EvidenceJoinResult:
  """Join R4 evidence to an ``EpochReport`` or validate it for a candidate id.

  Valid joins augment ``EpochReport.workload["r4_evidence"]``. Missing or
  invalid evidence leaves workload unchanged and appends structured blockers to
  the report.
  """
  if isinstance(target, EpochReport):
    joined = validate_r4_evidence(artifact, target.candidate_id)
    report = _augment_report(target, joined)
    return replace(joined, report=report)
  return validate_r4_evidence(artifact, str(target))


def _candidate_evidence(artifact: Mapping[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
  if any(field in artifact for field in TOP_LEVEL_EVIDENCE_FIELDS):
    return _copy_evidence_fields(artifact)

  rows = artifact.get("rows")
  if not isinstance(rows, list | tuple):
    return None

  for row in rows:
    if not isinstance(row, Mapping):
      continue
    explicit_row_candidate = _string_or_none(row.get("candidate_id"))
    row_candidate = explicit_row_candidate or _string_or_none(artifact.get("candidate_id"))
    if candidate_id is not None and row_candidate is not None and row_candidate != candidate_id:
      continue
    copied = _copy_evidence_fields(row)
    if copied or explicit_row_candidate is not None:
      copied.setdefault("schema", artifact.get("schema"))
      copied.setdefault("candidate_id", row_candidate)
      return copied
  return None


def _copy_evidence_fields(evidence: Mapping[str, Any]) -> dict[str, Any]:
  return {key: copy.deepcopy(evidence[key]) for key in EVIDENCE_FIELDS if key in evidence}


def _validate_evidence(evidence: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
  missing = tuple(field for field in ("schema", "candidate_id", "shape", "production_dispatch_changed", "status") if _is_absent(evidence.get(field)))
  invalid: list[dict[str, Any]] = []

  schema = evidence.get("schema")
  if schema is not None and schema not in SCHEMAS:
    invalid.append({"field": "schema", "value": schema, "reason": f"expected one of {sorted(SCHEMAS)}"})

  candidate_id = evidence.get("candidate_id")
  if candidate_id is not None and not _string_or_none(candidate_id):
    invalid.append({"field": "candidate_id", "value": candidate_id, "reason": "expected non-empty string"})

  backend = evidence.get("backend")
  if backend is not None and not _string_or_none(backend):
    invalid.append({"field": "backend", "value": backend, "reason": "expected non-empty string"})

  shape = evidence.get("shape")
  if shape is not None:
    invalid.extend(_validate_shape(shape))

  production_dispatch_changed = evidence.get("production_dispatch_changed")
  if production_dispatch_changed is not None and production_dispatch_changed is not False:
    invalid.append(
      {
        "field": "production_dispatch_changed",
        "value": production_dispatch_changed,
        "reason": "expected False; production route claims are not accepted as R4 evidence",
      }
    )

  status = evidence.get("status")
  if status is not None and status not in STATUSES:
    invalid.append({"field": "status", "value": status, "reason": f"expected one of {sorted(STATUSES)}"})

  exact_blocker = evidence.get("exact_blocker")
  if status in (FAIL, BLOCKED) and _is_absent(exact_blocker):
    missing += ("exact_blocker",)

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


def _augment_report(report: EpochReport, result: R4EvidenceJoinResult) -> EpochReport:
  if result.ok:
    workload = dict(report.workload)
    joined = dict(workload.get("r4_evidence", {}))
    joined[result.evidence_key] = copy.deepcopy(result.evidence)
    workload["r4_evidence"] = joined
    return replace(report, workload=workload)

  blockers = tuple(dict(blocker, source="r4_evidence") for blocker in result.blockers)
  return replace(report, blockers=(*report.blockers, *blockers))


def _evidence_key(evidence: Mapping[str, Any]) -> str:
  return _string_or_none(evidence.get("evidence_kind")) or str(evidence.get("schema"))


def _blocker(missing: str, reason: str, **extra: Any) -> dict[str, Any]:
  blocker = {"missing": missing, "reason": reason}
  blocker.update(extra)
  return blocker


def _is_absent(value: Any) -> bool:
  return value is None or value == "" or value == [] or value == {}


def _string_or_none(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value)
  return text if text else None


__all__ = ["R4EvidenceJoinResult", "join_r4_evidence", "validate_r4_evidence"]
