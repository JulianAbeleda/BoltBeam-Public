"""Immutable MMQ epoch specs and coverage helpers.

This module is the E0 taxonomy layer for the MMQ epoch model. It intentionally
stays independent of tinygrad, hardware collectors, and emitted kernel formats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"

KNOWN_STATUSES = frozenset((PASS, FAIL, BLOCKED, UNKNOWN))
BLOCKER_STATUSES = frozenset((FAIL, BLOCKED, UNKNOWN))


@dataclass(frozen=True)
class EpochSpec:
  id: str
  kind: str
  required_events: tuple[str, ...]
  required_categories: tuple[str, ...]
  entry_conditions: tuple[str, ...]
  exit_conditions: tuple[str, ...]

  def to_dict(self) -> dict[str, object]:
    return {
      "id": self.id,
      "kind": self.kind,
      "required_events": list(self.required_events),
      "required_categories": list(self.required_categories),
      "entry_conditions": list(self.entry_conditions),
      "exit_conditions": list(self.exit_conditions),
    }


@dataclass(frozen=True)
class EpochEvent:
  epoch_id: str
  event_kind: str
  source_anchor: str | None = None
  instruction_mnemonic: str | None = None
  logical_identity: Mapping[str, Any] | None = None
  physical_identity: Mapping[str, Any] | None = None
  resources: Mapping[str, Any] | None = None
  status: str = UNKNOWN


@dataclass(frozen=True)
class EpochAnswer:
  epoch_id: str
  status: str = UNKNOWN
  events: tuple[EpochEvent, ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class EpochReport:
  candidate_id: str
  workload: Mapping[str, Any]
  epochs: tuple[EpochAnswer, ...]
  missing_epochs: tuple[str, ...] = ()
  blockers: tuple[Mapping[str, Any], ...] = ()


MMQ_EPOCH_SPECS: tuple[EpochSpec, ...] = (
  EpochSpec(
    id="launch_ownership",
    kind="launch_ownership",
    required_events=("owner_map",),
    required_categories=("tile_geometry", "work_ownership"),
    entry_conditions=("candidate launch geometry is known",),
    exit_conditions=("each output tile and fragment has exactly one owner",),
  ),
  EpochSpec(
    id="q4k_tile_x_load_decode",
    kind="q4k_tile_x_load_decode",
    required_events=("q4k_global_load", "q4k_decode", "tile_x_stage"),
    required_categories=("data_layout", "shared_memory_lifecycle", "resource_model"),
    entry_conditions=("Q4_K packed layout and address formula are known",),
    exit_conditions=("decoded tile_x values are staged for consumers",),
  ),
  EpochSpec(
    id="q8_tile_y_stage",
    kind="q8_tile_y_stage",
    required_events=("q8_global_load", "tile_y_stage"),
    required_categories=("data_layout", "shared_memory_lifecycle", "resource_model"),
    entry_conditions=("Q8_1 or DS4 activation panel layout is known",),
    exit_conditions=("tile_y panel is staged for reuse",),
  ),
  EpochSpec(
    id="visibility_sync",
    kind="visibility_sync",
    required_events=("stage_wait", "barrier"),
    required_categories=("sync_cadence", "shared_memory_lifecycle"),
    entry_conditions=("producer stage events are complete",),
    exit_conditions=("staged data is visible to consuming dot work",),
  ),
  EpochSpec(
    id="dot_accumulate",
    kind="dot_accumulate",
    required_events=("dot_primitive", "accumulator_write"),
    required_categories=("dot_primitive", "accumulator_mapping"),
    entry_conditions=("staged operands are visible and accumulator identity is known",),
    exit_conditions=("dot work updates the expected accumulator slots",),
  ),
  EpochSpec(
    id="k_advance",
    kind="k_advance",
    required_events=("k_step", "accumulator_carry"),
    required_categories=("k_loop_cadence", "accumulator_mapping"),
    entry_conditions=("current K panel has been consumed",),
    exit_conditions=("next K panel is selected without losing accumulator identity",),
  ),
  EpochSpec(
    id="stage_reuse_or_overwrite",
    kind="stage_reuse_or_overwrite",
    required_events=("reuse_claim", "overwrite_boundary"),
    required_categories=("shared_memory_lifecycle", "k_loop_cadence", "sync_cadence"),
    entry_conditions=("staged tile lifetime and consumers are known",),
    exit_conditions=("staged data is either reused or safely invalidated before overwrite",),
  ),
  EpochSpec(
    id="writeback",
    kind="writeback",
    required_events=("global_store", "store_owner"),
    required_categories=("store_path", "work_ownership", "accumulator_mapping"),
    entry_conditions=("final accumulator values and owner identity are known",),
    exit_conditions=("each output element is stored by exactly one owner",),
  ),
  EpochSpec(
    id="epilogue",
    kind="epilogue",
    required_events=("final_wait", "kernel_exit"),
    required_categories=("sync_cadence", "stop_criteria"),
    entry_conditions=("writeback work has been issued",),
    exit_conditions=("outstanding stores are closed and the kernel exits",),
  ),
)


def mmq_epoch_specs() -> tuple[EpochSpec, ...]:
  return MMQ_EPOCH_SPECS


def mmq_epoch_ids() -> tuple[str, ...]:
  return tuple(spec.id for spec in MMQ_EPOCH_SPECS)


def epoch_spec_by_id(epoch_id: str) -> EpochSpec:
  for spec in MMQ_EPOCH_SPECS:
    if spec.id == epoch_id:
      return spec
  raise ValueError(f"unknown MMQ epoch {epoch_id!r}; expected one of {mmq_epoch_ids()}")


def epoch_coverage(report: EpochReport | Mapping[str, Any]) -> dict[str, list[str]]:
  """Return missing/unknown/blocked epoch ids for an answer map or report."""
  answers = _epoch_answers(report)
  known = set(mmq_epoch_ids())
  answered = {epoch_id for epoch_id, answer in answers.items() if _is_answered(answer)}
  missing = [epoch_id for epoch_id in mmq_epoch_ids() if epoch_id not in answered]
  unknown = sorted(epoch_id for epoch_id in answers if epoch_id not in known)
  blocked = [epoch_id for epoch_id, answer in answers.items() if epoch_id in known and _is_blocked(answer)]
  return {"missing": missing, "unknown": unknown, "blocked": blocked}


def validate_epoch_report(report: EpochReport | Mapping[str, Any]) -> dict[str, list[str]]:
  return epoch_coverage(report)


def blocked_epoch_reasons(report: EpochReport | Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
  reasons: list[dict[str, Any]] = []
  if isinstance(report, EpochReport):
    reasons.extend(dict(blocker) for blocker in report.blockers)
    epoch_iter = report.epochs
  else:
    reasons.extend(dict(blocker) for blocker in _as_tuple(report.get("blockers", ())))
    epoch_iter = _as_tuple(report.get("epochs", ()))

  for answer in epoch_iter:
    epoch_id = _get(answer, "epoch_id") or _get(answer, "id")
    status = _get(answer, "status")
    for blocker in _as_tuple(_get(answer, "blockers", ())):
      reason = dict(blocker) if isinstance(blocker, Mapping) else {"reason": str(blocker)}
      if epoch_id is not None:
        reason.setdefault("epoch_id", epoch_id)
      reasons.append(reason)
    if status in BLOCKER_STATUSES and not _as_tuple(_get(answer, "blockers", ())):
      reason = {"epoch_id": epoch_id, "status": status}
      missing = _get(answer, "missing")
      if missing:
        reason["missing"] = missing
      reasons.append(reason)
  return tuple(reasons)


def _epoch_answers(report: EpochReport | Mapping[str, Any]) -> dict[str, Any]:
  if isinstance(report, EpochReport):
    answers = {answer.epoch_id: answer for answer in report.epochs}
    for epoch_id in report.missing_epochs:
      answers.setdefault(epoch_id, None)
    return answers

  if "epochs" in report:
    epochs = report["epochs"]
    if isinstance(epochs, Mapping):
      return dict(epochs)
    return {
      _get(answer, "epoch_id") or _get(answer, "id"): answer
      for answer in _as_tuple(epochs)
      if _get(answer, "epoch_id") or _get(answer, "id")
    }
  return dict(report)


def _is_answered(answer: Any) -> bool:
  if answer in (None, "", [], {}, ()):
    return False
  status = _get(answer, "status")
  return status not in (None, "", UNKNOWN)


def _is_blocked(answer: Any) -> bool:
  status = _get(answer, "status")
  return status in (FAIL, BLOCKED) or bool(_as_tuple(_get(answer, "blockers", ())))


def _get(value: Any, key: str, default: Any = None) -> Any:
  if isinstance(value, Mapping):
    return value.get(key, default)
  return getattr(value, key, default)


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if value in (None, ""):
    return ()
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)
