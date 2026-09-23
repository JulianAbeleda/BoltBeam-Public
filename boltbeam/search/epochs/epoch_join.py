"""Pure joins from tinygrad AMD ISA proof manifests to MMQ epoch reports."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from boltbeam.search.epochs.epoch_model import BLOCKED, PASS, UNKNOWN, EpochAnswer, EpochEvent, EpochReport, mmq_epoch_ids
from boltbeam.search.mmq.mmq_epoch_oracle import llama_mmq_epoch_sequence


Row = Mapping[str, Any]


def amd_isa_manifest_epoch_report(
  manifest: Mapping[str, Any],
  candidate_metadata: Mapping[str, Any] | None = None,
  workload: Mapping[str, Any] | None = None,
) -> EpochReport:
  """Build an MMQ ``EpochReport`` from a tinygrad-like AMD ISA proof manifest.

  This is intentionally a structural, pure-Python join. It does not import
  tinygrad, inspect hardware, or treat heuristics as promotion-grade proof.
  """
  metadata = dict(candidate_metadata or {})
  workload_map = _normalize_workload(workload or metadata.get("workload") or manifest.get("workload") or {})
  candidate_id = str(metadata.get("candidate_id") or manifest.get("candidate_id") or "unknown")
  rows = tuple(row for row in _as_tuple(manifest.get("rows", ())) if isinstance(row, Mapping))

  epoch_ids = _epoch_ids_for_workload(workload_map)
  classifiers: dict[str, Callable[[tuple[Row, ...], Mapping[str, Any]], tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]]] = {
    "launch_ownership": _classify_launch_ownership,
    "q4k_tile_x_load_decode": _classify_q4k_tile_x_load_decode,
    "q8_tile_y_stage": _classify_q8_tile_y_stage,
    "visibility_sync": _classify_visibility_sync,
    "dot_accumulate": _classify_dot_accumulate,
    "k_advance": _classify_k_advance,
    "stage_reuse_or_overwrite": _classify_stage_reuse_or_overwrite,
    "writeback": _classify_writeback,
    "epilogue": _classify_epilogue,
  }

  answers: list[EpochAnswer] = []
  report_blockers: list[dict[str, Any]] = []
  for epoch_id in epoch_ids:
    status, evidence, blockers = classifiers.get(epoch_id, _classify_unknown)(rows, metadata)
    blockers = tuple(dict(blocker, epoch_id=epoch_id) if "epoch_id" not in blocker else blocker for blocker in blockers)
    events = tuple(_event_from_row(epoch_id, row, status) for row in evidence)
    answers.append(EpochAnswer(epoch_id=epoch_id, status=status, events=events, blockers=blockers))
    report_blockers.extend(blockers)

  extra_epoch_ids = sorted(
    {
      str(row.get("epoch_id"))
      for row in rows
      if row.get("epoch_id") and str(row.get("epoch_id")) not in set(epoch_ids)
    }
  )
  for epoch_id in extra_epoch_ids:
    evidence = tuple(row for row in rows if str(row.get("epoch_id")) == epoch_id)
    answers.append(
      EpochAnswer(
        epoch_id=epoch_id,
        status=UNKNOWN,
        events=tuple(_event_from_row(epoch_id, row, UNKNOWN) for row in evidence),
        blockers=({"epoch_id": epoch_id, "status": UNKNOWN, "missing": "known MMQ epoch id"},),
      )
    )

  known_answered = {answer.epoch_id for answer in answers if answer.epoch_id in set(mmq_epoch_ids())}
  missing_epochs = tuple(epoch_id for epoch_id in epoch_ids if epoch_id not in known_answered)
  return EpochReport(
    candidate_id=candidate_id,
    workload=workload_map,
    epochs=tuple(answers),
    missing_epochs=missing_epochs,
    blockers=tuple(report_blockers),
  )


def _epoch_ids_for_workload(workload: Mapping[str, Any]) -> tuple[str, ...]:
  try:
    return llama_mmq_epoch_sequence(int(workload["m"]), int(workload["n"]), int(workload["k"])).epoch_ids
  except Exception:
    return mmq_epoch_ids()


def _classify_launch_ownership(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  if metadata.get("owner_map_pass") is True:
    return PASS, (), ()
  evidence = _matching_rows(rows, lambda row: "owner" in _row_text(row))
  if evidence:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("owner_map_pass", "candidate metadata owner_map_pass=true or owner-map row"),)


def _classify_q4k_tile_x_load_decode(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  q4_rows = _matching_rows(rows, lambda row: _mentions(row, ("q4", "q4_k", "q4k", "tile_x")))
  has_global = any(_is_kind(row, "global_load") for row in q4_rows)
  has_stage = any(_is_kind(row, "ds_store", "ds_load") or "decode" in _row_text(row) for row in q4_rows)
  if has_global and has_stage:
    return PASS, q4_rows, ()
  return BLOCKED, q4_rows, (_blocker("q4k_global_load_decode_stage", "Q4_K global load/decode/stage rows"),)


def _classify_q8_tile_y_stage(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  q8_rows = _matching_rows(rows, lambda row: _mentions(row, ("q8", "q8_1", "ds4", "tile_y")))
  has_load = any(_is_kind(row, "global_load", "ds_load") for row in q8_rows)
  has_stage = any(_is_kind(row, "ds_store", "ds_load") for row in q8_rows)
  if has_load and has_stage:
    return PASS, q8_rows, ()
  return BLOCKED, q8_rows, (_blocker("q8_tile_y_stage", "Q8/DS4 global or LDS stage rows"),)


def _classify_visibility_sync(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  evidence = _matching_rows(rows, lambda row: _is_kind(row, "barrier", "waitcnt"))
  if evidence:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("visibility_sync", "barrier or waitcnt row"),)


def _classify_dot_accumulate(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  dot_rows = _matching_rows(rows, lambda row: _is_kind(row, "wmma", "dot") or "wmma" in _row_text(row) or "dot" in _row_text(row))
  accum_rows = _matching_rows(rows, lambda row: _is_kind(row, "accum_read", "accum_write") or "accum" in _row_text(row))
  if dot_rows and accum_rows:
    return PASS, _unique_rows((*dot_rows, *accum_rows)), ()
  return BLOCKED, _unique_rows((*dot_rows, *accum_rows)), (_blocker("dot_accumulate", "wmma or dot row plus accumulator identity rows"),)


def _classify_k_advance(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  evidence = _matching_rows(rows, lambda row: _mentions(row, ("k_advance", "k_step", "k_loop", "accumulator_carry")))
  if evidence or metadata.get("bounded_single_panel") is True:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("k_advance", "K loop/carry row or bounded_single_panel metadata"),)


def _classify_stage_reuse_or_overwrite(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  evidence = _matching_rows(rows, lambda row: _mentions(row, ("reuse", "overwrite", "stage_boundary")))
  if evidence:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("stage_reuse_or_overwrite", "reuse or overwrite boundary row"),)


def _classify_writeback(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  evidence = _matching_rows(rows, lambda row: _is_kind(row, "global_store"))
  if evidence:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("writeback", "global_store row"),)


def _classify_epilogue(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del metadata
  evidence = _matching_rows(rows, lambda row: _is_kind(row, "waitcnt", "kernel_exit") or "kernel_exit" in _row_text(row))
  if evidence:
    return PASS, evidence, ()
  return BLOCKED, (), (_blocker("epilogue", "waitcnt or kernel_exit row"),)


def _classify_unknown(rows: tuple[Row, ...], metadata: Mapping[str, Any]) -> tuple[str, tuple[Row, ...], tuple[dict[str, Any], ...]]:
  del rows, metadata
  return UNKNOWN, (), (_blocker("unknown_epoch", "known epoch classifier"),)


def _event_from_row(epoch_id: str, row: Row, status: str) -> EpochEvent:
  return EpochEvent(
    epoch_id=epoch_id,
    event_kind=str(row.get("kind") or row.get("logical_op") or "unknown"),
    source_anchor=_source_anchor(row),
    instruction_mnemonic=_instruction_mnemonic(row),
    logical_identity=_mapping_or_none(row.get("logical_identity")),
    physical_identity=_mapping_or_none(row.get("physical_identity")),
    resources=_mapping_or_none(row.get("resources")),
    status=status,
  )


def _blocker(missing: str, reason: str) -> dict[str, Any]:
  return {"status": BLOCKED, "missing": missing, "reason": reason}


def _matching_rows(rows: tuple[Row, ...], predicate: Callable[[Row], bool]) -> tuple[Row, ...]:
  return tuple(row for row in rows if predicate(row))


def _unique_rows(rows: tuple[Row, ...]) -> tuple[Row, ...]:
  unique: list[Row] = []
  seen: set[int] = set()
  for row in rows:
    row_id = id(row)
    if row_id not in seen:
      unique.append(row)
      seen.add(row_id)
  return tuple(unique)


def _is_kind(row: Row, *needles: str) -> bool:
  values = (str(row.get("kind", "")), str(row.get("logical_op", "")))
  normalized = tuple(value.lower().replace("-", "_") for value in values)
  return any(any(needle in value for value in normalized) for needle in needles)


def _mentions(row: Row, needles: tuple[str, ...]) -> bool:
  text = _row_text(row)
  return any(needle in text for needle in needles)


def _row_text(row: Row) -> str:
  parts = [
    row.get("kind"),
    row.get("logical_op"),
    row.get("emitted"),
    row.get("epoch_id"),
    row.get("source_anchor"),
    row.get("logical_identity"),
    row.get("physical_identity"),
    row.get("resources"),
  ]
  return " ".join(str(part).lower().replace("-", "_") for part in parts if part is not None)


def _source_anchor(row: Row) -> str | None:
  value = row.get("source_anchor") or row.get("epoch_id")
  return str(value) if value is not None else None


def _instruction_mnemonic(row: Row) -> str | None:
  emitted = row.get("emitted")
  if emitted is None:
    return None
  text = str(emitted).strip()
  return text.split(None, 1)[0] if text else None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
  return value if isinstance(value, Mapping) else None


def _normalize_workload(workload: Mapping[str, Any]) -> dict[str, Any]:
  return dict(workload)


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if value in (None, ""):
    return ()
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


__all__ = ["amd_isa_manifest_epoch_report"]
