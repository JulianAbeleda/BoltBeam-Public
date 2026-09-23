"""Measured cohort ranking for operand-path candidates.

Prediction never enters this module. Only correctness-eligible, identity-comparable
timings from one session can produce a winner; overlapping uncertainty remains
inconclusive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from boltbeam.kernel_analysis.contrast import timing_interval
from boltbeam.kernel_analysis.eligibility import performance_eligibility
from boltbeam.kernel_analysis.model import KernelEvidence


@dataclass(frozen=True)
class MeasuredCandidateRank:
  candidate_id: str
  eligible: bool
  median_ms: float | None
  interval_ms: tuple[float, float] | None
  blockers: tuple[str, ...] = ()

  def to_json(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "eligible": self.eligible, "median_ms": self.median_ms,
            "interval_ms": list(self.interval_ms) if self.interval_ms is not None else None,
            "blockers": list(self.blockers)}


@dataclass(frozen=True)
class MeasuredMatrixRanking:
  status: str
  baseline_id: str
  winner_id: str | None
  session_id: str | None
  rows: tuple[MeasuredCandidateRank, ...]
  selection_confidence: str
  mechanism_confidence: str = "unknown"
  blockers: tuple[str, ...] = ()

  def __post_init__(self):
    if self.status not in {"promote", "retain", "inconclusive", "unsupported", "blocked"}:
      raise ValueError("invalid measured matrix status")

  def to_json(self) -> dict[str, Any]:
    return {"schema": "boltbeam.measured_operand_matrix_ranking.v1", "status": self.status,
            "baseline_id": self.baseline_id, "winner_id": self.winner_id, "session_id": self.session_id,
            "selection_confidence": self.selection_confidence, "mechanism_confidence": self.mechanism_confidence,
            "rows": [row.to_json() for row in self.rows], "blockers": list(self.blockers)}


def rank_measured_matrix(evidence: Iterable[KernelEvidence], *, baseline_id: str) -> MeasuredMatrixRanking:
  """Rank one measured cohort, excluding invalid rows but never hiding why they were excluded."""
  cohort = tuple(evidence)
  by_id = {row.candidate.candidate_id: row for row in cohort}
  if not baseline_id or baseline_id not in by_id: raise ValueError("baseline_id must identify one cohort candidate")
  if len(by_id) != len(cohort): raise ValueError("candidate ids must be unique")
  baseline = by_id[baseline_id]
  ranked: list[MeasuredCandidateRank] = []
  for item in cohort:
    result = performance_eligibility(item, None if item is baseline else baseline)
    timing = item.timing
    interval = timing_interval(tuple(timing.normalized_samples_ms)) if result.eligible and timing is not None else None
    ranked.append(MeasuredCandidateRank(item.candidate.candidate_id, result.eligible,
      interval["median"] if interval else None, (interval["low"], interval["high"]) if interval else None,
      tuple(blocker.requirement_id for blocker in result.blockers)))
  valid = sorted((row for row in ranked if row.eligible and row.interval_ms is not None),
                 key=lambda row: (row.median_ms, row.candidate_id))
  ranked_rows = tuple(sorted(ranked, key=lambda row: (not row.eligible,
    row.median_ms if row.median_ms is not None else float("inf"), row.candidate_id)))
  session = baseline.identity.session_id
  if len(valid) < 2:
    unsupported = bool(ranked) and all(by_id[row.candidate_id].stages.compile.status == "unsupported" for row in ranked)
    return MeasuredMatrixRanking("unsupported" if unsupported else "blocked", baseline_id, None, session,
      ranked_rows, "unknown", blockers=tuple(sorted({b for row in ranked for b in row.blockers})))
  first, second = valid[0], valid[1]
  if first.interval_ms[1] >= second.interval_ms[0]:
    return MeasuredMatrixRanking("inconclusive", baseline_id, None, session, ranked_rows, "medium",
      blockers=("top_candidate_intervals_overlap",))
  status = "retain" if first.candidate_id == baseline_id else "promote"
  return MeasuredMatrixRanking(status, baseline_id, first.candidate_id, session, ranked_rows, "high")


__all__ = ["MeasuredCandidateRank", "MeasuredMatrixRanking", "rank_measured_matrix"]
