"""FutureSight: static rejection and ordering of a candidate population, before anything runs.

CPU-only and policy-free. Candidates arrive already built and hashed by BoltBeam;
this module reads their schedule and their supplied ``candidate_hash`` and never
derives, normalizes, expands or measures. Its ordering is a measurement order,
never a verdict. Target facts are read through BubbleBeam's reader, so both sides
judge against the same facts.

Ported from tinygrad-arkey ``extra/llm_research/bubblebeam_futuresight.py``
(35459866b) with the same names and behaviour. The flash builders
(``build_flash_legality``, ``build_flash_static_priority``) read flash
candidates only through BoltBeam's schema
(``boltbeam.search.flash_decode_candidate``), which owns their geometry,
identity and legality rules.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from boltbeam.search.bubblebeam import JSONValue, _legal_transform_sequence, _positive_int, _static_facts
from boltbeam.search.flash_decode_candidate import (
  FLASH_DECODE_CANDIDATE_SCHEMA_VERSION, FlashDecodeCandidate, flash_legality_violations,
)
from boltbeam.search.futuresight_evidence import ASSESSMENT_VERSION
from boltbeam.search.target_fact_keys import read_target_limits


CanonicalCandidate = Mapping[str, Any]


@dataclass(frozen=True)
class StaticRejection:
  candidate_hash: str
  reason: str


@dataclass(frozen=True)
class StaticAssessment:
  candidate_hash: str
  score: int
  reason: str


Legality = Callable[[CanonicalCandidate], str | None]
Priority = Callable[[CanonicalCandidate], tuple[int, str]]


def _at_path(value: Mapping[str, Any], path: str) -> Any:
  current: Any = value
  for part in path.split("."):
    if not isinstance(current, Mapping) or part not in current: return None
    current = current[part]
  return current


def build_static_legality(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> Legality:
  """Build a pure generic legality check from the same supplied live facts."""
  facts = _static_facts(workload_facts, target_facts)
  shape, vocabulary, supported = facts.shape, facts.vocabulary, facts.vocabulary.transforms

  def check(candidate: CanonicalCandidate) -> str | None:
    schedule = candidate.get("schedule", {})
    plan_kind = schedule.get("plan_kind") if isinstance(schedule, Mapping) else None
    if vocabulary.plan_kinds and plan_kind not in vocabulary.plan_kinds: return "unsupported_plan_kind"
    transforms = schedule.get("transforms") if isinstance(schedule, Mapping) else None
    if not _legal_transform_sequence(transforms, supported): return "unsupported_or_invalid_transform"
    if vocabulary.generic_control_plan_kind == plan_kind and transforms: return "heuristic_control_has_explicit_transforms"
    threads = _at_path(candidate, "schedule.launch.threads")
    if not _positive_int(threads): return "non_positive_threads"
    if facts.max_threads is not None and threads > facts.max_threads: return "over_threads"
    for axis in ("m", "n", "k"):
      tile, extent = _at_path(candidate, f"schedule.tile.{axis}"), shape.get(axis)
      if not _positive_int(tile): return f"non_positive_tile_{axis}"
      if _positive_int(extent) and extent % tile: return f"non_divisible_tile_{axis}"
    local_limit = _at_path(candidate, "static_constraints.max_local_memory_bytes")
    if local_limit is not None and not _positive_int(local_limit): return "non_positive_local_memory"
    if facts.max_local_memory is not None and _positive_int(local_limit) and local_limit > facts.max_local_memory: return "over_local_memory"
    return None
  return check


def build_static_priority(preferences: Mapping[str, Sequence[JSONValue]]) -> Priority:
  """Build deterministic preference scoring without target or backend policy."""
  for path, values in preferences.items():
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
      raise ValueError(f"priority preferences for {path!r} must be a non-empty sequence")
  ordered = tuple((path, tuple(values)) for path, values in sorted(preferences.items()))

  def priority(candidate: CanonicalCandidate) -> tuple[int, str]:
    score, matched = 0, []
    for path, values in ordered:
      actual = _at_path(candidate, path)
      try: index = next(index for index, value in enumerate(values) if actual == value)
      except StopIteration: continue
      score += len(values) - index
      matched.append(path)
    return score, "preferred:" + ",".join(matched) if matched else "no_static_preference"
  return priority


def _flash_candidate(candidate: CanonicalCandidate) -> FlashDecodeCandidate | str:
  """The schema's reading of one flash envelope, or the rejection reason when there is none."""
  if candidate.get("schema_version") != FLASH_DECODE_CANDIDATE_SCHEMA_VERSION: return "unsupported_schema_version"
  try: return FlashDecodeCandidate.from_envelope(dict(candidate))
  except ValueError: return "invalid_flash_geometry"


def build_flash_legality(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> Legality:
  """Build flash decode legality from supplied target facts only.

  The rules are the schema's (``flash_legality_violations``); this adapts them to
  one rejection reason per candidate, the first violated rule. The subgroup rule
  reads the candidate's own target, a missing thread or memory limit is a
  ``missing_fact:<key>`` rejection, and a present but invalid fact raises here.
  """
  facts = dict(target_facts)
  read_target_limits(facts)

  def check(candidate: CanonicalCandidate) -> str | None:
    flash = _flash_candidate(candidate)
    if isinstance(flash, str): return flash
    violations = flash_legality_violations(flash, facts)
    return violations[0][0] if violations else None
  return check


def build_flash_static_priority(target_facts: Mapping[str, Any]) -> Priority:
  """Cold-L2-aware static ordering for flash candidates, derived from facts and geometry.

  The estimate prefers wider column parallelism capped at the physical subgroup
  size, shorter shuffle ladders, and KV staging that keeps the score resident in
  local memory, while penalizing small ``stage_width`` values that force more
  cooperative staging passes (each pass is a cold-L2 relaunch). The subgroup
  size is the candidate's own ``target.subgroup_size``, the source legality
  uses. Residency uses the largest single launch, since launches are held to
  the limit one at a time. None of the weights encode a vendor win table;
  rank_static_candidates breaks score ties deterministically by candidate_hash.
  """
  limits = read_target_limits(target_facts)

  def priority(candidate: CanonicalCandidate) -> tuple[int, str]:
    flash = _flash_candidate(candidate)
    if flash == "unsupported_schema_version": return 0, "flash_candidate_required"
    if isinstance(flash, str): return 0, flash
    tile = flash.tile
    column = min(tile.group_width, flash.target["subgroup_size"])
    ladder = tile.reduce_stages
    local = max(flash.launch_local_memory_bytes(limits.reserved_local_memory).values())
    resident = 2 if limits.max_local_memory is not None and local * 2 <= limits.max_local_memory else 1
    hot = 1 if tile.staging == "KV_BOTH" else 0
    relaunch = max(0, (tile.threads // tile.stage_width - 1).bit_length())
    score = column - ladder + resident + hot - relaunch
    return score, f"flash:column={column},ladder={ladder},resident={resident},hot={hot},relaunch={relaunch}"
  return priority


def _candidate_hash(candidate: CanonicalCandidate) -> str:
  """Read BoltBeam's identity without deriving or normalizing it locally.

  The hash is the identity. A schedule is required to be a mapping only when
  present, because some candidate schemas carry none.
  """
  if not isinstance(candidate.get("schema_version"), str) or not candidate["schema_version"]:
    raise ValueError("canonical candidate requires schema_version")
  if not isinstance(candidate.get("candidate_hash"), str) or not candidate["candidate_hash"]:
    raise ValueError("canonical candidate requires candidate_hash")
  if candidate.get("schedule") is not None and not isinstance(candidate.get("schedule"), Mapping):
    raise ValueError("canonical candidate schedule must be a mapping when present")
  return candidate["candidate_hash"]


def classify_candidates(candidates: Iterable[CanonicalCandidate], legalities: Iterable[Legality] = ()) -> tuple[list[CanonicalCandidate], list[StaticRejection]]:
  """Classify an authoritative population without changing or expanding it."""
  accepted, rejected = [], []
  predicates = tuple(legalities)
  for candidate in candidates:
    candidate_hash = _candidate_hash(candidate)
    reason = next((result for check in predicates if (result := check(candidate)) is not None), None)
    (rejected if reason is not None else accepted).append(StaticRejection(candidate_hash, reason) if reason is not None else candidate)
  return accepted, rejected


def apply_coupled_row(baseline: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
  """Apply dotted schedule fields to a JSON baseline for static row legality."""
  value = json.loads(json.dumps(baseline))
  for path, replacement in sorted(row.items()):
    current = value
    parts = path.split(".")
    if not isinstance(path, str) or not parts or any(not part for part in parts): raise ValueError("invalid coupled row path")
    for part in parts[:-1]:
      if not isinstance(current, dict) or part not in current: raise ValueError("coupled row path does not exist")
      current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current: raise ValueError("coupled row path does not exist")
    current[parts[-1]] = replacement
  return value


def classify_coupled_rows(baseline: Mapping[str, Any], proposed_rows: Iterable[Mapping[str, Any]], workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  check = build_static_legality(workload_facts, target_facts); legal, rejected = [], []
  for row in proposed_rows:
    candidate = apply_coupled_row(baseline, row); reason = check(candidate)
    (legal if reason is None else rejected).append(dict(row) if reason is None else {"row":dict(row),"reason":reason})
  return legal, rejected


def rank_static_candidates(candidates: Iterable[CanonicalCandidate], priority: Priority) -> list[StaticAssessment]:
  """Deterministically order candidates for measurement; this is never a verdict."""
  scored = [StaticAssessment(_candidate_hash(candidate), *priority(candidate)) for candidate in candidates]
  return sorted(scored, key=lambda row: (-row.score, row.candidate_hash))


def candidate_report(candidates: Iterable[CanonicalCandidate], legalities: Iterable[Legality], priority: Priority) -> dict[str, Any]:
  """Report only static assessments keyed by BoltBeam-supplied identity."""
  accepted, rejected = classify_candidates(candidates, legalities)
  ranked = rank_static_candidates(accepted, priority)
  return {"assessment_version": ASSESSMENT_VERSION,
          "assessments": [{"candidate_hash": row.candidate_hash, "static_score": row.score,
                           "static_reason": row.reason} for row in ranked],
          "rejections": [{"candidate_hash": row.candidate_hash, "reason": row.reason} for row in rejected]}


__all__ = ["CanonicalCandidate", "Legality", "Priority", "StaticAssessment", "StaticRejection", "apply_coupled_row",
           "build_flash_legality", "build_flash_static_priority",
           "build_static_legality", "build_static_priority", "candidate_report", "classify_candidates",
           "classify_coupled_rows", "rank_static_candidates"]
