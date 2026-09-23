"""BubbleBeam: legal dimension values proposed from a target's declared facts.

CPU-only and policy-free. It reads the target facts an adapter supplies
(``boltbeam.search.futuresight_adapter.target_facts``) and filters caller-declared
axis values into the dimensions ``instantiate_candidates`` expands. It proposes
values per axis; it never builds, hashes or ranks a population. FutureSight
(``boltbeam.search.futuresight``) judges the population with the same facts.

Ported from tinygrad-arkey ``extra/llm_research/bubblebeam_futuresight.py``
(35459866b) with the same names and behaviour.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from boltbeam.core.canonical import canonical_json
from boltbeam.search.target_fact_keys import read_target_limits


JSONValue = str | int | float | bool | None | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]


@dataclass(frozen=True)
class LegalDimensionProposal:
  """Legal values proposed for one independent axis; not a population."""
  path: str
  values: tuple[JSONValue, ...]

  def __post_init__(self) -> None:
    if not self.path or any(not part for part in self.path.split(".")): raise ValueError("dimension proposal path is required")
    if not self.values: raise ValueError(f"dimension proposal {self.path!r} has no legal values")
    # canonical_json is an equality key for values here, never a candidate identity.
    if len({canonical_json(value) for value in self.values}) != len(self.values):
      raise ValueError(f"dimension proposal {self.path!r} contains duplicate values")


def dimension_mapping(proposals: Iterable[LegalDimensionProposal]) -> dict[str, tuple[JSONValue, ...]]:
  """Adapt proposal rows to ``instantiate_candidates`` dimensions."""
  rows = tuple(proposals)
  if len({row.path for row in rows}) != len(rows): raise ValueError("dimension proposal paths must be unique")
  return {row.path: row.values for row in sorted(rows, key=lambda row: row.path)}


def _positive_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _legal_transform_sequence(value: Any, supported: frozenset[str]) -> bool:
  if not isinstance(value, (list, tuple)): return False
  for transform in value:
    if not isinstance(transform, Mapping) or set(transform) != {"op", "axis", "arg"}: return False
    if transform["op"] not in supported: return False
    if transform["axis"] is not None and (not isinstance(transform["axis"], int) or isinstance(transform["axis"], bool)): return False
  return True


@dataclass(frozen=True)
class ScheduleVocabulary:
  """Adapter from target-admitted tinygrad transforms to data dimensions."""
  transforms: frozenset[str]
  plan_kinds: frozenset[str]
  generic_control_plan_kind: str | None

  def propose_dimension(self, path: str, values: Iterable[JSONValue]) -> LegalDimensionProposal:
    legal = tuple(value for value in values if isinstance(value, str) and value in self.transforms)
    return LegalDimensionProposal(path, legal)


def target_schedule_vocabulary(target_facts: Mapping[str, JSONValue]) -> ScheduleVocabulary:
  """Consume target facts supplied by an adapter; no backend table is embedded here."""
  transforms = target_facts.get("compiler_transforms", ())
  if not isinstance(transforms, (list, tuple, frozenset)): raise ValueError("target compiler_transforms must be a sequence")
  if not all(isinstance(transform, str) for transform in transforms): raise ValueError("target compiler_transforms must contain strings")
  plan_kinds = target_facts.get("supported_plan_kinds", ())
  if not isinstance(plan_kinds, (list, tuple, frozenset)) or not all(isinstance(kind, str) for kind in plan_kinds):
    raise ValueError("target supported_plan_kinds must be a sequence of strings")
  control = target_facts.get("generic_control_plan_kind")
  if control is not None and (not isinstance(control, str) or control not in plan_kinds):
    raise ValueError("generic_control_plan_kind must name a supported plan kind")
  return ScheduleVocabulary(frozenset(transforms), frozenset(plan_kinds), control)


@dataclass(frozen=True)
class _StaticFacts:
  shape: Mapping[str, Any]
  vocabulary: ScheduleVocabulary
  max_threads: int | None
  max_local_memory: int | None


def _static_facts(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> _StaticFacts:
  """The one reader of workload/target facts; FutureSight's legality uses it too."""
  shape = workload_facts.get("shape", {})
  if not isinstance(shape, Mapping): raise ValueError("workload shape must be a mapping")
  limits = read_target_limits(target_facts)
  return _StaticFacts(dict(shape), target_schedule_vocabulary(target_facts), limits.max_threads, limits.max_local_memory)


def propose_legal_dimensions(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any],
                             axis_choices: Mapping[str, Sequence[JSONValue]]) -> tuple[LegalDimensionProposal, ...]:
  """Filter caller-declared axes using only supplied compiler/resource facts.

  The result is a sorted set of path/value rows, not a candidate population.
  Quantization, scalar/accumulator dtype, layout, and schedule remain separate
  paths; this function never infers one axis from another.
  """
  facts = _static_facts(workload_facts, target_facts); shape, vocabulary = facts.shape, facts.vocabulary
  supported = vocabulary.transforms

  proposals = []
  for path in sorted(axis_choices):
    raw_values = axis_choices[path]
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
      raise ValueError(f"axis choices for {path!r} must be a sequence")
    values: list[JSONValue] = []
    for value in raw_values:
      legal = True
      if path == "schedule.plan_kind": legal = isinstance(value, str) and value in vocabulary.plan_kinds
      elif path == "schedule.transforms": legal = _legal_transform_sequence(value, supported)
      elif path == "schedule.launch.threads": legal = _positive_int(value) and (facts.max_threads is None or value <= facts.max_threads)
      elif path.startswith("schedule.tile.") and path.rsplit(".", 1)[-1] in {"m", "n", "k"}:
        axis = path.rsplit(".", 1)[-1]; extent = shape.get(axis)
        legal = _positive_int(value) and _positive_int(extent) and extent % value == 0
      elif path.endswith((".vector_width", ".alignment", ".stage_count")): legal = _positive_int(value)
      elif path == "static_constraints.max_local_memory_bytes":
        legal = value is None or (_positive_int(value) and (facts.max_local_memory is None or value <= facts.max_local_memory))
      if legal: values.append(value)
    proposals.append(LegalDimensionProposal(path, tuple(values)))
  return tuple(proposals)


__all__ = ["JSONValue", "LegalDimensionProposal", "ScheduleVocabulary", "dimension_mapping", "propose_legal_dimensions",
           "target_schedule_vocabulary"]
