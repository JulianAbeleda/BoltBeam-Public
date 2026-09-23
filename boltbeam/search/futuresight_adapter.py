"""BoltBeam's side of BubbleBeam/FutureSight: target facts in, campaign request and static evidence out.

BubbleBeam and FutureSight read chip facts from one flat ``target_facts`` mapping. BoltBeam holds
those facts in two places: the target registry (``boltbeam/data/targets.json``) and the provider's
``describe`` result, whose ``target`` object is what a resolved target records as observed facts.
This adapter reads both under the exact key names the port reads. It never derives one fact from
another (a per-CU LDS size is not a per-threadgroup limit), so a fact neither source carries stays
missing and is reported by name.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from boltbeam.plan.resolved_target import resolved_target_document, validate_candidate_target, validate_resolved_target_document
from boltbeam.search.bubblebeam import dimension_mapping, propose_legal_dimensions
from boltbeam.search.futuresight import build_static_legality, build_static_priority, candidate_report, classify_coupled_rows
from boltbeam.search.futuresight_evidence import bind_futuresight_evidence, row_hash
from boltbeam.search.semantic.semantic_candidate_plan import semantic_workload_facts
from boltbeam.search.semantic.semantic_identity import validate_exact_semantic_workload
from boltbeam.search.semantic.semantic_population_export import POPULATION_SCHEMA
from boltbeam.search.semantic_campaign_cli import REQUEST_FIELDS
from boltbeam.search.spec import FullKernelCandidate
from boltbeam.search.target_fact_keys import MAX_LOCAL_MEMORY, MAX_THREADS, RESERVED_LOCAL_MEMORY
from boltbeam.target.targets import get_target

# Every target_facts key BubbleBeam/FutureSight read: _static_facts and target_schedule_vocabulary
# (proposal and static legality), and the flash builders' limits and per-launch reservation. The
# subgroup size is not a target fact here: the flash builders read each candidate's own target block.
TARGET_FACT_KEYS = ("compiler_transforms", "generic_control_plan_kind", MAX_LOCAL_MEMORY, MAX_THREADS,
                    RESERVED_LOCAL_MEMORY, "supported_plan_kinds")

# A proposal spec is a campaign request before BubbleBeam runs: axis choices and coupled rows in,
# dimensions and legal/rejected rows out. compiler_facts are not an input; this adapter builds them.
_PROPOSED_FIELDS = frozenset({"dimensions", "legal_coupled_rows", "rejected_coupled_rows", "futuresight_evidence", "compiler_facts"})
SPEC_FIELDS = (REQUEST_FIELDS - _PROPOSED_FIELDS) | {"axis_choices", "coupled_rows"}


def target_facts(target_id:str, describe:Mapping[str, Any] | None = None) -> tuple[dict[str, Any], tuple[str, ...]]:
  """Return ``(target_facts, missing_keys)`` for a registry target and an optional provider describe result.

  Sources, in lookup order: the registry row's fields, its capabilities, the describe result's
  ``target`` object, and the describe result itself (compiler_transforms, supported_plan_kinds).
  Every source that carries a key must agree on its value, or the facts are refused.
  """
  if describe is not None and (not isinstance(describe, Mapping) or not isinstance(describe.get("target", {}), Mapping)):
    raise ValueError("provider describe result must be an object whose target is an object")
  row = get_target(target_id).to_json()
  provider = describe or {}
  sources = {"registry": row, "registry capabilities": row.get("capabilities") or {},
             "provider target": provider.get("target", {}), "provider": provider}
  facts: dict[str, Any] = {}
  for key in TARGET_FACT_KEYS:
    found = {name: json.loads(json.dumps(source[key])) for name, source in sources.items() if source.get(key) is not None}
    if len({json.dumps(value, sort_keys=True) for value in found.values()}) > 1:
      raise ValueError(f"target fact {key!r} disagrees across sources: {found}")
    if found: facts[key] = next(iter(found.values()))
  return facts, tuple(key for key in TARGET_FACT_KEYS if key not in facts)


def propose_request(spec:Mapping[str, Any], describe:Mapping[str, Any] | None = None) -> tuple[dict[str, Any], tuple[str, ...]]:
  """BubbleBeam: propose legal dimensions and split coupled rows for one exact workload on its resolved target.

  Returns the semantic campaign request and the target facts that were missing. Without ``describe``
  the provider facts are the resolved target's recorded observed facts; with it, its target object
  must reproduce that resolved target. The request carries no evidence: FutureSight assesses the
  population exported from it.
  """
  if not isinstance(spec, Mapping) or set(spec) != SPEC_FIELDS:
    raise ValueError("dimension proposal spec has missing or unknown fields")
  if not isinstance(spec["axis_choices"], Mapping): raise ValueError("axis_choices must be an object")
  if not isinstance(spec["coupled_rows"], list) or not all(isinstance(row, Mapping) for row in spec["coupled_rows"]):
    raise ValueError("coupled_rows must be a list of objects")
  if not isinstance(spec["schedule"], Mapping): raise ValueError("schedule must be an object")
  workload = spec["semantic_workload"]
  validate_exact_semantic_workload(workload)
  resolved = validate_resolved_target_document(spec["resolved_target"])
  validate_candidate_target(workload["target"], resolved)
  provider = describe if describe is not None else {"target": resolved["observed_facts"]}
  facts, missing = target_facts(resolved["target_id"], provider)
  if resolved_target_document(resolved["target_id"], provider.get("target")) != resolved:
    raise ValueError("provider describe target facts differ from the request's resolved target")
  shape = semantic_workload_facts(workload)
  dimensions = dimension_mapping(propose_legal_dimensions(shape, facts, spec["axis_choices"]))
  legal, rejected = classify_coupled_rows({"schedule": dict(spec["schedule"])}, spec["coupled_rows"], shape, facts)
  request = {key: spec[key] for key in REQUEST_FIELDS - _PROPOSED_FIELDS} | {
    "dimensions": dimensions, "legal_coupled_rows": legal, "rejected_coupled_rows": rejected,
    "futuresight_evidence": None, "compiler_facts": facts}
  return request, missing


def assess_population(population:Mapping[str, Any], preferences:Mapping[str, Any] | None = None) -> dict[str, Any]:
  """FutureSight: statically reject and order one exported population into semantic-campaign evidence.

  Facts come only from the population (the request's compiler_facts, exported with it); the ordering
  preferences are the caller's. The evidence passes semantic-campaign's own binding check before it
  is returned, so a file written from it is one the campaign accepts.
  """
  if not isinstance(population, Mapping) or population.get("schema") != POPULATION_SCHEMA:
    raise ValueError(f"FutureSight assessment requires a {POPULATION_SCHEMA} population")
  if preferences is not None and not isinstance(preferences, Mapping): raise ValueError("preferences must be an object")
  facts = population["compiler_facts"]
  if "static_preferences" in facts:
    raise ValueError("static_preferences are a caller choice, not a compiler fact; pass them as preferences")
  candidates = [FullKernelCandidate.from_dict(row["candidate"]) for row in population["candidates"]]
  if [c.candidate_hash for c in candidates] != [row["candidate_hash"] for row in population["candidates"]]:
    raise ValueError("population candidate_hash does not match its canonical candidate")
  if any(item["row_hash"] != row_hash(item["row"]) for item in population["coupled_rows"]):
    raise ValueError("population row_hash does not match its coupled row")
  legality = build_static_legality(population["workload_facts"], facts)
  report = candidate_report([c.to_dict() | {"candidate_hash": c.candidate_hash} for c in candidates], (legality,),
                            build_static_priority(preferences or {}))
  evidence = report | {"population_hash": population["population_hash"],
                       "rejected_coupled_rows": list(population["rejected_coupled_rows"])}
  bind_futuresight_evidence(candidates, evidence, [item["row"] for item in population["coupled_rows"]])
  return evidence


__all__ = ["SPEC_FIELDS", "TARGET_FACT_KEYS", "assess_population", "propose_request", "target_facts"]
