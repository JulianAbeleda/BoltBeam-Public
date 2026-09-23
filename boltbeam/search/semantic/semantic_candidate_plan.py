"""Finite, policy-free candidates for one exact semantic Tinygrad workload.

This is deliberately an authoring seam: it preserves the recorded workload and
delegates canonical expansion and hashing to :mod:`full_kernel_candidates`.
It neither selects a winner nor embeds backend resource/roofline policy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from boltbeam.search.full_kernel.full_kernel_candidates import FieldPath, instantiate_candidate_rows, instantiate_candidates
from boltbeam.search.futuresight_evidence import ASSESSMENT_VERSION
from boltbeam.search.semantic.semantic_identity import validate_exact_semantic_workload
from boltbeam.search.spec import FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION, FullKernelCandidate


SEMANTIC_WORKLOAD_SCHEMA = "tinygrad.semantic_provider_workload.v1"
# Names from tinygrad.codegen.opt.OptOps.  Keeping this vocabulary here makes
# the boundary explicit and prevents architecture-specific emitters leaking in.
GENERIC_TINYGRAD_OPT_OPS = frozenset(("TC", "UPCAST", "UNROLL", "LOCAL", "THREAD", "GROUP", "GROUPTOP", "NOLOCALS", "PADTO", "SWAP"))


def build_semantic_candidate_plan(workload: Mapping[str, Any], schedule: Mapping[str, Any],
                                  dimensions: Mapping[FieldPath, Sequence[Any]], *, max_candidates: int = 256,
                                  generator_revision: str = "semantic_candidate_plan.v1") -> tuple[FullKernelCandidate, ...]:
  """Expand a bounded generic-opt plan for exactly one recorded workload.

  ``workload`` is the output of ``semantic_identity_to_workload``.  The caller
  supplies an already-admitted target identity and a data-only baseline
  schedule; this function does not infer either.  Only ``schedule.*`` fields
  may vary, so no expansion can alter the exact model/tensor/MNK contract.
  """
  if not isinstance(workload, Mapping) or workload.get("schema") != SEMANTIC_WORKLOAD_SCHEMA:
    raise ValueError("semantic candidate plan requires an exact semantic workload")
  if workload.get("fixture_shape_substitution") != "forbidden":
    raise ValueError("semantic candidate plan forbids fixture shape substitution")
  if not isinstance(schedule, Mapping): raise ValueError("schedule must be a mapping")
  if not isinstance(dimensions, Mapping): raise ValueError("dimensions must be a mapping")
  if not isinstance(generator_revision, str) or not generator_revision: raise ValueError("generator_revision is required")
  if any(not _path_text(path).startswith("schedule.") for path in dimensions):
    raise ValueError("semantic candidate dimensions may only change schedule fields")

  identity, shape, operands, tolerance, target = (workload.get(key) for key in
                                                    ("semantic_identity", "shape", "operands", "tolerance", "target"))
  if not all(isinstance(value, Mapping) for value in (identity, shape, operands, tolerance, target)):
    raise ValueError("semantic workload is incomplete")
  identity = validate_exact_semantic_workload(workload)
  _validate_generic_transforms(schedule.get("transforms"))
  for path, values in dimensions.items():
    if _path_text(path) == "schedule.transforms" and isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
      for transforms in values: _validate_generic_transforms(transforms)

  seed = FullKernelCandidate({
    "schema_version": FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION,
    "workload": {
      "profile": identity.get("module_path"), "model_sha256": workload.get("model_hash"),
      "phase": identity.get("phase"), "role": identity.get("role"), "operation": workload.get("operation"),
      "shape": dict(shape),
      "operands": {
        "a": {"dtype": operands.get("a", {}).get("dtype"), "layout": "row_major", "quantization": "none"},
        "b": {"dtype": identity.get("source_quant_storage", "").lower(), "layout": operands.get("b", {}).get("layout"),
              "quantization": operands.get("b", {}).get("quantization")},
        "c": {"dtype": operands.get("c", {}).get("dtype"), "layout": "row_major", "quantization": "none"},
      }, "accumulator_dtype": identity.get("accumulator_dtype"), "target": dict(target),
    },
    "schedule": dict(schedule),
    "static_constraints": {"max_local_memory_bytes": None, "max_registers_per_thread": None, "spill_policy": "unknown"},
    "correctness": {"oracle": "exact_gguf_packed_reference", "atol": tolerance.get("atol"), "rtol": tolerance.get("rtol")},
    "memory_budget": {"status": "unavailable", "bytes": None},
    "provenance": {"generator_id": "semantic_candidate_plan", "generator_revision": generator_revision,
                   "schema_revision": FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION},
    "applicability": {"exact_shape": True, "profiles": [identity.get("module_path")], "roles": [identity.get("role")],
                      "targets": [f"{target.get('target_id')}:subgroup{target.get('subgroup_size')}"]},
  })
  if max_candidates < 2: raise ValueError("semantic plan requires room for the heuristic control")
  expanded = instantiate_candidates(seed, dimensions, max_candidates=max_candidates-2) if dimensions else ()
  control_payload = json.loads(seed.canonical_json())
  control_payload["schedule"]["plan_kind"] = "tinygrad_heuristic.v1"
  control_payload["schedule"]["transforms"] = []
  control = FullKernelCandidate(control_payload)
  # Canonical identity remains the only dedup authority; ordering by hash keeps
  # the finite population deterministic without encoding a measurement policy.
  return tuple(sorted({candidate.candidate_hash: candidate for candidate in (*expanded, seed, control)}.values(), key=lambda x: x.candidate_hash))

def assess_semantic_population(candidates: Sequence[FullKernelCandidate], *, legality, priority) -> dict[str, Any]:
  """Use injected BubbleBeam/FutureSight static facts without owning policy.

  ``legality`` returns a rejection reason or ``None``; ``priority`` returns a
  static score/reason.  This seam cannot expand, measure, select, or promote.
  """
  accepted, rejected = [], []
  for candidate in sorted(candidates, key=lambda row: row.candidate_hash):
    reason = legality(candidate.to_dict())
    if reason is None: accepted.append(candidate)
    else: rejected.append({"candidate_hash": candidate.candidate_hash, "reason": str(reason)})
  assessments = []
  for candidate in accepted:
    score, reason = priority(candidate.to_dict())
    assessments.append({"candidate_hash": candidate.candidate_hash, "static_score": score, "static_reason": reason})
  return {"assessment_version": ASSESSMENT_VERSION, "assessments": sorted(assessments, key=lambda x: (-x["static_score"], x["candidate_hash"])), "rejections": rejected}

def instantiate_legal_coupled_rows(seed: FullKernelCandidate, rows: Sequence[Mapping[FieldPath, Any]], *, legality, max_candidates: int = 256) -> tuple[FullKernelCandidate, ...]:
  """Prune FutureSight-coupled schedule rows before BoltBeam hashes them.

  The injected legality predicate sees a plain schedule-row mapping derived
  solely by the caller from exact workload and admitted compiler facts.
  """
  legal_rows = tuple(row for row in rows if legality(row) is None)
  if not legal_rows: return ()
  return instantiate_candidate_rows(seed, legal_rows, max_candidates=max_candidates)

def semantic_workload_facts(workload: Mapping[str, Any]) -> dict[str, Any]:
  """The exact shape/operand facts BubbleBeam and FutureSight receive for one workload; nothing else."""
  return {"shape": dict(workload.get("shape", {})), "operands": dict(workload.get("operands", {}))}

def build_semantic_population(workload: Mapping[str, Any], schedule: Mapping[str, Any], *, compiler_facts: Mapping[str, Any],
                              propose_dimensions, coupled_rows: Sequence[Mapping[FieldPath, Any]] = (), legality=lambda _row: None,
                              priority=lambda _candidate: (0, "no_static_preference"), max_candidates: int = 256) -> dict[str, Any]:
  """Central non-hardware bridge to injected FutureSight proposal/assessment.

  Proposal derivation is owned by the injected BubbleBeam function and receives
  only exact shape/operand facts plus adapter-supplied compiler facts.
  """
  if not isinstance(compiler_facts, Mapping): raise ValueError("compiler_facts must be a mapping")
  facts = semantic_workload_facts(workload)
  dimensions = propose_dimensions(facts, dict(compiler_facts))
  population = build_semantic_candidate_plan(workload, schedule, dimensions, max_candidates=max_candidates)
  if coupled_rows:
    # Coupled rows are additional legal alternatives; canonical hashes dedup
    # against independent-axis candidates and the mandatory heuristic control.
    baseline = next((candidate for candidate in population if candidate.payload["schedule"] == dict(schedule)), None)
    if baseline is None: raise ValueError("coupled rows require the canonical baseline schedule in the population")
    extras = instantiate_legal_coupled_rows(baseline, coupled_rows, legality=legality, max_candidates=max_candidates)
    population = tuple(sorted({x.candidate_hash:x for x in (*population, *extras)}.values(), key=lambda x:x.candidate_hash))
  if len(population) > max_candidates: raise ValueError("semantic population exceeds max_candidates")
  if sum(x.payload["schedule"].get("plan_kind") == "tinygrad_heuristic.v1" for x in population) != 1:
    raise ValueError("semantic population requires exactly one heuristic control")
  return {"candidates": population, "assessment": assess_semantic_population(population, legality=lambda c: legality(c.get("schedule", {})), priority=priority)}


def _path_text(path: FieldPath) -> str:
  return path if isinstance(path, str) else ".".join(path)


def _validate_generic_transforms(transforms: Any) -> None:
  if not isinstance(transforms, list): raise ValueError("schedule.transforms must be a list")
  for transform in transforms:
    if not isinstance(transform, Mapping) or transform.get("op") not in GENERIC_TINYGRAD_OPT_OPS:
      raise ValueError("schedule transforms must use generic tinygrad OptOps")


__all__ = ["GENERIC_TINYGRAD_OPT_OPS", "SEMANTIC_WORKLOAD_SCHEMA", "assess_semantic_population", "build_semantic_candidate_plan", "build_semantic_population", "instantiate_legal_coupled_rows", "semantic_workload_facts"]
