"""Bounded execution adapter for an exact semantic candidate population.

This module owns request assembly only.  Provider JSONL stages and the result
ledger remain authoritative in ``TinygradSearchProviderWorker`` and
``run_full_kernel_search`` respectively.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from boltbeam.plan.resolved_target import validate_candidate_target, validate_resolved_target_document
from boltbeam.search.full_kernel.full_kernel_search import REQUEST_SCHEMA, run_full_kernel_search, validate_execution
from boltbeam.search.semantic.semantic_candidate_plan import build_semantic_candidate_plan
from boltbeam.search.spec import FullKernelCandidate
from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession, TinygradSearchProviderWorker
from boltbeam.search.futuresight_evidence import bind_futuresight_evidence
from boltbeam.search.futuresight_evidence import row_hash


def run_semantic_search_campaign(workload: Mapping[str, Any], schedule: Mapping[str, Any],
                                 dimensions: Mapping[str | tuple[str, ...], Sequence[Any]], *,
                                 worker: Callable[[FullKernelCandidate], Any], request_id: str, run_id: str,
                                 timestamp: str, execution: Mapping[str, Any], resolved_target: Mapping[str, Any], max_candidates: int = 256,
                                 gguf_path: str | None = None,
                                 population_builder=None, compiler_facts: Mapping[str, Any] | None = None,
                                 timeout_s: float = 60.0,
                                 futuresight_evidence: Mapping[str, Any] | None = None, coupled_rows: Sequence[Mapping[str, Any]] = (),
                                 rejected_coupled_rows: Sequence[Mapping[str, Any]] = (),
                                 generator_revision: str = "semantic_search_campaign.v1",
                                 finalist_count: int = 0, finalist_repeats: int = 0,
                                 manage_provider_session: bool = True,
                                 requested_provider_revision: str | None = None,
                                 requested_boltbeam_revision: str | None = None) -> dict[str, Any]:
  """Run one finite, exact-workload campaign through the canonical ledger.

  Exact execution with at least five samples is mandatory.  The supplied worker
  is normally ``TinygradSearchProviderWorker``; it stops a candidate at the
  first failed JSONL stage, and its complete stage envelopes are retained by
  the canonical result ledger.
  """
  if not callable(worker) and not isinstance(worker, PersistentJSONLSession): raise ValueError("semantic campaign worker must be callable")
  normalized_execution = validate_execution(execution)
  if normalized_execution["shape_mode"] != "exact_workload":
    raise ValueError("semantic campaign requires exact_workload execution")
  if normalized_execution["samples"] < 5:
    raise ValueError("semantic campaign requires at least five raw samples")
  if not isinstance(finalist_count, int) or not isinstance(finalist_repeats, int) or finalist_count < 0 or finalist_repeats < 0:
    raise ValueError("semantic finalist repeat bounds must be non-negative integers")
  if (finalist_count == 0) != (finalist_repeats == 0):
    raise ValueError("semantic finalist count and repeats must both be zero or positive")
  if finalist_count and not isinstance(worker, PersistentJSONLSession):
    raise ValueError("semantic finalist repeats require one persistent provider session")
  resolved_target = validate_resolved_target_document(resolved_target)
  if population_builder is None:
    candidates = build_semantic_candidate_plan(workload, schedule, dimensions, max_candidates=max_candidates,
                                                generator_revision=generator_revision)
  else:
    if not callable(population_builder) or not isinstance(compiler_facts, Mapping): raise ValueError("population bridge requires callable and compiler_facts")
    population = population_builder(workload, schedule, dict(compiler_facts), max_candidates)
    if not isinstance(population, Mapping) or not isinstance(population.get("candidates"), (list, tuple)):
      raise ValueError("population bridge must return canonical candidates")
    candidates = tuple(population["candidates"])
  if not candidates: raise ValueError("semantic campaign requires at least one candidate")
  target, candidate_workload = candidates[0].target, candidates[0].workload
  if any(candidate.target != target or candidate.workload != candidate_workload for candidate in candidates):
    raise ValueError("semantic campaign candidate population drifted from its exact workload")
  validate_candidate_target(target, resolved_target)
  static_evidence = bind_futuresight_evidence(candidates, futuresight_evidence, coupled_rows) if futuresight_evidence is not None else None
  rejected_row_evidence = []
  if static_evidence is not None and "rejected_coupled_rows" in futuresight_evidence:
    raw = {row_hash(item["row"]): item for item in rejected_coupled_rows if isinstance(item, Mapping) and "row" in item}
    for item in futuresight_evidence["rejected_coupled_rows"]:
      if not isinstance(item, Mapping) or set(item) != {"row_hash", "row", "reason"} or item["row_hash"] != row_hash(item["row"]) or item["row_hash"] not in raw or raw[item["row_hash"]].get("reason") != item["reason"]: raise ValueError("rejected coupled-row evidence mismatch")
      rejected_row_evidence.append(dict(item))
  static_rejected = {row["candidate_hash"]: row for row in static_evidence["rejections"] if "candidate_hash" in row} if static_evidence else {}
  controls = {candidate.candidate_hash for candidate in candidates if candidate.payload["schedule"]["plan_kind"] == "tinygrad_heuristic.v1"}
  if controls & set(static_rejected): raise ValueError("ordinary heuristic control cannot be statically rejected")
  executable = tuple(candidate for candidate in candidates if candidate.candidate_hash not in static_rejected)
  request = {
    "schema": REQUEST_SCHEMA, "request_id": request_id, "run_id": run_id, "timestamp": timestamp,
    "candidate_space_status": "FINITE", "candidate_space": {"candidates": [candidate.to_dict() for candidate in candidates]},
    "target": target, "resolved_target": resolved_target, "workloads": [candidate_workload],
    "budget": {"max_candidates": max_candidates},
    "objective": {"metric": "median_ns", "direction": "minimize", "tie_break": "candidate_hash"},
    "execution": normalized_execution,
  }
  if requested_boltbeam_revision is not None: request["requested_boltbeam_revision"] = requested_boltbeam_revision
  if requested_provider_revision is not None: request["requested_provider_revision"] = requested_provider_revision
  if static_evidence is not None:
    scores={row["candidate_hash"]:row["static_score"] for row in static_evidence["assessments"]}
    request["execution_order"]=[c.candidate_hash for c in sorted(executable,key=lambda c:(-scores[c.candidate_hash],c.candidate_hash))]+sorted(static_rejected)
  def guarded(candidate):
    if candidate.candidate_hash in static_rejected: return {"candidate_hash":candidate.candidate_hash,"candidate_rejection":{"state":"REJECTED_STATIC","stage":"futuresight_static","code":"static_rejection","reason":str(static_rejected[candidate.candidate_hash].get("reason","static rejection"))}}
    return worker(candidate)
  if getattr(worker, "direct_worker", False): guarded.direct_worker = True
  if isinstance(worker, PersistentJSONLSession):
    # The process lifetime spans the whole population, so its exact-model cache
    # is reusable across all staged requests and is always torn down afterward.
    if not isinstance(gguf_path, str) or not gguf_path: raise ValueError("persistent semantic campaign requires a verified GGUF path")
    exact_gguf = {"workload": dict(workload), "path": gguf_path}
    def execute_persistent():
      staged = TinygradSearchProviderWorker(worker, timeout_s=timeout_s, resolved_target=resolved_target,
        requested_provider_revision=requested_provider_revision, execution=normalized_execution, exact_gguf=exact_gguf)
      def staged_guarded(candidate):
        if candidate.candidate_hash in static_rejected: return {"candidate_hash":candidate.candidate_hash,"candidate_rejection":{"state":"REJECTED_STATIC","stage":"futuresight_static","code":"static_rejection","reason":str(static_rejected[candidate.candidate_hash].get("reason","static rejection"))}}
        return staged(candidate)
      staged_guarded.direct_worker = True
      result = run_full_kernel_search(request, staged_guarded)
      if finalist_count:
        measured = sorted((row for row in result["population"] if row.get("state") == "MEASURED"), key=lambda row:row["rank"])
        finalists = {row["candidate_hash"] for row in measured[:finalist_count]}
        finalists |= {candidate.candidate_hash for candidate in candidates
                      if candidate.payload["schedule"].get("plan_kind") == "tinygrad_heuristic.v1" and
                      any(row["candidate_hash"] == candidate.candidate_hash for row in measured)}
        by_hash = {candidate.candidate_hash:candidate for candidate in candidates}
        result["finalist_repetitions"] = [{"candidate_hash":candidate_hash,
          "repetitions":[staged.measure_only(by_hash[candidate_hash]) for _ in range(finalist_repeats)]}
          for candidate_hash in sorted(finalists)]
      return result
    if manage_provider_session:
      with worker: result = execute_persistent()
    else:
      # A dead shared process is evidence, not an exception.  The staged worker
      # emits one session_unavailable terminal row for every remaining
      # candidate/population without launching a replacement provider.
      result = execute_persistent()
  else: result = run_full_kernel_search(request, guarded)
  if static_evidence is not None:
    result["futuresight_static_evidence"] = static_evidence
    result["futuresight_rejected_coupled_rows"] = rejected_row_evidence
  return result


__all__ = ["run_semantic_search_campaign"]
