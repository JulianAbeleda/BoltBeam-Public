"""KA-7: one-variable next-experiment recommendation.

Follows a fixed priority ladder — repair identity, establish execution, establish correctness,
establish comparability, then test the highest-impact unresolved mechanism, and only then expand
workload. Each recommendation names exactly one intended variable (or an explicit two-factor request
when factors cannot vary independently), the fixed invariants, and the expected outcome under each
hypothesis. When the question is already answered, no recommendation is emitted.
"""
from __future__ import annotations

from boltbeam.kernel_analysis.contrast import PairContrast
from boltbeam.kernel_analysis.diagnosis import Hypothesis
from boltbeam.kernel_analysis.eligibility import (
  correctness_eligibility, diagnosis_eligibility, performance_eligibility,
)
from boltbeam.kernel_analysis.model import KernelEvidence, KernelRecommendation

# mechanisms that cannot be varied independently in a single knob -> require a named two-factor request
_ENTANGLED = {("staging", "geometry")}


def _identity_corrupt(ev: KernelEvidence) -> bool:
  return any(b.scope == "identity" or b.code in ("identity_mismatch", "identity_corrupt")
             for b in ev.blockers)


def _workload_points(ev: KernelEvidence) -> tuple[dict, ...]:
  return (dict(ev.workload.shape),)


def recommend(candidate: KernelEvidence,
              comparator: KernelEvidence | None = None,
              contrast: PairContrast | None = None,
              hypotheses: tuple[Hypothesis, ...] = ()) -> KernelRecommendation | None:
  """Return the single next bounded experiment, or None when nothing is left to establish."""
  cand_id = candidate.candidate.candidate_id
  comp_id = comparator.candidate.candidate_id if comparator is not None else None
  invariants = {"workload": candidate.workload.to_json(),
                "target": candidate.identity.target_id}

  # 1. Repair experiment/harness identity.
  binary_mismatch = (candidate.identity.binary_hash is not None
                     and candidate.identity.executed_binary_hash is not None
                     and candidate.identity.binary_hash != candidate.identity.executed_binary_hash)
  if _identity_corrupt(candidate) or binary_mismatch:
    return KernelRecommendation(
      hypothesis="evidence identity is corrupted; the run is not attributable to the kernel",
      candidate=cand_id, comparator=comp_id, variable="experiment_identity",
      invariants=invariants,
      required_artifacts=("matched compiled/executed binary hashes", "single-session experiment manifest"),
      expected_outcomes=("after repair, the same candidate produces a self-consistent identity chain",),
      stop_condition="identity chain is internally consistent",
      reopen_condition="binary, compiler, or harness changes",
    )

  # 2. Establish execution (never request timing for a non-executing candidate).
  exec_status = candidate.stages.execution.status
  if exec_status != "pass":
    liveness = {
      "timeout": ("execution timed out; capture liveness evidence, not timing",
                  ("barrier/wait counts", "per-wave progress trace before hang", "K-loop exit evidence")),
      "runtime_fault": ("execution faulted; capture the fault class and address",
                        ("fault class/address", "preflight/postflight health")),
      "health_failure": ("device health failed; re-establish a clean device before any timing",
                         ("preflight/postflight health", "device recovery log")),
      "no_result": ("harness returned no structured result; re-run under isolation",
                    ("isolated re-run result", "harness logs")),
    }.get(exec_status, ("execution has not been established",
                        ("execution result under the declared protocol",)))
    return KernelRecommendation(
      hypothesis=liveness[0], candidate=cand_id, comparator=comp_id, variable="execution_liveness",
      invariants=invariants, required_artifacts=liveness[1],
      workloads=_workload_points(candidate),
      expected_outcomes=("execution either completes or yields a discriminating liveness signal",),
      stop_condition="execution status is pass or a mechanism is isolated",
      reopen_condition="kernel body or synchronization changes",
    )

  # 3. Establish correctness (must precede any timing comparison).
  corr = correctness_eligibility(candidate)
  if not corr.eligible:
    missing = [b.requirement_id for b in corr.blockers]
    return KernelRecommendation(
      hypothesis="correctness is not established for this candidate",
      candidate=cand_id, comparator=comp_id, variable="correctness",
      invariants=invariants,
      required_artifacts=("element count, tolerances, max error, finite-output check", "reference/seed identity"),
      workloads=_workload_points(candidate),
      seeds=({"seed": candidate.correctness.seed} if candidate.correctness and candidate.correctness.seed
             else {"seed": "declare a fixed seed"},),
      expected_outcomes=("candidate output matches the reference within tolerance, or a numerical defect is localized",),
      stop_condition="numerical comparison passes with required metrics present",
      reopen_condition="numerical policy, reference, or kernel body changes",
    )

  # 4. Establish comparability (a valid gap needs a same-session A/B under identical conditions).
  if comparator is not None:
    perf = performance_eligibility(candidate, comparator=comparator)
    if not perf.eligible:
      unmet = {b.requirement_id for b in perf.blockers}
      comparability_ids = {"perf.same_workload", "perf.same_system", "perf.same_session",
                           "perf.comparator_identity"}
      if unmet & comparability_ids:
        return KernelRecommendation(
          hypothesis="candidate and comparator are not yet comparable",
          candidate=cand_id, comparator=comp_id, variable="comparability",
          invariants=invariants,
          required_artifacts=("same-session A/B under identical system/clock/protocol",
                              "explicit comparator identity"),
          workloads=_workload_points(candidate),
          expected_outcomes=("a same-session A/B yields a comparable timing pair",),
          stop_condition="candidate and comparator share workload, system, session, and protocol",
          reopen_condition="system, session, or protocol changes",
        )

  # 5. Test the highest-impact unresolved mechanism.
  mechanism = _top_mechanism(contrast, hypotheses)
  if mechanism is not None:
    variable, statement, factors = mechanism
    two_factor = tuple(sorted(factors)) in _ENTANGLED or len(factors) > 1
    return KernelRecommendation(
      hypothesis=statement,
      candidate=cand_id, comparator=comp_id,
      variable=None if two_factor else variable,
      invariants={**invariants, "held_fixed": "all knobs except the intended factor(s)"},
      required_artifacts=("A/B evidence isolating the intended factor(s) under identical conditions",),
      workloads=_workload_points(candidate),
      expected_outcomes=(
        f"if the mechanism holds, varying {variable} moves timing in the predicted direction",
        "if refuted, timing is unchanged within the noise floor",
      ),
      hints={"factorial": sorted(factors)} if two_factor else {},
      stop_condition="the intended factor's effect is measured beyond the noise floor",
      reopen_condition="the isolated factor or the kernel body changes",
    )

  # A measured selection can be complete while its operand mechanism remains unresolved.
  unresolved = [(operand, path) for operand, path in candidate.operand_paths
                if path.classification.strategy == "unknown"
                or path.classification.missing_discriminators
                or path.classification.contradicting_evidence_ids]
  if unresolved:
    operand, path = unresolved[0]
    missing = path.classification.missing_discriminators or (
      "identity-bound per-operand final-program attribution",)
    return KernelRecommendation(
      hypothesis="the operand transport mechanism is unresolved independently of any timing result",
      candidate=cand_id, comparator=comp_id, variable=f"operand:{operand}:strategy",
      invariants={**invariants, "held_fixed": "workload, geometry, layout, ABI, and timing protocol"},
      required_artifacts=tuple(missing), workloads=_workload_points(candidate),
      expected_outcomes=("static/dynamic evidence classifies exactly one operand strategy or remains unknown",),
      stop_condition=f"operand {operand} has an exclusive classification without unresolved contradiction",
      reopen_condition="final binary, operand ownership map, or transport declaration changes",
    )

  # 6. Nothing left to establish for this pair/candidate.
  return None


def _top_mechanism(contrast: PairContrast | None,
                   hypotheses: tuple[Hypothesis, ...]):
  """Pick the highest-impact unresolved performance mechanism, if any. Returns (variable, statement, factors)."""
  if contrast is None:
    return None
  if contrast.timing.get("qualified") not in ("win", "regression"):
    return None  # tie/unavailable -> no mechanism worth an isolating experiment
  # Prefer a declared intended transport change (staging), which often entangles with geometry.
  mech = contrast.mechanism
  if mech.get("status") == "present":
    changed = mech.get("changed_operands", [])
    return ("operand_strategy", f"test whether the measured gap follows the declared operand strategy change on {changed}",
            {"staging", "geometry"})
  # Otherwise, escalate the highest-confidence performance hypothesis.
  perf = [h for h in hypotheses if h.category == "performance" and h.status in ("supported", "plausible")]
  if perf:
    order = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    top = max(perf, key=lambda h: order.get(h.confidence, 0))
    return (top.category, top.statement, {top.hypothesis_id.split(".")[-1]})
  return None
