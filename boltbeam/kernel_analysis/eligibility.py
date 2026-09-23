"""KA-4: decision-specific eligibility authority for kernel evidence.

One authority answers whether a candidate's evidence chain supports a given decision:

- diagnosis: can we reason about what happened at all?
- correctness: is the candidate's output trustworthy?
- performance: can this candidate's timing enter a comparison (optionally against a comparator)?
- promotion: could this evidence satisfy downstream promotion policy? (report-only; this package never promotes)

Missing evidence is never a pass. A single boolean is insufficient, so every decision returns the
per-requirement results (reusing core.requirements vocabulary) alongside an overall verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from boltbeam.core.requirements import RequirementResult, RequirementStatus
from boltbeam.kernel_analysis.model import KernelEvidence

SATISFIED = RequirementStatus.SATISFIED.value
MISSING = RequirementStatus.MISSING.value
INVALID = RequirementStatus.INVALID.value
MISMATCH = RequirementStatus.MISMATCH.value
UNSUPPORTED = RequirementStatus.UNSUPPORTED.value


class Decision(str, Enum):
  DIAGNOSIS = "diagnosis"
  CORRECTNESS = "correctness"
  PERFORMANCE = "performance"
  PROMOTION = "promotion"


@dataclass(frozen=True)
class EligibilityResult:
  decision: str
  eligible: bool
  results: tuple[RequirementResult, ...]
  candidate_id: str | None = None
  comparator_id: str | None = None

  @property
  def blockers(self) -> tuple[RequirementResult, ...]:
    return tuple(r for r in self.results if r.status != SATISFIED)

  def to_json(self) -> dict[str, Any]:
    out = {
      "decision": self.decision,
      "eligible": self.eligible,
      "results": [r.to_json() for r in self.results],
      "blockers": [r.requirement_id for r in self.blockers],
    }
    if self.candidate_id is not None:
      out["candidate_id"] = self.candidate_id
    if self.comparator_id is not None:
      out["comparator_id"] = self.comparator_id
    return out


def _r(req_id: str, ok: bool, bad_status: str, reason: str) -> RequirementResult:
  return RequirementResult(req_id, SATISFIED if ok else bad_status, "" if ok else reason)


def _verdict(decision: Decision, results: list[RequirementResult], ev: KernelEvidence,
             comparator: KernelEvidence | None = None) -> EligibilityResult:
  eligible = all(r.status == SATISFIED for r in results)
  return EligibilityResult(
    decision=decision.value, eligible=eligible, results=tuple(results),
    candidate_id=ev.candidate.candidate_id,
    comparator_id=comparator.candidate.candidate_id if comparator is not None else None,
  )


# --- shared checks -----------------------------------------------------------

def _identity_corrupt(ev: KernelEvidence) -> bool:
  return any(b.scope == "identity" or b.code in ("identity_mismatch", "identity_corrupt")
             for b in ev.blockers)


def _workload_identified(ev: KernelEvidence) -> bool:
  # Enough to reason about *what happened* — operation and role. A GEMM shape is not required to
  # diagnose a norm/elementwise kernel that has no m/n/k.
  w = ev.workload
  return bool(w.operation and w.role)


def _workload_known(ev: KernelEvidence) -> bool:
  # Enough to *compare* — a concrete shape is required for correctness/performance.
  w = ev.workload
  return bool(w.operation and w.role and w.shape)


def _has_source(ev: KernelEvidence) -> bool:
  if ev.provenance:
    return True
  return any(stage.sources for stage in
             (ev.stages.compile, ev.stages.execution, ev.stages.correctness, ev.stages.timing))


def _stage_known(ev: KernelEvidence) -> bool:
  # A stage disposition exists (something other than a uniform not_run), or a blocker was recorded.
  statuses = (ev.stages.compile.status, ev.stages.execution.status,
              ev.stages.correctness.status, ev.stages.timing.status)
  return any(s != "not_run" for s in statuses) or bool(ev.blockers)


def _executed_matches_compiled(ev: KernelEvidence) -> bool | None:
  b, x = ev.identity.binary_hash, ev.identity.executed_binary_hash
  if b is None or x is None:
    return None
  return b == x


def _health_invalidates(ev: KernelEvidence) -> bool:
  h = ev.health
  if h is None:
    return False
  return h.postflight is False or (h.timeout_contained is False)


# --- diagnosis ---------------------------------------------------------------

def diagnosis_eligibility(ev: KernelEvidence) -> EligibilityResult:
  """A compile/runtime failure is diagnosable when candidate, workload, stage, and source are known.
  Identity corruption is not evidence about the kernel, so it blocks diagnosis."""
  results = [
    _r("diag.candidate_known", bool(ev.candidate.candidate_id), MISSING, "candidate id missing"),
    _r("diag.workload_identified", _workload_identified(ev), MISSING, "operation/role missing"),
    _r("diag.stage_known", _stage_known(ev), MISSING, "no stage disposition or blocker recorded"),
    _r("diag.source_known", _has_source(ev), MISSING, "no evidence source"),
    _r("diag.identity_not_corrupt", not _identity_corrupt(ev), INVALID,
       "identity corruption is not evidence about the kernel"),
  ]
  return _verdict(Decision.DIAGNOSIS, results, ev)


# --- correctness -------------------------------------------------------------

def correctness_eligibility(ev: KernelEvidence) -> EligibilityResult:
  compile_ok = ev.stages.compile.status == "pass"
  exec_ok = ev.stages.execution.status == "pass"
  matches = _executed_matches_compiled(ev)
  c = ev.correctness
  results = [
    _r("corr.compile_pass", compile_ok, INVALID, f"compile status is {ev.stages.compile.status!r}"),
    _r("corr.execution_pass", exec_ok, INVALID, f"execution status is {ev.stages.execution.status!r}"),
    _r("corr.binary_match", matches is True, MISSING if matches is None else MISMATCH,
       "compiled/executed binary identity is missing" if matches is None else
       "executed binary does not match compiled binary"),
    _r("corr.workload_known", _workload_known(ev), MISSING, "workload incomplete"),
    _r("corr.reference_known", ev.identity.reference_identity is not None
       or (c is not None and c.seed is not None), MISSING, "no reference/seed identity"),
    _r("corr.metrics_present", c is not None and not c.missing_metrics, MISSING,
       "required numerical metrics missing"),
    _r("corr.numerical_pass", c is not None and c.status == "pass", INVALID,
       "numerical comparison did not pass" if c is not None else "no correctness metrics"),
    _r("corr.health_ok", not _health_invalidates(ev), INVALID, "invalidating health failure"),
  ]
  return _verdict(Decision.CORRECTNESS, results, ev)


# --- performance -------------------------------------------------------------

def _performance_readiness(ev: KernelEvidence) -> list[RequirementResult]:
  corr = correctness_eligibility(ev)
  t = ev.timing
  timing_measured = ev.stages.timing.status == "measured" and t is not None and bool(t.samples)
  scope_known = t is not None and bool(t.scope)
  no_timeout = ev.stages.execution.status not in ("timeout", "runtime_fault", "health_failure")
  results = [
    _r("perf.correctness", corr.eligible, INVALID, "candidate is not correctness-eligible"),
    _r("perf.timing_measured", timing_measured, MISSING if t is None else INVALID,
       "no measured timing samples"),
    _r("perf.timing_scope_known", scope_known, MISSING, "timing scope/inclusion unknown"),
    _r("perf.no_timeout_or_fault", no_timeout, INVALID,
       f"execution status is {ev.stages.execution.status!r}"),
    _r("perf.health_ok", not _health_invalidates(ev), INVALID, "invalidating health failure"),
  ]
  return results


def _comparable(candidate: KernelEvidence, comparator: KernelEvidence) -> list[RequirementResult]:
  ci, xi = candidate.identity, comparator.identity

  def _match(a, b):
    return a is not None and b is not None and a == b

  workload_equal = candidate.workload.to_json() == comparator.workload.to_json()
  results = [
    _r("perf.comparator_identity", comparator.candidate.candidate_id is not None
       and comparator.candidate.candidate_id != candidate.candidate.candidate_id, MISSING,
       "explicit distinct comparator identity required"),
    _r("perf.same_workload", workload_equal, MISMATCH, "candidate and comparator workloads differ"),
    _r("perf.same_system", _match(ci.system_snapshot_id, xi.system_snapshot_id), MISSING
       if ci.system_snapshot_id is None or xi.system_snapshot_id is None else MISMATCH,
       "system snapshot missing or mismatched"),
    _r("perf.same_session", _match(ci.session_id, xi.session_id), MISSING
       if ci.session_id is None or xi.session_id is None else MISMATCH,
       "session missing or mismatched"),
    _r("perf.same_clock_state", _match(ci.clock_state_id, xi.clock_state_id), MISSING
       if ci.clock_state_id is None or xi.clock_state_id is None else MISMATCH,
       "clock state missing or mismatched"),
  ]
  return results


def performance_eligibility(ev: KernelEvidence,
                            comparator: KernelEvidence | None = None) -> EligibilityResult:
  """Per-candidate performance readiness; if a comparator is supplied, also require the two chains be comparable.

  Raw timeout is diagnosis-eligible but never performance-eligible. A measured identity mismatch fails."""
  results = _performance_readiness(ev)
  if comparator is not None:
    results = results + _comparable(ev, comparator)
  return _verdict(Decision.PERFORMANCE, results, ev, comparator)


# --- promotion (report-only) -------------------------------------------------

def promotion_eligibility(ev: KernelEvidence) -> EligibilityResult:
  """Report-only: could this evidence satisfy downstream promotion policy? This package never promotes."""
  perf = performance_eligibility(ev)
  results = list(perf.results) + [
    _r("promo.experiment_identity", ev.identity.experiment_id is not None, MISSING,
       "durable experiment identity required for promotion evidence"),
    _r("promo.session_identity", ev.identity.session_id is not None, MISSING,
       "session identity required for promotion evidence"),
  ]
  return _verdict(Decision.PROMOTION, results, ev)


def evaluate_all(ev: KernelEvidence,
                 comparator: KernelEvidence | None = None) -> dict[str, EligibilityResult]:
  """Convenience: every decision for one candidate (performance uses comparator when given)."""
  return {
    Decision.DIAGNOSIS.value: diagnosis_eligibility(ev),
    Decision.CORRECTNESS.value: correctness_eligibility(ev),
    Decision.PERFORMANCE.value: performance_eligibility(ev, comparator),
    Decision.PROMOTION.value: promotion_eligibility(ev),
  }
