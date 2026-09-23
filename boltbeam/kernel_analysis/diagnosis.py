"""KA-6A: evidence-bounded diagnostic reasoning.

A typed rule registry turns normalized facts into hypotheses. Rules never consume candidate,
provider, or target names — only measured/derived facts. Every hypothesis records what supports it,
what contradicts it, what evidence is missing, and a calibrated confidence. A timeout with barriers
is a *plausible* synchronization/liveness hypothesis, never a *proven* divergent barrier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from boltbeam.core.facts import TruthStatus
from boltbeam.kernel_analysis.contrast import PairContrast
from boltbeam.kernel_analysis.model import KernelEvidence

# status / confidence vocabularies
SUPPORTED, PLAUSIBLE, CONTRADICTED, UNKNOWN = "supported", "plausible", "contradicted", "unknown"
HIGH, MEDIUM, LOW, CONF_UNKNOWN = "high", "medium", "low", "unknown"
_CONF_ORDER = [CONF_UNKNOWN, LOW, MEDIUM, HIGH]

MOSTLY_ZERO_THRESHOLD = 0.9
LOW_CORRELATION_THRESHOLD = 0.9


@dataclass(frozen=True)
class Hypothesis:
  hypothesis_id: str
  category: str
  statement: str
  status: str
  truth_status: str
  confidence: str
  supporting_fact_ids: tuple[str, ...] = ()
  contradicting_fact_ids: tuple[str, ...] = ()
  missing_evidence: tuple[str, ...] = ()
  scope: str = "candidate"

  def __post_init__(self):
    if self.status not in (SUPPORTED, PLAUSIBLE, CONTRADICTED, UNKNOWN):
      raise ValueError(f"invalid hypothesis status {self.status!r}")
    if self.confidence not in _CONF_ORDER:
      raise ValueError(f"invalid confidence {self.confidence!r}")
    if self.truth_status not in {s.value for s in TruthStatus}:
      raise ValueError(f"invalid truth status {self.truth_status!r}")

  def downgraded(self) -> "Hypothesis":
    """Lower confidence one step when a contradiction is present."""
    idx = max(0, _CONF_ORDER.index(self.confidence) - 1)
    from dataclasses import replace
    return replace(self, confidence=_CONF_ORDER[idx])

  def to_json(self) -> dict[str, Any]:
    return {
      "hypothesis_id": self.hypothesis_id,
      "category": self.category,
      "statement": self.statement,
      "status": self.status,
      "truth_status": self.truth_status,
      "confidence": self.confidence,
      "supporting_fact_ids": list(self.supporting_fact_ids),
      "contradicting_fact_ids": list(self.contradicting_fact_ids),
      "missing_evidence": list(self.missing_evidence),
      "scope": self.scope,
    }


@dataclass(frozen=True)
class DiagnosisContext:
  evidence: KernelEvidence
  contrast: PairContrast | None = None


Rule = Callable[[DiagnosisContext], list[Hypothesis]]
_RULES: dict[str, list[tuple[str, Rule]]] = {}


def register_rule(category: str, rule_id: str, rule: Rule) -> None:
  bucket = _RULES.setdefault(category, [])
  if any(rid == rule_id for rid, _ in bucket):
    raise ValueError(f"diagnosis rule already registered: {category}/{rule_id}")
  bucket.append((rule_id, rule))


def rule(category: str, rule_id: str):
  def _wrap(fn: Rule) -> Rule:
    register_rule(category, rule_id, fn)
    return fn
  return _wrap


# --- fact helpers ------------------------------------------------------------

def _regression(contrast: PairContrast | None) -> bool:
  return contrast is not None and contrast.timing.get("qualified") == "regression"


def _num(v):
  return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# --- admission rules ---------------------------------------------------------

@rule("admission", "compile_unsupported")
def _compile_unsupported(ctx: DiagnosisContext) -> list[Hypothesis]:
  if ctx.evidence.stages.compile.status != "unsupported":
    return []
  return [Hypothesis("admission.compile_unsupported", "admission",
                     "the kernel form is not supported by the compiler",
                     SUPPORTED, TruthStatus.MEASURED.value, HIGH,
                     supporting_fact_ids=("stage.compile=unsupported",))]


@rule("admission", "compile_timeout")
def _compile_timeout(ctx: DiagnosisContext) -> list[Hypothesis]:
  if ctx.evidence.stages.compile.status != "timeout":
    return []
  return [Hypothesis("admission.compile_timeout", "admission",
                     "compilation did not terminate within the bound",
                     SUPPORTED, TruthStatus.MEASURED.value, HIGH,
                     supporting_fact_ids=("stage.compile=timeout",))]


@rule("admission", "identity_failure")
def _identity_failure(ctx: DiagnosisContext) -> list[Hypothesis]:
  bad = [b for b in ctx.evidence.blockers if b.scope == "identity"
         or b.code in ("identity_mismatch", "identity_corrupt")]
  if not bad:
    return []
  return [Hypothesis("admission.identity_failure", "admission",
                     "evidence identity is corrupted; the outcome is not attributable to the kernel",
                     SUPPORTED, TruthStatus.MEASURED.value, HIGH,
                     supporting_fact_ids=tuple(f"blocker.{b.code}" for b in bad))]


# --- liveness rules ----------------------------------------------------------

@rule("liveness", "timeout_with_sync")
def _timeout_with_sync(ctx: DiagnosisContext) -> list[Hypothesis]:
  ev = ctx.evidence
  if ev.stages.execution.status != "timeout":
    return []
  s = ev.structure
  barriers = _num(getattr(s, "barriers", None)) if s is not None else None
  waits = _num(getattr(s, "waits", None)) if s is not None else None
  if not barriers and not waits:
    # unknown-mechanism timeout
    return [Hypothesis("liveness.unknown_timeout", "liveness",
                       "execution timed out; the mechanism cannot be isolated from current evidence",
                       PLAUSIBLE, TruthStatus.MEASURED.value, CONF_UNKNOWN,
                       supporting_fact_ids=("stage.execution=timeout",),
                       missing_evidence=("per-wave progress trace", "barrier/wait counts"))]
  support = ["stage.execution=timeout"]
  if barriers:
    support.append(f"structure.barriers={barriers}")
  if waits:
    support.append(f"structure.waits={waits}")
  return [Hypothesis("liveness.timeout_with_sync", "liveness",
                     "the execution timeout is consistent with a synchronization/liveness issue "
                     "(barriers/waits present); this is not proof of a divergent barrier",
                     PLAUSIBLE, TruthStatus.DERIVED.value, LOW,
                     supporting_fact_ids=tuple(support),
                     missing_evidence=("barrier-arrival / divergence trace",
                                       "per-wave progress before hang"))]


@rule("liveness", "runtime_fault")
def _runtime_fault(ctx: DiagnosisContext) -> list[Hypothesis]:
  if ctx.evidence.stages.execution.status != "runtime_fault":
    return []
  return [Hypothesis("liveness.runtime_fault", "liveness",
                     "execution faulted at runtime",
                     SUPPORTED, TruthStatus.MEASURED.value, MEDIUM,
                     supporting_fact_ids=("stage.execution=runtime_fault",),
                     missing_evidence=("fault class / address",))]


@rule("liveness", "health_failure")
def _health_failure(ctx: DiagnosisContext) -> list[Hypothesis]:
  ev = ctx.evidence
  h = ev.health
  bad = ev.stages.execution.status == "health_failure" or (h is not None and h.postflight is False)
  if not bad:
    return []
  return [Hypothesis("liveness.health_failure", "liveness",
                     "a device health failure invalidates this execution; it cannot prove correctness or speed",
                     SUPPORTED, TruthStatus.MEASURED.value, MEDIUM,
                     supporting_fact_ids=("health.postflight=false",))]


@rule("liveness", "no_result")
def _no_result(ctx: DiagnosisContext) -> list[Hypothesis]:
  if ctx.evidence.stages.execution.status != "no_result":
    return []
  return [Hypothesis("liveness.no_result", "liveness",
                     "the harness returned no structured result; treat as a harness failure, not a kernel verdict",
                     PLAUSIBLE, TruthStatus.MEASURED.value, LOW,
                     supporting_fact_ids=("stage.execution=no_result",),
                     missing_evidence=("harness logs", "re-run under isolation"))]


# --- numerical rules ---------------------------------------------------------

@rule("numerical", "nan_inf")
def _nan_inf(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.evidence.correctness
  if c is None:
    return []
  nan = _num(c.nan_fraction) or 0
  inf = _num(c.inf_fraction) or 0
  if nan <= 0 and inf <= 0:
    return []
  return [Hypothesis("numerical.nan_inf", "numerical",
                     "output contains NaN/Inf; consistent with an uninitialized accumulator or numeric overflow",
                     SUPPORTED, TruthStatus.MEASURED.value, MEDIUM,
                     supporting_fact_ids=tuple(
                       f"correctness.{k}={v}" for k, v in (("nan_fraction", nan), ("inf_fraction", inf)) if v > 0),
                     missing_evidence=("which elements are non-finite (tile/row map)",))]


@rule("numerical", "mostly_zero")
def _mostly_zero(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.evidence.correctness
  if c is None or _num(c.zero_fraction) is None:
    return []
  if c.zero_fraction < MOSTLY_ZERO_THRESHOLD:
    return []
  return [Hypothesis("numerical.mostly_zero", "numerical",
                     "output is mostly zero; consistent with a bounded write/address defect "
                     "(missing store coverage, wrong output pointer, or an unwritten tile)",
                     PLAUSIBLE, TruthStatus.MEASURED.value, MEDIUM,
                     supporting_fact_ids=(f"correctness.zero_fraction={c.zero_fraction}",),
                     missing_evidence=("per-store ownership / coverage map",
                                       "output address vs reference layout"))]


@rule("numerical", "low_correlation")
def _low_correlation(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.evidence.correctness
  if c is None or _num(c.correlation) is None:
    return []
  if c.correlation >= LOW_CORRELATION_THRESHOLD:
    return []
  return [Hypothesis("numerical.low_correlation", "numerical",
                     "output correlates poorly with the reference; consistent with a structural (not scaling) error",
                     PLAUSIBLE, TruthStatus.MEASURED.value, LOW,
                     supporting_fact_ids=(f"correctness.correlation={c.correlation}",),
                     missing_evidence=("error localization (tile/row)",))]


@rule("numerical", "deterministic_wrong")
def _deterministic_wrong(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.evidence.correctness
  if c is None or c.status != "fail":
    return []
  # only when finite (NaN/Inf handled separately) and deterministic
  if (_num(c.nan_fraction) or 0) > 0 or (_num(c.inf_fraction) or 0) > 0:
    return []
  if c.sample_deterministic is False:
    return [Hypothesis("numerical.nondeterministic", "numerical",
                       "output is incorrect and nondeterministic across runs; consistent with a race or "
                       "uninitialized read",
                       PLAUSIBLE, TruthStatus.MEASURED.value, LOW,
                       supporting_fact_ids=("correctness.sample_deterministic=false",),
                       missing_evidence=("repeated-run variance map",))]
  support = ["correctness.status=fail"]
  if c.max_error is not None:
    support.append(f"correctness.max_error={c.max_error}")
  return [Hypothesis("numerical.deterministic_wrong", "numerical",
                     "output is deterministically incorrect; consistent with a fixed algorithmic/layout error",
                     SUPPORTED, TruthStatus.MEASURED.value, MEDIUM,
                     supporting_fact_ids=tuple(support),
                     missing_evidence=("error localization (tile/row)",))]


# --- performance rules (need a contrast) -------------------------------------

@rule("performance", "scratch_pressure")
def _scratch_pressure(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.contrast
  if not _regression(c):
    return []
  r = ctx.evidence.resources
  scratch = _num(getattr(r, "scratch_bytes", None)) if r is not None else None
  if not scratch:
    return []
  support = [f"resource.scratch_bytes={scratch}", "timing.qualified=regression"]
  contradicting = []
  occ = c.resource_deltas.get("occupancy") if c else None
  if occ and _num(occ.get("abs_delta")) is not None and occ["abs_delta"] > 0:
    contradicting.append("resource.occupancy=increased")
  h = Hypothesis("performance.scratch_pressure", "performance",
                 "scratch spilling accompanies the measured regression (register/occupancy pressure)",
                 SUPPORTED, TruthStatus.DERIVED.value, MEDIUM,
                 supporting_fact_ids=tuple(support), contradicting_fact_ids=tuple(contradicting),
                 missing_evidence=("register allocation report",), scope="pair")
  return [h.downgraded() if contradicting else h]


@rule("performance", "occupancy")
def _occupancy(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.contrast
  if not _regression(c) or c is None:
    return []
  occ = c.resource_deltas.get("occupancy")
  if not occ or _num(occ.get("abs_delta")) is None:
    return []
  if occ["abs_delta"] >= 0:
    return []  # occupancy did not fall; this rule doesn't fire
  return [Hypothesis("performance.occupancy", "performance",
                     "lower occupancy accompanies the measured regression (resource pressure)",
                     SUPPORTED, TruthStatus.DERIVED.value, MEDIUM,
                     supporting_fact_ids=("resource.occupancy=decreased", "timing.qualified=regression"),
                     missing_evidence=("occupancy-limiting resource attribution",), scope="pair")]


@rule("performance", "synchronization")
def _synchronization(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.contrast
  if not _regression(c) or c is None:
    return []
  if "barriers increased" not in c.structure_deltas:
    return []
  return [Hypothesis("performance.synchronization", "performance",
                     "added synchronization accompanies the regression; a barrier-cost contribution is plausible",
                     PLAUSIBLE, TruthStatus.DERIVED.value, LOW,
                     supporting_fact_ids=("structure.barriers=increased", "timing.qualified=regression"),
                     missing_evidence=("per-barrier stall attribution",), scope="pair")]


@rule("performance", "memory_residency")
def _memory_residency(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.contrast
  if not _regression(c) or c is None:
    return []
  if "LDS increased" not in c.structure_deltas:
    return []
  return [Hypothesis("performance.memory_residency", "performance",
                     "increased LDS staging accompanies the regression; staging cost may exceed reuse benefit",
                     PLAUSIBLE, TruthStatus.DERIVED.value, LOW,
                     supporting_fact_ids=("structure.LDS=increased", "timing.qualified=regression"),
                     missing_evidence=("per-operand reuse vs staging-cost model",), scope="pair")]


@rule("performance", "noise")
def _noise(ctx: DiagnosisContext) -> list[Hypothesis]:
  c = ctx.contrast
  if c is None or c.timing.get("qualified") != "tie":
    return []
  return [Hypothesis("performance.noise", "performance",
                     "the timing difference is within the combined noise floor; no performance change is established",
                     SUPPORTED, TruthStatus.DERIVED.value, MEDIUM,
                     supporting_fact_ids=("timing.qualified=tie", "timing.noisy_tie=true"),
                     scope="pair")]


@rule("mechanism", "operand_contradiction")
def _operand_contradiction(ctx: DiagnosisContext) -> list[Hypothesis]:
  out = []
  for operand, path in ctx.evidence.operand_paths:
    contradictions = path.classification.contradicting_evidence_ids
    if path.declared_strategy == path.classification.strategy and not contradictions:
      continue
    support = (f"operand.{operand}.declared={path.declared_strategy}",
               f"operand.{operand}.classified={path.classification.strategy}")
    out.append(Hypothesis(
      f"mechanism.{operand}.strategy_contradiction", "mechanism",
      "the declared operand strategy is not established by the classified final-program evidence",
      CONTRADICTED, TruthStatus.DERIVED.value, path.classification.confidence,
      supporting_fact_ids=support, contradicting_fact_ids=contradictions,
      missing_evidence=path.classification.missing_discriminators,
      scope=f"operand:{operand}"))
  return out


@rule("performance", "operand_spill")
def _operand_spill(ctx: DiagnosisContext) -> list[Hypothesis]:
  if not _regression(ctx.contrast):
    return []
  out = []
  for operand, row in (ctx.contrast.operand_deltas if ctx.contrast else {}).items():
    spill = row.get("spill_bytes", {})
    delta = _num(spill.get("abs_delta"))
    if delta is None or delta <= 0:
      continue
    out.append(Hypothesis(
      f"performance.{operand}.spill", "performance",
      "increased operand-attributed spill traffic accompanies the measured regression",
      SUPPORTED, TruthStatus.DERIVED.value, MEDIUM,
      supporting_fact_ids=(f"operand.{operand}.spill_bytes=increased", "timing.qualified=regression"),
      missing_evidence=("isolated spill-free A/B with the same operand strategy",), scope=f"operand:{operand}"))
  return out


@rule("performance", "byte_amplification")
def _byte_amplification(ctx: DiagnosisContext) -> list[Hypothesis]:
  if not _regression(ctx.contrast):
    return []
  out = []
  for operand, row in (ctx.contrast.operand_deltas if ctx.contrast else {}).items():
    delta = _num(row.get("byte_amplification", {}).get("abs_delta"))
    if delta is None or delta <= 0:
      continue
    out.append(Hypothesis(
      f"performance.{operand}.byte_amplification", "performance",
      "higher transferred-to-useful byte amplification accompanies the measured regression",
      PLAUSIBLE, TruthStatus.DERIVED.value, LOW,
      supporting_fact_ids=(f"operand.{operand}.byte_amplification=increased", "timing.qualified=regression"),
      missing_evidence=("controlled per-operand traffic A/B with calibrated transfer bytes",),
      scope=f"operand:{operand}"))
  return out


def diagnose(evidence: KernelEvidence, contrast: PairContrast | None = None) -> tuple[Hypothesis, ...]:
  """Run every registered rule; return the hypotheses it produced."""
  ctx = DiagnosisContext(evidence=evidence, contrast=contrast)
  out: list[Hypothesis] = []
  for category in _RULES:
    for _rule_id, fn in _RULES[category]:
      out.extend(fn(ctx))
  return tuple(out)
