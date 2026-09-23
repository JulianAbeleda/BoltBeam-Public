"""KA-5: provider-neutral pairwise contrast.

Computes what changed structurally and measurably between a candidate and a comparator. It never
claims causality (that is diagnosis) and never fabricates a speedup when the pair is not
performance-eligible. Timing statistics are recomputed from raw samples; a difference inside the
combined noise floor is a tie, not a win.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from boltbeam.kernel_analysis.eligibility import performance_eligibility
from boltbeam.kernel_analysis.model import KernelEvidence, Measurement
from boltbeam.perf.fit import bootstrap_median_interval

# Semantic axes: (accessor, human noun). Direction words are derived, never causal.
_RESOURCE_AXES = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "scratch_loads", "scratch_stores",
                  "spilled_vgprs", "spilled_sgprs", "occupancy", "waves")
_STRUCTURE_AXES = ("barriers", "waits", "global_loads", "global_stores", "shared_loads",
                   "shared_stores", "tensor_ops", "loops", "pipeline_stages", "buffering",
                   "branches", "predicates", "instructions")
_AXIS_NOUN = {
  "lds_bytes": "LDS", "occupancy": "occupancy", "barriers": "barriers", "waits": "waits",
  "global_loads": "global loads", "global_stores": "global stores", "shared_loads": "shared loads",
  "shared_stores": "shared stores", "scratch_bytes": "scratch", "vgpr": "VGPR", "sgpr": "SGPR",
  "scratch_loads": "scratch loads", "scratch_stores": "scratch stores",
  "spilled_vgprs": "spilled VGPRs", "spilled_sgprs": "spilled SGPRs",
  "waves": "waves", "tensor_ops": "tensor ops", "loops": "loops", "pipeline_stages": "pipeline stages",
  "buffering": "buffering", "branches": "branches", "predicates": "predicates", "instructions": "instructions",
}


def median(samples: tuple[float, ...]) -> float | None:
  return statistics.median(samples) if samples else None


def mad(samples: tuple[float, ...]) -> float | None:
  """Median absolute deviation — a robust, deterministic spread estimate."""
  if not samples:
    return None
  med = statistics.median(samples)
  return statistics.median(tuple(abs(s - med) for s in samples))


def timing_interval(samples: tuple[float, ...]) -> dict[str, float] | None:
  """Deterministic median interval. Uses the repository bootstrap policy at >=30 samples,
  otherwise a median +/- MAD band (clamped at zero). Returns None without samples."""
  if not samples:
    return None
  if len(samples) >= 30:
    low, med, high = bootstrap_median_interval(samples, rounds=1000, seed=0)
    return {"low": low, "median": med, "high": high, "method": "bootstrap"}
  med = statistics.median(samples)
  spread = mad(samples) or 0.0
  return {"low": max(0.0, med - spread), "median": med, "high": med + spread, "method": "mad"}


@dataclass(frozen=True)
class PairContrast:
  candidate_id: str
  comparator_id: str
  performance_eligible: bool
  blockers: tuple[str, ...]
  identity_equality: dict[str, str]
  workload_equal: bool
  stage_diffs: dict[str, dict[str, str]]
  correctness_diffs: dict[str, Any]
  resource_deltas: dict[str, dict[str, Any]]
  structure_deltas: tuple[str, ...]
  timing: dict[str, Any]
  mechanism: dict[str, Any]
  operand_deltas: dict[str, Any] = field(default_factory=dict)
  used_facts: tuple[str, ...] = ()
  missing_facts: tuple[str, ...] = ()

  def to_json(self) -> dict[str, Any]:
    return {
      "candidate_id": self.candidate_id,
      "comparator_id": self.comparator_id,
      "performance_eligible": self.performance_eligible,
      "blockers": list(self.blockers),
      "identity_equality": self.identity_equality,
      "workload_equal": self.workload_equal,
      "stage_diffs": self.stage_diffs,
      "correctness_diffs": self.correctness_diffs,
      "resource_deltas": self.resource_deltas,
      "structure_deltas": list(self.structure_deltas),
      "timing": self.timing,
      "mechanism": self.mechanism,
      "operand_deltas": self.operand_deltas,
      "used_facts": list(self.used_facts),
      "missing_facts": list(self.missing_facts),
    }


_IDENTITY_AXES = ("provider", "compiler", "backend", "target_id", "device_id",
                  "system_snapshot_id", "session_id", "experiment_id")


def _identity_equality(a: KernelEvidence, b: KernelEvidence) -> dict[str, str]:
  out: dict[str, str] = {}
  for axis in _IDENTITY_AXES:
    av, bv = getattr(a.identity, axis), getattr(b.identity, axis)
    if av is None or bv is None:
      out[axis] = "unknown"
    else:
      out[axis] = "equal" if av == bv else "mismatch"
  return out


def _stage_diffs(a: KernelEvidence, b: KernelEvidence) -> dict[str, dict[str, str]]:
  out = {}
  for stage in ("compile", "execution", "correctness", "timing"):
    sa, sb = getattr(a.stages, stage).status, getattr(b.stages, stage).status
    if sa != sb:
      out[stage] = {"candidate": sa, "comparator": sb}
  return out


def _correctness_diffs(a: KernelEvidence, b: KernelEvidence) -> dict[str, Any]:
  ca, cb = a.correctness, b.correctness
  out: dict[str, Any] = {}
  if ca is None or cb is None:
    out["available"] = False
    return out
  out["available"] = True
  out["status"] = {"candidate": ca.status, "comparator": cb.status}
  for f in ("max_error", "mean_error", "zero_fraction", "nan_fraction", "inf_fraction"):
    va, vb = getattr(ca, f), getattr(cb, f)
    if va is not None and vb is not None and va != vb:
      out[f] = {"candidate": va, "comparator": vb}
  return out


def _num(v):
  return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _resource_deltas(a: KernelEvidence, b: KernelEvidence,
                     used: list[str], missing: list[str]) -> dict[str, dict[str, Any]]:
  ra, rb = a.resources, b.resources
  out: dict[str, dict[str, Any]] = {}
  for axis in _RESOURCE_AXES:
    va = _num(getattr(ra, axis, None)) if ra is not None else None
    vb = _num(getattr(rb, axis, None)) if rb is not None else None
    if va is None or vb is None:
      missing.append(f"resource.{axis}")
      continue
    used.append(f"resource.{axis}")
    abs_delta = va - vb
    pct = (abs_delta / vb * 100.0) if vb else None
    out[axis] = {"candidate": va, "comparator": vb, "abs_delta": abs_delta, "pct_delta": pct}
  return out


def _structure_deltas(a: KernelEvidence, b: KernelEvidence,
                      used: list[str], missing: list[str]) -> tuple[str, ...]:
  phrases: list[str] = []
  sa, sb = a.structure, b.structure
  # LDS/occupancy live on resources but read as structural change to a human.
  ra, rb = a.resources, b.resources
  pairs = [(axis, _num(getattr(ra, axis, None)) if ra is not None else None,
            _num(getattr(rb, axis, None)) if rb is not None else None) for axis in ("lds_bytes", "occupancy")]
  pairs += [(axis, _num(getattr(sa, axis, None)) if sa is not None else None,
             _num(getattr(sb, axis, None)) if sb is not None else None) for axis in _STRUCTURE_AXES]
  for axis, va, vb in pairs:
    if va is None or vb is None:
      continue
    used.append(f"structure.{axis}")
    if va == vb:
      continue
    direction = "increased" if va > vb else "decreased"
    phrases.append(f"{_AXIS_NOUN.get(axis, axis)} {direction}")
  return tuple(phrases)


def _mechanism(a: KernelEvidence, b: KernelEvidence) -> dict[str, Any]:
  ta, tb = a.candidate.transport_map, b.candidate.transport_map
  operands = sorted(set(ta) | set(tb))
  if not operands:
    return {"status": "unknown", "detail": "no operand transports declared"}
  changed = [op for op in operands if ta.get(op) != tb.get(op)]
  if not changed:
    return {"status": "absent", "detail": "candidate and comparator declare identical transports"}
  return {"status": "present", "changed_operands": changed,
          "candidate": {op: ta.get(op) for op in changed},
          "comparator": {op: tb.get(op) for op in changed}}


def _amplification(path) -> float | None:
  useful = _num(path.dynamic.useful_bytes)
  transferred = _num(path.dynamic.transferred_bytes)
  return transferred / useful if useful and transferred is not None else None


def _operand_deltas(a: KernelEvidence, b: KernelEvidence,
                    used: list[str], missing: list[str]) -> dict[str, Any]:
  """Contrast orthogonal per-operand facts without turning them into causal claims."""
  pa, pb = dict(a.operand_paths), dict(b.operand_paths)
  out: dict[str, Any] = {}
  for operand in sorted(set(pa) | set(pb)):
    ca, cb = pa.get(operand), pb.get(operand)
    if ca is None or cb is None:
      missing.append(f"operand.{operand}.path")
      out[operand] = {"available": False,
                      "missing_side": "candidate" if ca is None else "comparator"}
      continue
    used.append(f"operand.{operand}.path")
    row: dict[str, Any] = {
      "available": True,
      "declared_strategy": {"candidate": ca.declared_strategy, "comparator": cb.declared_strategy},
      "classified_strategy": {"candidate": ca.classification.strategy,
                              "comparator": cb.classification.strategy},
      "classification_confidence": {"candidate": ca.classification.confidence,
                                    "comparator": cb.classification.confidence},
    }
    tiers_a = {t.tier: t.to_json() for t in ca.dynamic.serving_tiers}
    tiers_b = {t.tier: t.to_json() for t in cb.dynamic.serving_tiers}
    row["serving_tiers"] = {"candidate": tiers_a, "comparator": tiers_b}
    amp_a, amp_b = _amplification(ca), _amplification(cb)
    row["byte_amplification"] = {
      "candidate": amp_a, "comparator": amp_b,
      "abs_delta": amp_a - amp_b if amp_a is not None and amp_b is not None else None,
    }
    if amp_a is None or amp_b is None:
      missing.append(f"operand.{operand}.byte_amplification")
    spill_a, spill_b = _num(ca.static.spill_bytes), _num(cb.static.spill_bytes)
    row["spill_bytes"] = {
      "candidate": spill_a, "comparator": spill_b,
      "abs_delta": spill_a - spill_b if spill_a is not None and spill_b is not None else None,
    }
    if spill_a is None or spill_b is None:
      missing.append(f"operand.{operand}.spill_bytes")
    out[operand] = row
  return out


def _timing_contrast(a: KernelEvidence, b: KernelEvidence, performance_eligible: bool,
                     used: list[str], missing: list[str]) -> dict[str, Any]:
  ta: Measurement | None = a.timing
  tb: Measurement | None = b.timing
  sa = ta.normalized_samples_ms if ta is not None else ()
  sb = tb.normalized_samples_ms if tb is not None else ()
  # Summary-only timing (no raw samples) is diagnostic, not comparison-grade.
  if not sa or not sb:
    if not sa:
      missing.append("timing.candidate_samples")
    if not sb:
      missing.append("timing.comparator_samples")
    return {"qualified": "unavailable", "reason": "missing raw timing samples on one or both candidates"}
  if not performance_eligible:
    return {"qualified": "unavailable", "reason": "pair is not performance-eligible; no speedup computed"}

  used.append("timing.candidate_samples")
  used.append("timing.comparator_samples")
  med_a = statistics.median(sa)
  med_b = statistics.median(sb)
  noise = max(mad(sa) or 0.0, mad(sb) or 0.0)
  delta = med_a - med_b
  speedup = (med_b / med_a) if med_a > 0 else None
  pct = (delta / med_b * 100.0) if med_b else None
  noisy_tie = abs(delta) <= noise
  if noisy_tie:
    qualified = "tie"
  elif delta < 0:
    qualified = "win"        # candidate faster (lower ms)
  else:
    qualified = "regression"
  return {
    "qualified": qualified,
    "candidate_median_ms": med_a,
    "comparator_median_ms": med_b,
    "candidate_interval_ms": timing_interval(sa),
    "comparator_interval_ms": timing_interval(sb),
    "delta_ms": delta,
    "speedup": speedup,
    "percent_delta": pct,
    "noise_floor_ms": noise,
    "noisy_tie": noisy_tie,
  }


def contrast_pair(candidate: KernelEvidence, comparator: KernelEvidence) -> PairContrast:
  """Provider-neutral contrast of `candidate` against `comparator`."""
  perf = performance_eligibility(candidate, comparator=comparator)
  used: list[str] = []
  missing: list[str] = []
  workload_equal = candidate.workload.to_json() == comparator.workload.to_json()
  resource_deltas = _resource_deltas(candidate, comparator, used, missing)
  structure_deltas = _structure_deltas(candidate, comparator, used, missing)
  timing = _timing_contrast(candidate, comparator, perf.eligible, used, missing)
  operand_deltas = _operand_deltas(candidate, comparator, used, missing)
  return PairContrast(
    candidate_id=candidate.candidate.candidate_id,
    comparator_id=comparator.candidate.candidate_id,
    performance_eligible=perf.eligible,
    blockers=tuple(b.requirement_id for b in perf.blockers),
    identity_equality=_identity_equality(candidate, comparator),
    workload_equal=workload_equal,
    stage_diffs=_stage_diffs(candidate, comparator),
    correctness_diffs=_correctness_diffs(candidate, comparator),
    resource_deltas=resource_deltas,
    structure_deltas=structure_deltas,
    timing=timing,
    mechanism=_mechanism(candidate, comparator),
    operand_deltas=operand_deltas,
    used_facts=tuple(used),
    missing_facts=tuple(dict.fromkeys(missing)),
  )
