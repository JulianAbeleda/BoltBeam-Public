"""Pure roofline and Amdahl math (audit-brain-build-scope BB9).

Everything is a plain function of caller-supplied numbers. There are deliberately NO model dimensions, NO target
identifiers, and NO baked-in bandwidths in this file: a decode roofline for any model on any target is just
bytes-read and a bandwidth. Keeping the math constant-free is what lets the same code cover decode, prefill, a
7900 XTX, an H100, or a hypothetical target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boltbeam.vocab import Confidence, PracticalRooflineClass, SCHEMA_CEILING_REPORT


def dual_roofline_placement(*, measured:float, raw_ceiling:float, practical_ceiling:float,
                            metric:str = "tok_s", raw_basis:str | None = None,
                            practical_basis:str | None = None) -> dict[str, Any]:
  """Place one measured throughput against both a raw hardware and practical measured ceiling.

  All values use the same higher-is-better metric. The practical ceiling must be no higher than the raw ceiling;
  its basis is supplied by the caller (for example a sustained-copy measurement or a same-route kernel oracle).
  """
  for name, value in (("measured", measured), ("raw_ceiling", raw_ceiling),
                      ("practical_ceiling", practical_ceiling)):
    _require_positive(name, value)
  if practical_ceiling > raw_ceiling:
    raise ValueError(f"practical_ceiling must be <= raw_ceiling, got {practical_ceiling} > {raw_ceiling}")
  if not metric:
    raise ValueError("metric must be non-empty")

  def _line(kind:str, ceiling:float, basis:str | None) -> dict[str, Any]:
    return {
      "kind": kind,
      "ceiling": ceiling,
      "metric": metric,
      "basis": basis,
      "measured_pct_of_ceiling": 100.0 * measured / ceiling,
      "headroom_x": ceiling / measured,
      "absolute_gap": ceiling - measured,
    }

  return {
    "schema": "boltbeam.dual_roofline_placement.v1",
    "metric": metric,
    "measured": measured,
    "raw_hardware": _line("raw_hardware", raw_ceiling, raw_basis),
    "practical": _line("practical", practical_ceiling, practical_basis),
  }


def floor_ms_per_token(bytes_read:float, bandwidth_bytes_per_s:float) -> float:
  """Lower bound on milliseconds per token from a bandwidth roofline.

  A memory-bound step cannot finish faster than the time it takes to stream `bytes_read` at
  `bandwidth_bytes_per_s`. Returns that time in milliseconds.
  """
  _require_positive("bytes_read", bytes_read)
  _require_positive("bandwidth_bytes_per_s", bandwidth_bytes_per_s)
  return (bytes_read / bandwidth_bytes_per_s) * 1000.0


def ceiling_tok_s(bytes_read:float, bandwidth_bytes_per_s:float) -> float:
  """Upper bound on tokens per second implied by the same roofline (1000 / floor_ms)."""
  return 1000.0 / floor_ms_per_token(bytes_read, bandwidth_bytes_per_s)


def amdahl_whole_gain(role_share_fraction:float, role_speedup:float) -> float:
  """Whole-model speedup from making a role `role_speedup`x faster.

  `role_share_fraction` is the fraction of the whole wall the role currently occupies. By Amdahl's law the new
  wall is (1 - share) + share / speedup, so the whole-model gain is the reciprocal. `role_speedup = inf` models
  a role driven to zero cost, giving the classic 1 / (1 - share) hard ceiling.
  """
  if not 0.0 <= role_share_fraction <= 1.0:
    raise ValueError(f"role_share_fraction must be in [0, 1], got {role_share_fraction!r}")
  if role_speedup <= 0.0:
    raise ValueError(f"role_speedup must be > 0 (use float('inf') for 'to zero'), got {role_speedup!r}")
  remaining = (1.0 - role_share_fraction) + (role_share_fraction / role_speedup)
  if remaining <= 0.0:
    raise ValueError("degenerate: role occupies the whole wall and is driven to zero (unbounded gain)")
  return 1.0 / remaining


@dataclass(frozen=True)
class CeilingReport:
  """Derived roofline + Amdahl ceiling for a single workload step.

  `memcpy_peak_bandwidth` is the hardware streaming peak (e.g. a raw device-copy benchmark). The floor and
  ceiling are the theoretical roofline computed against that peak. `dequant_achievable_bandwidth` is the
  distinct, lower effective bandwidth a real dequant+GEMV route sustains; keeping the two apart stops an audit
  from claiming a route is "at the roofline" when it is only near what a copy could do.
  """
  bytes_read:float
  memcpy_peak_bandwidth:float
  dequant_achievable_bandwidth:float
  floor_ms:float
  ceiling_tok_s:float
  role_share_fraction:float
  role_speedup:float
  amdahl_projected_gain:float
  assumptions:tuple[str, ...] = ()
  confidence:str = Confidence.DERIVED.value

  def to_json(self) -> dict[str, Any]:
    return {
      "schema": SCHEMA_CEILING_REPORT,
      "bytes_read": self.bytes_read,
      "memcpy_peak_bandwidth": self.memcpy_peak_bandwidth,
      "dequant_achievable_bandwidth": self.dequant_achievable_bandwidth,
      "floor_ms": self.floor_ms,
      "ceiling_tok_s": self.ceiling_tok_s,
      "role_share_fraction": self.role_share_fraction,
      "role_speedup": self.role_speedup,
      "amdahl_projected_gain": self.amdahl_projected_gain,
      "assumptions": list(self.assumptions),
      "confidence": self.confidence,
    }


def build_ceiling_report(bytes_read:float, memcpy_peak_bandwidth:float, dequant_achievable_bandwidth:float,
                         role_share_fraction:float, role_speedup:float, assumptions:tuple[str, ...] = (),
                         confidence:str = Confidence.DERIVED.value) -> CeilingReport:
  """Assemble a CeilingReport from raw bytes, both bandwidths, and an Amdahl (share, speedup) pair.

  The floor/ceiling roofline is taken against the hardware `memcpy_peak_bandwidth`; the distinct
  `dequant_achievable_bandwidth` is carried alongside so callers can compare a route's real sustained bandwidth
  against the peak the floor assumes.
  """
  _require_positive("dequant_achievable_bandwidth", dequant_achievable_bandwidth)
  if confidence not in {c.value for c in Confidence}:
    raise ValueError(f"unknown confidence {confidence!r}")
  floor = floor_ms_per_token(bytes_read, memcpy_peak_bandwidth)
  return CeilingReport(
    bytes_read=bytes_read,
    memcpy_peak_bandwidth=memcpy_peak_bandwidth,
    dequant_achievable_bandwidth=dequant_achievable_bandwidth,
    floor_ms=floor,
    ceiling_tok_s=1000.0 / floor,
    role_share_fraction=role_share_fraction,
    role_speedup=role_speedup,
    amdahl_projected_gain=amdahl_whole_gain(role_share_fraction, role_speedup),
    assumptions=tuple(assumptions),
    confidence=confidence,
  )


@dataclass(frozen=True)
class PracticalRooflinePoint:
  """One protected-context practical-roofline measurement.

  `practical_ceiling_tok_s` is the best measured same-scope implementation, not a raw hardware peak. The score
  answers whether the candidate is close enough to that practical ceiling that further route-family work is
  unlikely to pay.
  """
  candidate_tok_s: float
  practical_ceiling_tok_s: float
  context: int | None = None
  measured_spread_pct: float | None = None

  @property
  def pct_of_practical(self) -> float:
    _require_positive("candidate_tok_s", self.candidate_tok_s)
    _require_positive("practical_ceiling_tok_s", self.practical_ceiling_tok_s)
    return 100.0 * self.candidate_tok_s / self.practical_ceiling_tok_s


@dataclass(frozen=True)
class PracticalRooflineResult:
  """Conservative reduction of practical-roofline points across protected contexts."""
  classification: str
  p_worst_pct: float
  g_worst_pp: float
  a_worst_pp: float
  worst_context: int | None
  replacement_objective: bool
  action: str
  min_pct: float
  closeout_gap_pp: float

  def to_json(self) -> dict[str, Any]:
    return {
      "classification": self.classification,
      "P_worst_pct": self.p_worst_pct,
      "G_worst_pp": self.g_worst_pp,
      "A_worst_pp": self.a_worst_pp,
      "worst_context": self.worst_context,
      "replacement_objective": self.replacement_objective,
      "action": self.action,
      "min_pct": self.min_pct,
      "closeout_gap_pp": self.closeout_gap_pp,
    }


def practical_roofline_score(points: tuple[PracticalRooflinePoint, ...] | list[PracticalRooflinePoint], *,
                             replacement_objective: bool = False, min_pct: float = 98.0,
                             closeout_gap_pp: float = 2.0) -> PracticalRooflineResult:
  """Score a candidate against a practical promotion roofline.

  For each protected context:

    P = candidate / practical ceiling * 100
    G = 100 - P
    A = max(0, G - max(closeout_gap, measured spread))

  `A_worst == 0` means no actionable route-family headroom remains after the closeout tolerance/noise band.
  Promotion still requires the evaluator's hard guardrails and an explicit replacement objective.
  """
  if not points:
    return PracticalRooflineResult(PracticalRooflineClass.INCONCLUSIVE.value, 0.0, 0.0, 0.0, None,
                                   replacement_objective, "inconclusive", min_pct, closeout_gap_pp)
  _require_positive("min_pct", min_pct)
  _require_positive("closeout_gap_pp", closeout_gap_pp)
  reduced: list[tuple[float, float, float, int | None]] = []
  for p in points:
    pct = p.pct_of_practical
    gap = 100.0 - pct
    spread = float(p.measured_spread_pct) if p.measured_spread_pct is not None else 0.0
    if spread < 0.0:
      raise ValueError(f"measured_spread_pct must be >= 0, got {spread!r}")
    actionable = max(0.0, gap - max(closeout_gap_pp, spread))
    reduced.append((pct, gap, actionable, p.context))

  p_worst = min(r[0] for r in reduced)
  g_worst = max(r[1] for r in reduced)
  a_worst = max(r[2] for r in reduced)
  worst_context = min(reduced, key=lambda r: r[0])[3]
  if p_worst >= min_pct and a_worst == 0.0:
    cls = PracticalRooflineClass.PROMOTABLE.value if replacement_objective else PracticalRooflineClass.CLOSEOUT.value
    action = "promote" if replacement_objective else "closeout"
  else:
    cls = PracticalRooflineClass.HEADROOM_REMAINS.value
    action = "headroom-remains"
  return PracticalRooflineResult(cls, p_worst, g_worst, a_worst, worst_context, replacement_objective, action,
                                 min_pct, closeout_gap_pp)


def _require_positive(name:str, value:float) -> None:
  if value <= 0.0:
    raise ValueError(f"{name} must be > 0, got {value!r}")



def floor_us(bytes_read:float, peak_gbs:float) -> float:
  """Lower bound in microseconds to stream `bytes_read` at `peak_gbs` (bytes / GB/s / 1000)."""
  return bytes_read / peak_gbs / 1000.0


def effective_gbs(bytes_read:float, wall_us:float) -> float:
  """Achieved bandwidth in GB/s from bytes moved over a wall time in microseconds."""
  return bytes_read / wall_us / 1000.0


def bandwidth_utilization(bytes_read:float, wall_us:float, peak_gbs:float) -> float:
  """Achieved bandwidth as a fraction of peak (0 when wall time is zero)."""
  if wall_us <= 0:
    return 0.0
  return effective_gbs(bytes_read, wall_us) / peak_gbs
