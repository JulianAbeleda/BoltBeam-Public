"""Resolve the ACHIEVED streaming ceiling that a roofline is measured against — the denominator, done right.

`derive_roofline_plan` takes `achieved_gbs` as a given. Getting that number wrong silently invalidates every
downstream conclusion (a too-low ceiling makes a near-optimal kernel look like it has headroom; a too-high one
makes real headroom vanish). This module turns a set of streaming-copy samples into the correct denominator, baking
in two mistakes that each cost real measurement to learn:

  1. USE THE SUSTAINED CEILING, NOT A COLD BURST. On a power/thermal-managed target the memory clock RAMPS UP under
     continuous load — a few-iteration "cold" copy reads LOWER than the sustained rate a whole-model decode actually
     sees. (Measured on a 7900 XTX-class part: cold burst ~720-766 GB/s, sustained ~829 GB/s.) The decode runs
     sustained, so the sustained copy is the honest ceiling. The inverse regime — sustained materially BELOW cold —
     is real too (small parts that thermally throttle), and it is a THERMAL lever, not a codegen one; we flag it
     rather than silently using a throttled number.
  2. REJECT A WEAK COPY. A poorly-written copy kernel undershoots HBM and reads BELOW a real compute kernel's own
     achieved GB/s — physically impossible for the true ceiling (no kernel can exceed the streaming ceiling). If the
     best copy sample is under a known reference kernel's GB/s, the copy is the bottleneck, not the memory: the
     sample is discarded and the reference kernel becomes the floor on the true ceiling.

Pure functions of caller-supplied GB/s samples. No device access, no target constants — the measurement itself
lives in the target harness; the METHOD of turning samples into a trustworthy denominator lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# sustained must beat cold by at least this fraction to call it a clock-ramp regime (vs flat / noise).
_RAMP_EPS = 0.03
# sustained below cold by more than this fraction is a genuine sustained-throttle regime (a thermal lever).
_THROTTLE_EPS = 0.05


@dataclass(frozen=True)
class CeilingResolution:
  achieved_gbs: float                    # THE denominator to feed derive_roofline_plan(achieved_gbs=...)
  regime: str                            # clocks_ramp_up | sustained_throttle | flat
  cold_gbs: float | None
  sustained_gbs: float | None
  raw_peak_gbs: float | None
  pct_of_raw_peak: float | None
  reference_floor_gbs: float | None      # a real kernel's GB/s that the ceiling must not fall below
  warnings: list[str] = field(default_factory=list)

  def as_dict(self) -> dict[str, Any]:
    return {"schema": "boltbeam.roofline_ceiling.v1", "achieved_gbs": self.achieved_gbs, "regime": self.regime,
            "cold_gbs": self.cold_gbs, "sustained_gbs": self.sustained_gbs, "raw_peak_gbs": self.raw_peak_gbs,
            "pct_of_raw_peak": self.pct_of_raw_peak, "reference_floor_gbs": self.reference_floor_gbs,
            "warnings": list(self.warnings)}


def _best(samples:list[float] | None) -> float | None:
  vals = [float(s) for s in (samples or []) if s and s > 0]
  return max(vals) if vals else None


def resolve_achieved_ceiling(*, cold_samples:list[float] | None = None, sustained_samples:list[float] | None = None,
                             raw_peak_gbs:float | None = None,
                             reference_kernel_gbs:float | None = None) -> CeilingResolution:
  """Turn streaming-copy samples into the achieved ceiling to use as a roofline denominator.

  cold_samples / sustained_samples: measured copy GB/s (read+write counted) from a SHORT burst and a LONG sustained
  loop respectively — the harness measures both; take the best (fastest) of each here. raw_peak_gbs: the spec HBM
  peak, for context only (never the denominator). reference_kernel_gbs: the highest achieved GB/s of any REAL
  compute kernel you have measured — a hard floor on the true ceiling (nothing streams faster than the memory).
  Returns the denominator + regime + warnings. Prefers sustained; guards against cold-undershoot and a weak copy.
  """
  cold, sustained = _best(cold_samples), _best(sustained_samples)
  warnings: list[str] = []

  # pick the streaming ceiling: sustained is the honest one; fall back to cold with a warning if that's all we have.
  if sustained is not None and cold is not None:
    if sustained >= cold * (1.0 + _RAMP_EPS):
      regime, achieved = "clocks_ramp_up", sustained
    elif sustained <= cold * (1.0 - _THROTTLE_EPS):
      regime, achieved = "sustained_throttle", sustained
      warnings.append(f"sustained ({sustained:.0f}) is materially below cold ({cold:.0f}) GB/s: the target THROTTLES "
                      "under continuous load. This is a THERMAL/power lever, not a codegen one — the sustained number "
                      "is the real decode ceiling, but chasing codegen against it won't recover the throttled gap.")
    else:
      regime, achieved = "flat", max(sustained, cold)
  elif sustained is not None:
    regime, achieved = "flat", sustained
  elif cold is not None:
    regime, achieved = "flat", cold
    warnings.append("only a COLD burst was provided; a whole-model decode runs SUSTAINED and typically sees a higher "
                    "clock. Measure a long sustained copy loop — this denominator likely UNDERSHOOTS the real ceiling.")
  else:
    raise ValueError("need at least one of cold_samples / sustained_samples")

  # weak-copy guard: no copy can read slower than a real kernel achieves. If it does, the copy is the bottleneck.
  reference_floor = float(reference_kernel_gbs) if reference_kernel_gbs and reference_kernel_gbs > 0 else None
  if reference_floor is not None and achieved < reference_floor:
    warnings.append(f"the streaming-copy sample ({achieved:.0f} GB/s) is BELOW a real kernel's achieved "
                    f"{reference_floor:.0f} GB/s — impossible for the true ceiling, so the copy kernel is weak and "
                    "undershoots HBM. Using the reference kernel's GB/s as the ceiling floor; write a better copy.")
    achieved = reference_floor

  pct_peak = round(100.0 * achieved / raw_peak_gbs, 1) if raw_peak_gbs else None
  if raw_peak_gbs and achieved > raw_peak_gbs:
    warnings.append(f"achieved ({achieved:.0f}) exceeds the stated raw peak ({raw_peak_gbs:.0f}) GB/s — check the "
                    "byte count (a dequant kernel's LOGICAL bytes over-count) or the peak spec.")

  return CeilingResolution(achieved_gbs=round(achieved, 1), regime=regime,
                           cold_gbs=round(cold, 1) if cold else None,
                           sustained_gbs=round(sustained, 1) if sustained else None,
                           raw_peak_gbs=raw_peak_gbs, pct_of_raw_peak=pct_peak,
                           reference_floor_gbs=round(reference_floor, 1) if reference_floor else None,
                           warnings=warnings)
