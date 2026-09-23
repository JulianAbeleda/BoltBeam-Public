"""The ONE tier/threshold table (audit-brain-build-scope BB5). No thresholds live anywhere else."""
from __future__ import annotations

from dataclasses import dataclass

from boltbeam.vocab import Tier


# speed-movement tiers, in percent whole-workload gain
TIER_A_MIN_PCT = 5.0        # broad win, default-promotable if guardrails pass
TIER_B_MIN_PCT = 2.0        # residual win, promotable with clean rollback + no protected regression
# below TIER_B is diagnostic (Tier.C) unless strategic

# a candidate that is within +/- this band of baseline is "speed-equivalent" (promotable if it reduces
# hand-written surface / is purer), not a win and not a regression
SPEED_EQUIVALENT_BAND_PCT = 1.0

# any protected-context regression worse than this fails the guardrail
PROTECTED_REGRESSION_MAX_PCT = 1.0

# measurement noise floor: a |delta| smaller than the observed spread is inconclusive, not a win
NOISE_QUALIFY_SPREAD_MULT = 1.0


@dataclass(frozen=True)
class TierResult:
  tier: str
  is_win: bool
  is_regression: bool
  is_equivalent: bool


def classify_tier(delta_pct:float, *, spread_pct:float | None = None) -> TierResult:
  """Map a measured whole-workload %delta (candidate vs baseline) to a tier.

  A win must exceed the noise band when spread is known. Speed-equivalent is the +/-1% band. A regression is a
  negative delta beyond the equivalent band.
  """
  if spread_pct is not None and abs(delta_pct) < spread_pct * NOISE_QUALIFY_SPREAD_MULT:
    return TierResult(Tier.NONE.value, is_win=False, is_regression=False, is_equivalent=True)
  if abs(delta_pct) <= SPEED_EQUIVALENT_BAND_PCT:
    return TierResult(Tier.NONE.value, is_win=False, is_regression=False, is_equivalent=True)
  if delta_pct < 0:
    return TierResult(Tier.NONE.value, is_win=False, is_regression=True, is_equivalent=False)
  if delta_pct >= TIER_A_MIN_PCT:
    return TierResult(Tier.A.value, is_win=True, is_regression=False, is_equivalent=False)
  if delta_pct >= TIER_B_MIN_PCT:
    return TierResult(Tier.B.value, is_win=True, is_regression=False, is_equivalent=False)
  return TierResult(Tier.C.value, is_win=False, is_regression=False, is_equivalent=False)


def is_protected_regression(delta_pct:float) -> bool:
  return delta_pct < -PROTECTED_REGRESSION_MAX_PCT
