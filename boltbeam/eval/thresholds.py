"""Speed-movement tiers. The thresholds are stated choices in data/policy.json, read through core.policy."""
from __future__ import annotations

from dataclasses import dataclass

from boltbeam.core.policy import value
from boltbeam.vocab import Tier

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

  A win must exceed the noise band when spread is known. Speed-equivalent is the equal_speed_pct band. A regression is
  a negative delta beyond the equivalent band. Below tier_b_min_pct is diagnostic (Tier.C).
  """
  if spread_pct is not None and abs(delta_pct) < spread_pct * NOISE_QUALIFY_SPREAD_MULT:
    return TierResult(Tier.NONE.value, is_win=False, is_regression=False, is_equivalent=True)
  if abs(delta_pct) <= value("equal_speed_pct"):
    return TierResult(Tier.NONE.value, is_win=False, is_regression=False, is_equivalent=True)
  if delta_pct < 0:
    return TierResult(Tier.NONE.value, is_win=False, is_regression=True, is_equivalent=False)
  if delta_pct >= value("tier_a_min_pct"):
    return TierResult(Tier.A.value, is_win=True, is_regression=False, is_equivalent=False)
  if delta_pct >= value("tier_b_min_pct"):
    return TierResult(Tier.B.value, is_win=True, is_regression=False, is_equivalent=False)
  return TierResult(Tier.C.value, is_win=False, is_regression=False, is_equivalent=False)


def is_protected_regression(delta_pct:float) -> bool:
  return delta_pct < -value("protected_regression_max_pct")
