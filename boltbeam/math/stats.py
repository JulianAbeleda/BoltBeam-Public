"""Small-sample statistics with no dependency: Student's t quantile, a mean's t interval, Welch's test, a robust
outlier bound. Exact enough for a 95% interval (Simpson's rule on the t density, bisection on its CDF)."""
from __future__ import annotations

import functools
import math
import statistics


def _t_cdf(x:float, df:int) -> float:
  """P(T <= x) for x >= 0: one half plus the density integrated from 0 to x."""
  c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
  f = lambda u: c * (1 + u * u / df) ** (-(df + 1) / 2)  # noqa: E731
  n = 2000
  h = x / n
  s = f(0) + f(x) + sum((4 if k % 2 else 2) * f(k * h) for k in range(1, n))
  return 0.5 + s * h / 3


@functools.lru_cache(maxsize=None)
def t_quantile(p:float, df:int) -> float:
  """The p quantile (p > 0.5) of Student's t with df degrees of freedom."""
  if df < 1:
    raise ValueError("t needs at least one degree of freedom")
  lo, hi = 0.0, 1.0
  while _t_cdf(hi, df) < p:
    hi *= 2
  for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if _t_cdf(mid, df) < p else (lo, mid)
  return (lo + hi) / 2


def mean_ci(xs:list[float], level:float = 0.95) -> dict[str, float | int | None]:
  """The mean and its two-sided t interval; no interval from one value."""
  n = len(xs)
  mu = statistics.fmean(xs) if xs else None
  if n < 2:
    return {"mean": mu, "lo": None, "hi": None, "n": n}
  h = t_quantile(0.5 + level / 2, n - 1) * statistics.stdev(xs) / math.sqrt(n)
  return {"mean": mu, "lo": mu - h, "hi": mu + h, "n": n}


def welch_differs(a:list[float], b:list[float], floor:float = 0.0, level:float = 0.95) -> bool | None:
  """Whether two groups' means differ beyond Welch's two-sided interval and beyond floor (a sensor's step); None
  when a group has fewer than two values."""
  if len(a) < 2 or len(b) < 2:
    return None
  va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
  d = abs(statistics.fmean(b) - statistics.fmean(a))
  if va + vb == 0:
    return d > floor
  df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1)) + (vb ** 2 / (len(b) - 1)))
  return d > t_quantile(0.5 + level / 2, max(1, int(df))) * math.sqrt(va + vb) and d > floor


def outlier_bound(xs:list[float], family:float = 0.05, resolution:float = 0.0) -> float | None:
  """The value above which one of len(xs) samples is an outlier at the family-wise level: the median plus the
  normal quantile for family / n times the MAD's sigma (never under the measure's resolution)."""
  if len(xs) < 3:
    return None
  med = statistics.median(xs)
  sigma = max(1.4826 * statistics.median(abs(x - med) for x in xs), resolution)
  return med + statistics.NormalDist().inv_cdf(1 - family / len(xs)) * sigma
