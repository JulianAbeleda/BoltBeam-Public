"""How many tokens to draft per step (speculative decoding's k), from measured decode speeds: the one rule the
recommended setup, tune and the draft-count sweep read. Engine-blind: speeds per context and k in, a k and why out.
A speed is (per_prompt, value): per_prompt {prompt: tok/s} when measured on prompts, else None (a formula figure).

Measured figures are judged by their own prompts, paired: a larger k is faster when the bootstrap (prompts resampled,
seeded) lower bound of the median per-prompt ratio is above 1, and could still be faster while its upper bound is.
No tie constant: the prompts' own spread says what one measurement can tell apart. Formula figures have no prompts,
so they keep a stated default margin, FORMULA_TIE, said as such."""
from __future__ import annotations

import random, statistics
from typing import Any

FORMULA_TIE = 0.05  # a stated default for figures with no per-prompt data (formula figures), never for measured ones
K_MAX = 16  # a guard on the sweep's time (one engine start per k), not a measured limit; reaching it is said. The best k at a long context is unmeasured: the long depth that reached 10 was a repetition loop (collectors/mtp.py)
LEVEL, RESAMPLES = 0.95, 2000  # one-sided confidence of each bound, bootstrap resamples (seed 0)


def bounds(new:tuple[Any, float], old:tuple[Any, float], level:float = LEVEL) -> tuple[float, float]:
  """The one-sided `level` lower and upper bounds of how much faster `new` is than `old`: the paired bootstrap of the
  median per-prompt ratio when both ran the same prompts; else the value ratio less and plus FORMULA_TIE. A caller
  comparing several candidates at once passes a corrected level (corrected)."""
  if new[0] and set(new[0]) == set(old[0] or {}):
    ratios, rng = [new[0][p] / old[0][p] for p in sorted(new[0])], random.Random(0)
    meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(RESAMPLES))
    return meds[int((1 - level) * RESAMPLES)], meds[int(level * RESAMPLES) - 1]
  r = new[1] / old[1]
  return r / (1 + FORMULA_TIE), r * (1 + FORMULA_TIE)


def corrected(m:int) -> float:
  """The level each of m comparisons is held to so that all m together keep LEVEL (Bonferroni)."""
  return 1 - (1 - LEVEL) / max(1, m)


def best(speeds:dict[int, tuple[Any, float | None]]) -> int | None:
  """From the smallest k > 0 up, a larger k replaces the pick only when it is faster (lower bound above 1)."""
  ks = sorted(k for k, v in speeds.items() if k > 0 and v[1])
  pick = ks[0] if ks else None
  for k in ks[1:]:
    pick = k if bounds(speeds[k], speeds[pick])[0] > 1 else pick
  return pick


def open_at(depths:dict[float, dict[int, tuple]]) -> list[float]:
  """The contexts where a larger k than the largest measured could still win: the largest is the pick there, or its
  upper bound against the pick is above 1."""
  top = max(k for s in depths.values() for k in s)
  return sorted(c for c, s in depths.items() if top in s and (best(s) == top or bounds(s[top], s[best(s)])[1] > 1))


def sweep_next(depths:dict[float, dict[int, tuple]]) -> tuple[int | None, str]:
  """The next k to measure and why, or None and why the sweep is done: one more while any context is open, to K_MAX."""
  top, still = max(k for s in depths.values() for k in s), open_at(depths)
  if not still:
    return None, f"k={top} cannot beat the best k at any context measured"
  if top >= K_MAX:
    return None, f"stopped at k={K_MAX}, the sweep's cap, with k={top} still able to win at {still[0]:,.0f} tokens"
  return top + 1, f"k={top} could still win at {', '.join(f'{c:,.0f}' for c in still)} tokens"


def choose(depths:dict[float, dict[int, tuple]], context:int | None, fits, *, seqs:int = 1,
           limit:int | None = None) -> dict[str, Any] | None:
  """The draft count for a context and why it is or is not used. depths: {context measured: {k: speed}}, k=0 the
  baseline; fits(measured, asked): whether a measured depth is the asked context's own. Its own depth gives its best
  k; between two measured depths the smaller of their two best ks ("between", both said); past either end the end's
  ("beyond", "below"). "open": the largest k measured there could still win, so the best may be higher; "forgone":
  a larger k with a higher median that was not taken (forgone). Held: batch
  > 1, a context past the engine's MTP limit, or no win over k=0. None with no k > 0."""
  ctxs = sorted(depths)
  picks = {c: best(depths[c]) for c in ctxs}
  if not ctxs or all(k is None for k in picks.values()):
    return None
  near = [c for c in ctxs if context is not None and fits(c, context)]
  if context is None or near:
    at, asked = (near or ctxs)[0], "measured" if near else None
  elif context > ctxs[-1] or context < ctxs[0]:
    at, asked = (ctxs[-1], "beyond") if context > ctxs[-1] else (ctxs[0], "below")
  else:
    lo, hi = max(c for c in ctxs if c < context), min(c for c in ctxs if c > context)
    at, asked = (lo if picks[lo] <= picks[hi] else hi), "between"
  k = picks[at]
  held = "batch" if seqs != 1 else "context_limit" if limit and context and context > limit else \
    "k0" if bounds(depths[at][k], depths[at][0])[0] <= 1 else None
  return {"k": k, "at": at, "asked": asked, "picks": picks, "held": held, "open": at in open_at({at: depths[at]}),
          "forgone": forgone(depths[at], k)}


def forgone(speeds:dict[int, tuple], k:int) -> dict[str, Any] | None:
  """The larger k with the highest median that the pick did not take, its median and on how many prompts it was
  faster than the pick; None when no larger k has a higher median."""
  more = [j for j in speeds if j > k and speeds[j][1] and speeds[j][1] > speeds[k][1]]
  if not more:
    return None
  j = max(more, key=lambda x: speeds[x][1])
  a, b = speeds[j][0] or {}, speeds[k][0] or {}
  return {"k": j, "tok_s": speeds[j][1], "wins": sum(a[p] > b[p] for p in a if p in b), "n": len(a)}
