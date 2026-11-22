"""The paired non-inferiority rule (stage 2 of Speed up's gate, owner-approved 2026-11-14), in one place: quantization/
speed_up reads it with its table's threshold, search/identity judges a lossy kernel's quality evidence by it. Pure:
no model, engine or GPU; th is data/speed_up.json's threshold block."""
from __future__ import annotations

import math
import statistics
from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.math.stats import t_quantile


def z(confidence:float, df:int | None = None) -> float:
  """The one-sided quantile at confidence: normal, or Student t with df degrees of freedom (math/stats.t_quantile:
  its CDF integrated and inverted, exact to the 1e-3 a gate reads at every df: 6.314 at df 1, 2.920 at df 2, 1.895 at
  df 7, 0.95; a series in 1/df is not at df 1-2)."""
  return t_quantile(confidence, df) if df else statistics.NormalDist().inv_cdf(confidence)


def paired(a:list[dict], b:list[dict]) -> dict[str, Any]:
  """Two arms of a paired eval (rows id, correct, finish): b items only a got right, c only b; refused unless both arms
  answered the same items, each once, every row whole."""
  for rows in (a, b):
    if not rows or any(not isinstance(r, dict) or not isinstance(r.get("correct"), bool) or "id" not in r or "finish" not in r for r in rows):
      raise Refused("an eval arm is empty or has a row without id, a true/false correct and finish")
    if len({r["id"] for r in rows}) != len(rows):
      raise Refused("an eval arm answers an item twice")
  x, y = ({r["id"]: r for r in rows} for rows in (a, b))
  if x.keys() != y.keys():
    raise Refused(f"the two eval arms did not answer the same items: {len(x.keys() ^ y.keys())} are in one arm only")
  return {"n": len(x), "b": sum(x[i]["correct"] > y[i]["correct"] for i in x), "c": sum(y[i]["correct"] > x[i]["correct"] for i in x),
          "truncated": [sum(r["finish"] == "length" for r in rows) for rows in (a, b)]}


def noninferior(original:list[dict], copy:list[dict], noise:list[dict], th:dict[str, Any], tolerance:str | float | None = None, how:str = "on its noise run") -> dict[str, Any]:
  """Stage 2: the copy's paired eval loss against the engine's own eval noise (noise: the original run again under the
  engine's declared change, perturbation(), or as it was when none is declared), on the same items. The two pairs share
  the original, so L_copy - L_noise is exactly D = mean(noise_i - copy_i), each item 1 right, 0 wrong. The standard
  non-inferiority rule against the tolerance, computed exactly per item: u_i = (noise_i - copy_i) - t_i, t_i = |noise_i -
  orig_i| for "psi0" (so mean(u) = D - psi0, the allowance measured on the same items and its correlation with D carried
  in u's own SE), t_i = the share for a fixed tolerance (0: shown no worse than the noise run). fail: mean(u) - z SE_u > 0
  (proven worse than allowed); pass: mean(u) + z SE_u <= 0; else inconclusive, with the n at which an equal copy passes at
  threshold.eval.power, (z_a + z_b)^2 var(u) / allowance^2, as a hint. SE_u = sqrt(var(u) / n). tolerance
  (threshold.eval.tolerance): "psi0", the engine's own gross churn, the share of answers its noise run changes against
  the original (refused when 0: nothing to measure the copy against); 0; or a fixed share. Truncation (finish=length)
  over threshold.eval.max_truncation in any arm: inconclusive, BoltBeam's eval budget rule. A "run N more" hint re-rolls
  all three arms if they are rerun: repeating the whole eval until it passes inflates the error rate, so run the noise
  arm once and add the new items to all three arms together."""
  ev, conf, tol = th["eval"], th["confidence"], th["eval"]["tolerance"] if tolerance is None else tolerance
  p, p0, d = paired(original, copy), paired(original, noise), paired(noise, copy)
  n, churn, za = d["n"], (p0["b"] + p0["c"]) / d["n"], z(conf)
  if tol == "psi0" and not churn:
    raise Refused("this engine shows no eval noise: its run of the original again changed no answer, so there is nothing to measure the copy "
                  "against; declare a perturbation for it (threshold.noise.engines)")
  o, c, y = ({r["id"]: int(r["correct"]) for r in rows} for rows in (original, copy, noise))
  u = [y[i] - c[i] - (abs(y[i] - o[i]) if tol == "psi0" else float(tol)) for i in o]
  allow, loss, mean = churn if tol == "psi0" else float(tol), (d["b"] - d["c"]) / n, sum(u) / n
  var = sum((x - mean) ** 2 for x in u) / n
  se, cut = math.sqrt(var / n), max(x / n for x in p["truncated"] + p0["truncated"])
  need = math.ceil((za + z(ev["power"])) ** 2 * var / allow ** 2) if allow else None
  verdict = ("inconclusive" if cut > ev["max_truncation"] else "fail" if mean - za * se > 0 else "pass" if mean + za * se <= 0 else "inconclusive")
  said = (f"the engine's own churn, {allow * 100:.1f} points on this model (answers it changes against itself {how})" if tol == "psi0" else
          "none: a pass is shown no worse than the noise run" if not allow else f"a fixed {allow * 100:.1f} points")
  return {"verdict": verdict, "n": n, "copy": p, "noise": p0, "against_noise": d, "loss": loss, "margin": mean, "se": se, "z": za, "churn": churn, "tolerance": tol,
          "allowed": allow, "allowed_said": said, "n_needed": need, "power": ev["power"], "confidence": conf,
          "why": (f"truncation {cut:.1%} is over {ev['max_truncation']:.0%}" if cut > ev["max_truncation"] else None if verdict != "inconclusive" else
                  f"underpowered: run {need - n} more items, added to all three arms (rerunning the whole eval until it passes inflates the error rate)" if need and need > n else "its loss is neither shown within the allowance nor shown beyond it")}


def _mean_se(x:list[float], floor:float | None = None) -> tuple[float, float]:
  """Mean and its standard error sqrt(s^2 / n), s^2 the sample variance (n - 1: the t at n - 1 df it is read with);
  floor: the SE when s^2 is 0."""
  n = len(x)
  if n < 2:
    raise Refused(f"{n} value{'s' * (n != 1)}: a standard error needs at least 2")
  m = sum(x) / n
  se = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n)
  return m, (floor if floor is not None and se == 0 else se)


def noise_pair(original:list[dict], copy:list[dict], noise:list[dict], th:dict[str, Any]) -> dict[str, Any]:
  """The lossy gate's quality rule (owner 2026-11-21, threshold.lossy): the copy's paired loss against the original,
  non-inferior within a margin taken from the engine's own numerics-only noise pair (noise: the original again under
  its declared change) on the same items. Per item (1 right, 0 wrong) loss_i = orig_i - copy_i, noise_i = orig_i -
  noise_i; d, d0 their means, SE sqrt(s^2 / n) (sample variance; 1/n, one item's worth, when no item differs). margin = |d0| +
  z(two-sided) SE0: the noise pair's own upper bound on its effect. pass: d + z(one-sided) SE <= margin; fail: d - z SE
  > margin (shown worse than the margin); else inconclusive. Truncation over threshold.eval.max_truncation in any arm:
  inconclusive. It replaces the A1 margin 1.96 sqrt(2 p (1 - p) q / n): that is 1.96 s / sqrt(n) against a test of d +
  1.645 s / sqrt(n), so a true null fails 1 - Phi(0.315) = 37.6% of the time at any n."""
  conf = th["confidence"]
  p, p0 = paired(original, copy), paired(original, noise)
  if p0["n"] != p["n"] or {r["id"] for r in noise} != {r["id"] for r in copy}:
    raise Refused("the noise run did not answer the copy's items")
  o, c, y = ({r["id"]: int(r["correct"]) for r in rows} for rows in (original, copy, noise))
  n = p["n"]
  if n < 2:
    raise Refused(f"{n} paired item{'s' * (n != 1)}: the noise pair's margin needs at least 2")
  if all(o[i] == y[i] for i in o):
    raise Refused(f"the noise run changed no answer of the {n} against the original: its declared change did nothing on these items, so there is "
                  "no measured noise to take a margin from (a floor of one item's worth would be a picked margin); run the noise arm with a change "
                  "that moves the engine's numerics")
  d, se = _mean_se([o[i] - c[i] for i in o], 1 / n)
  d0, se0 = _mean_se([o[i] - y[i] for i in o], 1 / n)
  z1, z2 = z(conf), z(0.5 + conf / 2)
  margin, cut = abs(d0) + z2 * se0, max(x / n for x in p["truncated"] + p0["truncated"])
  verdict = ("inconclusive" if cut > th["eval"]["max_truncation"] else "pass" if d + z1 * se <= margin else "fail" if d - z1 * se > margin else "inconclusive")
  return {"verdict": verdict, "n": n, "copy": p, "noise": p0, "loss": d, "se": se, "noise_loss": d0, "noise_se": se0, "margin": margin,
          "worse_bound": d + z1 * se, "z": z1, "z_margin": z2, "confidence": conf,
          "why": (f"truncation {cut:.1%} is over {th['eval']['max_truncation']:.0%}" if cut > th["eval"]["max_truncation"] else None if verdict != "inconclusive" else
                  "its loss is neither shown within the noise pair's margin nor shown beyond it")}


FLOAT32 = 2.0 ** -24   # a listed probability's rounding (llama-server's logprobs are float32), relative to 1
ABSENT = 0.0           # the probability of a reference token a top list does not hold (it is in [0, the list's least]): dp is then a bound


def divergence(copy:list[dict], noise:list[dict], th:dict[str, Any]) -> dict[str, Any]:
  """The lossy gate's token-probability check (owner 2026-11-21; llama.cpp tools/perplexity's KL-divergence standard):
  the copy's and the noise run's divergence from the original, teacher-forced on the same positions, in blocks of
  consecutive positions (rows id, positions, kld: mean KL(original || run) over the block, dp: mean change of the
  reference token's probability, run less original; search/identity.judged makes them with blocks() from the raw rows).
  Blocks are the unit (positions in a block are not independent). Two checks, each the noise_pair form with Student t
  at k - 1 df for k blocks: kld, the copy's mean + t(one-sided) SE at or below the noise run's mean + t(two-sided) SE
  (its own upper bound); skew, the copy's loss of the reference token's probability (-dp, worse when positive) + t SE
  at or below |the noise run's| + t(two-sided) SE. Symmetric probability changes are noise; a skew toward worse beyond
  the noise run's own is harm. At least threshold.lossy.min_items blocks (the owner's 100 independent units): the line's
  t(two-sided) less the copy's t(one-sided) is the test's slack, 0.315 in the large-k limit, 6.39 at k 2 and 0.47 at k
  8, so fewer blocks pass a copy as divergent as the noise run more often (docs/region-search.md, the derivation).
  A noise run with no spread (SE 0) did nothing at this stride and is refused: there is no line to hold the copy to.
  Rows with rest (blocks()'s: the mass outside both top lists, which the KL's partition does not see) are also held to
  the noise run's: the copy's mean rest above the noise run's mean + t(two-sided) SE is refused, so the copy's KL lower
  bound is compared to the noise run's on no more unseen mass than the noise run's own upper bound on it."""
  conf, least = th["confidence"], th["lossy"]["min_items"]
  for name, rows in (("copy", copy), ("noise", noise)):
    if not rows or any(not isinstance(r, dict) or "id" not in r or not isinstance(r.get("positions"), int) or r["positions"] < 1
                       or not all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) and math.isfinite(r[k]) for k in ("kld", "dp"))
                       or r["kld"] < 0 or not -1 <= r["dp"] <= 1 for r in rows):
      raise Refused(f"the {name} run's divergence has a block without an id, a whole positions count, a kld >= 0 or a dp in [-1, 1]")
    if len({r["id"] for r in rows}) != len(rows):
      raise Refused(f"the {name} run's divergence names a block twice")
  x, y = ({r["id"]: r for r in rows} for rows in (copy, noise))
  if x.keys() != y.keys() or any(x[i]["positions"] != y[i]["positions"] for i in x):
    raise Refused("the copy's and the noise run's divergence are not on the same blocks of positions")
  k = len(x)
  if k < least:
    raise Refused(f"the divergence has {k} blocks of positions, under the {least} the lossy gate requires (threshold.lossy.min_items): with fewer, "
                  "the noise run's line loosens faster than the copy's bound, so less data would pass more easily; teacher-force more positions")
  t1, t2 = z(conf, k - 1), z(0.5 + conf / 2, k - 1)
  if all(isinstance(r.get("rest"), (int, float)) for r in (*copy, *noise)):   # blocks()'s rows: the mass the KL's lower bound does not see
    m, _ = _mean_se([x[i]["rest"] for i in x])
    m0, se0 = _mean_se([y[i]["rest"] for i in x])
    if m > m0 + t2 * se0:
      raise Refused(f"the copy's top lists leave {m:.3g} of the probability outside the tokens both lists hold (mean by block), over the noise run's "
                    f"{m0:.3g} + t {t2:.3f} SE {se0:.3g}: the KL's lower bound would not see what the copy moves there (disjoint lists read as KL ~0); "
                    "teacher-force again with more n_probs")
  out = {"blocks": k, "positions": sum(r["positions"] for r in copy), "t": t1, "t_line": t2, "confidence": conf}
  for key, sign in (("kld", 1), ("dp", -1)):
    m, se = _mean_se([sign * x[i][key] for i in x])
    m0, se0 = _mean_se([sign * y[i][key] for i in x])
    if se0 == 0:
      raise Refused(f"the noise run's {key} is {sign * m0:.4g} in every block (SE 0): its declared change did nothing at this stride, so there is no "
                    "noise line to hold the copy to; run the noise arm with a change that moves the engine's numerics at the kernel's shape")
    line = (m0 if key == "kld" else abs(m0)) + t2 * se0
    out[key] = {"copy": sign * m, "copy_se": se, "noise": sign * m0, "noise_se": se0, "line": line, "worse_bound": m + t1 * se,
                "verdict": "pass" if m + t1 * se <= line else "fail"}
  out["verdict"] = "pass" if out["kld"]["verdict"] == out["dp"]["verdict"] == "pass" else "fail"
  return out


def blocks(base:list[dict], run:list[dict], size:int, stride:int | None = None, keep:set[int] | None = None) -> dict[str, Any]:
  """divergence()'s rows from teacher-forced distributions (rows pos, ref, top [[id, logprob]], prompt_n; the original's
  base, the run's run, on the same positions): per position KL(base || run) on the partition {each token in both top
  lists} + {the rest, one bucket} (exact on that partition, a lower bound on the full-vocabulary KL), and dp, the
  reference token's probability change, run less base. Positions whose prompt_n is not stride (default: the most common;
  a cache miss) are dropped in both, and any not in keep (search/identity.judged: the positions every arm hit); blocks
  of size consecutive positions, the last partial one dropped. Explicit, not silent:
    rest   a mass at or under FLOAT32 x the lists' length is rounding: as 0 (p log p/q -> 0 as p -> 0). A base rest above
           that where the run's is not is refused: the original puts probability where the run's list leaves none, the
           KL there is unbounded at this n_probs (teacher-force again with more). A block's rest: the mean of max(base's,
           run's) mass outside the tokens both lists hold, the probability the KL's lower bound does not see
           (divergence() holds the copy's to the noise run's).
    ref    a reference token a list does not hold counts ABSENT (0; its probability is in [0, the list's least]): dp is
           then a bound, and ref_outside_top counts the positions where either list lacks it.
  Also llama.cpp's summary figures (tools/perplexity): mean, median, p99, max KL, same top token, RMS dp."""
  stride = statistics.mode(r["prompt_n"] for r in base) if stride is None else stride
  b, r = ({x["pos"]: x for x in rows if x["prompt_n"] == stride and (keep is None or x["pos"] in keep)} for rows in (base, run))
  pos = sorted(b.keys() & r.keys())
  if not pos:
    raise Refused(f"no teacher-forced position both runs hit with an append of {stride} tokens")
  kl, dp, rest, same, miss = [], [], [], 0, 0
  for i in pos:
    p, q = ({t: math.exp(lp) for t, lp in x[i]["top"]} for x in (b, r))
    both, guard = p.keys() & q.keys(), FLOAT32 * (len(p) + len(q))
    pr, qr = max(1 - sum(p[t] for t in both), 0.0), max(1 - sum(q[t] for t in both), 0.0)
    if pr > guard >= qr:
      raise Refused(f"position {i}: the original puts {pr:.3g} of its probability outside the tokens both top lists hold and the run's list leaves "
                    f"none there (under {guard:.2g}, rounding): the KL is unbounded at {len(p)} listed tokens; teacher-force again with more n_probs")
    kl.append(max(sum(p[t] * math.log(p[t] / q[t]) for t in both) + (pr * math.log(pr / qr) if pr > guard else 0.0), 0.0))
    rest.append(max(x if x > guard else 0.0 for x in (pr, qr)))   # rounding is 0, as above
    ref = b[i]["ref"]
    dp.append(q.get(ref, ABSENT) - p.get(ref, ABSENT))
    miss += ref not in p or ref not in q
    same += b[i]["top"][0][0] == r[i]["top"][0][0]
  rows = [{"id": k, "positions": size, **{f: sum(v[k * size:(k + 1) * size]) / size for f, v in (("kld", kl), ("dp", dp), ("rest", rest))}}
          for k in range(len(pos) // size)]
  s = sorted(kl)
  return {"blocks": rows, "positions": len(pos), "dropped": len(base) - len(pos), "ref_outside_top": miss,
          "kld": {"mean": sum(kl) / len(kl), "median": s[len(s) // 2], "p99": s[min(len(s) - 1, int(0.99 * len(s)))], "max": s[-1]},
          "same_top": same / len(pos), "dp_mean": sum(dp) / len(dp), "dp_rms": math.sqrt(sum(x * x for x in dp) / len(dp))}


__all__ = ["ABSENT", "FLOAT32", "blocks", "divergence", "noise_pair", "noninferior", "paired", "z"]
