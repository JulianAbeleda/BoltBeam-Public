"""A session benchmark's analysis: identity, conditions, contamination, per-turn and per-band paired deltas, a table.

Everything is read from the record collectors/session_ab.py wrote; nothing here measures.

  identity        each arm's replies against the base arm's first valid run, turn by turn (sha256): "59/59"
  conditions      the arms' start conditions (GPU temperature, load, other programs' CPU, the clock time) compared
                  arm against base with Welch's test (95%, Bonferroni over conditions and arms) beyond each sensor's
                  step, and the power limit exactly; a
                  comparison whose conditions differ is refused, never averaged
  contamination   a turn is contaminated when other programs' CPU during it lies beyond the run's own outlier bound
                  (median + MAD sigma at a 5% family-wise level over every turn of the run); the clean deltas drop a
                  turn contaminated in either arm of a pair
  bands           the warm turns (turn 1 is cold, reported apart) cut into quartiles of their prompt depth
  deltas          arm less base, per pair (pair k = block k's run of each), mean and 95% t interval over the pairs
"""
from __future__ import annotations

import math
import statistics
from typing import Any

from boltbeam.math.stats import mean_ci, outlier_bound, welch_differs

METRICS = (("prefill_ms", "prefill ms"), ("ttft_ms", "TTFT ms"), ("pre_server_ms", "pre-server ms"),
           ("decode_tok_s", "decode tok/s"))
BANDS = 4  # depth quartiles of the warm turns
# (condition, step): the sensor's resolution (a GPU temperature is read in whole C, load1 in hundredths); start_s is
# the arm's start in seconds since the epoch (the clock time, safe across midnight)
CONDITIONS = (("temp_start_c", 1.0), ("load1_start", 0.01), ("foreign_cores", 0.0), ("start_s", 0.0))



def _words(key:str, **kw:Any) -> str:
  from boltbeam.report.plain import words
  return words()["session_ab"][key].format(**kw)


def per_turn(t:dict[str, Any]) -> dict[str, float | None]:
  pm, tt = t.get("prompt_ms"), t.get("ttft_ms")
  return {"prefill_ms": pm, "ttft_ms": tt, "pre_server_ms": tt - pm if pm is not None and tt is not None else None,
          "wall_ms": t.get("wall_ms")}


def band_value(turns:list[dict[str, Any]], keep:set[int], metric:str) -> float | None:
  """A metric over the kept turns: the mean of a time, decode as all tokens over all decode time."""
  ts = [t for t in turns if t["i"] in keep]
  if metric == "decode_tok_s":
    ts = [t for t in ts if t.get("predicted_ms") and t.get("predicted_n")]
    ms = sum(t["predicted_ms"] for t in ts)
    return sum(t["predicted_n"] for t in ts) / ms * 1e3 if ms else None
  xs = [v for t in ts if (v := per_turn(t)[metric]) is not None]
  return statistics.fmean(xs) if xs else None


def bands(turns:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Turn 1 apart, the warm turns in BANDS quartiles of prompt depth, and all warm turns."""
  warm = sorted((t for t in turns if t["i"] > 1), key=lambda t: (t.get("prompt_tokens") or 0, t["i"]))
  out = [{"label": "turn 1", "turns": {1}}] if any(t["i"] == 1 for t in turns) else []
  for k in range(BANDS):
    part = warm[k * len(warm) // BANDS:(k + 1) * len(warm) // BANDS]
    if part:
      out.append({"label": f"<={(part[-1].get('prompt_tokens') or 0) / 1e3:.0f}k", "turns": {t["i"] for t in part}})
  out.append({"label": "warm", "turns": {t["i"] for t in warm}})
  return out


def arm_conditions(a:dict[str, Any]) -> dict[str, float | None]:
  fc = [t["foreign_cores"] for t in a.get("turns") or [] if t.get("foreign_cores") is not None]
  return {"temp_start_c": (a.get("gate") or {}).get("temp_c"), "load1_start": (a.get("start") or {}).get("load1"),
          "foreign_cores": statistics.fmean(fc) if fc else None, "start_s": (a.get("start") or {}).get("epoch"),
          "power_limit_w": (a.get("gate") or {}).get("power_limit_w")}


def condition_mismatches(base:list[dict[str, Any]], arm:list[dict[str, Any]], tests:int = 1) -> list[str]:
  """tests: how many comparisons the record makes in all; each condition is tested at 5% over all of them
  (Bonferroni), so matching arms are not refused by chance as conditions and arms are added."""
  out, level = [], 1 - 0.05 / (tests * len(CONDITIONS))
  ca, cb = [arm_conditions(a) for a in base], [arm_conditions(a) for a in arm]
  limits = {c["power_limit_w"] for c in ca + cb if c["power_limit_w"] is not None}
  if len(limits) > 1:
    out.append(_words("cond_cap", limits=", ".join(f"{x:.0f} W" for x in sorted(limits))))
  for key, step in CONDITIONS:
    xa, xb = [c[key] for c in ca if c[key] is not None], [c[key] for c in cb if c[key] is not None]
    if welch_differs(xa, xb, step, level):
      out.append(_words("cond_differs", what=key, a=statistics.fmean(xa), b=statistics.fmean(xb)))
  return out


def idle_bound(arms:list[dict[str, Any]], family:float = 0.05) -> dict[str, Any] | None:
  """The bound on other programs' CPU in a turn from the machine's own idle: the start gate's samples before each arm
  (one a GATE_INTERVAL_S, nothing of ours running). baseline: their mean; SE: their SD scaled to a typical turn's length
  (a turn's figure is a mean over its seconds), never under two ticks over it; bound: baseline + z SE, z the normal
  quantile for family / turns (family-wise over the run's turns). None with under 3 samples."""
  from boltbeam.collectors.session_ab import GATE_INTERVAL_S, HZ
  idle = [x for a in arms for x in (a.get("gate") or {}).get("foreign_cores") or [] if x is not None]
  secs = [t["wall_ms"] / 1e3 for a in arms for t in a.get("turns") or [] if t.get("wall_ms")]
  n = sum(1 for a in arms for t in a.get("turns") or [] if t.get("foreign_cores") is not None)
  if len(idle) < 3 or not secs or not n:
    return None
  turn = statistics.median(secs)
  se = max(statistics.stdev(idle) / math.sqrt(max(turn / GATE_INTERVAL_S, 1.0)), 2 / (HZ * turn))
  base = statistics.fmean(idle)
  z = statistics.NormalDist().inv_cdf(1 - family / n)
  return {"baseline_cores": base, "se_cores": se, "z": z, "bound_cores": base + z * se, "samples": len(idle), "source": "idle"}


def contamination(arms:list[dict[str, Any]]) -> tuple[dict[str, Any], dict[int, set[int]]]:
  """The bound on other programs' CPU per turn (idle_bound: the measured idle before the arms; without it the run's own
  outlier bound over its turns) and the turns over it, by arm index."""
  every = [t["foreign_cores"] for a in arms for t in a.get("turns") or [] if t.get("foreign_cores") is not None]
  from boltbeam.collectors.session_ab import HZ
  secs = [t["wall_ms"] / 1e3 for a in arms for t in a.get("turns") or [] if t.get("wall_ms")]
  got = idle_bound(arms) or {"baseline_cores": statistics.median(every) if every else None, "source": "turns",
                             "bound_cores": outlier_bound(every, resolution=2 / (HZ * statistics.median(secs)) if secs else 0.0)}  # two ticks over a typical turn
  bound = got["bound_cores"]
  flagged = {k: {t["i"] for t in a.get("turns") or [] if bound is not None and (t.get("foreign_cores") or 0) > bound}
             for k, a in enumerate(arms)}
  return got, flagged


def analyze(record:dict[str, Any]) -> dict[str, Any]:
  """The record's comparisons: every arm against the first (the base), on the valid runs."""
  names = [a["name"] for a in record["inputs"]["arms"]]
  valid = [a for a in record["arms"] if a["valid"]]
  by = {n: sorted((a for a in valid if a["name"] == n), key=lambda a: a["block"]) for n in names}
  got, flags = contamination(valid)
  flag = {id(a): flags[k] for k, a in enumerate(valid)}
  base = names[0]
  ref = by[base][0]["turns"] if by[base] else []
  ref_sha = {t["i"]: t["sha256"] for t in ref}
  identity = {n: [f"{sum(ref_sha.get(t['i']) == t['sha256'] for t in a['turns'])}/{len(ref)}" for a in by[n]] for n in names}
  out = {"base": base, "identity": identity, "contamination": {**got,
         "turns": {f"{a['name']}#{a['block'] + 1}": sorted(flag[id(a)]) for a in valid if flag[id(a)]}},
         "valid": {n: len(by[n]) for n in names}, "invalid": sum(not a["valid"] for a in record["arms"]),
         "exit_hangs": [f"{a['name']}#{a['block'] + 1}" for a in record["arms"] if (a.get("exit") or {}).get("killed")],
         "comparisons": []}
  cuts = bands(ref) if ref else []
  for n in names[1:]:
    blocks = sorted({a["block"] for a in by[n]} & {a["block"] for a in by[base]})
    pairs = [(next(a for a in by[base] if a["block"] == k), next(a for a in by[n] if a["block"] == k)) for k in blocks]
    cmp: dict[str, Any] = {"arm": n, "pairs": len(pairs), "refused": condition_mismatches(by[base], by[n], len(names) - 1) or None,
                           "conditions": {"base": [arm_conditions(a) for a in by[base]], "arm": [arm_conditions(a) for a in by[n]]}}
    if not pairs:
      cmp["refused"] = (cmp["refused"] or []) + [_words("no_pairs")]
    if cmp["refused"]:
      out["comparisons"].append(cmp)
      continue
    rows = []
    for cut in cuts:
      row = {"band": cut["label"], "turns": len(cut["turns"])}
      for clean in (False, True):
        for m, _ in METRICS:
          got = []
          for a, b in pairs:
            keep = cut["turns"] - (flag[id(a)] | flag[id(b)] if clean else set())
            va, vb = band_value(a["turns"], keep, m), band_value(b["turns"], keep, m)
            if va is not None and vb is not None:
              got.append((va, vb))
          key = m + ("_clean" if clean else "")
          row[key] = {"base": statistics.fmean(x for x, _ in got) if got else None,
                      "arm": statistics.fmean(y for _, y in got) if got else None, "delta": mean_ci([y - x for x, y in got])}
      rows.append(row)
    walls = [(sum(t["wall_ms"] for t in a["turns"]) / 1e3, sum(t["wall_ms"] for t in b["turns"]) / 1e3) for a, b in pairs]
    cmp["bands"] = rows
    cmp["session_wall_s"] = {"base": statistics.fmean(x for x, _ in walls), "arm": statistics.fmean(y for _, y in walls),
                             "delta": mean_ci([y - x for x, y in walls])}
    common = sorted(set.intersection(*({t["i"] for t in a["turns"]} for p in pairs for a in p)))
    cmp["turns"] = [{"i": i, **{m: mean_ci([per_turn(_turn(b, i))[m] - per_turn(_turn(a, i))[m] for a, b in pairs
                                             if per_turn(_turn(b, i))[m] is not None and per_turn(_turn(a, i))[m] is not None])
                                for m in ("prefill_ms", "ttft_ms", "pre_server_ms", "wall_ms")}} for i in common]
    out["comparisons"].append(cmp)
  return out


def _turn(a:dict[str, Any], i:int) -> dict[str, Any]:
  return next(t for t in a["turns"] if t["i"] == i)


def _ci(d:dict[str, Any], p:int) -> str:
  if d.get("mean") is None:
    return "-"
  return f"{d['mean']:+.{p}f}" + (f" [{d['lo']:+.{p}f}, {d['hi']:+.{p}f}]" if d.get("lo") is not None else "")


def lines(record:dict[str, Any], an:dict[str, Any]) -> list[str]:
  """The short table: identity, conditions, and per band each metric base -> arm with the paired delta."""
  inp = record["inputs"]
  out = [_words("head", arms=len(inp["arms"]), repeats=inp["repeats"], turns="{}-{}".format(*inp.get("turns") or (1, inp["session"]["turns"])),
                order=" | ".join(" ".join(n for k, n in inp["order"] if k == b) for b in range(inp["repeats"])),
                gaps="gaps from a file" if inp.get("gaps") else "back to back", history=inp["history"])]
  out.append(_words("identity", base=an["base"]) + "; ".join(f"{n} {' '.join(v)}" for n, v in an["identity"].items()))
  c = an["contamination"]
  out.append(_words("contamination", bound=c["bound_cores"] or 0, n=sum(len(v) for v in c["turns"].values()),
                    how=_words("contamination_idle", base=c["baseline_cores"], se=c["se_cores"], z=c["z"], k=c["samples"]) if c.get("source") == "idle"
                    else _words("contamination_turns"),
                    where=", ".join(f"{k}: {v}" for k, v in c["turns"].items()) or "none"))
  if an["invalid"] or an["exit_hangs"] or record.get("stopped"):
    out.append(_words("invalid", n=an["invalid"], hangs=", ".join(an["exit_hangs"]) or "none") +
               (f" {record['stopped']}" if record.get("stopped") else ""))
  for cmp in an["comparisons"]:
    if cmp["refused"]:
      out.append(_words("refused", arm=cmp["arm"], base=an["base"], why="; ".join(cmp["refused"])))
      continue
    out.append(_words("cmp_head", arm=cmp["arm"], base=an["base"], pairs=cmp["pairs"]))
    out.append(f"  {'band':<10}" + "".join(f"{lab:>34}" for _, lab in METRICS))
    for clean in (False, True):
      if clean:
        out.append("  " + _words("clean"))
      for r in cmp["bands"]:
        cells = []
        for m, _ in METRICS:
          x = r[m + ("_clean" if clean else "")]
          p = 1 if m == "decode_tok_s" else 0
          cells.append("-" if x["base"] is None else f"{x['base']:.{p}f}->{x['arm']:.{p}f} {_ci(x['delta'], p)}")
        band = f"{r['band']} ({r['turns']})"  # one quote level: Python 3.11 (the floor) rejects reused quotes
        out.append(f"  {band:<10}" + "".join(f"{c:>34}" for c in cells))
    w = cmp["session_wall_s"]
    out.append("  " + _words("wall", a=w["base"], b=w["arm"], d=_ci(w["delta"], 1)))
  return out
