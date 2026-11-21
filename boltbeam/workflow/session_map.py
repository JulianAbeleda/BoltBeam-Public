"""The session map: for a traced session, per kernel class and phase, one decode step's and one prefill turn's time now,
the floor, the reference ceiling, the gap in session seconds and whether the search can express the class
(boltbeam.session_map.v1; docs/session-map.md).

    captures    nsys sqlite exports, each a window of the session's turns (the driver tagged them), read once.
    turns       the session record's (collectors/replay.llama_turns): each turn's span on the host clock and the engine's
                own prompt and decode times place its prefill and decode windows (time_budget.turn_windows).
    decode      the kernel ledger over each window's decode windows joined (kernel_ledger.read with turns): its steps,
                rows, joins, bytes and coverage; rows roll up into classes (data/kernel_classes.json). Host idle per step
                is every gap in the decode windows, attributed by the call in flight (time_budget.attribute_gaps).
    prefill     each turn's prefill window: launches and copies in start order, each counting what it adds to the union
                (first-started wins); the target's launches on the stream with the most, the draft head's on the others
                (the engine's prefill_phase row); passes = the target's launches of its most launched attention kernel /
                the header's attention layers (a pass runs each attention layer once).
    floor       decode: the ledger's limits (bytes / measured bandwidth; derived, or estimated for experts and attention);
                a row with no bytes adds 0 (a lower bound), so that class's gap is an upper bound (gap_known false). Prefill (an estimate): per pass, the larger of the class's header bytes read
                once / bandwidth and its work (2 x K x N x tokens, experts at experts_used) / the measured matrix rate;
                a class that reads no weight (NO_WEIGHTS) 0, a lower bound; attention, indexer, recurrent unknown.
    reference   a measured row only (--reference: another engine's trace of the session, a microbench): its time, or a
                measured rate (GB/s) on the class's bytes. Anything else is unknown, with its test.
    gap         (now - floor) x the session's steps (decode) or turns (prefill), each turn at the nearest window by depth;
                reachable = (now - reference) the same way, never below 0.
    express     per class, the ledger's coverage (search_can_propose by row, time-weighted), its largest missing pieces,
                and the kernels an installed manifest already puts in place (generated).
    coverage    unattributed = the time in the classes data/kernel_classes.json lists as unattributed; the record fails when
                it is above UNATTRIBUTED_MAX_PCT of the step (the design's refuting line).
    audit       the per-role decode audit (record key `audit`; docs/session-map.md "The audit"): per class and window,
                the phases summed, now against a floor that counts a byte-less row's launch floor (and --floors' cited
                bytes); exposed copies and host idle as rows; the depth slope (z standard errors); shape flags whose
                thresholds are this machine's (compute units, the capture's best achieved bandwidth, one graph launch);
                the stock proof against --previous; the queue ranked by gap x the lever's measured realization
                (data/realization.json).

Engine-blind: an engine is a key into data rows; nothing here names one, a model or a GPU.
Labels: measured (source), derived (formula, inputs), estimate (formula, test), unknown (test, no value)."""
from __future__ import annotations

import bisect, collections, json, math, pathlib, re, statistics
from typing import Any

from boltbeam import formats
from boltbeam.collectors import time_budget as tb
from boltbeam.workflow import kernel_ledger as kl
from boltbeam.workflow.context_audit import derived, measured, unknown


SCHEMA = "boltbeam.session_map.v1"
UNATTRIBUTED_MAX_PCT = 5.0  # the machine-search design's S0 refuting line (2026-11-18): above it the map does not cover the step
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
GATES = ("join coverage: every main-file matrix joined", "draft passes against the counters")  # the ledger checks the map's exit code holds to
NO_WEIGHTS = ("small", "sampling", "expert_routing", "kv_write", "sparse_select")  # prefill classes that read no weight: floor 0, a lower bound
estimate = kl.estimate
AUDIT = json.loads((DATA / "realization.json").read_text())  # the audit's levers, measured realization and call words
Z = AUDIT["z"]
LETTER = {"measured": "M", "derived": "D", "estimate": "E", "unknown": "U"}


def table() -> dict[str, Any]:
  """data/kernel_classes.json, checked: every class a row names is listed (fail closed)."""
  t = json.loads((DATA / "kernel_classes.json").read_text())
  named = [*t["matmul"].values(), *t["families"].values(), *t["unattributed"],
           *(r["class"] for e in t["engines"].values() for r in e.get("rows") or [])]
  if bad := [c for c in named if c not in t["classes"]]:
    raise ValueError(f"data/kernel_classes.json: a row names a class the table does not list: {bad[0]}")
  return t


def engine_rows(t:dict[str, Any], engine:str | list[str]) -> tuple[list[tuple[re.Pattern, str]], dict[str, Any]]:
  """The engine's class rows along its `also` chain (kernel_families.json), and its prefill_phase row."""
  fam = tb._table("kernel_families.json")["engines"]
  rows, phase, todo, seen = [], {}, list(tb.engine_keys(engine)), set()
  while todo:
    if (key := todo.pop(0)) in seen:
      continue
    seen.add(key)
    e = t["engines"].get(key) or {}
    rows += [(re.compile(r["pattern"]), r["class"]) for r in e.get("rows") or []]
    phase = phase or e.get("prefill_phase") or {}
    todo += (fam.get(key) or {}).get("also") or []
  return rows, phase


def class_of(t:dict[str, Any], rows:list[tuple[re.Pattern, str]], name:str, family:str, op:str | None, fmts:list[str] | None, head:bool,
             joined:bool = True) -> str:
  """A launch's class (the module; data/kernel_classes.json note). fmts: the format families of the tensors a join tied it
  to (None: none); joined False: no join was tried (a prefill), so a dense matmul is the unsplit class."""
  if head:
    return "output_head"
  if hit := next((c for p, c in rows if p.search(name)), None):
    return hit
  if op == "expert_matmul" or family == "expert_mmq":
    return t["matmul"]["expert_matmul"]
  if op == "matmul" or (family == "dense_gemm" and not joined):
    return t["matmul"]["unsplit"] if not joined else t["matmul"]["unjoined"] if not fmts else t["matmul"]["float" if all(f == "float" for f in fmts) else "quantized"]
  return t["families"].get(family, "other")


# --- decode -------------------------------------------------------------------------------------------------------

def host_idle(tl:tb.Timeline, wins:list[tuple[int, int]], steps:int) -> dict[str, Any]:
  """Idle per decode step over the decode windows: every gap between GPU operations (kernels and copies) attributed by the
  call in flight (time_budget.attribute_gaps, the decode cause rows); API calls per step by name."""
  rows, cpu, _ = tb.causes(decode=True)
  ks = sorted((s, e, n, st) for s, e, n, st in tl.kernels)
  ops = sorted([(s, e) for s, e, *_ in ks] + [(c["start_ns"], c["end_ns"]) for c in tl.copies])
  calls = sorted(tl.calls)
  kst, ost, cst = [k[0] for k in ks], [o[0] for o in ops], [c[0] for c in calls]
  longest = max((e - s for s, e, _ in calls), default=0)  # a call in flight at a window's start began at most this long before it
  by_cause, api, idle, pairs = collections.Counter(), collections.Counter(), 0.0, collections.Counter()
  for a, b in wins:
    au = [x for x in tb.union(ops[bisect.bisect_left(ost, a):bisect.bisect_left(ost, b)]) if x[1] > a and x[0] < b]
    gaps = ([(a, au[0][0])] if au and au[0][0] > a else []) + tb.idle_gaps(au)
    cl = [c for c in calls[bisect.bisect_left(cst, a - longest):bisect.bisect_left(cst, b)] if c[1] > a]
    api.update(c[2] for c in cl if c[0] >= a)
    for g in tb.attribute_gaps(gaps, cl, ks[bisect.bisect_left(kst, a):bisect.bisect_left(kst, b)], rows, cpu):
      idle += (g["end_ns"] - g["start_ns"]) / 1e6
      by_cause.update({c: ns / 1e6 for c, ns in g["causes"].items()})
      pairs[(g["before"], g["after"])] += (g["end_ns"] - g["start_ns"]) / 1e6
  src = f"{pathlib.Path(tl.source).name}: decode windows, {steps} steps, traced"
  return {"idle_ms": measured(idle / steps, src), "by_cause_ms": {c: measured(v / steps, src) for c, v in by_cause.most_common()},
          "api_calls_a_step": {n: measured(v / steps, src) for n, v in api.most_common(8)},
          "pairs_ms": [[a, b, v / steps] for (a, b), v in pairs.most_common(10)]}


def decode_classes(led:dict[str, Any], t:dict[str, Any], rows:list[tuple[re.Pattern, str]], depth:str, manifest:list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
  """The ledger's rows at one depth rolled into (phase, class): launches, time, floor (bytes / bandwidth), bytes, the
  rows with no bytes, coverage and the installed kernels."""
  out: dict[tuple, dict[str, Any]] = {}
  gen = {r["re"].pattern: r["manifest"] for r in tb.shape_rows(None, None, manifest)}
  for r in led["byte_limited"] + led["bound_unknown"]:
    if not (x := r["depths"].get(depth)):
      continue
    serves = r["serves"] if isinstance(r["serves"], dict) else {}
    fmts = [(formats.row(s["format"]) or {}).get("family") for s in serves.get("tensors", [])] or None
    c = class_of(t, rows, r["kernel"], r["family"], None if r["op"].startswith("none") else r["op"], fmts and ["float" if f == "float" else "q" for f in fmts],
                 "lm_head" in serves.get("role", []))
    g = out.setdefault((r["phase"], c), {"phase": r["phase"], "class": c, "launches": 0.0, "now_ms": 0.0, "floor_ms": 0.0, "floor_labels": set(), "bytes": 0.0,
                                         "no_bytes_ms": 0.0, "propose_ms": 0.0, "missing": collections.Counter(), "generated": set(), "rows": 0,
                                         "launch_floor_ms": 0.0, "kernels": []})
    ms, lim = x["time_ms"]["value"], x["limit_ms"]
    g["launches"] += x["launches"]["value"]
    g["now_ms"] += ms
    g["rows"] += 1
    g["kernels"].append({"kernel": r["kernel"], "grid": r["grid"], "launches": x["launches"]["value"], "ms": ms, "step_sd": x["time_ms"].get("step_sd"),
                         "limit_ms": lim["value"], "label": lim["label"], "floor_ms": x["floor_ms"]["value"]})
    if lim["value"] is None:
      g["no_bytes_ms"] += ms
      g["launch_floor_ms"] += x["floor_ms"]["value"] or 0.0
    else:
      g["floor_ms"] += lim["value"]
      g["bytes"] += x["bytes"]["value"]
      g["floor_labels"].add(lim["label"])
    cov = x["coverage"]
    g["propose_ms"] += ms if cov["search_can_propose"] else 0.0
    for m in cov["missing"]:
      g["missing"][m.split(" (")[0]] += ms
    if (m := next((v for p, v in gen.items() if re.search(p, r["kernel"])), None)):
      g["generated"].add(m)
  return out


# --- prefill ------------------------------------------------------------------------------------------------------

def prefill_turns(tl:tb.Timeline, wins:list[tuple[int, int]], t:dict[str, Any], rows:list[tuple[re.Pattern, str]], phase_row:dict[str, Any],
                  shapes:list[dict[str, Any]], fams_rules:list, vocab:int | None, layers:int | None) -> dict[str, Any]:
  """Per prefill turn (mean over the window's turns): ms per (phase, class), exclusive with copies in the walk (their busy
  time is the row `copies`), and the target's passes (its most launched attention kernel / the attention layers)."""
  ks = sorted((s, e, n, st, g) for (s, e, n, st), g in zip(tl.kernels, tl.grids or [(0, 0, 0)] * len(tl.kernels)))
  cs = sorted((c["start_ns"], c["end_ns"], c.get("bytes") or 0, c["kind"], c.get("stream")) for c in tl.copies)
  kst, cst = [k[0] for k in ks], [c[0] for c in cs]
  acc, passes = collections.Counter(), 0.0
  for a, b in wins:
    kk, cc = ks[bisect.bisect_left(kst, a):bisect.bisect_left(kst, b)], cs[bisect.bisect_left(cst, a):bisect.bisect_left(cst, b)]
    main = collections.Counter(k[3] for k in kk).most_common(1)[0][0] if kk else None
    draft_apart = phase_row.get("draft") == "other_streams"
    fams = tb.launch_families([k[:4] for k in kk], fams_rules)
    heads = set(tb.output_layers([(k[2], k[4]) for k in kk], shapes, vocab)) if vocab else set()
    ops = sorted([(k[0], k[1], i, "k") for i, k in enumerate(kk)] + [(c[0], c[1], i, "c") for i, c in enumerate(cc)])
    names: list[tuple[str, str] | None] = [None] * len(kk)
    cur = a
    for s, e, i, kind in ops:
      add = max(0, min(e, b) - max(s, cur)) / 1e6
      cur = max(cur, e)
      if kind == "c":
        acc[("copies", "copies")] += add
        continue
      name, stream = kk[i][2], kk[i][3]
      sr = tb.shape_of(name, shapes) or {}
      c = class_of(t, rows, name, sr.get("family", fams[i]), sr.get("op"), None, i in heads, joined=False)
      names[i] = ("draft" if draft_apart and stream != main else "target", c)
      acc[names[i]] += add
    att = collections.Counter(k[2] for k, c in zip(kk, names) if c == ("target", "attention"))
    passes += att.most_common(1)[0][1] / layers if att and layers else 0
  n = max(1, len(wins))
  return {"ms": {k: v / n for k, v in acc.items()}, "passes": passes / n if layers else None}


def prefill_floor(cls:str, header:list[dict[str, Any]], passes:float, tokens:float, experts_used:int | None, target:dict[str, Any]) -> dict[str, Any]:
  """A matmul class's floor for one prefill turn: per pass the larger of its header bytes read once / bandwidth and its work
  / the measured matrix rate (int8 for a quantized format, bf16 for a float one); the head runs once a turn. Else unknown."""
  pick = {"expert_matmul": lambda x: x["experts"] > 0, "output_head": lambda x: x["role"] == "lm_head",
          "dense": lambda x: not x["experts"] and x["role"] != "lm_head" and x["n"] > 1}
  bw, rates = target.get("memory_bandwidth_gbs"), target.get("matrix_tflops") or {}
  rate = lambda f: rates.get("bf16" if (formats.row(f) or {}).get("family") == "float" else "int8")  # noqa: E731  (quantized formats run int8 MMA)
  cells = [x for x in header if pick[cls](x)] if cls in pick else []
  if not cells or not bw or passes is None or (cls == "expert_matmul" and not experts_used) or not all(rate(x["format"]) for x in cells):
    return unknown(f"the bytes and work of class {cls} at a prefill pass (its tensors, the passes, the target's matrix rates)")
  n = 1 if cls == "output_head" else passes
  t_bytes = sum(x["bytes"] for x in cells) / (bw * 1e9) * 1e3 * n
  t_work = sum(2 * x["k"] * x["n"] * x["count"] * (experts_used if x["experts"] else 1) / (rate(x["format"]) * 1e12) for x in cells) * 1e3 * (1 if cls == "output_head" else tokens)
  return estimate(max(t_bytes, t_work), "max(header bytes / bandwidth x passes, 2 x K x N x tokens / the measured matrix rate: bf16 for a float format, "
                  "int8 for a quantized one)", "every expert reached each pass (a routing profile's reach at the pass's tokens) and the rate the "
                  "engine's kernel runs a format at", inputs={"bytes_ms": t_bytes, "work_ms": t_work, "passes": n, "tokens": tokens})


def layer_count(table:dict[str, Any], name:str) -> int | None:
  """A layer count by data/min_memory.json's tensor-name pattern for it under the header's architecture."""
  from boltbeam.workflow.min_memory import block
  pat = (block((table or {}).get("meta"), (table or {}).get("tensors")) or {}).get("counts", {}).get(name)
  return sum(1 for n, *_ in table["tensors"] if re.search(pat, n)) if pat else None


# --- the map ------------------------------------------------------------------------------------------------------

def bands(turns:list[dict[str, Any]], depths:dict[str, float], row:dict[str, Any] | None) -> dict[str, dict[str, float]]:
  """Each turn at the window nearest its depth: the session's decode steps and prefill turns per window."""
  out = {w: {"steps": 0, "turns": 0} for w in depths}
  for x in turns:
    w = min(depths, key=lambda k: abs(depths[k] - x["depth"]))
    out[w]["steps"] += tb.decode_steps(x["tokens"], x["accepted"], row)
    out[w]["turns"] += 1
  return out


def session_map(captures:list[dict[str, Any]], turns:list[dict[str, Any]], *, main:dict[str, Any], draft:dict[str, Any] | None, engine:str | list[str],
                vendor:str | None, target:dict[str, Any], k:int, kv_type:str | None = None, selection:int | None = None,
                manifest:dict[str, Any] | None = None, routing:dict[str, Any] | None = None, untraced:list[list[dict[str, Any]]] | None = None,
                reference:list[dict[str, Any]] | None = None, traced_runs:list[list[dict[str, Any]]] | None = None,
                draft_selection:int | None = None, previous:dict[str, Any] | None = None, floors:list[dict[str, Any]] | None = None,
                h2d_gbs:float | None = None, first:bool = False, name:str | None = None) -> dict[str, Any]:
  """The record. captures: [{path, window, tl?}] (tl: a Timeline already read); turns: the traced record's
  (replay.llama_turns), tagged by window; untraced: the same session's untraced runs (each a list of turns): the
  untraced step per window and the session's steps. reference: [{part (decode or prefill), phase, classes (one, several, or ["*"]), ms or gbs, source,
  window?}] measured rows (ms a step, or a turn). draft_selection: the KV cells the draft's attention reads when it is sparse
  too. The audit's inputs (the module's audit lines): previous, an earlier map record of the same plan (the stock proof);
  floors, [{class, bytes, depth?, source}] bytes a class reads that the header does not give (scaled by depth when `depth`
  names the depth they were counted at); h2d_gbs, the GPU's measured host-to-device copy rate; first, rank with no
  previous record (the plan's first audit, said); name, this record's name for a later proof."""
  t, eng = table(), tb.engine_keys(engine)[-1]
  rows, phase_row = engine_rows(t, engine)
  mrows = kl.manifest_rows(manifest)
  shapes = tb.shape_rows(engine, target.get("id"), mrows)
  row_d = tb.decode_row(engine)
  vocab = next((x["n"] for x in kl.cells(main, "main") if x["role"] == "lm_head"), None)
  used, layers = kl.fact(main, "experts_used"), layer_count(main, "attention_layers")
  header = kl.cells(main, "main")
  windows, caps, tls = {}, [], {}
  for c in captures:
    w = c["window"]
    tw = [x for x in turns if x["window"] == w]
    if not tw:
      return {"schema": SCHEMA, "refused": f"window {w}: the session record tags no turn with it"}
    tl = c.get("tl") or tb.timeline_from_nsys(c["path"])
    tls[w] = tl
    depth = round(statistics.fmean(x["depth"] for x in tw))
    windows[w] = {"turns": [x["i"] for x in tw], "depth": depth, "steps": sum(tb.decode_steps(x["tokens"], x["accepted"], row_d) for x in tw),
                  "tokens": sum(x["prompt_n"] for x in tw) / len(tw)}
    caps.append({"cap": kl.read(c["path"], tw, "decode", tl=tl), "depth": depth, "window": w})
  caps.sort(key=lambda c: c["depth"])
  for w, info in windows.items():
    per = [sum(x["decode_ms"] for x in r if x["i"] in info["turns"]) / max(1, sum(tb.decode_steps(x["tokens"], x["accepted"], row_d) for x in r if x["i"] in info["turns"]))
           for r in (untraced or [])]
    info["untraced_runs_ms"] = per
    info["untraced_step_ms"] = measured(statistics.median(per), f"median of {len(per)} untraced runs, turns {info['turns']}") if per else \
      unknown("the same turns untraced (--untraced): the trace's overhead")
  leds = {}  # one ledger a window: its rankings hold every row of that window (a ledger over several depths ranks the deepest's)
  for c in caps:
    leds[c["window"]] = led = kl.ledger([{"cap": c["cap"], "depth": c["depth"], "untraced_ms": windows[c["window"]]["untraced_step_ms"]["value"]}], main=main, draft=draft,
                                        engine=engine, vendor=vendor, target=target, k=k, kv_type=kv_type, selection={q: v for q, v in (("verify", selection), ("draft", draft_selection)) if v} or None, manifest=manifest,
                                        routing=routing)
    if led.get("refused"):
      return {"schema": SCHEMA, "refused": f"kernel ledger, window {c['window']}: {led['refused']}"}
  run_turns = untraced or traced_runs or [turns]  # each run once: several traced records are several runs, never one
  depths = {w: windows[w]["depth"] for w in windows}
  per_run = [bands(r, depths, row_d) for r in run_turns]
  band = {w: {q: statistics.median(b[w][q] for b in per_run) for q in ("steps", "turns")} for w in windows}
  fams_rules = tb.families_for(engine, vendor)
  out_windows, aud = [], []
  for c in caps:
    w, d = c["window"], str(c["depth"])
    tot = leds[w]["depths"][0]
    step = tot["totals"]["step_ms"]["value"]
    cls = decode_classes(leds[w], t, rows, d, mrows)
    tw = [x for x in turns if x["window"] == w]
    wins = [x for x in tb.turn_windows(tls[w], tw) if x]
    host = host_idle(tls[w], [x["decode"] for x in wins], windows[w]["steps"])
    un = windows[w]["untraced_step_ms"]["value"]
    host["idle_untraced_ms"] = estimate(host["idle_ms"]["value"] - (step - un), "traced idle - (traced step - untraced step): the trace's overhead taken as host time",
                                        "a replayed step with no host work") if un is not None else unknown("the untraced step (--untraced)")
    pf = prefill_turns(tls[w], [x["prefill"] for x in wins], t, rows, phase_row, shapes, fams_rules, vocab, layers)
    out_windows.append({"window": w, "depth": c["depth"], "turns": windows[w]["turns"], "decode_steps": measured(tot["steps"]["value"], tot["source"]),
                        "step_ms": tot["totals"]["step_ms"], "untraced_step_ms": windows[w]["untraced_step_ms"], "kernels_ms": tot["totals"]["union_ms"],
                        "host": host, "classes": cls, "prefill": pf, "prefill_tokens": windows[w]["tokens"], "checks": tot["checks"]})
    aud.append(audit_window(out_windows[-1], leds[w], tls[w], [x["decode"] for x in wins], target=target, vendor=vendor, h2d_gbs=h2d_gbs, floors=floors or [],
                           counted=windows[w]["steps"]))
  out = finish(out_windows, band, t, target, header, used, reference or [], eng, vendor, k)
  out["audit"] = audit(sorted(aud, key=lambda a: a["depth"]), out["session"], windows, previous, first)
  out["name"] = name
  return out


def finish(wins:list[dict[str, Any]], band:dict[str, dict[str, float]], t:dict[str, Any], target:dict[str, Any],
           header:list[dict[str, Any]], used:int | None, reference:list[dict[str, Any]], eng:str, vendor:str | None, k:int) -> dict[str, Any]:
  """Rows per (phase, class) with each window's figures, session seconds over the bands, the non-kernel rows and checks."""
  steps_s = sum(b["steps"] for b in band.values())
  turns_s = sum(b["turns"] for b in band.values())
  ref = lambda part, ph, c, w: next((r for r in reference if r["part"] == part and r["phase"] == ph and r["classes"] == [c] and r.get("window") in (None, w)), None)  # noqa: E731
  decode, prefill, checks = {}, {}, []
  for x in wins:
    w, st = x["window"], band[x["window"]]
    for key, g in x["classes"].items():
      row = decode.setdefault(key, {"phase": key[0], "class": key[1], "windows": {}, "gap_s": 0.0, "gap_known": True, "reachable_s": 0.0, "reachable_known": True})
      row["gap_known"] &= not g["no_bytes_ms"]  # a row with no bytes floors at 0: the gap is then an upper bound
      lab = "derived" if g["floor_labels"] <= {"derived"} and not g["no_bytes_ms"] else "estimate"
      floor = {"value": g["floor_ms"], "label": lab, **({"formula": "sum of the ledger's limits (bytes / bandwidth)", "inputs": ["ledger rows"]} if lab == "derived" else
                                                       {"formula": "sum of the ledger's limits; a row with no bytes adds 0 (a lower bound)", "test": "bytes for the rows that have none (a shapes row, a routing count)",
                                                        "no_bytes_ms": g["no_bytes_ms"]})}
      rv = reference_of(ref("decode", key[0], key[1], w), g["bytes"] if not g["no_bytes_ms"] else None)
      row["windows"][w] = {"launches": measured(g["launches"], "the ledger"), "now_ms": measured(g["now_ms"], "the ledger: exclusive, traced"), "floor_ms": floor,
                           "reference_ms": rv, "can_express": express(g), "generated": sorted(g["generated"])}
      row["gap_s"] += (g["now_ms"] - g["floor_ms"]) * st["steps"] / 1e3
      if rv["value"] is None:
        row["reachable_known"] = False
      else:
        row["reachable_s"] += max(0.0, g["now_ms"] - rv["value"]) * st["steps"] / 1e3
    for (ph, c), ms in x["prefill"]["ms"].items():
      row = prefill.setdefault((ph, c), {"phase": ph, "class": c, "windows": {}, "gap_s": 0.0, "gap_known": True, "reachable_s": 0.0, "reachable_known": True})
      fl = prefill_floor(c, header, x["prefill"]["passes"], x["prefill_tokens"], used, target) if ph == "target" else unknown("the draft head's prefill bytes (its header's join)")
      if fl["value"] is None and c in NO_WEIGHTS:
        fl = estimate(0.0, "no weight bytes: 0 (a lower bound)", "the bytes these kernels read and write (the engine's graph)")
      rv = reference_of(ref("prefill", ph, c, w), None)
      row["windows"][w] = {"now_ms": measured(ms, "prefill windows: exclusive, traced, ms a turn"), "floor_ms": fl, "reference_ms": rv}
      if fl["value"] is None:
        row["gap_known"] = False
      else:
        row["gap_s"] += (ms - fl["value"]) * st["turns"] / 1e3
      if rv["value"] is None:
        row["reachable_known"] = False
      else:
        row["reachable_s"] += max(0.0, ms - rv["value"]) * st["turns"] / 1e3
    step = x["step_ms"]["value"]
    loose = sum(g["now_ms"] for (_, c), g in x["classes"].items() if c in t["unattributed"])
    checks.append({"check": f"window {w}: unattributed share of the step", "gate": True, "ok": 100 * loose / step <= UNATTRIBUTED_MAX_PCT, "pct": 100 * loose / step,
                   "why": f"{loose:.3f} of {step:.3f} ms a step in {', '.join(t['unattributed'])} ({100 * loose / step:.1f}%, at most {UNATTRIBUTED_MAX_PCT:g}%)"})
    checks += [{**c, "check": f"window {w}: {c['check']}"} for c in x["checks"]]
  # prefill classes are kept per window (the expression column is the decode's when the class runs in both)
  key_s = lambda r: -(r["gap_s"])  # noqa: E731
  out = {"schema": SCHEMA, "engine": eng, "vendor": vendor, "target": target.get("id"), "k": k,
         "session": {"decode_steps": derived(steps_s, "the session's decode steps (tokens - tokens_from_prompt - accepted), median over its runs", "turns"),
                     "turns": derived(turns_s, "the session's turns", "turns"),
                     "bands": {w: {"decode_steps": derived(b["steps"], "steps of the turns nearest this window by depth", "turns"),
                                   "turns": derived(b["turns"], "turns nearest this window by depth", "turns")} for w, b in band.items()}},
         "windows": [{q: v for q, v in x.items() if q not in ("classes", "prefill", "checks")} | {"prefill_passes": derived(x["prefill"]["passes"], "the target's launches of its most launched attention kernel a turn / the header's attention layers", "launches"),
                                                                                                  "prefill_tokens": measured(x["prefill_tokens"], "the record's prompt_n")} for x in wins],
         "decode": sorted(decode.values(), key=key_s), "prefill": sorted(prefill.values(), key=key_s),
         "reference_groups": groups(reference, {"decode": decode, "prefill": prefill}, band), "checks": checks}
  out["ok"] = all(c["ok"] for c in checks if c.get("gate", c["check"].split(": ", 1)[-1] in GATES))
  return out


def groups(reference:list[dict[str, Any]], parts:dict[str, dict[tuple, dict[str, Any]]], band:dict[str, dict[str, float]]) -> list[dict[str, Any]]:
  """A reference row measured over several classes of a phase ("*": all of them; another engine splits its time its own
  way): per window the classes' summed time against it, and the reachable seconds over the bands."""
  out = []
  for r in (r for r in reference if r["classes"] != [r["classes"][0]] or r["classes"] == ["*"]):
    rows = [x for (ph, c), x in parts[r["part"]].items() if ph == r["phase"] and (r["classes"] == ["*"] or c in r["classes"])]
    g = {"part": r["part"], "phase": r["phase"], "classes": r["classes"], "windows": {}, "reachable_s": 0.0}
    for w, b in band.items():
      if r.get("window") not in (None, w) or not (cells := [x["windows"][w] for x in rows if w in x["windows"]]):
        continue
      now = sum(c["now_ms"]["value"] for c in cells)
      rv = measured(r["ms"], r["source"]) if r.get("ms") is not None else unknown("a time for the group (a rate needs the classes' bytes)")
      g["windows"][w] = {"now_ms": derived(now, "sum of the classes' time", r["classes"]), "reference_ms": rv}
      if rv["value"] is not None:
        g["reachable_s"] += max(0.0, now - rv["value"]) * b["steps" if r["part"] == "decode" else "turns"] / 1e3
    out.append(g)
  return out


def reference_of(r:dict[str, Any] | None, by:float | None) -> dict[str, Any]:
  """A --reference row as the class's reference: its measured time, or its measured rate on the class's bytes (derived)."""
  if r and r.get("ms") is not None:
    return measured(r["ms"], r["source"])
  if r and r.get("gbs") and by:
    return derived(by / (r["gbs"] * 1e6), "class bytes / a measured rate", {"gbs": r["gbs"], "source": r["source"]})
  return unknown("a measured row of this class on its shapes: another engine's trace of the session, or a microbench (--reference)")


def express(g:dict[str, Any]) -> dict[str, Any]:
  """Whether the search can express the class: yes (every row), partial (some), no (none), with the largest missing pieces."""
  share = g["propose_ms"] / g["now_ms"] if g["now_ms"] else 0.0
  return {"answer": "yes" if share >= 0.999 else "no" if share <= 0.001 else "partial", "share": share,
          "missing": [m for m, _ in g["missing"].most_common(3)], "generated": sorted(g["generated"])}


# --- the audit ----------------------------------------------------------------------------------------------------

def realization(lever:str) -> dict[str, Any]:
  """A lever class's factor: the median of its measured rows' ratios, at most 1 (None with no row). A row with no ratio
  (no projection, or a prediction void for shape) is history. A regressed row is in: a change that made
  it slower realized below 0, and the median over every realized row is the honest search-realization figure."""
  rows = [r for r in AUDIT["realization"].get(lever) or [] if r.get("ratio") is not None]
  return {"factor": min(1.0, statistics.median(r["ratio"] for r in rows)) if rows else None, "ratios": [r["ratio"] for r in rows],
          "sources": [r["source"] for r in rows]}


def _calls(tl:tb.Timeline, wins:list[tuple[int, int]], word:str) -> tuple[int, float]:
  """Host API calls whose name holds `word`, started inside the windows: count and summed ms."""
  st = [c[0] for c in tl.calls]
  hit = [c for a, b in wins for c in tl.calls[bisect.bisect_left(st, a):bisect.bisect_left(st, b)] if word in c[2]]
  return len(hit), sum(e - s for s, e, _ in hit) / 1e6


def _h2d_bytes(tl:tb.Timeline, wins:list[tuple[int, int]]) -> int:
  return sum(c["bytes"] or 0 for c in tl.copies if c["kind"] == "HtoD" and any(a <= c["start_ns"] < b for a, b in wins))


def audit_window(x:dict[str, Any], led:dict[str, Any], tl:tb.Timeline, wins:list[tuple[int, int]], *, target:dict[str, Any], vendor:str | None,
                 h2d_gbs:float | None, floors:list[dict[str, Any]], counted:int) -> dict[str, Any]:
  """One window: per class now / floor / se and its phases, the copies and host rows, and the shape flags."""
  d = led["depths"][0]
  n, bw = d["steps"]["value"], target.get("memory_bandwidth_gbs")
  classes: dict[str, dict[str, Any]] = {}
  for (phase, cls), g in x["classes"].items():
    c = classes.setdefault(cls, {"now_ms": 0.0, "floor_ms": 0.0, "launches": 0.0, "sd": 0.0, "label": "derived", "phases": {}, "kernels": []})
    c["now_ms"] += g["now_ms"]
    c["floor_ms"] += g["floor_ms"] + g["launch_floor_ms"]
    c["launches"] += g["launches"]
    c["phases"][phase] = g["now_ms"]
    c["sd"] = None if c["sd"] is None or any(k["step_sd"] is None for k in g["kernels"]) else c["sd"] + sum(k["step_sd"] for k in g["kernels"])
    if g["no_bytes_ms"] or not g["floor_labels"] <= {"derived"}:
      c["label"] = "estimate"
    c["kernels"] += [{**k, "phase": phase} for k in g["kernels"]]
  for f in floors:
    if (c := classes.get(f["class"])) and bw:
      by = f["bytes"] * (x["depth"] / f["depth"] if f.get("depth") else 1.0)
      if (ms := by / (bw * 1e9) * 1e3) > c["floor_ms"]:
        c.update(floor_ms=ms, label="estimate", floor_source=f"{by / 1e6:.1f} MB / {bw} GB/s: {f['source']}")
  for c in classes.values():  # the rows' per-step sds summed: rows of one step co-vary (clocks, heat), so the sum is the safe side
    sd = c.pop("sd")
    c["se_ms"] = sd / math.sqrt(n) if sd is not None and n > 1 else None
  host = x["host"]
  idle_t, idle_u = host["idle_ms"]["value"], host["idle_untraced_ms"]["value"]  # a step of the counters' (`counted` steps)
  launches, launch_ms = _calls(tl, wins, AUDIT["calls"].get(vendor or "", {}).get("graph_launch", "\0"))
  exposed = max(0.0, d["totals"]["idle_ms"]["value"] - idle_t * counted / n)  # both over the ledger's n steps
  h2d = _h2d_bytes(tl, wins) / n
  classes["exposed_copies"] = {"now_ms": exposed, "floor_ms": h2d / (h2d_gbs * 1e9) * 1e3 if h2d_gbs else 0.0, "label": "derived" if h2d_gbs else "unknown", "se_ms": None,
                       "h2d_bytes": h2d, "phases": {}, "kernels": []}
  classes["host"] = {"now_ms": idle_u if idle_u is not None else idle_t, "now_label": "estimate" if idle_u is not None else "measured",
                     "floor_ms": launch_ms / launches if launches else 0.0, "label": "derived" if launches else "unknown", "se_ms": None, "phases": {}, "kernels": [],
                     "traced_ms": idle_t, "by_cause_ms": {k: v["value"] for k, v in host["by_cause_ms"].items()}}
  for c in classes.values():
    c["gap_ms"] = c["now_ms"] - c["floor_ms"]
  return {"window": x["window"], "depth": x["depth"], "steps": n, "step_ms": d["totals"]["step_ms"]["value"],
          "untraced_step_ms": x["untraced_step_ms"]["value"], "launch_floor_us": d["launch_floor_us"]["value"], "classes": classes,
          "flags": flags(classes, host, tl, wins, n, target, vendor)}


def flags(classes:dict[str, dict[str, Any]], host:dict[str, Any], tl:tb.Timeline, wins:list[tuple[int, int]], n:int, target:dict[str, Any],
          vendor:str | None) -> list[dict[str, Any]]:
  """The shape flags of one window (see the module), each with its class and its ms a step."""
  sms, bw, out = target.get("compute_units"), target.get("memory_bandwidth_gbs"), []
  rows = [(cls, k) for cls, c in classes.items() for k in c["kernels"]]
  se = lambda k: (k["step_sd"] or 0.0) / math.sqrt(n)  # noqa: E731
  for cls, k in rows:
    blocks = math.prod(k["grid"])
    if sms and blocks < sms and k["ms"] - (k["floor_ms"] or 0.0) > Z * se(k):
      out.append({"flag": "grid", "class": cls, "phase": k["phase"], "kernel": k["kernel"], "grid": k["grid"], "ms": k["ms"] * (1 - blocks / sms), "time_ms": k["ms"],
                  "why": f"{blocks} blocks on {sms} compute units: {1 - blocks / sms:.0%} of the GPU idle while it runs"})
  byte_rows = [(cls, k) for cls, k in rows if k["limit_ms"] and k["ms"] > Z * se(k)]
  reach = max((min(1.0, k["limit_ms"] / k["ms"]) for _, k in byte_rows if k["label"] == "derived"), default=None)
  for cls, k in byte_rows if reach and bw else []:
    if k["ms"] - k["limit_ms"] / reach > Z * se(k):
      out.append({"flag": "bandwidth", "class": cls, "phase": k["phase"], "kernel": k["kernel"], "grid": k["grid"], "ms": k["ms"] - k["limit_ms"] / reach, "time_ms": k["ms"],
                  "why": f"{bw * k['limit_ms'] / k['ms']:.0f} GB/s ({k['label']} bytes) against the {bw * reach:.0f} GB/s the best byte-bound group of this capture reaches"})
  one = classes["host"]["floor_ms"]
  for a, b, ms in host.get("pairs_ms") or []:
    if one and ms > one:
      out.append({"flag": "host gap", "class": "host", "kernel": f"{a} -> {b}", "ms": ms, "why": f"{ms:.3f} ms a step of idle between them, above one graph launch ({one:.3f})"})
  words = AUDIT["calls"].get(vendor or "", {})
  for what, cause in (("reinstantiate", None), ("sync", "sync")):  # a sync's call time is waiting: its cost is the idle it leaves
    if words.get(what) and (got := _calls(tl, wins, words[what]))[0]:
      ms = classes["host"]["by_cause_ms"].get(cause, 0.0) if cause else got[1] / n
      out.append({"flag": what, "class": "host", "kernel": words[what], "ms": ms, "per_step": got[0] / n,
                  "why": f"{got[0] / n:.2f} calls a step; {ms:.3f} ms a step of {'idle in them (traced)' if cause else 'call time'}"})
  return sorted(out, key=lambda f: -f["ms"])  # ms: what the shape costs a step (idle compute units' share, time over the reach, idle)


def slopes(ws:list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  """Per class, the deepest window less the shallowest: grows, flat, or unknown (no spread)."""
  a, b = ws[0], ws[-1]
  out = {}
  for cls in set(a["classes"]) & set(b["classes"]) if len(ws) > 1 and b["depth"] > a["depth"] else ():
    x, y = a["classes"][cls], b["classes"][cls]
    delta = y["now_ms"] - x["now_ms"]
    se = math.hypot(x["se_ms"], y["se_ms"]) if x["se_ms"] is not None and y["se_ms"] is not None else None
    out[cls] = {"delta_ms": delta, "per_100k_ms": delta / (b["depth"] - a["depth"]) * 1e5, "se_ms": se,
                "trend": "unknown" if se is None else "grows" if delta > Z * se else "shrinks" if delta < -Z * se else "flat"}
  return out


def run_spread(info:dict[str, Any]) -> float | None:
  """The untraced runs' step spread as a share (pooled over windows with two runs or more), None without."""
  vs = [statistics.variance(p) / statistics.fmean(p) ** 2 for p in (w.get("untraced_runs_ms") or [] for w in info.values()) if len(p) > 1]
  return math.sqrt(statistics.fmean(vs)) if vs else None


def proof(ws:list[dict[str, Any]], slope:dict[str, dict[str, Any]], spread:float | None, previous:dict[str, Any] | None) -> dict[str, Any]:
  """The stock proof (see the module): ok, and per flat class its difference and tolerance; ok None when it cannot be judged."""
  if previous is None:
    return {"ok": None, "why": "no previous audit named (--previous), or its record has no audit part: the depth-flat classes have nothing to match"}
  if not slope:
    return {"ok": None, "why": "one depth: no slope tells which classes are flat (trace two depths)"}
  flat = [c for c, s in slope.items() if s["trend"] == "flat"]
  sp = [s for s in (spread, previous.get("run_spread")) if s is not None]
  if not sp:
    return {"ok": None, "why": "no run-to-run spread in either audit (--untraced with two runs or more): the tolerance cannot be derived"}
  rows = []
  for x in ws:
    pw = min(previous["windows"], key=lambda p: abs(p["depth"] - x["depth"]))
    for c in flat:
      a, b = x["classes"].get(c), pw["classes"].get(c)
      if not a or not b or a["se_ms"] is None or b["se_ms"] is None:
        continue
      tol = Z * math.sqrt(a["se_ms"] ** 2 + b["se_ms"] ** 2 + sum(s ** 2 for s in (sp * 2)[:2]) * ((a["now_ms"] + b["now_ms"]) / 2) ** 2)
      rows.append({"class": c, "window": x["window"], "previous_window": pw["window"], "now_ms": a["now_ms"], "previous_ms": b["now_ms"],
                   "delta_ms": a["now_ms"] - b["now_ms"], "tolerance_ms": tol, "ok": abs(a["now_ms"] - b["now_ms"]) <= tol})
  bad = [r for r in rows if not r["ok"]]
  why = "; ".join(f"{r['class']} {r['window']} {r['now_ms']:.3f} vs {r['previous_ms']:.3f} ({r['previous_window']}), {r['delta_ms']:+.3f} beyond {r['tolerance_ms']:.3f}" for r in bad)
  return {"ok": bool(rows) and not bad, "previous": previous.get("name"), "rows": rows, "run_spread": sp,
          "why": why or (f"{len(rows)} flat class cells within tolerance" if rows else "no flat class in both audits")}


def audit(ws:list[dict[str, Any]], session:dict[str, Any], info:dict[str, Any], previous:dict[str, Any] | None, first:bool) -> dict[str, Any]:
  """The audit part of the record: windows (classes, copies, host, flags), slopes, the stock proof and the ranked queue."""
  band = {w: b["decode_steps"]["value"] for w, b in session["bands"].items()}
  slope, spread = slopes(ws), run_spread(info)
  pr = proof(ws, slope, spread, (previous or {}).get("audit") and {**previous["audit"], "name": previous.get("name")})
  queue = []
  for cls in sorted({c for w in ws for c in w["classes"]}):
    cells = {w["window"]: w["classes"][cls] for w in ws if cls in w["classes"]}
    gap_s = sum(c["gap_ms"] * band[w] for w, c in cells.items()) / 1e3
    lever = AUDIT["levers"].get(cls, "kernel")
    r = realization(lever)
    queue.append({"target": cls, "lever": lever, "gap_s": gap_s, "realization": r, "score_s": None if r["factor"] is None else max(0.0, gap_s) * r["factor"],
                  "slope": slope.get(cls), "flags": [f for w in ws for f in w["flags"] if f["class"] == cls][:5],
                  "windows": {w: {"now_ms": c["now_ms"], "floor_ms": c["floor_ms"], "gap_ms": c["gap_ms"],
                                  "label": "E" if c.get("now_label") == "estimate" and c["label"] == "derived" else LETTER[c["label"]]} for w, c in cells.items()},
                  "labels": {"gap_s": "derived: sum over windows of (now - floor) x the band's steps", "score_s": "estimate: gap_s x the lever's median realization"}})
  queue.sort(key=lambda q: (q["score_s"] is None, -(q["score_s"] or 0.0), -q["gap_s"]))
  ranked = pr["ok"] is True or (previous is None and first)
  for i, q in enumerate([q for q in queue if ranked and (q["score_s"] or 0) > 0], 1):
    q["rank"] = i
  for w in ws:
    for c in w["classes"].values():
      c.pop("kernels", None)
  return {"windows": ws, "slopes": slope, "run_spread": spread, "stock_proof": pr, "ranked": bool(ranked),
          "refused_ranking": None if ranked else f"not ranked: stock proof {'failed' if pr['ok'] is False else 'not judged'}: {pr['why']}" + ("" if previous else " (or --first)"),
          "queue": queue}


def lines(rec:dict[str, Any], top:int = 14) -> list[str]:
  """The short table: decode classes (per window ms: now / floor / reference; session s of gap; express), prefill, host, checks."""
  from boltbeam.report import plain
  wd = plain.words()["session_map"]
  if rec.get("refused"):
    return [wd["refused"].format(why=rec["refused"])]
  v = lambda f: "U" if f is None or f.get("value") is None else f"{f['value']:.2f}"  # noqa: E731
  ws = [x["window"] for x in rec["windows"]]
  out = [wd["head"].format(engine=rec["engine"], target=rec["target"], steps=rec["session"]["decode_steps"]["value"], turns=rec["session"]["turns"]["value"],
                           windows=", ".join(f"{x['window']} {x['depth'] / 1e3:.0f}k" for x in rec["windows"]))]
  for x in rec["windows"]:
    h = x["host"]
    out.append(wd["window"].format(w=x["window"], step=v(x["step_ms"]), untraced=v(x["untraced_step_ms"]), kernels=v(x["kernels_ms"]), idle=v(h["idle_ms"]),
                                   idle_u=v(h["idle_untraced_ms"]), causes=" ".join(f"{c} {m['value']:.2f}" for c, m in list(h["by_cause_ms"].items())[:6]),
                                   passes=v(x["prefill_passes"]), tokens=v(x["prefill_tokens"])))
  out.append(wd["decode_cols"].format(windows=" | ".join(ws)))
  for r in rec["decode"][:top]:
    cells = " | ".join(f"{v(c['now_ms'])}/{v(c['floor_ms'])}/{v(c['reference_ms'])}" for c in (r["windows"].get(w, {}) for w in ws) if c) or "-"
    ex = r["windows"][ws[-1]]["can_express"] if ws[-1] in r["windows"] else next(iter(r["windows"].values()))["can_express"]
    out.append(f"  {r['phase'][0]} {wd['classes'].get(r['class'], r['class']):<34} {cells}  gap {'' if r['gap_known'] else '<='}{r['gap_s']:.1f} s  reach {r['reachable_s']:.1f}{'' if r['reachable_known'] else '+U'} s  "
               f"{ex['answer']}{(': ' + '; '.join(ex['missing'][:2])) if ex['missing'] else ''}{(' [' + ','.join(ex['generated']) + ']') if ex['generated'] else ''}")
  out.append(wd["prefill_cols"].format(windows=" | ".join(ws)))
  for r in rec["prefill"][:top]:
    cells = " | ".join(f"{v(c['now_ms'])}/{v(c['floor_ms'])}/{v(c['reference_ms'])}" for c in (r["windows"].get(w, {}) for w in ws) if c) or "-"
    out.append(f"  {r['phase'][0]} {wd['classes'].get(r['class'], r['class']):<34} {cells}  gap {r['gap_s']:.1f}{'' if r['gap_known'] else '+U'} s  "
               f"reach {r['reachable_s']:.1f}{'' if r['reachable_known'] else '+U'} s")
  for g in rec.get("reference_groups") or []:
    cells = " | ".join(f"{v(c['now_ms'])}/{v(c['reference_ms'])}" for c in (g["windows"].get(w, {}) for w in ws) if c) or "-"
    out.append(wd["group"].format(part=g["part"], phase=g["phase"], classes="+".join(g["classes"]), cells=cells, reach=g["reachable_s"]))
  return out + audit_lines(rec.get("audit"), wd, top) + [f"{'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['why']}" for c in rec["checks"]]


def audit_lines(a:dict[str, Any] | None, wd:dict[str, Any], top:int) -> list[str]:
  """The audit: per target and window now / floor / gap (label), session s of gap, slope, lever x factor = score; the
  flags; the stock proof; the ranking's state."""
  if not a:
    return []
  ws = [w["window"] for w in a["windows"]]
  out = [wd["audit_cols"].format(windows=" | ".join(ws))]
  for q in a["queue"][:top]:
    cells = " | ".join(f"{c['now_ms']:.2f}/{c['floor_ms']:.3f}/{c['gap_ms']:.2f} {c['label']}" for c in (q["windows"].get(w) for w in ws) if c)
    s, f = q["slope"], q["realization"]["factor"]
    slope = f"{s['delta_ms']:+.2f} {s['trend']}" if s else "-"  # one quote level: Python 3.11 (the floor) rejects reused quotes
    fac, score = "U" if f is None else f"{f:.2f}", "U" if q["score_s"] is None else f"{q['score_s']:.1f}"
    out.append(f"  {q.get('rank') or '-':>2} {wd['classes'].get(q['target'], q['target'])[:40]:<40} {cells}  gap {q['gap_s']:.1f} s  "
               f"slope {slope}  {q['lever']} x{fac} = {score} s")
  out.append(wd["flags"])
  out += [f"  {w['window']} {f['flag']:<13} {f['class']:<14} {f['kernel'][:44]} {f.get('grid') or ''} {f['ms']:.3f} ms: {f['why']}" for w in a["windows"] for f in w["flags"][:top]]
  pr = a["stock_proof"]
  return out + [wd["proof"].format(state="ok" if pr["ok"] else "FAIL" if pr["ok"] is False else "not judged", why=pr["why"]), a["refused_ranking"] or wd["ranked"]]
