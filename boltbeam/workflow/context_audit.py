"""The long-context audit: where a session's speed goes as its context grows, and whether the figures reconcile.

    depth buckets    a session's turns by the context they ran at (BUCKETS equal parts of the context, and the session):
                     tok/s, ms per step, tokens per step, acceptance, prompt ms per turn. A turn sits at turn_depth; a
                     step is collectors/time_budget.decode_steps with the engine's decode row (the one step rule).
    host time        time per step with no GPU work in flight, as a role: its parts at each depth, and its growth, slope
                     (ms per 1k) and constant from at least two depths. One depth gives no slope: "unknown", with the test.
    memory           the device's samples over a session: used at load, peak, the lowest free memory, what the device
                     keeps for itself (total - used - free, per sample). The margin is an estimate with its test. The
                     ledger: bytes held per role, and the unaccounted line up to the device's used memory.
    reconciliation   failing checks, never notes: roles sum to the step, the ledger sums to the device's used, the lowest
                     free memory covers the margin, every figure is labelled. A tolerance is the range of the record's
                     own repeats of that quantity on that configuration (workflow/repeat_bands.RULE); without them the
                     check fails and says the test. A failed check refuses the conclusion.
    output labels    each setting's effect on the answers: identical, quality-gated or untested, from the row that owns
                     the setting (arkey_ie_mods switches through workflow/mods, data/engine_knobs.json knobs and audit_flags). A
                     ranking's arms are run folders, their settings the recorded environment and command.
    placement        per class of a step's host operations, where its work goes and what removing it saves
                     (workflow/step_placement: a replay of the captured step, gated by G1 and G2; else idle upper bounds).
    expert cache     the covered share of the expert bytes held on the host and the engine's logged hit rates. An audit
                     metric only: nothing that sets a value reads it (docs/context-audit.md says what would change that).

Engine-blind: an engine is a key into the data rows; nothing here names one. Nothing here measures.
Labels: measured (source), derived (formula, inputs), estimate (formula, test), unknown (test, no value)."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from boltbeam.collectors.capture_scope import spread, strength
from boltbeam.collectors.time_budget import decode_steps  # the step rule's owner
from boltbeam.core.refused import Refused
from boltbeam.workflow.diff import refuse_unless_same  # the one authority for comparing runs
from boltbeam.workflow.repeat_bands import RULE

SCHEMA = "boltbeam.context_audit.v1"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
BUCKETS = 4  # a named constant, not derived: the context cut into this many equal parts (a 128k session: 32k each)
IDENTICAL, GATED, UNTESTED, UNJUDGED = OUTPUTS = ("identical", "quality-gated", "untested", "unjudged")  # best first
NEEDS = {"measured": ("source",), "derived": ("formula", "inputs"), "estimate": ("formula", "test"), "unknown": ("test",)}


def measured(value:float, source:str) -> dict[str, Any]:
  return {"value": value, "label": "measured", "source": source}


def derived(value:float | None, formula:str, inputs:Any) -> dict[str, Any]:
  return {"value": value, "label": "derived", "formula": formula, "inputs": inputs}


def unknown(test:str) -> dict[str, Any]:
  return {"value": None, "label": "unknown", "test": test}


def unlabelled(node:Any, path:str = "", ok:bool = False) -> list[str]:
  """Where a record breaks the label rule: a number whose own dict carries no label (a row or a figure; a list passes its
  dict's on, `inputs` are not figures), a label that is none of NEEDS or lacks what it must carry, an unknown with a value."""
  if isinstance(node, dict):
    bad = []
    if "label" in node:
      need = NEEDS.get(node["label"])
      bad = [f"{path}: label {node['label']!r}"] if need is None else [f"{path}: {node['label']} without its {k}" for k in need if not node.get(k)]
      bad += [f"{path}: unknown with a value"] if node["label"] == "unknown" and node.get("value") is not None else []
    return bad + [b for k, v in node.items() if k != "inputs" for b in unlabelled(v, f"{path}.{k}" if path else str(k), "label" in node)]
  if isinstance(node, (list, tuple)):
    return [b for i, v in enumerate(node) for b in unlabelled(v, f"{path}[{i}]", ok)]
  return [f"{path}: a number with no label"] if isinstance(node, (int, float)) and not isinstance(node, bool) and not ok else []


# --- tolerances: the record's own repeats ----------------------------------------------------------------------------

def band(rec:dict[str, Any], quantity:str) -> dict[str, Any]:
  """The tolerance for one quantity: the range of the record's repeats of it ({quantity, configuration, values}, n >= 2)
  on the record's own configuration. Repeats of another quantity or configuration are refused: unknown, with the test."""
  test = f"at least two more runs of this configuration, recording {quantity}"
  rows = [r for r in rec.get("repeats") or [] if r.get("quantity") == quantity]
  if not rows:
    return unknown(test + " (the record carries no repeats of it)")
  same = [r for r in rows if rec.get("configuration") is not None and r.get("configuration") == rec["configuration"]]
  if not same or len(same[0].get("values") or []) < 2:
    return unknown(test + " (its repeats are of another configuration, or fewer than two)")
  r, n = same[0], len(same[0]["values"])
  if not r.get("from"):
    raise Refused(f"a repeat row of {quantity} names no source (its `from`): a tolerance must say which runs it came from")
  try:
    width = spread(r["values"])["width_ms"]
  except ValueError as exc:
    raise Refused(f"the repeats of {quantity}: {exc}") from exc
  return {**measured(width, f"{r['from']}: the range of {n} repeats; {RULE}"), **strength(n)}


def within(name:str, left:float, tol:dict[str, Any], said:str, unit:str = "") -> dict[str, Any]:
  """One check for every sum: what is left over must lie within the tolerance; an unknown tolerance fails with its test."""
  ok = tol["value"] is not None and abs(left) <= tol["value"]
  tail = f"not judged to hold: the tolerance is unknown; test: {tol['test']}" if tol["value"] is None else \
    ("within" if ok else "beyond") + f" the range of {tol['n']} repeats, {tol['value']:,.2f}{unit}"
  return {"check": name, "ok": ok, "why": f"{said}; {tail}"}


# --- depth buckets -------------------------------------------------------------------------------------------------

def turn_depth(t:dict[str, Any]) -> float:
  """Where a turn sits, the one rule: the mean context of its decode, its prefix plus half its reply."""
  return t["prefix"] + t["predicted_n"] / 2


def depth_buckets(turns:list[dict[str, Any]], *, context:int, row:dict[str, Any], source:str) -> dict[str, Any]:
  """A session's turns (a replay's turn rows: prefix, predicted_n, predicted_ms, prompt_ms, draft_n, draft_n_accepted)
  by turn_depth, and the session row. row: the engine's decode row (data/kernel_families.json), for decode_steps."""
  how = {"label": "derived", "inputs": source,
         "formula": "tokens = decode_steps(predicted_n, 0), steps = decode_steps(predicted_n, accepted): tok/s = tokens / predicted_ms, step ms = "
                    "predicted_ms / steps, tokens per step = tokens / steps, acceptance = accepted / drafted, prompt ms per turn = mean(prompt_ms)"}

  for n, t in enumerate(turns, 1):
    if decode_steps(t["predicted_n"], t.get("draft_n_accepted") or 0, row) <= 0 or t["predicted_ms"] <= 0:
      raise Refused(f"turn {t.get('i', n)} has no decode step to time: {t['predicted_n']} tokens, "
                    f"{t.get('draft_n_accepted') or 0} accepted, {t['predicted_ms']:.1f} ms of decode")

  def cut(lo:float, hi:float | None) -> dict[str, Any] | None:
    ts = [t for t in turns if lo <= turn_depth(t) and (hi is None or turn_depth(t) < hi)]
    if not ts:
      return None
    ms, acc, drafted = sum(t["predicted_ms"] for t in ts), sum(t.get("draft_n_accepted") or 0 for t in ts), sum(t.get("draft_n") or 0 for t in ts)
    n, steps = sum(decode_steps(t["predicted_n"], 0, row) for t in ts), sum(decode_steps(t["predicted_n"], t.get("draft_n_accepted") or 0, row) for t in ts)
    return {**how, "lo": lo, "hi": hi, "turns": len(ts), "tokens": n, "tok_s": 1e3 * n / ms, "step_ms": ms / steps, "tokens_per_step": n / steps,
            "acceptance": acc / drafted if drafted else None, "prompt_ms_per_turn": sum(t.get("prompt_ms") or 0.0 for t in ts) / len(ts)}
  cuts = [0, *(context * i // BUCKETS for i in range(1, BUCKETS)), None]
  return {**how, "context": context, "rows": [r for lo, hi in zip(cuts, cuts[1:]) if (r := cut(lo, hi))], "session": cut(0, None),
          "step_rule": unknown("decode_budget's steps_expected against the verify forwards counted in a trace of the same turns")}


# --- host time as a role -------------------------------------------------------------------------------------------

def host_role(points:list[dict[str, Any]], *, apart:tuple[str, ...] = ()) -> dict[str, Any]:
  """Host time per step (no GPU work in flight) at each depth, with its growth, slope and constant. points: {depth
  (tokens), parts_ms {name: ms per step}, trace, traced, configuration}; apart: parts that are another role's (experts
  computed on the host), kept as lines and left out of the role's total. Depths are fitted only when captured alike:
  one configuration, all traced or none. One depth gives no slope; two give a line with no spread."""
  pts = sorted(points, key=lambda p: p["depth"])
  for p in pts[1:]:
    refuse_unless_same([(k, pts[0].get(k), p.get(k)) for k in ("traced", "configuration")])
  xs = [p["depth"] / 1e3 for p in pts]
  src = "; ".join(p["trace"] for p in pts) if all(p.get("trace") for p in pts) else ""  # a point that names no trace leaves the figure unsourced
  test = "the same capture at a second depth (one depth cannot tell a constant from growth)"

  def line(ys:list[float], **more) -> dict[str, Any]:
    if len(set(xs)) < 2:
      return {**measured(None, src), "at": ys, "growth_ms": unknown(test), "slope_ms_per_1k": unknown(test), "constant_ms": unknown(test), **more}
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    how = "the line through 2 points, no spread" if len(xs) == 2 else f"least squares over {len(xs)} depths"
    return {**measured(None, src), "at": ys, "growth_ms": derived(ys[-1] - ys[0], "deepest - shallowest", "at"),
            "slope_ms_per_1k": derived(slope, how, "at, depths"), "constant_ms": derived(my - slope * mx, how + ", at context 0", "at, depths"), **more}
  names = dict.fromkeys(n for p in pts for n in p["parts_ms"])
  return {**measured(None, src), "depths": [p["depth"] for p in pts], "traced": bool(pts[0].get("traced")),
          "host": line([sum(v for n, v in p["parts_ms"].items() if n not in apart) for p in pts]),
          "parts": {n: line([p["parts_ms"].get(n, 0.0) for p in pts], apart=n in apart) for n in names}}


def host_functions(spec:dict[str, Any], rec:dict[str, Any], *, engine:str, row:dict[str, Any], base:pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Host time by engine function (collectors/host_functions) and its checks. spec: records (the timer's file), release
  (its rows), and optionally capture (the same run's GPU captures), capture_reference (an untraced run's turn rows of the
  same turns: the capture's overhead), overhead {with_ms, without_ms, source} (the timer's), missing (the rows the
  build could not time, from tools/host_functions). Turns carry span_ns. Per depth: CPU time per function as host_role
  lines, traced and untraced apart (device waits apart); where a capture covers turns, their part in GPU-idle gaps,
  ranked: an upper bound on what removing a function saves, qualified only under a capture within the ceiling."""
  from boltbeam.collectors import host_functions as hf
  frow = hf.row(engine, spec["release"])
  turns = [{**t, "depth": turn_depth(t), "expect_steps": decode_steps(t["predicted_n"], t.get("draft_n_accepted") or 0, row)} for t in rec["turns"] if t.get("span_ns")]
  cap = hf.capture([base / c for c in spec["capture"]]) if spec.get("capture") else None
  try:
    with open(base / spec["records"]) as f:
      folded = hf.fold(hf.read(f, spec["records"]), turns, frow, cap)
  except OSError as exc:
    raise Refused(f"the timer's records cannot be read: {exc}") from None
  ref = {t["i"]: t for t in spec.get("capture_reference") or []}

  def cost(ids:list[int]) -> dict[str, Any]:  # the capture's: traced steps of these turns against the untraced run's
    step = lambda ts: sum(t["predicted_ms"] for t in ts) / sum(decode_steps(t["predicted_n"], t.get("draft_n_accepted") or 0, row) for t in ts)  # noqa: E731
    mine, theirs = [t for t in rec["turns"] if t.get("i") in ids], [ref[i] for i in ids if i in ref]
    return hf.overhead(step(mine), [step(theirs)] if len(theirs) == len(ids) else [], source="capture_reference", reference="the same turns untraced")
  edges = [rec["context"] * i // BUCKETS for i in range(1, BUCKETS)]
  pts = hf.points(folded, edges, trace=spec["records"], configuration=rec.get("configuration"), qualified=lambda ids: cost(ids).get("qualifies") is True)
  apart = tuple(k for p in pts for k in p["parts_ms"] if k.split("@")[0] in frow["device_wait"])
  idle = []
  for lo, hi in zip([0, *edges], [*edges, None]):
    got = [p for p in pts if p["lo"] == lo and p["traced"]]
    idle += [{"lo": lo, "hi": hi, "depth": p["depth"], "turns": p["turns"], "capture": cost(p["ids"]), **{k: v for k, v in p["idle"].items() if k not in ("parts_ns", "parts_ms", "window_ns")}}
             for p in got] or [{"lo": lo, "hi": hi, "value": None, "label": "unknown", "test": hf.IDLE_TEST}]
  out = {"untraced": host_role(u, apart=apart) if (u := [p for p in pts if not p["traced"]]) else None,
         "traced": host_role(t, apart=apart) if (t := [p for p in pts if p["traced"]]) else None, "idle": idle,
         "left_out": [f"turn {t.get('i')}: no span_ns" for t in rec["turns"] if not t.get("span_ns")], "skipped": folded["skipped"],
         "never_fired": folded["never_fired"], "missing": spec.get("missing") or [],
         "overhead": hf.overhead((o := spec.get("overhead") or {}).get("with_ms"), o.get("without_ms") or [], source=o.get("source", ""), reference="the run without the timer")}
  return out, [hf.closes(pts), hf.steps_check(folded), *([hf.clock_check(folded)] if cap else [])]


# --- memory: the device's samples, the margin, the ledger ------------------------------------------------------------

def memory_margin(m:dict[str, Any], peak_band:dict[str, Any]) -> dict[str, Any]:
  """The device's memory over one session. m: samples (time, used, free; one unit) from before the load to the end,
  total, load_at, end_at, warm_at (the first turn's end), from. Measured: used and free at load, the peak, the lowest
  free. Derived: growth, the largest rise between consecutive samples (over the session, and after warm-up), and what the
  device keeps for itself, total - used - free per sample (so free is what an allocation can get). The margin is an
  estimate: the range of the peak across repeats of this configuration (peak_band) plus the largest rise after warm-up,
  as the bound on a spike between two samples. The first request's growth is already in the peak, so it is not added."""
  samples, load, end, warm = m["samples"], m["load_at"], m.get("end_at"), m.get("warm_at")
  pre = [s for s in samples if s[0] <= load]
  post = [s for s in samples if s[0] > load and (end is None or s[0] <= end)]
  if not pre or not post:
    return {"refused": "no sample before the load" if not pre else "no sample after the load"}
  rises = [(b[0], b[1] - a[1]) for a, b in zip([pre[-1], *post], post)]
  late = None if warm is None else max([0.0, *(d for t, d in rises if t > warm)])
  kept = sorted(m["total"] - s[1] - s[2] for s in samples)
  got = {**measured(None, m.get("from", "")), "samples": len(samples), "after_load": len(post), "total": m["total"], "used_at_load": pre[-1][1],
         "free_at_load": pre[-1][2], "peak": max(s[1] for s in [pre[-1], *post]), "lowest_free": min(s[2] for s in [pre[-1], *post])}
  how = "growth = peak - used at load; rise = used less the sample before; reserved = total - used - free, per sample"
  test = "sample the device at 10 times the rate: a rise between two samples larger than the largest seen refutes it"
  if late is None or peak_band["value"] is None:
    margin = unknown(peak_band.get("test") if late is not None else "record when the first turn ended (warm_at), so the rises after warm-up can be told apart")
  else:
    margin = {"value": peak_band["value"] + late, "label": "estimate", "formula": "range of the peak across repeats + largest rise after warm-up", "test": test}
  return {"measured": got, "margin": margin, "peak_band": peak_band,
          "derived": {**derived(None, how, "measured, the samples"), "growth": got["peak"] - got["used_at_load"], "largest_rise": max([0.0, *(d for _, d in rises)]),
                      "largest_rise_after_warm_up": late, "reserved_median": kept[len(kept) // 2], "reserved_min": kept[0], "reserved_max": kept[-1]}}


def memory_ledger(rows:list[dict[str, Any]], *, used:float, tol:dict[str, Any]) -> dict[str, Any]:
  """Bytes held per role against the device's used memory. rows: figures (measured or derived) with a role. What they do
  not cover is the `unaccounted` line, never dropped. tol: band(record, "used_at_load")."""
  held = sum(r["value"] for r in rows)
  said = f"rows {held:,.1f} + unaccounted {used - held:,.1f} = used {used:,.1f}; what settles the unaccounted part: a measured row for it (the engine's allocations by owner)"
  return {**derived(None, "unaccounted = used - sum(rows)", "rows, the device's used memory"), "rows": rows, "held": held, "used": used,
          "unaccounted": used - held, "tolerance": tol, "check": within("ledger sums to device used", used - held, tol, said)}


def roles_sum(roles:dict[str, float], step_ms:float, tol:dict[str, Any], *, what:str = "") -> dict[str, Any]:
  """Roles against the measured step: the remainder is a named line (negative: the roles overlap in time, summed
  durations; positive: time no role took). tol: band(record, "step_ms")."""
  left = step_ms - sum(roles.values())
  said = f"{what + ': ' if what else ''}roles {sum(roles.values()):.2f} ms against the step {step_ms:.2f} ms: remainder {left:+.2f} ms" + \
    (" (the roles overlap: summed durations, not wall time)" if left < 0 else "")
  return {**within("roles sum to the step", left, tol, said, " ms"), **derived(None, "remainder = step - sum(roles)", "roles, step"), "remainder_ms": left}


def reconcile(checks:list[dict[str, Any]], record:Any = None) -> dict[str, Any]:
  """The checks, the label rule over the record, and the verdict: one failed check refuses the conclusion."""
  bad = unlabelled(record) if record is not None else []
  checks = [*checks, {"check": "every figure is labelled", "ok": not bad, "why": "all measured, derived, estimate or unknown" if not bad else "; ".join(bad[:5])}]
  failed = [c["check"] for c in checks if not c["ok"]]
  return {"checks": checks, "ok": not failed, "refused": "does not reconcile: " + "; ".join(failed) if failed else None}


# --- output labels ---------------------------------------------------------------------------------------------------

def _rows(name:str, key:str) -> list[dict[str, Any]]:
  return json.loads((DATA / name).read_text()).get(key) or []


def output_label(engine:str, setting:str, value:Any = None) -> dict[str, Any] | None:
  """What a setting does to an engine's answers, from the row that owns it (tune.output_of reads it): a mod's switch
  (arkey_ie_mods; off, empty or 0, is the stock engine), an audit flag, a knob by its name or an alias. None: no row names
  it. A switch that is both a mod's and a knob's takes the mod's label (the mods repo checks it), the knob's evidence
  kept beside it. The install-hook mod (install_paths) is untested unless every kernel its manifest names is a kernel
  row of the index, then the least of their labels."""
  from boltbeam.workflow.tune import output_of
  from boltbeam.workflow import mods
  knob = next((r for r in [*_rows("engine_knobs.json", "audit_flags"), *_rows("engine_knobs.json", "knobs")] if r["engine"] == engine
               and setting in (*(r.get("names") or [r.get("name")]), *(r.get("aliases") or []))), None)
  m = next((m for m in mods.mods(engine) if setting == mods.switch(m)), None)
  if m is None:
    return output_of(knob, value) if knob else None
  if str(value if value is not None else 1).strip() in ("", "0"):
    return {"output": IDENTICAL, "why": "off: the stock engine"}
  label, why = m["output"], f"arkey_ie_mods {m['id']}: its output evidence (mods.json at the pin)"
  if m["name"] == ((mods.pointer()["install_paths"].get(engine) or {}).get("mod")):
    label, why = UNTESTED, "the hook runs the kernels its manifest names; arkey_ie_mods has no kernel row at the pin to label one"
  return {"output": label, "why": why + (f"; the knob row: {knob['output_why']}" if knob and knob.get("output_why") else "")}


def _flags(engine:str) -> tuple[set[str], set[str]]:
  """The engine's switches (flags that take no value: knob rows of kind switch, the audit rows' and the neutral row's) and
  its output-neutral flags (data/engine_knobs.json output_neutral)."""
  rows = [r for r in [*_rows("engine_knobs.json", "audit_flags"), *_rows("engine_knobs.json", "output_neutral")] if r["engine"] == engine]
  knobs = [r for r in _rows("engine_knobs.json", "knobs") if r["engine"] == engine and r["kind"] == "switch"]
  switches = {f for r in rows for f in r.get("switches") or []} | {f for r in knobs for f in (r["name"], *(r.get("aliases") or []))}
  return switches, {f for r in _rows("engine_knobs.json", "output_neutral") if r["engine"] == engine for f in r["names"]}


def arm_of(run:pathlib.Path, engine:str, row:dict[str, Any]) -> dict[str, Any]:
  """A run folder as a ranking's arm: its session tok/s from its replay turns, and its settings as recorded: every engine
  environment variable (one no row names is untested), and every command flag (expert_placement.flag_pairs, the one
  parser; a switch present is on). A flag no row judges and not on the engine's output-neutral list is `unjudged`. A run
  whose command was not recorded has no settings: refused, never read as stock."""
  from boltbeam.collectors.expert_placement import flag_pairs
  from boltbeam.workflow.tune import output_of

  def read(name:str) -> dict[str, Any]:
    return json.loads((run / name).read_text()) if (run / name).is_file() else {}
  wp, trace = read("workload_profile.json"), read("timing_trace.json")
  args = (wp.get("engine_settings") or {}).get("engine_args")
  turns = [r for r in trace.get("rows") or [] if r.get("workload") == "replay" and r.get("scope") == "turn"]
  out = {**measured(None, f"{run.name}: timing_trace.json, workload_profile.json"), "name": run.name,
         "tok_s": depth_buckets(turns, context=max(t["prefix"] for t in turns), row=row, source=run.name)["session"]["tok_s"] if turns else None}
  if args is None:
    return {**out, "settings": None, "unjudged": []}
  switches, neutral = _flags(engine)
  pairs = [(f, "on" if f in switches else v) for f, v in flag_pairs([str(a) for a in args], switches)]
  env = {k: output_label(engine, k, v) or output_of(None) for k, v in (wp.get("engine_env") or {}).items()}
  named = {f: l for f, v in pairs if (l := output_label(engine, f, v))}
  return {**out, "settings": {**env, **named}, "unjudged": [f for f, _ in pairs if f not in named and f not in neutral]}


def rank(arms:list[dict[str, Any]]) -> dict[str, Any]:
  """Arms (arm_of) fastest first, each with the worst output label among its settings. An arm with an untested setting,
  an unjudged flag, no recorded settings or no session speed is refused, with the reason (fail closed)."""
  ranked, refused = [], []
  for a in arms:
    s = a["settings"]
    worst = UNTESTED if s is None else UNJUDGED if a["unjudged"] else max((l["output"] for l in s.values()), key=OUTPUTS.index, default=IDENTICAL)
    why = ("its settings are not recorded" if s is None else "flags no row judges: " + ", ".join(a["unjudged"]) if a["unjudged"] else
           "output untested: " + ", ".join(k for k, l in s.items() if l["output"] == UNTESTED) if worst == UNTESTED else
           "no replay turns to time" if a["tok_s"] is None else None)
    (refused if why else ranked).append({**a, "output": worst, **({"why": why} if why else {})})
  return {"ranked": sorted(ranked, key=lambda r: -r["tok_s"]), "refused": refused}


# --- expert cache coverage (audit only) ------------------------------------------------------------------------------

def expert_cache(engine:str, *, log:str = "", cache:float | None = None, host:float | None = None, source:str = "") -> dict[str, Any] | None:
  """The engine's expert cache: the covered share of the expert bytes held on the host, and its hit rate per batch class
  when the engine logs them. The engine's audit_flags row (role expert_cache) gives the log patterns; cache and host
  (one unit) stand in where the log does not say them. None for an engine with no such row."""
  row = next((r for r in _rows("engine_knobs.json", "audit_flags") if r["engine"] == engine and r.get("role") == "expert_cache"), None)
  if row is None:
    return None
  pats = row.get("log") or {}
  if size := re.search(pats.get("size", "$^"), log):
    cache, host = float(size["cache"]), float(size["host"])
  rates = {m["batch"]: measured(int(m["hits"]) / max(1, int(m["hits"]) + int(m["misses"])), source or "the engine's log") for m in re.finditer(pats.get("hits", "$^"), log)}
  share = derived(min(1.0, cache / host), "cache bytes / expert bytes on the host", {"cache": cache, "host": host}) if cache and host else \
    unknown("the cache's size and the host experts' bytes: the engine's log line, or the device's used memory with and without the host layers")
  from boltbeam.workflow.tune import output_of
  return {"coverage": share, "hit_rate": rates or unknown("run the engine at the verbosity that logs its cache's hits and misses"), **output_of(row)}


# --- the audit of one record -----------------------------------------------------------------------------------------

def audit(rec:dict[str, Any], *, engine:str, row:dict[str, Any], base:pathlib.Path = pathlib.Path(".")) -> dict[str, Any]:
  """One input record's audit. rec: context, turns, source, configuration, repeats, and any of memory, ledger {rows,
  used, outside?, shapes + unit_bytes?}, host_points (+ host_apart), host_functions (host_functions()), roles {roles, step_ms}, arms (run folders, relative to base), expert_cache {log |
  cache, host}. The token rule is the engine's row: a record that states its own is refused. A failed check sets `refused`."""
  if "tokens_from_prompt" in rec:
    raise Refused("the record states tokens_from_prompt: the step rule is the engine's decode row (data/kernel_families.json), not the record's")
  out: dict[str, Any] = {"schema": SCHEMA}
  checks = []
  if rec.get("turns"):
    out["buckets"] = depth_buckets(rec["turns"], context=rec["context"], row=row, source=rec.get("source", ""))
  if m := rec.get("memory"):
    g = out["memory"] = memory_margin(m, band(rec, "peak"))
    if "refused" in g:
      checks.append({"check": "memory margin", "ok": False, "why": g["refused"]})
    else:
      low, need = g["measured"]["lowest_free"], g["margin"]
      checks.append({"check": "memory margin", "ok": need["value"] is not None and low >= need["value"],
                     "why": f"lowest free over the session {low:,.0f} against " + (f"the margin {need['value']:,.0f} (estimate)" if need["value"] is not None else f"an unknown margin; test: {need['test']}")})
  if l := rec.get("ledger"):
    rows, beside = l["rows"], l.get("outside") or []  # outside: memory the device keeps beyond used (reserved), measured
    if l.get("shapes"):  # each held row beside the least the model's shapes need (workflow/min_memory)
      from boltbeam.workflow import min_memory as mm
      least = mm.of_record(l, base)
      rows, beside = mm.gaps(rows, least), mm.gaps(beside, least)
      checks += mm.checks([*rows, *beside], least)
    out["ledger"] = {**memory_ledger(rows, used=l["used"], tol=band(rec, "used_at_load")), "outside": beside}
    checks.append(out["ledger"]["check"])
  if rec.get("host_points"):
    out["host"] = host_role(rec["host_points"], apart=tuple(rec.get("host_apart") or ()))
  if h := rec.get("host_functions"):
    out["host_functions"], more = host_functions(h, rec, engine=engine, row=row, base=base)
    checks += more
  if r := rec.get("roles"):
    checks.append(roles_sum(r["roles"], r["step_ms"], band(rec, "step_ms"), what=r.get("what", "")))
  if rec.get("arms"):
    out["ranking"] = rank([arm_of(base / a, engine, row) for a in rec["arms"]])
  if e := rec.get("expert_cache"):
    out["expert_cache"] = expert_cache(engine, **e)
  if p := rec.get("placement"):  # where each host operation of a step runs (workflow/step_placement)
    from boltbeam.workflow import step_placement
    out["placement"] = step_placement.section(p, rec, engine=engine, base=base)
  return {**out, **reconcile(checks, out)}


def lines(out:dict[str, Any]) -> list[str]:
  """The audit as text: the bucket table, host time, memory, the ranking's refusals, then every check and the verdict."""
  said = []
  if b := out.get("buckets"):
    said.append("| context | turns | tok/s | step ms | tokens/step | acceptance | prompt ms/turn |\n|---|---|---|---|---|---|---|")
    for r in [*b["rows"], b["session"]]:
      name = "session" if r is b["session"] else f"{r['lo']:,} - " + ("end" if r["hi"] is None else f"{r['hi']:,}")
      acc = "-" if r["acceptance"] is None else f"{r['acceptance']:.4f}"
      said.append(f"| {name} | {r['turns']} | {r['tok_s']:.1f} | {r['step_ms']:.2f} | {r['tokens_per_step']:.3f} | {acc} | {r['prompt_ms_per_turn']:.0f} |")
    said.append(f"derived by the engine's step rule; the step count is unknown against a trace; test: {b['step_rule']['test']}")
  if h := out.get("host"):
    v = lambda f: "unknown: needs " + f["test"] if f["value"] is None else f"{f['value']:.4g} ({f['formula']})"  # noqa: E731
    at = ", ".join(f"{y:.2f} ms at {d / 1e3:.1f}k" for y, d in zip(h["host"]["at"], h["depths"]))
    said.append(f"host time per step ({'traced' if h['traced'] else 'untraced'}): {at}; growth {v(h['host']['growth_ms'])} ms; slope {v(h['host']['slope_ms_per_1k'])} ms per 1k")
  if f := out.get("host_functions"):
    for p in f["idle"]:
      at = f"context {p['lo']:,} - " + ("end" if p["hi"] is None else f"{p['hi']:,}")
      said.append(f"host functions, {at}: " + (p["test"] if p["label"] == "unknown" else
                  f"{p['turns']} captured turn(s) at {p['depth'] / 1e3:.1f}k, traced ({p['capture']['words'] if p['capture']['value'] is not None else 'capture overhead unknown'}); "
                  f"GPU idle {p['gpu_idle_ms']:.3f} ms a step; by the part in GPU-idle gaps, an upper bound on what a fix saves"
                  + ("" if p["label"] == "measured" else " (unqualified: the capture's overhead is unknown or over the ceiling)") + ", CPU beside: "
                  + ", ".join(f"{r['part']} {r['idle_ms']:.3f} ({r['cpu_ms']:.3f})" for r in p["rank"][:8])))
    o = f["overhead"]
    said.append("the timer's overhead: " + (o["words"] if o["value"] is not None else f"unknown; test: {o['test']}"))
    said += [f"host functions {what}: " + "; ".join(f[k]) for what, k in (("not timed", "missing"), ("never fired", "never_fired"), ("left out", "left_out"), ("skipped", "skipped")) if f[k]]
  if (m := out.get("memory")) and "refused" not in m:
    g, d = m["measured"], m["derived"]
    said.append(f"memory: used at load {g['used_at_load']:,.0f}, peak {g['peak']:,.0f}, growth {d['growth']:,.0f}, largest rise {d['largest_rise']:,.0f}, "
                f"free at load {g['free_at_load']:,.0f}, lowest free {g['lowest_free']:,.0f}; the device keeps {d['reserved_median']:,.0f}")
  if (l := out.get("ledger")) and any("minimum" in r for r in l["rows"]):
    v = lambda f: "-" if f["value"] is None else f"{round(f['value'], 1) + 0.0:,.1f}"  # noqa: E731
    said.append("| role | held | minimum | gap | class |\n|---|---|---|---|---|")
    for r in [*l["rows"], *l["outside"]]:
      gap = f"<= {v(r['gap'])}" if r["class"] == "bounded" else v(r["gap"])
      why = ": " + (r["minimum"].get("rest") or r["minimum"].get("test")) if r["class"] != "exact" else ""
      more = [w for w, on in ((f"output {r['minimum'].get('output')}", "output" in r["minimum"]), ("outside used", r in l["outside"]),
                              (f"held is an estimate: {r.get('test')}", r["label"] == "estimate")) if on]
      said.append(f"| {r['role']} | {r['value']:,.1f} ({r['label']}) | {v(r['minimum'])} | {gap} | {r['class']}{why}" + (f" ({'; '.join(more)})" if more else "") + " |")
    said.append(f"exact gaps {sum(r['gap']['value'] for r in l['rows'] if r['class'] == 'exact'):,.1f}; unaccounted {l['unaccounted']:,.1f}")
  rk = out.get("ranking") or {}
  for r in rk.get("ranked", []):
    said.append(f"ranked {r['name']}: {r['tok_s']:.1f} tok/s, output {r['output']}; " + ", ".join(f"{k} {l['output']}" for k, l in r["settings"].items()))
  said += [f"not ranked, {r['name']}: {r['why']}" + (f"; labels: " + ", ".join(f"{k} {l['output']}" for k, l in r["settings"].items()) if r.get("settings") else "")
           for r in rk.get("refused", [])]
  if e := out.get("expert_cache"):
    c, h = e["coverage"], e["hit_rate"]
    said.append(f"expert cache (audit only), output {e['output']}: coverage " + (f"{c['value']:.0%} ({c['formula']})" if c["value"] is not None else f"unknown; test: {c['test']}")
                + "; hit rate " + (", ".join(f"batches {b} {v['value']:.1%}" for b, v in h.items()) if "label" not in h else f"unknown; test: {h['test']}"))
  if pl := out.get("placement"):
    said += [f"placement {'pass' if g['ok'] else 'FAIL'} {g['gate']}: {g['why']}" for g in pl["gates"]]
    said.append("placement: " + ("the replay decides (G1 and G2 pass): savings are its predictions" if pl["decides"] else
                                 "the replay decides nothing (a gate failed or is missing): savings are host time in GPU-idle gaps, upper bounds"))
    for name, w in pl["windows"].items():
      said.append(f"| {name}: class | rate | goes to | can | saving ms a step |\n|---|---|---|---|---|")
      for r in w["rows"]:
        v = r["saving_ms"]
        save = "-" if v is None else f"unknown: {v['test']}" if v["value"] is None else f"{v['value']:.3f} ({v['label']})"
        said.append(f"| {r['class']} ({', '.join(r['labels'][:4])}{', ...' if len(r['labels']) > 4 else ''}) | {r['rate']} | {r['side']} | {r['can']} | {save} |")
      said.append(f"{name}: cuts a step " + ", ".join(f"{k} {v['value']:.1f}" for k, v in sorted(w["cuts"].items()))
                  + (f"; every cut not held at home removed: {w['all_cuts_ms']['value']:.3f} ms a step (derived, if the unknown preconditions hold)" if "all_cuts_ms" in w else "")
                  + (f"; {len(w['refused'])} step(s) refused: {w['refused'][0]}" if w["refused"] else ""))
  said += [f"{'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['why']}" for c in out["checks"]]
  return said + ([out["refused"]] if out["refused"] else [])
