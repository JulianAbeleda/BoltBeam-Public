"""`boltbeam tune MODEL --engine E`: this machine's values for an engine's knobs, each with where it came from, from one table
(data/engine_knobs.json): a row per knob, keyed by engine, says what it controls and how it is set.

  facts     a formula over named facts (host RAM and L3, the model's bytes, the GPU, the engine's own facts): instant
  measured  a short real-text prefill sweep under --measure, over the row's axis knob, with the row's points, tie and
            stop rule; without --measure the facts value or the default stands
  default   left at the default, with the evidence why

This module is engine-blind (policy/guards.py checks it): the table names engines, and each engine's adapter (the
module its table row names) does the four engine-specific things through one interface:

  detect(build) -> features       what the build is and the features it has, read without running a model
  facts(model, host, features, options) -> facts   the model's facts by their generic names, the engine's own as ns.name
  launch_measure(model, features, facts, port, out, extra) -> run(values, prompt, reps, tag) -> wall ms
  render(values, facts, features, model, extra) -> {env, flags, config, argv, command}

and declares NS (its fact namespace) and FACTS (its facts and what they are). Adding an engine is an adapter module and
rows; nothing here changes."""
from __future__ import annotations

import ast
import datetime
import importlib
import json
import os
import pathlib
from typing import Any, Callable

from boltbeam.core.refused import Refused

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "engine_knobs.json"
KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
PROMPT = 10000  # the prompt length the setup is stated at (the evidence's 10k)
FACTS = {  # the engine-neutral names a formula may read: the fact and what it is
  "host_expert_bytes": "the experts this placement keeps in host RAM (the model's tensor table, per layer)",
  "model_experts": "experts per layer (0: a dense model)",
  "model_weight_bytes": "the weights the engine holds on the GPU",
  "model_draft_layers": "the checkpoint's draft (multi-token prediction) layers",
  "kv_bytes_per_token": "KV cache bytes per token at the model's own type",
  "usable_ram_bytes": "available RAM less the margin, floored to a GiB (target/host.usable_ram)",
  "ram_available_bytes": "RAM available now (/proc/meminfo MemAvailable, macOS vm_stat)",
  "ram_total_bytes": "RAM installed", "ram_margin_bytes": "RAM kept free for the system and the engine (target/host.RAM_MARGIN_BYTES)",
  "l3_bytes": "one L3 instance (target/host.l3_cache)", "page_bytes": "the memory page size",
  "gpu_total_bytes": "the GPU's memory", "gpu_free_bytes": "the GPU's memory free now",
  "prompt_tokens": "the prompt length the setup is stated at", "context_tokens": "the context the setup must hold",
  "unified_memory": "the GPU reads host memory directly (no copies)",
  "value": "the row's own value (in sets)"}
FUNCS: dict[str, Callable[..., Any]] = {"min": min, "max": max, "pow2_floor": lambda x: 1 << (int(x).bit_length() - 1) if x >= 1 else 0,
                                         "gib_floor": lambda x: int(x // GiB), "gib_ceil": lambda x: int(-(-x // GiB)),
                                         "floor_to": lambda x, step: round(int(x / step) * step, 6)}
CONSTS = {"KiB": KiB, "MiB": MiB, "GiB": GiB, "True": True, "False": False, "None": None}


def table(path:pathlib.Path = TABLE) -> dict[str, Any]:
  return json.loads(path.read_text())


def engines(tab:dict[str, Any] | None = None) -> list[str]:
  return list((tab or table())["engines"])


def adapter(engine:str, tab:dict[str, Any] | None = None):
  """The engine's adapter module, as its table row names it."""
  tab = tab or table()
  if engine not in tab["engines"]:
    raise Refused(f"BoltBeam has no knob table for {engine}; it has one for {', '.join(tab['engines'])}")
  return importlib.import_module(tab["engines"][engine]["adapter"])


def knobs(engine:str, tab:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  return [k for k in (tab or table())["knobs"] if k["engine"] == engine]


def fact_names(engine:str, tab:dict[str, Any] | None = None) -> set[str]:
  """The facts a row of this engine may read: the generic ones and its adapter's namespaced ones."""
  ad = adapter(engine, tab)
  return set(FACTS) | {f"{ad.NS}.{n}" for n in ad.FACTS}


# --- formulas: a restricted expression over named facts -------------------------------------------------------------------

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: _div(a, b, False),
        ast.FloorDiv: lambda a, b: _div(a, b, True)}
_CMP = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
        ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


class Missing(Exception):
  """A formula read a fact this machine or model did not give (or divided by a zero one)."""


def _div(a, b, floor:bool):
  if not b:
    raise Missing("a zero divisor")
  return a // b if floor else a / b


def _name(n:ast.AST) -> str | None:
  """A fact's name: x, or an engine's own ns.x."""
  if isinstance(n, ast.Name):
    return n.id
  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
    return f"{n.value.id}.{n.attr}"
  return None


def names_in(expr:str) -> list[str]:
  """The facts a formula reads, in order, without functions and constants."""
  out = []
  for n in ast.walk(ast.parse(expr, mode="eval")):
    if isinstance(n, ast.Attribute) and (got := _name(n)):
      out.append(got)
    elif isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTS:
      out.append(n.id)
  dotted = {o.split(".")[0] for o in out if "." in o}
  return list(dict.fromkeys(o for o in out if o not in dotted))


def evaluate(expr:str, facts:dict[str, Any], allowed:set[str] | None = None) -> Any:
  """The formula's value over facts. Only fact names (FACTS, or `allowed` when given), numbers, strings, arithmetic,
  comparisons, and/or/not, x if c else y and FUNCS: anything else is a ValueError (the table is data, never code). A
  fact that is None raises Missing."""
  allowed = allowed if allowed is not None else set(FACTS) | {k for k in facts if "." in k}

  def ev(n:ast.AST) -> Any:
    if isinstance(n, ast.Expression):
      return ev(n.body)
    if isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, (int, float, bool, str))):
      return n.value
    name = _name(n)
    if name is not None:
      if name in CONSTS:
        return CONSTS[name]
      if name not in allowed and name != "value":
        raise ValueError(f"{name!r} is not a fact a knob formula may read")
      if name == "value":  # a row's own value is None when it keeps the default
        return facts.get("value")
      if facts.get(name) is None:
        raise Missing(name)
      return facts[name]
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
      return _BIN[type(n.op)](ev(n.left), ev(n.right))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.Not)):
      v = ev(n.operand)
      return -v if isinstance(n.op, ast.USub) else not v
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
      return _CMP[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
    if isinstance(n, ast.BoolOp):
      vals = (ev(v) for v in n.values)
      return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.IfExp):
      return ev(n.body) if ev(n.test) else ev(n.orelse)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
      return FUNCS[n.func.id](*(ev(a) for a in n.args))
    raise ValueError(f"{ast.dump(n)[:60]} is not allowed in a knob formula")
  return ev(ast.parse(expr, mode="eval"))


class _Unknown:
  def __format__(self, spec:str) -> str:
    return "unknown"

  def __str__(self) -> str:
    return "unknown"


class _View(dict):
  def __missing__(self, key):
    return _Unknown()

  def __getattr__(self, key):
    return self[key]


def say(template:str, facts:dict[str, Any]) -> str:
  """A row's reason with its facts filled in: *_bytes also as *_gib and *_mib, ns.x as an engine's own, a missing or
  None fact as "unknown"."""
  view = _View()
  for k, v in facts.items():
    ns, _, key = k.rpartition(".")
    into = view.setdefault(ns, _View()) if ns else view
    if not isinstance(into, _View):
      continue
    into[key] = _Unknown() if v is None else v
    if key.endswith("_bytes"):
      base = key[:-6]
      into[f"{base}_gib"], into[f"{base}_mib"] = (_Unknown(), _Unknown()) if not isinstance(v, (int, float)) else (v / GiB, v / MiB)
  return template.format_map(view)


# --- the host: engine-neutral facts --------------------------------------------------------------------------------------

def host_facts(*, memory:dict[str, Any] | None = None, l3:dict[str, Any] | None = None, prompt:int = PROMPT,
               context:int | None = None) -> dict[str, Any]:
  """RAM, L3, the page size and the prompt and context the setup is stated at, with "sources". memory and l3 stand in
  for the readings (tests)."""
  from boltbeam.target import host
  mem = memory or host.memory()
  ram = host.usable_ram(mem["total_bytes"], mem.get("available_bytes"))
  l3 = l3 or host.l3_cache()
  page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
  out = {"usable_ram_bytes": ram["usable_bytes"], "ram_available_bytes": ram.get("available_bytes"), "ram_total_bytes": ram["total_bytes"],
         "ram_margin_bytes": ram["margin_bytes"], "l3_bytes": l3.get("bytes"), "l3_source": l3["source"], "page_bytes": page,
         "prompt_tokens": prompt, "context_tokens": context or prompt}
  out["sources"] = {"usable_ram_bytes": ram["says"], "ram_available_bytes": mem["source"], "ram_total_bytes": mem["source"],
                    "ram_margin_bytes": "target/host.RAM_MARGIN_BYTES", "l3_bytes": l3["source"], "page_bytes": "os.sysconf SC_PAGE_SIZE",
                    "prompt_tokens": "--prompt", "context_tokens": "--context" if context else "--prompt (no --context given)"}
  return out


# --- the rows ----------------------------------------------------------------------------------------------------------

def _shown(v:Any) -> Any:
  return round(v / GiB, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= GiB else v


def groups_in(engine:str, features:dict[str, Any], tab:dict[str, Any]) -> dict[str, bool]:
  """The engine's named extras (local patches) and whether the build has each (its marker is a feature)."""
  return {g: v["marker"] in features["features"] for g, v in tab["engines"][engine]["groups"].items()}


def derive(f:dict[str, Any], features:dict[str, Any], engine:str, tab:dict[str, Any] | None = None,
           measured:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """Every row of the engine on this build and these facts: {name, kind, group, status (set | default | not_applicable |
  not_in_build), value, default, source (derived | measured | default), reason, provenance}. measured: --measure's
  result per knob name ({value, ...} or {status: failed}), which overrides the facts value."""
  tab, measured, f = tab or table(), measured or {}, dict(f)
  allowed = fact_names(engine, tab)
  rows = []
  for k in knobs(engine, tab):
    row = {"name": k["name"], "kind": k["kind"], "group": k.get("group"), "default": k["default"], "controls": k["controls"],
           "bitwise": k.get("bitwise", True), "with": k.get("with")}
    lacks = [r for r in k.get("requires", []) if r not in features["features"]]
    if lacks:
      rows.append({**row, "status": "not_in_build", "value": None, "source": None, "reason": None,
                   "provenance": {"build": features["source"], "lacks": lacks,
                                  **({"group": tab["engines"][engine]["groups"][k["group"]]["file"]} if k.get("group") else {})}})
      continue
    unread = None
    try:
      applies = evaluate(k.get("applies", "True"), f, allowed)
    except Missing as m:
      applies, unread = False, str(m)
    if not applies:
      reason = _missing(unread, adapter(engine, tab).FACTS) if unread else say(k.get("not_applicable", ""), f)
      rows.append({**row, "status": "not_applicable", "value": None, "source": None, "reason": reason,
                   "provenance": {"applies": k.get("applies"), "inputs": {n: _shown(f.get(n)) for n in names_in(k.get("applies", "True"))}}})
      continue
    value, source, reason = None, "default", say(k.get("reason_default") or k.get("reason", ""), f)
    prov: dict[str, Any] = {"evidence": k.get("evidence"), "features": {r: features["features"][r] for r in k.get("requires", [])}}
    if k["derive"] in ("facts", "measured") and k.get("formula"):
      inputs = names_in(k["formula"])
      prov.update(formula=k["formula"], inputs={n: {"value": _shown(f.get(n)), "source": f["sources"].get(n)} for n in inputs})
      try:
        value = evaluate(k["formula"], f, allowed)
        source = "derived"
        reason = say(k["reason"] if value is not None else k.get("reason_default") or k["reason"], f)
      except Missing as m:
        prov["missing"] = str(m)
        if k.get("reason_missing"):
          reason = say(k["reason_missing"], f)
    got = measured.get(k["name"])
    if got and got.get("value") is not None:
      value, source = got["value"], "measured"
      reason = got["says"]
      prov["sweep"] = got
    elif got:
      prov["sweep"] = got
      reason += f" (--measure: {got.get('says') or got.get('why')})"
    if k["derive"] == "measured" and not got:
      prov["measure"] = f"confirm with --measure: {k['sweep']['type']} over {k['sweep']['axis']} {k['sweep']['points']}"
    for name, expr in (k.get("sets") or {}).items():
      try:
        f[name] = evaluate(expr, {**f, "value": value}, allowed)
      except Missing:
        f[name] = None
    shown = str(value) if value is not None else k["default"]
    rows.append({**row, "status": "set" if value is not None and shown != k["default"] else "default", "value": shown, "source": source,
                 "reason": reason, "provenance": prov})
  return rows


def _words() -> dict[str, Any]:
  from boltbeam.report.plain import words
  return words()


def fact_words(fact:str, engine_facts:dict[str, str] | None = None) -> str:
  """A fact as a reader knows it: what it is (FACTS, or the adapter's), with no code names or module paths."""
  import re
  what = FACTS.get(fact) or (engine_facts or {}).get(fact.rpartition(".")[2]) or fact.rpartition(".")[2].replace("_", " ")
  return re.sub(r"\s*\([^()]*[/.][^()]*\)", "", what).strip()


def _missing(fact:str, engine_facts:dict[str, str] | None = None) -> str:
  return _words()["tune"]["missing"].format(fact=fact_words(fact, engine_facts))


def values_of(rows:list[dict[str, Any]], **over:Any) -> list[dict[str, Any]]:
  """The rows set away from their defaults as {name, kind, value, with}, with `over` (name -> value) replacing or adding."""
  kinds = {r["name"]: r for r in rows}
  out = {r["name"]: r["value"] for r in rows if r["status"] == "set"}
  out.update({k: str(v) for k, v in over.items()})
  return [{"name": n, "kind": kinds[n]["kind"], "value": v, "with": kinds[n].get("with")} for n, v in out.items()]


# --- --measure: sweeps over a knob's axis --------------------------------------------------------------------------------

def sweep(knob:dict[str, Any], run:Callable[..., float], rows:list[dict[str, Any]], *, prompt:int) -> dict[str, Any]:
  """A row's sweep. crossover: per point of the axis knob (ascending) the knob's on and off arms, alternating order; the
  answer is the first point from which on beats off by more than the tie at two points in a row (or at the last point).
  best: one arm per point, the smallest point within the tie of the fastest; it stops after two points in a row slower
  than the best by more than the tie. run(values=, prompt=, reps=, tag=) -> wall ms; values are the setup's with the
  axis (and the arm) replaced."""
  s = knob["sweep"]
  tie, reps = s["tie_pct"] / 100, s["reps"]
  prompt_n = prompt if s.get("prompt") == "prompt_tokens" else int(s.get("prompt") or prompt)
  pts = [p for p in s["points"] if p < prompt_n] + [p for p in s["points"] if p >= prompt_n][:1]  # a chunk past the prompt is the prompt
  out, answer = [], None
  if s["type"] == "crossover":
    wins = 0
    for i, p in enumerate(pts):
      arms = {"on": s["on"].format(point=p), "off": s["off"]}
      ms = {}
      for arm in (("off", "on") if i % 2 == 0 else ("on", "off")):
        ms[arm] = run(values=values_of(rows, **{s["axis"]: p, knob["name"]: arms[arm]}), prompt=prompt_n, reps=reps, tag=f"{knob['name']}-{p}-{arm}")
      win = ms["on"] < ms["off"] * (1 - tie)
      out.append({"point": p, "on_ms": round(ms["on"], 1), "off_ms": round(ms["off"], 1), "on_wins": win})
      wins = wins + 1 if win else 0
      if wins == 2:
        answer = pts[i - 1]
        break
    if answer is None and out and out[-1]["on_wins"]:
      answer = out[-1]["point"]
    low = " (or lower: it already wins at the smallest point swept)" if answer is not None and answer == pts[0] else ""
    says = (f"measured crossover: on beats off by more than {s['tie_pct']}% from {s['axis']} {answer}{low} ({prompt_n}-token real-text prompt, wall: "
            + ", ".join(f"{r['point']}: {r['on_ms']:.0f} vs {r['off_ms']:.0f} ms" for r in out) + ")"
            if answer is not None else f"measured: on never beat off by more than {s['tie_pct']}% up to {s['axis']} {pts[-1] if pts else '-'}; "
            "the default is not confirmed (the crossover may be higher)")
  else:
    best = None
    worse = 0
    for p in pts:
      ms = run(values=values_of(rows, **{s["axis"]: p}), prompt=prompt_n, reps=reps, tag=f"{knob['name']}-{p}")
      out.append({"point": p, "ms": round(ms, 1), "tok_s": round(prompt_n * 1000 / ms, 1)})
      best = ms if best is None else min(best, ms)
      worse = worse + 1 if ms > best * (1 + tie) else 0
      if worse == 2:
        break
    answer = next(r["point"] for r in out if r["ms"] <= best * (1 + tie)) if out else None
    if answer is None:
      return {"value": None, "points": [], "prompt": prompt_n, "tie_pct": s["tie_pct"], "stop": s["stop"], "says": f"no {s['axis']} point to sweep"}
    says = f"measured: {s['axis']} {answer} is the smallest within {s['tie_pct']}% of the fastest ({prompt_n}-token real-text prompt, wall: " + \
           ", ".join(f"{r['point']}: {r['tok_s']:.0f} tok/s" for r in out) + ")"
  return {"value": answer, "points": out, "prompt": prompt_n, "tie_pct": s["tie_pct"], "stop": s["stop"], "says": says}


def tune(model:str, engine:str, build:str | None = None, *, prompt:int = PROMPT, context:int | None = None, measure:bool = False,
         knobs_only:list[str] | None = None, port:int = 0, out:pathlib.Path | None = None, extra:list[str] | None = None,
         experts_in_ram:str | None = None, run:Callable[..., float] | None = None, tab:dict[str, Any] | None = None,
         memory:dict[str, Any] | None = None, l3:dict[str, Any] | None = None, say:Callable[[str], None] | None = None,
         **options) -> dict[str, Any]:
  """The setup: the build, the facts, every row, the rendered command, and (measure) the sweeps' records. A Refused when
  the engine, its build or the model cannot be read. options go to the adapter's facts (test stand-ins). say: --measure's
  progress as a screen's pipeline lines (a stage per swept knob, a progress line per point, at most the row's points)."""
  tab = tab or table()
  ad = adapter(engine, tab)
  if knobs_only and not measure:  # --knob picks what --measure sweeps; alone it would be silently ignored
    raise Refused("--knob names the knobs --measure sweeps; without --measure nothing is swept, so it is refused. Add --measure, or drop --knob")
  if knobs_only:  # a flag's name may be given without its leading dashes
    names = {k["name"].lstrip("-"): k["name"] for k in knobs(engine, tab)}
    if unknown := sorted(q for q in knobs_only if q.lstrip("-") not in names):
      raise Refused(f"{engine} has no knob named {', '.join(unknown)} in BoltBeam's knob table")
    knobs_only = [names[q.lstrip("-")] for q in knobs_only]
  feats = ad.detect(build)
  f = host_facts(memory=memory, l3=l3, prompt=prompt, context=context)
  got = ad.facts(model, f, feats, {"experts_in_ram": experts_in_ram, **options})
  f = {**f, **got, "sources": {**f["sources"], **got.get("sources", {})}}
  rows = derive(f, feats, engine, tab)
  measured: dict[str, Any] = {}
  if measure:
    out = out or pathlib.Path.home() / ".boltbeam" / "tune" / f"{engine}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    go = run or ad.launch_measure(model, feats, f, port=port, out=out, extra=extra)
    todo = [k for k in knobs(engine, tab) if k["derive"] == "measured" and (not knobs_only or k["name"] in knobs_only)
            and next(x for x in rows if x["name"] == k["name"])["status"] not in ("not_in_build", "not_applicable")]
    say = say or (lambda _line: None)
    say(f"pipeline steps: {len(todo)}")
    for k in todo:
      most = len(k["sweep"]["points"]) * (2 if k["sweep"]["type"] == "crossover" else 1)
      point = [0]
      def timed(*a, _go=go, _most=most, _k=k["name"], _point=point, **kw):
        _point[0] += 1
        say(f"stage {_k}: progress {min(_point[0], _most) - 1}/{_most}")
        return _go(*a, **kw)
      say(f"stage {k['name']}: start")
      try:
        measured[k["name"]] = sweep(k, timed, rows, prompt=prompt)
      except Exception as exc:  # noqa: BLE001  a failed sweep keeps the facts value and says why (in its row, not the stage)
        measured[k["name"]] = {"value": None, "status": "failed", "why": _words()["tune"]["sweep_failed"].format(out=short_path(str(out))),
                               "detail": f"{type(exc).__name__}: {str(exc)[-300:]}"}  # the raw text: provenance, not the reason
      say(f"stage {k['name']}: done")
    say("pipeline done")
    rows = derive(f, feats, engine, tab, measured)
    (out / "tune.json").write_text(json.dumps({"engine": engine, "rows": rows, "measured": measured, "facts": f}, indent=1, default=str))
  shown = ad.render(values_of(rows), f, feats, model, extra)
  return {"schema": "boltbeam.tune.v1", "engine": engine, "model": model, "build": {**{k: v for k, v in feats.items() if k != "features"},
          "groups": groups_in(engine, feats, tab), "feature_count": len(feats["features"])}, "facts": f, "rows": rows, **shown,
          "measured": measured, "out": str(out) if out else None}


# --- the words ---------------------------------------------------------------------------------------------------------

def short_path(path:str) -> str:
  """A build's path as a reader sees it: the home folder as ~, and a deep path as its first two folders, ... and the
  program (the full path stays in the command and in the record)."""
  p = pathlib.Path(path).expanduser()
  try:
    parts, head = p.relative_to(pathlib.Path.home()).parts, "~"
  except ValueError:
    parts, head = (p.parts[1:], "") if p.is_absolute() else (p.parts, ".")
  if len(parts) > 4:
    parts = (*parts[:2], "...", parts[-1])
  return "/".join((head, *parts)) if head else "/" + "/".join(parts)


def text(res:dict[str, Any], tab:dict[str, Any] | None = None) -> str:
  """The setup as plain lines (data/plain_words.json tune)."""
  from boltbeam.report.plain import words
  tab = tab or table()
  w, b, f, rows, eng = words()["tune"], res["build"], res["facts"], res["rows"], tab["engines"][res["engine"]]
  have = [k for k, v in b["groups"].items() if v]
  ver = f", {b['version']}" if b.get("version") else ""
  lines = [w["build"].format(engine=res["engine"], path=short_path(b["build"]), backend=(b.get("backend") or "?").upper(), version=ver,
                             fork=b.get("fork") or w["unlisted"])]
  if eng["groups"]:
    lines[-1] += " " + w["groups"].format(what=eng["groups_words"], groups=", ".join(have) if have else w["none"])
  lines += list(b.get("notes") or []) + list(f.get("notes") or [])
  lines.append(w["host"].format(total=f["ram_total_bytes"] / GiB, usable=f["usable_ram_bytes"] / GiB,
                                avail="unknown" if f["ram_available_bytes"] is None else f"{f['ram_available_bytes'] / GiB:.1f}",
                                l3="unknown" if f["l3_bytes"] is None else f"{f['l3_bytes'] / MiB:.0f} MiB", l3_source=f["l3_source"]))
  lines += ["", w["setup"]]
  for r in rows:
    if r["status"] in ("set", "default"):
      val = w["unset"] if r["value"] is None else r["value"]
      name = f"{r['name']}={val}" if r["kind"] in ("env", "config") else f"{r['name']} {val}"
      tail = "" if r["bitwise"] else " " + w["not_bitwise"]
      lines.append(f"  {name:<34} {w['source_' + r['source']]}: {r['reason']}{tail}")
  na = [r for r in rows if r["status"] == "not_applicable"]
  if na:
    lines += ["", w["not_applicable"]] + [f"  {r['name']}: {r['reason']}" for r in na]
  absent = [r for r in rows if r["status"] == "not_in_build"]
  gone = sorted({r["group"] for r in absent if r["group"] and not b["groups"].get(r["group"])})
  older = [r["name"] for r in absent if r["group"] and b["groups"].get(r["group"])]
  lacks = [r["name"] for r in absent if not r["group"]]
  if older:
    lines += ["", w["older_group"].format(noun=eng["group_noun"], knobs=", ".join(older))]
  if gone:
    lines += ["", w["not_in_build"].format(groups=", ".join(gone))]
  if lacks:
    lines += ["", w["lacks"].format(engine=res["engine"], knobs=", ".join(lacks))]
  lines += ["", w["command"], "  " + res["command"]]
  if not res["measured"] and any(r["provenance"].get("measure") for r in rows):
    lines.append(w["measure_hint"])
  return "\n".join(lines)


def page(res:dict[str, Any], setting_of:Callable[[dict[str, Any]], str | None] | None = None,
         tab:dict[str, Any] | None = None) -> dict[str, Any]:
  """The setup as a screen lays it out (data/plain_words.json tune.page): the build line, the notes, one row per knob set
  or left at its default (its value, source, reason and tags), the knobs not used, the command, what --measure would
  confirm, and what "use these" copies. setting_of(row) names the run setting that takes a flag or config row's value
  (None: a run does not pass it); an env row is passed as an environment variable. Engine-blind, as the rest."""
  from boltbeam.report.plain import words
  w, wp = words()["tune"], words()["tune"]["page"]
  lines = text(res, tab).split("\n")
  head = lines[0]
  notes = [line for line in lines[1:lines.index(w["setup"])] if line.strip()]
  rows, settings, env, not_passed = [], {}, {}, []
  uses: dict[str, dict[str, Any]] = {}
  for r in res["rows"]:
    if r["status"] == "set" and r["value"] is not None:
      if r["kind"] == "env":
        uses[r["name"]] = {"env": r["name"], "value": r["value"]}
      elif name := (setting_of(r) if setting_of else None):
        uses[r["name"]] = {"setting": name, "value": r["value"]}
      else:
        not_passed.append(r["name"])
  placed = [{**p, "use": setting_of({**p, "kind": "flag"}) if setting_of else None} for p in res.get("placement") or []]
  held = held_back(res, uses, placed, tab)  # a value whose formula assumes a row (or a placement) a run cannot pass
  for p in placed:
    if p["use"] and not held.get(None):
      settings[p["use"]] = p["value"]
  for r in res["rows"]:
    if r["status"] not in ("set", "default"):
      continue
    tags = [] if r["bitwise"] else [w["not_bitwise"].strip("[]")]
    use = None if r["name"] in held else uses.get(r["name"])
    if use and "env" in use:
      env[use["env"]] = use["value"]
    elif use:
      settings[use["setting"]] = use["value"]
    confirm = (r["provenance"] or {}).get("measure")
    rows.append({"name": r["name"], "kind": r["kind"], "status": r["status"], "value": w["unset"] if r["value"] is None else r["value"],
                 "source": r["source"], "source_words": w["source_" + r["source"]], "reason": r["reason"], "tags": tags,
                 "group": r["group"], "use": use, "confirm": wp["confirm"].format(how=confirm.removeprefix("confirm with --measure: ")) if confirm else None,
                 "held": wp["held"].format(knob=r["name"], needs=", ".join(held[r["name"]])) if r["name"] in held else None})
  measured = [k for k, v in (res.get("measured") or {}).items() if v.get("value") is not None]
  to_measure = [r["name"] for r in res["rows"] if (r["provenance"] or {}).get("measure")]
  tail = [line for line in lines[lines.index(w["setup"]) + 1:] if line.strip()]
  absent = [line for line in tail if not line.startswith("  ") and line not in (w["not_applicable"], w["command"], w["measure_hint"])]
  return {"title": wp["title"], "head": head, "notes": notes, "says": wp["measured"].format(knobs=", ".join(measured)) if measured else wp["facts"],
          "rows": rows, "not_used_title": wp["not_used"],
          "not_used": [{"name": r["name"], "reason": r["reason"]} for r in res["rows"] if r["status"] == "not_applicable"],
          "absent": absent, "command": res["command"],
          "measure": {"label": wp["measure"], "helper": wp["measure_helper"], "knobs": to_measure, "none": None if to_measure else wp["measure_none"],
                      "running": wp["running"], "stop": wp["stop"], "stop_helper": wp["stop_helper"], "stopped": wp["stopped"],
                      "failed": wp["failed"], "starting": wp["starting"], "progress": wp["progress"]},
          "use": {"label": wp["use"], "helper": wp["use_helper"], "settings": settings, "env": env,
                  "none": None if settings or env else wp["use_none"], "used": wp["used"],
                  "not_passed": " ".join(filter(None, [wp["not_passed"].format(knobs=", ".join(not_passed)) if not_passed else None,
                                                       *(wp["held"].format(knob=k, needs=", ".join(v)) for k, v in held.items() if k)])) or None,
                  "env_about": wp["env"]}}


def held_back(res:dict[str, Any], uses:dict[str, dict[str, Any]], placed:list[dict[str, Any]], tab:dict[str, Any] | None = None) -> dict[str | None, list[str]]:
  """The rows "use these" must not copy, each with what its value assumes that a run cannot pass: a row whose formula (or
  `applies`) reads a fact another row's `sets` writes, when that row is set and not copied (through the chain), or a fact
  of the experts' placement (the adapter's PLACEMENT_FACTS) when the placement cannot be passed. Key None: the placement
  itself is held (a part of it cannot be passed, so none of it is copied)."""
  tab = tab or table()
  ks = {k["name"]: k for k in knobs(res["engine"], tab)}
  writes = {fact: k["name"] for k in ks.values() for fact in (k.get("sets") or {})}
  status = {r["name"]: r for r in res["rows"]}
  out: dict[str | None, list[str]] = {}
  stuck = [p for p in placed if not p["use"]]
  if stuck:
    out[None] = [f"{p['name']} {p['value']}" for p in stuck]
  where = set(getattr(adapter(res["engine"], tab), "PLACEMENT_FACTS", ())) if placed else set()

  def needs(name:str, seen:set[str]) -> list[str]:
    k = ks.get(name) or {}
    reads = names_in(k["formula"]) if k.get("formula") else []
    reads += names_in(k.get("applies", "True"))
    got = [" ".join(f"{p['name']} {p['value']}" for p in placed)] if stuck and where & set(reads) else []
    for fact in reads:
      dep = writes.get(fact)
      if not dep or dep == name or dep in seen:
        continue
      d = status.get(dep) or {}
      if d.get("status") == "set" and dep not in uses:
        got.append(f"{dep} {d['value']}")
      got += needs(dep, seen | {name})
    return list(dict.fromkeys(got))
  for name in uses:
    if got := needs(name, set()):
      out[name] = got
  return out
