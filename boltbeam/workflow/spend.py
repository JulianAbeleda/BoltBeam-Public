"""The spend fit: how to spend the GPU's memory at a target context, ranked by projected session time from measured rows.

Each way to trade memory for speed or context is a row of the engine's table (data/engine_knobs.json engines.<E>.spend):
how a command sets it (read), the memory it takes at a value (bytes, a formula over the facts), the session metrics it
can move (moves), and whether the build has its switch (requires). A combination takes one value per row. Engine-blind:
the rows, the flag names and the formulas are data; the adapter only reads files the rows name (spend_value).

  anchor     the memory a combination needs is measured, not estimated: a session-ab arm of the same model that ran to
             the session's last turn gives its lowest free memory over the session (telemetry); a combination's free =
             that, less what its rows take beyond the arm's (bytes at each value), less the cells beyond the arm's
             context (cell_bytes); the sink row's leftover goes to its sink (the expert cache). The nearest arm (fewest
             rows apart) anchors. margin (margin_of): the larger of the spread of the lowest free over runs of one setup
             and the formulas' worst error predicting one measured arm from another; with neither it is unknown and
             nothing is ranked.
  units      a measured effect: an arm against its base (session-ab: step ms = decode ms / (tokens - accepted), tokens a
             step, prefill ms a turn, pooled over valid runs) or a measured row file (boltbeam.spend_measured.v1). Each
             metric's effect is keyed by the rows that move it whose value differs; a unit applies only to the same model,
             context and session, over turns that cover the projection's.
  projection the base arm's session (prefill per turn, decode tokens, step ms, tokens a step) with the metric effects
             of a combination: a disjoint cover of its changed rows by units, per metric (deltas add, ratios multiply).
             No cover: the combination is unmeasured, shown with its memory, never ranked.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from typing import Any

from boltbeam.collectors.replay import flag_value

MiB = 1 << 20
METRICS = ("step_ms", "prefill_ms", "tokens_per_step")  # tokens_per_step is a ratio; the others add
SCHEMAS = ("boltbeam.session_ab.v1", "boltbeam.spend_measured.v1")
STORES = (pathlib.Path(__file__).resolve().parents[1] / "data" / "spend", pathlib.Path.home() / ".boltbeam" / "spend")  # shipped, then the user's


def rows_of(engine:str, tab:dict[str, Any]) -> dict[str, Any]:
  return (tab["engines"].get(engine) or {}).get("spend") or {}


def read_records(paths:list[str]) -> list[dict[str, Any]]:
  """Every record under the paths (a file, or a folder searched for *.json) whose schema is one the fit reads."""
  out = []
  for p in (pathlib.Path(x).expanduser() for x in paths):
    for f in sorted(p.rglob("*.json")) if p.is_dir() else [p]:
      try:
        d = json.loads(f.read_text())
      except (OSError, ValueError):
        continue
      if isinstance(d, dict) and d.get("schema") in SCHEMAS:
        out.append({**d, "_path": str(f)})
  return out


def model_of(d:dict[str, Any]) -> str | None:
  """The model file a record measured (its name: records travel without their machine's folders)."""
  if d.get("schema") == "boltbeam.spend_measured.v1":
    return pathlib.Path(d["scope"]["model"]).name
  m = flag_value([str(x) for x in d["inputs"]["server_argv"]], ("-m", "--model"))
  return pathlib.Path(m).name if m else None


def defaults(model:str, gpu_total, features:set[str]) -> list[str]:
  """The measured records of this model file, from the stores (STORES: BoltBeam's own measured rows, then
  ~/.boltbeam/spend): what tune ranks on when no --measured is given. Only a session-ab record of a GPU of this size
  (its lowest free memory is that GPU's; an unread size takes none) whose base switches this build has (its times are
  that build's). gpu_total: its bytes, read (a call) only when a record names the model."""
  out = []
  mine = [d for d in read_records([str(x) for x in STORES if x.is_dir()]) if model_of(d) == pathlib.Path(model).name]
  gpu_total = (gpu_total() if callable(gpu_total) else gpu_total) if mine else None
  for d in mine if gpu_total else []:
    tot = {round(((r.get("telemetry") or {}).get("memory_mib") or {}).get("total") or 0) for r in d.get("arms") or []}
    env = {k for k in (d.get("inputs") or {}).get("server_env") or {} if k.isupper()}
    if tot == {round(gpu_total / MiB)} and env <= features:
      out.append(d["_path"])
  return out


def where_of(path:str) -> str:
  """A record by its run folder's name; a stored one (STORES) by its own name."""
  p = pathlib.Path(path)
  return p.stem if p.parent in STORES else p.parent.name


# --- reading a command into row values --------------------------------------------------------------------------------

def value_of(row:dict[str, Any], argv:list[str], env:dict[str, str], ad, known:dict[str, Any] | None = None) -> Any:
  """A row's value in a command: its flag's (an int), its env switch (on/off), or what the adapter reads from the file a
  flag names (known: the record's own reading of it, by file name, for a record measured elsewhere); the row's default
  where the command does not set it."""
  r = row.get("read") or {}
  if "flags" in r:
    v = flag_value(argv, tuple(r["flags"]))
    return int(v) if v is not None and v.lstrip("-").isdigit() else r.get("default")
  if "env" in r:
    return "on" if env.get(r["env"]) == r.get("on", "1") else "off"
  if "file_flags" in r:
    f = flag_value(argv, tuple(r["file_flags"]))
    got = ((known or {}).get(pathlib.Path(f).name) or {}).get(r["fact"]) if f else None
    return got if got is not None else ad.spend_value(r["fact"], f) if f else r.get("default")
  return r.get("default")


def config_of(spend:dict[str, Any], argv:list[str], env:dict[str, str], ad, f:dict[str, Any], allowed:set[str], known:dict[str, Any] | None = None) -> dict[str, Any]:
  """A command's row values, its sink's bytes (_sink), the files its rows read (_files) and its argv_facts (a flag's
  value, else the default formula over the facts)."""
  from boltbeam.workflow import tune
  out: dict[str, Any] = {row["name"]: value_of(row, argv, env, ad, known) for row in spend["rows"]}
  out["_files"] = {row["name"]: flag_value(argv, tuple(row["read"]["file_flags"])) for row in spend["rows"] if "file_flags" in (row.get("read") or {})}
  sink = spend.get("sink")
  if sink:
    v = flag_value(argv, tuple(sink["flags"]))
    out["_sink"] = int(v) * sink["unit_bytes"] if v and v.isdigit() else 0
  for name, rule in (spend.get("argv_facts") or {}).items():
    v = flag_value(argv, tuple(rule["flags"]))
    v = v if v is not None else tune.evaluate(rule["default"], f, allowed)
    out[name] = int(v) if isinstance(v, str) and v.isdigit() else v
  return out


def available(row:dict[str, Any], features:set[str]) -> bool:
  """A row the fit may move here: its switch exists (built) and the build has it (requires)."""
  return row.get("built", True) and (row.get("requires") is None or row["requires"] in features)


# --- the measured inputs: session-ab arms and measured rows ------------------------------------------------------------

def _arm_metrics(arm_runs:list[dict[str, Any]]) -> dict[str, Any]:
  turns = [t for a in arm_runs for t in a["turns"] if t.get("predicted_n") is not None and t.get("prompt_ms") is not None]
  n = sum(t["predicted_n"] for t in turns)
  steps = sum(t["predicted_n"] - (t.get("draft_accepted") or 0) for t in turns)
  if not turns or steps <= 0:
    return {}
  return {"step_ms": sum(t["predicted_ms"] for t in turns) / steps, "tokens_per_step": n / steps,
          "prefill_ms": sum(t["prompt_ms"] for t in turns) / len(turns), "turns": len(turns) // len(arm_runs),
          "tokens": n / len(arm_runs), "prefill_total_ms": sum(t["prompt_ms"] for t in turns) / len(arm_runs)}


def sessions(records:list[dict[str, Any]], spend:dict[str, Any], ad, model:str, f:dict[str, Any], allowed:set[str]) -> list[dict[str, Any]]:
  """Each session-ab record of this model as {scope, base, arms: {name: config, metrics, free_min, runs}}."""
  out = []
  for d in records:
    if d.get("schema") != "boltbeam.session_ab.v1":
      continue
    inp = d["inputs"]
    argv, env = [str(x) for x in inp["server_argv"]], dict(inp.get("server_env") or {})
    if model_of(d) != pathlib.Path(model).name:
      continue
    ctx = flag_value(argv, ("-c", "--ctx-size"))
    session = inp.get("session") or {}
    scope = {"context": int(ctx) if ctx else None, "session": session.get("sha256"), "turns": tuple(inp.get("turns") or (1, session.get("turns"))),
             "last": session.get("turns"), "path": d["_path"]}
    refused = {c["arm"] for c in (d.get("analysis") or {}).get("comparisons") or [] if c.get("refused")}
    arms = {}
    for a in inp["arms"]:
      runs = [r for r in d["arms"] if r["name"] == a["name"] and r.get("valid")]
      if not runs or a["name"] in refused:
        continue
      cfg = config_of(spend, argv + [str(x) for x in a.get("flags") or []], {**env, **(a.get("env") or {})}, ad, f, allowed, inp.get("file_facts"))
      used = [((r.get("telemetry") or {}).get("memory_mib") or {}) for r in runs]
      free = [(u["total"] - u["used_max"]) * MiB for u in used if u.get("total") and u.get("used_max")]
      arms[a["name"]] = {"config": cfg, "metrics": _arm_metrics(runs), "free_min": min(free) if free else None,
                         "free_spread": max(free) - min(free) if len(free) > 1 else None, "runs": len(runs)}
    base = (d.get("analysis") or {}).get("base") or (inp.get("order") or [None])[0]
    if base in arms:
      out.append({"scope": scope, "base": base, "arms": arms})
  return out


def units(spend:dict[str, Any], sess:list[dict[str, Any]], records:list[dict[str, Any]], model:str) -> list[dict[str, Any]]:
  """The measured effects: each session-ab arm against its base, and each measured row of a matching scope, per metric
  keyed by the rows that move it and differ ({rows: {row: value}, base: {row: value}, metric, value, sink, where})."""
  moves = {r["name"]: set(r.get("moves") or []) for r in spend["rows"]}
  out = []
  for s in sess:
    b = s["arms"][s["base"]]
    for name, a in s["arms"].items():
      if name == s["base"] or not a["metrics"] or not b["metrics"]:
        continue
      for m in METRICS:
        diff = {r: a["config"][r] for r in moves if m in moves[r] and a["config"][r] != b["config"][r]}
        if not diff:
          continue
        val = a["metrics"][m] / b["metrics"][m] if m == "tokens_per_step" else a["metrics"][m] - b["metrics"][m]
        out.append({"rows": diff, "base": {r: b["config"][r] for r in diff}, "metric": m, "value": val, "sink": a["config"].get("_sink"),
                    "scope": s["scope"], "where": f"{where_of(s['scope']['path'])} {name} vs {s['base']}", "files": a["config"]["_files"],
                    "runs": min(a["runs"], b["runs"]), "at": {}})
  for d in records:
    if d.get("schema") != "boltbeam.spend_measured.v1" or pathlib.Path(d["scope"]["model"]).name != pathlib.Path(model).name:
      continue
    sc = {"context": d["scope"]["context"], "session": d["scope"]["session_sha256"], "turns": tuple(d["scope"]["turns"]), "path": d["_path"]}
    for row in d["rows"]:
      for m, (base_v, arm_v) in row["metrics"].items():
        diff = {r: v for r, v in row["arm"].items() if m in moves.get(r, set())}
        if diff:
          out.append({"rows": diff, "base": {r: row["base"][r] for r in diff}, "metric": m, "value": arm_v / base_v if m == "tokens_per_step" else arm_v - base_v,
                      "sink": None, "scope": sc, "where": row.get("evidence") or d["_path"], "files": row.get("files") or {}, "runs": row.get("runs", 1),
                      "at": {r: v for r, v in (row.get("at") or {}).items() if m in moves.get(r, set())}})
  return out


# --- memory -------------------------------------------------------------------------------------------------------------

def _facts(f:dict[str, Any], cfg:dict[str, Any], context:int) -> dict[str, Any]:
  """The facts with the combination's values: spend.<row> for each row, its argv_facts, the context."""
  return {**f, **{f"spend.{k}": v for k, v in cfg.items() if not k.startswith("_") and "." not in k},
          **{k: v for k, v in cfg.items() if "." in k}, "context_tokens": context}


def taken(spend:dict[str, Any], cfg:dict[str, Any], f:dict[str, Any], context:int, allowed:set[str], at:tuple) -> dict[str, float]:
  """What each row takes at the combination's values, and the context's cells (cell_bytes x cells), in bytes: the facts
  with the combination's values and the spend sets, and the engine's after-load terms over them (at: engine, table)."""
  from boltbeam.workflow import tune
  g = _facts(f, cfg, context)
  ok = allowed | set(g)
  g |= {k: tune.evaluate(e, g, ok) for k, e in (spend.get("sets") or {}).items()}
  g |= tune.after_load(g, *at)
  ok |= set(g)
  out = {r["name"]: float(tune.evaluate(r["bytes"], {**g, "value": cfg[r["name"]]}, ok)) for r in spend["rows"]}
  out["cells"] = float(tune.evaluate(spend["cell_bytes"], g, ok)) * context
  return out


def free_at(spend:dict[str, Any], cfg:dict[str, Any], anchor:dict[str, Any], f:dict[str, Any], context:int, allowed:set[str], at:tuple) -> dict[str, Any]:
  """A combination's free memory from its anchor arm: the arm's lowest free, less what the combination's rows and cells
  take beyond the arm's. The sink (expert cache) is the arm's, then: none where the sink row says none, else the arm's
  plus everything left (the memory note: spare MiB to the cache)."""
  from boltbeam.workflow import tune
  a, x = taken(spend, anchor["config"], f, anchor["context"], allowed, at), taken(spend, cfg, f, context, allowed, at)
  free = anchor["free_min"] - sum(x[k] - a[k] for k in x)
  sink = spend.get("sink")
  has = bool(sink) and bool(tune.evaluate(sink["when"], {"value": cfg[sink["row"]]}, {"value"}))
  free += anchor["config"].get("_sink") or 0  # the arm's cache, given back
  return {"free": free, "sink": max(free - anchor["margin"], 0) if has else 0, "has_sink": has, "by_row": {k: x[k] - a[k] for k in x}}


# --- the fit --------------------------------------------------------------------------------------------------------------

def candidates(spend:dict[str, Any], base:dict[str, Any], measured:list[dict[str, Any]], f:dict[str, Any], features:set[str]) -> dict[str, list]:
  """Each row's values: the base arm's, and where the row is available here every measured one, its own values and its
  range (a count fact: 0..N); values_from adds another row's (a batch is at least a chunk)."""
  out = {}
  for r in spend["rows"]:
    vals = [base[r["name"]]]
    if available(r, features):
      vals += [u["rows"][r["name"]] for u in measured if r["name"] in u["rows"]] + list(r.get("values") or [])
      vals += list(range(int(f.get(r["range"]) or 0) + 1)) if r.get("range") else []
    out[r["name"]] = list(dict.fromkeys(v for v in vals if v is not None))
  for r in spend["rows"]:
    if r.get("values_from") and available(r, features):
      out[r["name"]] = list(dict.fromkeys(out[r["name"]] + out[r["values_from"].removeprefix("spend.")]))
  return out


def _cover(changed:dict[str, Any], base:dict[str, Any], cands:list[dict[str, Any]], sink:int | None, sink_row:str | None) -> list[dict[str, Any]] | None:
  """A disjoint set of units whose rows are exactly the changed rows at the combination's values, measured against the
  projection's base values; the sink row's unit only where the combination's sink holds at least the unit's."""
  if not changed:
    return []
  ok = [u for u in cands if all(changed.get(r) == v for r, v in u["rows"].items()) and all(base.get(r) == v for r, v in u["base"].items())
        and all(r in u["rows"] or base.get(r) == v for r, v in (u.get("at") or {}).items())
        and (sink_row not in u["rows"] or not u["sink"] or (sink is not None and sink >= u["sink"]))]
  for n in range(1, len(ok) + 1):
    for pick in itertools.combinations(ok, n):
      keys = [r for u in pick for r in u["rows"]]
      if len(keys) == len(set(keys)) and set(keys) == set(changed):
        return list(pick)
  return None


def project(spend:dict[str, Any], cfg:dict[str, Any], fit:dict[str, Any], base:dict[str, Any], us:list[dict[str, Any]], sink:int | None) -> dict[str, Any]:
  """The combination's session time from the base session and the measured effects; {unmeasured: rows} without a cover."""
  moves = {r["name"]: set(r.get("moves") or []) for r in spend["rows"]}
  srow = (spend.get("sink") or {}).get("row")
  eff, used, missing = {}, [], []
  for m in METRICS:
    changed = {r: cfg[r] for r in moves if m in moves[r] and cfg[r] != base["config"][r]}
    if srow and m in moves.get(srow, set()) and cfg[srow] == base["config"][srow] and fit["has_sink"] and sink < (base["config"].get("_sink") or 0):
      changed[srow] = cfg[srow]  # the same layers in RAM behind a smaller cache: not the base's measurement
    got = _cover(changed, base["config"], [u for u in us if u["metric"] == m], sink, srow)
    if got is None:
      missing += [r for r in changed if r not in missing]
      continue
    vals = [u["value"] for u in got]
    eff[m] = (1.0 if m == "tokens_per_step" else 0.0)
    for v in vals:
      eff[m] = eff[m] * v if m == "tokens_per_step" else eff[m] + v
    used += [u["where"] for u in got if u["where"] not in used]
  if missing:
    return {"unmeasured": missing}
  bm = base["metrics"]
  step, tps = bm["step_ms"] + eff["step_ms"], bm["tokens_per_step"] * eff["tokens_per_step"]
  decode_s, prefill_s = bm["tokens"] / tps * step / 1000, (bm["prefill_total_ms"] + bm["turns"] * eff["prefill_ms"]) / 1000
  return {"seconds": decode_s + prefill_s, "decode_s": decode_s, "prefill_s": prefill_s, "step_ms": step, "tokens_per_step": tps,
          "prefill_ms": bm["prefill_ms"] + eff["prefill_ms"], "from": used, "runs": min([base["runs"]] + [u.get("runs", 1) for u in us if u["where"] in used])}


def plan(f:dict[str, Any], engine:str, tab:dict[str, Any], *, model:str, records:list[dict[str, Any]], context:int | None,
         features:set[str], ad, allowed:set[str]) -> dict[str, Any]:
  """The spend fit at a context (None: the most cells each combination holds): every combination costed from its nearest
  anchor, the measured ones ranked by projected session time; the rows not available here with the memory they would
  free. {why} when there is nothing to anchor on."""
  spend, at_ = rows_of(engine, tab), (engine, tab)
  if not spend:
    return {"why": "no_rows"}
  sess = sessions(records, spend, ad, model, f, allowed)
  anchors = [{**a, "context": s["scope"]["context"], "name": n, "where": s["scope"]["path"]} for s in sess for n, a in s["arms"].items()
             if a["free_min"] is not None and s["scope"]["turns"][-1] == s["scope"]["last"] and s["scope"]["context"]]
  if not anchors:
    return {"why": "no_anchor"}
  try:
    margin = margin_of(spend, anchors, f, allowed, at_)
  except Exception as exc:  # noqa: BLE001  a fact the rows read is missing (no draft head given): said, never a crash
    return {"why": "no_facts", "what": f"{type(exc).__name__}: {exc}"}
  for a in anchors:
    a["margin"] = margin["bytes"] if margin["bytes"] is not None else 0
  timed = [s for s in sess if s["arms"][s["base"]]["metrics"]]
  if not timed:
    return {"why": "no_anchor"}
  ref = max(timed, key=lambda s: (s["scope"]["turns"][-1] == s["scope"]["last"], s["scope"]["turns"][-1] - s["scope"]["turns"][0]))
  base = {**ref["arms"][ref["base"]], "scope": ref["scope"]}
  us = [u for u in units(spend, sess, records, model) if u["scope"]["context"] == base["scope"]["context"]  # the same context, session and turns
        and u["scope"]["session"] == base["scope"]["session"] and tuple(u["scope"]["turns"]) == tuple(base["scope"]["turns"])]
  cands = candidates(spend, base["config"], us, f, features)
  names = list(cands)
  cap = int(f.get(spend["context_cap"]) or 0) if spend.get("context_cap") else 0
  out = []
  for vals in itertools.product(*(cands[n] for n in names)):
    cfg = {**base["config"], **dict(zip(names, vals))}
    if any(not _holds(r, cfg, allowed) for r in spend["rows"]):
      continue
    anchor = min(anchors, key=lambda a: (sum(a["config"][n] != cfg[n] for n in names), -a["runs"]))
    try:
      cells = context or _max_cells(spend, cfg, anchor, f, allowed, cap, at_)
      fit = free_at(spend, cfg, anchor, f, cells, allowed, at_)
    except Exception as exc:  # noqa: BLE001  a row whose memory cannot be read is never chosen, and said
      out.append({"values": dict(zip(names, vals)), "fits": False, "why": f"{type(exc).__name__}: {exc}"})
      continue
    row = {"values": dict(zip(names, vals)), "context": cells, "free": fit["free"] - fit["sink"], "sink": fit["sink"], "fits": fit["free"] >= anchor["margin"] and cells > 0,
           "anchor": anchor["name"], "by_row": fit["by_row"], "layers": cfg.get(spend["sink"]["row"]) if spend.get("sink") else None,
           "files": {r: next((u["files"][r] for u in us if u["rows"].get(r) == cfg[r] and u["files"].get(r)), None) for r in cfg.get("_files") or {}}}
    if row["fits"] and margin["bytes"] is None:
      row["projection"] = {"unmeasured": ["margin"]}  # no measured spread or residual: whether it fits is not known well enough to rank
    elif row["fits"]:
      row["projection"] = project(spend, cfg, fit, base, us, fit["sink"] if fit["has_sink"] else None) if cells == base["scope"]["context"] \
        else {"unmeasured": ["context"]}
    out.append(row)
  ranked = sorted((r for r in out if r["fits"] and "seconds" in r["projection"]), key=lambda r: r["projection"]["seconds"])
  unranked = sorted((r for r in out if r["fits"] and "seconds" not in r["projection"]), key=lambda r: (-r["context"], r["layers"] or 0, -r["free"] - r["sink"]))
  return {"context": context, "measured_context": base["scope"]["context"], "margin": margin,
          "base": {"name": ref["base"], "config": {n: base["config"][n] for n in names}, "metrics": base["metrics"], "where": ref["scope"]["path"], "sink": base["config"].get("_sink") or 0,
                   "turns": list(ref["scope"]["turns"])},
          "ranked": ranked, "unmeasured_fits": unranked, "combinations": len(out), "errors": [r for r in out if r.get("why")][:3],
          "pending": pending(spend, features, f, base["config"], context or base["scope"]["context"], allowed, at_),
          "units": [{k: u[k] for k in ("rows", "metric", "value", "where")} for u in us]}


def margin_of(spend:dict[str, Any], anchors:list[dict[str, Any]], f:dict[str, Any], allowed:set[str], at_:tuple) -> dict[str, Any]:
  """What a combination must keep free beyond its predicted lowest: the larger of the spread of the lowest free over runs
  of one setup (repeats, and arms of equal setups) and the worst error of the memory formulas when one measured arm is
  predicted from another (leave one out). Unknown (bytes None) with neither: one reading of one setup."""
  key = lambda a: tuple(sorted((k, str(v)) for k, v in a["config"].items() if not k.startswith("_")) + [("ctx", str(a["context"]))])  # noqa: E731
  groups: dict[tuple, list[float]] = {}
  for a in anchors:
    groups.setdefault(key(a), []).append(a["free_min"])
  spread = [max(g) - min(g) for g in groups.values() if len(g) > 1] + [a["free_spread"] for a in anchors if a["free_spread"] is not None]
  resid = []
  for a, b in itertools.permutations(anchors, 2):
    if key(a) != key(b):
      got = free_at(spend, b["config"], {**a, "margin": 0}, f, b["context"], allowed, at_)["free"]
      resid.append(abs(got - (b["free_min"] + (b["config"].get("_sink") or 0))))
  both = spread + resid
  return {"bytes": max(both) if both else None, "spread": max(spread) if spread else None, "residual": max(resid) if resid else None}


def _holds(row:dict[str, Any], cfg:dict[str, Any], allowed:set[str]) -> bool:
  from boltbeam.workflow import tune
  g = _facts({}, cfg, 0)
  return not row.get("valid") or bool(tune.evaluate(row["valid"], {**g, "value": cfg[row["name"]]}, allowed | set(g)))


def _max_cells(spend:dict[str, Any], cfg:dict[str, Any], anchor:dict[str, Any], f:dict[str, Any], allowed:set[str], cap:int, at_:tuple) -> int:
  """The most cells (a multiple of the engine's cell step, at most the cap) the combination holds: its free memory is
  affine in the cells, so two points (the anchor's context and twice it) give it."""
  c0 = anchor["context"]
  lo, hi = (free_at(spend, cfg, anchor, f, c, allowed, at_)["free"] - anchor["margin"] for c in (c0, 2 * c0))
  per = (lo - hi) / c0
  cells = int(c0 + lo / per) // spend["cells_step"] * spend["cells_step"] if per > 0 else 0
  return max(0, min(cells, cap) if cap else cells)


def pending(spend:dict[str, Any], features:set[str], f:dict[str, Any], base:dict[str, Any], context:int, allowed:set[str], at_:tuple) -> list[dict[str, Any]]:
  """The rows the fit cannot choose here (no switch built yet, or not in this build), each with what it would free at the
  base setup and the context. Never ranked."""
  out = []
  for r in spend["rows"]:
    if available(r, features) or "on" not in r:
      continue
    try:
      on, off = (taken(spend, {**base, r["name"]: v}, f, context, allowed, at_)[r["name"]] for v in (r["on"], base[r["name"]]))
      frees = off - on
    except Exception:  # noqa: BLE001  unread: said as unknown
      frees = None
    out.append({"row": r["name"], "words": r.get("words", r["name"]), "frees": frees, "built": r.get("built", True), "requires": r.get("requires"),
                "moves": r.get("moves") or [], "evidence": r.get("evidence", "")})
  return out


# --- the words ----------------------------------------------------------------------------------------------------------

def label(values:dict[str, Any], spend:dict[str, Any], files:dict[str, Any] | None = None) -> str:
  """A combination as the reader reads it: each row by name and value; a file row names its file."""
  out = []
  for r in spend["rows"]:
    v = values.get(r["name"])
    if v in (None, "off"):
      continue
    f, env = (files or {}).get(r["name"]), (r.get("read") or {}).get("env")
    if env and str(v) == str(r["read"].get("on")):  # a switch is said as the command sets it: NAME=on
      out.append(f"{r['name']} ({env}={v})")
      continue
    out.append(f"{r['name']} {v}" + (f" ({pathlib.Path(f).name})" if f else ""))
  return ", ".join(out)


def words(p:dict[str, Any], spend:dict[str, Any]) -> list[str]:
  """The spend fit in plain lines (data/plain_words.json spend)."""
  from boltbeam.report.plain import words as all_words
  w = all_words()["spend"]
  if "why" in p:
    return [w[p["why"]].format(what=p.get("what", ""))]
  b, ranked = p["base"], p["ranked"]
  ctx = w["context"].format(cells=p["context"]) if p["context"] else w["max_context"]
  out = [w["head" if p["context"] else "head_max"].format(context=ctx, n=p["combinations"], rows=", ".join(r["name"] for r in spend["rows"]),
                                                        measured=f"{p['measured_context']:,}", where=where_of(b["where"]),
                                                        turns="-".join(map(str, b["turns"]))),
         w["margin_unknown"] if p["margin"]["bytes"] is None else
         w["margin"].format(mib=p["margin"]["bytes"] / MiB, spread="?" if p["margin"]["spread"] is None else f"{p['margin']['spread'] / MiB:.0f}",
                            resid="?" if p["margin"]["residual"] is None else f"{p['margin']['residual'] / MiB:.0f}")]
  bm = b["metrics"]
  dec = bm["tokens"] / bm["tokens_per_step"] * bm["step_ms"] / 1000
  base_s = bm["prefill_total_ms"] / 1000 + dec
  out.append(w["base"].format(name=b["name"], config=label(b["config"], spend) + (w["sink"].format(mib=b["sink"] / MiB) if b["sink"] else ""), s=base_s, p=bm["prefill_total_ms"] / 1000, d=dec,
                              step=bm["step_ms"], tps=bm["tokens_per_step"], pre=bm["prefill_ms"]))
  sink = lambda r: w["sink"].format(mib=r["sink"] / MiB) if r.get("sink") else ""  # noqa: E731
  fmt = lambda m, v: f"{v:.3f}" if m == "tokens_per_step" else f"{v:.2f}"  # noqa: E731
  if ranked:
    best, pr = ranked[0], ranked[0]["projection"]
    same = all(best["values"][n] == b["config"][n] for n in b["config"])  # the measured setup itself is the fastest measured
    out.append(w["best_is_base" if same else "best" if pr["runs"] > 1 else "candidate"].format(config=label(best["values"], spend, best["files"]), s=pr["seconds"], ds=pr["seconds"] - base_s, pct=100 * (pr["seconds"] / base_s - 1),
                                step=pr["step_ms"], tps=pr["tokens_per_step"], pre=pr["prefill_ms"], free=best["free"] / MiB, sink=sink(best)))
    out.append(w["from"].format(units="; ".join(pr["from"]) or "the base run itself"))
    out += [w["derive"].format(file=pathlib.Path(best["files"][r["name"]]).name, how=r["derive"]) for r in spend["rows"]
            if r.get("derive") and (best["files"] or {}).get(r["name"]) and best["values"][r["name"]] != b["config"][r["name"]]]
    if len(ranked) > 1:
      r2, q = ranked[1], ranked[1]["projection"]
      out.append(w["runner"].format(config=label(r2["values"], spend, r2["files"]), s=q["seconds"], ds=q["seconds"] - pr["seconds"], step=q["step_ms"],
                                    tps=q["tokens_per_step"], pre=q["prefill_ms"], free=r2["free"] / MiB, sink=sink(r2)))
      why = [w["why_metric"].format(what=w[k], base=fmt(m, q[m]), best=fmt(m, pr[m]))
             for m, k in (("step_ms", "metric_step"), ("tokens_per_step", "metric_tps"), ("prefill_ms", "metric_prefill")) if fmt(m, q[m]) != fmt(m, pr[m])]
      out.append(w["why"].format(why="; ".join(why), s=q["seconds"] - pr["seconds"]))
  else:
    out.append(w["none_ranked"].format(context=ctx))
  if un := p["unmeasured_fits"]:
    items = [w["unranked_item"].format(config=label(r["values"], spend), free=r["free"] / MiB, sink=sink(r),
                                       ctx="" if p["context"] else w["ctx"].format(cells=r["context"])) for r in un[:3]]
    out.append(w["unranked"].format(n=len(un), lead=w["unranked_lead"] if p["context"] else "", items="; ".join(items)))
    if not p["context"]:  # the most context: also what the fewest expert layers in RAM hold
      few = max((r for r in un + ranked if r["layers"] == min(x["layers"] or 0 for x in un + ranked)), key=lambda r: r["context"])
      out.append(w["fewest_layers"].format(layers=few["layers"], cells=few["context"], config=label(few["values"], spend)))
    out.append(w["measure_hint"].format(context=f"{p['context'] or un[0]['context']:,}"))
  if p["pending"]:
    out.append(w["pending_head"])
    for r in p["pending"]:
      why = w["pending_unbuilt"] if not r["built"] else w["pending_built"].format(switch=r["requires"])
      cost = w["pending_cost"] if any(r["row"] in u["rows"] for u in p["units"]) else w["pending_nocost"]
      out.append(w["pending"].format(words=r["words"], mib="?" if r["frees"] is None else f"{r['frees'] / MiB:,.0f}", why=f"{why}; {cost}"))
  out.append(w["command_apart"])
  return out
