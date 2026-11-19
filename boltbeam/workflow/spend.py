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
  fit_value  a row valued per machine (fit_value {min, step}): each combination takes the largest value that fits its free
             memory (bytes affine in the value: two points), projected from the largest measured value at or below it
             (a larger value is no slower); only_where_default_fails: a value other than the row's read default is kept
             only where the same combination at the default does not fit.
"""
from __future__ import annotations

import functools
import itertools
import json
import pathlib
import re
import shlex
from typing import Any

from boltbeam.collectors.replay import flag_value
from boltbeam.vocab import SCHEMA_SPEND_PROBE

MiB = 1 << 20
DERIVE = "_derive"  # a fit_value row's candidate: the largest value the combination fits (_derive)
METRICS = ("step_ms", "prefill_ms", "tokens_per_step")  # tokens_per_step is a ratio; the others add
PROBE = SCHEMA_SPEND_PROBE  # a load probe: a command, whether it loaded and ran its session, its lowest free memory
SCHEMAS = ("boltbeam.session_ab.v1", "boltbeam.spend_measured.v1", PROBE)
STORES = (pathlib.Path(__file__).resolve().parents[1] / "data" / "spend", pathlib.Path.home() / ".boltbeam" / "spend")  # shipped, then the user's


def rows_of(engine:str, tab:dict[str, Any]) -> dict[str, Any]:
  return (tab["engines"].get(engine) or {}).get("spend") or {}


def probe_of(result:pathlib.Path) -> dict[str, Any] | None:
  """A load probe's folder as a spend_probe record: cmd.txt the command as run (`env K=V ... SERVER ARGS`), result.json
  its reading (status ready or not, session ok or not, the GPU's free MiB at ready and its lowest, min_free_mib). None
  when the folder is not one."""
  cmd = result.with_name("cmd.txt")
  try:
    r, words = json.loads(result.read_text()), shlex.split(cmd.read_text())
  except (OSError, ValueError):
    return None
  if not isinstance(r, dict) or "min_free_mib" not in r or "status" not in r:
    return None
  words = words[1:] if words[:1] == ["env"] else words
  env = dict(w.split("=", 1) for w in itertools.takewhile(lambda w: re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", w), words))
  from boltbeam.collectors.load_probe import out_of_memory
  log = result.with_name("srv.log")  # a failed load's cause, by the engine's own row: its status line alone is no fit verdict
  oom = out_of_memory(None, log.read_text(errors="replace")) if r["status"] != "ready" and log.is_file() else {}
  return {"schema": PROBE, "argv": words[len(env):], "env": env, "loaded": r["status"] == "ready", "session": r.get("session") == "ok",
          "free_min_mib": r["min_free_mib"], "free_ready_mib": r.get("free_at_ready_mib"), "name": r.get("name") or result.parent.name,
          **({"failure": oom} if oom else {})}


def read_records(paths:list[str]) -> list[dict[str, Any]]:
  """Every record under the paths (a file, or a folder searched for *.json) whose schema is one the fit reads, and every
  load probe folder (probe_of)."""
  out = []
  for p in (pathlib.Path(x).expanduser() for x in paths):
    for f in sorted(p.rglob("*.json")) if p.is_dir() else [p]:
      try:
        d = json.loads(f.read_text())
      except (OSError, ValueError):
        continue
      if isinstance(d, dict) and d.get("schema") in SCHEMAS:
        out.append({**d, "_path": str(f)})
      elif f.name == "result.json" and (got := probe_of(f)):
        out.append({**got, "_path": str(f)})
  return out


def model_of(d:dict[str, Any]) -> str | None:
  """The model file a record measured (its name: records travel without their machine's folders)."""
  if d.get("schema") == "boltbeam.spend_measured.v1":
    return pathlib.Path(d["scope"]["model"]).name
  m = flag_value([str(x) for x in (d["argv"] if d.get("schema") == PROBE else d["inputs"]["server_argv"])], ("-m", "--model"))
  return pathlib.Path(m).name if m else None


def defaults(model:str, gpu_total, features:set[str]) -> list[str]:
  """The measured records of this model file, from the stores (STORES: BoltBeam's own measured rows, then
  ~/.boltbeam/spend): what tune ranks on when no --measured is given. Only a session-ab record of a GPU of this size
  (its lowest free memory is that GPU's; an unread size takes none) whose base switches this build has (its times are
  that build's). gpu_total: its bytes, read (a call) only when a record names the model."""
  out = []
  mine = [d for d in read_records([str(x) for x in STORES if x.is_dir()]) if model_of(d) == pathlib.Path(model).name
          and d.get("schema") == SCHEMAS[0]]  # a stored stress probe is headroom only (headroom_of), never an anchor
  gpu_total = (gpu_total() if callable(gpu_total) else gpu_total) if mine else None
  for d in mine if gpu_total else []:
    tot = {round(((r.get("telemetry") or {}).get("memory_mib") or {}).get("total") or 0) for r in d.get("arms") or []}
    env = {k for k in (d.get("inputs") or {}).get("server_env") or {} if k.isupper()}
    if tot == {round(gpu_total / MiB)} and env <= features:
      out.append(d["_path"])
  return out


def headroom_of(spend:dict[str, Any], records:list[dict[str, Any]], model:str, features:set[str], skip:tuple = ()) -> dict[str, Any]:
  """What a session can add after its lowest, kept free by every fit: the first spend.headroom row this build has. bound:
  its bytes formula over its params (inline, else from its record in the stores; a param may itself be a formula over the
  combination: room_at), taken only once every param is known; mitigation: the most any load-probe --stress of this model
  grew, given or stored (a crash's lower bound included). The stored probes count here only, never as anchors. skip: rows
  that could not be read (a bound whose formula failed falls to the next row)."""
  stored = [d for d in read_records([str(x) for x in STORES if x.is_dir()]) if d.get("schema") == PROBE]
  for r in spend.get("headroom") or []:
    if not available(r, features) or r["name"] in skip:
      continue
    if r["kind"] == "bound":
      try:
        rec = next((json.loads((x / r["record"]).read_text()) for x in STORES if r.get("record") and (x / r["record"]).is_file()), {})
      except (OSError, ValueError):
        continue  # an unreadable record: the next row
      got = {k: v if v is not None else rec.get(k) for k, v in r["params"].items()}
      if None not in got.values():
        return {"bytes": None, "kind": "bound", "row": r["name"], "formula": r["bytes"], "params": got, "switch": r.get("switch") or {}}
      continue  # its record has not landed: the next row
    grow = [d["stress_growth_mib"] for d in {d["_path"]: d for d in stored + records}.values()
            if d.get("schema") == PROBE and d.get("stress_growth_mib") is not None and model_of(d) == pathlib.Path(model).name]
    return {"bytes": max(grow) * MiB if grow else None, "kind": r["kind"], "row": r["name"]}
  return {"bytes": None, "kind": None, "row": None}


def room_at(room:dict[str, Any], spend:dict[str, Any], cfg:dict[str, Any], f:dict[str, Any], context:int, allowed:set[str]) -> float:
  """The headroom a combination keeps (bytes): a bound row's formula over its params (each a number, or a formula over the
  combination's facts: spend.<row>, context_tokens), else the row's one figure."""
  if not room.get("formula"):
    return room["bytes"] or 0
  from boltbeam.workflow import tune
  g = _facts(f, cfg, context)
  ok = allowed | set(g)
  g |= {k: tune.evaluate(e, g, ok) for k, e in (spend.get("sets") or {}).items()}
  g |= {f"headroom.{k}": tune.evaluate(v, g, ok | set(g)) if isinstance(v, str) else v for k, v in room["params"].items()}
  return float(tune.evaluate(room["formula"], g, ok | set(g)))


def recorded(spend:dict[str, Any], records:list[dict[str, Any]], model:str, fact:str) -> dict[str, list[str]]:
  """Each value of an argv fact (spend.argv_facts: a flag and its default) that the measured commands of this model set,
  with the records that set it: what a setup takes when nothing else names it (one value), or asks about (several). A
  command that leaves the flag at its default sets nothing here: the default is the engine's, not a measurement."""
  rule, out = (spend.get("argv_facts") or {}).get(fact), {}
  for d in records if rule else []:
    if model_of(d) != pathlib.Path(model).name or (d.get("schema") == PROBE and not d.get("loaded")):  # a probe that never loaded measured nothing
      continue
    inp = d.get("inputs") or d
    argv = [str(x) for x in inp.get("server_argv") or inp.get("argv") or []]
    for a in inp.get("arms") or [{}]:
      v = flag_value(argv + [str(x) for x in a.get("flags") or []], tuple(rule["flags"]))
      if v is not None:
        out.setdefault(v, []).append(where_of(d["_path"]))
  return {v: list(dict.fromkeys(w)) for v, w in out.items()}


def where_of(path:str) -> str:
  """A record by its run folder's name; a stored one (STORES) by its own name."""
  p = pathlib.Path(path)
  return p.stem if p.parent in STORES else p.parent.name


# --- reading a command into row values --------------------------------------------------------------------------------

def value_of(row:dict[str, Any], argv:list[str], env:dict[str, str], ad, known:dict[str, Any] | None = None) -> Any:
  """A row's value in a command: its flag's (an int), its env switch (on/off; no on value: the count set), or what the
  adapter reads from the file a flag names (known: the record's own reading of it, by file name, for a record measured elsewhere); the row's default
  where the command does not set it."""
  r = row.get("read") or {}
  if "flags" in r:
    v = flag_value(argv, tuple(r["flags"]))
    return int(v) if v is not None and v.lstrip("-").isdigit() else r.get("default")
  if "env" in r:
    v = str(env.get(r["env"]) or "")
    return ("on" if v == r["on"] else "off") if "on" in r else int(v) if v.isdigit() and int(v) > 0 else r.get("default")
  if "file_flags" in r:
    f = flag_value(argv, tuple(r["file_flags"]))
    got = ((known or {}).get(pathlib.Path(f).name) or {}).get(r["fact"]) if f else None
    return got if got is not None else ad.spend_value(r["fact"], f) if f else r.get("default")
  return r.get("default")


def layer_set(row:dict[str, Any], argv:list[str], f:dict[str, Any]) -> tuple[int, ...] | None:
  """The layers a command keeps in host RAM by a layers row (its `layers`): every whole-layer override it passes and the
  first N its count flag names. None: the row places no layers, or the model's per-layer bytes were not read. An expert
  override it cannot read raises ValueError (the record is skipped, said)."""
  lay = row.get("layers")
  by = f.get(lay["bytes"]) if lay else None
  if not by:
    return None
  pat, got = re.compile(lay["pattern"]), set()
  vals = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in lay["flags"]] + [a.split("=", 1)[1] for a in argv if a.split("=", 1)[0] in lay["flags"] and "=" in a]
  for v in (v for v in vals if lay["targets"].strip("_") in v):
    if not (m := pat.match(v)):  # an override this row cannot read (all experts, a range): its memory is unknown, never read as no layers
      raise ValueError(f"{v}: an expert override the layers row cannot read (it reads blk\\.(N|M)\\.ffn_.*_exps=CPU)")
    got |= {int(x) for x in m.group(1).split("|") if x}
  n = flag_value(argv, tuple(lay["count_flags"]))
  if n is not None and n.isdigit():
    got |= {layer for layer in by if layer < int(n)}
  return tuple(sorted(x for x in got if x in by))


def config_of(spend:dict[str, Any], argv:list[str], env:dict[str, str], ad, f:dict[str, Any], allowed:set[str], known:dict[str, Any] | None = None) -> dict[str, Any]:
  """A command's row values, its sink's bytes (_sink), the files its rows read (_files) and its argv_facts (a flag's
  value, else the default formula over the facts)."""
  from boltbeam.workflow import tune
  out: dict[str, Any] = {row["name"]: value_of(row, argv, env, ad, known) for row in spend["rows"]}
  for row in spend["rows"]:  # a layers row: its count is the layers the command keeps in RAM, by -ot or the count flag
    if (s := layer_set(row, argv, f)) is not None:
      out[row["name"]] = len(s)
      out.setdefault("_layers", {})[row["name"]] = s
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


def sessions(records:list[dict[str, Any]], spend:dict[str, Any], ad, model:str, f:dict[str, Any], allowed:set[str],
             known:dict[str, Any] | None = None, skipped:list | None = None) -> list[dict[str, Any]]:
  """Each session-ab record of this model as {scope, base, arms: {name: config, metrics, free_min, runs}}."""
  out = []
  for d in records:
    if d.get("schema") != "boltbeam.session_ab.v1":
      continue
    inp = d["inputs"]
    argv, env = [str(x) for x in inp["server_argv"]], dict(inp.get("server_env") or {})
    if model_of(d) != pathlib.Path(model).name:
      continue
    ctx = flag_value(argv, tuple(spend["context_flags"]))
    session = inp.get("session") or {}
    scope = {"context": int(ctx) if ctx else None, "session": session.get("sha256"), "turns": tuple(inp.get("turns") or (1, session.get("turns"))),
             "last": session.get("turns"), "path": d["_path"]}
    refused = {c["arm"] for c in (d.get("analysis") or {}).get("comparisons") or [] if c.get("refused")}
    refused |= set(re.findall(r"--floor[= ](\S+)", str(inp.get("hook") or "")))  # the gate's floor arm: a no-op perturbation, not a setup
    arms = {}
    for a in inp["arms"]:
      runs = [r for r in d["arms"] if r["name"] == a["name"] and r.get("valid")]
      if a.get("binary") and pathlib.Path(a["binary"]).name != pathlib.Path(argv[0]).name:  # another program (another engine): not this engine's setup
        continue
      if not runs or a["name"] in refused:
        continue
      try:
        cfg = config_of(spend, argv + [str(x) for x in a.get("flags") or []], {**env, **(a.get("env") or {})}, ad, f, allowed, {**(known or {}), **(inp.get("file_facts") or {})})
      except (OSError, ValueError) as exc:  # a file it names is not here and not known by name: the arm is not read, said with the record's own figures
        cmp_ = next((c for c in (d.get("analysis") or {}).get("comparisons") or [] if c.get("arm") == a["name"]), None)
        if skipped is not None and a["name"] not in refused:
          skipped.append({"where": f"{where_of(d['_path'])} {a['name']}", "why": f"{type(exc).__name__}: {exc}", "against": cmp_})
        continue
      used = [((r.get("telemetry") or {}).get("memory_mib") or {}) for r in runs]
      free = [(u["total"] - u["used_max"]) * MiB for u in used if u.get("total") and u.get("used_max")]
      arms[a["name"]] = {"config": cfg, "metrics": _arm_metrics(runs), "free_min": min(free) if free else None,
                         "free_spread": max(free) - min(free) if len(free) > 1 else None, "runs": len(runs)}
    first = (inp.get("order") or [None])[0]  # the arm order, or (block, arm) pairs
    base = (d.get("analysis") or {}).get("base") or (first[-1] if isinstance(first, list) and first else first)
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
  out = {r["name"]: (-sum(f[r["layers"]["bytes"]][x] for x in layers_of(r, cfg, f)) if layers_of(r, cfg, f) is not None
                     else float(tune.evaluate(r["bytes"], {**g, "value": cfg[r["name"]]}, ok))) for r in spend["rows"]}
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


# --- which layers: a layers row's placement -------------------------------------------------------------------------------

def layers_of(row:dict[str, Any], cfg:dict[str, Any], f:dict[str, Any]) -> tuple[int, ...] | None:
  """The layers a combination keeps in RAM by this row: the set its command read or its placement chose (_layers), else
  the `value` layers that free the most (largest first, ties to the lower layer): what that count can free at most.
  None: not a layers row, or no per-layer bytes."""
  by = f.get(row["layers"]["bytes"]) if row.get("layers") else None
  if not by:
    return None
  if (s := (cfg.get("_layers") or {}).get(row["name"])) is not None:
    return s
  return tuple(sorted(sorted(by, key=lambda x: (-by[x], x))[:int(cfg[row["name"]] or 0)]))


def layer_costs(row:dict[str, Any], cfg:dict[str, Any], f:dict[str, Any], split:tuple[float, float]) -> dict[int, float]:
  """Each layer's projected host traffic in RAM, in bytes a session-share: decode reads top-k experts of every layer a
  token (its bytes, the same share for each), a prompt chunk reaches a share of each layer's experts (layer_reach at the
  combination's chunk, the largest measured chunk not above it; uniform without a profile). split: the base session's
  (decode s, prefill s), which weighs the two."""
  lay, by = row["layers"], f[row["layers"]["bytes"]]
  reach = f.get(lay["reach"]) or None
  d, p = split
  if not reach or not (d + p):
    return {x: float(b) for x, b in by.items()}
  chunk = int(cfg.get(lay["chunk"].removeprefix("spend.")) or 0)
  at = reach[max([w for w in reach if w <= chunk] or [min(reach)])]
  mean = sum(at.values()) / len(at)
  return {x: b * (d + p * (at.get(x, mean) / mean)) / (d + p) for x, b in by.items()}


def choose(by:dict[int, float], cost:dict[int, float], k:int, need:float) -> tuple[int, ...] | None:
  """The k layers whose bytes reach `need` at the least cost (ties: the lower layers), exactly (a table over the layers,
  their count and their bytes capped at need). None: no k layers reach it."""
  unit = MiB  # the table counts whole MiB (a layer's bytes rounded down, need up, so a set it accepts frees at least need): bounded
  return _choose(tuple((x, int(by[x]) // unit, cost[x]) for x in sorted(by)), k, max(-(-int(need) // unit), 0))


@functools.lru_cache(maxsize=4096)
def _choose(layers:tuple[tuple[int, int, float], ...], k:int, need:int) -> tuple[int, ...] | None:
  """choose's table, cached: every combination of the other rows asks the same (layers, k, need) again."""
  best: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
  for x, mib, cost in layers:
    for (n, b), (c, ids) in list(best.items()):
      if n < k:
        key, val = (n + 1, min(b + mib, need)), (c + cost, ids + (x,))
        if key not in best or val < best[key]:
          best[key] = val
  got = best.get((k, need))
  return got[1] if got else None


def place(spend:dict[str, Any], cfg:dict[str, Any], fit:dict[str, Any], margin:float, f:dict[str, Any], split:tuple[float, float]) -> dict[str, Any]:
  """A fitting combination's layers rows placed: of `value` layers, the set that still frees what the combination needs
  (its free above the margin is what it may give back) at the least cost (layer_costs). The free memory and the sink
  follow; the set and the flags that pass it are kept (placed). A set the command read stays as it is."""
  out = {**fit, "placed": {}}
  for r in spend["rows"]:
    lay = r.get("layers")
    if not lay or not f.get(lay["bytes"]) or (cfg.get("_layers") or {}).get(r["name"]) is not None or not cfg[r["name"]]:
      continue
    by, most = f[lay["bytes"]], layers_of(r, cfg, f)
    spare = out["free"] - margin
    got = choose(by, layer_costs(r, cfg, f, split), len(most), sum(by[x] for x in most) - spare) if spare >= 0 else None
    got = got or most
    back = sum(by[x] for x in most) - sum(by[x] for x in got)
    out["free"] -= back
    out["by_row"] = {**out["by_row"], r["name"]: out["by_row"][r["name"]] + back}
    out["placed"][r["name"]] = {"layers": list(got), "argv": [lay["write"][0], lay["write"][1].format(ids=lay["sep"].join(map(str, got)))],
                               "reach": f.get(lay["reach_source"])}
  if out["has_sink"]:
    out["sink"] = max(out["free"] - margin, 0)
  return out


# --- the fit --------------------------------------------------------------------------------------------------------------

def candidates(spend:dict[str, Any], base:dict[str, Any], measured:list[dict[str, Any]], f:dict[str, Any], features:set[str]) -> dict[str, list]:
  """Each row's values: the base arm's, and where the row is available here every measured one, its own values (a fit_value row: its derive marker)
  and its range (a count fact: 0..N); values_from adds another row's (a batch is at least a chunk)."""
  out = {}
  for r in spend["rows"]:
    vals = [base[r["name"]]]
    if available(r, features) and r.get("fit_value"):
      vals += list(r.get("values") or []) + [DERIVE]  # the value each combination fits, not the ones measured elsewhere
    elif available(r, features):
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
  drv = {r["name"] for r in spend["rows"] if r.get("fit_value")}
  for m in METRICS:
    changed = {r: cfg[r] for r in moves if m in moves[r] and cfg[r] != base["config"][r]}
    tries = [changed]
    for r in drv & set(changed):  # a derived value: a measured one at or below it, largest first (a larger value is no slower)
      if isinstance(base["config"][r], int) and isinstance(changed[r], int) and base["config"][r] <= changed[r]:
        tries = [{k: v for k, v in t.items() if k != r} for t in tries]  # the base's own value covers it
        continue
      got = sorted({u["rows"][r] for u in us if u["metric"] == m and isinstance(u["rows"].get(r), int) and u["rows"][r] <= changed[r]}, reverse=True)
      tries = [{**t, r: v} for t in tries for v in got] or tries if isinstance(changed[r], int) else tries
    changed = next((t for t in tries if _cover(t, base["config"], [u for u in us if u["metric"] == m], sink, srow) is not None), tries[0])
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


def probes(records:list[dict[str, Any]], spend:dict[str, Any], ad, model:str, f:dict[str, Any], allowed:set[str], known:dict[str, Any]) -> list[dict[str, Any]]:
  """Each load probe of this model as {config, context, loaded (it ran its session), free_min (its lowest free, bytes),
  name, where}: measured memory without times, read as a session-ab arm's command is."""
  out = []
  for d in records:
    if d.get("schema") != PROBE or model_of(d) != pathlib.Path(model).name or (d["loaded"] and d.get("free_min_mib") is None) \
       or (not d["loaded"] and (d.get("failure") or {}).get("out_of_memory") is not True):  # a load with no reading, or a failed load not out of memory: no verdict
      continue
    argv = [str(x) for x in d["argv"]]
    ctx = flag_value(argv, tuple(spend["context_flags"]))
    try:
      cfg = config_of(spend, argv, dict(d.get("env") or {}), ad, f, allowed, {**known, **(d.get("file_facts") or {})})
    except (OSError, ValueError) as exc:  # a file it names is not here and not known by name: skipped, said
      out.append({"skipped": f"{where_of(d['_path'])}: {type(exc).__name__}: {exc}"})
      continue
    out.append({"config": cfg, "context": int(ctx) if ctx else None, "loaded": bool(d["loaded"] and d["session"]), "no_load": not d["loaded"],
                "stress_oom": ((d.get("stress") or {}).get("failure") or {}).get("out_of_memory") is True,
                "free_min": d["free_min_mib"] * MiB if d.get("free_min_mib") is not None else None, "free_spread": None, "runs": 1, "name": f"probe {d.get('name')}", "where": d["_path"], "probe": True})
  return out


def probe_fall(records:list[dict[str, Any]], model:str, context:int | None, spend:dict[str, Any], gap:float = 0.0) -> dict[str, Any] | None:
  """The most a session took below the free memory at load, over the load probes of this model at this context that
  ran their session (free at ready less its lowest after ready: a dip while loading is not the session's), plus how much lower a full session went than a probe's short one
  (gap: probe_gap): what tune's memory fit keeps free for a session when no run of its own setup was read. None: no such probe."""
  got = [(d["free_ready_mib"] - d.get("free_min_after_ready_mib", d["free_min_mib"]), d.get("name")) for d in records if d.get("schema") == PROBE and d["loaded"] and d["session"]
         and d.get("free_ready_mib") is not None and model_of(d) == pathlib.Path(model).name
         and context is not None and str(flag_value([str(x) for x in d["argv"]], tuple(spend["context_flags"]))) == str(context)]
  if not got:
    return None
  mib, name = max(got)
  return {"bytes": max(mib, 0) * MiB + gap, "where": f"the most of {len(got)} load probes at this context ({name})"
          + (f", plus {gap / MiB:,.0f} MiB a full session went below a probe's short one" if gap else "")}


def probe_gap(short:list[dict[str, Any]], full:list[dict[str, Any]], names:list[str]) -> dict[str, Any]:
  """How much lower a full session's lowest free memory (its cache added back) went than a load probe's short session at the same setup (same
  rows, layers and context): the most over such pairs, taken off every probe's reading before it anchors. None when no
  probe has a full-session twin: the probes' readings stand as they are, an upper bound, said."""
  pairs = [(p["free_min"] + (p["config"].get("_sink") or 0) - a["free_min"] - (a["config"].get("_sink") or 0), p["name"], a["name"]) for p in short for a in full
           if a["context"] == p["context"] and _same(a["config"], p["config"], names)]
  if not pairs:
    return {"bytes": None, "pairs": 0}
  most = max(pairs)
  return {"bytes": max(most[0], 0.0), "pairs": len(pairs), "from": f"{most[1]} against {most[2]}"}


def plan_gap(spend:dict[str, Any], records:list[dict[str, Any]], model:str, f:dict[str, Any], ad, allowed:set[str]) -> float:
  """probe_gap over these records, as plan reads them (the full sessions' arms and the probes): bytes, 0 when unmeasured."""
  known = {k: v for d in records for k, v in ((d.get("inputs") or d).get("file_facts") or {}).items()}
  full = [{**a, "context": s["scope"]["context"], "name": n} for s in sessions(records, spend, ad, model, f, allowed, known) for n, a in s["arms"].items()
          if a["free_min"] is not None and s["scope"]["turns"][-1] == s["scope"]["last"] and s["scope"]["context"]]
  short = [p for p in probes(records, spend, ad, model, f, allowed, known) if "config" in p and p["loaded"] and p["context"]]
  return probe_gap(short, full, [r["name"] for r in spend["rows"]])["bytes"] or 0.0


def known_files(spend:dict[str, Any], ad, local:list[str] | None) -> dict[str, Any]:
  """The facts the file rows read from the files given here (tune's own draft), by file name: what a record measured
  elsewhere names by its own path."""
  out: dict[str, Any] = {}
  for path in local or []:
    for r in spend["rows"]:
      if "file_flags" in (r.get("read") or {}):
        try:
          out.setdefault(pathlib.Path(path).name, {})[r["read"]["fact"]] = ad.spend_value(r["read"]["fact"], path)
        except (OSError, ValueError):
          pass
  return out


def _seconds(m:dict[str, Any]) -> float:
  return (m["tokens"] / m["tokens_per_step"] * m["step_ms"] + m["prefill_total_ms"]) / 1000


def project_any(spend:dict[str, Any], cfg:dict[str, Any], fit:dict[str, Any], base:dict[str, Any], us:list[dict[str, Any]],
                others:list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
  """project() on the base session, else on another record whose base is the same setup: its projection over its own
  turns, scaled to the base session by the two bases' measured times (the same setup measured on both)."""
  sink = fit["sink"] if fit["has_sink"] else None
  got = project(spend, cfg, fit, base, us, sink)
  for b, o in others if "unmeasured" in got else []:
    q = project(spend, cfg, fit, b, o, sink)
    if "seconds" in q:
      k = _seconds(base["metrics"]) / _seconds(b["metrics"])
      return {**q, "seconds": q["seconds"] * k, "decode_s": q["decode_s"] * k, "prefill_s": q["prefill_s"] * k,
              "scaled": f"turns {'-'.join(map(str, b['scope']['turns']))} of {where_of(b['scope']['path'])}, scaled by its base's time ({k:.2f}x)"}
  return got


def plan(f:dict[str, Any], engine:str, tab:dict[str, Any], *, model:str, records:list[dict[str, Any]], context:int | None,
         features:set[str], ad, allowed:set[str], local:list[str] | None = None) -> dict[str, Any]:
  """The spend fit at a context (None: the most cells each combination holds): every combination costed from its nearest
  anchor, the measured ones ranked by projected session time; the rows not available here with the memory they would
  free. {why} when there is nothing to anchor on."""
  spend, at_ = rows_of(engine, tab), (engine, tab)
  if not spend:
    return {"why": "no_rows"}
  known = {k: v for d in records for k, v in ((d.get("inputs") or d).get("file_facts") or {}).items()}  # a file another record read, by name
  known = {**known, **known_files(spend, ad, local)}
  unread: list[dict[str, Any]] = []
  sess = sessions(records, spend, ad, model, f, allowed, known, unread)
  probed = probes(records, spend, ad, model, f, allowed, known)
  failed = [p for p in probed if "config" in p and p["no_load"] and p["context"]]  # never ready: out of memory at load (a later error is not a fit verdict)
  oom = [p for p in probed if "config" in p and p.get("stress_oom") and p["context"]]  # ran its session, then ran out under the stress: never fits
  anchors = [{**a, "context": s["scope"]["context"], "name": n, "where": s["scope"]["path"]} for s in sess for n, a in s["arms"].items()
             if a["free_min"] is not None and s["scope"]["turns"][-1] == s["scope"]["last"] and s["scope"]["context"]]
  short = [p for p in probed if "config" in p and p["loaded"] and p["context"]]  # a probe that ran its (short) session: its lowest free
  gap = probe_gap(short, anchors, [r["name"] for r in spend["rows"]])
  anchors += [{**p, "free_min": p["free_min"] - (gap["bytes"] or 0)} for p in short]
  if not anchors:
    return {"why": "no_anchor"}
  try:
    margin = margin_of(spend, anchors, f, allowed, at_, failed)
  except Exception as exc:  # noqa: BLE001  a fact the rows read is missing (no draft head given): said, never a crash
    return {"why": "no_facts", "what": f"{type(exc).__name__}: {exc}"}
  # runtime headroom: what new and resumed prompts add beyond a session's lowest, kept by every fit: the build's row (headroom_of)
  room = headroom_of(spend, records, model, features)
  for a in anchors:
    a["margin"] = margin["bytes"] if margin["bytes"] is not None else 0
  timed = [s for s in sess if s["arms"][s["base"]]["metrics"]]
  if not timed:
    return {"why": "no_anchor"}
  ref = max(timed, key=lambda s: (s["scope"]["context"] == context, s["scope"]["turns"][-1] == s["scope"]["last"], s["scope"]["turns"][-1] - s["scope"]["turns"][0]))  # at the asked context first
  base = {**ref["arms"][ref["base"]], "scope": ref["scope"]}
  try:  # said at the base's setup; each combination keeps its own (room_at)
    head = room_at(room, spend, base["config"], f, context or (int(f.get(spend["context_cap"]) or 0) if spend.get("context_cap") else 0) or base["scope"]["context"], allowed)
  except Exception:  # noqa: BLE001  a bound whose formula cannot be read here: the next row (the mitigation)
    room = headroom_of(spend, records, model, features, skip=(room["row"],))
    head = room["bytes"] or 0
  margin |= {"headroom": head if room["kind"] == "bound" else room["bytes"], "headroom_kind": room["kind"], "headroom_row": room["row"],
             "headroom_switch": " ".join(f"{k}={v}" for k, v in (room.get("switch") or {}).items()), "switch": room.get("switch") or {}}  # tune's command sets it
  every = units(spend, sess, records, model)
  inside = lambda sc: [u for u in every if u["scope"]["context"] == sc["context"] and u["scope"]["session"] == sc["session"]  # noqa: E731
                       and tuple(u["scope"]["turns"]) == tuple(sc["turns"])]  # the same context, session and turns
  us = inside(base["scope"])
  # another timed record at this context whose base ran the same setup (rows, layers, cache): its arms project through
  # the ratio of its own base's time, so a screen over other turns ranks beside the full session
  same = lambda a: _same(a["config"], base["config"], [r["name"] for r in spend["rows"]]) and a["config"].get("_sink") == base["config"].get("_sink")  # noqa: E731
  others = [({**s["arms"][s["base"]], "scope": s["scope"]}, inside(s["scope"])) for s in timed
            if s is not ref and s["scope"]["context"] == base["scope"]["context"] and same(s["arms"][s["base"]])]
  cap = int(f.get(spend["context_cap"]) or 0) if spend.get("context_cap") else 0
  cands = candidates(spend, base["config"], us + [u for _, o in others for u in o], f, features)
  names = list(cands)
  bm = base["metrics"]
  split = (bm["tokens"] / bm["tokens_per_step"] * bm["step_ms"] if bm.get("tokens_per_step") else 0.0, bm["prefill_total_ms"])  # the base session's decode and prefill ms
  out = []
  for vals in itertools.product(*(cands[n] for n in names)):
    cfg = {**base["config"], **dict(zip(names, vals))}
    ran = next((a["config"] for a in [{"config": base["config"]}, *anchors] if all(a["config"][n] == cfg[n] for n in names)), None)
    cfg["_layers"] = dict((ran or {}).get("_layers") or {})  # a measured setup's values: its own layers as it ran them; else placed
    drv = [r for r in spend["rows"] if r.get("fit_value") and cfg[r["name"]] == DERIVE]
    cfg |= {r["name"]: r["fit_value"]["min"] for r in drv}
    if any(not _holds(r, cfg, allowed, context or cap, f) for r in spend["rows"]):
      continue
    apart = lambda a: (sum(a["config"][n] != cfg[n] for n in names), not _same(a["config"], cfg, names), a["context"] != context)  # noqa: E731
    anchor = min(anchors, key=lambda a: (*apart(a), -a["runs"]))
    vals = tuple(cfg[n] for n in names)
    # the measured setup itself (same rows, layers and context): its own reading, so only the run-to-run spread is kept
    exact = not any(apart(anchor)) and not anchor.get("probe")  # a full session of this very setup (a probe's short one is not)
    try:
      head = room_at(room, spend, cfg, f, context or cap, allowed)
      anchor = {**anchor, "margin": anchor["margin"] + head}
      keep = ((margin["spread"] or 0) + head) if exact else anchor["margin"]
      cells = context or _max_cells(spend, cfg, anchor, f, allowed, cap, at_)
      if drv and free_at(spend, cfg, anchor, f, cells, allowed, at_)["has_sink"]:
        continue  # behind a cache the derived value would take the cache's memory: a trade nothing measured
      for r in drv:
        cfg[r["name"]] = _derive(spend, r, cfg, anchor, f, cells, allowed, at_, keep)
      if not context and any(not _holds(r, cfg, allowed, cells, f) for r in spend["rows"]):  # a value legal only below the context it holds
        continue
      fit = place(spend, cfg, free_at(spend, cfg, anchor, f, cells, allowed, at_), keep, f, split)
      if exact and fit["has_sink"]:  # its cache as it ran, plus what its measured lowest leaves above the full margin
        own = anchor["config"].get("_sink") or 0
        fit = {**fit, "sink": max(own, min(fit["sink"], fit["free"] - margin["bytes"])) if margin["bytes"] is not None else own}
        fit["sink"] = max(min(fit["sink"], fit["free"] - keep), 0)  # never into the headroom: the cache gives it back
    except Exception as exc:  # noqa: BLE001  a row whose memory cannot be read is never chosen, and said
      out.append({"values": {**dict(zip(names, vals)), **{r["name"]: DERIVE for r in drv}}, "fits": False, "why": f"{type(exc).__name__}: {exc}"})
      continue
    vals = tuple(cfg[n] for n in names)
    lay = {**cfg["_layers"], **{r: tuple(v["layers"]) for r, v in fit["placed"].items()}}
    veto = next((p for p in failed + oom if p["context"] == cells and (_same(p["config"], {**cfg, "_layers": lay}, names) or drv and all(  # a derived value at or above a failed one fails too
      isinstance(p["config"].get(r["name"]), int) and cfg[r["name"]] >= p["config"][r["name"]] for r in drv) and _same(p["config"], {**cfg, "_layers": lay}, [n for n in names if n not in {r["name"] for r in drv}]))), None)
    row = {"values": dict(zip(names, vals)), "derived": [r["name"] for r in drv], "context": cells, "free": fit["free"] - fit["sink"], "sink": fit["sink"], "fits": fit["free"] >= keep and cells > 0 and not veto,
           "placed": fit["placed"], "failed": veto and veto["name"],
           "anchor": anchor["name"], "by_row": fit["by_row"], "layers": cfg.get(spend["sink"]["row"]) if spend.get("sink") else None,
           "files": {r: next((u["files"][r] for u in us if u["rows"].get(r) == cfg[r] and u["files"].get(r)), None) for r in cfg.get("_files") or {}}}
    if row["fits"] and margin["bytes"] is None:
      row["projection"] = {"unmeasured": ["margin"]}  # no measured spread or residual: whether it fits is not known well enough to rank
    elif row["fits"]:
      row["projection"] = project_any(spend, cfg, fit, base, us, others) if cells == base["scope"]["context"] else {"unmeasured": ["context"]}
    out.append(row)
  held, every_n = _held_off(spend, out), len(out)
  out = [r for r in out if id(r) not in held]
  ranked = sorted((r for r in out if r["fits"] and "seconds" in r["projection"]), key=lambda r: r["projection"]["seconds"])
  unranked = sorted((r for r in out if r["fits"] and "seconds" not in r["projection"]), key=lambda r: (-r["context"], r["layers"] or 0, -r["free"] - r["sink"]))
  return {"context": context, "measured_context": base["scope"]["context"], "margin": margin,
          "base": {"name": ref["base"], "config": {n: base["config"][n] for n in names}, "metrics": base["metrics"], "where": ref["scope"]["path"], "sink": base["config"].get("_sink") or 0,
                   "turns": list(ref["scope"]["turns"])},
          "ranked": ranked, "unmeasured_fits": unranked, "combinations": every_n, "held_off": len(held), "errors": [r for r in out if r.get("why")][:3],
          "pending": pending(spend, features, f, base["config"], context or base["scope"]["context"], allowed, at_),
          "unread": unread, "probes": {"gap": gap, "anchors": sum(1 for p in probed if p.get("loaded")), "failed": len(failed), "skipped": [p["skipped"] for p in probed if "skipped" in p]},
          "units": [{k: u[k] for k in ("rows", "metric", "value", "where")} for u in us]}


def margin_of(spend:dict[str, Any], anchors:list[dict[str, Any]], f:dict[str, Any], allowed:set[str], at_:tuple,
              failed:list[dict[str, Any]] | None = None) -> dict[str, Any]:
  """What a combination must keep free beyond its predicted lowest: the larger of the spread of the lowest free over runs
  of one setup (repeats, and arms of equal setups) and the worst error of the memory formulas when one measured arm is
  predicted from another (leave one out), and when a probe that did not load is predicted from each arm (it had less
  than nothing free, so the free predicted for it is a lower bound of the error). Unknown (bytes None) with neither."""
  key = lambda a: tuple(sorted((k, str(v)) for k, v in a["config"].items() if not k.startswith("_")) + [("ctx", str(a["context"])),  # noqa: E731
                                                                                                         ("layers", str(a["config"].get("_layers")))])
  groups: dict[tuple, list[float]] = {}
  for a in anchors:
    if not a.get("probe"):  # a probe's short session is not a repeat of a full one (probe_gap carries that difference)
      groups.setdefault(key(a), []).append(a["free_min"] + (a["config"].get("_sink") or 0))  # the cache it ran is memory it chose to spend
  spread = [max(g) - min(g) for g in groups.values() if len(g) > 1] + [a["free_spread"] for a in anchors if a["free_spread"] is not None]
  resid = []
  for a, b in itertools.permutations(anchors, 2):
    if key(a) != key(b):
      got = free_at(spend, b["config"], {**a, "margin": 0}, f, b["context"], allowed, at_)["free"]
      resid.append(abs(got - (b["free_min"] + (b["config"].get("_sink") or 0))))
  for b in failed or []:
    for a in anchors:
      resid.append(max(free_at(spend, b["config"], {**a, "margin": 0}, f, b["context"], allowed, at_)["free"] - (b["config"].get("_sink") or 0), 0.0))  # its own cache was taken too
  both = spread + resid
  return {"bytes": max(both) if both else None, "spread": max(spread) if spread else None, "residual": max(resid) if resid else None}


def _same(a:dict[str, Any], b:dict[str, Any], names:list[str]) -> bool:
  """Two combinations with the same row values and the same layers in RAM."""
  la, lb = a.get("_layers") or {}, b.get("_layers") or {}
  return all(a.get(n) == b.get(n) for n in names) and all(tuple(la.get(r) or ()) == tuple(lb.get(r) or ()) for r in set(la) | set(lb))


def _holds(row:dict[str, Any], cfg:dict[str, Any], allowed:set[str], context:int, f:dict[str, Any] | None = None) -> bool:
  from boltbeam.workflow import tune
  g = _facts({}, cfg, context)
  return not row.get("valid") or bool(tune.evaluate(row["valid"], {**(f or {}), **g, "value": cfg[row["name"]]}, allowed | set(g)))


def _max_cells(spend:dict[str, Any], cfg:dict[str, Any], anchor:dict[str, Any], f:dict[str, Any], allowed:set[str], cap:int, at_:tuple) -> int:
  """The most cells (a multiple of the engine's cell step, at most the cap) the combination holds: its free memory is
  affine in the cells, so two points (the anchor's context and twice it) give it."""
  c0 = anchor["context"]
  lo, hi = (free_at(spend, cfg, anchor, f, c, allowed, at_)["free"] - anchor["margin"] for c in (c0, 2 * c0))
  per = (lo - hi) / c0
  cells = int(c0 + lo / per) // spend["cells_step"] * spend["cells_step"] if per > 0 else 0
  return max(0, min(cells, cap) if cap else cells)


def _derive(spend:dict[str, Any], row:dict[str, Any], cfg:dict[str, Any], anchor:dict[str, Any], f:dict[str, Any], cells:int,
            allowed:set[str], at_:tuple, keep:float) -> int:
  """A fit_value row's value for the combination: the largest multiple of its step (at least its min, below the cells) whose
  free memory keeps `keep`; the free is affine in the value, so two points (min and twice it) give it."""
  lo_v, step = row["fit_value"]["min"], row["fit_value"]["step"]
  lo, hi = (free_at(spend, {**cfg, row["name"]: v}, anchor, f, cells, allowed, at_)["free"] - keep for v in (lo_v, 2 * lo_v))
  per = (lo - hi) / lo_v
  top = (cells - 1) // step * step
  return max(lo_v, min(top, int(lo_v + lo / per) // step * step)) if per > 0 and lo > 0 else lo_v


def _held_off(spend:dict[str, Any], out:list[dict[str, Any]]) -> set[int]:
  """The combinations an only_where_default_fails row holds off: a value other than its read default where the same
  combination at the default fits (it frees memory nothing needs and costs time)."""
  held = set()
  for r in (r for r in spend["rows"] if r.get("only_where_default_fails")):
    n, d = r["name"], (r.get("read") or {}).get("default")
    key = lambda x: tuple(sorted((k, str(v)) for k, v in x["values"].items() if k != n))  # noqa: E731
    fits: dict[tuple, int] = {}
    for x in out:
      if x["values"].get(n) == d and x["fits"]:
        fits[key(x)] = max(fits.get(key(x), 0), x["context"])
    held |= {id(x) for x in out if x["values"].get(n) != d and not x.get("why") and fits.get(key(x), -1) >= x["context"]}  # its twin holds as much context
  return held


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

def label(values:dict[str, Any], spend:dict[str, Any], files:dict[str, Any] | None = None, placed:dict[str, Any] | None = None) -> str:
  """A combination as the reader reads it: each row by name and value; a file row names its file; a placed layers row
  the flags that keep its layers in RAM."""
  out = []
  for r in spend["rows"]:
    v = values.get(r["name"])
    if v in (None, "off"):
      continue
    if (pl := (placed or {}).get(r["name"])) and v:
      out.append(f"{r['name']} {v} ({' '.join(pl['argv'])})")
      continue
    f, env = (files or {}).get(r["name"]), (r.get("read") or {}).get("env")
    if env and ("on" not in r["read"] or str(v) in ("on", str(r["read"]["on"]))):  # a switch is said as the command sets it: NAME=on, or NAME=N
      out.append(f"{r['name']} ({env}={r['read'].get('on', v)})")
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
                            resid="?" if p["margin"]["residual"] is None else f"{p['margin']['residual'] / MiB:.0f}",
                            head=w[f"headroom_{p['margin'].get('headroom_kind') or 'mitigation'}"].format(mib=p["margin"]["headroom"] / MiB, switch=p["margin"].get("headroom_switch") or "its switch") if p["margin"].get("headroom") is not None else w["headroom_none"])]
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
    out.append(w["best_is_base" if same else "best" if pr["runs"] > 1 else "candidate"].format(config=label(best["values"], spend, best["files"], best.get("placed")), s=pr["seconds"], ds=pr["seconds"] - base_s, pct=100 * (pr["seconds"] / base_s - 1),
                                step=pr["step_ms"], tps=pr["tokens_per_step"], pre=pr["prefill_ms"], free=best["free"] / MiB, sink=sink(best)))
    out.append(w["from"].format(units="; ".join(pr["from"]) or "the base run itself") + (f" ({pr['scaled']})" if pr.get("scaled") else ""))
    out += [w["derive"].format(file=pathlib.Path(best["files"][r["name"]]).name, how=r["derive"]) for r in spend["rows"]
            if r.get("derive") and (best["files"] or {}).get(r["name"]) and best["values"][r["name"]] != b["config"][r["name"]]]
    out += [w["fit_value"].format(row=n, v=best["values"][n]) for n in best.get("derived") or []]
    if len(ranked) > 1:
      r2, q = ranked[1], ranked[1]["projection"]
      out.append(w["runner"].format(config=label(r2["values"], spend, r2["files"], r2.get("placed")), s=q["seconds"], ds=q["seconds"] - pr["seconds"], step=q["step_ms"],
                                    tps=q["tokens_per_step"], pre=q["prefill_ms"], free=r2["free"] / MiB, sink=sink(r2)))
      why = [w["why_metric"].format(what=w[k], base=fmt(m, q[m]), best=fmt(m, pr[m]))
             for m, k in (("step_ms", "metric_step"), ("tokens_per_step", "metric_tps"), ("prefill_ms", "metric_prefill")) if fmt(m, q[m]) != fmt(m, pr[m])]
      out.append(w["why"].format(why="; ".join(why), s=q["seconds"] - pr["seconds"]))
  else:
    out.append(w["none_ranked"].format(context=ctx))
  if p.get("held_off"):
    out.append(w["held_off"].format(n=p["held_off"], rows=", ".join(r["name"] for r in spend["rows"] if r.get("only_where_default_fails"))))
  if un := p["unmeasured_fits"]:
    items = [w["unranked_item"].format(config=label(r["values"], spend, None, r.get("placed")), free=r["free"] / MiB, sink=sink(r),
                                       ctx="" if p["context"] else w["ctx"].format(cells=r["context"])) for r in un[:3]]
    out.append(w["unranked"].format(n=len(un), lead=w["unranked_lead"] if p["context"] else "", items="; ".join(items)))
    if not p["context"]:  # the most context: also what the fewest expert layers in RAM hold
      few = max((r for r in un + ranked if r["layers"] == min(x["layers"] or 0 for x in un + ranked)), key=lambda r: r["context"])
      out.append(w["fewest_layers"].format(layers=few["layers"], cells=few["context"], config=label(few["values"], spend, None, few.get("placed"))))
    out.append(w["measure_hint"].format(context=f"{p['context'] or un[0]['context']:,}"))
  for u in p.get("unread") or []:  # measured, not costed: the record's own comparison, its deepest band
    c = u["against"] or {}
    band = (c.get("bands") or [{}])[-1] if c else {}
    fig = lambda k: band.get(k) or {}  # noqa: E731
    out.append(w["unread"].format(where=u["where"], why=u["why"]) + (w["unread_cmp"].format(
      band=band.get("band", "?"), ttft_b=fig("ttft_ms").get("base", 0), ttft_a=fig("ttft_ms").get("arm", 0),
      dec_b=fig("decode_tok_s").get("base", 0), dec_a=fig("decode_tok_s").get("arm", 0)) if band.get("ttft_ms") else ""))
  pr_ = p.get("probes") or {}
  if pr_.get("anchors") or pr_.get("failed") or pr_.get("skipped"):
    g = pr_.get("gap") or {}
    out.append(w["probes"].format(n=pr_.get("anchors", 0), failed=pr_.get("failed", 0))
               + (w["probes_gap"].format(mib=g["bytes"] / MiB, src=g["from"]) if g.get("bytes") is not None else w["probes_nogap"] if pr_.get("anchors") else "")
               + (w["probes_skipped"].format(what="; ".join(pr_["skipped"])) if pr_.get("skipped") else ""))
  if placed := next((r["placed"] for r in [*ranked, *un] if r.get("placed")), None):
    for name, pl in placed.items():
      row = next(r for r in spend["rows"] if r["name"] == name)
      out.append(w["placed"].format(why=row["layers"]["why"], reach=pl.get("reach") or ""))
  if p["pending"]:
    out.append(w["pending_head"])
    for r in p["pending"]:
      why = w["pending_unbuilt"] if not r["built"] else w["pending_built"].format(switch=r["requires"])
      cost = w["pending_cost"] if any(r["row"] in u["rows"] for u in p["units"]) else w["pending_nocost"]
      out.append(w["pending"].format(words=r["words"], mib="?" if r["frees"] is None else f"{r['frees'] / MiB:,.0f}", why=f"{why}; {cost}"))
  out.append(w["command_apart"])
  return out
