"""Measured uncertainty for an exact run setup, never a chip-wide default."""
from __future__ import annotations

import datetime
import math
import pathlib
from typing import Any

from boltbeam.artifacts.base import sha256_json
from boltbeam.workflow.common import load_manifest, read_json, write_json

SCHEMA = "boltbeam.repeat_bands.v1"
ARTIFACT = "repeat_bands.json"
PLAN_SCHEMA = "boltbeam.repeat_plan.v1"
PIN_RUNS = 4  # runs a band needs, fixed before collection, before it may justify an engine pin; fewer is a weak verdict
RULE = ("a row moved when the after value lies outside the range of n independent before runs of its exact setup; "
        "with no change, that happens with probability 2/(n+1) per row (exchangeable runs)")


def strength(n:int) -> dict:
  """The band's n, its false-move rate per row and whether its verdicts are weak (fewer runs than PIN_RUNS)."""
  return {"n": n, "false_move_rate": 2 / (n + 1), "weak": n < PIN_RUNS}


def plan(n:int, out:pathlib.Path) -> dict:
  """Fix n before any repeat is collected: the plan's time precedes every run a band built on it may use."""
  if n < 2:
    raise ValueError("a band needs at least two runs")
  record = {"schema": PLAN_SCHEMA, "n": n, "planned_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
  write_json(out, record)
  return record


def _read(run:pathlib.Path, name:str) -> dict:
  path = run / name
  return read_json(path) if path.exists() else {}


def scope(run:pathlib.Path) -> dict:
  from boltbeam.collectors import providers
  from boltbeam.workflow.screen import run_provider
  manifest = load_manifest(run)
  profile = _read(run, "model_profile.json")
  profile = {k: v for k, v in profile.items() if k not in ("source", "model_id")}
  request = _read(run, "trace_request.json")
  timing = _read(run, "timing_trace.json")
  machine = _read(run, "machine_facts.json")
  status = _read(run, "measure_status.json")
  capture_file = {"prefill": "prefill_trace.json", "session": "session_trace.json",
                  "replay": "replay_capture.json"}.get(manifest.get("workload"), providers.TRACES[run_provider(run)])
  capture = _read(run, capture_file)
  whole = next((r for r in capture.get("rows", []) if r.get("scope") == "whole_step"), {})
  methods = {length: {k: attr[k] for k in ("method", "formats") if k in attr}
             for length, attr in capture.get("lengths", {}).items()}
  return {"target": manifest.get("target_id"), "devices": sorted(machine.get("uuids") or []),
          "engine": run_provider(run), "build": providers.run_build(run) or (manifest.get("engine") or {}).get("build"),
          "model": {"path": manifest.get("model_path"), "profile": sha256_json(profile) if profile else None},
          "workload": manifest.get("workload"),
          "layout": {k: status.get(k) for k in ("layout", "gpus", "batches")},
          "capture": {"method": capture.get("capture", {}),
                      "window": {k: whole[k] for k in ("context", "decode_tokens", "batch") if k in whole},
                      "parts": methods, "depth": capture.get("depth", 0)},
          "replay": timing.get("replay_trace") or {},
          "request": {k: request[k] for k in ("contexts", "execution", "engine_settings") if k in request},
          "settings": [{"context": r.get("context"), "engine": {k: v for k, v in (r.get("engine") or {}).items()
                         if k not in ("command", "wall_samples_ms")}}
                       for r in timing.get("rows", []) if r.get("scope") == "whole_step" or r.get("prefill_ms")],
          "toolchain": {"versions": (manifest.get("toolchain") or {}).get("versions", {}),
                        "runtime": (manifest.get("toolchain") or {}).get("runtime", {})}}


def _date(run:pathlib.Path) -> str:
  manifest = load_manifest(run)
  date = (manifest.get("toolchain") or {}).get("recorded_at") or _read(run, "machine_facts.json").get("measured_at")
  if not date:
    raise ValueError(f"{run.name}: no recorded measurement date; file modification time is not a run date")
  datetime.date.fromisoformat(str(date)[:10])
  return str(date)


def _spread(xs:list[float]) -> dict:
  if len(xs) < 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in xs):
    raise ValueError("repeat measurements must be at least two finite nonnegative times")
  mean = sum(xs) / len(xs)
  return {"runs_ms": list(xs), "low_ms": min(xs), "high_ms": max(xs), "reference_ms": mean, "width_ms": max(xs) - min(xs),
          "width_pct": 100 * (max(xs) - min(xs)) / mean if mean else None}


def _runs(row:dict) -> list:
  return row.get("runs_ms") or [row.get("before_ms"), row.get("after_ms")]  # a band saved as a pair: n = 2


def _when(date:str) -> datetime.datetime:
  d = datetime.datetime.fromisoformat(str(date).replace("Z", "+00:00"))
  return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def calibrate(runs:list[pathlib.Path], fixed:dict | None = None) -> dict:
  """Each row's range over n independent repeats of one setup (n >= 2), with n and its false-move rate. With a plan
  (plan()), n must be the planned n and every run must come after the plan."""
  from boltbeam.workflow import diff
  if len(runs) < 2 or len({r.resolve() for r in runs}) != len(runs):
    raise ValueError("a band needs at least two distinct runs")
  a = runs[0]
  sa = scope(a)
  for b in runs[1:]:
    sb = scope(b)
    diff.refuse_unless_same([(k, sa[k], sb[k]) for k in sa])
  if not sa["model"]["profile"] or not sa["model"]["path"] or not sa["request"].get("contexts"):
    raise ValueError("repeat scope needs model profile, model path and requested contexts")
  if not sa["build"] or sa["build"] == "unknown":
    raise ValueError("repeat scope needs a recorded engine build")
  if sa["workload"] == "decode":
    base = sa["capture"]["window"].get("context")
    if base not in sa["request"]["contexts"]:
      raise ValueError("capture context differs from the requested setup; this pair cannot calibrate that setup")
  S = [sides(r) for r in runs]
  x = S[0]
  if not x or any(y.keys() != x.keys() for y in S):
    raise ValueError("repeat captures must cover the same nonempty workload parts")
  parts = {}
  for part in x:
    if any(y[part]["rows"].keys() != x[part]["rows"].keys() for y in S):
      raise ValueError(f"{part}: repeat row sets differ")
    rows = {}
    for key, row in x[part]["rows"].items():
      if any(diff._differs(row.get(f), y[part]["rows"][key].get(f)) for y in S for f in diff.WORK):
        raise ValueError(f"{part}/{key}: repeat work differs")
      rows[key] = {**_spread([y[part]["rows"][key]["now_ms"] for y in S]), "work": {f: row[f] for f in diff.WORK if f in row}}
    timing = {}
    for key in ("captured_ms", "untraced_ms", "gaps_ms"):
      if all(y[part]["tie"].get(key) is not None for y in S):
        timing[key] = _spread([y[part]["tie"][key] for y in S])
    parts[part] = {"rows": rows, "timing": timing}
  from boltbeam.collectors.providers import TRACES
  def capture_evidence(run):
    if sa["workload"] == "replay":
      captured = _read(run, "replay_capture.json")
      return {"index": captured, "turns": {t["file"]: _read(run, t["file"]) for t in captured.get("turns", []) if t.get("file")}}
    filename = {"prefill": "prefill_trace.json", "session": "session_trace.json"}.get(sa["workload"], TRACES[sa["engine"]])
    return _read(run, filename)
  sources = [{"run": r.name, "date": _date(r), "build": sa["build"],
              "timing_sha256": sha256_json(_read(r, "timing_trace.json")),
              "capture_sha256": sha256_json(capture_evidence(r))} for r in runs]
  if fixed is not None:
    if fixed.get("schema") != PLAN_SCHEMA or fixed.get("n") != len(runs):
      raise ValueError(f"the plan fixed n = {fixed.get('n')} before collection; {len(runs)} runs were given")
    if any(_when(s["date"]) < _when(fixed["planned_at"]) for s in sources):
      raise ValueError("a run predates the plan: n was not fixed before collection")
  return {"schema": SCHEMA, "scope": sa, "sources": sources, "parts": parts, **strength(len(runs)),
          "plan": fixed, "rule": RULE}


def save(runs:list[pathlib.Path], out:pathlib.Path, fixed:dict | None = None) -> dict:
  record = calibrate(runs, fixed)
  existing = read_json(out) if out.exists() else {}
  history = existing.get("calibrations", [existing] if existing.get("scope") else [])
  write_json(out, {"schema": SCHEMA, "calibrations": [*history, record]})
  return record


def find(run:pathlib.Path, part:str, *, record:dict | None = None) -> dict | None:
  record = record or _read(run, ARTIFACT)
  if not record:
    return None
  if record.get("schema") != SCHEMA:
    return None
  setup = scope(run)
  candidates = record.get("calibrations", [record])
  record = next((r for r in reversed(candidates) if r.get("scope") == setup), {})
  if not record:
    return None
  n = len(record.get("sources", []))
  if n < 2 or any(not r.get("date") or not r.get("build") for r in record["sources"]):
    return None
  result = record.get("parts", {}).get(part)
  if not result:
    return None
  for row in [*result.get("rows", {}).values(), *result.get("timing", {}).values()]:
    try:
      measured = _spread(_runs(row))
    except (KeyError, ValueError, TypeError):
      return None
    if len(measured["runs_ms"]) != n or any(row.get(k) != v for k, v in measured.items() if k in row):
      return None
    row.update(measured)  # a pair saved before n was recorded reads as its two runs
  return {**result, **strength(n), "plan": record.get("plan"), "sources": record["sources"]}


def pin_words(band:dict | None) -> str | None:
  """Why this band cannot justify an engine pin, or None when it can: n fixed by a plan before collection, at least
  PIN_RUNS, and every run after the plan."""
  if not band:
    return "no matching repeat band for the before build"
  fixed, n = band.get("plan") or {}, band.get("n", 0)
  if n < PIN_RUNS:
    return f"the before band has {n} runs; a pin change needs at least {PIN_RUNS}, fixed before collection"
  if not fixed or fixed.get("n") != n:
    return f"the before band's n = {n} was not fixed before collection (screen repeat-plan)"
  return None


def width(band:dict | None, key:str, *, work:dict | None = None) -> float | None:
  row = (band or {}).get("rows", {}).get(key)
  if row is None or work is not None and any(row.get("work", {}).get(f) != work.get(f) for f in row.get("work", {})):
    return None
  value = row.get("width_ms")
  return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def end_to_end_width(band:dict | None) -> float | None:
  timing = (band or {}).get("timing", {})
  values = [timing.get(k, {}).get("width_ms") for k in ("captured_ms", "untraced_ms")]
  return max(values) if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values) else None


def for_decode(run:pathlib.Path, trace:dict | None = None) -> dict | None:
  from boltbeam.collectors.providers import TRACES
  from boltbeam.workflow.screen import run_provider
  from boltbeam.workflow.tie_out import attended_context
  provider = run_provider(run)
  trace = trace or _read(run, TRACES[provider])
  return find(run, f"decode {attended_context(trace, provider)}")


def sides(run:pathlib.Path) -> dict:
  from boltbeam.workflow import diff
  from boltbeam.collectors import replay
  workload = load_manifest(run)["workload"]
  if workload == "prefill":
    return diff._prefill_sides(run)
  if workload == "decode":
    return diff._decode_side(run)
  if workload not in ("replay", "session"):
    raise ValueError(f"no repeat evidence reader for workload {workload}")
  if workload == "session":
    from boltbeam.collectors.session import TRACE as capture_file
    captures = _read(run, capture_file)
    timing = _read(run, "timing_trace.json")
    focus = timing.get("focus")
    result = {}
    for length, attr in captures.get("lengths", {}).items():
      if not attr.get("rows"):
        raise ValueError("cannot calibrate missing session rows")
      truth = next((r.get("turn_ms") for r in timing.get("rows", [])
                    if r.get("scope") == "whole_step" and r.get("context") == focus), None)
      result[f"prefill {length}"] = {
        "rows": {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in diff.WORK}} for r in attr["rows"]},
        "tie": {"captured_ms": attr.get("profiled_wall_ms"), "untraced_ms": truth,
                "gaps_ms": attr.get("gaps_ms")}}
    return result
  captures = _read(run, replay.CAPTURE)
  result = {}
  for turn in captures.get("turns", []):
    if turn.get("failed"):
      raise ValueError("cannot calibrate a failed replay capture")
    attr = _read(run, turn["file"]).get("lengths", {}).get(str(turn["prompt_n"]), {})
    if not attr.get("rows"):
      raise ValueError("cannot calibrate missing replay rows")
    result[f"turn {turn['i']}"] = {
      "rows": {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in diff.WORK}} for r in attr["rows"]},
      "tie": {"captured_ms": attr.get("profiled_wall_ms"), "untraced_ms": turn.get("server_ms"),
              "gaps_ms": attr.get("gaps_ms")}}
  return result
