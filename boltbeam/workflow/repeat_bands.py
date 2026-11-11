"""Measured uncertainty for an exact run setup, never a chip-wide default: the plan that fixes n, the calibration that
writes a band (repeat_bands.json) and its summary. A run finds its band in collectors/capture_scope.py."""
from __future__ import annotations

import datetime
import statistics
import pathlib

from boltbeam.artifacts.base import sha256_json
from boltbeam.artifacts.run_io import load_manifest, optional, read_json, write_json
from boltbeam.collectors.capture_scope import ARTIFACT, PIN_RUNS, SCHEMA, find, scope, spread, strength

PLAN_SCHEMA = "boltbeam.repeat_plan.v1"
PLAN = "repeat_plan.json"  # the plan a repeated pipeline (screen pipeline --repeats) fixes in its first run
RULE = ("a row moved when the after value lies outside the range of n independent before runs of its exact setup; "
        "with no change, that happens with probability 2/(n+1) per row (exchangeable runs)")


def plan(n:int, out:pathlib.Path) -> dict:
  """Fix n before any repeat is collected: the plan's time precedes every run a band built on it may use."""
  if n < 2:
    raise ValueError("a band needs at least two runs")
  record = {"schema": PLAN_SCHEMA, "n": n, "planned_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
  write_json(out, record)
  return record


def _date(run:pathlib.Path) -> str:
  manifest = load_manifest(run)
  date = (manifest.get("toolchain") or {}).get("recorded_at") or optional(run, "machine_facts.json").get("measured_at")
  if not date:
    raise ValueError(f"{run.name}: no recorded measurement date; file modification time is not a run date")
  datetime.date.fromisoformat(str(date)[:10])
  return str(date)


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
    diff.refuse_unless_same([(k, sa.get(k), sb.get(k)) for k in sa.keys() | sb.keys()])
  if not sa["model"]["profile"] or not sa["model"]["path"] or not sa["request"].get("contexts"):
    raise ValueError("repeat scope needs model profile, model path and requested contexts")
  if not sa["build"] or sa["build"] == "unknown":
    raise ValueError("repeat scope needs a recorded engine build")
  if sa["workload"] == "decode":
    base = sa["capture"]["window"].get("context")
    if base is None:
      raise ValueError(f"{a.name}: its decode capture names no context, so nothing says which setup it calibrates")
    if any(c not in sa["request"]["contexts"] for c in (base if isinstance(base, list) else [base])):
      raise ValueError(f"capture context {base} is not among the requested {sa['request']['contexts']}; this pair "
                       "cannot calibrate that setup")
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
      rows[key] = {**spread([y[part]["rows"][key]["now_ms"] for y in S]), "work": {f: row[f] for f in diff.WORK if f in row}}
    timing = {}
    for key in ("captured_ms", "engine_elapsed_ms", "untraced_ms", "gaps_ms"):
      if all(y[part]["tie"].get(key) is not None for y in S):
        timing[key] = spread([y[part]["tie"][key] for y in S])
    parts[part] = {"rows": rows, "timing": timing}
  from boltbeam.collectors.providers import TRACES
  def capture_evidence(run):
    if sa["workload"] == "replay":
      captured = optional(run, "replay_capture.json")
      return {"index": captured, "turns": {t["file"]: optional(run, t["file"]) for t in captured.get("turns", []) if t.get("file")}}
    filename = {"prefill": "prefill_trace.json", "session": "session_trace.json"}.get(sa["workload"], TRACES[sa["engine"]])
    return optional(run, filename)
  sources = [{"run": r.name, "date": _date(r), "build": sa["build"],
              "timing_sha256": sha256_json(optional(r, "timing_trace.json")),
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


def summary(run:pathlib.Path) -> dict | None:
  """This run's repeat band as the screen shows it: n, the chance a row reads moved with no change, why it cannot
  justify an engine pin (pin_words), and per part the untraced time over the n runs: its median, range and spread
  ((high - low) / median). None without a band for this exact setup."""
  record = optional(run, ARTIFACT)
  names = sorted({p for c in record.get("calibrations", [record] if record.get("scope") else []) for p in c.get("parts", {})})
  bands = {p: band for p in names if (band := find(run, p, record=record))}
  if not bands:
    return None
  lines = []
  for part, band in bands.items():
    if t := band.get("timing", {}).get("untraced_ms"):
      med = statistics.median(t["runs_ms"])
      lines.append(f"{part}: median {med:.3f} ms, {t['low_ms']:.3f} to {t['high_ms']:.3f} ms over {band['n']} runs "
                   f"(spread {100 * (t['high_ms'] - t['low_ms']) / med:.1f}%)")
  first = next(iter(bands.values()))
  return {**strength(first["n"]), "pin": pin_words(first), "lines": lines}


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
    captures = optional(run, capture_file)
    timing = optional(run, "timing_trace.json")
    focus = timing.get("focus")
    result = {}
    for length, attr in captures.get("lengths", {}).items():
      if not attr.get("rows"):
        raise ValueError("cannot calibrate missing session rows")
      truth = next((r.get("qualification_ms") or r.get("turn_ms") for r in timing.get("rows", [])
                    if r.get("scope") == "whole_step" and r.get("context") == focus), None)
      result[f"prefill {length}"] = {
        "rows": {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in diff.WORK}} for r in attr["rows"]},
        "tie": {"captured_ms": attr.get("profiled_wall_ms"), "engine_elapsed_ms": attr.get("measured_ms"), "untraced_ms": truth,
                "gaps_ms": attr.get("gaps_ms")}}
    return result
  captures = optional(run, replay.CAPTURE)
  result = {}
  for turn in captures.get("turns", []):
    if turn.get("failed"):
      raise ValueError("cannot calibrate a failed replay capture")
    attr = optional(run, turn["file"]).get("lengths", {}).get(str(turn["prompt_n"]), {})
    if not attr.get("rows"):
      raise ValueError("cannot calibrate missing replay rows")
    result[f"turn {turn['i']}"] = {
      "rows": {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in diff.WORK}} for r in attr["rows"]},
      "tie": {"captured_ms": attr.get("profiled_wall_ms"), "engine_elapsed_ms": attr.get("measured_ms"), "untraced_ms": turn.get("server_ms"),
              "gaps_ms": attr.get("gaps_ms")}}
  return result
