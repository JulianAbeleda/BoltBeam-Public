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


def _spread(a:float, b:float) -> dict:
  if not all(math.isfinite(x) and x >= 0 for x in (a, b)):
    raise ValueError("repeat measurements must be finite nonnegative times")
  mean = (a + b) / 2
  return {"before_ms": a, "after_ms": b, "reference_ms": mean, "width_ms": abs(b - a),
          "width_pct": 100 * abs(b - a) / mean if mean else None}


def calibrate(a:pathlib.Path, b:pathlib.Path) -> dict:
  """Absolute row spread and its relative equivalent, from two independent repeats of one setup."""
  from boltbeam.workflow import diff
  if a.resolve() == b.resolve():
    raise ValueError("a repeat pair needs two distinct runs")
  sa, sb = scope(a), scope(b)
  diff.refuse_unless_same([(k, sa[k], sb[k]) for k in sa])
  if not sa["model"]["profile"] or not sa["model"]["path"] or not sa["request"].get("contexts"):
    raise ValueError("repeat scope needs model profile, model path and requested contexts")
  if not sa["build"] or sa["build"] == "unknown":
    raise ValueError("repeat scope needs a recorded engine build")
  if sa["workload"] == "decode":
    base = sa["capture"]["window"].get("context")
    if base not in sa["request"]["contexts"]:
      raise ValueError("capture context differs from the requested setup; this pair cannot calibrate that setup")
  A, B = sides(a), sides(b)
  if not A or A.keys() != B.keys():
    raise ValueError("repeat captures must cover the same nonempty workload parts")
  parts = {}
  for part in A:
    x, y = A[part], B[part]
    if x["rows"].keys() != y["rows"].keys():
      raise ValueError(f"{part}: repeat row sets differ")
    rows = {}
    for key, row in x["rows"].items():
      if any(diff._differs(row.get(f), y["rows"][key].get(f)) for f in diff.WORK):
        raise ValueError(f"{part}/{key}: repeat work differs")
      rows[key] = {**_spread(row["now_ms"], y["rows"][key]["now_ms"]),
                   "work": {f: row[f] for f in diff.WORK if f in row}}
    timing = {}
    for key in ("captured_ms", "untraced_ms", "gaps_ms"):
      if x["tie"].get(key) is not None and y["tie"].get(key) is not None:
        timing[key] = _spread(x["tie"][key], y["tie"][key])
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
              "capture_sha256": sha256_json(capture_evidence(r))} for r in (a, b)]
  return {"schema": SCHEMA, "scope": sa, "sources": sources, "parts": parts,
          "rule": "absolute difference between two independent repeats; no statistical confidence claim or constant floor"}


def save(a:pathlib.Path, b:pathlib.Path, out:pathlib.Path) -> dict:
  record = calibrate(a, b)
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
  if len(record.get("sources", [])) != 2 or any(not r.get("date") or not r.get("build") for r in record["sources"]):
    return None
  result = record.get("parts", {}).get(part)
  if not result:
    return None
  for row in [*result.get("rows", {}).values(), *result.get("timing", {}).values()]:
    try:
      measured = _spread(row["before_ms"], row["after_ms"])
    except (KeyError, ValueError, TypeError):
      return None
    if any(row.get(k) != v for k, v in measured.items()):
      return None
  return result


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
