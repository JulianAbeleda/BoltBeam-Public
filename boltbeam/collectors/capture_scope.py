"""A run's exact capture setup (scope) and the repeat band recorded for it (find, for_decode, a row's width): measured
uncertainty for one setup, never a chip-wide default. workflow/repeat_bands.py calibrates and saves the bands."""
from __future__ import annotations

import math
import pathlib

from boltbeam.artifacts.base import sha256_json
from boltbeam.artifacts.run_io import load_manifest, optional, read_json

SCHEMA = "boltbeam.repeat_bands.v1"
ARTIFACT = "repeat_bands.json"
PIN_RUNS = 4  # runs a band needs, fixed before collection, before it may justify an engine pin; fewer is a weak verdict


def strength(n:int) -> dict:
  """The band's n, its false-move rate per row and whether its verdicts are weak (fewer runs than PIN_RUNS)."""
  return {"n": n, "false_move_rate": 2 / (n + 1), "weak": n < PIN_RUNS}


def scope(run:pathlib.Path) -> dict:
  from boltbeam.collectors import providers
  manifest = load_manifest(run)
  profile = optional(run, "model_profile.json")
  profile = {k: v for k, v in profile.items() if k not in ("source", "model_id")}
  request = optional(run, "trace_request.json")
  timing = optional(run, "timing_trace.json")
  machine = optional(run, "machine_facts.json")
  status = optional(run, "measure_status.json")
  capture_file = {"prefill": "prefill_trace.json", "session": "session_trace.json",
                  "replay": "replay_capture.json"}.get(manifest.get("workload"), providers.TRACES[providers.run_provider(run)])
  capture = optional(run, capture_file)
  whole = next((r for r in capture.get("rows", []) if r.get("scope") == "whole_step"), {})
  labeled = optional(run, "decode_capture.json") if not capture else {}
  if labeled.get("schema") == "boltbeam.decode_capture.v1" and labeled.get("contexts"):
    # a labeled vLLM capture (collectors/vllm_semantic_decode) holds one capture per context, not the engine's trace:
    # its window is the contexts captured, each with its tokens and batch (a 5090's repeat band, 2026-11-09)
    caps = [labeled["contexts"][k] for k in sorted(labeled["contexts"], key=int)]
    whole = {"context": [c.get("context") for c in caps] if len(caps) > 1 else caps[0].get("context"),
             "decode_tokens": caps[0].get("length"), "batch": (caps[0].get("timing_window") or {}).get("batch")}
  methods = {length: {k: attr[k] for k in ("method", "formats", "timing_window") if k in attr}
             for length, attr in capture.get("lengths", {}).items()}
  result = {"target": manifest.get("target_id"), "devices": sorted(machine.get("uuids") or []),
          "engine": providers.run_provider(run), "build": providers.run_build(run) or (manifest.get("engine") or {}).get("build"),
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
  if env := optional(run, "workload_profile.json").get("engine_env"):
    result["engine_env"] = dict(sorted(env.items()))  # the engine's environment is part of the setup; absent when none, so older bands match
  from boltbeam.collectors.gpu_telemetry import measurement_clock_scope
  if (clock := measurement_clock_scope(run)) is not None:
    result["measurement_clock"] = clock
  identities = {}
  required = {key + "_sha256" for key in ("helper_source", "helper_binary", "collector_source",
              "cupti_library", "nvperf_host", "nvperf_target", "symbol_decoder")}
  for path in sorted(run.rglob("capture_identity.json")):
    identity = read_json(path)
    stable = {k: identity[k] for k in required | {"method", "cupti_version", "compiler_flags"} if k in identity}
    valid = stable.get("method") == "cupti-hes" and required <= stable.keys() and all(
        isinstance(stable[k], str) and len(stable[k]) == 64 and all(c in "0123456789abcdef" for c in stable[k])
        for k in required)
    valid = valid and type(stable.get("cupti_version")) is int and stable["cupti_version"] > 0
    valid = valid and isinstance(stable.get("compiler_flags"), list) and bool(stable["compiler_flags"])
    if not valid or not all(isinstance(flag, str) for flag in stable["compiler_flags"]):
      raise ValueError("invalid retained capture identity")
    identities[sha256_json(stable)] = stable
  if identities:
    result["capture_identities"] = [identities[k] for k in sorted(identities)]
  hes = capture.get("capture", {}).get("method") == "cupti-hes" or any(attr.get("method") == "cupti-hes"
        for attr in capture.get("lengths", {}).values())
  if hes and not identities:
    raise ValueError("cupti-hes requires retained capture identity")
  return result


def spread(xs:list[float]) -> dict:
  if len(xs) < 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in xs):
    raise ValueError("repeat measurements must be at least two finite nonnegative times")
  mean = sum(xs) / len(xs)
  return {"runs_ms": list(xs), "low_ms": min(xs), "high_ms": max(xs), "reference_ms": mean, "width_ms": max(xs) - min(xs),
          "width_pct": 100 * (max(xs) - min(xs)) / mean if mean else None}


def row_runs(row:dict) -> list:
  return row.get("runs_ms") or [row.get("before_ms"), row.get("after_ms")]  # a band saved as a pair: n = 2


def find(run:pathlib.Path, part:str, *, record:dict | None = None) -> dict | None:
  record = record or optional(run, ARTIFACT)
  if not record:
    return None
  if record.get("schema") != SCHEMA:
    return None
  try:
    setup = scope(run)
  except (ValueError, TypeError, KeyError):
    return None
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
      measured = spread(row_runs(row))
    except (KeyError, ValueError, TypeError):
      return None
    if len(measured["runs_ms"]) != n or any(row.get(k) != v for k, v in measured.items() if k in row):
      return None
    row.update(measured)  # a pair saved before n was recorded reads as its two runs
  return {**result, **strength(n), "plan": record.get("plan"), "sources": record["sources"]}


def width(band:dict | None, key:str, *, work:dict | None = None) -> float | None:
  row = (band or {}).get("rows", {}).get(key)
  if row is None or work is not None and any(row.get("work", {}).get(f) != work.get(f) for f in row.get("work", {})):
    return None
  value = row.get("width_ms")
  return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def end_to_end_width(band:dict | None, *, captured_key:str = "captured_ms") -> float | None:
  timing = (band or {}).get("timing", {})
  values = [timing.get(k, {}).get("width_ms") for k in (captured_key, "untraced_ms")]
  return max(values) if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values) else None


def for_decode(run:pathlib.Path, trace:dict | None = None) -> dict | None:
  from boltbeam.collectors.capture_settings import attended_context
  from boltbeam.collectors.providers import TRACES, run_provider
  provider = run_provider(run)
  trace = trace or optional(run, TRACES[provider])
  return find(run, f"decode {attended_context(trace, provider)}")
