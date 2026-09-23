from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.trace.timing import build_timing_profile
from boltbeam.vocab import SCHEMA_HW_TRACE, SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, read_json, run_dir, update_manifest, write_json

TIMING_ARTIFACTS = ("timing_trace.json", "timing_profile.json")


def ingest_timing_run(run:str | pathlib.Path, trace_path:str | pathlib.Path) -> dict[str, Any]:
  out = run_dir(run)
  manifest = load_manifest(out)
  trace_p = pathlib.Path(trace_path).expanduser()
  trace = read_json(trace_p)
  for key, manifest_key in (("model_id", "model_id"), ("target_id", "target_id"), ("workload", "workload")):
    if trace.get(key) != manifest.get(manifest_key):
      raise ValueError(f"timing trace {key} mismatch ({trace.get(key)!r} != {manifest.get(manifest_key)!r})")
  request = read_json(out / "trace_request.json") if (out / "trace_request.json").exists() else None
  profile = build_timing_profile(trace, request=request, source_path=trace_p)
  raw_name = "hw_trace.json" if trace.get("schema") == SCHEMA_HW_TRACE else "timing_trace.json"
  if trace.get("schema") not in {SCHEMA_TIMING_TRACE, SCHEMA_HW_TRACE}:
    raise ValueError(f"timing trace schema mismatch ({trace.get('schema')!r})")
  write_json(out / raw_name, trace)
  write_json(out / "timing_profile.json", profile)
  artifacts = [raw_name, "timing_profile.json"]
  if raw_name == "hw_trace.json" and (out / "timing_trace.json").exists():
    artifacts.insert(0, "timing_trace.json")
  update_manifest(out, stage="ingest_timing", artifacts=artifacts)
  return profile
