from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.quantization.quant_gemv import build_primitive_profile
from boltbeam.workflow.common import load_manifest, read_json, run_dir, update_manifest, write_json

PROBE_ARTIFACTS = ("primitive_profile.json",)


def ingest_probe_run(run:str | pathlib.Path, evidence_path:str | pathlib.Path) -> dict[str, Any]:
  out = run_dir(run)
  manifest = load_manifest(out)
  evidence_p = pathlib.Path(evidence_path).expanduser()
  evidence = read_json(evidence_p)
  for key, manifest_key in (("model_id", "model_id"), ("target_id", "target_id"), ("workload", "workload")):
    if evidence.get(key) != manifest.get(manifest_key):
      raise ValueError(f"probe evidence {key} mismatch ({evidence.get(key)!r} != {manifest.get(manifest_key)!r})")
  request = read_json(out / "probe_request.json") if (out / "probe_request.json").exists() else None
  profile = build_primitive_profile(evidence, request=request, source_path=evidence_p)
  write_json(out / "primitive_profile.json", profile)
  update_manifest(out, stage="ingest_probe", artifacts=list(PROBE_ARTIFACTS))
  return profile
