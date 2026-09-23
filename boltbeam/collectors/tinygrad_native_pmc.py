from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

from boltbeam.collectors.tinygrad_rocprof import TinygradRocprofCapture
from boltbeam.vocab import SCHEMA_TIMING_TRACE


def _failure(cfg:TinygradRocprofCapture, exit_code:int, hw_json:pathlib.Path) -> dict[str, Any]:
  return {
    "schema": SCHEMA_TIMING_TRACE,
    "model_id": cfg.model_id,
    "target_id": cfg.target_id,
    "workload": cfg.workload,
    "provider_id": cfg.provider_id,
    "timing_source": "collector_failure",
    "contexts": [cfg.context],
    "rows": [{"scope": "failure", "context": cfg.context, "stage": "tinygrad_native_pmc_command",
              "exit_code": exit_code}],
    "aux_sources": {"trace_dir": str(cfg.trace_dir), "hw_json": str(hw_json)},
  }


def capture_tinygrad_native_pmc(cfg:TinygradRocprofCapture) -> dict[str, Any]:
  """Run the tinygrad-native PMC trace (extra/qk/prefill/prefill_boltbeam_trace.py --hw-trace).

  Reads AMD hardware counters through tinygrad's own KFD interface (no rocprofv3), so it works
  with the DEV=AMD backend. The script emits boltbeam.hw_trace.v1 directly with counters on rows.
  """
  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  # The subprocess runs with cwd=tinygrad_root, so --out must be absolute or it lands under the
  # tinygrad tree instead of here.
  hw_json = (cfg.trace_dir / f"{cfg.prefix}_native_pmc.json").resolve()

  env = dict(cfg.env)
  # pmc_enabled requires PROFILE>0 and PMC>0; the shared tinygrad env pins PROFILE=0, so set both here.
  env["PMC"] = "1"
  env["PROFILE"] = "1"
  cmd = [
    cfg.python, "extra/qk/prefill/prefill_boltbeam_trace.py",
    "--mode", "full", "--hw-trace",
    "--model", cfg.model,
    "--model-id", cfg.model_id,
    "--context", str(cfg.context),
    "--max-context", str(cfg.max_context),
    "--peak-gbs", str(cfg.peak_gbs),
    "--out", str(hw_json),
  ]
  proc = subprocess.run(cmd, cwd=str(cfg.tinygrad_root), env=env)
  if proc.returncode != 0:
    return _failure(cfg, proc.returncode, hw_json)
  if not hw_json.exists():
    raise FileNotFoundError(f"expected native PMC trace not found: {hw_json}")
  trace = json.loads(hw_json.read_text(encoding="utf-8"))
  trace["provider_id"] = cfg.provider_id or trace.get("provider_id")
  trace["source_path"] = str(hw_json)
  trace.setdefault("aux_sources", {})["native_pmc_json"] = str(hw_json)
  return trace
