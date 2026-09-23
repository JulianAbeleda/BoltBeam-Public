from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

from boltbeam.artifacts.tinygrad_profile_events import reconcile_authority_wall, timing_trace_from_tinygrad_profile_events
from boltbeam.collectors.tinygrad_rocprof import TinygradRocprofCapture, tinygrad_authority_command
from boltbeam.vocab import SCHEMA_TIMING_TRACE


def _profile_temp_path() -> pathlib.Path:
  return pathlib.Path(tempfile.gettempdir()) / f"profile.pkl.{pathlib.Path.home().name}"


_authority_command = tinygrad_authority_command


def capture_tinygrad_profile_events(cfg:TinygradRocprofCapture) -> dict[str, Any]:
  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  raw_profile = cfg.trace_dir / f"{cfg.prefix}_profile.pkl"
  tmp_profile = _profile_temp_path()
  if tmp_profile.exists():
    tmp_profile.unlink()

  env = dict(cfg.env)
  env["PROFILE"] = "1"
  env.setdefault("VIZ", "0")
  cmd = _authority_command(cfg)
  proc = subprocess.run(cmd, cwd=str(cfg.tinygrad_root), env=env, capture_output=True, text=True)
  if proc.returncode != 0:
    return {
      "schema": SCHEMA_TIMING_TRACE,
      "model_id": cfg.model_id,
      "target_id": cfg.target_id,
      "workload": cfg.workload,
      "provider_id": cfg.provider_id,
      "timing_source": "collector_failure",
      "contexts": [cfg.context],
      "rows": [{
        "scope": "failure",
        "context": cfg.context,
        "stage": "tinygrad_profile_events_command",
        "exit_code": proc.returncode,
      }],
      "aux_sources": {"trace_dir": str(cfg.trace_dir), "authority_command": cmd},
    }
  if not tmp_profile.exists():
    raise FileNotFoundError(f"expected tinygrad PROFILE pickle not found: {tmp_profile}")
  shutil.copy2(tmp_profile, raw_profile)
  trace = timing_trace_from_tinygrad_profile_events(
    raw_profile,
    model_id=cfg.model_id,
    target_id=cfg.target_id,
    workload=cfg.workload,
    context=cfg.context,
    provider_id=cfg.provider_id,
    peak_gbs=cfg.peak_gbs,
    tinygrad_root=cfg.tinygrad_root,
    weight_inventory=cfg.weight_inventory,
  )
  return reconcile_authority_wall(trace, proc.stdout or "", context=cfg.context)
