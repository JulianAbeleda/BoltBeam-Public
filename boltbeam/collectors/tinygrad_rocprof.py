from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from boltbeam.artifacts.tinygrad_rocprof import timing_trace_from_tinygrad_rocprof_csv
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.target.tinygrad_root import resolve_tinygrad_root
from boltbeam.workflow.common import load_manifest, run_dir
from boltbeam.core.canonical import pretty_json
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


@dataclass(frozen=True)
class TinygradRocprofCapture:
  model:str
  model_id:str
  target_id:str
  workload:str
  context:int
  max_context:int
  out:pathlib.Path
  trace_dir:pathlib.Path
  prefix:str
  tinygrad_root:pathlib.Path
  python:str
  rocprofv3:str
  weight_inventory:pathlib.Path | None
  prefill_chunked:str
  env:dict[str, str]
  prefill_k:int | None = None
  prefill_warmups:int | None = None
  prefill_rounds:int | None = None
  prefill_start_positions:str | None = None
  prefill_whole_lengths:str | None = None
  provider_id:str = "tinygrad/rocprofv3"
  peak_gbs:float = DEFAULT_PEAK_MEM_GBS


def tinygrad_authority_command(cfg:TinygradRocprofCapture) -> list[str]:
  """Build the bounded command shared by profile-events and rocprof collectors."""
  cmd = [cfg.python, "extra/qk/bench.py", "--model", cfg.model]
  if cfg.workload == "prefill":
    # Roofline attribution must observe the same production authority as the
    # prefill_whole_synced benchmark.  ``smoke`` is a quick reduced profile
    # and can report materially higher tok/s than the production protocol.
    bounded = []
    for attr, flag in (("prefill_k", "--prefill-K"), ("prefill_warmups", "--prefill-warmups"),
                       ("prefill_rounds", "--prefill-rounds"),
                       ("prefill_start_positions", "--prefill-start-positions"),
                       ("prefill_whole_lengths", "--prefill-whole-lengths")):
      value = getattr(cfg, attr, None)
      if value is not None: bounded.extend([flag, str(value)])
    return cmd + ["--prefill", "--prefill-mode", "authority", *bounded, "--prefill-no-artifact"]
  if cfg.workload == "decode":
    return cmd + ["--decode", "--decode-ckpts", str(cfg.context), "--decode-nmeas", "4",
                  "--decode-max-context", str(cfg.max_context)]
  raise ValueError(f"unsupported tinygrad workload {cfg.workload!r}")


def _parse_env_overrides(raw:list[str] | None) -> dict[str, str]:
  out: dict[str, str] = {}
  for item in raw or []:
    if "=" not in item:
      raise ValueError(f"invalid env override {item!r}; expected KEY=VALUE")
    k, v = item.split("=", 1)
    out[k] = v
  return out


def _resolve(args:argparse.Namespace) -> TinygradRocprofCapture:
  run_manifest: dict[str, Any] = {}
  model = args.model
  out = pathlib.Path(args.out).expanduser() if args.out else None
  weight_inventory = pathlib.Path(args.weight_inventory).expanduser() if args.weight_inventory else None
  if args.run:
    rp = run_dir(args.run)
    run_manifest = load_manifest(rp)
    model = model or run_manifest.get("model_path")
    out = out or (rp / "timing_trace.json")
    if weight_inventory is None and (rp / "weight_inventory.json").exists():
      weight_inventory = rp / "weight_inventory.json"
  if not model:
    raise ValueError("--model is required unless --run supplies one")
  if out is None:
    raise ValueError("--out is required unless --run is supplied")
  model_id = args.model_id or run_manifest.get("model_id") or pathlib.Path(model).stem
  trace_dir = pathlib.Path(args.trace_dir).expanduser() if args.trace_dir else out.with_suffix(".rocprof.d")
  prefix = args.prefix or f"{model_id}_pp{args.context}"
  tinygrad_root = resolve_tinygrad_root(args.tinygrad_root, require_checkout=True)
  env = os.environ.copy()
  env.update({
    "DEV": "AMD",
    "PYTHONPATH": str(tinygrad_root),
    "PREFILL_V2": "1",
    "PREFILL_CHUNKED": str(args.prefill_chunked),
    "PREFILL_GRAPH_GEMM": "1",
    "PROFILE": "0",
  })
  env.update(_parse_env_overrides(args.env))
  return TinygradRocprofCapture(
    model=str(model),
    model_id=model_id,
    target_id=args.target_id or run_manifest.get("target_id") or "amd_gfx1100",
    workload=args.workload or run_manifest.get("workload") or "prefill",
    context=args.context,
    max_context=args.max_context,
    out=out,
    trace_dir=trace_dir,
    prefix=prefix,
    tinygrad_root=tinygrad_root,
    python=args.python,
    rocprofv3=args.rocprofv3,
    weight_inventory=weight_inventory,
    prefill_chunked=str(args.prefill_chunked),
    prefill_k=getattr(args, "prefill_k", None),
    prefill_warmups=getattr(args, "prefill_warmups", None),
    prefill_rounds=getattr(args, "prefill_rounds", None),
    prefill_start_positions=getattr(args, "prefill_start_positions", None),
    prefill_whole_lengths=getattr(args, "prefill_whole_lengths", None),
    env=env,
    provider_id=args.provider_id,
    peak_gbs=args.peak_gbs,
  )


def capture_tinygrad_rocprof(cfg:TinygradRocprofCapture) -> dict[str, Any]:
  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  wall_json = cfg.trace_dir / f"{cfg.prefix}_tinygrad_wall.json"
  kernel_stats = cfg.trace_dir / f"{cfg.prefix}_kernel_stats.csv"
  memory_copy_stats = cfg.trace_dir / f"{cfg.prefix}_memory_copy_stats.csv"
  tinygrad_cmd = tinygrad_authority_command(cfg)
  cmd = [
    cfg.rocprofv3, "--kernel-trace", "--memory-copy-trace", "--stats", "-f", "csv",
    "-d", str(cfg.trace_dir), "-o", cfg.prefix, "--",
  ] + tinygrad_cmd
  proc = subprocess.run(cmd, cwd=str(cfg.tinygrad_root), env=cfg.env)
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
        "stage": "rocprofv3_tinygrad_command",
        "exit_code": proc.returncode,
      }],
      "aux_sources": {"trace_dir": str(cfg.trace_dir), "wall_json": str(wall_json)},
    }
  if not kernel_stats.exists():
    raise FileNotFoundError(f"expected kernel stats not found: {kernel_stats}")
  return timing_trace_from_tinygrad_rocprof_csv(
    kernel_stats,
    model_id=cfg.model_id,
    target_id=cfg.target_id,
    workload=cfg.workload,
    context=cfg.context,
    provider_id=cfg.provider_id,
    peak_gbs=cfg.peak_gbs,
    tinygrad_trace_json=wall_json if wall_json.exists() else None,
    weight_inventory=cfg.weight_inventory,
    memory_copy_csv=memory_copy_stats if memory_copy_stats.exists() else None,
  )


def _write(obj:dict[str, Any], out:pathlib.Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(obj))


def main(argv:list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description="Collect tinygrad prefill rocprofv3 data and emit boltbeam.timing_trace.v1")
  ap.add_argument("--run", default=None, help="optional BoltBeam run directory")
  ap.add_argument("--model", default=None, help="GGUF model path; required unless --run supplies one")
  ap.add_argument("--model-id", default=None)
  ap.add_argument("--target-id", default=None)
  ap.add_argument("--workload", default=None, choices=["decode", "prefill"])
  ap.add_argument("--context", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--out", default=None, help="write boltbeam.timing_trace.v1 here; defaults to <run>/timing_trace.json")
  ap.add_argument("--trace-dir", default=None, help="directory for raw rocprof CSV and tinygrad wall JSON")
  ap.add_argument("--prefix", default=None)
  ap.add_argument("--tinygrad-root", default=None,
                  help="tinygrad checkout (default: TINYGRAD_ROOT or recognized sibling)")
  ap.add_argument("--python", default=".venv/bin/python")
  ap.add_argument("--rocprofv3", default="rocprofv3")
  ap.add_argument("--weight-inventory", default=None)
  ap.add_argument("--prefill-chunked", default="1")
  ap.add_argument("--prefill-K", dest="prefill_k", type=int, default=None,
                  help="optional bounded authority burst count")
  ap.add_argument("--prefill-warmups", dest="prefill_warmups", type=int, default=None)
  ap.add_argument("--prefill-rounds", dest="prefill_rounds", type=int, default=None)
  ap.add_argument("--prefill-start-positions", dest="prefill_start_positions", default=None)
  ap.add_argument("--prefill-whole-lengths", dest="prefill_whole_lengths", default=None)
  ap.add_argument("--provider-id", default="tinygrad/rocprofv3")
  ap.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  ap.add_argument("--env", action="append", default=[], help="extra tinygrad env override KEY=VALUE")
  args = ap.parse_args(argv)
  try:
    cfg = _resolve(args)
    trace = capture_tinygrad_rocprof(cfg)
    _write(trace, cfg.out)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"tinygrad_rocprof collector: {exc}\n")
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
