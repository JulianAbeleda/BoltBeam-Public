from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from boltbeam.artifacts.llama_rocprof import timing_trace_from_rocprof_csv
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, run_dir
from boltbeam.core.canonical import pretty_json
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


@dataclass(frozen=True)
class LlamaRocprofCapture:
  model:str
  model_id:str
  target_id:str
  workload:str
  context:int
  out:pathlib.Path
  trace_dir:pathlib.Path
  prefix:str
  llama_bench:str
  rocprofv3:str
  weight_inventory:pathlib.Path | None
  provider_id:str = "llama.cpp/rocprofv3"
  peak_gbs:float = DEFAULT_PEAK_MEM_GBS


def _resolve(args:argparse.Namespace) -> LlamaRocprofCapture:
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
  workload = args.workload or run_manifest.get("workload") or "prefill"
  return LlamaRocprofCapture(
    model=str(model),
    model_id=model_id,
    target_id=args.target_id or run_manifest.get("target_id") or "amd_gfx1100",
    workload=workload,
    context=args.context,
    out=out,
    trace_dir=trace_dir,
    prefix=args.prefix or f"{model_id}_{'tg' if workload == 'decode' else 'pp'}{args.context}",
    llama_bench=args.llama_bench,
    rocprofv3=args.rocprofv3,
    weight_inventory=weight_inventory,
    provider_id=args.provider_id,
    peak_gbs=args.peak_gbs,
  )


def capture_llama_rocprof(cfg:LlamaRocprofCapture) -> dict[str, Any]:
  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  bench_json = cfg.trace_dir / f"{cfg.prefix}_llama_bench.json"
  kernel_stats = cfg.trace_dir / f"{cfg.prefix}_kernel_stats.csv"
  memory_copy_stats = cfg.trace_dir / f"{cfg.prefix}_memory_copy_stats.csv"
  workload_args = (["-p", "0", "-n", "10", "-d", str(cfg.context)] if cfg.workload == "decode" else
                   ["-p", str(cfg.context), "-n", "0"])
  cmd = [
    cfg.rocprofv3, "--kernel-trace", "--memory-copy-trace", "--stats", "-f", "csv",
    "-d", str(cfg.trace_dir), "-o", cfg.prefix, "--", cfg.llama_bench,
    "-m", cfg.model, "-ngl", "99", *workload_args, "-r", "1", "-o", "json",
  ]
  with bench_json.open("w") as f:
    proc = subprocess.run(cmd, stdout=f)
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
        "stage": "rocprofv3_llama_bench_command",
        "exit_code": proc.returncode,
      }],
      "aux_sources": {"trace_dir": str(cfg.trace_dir), "llama_bench_json": str(bench_json)},
    }
  if not kernel_stats.exists():
    raise FileNotFoundError(f"expected kernel stats not found: {kernel_stats}")
  return timing_trace_from_rocprof_csv(
    kernel_stats,
    model_id=cfg.model_id,
    target_id=cfg.target_id,
    workload=cfg.workload,
    context=cfg.context,
    provider_id=cfg.provider_id,
    peak_gbs=cfg.peak_gbs,
    llama_bench_json=bench_json,
    weight_inventory=cfg.weight_inventory,
    memory_copy_csv=memory_copy_stats if memory_copy_stats.exists() else None,
  )


def _write(obj:dict[str, Any], out:pathlib.Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(obj))


def main(argv:list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description="Collect llama.cpp rocprofv3 data and emit boltbeam.timing_trace.v1")
  ap.add_argument("--run", default=None, help="optional BoltBeam run directory")
  ap.add_argument("--model", default=None, help="GGUF model path; required unless --run supplies one")
  ap.add_argument("--model-id", default=None)
  ap.add_argument("--target-id", default=None)
  ap.add_argument("--workload", default=None, choices=["decode", "prefill"])
  ap.add_argument("--context", type=int, default=512)
  ap.add_argument("--out", default=None, help="write boltbeam.timing_trace.v1 here; defaults to <run>/timing_trace.json")
  ap.add_argument("--trace-dir", default=None, help="directory for raw rocprof CSV and llama-bench JSON")
  ap.add_argument("--prefix", default=None)
  ap.add_argument("--llama-bench", default="llama-bench")
  ap.add_argument("--rocprofv3", default="rocprofv3")
  ap.add_argument("--weight-inventory", default=None)
  ap.add_argument("--provider-id", default="llama.cpp/rocprofv3")
  ap.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  args = ap.parse_args(argv)
  try:
    cfg = _resolve(args)
    trace = capture_llama_rocprof(cfg)
    _write(trace, cfg.out)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"llama_rocprof collector: {exc}\n")
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
