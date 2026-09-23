from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any

from boltbeam.profiler.importers.ncu import import_profiler_ncu_csv
from boltbeam.workflow.common import load_manifest, run_dir
from boltbeam.core.canonical import pretty_json


@dataclass(frozen=True)
class LlamaNcuCapture:
  model: str
  model_id: str
  target_id: str
  workload: str
  context: int
  out: pathlib.Path
  trace_dir: pathlib.Path
  prefix: str
  llama_bench: str
  ncu: str = "ncu"
  ncu_set: str = "basic"
  kernel_name: str | None = None
  launch_count: int = 5
  provider_id: str = "nvidia/ncu"


def _default_kernel_filter(workload: str) -> str:
  if workload == "decode":
    return "regex:^mul_mat_vec_q"
  return "regex:^(quantize_mmq_q8_1|mul_mat_q|mul_mat_q_stream_k_fixup|flash_attn)"


def _resolve(args: argparse.Namespace) -> LlamaNcuCapture:
  manifest: dict[str, Any] = {}
  model = args.model
  out = pathlib.Path(args.out).expanduser() if args.out else None
  if args.run:
    rp = run_dir(args.run)
    manifest = load_manifest(rp)
    model = model or manifest.get("model_path")
    out = out or (rp / "hw_trace.json")
  if not model:
    raise ValueError("--model is required unless --run supplies one")
  if out is None:
    raise ValueError("--out is required unless --run is supplied")
  workload = args.workload or manifest.get("workload") or "prefill"
  model_id = args.model_id or manifest.get("model_id") or pathlib.Path(model).stem
  trace_dir = pathlib.Path(args.trace_dir).expanduser() if args.trace_dir else out.with_suffix(".ncu.d")
  launch_count = int(args.ncu_launch_count)
  if launch_count <= 0:
    raise ValueError("--ncu-launch-count must be positive")
  return LlamaNcuCapture(
    model=str(model), model_id=model_id,
    target_id=args.target_id or manifest.get("target_id") or "nvidia_unknown",
    workload=workload, context=int(args.context), out=out, trace_dir=trace_dir,
    prefix=args.prefix or f"{model_id}_{'tg' if workload == 'decode' else 'pp'}{args.context}",
    llama_bench=args.llama_bench, ncu=args.ncu, ncu_set=args.ncu_set,
    kernel_name=args.ncu_kernel_name or _default_kernel_filter(workload),
    launch_count=launch_count, provider_id=args.provider_id or "nvidia/ncu",
  )


def _bench_args(cfg: LlamaNcuCapture) -> list[str]:
  workload = (["-p", "0", "-n", "1", "-d", str(cfg.context)] if cfg.workload == "decode"
              else ["-p", str(cfg.context), "-n", "0"])
  return [cfg.llama_bench, "-m", cfg.model, "-ngl", "999", "-fa", "on",
          *workload, "-r", "1", "-o", "json"]


def _write_text(path: pathlib.Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _parse_bench_json(stdout: str, path: pathlib.Path) -> list[dict[str, Any]]:
  """Extract llama-bench JSON when NCU wraps stdout with profiler banners."""
  decoder = json.JSONDecoder()
  start = stdout.find("[")
  while start >= 0:
    try:
      data, _ = decoder.raw_decode(stdout[start:])
    except json.JSONDecodeError:
      start = stdout.find("[", start + 1)
      continue
    if isinstance(data, list) and data:
      return data
    start = stdout.find("[", start + 1)
  raise RuntimeError(f"llama-bench did not emit valid JSON rows; raw output: {path}")


def capture_llama_ncu(cfg: LlamaNcuCapture) -> dict[str, Any]:
  """Capture a bounded NCU report and normalize its long-form details CSV.

  Kernel replay changes whole-model latency, so llama-bench timing is retained only as an auxiliary artifact.
  The returned hardware trace contains per-dispatch NCU measurements and explicitly disclaims authority for
  clean model throughput.
  """
  cfg.trace_dir.mkdir(parents=True, exist_ok=True)
  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  base = (cfg.trace_dir / cfg.prefix).resolve()
  report = base.with_suffix(".ncu-rep")
  details_csv = base.with_name(base.name + "_details.csv")
  bench_json = base.with_name(base.name + "_llama_bench_profiled.json")
  profiler_stdout = base.with_name(base.name + "_ncu_stdout.log")
  profiler_log = base.with_name(base.name + "_ncu.log")

  cmd = [cfg.ncu, "--target-processes", "all", "--set", cfg.ncu_set,
         "--kernel-name-base", "function", "--kernel-name", str(cfg.kernel_name),
         "--launch-count", str(cfg.launch_count), "--force-overwrite", "-o", str(base),
         *_bench_args(cfg)]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  _write_text(profiler_stdout, proc.stdout)
  _write_text(profiler_log, proc.stderr)
  if "ERR_NVGPUCTRPERM" in proc.stderr:
    raise RuntimeError("NCU performance counters are restricted (ERR_NVGPUCTRPERM); run the bounded collector "
                       "with sufficient privilege or enable NVIDIA counter access")
  if proc.returncode != 0:
    raise RuntimeError(f"NCU capture exited with status {proc.returncode}; log: {profiler_log}")
  bench_rows = _parse_bench_json(proc.stdout, profiler_stdout)
  _write_text(bench_json, pretty_json(bench_rows))
  if not report.exists():
    raise FileNotFoundError(f"expected NCU report not found: {report}")

  export_cmd = [cfg.ncu, "--import", str(report), "--csv", "--page", "details",
                "--log-file", str(details_csv)]
  exported = subprocess.run(export_cmd, capture_output=True, text=True)
  if exported.returncode != 0:
    raise RuntimeError(f"NCU CSV export exited with status {exported.returncode}: {exported.stderr.strip()}")
  if not details_csv.exists():
    raise FileNotFoundError(f"expected NCU details CSV not found: {details_csv}")

  trace = import_profiler_ncu_csv(
    details_csv, model_id=cfg.model_id, target_id=cfg.target_id, workload=cfg.workload,
    context=cfg.context, provider_id=cfg.provider_id,
  )
  trace["trace_source"] = "ncu_report_details"
  trace.setdefault("aux_sources", {}).update({
    "ncu_report": str(report), "ncu_details_csv": str(details_csv),
    "llama_bench_profiled_json": str(bench_json), "ncu_stdout_log": str(profiler_stdout),
    "ncu_log": str(profiler_log),
  })
  trace["capture"] = {
    "ncu_set": cfg.ncu_set, "kernel_name": cfg.kernel_name, "launch_count": cfg.launch_count,
    "profiled_kernel_rows": len(trace.get("rows", [])), "command": cmd,
  }
  trace.setdefault("notes", []).append(
    "NCU kernel replay perturbs whole-model timing; use a clean llama-bench run for authoritative tok/s."
  )
  return trace
