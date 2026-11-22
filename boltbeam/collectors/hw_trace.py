from __future__ import annotations

import argparse
import json
import pathlib, sys
from typing import Any

from boltbeam.runtime.gpu_health import assert_gpu_health
from boltbeam.trace.hw_trace import timing_trace_to_hw_trace
from boltbeam.profiler.capabilities import normalize_collector_id, normalize_provider_id, resolve_profiler_capability
from boltbeam.artifacts.run_io import load_manifest, run_dir
from boltbeam.collectors.engine_families import provider_name
from boltbeam.core.canonical import pretty_json


# The tinygrad fork's bench harnesses (bench.py, prefill_boltbeam_trace.py) left tinygrad-arkey with the kernel generator.
RETIRED_ARKEY = ("retired 2026-11-21: the arkey hardware-trace samplers (rocprofv3, tinygrad-profile-events, tinygrad-native-pmc) ran "
                 "the tinygrad fork's bench harnesses, which left tinygrad-arkey with the kernel generator; arkey's role times "
                 "come from role-time (collectors/arkey_role_time), and metal-system-trace still runs")


def _write(obj:dict[str, Any], out:pathlib.Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(obj))


def _timing_out_for(hw_out:pathlib.Path, explicit:str | None) -> pathlib.Path:
  if explicit:
    return pathlib.Path(explicit).expanduser()
  if hw_out.name == "hw_trace.json":
    return hw_out.with_name("timing_trace.json")
  return hw_out.with_name(hw_out.stem + "_timing_trace.json")


def _target_id(args:argparse.Namespace) -> str:
  if getattr(args, "target_id", None): return str(args.target_id)
  if getattr(args, "run", None) and (target := load_manifest(run_dir(args.run)).get("target_id")): return str(target)
  raise ValueError("--target-id is required unless --run supplies a target; collector selection has no backend default")


def _default_sampler(provider:str, target_id:str) -> str:
  return resolve_profiler_capability(normalize_provider_id(provider), target_id).collector_id


def collect_hw_trace(args:argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], pathlib.Path, pathlib.Path]:
  target_id, args.provider = _target_id(args), provider_name(args.provider)  # old callers may still say tinygrad
  sampler = normalize_collector_id(getattr(args, "sampler", None) or _default_sampler(args.provider, target_id))
  if args.provider == "arkey" and sampler != "metal-system-trace":
    raise ValueError(RETIRED_ARKEY)   # before the GPU health check: nothing runs
  capability = resolve_profiler_capability(normalize_provider_id(args.provider), target_id, collector_id=sampler)
  if getattr(args, "backend_tool", None) is None: args.backend_tool = capability.tool
  health_mode = getattr(args, "gpu_health", "warn")
  pre_health = assert_gpu_health(health_mode, stage="preflight")
  if sampler == "metal-system-trace":
    from boltbeam.collectors.metal_system_trace import capture_metal_system_trace
    timing, out = capture_metal_system_trace(args)
  elif args.provider == "llama":
    if sampler != "rocprofv3":
      raise ValueError(f"sampler {sampler!r} is not implemented for llama yet")
    from boltbeam.collectors.llama_rocprof import _resolve, capture_llama_rocprof
    args.rocprofv3 = args.backend_tool
    args.provider_id = args.provider_id or "llama.cpp/boltbeam"
    cfg = _resolve(args)
    timing = capture_llama_rocprof(cfg)
    out = cfg.out
  else:
    raise ValueError(f"unknown provider {args.provider!r}")
  post_health = assert_gpu_health(health_mode, stage="postrun")
  timing.setdefault("aux_sources", {})
  timing["aux_sources"]["gpu_health"] = {"preflight": pre_health, "postrun": post_health}
  hw_trace = timing_trace_to_hw_trace(
    timing,
    backend_counter_csv=args.backend_counter_csv,
    backend_id=args.backend_id,
    provider_id=args.provider_id,
  )
  return timing, hw_trace, out, _timing_out_for(out, getattr(args, "timing_out", None))


def main(argv:list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description="Collect a provider trace and emit boltbeam.hw_trace.v1")
  ap.add_argument("--provider", required=True, choices=["llama", "arkey"], type=provider_name)
  ap.add_argument("--run", default=None, help="optional BoltBeam run directory")
  ap.add_argument("--model", default=None, help="GGUF model path; required unless --run supplies one")
  ap.add_argument("--model-id", default=None)
  ap.add_argument("--target-id", default=None)
  ap.add_argument("--workload", default=None, choices=["decode", "prefill"])
  ap.add_argument("--context", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--out", default=None, help="write boltbeam.hw_trace.v1 here; defaults to <run>/hw_trace.json")
  ap.add_argument("--timing-out", default=None,
                  help="write raw-normalized boltbeam.timing_trace.v1 here; defaults next to --out as timing_trace.json")
  ap.add_argument("--trace-dir", default=None, help="directory for raw backend outputs")
  ap.add_argument("--prefix", default=None)
  ap.add_argument("--llama-bench", default="llama-bench")
  ap.add_argument("--tinygrad-root", default=None,
                  help="tinygrad checkout (default: TINYGRAD_ROOT or recognized sibling)")
  ap.add_argument("--python", default=".venv/bin/python")
  ap.add_argument("--backend-tool", default=None,
                  help="optional collector executable override; defaults to the resolved capability tool")
  ap.add_argument("--sampler", default=None,
                  help="collector id; default is resolved from provider + target capabilities")
  ap.add_argument("--xcrun", default="xcrun", help="Apple developer-tool launcher for metal-system-trace")
  ap.add_argument("--trace-time-limit", type=int, default=120)
  ap.add_argument("--reuse-capture", action="store_true",
                  help="normalize an existing complete backend capture without launching the runtime")
  ap.add_argument("--backend-id", default="amd_kernel_csv")
  ap.add_argument("--backend-counter-csv", default=None)
  ap.add_argument("--weight-inventory", default=None)
  ap.add_argument("--prefill-chunked", default="1")
  ap.add_argument("--provider-id", default=None)
  ap.add_argument("--peak-gbs", type=float, default=None, help="defaults to the target's registry bandwidth")
  ap.add_argument("--env", action="append", default=[], help="extra provider env override KEY=VALUE")
  ap.add_argument("--gpu-health", default="warn", choices=["off", "warn", "fail"],
                  help="check for missing render nodes, D-state GPU tasks, and stale GPU fd holders around collection")
  args = ap.parse_args(argv)
  if args.run and args.out is None:
    args.out = str(run_dir(args.run) / "hw_trace.json")
  try:
    timing, hw_trace, out, timing_out = collect_hw_trace(args)
    _write(timing, timing_out)
    _write(hw_trace, out)
  except (FileNotFoundError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"hw_trace collector: {exc}\n")
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
