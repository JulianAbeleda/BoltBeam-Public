from __future__ import annotations

import json
import pathlib

from boltbeam.cli._common import _write, _fail, _load_json, _run_manifest_defaults
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

def _register_schedule_trace(sub) -> None:
  p = sub.add_parser("schedule-trace", help="ingest a scheduler trace -> diagnose occupancy_starved/materialization/fused_ok + fix")
  p.add_argument("trace", help="schedule trace JSON from tinygrad extra/qk_schedule_trace.py")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_schedule_trace)



def cmd_schedule_trace(args) -> int:
  from boltbeam.trace.schedule_trace import ingest_schedule_trace
  _write(ingest_schedule_trace(_load_json(args.trace)), args.out)
  return 0

def _register_import_hw_trace(sub) -> None:
  p = sub.add_parser("import-hw-trace", help="convert a provider/backend kernel CSV into boltbeam.hw_trace.v1")
  p.add_argument("csv", help="kernel stats/trace CSV from a BoltBeam-supported backend adapter")
  p.add_argument("--provider", required=True, choices=["llama", "tinygrad"],
                 help="runtime provider that produced the trace")
  p.add_argument("--run", default=None, help="optional run directory; fills model/target/workload and defaults --out to hw_trace.json")
  p.add_argument("--model-id", default=None, help="BoltBeam model id for the trace; required unless --run supplies it")
  p.add_argument("--target-id", default=None, help="target id for the trace; defaults to run target or amd_gfx1100")
  p.add_argument("--workload", default=None, choices=["decode", "prefill"], help="workload represented by the trace; defaults to run workload or prefill")
  p.add_argument("--context", type=int, default=None, help="prompt/context tokens represented by this trace")
  p.add_argument("--provider-id", default=None, help="provider id to stamp in the trace; defaults to provider/boltbeam")
  p.add_argument("--backend-id", default="amd_kernel_csv",
                 help="private backend adapter id used for this import; backend counter names are normalized before output")
  p.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS, help="HBM peak GB/s used by downstream roofline analysis")
  p.add_argument("--llama-bench-json", default=None, help="optional llama-bench -o json output for wall elapsed/tok/s")
  p.add_argument("--tinygrad-trace-json", default=None, help="optional tinygrad timing JSON for wall elapsed/tok/s normalization")
  p.add_argument("--weight-inventory", default=None, help="optional weight_inventory.json for estimated physical bytes")
  p.add_argument("--backend-json", default=None, help="optional backend JSON with transfer/allocation data")
  p.add_argument("--memory-copy-csv", default=None, help="optional backend memory-copy stats/trace CSV")
  p.add_argument("--backend-counter-csv", default=None,
                 help="optional backend counter CSV; output counters use BoltBeam-normalized names")
  p.add_argument("--out", default=None, help="write hardware trace JSON here instead of stdout")
  p.add_argument("--timing-out", default=None,
                 help="write raw-normalized timing trace JSON here; defaults next to --out as timing_trace.json")
  p.set_defaults(fn=cmd_import_hw_trace)



def cmd_import_hw_trace(args) -> int:
  from boltbeam.artifacts.llama_rocprof import timing_trace_from_rocprof_csv
  from boltbeam.artifacts.tinygrad_rocprof import timing_trace_from_tinygrad_rocprof_csv
  from boltbeam.trace.hw_trace import timing_trace_to_hw_trace
  run_manifest, out = _run_manifest_defaults(args, "hw_trace.json")
  model_id = args.model_id or run_manifest.get("model_id")
  if not model_id:
    return _fail("import-hw-trace: --model-id is required unless --run supplies one")
  provider_id = args.provider_id
  if provider_id is None:
    provider_id = "llama.cpp/boltbeam" if args.provider == "llama" else "tinygrad/boltbeam"
  common = {
    "model_id": model_id,
    "target_id": args.target_id or run_manifest.get("target_id") or "amd_gfx1100",
    "workload": args.workload or run_manifest.get("workload") or "prefill",
    "context": args.context,
    "provider_id": provider_id,
    "peak_gbs": args.peak_gbs,
  }
  try:
    if args.provider == "llama":
      timing = timing_trace_from_rocprof_csv(
        args.csv, **common,
        llama_bench_json=args.llama_bench_json,
        weight_inventory=args.weight_inventory,
        rocprof_json=args.backend_json,
        memory_copy_csv=args.memory_copy_csv,
      )
    elif args.provider == "tinygrad":
      timing = timing_trace_from_tinygrad_rocprof_csv(
        args.csv, **common,
        tinygrad_trace_json=args.tinygrad_trace_json,
        weight_inventory=args.weight_inventory,
        memory_copy_csv=args.memory_copy_csv,
      )
    else:
      return _fail(f"import-hw-trace: unknown provider {args.provider!r}")
    trace = timing_trace_to_hw_trace(
      timing,
      backend_counter_csv=args.backend_counter_csv,
      backend_id=args.backend_id,
      provider_id=provider_id,
    )
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"import-hw-trace: {exc}")
  _write(trace, out)
  return 0

def _register_import_ncu(sub) -> None:
  p = sub.add_parser("import-ncu", help="convert an Nsight Compute CSV into boltbeam.hw_trace.v1")
  p.add_argument("csv", help="CSV exported by NVIDIA Nsight Compute")
  p.add_argument("--run", default=None,
                 help="optional run directory; fills model/target/workload and defaults --out to hw_trace.json")
  p.add_argument("--model-id", default=None,
                 help="BoltBeam model id for the trace; required unless --run supplies it")
  p.add_argument("--target-id", default=None,
                 help="target id for the trace; defaults to run target or nvidia_unknown")
  p.add_argument("--workload", default=None, choices=["decode", "prefill"],
                 help="workload represented by the trace; defaults to run workload or decode")
  p.add_argument("--context", type=int, default=None,
                 help="prompt/context tokens represented by this trace")
  p.add_argument("--provider-id", default=None,
                 help="provider id to stamp in the trace; defaults to nvidia/ncu")
  p.add_argument("--out", default=None, help="write hardware trace JSON here instead of stdout")
  p.set_defaults(fn=cmd_import_ncu)



def cmd_import_ncu(args) -> int:
  from boltbeam.profiler.importers.ncu import import_profiler_ncu_csv
  run_manifest, out = _run_manifest_defaults(args, "hw_trace.json")
  model_id = args.model_id or run_manifest.get("model_id")
  if not model_id:
    return _fail("import-ncu: --model-id is required unless --run supplies one")
  try:
    trace = import_profiler_ncu_csv(
      args.csv,
      model_id=model_id,
      target_id=args.target_id or run_manifest.get("target_id") or "nvidia_unknown",
      workload=args.workload or run_manifest.get("workload") or "decode",
      context=args.context,
      provider_id=args.provider_id or "nvidia/ncu",
    )
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"import-ncu: {exc}")
  _write(trace, out)
  return 0

def _register_collect_hw_trace(sub) -> None:
  p = sub.add_parser("collect-hw-trace", help="run a BoltBeam provider trace collector and emit boltbeam.hw_trace.v1")
  p.add_argument("--provider", required=True, choices=["llama", "tinygrad"], help="runtime provider to trace")
  p.add_argument("--run", default=None, help="optional run directory; fills model/target/workload and defaults --out to hw_trace.json")
  p.add_argument("--model", default=None, help="GGUF model path; required unless --run supplies one")
  p.add_argument("--model-id", default=None)
  p.add_argument("--target-id", default=None)
  p.add_argument("--workload", default=None, choices=["decode", "prefill"])
  p.add_argument("--context", type=int, default=512)
  p.add_argument("--max-context", type=int, default=4608)
  p.add_argument("--out", default=None, help="write hardware trace JSON here instead of stdout")
  p.add_argument("--timing-out", default=None,
                 help="write raw-normalized timing trace JSON here; defaults next to --out")
  p.add_argument("--trace-dir", default=None, help="directory for raw backend outputs")
  p.add_argument("--prefix", default=None)
  p.add_argument("--llama-bench", default="llama-bench")
  p.add_argument("--tinygrad-root", default=None,
                 help="tinygrad checkout (default: TINYGRAD_ROOT or recognized sibling)")
  p.add_argument("--python", default=".venv/bin/python")
  p.add_argument("--backend-tool", default=None,
                 help="optional collector executable override; defaults to the resolved capability tool")
  p.add_argument("--sampler", default=None,
                 help="collector id; default is resolved from provider + target capabilities")
  p.add_argument("--xcrun", default="xcrun", help="Apple developer-tool launcher for metal-system-trace")
  p.add_argument("--trace-time-limit", type=int, default=120)
  p.add_argument("--reuse-capture", action="store_true",
                 help="normalize an existing complete backend capture without launching the runtime")
  p.add_argument("--backend-id", default="amd_kernel_csv")
  p.add_argument("--backend-counter-csv", default=None)
  p.add_argument("--weight-inventory", default=None)
  p.add_argument("--prefill-chunked", default="1")
  p.add_argument("--prefill-K", dest="prefill_k", type=int, default=None,
                 help="optional bounded authority burst count")
  p.add_argument("--prefill-warmups", dest="prefill_warmups", type=int, default=None)
  p.add_argument("--prefill-rounds", dest="prefill_rounds", type=int, default=None)
  p.add_argument("--prefill-start-positions", dest="prefill_start_positions", default=None)
  p.add_argument("--prefill-whole-lengths", dest="prefill_whole_lengths", default=None)
  p.add_argument("--provider-id", default=None)
  p.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  p.add_argument("--env", action="append", default=[], help="extra provider env override KEY=VALUE")
  p.add_argument("--gpu-health", default="warn", choices=["off", "warn", "fail"],
                 help="check for missing render nodes, D-state GPU tasks, and stale GPU fd holders around collection")
  p.set_defaults(fn=cmd_collect_hw_trace)



def cmd_collect_hw_trace(args) -> int:
  from boltbeam.collectors.hw_trace import collect_hw_trace
  if args.run and args.out is None:
    from boltbeam.workflow.common import run_dir
    args.out = str(run_dir(args.run) / "hw_trace.json")
  try:
    timing, trace, out, timing_out = collect_hw_trace(args)
  except (FileNotFoundError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
    return _fail(f"collect-hw-trace: {exc}")
  _write(timing, str(timing_out))
  _write(trace, str(out))
  return 0

def _register_compare_hw_trace(sub) -> None:
  p = sub.add_parser("compare-hw-trace", help="compare two boltbeam.hw_trace.v1 files, including timing and normalized counters")
  p.add_argument("--baseline", required=True, help="baseline boltbeam.hw_trace.v1 JSON, usually llama.cpp")
  p.add_argument("--candidate", required=True, help="candidate boltbeam.hw_trace.v1 JSON, usually tinygrad")
  p.add_argument("--context", type=int, default=None, help="context length to compare; defaults to the candidate's first context")
  p.add_argument("--out", default=None, help="write comparison JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional timing markdown report path")
  p.set_defaults(fn=cmd_compare_hw_trace)



def cmd_compare_hw_trace(args) -> int:
  from boltbeam.trace.hw_trace import command_buffer_compare_markdown, compare_hw_traces
  from boltbeam.trace.timing_compare import compare_markdown
  try:
    report = compare_hw_traces(_load_json(args.baseline), _load_json(args.candidate), context=args.context)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"compare-hw-trace: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(compare_markdown(report["timing_compare"]) + "\n" +
                                           command_buffer_compare_markdown(report))
  return 0

def _register_compare_substrate(sub) -> None:
  p = sub.add_parser("compare-substrate", help="compare hot packed-prefill GEMM substrate rows against a baseline trace")
  p.add_argument("--baseline", required=True, help="baseline boltbeam.timing_trace.v1/hw_trace.v1 JSON, usually llama.cpp")
  p.add_argument("--candidate", required=True, help="candidate boltbeam.timing_trace.v1/hw_trace.v1 JSON, usually tinygrad")
  p.add_argument("--context", type=int, default=None, help="context length to compare; defaults to the candidate's first context")
  p.add_argument("--min-candidate-pct", type=float, default=3.0,
                 help="minimum candidate step percentage for a packed GEMM row to appear in the hot table")
  p.add_argument("--out", default=None, help="write substrate comparison JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_compare_substrate)



def cmd_compare_substrate(args) -> int:
  from boltbeam.trace.substrate_compare import compare_substrate, substrate_markdown
  try:
    report = compare_substrate(_load_json(args.baseline), _load_json(args.candidate),
                               context=args.context, min_candidate_pct=args.min_candidate_pct)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"compare-substrate: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(substrate_markdown(report))
  return 0

def _register_profiler_report(sub) -> None:
  p = sub.add_parser("profiler-report", help="model-aware profiler report over hw_trace: top kernels, speed-of-light, roofline, causal verdict, missing evidence")
  p.add_argument("trace", help="boltbeam.hw_trace.v1 JSON")
  p.add_argument("--target-id", default=None, help="override target id (else from trace)")
  p.add_argument("--top", type=int, default=5, help="number of hot kernels to detail")
  p.add_argument("--baseline", default=None, help="optional baseline hw_trace.v1 for more-work-vs-lost-efficiency")
  p.add_argument("--roofline-input", default=None,
                 help="optional current-model JSON with measured/raw_ceiling/practical_ceiling/metric and bases")
  p.add_argument("--tiered-roofline", default=None,
                 help="optional route-aware roofline-theoretical --throughput JSON for operand/cache-tier reporting")
  p.add_argument("--out", default=None, help="write report JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_profiler_report)



def cmd_profiler_report(args) -> int:
  from boltbeam.profiler.report import profiler_report, profiler_report_markdown
  try:
    baseline = _load_json(args.baseline) if args.baseline else None
    report = profiler_report(_load_json(args.trace), target_id=args.target_id, top=args.top, baseline=baseline,
                             roofline_input=_load_json(args.roofline_input) if args.roofline_input else None,
                             tiered_roofline=_load_json(args.tiered_roofline) if args.tiered_roofline else None)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"profiler-report: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(profiler_report_markdown(report))
  return 0

def _register_prefill_role_trace(sub) -> None:
  p = sub.add_parser("prefill-role-trace", help="extract one prefill role and require memory/VALU/MFMA counter coverage")
  p.add_argument("trace", help="BoltBeam timing/hw trace JSON")
  p.add_argument("--role", required=True, help="role to isolate, e.g. ffn_gate_up")
  p.add_argument("--quant", default=None, help="quant to isolate, e.g. Q4_K")
  p.add_argument("--shape", default=None, help="optional M,N,K shape, e.g. 512,17408,5120")
  p.add_argument("--weight-inventory", default=None,
                 help="optional weight inventory used for exact bytes and llama grid-to-role matching")
  p.add_argument("--rows-per-grid-x", type=int, default=4,
                 help="llama mul_mat_q role mapping: output rows per grid_x entry")
  p.add_argument("--out", default=None, help="write role trace JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown role trace")
  p.set_defaults(fn=cmd_prefill_role_trace)



def cmd_prefill_role_trace(args) -> int:
  from boltbeam.experiment.prefill_role_trace import prefill_role_trace_markdown, prefill_role_trace_report
  try:
    report = prefill_role_trace_report(
      _load_json(args.trace),
      role=args.role,
      quant=args.quant,
      shape=args.shape,
      weight_inventory=args.weight_inventory,
      rows_per_grid_x=args.rows_per_grid_x,
    )
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"prefill-role-trace: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(prefill_role_trace_markdown(report))
  return 0


def register(sub) -> None:
  _register_schedule_trace(sub)
  _register_import_hw_trace(sub)
  _register_import_ncu(sub)
  _register_collect_hw_trace(sub)
  _register_compare_hw_trace(sub)
  _register_compare_substrate(sub)
  _register_profiler_report(sub)
  _register_prefill_role_trace(sub)
