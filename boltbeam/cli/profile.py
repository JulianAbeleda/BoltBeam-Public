from __future__ import annotations

import json, pathlib

from boltbeam.cli._common import _write, _fail, _load_json, _run_manifest_defaults
from boltbeam.target.targets import need_target_id
from boltbeam.collectors.engine_families import provider_name

def _register_schedule_trace(sub) -> None:
  p = sub.add_parser("schedule-trace", help="ingest a scheduler trace -> diagnose occupancy_starved/materialization/fused_ok + fix")
  p.add_argument("trace", help="schedule trace JSON from tinygrad extra/qk_schedule_trace.py")
  p.add_argument("--peak-gbs", type=float, required=True, help="the traced GPU's peak bandwidth, GB/s")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_schedule_trace)



def cmd_schedule_trace(args) -> int:
  from boltbeam.trace.schedule_trace import ingest_schedule_trace
  _write(ingest_schedule_trace(_load_json(args.trace), peak_gbs=args.peak_gbs), args.out)
  return 0

def _register_import_hw_trace(sub) -> None:
  p = sub.add_parser("import-hw-trace", help="convert a provider/backend kernel CSV into boltbeam.hw_trace.v1")
  p.add_argument("csv", help="kernel stats/trace CSV from a BoltBeam-supported backend adapter")
  p.add_argument("--provider", required=True, choices=["llama", "arkey"], type=provider_name,
                 help="runtime provider that produced the trace")
  p.add_argument("--run", default=None, help="optional run directory; fills model/target/workload and defaults --out to hw_trace.json")
  p.add_argument("--model-id", default=None, help="BoltBeam model id for the trace; required unless --run supplies it")
  p.add_argument("--target-id", default=None, help="target id for the trace; required unless --run supplies one")
  p.add_argument("--workload", default=None, choices=["decode", "prefill"], help="workload represented by the trace; defaults to run workload or prefill")
  p.add_argument("--context", type=int, default=None, help="prompt/context tokens represented by this trace")
  p.add_argument("--provider-id", default=None, help="provider id to stamp in the trace; defaults to provider/boltbeam")
  p.add_argument("--backend-id", default="amd_kernel_csv",
                 help="private backend adapter id used for this import; backend counter names are normalized before output")
  p.add_argument("--peak-gbs", type=float, default=None, help="HBM peak GB/s used by downstream roofline analysis; defaults to the target's")
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
  try:
    target_id = need_target_id(args.target_id or run_manifest.get("target_id"), "import-hw-trace")
  except ValueError as exc:
    return _fail(str(exc))
  common = {
    "model_id": model_id,
    "target_id": target_id,
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
    elif args.provider == "arkey":
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
                 help="target id for the trace; required unless --run supplies one (no chip is assumed)")
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
      target_id=need_target_id(args.target_id or run_manifest.get("target_id"), "import-ncu"),
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
  p.add_argument("--provider", required=True, choices=["llama", "arkey"], type=provider_name, help="runtime provider to trace")
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
  p.add_argument("--peak-gbs", type=float, default=None, help="defaults to the target's registry bandwidth")
  p.add_argument("--env", action="append", default=[], help="extra provider env override KEY=VALUE")
  p.add_argument("--gpu-health", default="warn", choices=["off", "warn", "fail"],
                 help="check for missing render nodes, D-state GPU tasks, and stale GPU fd holders around collection")
  p.set_defaults(fn=cmd_collect_hw_trace)



def cmd_collect_hw_trace(args) -> int:
  from boltbeam.collectors.hw_trace import collect_hw_trace
  if args.run and args.out is None:
    from boltbeam.artifacts.run_io import run_dir
    args.out = str(run_dir(args.run) / "hw_trace.json")
  try:
    timing, trace, out, timing_out = collect_hw_trace(args)
  except (FileNotFoundError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
    return _fail(f"collect-hw-trace: {exc}")
  _write(timing, str(timing_out))
  _write(trace, str(out))
  return 0

def _register_metal_measure(sub) -> None:
  from boltbeam.collectors.metal_native import add_arguments
  p = sub.add_parser("metal-measure", help="measure a run on this Mac's GPU with BoltBeam's own Metal kernels: "
                                           "probe_evidence.json and timing_trace.json, no Xcode needed")
  add_arguments(p)
  p.set_defaults(fn=cmd_metal_measure)


def _register_metal_bandwidth(sub) -> None:
  p = sub.add_parser("metal-bandwidth", help="measure this Mac GPU's read-only memory speed, the number the speed limit needs")
  p.add_argument("--runs", type=int, default=1, help="whole runs of the read and matrix probes; the uncertainty is their spread at the best shape")
  p.set_defaults(fn=cmd_metal_bandwidth)


def cmd_metal_bandwidth(args) -> int:
  from boltbeam.collectors.metal_bandwidth import run_cli
  return run_cli(args)


def cmd_metal_measure(args) -> int:
  from boltbeam.collectors.metal_native import run_cli
  return run_cli(args)


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


def _register_time_budget(sub) -> None:
  p = sub.add_parser("time-budget", help="where one request's wall time went, from an nsys sqlite export: kernels by family, "
                     "exposed copy, GPU idle by cause, host time outside the GPU work (boltbeam.time_budget.v1)")
  p.add_argument("trace", help="an nsys capture's sqlite export (capture.sqlite)")
  p.add_argument("--wall-ms", type=float, default=None, help="the client's wall time for the request (default: from --driver, else unknown)")
  p.add_argument("--driver", default=None, help="a prefill driver's JSON line, or an MTP decode driver's output (llama_server_decode_drive.py "
                 "--mtp-prompts, vllm_mtp_drive.py), as a file: the request's span and wall time come from it; a decode row adds its steps")
  p.add_argument("--request", type=int, default=None, help="with --driver: which timed request (default: the median wall; a decode: the median tok/s)")
  p.add_argument("--untraced", default=None, help="a decode: the same driver command's output run without the profiler; the capture overhead "
                 "is measured against the same prompt, depth and k in it")
  p.add_argument("--model", default=None, help="a decode: the model file or folder, for the weight bytes a verify forward reads")
  p.add_argument("--draft", default=None, help="a decode: the MTP draft head's GGUF, for the weight bytes a draft pass reads")
  p.add_argument("--spans-from-trace", action="store_true", help="a decode whose driver recorded no span_ns (an older driver): the requests "
                 "are the trace's last ones, each its client wall long and ending where its GPU work ends (time_budget.spans_from_trace)")
  p.add_argument("--prompt-from-counts", action="store_true", help="a decode whose driver recorded no first-token mark: the prompt is the "
                 "fewest leading forwards after which the rest give the engine's counts (said in the output)")
  p.add_argument("--target", default=None, help="a decode: the GPU's target id (data/targets.json), for its measured read rate")
  p.add_argument("--engine", action="append", required=True, help="kernel family rows to read (data/kernel_families.json engines: a fork, then its root), in order; repeatable")
  p.add_argument("--vendor", default="nvidia", help="the GPU vendor's rows (target/vendors.py; nsys traces are nvidia)")
  p.add_argument("--gap-ms", type=float, default=None, help="a prefill: attribute idle gaps at least this long one by one (default time_budget.GAP_MS, a named constant); a decode attributes every gap")
  p.add_argument("--out", default=None, help="write the record here (JSON)")
  p.set_defaults(fn=cmd_time_budget)


def cmd_time_budget(args) -> int:
  import statistics
  from boltbeam.collectors import time_budget as tb
  from boltbeam.report.plain import time_budget_words
  if len(missing := tb.unknown_engines(args.engine)) == len(args.engine):  # a fork without rows of its own reads its root's
    return _fail(f"BoltBeam has no kernel family rows for {', '.join(missing)}; engines it has rows for: {', '.join(tb.engine_names())}")
  kw = dict(engine=args.engine, vendor=args.vendor, **({"gap_ms": args.gap_ms} if args.gap_ms is not None else {}),
            what=f"from {pathlib.Path(args.trace).name}")
  from boltbeam.collectors.engine_common import json_lines  # a profiler's own lines may sit among the driver's
  lines = json_lines(pathlib.Path(args.driver).read_text()) if args.driver else []
  if rows := [r for r in lines if (r.get("span_ns") or args.spans_from_trace) and "spec" in r and r.get("decode_tok_s")]:
    return _decode_time_budget(args, rows, lines, kw)
  if args.driver:
    line = json.loads(pathlib.Path(args.driver).read_text().strip().splitlines()[-1])
    walls = line.get("wall_samples_ms") or []
    i = args.request if args.request is not None else (walls.index(statistics.median_low(walls)) if walls else 0)
    rec = tb.request_budget(args.trace, line, i, **kw)
    if rec is None:
      return _fail("the driver line has no request span for that request, or the trace no clock origin: pass --wall-ms instead")
  else:
    rec = tb.from_nsys(args.trace, wall_ms=args.wall_ms, **kw)
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
  got = time_budget_words(rec)
  print(got["line"])
  for l in got["lines"]:
    print("  " + l)
  from boltbeam.report.plain import budget_unfit, words
  unknown_row = words()["time_budget"]["rows"]["host"] if budget_unfit(rec) else None
  print("| part | ms |\n|---|---|")
  for label, v in got["rows"]:
    print(f"| {label} | {'unknown' if v is None and label == unknown_row else '-' if v is None else f'{round(v, 1) + 0.0:,.1f}'} |")
  return 0


def _decode_time_budget(args, rows:list[dict], lines:list[dict], kw:dict) -> int:
  """time-budget of one decode request of an MTP driver's output (rows: its decode rows; lines: every JSON line, the
  engine's among them): its steps, what bounds them, the capture overhead."""
  import statistics
  from boltbeam.collectors import time_budget as tb
  from boltbeam.collectors.engine_common import json_lines
  from boltbeam.core.refused import Refused
  from boltbeam.report.plain import time_budget_decode_words, time_budget_words
  speeds = [r["decode_tok_s"] for r in rows]
  i = args.request if args.request is not None else speeds.index(statistics.median_low(speeds))
  if not 0 <= i < len(rows):
    return _fail(f"the driver output has {len(rows)} decode requests; --request {i} is not one of them")
  row = rows[i]
  try:
    read = _decode_read(args, row, lines)
  except (Refused, ValueError, OSError) as exc:
    return _fail(f"time-budget: {exc}")
  overhead = _decode_overhead(row, rows, json_lines(pathlib.Path(args.untraced).read_text()) if args.untraced else None)
  kw = {k: v for k, v in kw.items() if k != "gap_ms"}  # a decode attributes every gap
  more = {}
  if args.model:  # the output row in f32: a read of whole rows is a result, other reads are layers handed to the host
    from boltbeam.profile.loaders import profile_from_model
    more["output_row_bytes"] = 4 * (profile_from_model(args.model).vocab_size or 0) or None
  if args.spans_from_trace:
    tl = tb.timeline_from_nsys(args.trace)
    more = {"tl": tl, "trace_span": tb.spans_from_trace(tl, [r["wall_s"] * 1e3 for r in rows])[i]}
  rec = tb.request_decode_budget(args.trace, row, read=read, overhead=overhead, input_rows=_input_rows(args, int(row.get("k") or 0)),
                                 prompt_from_counts=args.prompt_from_counts, **more,
                                 **{**kw, "what": f"one decode request ({row.get('prompt')}, {row.get('depth')}, k={row.get('k')}), from {pathlib.Path(args.trace).name}"})
  if rec is None:
    return _fail("the trace has no clock origin (TARGET_INFO_SESSION_START_TIME): the request's span cannot be placed in it")
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
  if not rec["decode"].get("refused"):  # a refused cut prints no shares
    print(time_budget_words(rec)["line"])
  for line in time_budget_decode_words(rec):
    print("  " + line)
  return 0


def _decode_read(args, row:dict, lines:list[dict]) -> dict | None:
  """What bounds a step: the weight bytes a verify of the engine's own tokens per verify (1 + draft tokens / verify steps,
  through the model's saved routing profile when there is one) and a draft pass read, at the GPU's measured read rate.
  Where the weights went, as the engine's row says to read it: its weight buffers in the driver's engine line for this
  k (residency_ok); an engine whose row reads none gives the bound with residency "not checked". {"refused": key}
  when it is read and not every weight is on the GPU, or cannot be read; None without --model and --target."""
  if not (args.model and args.target):
    return None
  from boltbeam.collectors import time_budget as tb
  from boltbeam.profile.loaders import profile_from_model
  from boltbeam.roofline import forward_read as fr, routing_profile
  rule = (tb.decode_row(args.engine) or {}).get("residency")
  checked = "not checked: BoltBeam does not read where this engine put its weights"
  if rule:
    eng = next((l["engine"] for l in lines if isinstance(l.get("engine"), dict) and l["engine"].get("k") == row.get("k")), None)
    if eng is None:
      return {"refused": "no_engine_line"}
    got = residency_ok(eng.get("residency") or {}, rule["host_buffers"], _host_table_bytes(args.model),
                       _host_table_bytes(args.draft) if args.draft else 0.0, bool(row.get("k")))
    if got is not True:
      return {"refused": got}
    checked = "read: the target model's and the draft head's weight buffers hold no weight in host memory beyond their host input tables"
  spec = row.get("spec") or {}
  w = 1.0 + spec["draft_tokens"] / spec["drafts"] if spec.get("drafts") else 1.0
  found = routing_profile.find(args.model)
  v = fr.verify_bytes(profile_from_model(args.model).to_json(), w, found["layers"] if found else None)
  d = fr.draft_bytes(args.draft) if args.draft else None
  return {**fr.measured_read(args.target), "verify_bytes": v["bytes"], "verify_tokens": w, "draft_bytes": d and d["bytes"],
          "how": v["how"] + ("; " + d["how"] if d else ""), "residency": checked}


def residency_ok(r:dict, host:str, target_tables:float, draft_tables:float = 0.0, draft:bool = False) -> bool | str:
  """True when the weight buffers the driver read (residency) put nothing in host memory but each model's host input
  tables: per model (the target, and with a draft head its own), the MiB of every host-memory buffer (host: the engine
  row's pattern) at most its tables' bytes, within the print's own rounding (rounding_mib). Else the refusal's key."""
  import re
  if not r.get("target") or (draft and not r.get("draft")):
    return "residency_not_read"
  for part, tables in (("target", target_tables), ("draft", draft_tables)):
    held = sum(v for n, v in (r.get(part) or {}).items() if re.search(host, n))
    if (held - (r.get("rounding_mib") or {}).get(part, 0.0)) * 2**20 > tables:
      return "not_all_on_gpu"
  return True


def _host_table_bytes(model:str) -> float:
  """The bytes of the model file's host input tables (workflow/recommend.HOST_ONLY), as stored."""
  if not str(model).endswith(".gguf"):
    return 0.0
  import math
  from boltbeam.profile.gguf import read_gguf_split
  from boltbeam.profile.weight_ledger import _physical_bytes
  from boltbeam.workflow.recommend import HOST_ONLY
  return float(sum(_physical_bytes(math.prod(d), t, tensor_name=n) for n, d, t, _ in read_gguf_split(model)[1] if n.split(".")[0] in HOST_ONLY))


def _input_rows(args, k:int) -> dict[str, int]:
  """The per-token uploads a copy is matched against (time_budget.input_copies): each host input table of the model
  file (workflow/recommend.HOST_ONLY), its row's bytes on the way in (its elements x the engine row's
  host_rows_element_bytes), and with a draft head (k > 0) and an engine row that uploads one, the draft context's hidden
  row (the model's embedding_length x the same bytes). {} when the engine keeps no host table or no GGUF was given."""
  from boltbeam.collectors import time_budget as tb
  row = tb.decode_row(args.engine) or {}
  per = row.get("host_rows_element_bytes")
  if not (per and args.model and str(args.model).endswith(".gguf")):
    return {}
  from boltbeam.profile.gguf import read_gguf_split
  from boltbeam.workflow.recommend import HOST_ONLY
  kv, tensors, _ = read_gguf_split(args.model)
  out = {n.split(".")[0]: int(dims[0]) * per for n, dims, _, _ in tensors if n.split(".")[0] in HOST_ONLY and len(dims) == 2}
  if k and row.get("draft_hidden_rows") and (width := kv.get(f"{kv.get('general.architecture')}.embedding_length")):
    out["draft_hidden"] = int(width) * per
  return out


def _decode_overhead(row:dict, traced:list[dict], untraced:list[dict] | None) -> dict:
  """The capture's overhead: every prompt at this request's draft count and depth, traced against untraced (the
  engine's own decode tok/s, paired by prompt); the median pair's said in replay_overhead's words, every pair kept."""
  from boltbeam.collectors.replay_overhead import overhead_words
  import statistics
  if untraced is None:
    return {"pct": None, "words": None, "why": "no_untraced"}
  same = lambda r: (r.get("k"), r.get("depth")) == (row.get("k"), row.get("depth")) and r.get("decode_tok_s")  # noqa: E731
  base = {r.get("prompt"): r["decode_tok_s"] for r in untraced if same(r)}
  pairs = {r["prompt"]: 100.0 * (base[r["prompt"]] / r["decode_tok_s"] - 1) for r in traced if same(r) and r.get("prompt") in base}
  if not pairs:
    return {"pct": None, "words": None, "why": "untraced_no_row"}
  pct = statistics.median(pairs.values())
  return {"pct": pct, "words": overhead_words(pct, f"the untraced run (median of {len(pairs)} prompts)"), "pairs": pairs,
          "range": [min(pairs.values()), max(pairs.values())], "k": row.get("k"), "depth": row.get("depth"),
          "how": "the engine's own decode tok/s per prompt at this draft count and depth, traced against untraced"}


def cmd_context_audit(args) -> int:
  from boltbeam.collectors import time_budget as tb
  from boltbeam.core.refused import Refused
  from boltbeam.workflow import context_audit as ca
  from boltbeam.workflow.diff import DiffRefused
  path = pathlib.Path(args.record)
  if (row := tb.decode_row(args.engine)) is None:
    return _fail(f"BoltBeam has no decode row for {', '.join(args.engine)} (data/kernel_families.json): the step rule comes from it")
  try:
    out = ca.audit({"source": path.name, **json.loads(path.read_text())}, engine=args.engine[-1], row=row, base=path.parent)
  except (Refused, DiffRefused) as exc:
    return _fail(f"context-audit: {exc}")
  except (KeyError, TypeError, ValueError, AttributeError) as exc:  # a record that lacks or mistypes a field: said, never a traceback
    return _fail(f"context-audit: the record is malformed ({type(exc).__name__}: {exc}); docs/context-audit.md lists its fields")
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
  print("\n".join(ca.lines(out)))
  return 0 if out["ok"] else 1


LOWERING_HELP = "gemm-space's lowering facts (boltbeam.gemm_lowering_facts.v1): whether it admits a float row's shape"


def _lowering(path:str | None):
  """The lowering facts kernel-ledger and session-map hand the ledger (kernel_ledger.coverage), or None."""
  from boltbeam.search.gemm_strategy import GemmLowering
  return GemmLowering.from_json(json.loads(pathlib.Path(path).read_text())) if path else None


def cmd_kernel_ledger(args) -> int:
  from boltbeam.collectors import time_budget as tb
  from boltbeam.roofline import limit
  from boltbeam.workflow import kernel_ledger as kl
  paths = [c.partition("@")[0] for c in args.capture] + [p for p in (args.model, args.draft, args.lowering, args.session, args.manifest, args.routing, *(args.counters or [])) if p]
  if missing := [p for p in paths if not pathlib.Path(p).is_file()]:
    return _fail(f"kernel-ledger: no such file: {', '.join(missing)}")
  if bad := tb.unknown_engines(args.engine):
    return _fail(f"kernel-ledger: data/kernel_families.json has no rows for {', '.join(bad)}: the steps and the join come from them")
  if (target := next((t for t in limit.targets()["targets"] if t["id"] == args.target), None)) is None:
    return _fail(f"kernel-ledger: no target {args.target} (python -m boltbeam.workflow.screen targets lists them)")
  if not kl.shape_rows(args.engine, args.target):
    return _fail(f"kernel-ledger: {', '.join(args.engine)} has no shapes rows (data/kernel_families.json): no output layer to cut steps by, no join")
  if len(args.untraced_ms or []) not in (0, len(args.capture)) or len(args.counters or []) not in (0, len(args.capture)):
    return _fail("kernel-ledger: give --untraced-ms and --counters once per capture, in the captures' order, or not at all")
  if bool(args.session) != bool(args.window) or (args.window and len(args.window) != len(args.capture)):
    return _fail("kernel-ledger: give --session with one --window per capture (the window its turns are tagged with), or neither")
  load = lambda p: json.loads(pathlib.Path(p).read_text()) if p else None  # noqa: E731
  low = _lowering(args.lowering)
  from boltbeam.collectors import replay
  turns = replay.llama_turns(load(args.session)["turns"]) if args.session else None
  caps = []
  for i, (p, _, d) in enumerate(c.partition("@") for c in args.capture):
    tw = [t for t in turns if t["window"] == args.window[i]] if turns else None
    if turns and not tw:
      return _fail(f"kernel-ledger: the session record tags no turn with window {args.window[i]}")
    caps.append({"cap": kl.read(p, tw), "depth": int(d) if d else round(sum(t["depth"] for t in tw) / len(tw)) if tw else None,
                 "untraced_ms": (args.untraced_ms or [None] * len(args.capture))[i], "counters": load((args.counters or [None] * len(args.capture))[i])})
  out = kl.ledger(caps, main=load(args.model), draft=load(args.draft), engine=args.engine, vendor=args.vendor, target=target, k=args.k, kv_type=args.kv_type,
                  selection={"verify": args.selection} if args.selection else None, lowering=low, manifest=load(args.manifest), routing=load(args.routing))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=list) + "\n")
  print("\n".join(kl.lines(out, args.top)))
  return 0 if out.get("ok") else 1


def _saved_h2d() -> float | None:
  """This machine's saved pinned host-to-device rate when the copy probe holds one GPU (target/host.only_h2d), else None."""
  from boltbeam.target import host
  return ((host.only_h2d() or {}).get("h2d_gbs") or {}).get("pinned")


def cmd_session_map(args) -> int:
  from boltbeam.collectors import replay
  from boltbeam.collectors import time_budget as tb
  from boltbeam.roofline import limit
  from boltbeam.workflow import session_map as sm
  caps = [c.partition("@") for c in args.capture]
  paths = [p for p, _, _ in caps] + [p.partition("@")[0] for p in (*args.session, args.model, args.draft, args.lowering, args.manifest, args.routing, args.reference, args.previous,
                                                                    args.floors, *(args.untraced or [])) if p]
  if missing := [p for p in paths if not pathlib.Path(p).is_file()]:
    return _fail(f"session-map: no such file: {', '.join(missing)}")
  if bad := [p for p, _, w in caps if not w]:
    return _fail(f"session-map: give each capture as PATH@WINDOW (the window the session record tags its turns with): {', '.join(bad)}")
  if bad := tb.unknown_engines(args.engine):
    return _fail(f"session-map: data/kernel_families.json has no rows for {', '.join(bad)}")
  if (target := next((t for t in limit.targets()["targets"] if t["id"] == args.target), None)) is None:
    return _fail(f"session-map: no target {args.target} (data/targets.json lists them)")
  from boltbeam.target.targets import get_target
  target = {**target, "compute_units": get_target(args.target).compute_units}  # SMs / CUs: the audit's grid flag
  load = lambda p: json.loads(pathlib.Path(p).read_text()) if p else None  # noqa: E731

  def runs(spec:str) -> list[list[dict]]:  # a driver record (its turns), or session-ab's record @ an arm (each valid run of it)
    path, _, arm = spec.partition("@")
    d = load(path)
    if arm:
      return [replay.llama_turns(a["turns"]) for a in d.get("arms") or [] if a.get("name") == arm and a.get("valid")]
    return [replay.llama_turns(d["turns"])]
  try:
    recs = [r for x in args.session for r in runs(x)[:1]]
    if len(recs) < len(args.session):
      return _fail("session-map: a --session names no valid run")
    tags = [{t["window"] for t in r if t["window"]} for r in recs]
    if dup := sorted(set.intersection(*tags) if len(tags) > 1 else set()):
      return _fail(f"session-map: window {', '.join(dup)} is tagged in more than one --session record: a window names one traced run's turns")
    turns = [t for r in recs for t in r]
    untraced = [r for u in args.untraced or [] for r in runs(u)]
    out = sm.session_map([{"path": p, "window": w} for p, _, w in caps], turns, main=load(args.model), draft=load(args.draft), engine=args.engine,
                         vendor=args.vendor, target=target, k=args.k, kv_type=args.kv_type, selection=args.selection, manifest=load(args.manifest), routing=load(args.routing),
                         lowering=_lowering(args.lowering), untraced=untraced or None, traced_runs=recs, reference=(load(args.reference) or {}).get("rows") if args.reference else None,
                         draft_selection=args.draft_selection, previous=load(args.previous), floors=(load(args.floors) or {}).get("rows"), first=args.first,
                         h2d_gbs=args.h2d_gbs or _saved_h2d(), name=args.name or (pathlib.Path(args.out).stem if args.out else None))
  except (KeyError, ValueError) as exc:
    return _fail(f"session-map: {type(exc).__name__}: {exc}")
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=sorted) + "\n")
  print("\n".join(sm.lines(out, args.top)))
  return 0 if out.get("ok") else 1


def cmd_span_saving(args) -> int:
  from boltbeam.collectors import time_budget as tb
  from boltbeam.search import span_saving as ss
  if not pathlib.Path(args.capture).is_file():
    return _fail(f"span-saving: no such file: {args.capture}")
  if (args.candidate_us is None) != (args.candidate_source is None) or (args.per_step is not None and args.candidate_us is None):
    return _fail("span-saving: --candidate-us takes --candidate-source (the bench file, or what the estimate is), and --per-step needs both")
  tl = tb.timeline_from_nsys(args.capture)
  s = ss.summary(ss.spans(tl.kernels, tl.grids, args.run), pathlib.Path(args.capture).name)
  rec = {"schema": ss.SCHEMA, "source": pathlib.Path(args.capture).name, "run": args.run, "spans": s}
  if args.stock_for:  # the candidate and PASS:SHAPE the run stands for, written into the record
    rec |= dict(zip(("candidate", "at"), args.stock_for))
  if args.candidate_us is not None and s.get("occurrences"):
    rec["prediction"] = ss.predict(s, args.candidate_us, args.candidate_source, args.per_step, "estimate" if args.candidate_estimate else "measured")
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
  print("\n".join(ss.lines(rec)))
  return 0 if s.get("occurrences") else 1


def _register_span_saving(sub) -> None:
  p = sub.add_parser("span-saving", help="what one candidate kernel replacing a run of stock launches saves in the model: the run's measured wall "
                     "(PDL overlap counted once, idle gaps in) minus the candidate's time, beside the summed-time projection (boltbeam.span_saving.v1); exits 1 when the run never occurs")
  p.add_argument("capture", help="an nsys sqlite export")
  p.add_argument("--run", action="append", required=True, help="one launch of the run, in order: a regular expression searched in kernel names (anchor it with ^...$), NAME@gridX to key the grid; repeatable")
  p.add_argument("--candidate-us", type=float, default=None, help="the candidate's in-model time a copy, us (its bench, back-to-back copies in one graph)")
  p.add_argument("--candidate-source", default=None, help="where the candidate's time comes from (a run folder file, or the model that estimates it)")
  p.add_argument("--candidate-estimate", action="store_true", help="the candidate's time is an estimate (a model's figure), not a bench measurement")
  p.add_argument("--per-step", type=float, default=None, help="occurrences a decode step (or a turn): gives the saving a step")
  p.add_argument("--stock-for", nargs=2, default=None, metavar=("HASH", "PASS:SHAPE"), help="the candidate (its hash) whose stock side this run is, and its pass and shape: written into the record")
  p.add_argument("--out", default=None, help="write the record here (JSON)")
  p.set_defaults(fn=cmd_span_saving)


def _register_session_map(sub) -> None:
  p = sub.add_parser("session-map", help="per kernel class of a traced session, for the decode step and the prefill turn: now, floor, reference ceiling, "
                     "gap in session seconds, and whether the search can express it (boltbeam.session_map.v1; docs/session-map.md); exits 1 when "
                     "more than 5% of a step is unattributed or a count check fails")
  p.add_argument("capture", nargs="+", help="an nsys sqlite export as PATH@WINDOW: the window the session record tags its turns with")
  p.add_argument("--session", action="append", required=True, help="a traced run's driver record (turns with span_ns, win and llama-server's timings); "
                 "repeatable when the captures come from several traced runs, each tagging its own windows")
  p.add_argument("--model", required=True, help="the main model's header table ({meta, tensors: [name, dims, type id]})")
  p.add_argument("--draft", default=None, help="the draft head's header table")
  p.add_argument("--lowering", default=None, help=LOWERING_HELP + " (the map's can_express for float rows)")
  p.add_argument("--engine", action="append", required=True, help="the engine's rows (data/kernel_families.json, data/kernel_classes.json): a fork, then its root")
  p.add_argument("--vendor", default=None, help="the GPU vendor's family rows (nvidia, amd)")
  p.add_argument("--target", required=True, help="the chip (data/targets.json): its measured bandwidth, matrix rates and shapes facts")
  p.add_argument("--k", type=int, required=True, help="draft tokens a step (0 without a draft)")
  p.add_argument("--kv-type", default=None, help="the KV cache's format (Q8_0, F16, ...)")
  p.add_argument("--selection", type=int, default=None, help="the KV cells a verify attention launch reads when the engine runs sparse attention")
  p.add_argument("--manifest", default=None, help="the kernel manifest the engine ran with (boltbeam.kernel_manifest.v1): its kernels are joined by what they compute")
  p.add_argument("--routing", default=None, help="the model's measured routing profile (boltbeam.routing_profile.v1, `boltbeam routing`): expert bytes "
                 "from the experts a verify reaches under it, not the upper estimate")
  p.add_argument("--untraced", action="append", help="the same session untraced: a driver record, or session-ab's record as PATH@ARM; repeatable. "
                 "Gives the untraced step per window and the session's steps")
  p.add_argument("--reference", default=None, help="measured reference rows (JSON {rows: [{part (decode, prefill), phase, classes (or [\"*\"]), ms or gbs, source, window?}]})")
  p.add_argument("--draft-selection", type=int, default=None, help="the KV cells a draft attention launch reads, when the draft's attention is sparse too")
  p.add_argument("--previous", default=None, help="the audit's stock proof: an earlier session-map record of the same plan; the classes flat across this "
                 "map's depths must match it within the derived tolerance, or the queue is not ranked")
  p.add_argument("--first", action="store_true", help="rank with no --previous (this plan's first audit; the record says so)")
  p.add_argument("--floors", default=None, help="bytes a class reads that the header does not give (JSON {rows: [{class, bytes, depth?, source}]}; "
                 "scaled by depth when depth names the depth they were counted at)")
  p.add_argument("--h2d-gbs", type=float, default=None, help="the GPU's measured host-to-device copy rate (default: this machine's saved copy probe, "
                 "target/host.h2d, when it holds one GPU)")
  p.add_argument("--name", default=None, help="this record's name, for a later --previous")
  p.add_argument("--top", type=int, default=14, help="rows shown per part (the record holds all)")
  p.add_argument("--out", default=None, help="write the record here (JSON)")
  p.set_defaults(fn=cmd_session_map)


def register(sub) -> None:
  _register_time_budget(sub)
  _register_session_map(sub)
  _register_span_saving(sub)
  p = sub.add_parser("kernel-ledger", help="every kernel of a decode step by shape: launches, exclusive time, bytes, limit, distance, in two rankings (bytes known, "
                     "bound unknown), with search and install coverage (boltbeam.kernel_ledger.v1; docs/kernel-ledger.md); exits 1 when a check fails")
  p.add_argument("capture", nargs="+", help="an nsys sqlite export or its extract (JSON[.gz]), as PATH or PATH@DEPTH (tokens of context); two depths give slopes")
  p.add_argument("--model", required=True, help="the main model's header table ({meta, tensors: [name, dims, type id]})")
  p.add_argument("--draft", default=None, help="the draft head's header table, for the draft passes' join")
  p.add_argument("--engine", action="append", required=True, help="the engine's rows (data/kernel_families.json): a fork, then its root; repeatable")
  p.add_argument("--vendor", default=None, help="the GPU vendor's family rows (nvidia, amd)")
  p.add_argument("--target", required=True, help="the chip (python -m boltbeam.workflow.screen targets): its measured bandwidth and its shapes facts")
  p.add_argument("--k", type=int, required=True, help="draft tokens a step: the verify runs k + 1 columns (0 without a draft)")
  p.add_argument("--kv-type", default=None, help="the KV cache's format (Q8_0, F16, ...): attention bytes and flash_space's coverage")
  p.add_argument("--selection", type=int, default=None, help="the KV cells a verify attention launch reads when the engine runs sparse attention")
  p.add_argument("--untraced-ms", type=float, action="append", help="the same turns' step untraced, ms, once per capture: discloses the trace's overhead")
  p.add_argument("--counters", action="append", help="JSON [{tokens, accepted, drafted}] of the capture's turns (the engine's counters), once per capture")
  p.add_argument("--lowering", default=None, help=LOWERING_HELP)
  p.add_argument("--session", default=None, help="the traced run's driver record (turns with span_ns, win, llama-server's timings): only the turns' "
                 "decode windows are read, so a capture that holds prefills is cut right; the counters are the turns'")
  p.add_argument("--window", action="append", help="with --session, the window each capture's turns are tagged with, once per capture")
  p.add_argument("--manifest", default=None, help="the kernel manifest the engine ran with (boltbeam.kernel_manifest.v1): its kernels joined by what they compute")
  p.add_argument("--routing", default=None, help="the model's measured routing profile: expert bytes from the experts a verify reaches under it")
  p.add_argument("--top", type=int, default=20, help="rows shown per ranking (the record holds all)")
  p.add_argument("--out", default=None, help="write the record here (JSON)")
  p.set_defaults(fn=cmd_kernel_ledger)
  p = sub.add_parser("context-audit", help="a session by context depth, host time as a role, the memory margin and ledger, each setting's output label, "
                     "and the checks that must reconcile (boltbeam.context_audit.v1); exits 1 when one fails")
  p.add_argument("record", help="the audit's input (JSON): context, turns, configuration, repeats and any of memory, ledger, host_points, host_functions (the host-function timer's records, optional captures), roles, arms (run folders), expert_cache (docs/context-audit.md)")
  p.add_argument("--engine", action="append", required=True, help="the engine's rows (data/kernel_families.json, data/engine_knobs.json): a fork, then its root; repeatable")
  p.add_argument("--out", default=None, help="write the record here (JSON)")
  p.set_defaults(fn=cmd_context_audit)
  _register_schedule_trace(sub)
  _register_import_hw_trace(sub)
  _register_import_ncu(sub)
  _register_collect_hw_trace(sub)
  _register_metal_measure(sub)
  _register_metal_bandwidth(sub)
  _register_compare_hw_trace(sub)
  _register_compare_substrate(sub)
  _register_profiler_report(sub)
  _register_prefill_role_trace(sub)
