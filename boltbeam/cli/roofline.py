from __future__ import annotations

import json
import pathlib

from boltbeam.cli._common import _write, _fail, _load_json, _profile, _add_common
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

def _register_roofline(sub) -> None:
  p = sub.add_parser("roofline", help="ingest a per-kernel roofline trace -> whole-model %%-of-HBM-peak + attribute the gap (gemv_codegen_capped/elementwise_dilution/latency_bound) with reclaimable us + fix")
  p.add_argument("trace", help="roofline trace JSON from tinygrad extra/qk_roofline_trace.py (per-kernel us + phys_bytes + kind, whole-step total_bytes)")
  p.add_argument("--peak", type=float, default=DEFAULT_PEAK_MEM_GBS, help="HBM peak GB/s (gfx1100 XTX = 960)")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_roofline)



def cmd_roofline(args) -> int:
  from boltbeam.roofline.roofline_trace import ingest_roofline
  _write(ingest_roofline(_load_json(args.trace), peak_gbs=args.peak), args.out)
  return 0

def _register_roofline_plan(sub) -> None:
  p = sub.add_parser("roofline-plan", help="START HERE: derive the achievable-ceiling roofline, place the model, work DOWN to per-role regime+lever (model-size-aware int-dot prize)")
  p.add_argument("plan", help="plan JSON: legacy {weight_bytes,...}; or GGUF decode {gguf, raw_bandwidth_gbs?, raw_compute_tflops?, practical_streaming?, measured_tok_s?|measurements?}")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_roofline_plan)



def cmd_roofline_plan(args) -> int:
  from boltbeam.roofline.roofline_plan import derive_decode_roofline, derive_roofline_plan
  p = _load_json(args.plan)
  if p.get("gguf"):
    from boltbeam.profile.decode_roles import decode_roofline_inventory_from_gguf
    inventory = decode_roofline_inventory_from_gguf(p["gguf"], model_id=p.get("model_id"))
    _write(derive_decode_roofline(inventory=inventory, raw_bandwidth_gbs=p.get("raw_bandwidth_gbs"),
      raw_compute_tflops=p.get("raw_compute_tflops"), practical_streaming=p.get("practical_streaming"),
      measured_tok_s=p.get("measured_tok_s"), measurements=p.get("measurements")), args.out)
    return 0
  _write(derive_roofline_plan(weight_bytes=p["weight_bytes"], kv_bytes=p.get("kv_bytes", 0.0),
    achieved_gbs=p["achieved_gbs"], measured_ms_per_token=p["measured_ms_per_token"], roles=p["roles"],
    dequant_tax_frac=p.get("dequant_tax_frac", 0.14), raw_peak_gbs=p.get("raw_peak_gbs")), args.out)
  return 0

def _register_prefill_roofline(sub) -> None:
  p = sub.add_parser("prefill-roofline", help="derive a practical packed-prefill roofline from baseline/candidate traces")
  p.add_argument("--baseline", required=True, help="baseline boltbeam.timing_trace.v1/hw_trace.v1 JSON, usually llama.cpp")
  p.add_argument("--candidate", required=True, help="candidate trace with kernel attribution/resources")
  p.add_argument("--wall-trace", default=None, help="optional clean timing trace used to rescale profile attribution")
  p.add_argument("--context", type=int, default=None, help="context length to compare; defaults to the candidate's first context")
  p.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  p.add_argument("--out", default=None, help="write roofline report JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_prefill_roofline)



def cmd_prefill_roofline(args) -> int:
  from boltbeam.roofline.prefill_roofline import prefill_roofline_markdown, prefill_roofline_report
  try:
    report = prefill_roofline_report(
      _load_json(args.baseline),
      _load_json(args.candidate),
      context=args.context,
      peak_gbs=args.peak_gbs,
      wall_trace=_load_json(args.wall_trace) if args.wall_trace else None,
    )
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"prefill-roofline: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(prefill_roofline_markdown(report))
  return 0

def _register_prefill_roofline_ladder(sub) -> None:
  p = sub.add_parser("prefill-roofline-ladder", help="rank packed prefill kernels from raw-HBM floors through llama practical rate")
  p.add_argument("--baseline", required=True, help="baseline boltbeam.timing_trace.v1/hw_trace.v1 JSON, usually llama.cpp")
  p.add_argument("--candidate", required=True, help="candidate trace with kernel attribution/resources")
  p.add_argument("--wall-trace", default=None, help="optional clean timing trace used to rescale profile attribution")
  p.add_argument("--context", type=int, default=None, help="context length to compare; defaults to the candidate's first context")
  p.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  p.add_argument("--scalar-tflops", type=float, default=None,
                 help="optional calibrated scalar/vector FMA throughput; omitted means no scalar-FMA timing floor")
  p.add_argument("--dequant-gops", type=float, default=None,
                 help="optional calibrated dequant/unpack throughput; omitted means no dequant timing floor")
  p.add_argument("--q4-dequant-ops-per-weight", type=float, default=8.0,
                 help="estimated unpack/dequant ops per Q4_K weight value for optional dequant floor")
  p.add_argument("--q6-dequant-ops-per-weight", type=float, default=10.0,
                 help="estimated unpack/dequant ops per Q6_K weight value for optional dequant floor")
  p.add_argument("--out", default=None, help="write roofline ladder JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_prefill_roofline_ladder)



def cmd_prefill_roofline_ladder(args) -> int:
  from boltbeam.roofline.prefill_roofline_ladder import prefill_roofline_ladder_markdown, prefill_roofline_ladder_report
  try:
    report = prefill_roofline_ladder_report(
      _load_json(args.baseline),
      _load_json(args.candidate),
      context=args.context,
      peak_gbs=args.peak_gbs,
      wall_trace=_load_json(args.wall_trace) if args.wall_trace else None,
      scalar_tflops=args.scalar_tflops,
      dequant_gops=args.dequant_gops,
      q4_dequant_ops_per_weight=args.q4_dequant_ops_per_weight,
      q6_dequant_ops_per_weight=args.q6_dequant_ops_per_weight,
    )
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"prefill-roofline-ladder: {exc}")
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(prefill_roofline_ladder_markdown(report))
  return 0

def _register_roofline_theoretical(sub) -> None:
  p = sub.add_parser("roofline-theoretical",
                     help="theoretical compute/memory roofline per GEMM role from a GGUF + target (no GPU)")
  _add_common(p)
  p.add_argument("--context", type=int, default=512, help="prefill token dimension (M)")
  p.add_argument("--dtype", default="fp16", help="matrix-engine dtype key into the target's peak_tflops")
  p.add_argument("--peak-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS, help="fallback HBM peak GB/s if target lacks one")
  p.add_argument("--peak-tflops", type=float, default=None,
                 help="measured compute ceiling in TFLOP/s; required for targets whose registry row "
                      "carries no peak_tflops (descriptor-only targets)")
  p.add_argument("--tiered", action="store_true",
                 help="cache-aware roofline: step the memory ceiling by working-set residency (L2/Infinity Cache/DRAM)")
  p.add_argument("--throughput", action="store_true",
                 help="roll the per-role roofline up to decode + prefill tok/s with a per-role breakdown")
  p.add_argument("--execution-profile", default=None,
                 help="route facts JSON: {prefill:{weight_bits,tile_m,transport,roles...}, decode:{...}}")
  p.add_argument("--memory-profile", default=None,
                 help="optional memory_hierarchy_profile from ingest-mem-sweep; replaces modeled tier bandwidths")
  p.set_defaults(fn=cmd_roofline_theoretical)



def _resolve_peak_flops(target, dtype:str, override_tflops:float | None) -> float:
  """Compute ceiling in FLOP/s, or a stated refusal.

  A target whose registry row carries no `peak_tflops` has no compute ceiling to place a
  model against. Descriptor-only rows are exactly that case, and the registry already says
  so via `backend_status`. Reporting the absence beats both crashing on an empty `max()`
  and inventing a vendor figure — "a capability a backend lacks is stated explicitly,
  never faked".

  Precedence: what the caller measured, then what the row measured on its matrix unit, then the
  row's ALU rates. A measured figure outranks a sheet figure for the same silicon.
  """
  if override_tflops is not None:
    if override_tflops <= 0: raise SystemExit("--peak-tflops must be positive")
    return override_tflops * 1e12
  measured = target.matrix_tflops_for(dtype)
  if measured:
    return measured * 1e12
  peaks = target.peak_tflops or {}
  value = peaks.get(dtype) or peaks.get("fp16") or (max(peaks.values()) if peaks else None)
  if value:
    return value * 1e12
  raise SystemExit(
    f"target {target.target_id!r} carries no peak_tflops, so there is no compute ceiling to "
    f"place this model against (backend_status={target.backend_status!r}).\n"
    f"Supply a measured ceiling with --peak-tflops <TFLOP/s>, or add one to "
    f"boltbeam/data/targets.json once it has been measured on the hardware.")


def cmd_roofline_theoretical(args) -> int:
  from boltbeam.kernel_analysis.theoretical_roofline import (
    calibrated_memory_tiers, model_roofline, registered_memory_tiers, throughput_rollup, tiered_model_roofline,
  )
  profile, target = _profile(args)
  execution = _load_json(args.execution_profile) if args.execution_profile else {}
  if execution.get("model_id") not in (None, profile.model_id):
    raise ValueError(f"execution profile model_id={execution.get('model_id')!r} does not match {profile.model_id!r}")
  peak_flops = _resolve_peak_flops(target, args.dtype, args.peak_tflops)
  bw = target.memory_bandwidth_gbs or args.peak_gbs
  if args.throughput:
    tiers = (calibrated_memory_tiers(target, _load_json(args.memory_profile), dram_gbs=bw)
             if args.memory_profile else registered_memory_tiers(target, dram_gbs=bw))
    report = throughput_rollup(profile.to_json(), peak_flops=peak_flops, tiers=tiers,
                               prefill_context=args.context, prefill_execution=execution.get("prefill"),
                               decode_execution=execution.get("decode"))
  elif args.tiered:
    tiers = (calibrated_memory_tiers(target, _load_json(args.memory_profile), dram_gbs=bw)
             if args.memory_profile else registered_memory_tiers(target, dram_gbs=bw))
    report = tiered_model_roofline(profile.to_json(), peak_flops=peak_flops, tiers=tiers,
                                   context=args.context, execution=execution.get("prefill") or execution or None)
  else:
    report = model_roofline(profile.to_json(), peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9,
                            context=args.context)
  _write(report, args.out)
  return 0


def register(sub) -> None:
  _register_roofline(sub)
  _register_roofline_plan(sub)
  _register_prefill_roofline(sub)
  _register_prefill_roofline_ladder(sub)
  _register_roofline_theoretical(sub)
