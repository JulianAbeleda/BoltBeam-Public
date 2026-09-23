"""Ingest a per-kernel decode/prefill trace and attribute the roofline gap: WHERE the model loses vs HBM peak.

Complements boltbeam/math/roofline.py (theoretical floor/ceiling/Amdahl/practical-roofline-score): that answers
"what is the ceiling"; this answers "of the measured gap to it, which KERNELS/buckets own it and what's the fix."
Parallel to boltbeam/schedule_trace.py (which classifies ONE kernel's schedule); this attributes the WHOLE step's
roofline gap across buckets. Model-agnostic, pure analysis, no tinygrad dependency.

CRITICAL: use PHYSICAL bytes (packed quantized weights actually read from HBM), NOT the logical/dequantized byte
counter a runtime may report -- a Q4_K GEMV's logical view over-counts (the dequant expands 4-bit -> f32), so a
logical GB/s reads >100% of peak and is meaningless for roofline. The trace producer must supply `phys_bytes` per
kernel (weight bytes for a GEMV) and the whole-step `total_bytes` (e.g. GlobalCounters over one step with packed
weights). This module then computes the physically-meaningful utilization and attributes the loss.

Roofline decomposition. ideal_us = total_bytes / peak is the minimum time if every byte moved at peak. The gap
(actual_us - ideal_us) is attributed to three buckets:
  gemv_codegen_capped   memory-bound kernels running below peak (weight reads at ~half peak) -> the dominant,
                        codegen lever (better tiling / dot lowering); loss = us - phys_bytes/peak.
  elementwise_dilution  kernels moving ~no weight bytes (norms/rope/act/residual/small reduce) whose time is pure
                        launch/latency dead time -> fold into the producer/consumer (fusion), reclaimable.
  latency_bound         kernels whose time far exceeds their byte-floor by nature (attention reduce/softmax) ->
                        NOT a bandwidth lever; near its own ceiling already.
"""
from __future__ import annotations
from typing import Any
from boltbeam.math.roofline import bandwidth_utilization
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS
from boltbeam.math import roofline as _roofline_math

# kind -> which loss bucket a sub-peak kernel belongs to. The producer tags each kernel's `kind`.
_MEMORY_KINDS = {"gemv", "gemm", "matvec", "weight"}          # weight-read bound -> gemv_codegen_capped
_LATENCY_KINDS = {"attention", "flash", "reduce_softmax"}     # inherently compute/latency bound
_OVERHEAD_KINDS = {"elementwise", "reduce", "norm", "rope", "activation", "copy", "cast"}


def _util(phys_bytes: float, us: float, peak_gbs: float) -> float:
  return bandwidth_utilization(phys_bytes, us, peak_gbs)


def ingest_roofline(trace: dict[str, Any], peak_gbs: float = DEFAULT_PEAK_MEM_GBS,
                    gemv_target: float = 0.85, dead_time_frac: float = 0.10) -> dict[str, Any]:
  """trace = {label?, total_us?, total_bytes, kernels:[{name, us, phys_bytes, kind}]}. Pure analysis.

  A non-weight kernel (norm/rope/act/reduce) is split by measured bandwidth: below `dead_time_frac` of peak it is
  launch/latency DEAD TIME (fusible -> elementwise_dilution); at or above it is REAL activation work
  (activation_bound, not free to fuse away). This avoids mislabeling a genuine reduce as reclaimable dilution."""
  kernels = trace.get("kernels", [])
  if not kernels:
    return {"schema": "boltbeam.roofline_report.v1", "label": trace.get("label"), "diagnosis": "no_kernels"}
  total_us = trace.get("total_us") or sum(k.get("us", 0.0) for k in kernels)
  total_bytes = trace.get("total_bytes") or sum(k.get("phys_bytes", 0.0) for k in kernels)
  ideal_us = _roofline_math.floor_us(total_bytes, peak_gbs) if peak_gbs else 0.0
  whole_util = round(100 * ideal_us / total_us, 1) if total_us else 0.0

  # per-kernel classification + loss (us above its own physical byte-floor)
  buckets = {"gemv_codegen_capped": 0.0, "elementwise_dilution": 0.0, "activation_bound": 0.0,
             "latency_bound": 0.0, "at_peak": 0.0}
  rows = []
  for k in kernels:
    us, pb, kind = k.get("us", 0.0), k.get("phys_bytes", 0.0), (k.get("kind") or "elementwise").lower()
    floor_us = _roofline_math.floor_us(pb, peak_gbs) if peak_gbs else 0.0
    loss = max(0.0, us - floor_us)
    util_f = _util(pb, us, peak_gbs)
    util = round(100 * util_f, 1)
    if kind in _MEMORY_KINDS:
      bucket = "at_peak" if util_f >= gemv_target else "gemv_codegen_capped"
    elif kind in _LATENCY_KINDS:
      bucket = "latency_bound"
    elif util_f < dead_time_frac:
      bucket = "elementwise_dilution"        # launch/latency-bound, moves ~no bytes -> fusible dead time
    else:
      bucket = "activation_bound"            # a real norm/rope/reduce doing genuine activation work
    # dilution's whole time is reclaimable dead time; other buckets' loss is time above the byte-floor
    buckets[bucket] += us if bucket == "elementwise_dilution" else loss
    rows.append({"name": k.get("name"), "us": round(us, 1), "pct_step": round(100 * us / total_us, 1) if total_us else 0.0,
                 "phys_util_pct": util, "kind": kind, "bucket": bucket, "loss_us": round(loss, 1)})
  rows.sort(key=lambda r: r["loss_us"], reverse=True)

  # rank the buckets by reclaimable time; each carries the actionable lever
  fixes = {
    "gemv_codegen_capped": "dominant weight-read GEMVs run below peak -> codegen lever: better tiling / dot lowering "
                           "(v_dot2), coalesced packed loads. This is the main roofline gap.",
    "elementwise_dilution": "tiny elementwise/reduce kernels at near-zero bandwidth -> pure launch/latency dead time; "
                            "fuse into the producer/consumer GEMV epilogue/prologue to reclaim it (fully reclaimable).",
    "activation_bound": "real norm/rope/reduce kernels doing genuine activation work at moderate bandwidth -> NOT free "
                        "dead time; reclaim only via better tiling/occupancy or folding into an adjacent reduce.",
    "latency_bound": "attention reduce/softmax is inherently latency-bound, near its own ceiling -> NOT a bandwidth "
                     "lever; do not chase here.",
    "at_peak": "already near the bandwidth ceiling.",
  }
  ranked = sorted(((round(v, 1), b) for b, v in buckets.items() if b != "at_peak" and v > 0), reverse=True)
  dominant = ranked[0][1] if ranked else "at_peak"
  # eager traces inflate launch-bound dilution: in a JIT/graph run those tiny kernels batch/overlap and their
  # per-kernel time (hence dilution) mostly disappears. Weight-read GEMV loss is JIT-persistent (read once, not
  # batch-hidden). So on an eager trace, treat elementwise_dilution as an UPPER BOUND and trust gemv_codegen_capped.
  timing = (trace.get("timing_source") or "unknown").lower()
  caveat = None
  if timing not in ("jit", "graph", "profile"):
    caveat = ("timing_source is not jit/profile -> eager per-kernel times inflate elementwise_dilution (launch-bound "
              "tiny kernels batch-hide in a graph run). Treat dilution as an upper bound; gemv_codegen_capped is "
              "JIT-persistent. Use a PROFILE=1 producer for accurate reclaimable-us.")
  return {
    "schema": "boltbeam.roofline_report.v1", "label": trace.get("label"), "peak_gbs": peak_gbs,
    "timing_source": timing, "caveat": caveat,
    "whole_model": {"total_us": round(total_us, 1), "total_GB": round(total_bytes / 1e9, 3),
                    "eff_GBps": round((total_bytes / 1e3) / total_us, 1) if total_us else 0.0,
                    "pct_of_peak": whole_util, "ideal_us": round(ideal_us, 1),
                    "headroom_x": round(total_us / ideal_us, 2) if ideal_us else None},
    "loss_by_bucket_us": {b: round(v, 1) for b, v in buckets.items()},
    "dominant_bucket": dominant, "dominant_fix": fixes[dominant],
    "ranked_levers": [{"bucket": b, "reclaimable_us": v, "fix": fixes[b]} for v, b in ranked],
    "kernels": rows[:24],
  }
