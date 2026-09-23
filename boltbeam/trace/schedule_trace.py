"""Ingest a scheduler trace and classify WHY a generated kernel is (in)efficient.

BoltBeam had ISA capture (what instructions a kernel emits) and reduce classification, but not the scheduler-
decision trace — the per-kernel time/bandwidth that reveals a bad SCHEDULE (an occupancy-starved fused reduce)
vs a MATERIALIZATION (a big elementwise a reduce then re-reads) vs a fused-OK kernel. This closes that gap so
the search can target the actual bottleneck (schedule opts vs fusion). Pure analysis.

Buckets each kernel and diagnoses the dominant one:
  occupancy_starved  a slow reduce at low achieved bandwidth -> needs tiling/UPCAST/LOCAL schedule opts
  materialization    a large elementwise that dominates -> fusion failure (should fold into its consumer)
  fused_ok           the dominant kernel already runs near the bandwidth ceiling
"""
from __future__ import annotations
from typing import Any
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

def ingest_schedule_trace(trace: dict[str, Any], peak_gbs: float = DEFAULT_PEAK_MEM_GBS) -> dict[str, Any]:
  kernels = trace.get("kernels", [])
  total_us = trace.get("total_us") or sum(k.get("us", 0) for k in kernels)
  if not kernels:
    return {"schema": "boltbeam.schedule_report.v1", "label": trace.get("label"), "diagnosis": "no_kernels"}
  dom = max(kernels, key=lambda k: k.get("us", 0))
  dom_share = round(100 * dom["us"] / total_us, 1) if total_us else 0.0
  gbs = dom.get("gbs") or 0.0
  bw_util = round(100 * gbs / peak_gbs, 1)
  if dom["kind"] == "reduce" and bw_util < 40:
    diag, fix = "occupancy_starved", ("the dominant reduce runs far below the bandwidth ceiling — it is fused but "
      "lacks tiling/occupancy; apply/search UPCAST + LOCAL schedule opts (not more fusion)")
  elif dom["kind"] == "elementwise":
    diag, fix = "materialization", ("a large elementwise dominates — it should fuse into its consumer reduce; "
      "remove the realize barrier / restructure so the producer streams into the reduce")
  elif bw_util >= 70:
    diag, fix = "fused_ok", "the dominant kernel is near the bandwidth ceiling; little schedule headroom left"
  else:
    diag, fix = "mixed", "no single dominant pathology; inspect the per-kernel breakdown"
  # fusion-failure flags: slow kernels well below the ceiling
  failures = [{"name": k["name"], "kind": k["kind"], "us": k["us"], "gbs": k.get("gbs"),
               "bw_util_pct": round(100 * (k.get("gbs") or 0) / peak_gbs, 1)}
              for k in kernels if k.get("us", 0) > 100 and (k.get("gbs") or 0) < 0.4 * peak_gbs]
  return {"schema": "boltbeam.schedule_report.v1", "label": trace.get("label"), "total_us": total_us,
          "dominant": {"name": dom["name"], "kind": dom["kind"], "us": dom["us"], "share_pct": dom_share,
                       "gbs": gbs, "bw_util_pct": bw_util},
          "diagnosis": diag, "fix": fix,
          "fusion_failures": sorted(failures, key=lambda x: -x["us"])}
