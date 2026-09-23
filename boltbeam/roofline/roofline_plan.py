"""Roofline-FIRST decode analysis: start from the achievable ceiling, then work DOWN to the per-role levers.

This is the entry point of a decode-perf investigation, codifying the method that produced the Q4_K findings:
derive the roofline first, place the measured step against it, attribute the gap by role, classify each role's
REGIME (what actually binds it), and emit the lever for that regime — with the levers ranked by Amdahl whole-gain
and the weight-read/int-dot prize made MODEL-SIZE-AWARE (it grows with the streaming-bound weight-byte share).

Three hard lessons baked in (each cost real measurement to learn):
  1. Denominator = ACHIEVED, WORKLOAD-COMPARABLE streaming bandwidth, NOT the raw HBM peak. A dequant GEMV can't
     reach raw peak; a runtime's logical GB/s over-counts a 4-bit dequant (>100% of peak, meaningless). An aggregate
     read+write copy remains a useful proxy, but it is not a weight-read roof until its traffic scope is comparable.
  2. Regime decides the lever, not the kernel name:
       - streaming-bound weight-read GEMV  -> the int-dot / no-decode dequant lever (removes the ~14% decode tax);
       - latency-bound attention (reduce/softmax, tiny KV bytes) -> NOT a bandwidth lever, it's near its own floor;
       - launch-bound elementwise (norm/rope/act, ~no weight bytes) -> graph capture + fusion, not a kernel rewrite.
  3. The int-dot prize SCALES with model size: per-token latency ~ model_bytes / bandwidth, and every
     streaming-bound weight matrix pays the decode tax as direct bandwidth loss. Its whole-decode gain is
     dequant_tax applied to the weight-GEMV time share — which is small on a model whose GEMVs are already
     near-ceiling and large on a bigger model where more of the step is streaming-bound weight reads.
"""
from __future__ import annotations
import math
from typing import Any, Mapping

from boltbeam.math.roofline import floor_ms_per_token, ceiling_tok_s, amdahl_whole_gain
from boltbeam.math.roofline import dual_roofline_placement


def _finite_positive(name:str, value:Any, *, allow_none:bool = False) -> float | None:
  if value is None and allow_none: return None
  try: number = float(value)
  except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc
  if not math.isfinite(number) or number <= 0:
    raise ValueError(f"{name} must be a finite positive number, got {value!r}")
  return number


def _finite_nonnegative(name:str, value:Any) -> float:
  try: number = float(value)
  except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be a finite non-negative number, got {value!r}") from exc
  if not math.isfinite(number) or number < 0:
    raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
  return number


_PLACEMENT_DERIVED_FIELDS = frozenset(("achieved_weight_equivalent_gbs", "achieved_semantic_tflops",
                                       "pct_of_raw_bandwidth_roof", "pct_of_raw_compute_roof",
                                       "pct_of_practical_bandwidth_roof"))
_PRACTICAL_DERIVED_FIELDS = frozenset(("bandwidth_floor_ms_per_token", "bandwidth_ceiling_tok_s",
                                       "practical_roof_status"))


def _placement_metrics(tok_s:float, *, weight_bytes:float, flops:float, raw_bandwidth_gbs:float | None,
                       raw_compute_tflops:float | None, practical_gbs:float | None) -> dict[str, float]:
  out = {"achieved_weight_equivalent_gbs": weight_bytes * tok_s / 1e9,
         "achieved_semantic_tflops": flops * tok_s / 1e12}
  if raw_bandwidth_gbs is not None:
    out["pct_of_raw_bandwidth_roof"] = 100.0 * out["achieved_weight_equivalent_gbs"] / raw_bandwidth_gbs
  if raw_compute_tflops is not None:
    out["pct_of_raw_compute_roof"] = 100.0 * out["achieved_semantic_tflops"] / raw_compute_tflops
  if practical_gbs is not None:
    out["pct_of_practical_bandwidth_roof"] = 100.0 * out["achieved_weight_equivalent_gbs"] / practical_gbs
  return out


def _named_measurements(value:Mapping[str, Any] | None, *, weight_bytes:float, flops:float,
                        raw_bandwidth_gbs:float | None, raw_compute_tflops:float | None,
                        practical_gbs:float | None) -> dict[str, dict[str, Any]]:
  """Preserve caller-owned whole-model identity while deriving shared placement math once."""
  if value is None: return {}
  if not isinstance(value, Mapping): raise ValueError("measurements must be an object keyed by measurement name")
  out: dict[str, dict[str, Any]] = {}
  scopes: set[str] = set()
  if any(not isinstance(name, str) or not name for name in value):
    raise ValueError("measurement names must be non-empty strings")
  for name in sorted(value):
    record = value[name]
    if not isinstance(record, Mapping): raise ValueError(f"measurements.{name} must be an object")
    if _PLACEMENT_DERIVED_FIELDS & record.keys():
      raise ValueError(f"measurements.{name} contains caller-supplied derived placement fields")
    scope = record.get("scope")
    if not isinstance(scope, str) or not scope:
      raise ValueError(f"measurements.{name}.scope must be a non-empty string")
    scopes.add(scope)
    tok_s = _finite_positive(f"measurements.{name}.tok_s", record.get("tok_s"))
    out[name] = {**record, "tok_s": tok_s,
                 **_placement_metrics(tok_s, weight_bytes=weight_bytes, flops=flops,
                                      raw_bandwidth_gbs=raw_bandwidth_gbs,
                                      raw_compute_tflops=raw_compute_tflops, practical_gbs=practical_gbs)}
  if len(scopes) > 1:
    raise ValueError("named whole-model measurements must share one matched workload scope")
  return out


def derive_decode_roofline(*, inventory:dict[str, Any], raw_bandwidth_gbs:float | None = None,
                           raw_compute_tflops:float | None = None, practical_streaming:dict[str, Any] | None = None,
                           measured_tok_s:float | None = None,
                           measurements:Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Place a GGUF-derived decode inventory on raw and explicitly comparable practical roofs.

  This is intentionally a thin report composer over the shared constant-free
  roofline math. Target facts and all measurements remain caller supplied. A
  stream requires explicit scope/comparability before it becomes a denominator;
  named whole-model measurements retain their caller-owned scope metadata.
  """
  weight_bytes = _finite_positive("inventory.packed_weight_bytes_per_token", inventory["packed_weight_bytes_per_token"])
  flops = _finite_nonnegative("inventory.dense_matrix_flops_per_token", inventory["dense_matrix_flops_per_token"])
  raw_bandwidth_gbs = _finite_positive("raw_bandwidth_gbs", raw_bandwidth_gbs, allow_none=True)
  raw_compute_tflops = _finite_positive("raw_compute_tflops", raw_compute_tflops, allow_none=True)
  measured_tok_s = _finite_positive("measured_tok_s", measured_tok_s, allow_none=True)
  if measured_tok_s is not None and measurements is not None:
    raise ValueError("use either legacy measured_tok_s or named measurements, not both")
  raw: dict[str, Any] = {"level": "raw_hardware", "bandwidth_gbs": raw_bandwidth_gbs,
                          "compute_tflops": raw_compute_tflops,
                          "bandwidth_basis": "advertised/observed target fact" if raw_bandwidth_gbs else "unknown",
                          "compute_basis": "independently sourced/observed target fact" if raw_compute_tflops else "unknown"}
  if raw_bandwidth_gbs is not None:
    raw["bandwidth_floor_ms_per_token"] = floor_ms_per_token(weight_bytes, raw_bandwidth_gbs * 1e9)
    raw["bandwidth_ceiling_tok_s"] = ceiling_tok_s(weight_bytes, raw_bandwidth_gbs * 1e9)
  if raw_compute_tflops is not None:
    raw["compute_floor_ms_per_token"] = flops / (raw_compute_tflops * 1e12) * 1000.0
    raw["compute_ceiling_tok_s"] = raw_compute_tflops * 1e12 / flops if flops else None
  practical_gbs = None
  practical = {"level": "practical_streaming", "status": "unknown", "practical_roof_status": "unknown",
               "reason": "sustained same-session stream measurement has not been supplied"}
  if practical_streaming is not None:
    if not isinstance(practical_streaming, Mapping): raise ValueError("practical_streaming must be an object")
    if _PRACTICAL_DERIVED_FIELDS & practical_streaming.keys():
      raise ValueError("practical_streaming contains caller-supplied derived roofline fields")
    scope = practical_streaming.get("scope")
    if not isinstance(scope, str) or not scope:
      raise ValueError("practical_streaming.scope must be a non-empty string")
    comparable = practical_streaming.get("workload_comparable")
    if not isinstance(comparable, bool):
      raise ValueError("practical_streaming.workload_comparable must be an explicit boolean")
    practical = {"level": "practical_streaming", **practical_streaming}
    gbs = practical.get("sustained_gbs")
    if gbs is not None:
      gbs = _finite_positive("practical_streaming.sustained_gbs", gbs)
      practical["sustained_gbs"] = gbs
      practical.setdefault("status", "measured")
      if comparable:
        practical_gbs = gbs
        practical.update({"practical_roof_status": "measured_workload_comparable",
                          "bandwidth_floor_ms_per_token": floor_ms_per_token(weight_bytes, gbs * 1e9),
                          "bandwidth_ceiling_tok_s": ceiling_tok_s(weight_bytes, gbs * 1e9)})
      else:
        practical["practical_roof_status"] = "not_workload_comparable"
        practical.setdefault("reason", "measured streaming proxy is not comparable to the workload byte scope")
    else:
      practical.setdefault("status", "incomplete")
      practical["practical_roof_status"] = "incomplete"
      practical.setdefault("reason", "sustained_gbs is absent")
  placement: dict[str, Any] = {"level": "workload_placement", "packed_weight_bytes_per_token": weight_bytes,
                                "semantic_flops_per_token": flops,
                                "arithmetic_intensity_flop_per_byte": flops / weight_bytes,
                                "traffic": {"activation_bytes": None, "kv_bytes": None, "intermediate_bytes": None},
                                "role_contributions": inventory["role_contributions"],
                                "provenance": inventory["provenance"], "caveats": inventory["caveats"]}
  if measured_tok_s is not None:
    placement["measured_tok_s"] = measured_tok_s
    placement.update(_placement_metrics(measured_tok_s, weight_bytes=weight_bytes, flops=flops,
                                        raw_bandwidth_gbs=raw_bandwidth_gbs,
                                        raw_compute_tflops=raw_compute_tflops, practical_gbs=practical_gbs))
  named = _named_measurements(measurements, weight_bytes=weight_bytes, flops=flops,
                              raw_bandwidth_gbs=raw_bandwidth_gbs,
                              raw_compute_tflops=raw_compute_tflops, practical_gbs=practical_gbs)
  if named: placement["measurements"] = named
  return {"schema": "boltbeam.decode_roofline.v1", "raw_hardware": raw, "practical_streaming": practical,
          "workload_placement": placement}

# role kind -> (regime, lever). The caller tags each role by kind (from a per-kernel trace classifier).
_REGIME = {
  "weight_read_gemv": ("streaming_bound",
    "int-dot / no-decode dequant: dot the packed 4-bit codes directly (int8 activations, scale once per group), "
    "removing the ~14% decode tax. Payoff SCALES with model size (weight-byte share). Route shape-selectively — "
    "only where the GEMV is genuinely streaming-bound; keep the float path where it's already near-ceiling."),
  "attention": ("latency_bound",
    "reduce/softmax-bound, tiny KV byte share -> NOT a bandwidth lever; it sits near its own latency floor. "
    "Do not chase bandwidth here."),
  "elementwise": ("launch_bound",
    "moves ~no weight bytes -> launch/latency dead time. Capture the whole step in a graph + fuse adjacent "
    "small linears / fold quantize+reduce into producers. Benefits the entire step, both paths."),
  "reduce": ("launch_bound", "small reduce moving ~no weight bytes -> graph capture + fusion."),
}


def derive_roofline_plan(*, weight_bytes: float, kv_bytes: float, achieved_gbs: float,
                         measured_ms_per_token: float, roles: list[dict[str, Any]],
                         dequant_tax_frac: float = 0.14, raw_peak_gbs: float | None = None) -> dict[str, Any]:
  """START HERE. weight_bytes/kv_bytes read per token; achieved_gbs = measured streaming copy (NOT raw peak);
  roles = [{name, kind in {weight_read_gemv,attention,elementwise,reduce}, bytes, ms}]. Returns the ceiling, the
  placement, the per-role regime+lever, and the levers ranked by Amdahl whole-gain (int-dot prize model-size-aware)."""
  bw = achieved_gbs * 1e9
  stream_bytes = weight_bytes + kv_bytes
  floor_ms = floor_ms_per_token(stream_bytes, bw)
  ceil_toks = ceiling_tok_s(stream_bytes, bw)
  measured_toks = 1000.0 / measured_ms_per_token
  pct_of_ceiling = round(100.0 * floor_ms / measured_ms_per_token, 1)
  dual = None
  if raw_peak_gbs is not None:
    raw_ceil = ceiling_tok_s(stream_bytes, raw_peak_gbs * 1e9)
    dual = dual_roofline_placement(
      measured=measured_toks, raw_ceiling=raw_ceil, practical_ceiling=ceil_toks, metric="tok_s",
      raw_basis=f"raw target memory bandwidth ({raw_peak_gbs} GB/s)",
      practical_basis=f"measured achievable streaming bandwidth ({achieved_gbs} GB/s)",
    )

  total_ms = sum(r.get("ms", 0.0) for r in roles) or measured_ms_per_token
  rows, levers = [], []
  for r in roles:
    kind = (r.get("kind") or "elementwise").lower()
    regime, lever = _REGIME.get(kind, ("launch_bound", _REGIME["elementwise"][1]))
    ms, rb = r.get("ms", 0.0), r.get("bytes", 0.0)
    time_share = ms / total_ms if total_ms else 0.0
    role_gbs = round((rb / 1e9) / (ms / 1e3), 1) if ms and rb else 0.0
    # the int-dot lever's whole-decode gain: remove the dequant tax on the streaming-bound weight-read time share
    whole_gain = None
    if regime == "streaming_bound" and 0 < time_share < 1:
      whole_gain = round(amdahl_whole_gain(time_share, 1.0 / (1.0 - dequant_tax_frac)), 3)
      levers.append({"role": r.get("name"), "regime": regime, "lever": lever,
                     "whole_decode_gain_x": whole_gain, "prize_pct": round(100 * (whole_gain - 1), 1),
                     "scales_with_model_size": True})
    elif regime == "launch_bound" and time_share > 0.03:
      levers.append({"role": r.get("name"), "regime": regime, "lever": lever,
                     "reclaimable_time_share_pct": round(100 * time_share, 1), "scales_with_model_size": False,
                     "caveat": "UPPER BOUND — launch dead-time is largely removed by whole-step graph capture (JIT); "
                               "measure JIT-timed, not eager, before chasing it."})
    rows.append({"name": r.get("name"), "kind": kind, "regime": regime, "ms": round(ms, 1),
                 "time_share_pct": round(100 * time_share, 1), "eff_GBps": role_gbs,
                 "pct_of_achieved": round(100 * role_gbs / achieved_gbs, 1) if role_gbs else 0.0})

  # rank: the DURABLE bandwidth-fundamental streaming-bound lever (int-dot) first, then launch-bound (graph-hidden
  # upper bound); within a regime, by whole-decode gain. This encodes the session finding that graphs already
  # capture most launch dead-time, so the int-dot lever is the real, durable roofline lever.
  _prio = {"streaming_bound": 2, "launch_bound": 1}
  levers.sort(key=lambda l: (_prio.get(l["regime"], 0),
                             l.get("whole_decode_gain_x", 1 + l.get("reclaimable_time_share_pct", 0) / 100)),
              reverse=True)
  out = {
    "schema": "boltbeam.roofline_plan.v1",
    "step_1_ceiling": {"stream_GB_per_tok": round(stream_bytes / 1e9, 3), "achieved_GBps": achieved_gbs,
                       "floor_ms_per_tok": round(floor_ms, 3), "ceiling_tok_s": round(ceil_toks, 1),
                       "note": "denominator is ACHIEVED streaming copy, not raw HBM peak"},
    "step_2_placement": {"measured_ms_per_tok": measured_ms_per_token, "measured_tok_s": round(measured_toks, 1),
                         "pct_of_achievable_ceiling": pct_of_ceiling,
                         "headroom_x": round(measured_ms_per_token / floor_ms, 2)},
    "step_3_roles": rows,
    "step_4_levers": levers,
    "dominant_lever": levers[0] if levers else None,
    "model_size_note": ("the int-dot/no-decode weight-GEMV prize grows with the weight-byte time share -> re-run this "
                        "on a larger model before concluding 'near-ceiling'; the same kernel that ties on a small "
                        "model wins on a bigger one where more of the step is streaming-bound weight reads."),
  }
  if dual is not None:
    out["dual_roofline"] = dual
  return out
