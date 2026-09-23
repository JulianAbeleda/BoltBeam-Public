from __future__ import annotations

from typing import Any

from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS
from boltbeam.roofline.prefill_roofline import (
  _context, _eff_gbs, _f, _floor_us, _is_packed_prefill, _kernel_rows, _scale_candidate,
  _trusted_baseline_rate, _whole,
)


def _shape(row:dict[str, Any]) -> tuple[int, int, int] | None:
  s = row.get("shape") or ()
  if len(s) != 3:
    return None
  try:
    return int(s[0]), int(s[1]), int(s[2])
  except (TypeError, ValueError):
    return None


def _calls(row:dict[str, Any]) -> int:
  try:
    return max(1, int(row.get("calls") or 1))
  except (TypeError, ValueError):
    return 1


def _logical_ops(row:dict[str, Any]) -> float | None:
  shp = _shape(row)
  if shp is None:
    return None
  m, n, k = shp
  return float(2 * m * n * k * _calls(row))


def _weights(row:dict[str, Any]) -> float | None:
  shp = _shape(row)
  if shp is None:
    return None
  _m, n, k = shp
  return float(n * k * _calls(row))


def _floor_from_ops(ops:float | None, rate_per_s:float | None) -> float | None:
  if ops is None or rate_per_s is None or rate_per_s <= 0:
    return None
  return ops / rate_per_s * 1e6


def _max_present(values:list[float | None]) -> float | None:
  present = [v for v in values if v is not None]
  return max(present) if present else None


def _classify(*, wall_us:float, theoretical_us:float | None, raw_us:float | None, reference_us:float | None,
              eff_gbs:float | None, llama_gbs:float | None) -> str:
  if not wall_us:
    return "missing_timing"
  if theoretical_us and wall_us / theoretical_us <= 1.25:
    return "near_theoretical_roofline"
  if reference_us and wall_us / reference_us <= 1.25:
    return "near_reference_impl"
  if llama_gbs and eff_gbs and eff_gbs < llama_gbs * 0.25:
    return "far_below_reference_impl_rate"
  if raw_us and wall_us / raw_us > 10.0:
    return "raw_hbm_not_binding"
  if reference_us:
    return "reference_impl_gap"
  return "uncalibrated_roofline"


def _fmt(value:Any) -> str:
  v = _f(value)
  return "" if v is None else f"{v:.3f}"


def prefill_roofline_ladder_report(
  baseline:dict[str, Any],
  candidate:dict[str, Any],
  *,
  context:int | None = None,
  peak_gbs:float = DEFAULT_PEAK_MEM_GBS,
  wall_trace:dict[str, Any] | None = None,
  scalar_tflops:float | None = None,
  dequant_gops:float | None = None,
  q4_dequant_ops_per_weight:float = 8.0,
  q6_dequant_ops_per_weight:float = 10.0,
) -> dict[str, Any]:
  """Build a per-kernel ladder: theoretical floors -> reference implementation -> candidate.

  The roofline rung is the maximum available theoretical floor: raw source-byte HBM plus
  optional scalar-FMA/dequant floors when calibrated rates are supplied. The llama rows are
  not the roofline; they are a measured reference implementation below that ceiling.
  """
  ctx = _context(candidate, context)
  cand = _scale_candidate(candidate, wall_trace, ctx)
  bwhole, cwhole = _whole(baseline, ctx), _whole(cand, ctx)
  brows, crows = _kernel_rows(baseline, ctx), _kernel_rows(cand, ctx)
  q4_rate = _trusted_baseline_rate(brows, quant="Q4_K", peak_gbs=peak_gbs)
  q4_gbs = q4_rate["eff_gbs"] if q4_rate else None
  scalar_ops_per_s = scalar_tflops * 1e12 if scalar_tflops and scalar_tflops > 0 else None
  dequant_ops_per_s = dequant_gops * 1e9 if dequant_gops and dequant_gops > 0 else None

  rows = []
  for row in crows:
    if not _is_packed_prefill(row):
      continue
    wall_us = _f(row.get("wall_us")) or 0.0
    bytes_ = _f(row.get("phys_bytes")) or 0.0
    eff = _eff_gbs(row)
    ops = _logical_ops(row)
    weights = _weights(row)
    dequant_ops_per_weight = q4_dequant_ops_per_weight if row.get("quant") == "Q4_K" else q6_dequant_ops_per_weight
    dequant_ops = weights * dequant_ops_per_weight if weights is not None else None
    raw_us = _floor_us(bytes_, peak_gbs) if bytes_ and peak_gbs else None
    fma_us = _floor_from_ops(ops, scalar_ops_per_s)
    dequant_us = _floor_from_ops(dequant_ops, dequant_ops_per_s)
    theoretical_us = _max_present([raw_us, fma_us, dequant_us])
    llama_gbs = q4_gbs if row.get("quant") == "Q4_K" else None
    llama_us = _floor_us(bytes_, llama_gbs) if bytes_ and llama_gbs else None
    reclaim = max(0.0, wall_us - llama_us) if llama_us is not None else None
    rows.append({
      "kernel": row.get("kernel"),
      "role": row.get("role"),
      "quant": row.get("quant"),
      "shape": row.get("shape") or [],
      "wall_us": wall_us,
      "pct_step": 100.0 * wall_us / float(cwhole.get("wall_us") or 0.0) if cwhole.get("wall_us") else None,
      "phys_bytes": bytes_,
      "logical_fma_ops": ops,
      "dequant_ops_est": dequant_ops,
      "current_eff_gbs": eff,
      "raw_hbm_floor_us": raw_us,
      "scalar_fma_floor_us": fma_us,
      "dequant_floor_us": dequant_us,
      "theoretical_floor_us": theoretical_us,
      "llama_practical_gbs": llama_gbs,
      "llama_practical_us": llama_us,
      "reference_impl": baseline.get("provider_id"),
      "reference_impl_gbs": llama_gbs,
      "reference_impl_us": llama_us,
      "reclaimable_to_llama_us": reclaim,
      "reclaimable_to_reference_us": reclaim,
      "gap_vs_raw_x": wall_us / raw_us if raw_us else None,
      "gap_vs_theoretical_x": wall_us / theoretical_us if theoretical_us else None,
      "gap_vs_llama_x": wall_us / llama_us if llama_us else None,
      "gap_vs_reference_x": wall_us / llama_us if llama_us else None,
      "gap_class": _classify(wall_us=wall_us, theoretical_us=theoretical_us, raw_us=raw_us,
                             reference_us=llama_us, eff_gbs=eff, llama_gbs=llama_gbs),
      "resources": row.get("resources") or {},
      "counters": row.get("counters") or {},
    })
  rows.sort(key=lambda r: r.get("reclaimable_to_llama_us") or 0.0, reverse=True)
  total_reclaim = sum(float(r.get("reclaimable_to_llama_us") or 0.0) for r in rows)
  cwall = _f(cwhole.get("wall_us")) or 0.0
  return {
    "schema": "boltbeam.prefill_roofline_ladder.v1",
    "context": ctx,
    "model_id": cand.get("model_id") or baseline.get("model_id"),
    "target_id": cand.get("target_id") or baseline.get("target_id"),
    "baseline_provider": baseline.get("provider_id"),
    "candidate_provider": cand.get("provider_id"),
    "assumptions": {
      "peak_gbs": peak_gbs,
      "scalar_tflops": scalar_tflops,
      "dequant_gops": dequant_gops,
      "q4_dequant_ops_per_weight": q4_dequant_ops_per_weight,
      "q6_dequant_ops_per_weight": q6_dequant_ops_per_weight,
      "note": "Roofline is the max of available theoretical floors. Llama is a measured reference implementation, not the roofline.",
    },
    "summary": {
      "candidate_wall_us": cwall,
      "candidate_tok_s": cwhole.get("tok_s"),
      "baseline_wall_us": bwhole.get("wall_us"),
      "baseline_tok_s": bwhole.get("tok_s"),
      "baseline_role": "reference_impl",
      "trusted_llama_q4_eff_gbs": q4_gbs,
      "reference_q4_eff_gbs": q4_gbs,
      "packed_reclaimable_to_llama_us": total_reclaim,
      "packed_reclaimable_to_reference_us": total_reclaim,
      "candidate_at_llama_q4_for_q4_rows_wall_us": max(0.0, cwall - total_reclaim) if cwall else None,
      "candidate_at_llama_q4_for_q4_rows_tok_s": (ctx or 0) / max(0.0, cwall - total_reclaim) * 1e6 if ctx and cwall and cwall > total_reclaim else None,
      "top_gap_class": rows[0]["gap_class"] if rows else "no_packed_rows",
    },
    "rows": rows,
    "next_work": [{
      "role": r.get("role"),
      "quant": r.get("quant"),
      "shape": r.get("shape"),
      "gap_class": r.get("gap_class"),
      "reclaimable_to_llama_us": r.get("reclaimable_to_llama_us"),
      "reclaimable_to_reference_us": r.get("reclaimable_to_reference_us"),
      "suggested_resolution": "fix the packed-prefill matmul schedule/codegen for this shape before lifecycle or broad tuning",
    } for r in rows[:5]],
  }


def prefill_roofline_ladder_markdown(report:dict[str, Any]) -> str:
  s, a = report.get("summary", {}), report.get("assumptions", {})
  lines = [
    f"# Prefill Roofline Ladder: {report.get('candidate_provider')} vs {report.get('baseline_provider')}",
    "",
    f"- context: `{report.get('context')}`",
    f"- candidate tok/s: `{_fmt(s.get('candidate_tok_s'))}`",
    f"- reference implementation tok/s: `{_fmt(s.get('baseline_tok_s'))}`",
    f"- reference Q4 effective GB/s: `{_fmt(s.get('reference_q4_eff_gbs'))}`",
    f"- packed reclaimable to reference Q4 rate: `{_fmt(s.get('packed_reclaimable_to_reference_us'))} us`",
    f"- scalar/dequant calibrated: `scalar_tflops={a.get('scalar_tflops')}`, `dequant_gops={a.get('dequant_gops')}`",
    f"- note: `{a.get('note')}`",
    "",
    "## Ladder",
    "",
    "| role | quant | shape | raw HBM us | FMA floor us | dequant floor us | theoretical floor us | reference us | current us | current GB/s | gap vs roofline | gap vs reference | class |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
  ]
  for row in report.get("rows", [])[:12]:
    lines.append("| " + " | ".join([
      str(row.get("role") or ""),
      str(row.get("quant") or ""),
      "`" + str(row.get("shape") or []) + "`",
      _fmt(row.get("raw_hbm_floor_us")),
      _fmt(row.get("scalar_fma_floor_us")),
      _fmt(row.get("dequant_floor_us")),
      _fmt(row.get("theoretical_floor_us")),
      _fmt(row.get("reference_impl_us")),
      _fmt(row.get("wall_us")),
      _fmt(row.get("current_eff_gbs")),
      _fmt(row.get("gap_vs_theoretical_x")),
      _fmt(row.get("gap_vs_reference_x")),
      str(row.get("gap_class") or ""),
    ]) + " |")
  lines += ["", "## Next Work", ""]
  for item in report.get("next_work", []):
    lines.append(f"- {item['role']} {item['quant']} `{item['shape']}`: {item['gap_class']}; "
                 f"{_fmt(item.get('reclaimable_to_reference_us'))} us reclaimable to reference implementation.")
  return "\n".join(lines) + "\n"
