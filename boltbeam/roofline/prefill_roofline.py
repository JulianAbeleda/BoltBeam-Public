from __future__ import annotations

import copy
from typing import Any
from boltbeam.math.roofline import effective_gbs, floor_us
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


def _f(value:Any) -> float | None:
  try:
    return None if value is None else float(value)
  except (TypeError, ValueError):
    return None


def _context(trace:dict[str, Any], requested:int | None) -> int | None:
  if requested is not None:
    return requested
  for c in trace.get("contexts", []):
    if c is not None:
      return int(c)
  for row in trace.get("rows", []):
    if row.get("context") is not None:
      return int(row["context"])
  return None


def _whole(trace:dict[str, Any], context:int | None) -> dict[str, Any]:
  for row in trace.get("rows", []):
    if row.get("scope") != "whole_step":
      continue
    if context is None or row.get("context") == context:
      return row
  return {}


def _kernel_rows(trace:dict[str, Any], context:int | None) -> list[dict[str, Any]]:
  return [r for r in trace.get("rows", []) if r.get("scope") == "kernel" and (context is None or r.get("context") == context)]


def _eff_gbs(row:dict[str, Any]) -> float | None:
  bytes_ = _f(row.get("phys_bytes"))
  wall_us = _f(row.get("wall_us"))
  if not bytes_ or not wall_us or wall_us <= 0:
    return None
  return effective_gbs(bytes_, wall_us)


def _is_packed_prefill(row:dict[str, Any]) -> bool:
  kernel = str(row.get("kernel") or "")
  if "direct_packed" in kernel or "prefill_q4k" in kernel or "prefill_q6k" in kernel:
    return True
  return row.get("role") in {"ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "quantized_matmul"} and row.get("quant") in {"Q4_K", "Q6_K"}


def _scale_candidate(candidate:dict[str, Any], wall_trace:dict[str, Any] | None, context:int | None) -> dict[str, Any]:
  if wall_trace is None:
    return copy.deepcopy(candidate)
  out = copy.deepcopy(candidate)
  old_whole = _whole(out, context)
  new_whole = _whole(wall_trace, context)
  old_us, new_us = _f(old_whole.get("wall_us")), _f(new_whole.get("wall_us"))
  if not old_us or not new_us:
    return out
  scale = new_us / old_us
  for row in out.get("rows", []):
    if row.get("scope") == "whole_step" and (context is None or row.get("context") == context):
      row.update(new_whole)
      row["time_source"] = "synced_wall_override_plus_scaled_profile_attribution"
    elif row.get("scope") == "kernel" and (context is None or row.get("context") == context):
      if row.get("wall_us") is not None:
        row["wall_us"] = float(row["wall_us"]) * scale
      if row.get("raw_wall_us") is not None:
        row["raw_wall_us"] = float(row["raw_wall_us"]) * scale
      if row.get("phys_bytes") and row.get("wall_us"):
        row["gbs"] = float(row["phys_bytes"]) / float(row["wall_us"]) / 1000.0
      row["time_source"] = "profile_attribution_scaled_to_synced_wall_override"
  out.setdefault("metadata", {})["wall_override_scale"] = scale
  return out


def _trusted_baseline_rate(rows:list[dict[str, Any]], *, quant:str, peak_gbs:float) -> dict[str, Any] | None:
  matches = []
  for row in rows:
    if row.get("quant") != quant or row.get("role") != "quantized_matmul":
      continue
    eff = _eff_gbs(row)
    if eff is None or eff <= 0 or eff > peak_gbs:
      continue
    matches.append(row)
  bytes_ = sum(float(r.get("phys_bytes") or 0.0) for r in matches)
  us = sum(float(r.get("wall_us") or 0.0) for r in matches)
  if not bytes_ or not us:
    return None
  return {"quant": quant, "eff_gbs": effective_gbs(bytes_, us), "rows": len(matches), "bytes": bytes_, "wall_us": us}


def _floor_us(bytes_:float, gbs:float) -> float:
  return floor_us(bytes_, gbs)


def prefill_roofline_report(baseline:dict[str, Any], candidate:dict[str, Any], *, context:int | None = None,
                            peak_gbs:float = DEFAULT_PEAK_MEM_GBS, wall_trace:dict[str, Any] | None = None) -> dict[str, Any]:
  ctx = _context(candidate, context)
  cand = _scale_candidate(candidate, wall_trace, ctx)
  bwhole, cwhole = _whole(baseline, ctx), _whole(cand, ctx)
  ctotal = _f(cwhole.get("wall_us")) or 0.0
  context_tokens = int(ctx or cwhole.get("context") or 0)
  brows, crows = _kernel_rows(baseline, ctx), _kernel_rows(cand, ctx)
  packed = [r for r in crows if _is_packed_prefill(r)]
  q4_rate = _trusted_baseline_rate(brows, quant="Q4_K", peak_gbs=peak_gbs)
  rates = {"Q4_K": q4_rate}

  total_bytes = _f(cwhole.get("total_bytes")) or sum(float(r.get("phys_bytes") or 0.0) for r in crows)
  raw_floor = _floor_us(total_bytes, peak_gbs) if total_bytes and peak_gbs else None
  raw_tok_s = context_tokens / raw_floor * 1e6 if raw_floor and context_tokens else None

  rows = []
  q4_reclaim = 0.0
  optimistic_reclaim = 0.0
  q4_target_gbs = q4_rate["eff_gbs"] if q4_rate else None
  for row in packed:
    wall_us = _f(row.get("wall_us")) or 0.0
    bytes_ = _f(row.get("phys_bytes")) or 0.0
    eff = _eff_gbs(row)
    target_gbs = q4_target_gbs if row.get("quant") == "Q4_K" else None
    target_us = _floor_us(bytes_, target_gbs) if target_gbs and bytes_ else None
    reclaim = max(0.0, wall_us - target_us) if target_us is not None else None
    if reclaim is not None:
      q4_reclaim += reclaim
    optimistic_us = _floor_us(bytes_, q4_target_gbs) if q4_target_gbs and bytes_ else None
    optimistic_row_reclaim = max(0.0, wall_us - optimistic_us) if optimistic_us is not None else None
    if optimistic_row_reclaim is not None:
      optimistic_reclaim += optimistic_row_reclaim
    rows.append({
      "kernel": row.get("kernel"),
      "role": row.get("role"),
      "quant": row.get("quant"),
      "shape": row.get("shape") or [],
      "wall_us": wall_us,
      "pct_step": 100.0 * wall_us / ctotal if ctotal else None,
      "phys_bytes": bytes_,
      "current_eff_gbs": eff,
      "trusted_target_gbs": target_gbs,
      "trusted_target_us": target_us,
      "trusted_reclaimable_us": reclaim,
      "optimistic_q4_rate_us": optimistic_us,
      "optimistic_reclaimable_us": optimistic_row_reclaim,
      "resources": row.get("resources") or {},
    })
  rows.sort(key=lambda r: r.get("optimistic_reclaimable_us") or 0.0, reverse=True)

  q4_wall = max(0.0, ctotal - q4_reclaim)
  optimistic_wall = max(0.0, ctotal - optimistic_reclaim)
  return {
    "schema": "boltbeam.prefill_roofline_report.v1",
    "context": ctx,
    "model_id": cand.get("model_id") or baseline.get("model_id"),
    "target_id": cand.get("target_id") or baseline.get("target_id"),
    "baseline_provider": baseline.get("provider_id"),
    "candidate_provider": cand.get("provider_id"),
    "peak_gbs": peak_gbs,
    "summary": {
      "current_wall_us": ctotal,
      "current_tok_s": cwhole.get("tok_s"),
      "current_total_bytes": total_bytes,
      "raw_hbm_source_byte_floor_us": raw_floor,
      "raw_hbm_source_byte_tok_s": raw_tok_s,
      "current_pct_of_raw_hbm_source_byte_ceiling": 100.0 * float(cwhole.get("tok_s") or 0.0) / raw_tok_s if raw_tok_s else None,
      "raw_hbm_binding": False,
      "raw_hbm_note": "Packed source-byte HBM peak is a sanity upper bound, not the binding ceiling for Q4/Q6 unpack+dequant matmul.",
      "trusted_llama_q4_eff_gbs": q4_target_gbs,
      "q4_only_llama_rate_wall_us": q4_wall,
      "q4_only_llama_rate_tok_s": context_tokens / q4_wall * 1e6 if q4_wall and context_tokens else None,
      "q4_only_speedup": ctotal / q4_wall if q4_wall else None,
      "all_packed_at_llama_q4_rate_wall_us": optimistic_wall,
      "all_packed_at_llama_q4_rate_tok_s": context_tokens / optimistic_wall * 1e6 if optimistic_wall and context_tokens else None,
      "all_packed_at_llama_q4_rate_speedup": ctotal / optimistic_wall if optimistic_wall else None,
      "packed_pct_step": sum(float(r.get("pct_step") or 0.0) for r in rows),
    },
    "baseline_rates": {"Q4_K": q4_rate, "Q6_K": {"trusted": False, "reason": "baseline trace maps Q6_K to mul_mat_vec_q rows above raw HBM peak; excluded from trusted ceiling"}},
    "hot_rows": rows,
    "next_work": _next_work(rows),
  }


def _next_work(rows:list[dict[str, Any]]) -> list[dict[str, Any]]:
  out = []
  for row in rows[:5]:
    out.append({
      "role": row.get("role"),
      "quant": row.get("quant"),
      "shape": row.get("shape"),
      "reclaimable_us_to_optimistic_q4_rate": row.get("optimistic_reclaimable_us"),
      "why": "largest roofline gap row" if len(out) == 0 else "next largest packed-prefill roofline gap",
      "suggested_resolution": "generated tiled quantized prefill matmul topology; match llama-class workgroup/token tiling before lifecycle work",
    })
  return out


def prefill_roofline_markdown(report:dict[str, Any]) -> str:
  s = report.get("summary", {})
  lines = [
    f"# Prefill Practical Roofline: {report.get('candidate_provider')} vs {report.get('baseline_provider')}",
    "",
    f"- context: `{report.get('context')}`",
    f"- current tok/s: `{_fmt(s.get('current_tok_s'))}`",
    f"- raw HBM source-byte ceiling tok/s: `{_fmt(s.get('raw_hbm_source_byte_tok_s'))}` (non-binding)",
    f"- trusted llama Q4 effective GB/s: `{_fmt(s.get('trusted_llama_q4_eff_gbs'))}`",
    f"- Q4-only at llama Q4 rate tok/s: `{_fmt(s.get('q4_only_llama_rate_tok_s'))}` "
    f"({_fmt(s.get('q4_only_speedup'))}x)",
    f"- all packed at llama Q4 rate tok/s: `{_fmt(s.get('all_packed_at_llama_q4_rate_tok_s'))}` "
    f"({_fmt(s.get('all_packed_at_llama_q4_rate_speedup'))}x)",
    f"- packed pct step: `{_fmt(s.get('packed_pct_step'))}`",
    "",
    "## Interpretation",
    "",
    "- Raw HBM is not the useful target here; Q4/Q6 unpack/dequant plus matmul is the binding substrate.",
    "- The practical roofline is llama's measured Q4 quantized matmul rate for the same workload family.",
    "- Q6 is left untrusted in this trace because the imported llama Q6 rows exceed raw HBM peak.",
    "",
    "## Hot Rows",
    "",
    "| role | quant | shape | current us | current GB/s | target GB/s | target us | reclaim us | resources |",
    "|---|---|---|---:|---:|---:|---:|---:|---|",
  ]
  for row in report.get("hot_rows", [])[:8]:
    res = row.get("resources") or {}
    lines.append("| " + " | ".join([
      str(row.get("role") or ""),
      str(row.get("quant") or ""),
      "`" + str(row.get("shape") or []) + "`",
      _fmt(row.get("wall_us")),
      _fmt(row.get("current_eff_gbs")),
      _fmt(row.get("trusted_target_gbs")),
      _fmt(row.get("trusted_target_us")),
      _fmt(row.get("trusted_reclaimable_us")),
      _resource_text(res),
    ]) + " |")
  lines += ["", "## Next Work", ""]
  for item in report.get("next_work", []):
    lines.append(f"- {item['role']} {item['quant']} `{item['shape']}`: {_fmt(item.get('reclaimable_us_to_optimistic_q4_rate'))} us reclaimable; {item['suggested_resolution']}")
  return "\n".join(lines) + "\n"


def _fmt(value:Any) -> str:
  v = _f(value)
  return "" if v is None else f"{v:.3f}"


def _resource_text(resources:dict[str, Any]) -> str:
  if not resources:
    return "missing"
  parts = []
  if resources.get("global_size"):
    parts.append("global=" + "x".join(str(x) for x in resources["global_size"]))
  if resources.get("local_size"):
    parts.append("local=" + "x".join(str(x) for x in resources["local_size"]))
  if resources.get("workgroup_threads") is not None:
    parts.append(f"threads={resources['workgroup_threads']}")
  for key in ("lds_bytes", "scratch_bytes", "vgpr", "sgpr"):
    if resources.get(key) is not None:
      parts.append(f"{key}={resources[key]}")
  return ", ".join(parts) or "present"
