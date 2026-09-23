from __future__ import annotations

from typing import Any

from boltbeam.vocab import SCHEMA_HW_TRACE, SCHEMA_SUBSTRATE_COMPARE, SCHEMA_TIMING_TRACE


_PACKED_PREFILL_ROLES = {
  ("ffn_down", "Q4_K"),
  ("ffn_down", "Q6_K"),
  ("ffn_gate_up", "Q4_K"),
  ("attn_qo", "Q4_K"),
  ("attn_kv", "Q4_K"),
  ("attn_kv", "Q6_K"),
  ("quantized_matmul", "Q4_K"),
  ("quantized_matmul", "Q6_K"),
}


def _f(value:Any) -> float | None:
  try:
    return None if value is None else float(value)
  except (TypeError, ValueError):
    return None


def _schema_ok(trace:dict[str, Any]) -> bool:
  return trace.get("schema") in {SCHEMA_TIMING_TRACE, SCHEMA_HW_TRACE}


def _kernel_rows(trace:dict[str, Any], context:int | None) -> list[dict[str, Any]]:
  rows = []
  for row in trace.get("rows", []):
    if row.get("scope") != "kernel":
      continue
    if context is not None and row.get("context") != context:
      continue
    rows.append(row)
  return rows


def _whole(trace:dict[str, Any], context:int | None) -> dict[str, Any]:
  for row in trace.get("rows", []):
    if row.get("scope") != "whole_step":
      continue
    if context is None or row.get("context") == context:
      return row
  return {}


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


def _eff_gbs(row:dict[str, Any]) -> float | None:
  bytes_ = _f(row.get("phys_bytes"))
  wall_us = _f(row.get("wall_us"))
  if not bytes_ or not wall_us or wall_us <= 0:
    return None
  return bytes_ / wall_us / 1000.0


def _pct_step(row:dict[str, Any], total_us:float | None) -> float | None:
  wall_us = _f(row.get("wall_us"))
  if not wall_us or not total_us:
    return None
  return 100.0 * wall_us / total_us


def _is_candidate_packed(row:dict[str, Any]) -> bool:
  role, quant = row.get("role"), row.get("quant")
  kernel = str(row.get("kernel") or "")
  if (role, quant) in _PACKED_PREFILL_ROLES:
    return True
  return "direct_packed" in kernel or "prefill_q4k" in kernel or "prefill_q6k" in kernel


def _is_baseline_quantized_matmul(row:dict[str, Any]) -> bool:
  return row.get("role") in {"quantized_matmul", "quantized_matvec"} and row.get("quant") in {"Q4_K", "Q6_K"}


def _shape(row:dict[str, Any]) -> list[int]:
  try:
    return [int(x) for x in (row.get("shape") or [])]
  except (TypeError, ValueError):
    return []


def _resources(row:dict[str, Any]) -> dict[str, Any]:
  res = dict(row.get("resources") or {})
  if row.get("counters"):
    res["counters"] = row.get("counters")
  return res


def _substrate(row:dict[str, Any], provider_id:str | None) -> str:
  kernel = str(row.get("kernel") or "")
  if "mul_mat_q<" in kernel:
    return "llama mul_mat_q quantized matmul"
  if "prefill_q4k" in kernel or "prefill_q6k" in kernel:
    return "tinygrad generated packed prefill GEMM"
  if "prefill_gen_sched" in kernel or "prefill_graph" in kernel:
    return "tinygrad generated fp16 graph GEMM"
  if provider_id and "llama" in provider_id:
    return "llama provider kernel"
  if provider_id and "tinygrad" in provider_id:
    return "tinygrad provider kernel"
  return "provider kernel"


def _short_kernel(name:Any, limit:int=96) -> str:
  s = str(name or "")
  return s if len(s) <= limit else s[:limit-3] + "..."


def _row_summary(row:dict[str, Any], total_us:float | None, provider_id:str | None) -> dict[str, Any]:
  return {
    "kernel": row.get("kernel"),
    "role": row.get("role"),
    "quant": row.get("quant"),
    "shape": _shape(row),
    "wall_us": _f(row.get("wall_us")),
    "calls": row.get("calls"),
    "phys_bytes": _f(row.get("phys_bytes")),
    "pct_step": _pct_step(row, total_us),
    "eff_gbs": _eff_gbs(row),
    "substrate": _substrate(row, provider_id),
    "resources": _resources(row),
  }


def compare_substrate(baseline:dict[str, Any], candidate:dict[str, Any], *, context:int | None = None,
                      min_candidate_pct:float = 3.0) -> dict[str, Any]:
  if not _schema_ok(baseline):
    raise ValueError(f"baseline must be {SCHEMA_TIMING_TRACE} or {SCHEMA_HW_TRACE}, got {baseline.get('schema')!r}")
  if not _schema_ok(candidate):
    raise ValueError(f"candidate must be {SCHEMA_TIMING_TRACE} or {SCHEMA_HW_TRACE}, got {candidate.get('schema')!r}")
  ctx = _context(candidate, context)
  bwhole, cwhole = _whole(baseline, ctx), _whole(candidate, ctx)
  btotal, ctotal = _f(bwhole.get("wall_us")), _f(cwhole.get("wall_us"))
  brows = _kernel_rows(baseline, ctx)
  crows = _kernel_rows(candidate, ctx)
  baseline_quant = sorted([r for r in brows if _is_baseline_quantized_matmul(r)],
                          key=lambda r: _f(r.get("wall_us")) or 0.0, reverse=True)
  candidate_hot = [
    r for r in sorted([r for r in crows if _is_candidate_packed(r)],
                      key=lambda r: _f(r.get("wall_us")) or 0.0, reverse=True)
    if (_pct_step(r, ctotal) or 0.0) >= min_candidate_pct
  ]
  best_by_quant: dict[str, dict[str, Any]] = {}
  for row in baseline_quant:
    quant = row.get("quant")
    if quant and quant not in best_by_quant:
      best_by_quant[quant] = row
  top_baseline = baseline_quant[0] if baseline_quant else None

  rows = []
  for cand in candidate_hot:
    base = best_by_quant.get(cand.get("quant")) or top_baseline
    ce, be = _eff_gbs(cand), _eff_gbs(base or {})
    rows.append({
      "candidate": _row_summary(cand, ctotal, candidate.get("provider_id")),
      "matched_baseline": _row_summary(base or {}, btotal, baseline.get("provider_id")) if base else None,
      "baseline_eff_gbs_over_candidate_eff_gbs": (be / ce) if be and ce else None,
      "candidate_missing_resources": not bool(_resources(cand)),
      "baseline_missing_resources": not bool(_resources(base or {})),
    })

  packed_pct = sum((_pct_step(r, ctotal) or 0.0) for r in candidate_hot)
  candidate_eff = [_eff_gbs(r) for r in candidate_hot if _eff_gbs(r) is not None]
  baseline_eff = [_eff_gbs(r) for r in baseline_quant if _eff_gbs(r) is not None]
  return {
    "schema": SCHEMA_SUBSTRATE_COMPARE,
    "model_id": candidate.get("model_id") or baseline.get("model_id"),
    "target_id": candidate.get("target_id") or baseline.get("target_id"),
    "workload": candidate.get("workload") or baseline.get("workload"),
    "context": ctx,
    "baseline_provider": baseline.get("provider_id"),
    "candidate_provider": candidate.get("provider_id"),
    "summary": {
      "baseline_tok_s": bwhole.get("tok_s"),
      "candidate_tok_s": cwhole.get("tok_s"),
      "baseline_quantized_matmul_rows": len(baseline_quant),
      "candidate_hot_packed_rows": len(candidate_hot),
      "candidate_hot_packed_pct_step": packed_pct,
      "baseline_eff_gbs_max": max(baseline_eff) if baseline_eff else None,
      "candidate_eff_gbs_max": max(candidate_eff) if candidate_eff else None,
      "eff_gbs_gap_at_best": (max(baseline_eff) / max(candidate_eff)) if baseline_eff and candidate_eff and max(candidate_eff) else None,
    },
    "classification": _classify(rows, packed_pct),
    "baseline_templates": [_row_summary(r, btotal, baseline.get("provider_id")) for r in baseline_quant[:8]],
    "candidate_hot_rows": rows,
    "next_actions": _next_actions(rows),
  }


def _classify(rows:list[dict[str, Any]], packed_pct:float) -> str:
  gaps = [r.get("baseline_eff_gbs_over_candidate_eff_gbs") for r in rows if r.get("baseline_eff_gbs_over_candidate_eff_gbs")]
  if packed_pct >= 50.0 and gaps and max(gaps) >= 4.0:
    return "packed_prefill_substrate_gap"
  if packed_pct >= 50.0:
    return "packed_prefill_resource_incomplete"
  return "substrate_mixed_or_inconclusive"


def _next_actions(rows:list[dict[str, Any]]) -> list[str]:
  if any(r.get("candidate_missing_resources") for r in rows):
    return [
      "capture candidate workgroup/grid/VGPR/LDS/scratch for the hot packed GEMM kernels",
      "compare candidate tile/resource shape against the matched llama quantized matmul template",
      "change the generated packed-prefill schedule only after the resource delta is visible",
    ]
  return [
    "prioritize the hot candidate row with the largest baseline/candidate effective-GB/s gap",
    "adjust generated packed-prefill tile/resource topology, then rerun this substrate compare",
    "promote only if route-bound whole-prefill and per-kernel GB/s both move",
  ]


def substrate_markdown(report:dict[str, Any]) -> str:
  s = report.get("summary", {})
  lines = [
    f"# Substrate Compare: {report.get('candidate_provider')} vs {report.get('baseline_provider')}",
    "",
    f"- context: `{report.get('context')}`",
    f"- classification: `{report.get('classification')}`",
    f"- baseline tok/s: `{s.get('baseline_tok_s')}`",
    f"- candidate tok/s: `{s.get('candidate_tok_s')}`",
    f"- hot packed candidate pct: `{s.get('candidate_hot_packed_pct_step')}`",
    f"- best effective GB/s gap: `{s.get('eff_gbs_gap_at_best')}`",
    "",
    "## Baseline Templates",
    "",
    "| kernel | role | quant | us | calls | eff GB/s | resources |",
    "|---|---|---|---:|---:|---:|---|",
  ]
  for row in report.get("baseline_templates", []):
    lines.append("| " + " | ".join([
      f"`{_short_kernel(row.get('kernel'))}`",
      str(row.get("role") or ""),
      str(row.get("quant") or ""),
      _fmt(row.get("wall_us")),
      str(row.get("calls") or ""),
      _fmt(row.get("eff_gbs")),
      _resource_text(row.get("resources") or {}),
    ]) + " |")
  lines += [
    "",
    "## Candidate Hot Rows",
    "",
    "| role | quant | shape | us | pct | eff GB/s | gap | candidate resources | matched baseline |",
    "|---|---|---|---:|---:|---:|---:|---|---|",
  ]
  for pair in report.get("candidate_hot_rows", []):
    cand = pair.get("candidate") or {}
    base = pair.get("matched_baseline") or {}
    lines.append("| " + " | ".join([
      str(cand.get("role") or ""),
      str(cand.get("quant") or ""),
      "`" + str(cand.get("shape") or []) + "`",
      _fmt(cand.get("wall_us")),
      _fmt(cand.get("pct_step")),
      _fmt(cand.get("eff_gbs")),
      _fmt(pair.get("baseline_eff_gbs_over_candidate_eff_gbs")),
      _resource_text(cand.get("resources") or {}),
      f"`{_short_kernel(base.get('kernel'))}`",
    ]) + " |")
  if report.get("next_actions"):
    lines += ["", "## Next Actions", ""]
    lines += [f"- {x}" for x in report["next_actions"]]
  return "\n".join(lines) + "\n"


def _fmt(value:Any) -> str:
  v = _f(value)
  if v is None:
    return ""
  return f"{v:.3f}"


def _resource_text(resources:dict[str, Any]) -> str:
  if not resources:
    return "missing"
  parts = []
  if resources.get("workgroup"):
    parts.append("WG=" + "x".join(str(x) for x in resources["workgroup"]))
  if resources.get("grid"):
    parts.append("grid=" + "x".join(str(x) for x in resources["grid"]))
  if resources.get("global_size"):
    parts.append("global=" + "x".join(str(x) for x in resources["global_size"]))
  if resources.get("local_size"):
    parts.append("local=" + "x".join(str(x) for x in resources["local_size"]))
  if resources.get("workgroup_threads") is not None and not resources.get("workgroup"):
    parts.append(f"threads={resources['workgroup_threads']}")
  for key in ("lds_bytes", "scratch_bytes", "vgpr", "sgpr"):
    if resources.get(key) is not None:
      parts.append(f"{key}={resources[key]}")
  return ", ".join(parts) or "present"
