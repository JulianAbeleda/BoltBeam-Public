from __future__ import annotations

from typing import Any

_PREFILL_PACKED_ROLES = {
  ("ffn_down", "Q6_K"),
  ("ffn_gate_up", "Q4_K"),
  ("attn_qo", "Q4_K"),
  ("attn_kv", "Q6_K"),
  ("quantized_matmul", "Q4_K"),
  ("quantized_matmul", "Q6_K"),
}


def _ctx_summary(profile:dict[str, Any], ctx:int | None = None) -> dict[str, Any]:
  rows = profile.get("context_summaries", [])
  if not rows:
    return {}
  if ctx is not None:
    for row in rows:
      if row.get("context") == ctx:
        return row
  return rows[0]


def _role_index(profile:dict[str, Any], ctx:int | None = None) -> dict[tuple[str | None, str | None], dict[str, Any]]:
  out: dict[tuple[str | None, str | None], dict[str, Any]] = {}
  for row in profile.get("role_timing", []):
    if ctx is not None and row.get("context") != ctx:
      continue
    key = (row.get("role"), row.get("quant"))
    if key not in out:
      out[key] = dict(row)
    else:
      cur = out[key]
      cur["wall_us"] = float(cur.get("wall_us") or 0.0) + float(row.get("wall_us") or 0.0)
      cur["phys_bytes"] = float(cur.get("phys_bytes") or 0.0) + float(row.get("phys_bytes") or 0.0)
      cur["pct_step"] = float(cur.get("pct_step") or 0.0) + float(row.get("pct_step") or 0.0)
      cur["kernels"] = list(cur.get("kernels", [])) + list(row.get("kernels", []))
  return out


def _f(v:Any) -> float | None:
  try:
    return None if v is None else float(v)
  except (TypeError, ValueError):
    return None


def _pct_delta(candidate:float | None, baseline:float | None) -> float | None:
  if candidate is None or baseline is None or baseline == 0:
    return None
  return 100.0 * (candidate - baseline) / baseline


def _is_tinygrad_prefill(candidate:dict[str, Any]) -> bool:
  return "tinygrad" in str(candidate.get("provider_id") or "").lower() and candidate.get("workload") == "prefill"


def _row_kernels(row:dict[str, Any]) -> list[str]:
  kernels = row.get("candidate_kernels", row.get("kernels", []))
  return [str(k) for k in kernels if k is not None]


def _is_packed_prefill_row(row:dict[str, Any]) -> bool:
  if (row.get("role"), row.get("quant")) in _PREFILL_PACKED_ROLES:
    return True
  return any("direct_packed" in k or "prefill_q4k" in k or "prefill_q6k" in k for k in _row_kernels(row))


def _packed_prefill_trigger(rows:list[dict[str, Any]], candidate:dict[str, Any]) -> dict[str, Any] | None:
  if not _is_tinygrad_prefill(candidate):
    return None
  hot = [r for r in rows if _is_packed_prefill_row(r) and (_f(r.get("candidate_pct_step")) or 0.0) >= 3.0]
  packed_pct = sum((_f(r.get("candidate_pct_step")) or 0.0) for r in hot)
  slow_packed = [r for r in hot if (_f(r.get("candidate_minus_baseline_pct_step")) or 0.0) > 5.0]
  kernel_hint = any("direct_packed" in k or "prefill_q4k" in k or "prefill_q6k" in k for r in hot for k in _row_kernels(r))
  if packed_pct < 50.0 or (not slow_packed and not kernel_hint) or (not kernel_hint and len(hot) < 2):
    return None
  hot_roles = [{
    "role": r.get("role"),
    "quant": r.get("quant"),
    "candidate_pct_step": r.get("candidate_pct_step"),
    "candidate_minus_baseline_pct_step": r.get("candidate_minus_baseline_pct_step"),
  } for r in hot[:8]]
  return {
    "trigger_id": "tinygrad_packed_prefill_gemm_schedule",
    "classification": "codegen_or_scheduler",
    "reason": "tinygrad prefill is dominated by packed/direct Q4/Q6 GEMM roles, so the gap is in packed prefill GEMM schedule/codegen before memory lifecycle work",
    "packed_prefill_pct_step": round(packed_pct, 3),
    "hot_roles": hot_roles,
    "recommended_focus": [
      "Q6_K ffn_down packed-load/generated schedule",
      "Q4_K ffn_gate_up packed-load/generated schedule",
      "direct packed GEMM tile/resource shape versus llama/library roofline",
    ],
  }


def _gap_class(rows:list[dict[str, Any]], cand:dict[str, Any], base:dict[str, Any], candidate:dict[str, Any] | None = None) -> str:
  if candidate is not None and _packed_prefill_trigger(rows, candidate):
    return "packed_prefill_gemm_schedule_gap"
  slow = [r for r in rows if (r.get("candidate_minus_baseline_pct_step") or 0.0) > 5.0]
  roles = {r.get("role") for r in slow}
  if any(r in roles for r in ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "quantized_matmul")):
    return "codegen_or_schedule_gemm_gap"
  if any(r in roles for r in ("fp16_overlay", "dequant", "activation_quantize")):
    return "materialization_lifecycle_gap"
  if any(r in roles for r in ("attention", "attention_qk", "attention_pv")):
    return "attention_context_gap"
  if any(r in roles for r in ("norm", "rope", "elementwise", "ffn_activation", "copy")):
    return "graph_fragmentation_or_fusion_gap"
  c_bytes, b_bytes = _f(cand.get("total_bytes")), _f(base.get("total_bytes"))
  if c_bytes and b_bytes and c_bytes > 1.10 * b_bytes:
    return "memory_residency_gap"
  return "mixed_or_inconclusive_gap"


def compare_timing_profiles(baseline:dict[str, Any], candidate:dict[str, Any], *, context:int | None = None) -> dict[str, Any]:
  bsum, csum = _ctx_summary(baseline, context), _ctx_summary(candidate, context)
  bidx, cidx = _role_index(baseline, bsum.get("context")), _role_index(candidate, csum.get("context"))
  keys = sorted(set(bidx) | set(cidx), key=lambda k: (-(cidx.get(k, {}).get("pct_step") or 0.0), str(k)))
  rows = []
  for key in keys:
    b, c = bidx.get(key, {}), cidx.get(key, {})
    bp, cp = _f(b.get("pct_step")) or 0.0, _f(c.get("pct_step")) or 0.0
    bw, cw = _f(b.get("wall_us")) or 0.0, _f(c.get("wall_us")) or 0.0
    rows.append({
      "role": key[0],
      "quant": key[1],
      "baseline_pct_step": round(bp, 3),
      "candidate_pct_step": round(cp, 3),
      "candidate_minus_baseline_pct_step": round(cp - bp, 3),
      "baseline_wall_us": round(bw, 3),
      "candidate_wall_us": round(cw, 3),
      "candidate_vs_baseline_wall_pct": None if bw == 0 else round(100.0 * (cw - bw) / bw, 3),
      "baseline_phys_bytes": round(_f(b.get("phys_bytes")) or 0.0, 3),
      "candidate_phys_bytes": round(_f(c.get("phys_bytes")) or 0.0, 3),
      "candidate_kernels": list(c.get("kernels", [])),
    })
  b_tok, c_tok = _f(bsum.get("tok_s")), _f(csum.get("tok_s"))
  b_us, c_us = _f(bsum.get("total_us")), _f(csum.get("total_us"))
  triggers = [t for t in [_packed_prefill_trigger(rows, candidate)] if t is not None]
  report = {
    "schema": "boltbeam.timing_compare.v1",
    "model_id": candidate.get("model_id") or baseline.get("model_id"),
    "target_id": candidate.get("target_id") or baseline.get("target_id"),
    "workload": candidate.get("workload") or baseline.get("workload"),
    "context": csum.get("context", bsum.get("context")),
    "baseline_provider": baseline.get("provider_id"),
    "candidate_provider": candidate.get("provider_id"),
    "summary": {
      "baseline_tok_s": b_tok,
      "candidate_tok_s": c_tok,
      "candidate_vs_baseline_tok_s_pct": None if not b_tok else round(100.0 * (c_tok - b_tok) / b_tok, 3),
      "baseline_total_us": b_us,
      "candidate_total_us": c_us,
      "candidate_vs_baseline_elapsed_pct": _pct_delta(c_us, b_us),
      "baseline_total_bytes": bsum.get("total_bytes"),
      "candidate_total_bytes": csum.get("total_bytes"),
      "baseline_launch_count": bsum.get("launch_count"),
      "candidate_launch_count": csum.get("launch_count"),
    },
    "dominant_gap": _gap_class(rows, csum, bsum, candidate),
    "triggers": triggers,
    "role_deltas": rows[:32],
  }
  report["next_actions"] = _next_actions(report["dominant_gap"])
  return report


def _next_actions(gap:str) -> list[str]:
  return {
    "packed_prefill_gemm_schedule_gap": [
      "treat this as a packed prefill GEMM schedule/codegen issue, not a planner or fp16-overlay lifecycle issue",
      "measure Q6_K ffn_down with packed-load/generated schedule first, then Q4_K ffn_gate_up",
      "compare per-role TFLOPS/tile/resource shape against llama or library kernels before changing routing",
    ],
    "codegen_or_schedule_gemm_gap": [
      "inspect generated GEMM schedule/resource descriptors for the hot prefill roles",
      "compare tinygrad GEMM TFLOPS per role against llama/libraries before changing lifecycle",
    ],
    "materialization_lifecycle_gap": [
      "find dequant/fp16-overlay materialization rows and remove or make them replay-scratch-local",
      "do not chase matmul kernels until the extra lifecycle rows disappear",
    ],
    "graph_fragmentation_or_fusion_gap": [
      "reduce launch/tail elementwise overhead through graph capture or fusion boundaries",
      "verify with the same timing profile before route promotion",
    ],
    "attention_context_gap": [
      "profile start_pos/context growth and separate QK/PV/softmax buckets",
      "only tune attention if it dominates the context where 14B misses the roofline",
    ],
    "memory_residency_gap": [
      "audit resident fp16 overlay and packed clone bytes before optimizing kernels",
      "require peak VRAM and transfer evidence with the timing profile",
    ],
  }.get(gap, ["collect a finer timing trace; no single subsystem owns the gap"])


def compare_markdown(report:dict[str, Any]) -> str:
  s = report.get("summary", {})
  lines = [
    f"# Timing Compare: {report.get('candidate_provider')} vs {report.get('baseline_provider')}",
    "",
    f"- context: `{report.get('context')}`",
    f"- dominant gap: `{report.get('dominant_gap')}`",
    f"- baseline tok/s: `{s.get('baseline_tok_s')}`",
    f"- candidate tok/s: `{s.get('candidate_tok_s')}`",
    f"- candidate vs baseline tok/s: `{s.get('candidate_vs_baseline_tok_s_pct')}`%",
  ]
  if report.get("triggers"):
    lines += ["", "## Triggers", ""]
    for t in report.get("triggers", []):
      lines.append(f"- `{t.get('trigger_id')}`: {t.get('reason')} ({t.get('packed_prefill_pct_step')}% of candidate step)")
      for r in t.get("hot_roles", [])[:4]:
        lines.append(f"  - {r.get('role')} {r.get('quant') or ''}: {r.get('candidate_pct_step')}% candidate, delta {r.get('candidate_minus_baseline_pct_step')} pp")
  lines += [
    "",
    "## Role Deltas",
    "",
    "| role | quant | baseline % | candidate % | delta pp | baseline us | candidate us |",
    "|---|---|---:|---:|---:|---:|---:|",
  ]
  for r in report.get("role_deltas", [])[:16]:
    lines.append(f"| {r.get('role')} | {r.get('quant') or ''} | {r.get('baseline_pct_step')} | "
                 f"{r.get('candidate_pct_step')} | {r.get('candidate_minus_baseline_pct_step')} | "
                 f"{r.get('baseline_wall_us')} | {r.get('candidate_wall_us')} |")
  lines += ["", "## Next Actions", ""]
  lines += [f"- {a}" for a in report.get("next_actions", [])]
  return "\n".join(lines) + "\n"
