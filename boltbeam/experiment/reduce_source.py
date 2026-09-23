"""L1A reduce-source resolution (parity track).

Classify each hot `r_*` reduce kernel from a tinygrad RSR0 ordered-trace artifact into a source class, using
PRODUCER/CONSUMER kernel identity (the nearest non-reduce neighbors and the +/-window) plus shape facts — not
string-guessing on the reduce name alone. The reduce's neighbors ARE its producer/consumer at kernel
granularity: a `*_partial` producer means a coop-partial GEMV combine; a `flash_*`/`start_pos` neighbor means
attention; a reduce to hidden_size between elementwise kernels is an RMSNorm sum-of-squares; a reduce over the
vocab dimension with no weight producer is sampling/gumbel.

This is diagnosis, so it lives in BoltBeam (the brain), not in tinygrad. tinygrad only captured the ordered
graph facts. The output is per-kernel {source_class, removable, proposed_capability} + a coverage verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boltbeam.vocab import Verdict

# source classes (the L1A taxonomy)
RMSNORM = "rmsnorm"
ATTENTION_COMBINE = "attention_combine"
COOP_PARTIAL_COMBINE = "coop_partial_combine"
SAMPLING_GUMBEL = "sampling_gumbel"
MATVEC_EPILOGUE = "matvec_epilogue"
UNKNOWN = "unknown"

# generic codegen capability proposed to remove each removable class (never a handwritten/model-specific kernel)
_CAPABILITY = {
  RMSNORM: "generated_rmsnorm_reduce_scale_fusion",
  ATTENTION_COMBINE: "generated_attention_combine_reduce",
  COOP_PARTIAL_COMBINE: "generated_coop_partial_reduce_inline",
  MATVEC_EPILOGUE: "generated_matvec_partial_finalize",
}
# risk of the removal (attention fusion touches the flash kernel -> high; rmsnorm reduce+scale -> low)
_RISK = {RMSNORM: "low", COOP_PARTIAL_COMBINE: "medium", MATVEC_EPILOGUE: "medium",
         ATTENTION_COMBINE: "high", SAMPLING_GUMBEL: "none", UNKNOWN: "unknown"}


@dataclass(frozen=True)
class ReduceClass:
  source_class: str
  removable: bool | None            # True / False / None(unknown) — UNKNOWN is never treated as removable
  capability: str | None
  risk: str
  reason: str


def _has_subproduct(factors:list[int], target:int) -> bool:
  """True if a contiguous run of shape factors multiplies to `target` (e.g. 16*320==5120==hidden)."""
  if not target: return False
  for i in range(len(factors)):
    p = 1
    for j in range(i, len(factors)):
      p *= factors[j]
      if p == target: return True
      if p > target: break
  return False


def _name_factors(name:str) -> list[int]:
  return [int(x) for x in name.split("_") if x.isdigit()]


def classify_reduce(kernel:str, immediate:list[str], window:list[str], shape_factors:list[int], *,
                    hidden_size:int | None = None, vocab_size:int | None = None) -> ReduceClass:
  """Classify one reduce kernel. `immediate` = nearest prev/next NON-reduce kernels (the true producer/consumer);
  `window` = the +/-4 kernels (weaker context). Immediate signals dominate; the window only breaks ties the
  immediate neighbors leave open (e.g. an attention reduce whose immediate neighbors are plain elementwise but
  which sits next to a flash kernel). Order is by signal strength."""
  imm = [n for n in immediate if n]
  win = [n for n in (immediate + window) if n]
  has_partial = any("partial" in n for n in imm)   # the *immediate* producer, not a window neighbor
  attn_imm = any(("start_pos" in n) or ("flash" in n) for n in imm)
  attn_win = any(("start_pos" in n) or ("flash" in n) for n in win)
  has_gemv = any("gemv" in n for n in imm)
  # rmsnorm: the reduce's ENTIRE loop nest is exactly hidden elements (a pure per-token hidden reduce) — this
  # cleanly separates it from a large attention reduce that merely has a spurious hidden subproduct.
  prod = 1
  for f in shape_factors: prod *= f
  rmsnorm_sig = hidden_size and prod == hidden_size
  # sampling: a neighbor is vocab-shaped and there is no weight producer
  vocab_factor = vocab_size and any(_has_subproduct(_name_factors(n), vocab_size) for n in win)

  if has_partial:
    return ReduceClass(COOP_PARTIAL_COMBINE, True, _CAPABILITY[COOP_PARTIAL_COMBINE], _RISK[COOP_PARTIAL_COMBINE],
                       "a *_partial GEMV produces/consumes this reduce (coop-partial combine)")
  if rmsnorm_sig:
    return ReduceClass(RMSNORM, True, _CAPABILITY[RMSNORM], _RISK[RMSNORM],
                       "reduces to hidden_size next to a hidden-sized elementwise (RMSNorm sum-of-squares -> scale)")
  if attn_imm or attn_win:
    return ReduceClass(ATTENTION_COMBINE, True, _CAPABILITY[ATTENTION_COMBINE], _RISK[ATTENTION_COMBINE],
                       "sits in the attention block (flash_/start_pos neighbor); removable only by attention/flash fusion")
  if vocab_factor and not has_gemv:
    return ReduceClass(SAMPLING_GUMBEL, False, None, _RISK[SAMPLING_GUMBEL],
                       "reduces over the vocab dimension with no weight producer (sampling/gumbel); not removable")
  if has_gemv:
    return ReduceClass(MATVEC_EPILOGUE, True, _CAPABILITY[MATVEC_EPILOGUE], _RISK[MATVEC_EPILOGUE],
                       "a GEMV produces/consumes this reduce (matvec epilogue)")
  return ReduceClass(UNKNOWN, None, None, _RISK[UNKNOWN], "no producer/consumer/shape signal resolved the source")


def _row_immediate(row:dict[str, Any]) -> list[str]:
  out:list[str] = []
  for k in ("prev_nonreduce", "next_nonreduce"):
    for pair in row.get(k, []):
      if pair and pair[0]: out.append(pair[0])
  return out


def _row_window(row:dict[str, Any]) -> list[str]:
  out:list[str] = list(row.get("example_window", []))
  for w in row.get("windows", []):
    out += w
  return out


def resolve_trace(reduce_rows:list[dict[str, Any]], *, hidden_size:int | None = None,
                  vocab_size:int | None = None, unknown_ceiling_pct:float = 10.0) -> dict[str, Any]:
  """Resolve an RSR0 reduce_rows list into classified rows + coverage stats + an L1A verdict.

  Acceptance encoded: >=90% of the reduce bucket source-resolved (unknown < ceiling), sampling never removable.
  """
  total = sum(r.get("pct_gpu", 0.0) for r in reduce_rows) or 1.0
  classified = []
  by_class:dict[str, float] = {}
  removable_pct = 0.0
  for r in reduce_rows:
    rc = classify_reduce(r["kernel"], _row_immediate(r), _row_window(r), r.get("shape_factors", []),
                         hidden_size=hidden_size, vocab_size=vocab_size)
    pct = r.get("pct_gpu", 0.0)
    by_class[rc.source_class] = by_class.get(rc.source_class, 0.0) + pct
    if rc.removable is True: removable_pct += pct
    classified.append({"kernel": r["kernel"], "pct_gpu": pct, "calls_in_step": r.get("calls_in_step"),
                       "source_class": rc.source_class, "removable": rc.removable, "capability": rc.capability,
                       "risk": rc.risk, "reason": rc.reason})
  # (continue below)
  unknown_pct = by_class.get(UNKNOWN, 0.0)
  unknown_frac = 100.0 * unknown_pct / total
  resolved_frac = 100.0 - unknown_frac
  if unknown_frac >= unknown_ceiling_pct:
    verdict = "L1A_BLOCKED_SOURCE_RESOLUTION"
  elif removable_pct <= 0.0:
    verdict = "L1A_ABORT_NOT_REMOVABLE"
  else:
    verdict = "L1A_PASS_REDUCE_SOURCE_RESOLVED"
  return {
    "verdict": verdict,
    "reduce_bucket_pct_gpu": round(total, 2),
    "resolved_pct_of_bucket": round(resolved_frac, 1),
    "unknown_pct_of_bucket": round(unknown_frac, 1),
    "removable_pct_gpu": round(removable_pct, 2),
    "by_class_pct_gpu": {k: round(v, 2) for k, v in sorted(by_class.items(), key=lambda x: -x[1])},
    "rows": classified,
  }


# ---- before/after mechanism evidence (feeds the evaluator's reduce_eliminated guardrail) -------------------

def bucket_delta(baseline:dict[str, Any], candidate:dict[str, Any]) -> dict[str, dict[str, float]]:
  """Per-source-class {baseline_pct, candidate_pct, shrink_pct} from two resolve_trace reports (before/after a
  reduce-elimination capability). shrink_pct > 0 means the class got smaller (the reduce was removed/reduced)."""
  b = baseline.get("by_class_pct_gpu", {})
  c = candidate.get("by_class_pct_gpu", {})
  out:dict[str, dict[str, float]] = {}
  for cls in set(b) | set(c):
    bp, cp = float(b.get(cls, 0.0)), float(c.get(cls, 0.0))
    out[cls] = {"baseline_pct": round(bp, 3), "candidate_pct": round(cp, 3), "shrink_pct": round(bp - cp, 3)}
  return out


def reduce_bucket_evidence_extra(source_class:str, baseline_pct:float, candidate_pct:float) -> dict[str, Any]:
  """The `extra` payload for a reduce_source evidence row that lets the evaluator check the reduce_eliminated
  guardrail: the targeted class plus its baseline%. The row's own value carries the candidate% (after)."""
  return {"reduce_class": source_class, "baseline_pct_gpu": round(float(baseline_pct), 3)}
