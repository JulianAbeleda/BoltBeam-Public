"""The measured token, tied out against the roofline line by line, and why each role is where it is.

    limit at context N (ideal)            derived: weight bytes and the KV cache read at the measured context,
                                          over the chip's measured bandwidth
  + weight kernels above their ideal      measured: each attributed role's kernel time minus its ideal
  + other kernels above their ideal       measured: attention, norms, quantize, rope, copies, minus their ideal
                                          (the limit's KV read and any role that could not be split)
  + gaps between kernels (GPU idle)       the difference: the measured token minus everything above
  = measured token                        the captured run's own time per token

Every line is measured or derived from measured facts. The gaps line is the difference and is labelled so; it
is never presented as "nothing left over". Where a line cannot be measured, it is merged with the next and the
reason is given.

The per-role reason follows one rule, with its numbers here and nowhere else (ROLE_RULE).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

# KV cache element size per provider, and where that fact comes from
KV_ELEMENT = {"llama.cpp": (2, "llama-bench's default KV cache type, f16"),
              "tinygrad": (2, "tinygrad's fp16 KV cache on its generated decode path")}
TINYGRAD_WARM = 3  # warm tokens before the measured window (runtime/tinygrad_decode_profile.py --warm)
SHOW_BOTH_SHARE = 0.01  # show the limit at context 1 beside context N when the KV read changes it by 1% or more

# One rule for the reason word per role. Little's law: to keep memory busy at bandwidth B with round-trip latency
# L, B x L bytes must be in flight. A call whose bytes are only a few times that spends a large part of its time
# filling and draining the pipe, so it cannot get near peak however good its code is.
ROLE_RULE = {
  "at_limit_pct": 85.0,  # at or above this share of peak bandwidth: at the limit
  "latency_us": 1.0,  # assumed DRAM round trip plus launch ramp; not measured on these chips
  "fill_factor": 10.0,  # a call must move this many times B x L bytes to be able to reach about 90% of peak
}
OWN_TIMING = "tinygrad-profile-events"  # the engine's own profiling, not an outside capture
REASONS = {"at_limit": "at the limit", "small": "too small to fill memory", "slow": "slow kernel"}

# what a kernel that is not a weight role is, by words in its name; for labels only, never for attribution
KIND_WORDS = (("flash", "attention"), ("attention", "attention"), ("softmax", "attention"), ("rms_norm", "norm"),
              ("norm", "norm"), ("quantize", "quantize"), ("rope", "rope"), ("set_rows", "KV cache write"),
              ("copy", "copy"), ("cpy", "copy"), ("COPY", "copy"), ("silu", "elementwise"), ("glu", "elementwise"),
              ("add", "elementwise"), ("mul", "elementwise"))
KINDS = ("attention", "norm", "quantize", "rope", "KV cache write", "copy", "elementwise", "reduce")
KIND_PREFIX = (("r_", "reduce"), ("E_", "elementwise"))  # tinygrad's generated kernel names


def kind_of(row:dict[str, Any]) -> str:
  if row.get("kind") in KINDS:
    return row["kind"]
  name = str(row.get("kernel") or "")
  if row.get("role") in KINDS:
    return row["role"]
  for word, kind in KIND_WORDS:
    if word in name:
      return kind
  for prefix, kind in KIND_PREFIX:
    if name.startswith(prefix):
      return kind
  return "other"


def kv_ms(profile:dict[str, Any], context:float, element_bytes:int, bandwidth_gbs:float) -> float:
  """Time to read the KV cache once for one token at this context: layers x K and V x kv heads x head dim x
  context x element size, over the bandwidth."""
  att = (profile.get("metadata") or {}).get("attention") or {}
  layers, heads, dim = profile.get("layer_count"), att.get("head_count_kv"), att.get("head_dim")
  if not (layers and heads and dim and bandwidth_gbs):
    return 0.0
  return layers * 2 * heads * dim * context * element_bytes / (bandwidth_gbs * 1e9) * 1e3


def attended_context(trace:dict[str, Any], provider:str) -> float | None:
  """The mean context the measured tokens attended to. Captures subtract a 1-token run from an N-token run at the
  same depth, so the window is tokens 2..N; tinygrad's own timing measures tokens 1..N after its warm tokens."""
  whole = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step"), None)
  if not whole or whole.get("context") is None:
    return None
  n = whole.get("decode_tokens") or 1
  base = float(whole["context"]) + (TINYGRAD_WARM if provider == "tinygrad" else 0)
  captured = (trace.get("capture") or {}).get("method") not in (None, "tinygrad-profile-events")
  return base + ((n + 3) / 2 if captured else (n + 1) / 2)


def role_why(roles:list[dict[str, Any]], bandwidth_gbs:float) -> tuple[list[dict[str, Any]], str]:
  """Each role with % of peak, µs per call and its reason word; and the rule as one sentence with its numbers."""
  in_flight = bandwidth_gbs * 1e9 * ROLE_RULE["latency_us"] * 1e-6  # bytes
  small = in_flight * ROLE_RULE["fill_factor"]
  out = []
  for r in roles:
    calls = r.get("calls_per_token") or 0
    pct = 100.0 * r["ideal_ms"] / r["actual_ms"] if r["actual_ms"] > 0 else None
    per_call_bytes = r["ideal_ms"] * 1e-3 * bandwidth_gbs * 1e9 / calls if calls else None
    if pct is not None and round(pct, 1) >= ROLE_RULE["at_limit_pct"]:
      why = REASONS["at_limit"]
    elif per_call_bytes is not None and per_call_bytes < small:
      why = REASONS["small"]
    else:
      why = REASONS["slow"]
    out.append({**r, "pct_peak": round(pct, 1) if pct is not None else None, "us_per_call": r["actual_ms"] * 1e3 / calls if calls else None,
                "mb_per_call": per_call_bytes / 1e6 if per_call_bytes else None, "reason": why})
  rule = (f"At the limit: {ROLE_RULE['at_limit_pct']:.0f}% of peak or more. Too small to fill memory: a call moves "
          f"under {small / 1e6:.1f} MB, which is {ROLE_RULE['fill_factor']:.0f} x the {in_flight / 1e6:.2f} MB that must "
          f"be in flight ({bandwidth_gbs:.0f} GB/s x {ROLE_RULE['latency_us']:.1f} µs assumed latency, Little's law). "
          "Slow kernel: anything else below the limit.")
  return out, rule


def _step4(run:pathlib.Path, context:float | None, provider:str) -> tuple[float | None, int | None]:
  """The untraced whole-step ms per token from step 4, at the context nearest the measured one, only when step 4
  measured the same provider: one provider's kernels never tie out against another's token."""
  p = run / "timing_trace.json"
  if not p.exists():
    return None, None
  trace = json.loads(p.read_text())
  measured_by = "tinygrad" if str(trace.get("provider_id", "")).startswith("tinygrad") else "llama.cpp"
  if measured_by != provider:
    return None, None
  rows = [r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("tok_s")]
  if not rows:
    return None, None
  row = min(rows, key=lambda r: abs(r["context"] - (context or 0)))
  return 1000.0 / row["tok_s"], row["context"]


def tie_out(run:pathlib.Path, *, provider:str, table:dict[str, Any] | None, trace:dict[str, Any] | None,
            limit_ms:float, profile:dict[str, Any], bandwidth_gbs:float, missing:str | None = None) -> dict[str, Any]:
  """The tie-out for one provider's measurement. `table` is loss()'s measured table (or None: no kernels here)."""
  element, element_source = KV_ELEMENT.get(provider, (2, "f16 assumed"))
  context = attended_context(trace or {}, provider) if trace else None
  untraced, untraced_ctx = _step4(run, context, provider)
  if context is None:
    context = float(untraced_ctx or 1)
  kv = kv_ms(profile, context, element, bandwidth_gbs)
  limit = limit_ms + kv
  out = {"provider": provider, "context": context, "limit_ms_ctx1": limit_ms, "limit_ms": limit, "kv_ms": kv,
         "kv_source": f"KV cache read at context {context:.0f}: {element} bytes per element, {element_source}",
         "show_both": kv >= SHOW_BOTH_SHARE * limit_ms, "untraced_ms": untraced, "untraced_context": untraced_ctx,
         "lines": [], "token_ms": None, "token_source": None, "busy_ms": None, "missing": missing, "refused": None}
  whole = next((r for r in (trace or {}).get("rows", []) if r.get("scope") == "whole_step"), {})
  if table is None:  # no kernel view here: the whole step from step 4 is the token
    if untraced is None:
      return out
    out.update(token_ms=untraced, token_source=f"the untraced whole step at context {untraced_ctx}")
    out["lines"] = [{"label": f"limit at context {context:.0f} (ideal)", "ms": limit, "how": "derived"},
                    {"label": "kernels and gaps, not split", "ms": untraced - limit, "how": "difference"}]
    return out
  busy = table["kernel_ms"]
  # An outside capture (nsys, xctrace) runs the program nearly as it is: its own token is the one its kernels fit.
  # The engine's own profiling (tinygrad PROFILE=1, one command buffer per kernel) slows the run many times over,
  # so its wall time is mostly profiling: the token is the untraced whole step, and its idle is never "gaps".
  outside = (trace or {}).get("capture", {}).get("method") not in (None, OWN_TIMING)
  if outside and whole.get("token_ms"):
    token, source = whole["token_ms"], "the captured run's own time per token"
  else:
    token, source = untraced, f"the untraced whole step at context {untraced_ctx}"
  out.update(token_ms=token, token_source=source, busy_ms=busy)
  if token is None:
    out["refused"] = f"no measured token for {provider} in this run to tie the kernels out against"
    return out
  if busy > token:  # idle time cannot be negative: the two numbers are not from comparable runs
    numbers = (f"profiled kernels do not fit the real token: they sum to {busy:.3f} ms, more than the "
               f"{token:.3f} ms token ({source})")
    out["refused"] = (f"{provider}'s profiling mode inflates kernel times on this GPU, so its tie-out cannot be shown "
                      f"({numbers})" if not outside else numbers)
    return out
  roles = table["roles"]
  weight_ideal = sum(r["ideal_ms"] for r in roles)
  weight_above = sum(r["actual_ms"] - r["ideal_ms"] for r in roles)
  other_busy = busy - sum(r["actual_ms"] for r in roles)
  other_ideal = limit - weight_ideal
  lines = [{"label": f"limit at context {context:.0f} (ideal)", "ms": limit, "how": "derived"}]
  if roles:
    taken = {(r["role"], r["quant"]) for r in roles}
    parts: dict[str, float] = {}
    tokens = table["tokens"]
    for r in (trace or {}).get("rows", []):
      if r.get("scope") == "kernel" and (r.get("role"), r.get("quant")) not in taken:
        k = kind_of(r)
        parts[k] = parts.get(k, 0.0) + float(r["wall_us"]) / tokens / 1000.0
    lines += [{"label": "weight kernels above their ideal", "ms": weight_above, "how": "measured"},
              {"label": "other kernels above their ideal", "ms": other_busy - other_ideal, "how": "measured",
               "busy_ms": other_busy, "ideal_ms": other_ideal,
               "parts": [{"kind": k, "ms": v} for k, v in sorted(parts.items(), key=lambda kv_: -kv_[1])]}]
  else:
    lines.append({"label": "all kernels above the ideal, not split by role", "ms": busy - limit, "how": "measured"})
  lines.append({"label": "gaps between kernels (GPU idle)", "ms": token - limit - sum(l["ms"] for l in lines[1:]),
                "how": "difference"})
  out["lines"] = lines
  return out
