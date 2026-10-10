"""The measured token, tied out against the roofline line by line, and why each role is where it is.

    limit at context N (ideal)            derived: weight bytes and the KV cache read at the measured context,
                                          over the chip's measured bandwidth
  + weight kernels above their ideal      measured: each attributed role's kernel time minus its ideal
  + weight kernels, unattributed          measured, only when there are any: GEMVs no role took (UNATTRIBUTED),
                                          named; the limit has no bytes for them, so all their time is above it
  + other kernels above their ideal       measured: attention, norms, quantize, rope, copies, minus their ideal
                                          (the limit's KV read and any role that could not be split); a GEMV of a
                                          profile role the limit leaves out (an `ssm_in`, a MoE router) is named
                                          after it here
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

TINYGRAD_WARM = 3  # warm tokens before the measured window (runtime/tinygrad_decode_profile.py --warm)
SHOW_BOTH_SHARE = 0.01  # show the limit at context 1 beside context N when the KV read changes it by 1% or more

# One rule for the reason word per role. Little's law: to keep memory busy at bandwidth B with round-trip latency
# L, B x L bytes must be in flight. A call whose bytes are only a few times that spends a large part of its time
# filling and draining the pipe, so it cannot get near peak however good its code is.
ROLE_RULE = {
  "at_limit_pct": 85.0,  # at or above this share of peak bandwidth: at the limit
  "latency_us": 1.0,  # assumed DRAM round trip plus launch ramp, used only when the run measured no dispatch floor
  "fill_factor": 10.0,  # a call must move this many times B x L bytes to be able to reach about 90% of peak
}
LATENCY_ASSUMED = "assumed"
LATENCY_MEASURED = "the dispatch floor the probe measured on this GPU"
OWN_TIMING = "tinygrad-profile-events"  # the engine's own profiling, not an outside capture
from boltbeam.collectors.engine_kernels import METHOD as ISOLATED  # the engine's kernels timed alone by BoltBeam's kernel timer
REASONS = {"at_limit": "at the limit", "small": "too small to fill memory", "slow": "slow kernel",
           "compute": "compute bound", "unexplained": "unexplained"}
# a role timed alone whose reading is not firm gets this suffix after its reason word: the word is the rule's
# reading, the suffix says why it is not firm (noisy_words). One rule for both backends; reason_word keeps the firm
# word for the readers that key on it (the HTML rule tables).
NOISY = "{word}; noisy: {why}"
NOISY_RULE = {
  "spread_pct": 10.0,  # ± (half of P90 - P10 over the median) above this can move the word: 85% is at the limit, 75% clearly slow
  "floor_share": 1 / 3,  # the dispatch floor over the measured time above this: the kernel is mostly floor, the floor's jitter is its spread
}
NOISY_FLOOR = "{floor:.1f} µs floor under a {us:.1f} µs kernel"
# an isolated row's rule time: its time less the dispatch floor, the figure that matched nsys per role on the 5090
# (docs/in-model-vs-generic-rtx5090-20261010.md); the measured time, floor included, stays in the row as us_per_call
ISOLATED_COLUMNS = "µs/call is the measured time, floor included; less floor, GB/s and % of peak are {floor_words}"
ISOLATED_RULE = " Timed alone: % of peak and GB/s are on the time {floor_words}; µs/call is the measured time with the floor in it."
# the isolated tie-out: only the ideal and the token are compared; the split is an estimate, labelled so
NOT_SPLIT_LABEL = "kernels and gaps, not split"
ESTIMATE_LABEL = "(estimate from isolated times)"
OTHER_HOW_FLOOR = "at least: KV reads at the limit; attention, norms, launches and gaps were not timed"
OTHER_HOW_DIFF = "difference: the token less the kernels timed alone{floor}; attention, norms, launches and gaps were not timed"
ROLE_SOURCE_ISOLATED, ROLE_SOURCE_IN_MODEL = "isolated", "in_model"

UNATTRIBUTED = "weight kernel, unattributed"  # a GEMV no role took: weight time outside the limit; the capture has no bytes for it
UNATTRIBUTED_LABEL = "weight kernels, unattributed"  # its own tie-out line, the kernels named
GEMV_WORDS = ("mul_mat_vec", "kernel_mul_mv", "gemv")  # the GEMV names the adapters know (collectors/engine_kernels.py)
# what a kernel that is not a weight role is, by words in its name; for labels only, never for attribution. The GEMV
# words come before the elementwise ones: "mul_mat_vec" holds "mul", and a 2 ms GEMV is not elementwise work
KIND_WORDS = (("flash", "attention"), ("attention", "attention"), ("softmax", "attention"), ("rms_norm", "norm"),
              ("norm", "norm"), ("quantize", "quantize"), ("rope", "rope"), ("set_rows", "KV cache write"),
              ("copy", "copy"), ("cpy", "copy"), ("COPY", "copy"), *((w, UNATTRIBUTED) for w in GEMV_WORDS),
              ("silu", "elementwise"), ("glu", "elementwise"), ("add", "elementwise"), ("mul", "elementwise"))
KINDS = ("attention", "norm", "quantize", "rope", "KV cache write", "copy", "elementwise", "reduce", UNATTRIBUTED)
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


def roles_outside_limit(profile:dict[str, Any] | None, limit_keys:set[tuple[str, str]]) -> dict[int, str]:
  """The profile's matrix roles the limit has no bytes for (not in vocab.WEIGHT_GEMV_ROLES: an `ssm_in` kept as
  `ssm_projection`, a MoE router), named by their calls per token: {48: "ssm projection"}.
  A GEMV no role took that runs such a role's count of times per token is labelled with the role's name instead of
  UNATTRIBUTED (a label only: its time stays in other kernels, nothing is attributed). A count two such roles share
  names neither. A role whose quant the registry does not know (an F32 conv weight) is not a matrix role here."""
  from boltbeam.quantization.quant import quant_capability
  by_count: dict[int, set[str]] = {}
  for r in (profile or {}).get("roles") or []:
    if r.get("count") and r.get("rows") and r.get("cols") and (r.get("role"), r.get("quant")) not in limit_keys \
        and quant_capability(str(r.get("quant"))) is not None:
      by_count.setdefault(int(r["count"]), set()).add(str(r["role"]).replace("_", " "))
  return {c: next(iter(names)) for c, names in by_count.items() if len(names) == 1}


def kernel_words(name:str) -> str:
  """A captured kernel's name cut to its template (runtime/cuda_device.kernel_name) with its launch grid kept, the
  part that tells two launches of one program apart: `mul_mat_vec_q<(ggml_type)12, ...> grid=17408x1x1`."""
  import re
  from boltbeam.runtime.cuda_device import kernel_name
  grid = re.search(r"grid=\S+", name)
  return kernel_name(name.split(" grid=", 1)[0]) + (f" {grid.group(0)}" if grid else "")


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
  method = (trace.get("capture") or {}).get("method")
  if method == ISOLATED:  # the kernels ran alone: the context is step 4's, as the row carries it
    return float(whole["context"])
  n = whole.get("decode_tokens") or 1
  base = float(whole["context"]) + (TINYGRAD_WARM if provider == "tinygrad" else 0)
  captured = method not in (None, OWN_TIMING)
  return base + ((n + 3) / 2 if captured else (n + 1) / 2)


def batch_limit(*, weight_ms:float, profile:dict[str, Any], context:float, batch:int, element_bytes:int,
                bandwidth_gbs:float, params:float | None = None, peak_flops:float | None = None) -> dict[str, Any]:
  """The decode limit for B streams at one context. One step reads the weights once for B tokens and each stream's
  own KV cache, so step ms = weight ms + B x KV ms. A matrix product of B columns also does 2 x params x B flops;
  when that takes longer than the reads, the step is compute bound and that is its floor. tokens/s per stream is
  1000 / step ms; in total it is B times that."""
  mem = weight_ms + batch * kv_ms(profile, context, element_bytes, bandwidth_gbs)
  compute = 2.0 * params * batch / peak_flops * 1e3 if params and peak_flops else 0.0
  step = max(mem, compute)
  return {"batch": batch, "context": context, "step_ms": step, "memory_ms": mem, "compute_ms": compute,
          "bound": "compute" if compute > mem else "memory", "tok_s_stream": 1000.0 / step,
          "tok_s_total": batch * 1000.0 / step}


def noisy_words(spread_pct:float | None, us_per_call:float | None = None, floor_us:float | None = None) -> tuple[bool, float | None, str | None]:
  """Whether a role's isolated reading is firm: (noisy, ±pct, why). The problem flagged is a kernel too short for
  the clock: on a 5090 a 3 MB kernel under a 4 µs floor swings 8 to 10 µs between passes while the chip's
  run-to-run band is ±1.4%, so the band is the wrong yardstick (a sample spread and a run-to-run band are two
  different statistics). Noisy when the samples' half-spread (P90 - P10 over the median, halved, as ±) is over
  NOISY_RULE["spread_pct"], the gap between "at the limit" and a clearly slow row; or when the dispatch floor is over
  NOISY_RULE["floor_share"] of the measured time. why names what fired: "±14.0%", "4.1 µs floor under a 7.5 µs
  kernel", or both joined by a comma. Nothing without a spread or a floor (an in-model row)."""
  half = float(spread_pct) / 2.0 if spread_pct is not None else None
  wide = half is not None and half > NOISY_RULE["spread_pct"]
  floored = floor_us is not None and us_per_call is not None and us_per_call > 0 and floor_us > NOISY_RULE["floor_share"] * us_per_call
  if not (wide or floored):
    return False, half, None
  why = ([f"±{half:.1f}%"] if half is not None else []) + ([NOISY_FLOOR.format(floor=floor_us, us=us_per_call)] if floored else [])
  return True, half, ", ".join(why)


def role_why(roles:list[dict[str, Any]], bandwidth_gbs:float, *, regimes:dict[tuple[str, str], str] | None = None,
             throttled:bool = False, latency_us:float | None = None, floor_us:float | None = None) -> tuple[list[dict[str, Any]], str]:
  """Each role with % of peak, µs per call and its reason word; and the rule as one sentence with its numbers.
  regimes is the roofline regime per (role, quant) at this context; throttled says the chip throttled while read.
  latency_us is the dispatch floor the run's probe measured on this GPU; without one the assumed figure is used,
  and the sentence says which. An isolated row (less_floor_ms set) is judged on its time less the dispatch floor:
  % of peak and GB/s are on that time, us_per_call stays the measured time and us_per_call_less_floor joins it.
  floor_us is the dispatch floor the rows carry: a role whose reading is not firm (noisy_words) carries the NOISY
  suffix, and reason_word keeps the firm word."""
  latency, latency_source = (latency_us, LATENCY_MEASURED) if latency_us else (ROLE_RULE["latency_us"], LATENCY_ASSUMED)
  in_flight = bandwidth_gbs * 1e9 * latency * 1e-6  # bytes
  small = in_flight * ROLE_RULE["fill_factor"]
  out = []
  for r in roles:
    calls = r.get("calls_per_token") or 0
    isolated = r.get("less_floor_ms") is not None
    rule_ms = r["less_floor_ms"] if isolated else r["actual_ms"]  # the time the rule judges
    pct = 100.0 * r["ideal_ms"] / rule_ms if rule_ms > 0 else None
    per_call_bytes = r["ideal_ms"] * 1e-3 * bandwidth_gbs * 1e9 / calls if calls else None
    if r.get("within_noise"):  # below its floor, inside the chip's plausibility band (tinygrad_role_time.loss)
      why = r.get("label") or REASONS["at_limit"]
    elif pct is not None and round(pct, 1) >= ROLE_RULE["at_limit_pct"]:
      why = REASONS["at_limit"]
    elif per_call_bytes is not None and per_call_bytes < small:
      why = REASONS["small"]
    elif str((regimes or {}).get((r["role"], r["quant"])) or "").startswith("compute"):
      why = REASONS["compute"]
    elif per_call_bytes is not None and not throttled:  # below the limit, big enough, memory bound, not throttled
      why = REASONS["slow"]
    else:
      why = REASONS["unexplained"]
    us = r["actual_ms"] * 1e3 / calls if calls else None  # measured, floor included
    noisy, half, why_noisy = noisy_words(r.get("spread_pct"), us, floor_us if isolated else None)
    out.append({**r, "pct_peak": round(pct, 1) if pct is not None else None, "us_per_call": us,
                "us_per_call_less_floor": rule_ms * 1e3 / calls if calls and isolated else None,
                "mb_per_call": per_call_bytes / 1e6 if per_call_bytes else None,
                "gbs": per_call_bytes / (rule_ms * 1e6 / calls) if per_call_bytes and rule_ms > 0 else None,
                "reason": NOISY.format(word=why, why=why_noisy) if noisy else why, "reason_word": why, "noisy": noisy})
  rule = (f"At the limit: {ROLE_RULE['at_limit_pct']:.0f}% of peak or more. Too small to fill memory: a call moves "
          f"under {small / 1e6:.1f} MB, which is {ROLE_RULE['fill_factor']:.0f} x the {in_flight / 1e6:.2f} MB that must "
          f"be in flight ({bandwidth_gbs:.0f} GB/s x {latency:.1f} µs latency, {latency_source}, Little's law). "
          "Compute bound: the roofline regime is compute. Slow kernel: below the limit, the call big enough, not "
          "compute bound, the chip not throttled. Anything else: unexplained.")
  if any(r.get("less_floor_ms") is not None for r in roles):
    rule += ISOLATED_RULE.format(floor_words=floor_words(floor_us) or "less the dispatch floor")
  if floor_us is not None or any(r.get("spread_pct") is not None for r in roles):
    rule += (f" Noisy: the role's own samples spread more than ±{NOISY_RULE['spread_pct']:.0f}% (half of P90 - P10 over the "
             f"median), or the dispatch floor is over {100 * NOISY_RULE['floor_share']:.0f}% of its measured time (the kernel is "
             "mostly floor), so the word is not firm.")
  return out, rule


def measured_step(trace:dict[str, Any] | None, provider:str, near:float | None = None) -> dict[str, Any] | None:
  """THE measured token of step 4: one whole-step row, picked by one rule for every reader (the headline, the Run
  line, the tie-out, summary.txt, results.json). Rows of another provider never count: one engine's kernels never
  tie out against another's token. With a per-role capture, the row nearest the context that capture attended
  (near); without one, the row at the smallest context, the token closest to the context-1 limit. The other rows
  are "also measured", labelled, never the headline."""
  if not trace:
    return None
  measured_by = trace.get("provider") or ("tinygrad" if str(trace.get("provider_id", "")).startswith("tinygrad")
                                          else "llama.cpp")
  if measured_by != provider:
    return None
  rows = [r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("tok_s")]
  if not rows:
    return None
  key = (lambda r: abs(float(r.get("context") or 0) - near)) if near is not None else (lambda r: float(r.get("context") or 0))
  row = min(rows, key=key)
  index = trace["rows"].index(row)
  return {"context": row.get("context"), "tok_s": row["tok_s"], "ms": 1000.0 / row["tok_s"], "wall_us": row.get("wall_us"),
          "source": row.get("source"),
          "graph_error": row.get("graph_error"), "index": index, "decode_tokens": row.get("decode_tokens"),
          "rule": "nearest the context the per-role capture attended" if near is not None else "the smallest context",
          "also": [{"context": r.get("context"), "tok_s": r["tok_s"], "ms": 1000.0 / r["tok_s"]} for r in rows if r is not row]}


def _step4(run:pathlib.Path, context:float | None, provider:str) -> tuple[float | None, int | None]:
  """The untraced whole-step ms per token from step 4, at the context nearest the measured one, only when step 4
  measured the same provider: one provider's kernels never tie out against another's token."""
  p = run / "timing_trace.json"
  if not p.exists():
    return None, None
  token = measured_step(json.loads(p.read_text()), provider, near=context)
  return (token["ms"], token["context"]) if token else (None, None)


def tie_out(run:pathlib.Path, *, provider:str, table:dict[str, Any] | None, trace:dict[str, Any] | None,
            limit_ms:float, profile:dict[str, Any], bandwidth_gbs:float, missing:str | None = None) -> dict[str, Any]:
  """The tie-out for one provider's measurement. `table` is loss()'s measured table (or None: no kernels here).
  For a capture of B streams the token is one decode step: the weights once, and B streams' KV caches."""
  from boltbeam.collectors.providers import kv_element
  element, element_source = kv_element(provider)
  context = attended_context(trace or {}, provider) if trace else None
  batch = int(next((r.get("batch") or 1 for r in (trace or {}).get("rows", []) if r.get("scope") == "whole_step"), 1))
  untraced, untraced_ctx = _step4(run, context, provider) if batch == 1 else (None, None)
  if context is None:
    context = float(untraced_ctx or 1)
  kv = batch * kv_ms(profile, context, element, bandwidth_gbs)
  limit = limit_ms + kv
  streams = f" for each of {batch} streams" if batch > 1 else ""
  out = {"provider": provider, "context": context, "limit_ms_ctx1": limit_ms, "limit_ms": limit, "kv_ms": kv,
         "batch": batch,
         "kv_source": f"KV cache read at context {context:.0f}{streams}: {element} bytes per element, {element_source}",
         "show_both": kv >= SHOW_BOTH_SHARE * limit_ms, "untraced_ms": untraced, "untraced_context": untraced_ctx,
         "lines": [], "token_ms": None, "token_source": None, "busy_ms": None, "missing": missing, "refused": None,
         "isolated": (trace or {}).get("capture", {}).get("method") == ISOLATED}
  whole = next((r for r in (trace or {}).get("rows", []) if r.get("scope") == "whole_step"), {})
  if table is None:  # no kernel view here: the whole step from step 4 is the token
    if untraced is None:
      return out
    out.update(token_ms=untraced, token_source=f"the untraced whole step at context {untraced_ctx}")
    out["lines"] = [{"label": f"limit at context {context:.0f} (ideal)", "ms": limit, "how": "derived"},
                    {"label": "kernels and gaps, not split", "ms": untraced - limit, "how": "difference"}]
    return out
  busy = table["kernel_ms"]
  out["band"] = (table.get("band") or {}).get("words")
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
  if out["isolated"]:
    return _isolated(out, table, token, limit, context)
  band = float((table.get("band") or {}).get("band") or 0.01)
  if busy > token * (1 + band):  # idle time cannot be negative beyond noise: the two numbers are not comparable
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
    outside = roles_outside_limit(profile, taken | {(u["role"], u["quant"]) for u in table.get("unpaired_roles") or []})
    parts: dict[str, float] = {}
    loose = []  # GEMVs no role took and no outside role names: weight time the limit does not know, its own line
    tokens = table["tokens"]
    for r in (trace or {}).get("rows", []):
      if r.get("scope") == "kernel" and (r.get("role"), r.get("quant")) not in taken:
        ms, calls = float(r["wall_us"]) / tokens / 1000.0, float(r.get("calls") or 0)
        k = kind_of(r)
        if k == UNATTRIBUTED:
          k = outside.get(round(calls / tokens), UNATTRIBUTED)
          if k == UNATTRIBUTED:
            loose.append({"kernel": kernel_words(str(r.get("kernel") or "")), "calls_per_token": calls / tokens,
                          "us_per_call": float(r["wall_us"]) / calls if calls else None, "ms": ms})
            continue
        parts[k] = parts.get(k, 0.0) + ms
    loose_ms = sum(l["ms"] for l in loose)
    lines.append({"label": "weight kernels above their ideal", "ms": weight_above, "how": "measured"})
    if loose:
      lines.append({"label": UNATTRIBUTED_LABEL, "ms": loose_ms, "how": "measured", "kernels": sorted(loose, key=lambda l: -l["ms"])})
    lines.append({"label": "other kernels above their ideal", "ms": other_busy - loose_ms - other_ideal, "how": "measured",
                  "busy_ms": other_busy - loose_ms, "ideal_ms": other_ideal,
                  "parts": [{"kind": k, "ms": v} for k, v in sorted(parts.items(), key=lambda kv_: -kv_[1])]})
  else:
    lines.append({"label": "all kernels above the ideal, not split by role", "ms": busy - limit, "how": "measured"})
  gaps = token - limit - sum(l["ms"] for l in lines[1:])
  lines.append({"label": "gaps between kernels (GPU idle)" if gaps >= 0 else
                f"gaps between kernels (below 0, within ±{100 * band:.1f}%)", "ms": gaps, "how": "difference"})
  out["lines"] = lines
  return out


def scale_roles(roles:list[dict[str, Any]], weight_ms:float | None, isolated_sum_ms:float) -> tuple[list[dict[str, Any]], float | None]:
  """An isolated run's estimated split: each role's isolated time, less the dispatch floor where the rows carry one
  (less_floor_ms), scaled by weight_ms / that sum (1.0 when the sum fits the token). est_ms and est_lost_ms carry
  estimate: True. The isolated times stay as they are, measured."""
  if not weight_ms or not isolated_sum_ms:
    return roles, None
  k = weight_ms / isolated_sum_ms
  base = lambda r: r["less_floor_ms"] if r.get("less_floor_ms") is not None else r["actual_ms"]  # noqa: E731
  return [{**r, "est_ms": base(r) * k, "est_lost_ms": base(r) * k - r["ideal_ms"], "estimate": True}
          for r in roles], k


def floor_words(floor_us:float | None) -> str:
  """"less the 3.9 µs dispatch floor per launch", or nothing when the rows carry no floor."""
  return f"less the {floor_us:.1f} µs dispatch floor per launch" if floor_us is not None else ""


def _isolated(out:dict[str, Any], table:dict[str, Any], token:float, limit:float, context:float) -> dict[str, Any]:
  """The tie-out for kernels timed alone. Only what is measured is tied out: the ideal and the token, with their
  difference as one line. No floor or over-count refusal runs. The split by role is an estimate, said once,
  labelled. The timer times weight kernels only, so the rest of the token (attention, norms, launches, gaps) is
  never 0. The estimate takes each row's time less the dispatch floor the run measured (a 4 µs event floor on a
  5090 is 30 to 130% of a small role's time; on an M3 2.4 µs barely moves it): the measured times stay as measured
  and the words say "less the N µs dispatch floor" wherever the estimate uses them. When that sum fits inside the
  token the times stand and the rest is the difference. When it over-counts (alone and cold can be slower than
  inside the token) the times are scaled down to the token less the KV reads at the limit, the one part of the rest
  the roofline knows."""
  floor_us = table.get("floor_us")
  base = (lambda r: r["less_floor_ms"]) if floor_us is not None else (lambda r: r["actual_ms"])
  weight_iso = sum(r["actual_ms"] for r in table["roles"])
  weight_base = sum(base(r) for r in table["roles"])
  other_floor = max(0.0, limit - sum(r["ideal_ms"] for r in table["roles"]))
  room = max(0.0, token - other_floor)
  scale = min(1.0, room / weight_base) if weight_base else None
  weight_est = weight_base * scale if scale is not None else None
  out["show_both"] = True
  out["lines"] = [{"label": f"limit at context {context:.0f} (ideal)", "ms": limit, "how": "derived"},
                  {"label": NOT_SPLIT_LABEL, "ms": token - limit, "how": "difference"}]
  scaled = scale is not None and scale < 1.0
  floor = f" {floor_words(floor_us)}" if floor_us is not None else ""  # in weight_how; other_how formats its own
  out["estimate"] = {"estimate": True, "label": ESTIMATE_LABEL, "scale": scale, "isolated_sum_ms": weight_iso, "token_ms": token,
                     "floor_us": floor_us, "floor_words": floor_words(floor_us) or None,
                     "columns_words": ISOLATED_COLUMNS.format(floor_words=floor_words(floor_us)) if floor_us is not None else None,
                     "isolated_sum_less_floor_ms": weight_base if floor_us is not None else None,
                     "weight_ms": weight_est, "scaled": scaled,
                     "weight_how": (f"upper bound: the token less the KV-read floor" if scaled else
                                    f"as timed alone{floor}, fits inside the token"),
                     "other_ms": token - weight_est if weight_est is not None else None,
                     "other_how": OTHER_HOW_FLOOR if scaled else OTHER_HOW_DIFF.format(floor=f", {floor_words(floor_us)}" if floor_us is not None else "")}
  return out


# --- where the token goes: the tie-out as one ranked table, read by every renderer ---------------------------------

WHERE_TITLE = "Where the token goes"
NO_LIMIT = "no limit"  # a row the roofline has no bytes for: all its time is above the limit
GAPS_ROW = "gaps (GPU idle)"  # the tie-out's gaps line, short enough for an 80-column table
REST_ROW = "other kernels, not named"  # kernel time no kind named
# the part of the limit no row took: limit roles the capture did not split out. Their kernels' time is in the other
# rows, so this row has no time now, a negative loss, and no tok/s if fixed: there is nothing to fix in it
LIMIT_ROW = "limit of roles not split out"
ESTIMATE_REST_ROW = "rest, not timed"  # attention, norms, launches and gaps: the token less the weight estimate
WHERE_COLUMNS = ("row", "ms now", "ms at limit", "ms lost", "share", "tok/s if fixed", "% of peak")  # share: of all the ms lost


def where_token_goes(tie:dict[str, Any] | None, roles:list[dict[str, Any]]) -> dict[str, Any] | None:
  """The tie-out as one table of what each part costs the token, worst first. A row is a weight role, a kind of
  other kernel, the weight kernels no role took, or the gaps (and LIMIT_ROW when the limit holds roles the capture
  did not split out). Each row: ms per token now, ms at the limit (None: the
  limit has no bytes for it, NO_LIMIT; 0 for the gaps), ms lost (now less the limit), its share of all the ms lost,
  and tok/s if this row alone reached its limit (the headline token less this row's lost). Nothing is derived here:
  the roles' times and losses (lost_ms, or est_ms and est_lost_ms for kernels timed alone) and the tie-out's lines
  are read as they are. The rows sum to the tie-out's measured token, and the limits to its limit.

  The headline token (measured_step, untraced) is the base for tok/s. When the tie-out's token is a captured run's
  (tracing slows it), the rows sum to that one and the words say so. Kernels timed alone give an estimate, said."""
  if not tie or tie.get("refused") or tie.get("token_ms") is None or not tie.get("lines"):
    return None
  token, limit = float(tie["token_ms"]), float(tie["limit_ms"])
  base = float(tie["untraced_ms"]) if tie.get("untraced_ms") is not None else token  # measured_step's token
  est = tie.get("estimate")
  rows: list[dict[str, Any]] = []
  add = lambda name, what, now, at, **kw: rows.append({"name": name, "what": what, "now_ms": now, "limit_ms": at, **kw})  # noqa: E731
  lines = tie["lines"]
  other = next((l for l in lines if l.get("parts") is not None), None)
  if est and roles and est.get("weight_ms") is not None:  # kernels timed alone: the estimate's split, labelled
    for r in roles:
      add(f"{r['role']} {r['quant']}", "role", r["est_ms"], r["ideal_ms"], lost_ms=r["est_lost_ms"], pct_peak=r.get("pct_peak"))
    add(ESTIMATE_REST_ROW, "rest", est["other_ms"], limit - sum(r["ideal_ms"] for r in roles))
  elif other is not None and roles:  # in the model: every role, every kind of other kernel, the gaps
    for r in roles:
      add(f"{r['role']} {r['quant']}", "role", r["actual_ms"], r["ideal_ms"], lost_ms=r["lost_ms"], pct_peak=r.get("pct_peak"))
    for l in lines:
      if l.get("kernels"):
        add(l["label"], "kernels", l["ms"], None)
    kv = float(tie.get("kv_ms") or 0.0)
    named = 0.0
    for p in other["parts"]:
      named += p["ms"]
      if p["kind"] == "attention":  # the limit's KV read is attention's ideal
        add(p["kind"], "kernels", p["ms"], kv)
        kv = 0.0
      else:
        add(p["kind"], "kernels", p["ms"], None)
    rest_now, rest_limit = other["busy_ms"] - named, other["ideal_ms"] - (float(tie.get("kv_ms") or 0.0) - kv)
    if abs(rest_now) > 1e-9:
      add(REST_ROW, "rest", rest_now, None)
    if abs(rest_limit) > 1e-9:
      add(LIMIT_ROW, "limit", 0.0, rest_limit)
    gaps = lines[-1]
    add(GAPS_ROW, "gaps", gaps["ms"], 0.0)
  else:  # not split by role: the tie-out's own lines, the ideal on the first one that holds kernels
    for i, l in enumerate(lines[1:]):
      gaps = l["how"] == "difference" and len(lines) > 2
      now = l["ms"] + (limit if i == 0 else 0.0)
      add(l["label"].replace(" above the ideal", "").replace(" above their ideal", ""), "gaps" if gaps else "rest",
          now, 0.0 if gaps else (limit if i == 0 else None))
  for r in rows:
    r.setdefault("lost_ms", r["now_ms"] - (r["limit_ms"] or 0.0))
    r.setdefault("pct_peak", None)
  total = sum(r["lost_ms"] for r in rows)
  for r in rows:
    r["share"] = r["lost_ms"] / total if total > 0 else None
    r["tok_s_if_fixed"] = 1000.0 / (base - r["lost_ms"]) if base - r["lost_ms"] > 0 and r["what"] != "limit" else None
  rows.sort(key=lambda r: (-r["lost_ms"], r["name"]))
  words = [f"ms per token at context {tie['context']:.0f}, worst first; tok/s if fixed is the {base:.1f} ms token "
           f"({1000.0 / base:.1f} tok/s) less that row's lost ms."]
  if abs(base - token) > 1e-9:
    words.append(f"The rows sum to the {token:.1f} ms token the kernels were measured in ({tie.get('token_source')}); "
                 f"the headline token is {base:.1f} ms untraced.")
    if token > base:
      words.append(f"The gaps row includes the capture's own overhead, about {token - base:.1f} ms (the captured token less "
                   "the untraced one), so the gaps' tok/s if fixed is an upper bound.")
  if est:
    words.append(f"Estimate: each weight kernel was timed alone; the rest of the token (attention, norms, launches and gaps) is the difference {ESTIMATE_LABEL}.")
  return {"title": WHERE_TITLE, "estimate": bool(est), "token_ms": token, "base_ms": base, "limit_ms": limit,
          "lost_ms": total, "columns": list(WHERE_COLUMNS), "no_limit": NO_LIMIT, "words": " ".join(words), "rows": rows}
