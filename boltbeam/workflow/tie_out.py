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
  + gaps between kernels       the difference: the measured token minus everything above
  = measured token                        the captured run's own time per token

Every line is measured or derived from measured facts. The gaps line is the difference and is labelled so; it
is never presented as "nothing left over". Where a line cannot be measured, it is merged with the next and the
reason is given.

The per-role reason follows one rule, with its numbers here and nowhere else (ROLE_RULE).
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any

from boltbeam.collectors.capture_settings import attended_context, measured_step, step4_token, window_context
from boltbeam.collectors.kernel_kind import UNATTRIBUTED, kind_of
from boltbeam.collectors.replay_overhead import CEILING_PCT, end_to_end, overhead_words, within_ceiling
from boltbeam.math.numerics import ACCOUNTING_ATOL, ACCOUNTING_RTOL

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
from boltbeam.collectors.tinygrad_role_time import OWN_TIMING  # the engine's own profiling, not an outside capture
from boltbeam.collectors.engine_kernels import METHOD as ISOLATED  # the engine's kernels timed alone by BoltBeam's kernel timer
# each engine's own decode timing window, recorded alike by its untraced run and its outside capture (another engine's never matches)
WINDOWS = {"llama.cpp": ("long_minus_short", "differential decode window"), "tinygrad": ("engine_decode_loop", "decode-loop window")}
REASONS = {"at_limit": "at the limit", "small": "too small to fill memory", "slow": "slow kernel",
           "compute": "compute bound", "unexplained": "unexplained"}
# a role timed alone whose reading is not firm gets this suffix after its reason word: the word is the rule's
# reading, the suffix says why it is not firm (noisy_words). One rule for both backends; reason_word keeps the firm
# word for the readers that key on it (the HTML rule tables).
NOISY = "{word}; noisy: {why}"
from boltbeam.core.policy import value

NOISY_FLOOR = "{floor:.1f} µs floor under a {us:.1f} µs kernel"
# an isolated row's rule time: its time less the dispatch floor, the figure that matched nsys per role on the 5090
# (docs/in-model-vs-generic-rtx5090-20261010.md); the measured time, floor included, stays in the row as us_per_call
ISOLATED_COLUMNS = "µs/call is the measured time, floor included; less floor, GB/s and % of peak are {floor_words}"
ISOLATED_RULE = " Timed alone: % of peak and GB/s are on the time {floor_words}; µs/call is the measured time with the floor in it. Empty-launch subtraction is an estimate, not a measured additive correction."
# the isolated tie-out: only the ideal and the token are compared; the split is an estimate, labelled so
NOT_SPLIT_LABEL = "kernels and gaps, not split"
ESTIMATE_LABEL = "(estimate from isolated times)"
OTHER_HOW_FLOOR = "at least: untimed work at the limit; untimed weights, attention, norms, launches and gaps were not timed"
OTHER_HOW_DIFF = "difference: the token less the kernels timed alone{floor}; attention, norms, launches and gaps were not timed"
ROLE_SOURCE_ISOLATED = "isolated"

UNATTRIBUTED_LABEL = "weight kernels, unattributed"  # its own tie-out line, the kernels named


def roles_outside_limit(profile:dict[str, Any] | None, limit_keys:set[tuple[str, str]]) -> dict[int, str]:
  """The profile's matrix roles the limit has no bytes for (not vocab.is_weight_gemv_role: an ssm tensor of unknown
  shape kept as `ssm_projection`), named by their calls per token: {48: "ssm projection"}.
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
  """Time to read the KV cache once for one token at this context: the KV bytes per token (prefill_work) x context,
  over the bandwidth."""
  from boltbeam.roofline.prefill_work import kv_bytes_per_token
  per = kv_bytes_per_token(profile, element_bytes=element_bytes)
  if not (per and bandwidth_gbs):
    return 0.0
  return per * context / (bandwidth_gbs * 1e9) * 1e3


def state_ms(profile:dict[str, Any], provider:str, run:pathlib.Path | None, bandwidth_gbs:float) -> tuple[float, str | None]:
  """One stream's recurrent state per decode step, read once and written once (llama.cpp: build_rs, then ggml_cpy of
  the new state; vLLM: its decode kernels load and store the state in place), at the engine's dtypes
  (providers.state_element), and the words; (0, None) without recurrent layers, (0, why) when the dtype is unknown."""
  from boltbeam.collectors.providers import state_element
  from boltbeam.roofline.prefill_work import recurrent_state_bytes
  if not (recurrent_state_bytes(profile) and bandwidth_gbs):
    return 0.0, None
  element, how = state_element(provider, run, profile)
  if element is None:
    return 0.0, f"recurrent state not counted: {how}"
  b = recurrent_state_bytes(profile, element[1], conv_bytes=element[0])
  return 2 * b / (bandwidth_gbs * 1e9) * 1e3, f"recurrent state read and written, {b / 1e6:.1f} MB each way per stream: {how}"




def step_limit_ms(weight_ms:float, batch:int, kv_ms_stream:float, state_ms_stream:float, shards:int = 1) -> float:
  """The one decode limit rule: a step reads the weights once, and each of its B streams' KV cache and recurrent state;
  a tensor-parallel split of `shards` GPUs reads its share of those at once. Every decode limit is this (batch_limit,
  tie_out, results.decode_ceiling)."""
  return weight_ms + batch * (kv_ms_stream + state_ms_stream) / shards


def batch_limit(*, weight_ms:float, profile:dict[str, Any], context:float, batch:int, element_bytes:int,
                bandwidth_gbs:float, params:float | None = None, peak_flops:float | None = None,
                state:tuple[float, str | None] = (0.0, None)) -> dict[str, Any]:
  """The decode limit for B streams at one context. One step reads the weights once for B tokens and each stream's
  own KV cache and recurrent state (state_ms: ms per stream, words), so step ms = weight ms + B x (KV ms + state ms). A matrix product of B columns also does 2 x params x B flops;
  when that takes longer than the reads, the step is compute bound and that is its floor. tokens/s per stream is
  1000 / step ms; in total it is B times that."""
  mem = step_limit_ms(weight_ms, batch, kv_ms(profile, context, element_bytes, bandwidth_gbs), state[0])
  compute = 2.0 * params * batch / peak_flops * 1e3 if params and peak_flops else 0.0
  step = max(mem, compute)
  return {"batch": batch, "context": context, "step_ms": step, "memory_ms": mem, "compute_ms": compute,
          "bound": "compute" if compute > mem else "memory", "tok_s_stream": 1000.0 / step,
          "tok_s_total": batch * 1000.0 / step, **({"state_ms": batch * state[0], "state_words": state[1]} if state[1] else {})}


def noisy_words(spread_pct:float | None, us_per_call:float | None = None, floor_us:float | None = None) -> tuple[bool, float | None, str | None]:
  """Whether a role's isolated reading is firm: (noisy, ±pct, why). The problem flagged is a kernel too short for
  the clock: on a 5090 a 3 MB kernel under a 4 µs floor swings 8 to 10 µs between passes while the chip's
  run-to-run band is ±1.4%, so the band is the wrong yardstick (a sample spread and a run-to-run band are two
  different statistics). Noisy when the samples' half-spread (P90 - P10 over the median, halved, as ±) is over
  the policy noisy_spread_pct, the gap between "at the limit" and a clearly slow row; or when the dispatch floor is over
  1 / noisy_floor_multiple of the measured time. why names what fired: "±14.0%", "4.1 µs floor under a 7.5 µs
  kernel", or both joined by a comma. Nothing without a spread or a floor (an in-model row)."""
  half = float(spread_pct) / 2.0 if spread_pct is not None else None
  wide = half is not None and half > value("noisy_spread_pct")
  floored = floor_us is not None and us_per_call is not None and us_per_call > 0 and floor_us * value("noisy_floor_multiple") > us_per_call
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
    rule += (f" Noisy: the role's own samples spread more than ±{value('noisy_spread_pct'):.0f}% (half of P90 - P10 over the "
             f"median), or the dispatch floor is over {100 / value('noisy_floor_multiple'):.0f}% of its measured time (the kernel is "
             "mostly floor), so the word is not firm.")
  return out, rule


def tie_out(run:pathlib.Path, *, provider:str, table:dict[str, Any] | None, trace:dict[str, Any] | None,
            limit_ms:float, profile:dict[str, Any], bandwidth_gbs:float, missing:str | None = None) -> dict[str, Any]:
  """The tie-out for one provider's measurement. `table` is loss()'s measured table (or None: no kernels here).
  For a capture of B streams the token is one decode step: the weights once, and B streams' KV caches."""
  from boltbeam.collectors.providers import kv_element
  element, element_source = kv_element(provider, run)
  context = attended_context(trace or {}, provider) if trace else None
  batch = int(next((r.get("batch") or 1 for r in (trace or {}).get("rows", []) if r.get("scope") == "whole_step"), 1))
  untraced, untraced_ctx = step4_token(run, context, provider) if batch == 1 else (None, None)
  if context is None:
    context = float(untraced_ctx or 1)
  kv_stream = kv_ms(profile, context, element, bandwidth_gbs)
  kv = batch * kv_stream
  st, st_words = state_ms(profile, provider, run, bandwidth_gbs)
  limit = step_limit_ms(limit_ms, batch, kv_stream, st)
  streams = f" for each of {batch} streams" if batch > 1 else ""
  out = {"provider": provider, "context": context, "limit_ms_ctx1": limit_ms, "limit_ms": limit, "kv_ms": kv,
         "batch": batch, **({"state_ms": batch * st, "state_source": st_words} if st_words else {}),
         "kv_source": f"KV cache read at context {context:g}{streams}: {element} bytes per element, {element_source}",
         "show_both": kv >= SHOW_BOTH_SHARE * limit_ms, "untraced_ms": untraced, "untraced_context": untraced_ctx,
         "lines": [], "token_ms": None, "token_source": None, "busy_ms": None, "missing": missing, "refused": None,
         "isolated": (trace or {}).get("capture", {}).get("method") == ISOLATED}
  whole = next((r for r in (trace or {}).get("rows", []) if r.get("scope") == "whole_step"), {})
  if (kind := WINDOWS.get(provider, (None,))[0]) and (trace or {}).get("capture", {}).get("method") not in (None, ISOLATED, OWN_TIMING):
    path = run / "timing_trace.json"
    native = measured_step(json.loads(path.read_text()), provider, near=context) if path.exists() else None
    captured_window = (whole.get("engine") or {}).get("timing_window")
    matches = bool((window_context(whole) is not None if kind == "long_minus_short" else (captured_window or {}).get("method") == kind) and native and native.get("timing_window") == captured_window)
    out["timing_match"] = {"matches": matches, "native_window": (native or {}).get("timing_window"),
                           "captured_window": captured_window,
                           "words": f"same {WINDOWS[provider][1]} and fresh-process lifecycle" if matches else
                                    "native and captured decode timing need matching explicit windows and lifecycles"}
  if table is None:  # no kernel view here: the whole step from step 4 is the token
    if untraced is None:
      return out
    out.update(token_ms=untraced, token_source=f"the untraced whole step at context {untraced_ctx}")
    out["lines"] = [{"label": f"limit at context {context:g} (ideal)", "ms": limit, "how": "derived"},
                    {"label": "kernels and gaps, not split", "ms": untraced - limit, "how": "difference"}]
    return out
  busy = table["kernel_ms"]
  out["band"] = (table.get("band") or {}).get("words")
  # Every capture reconciles to its own wall. Profiling perturbation is a separate comparison with
  # the untraced step; kernel busy time is never substituted for measured elapsed time.
  outside = (trace or {}).get("capture", {}).get("method") not in (None, OWN_TIMING)
  if whole.get("token_ms"):
    token, source = whole["token_ms"], "the captured run's own time per token" if batch == 1 else "the captured run's own time per step"
    if window_context(whole) is not None:
      source = "the captured long-minus-short elapsed estimate per token"
  else:
    token, source = untraced, f"the untraced whole step at context {untraced_ctx}"
  if not outside:
    out["profiled_wall_ms"] = whole.get("token_ms")
    out["profile_overhead_ms"] = token - untraced if token is not None and untraced is not None else None
    out["measurement_scope"] = "profiled wall; residual includes host execution and profiling overhead"
  out.update(token_ms=token, token_source=source, busy_ms=busy)
  if token is None:
    out["refused"] = f"no measured token for {provider} in this run to tie the kernels out against"
    return out
  if out["isolated"]:
    if cap := (trace or {}).get("captured_token"):  # a capture held the whole token but named no kernel (attribution.UnnamedCapture):
      # its host step against the untraced one over identical recorded windows, GPU busy beside it; the roles stay the estimate
      native = measured_step(json.loads((run / "timing_trace.json").read_text()), provider, near=context) if untraced is not None else None
      cap_ms, busy_ms = float(cap["token_ms"]), float(cap["busy_ms"])
      out["end_to_end"] = e2e = end_to_end({"measured_ms": cap_ms, "wall_ms": cap_ms, "busy_ms": busy_ms, "gaps_ms": cap_ms - busy_ms, "timing_window": cap.get("timing_window")},
                                           untraced, "token", reference="untraced reference", reference_window=(native or {}).get("timing_window"))
      out["captured_token"] = {**cap, "untraced_ms": untraced, "words": f"Whole token captured by {cap['method']}, not split by role: captured {cap_ms:.2f} ms vs untraced "
                               + (f"{untraced:.2f} ms" if untraced else "not measured") + f", GPU busy {busy_ms:.2f} ms of it; " + verdict_words(e2e) + "."}
    return _isolated(out, table, token, limit, context)
  match = out.get("timing_match") or {}
  out["end_to_end"] = end_to_end({"measured_ms": whole.get("token_ms"), "wall_ms": token,
                                  "busy_ms": busy, "gaps_ms": token - busy, "timing_window": match.get("captured_window")},
                                 untraced, "token", band=(table.get("band") or {}).get("calibration"),
                                 reference="untraced reference", reference_window=match.get("native_window"))
  if outside and out["end_to_end"].get("confirms") is False and (cost := launch_cost(trace, out["end_to_end"]["diff_ms"])):
    out["launch_cost"] = cost
  if outside and out["end_to_end"].get("confirms") is False and untraced is not None:  # the capture's gaps are not the engine's
    out.update(over_ceiling(busy=busy, gaps=token - busy, untraced=untraced, replicas=(trace or {}).get("replicas"), of=(trace or {}).get("replicas_of")))
    if out.get("refused"):
      return out
  spread = (table.get("band") or {}).get("width_ms")
  out["check_status"] = table.get("check_status", "not checked")
  if spread is not None and busy - token > spread:  # idle time cannot be negative beyond noise: the two numbers are not comparable
    numbers = (f"profiled kernels do not fit the real token: they sum to {busy:.3f} ms, more than the "
               f"{token:.3f} ms token ({source})")
    out["refused"] = (f"{provider}'s profiling mode inflates kernel times on this GPU, so its tie-out cannot be shown "
                      f"({numbers})" if not outside else numbers)
    return out
  roles, doubted = table["roles"], table.get("ceiling_refused") or []
  weight_ideal = sum(r["ideal_ms"] for r in roles + doubted)
  weight_above = sum(r["actual_ms"] - r["ideal_ms"] for r in roles)
  other_busy = busy - sum(r["actual_ms"] for r in roles + doubted)
  other_ideal = limit - weight_ideal
  lines = [{"label": f"limit at context {context:g} (ideal)", "ms": limit, "how": "derived"}]
  if roles:
    taken = {(r["role"], r["quant"]) for r in roles + doubted}
    outside_roles = roles_outside_limit(profile, taken | {(u["role"], u["quant"]) for u in table.get("unpaired_roles") or []})
    parts: dict[str, float] = {}
    loose = []  # GEMVs no role took and no outside role names: weight time the limit does not know, its own line
    tokens = table["tokens"]
    for r in (trace or {}).get("rows", []):
      if r.get("scope") == "kernel" and (r.get("role"), r.get("quant")) not in taken:
        ms, calls = float(r["wall_us"]) / tokens / 1000.0, float(r.get("calls") or 0)
        k = kind_of(r)
        if k == UNATTRIBUTED:
          k = outside_roles.get(round(calls / tokens), UNATTRIBUTED)
          if k == UNATTRIBUTED:
            loose.append({"kernel": kernel_words(str(r.get("kernel") or "")), "calls_per_token": calls / tokens,
                          "us_per_call": float(r["wall_us"]) / calls if calls else None, "ms": ms})
            continue
        parts[k] = parts.get(k, 0.0) + ms
    loose_ms = sum(l["ms"] for l in loose)
    lines.append({"label": "weight kernels above their ideal", "ms": weight_above, "how": "measured"})
    if doubted:  # a row below its own floor: its time is measured, its ceiling refused (tinygrad_role_time.loss)
      lines.append({"label": "ceiling refused: " + ", ".join(f"{r['role']} {r['quant']}" for r in doubted),
                    "ms": sum(r["actual_ms"] - r["ideal_ms"] for r in doubted), "how": "measured, below its floor"})
    if loose:
      lines.append({"label": UNATTRIBUTED_LABEL, "ms": loose_ms, "how": "measured", "kernels": sorted(loose, key=lambda l: -l["ms"])})
    lines.append({"label": "other kernels above their ideal", "ms": other_busy - loose_ms - other_ideal, "how": "measured",
                  "busy_ms": other_busy - loose_ms, "ideal_ms": other_ideal,
                  "parts": [{"kind": k, "ms": v} for k, v in sorted(parts.items(), key=lambda kv_: -kv_[1])]})
  else:
    lines.append({"label": "all kernels above the ideal, not split by role", "ms": busy - limit, "how": "measured"})
  gaps = token - limit - sum(l["ms"] for l in lines[1:])
  lines.append({"label": ("gaps between kernels" if outside else "captured wall outside GPU kernels") if gaps >= 0 else
                (f"gaps between kernels (below 0, within measured repeat spread ±{spread:.6f} ms)" if spread is not None else
                 "gaps between kernels (below 0; not checked: no repeat band)"), "ms": gaps, "how": "difference"})
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
  out["lines"] = [{"label": f"limit at context {context:g} (ideal)", "ms": limit, "how": "derived"},
                  {"label": NOT_SPLIT_LABEL, "ms": token - limit, "how": "difference"}]
  scaled = scale is not None and scale < 1.0
  floor = f" {floor_words(floor_us)}" if floor_us is not None else ""  # in weight_how; other_how formats its own
  out["estimate"] = {"estimate": True, "label": ESTIMATE_LABEL, "scale": scale, "isolated_sum_ms": weight_iso, "token_ms": token,
                     "floor_us": floor_us, "floor_words": floor_words(floor_us) or None,
                     "columns_words": ISOLATED_COLUMNS.format(floor_words=floor_words(floor_us)) if floor_us is not None else None,
                     "isolated_sum_less_floor_ms": weight_base if floor_us is not None else None,
                     "weight_ms": weight_est, "scaled": scaled,
                     "weight_how": (f"upper bound: the token less the untimed-work floor" if scaled else
                                    f"as timed alone{floor}, fits inside the token"),
                     "other_ms": token - weight_est if weight_est is not None else None,
                     "other_how": OTHER_HOW_FLOOR if scaled else OTHER_HOW_DIFF.format(floor=f", {floor_words(floor_us)}" if floor_us is not None else "")}
  return out


# --- where the token goes: the tie-out as one ranked table, read by every renderer ---------------------------------

WHERE_TITLE = "Where the token goes"
NO_LIMIT = "no model"  # a row the roofline has no bytes for: all its time is above the limit
GAPS_ROW = "kernel gaps"  # the tie-out's gaps line, short enough for an 80-column table
GAPS_BOUND_ROW = "kernel gaps, at most"  # over the ceiling: the untraced token less the kernels (tie["gaps_bound"])
REST_ROW = "other kernels, not named"  # kernel time no kind named
# the part of the limit no row took: limit roles the capture did not split out. Their kernels' time is in the other
# rows, so this row has no time now, a negative loss, and no tok/s if fixed: there is nothing to fix in it
LIMIT_ROW = "limit of roles not split out"
ESTIMATE_REST_ROW = "rest, not timed"  # attention, norms, launches and gaps: the token less the weight estimate
WHERE_COLUMNS = ("row", "ms now", "ms at limit", "ms at chip best", "ms lost", "share", "tok/s if fixed",
                 "% of peak")  # share: of all the ms lost


def role_name(r:dict[str, Any]) -> str:
  """A role row's name in the where table: role and format. A role and format the model holds at two shapes names the
  row's part (profile/roles.shape_parts): its fine class alone ("attention_q Q5_K"), else the role and the tensor's own
  word ("hyper_connection_attn.down Q8_0"). report/plain.part_name reads it back."""
  return f"{role_label(r)} {r['quant']}"


def role_label(r:dict[str, Any]) -> str:
  """role_name without its format: the role, its fine class, or role.part. The summary's role list writes it."""
  from boltbeam.vocab import ROLE_GROUP_OF
  part = r.get("part")
  return r["role"] if not part else part if part in ROLE_GROUP_OF else r["role"] + "." + part


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
  captured, limit = float(tie["token_ms"]), float(tie["limit_ms"])
  base = float(tie["untraced_ms"]) if tie.get("untraced_ms") is not None else captured  # measured_step's token
  est, bound = tie.get("estimate"), tie.get("gaps_bound")
  token = base if bound else captured  # over the ceiling, the gaps are the bound and the rows sum to the untraced token
  project = bool(est) or ((tie.get("timing_match") or {}).get("matches") is True
                         and ((tie.get("end_to_end") or {}).get("confirms") is True or rows_stand(tie)))
  rows: list[dict[str, Any]] = []
  add = lambda name, what, now, at, **kw: rows.append({"name": name, "what": what, "now_ms": now, "limit_ms": at, **kw})  # noqa: E731
  lines = tie["lines"]
  other = next((l for l in lines if l.get("parts") is not None), None)
  if est and roles and est.get("weight_ms") is not None:  # kernels timed alone: the estimate's split, labelled
    for r in roles:
      add(role_name(r), "role", r["est_ms"], r["ideal_ms"], lost_ms=r["est_lost_ms"], pct_peak=r.get("pct_peak"))
    add(ESTIMATE_REST_ROW, "rest", est["other_ms"], limit - sum(r["ideal_ms"] for r in roles))
  elif other is not None and roles:  # in the model: every role, every kind of other kernel, the gaps
    for r in roles:
      add(role_name(r), "role", r["actual_ms"], r["ideal_ms"], lost_ms=r["lost_ms"], pct_peak=r.get("pct_peak"))
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
    if kv:
      add("KV-read ideal, attention not identified", "limit", 0.0, kv)
      rest_limit -= kv
    if abs(rest_limit) > 1e-9:
      add(LIMIT_ROW, "limit", 0.0, rest_limit)
    gaps = lines[-1]
    add("outside GPU kernels (profiled wall)" if tie.get("profiled_wall_ms") is not None else GAPS_BOUND_ROW if bound else GAPS_ROW,
        "gaps", bound["ms"] if bound else gaps["ms"], 0.0)
  else:  # not split by role: the tie-out's own lines, the ideal on the first one that holds kernels
    for i, l in enumerate(lines[1:]):
      gaps = l["how"] == "difference" and len(lines) > 2
      now = l["ms"] + (limit if i == 0 else 0.0)
      add(GAPS_BOUND_ROW if gaps and bound else l["label"].replace(" above the ideal", "").replace(" above their ideal", ""),
          "gaps" if gaps else "rest", bound["ms"] if gaps and bound else now, 0.0 if gaps else (limit if i == 0 else None))
  total = rank(rows, lambda r: 1000.0 / (base - r["lost_ms"]) if project and base - r["lost_ms"] > 0 and r["what"] != "limit" else None,
               "tok_s_if_fixed")
  per = f"ms per step of {tie['batch']} streams" if (tie.get("batch") or 1) > 1 else "ms/token"
  words = [f"Context {tie['context']:g} · {token:.2f} {per}. No model = measured time, unknown savings. Roofline is an upper bound."]
  if abs(base - captured) > 1e-9:
    words.append(f"Captured {captured:.1f} ms; untraced {base:.1f} ms.")
    words.append(f"Difference: {captured-base:+.1f} ms; " + (tie["launch_cost"]["words"] if tie.get("launch_cost") else "its cause is unassigned."))
  if not project:
    words.append("Native throughput allocation unavailable: capture timing is unqualified.")
  if (e2e := tie.get("end_to_end") or {}).get("status") == "checked":  # a checked verdict is always disclosed, over the ceiling too
    words.append(verdict_words(e2e) + ".")
  if est:
    words.append("Estimate from isolated weight timings; the remaining time is inferred.")
    words.append("% of peak uses isolated time minus the measured dispatch floor.")
  if any(r["name"] == LIMIT_ROW for r in rows):
    words.append("The negative unsplit-limit row balances the modeled budget; read it with the unsplit kernels.")
  qualification = qualification_words(tie)  # which rows stand, said beside the table
  return decode_progress(table(WHERE_TITLE, WHERE_COLUMNS, rows, now=token, limit=limit, lost=total, no_limit=NO_LIMIT,
               if_fixed=("tok_s_if_fixed", "{:.1f}"), peak=("pct_peak", "{:.1f}%"), sum_name="= token",
               estimate=bool(est), token_ms=token, base_ms=base, limit_ms=limit, words=" ".join(words),
               **({"qualification": qualification} if qualification else {})), allocate=project)


def rank(rows:list[dict[str, Any]], if_fixed, key:str) -> float:
  """THE ranking of a where table, for decode and prefill alike: each row's ms lost (now less its limit), its share
  of all the ms lost, what the headline would be with that row alone fixed (if_fixed, kept under key), worst first.
  Returns the ms lost in all."""
  for r in rows:
    r.setdefault("lost_ms", r["now_ms"] - (r["limit_ms"] or 0.0))
    r.setdefault("pct_peak", None)
  total = sum(r["lost_ms"] for r in rows)
  for r in rows:
    r["share"] = r["lost_ms"] / total if total > 0 else None
    r[key] = if_fixed(r)
  rows.sort(key=lambda r: (-r["lost_ms"], r["name"]))
  return total


def table(title:str, columns:tuple[str, ...], rows:list[dict[str, Any]], *, now:float, limit:float, lost:float,
          no_limit:str, if_fixed:tuple[str, str], peak:tuple[str, str], sum_name:str, **fields:Any) -> dict[str, Any]:
  """A where table as every renderer reads it: the rows' numbers and, beside them, each row's cells as text in the
  order of `columns` (after the row's name) and the sum row's cells. Each row has two ceilings: limit_ms, the engine's
  own path (decode: the bytes at the bandwidth), and best_ms, the chip-best path for its data (prefill_work.chip_best;
  the limit when the row has none). The text summary, the HTML report, the TUI and Emit print the cells; none formats
  or recomputes a number."""
  for row in rows:
    row['section']=('Timing adjustments' if row['what'] in ('gaps','overlap','limit') else
                    'Model operations' if row['limit_ms'] is not None else 'Supporting kernels — no model')
  section_order={'Model operations':0,'Supporting kernels — no model':1,'Timing adjustments':2}
  rows.sort(key=lambda row:(section_order[row['section']],-row['lost_ms'],row['name']))
  f = lambda v, fmt: fmt.format(v) if v is not None else ""  # noqa: E731
  for r in rows:
    r.setdefault("best_ms", r["limit_ms"])
    r["cells"] = [f(r["now_ms"], "{:.2f}"), f(r["limit_ms"], "{:.2f}") or no_limit, f(r["best_ms"], "{:.2f}") or no_limit,
                  f(r["lost_ms"], "{:.2f}"), f(None if r["share"] is None else 100 * r["share"], "{:.0f}%"),
                  f(r.get(if_fixed[0]), if_fixed[1]), f(r.get(peak[0]), peak[1])]
  best = sum(r["best_ms"] or 0.0 for r in rows)
  return {"title": title, "lost_ms": lost, "columns": list(columns), "no_limit": no_limit, "rows": rows, "best_ms": best,
          "sum_cells": [sum_name, f"{now:.2f}", f"{limit:.2f}", f"{best:.2f}", f"{lost:.2f}", "100%", "", ""], **fields}


# --- where the prefill time goes: the same table, with each kernel's ceiling at its path's measured peak ------------

PREFILL_TITLE = "Where the prefill time goes"
PREFILL_COLUMNS = ("row", "ms now", "ms at engine path", "ms at chip best", "ms lost", "share", "prefill ms if fixed", "MFU")
NO_CEILING = "no model"  # a kernel with no operations or bytes the model knows (norms, quantize, copies): all its time is lost



def verdict_words(e2e:dict[str, Any]) -> str:
  """An end-to-end verdict as every table says it: its words before the parenthesised numbers."""
  return e2e["words"].split(" (", 1)[0]


def check_end_to_end(e:dict[str, Any]) -> list[str]:
  """Why an end_to_end verdict is forged or incoherent: diff_pct and diff_ms are recomputed from captured_ms and
  untraced_ms, and `confirms` must be exactly the ceiling rule (None only when the status is not checked)."""
  c, u = e.get("captured_ms"), e.get("untraced_ms")
  if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (c, u)): return [f"captured_ms {c!r} and untraced_ms {u!r} must be present, finite and positive"]
  pct, ok = 100 * (c - u) / u, e.get("confirms")
  bad = [f"{k}: reported {e[k]!r}, derived {d!r}" for k, d in (("diff_ms", c - u), ("diff_pct", pct))
         if k in e and not math.isclose(e[k], d, rel_tol=ACCOUNTING_RTOL, abs_tol=ACCOUNTING_ATOL)]
  if (ok is None) == (e.get("status") == "checked") or ok is not None and ok != within_ceiling(pct):
    bad.append(f"confirms {ok!r} with status {e.get('status')!r} is not the {CEILING_PCT:g}% capture-overhead rule at {pct:+.3f}%")
  if e.get("status") == "checked" and overhead_words(pct, "native" if (ref := e.get("reference", "native")).startswith("untraced") else ref) not in str(e.get("words")): bad.append("words do not disclose the capture overhead derived from captured_ms and untraced_ms")
  return bad


SHORT_LAUNCH_US = 10.0  # a launch this short is mostly fixed cost: what a per-launch capture cost weighs on


def launch_cost(trace:dict[str, Any] | None, diff_ms:float) -> dict[str, Any] | None:
  """An over-ceiling capture's extra time per token, per launch of the engine's token, from the capture's own counts:
  the whole-step row's launches over its tokens, and the kernel rows' calls and time. A capture that intercepts every
  dispatch (rocprofv3 --kernel-trace) costs about the same per launch on any engine, so how far it lands over the
  ceiling is the engine's launch count and length: said, as measured, never as the run's fault. None without counts."""
  rows = (trace or {}).get("rows") or []
  whole = next((r for r in rows if r.get("scope") == "whole_step" and r.get("launch_count") and r.get("decode_tokens")), None)
  kernels = [r for r in rows if r.get("scope") == "kernel" and r.get("calls")]
  if whole is None or not kernels or diff_ms <= 0:
    return None
  per_token = whole["launch_count"] / whole["decode_tokens"]
  calls = sum(r["calls"] for r in kernels)
  short = sum(r["calls"] for r in kernels if r["wall_us"] / r["calls"] < SHORT_LAUNCH_US)
  us = 1000.0 * diff_ms / per_token
  return {"launches_per_token": per_token, "us_per_launch": us, "short_share": short / calls, "short_us": SHORT_LAUNCH_US,
          "words": (f"the capture adds {diff_ms:.1f} ms a token over the engine's {per_token:,.0f} launches a token, "
                    f"{us:.1f} us each, and {100 * short / calls:.0f}% of those launches take under {SHORT_LAUNCH_US:g} us: "
                    "a cost per launch weighs on an engine that splits its token into many short kernels.")}


# Over the ceiling, kernel rows stand when the run's repeats (over_ceiling names each engine's) agree per row:
# nsys times kernels on the GPU, its cost is on the CPU between launches, in the gaps. Qwen3.8-27B on two H100 pods captured
# at +25.5% and +11.9%: kernels 12.012 vs 12.023 ms, the 11 roles of 3% of kernel time or more within 0.8%, all 30 within
# 2.4%, gaps 3.411 vs 1.778 ms (docs/pods.md, in-model).
# a row of this share of the kernel time or more within this %, any row within the last (data/policy.json)
REPLICA_TOL = (value("replica_large_row_share"), value("replica_large_row_pct"), value("replica_any_row_pct"))




REPLICA_FLOOR = REPLICA_TOL[0] * REPLICA_TOL[1] / 100  # of the kernel time: what the rule allows its smallest large row


def replication(replicas:list[dict[str, float]] | None, of:str | None = None) -> dict[str, Any] | None:
  """Whether a run's repeats agree per kernel row: each row's spread (max - min over the mean; a row one repeat lacks
  is 0 there) within REPLICA_TOL. A row further apart in %, but by less than REPLICA_FLOOR of the kernel time, is finer
  than the rule resolves and is named, not counted. `of` says what the repeats are when they are not fresh captures
  (a semantic capture's step groups). None with fewer than two."""
  if len(replicas or []) < 2:
    return None
  names = sorted(set().union(*replicas))
  mean = {k: sum(r.get(k, 0.0) for r in replicas) / len(replicas) for k in names}
  apart = {k: max(r.get(k, 0.0) for r in replicas) - min(r.get(k, 0.0) for r in replicas) for k in names}
  spread = {k: 100 * apart[k] / mean[k] if mean[k] else 0.0 for k in names}
  tol = {k: REPLICA_TOL[1] if mean[k] >= REPLICA_TOL[0] * sum(mean.values()) else REPLICA_TOL[2] for k in names}
  over = [k for k in names if spread[k] > tol[k] + 1e-9]
  fine = [k for k in over if apart[k] <= REPLICA_FLOOR * sum(mean.values())]
  judged = [k for k in names if k not in fine] or names
  worst, ok = max(judged, key=lambda k: spread[k] / tol[k]), len(fine) == len(over)
  rule = f"{REPLICA_TOL[1]:g}% for a row of {100 * REPLICA_TOL[0]:g}% of kernel time or more, {REPLICA_TOL[2]:g}% for any"
  what = f"{len(replicas)} {of or 'captures'}"
  below = (f" Not counted: {len(fine)} row{'s' * (len(fine) > 1)} apart by under {100 * REPLICA_FLOOR:g}% of the kernel time (the largest, "
           f"{(top := max(fine, key=apart.get))}, by {apart[top]:.4f} ms)." if fine and ok else "")
  return {"captures": len(replicas), **({"of": of} if of else {}), "rows": len(judged), "worst_row": worst, "worst_pct": spread[worst],
          "tolerance_pct": tol[worst], "qualifies": ok, **({"below_resolution": fine} if fine else {}),
          "words": (f"Kernel rows qualified by replication: {what} agree, worst row {spread[worst]:.1f}% apart "
                    f"({worst}) over {len(judged)} rows (tolerance {rule}).{below}" if ok else
                    f"Kernel rows unqualified: {what} disagree, {worst} {spread[worst]:.1f}% apart, over "
                    f"its {tol[worst]:g}% tolerance ({rule}).")}


def over_ceiling(*, busy:float, gaps:float, untraced:float, replicas:list[dict[str, float]] | None, of:str | None = None) -> dict[str, Any]:
  """The one rule for a decode capture over the capture-overhead ceiling, for every engine and both table builders
  (tie_out for a trace, where_decode_capture_goes for a semantic capture). Recording slows the CPU between launches,
  so the captured gaps are not the engine's: they become a bound, the untraced token less the kernels (gaps_bound),
  and the kernel rows stand only when the run's repeats agree (replication). An engine hands in its repeats as rows
  of ms and says what they are (`of`): fresh captures (llama_bench_decode, engine_common.captured_role_time) or groups
  of one capture's decode steps (semantic_attribution.step_replicas). Refused when the kernels alone exceed the
  untraced token by more than the rows' own precision."""
  if busy > untraced * (1 + REPLICA_TOL[1] / 100):
    return {"refused": (f"the capture is over the {CEILING_PCT:g}% capture-overhead ceiling and its kernels alone sum to {busy:.3f} ms, "
                        f"{100 * (busy / untraced - 1):.1f}% more than the {untraced:.3f} ms untraced token (over the rows' "
                        f"{REPLICA_TOL[1]:g}%), so the gaps cannot be bounded")}
  return {"gaps_bound": {"ms": untraced - busy, "captured_ms": gaps, "untraced_ms": untraced, "busy_ms": busy},
          "replication": replication(replicas, of)}


rows_stand = lambda rule: bool(rule.get("gaps_bound")) and (rule.get("replication") or {}).get("qualifies") is True  # noqa: E731


def qualification_words(rule:dict[str, Any]) -> str | None:
  """What over_ceiling found, as the note above the table: which rows stand, and the gaps as a bound."""
  if not (bound := rule.get("gaps_bound")):
    return None
  return " ".join([(rule.get("replication") or {}).get("words") or "Kernel rows unqualified: one capture; a second capture of this run would qualify them by replication.",
    (f"Gaps shown as a bound: at most {bound['ms']:.2f} ms (untraced token − kernels); the capture showed {bound['captured_ms']:.2f} ms, added by recording."
     if bound['ms'] >= 0 else
     f"Gaps ≈ 0: the kernels sum {100 * -bound['ms'] / bound['untraced_ms']:.2f}% above the untraced token (recording slowed them that "
     f"much, within the rows' {REPLICA_TOL[1]:g}%); a gaps row of {bound['ms']:.2f} ms reconciles the table to it.")])


def capture_rows(attr:dict[str, Any]) -> list[dict[str, Any]]:
  """One numeric row schema for captured decode and prefill tables."""
  rows = [{"name": r["name"], "what": r["what"], "now_ms": r["ms"], "limit_ms": r.get("ceiling_ms"),
           "best_ms": r.get("best_ms", r.get("ceiling_ms")), "best_path": r.get("best_path"), "mfu_pct": r.get("mfu_pct"),
           "path": r.get("path"), "bound": r.get("bound"), "calls": r.get("calls"), "refused": r.get("refused"),
           "flops": r.get("flops"), "compute_ms": r.get("compute_ms"), "bytes": r.get("bytes"), "memory_ms": r.get("memory_ms"),
           "peak_tflops": r.get("peak_tflops"), "template": r.get("template"),
           "limit_range_ms":r.get("ceiling_range_ms")} for r in attr["rows"]]
  for row in rows:
    if interval:=row.get('limit_range_ms'):
      row['name']+=f' [instruction path unconfirmed; limit {interval[0]:.6g}–{interval[1]:.6g} ms]'
  rows.append({"name": GAPS_ROW, "what": "gaps", "now_ms": attr["gaps_ms"], "limit_ms": 0.0, "mfu_pct": None})
  return rows


def capture_limit_notes(attr:dict[str, Any]) -> list[str]:
  """Explain signed nominal-limit overruns without changing accounting or validating evidence."""
  notes = sorted({message for row in attr['rows'] for message in row.get('limitations', [])})
  peaks = (attr.get('uncertainty') or {}).get('peaks') or {}
  for row in attr['rows']:
    if interval:=row.get('ceiling_range_ms'):
      notes.append(f"{row['name']}: instruction path unconfirmed; limit {interval[0]:.6g}–{interval[1]:.6g} ms from measured SIMT and tensor-core rates. Totals and efficiency use the optimistic endpoint.")
    limit = row.get('ceiling_ms')
    if row.get('what') == 'overlap' or limit is None or limit <= row['ms']:
      continue
    recorded = ', '.join(f'{path} {100 * peaks[path]:g}%' for path in sorted(row.get('compute_paths') or {}) if peaks.get(path) is not None)
    notes.append(f"{row['name']}: nominal limit exceeds measured row time by {limit-row['ms']:.6g} ms; "
                 "the signed loss is retained in the totals even when rounded to negative zero. "
                 + (f"Recorded compute peak uncertainty: {recorded}." if recorded else "No compute peak uncertainty recorded for this row."))
  return notes


def where_decode_capture_goes(attr:dict[str, Any], *, untraced_ms:float | None,
                             context:int) -> dict[str, Any]:
  """A semantically attributed decode step, using the shared row/table arithmetic."""
  rows=capture_rows(attr)
  wall=float(attr['wall_ms'])
  base=float(untraced_ms) if untraced_ms is not None else wall
  comparable=(attr.get('end_to_end') or {}).get('confirms') is True or (untraced_ms is not None and abs(base-wall)<1e-9)
  rule={}
  if (attr.get('end_to_end') or {}).get('confirms') is False and untraced_ms is not None:  # over the ceiling: the one rule
    gaps=rows[-1]
    rule=over_ceiling(busy=wall-gaps['now_ms'],gaps=gaps['now_ms'],untraced=base,replicas=attr.get('replicas'),of=attr.get('replicas_of'))
    if bound:=rule.get('gaps_bound'):  # the gaps are the bound and the rows sum to the untraced token
      gaps.update(name=GAPS_BOUND_ROW,now_ms=bound['ms'])
  token=base if rule.get('gaps_bound') else wall
  overlap=any(r['what']=='overlap' for r in rows)
  for row in rows:
    row['pct_peak']=(100*row['limit_ms']/row['now_ms']
                     if row['limit_ms'] is not None and row['now_ms']>0 else None)
  total=rank(rows,lambda r:None,'tok_s_if_fixed')
  limit=sum(r['limit_ms'] or 0 for r in rows)
  words=f"Context {context} · captured {wall:.3f} ms/token. "
  if untraced_ms is not None and abs(base-wall)>1e-9:
    words+=f"Untraced {base:.3f} ms. "
  if not (comparable or rows_stand(rule)):
    words+="Capture timing is unqualified"+(f": {rule['refused']}. " if rule.get('refused') else ". ")
  if (attr.get('end_to_end') or {}).get('status')=='checked':  # a checked verdict is always disclosed, over the ceiling too
    words+=verdict_words(attr['end_to_end'])+'. '
  if overlap:
    words+="Overlap subtracts concurrent time. "
  words+="No model = measured time, unknown savings. Roofline is an upper bound. "
  words+=' '.join(capture_limit_notes(attr))
  result=table(WHERE_TITLE,WHERE_COLUMNS,rows,now=token,limit=limit,lost=total,no_limit="no model",
               if_fixed=('tok_s_if_fixed','≤{:.1f}' if overlap else '{:.1f}'),peak=('pct_peak','{:.1f}%'),sum_name='= token',
               estimate=False,token_ms=token,base_ms=base,limit_ms=limit,words=words,
               workload='decode',context=context,**({'qualification':q,**rule} if (q:=qualification_words(rule)) else {}))
  return decode_progress(result, allocate=comparable or rows_stand(rule))


GAIN_COLUMN = "+ Tok/S if roofline"  # a decode row's share of the headroom to the limit (report/plain.py reads this cell)


def decode_progress(result:dict[str, Any], *, allocate:bool = True) -> dict[str, Any]:
  """Engine-neutral additive allocation and cumulative decode throughput, in displayed order."""
  rows=result['rows']
  limit=result['limit_ms'];base=result['base_ms'];total=result['lost_ms']
  result['columns']=[('ms above limit' if column=='ms lost' else GAIN_COLUMN if column=='tok/s if fixed' else column) for column in result['columns']]
  result['roofline_tok_s']=1000/limit if limit>0 else None
  result['sum_cells'][0]='= total at roofline'
  result['sum_cells'][6]=f"{1000/limit:.1f}" if limit>0 else ''
  if limit>0 and base>0:
    from decimal import Decimal, ROUND_HALF_UP
    measured=1000/base
    gain=1000/limit-measured
    result['measured_tok_s']=measured
    result['baseline_cells']=['measured throughput', '', '', '', '', '', f'{measured:.1f}', '']
    if not allocate:
      for row in rows:
        row.update(tok_s_gain=None, cumulative_tok_s=None)
      return result
    # Round cumulative allocation, then difference adjacent totals so displayed tenths add exactly.
    rounded=lambda value:Decimal(str(value)).quantize(Decimal('.1'), rounding=ROUND_HALF_UP)
    previous=rounded(measured)
    cumulative=measured
    for index,row in enumerate(rows):
      allocation=gain*row['lost_ms']/total if abs(total)>1e-12 else 0.0
      row['tok_s_gain']=allocation
      cumulative+=allocation
      current=rounded(1000/limit if index==len(rows)-1 else cumulative)
      row['cells'][5]=f'{current-previous:+.1f}'
      row['cumulative_tok_s']=float(current)
      row['cells'].insert(6,f'{current:.1f}')
      previous=current
    result['columns'].insert(7,'cumulative tok/s')
    result['baseline_cells'].insert(7,f'{measured:.1f}')
    result['sum_cells'].insert(7,result['sum_cells'][6])
    result['words']+=' +tok/s allocates headroom by time difference; cumulative reconciles the roofline. These are accounting shares, not predicted speedups.'
  return result


def where_prefill_goes(attr:dict[str, Any] | None, *, untraced_ms:float | None, length:int,
                       depth:int = 0, project_speedup:bool = True) -> dict[str, Any] | None:
  """A captured prefill (prefill_attribution.attribute_prefill) as the where table: every role row, attention, every
  other kernel and the gaps, worst first, summing to the captured prefill's wall time. ms at ceiling is each row's
  max(operations / its path's peak, bytes / bandwidth); MFU is its operations over its time at that peak. prefill ms if
  fixed is the untraced prefill less the row's lost ms. The ranking and the cells are where_token_goes' own (rank,
  table). depth: a session's turn, `length` tokens appended after `depth` in the cache (collectors/session.py)."""
  if not attr or not attr.get("rows"):
    return None
  rows = capture_rows(attr)
  wall = float(attr["wall_ms"])
  base = float(untraced_ms) if untraced_ms else wall
  e2e = attr.get("end_to_end") or {}
  comparable = e2e.get('confirms') is True or (bool(untraced_ms) and abs(base-wall) < 1e-9)
  rule = {}
  if e2e.get("confirms") is False and untraced_ms:  # over the ceiling: the one rule decode tables go through (over_ceiling)
    gaps = rows[-1]
    rule = over_ceiling(busy=wall - gaps["now_ms"], gaps=gaps["now_ms"], untraced=base, replicas=attr.get("replicas"), of=attr.get("replicas_of"))
    if bound := rule.get("gaps_bound"):  # the gaps are the bound and the rows sum to the untraced prefill
      gaps.update(name=GAPS_BOUND_ROW, now_ms=bound["ms"])
  token = base if rule.get("gaps_bound") else wall
  can_project = project_speedup and (comparable or rows_stand(rule))
  total = rank(rows, lambda r: base - r["lost_ms"] if can_project and r["what"] != "limit" else None, "ms_if_fixed")
  limit = sum(r["limit_ms"] or 0.0 for r in rows)
  compute = sum(r.get("compute_ms") or 0.0 for r in rows)
  what = f"{length}-token turn after {depth} tokens in the cache" if depth else f"{length}-token prefill"
  projection=("Projected time fixes one row only. " if can_project else
              "Projected time unavailable: capture timing is unqualified" + (f" ({rule['refused']}). " if rule.get("refused") else ". ")
              if project_speedup else
              "Server gains unavailable: attribution is unvalidated. ")
  words=[f"{what} · captured {wall:.1f} ms. "+projection+
         "Limits use compute and memory; chip best uses the fastest exact path. "
         "No model = measured time, unknown savings."]
  words.extend(capture_limit_notes(attr))
  unpaired = [r["name"] for r in rows if str(r.get("refused") or "").startswith("no ceiling: a matrix kernel")]
  if unpaired:
    words.append(f"{', '.join(unpaired)}: matrix kernels no role could be paired with ({attr.get('ambiguous', {}).get('reason') if attr.get('ambiguous') else 'their launches fit no role'}); "
                 "their time is real, their ceiling is not known.")
  if abs(base - wall) > 1e-9:
    words.append(f"Untraced prefill: {base:.1f} ms.")
  if e2e.get("confirms") is None and e2e:
    words.append(e2e["words"][:1].upper() + e2e["words"][1:] + ".")
  if e2e.get("confirms") is True:
    words.append(verdict_words(e2e) + ".")
  if e2e.get("confirms") is False:
    words.append(f"Not qualified end to end: {e2e['words']}.")
  mixed = [r["name"] for r in rows if " + " in str(r.get("path") or "") and r.get("limit_ms") is not None]
  if mixed:
    words.append(f"{', '.join(mixed)}: instructions on more than one path; its operations are split over them by its instruction sites, each part at its own path's peak.")
  title = f"Where the turn goes at depth {depth:,}" if depth else PREFILL_TITLE
  columns = PREFILL_COLUMNS if project_speedup else (*PREFILL_COLUMNS[:6], "server ms if fixed", PREFILL_COLUMNS[7])
  return table(title, columns, rows, now=token, limit=limit, lost=total, no_limit=NO_CEILING,
               if_fixed=("ms_if_fixed", "{:.1f}"), peak=("mfu_pct", "{:.1f}%"), sum_name="= prefill", estimate=False,
               token_ms=token, base_ms=base, limit_ms=limit, words=" ".join(words), workload="prefill", length=length,
               mfu_pct=100.0 * compute / base if base else None, **({"depth": depth} if depth else {}),
               **({"end_to_end": e2e} if e2e else {}), **({"qualification": q, **rule} if (q := qualification_words(rule)) else {}))
