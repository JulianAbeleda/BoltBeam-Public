"""A run's results: the measured token against the limit, the per-role table and the tie-out (loss_block), and the
one `results` document every reader shows: the TUI, report.html, summary.txt and Emit. The stage order, the
next-step ladder and the hot-kernel pick are the report's (report/html.py); the tie-out and the where table are
workflow/tie_out.py's. Nothing here computes a new kind of fact."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.collectors import capture_settings, engine_builds, engine_kernels, gpu_telemetry, providers, tinygrad_role_time
from boltbeam.collectors.pairing import ambiguity_kind
from boltbeam.collectors.vllm_speculation import measured_words
from boltbeam.report.html import next_step, roofline_kernels
from boltbeam.role_key import role_key
from boltbeam.profile.roles import shape_parts
from boltbeam.search import search_stage
from boltbeam.target import gpu as tgpu, vendors
from boltbeam.target.targets import get_target
from boltbeam.roofline import limit
from boltbeam.roofline.limit import resolve_peak_flops
from boltbeam.workflow import captured_decode, evidence as ev, layout as lay, offload, tie_out as tie
from boltbeam.workflow import prefill as pf, replay as rw, session as ss
from boltbeam.artifacts.run_io import DECODE_CAPTURE, MACHINE, MEASURE_STATUS, load_manifest, optional
from boltbeam.collectors.capture_scope import for_decode
from boltbeam.workflow.repeat_bands import summary as repeat_summary
from boltbeam.workflow.report_validation import require
from boltbeam.core.refused import Refused
from boltbeam.vocab import SCHEMA_TUI as SCHEMA
from boltbeam.collectors.providers import run_provider
from boltbeam.workflow.run_folder import blocked, is_run, measurement_of, not_measured, summary

def _provider_table(run:pathlib.Path, provider:str, ceil:dict[str, Any], limit_ms:float, *, repeat_evidence:bool = False) -> tuple[dict | None, dict]:
  """One provider's per-role table (or None) and the capture its trace names."""
  trace = optional(run, providers.TRACES[provider])
  if not trace:
    return None, {}
  capture = trace.get("capture") or {"method": tinygrad_role_time.OWN_TIMING, "reason": None}
  return tinygrad_role_time.loss(ceil.get("_roles") or [], trace, limit_ms, None if repeat_evidence else for_decode(run, trace)), capture


def _no_table() -> dict[str, Any]:
  """A loss block's per-role fields before any table fills them: every key present, so a screen reads them all."""
  return {"not_attributed_ms": None, "source": None, "missing": None, "refused": None, "roles_provider": None,
          "provider_missing": None, "others": [], "unpaired_roles": [], "not_timed": [], "role_rule": None,
          "latency": None, "cross_check": None, "role_source": None, "role_source_words": None, "estimate": None}


def loss_block(run:pathlib.Path, manifest:dict[str, Any], ceil:dict[str, Any], token:dict[str, Any] | None, *,
               beside:bool = False, repeat_evidence:bool = False) -> dict[str, Any]:
  """The end result for the provider step 4 measured with: its speed against the limit, ms per token lost, and per
  role where it loses it. Another provider's per-role table, when the run holds one, is shown beside it, labelled.
  The floor rule (tinygrad_role_time.refusal) holds for every provider's table."""
  provider = run_provider(run)
  target = get_target(manifest.get("target_id"))
  if manifest.get('workload') == 'decode' and (run/DECODE_CAPTURE).is_file():
    return captured_decode.loss_block(run,provider,token)
  # facts of the run only: what this machine could capture is `providers`, never part of the pinned results
  plan = None
  if manifest.get("workload") in ("prefill", "session", "replay"):  # the untraced prompt and where its time goes
    empty = {"status": "absent", "reason": None, "limit_ms": None, "limit_tok_s": None, **_no_table(),
             "provider": provider, "capture": plan, "findings": []}
    return {"prefill": pf, "session": ss, "replay": rw}[manifest["workload"]].loss_block(run, manifest, empty)
  if ceil.get("status") != "modeled" or not ceil.get("floor_ms"):
    # no limit, but a measured speed stands: the headline says both (a Mamba2 hybrid with no capture, mamba2.FORMULA_REFUSED)
    trace = optional(run, providers.TRACES[provider]) if provider in providers.TRACES else {}  # its pairing, shown as on any run
    key, missing = (("no_limit", "each role was timed, but a per-role table is held against the limit, which this run does not have")
                    if trace.get("rows") else missing_reason(provider, target, manifest, optional(run, MEASURE_STATUS)))
    return {"status": "absent", "reason": ceil.get("reason"), **({"reason_key": ceil["reason_key"]} if ceil.get("reason_key") else {}),
            "missing": missing, **_missing_key(key), "roles": [], "provider": provider,
            "ambiguous": _ambiguous(trace.get("ambiguous")), "unpaired_roles": trace.get("unpaired_roles") or [],
            "capture": plan, "others": [], "runtimes": [{"provider": provider, "tok_s": token["tok_s"], "ms": token["ms"], "lost_ms": None,
            "per_role": False, "note": f"whole step, measured untraced, at context {token['context']}; no limit"}] if token else []}
  limit_ms = ceil["floor_ms"]
  runtimes = []
  if token:
    runtimes.append({"provider": provider, "tok_s": token["tok_s"], "ms": token["ms"], "lost_ms": token["ms"] - limit_ms,
                     "per_role": False, "note": f"whole step, measured untraced, at context {token['context']}"
                     + (f"; {state}" if (state := gpu_telemetry.run_state_words(run)) else "")})
  out = {"status": "modeled", "limit_tok_s": ceil["tok_s"], "limit_ms": limit_ms, "runtimes": runtimes, "roles": [],
         **({"batch": b, "per": f"per step of {b} streams"} if (b := (token or {}).get("batch") or 1) > 1 else {}),
         **_no_table(), "provider": provider, "capture": plan, "tie_out": None, "step": step_facts(token),
         "where_token_goes": None}
  profile, bw = optional(run, "model_profile.json"), ceil.get("peak_bandwidth_gbs")
  measure = optional(run, MEASURE_STATUS)
  floor_us = optional(run, "probe_evidence.json").get("dispatch_floor_us")
  out["latency"] = {"us": floor_us or tie.ROLE_RULE["latency_us"],
                    "source": tie.LATENCY_MEASURED if floor_us else tie.LATENCY_ASSUMED,
                    "evidence": [{"file": "probe_evidence.json", "path": "dispatch_floor_us"}] if floor_us else []}
  out["layout"], out["machine"] = _layout_limit(run, profile, ceil)
  out["search"] = search_block(run, provider)  # searched, skipped with its reason, or not run: said with or without roles
  state = out["search"].pop("state")
  if out["layout"] and out["layout"]["gpus"] > 1:
    if out["layout"].get("ms"):
      limit_ms = out["layout"]["ms"]  # the layout's limit is the one the measurement is compared with
      out.update(limit_ms=limit_ms, limit_tok_s=1000.0 / limit_ms)
    elif w := out["layout"].get("weights_ms"):
      # never the one-GPU limit for a split: a 2 x RTX 3090 tensor split read 145% of it (2026-11-09). The split's
      # weight reads alone still bound its time from below (its copies only add), so they are its limit, said so
      limit_ms = w
      out.update(limit_ms=w, limit_tok_s=1000.0 / w, limit_basis=(
        f"{out['layout']['label']}: weights only, {out['layout'].get('reason') or 'its copies not derived'}; "
        "the copies between GPUs would only add time, so this limit is looser than the layout's"))
      for rt in runtimes:
        rt["lost_ms"] = rt["ms"] - w
  tables = {name: _provider_table(run, name, ceil, limit_ms, repeat_evidence=repeat_evidence) for name in providers.NAMES}
  for name in providers.NAMES:
    table, capture = tables[name]
    if table is None or table["status"] != "measured":
      continue
    runtimes.append({"provider": name, "tok_s": table["tok_s"], "ms": table["kernel_ms"],
                     "lost_ms": table["kernel_ms"] - limit_ms, "per_role": True, "capture": capture.get("method"),
                     "note": f"GPU kernel time {out.get('per') or 'per token'}, " + _capture_words(capture.get("method"))})
    if name != provider:
      out["others"].append({"provider": name, "capture": capture, "tok_s": table["tok_s"], "ms": table["kernel_ms"],
                            "roles": table["roles"], "not_attributed_ms": table["not_attributed_ms"]})
  if beside:
    out["others"] += sibling_runs(run, manifest, provider)
  # the chosen provider's table; before any, an earlier run's other table keeps `roles` as it was, labelled
  order = [provider] + [n for n in providers.NAMES if n != provider]
  shown = next((name for name in order if tables[name][0] is not None), None)
  if tables[provider][0] is None:  # said beside the other provider's table, or alone when there is none
    key, out["provider_missing"] = missing_reason(provider, target, manifest, measure)
  if shown is None:
    out["missing"] = out["provider_missing"]
    out.update(_missing_key(key))
    out["tie_out"] = tie.tie_out(run, provider=provider, table=None, trace=None, limit_ms=limit_ms, profile=profile,
                                 bandwidth_gbs=bw, missing=out["missing"])
    out["where_token_goes"] = tie.where_token_goes(out["tie_out"], [])
    facts = ev.Facts(run, None)
    out["evidence"] = {"whole_step": facts.step4(), "other_kernels": [], "common": facts.common()}
    out["findings"] = ev.items(facts, out)
    return out
  table, capture = tables[shown]
  out["roles_provider"] = shown
  if table["status"] != "measured":  # a floor broken: the reason is shown, the numbers never are
    out["refused"] = table["reason"]
    return out
  facts = ev.Facts(run, providers.TRACES[shown])
  regimes = {role_key(r): r.get("regime") for r in ceil.get("_roles") or []}
  roles, rule = tie.role_why(table["roles"], bw, regimes=regimes, throttled=facts.throttle() is not None, latency_us=floor_us,
                             floor_us=table.get("floor_us"))
  out["cross_check"] = cross_check(run, bw)
  parts = shape_parts(profile.get("roles") or [])  # a role and format at two shapes: each row names its part
  roles = [{**r, **({"part": parts[role_key(r)]} if role_key(r) in parts else {}), "evidence": facts.role(r["role"], r["quant"], r.get("shape"))}
           for r in roles]
  if shown != provider:
    out["search"] = search_block(run, shown)
    state = out["search"].pop("state")
  by_key = {role_key(r): r for r in optional(run, "route_policy.json").get("routes", []) or []}
  if shown != tinygrad_role_time.PROVIDER:  # the search compares tinygrad kernels; this table is another engine's
    state, by_key = {"status": "skipped", "reason": search_stage.search_applies(None, shown)}, {}
  roles = [{**r, **search_stage.role_verdict(by_key.get(role_key(r)), state)} for r in roles]
  out.update(roles=roles, role_rule=rule, not_attributed_ms=table["not_attributed_ms"], capture=capture,
             check_status=table.get("check_status", "not checked"), band=table.get("band"),
             unpaired_roles=table.get("unpaired_roles") or [], not_timed=table.get("not_timed") or [],
             ceiling_refused=table.get("ceiling_refused") or [], attribution=table.get("attribution"), unpaired_launches=table.get("unpaired_launches") or [],
             joint_role_groups=table.get("joint_role_groups") or [],
             ambiguous=_ambiguous((optional(run, providers.TRACES[shown]) or {}).get("ambiguous")),
             source=_source(shown, capture, target))
  out["tie_out"] = tie.tie_out(run, provider=shown, table=table, trace=optional(run, providers.TRACES[shown]),
                               limit_ms=limit_ms, profile=profile, bandwidth_gbs=bw,
                               missing=None if roles else "the capture could not split the kernels by role")
  trace = optional(run, providers.TRACES[shown])
  out["role_source"], out["role_source_words"] = role_source(trace, capture)
  est = out["tie_out"].get("estimate")
  if out["role_source"] == tie.ROLE_SOURCE_ISOLATED and est:  # scaled from the sum the estimate was made of: less the floor when the rows carry one
    base = est["isolated_sum_less_floor_ms"] if est.get("floor_us") is not None else est["isolated_sum_ms"]
    out["roles"], _ = tie.scale_roles(out["roles"], est["weight_ms"], base)
    out["estimate"] = {**est, "method": method_line(trace, est, measurement_of(run))}
    for r in runtimes:  # the isolated sum is not a token: it carries no "lost", it is said as a sum
      if r.get("capture") == engine_kernels.METHOD:
        r.update(isolated=True, lost_ms=None, note="the kernels' times summed, each timed alone; attention, norms and gaps are not in it")
  out["where_token_goes"] = tie.where_token_goes(out["tie_out"], out["roles"])
  out["findings"] = ev.items(facts, out)
  taken = {role_key(r) for r in roles}
  out["evidence"] = {"whole_step": facts.whole(), "other_kernels": facts.others(taken), "common": facts.common()}
  if out["search"].get("next"):
    out["search"]["next"]["evidence"] = [p for r in roles if f"{r['role']} {r['quant']}" in out["search"]["next"]["roles"]
                                         for p in r["evidence"] if p["file"] == ev.COMPARE]
  return out


def search_block(run:pathlib.Path, provider:str) -> dict[str, Any]:
  """Run's kernel search for this run: searched, skipped (with the reason) or not run, and the next-step item for
  the roles where a faster kernel was found but not applied. Measured numbers only."""
  state = search_stage.read_status(run)
  routes = optional(run, "route_policy.json").get("routes", []) or []
  if state is None and any(isinstance(r.get("compare"), dict) for r in routes):
    state = {"status": "searched", "reason": None, "seconds": None}  # `screen compare` ran before search was a stage
  if state is None:
    state = {"status": "not_run", "reason": "the kernel search did not run for this run", "seconds": None}
  out = {"status": state["status"], "reason": state.get("reason"), "seconds": state.get("seconds"), "next": None,
         "state": state}
  if state["status"] == "searched" and provider == tinygrad_role_time.PROVIDER and (f := search_stage.found_summary(routes)):
    n = len(f["roles"])
    what = f"{n} {'role has' if n == 1 else 'roles have'} a faster kernel found: ~{f['ms']:.1f} ms per token if applied"
    if f["calls"] is not None:
      do = (f"Applying needs the fork's plan to bind to the decode kernels (reached {f['calls_reached']} of "
            f"{f['calls']:.0f} calls).")
    else:
      do = "Applying needs the fork's plan to bind to the decode kernels; no whole-model A/B ran for these roles."
    out["next"] = {"what": what, "ms": f["ms"], "do": do, "roles": f["roles"], "calls_reached": f["calls_reached"],
                   "calls": f["calls"], "rule": "a faster kernel found, not applied"}
  return out


def _layout_limit(run:pathlib.Path, profile:dict[str, Any], ceil:dict[str, Any]) -> tuple[dict | None, dict | None]:
  """The layout the run was measured with and its derived limit, and the machine facts behind it; None, None
  for a run without machine facts (single-GPU runs keep the registry limit either way)."""
  facts = optional(run, MACHINE)
  if not facts:
    return None, None
  status = optional(run, MEASURE_STATUS)
  layout = status.get("layout") or "one"
  got = lay.limit(layout, bytes_per_token=ceil.get("bytes_moved") or 0, facts=facts,
                  hidden_size=profile.get("hidden_size"), layers=profile.get("layer_count"),
                  gpus_used=status.get("gpus"), provider=status.get("provider"))
  shown = {"gpus": [{k: g.get(k) for k in ("index", "name", "target_id", "read_gbs", "read_source")} for g in facts["gpus"]],
           "pairs": facts.get("pairs", []), "measured_at": facts.get("measured_at"),
           "support": tgpu.LIMITED if len(facts["gpus"]) > 1 else None}
  return got, shown


def _source(provider:str, capture:dict[str, Any], target) -> str:
  if capture.get("method") == tinygrad_role_time.OWN_TIMING:
    return "measured in " + tinygrad_role_time.runtime_name(target)
  return f"measured in {provider}, {_capture_words(capture.get('method'))}"


def sibling_runs(run:pathlib.Path, manifest:dict[str, Any], provider:str) -> list[dict[str, Any]]:
  """The newest other-provider run of the same model on the same chip in the same runs folder, as one row to
  show beside this run, labelled with its provider and its run id. Its own floor rule has already run."""
  out = []
  seen = {provider}
  for other in sorted((p for p in run.parent.iterdir() if p != run and is_run(p)), reverse=True):
    m = load_manifest(other)
    name = run_provider(other)
    if name in seen or m.get("model_id") != manifest.get("model_id") or m.get("target_id") != manifest.get("target_id"):
      continue
    seen.add(name)
    res = results_core(other)
    loss, tok_s = res["loss"], res["timing"]["tok_s"]
    out.append({"provider": name, "run": other.name, "capture": loss.get("capture") or {"method": None, "reason": None},
                "tok_s": tok_s, "ms": 1000.0 / tok_s if tok_s else None,
                # an absent speed is said with its reason, never as 0.0
                "missing": None if tok_s else not_measured(other),
                "roles": loss.get("roles") if loss.get("roles_provider") == name else [],
                "not_attributed_ms": loss.get("not_attributed_ms") if loss.get("roles_provider") == name else None})
  return out

# how per-role time was taken, in the screen's words: tinygrad's own events, each vendor's capture
# (target/vendors.py), and the engine's kernels timed alone
CAPTURE_WORDS = {tinygrad_role_time.OWN_TIMING: "tinygrad's own timing",
                 **{v.capture.method: v.capture.words for v in vendors.VENDORS.values()},
                 engine_kernels.METHOD: engine_kernels.WORDS}


IN_MODEL_TOOL = {tinygrad_role_time.OWN_TIMING: "tinygrad events",
                 **{v.capture.method: v.capture.tool_word for v in vendors.VENDORS.values()}}


def _kernel_names(trace:dict[str, Any]) -> list[str]:
  names = []
  for r in trace.get("rows") or []:
    if r.get("scope") == "kernel" and r.get("kernel") and r["kernel"] not in names:
      names.append(r["kernel"])
  return names


def role_source(trace:dict[str, Any], capture:dict[str, Any]) -> tuple[str | None, str | None]:
  """Which kind of per-role measurement this is, and its words for the Facts: "isolated (BoltBeam kernel timer,
  llama.cpp kernels from ggml 0.19.0 at <path>)" or "in-model (xctrace|nsys|tinygrad events)"."""
  method = (capture or {}).get("method")
  if method == engine_kernels.METHOD:
    src = (trace.get("engine") or {}).get("source") or {}
    where = " from " + " at ".join(x for x in (src.get("version"), src.get("path")) if x) if src.get("version") or src.get("path") else ""
    return "isolated", f"isolated (BoltBeam kernel timer, {trace.get('provider_id') or 'engine'} kernels{where})"
  if method is None and not trace:
    return None, None
  return "in_model", f"in-model ({IN_MODEL_TOOL.get(method or tinygrad_role_time.OWN_TIMING, method)})"

def measurement_words(m:dict[str, Any] | None) -> str | None:
  """"Measurement: generic (kernel timer), chosen in Setup." or, after a fallback, "(in-model not available here: ...)"."""
  if not m or not m.get("label"):
    return None
  if m.get("fallback"):
    return f"Measurement: {m['label']} (in-model not available here: {m['fallback']})."
  return f"Measurement: {m['label']}" + (", chosen in Setup." if m.get("choice") not in (None, "auto") else ".")


def estimate_how(est:dict[str, Any]) -> str:
  """How the estimate's numbers were made from the isolated times, in one clause: the scale and its inputs, or "not
  scaled" with the sum that fits; the sum is the one less the dispatch floor when the rows carry one, said so.
  summary_text, report/html.py and boltbeam-tui (render.go estimateHow) print the same words."""
  base = est.get("isolated_sum_less_floor_ms") if est.get("floor_us") is not None else est["isolated_sum_ms"]
  floor = f" {est['floor_words']}" if est.get("floor_words") else ""
  if est["scaled"]:
    return f"scale {est['scale']:.3f} = ({est['token_ms']:.1f} - {est['other_ms']:.1f}) / {base:.1f}{floor}"
  return f"not scaled: {base:.1f} ms alone{floor} fits the {est['token_ms']:.1f} ms token"


def method_line(trace:dict[str, Any], est:dict[str, Any], measurement:dict[str, Any] | None = None) -> str:
  """How an isolated per-role table was measured, in one short paragraph, with the run's two sums."""
  names = _kernel_names(trace)
  src = ((trace.get("engine") or {}).get("source") or {})
  where = f", from the installed {src['version'].split()[0]} source" if src.get("version") else ""
  kernels = ", ".join(names[:1]) + (" and others" if len(names) > 1 else "")
  return (f"Each kernel timed alone by BoltBeam's kernel timer, cold (the cache swept by a read before every call), outside the engine: "
          f"{trace.get('provider_id') or 'the engine'}'s own {kernels}{where}. Alone and cold is slower than inside the token, "
          f"where the cache is warm and launches overlap. Sum alone: {est['isolated_sum_ms']:.1f} ms"
          + (f", {est['floor_words']}: {est['isolated_sum_less_floor_ms']:.1f} ms (the estimate uses this)"
             if est.get("floor_words") else "")
          + f"; the real token: {est['token_ms']:.1f} ms." + (f" {w}" if (w := measurement_words(measurement)) else ""))


def _capture_words(method:str | None) -> str:
  return CAPTURE_WORDS.get(method or "", method or "no capture")


MISSING_GENERIC = "no per-role time yet: Run with {provider} times each role"


def _missing(provider:str, target, manifest:dict[str, Any], measure:dict[str, Any] | None = None) -> str:
  return missing_reason(provider, target, manifest, measure)[1]


MISSING_GENERIC_KEY = "not_run"  # the generic reason: its key is left out of the results (missing_key absent)


def missing_reason(provider:str, target, manifest:dict[str, Any], measure:dict[str, Any] | None = None) -> tuple[str, str]:
  """Why this run has no per-role table for its provider, as (key, words): the key names the reason for a plain screen
  (data/plain_words.json not_split), the words are the report's. The run's own record comes first: the capture the
  measuring machine had (measure_status.json capture, written by the pipeline). Never the action just taken: a run
  that recorded "no capture here" is told what is missing, not to Run again."""
  if (measure or {}).get("not_split_by_role"):  # 2 x RTX 3090, vLLM split by rows read "Run with vllm times each role"
    return measure.get("not_split_key") or "not_split", f"not split by role: {measure['not_split_by_role']}"
  if provider == tinygrad_role_time.PROVIDER and not tinygrad_role_time.device_for(target):
    return "no_device", f"{manifest.get('target_id')} names no tinygrad device, so no per-role time can be taken here"
  cap = (measure or {}).get("capture") or {}
  if cap and cap.get("method") is None and cap.get("reason"):
    return "no_capture", f"no per-role time on the measuring machine: {cap['reason']}"
  return MISSING_GENERIC_KEY, MISSING_GENERIC.format(provider=provider)


def _missing_key(key:str) -> dict[str, str]:
  return {} if key == MISSING_GENERIC_KEY else {"missing_key": key}

def step_facts(token:dict[str, Any] | None) -> dict[str, Any] | None:
  """THE measured token's own facts for a screen: context, speed, how it was run (the trace row's source), the
  graph state when the engine says it, the other contexts measured, and the pointer to the row."""
  if not token:
    return None
  graph_failed = "graph replay failed" in str(token.get("source") or "")
  return {"context": token["context"], "tok_s": token["tok_s"], "ms": token["ms"], "source": token.get("source"),
          **({"requested_context": token["requested_context"]} if token.get("requested_context") not in (None, token["context"]) else {}),
          "rule": token["rule"], "graph_failed": graph_failed, "graph_error": token.get("graph_error"),
          "also": token["also"], "evidence": [{"file": "timing_trace.json", "path": token.get("evidence_path") or f"rows[{token['index']}]"}]}


def cross_check(run:pathlib.Path, bandwidth_gbs:float | None) -> dict[str, Any] | None:
  """The engine's kernels timed alone beside an in-model capture (providers.CROSS_CHECK), as rows a screen shows:
  role, quant, µs per call (measured, floor included), the same less the dispatch floor, GB/s and the share of peak;
  None when the run has none. The same rule as the isolated per-role table (tie_out.role_why): GB/s and the share of
  peak are on the time less the floor when the row carries one; columns_words says so once."""
  trace = optional(run, providers.CROSS_CHECK)
  if not trace:
    return None
  cap = trace.get("capture") or {}
  rows, not_measured, floor_us = [], [], None
  for i, r in enumerate(trace.get("rows") or []):
    if r.get("scope") != "kernel" or r.get("status") != "measured":
      if r.get("scope") == "kernel":
        not_measured.append({"role": r.get("role"), "quant": r.get("quant"),
                             "reason": r.get("reason") or r.get("status") or "not measured"})
      continue
    less = r.get("us_per_call_less_floor")
    if less is not None:
      floor_us = (r.get("timing") or {}).get("dispatch_floor_us", floor_us)
      gbs = r["bytes"] / (less * 1e3) if less > 0 and r.get("bytes") else None
    else:
      gbs = r.get("gbs")
    rows.append({"role": r["role"], "quant": r["quant"], "us_per_call": r.get("us_per_call"), "us_per_call_less_floor": less,
                 "gbs": gbs, "pct_peak": (100.0 * gbs / bandwidth_gbs) if gbs and bandwidth_gbs else None,
                 "evidence": [{"file": providers.CROSS_CHECK, "path": f"rows[{i}]"}]})
  return {"method": cap.get("method"), "words": _capture_words(cap.get("method")), "reason": cap.get("reason"), "rows": rows,
          "not_measured": not_measured,
          "columns_words": (tie.ISOLATED_COLUMNS.format(floor_words=tie.floor_words(floor_us)) +
                            ". Empty-launch subtraction is an estimate, not a measured additive correction") if floor_us is not None else None}


def probe_rows(run:pathlib.Path, bandwidth_gbs:float | None) -> dict[str, Any]:
  """BoltBeam's own GEMV per role (the building-block probe, collectors/boltbeam_gemv.py): GB/s and the share of
  peak, as a reference row group. Not the engine's kernel. absent names what Metal cannot report, so nobody asks
  for a probe this GPU cannot give."""
  ev = optional(run, "probe_evidence.json")
  measure = optional(run, MEASURE_STATUS)
  if not ev:
    return {"status": "absent", "reason": measure.get("probe_reason") or "the run holds no probe evidence", "rows": [], "absent": {}}
  rows = []
  absent:dict[str, str] = {}
  for i, r in enumerate(ev.get("probes") or []):
    if r.get("status") != "measured":
      continue
    gbs = (r.get("throughput") or {}).get("achieved_gbs")
    rows.append({"role": r["role"], "quant": r["quant"], "shape": r.get("shape"), "gbs": gbs,
                 "pct_peak": (100.0 * gbs / bandwidth_gbs) if gbs and bandwidth_gbs else None,
                 "us_per_call": (r.get("timing") or {}).get("candidate_us"),
                 "evidence": [{"file": "probe_evidence.json", "path": f"probes[{i}]"}]})
    absent.update(r.get("absent") or {})
  return {"status": "measured" if rows else "absent", "reason": None if rows else "no probe row was measured",
          "label": "BoltBeam's own kernel (reference), not the engine's", "dispatch_floor_us": ev.get("dispatch_floor_us"),
          "rows": rows, "absent": absent}


def _route_compare(c:Any) -> dict[str, Any] | None:
  """One role's kernel comparison as the screen shows it, or None before any comparison ran."""
  if not isinstance(c, dict): return None
  ab = c.get("ab") if isinstance(c.get("ab"), dict) else None
  return {"plan_id": c.get("plan_id"), "plan": c.get("plan"), "search_median_ns": c.get("search_median_ns"),
          "default_median_ns": c.get("default_median_ns"), "measured_correct": c.get("measured_correct"),
          "candidates": c.get("candidates"), "reason": c.get("reason"), "kernel": c.get("kernel"),
          "role_calls_per_token": c.get("role_calls_per_token"),
          "decided_by": c.get("decided_by"), "timing_source": c.get("timing_source"),
          "ab": None if ab is None else {k: ab.get(k) for k in ("baseline_tok_s", "candidate_tok_s", "delta_pct",
                                                                   "token_match", "route_bound")}}

def results(run:pathlib.Path, *, repeat_evidence:bool = False) -> dict[str, Any]:
  res = results_core(run, beside=True, repeat_evidence=repeat_evidence)
  res['validation'] = require(run, res)
  return res


def the_token(run:pathlib.Path) -> dict[str, Any] | None:
  """THE measured token of this run: step 4's whole-step row that capture_settings.measured_step picks for the run's
  provider, near the context its per-role capture attended when it has one. Every headline reads this."""
  provider = run_provider(run)
  role_trace = optional(run, providers.TRACES[provider])
  near = capture_settings.attended_context(role_trace, provider) if role_trace else None
  # a per-role capture of B streams ties out against step 4's step of B streams at its requested context
  whole = next((r for r in role_trace.get("rows", []) if r.get("scope") == "whole_step"), {}) if role_trace else {}
  batch, at = whole.get("batch") or 1, whole.get("context")
  captured=optional(run,'decode_capture.json')
  if captured.get('contexts'):
    near=at=float(next(iter(captured['contexts'])))
    batch=(next(iter(captured['contexts'].values())).get('engine') or {}).get('batch') or 1
  if batch>1:
    step=batch_step(optional(run,"timing_trace.json"),provider,int(at),batch)
    if step is None:
      raise Refused(f"the per-role capture is of {batch} streams at context {int(at)}, and step 4 timed no step of "
                    f"{batch} streams there: time that batch (--batch 1,{batch}) before reading the capture")
    return step
  return capture_settings.measured_step(optional(run, "timing_trace.json"), provider, near=near)


def batch_step(trace:dict[str, Any], provider:str, context:int, batch:int) -> dict[str, Any] | None:
  """The measured step a batched capture ties out against: step 4's point of `batch` streams at the capture's
  context (providers.engine_trace keeps every batch under "batches"), its time per step; None when there is none."""
  if (trace.get("provider") or "llama.cpp") != provider:
    return None
  points = [(i, p) for i, p in enumerate(trace.get("batches") or []) if p.get("context") == context and p.get("batch") == batch]
  if len(points) != 1:
    return None
  i, p = points[0]
  return {"context": context, "requested_context": context, "tok_s": p["tok_s_stream"], "ms": p["step_ms"],
          "wall_us": p["step_ms"] * 1000.0, "source": p.get("source"), "batch": batch,
          "timing_window": (p.get("engine") or {}).get("timing_window"), "graph_error": None, "index": None,
          "evidence_path": f"batches[{i}]", "decode_tokens": p.get("tokens"),
          "rule": f"the step of {batch} streams at the context the per-role capture attended", "also": []}


def results_core(run:pathlib.Path, beside:bool = False, *, repeat_evidence:bool = False) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = optional(run, "model_profile.json")
  policy = optional(run, "route_policy.json")
  plan = optional(run, "measurement_plan.json")
  primitive = optional(run, "primitive_profile.json")
  timing = optional(run, "timing_profile.json")
  ceil = limit.measured_vs_ceiling(manifest, profile, run)
  if manifest.get("workload") == "decode" and (refused := limit.decode_refused(profile, run)):
    ceil = {"status": "absent", "reason": refused, "reason_key": "formula_refused"}  # speed only, said: never the formula's limit for a Mamba2 hybrid
  token = the_token(run)
  context = token["context"] if token else None
  kernels, context = roofline_kernels(timing, context) if timing else ([], context)
  chosen = next((s for s in timing.get("context_summaries", []) if s.get("context") == context), {}) if timing else {}
  loss = loss_block(run, manifest, ceil, token, beside=beside, repeat_evidence=repeat_evidence)
  if manifest.get("workload") == "decode" and token and (why := limit.pct_withheld(run)):  # the limit stands for the parts;
    from boltbeam.report.plain import words
    loss.update(pct_withheld=why, pct_withheld_says=words()["no_limit"][why])  # the speed is not a % of it (a speculative run)
  if clock_words := gpu_telemetry.measurement_clock_words(run):
    loss["timing_scope"] = clock_words
  if manifest.get("workload") in ("prefill", "session"):  # one ceiling: the sum of the kernels' own, as the where table has it
    ceil = prefill_ceiling(ceil, loss, token)
  elif manifest.get("workload") == "replay":  # one ceiling: the sum of the turns' rooflines (workflow/replay.py)
    ceil = replay_ceiling(ceil, loss)
  elif manifest.get("workload") == "decode" and token and ceil.get("status") == "modeled":
    ceil = decode_ceiling(ceil, loss, token, profile, run_provider(run), run)
  moe = offload.run_block(run, profile, token, ceil.get("peak_bandwidth_gbs"), withheld=loss.get("pct_withheld")) if manifest.get("workload") == "decode" else None  # experts in RAM: its own limit
  engine_build = manifest.get("engine")
  if not (engine_build or {}).get("build"):
    recorded = providers.run_build(run)
    if recorded:
      engine_build = {"provider": run_provider(run), **providers.pin_check(
        run_provider(run), recorded, providers.run_build_number(run), fork=providers.run_fork(run))}
  return {
    "schema": SCHEMA, "kind": "results", "id": run.name, "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "measured": bool(timing) or bool(primitive),
    "ceiling": {k: v for k, v in ceil.items() if not k.startswith("_")},
    "loss": loss,
    **({"offload": moe} if moe else {}),
    **({"placement": lost} if not moe and manifest.get("workload") == "decode" and (lost := offload.placement(run, profile)) else {}),
    "measurement": measurement_of(run),  # the Setup choice of how roles were timed, as the run recorded it
    "probe": probe_rows(run, ceil.get("peak_bandwidth_gbs")),
    "routes": [{"role": r.get("role"), "quant": r.get("quant"), "shape": r.get("shape"),
                "selected_route": r.get("selected_route"), "status": r.get("status"),
                "candidates": list(r.get("candidates", []) or []), "evidence_refs": list(r.get("evidence_refs", []) or []),
                "compare": _route_compare(r.get("compare"))}
               for r in policy.get("routes", []) or []],
    "timing": {
      "status": "classified" if timing else "absent",
      "dominant_bucket": timing.get("dominant_timing_bucket"),
      "next_actions": (["Review the captured operation table; the separate whole-step timing profile does not classify individual kernels.",
                        *list(timing.get("next_actions", []) or [])[1:]]
                       if loss.get("semantic_capture") and timing.get("dominant_timing_bucket")=="timing_inconclusive"
                       else list(timing.get("next_actions", []) or [])),
      # THE measured token (capture_settings.measured_step): the same row the tie-out uses; also lists the other contexts
      "context": context, "total_us": (token.get("wall_us") or 1000.0 * token["ms"]) if token else chosen.get("total_us"),
      "tok_s": token["tok_s"] if token else None,
      "also": [{"context": a["context"], "tok_s": a["tok_s"]} for a in token["also"]] if token else [],
      "roles": [{"role": r.get("role"), "quant": r.get("quant"), "shape": r.get("shape"), "context": r.get("context"),
                 "wall_us": r.get("wall_us"), "pct_step": r.get("pct_step"), "classification": r.get("classification")}
                for r in timing.get("role_timing", []) or []],
      "kernels": [{"name": k.get("name"), "kind": k.get("kind"), "us": k.get("us"), "pct_step": k.get("pct_step"),
                   "phys_util_pct": k.get("phys_util_pct"), "bucket": k.get("bucket"), "loss_us": k.get("loss_us")}
                  for k in kernels],
    },
    "regimes": [{"role": r.get("role"), "quant": r.get("quant"), "shape": r.get("shape"),
                 "classification": r.get("classification"), "visible_bottleneck": r.get("visible_bottleneck"),
                 "next_action": r.get("next_action")} for r in primitive.get("quant_gemv_regimes", []) or []],
    "blocked": blocked(plan, run),
    "report": "report.html" if (run / "report.html").exists() else None,
    "engine": {"provider": run_provider(run), "weight_format": optional(run, MEASURE_STATUS).get("weight_format")
               or (providers.weight_format(profile) if profile else None),
               # the build the run measured against the pin table, and the toolchain it recorded (providers.record_build,
               # collectors/toolchain.record); None when the run predates them
               "build": engine_build, "toolchain": manifest.get("toolchain"),
               # the one-time comparison with the previous build, when this run made it (collectors/engine_builds.py)
               "comparison": (cmp := engine_builds.comparison(run)),
               "build_lines": engine_builds.build_lines({"build": engine_build, "comparison": cmp}),
               # what the run asked the engine for (trace_request.json); a setting changes what is measured
               "settings_words": setting_words(run),
               # what speculation did (MTP acceptance, collectors/vllm_speculation): measured, beside what was asked
               "speculation_words": _speculation_words(run),
               # how a prefill sent its prompt (collectors/prefill.prompt_form): token ids or a chat request
               **_prompt_form(loss)},
    "batches": batch_rows(run, manifest, profile, ceil),
    # the band of n runs of this exact setup (screen pipeline --repeats), when there is one
    **({"repeats": band} if (band := repeat_summary(run)) else {}),
  }


PREFILL_CEILING_BASIS = ("the sum of the prefill's kernels' ceilings: each max(operations / the measured peak of its "
                         "instructions' path, bytes / bandwidth)")


def decode_ceiling(ceil:dict[str, Any], loss:dict[str, Any], token:dict[str, Any],
                   profile:dict[str, Any], provider:str, run:pathlib.Path | None = None) -> dict[str, Any]:
  """Compare the untraced token with its own context's limit, keeping the capture's separate basis.

  The weight-only floor remains the input to batch limits; otherwise adding each batch's KV read
  would count the headline's context twice.
  """
  if loss.get('semantic_capture'):
    return {**ceil,'context':token['context'],'floor_ms':loss['limit_ms'],
            'tok_s':loss['limit_tok_s'],'bytes_moved':None,
            '_operation_limit':True,**({'_operation_batch':loss['batch']} if loss.get('batch',1)>1 else {}),
            'basis':"sum of captured operation limits, including attention and recurrent state work"}
  layout = loss.get("layout") or {}
  # a split's limit is its layout's (weights and joins), else its weights alone (results: limit_basis), never one GPU's
  layout_ms = layout.get("ms") or layout.get("weights_ms")
  split = layout["gpus"] if layout_ms and (layout.get("gpus") or 1) > 1 else 1
  weight = layout_ms if split > 1 else ceil["floor_ms"]
  element, _ = providers.kv_element(provider, run)
  state, words = tie.state_ms(profile, provider, run, ceil["peak_bandwidth_gbs"])
  # a tensor-parallel split shards the KV and state heads: each GPU reads its share at once; a step of B streams reads B
  limit = tie.step_limit_ms(weight, token.get("batch") or 1, tie.kv_ms(profile, token["context"], element, ceil["peak_bandwidth_gbs"]), state,
                            split if provider in lay.ENGINE_JOINS else 1)
  loss.update(limit_ms=limit, limit_tok_s=1000.0 / limit, limit_context=token["context"], **({"state_words": words} if words else {}))
  for runtime in loss.get("runtimes", []):
    if runtime.get("isolated"):
      continue
    basis = (loss.get("tie_out") or {}).get("limit_ms") if runtime.get("per_role") else limit
    if basis is not None and runtime.get("provider") == provider:
      runtime["lost_ms"] = runtime["ms"] - basis
  return {**ceil, "_weight_floor_ms": weight, "context": token["context"],
          "floor_ms": limit, "tok_s": 1000.0 / limit,
          "bytes_moved": limit * ceil["peak_bandwidth_gbs"] * 1e6}


def prefill_ceiling(ceil:dict[str, Any], loss:dict[str, Any], token:dict[str, Any] | None) -> dict[str, Any]:
  """A prefill run's ceiling is its where table's (workflow/prefill.py); the whole-model roofline at one fp16 peak
  is not shown beside it. Absent, with its reason, until the prefill was captured in the model."""
  w = loss.get("where_token_goes")
  if not w or not w.get("limit_ms"):
    return {"status": "absent", "reason": loss.get("missing") or ("no prefill captured in the model yet" if not w else
                                                                  "no captured row has a limit, so the prefill has none")}
  length = w["length"]
  return {"status": "modeled", "context": length, "floor_ms": w["limit_ms"], "tok_s": length / w["limit_ms"] * 1e3,
          "bytes_moved": None, "peak_bandwidth_gbs": ceil.get("peak_bandwidth_gbs"), "bandwidth_source": ceil.get("bandwidth_source"),
          "uncertainty": ceil.get("uncertainty"), "basis": PREFILL_CEILING_BASIS}


def replay_ceiling(ceil:dict[str, Any], loss:dict[str, Any]) -> dict[str, Any]:
  """A replay's ceiling is the sum of its warm turns' rooflines (workflow/replay.py), not one token's."""
  h = (loss.get("replay") or {}).get("headline") or {}
  if not h.get("roofline_ms"):
    return {"status": "absent", "reason": loss.get("reason") or "no turn ceiling: the run lacks the profile or the ledger"}
  return {"status": "modeled", "context": h["deepest_prefix"], "floor_ms": h["roofline_ms"], "tok_s": None, "bytes_moved": None,
          "peak_bandwidth_gbs": ceil.get("peak_bandwidth_gbs"), "bandwidth_source": ceil.get("bandwidth_source"),
          "uncertainty": ceil.get("uncertainty"), "basis": h.get("basis")}


def batch_rows(run:pathlib.Path, manifest:dict[str, Any], profile:dict[str, Any], ceil:dict[str, Any]) -> list[dict[str, Any]]:
  """Every measured point (context, batch) of step 4 beside its own limit (tie_out.batch_limit): tokens/s per
  stream and in total, and the share of the limit reached. Empty when step 4 kept no points."""
  trace = optional(run, "timing_trace.json")
  points = trace.get("batches") or []
  if not points or ceil.get("status") != "modeled":
    return []
  provider = trace.get("provider") or run_provider(run)
  element, _ = providers.kv_element(provider, run)
  state = tie.state_ms(profile, provider, run, ceil.get("peak_bandwidth_gbs") or 0.0)
  params = sum(float(r.get("rows") or 0) * float(r.get("cols") or 0) * float(r.get("count") or 1)
               for r in profile.get("roles") or [])
  try:
    peak_flops = resolve_peak_flops(get_target(manifest.get("target_id")), "fp16", None)
  except SystemExit:
    peak_flops = None
  out = []
  for p in points:
    if ceil.get('_operation_limit'):
      captured = ceil.get('_operation_batch', 1)
      if p['context']!=ceil['context'] or p['batch']!=captured:
        lim={'status':'absent','reason':'no matching captured operation limit','bound':'not captured'}
      else:
        lim={'batch':captured,'context':p['context'],'step_ms':ceil['floor_ms'],
             'tok_s_stream':ceil['tok_s'],'tok_s_total':captured*ceil['tok_s'],'bound':'per operation'}
    else:
      lim = tie.batch_limit(weight_ms=ceil.get("_weight_floor_ms", ceil["floor_ms"]), profile=profile, context=float(p["context"]), batch=int(p["batch"]),
                          element_bytes=element, bandwidth_gbs=ceil["peak_bandwidth_gbs"], params=params or None,
                          peak_flops=peak_flops, state=state)
    out.append({"context": p["context"], "batch": p["batch"], "step_ms": p["step_ms"], "tok_s_stream": p["tok_s_stream"],
                "tok_s_total": p["tok_s_total"], "source": p.get("source"), "limit": lim,
                "pct_of_limit": 100.0 * lim["step_ms"] / p["step_ms"] if p["step_ms"] and 'step_ms' in lim else None})
  return out


def show(run:pathlib.Path) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = optional(run, "model_profile.json")
  report = optional(run, "analysis_report.json")
  plan = optional(run, "measurement_plan.json")
  meta = profile.get("metadata", {}) or {}
  out, res = summary(run), results(run)
  out.update({
    "schema": SCHEMA, "kind": "run", "model_path": manifest.get("model_path"),
    "model": {"architecture": profile.get("architecture"), "architecture_class": profile.get("architecture_class"),
              "layer_count": profile.get("layer_count"), "hidden_size": profile.get("hidden_size"),
              "ffn_size": profile.get("ffn_size"), "vocab_size": profile.get("vocab_size"),
              "role_count": len(profile.get("roles", []) or []), "quant_types": list(meta.get("quant_types", []) or [])},
    # the report's own ladder, text only: the same sentence report.html and summary.md print, without the code marks
    "next_step": ((res.get("loss") or {}).get("replay") or {}).get("next_step") or next_step(report, plan, res.get("probe"), optional(run, MEASURE_STATUS)).replace("`", ""),
    "artifacts": list(manifest.get("artifacts", []) or []),
    "results": res,
  })
  if env := optional(run, "workload_profile.json").get("engine_env"):
    out["engine_env"] = dict(env)  # only a run that set some, so a record without any is as it was
  return out

def setting_words(run:pathlib.Path) -> list[str]:
  """The settings a run asked its engine for, and the environment it set for it (pipeline --engine-env), as NAME=VALUE."""
  env = [f"{k}={v}" for k, v in (optional(run, "workload_profile.json").get("engine_env") or {}).items()]
  try:
    return capture_settings.setting_words(capture_settings.requested(run)) + env
  except (OSError, ValueError):
    return env


def _prompt_form(loss:dict[str, Any]) -> dict[str, Any]:
  """A prefill's prompt form: the key and its sentence ({} for a run with no prefill). A record from before the form was
  recorded sent token ids."""
  from boltbeam.report.plain import words
  lengths = (loss.get("prefill") or {}).get("lengths") or {}
  if not lengths:
    return {}
  form = (next(iter(lengths.values())).get("engine") or {}).get("prompt_form") or "ids"
  hw = words()["head_to_head"]["prompt_forms"]
  return {"prompt_form": form, "prompt_form_words": words()["machine"]["prompt_form"].format(form=hw.get(form, form))}


def _speculation_words(run:pathlib.Path) -> list[str]:
  try:
    return measured_words(run)
  except (OSError, ValueError):
    return []

def _ambiguous(a:dict[str, Any] | None) -> dict[str, Any] | None:
  """The trace's unresolved pairing with its kind; a trace from before the kind was recorded has it read from its reason."""
  return {**a, "kind": ambiguity_kind(a)} if a else None
