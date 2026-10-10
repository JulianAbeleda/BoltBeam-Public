"""Emit: the run on screen turned into a gameplan, `python -m boltbeam.workflow.screen emit --run RUN`.

Run measures the model's roles against the roofline and, with the kernel search on, finds faster kernel plans. Emit
reads that run and writes `<run>/gameplan.json` and `<run>/gameplan.md`: per role, worst first, what kernel to emit
for this exact shape, built only from what the run recorded, with the check that must pass. A person or an agent
follows it to write the kernel into the tinygrad fork. Emit writes the plan, not the kernel.

Each role block shows the search's lineage in order, every layer read off the run's own files:
  1. BubbleBeam   the legal dimensions proposed for this shape on this chip (the search request's dimensions,
                  coupled rows and compiler facts: kernel_compare/<role>-<quant>-search-request.json)
  2. FutureSight  the static rejections, their top reasons and the measurement order it set (the campaign result's
                  futuresight_static_evidence, or the request's futuresight_evidence)
  3. Measured     the campaign's results for this role: candidates measured, the best median against the model's own
                  kernel, the spread (kernel_compare/<stem>-search-result.json, route_policy.json compare)
  4. Promotion    the verdict (search/role_compare.py role_verdict) and, for a winner, the promotion record to add,
                  the fork emitter with its spec arguments, and the check
A layer the run does not hold says "not recorded in this run" and names the file it would be in. Nothing here is
filled from the code's defaults: a missing field reads "not recorded" with its file. Every number carries its
evidence pointer {file, path} into the run, like the report does (workflow/evidence.py). The markdown is the
on-paper version of the JSON, nothing more.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from collections import Counter
from typing import Any

from boltbeam.core.canonical import pretty_json
from boltbeam.report.html import PLAIN_VERDICT
from boltbeam.search import role_compare
from boltbeam.workflow import evidence as ev
from boltbeam.workflow.common import load_manifest

SCHEMA = "boltbeam.gameplan.v1"
JSON_FILE, MD_FILE = "gameplan.json", "gameplan.md"
CLOSING = "Emit writes the plan, not the kernel."
NOT_RECORDED = "not recorded"
LAYERS = ("bubblebeam", "futuresight", "measured", "promotion")
LAYER_TITLE = {"bubblebeam": "BubbleBeam", "futuresight": "FutureSight", "measured": "Measured", "promotion": "Promotion"}
AT_LIMIT = "at the limit, nothing to gain"
NOT_SEARCHED = "not searched: run with the kernel search on"
TOP_REASONS = 2  # rejection reasons named per layer
FIRST_ORDER = 3  # candidates named from FutureSight's measurement order

# where each layer's facts live in a run; role_compare.compare_run writes the per-role files by stem
POLICY, STATUS = "route_policy.json", f"{role_compare.FOLDER}/{role_compare.STATUS_FILE}"
REQUEST, RESULT = role_compare.FOLDER + "/{stem}-search-request.json", role_compare.FOLDER + "/{stem}-search-result.json"
NORM_ROLES = ("attn_norm", "ffn_norm", "output_norm", "norm")

# The fork emitter that produces a kernel of each kind (tinygrad/llm/decode_kernels.py), by quant and role: the
# emitter's name, its spec arguments for this shape, and the pins its validate (or the decode admission in
# decode_routes.py) puts on the shape. A table, not code: a new kind is a row.
EMITTERS: tuple[tuple[str, tuple[str, ...] | None, str, str, str], ...] = (
  ("Q4_K", ("ffn_gate_up",), "q4k_g3_lanemap_gemv_w1w3_kernel", 'rows={rows}, k={k}, load_style="scalar", store_fp16=False',
   "k % 1024 == 0 (Q4KGateUpLaneMap.validate: 256-element blocks in 4 block groups); the decode binding admits k % 1024 == 0 "
   "and rows % 32 == 0 (_Q4KDecodeCandidate)"),
  ("Q4_K", None, "q4k_g3_lanemap_gemv_kernel", 'rows={rows}, k={k}, lanes=32, epilogue=None, load_style="scalar"',
   "k % 1024 == 0 (Q4KGateUpLaneMap.validate: 256-element blocks in 4 block groups); the decode binding admits k % 1024 == 0 "
   "and rows % 32 == 0 (_Q4KDecodeCandidate)"),
  ("Q6_K", None, "emit_q6k_gemv_kernel", 'Q6KGEMVRouteSpec(rows={rows}, k={k}, role="{role}", route_family="q6k_coop", row_tile=q6k_coop_row_tile_for_target(backend, architecture), target="{target}")',
   "k % 256 == 0, rows % row_tile == 0, lane_extent 16 (Q6KGEMVRouteSpec.validate)"),
  *(((q, None, "emit_block_quant_gemv_kernel",
      'BlockQuantGEMVRouteSpec(rows={rows}, k={k}, quant=QUANT_FORMATS["' + q + '"], wave_size=<the device wave size>, '
      'lanes_per_block=<the winner\'s code bytes / memory.b.vector_width>, warps_per_cta=<the winner\'s tile.n>)',
      "k % block_elems == 0, rows >= 1, lanes_per_block divides the block's code units and the wave size, wave size a power of "
      "two (BlockQuantGEMVRouteSpec.validate)") for q in ("Q8_0", "Q5_0", "Q4_0", "Q5_K"))),
  ("*", NORM_ROLES, "emit_decode_rmsnorm_kernel", "DecodeRMSNormSpec(rows={rows}, dim={k}, eps=<the model's rms epsilon>)",
   "dim % 32 == 0, eps > 0 (DecodeRMSNormSpec.validate)"),
)


class Refused(ValueError):
  """A run Emit cannot plan from: no per-role table yet."""


def _ptr(file:str, path:str | None = None) -> dict[str, Any]:
  return {"file": file, "path": path} if path else {"file": file}


def _nr(file:str, field:str) -> str:
  return f"{NOT_RECORDED} ({file}: {field})"


def _f(v:Any, fmt:str, file:str, field:str) -> str:
  """A number from the run, formatted; a missing one says so and names its file."""
  return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else _nr(file, field)


class _Files:
  """The run's files, read once, and the list of those that were there (the footer names them)."""

  def __init__(self, run:pathlib.Path):
    self.run, self.read, self._cache = run, [], {}

  def get(self, name:str) -> Any:
    if name not in self._cache:
      p = self.run / name
      self._cache[name] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
      if self._cache[name] is not None:
        self.read.append(name)
    return self._cache[name]


def _stem(role:str, quant:str, shape:Any = None) -> str:
  from boltbeam.role_key import role_stem
  return role_stem(role, quant, shape)


def _layer(key:str, recorded:bool, lines:list[str], facts:dict[str, Any] | None = None,
           evidence:list[dict[str, Any]] | None = None) -> dict[str, Any]:
  return {"layer": key, "title": LAYER_TITLE[key], "recorded": recorded, "lines": lines, "facts": facts or {},
          "evidence": evidence or []}


# --- the four layers ----------------------------------------------------------------------------------------------

def _bubblebeam(stem:str, request:dict[str, Any] | None) -> dict[str, Any]:
  file = REQUEST.format(stem=stem)
  if request is None:
    return _layer("bubblebeam", False, [f"{NOT_RECORDED} in this run: no {file}"])
  space = request.get("candidate_space") or {}
  dims = request.get("dimensions") if isinstance(request.get("dimensions"), dict) else space.get("dimensions")
  coupled = request.get("legal_coupled_rows")
  # a proposal is recorded when BubbleBeam wrote dimensions or split coupled rows (a proposal of rows alone has {})
  if not isinstance(dims, dict) or (not dims and not isinstance(coupled, list)):
    rows = space.get("rows")
    how = (f"the search used a finite space of {len(rows)} rows ({file} candidate_space.rows), not a BubbleBeam proposal"
           if isinstance(rows, list) else f"no dimensions in {file}")
    return _layer("bubblebeam", False, [f"{NOT_RECORDED} in this run: {how}; a proposal would be {file} dimensions and legal_coupled_rows"],
                  {"finite_rows": len(rows) if isinstance(rows, list) else None},
                  [_ptr(file, "candidate_space.rows")] if isinstance(rows, list) else [])
  where = "dimensions" if isinstance(request.get("dimensions"), dict) else "candidate_space.dimensions"
  size = 1
  for vals in dims.values():
    size *= max(len(vals), 1) if isinstance(vals, list) else 1
  rows_n = len(coupled) if isinstance(coupled, list) else 0
  size = (size if dims else 0) + rows_n + 2  # the independent axes, the legal coupled rows, the seed and the control
  axes = ", ".join(f"{path} {len(vals) if isinstance(vals, list) else 1} value{'s' if isinstance(vals, list) and len(vals) != 1 else ''}"
                   for path, vals in sorted(dims.items())) or "no independent axes (every value rides in a coupled row)"
  lines = [f"legal dimensions: {axes}; population up to {size} candidates"]
  evidence = [_ptr(file, where)]
  legal, rejected = request.get("legal_coupled_rows"), request.get("rejected_coupled_rows")
  if isinstance(legal, list) or isinstance(rejected, list):
    n, m = len(legal or []), len(rejected or [])
    reasons = Counter(str((r or {}).get("reason")) for r in (rejected or []) if isinstance(r, dict))
    top = "; ".join(f"{k} x {why}" for why, k in reasons.most_common(TOP_REASONS))
    lines.append(f"coupled rows: {n} legal, {m} rejected" + (f" ({top})" if top else ""))
    evidence += [_ptr(file, "legal_coupled_rows"), _ptr(file, "rejected_coupled_rows")]
  else:
    lines.append(f"coupled rows: {_nr(file, 'legal_coupled_rows')}")
  facts = request.get("compiler_facts")
  if isinstance(facts, dict) and facts:
    lines.append("chip facts used: " + ", ".join(f"{k} {json.dumps(v, sort_keys=True)}" if not isinstance(v, (list, dict)) else k for k, v in sorted(facts.items())))
    evidence.append(_ptr(file, "compiler_facts"))
  else:
    lines.append(f"chip facts used: {_nr(file, 'compiler_facts')}")
  return _layer("bubblebeam", True, lines, {"dimensions": {k: (len(v) if isinstance(v, list) else 1) for k, v in dims.items()},
                                           "population_size": size, "legal_coupled_rows": len(legal or []) if isinstance(legal, list) else None,
                                           "rejected_coupled_rows": len(rejected or []) if isinstance(rejected, list) else None}, evidence)


def _futuresight(stem:str, request:dict[str, Any] | None, result:dict[str, Any] | None) -> dict[str, Any]:
  file_r, file_q = RESULT.format(stem=stem), REQUEST.format(stem=stem)
  if isinstance((result or {}).get("futuresight_static_evidence"), dict):
    evidence, file, path = result["futuresight_static_evidence"], file_r, "futuresight_static_evidence"
  elif isinstance((request or {}).get("futuresight_evidence"), dict):
    evidence, file, path = request["futuresight_evidence"], file_q, "futuresight_evidence"
  else:
    return _layer("futuresight", False, [f"{NOT_RECORDED} in this run: no FutureSight evidence in {file_r} (futuresight_static_evidence) "
                                         f"or {file_q} (futuresight_evidence)"])
  rejections = [r for r in evidence.get("rejections") or [] if isinstance(r, dict)]
  assessments = [a for a in evidence.get("assessments") or [] if isinstance(a, dict)]
  reasons = Counter(str(r.get("reason")) for r in rejections)
  top = "; ".join(f"{k} x {why}" for why, k in reasons.most_common(TOP_REASONS))
  lines = [f"rejected statically: {len(rejections)} of {len(rejections) + len(assessments)} candidates" + (f" ({top})" if top else "")]
  if assessments:
    first = ", ".join(f"{i + 1}. {str(a.get('candidate_hash'))[:12]} (score {a.get('static_score')}, {a.get('static_reason')})"
                      for i, a in enumerate(assessments[:FIRST_ORDER]))
    lines.append(f"measurement order by static priority: {first}" + (f"; {len(assessments)} in all" if len(assessments) > FIRST_ORDER else ""))
  else:
    lines.append("measurement order: no candidate passed the static check")
  ptrs = [_ptr(file, f"{path}.rejections"), _ptr(file, f"{path}.assessments")]
  rows = (result or {}).get("futuresight_rejected_coupled_rows")
  if isinstance(rows, list):
    lines.append(f"coupled rows rejected statically: {len(rows)}")
    ptrs.append(_ptr(file_r, "futuresight_rejected_coupled_rows"))
  return _layer("futuresight", True, lines, {"rejected": len(rejections), "assessed": len(assessments),
                                            "top_reasons": [{"reason": why, "count": k} for why, k in reasons.most_common(TOP_REASONS)],
                                            "order": [a.get("candidate_hash") for a in assessments[:FIRST_ORDER]]}, ptrs)


def _measured(stem:str, result:dict[str, Any] | None, route:dict[str, Any] | None, route_index:int | None,
              status:dict[str, Any] | None, compare_index:int | None) -> dict[str, Any]:
  file_r = RESULT.format(stem=stem)
  c = (route or {}).get("compare") if isinstance((route or {}).get("compare"), dict) else None
  if result is None and c is None:
    why = f"; the search {status['status']}: {status.get('reason')}" if status and status.get("status") != "searched" else ""
    return _layer("measured", False, [f"{NOT_RECORDED} in this run: no {file_r}{why}"],
                  evidence=[_ptr(STATUS)] if status else [])
  lines, ptrs, facts = [], [], {}
  if result is not None:
    counts = result.get("counts") or {}
    rows = [(i, r) for i, r in enumerate(result.get("population") or []) if isinstance(r, dict)]
    measured = sorted(((i, r) for i, r in rows if r.get("state") == "MEASURED"), key=lambda ir: ir[1].get("rank") or 1 << 30)
    rejected = sum(int(counts.get(k) or 0) for k in ("rejected_invalid", "rejected_unsupported", "rejected_static", "rejected_incorrect"))
    lines.append(f"{_f(counts.get('total'), '{}', file_r, 'counts.total')} candidates: {_f(counts.get('measured_correct'), '{}', file_r, 'counts.measured_correct')} "
                 f"measured correct, {rejected} rejected, {int(counts.get('blocked') or 0)} blocked")
    ptrs.append(_ptr(file_r, "counts"))
    facts.update(total=counts.get("total"), measured_correct=counts.get("measured_correct"), rejected=rejected, blocked=counts.get("blocked"))
    if measured:
      i, best = measured[0]
      sched = ((best.get("candidate") or {}).get("schedule") or {})
      m = best.get("measurement") or {}
      summary = m.get("summary_ns") if isinstance(m.get("summary_ns"), dict) else {}
      spread = (f"min {summary['min'] / 1000:.1f}, max {summary['max'] / 1000:.1f} µs over {m.get('samples') or m.get('sample_count') or '?'} samples"
                if {"min", "max"} <= set(summary) else _nr(file_r, f"population[{i}].measurement.summary_ns"))
      text = role_compare.plan_text(sched)
      lines.append(f"best plan alone: {text} at {_f(m.get('median_ns'), '{:.0f}', file_r, f'population[{i}].measurement.median_ns')} ns median "
                   f"({spread}); plan_kind {sched.get('plan_kind')}, transforms {json.dumps(sched.get('transforms') or [], sort_keys=True)}")
      ptrs.append(_ptr(file_r, f"population[{i}]"))
      facts.update(best_plan=text, best_median_ns=m.get("median_ns"), plan_kind=sched.get("plan_kind"), transforms=sched.get("transforms"),
                   spread_ns=summary or None)
      default = next(((j, r) for j, r in measured if ((r.get("candidate") or {}).get("schedule") or {}).get("plan_kind") == role_compare.DEFAULT_KIND), None)
      if default is not None:
        j, d = default
        lines.append(f"the search's own reference kernel: {_f((d.get('measurement') or {}).get('median_ns'), '{:.0f}', file_r, f'population[{j}].measurement.median_ns')} ns median (not the model's kernel)")
        ptrs.append(_ptr(file_r, f"population[{j}]"))
        facts["reference_median_ns"] = (d.get("measurement") or {}).get("median_ns")
  if c is not None:
    k = c.get("kernel") if isinstance(c.get("kernel"), dict) else {}
    where = f"routes[{route_index}].compare" if route_index is not None else "routes[].compare"
    plan_us, model_us, calls = k.get("plan_us"), k.get("model_us_per_call"), c.get("role_calls_per_token")
    speed = f", {model_us / plan_us:.2f}x" if isinstance(plan_us, (int, float)) and isinstance(model_us, (int, float)) and plan_us else ""
    lines.append(f"the model's own kernels: {_f(model_us, '{:.1f}', POLICY, where + '.kernel.model_us_per_call')} µs per tensor x "
                 f"{_f(calls, '{:.0f}', POLICY, where + '.role_calls_per_token')} tensors per token; the plan alone "
                 f"{_f(plan_us, '{:.1f}', POLICY, where + '.kernel.plan_us')} µs{speed}")
    ptrs.append(_ptr(POLICY, where))
    facts.update(model_us_per_call=model_us, plan_us=plan_us, calls_per_token=calls, timing_source=c.get("timing_source"))
    ab = c.get("ab") if isinstance(c.get("ab"), dict) else None
    if ab:
      binding = ab.get("binding") if isinstance(ab.get("binding"), dict) else {}
      lines.append(f"whole-model A/B: {_f(ab.get('baseline_tok_s'), '{:.2f}', POLICY, where + '.ab.baseline_tok_s')} to "
                   f"{_f(ab.get('candidate_tok_s'), '{:.2f}', POLICY, where + '.ab.candidate_tok_s')} tok/s "
                   f"({_f(ab.get('delta_pct'), '{:+.1f}', POLICY, where + '.ab.delta_pct')}%); the plan reached "
                   f"{binding.get('calls_reached', NOT_RECORDED)} calls; token match {ab.get('token_match', NOT_RECORDED)}")
      ptrs.append(_ptr(POLICY, where + ".ab"))
      facts["ab"] = {k_: ab.get(k_) for k_ in ("baseline_tok_s", "candidate_tok_s", "delta_pct", "token_match", "route_bound")}
    elif result is not None or c.get("decided_by"):
      lines.append(f"whole-model A/B: {NOT_RECORDED} ({POLICY}: {where}.ab)" + (f"; decided by the {c['decided_by']}" if c.get("decided_by") else ""))
    chk = c.get("check") if isinstance(c.get("check"), dict) else None
    if chk:
      for who in ("winner", "model_kernel"):
        v = chk.get(who) if isinstance(chk.get(who), dict) else None
        if v:
          ours = v.get("boltbeam_us_less_floor")
          lines.append(f"BoltBeam's check of the {'winner' if who == 'winner' else 'model kernel'}: provider "
                       f"{_f(v.get('provider_us'), '{:.1f}', POLICY, where + f'.check.{who}.provider_us')} µs, BoltBeam "
                       f"{_f(ours, '{:.1f}', POLICY, where + f'.check.{who}.boltbeam_us_less_floor')} µs less the floor: {v.get('verdict')}"
                       + (f" ({v['reason']})" if v.get("reason") else ""))
      ptrs.append(_ptr(POLICY, where + ".check"))
      facts["check"] = chk
    if c.get("timing_source"):
      lines.append(f"every time here is a {c['timing_source']} time")
  if compare_index is not None:
    ptrs.append(_ptr(ev.COMPARE, f"roles[{compare_index}]"))
  return _layer("measured", True, lines, facts, ptrs)


def _emitter(role:str, quant:str, rows:int | None, k:int | None, target:str) -> dict[str, Any] | None:
  for q, roles, name, args, pins in EMITTERS:
    if (q == quant or q == "*") and (roles is None or role in roles):
      return {"name": name, "args": args.format(rows=rows if rows is not None else "<rows>", k=k if k is not None else "<k>", role=role, target=target),
              "pins": pins, "module": "tinygrad/llm/decode_kernels.py"}
  return None


def _promotion(role:str, quant:str, r:dict[str, Any], route:dict[str, Any] | None, route_index:int | None,
               request:dict[str, Any] | None, status:dict[str, Any] | None, shape:tuple[int | None, int | None],
               target:dict[str, Any], band:float | None, stem:str) -> dict[str, Any]:
  verdict = str(r.get("verdict") or "not_searched")
  reason = r.get("verdict_reason")
  where = f"routes[{route_index}].compare" if route_index is not None else "routes[].compare"
  lines = [f"verdict: {PLAIN_VERDICT.get(verdict, verdict)}" + (f" ({reason})" if reason else "")]
  ptrs = [_ptr(POLICY, f"routes[{route_index}]")] if route_index is not None else []
  facts: dict[str, Any] = {"verdict": verdict, "verdict_reason": reason}
  if (r.get("reason_word") or r.get("reason")) == "at the limit":
    lines.append(AT_LIMIT)
    return _layer("promotion", True, lines, {**facts, "plan": AT_LIMIT}, ptrs)
  c = (route or {}).get("compare") if isinstance((route or {}).get("compare"), dict) else {}
  k = c.get("kernel") if isinstance(c.get("kernel"), dict) else {}
  best = r.get("best_found") or {}
  plan_us, model_us, calls = best.get("plan_us", k.get("plan_us")), best.get("model_us", k.get("model_us_per_call")), best.get("calls_per_token", c.get("role_calls_per_token"))
  if verdict == "none_faster":
    n = r.get("candidates") if r.get("candidates") is not None else c.get("candidates")
    lines.append(f"nothing to emit: {_f(n, '{}', POLICY, where + '.candidates')} candidates measured, best "
                 f"{_f(plan_us, '{:.1f}', POLICY, where + '.kernel.plan_us')} µs against the model's own "
                 f"{_f(model_us, '{:.1f}', POLICY, where + '.kernel.model_us_per_call')} µs")
    ptrs.append(_ptr(POLICY, where))
    return _layer("promotion", True, lines, {**facts, "plan": "nothing to emit", "candidates": n, "plan_us": plan_us, "model_us": model_us}, ptrs)
  if verdict == "not_reproduced":
    lines.append("nothing to emit: BoltBeam's own check did not reproduce the provider's claim, so the plan is never promoted")
    ptrs.append(_ptr(POLICY, where + ".check"))
    return _layer("promotion", True, lines, {**facts, "plan": "nothing to emit"}, ptrs)
  if verdict == "not_searched":
    why = (status or {}).get("reason") or reason
    lines.append(NOT_SEARCHED + (f" ({why})" if why and why != reason else ""))  # the verdict line said it once
    return _layer("promotion", True, lines, {**facts, "plan": NOT_SEARCHED, "why_off": why}, ptrs + ([_ptr(STATUS, "reason")] if status else []))
  # applied or found, not applied: the plan, its gain, the emitter, the record, the check
  rows, kk = shape
  gain = ((model_us - plan_us) * calls / 1000.0) if all(isinstance(x, (int, float)) for x in (model_us, plan_us, calls)) else None
  lines.append(f"plan: {best.get('plan') or c.get('plan') or _nr(POLICY, where + '.plan')}; the plan alone "
               f"{_f(plan_us, '{:.1f}', POLICY, where + '.kernel.plan_us')} µs against the model's own "
               f"{_f(model_us, '{:.1f}', POLICY, where + '.kernel.model_us_per_call')} µs per call; gain if applied "
               + (f"{gain:.2f} ms per token ({calls:.0f} calls)" if gain is not None else _nr(POLICY, where + ".role_calls_per_token")))
  ptrs.append(_ptr(POLICY, where))
  if verdict == "found_not_applied":
    lines.append("found, not applied in this run" + (f": {c.get('reason')}" if c.get("reason") else ""))
  space = c.get("space") if isinstance(c.get("space"), dict) else {}
  emitter = ({"name": space["family"], "args": f"the winner's schedule ({c.get('plan')})",
              "pins": "the emitter's own validate on this shape (the search compiled it)", "module": "extra/llm_research/semantic_kernel_lowering.py"}
             if space.get("family") else _emitter(role, quant, rows, kk, str(target.get("target_id") or "")))
  if emitter:
    lines.append(f"emit with {emitter['name']}({emitter['args']}) from {emitter['module']}; shape pins: {emitter['pins']}")
  else:
    lines.append(f"emitter: no emitter in the fork for {quant} {role} ({NOT_RECORDED} in EMITTERS)")
  arch = (request or {}).get("target", {}).get("arch") if isinstance((request or {}).get("target"), dict) else None
  record = {"route": c.get("plan_id") or route.get("selected_route") if route else None, "backend": target.get("backend"),
            "architecture": arch, "target_id": target.get("target_id"), "shape": {"rows": rows, "k": kk}, "role": role, "quant": quant}
  lines.append(f"promotion record to add: route {record['route'] or _nr(POLICY, where + '.plan_id')}, backend {record['backend'] or _nr(POLICY, 'target.backend')}, "
               f"architecture {arch or _nr(REQUEST.format(stem=stem), 'target.arch')}, shape {rows} x {kk} ({role} {quant}); the record's shape gate admits this shape only")
  corr = (((request or {}).get("candidate_space") or {}).get("seed") or {}).get("correctness") if request else None
  oracle = (f"{corr.get('oracle')}, atol {corr.get('atol')}, rtol {corr.get('rtol')}" if isinstance(corr, dict)
            else _nr(REQUEST.format(stem=stem), "candidate_space.seed.correctness"))
  band_words = f"±{100 * band:.1f}%" if isinstance(band, (int, float)) else _nr("results", "ceiling.band")
  lines.append(f"check: correctness against the reference ({oracle}), then a Run whose tie-out puts {role} {quant} within {band_words} "
               f"of {_f(plan_us, '{:.1f}', POLICY, where + '.kernel.plan_us')} µs per call (the chip's band)")
  if request:
    ptrs.append(_ptr(REQUEST.format(stem=stem), "target"))
  return _layer("promotion", True, lines, {**facts, "plan": best.get("plan") or c.get("plan"), "plan_us": plan_us, "model_us": model_us,
                                           "calls_per_token": calls, "gain_ms_per_token": gain, "emitter": emitter,
                                           "promotion_record": record, "check": {"correctness": oracle, "band": band, "plan_us": plan_us}}, ptrs)


# --- fusions --------------------------------------------------------------------------------------------------------

FUSIONS = f"{role_compare.FOLDER}/{role_compare.FUSIONS_FILE}"


def _fusion_block(f:dict[str, Any], i:int) -> dict[str, Any]:
  """One fusion as Emit shows it: the roles it replaces, their combined ms lost in the model, BoltBeam's fused and
  unfused times per token, the proposal's rejections, the verdict. Read off kernel_compare/fusions.json only."""
  where = f"fusions[{i}]"
  name = " + ".join(f.get("covers") or [])
  lines = [f"replaces {name} ({f.get('quant')}) with one launch, {_f((f.get('instance') or {}).get('calls_per_token'), '{}', FUSIONS, where + '.instance.calls_per_token')} launches per token"]
  rows = f.get("in_model_rows") or []
  if rows:
    lines.append("in the model: " + ", ".join(f"{r['row']} {r['ms']:.3f} ms ({r['lost_ms']:.3f} ms lost)" for r in rows)
                 + f"; combined {_f(f.get('in_model_ms'), '{:.3f}', FUSIONS, where + '.in_model_ms')} ms, "
                 + f"{_f(f.get('lost_ms'), '{:.3f}', FUSIONS, where + '.lost_ms')} ms lost")
  if f.get("in_model_note"):
    lines.append(str(f["in_model_note"]))
  fused, unfused = f.get("fused") or {}, f.get("unfused") or {}
  if fused:
    lines.append(f"BoltBeam's fastest fused kernel: {fused.get('plan')}, {_f(fused.get('us_less_floor'), '{:.1f}', FUSIONS, where + '.fused.us_less_floor')} µs "
                 f"less the floor, {_f(fused.get('ms_per_token'), '{:.3f}', FUSIONS, where + '.fused.ms_per_token')} ms per token; "
                 "checked against the unfused nodes computed by BoltBeam")
  if unfused.get("ms_per_token") is not None:
    lines.append("unfused, each node alone: " + ", ".join(f"{n['node']} {n['us_less_floor']:.1f} µs" for n in unfused.get("nodes") or [])
                 + f" less the floor; {unfused['ms_per_token']:.3f} ms per token")
  elif unfused.get("lower_bound_ms") is not None:
    lines.append("unfused, each node alone: " + ", ".join(f"{n['node']} {n['us_less_floor']:.1f} µs" + ("" if n.get("verdict") == "reproduced" else " (noisy, not counted)")
                                                    for n in unfused.get("nodes") or [] if n.get("us_less_floor") is not None)
                 + f" less the floor; at least {unfused['lower_bound_ms']:.3f} ms per token")
  elif unfused.get("why"):
    lines.append(f"unfused: {unfused['why']}")
  rejected = f.get("rejected_rows") or []
  if rejected:
    lines.append(f"rows the emitter refused: {len(rejected)} (" + "; ".join(sorted({str(r.get('reason')) for r in rejected})) + ")")
  refused = (f.get("search") or {}).get("refused") or []
  if refused:
    lines.append(f"candidates refused in the search: {len(refused)} (" + "; ".join(sorted({f"{r.get('plan')}: {r.get('reason')}" for r in refused})[:3]) + ")")
  lines.append(f"verdict: {PLAIN_VERDICT.get(str(f.get('verdict')), str(f.get('verdict')))}" + (f" ({f['reason']})" if f.get("reason") else ""))
  return {"name": name, "covers": list(f.get("covers") or []), "quant": f.get("quant"), "family": f.get("family"),
          "lost_ms": f.get("lost_ms"), "in_model_ms": f.get("in_model_ms"), "fused_ms": fused.get("ms_per_token"),
          "unfused_ms": unfused.get("ms_per_token"), "verdict": f.get("verdict"), "lines": lines,
          "evidence": [_ptr(FUSIONS, where)] + ([_ptr(f["check_result"])] if f.get("check_result") else [])}


# --- the plan ----------------------------------------------------------------------------------------------------

def _shape(route:dict[str, Any] | None, profile_role:dict[str, Any] | None) -> tuple[int | None, int | None]:
  s = (route or {}).get("shape")
  if isinstance(s, list) and len(s) == 2:
    return int(s[0]), int(s[1])
  if profile_role and profile_role.get("rows") is not None:
    return profile_role.get("rows"), profile_role.get("cols")
  return None, None


def _role_block(files:_Files, r:dict[str, Any], i:int, est:dict[str, Any] | None, policy:dict[str, Any],
                status:dict[str, Any] | None, compare:dict[str, Any] | None, band:float | None) -> dict[str, Any]:
  role, quant = str(r["role"]), str(r["quant"])
  from boltbeam.role_key import role_key
  rk = role_key(r)
  routes = policy.get("routes") or []
  route_index = next((j for j, x in enumerate(routes) if role_key(x) == rk), None)
  if route_index is None and len(rk) == 2:  # a table row without a shape: the one route of its role and format
    route_index = next((j for j, x in enumerate(routes) if role_key(x)[:2] == rk), None)
  stem = _stem(role, quant, (routes[route_index].get("shape") if route_index is not None else r.get("shape")))
  if not (files.run / REQUEST.format(stem=stem)).is_file() and (files.run / REQUEST.format(stem=_stem(role, quant))).is_file():
    stem = _stem(role, quant)  # a run searched before roles carried their shape in the file names
  route = routes[route_index] if route_index is not None else None
  profile_role = None
  for p in r.get("evidence") or []:
    if p.get("file") == ev.PROFILE:
      try:
        profile_role = ev.resolve(files.run, p)
      except (KeyError, IndexError, OSError, ValueError):
        profile_role = None
  rows, k = _shape(route, profile_role)
  request, result = files.get(REQUEST.format(stem=stem)), files.get(RESULT.format(stem=stem))
  compare_index = next((j for j, x in enumerate((compare or {}).get("roles") or []) if role_key(x) == role_key(route or r)), None)
  lost = r.get("est_lost_ms") if est else r.get("lost_ms")
  floor = bool(est and est.get("columns_words"))
  basis = ("isolated, less the dispatch floor" if floor else "isolated" if r.get("est_ms") is not None or est else "in the model")
  target = policy.get("target") or {}
  target = {"backend": target.get("backend"), "target_id": target.get("target_id") or load_manifest(files.run).get("target_id")}
  layers = [_bubblebeam(stem, request), _futuresight(stem, request, result),
            _measured(stem, result, route, route_index, status, compare_index),
            _promotion(role, quant, r, route, route_index, request, status, (rows, k), target, band, stem)]
  evidence = list(r.get("evidence") or []) + ([_ptr(POLICY, f"routes[{route_index}]")] if route_index is not None else [])
  return {"rank": i + 1, "role": role, "quant": quant, "rows": rows, "cols": k, "lost_ms": lost, "lost_is_estimate": bool(est),
          "mb_per_call": r.get("mb_per_call"), "calls_per_token": r.get("calls_per_token"), "us_per_call": r.get("us_per_call"),
          "us_per_call_less_floor": r.get("us_per_call_less_floor"), "gbs": r.get("gbs"), "pct_peak": r.get("pct_peak"),
          "rate_basis": basis, "ideal_ms": r.get("ideal_ms"), "reason": r.get("reason"), "reason_word": r.get("reason_word"),
          "verdict": r.get("verdict") or "not_searched", "plan": layers[3]["facts"].get("plan"), "layers": layers, "evidence": evidence}


def gameplan(run:pathlib.Path, res:dict[str, Any], *, now:_dt.date | None = None) -> dict[str, Any]:
  """The plan for one run from the seam's results (workflow/screen.py results) and the run's files. Refused when the
  run holds no per-role table: there is nothing to plan from."""
  loss = res.get("loss") or {}
  roles = loss.get("roles") or []
  if not roles:
    raise Refused("no per-role table in this run: Run first" + (f" ({loss['missing']})" if loss.get("missing") else ""))
  files = _Files(run)
  manifest = load_manifest(run)
  files.read.append("run_manifest.json")
  policy = files.get(POLICY) or {}
  status = files.get(STATUS)
  compare = files.get(ev.COMPARE)
  est = loss.get("estimate")
  t = loss.get("tie_out") or {}
  token = next((x for x in loss.get("runtimes") or [] if not x.get("per_role")), None)
  ceil = res.get("ceiling") or {}
  band = ceil.get("band")
  from boltbeam.workflow.screen import measurement_words
  header = {
    "model_id": res.get("model_id"), "target_id": res.get("target_id"), "engine": (res.get("engine") or {}).get("provider") or loss.get("provider"),
    "weight_format": (res.get("engine") or {}).get("weight_format"), "batch": t.get("batch") or 1,
    "measurement": measurement_words(res.get("measurement")), "role_source": loss.get("role_source_words"),
    "token_ms": token["ms"] if token else None, "tok_s": token["tok_s"] if token else None,
    "limit_ms": loss.get("limit_ms"), "limit_tok_s": loss.get("limit_tok_s"),
    "pct_of_limit": (100.0 * loss["limit_ms"] / token["ms"]) if token and loss.get("limit_ms") and token.get("ms") else None,
    "lost_ms": token["lost_ms"] if token else None,
    "where_token_goes": loss.get("where_token_goes"),
    "evidence": list((loss.get("step") or {}).get("evidence") or []) + list((loss.get("evidence") or {}).get("common") or []),
  }
  order = lambda r: -((r.get("est_lost_ms") if est else r.get("lost_ms")) or 0.0)  # noqa: E731
  ordered = sorted(roles, key=lambda r: (order(r), str(r["role"]), str(r["quant"])))
  blocks = [_role_block(files, r, i, est, policy, status, compare, band) for i, r in enumerate(ordered)]
  found = files.get(FUSIONS) or {}
  fusions = sorted((_fusion_block(f, i) for i, f in enumerate(found.get("fusions") or [])),
                   key=lambda b: (-(b["lost_ms"] or 0.0), b["name"], str(b["quant"])))
  for name in ("model_profile.json", "measure_status.json", ev.MACHINE, ev.STEP4):
    if (run / name).is_file() and name not in files.read:
      files.read.append(name)
  for p in header["evidence"] + [p for b in blocks for p in b["evidence"]]:
    if (run / p["file"]).is_file() and p["file"] not in files.read:
      files.read.append(p["file"])
  today = (now or _dt.date.today()).isoformat()
  return {"schema": SCHEMA, "kind": "gameplan", "id": run.name, "header": header,
          "ordered_by": "est_lost_ms (an estimate from isolated times)" if est else "lost_ms",
          "roles": blocks, "fusions": fusions,
          "footer": {"files_read": sorted(set(files.read)), "date": today, "sentence": CLOSING}}


# --- the paper version ---------------------------------------------------------------------------------------------

def _fold(ptrs:list[dict[str, Any]]) -> str:
  by_file:dict[str, list[str]] = {}
  for p in ptrs:
    by_file.setdefault(str(p["file"]), [])
    if p.get("path") and p["path"] not in by_file[str(p["file"])]:
      by_file[str(p["file"])].append(str(p["path"]))
  parts = []
  for file, paths in by_file.items():
    parts.append(file + (" › " + ", ".join(paths[:4]) + (f", +{len(paths) - 4} more" if len(paths) > 4 else "") if paths else ""))
  return "evidence: " + ", ".join(parts) if parts else ""


def _num(v:Any, fmt:str) -> str:
  return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else NOT_RECORDED


def to_markdown(plan:dict[str, Any]) -> str:
  from boltbeam.workflow.screen import where_lines
  h = plan["header"]
  out = [f"# Gameplan: {h.get('model_id')} on {h.get('target_id')}", "",
         f"Run {plan['id']} · {h.get('engine')} · batch {h.get('batch')} · {h.get('weight_format') or 'weights ' + NOT_RECORDED}"
         + (f" · {h['measurement']}" if h.get("measurement") else "")]
  if h.get("token_ms") is not None:
    out.append(f"Measured {_num(h['token_ms'], '{:.1f}')} ms per token ({_num(h['tok_s'], '{:.1f}')} tok/s); the limit is "
               f"{_num(h['limit_ms'], '{:.1f}')} ms ({_num(h['limit_tok_s'], '{:.1f}')} tok/s); {_num(h['pct_of_limit'], '{:.0f}')}% of the limit; "
               f"{_num(h['lost_ms'], '{:.1f}')} ms per token lost.")
  else:
    out.append(f"Measured token: {NOT_RECORDED} (timing_trace.json: rows[].tok_s).")
  if h.get("role_source"):
    out.append(f"Per-role source: {h['role_source']}.")
  if fold := _fold(h.get("evidence") or []):
    out.append(f"  {fold}")
  if where := where_lines(h.get("where_token_goes")):
    out += ["", f"## {where[0].rstrip(':')}", where[1].strip(), "", "```", *(l[2:] for l in where[2:]), "```"]
  out += ["", f"Roles ordered by {plan['ordered_by']}, worst first. Each role shows the search's lineage: BubbleBeam, FutureSight, Measured, Promotion.", ""]
  for b in plan["roles"]:
    est = " est." if b.get("lost_is_estimate") else ""
    out.append(f"## {b['rank']}. {b['role']} {b['quant']} · {_num(b.get('lost_ms'), '{:.2f}')} ms lost{est}")
    less = f" ({_num(b['us_per_call_less_floor'], '{:.1f}')} less floor)" if b.get("us_per_call_less_floor") is not None else ""
    out.append(f"{_num(b.get('rows'), '{}')} x {_num(b.get('cols'), '{}')} · {_num(b.get('mb_per_call'), '{:.1f}')} MB per call · "
               f"{_num(b.get('calls_per_token'), '{:.0f}')} calls per token · {_num(b.get('us_per_call'), '{:.1f}')} µs/call{less} · "
               f"{_num(b.get('gbs'), '{:.1f}')} GB/s · {_num(b.get('pct_peak'), '{:.1f}')}% of peak ({b.get('rate_basis')}) · "
               f"{b.get('reason') or NOT_RECORDED} · {PLAIN_VERDICT.get(str(b.get('verdict')), str(b.get('verdict')))}")
    if fold := _fold(b.get("evidence") or []):
      out.append(f"  {fold}")
    for layer in b["layers"]:
      out.append(f"- **{layer['title']}**: " + layer["lines"][0])
      for line in layer["lines"][1:]:
        out.append(f"  {line}")
      if fold := _fold(layer.get("evidence") or []):
        out.append(f"  {fold}")
    out.append("")
  for b in plan.get("fusions") or []:
    out.append(f"## Fusion: {b['name']} {b['quant']} · {_num(b.get('lost_ms'), '{:.3f}')} ms lost combined")
    out += [f"- {line}" for line in b["lines"]]
    if fold := _fold(b.get("evidence") or []):
      out.append(f"  {fold}")
    out.append("")
  f = plan["footer"]
  out += ["---", f"Files read: {', '.join(f['files_read'])}.", f"Date: {f['date']}.", f["sentence"], ""]
  return "\n".join(out)


def emit(run:pathlib.Path, res:dict[str, Any], *, now:_dt.date | None = None) -> dict[str, Any]:
  """Write gameplan.json and gameplan.md into the run (the one writer) and return where, with the plan and its paper
  version, so a screen shows the same text that was written."""
  plan = gameplan(run, res, now=now)
  md = to_markdown(plan)
  (run / JSON_FILE).write_text(pretty_json(plan), encoding="utf-8")
  (run / MD_FILE).write_text(md, encoding="utf-8")
  return {"schema": SCHEMA, "kind": "emitted", "id": run.name, "dir": str(run), "files": [JSON_FILE, MD_FILE], "plan": plan,
          "markdown": md}
