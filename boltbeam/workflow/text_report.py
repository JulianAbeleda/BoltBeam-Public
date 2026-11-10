"""The results as plain text, to paste into a message: summary.txt (screen save) and the summary the report and
Emit print. The cells are the tables' own (tie_out.table); nothing here formats a number the seam did not."""
from __future__ import annotations

import pathlib
import sys
from typing import Any

from boltbeam.collectors import providers
from boltbeam.report.html import sectioned_cells
from boltbeam.target.targets import get_target
from boltbeam.workflow import prefill_audit
from boltbeam.artifacts.run_io import load_manifest
from boltbeam.workflow.replay import lines as replay_lines
from boltbeam.workflow.results import estimate_how, measurement_words
from boltbeam.workflow.session import lines as session_lines

def where_lines(w:dict[str, Any] | None) -> list[str]:
  """A where table (tie_out.where_token_goes, where_prefill_goes) as plain text: one table, worst first, each column
  as wide as its widest cell. The summary and the gameplan print these lines; the cells are the table's own."""
  if not w:
    return []
  cells = [list(w["columns"]), *sectioned_cells(w)]
  width = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
  out = [w["title"] + (" (estimate)" if w["estimate"] else "") + ":", *([f"  {w['qualification']}"] if w.get("qualification") else []), f"  {w['words']}"]
  for row in cells:
    out.append(("  " + row[0].ljust(width[0]) + " " + " ".join(c.rjust(width[i + 1]) for i, c in enumerate(row[1:]))).rstrip())
  return out


def curve_lines(c:dict[str, Any] | None) -> list[str]:
  """workflow/prefill.curve as plain text: one line per prompt length, then ms and MFU per row at every length."""
  if not c or not c.get("points"):
    return []
  f = lambda v, fmt: fmt.format(v) if v is not None else "-"  # noqa: E731
  out = ["Prefill against prompt length:", f"  {c['words']}"]
  for p in c["points"]:
    out.append(f"  {p['length']:>6} tokens: {p['ms']:9.1f} ms, {p['tok_s']:8.0f} tok/s, MFU {f(p.get('mfu_pct'), '{:.1f}%')}"
               + (f" ({p['mfu_how']})" if p.get("mfu_how") else "") + (f", {p['sm_mhz']:.0f} MHz mean" if p.get("sm_mhz") else "")
               + ("" if p.get("batched") else ", not batched") + (", THROTTLED" if p.get("throttled") else ""))
  out += [f"  {x['length']:>6} tokens: not read: {x['words']}" for x in c.get("failed") or []]
  width = max(len(n) for n in c["rows"]) if c["rows"] else 0
  for n in c["rows"]:
    out.append(f"  {n:<{width}} " + "  ".join(
      f"{f((p['rows'].get(n) or {}).get('ms'), '{:.2f}')} ms ({f((p['rows'].get(n) or {}).get('mfu_pct'), '{:.0f}%')})" for p in c["points"]))
  return out


def _tie_line(i:int, line:dict[str, Any]) -> str:
  """One tie-out line: the limit first, then each measured line added to it."""
  return f"  {'  ' if i == 0 else '+ '}{line['label']:<48} {line['ms']:9.3f}  {line['how']}"


def _token_line(t:dict[str, Any], batch:int) -> str:
  """The tie-out's sum line: the measured token, or the measured step of B streams."""
  what = "measured token" if batch == 1 else f"measured step of {batch} streams"
  return f"  = {what:<48} {t['token_ms']:9.3f}  {t.get('token_source') or ''}"


def summary_text(res:dict[str, Any], why_no_roles:str | None = None) -> str:
  """The tie-out and the per-role table as plain text, to paste into a message. why_no_roles is this machine's
  reason per-role time cannot be taken here; it replaces the generic "Run times each role" sentence."""
  loss = res.get("loss") or {}
  t = loss.get("tie_out") or {}
  m = next((r for r in loss.get("runtimes") or [] if not r.get("per_role")), None)
  batch = loss.get("batch") or 1  # the batch every figure below is of (the token's, loss_block)
  head = f"{res.get('model_id')} on {res.get('target_id')} with {loss.get('provider')}, batch {batch}"
  if loss.get("workload") == "replay" and (loss.get("replay") or {}).get("headline"):  # workflow/replay.py: per session
    h = loss["replay"]["headline"]
    head += (f": a {h['warm_turns']}-turn replay at {h['e_pct']:.1f}% of its roofline, {(h['time_ms'] - h['roofline_ms']) / 1e3:.1f} s "
             f"of {h['time_ms'] / 1e3:.1f} s lost" if h.get("e_pct") is not None else
             f": a {h['warm_turns']}-turn replay in {h['time_ms'] / 1e3:.1f} s, no ceiling ({loss.get('reason') or 'see below'})")
  elif m and loss.get("limit_tok_s"):
    pct = 100.0 * loss["limit_ms"] / m["ms"] if m.get("ms") else 0.0
    each = f" per stream ({batch * m['tok_s']:.1f} in total)" if batch > 1 else ""  # tools/check_run_math.py HEAD
    head += (f": {m['tok_s']:.1f} tok/s{each} measured, limit {loss['limit_tok_s']:.1f} tok/s{each and ' per stream'}, {pct:.0f}% of roofline, "
             f"{m['lost_ms']:.1f} ms {loss.get('per') or 'per token'} lost")
  elif m:  # measured, with no limit to hold it against yet (a prefill not yet captured in the model)
    head += f": {m['tok_s']:.1f} tok/s measured, {m['ms']:.1f} ms {loss.get('per') or 'per token'}; no limit yet: {loss.get('reason') or loss.get('missing')}"
  else:
    head += ": not measured"
  lines = [head]
  if state := loss.get("state_words"):  # the limit counts each stream's recurrent state, or says why it does not
    lines.append(f"Limit: {state}.")
  if basis := loss.get("limit_basis"):  # a split whose copies were not measured: its limit is its weights alone
    lines.append(f"Limit: {basis}.")
  if spec := loss.get("speculation"):  # the limit is this speculation's, at the acceptance the capture measured
    lines.append(f"Speculative decoding: {spec['words']}. The limit is per token at that acceptance: each verify step "
                 f"reads the target's weights once and the draft head's once per draft, over the tokens the step emits.")
  elif words := (res.get("engine") or {}).get("speculation_words"):
    lines.append("Speculative decoding: " + "; ".join(words) + ".")
  step = loss.get("step") or {}
  for a in step.get("also") or []:
    lines.append(f"Also measured at context {a['context']}: {a['tok_s']:.1f} tok/s (not the headline: {step.get('rule')})")
  if step.get("graph_failed"):
    lines.append("Graph replay failed on this run: the token ran without graphs" + (f" ({step['graph_error']})" if step.get("graph_error") else ""))
  est = loss.get("estimate")
  if loss.get("check_status") == "not checked":
    lines.append("Not checked: no matching repeat band for one or more timing rows.")
  if t.get("refused"):
    lines.append(f"Not tied out: {t['refused']}")
  elif est and t.get("token_ms") is not None:
    lines.append(f"Tie-out, ms {loss.get('per') or 'per token'} at context {t['context']:.0f} (only what is measured):")
    lines.append(f"    {'limit at context 1 (ideal)':<48} {t['limit_ms_ctx1']:9.3f}  derived")
    for i, l in enumerate(t["lines"]):
      lines.append(_tie_line(i, l))
    lines.append(_token_line(t, batch))
    lines.append(f"  estimated split (isolated): weight kernels {'up to ' if est['scaled'] else ''}{est['weight_ms']:.1f} ms, "
                 f"other and gaps {'at least ' if est['scaled'] else ''}{est['other_ms']:.1f} ms ({est['other_how']})")
    if ct := t.get("captured_token"): lines.append(f"  {ct['words']}")
  elif t.get("lines") and t.get("token_ms") is not None:
    lines.append(f"Tie-out, ms {loss.get('per') or 'per token'} at context {t['context']:.0f}:")
    for i, l in enumerate(t["lines"]):
      lines.append(_tie_line(i, l))
      for k in l.get("kernels") or []:  # GEMVs no role took, named with their launches
        lines.append(f"      {k['kernel']}: {k['calls_per_token']:.0f} per token, {k['us_per_call']:.1f} us each, {k['ms']:.3f} ms")
      if l.get("parts"):
        lines.append("      " + ", ".join(f"{p['kind']} {p['ms']:.3f}" for p in l["parts"]))
    lines.append(_token_line(t, batch))
    if b := t.get("gaps_bound"):  # over the ceiling: the gaps the untraced token leaves room for, and what recording added
      lines.append(f"  − {'added by recording (captured gaps − bound)':<48} {b['captured_ms'] - b['ms']:9.3f}  difference")
      lines.append(f"  = {'untraced token':<48} {b['untraced_ms']:9.3f}  gaps at most {b['ms']:.3f} ms (untraced token − kernels)")
  lines += where_lines(loss.get("where_token_goes"))
  lines += where_lines(loss.get("capture_comparison"))
  lines += curve_lines(loss.get("prefill_curve"))
  lines += prefill_audit.lines(loss.get("prefill_audit"))
  lines += session_lines(loss.get("session"))
  lines += replay_lines(loss.get("replay"))
  if res.get("batches"):  # step 4's points, each against its own limit (batch_rows); tools/check_run_math.py BATCH
    lines.append("Per batch, untraced, each beside its own limit (the weights read once, each stream's own reads B times):")
    for b in res["batches"]:
      lim = b["limit"]
      lines.append(f"  context {b['context']:g}, batch {b['batch']}: {b['tok_s_stream']:.2f} tok/s per stream, "
                   f"{b['tok_s_total']:.1f} tok/s in total, step {b['step_ms']:.3f} ms; " +
                   (f"limit {lim['step_ms']:.3f} ms, {lim['tok_s_total']:.1f} tok/s in total; {b['pct_of_limit']:.1f}% of limit"
                    if b.get("pct_of_limit") is not None else f"no limit: {lim.get('reason')}"))
  if t.get("band"):
    lines.append(f"Band: {t['band']}")
  roles = loss.get("roles") or []
  w = max([12] + [len(str(r["role"])) for r in roles] + [len(str(n.get("role") or "")) for n in loss.get("not_timed") or []])  # the role column fits its longest name
  if roles and est:
    lines.append("Per role, estimated from isolated kernel times:")
    lines.append(f"  {est['method']}")
    lines.append(f"  est. columns {est['label']}; " + estimate_how(est))
    floor = est.get("columns_words") is not None  # the rows carry a dispatch floor: a less-floor column, said once
    if floor:
      lines.append(f"  {est['columns_words']}")
    lines.append(f"  {'role':<12} {'quant':<5} {'MB/call':>8} {'us/call':>8}{' less floor' if floor else ''} {'GB/s':>6} {'% peak':>7} "
                 f"{'ideal':>7} {'est. ms in token':>16} {'est. lost':>9}  why")
    for r in roles:
      mb = f"{r['mb_per_call']:.1f}" if r.get("mb_per_call") is not None else ""
      us = f"{r['us_per_call']:.1f}" if r.get("us_per_call") is not None else ""
      less = (f" {r['us_per_call_less_floor']:10.1f}" if r.get("us_per_call_less_floor") is not None else f" {'':>10}") if floor else ""
      g = f"{r['gbs']:.1f}" if r.get("gbs") is not None else ""
      pct = f"{r['pct_peak']:.1f}%" if r.get("pct_peak") is not None else ""
      lines.append(f"  {r['role']:<{w}} {r['quant']:<5} {mb:>8} {us:>8}{less} {g:>6} {pct:>7} {r['ideal_ms']:7.3f} "
                   f"{r['est_ms']:16.3f} {r['est_lost_ms']:9.3f}  {r.get('reason') or ''}")
  elif roles:
    lines.append(f"Per role ({loss.get('source') or ''}):")
    lines.append(f"  {'role':<{w}} {'quant':<5} {'ideal':>7} {'actual':>7} {'lost':>6} {'% peak':>7} {'us/call':>8}  why")
    for r in roles:
      pct = f"{r['pct_peak']:.1f}%" if r.get("pct_peak") is not None else ""
      us = f"{r['us_per_call']:.1f}" if r.get("us_per_call") is not None else ""
      found = (r.get("best_found") or {}).get("text")
      verdict = (r.get("verdict") or "").replace("_", " ") + (f" ({found})" if found else "")
      lines.append(f"  {r['role']:<{w}} {r['quant']:<5} {r['ideal_ms']:7.3f} {r['actual_ms']:7.3f} {r['lost_ms']:6.3f} "
                   f"{pct:>7} {us:>8}  {r.get('reason') or ''}" + (f"; kernel search: {verdict}" if verdict else ""))
    if loss.get("not_attributed_ms") is not None:
      lines.append(f"  {'not attributed':<{w + 6}} {'':>7} {loss['not_attributed_ms']:7.3f}")
    if nxt := (loss.get("search") or {}).get("next"):
      lines.append(f"Next: {nxt['what']}. {nxt['do']}")
  if coverage := (loss.get("attribution") or {}).get("matrix_work"):
    if not coverage["complete"]: lines.append(coverage["words"])
  if unresolved := loss.get("unpaired_launches"):
    lines.append(f"Attribution incomplete: {sum(r['calls'] for r in unresolved)} launches lack logical program metadata; work is not checked.")
  for n in loss.get("not_timed") or []:  # a limit role the timer had no adapter for: its own row, out of the floor rule and the sums
    lines.append(f"  {n['role']:<{w}} {n['quant']:<5} not timed: {n.get('reason') or n.get('status')}; left out of the floor rule and the sums")
  for n in loss.get("ceiling_refused") or []:  # a row below its own floor: out of the losses and shares, said here
    lines.append(f"  {n['role']:<{w}} {n['quant']:<5} {n['ideal_ms']:7.3f} {n['actual_ms']:7.3f}  {n['reason']}")
  if refused := loss.get("ceiling_refused"):
    lines.append(f"Totals exclude {len(refused)} row{'s' if len(refused) > 1 else ''} whose ceiling is refused: "
                 + ", ".join(f"{n['role']} {n['quant']}" for n in refused) + ". The other rows keep their ceilings.")
  if a := loss.get("ambiguous"):  # the pairing the solver could not settle (collectors/pairing.py), as the TUI says it
    what = {"tie": "several pairings fit equally", "search_limit": "the search stopped before it finished",
            "no_fit": "it found no fit"}.get(a.get("kind"), "it is not certain")
    lines.append(f"Pairing: {what}: {a.get('reason')}. Read the per-role rows as unsure.")
  for u in loss.get("unpaired_roles") or []:
    lines.append(f"Unpaired, so in not attributed: {u.get('role')} {u.get('quant')}" + (f": {u['reason']}" if u.get("reason") else "") + ".")
  if not roles and why_no_roles:
    lines.append(f"Per role: {why_no_roles}")
  elif not roles and loss.get("missing"):
    lines.append(f"Per role: {loss['missing']}")
  if (cc := loss.get("cross_check")) and (cc.get("rows") or cc.get("not_measured")):
    lines.append(f"Cross-check ({cc.get('words')}):")
    if cc.get("columns_words"):
      lines.append(f"  {cc['columns_words']}")
    for r in cc["rows"]:
      less = f" {r['us_per_call_less_floor']:8.1f} us less floor" if r.get("us_per_call_less_floor") is not None else ""
      g = f" {r['gbs']:6.1f} GB/s" if r.get("gbs") is not None else ""
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {r['us_per_call']:8.1f} us{less}{g}")
    for r in cc.get("not_measured") or []:
      lines.append(f"  {r['role']} {r['quant']}: not measured: {r['reason']}")
  probe = res.get("probe") or {}
  if probe.get("rows"):
    lines.append(f"{probe.get('label')}:")
    for r in probe["rows"]:
      pct = f"{r['pct_peak']:.0f}%" if r.get("pct_peak") is not None else ""
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {r['gbs']:6.1f} GB/s {pct:>5} of peak")
  if loss.get("role_source_words"):
    lines.append(f"Per-role source: {loss['role_source_words']}")
  if words := measurement_words(res.get("measurement")):
    lines.append(words)
  if lat := loss.get("latency"):
    lines.append(f"Latency in the reason rule: {lat['us']:.1f} us, {lat['source']}")
  return "\n".join(lines) + "\n"


def why_no_roles(run:pathlib.Path, res:dict[str, Any]) -> str | None:
  """The reason per-role time cannot be taken on this machine for the run's engine, as the Run screen says it."""
  loss = res.get("loss") or {}
  if loss.get("roles"):
    return None
  try:
    target = get_target(load_manifest(run).get("target_id"))
  except SystemExit:
    return None
  plan = providers.capture_method(loss.get("provider") or providers.DEFAULT, target.backend)
  if plan.get("method"):
    return None
  where = "this Mac" if sys.platform == "darwin" else "this machine"
  return f"Not possible on {where}: {plan.get('reason')}."
