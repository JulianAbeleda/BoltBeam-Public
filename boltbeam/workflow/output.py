from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.report.html import next_step, render_run_html
from boltbeam.vocab import SCHEMA_OUTPUT_MANIFEST
from boltbeam.workflow.common import load_manifest, read_json, run_dir, update_manifest, write_json

OUTPUT_ARTIFACTS = ("output_manifest.json", "summary.md", "rollback.md", "provider_plan.json", "report.html")


def _load_optional(run:pathlib.Path, name:str) -> dict[str, Any]:
  p = run / name
  return read_json(p) if p.exists() else {}


def _regime_label(row:dict[str, Any]) -> str:
  shape = row.get("shape") or []
  shape_s = "x".join(str(x) for x in shape) if shape else "unknown"
  role = row.get("role") or "unknown"
  quant = row.get("quant") or "unknown"
  regime = row.get("classification") or "unknown"
  bottleneck = row.get("visible_bottleneck") or "unknown"
  action = row.get("next_action") or "collect more evidence"
  return f"- `{role}` `{quant}` `{shape_s}`: `{regime}` via `{bottleneck}`; {action}"


def _timing_label(row:dict[str, Any]) -> str:
  shape = row.get("shape") or []
  shape_s = "x".join(str(x) for x in shape) if shape else "unknown"
  role = row.get("role") or "unknown"
  pct = row.get("pct_step")
  pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "unknown"
  wall = row.get("wall_us")
  wall_s = f"{wall:.1f}us" if isinstance(wall, (int, float)) else "unknown"
  return f"- `{role}` `{row.get('quant') or 'unknown'}` `{shape_s}`: `{row.get('classification')}` {pct_s}, {wall_s}"


def _missing_runner_outputs(primitive:dict[str, Any], timing:dict[str, Any]) -> list[str]:
  missing = []
  if not primitive:
    missing.append("probe_evidence")
  if not timing:
    missing.append("timing_trace")
  return missing


def measured_line(report:dict[str, Any], res:dict[str, Any], measure:dict[str, Any]) -> str | None:
  """The run's measured status in one line, from measure_status.json and the results: "measured, in-model (nsys):
  7 of 7 weight roles timed". None when the run measured nothing; the analysis report's route status is then the
  status to show."""
  if measure.get("status") != "measured":
    return None
  from boltbeam.collectors import engine_kernels
  cap = measure.get("capture") or {}
  label = (cap.get("measurement") or {}).get("label") or cap.get("method") or "whole step"
  method = cap.get("method")  # the vendor tool is named beside "in-model"; the kernel timer's label already says itself
  how = f"{label} ({method})" if method and method != engine_kernels.METHOD and method not in label else label
  roles = (res.get("loss") or {}).get("roles") or []
  total = report.get("role_count") or len(roles)
  per_role = f": {len(roles)} of {total} weight roles timed" if roles else ": the whole step only, no per-role time"
  return f"measured, {how}{per_role}"


def _summary_md(manifest:dict[str, Any], profile:dict[str, Any], report:dict[str, Any],
                plan:dict[str, Any], primitive:dict[str, Any], timing:dict[str, Any],
                runner:dict[str, Any], res:dict[str, Any] | None = None, measure:dict[str, Any] | None = None) -> str:
  from boltbeam.workflow.screen import summary_text
  res, measure = res or {}, measure or {}
  if replay:=(res.get('loss') or {}).get('replay'):
    if replay.get('forced'):
      return '\n'.join([f"# BoltBeam Replay Summary: {manifest.get('model_id', 'unknown')}", '',
                        *replay['headline_lines'], '', replay['words'], '', replay['next_step'], ''])
  probe = res.get("probe")
  measured = measured_line(report, res, measure)
  roles = (res.get("loss") or {}).get("roles") or []
  lines = [
    f"# BoltBeam Run Summary: {manifest.get('model_id', profile.get('model_id', 'unknown'))}",
    "",
    f"- Latest stage: `{manifest.get('latest_stage')}`",
    f"- Model format: `{manifest.get('model_format', 'unknown')}`",
    f"- Target: `{manifest.get('target_id', 'unknown')}`",
    f"- Workload: `{manifest.get('workload', 'unknown')}`",
    f"- Architecture: `{profile.get('architecture_class', 'unknown')}`",
    # a run that measured says so; the analysis report's status is the route policy's (no kernel compared yet)
    f"- Measurement: {measured}" if measured else f"- Analysis status: `{report.get('status', 'not_analyzed')}`",
    "",
    "## Next Step",
    "",
  ]
  lines.append(next_step(report, plan, probe, measure))  # the one ladder (report/html.py), the same words everywhere
  if roles:  # the measured per-role table, the same text the screen and the TUI print (screen.summary_text)
    lines += ["", "## Measured", "", "```", summary_text(res).rstrip(), "```"]
  if timing and not roles:  # the whole-step profile's buckets say something only before a per-role table exists
    lines += ["", "## Timing Profile", ""]
    lines.append(f"- Dominant bucket: `{timing.get('dominant_timing_bucket', 'timing_inconclusive')}`")
    for action in timing.get("next_actions", [])[:3]:
      lines.append(f"- {action}")
    role_rows = timing.get("role_timing", [])
    for row in role_rows[:8]:
      lines.append(_timing_label(row))
    if len(role_rows) > 8:
      lines.append(f"- ... {len(role_rows) - 8} more timing rows in `timing_profile.json`")
  if runner:
    lines += ["", "## Runner Handoff", ""]
    lines.append(f"- Purpose: `{runner.get('purpose', 'audit_tracer')}`")
    lines.append(f"- Runtime executor: `{runner.get('runtime_boundary', {}).get('runtime_executor', 'tinygrad')}`")
    missing = _missing_runner_outputs(primitive, timing)
    lines.append(f"- Missing returned evidence: `{', '.join(missing) if missing else 'none'}`")
  regimes = primitive.get("quant_gemv_regimes", [])
  if regimes:
    lines += ["", "## Quant GEMV Regimes", ""]
    absent = (probe or {}).get("absent") or {}
    only_missing = all(r.get("classification") == "inconclusive" and r.get("visible_bottleneck") == "missing_evidence" for r in regimes)
    if only_missing and absent:  # the probe's own record says this GPU never gives those fields: one line, not a row per role
      lines.append(f"- {len(regimes)} roles probed, none classified: the regime needs fields this GPU does not report "
                   f"({'; '.join(sorted(set(absent.values())))}). The measured table above is the per-role result.")
    else:
      for row in regimes[:12]:
        lines.append(_regime_label(row))
      if len(regimes) > 12:
        lines.append(f"- ... {len(regimes) - 12} more regimes in `primitive_profile.json`")
  lines += ["", "## Artifacts", ""]
  for artifact in manifest.get("artifacts", []):
    lines.append(f"- `{artifact}`")
  if (res.get("engine_update") or {}).get("words"):
    lines += ["", res["engine_update"]["words"]]
  return "\n".join(lines) + "\n"


def _rollback_md(policy:dict[str, Any]) -> str:
  rows = [r for r in policy.get("routes", []) if r.get("selected_route") and r.get("rollback")]
  lines = ["# Rollback", ""]
  if not rows:
    lines.append("No selected rollback commands are present in the current route policy.")
    return "\n".join(lines) + "\n"
  for row in rows:
    cmd = " ".join(f"{k}={row['rollback'][k]}" for k in sorted(row["rollback"]))
    lines += [
      f"## {row['selected_route']}",
      "",
      "```bash",
      cmd,
      "```",
      "",
    ]
  return "\n".join(lines)


def _provider_plan(manifest:dict[str, Any], providers:dict[str, Any], plan:dict[str, Any],
                   primitive:dict[str, Any], timing:dict[str, Any], runner:dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "boltbeam.provider_plan.v1",
    "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"),
    "providers": providers.get("providers", []),
    "measurement_plan": {
      "artifact": "measurement_plan.json" if plan else None,
      "phase_order": plan.get("phase_order", []),
      "provider_neutral": True,
    },
    "primitive_profile": {
      "artifact": "primitive_profile.json" if primitive else None,
      "quant_gemv_regime_count": len(primitive.get("quant_gemv_regimes", [])),
    },
    "timing_profile": {
      "artifact": "timing_profile.json" if timing else None,
      "dominant_timing_bucket": timing.get("dominant_timing_bucket"),
      "role_timing_count": len(timing.get("role_timing", [])),
      "candidate_timing_count": len(timing.get("candidate_timing", [])),
    },
    "runner_plan": {
      "artifact": "runner_plan.json" if runner else None,
      "purpose": runner.get("purpose"),
      "status": runner.get("status"),
      "bundle": runner.get("bundle", {}),
      "runtime_boundary": runner.get("runtime_boundary", {}),
      "missing_outputs": _missing_runner_outputs(primitive, timing),
    },
  }


def output_run(run:str | pathlib.Path) -> dict[str, Any]:
  from boltbeam.workflow import gameplan
  out = run_dir(run)
  manifest = load_manifest(out)
  profile = _load_optional(out, "model_profile.json")
  report = _load_optional(out, "analysis_report.json")
  plan = _load_optional(out, "measurement_plan.json")
  policy = _load_optional(out, "route_policy.json")
  providers = _load_optional(out, "provider_capabilities.json")
  primitive = _load_optional(out, "primitive_profile.json")
  timing = _load_optional(out, "timing_profile.json")
  runner = _load_optional(out, "runner_plan.json")

  provider_plan = _provider_plan(manifest, providers, plan, primitive, timing, runner)
  output_manifest = {
    "schema": SCHEMA_OUTPUT_MANIFEST,
    "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"),
    "source_run": str(out),
    "artifacts": list(OUTPUT_ARTIFACTS),
    "includes_final_policy": bool(policy),
    "includes_measurement_request": bool(plan),
    "includes_primitive_profile": bool(primitive),
    "includes_timing_profile": bool(timing),
    "includes_runner_plan": bool(runner),
  }
  write_json(out / "provider_plan.json", provider_plan)
  from boltbeam.workflow.screen import results as screen_results
  from boltbeam.workflow.report_validation import ReportInvalid
  try:
    res = screen_results(out)
  except Exception as exc:  # report boundary: invalidate stale output on malformed evidence too
    from html import escape
    message = str(exc) if isinstance(exc, ReportInvalid) else f"FAILED: report generation: {exc}"
    write_json(out / 'report_validation.json', {'status': 'failed', 'error': message})
    (out / 'summary.md').write_text(message + '\n')
    context = f"{manifest.get('model_id')} · {manifest.get('workload')} · {out.name}"
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>FAILED: report validation</title>'
        '<h1>FAILED: report does not reconcile with evidence</h1><p>' + escape(context) +
        '</p><pre style="white-space:pre-wrap">' + escape(message) + '</pre>')
    raise
  write_json(out / 'report_validation.json', res['validation'])
  import time
  from boltbeam.collectors.providers import latest_llama, run_build
  from boltbeam.workflow.screen import run_provider
  update = (_load_optional(out, "output_manifest.json").get("engine_update") or {})
  if run_provider(out) == "llama.cpp" and (update.get("build") != run_build(out) or
      str(update.get("checked_at", ""))[:10] != time.strftime("%Y-%m-%d", time.gmtime())):
    update = latest_llama(run_build(out))
  res["engine_update"] = output_manifest["engine_update"] = update if run_provider(out) == "llama.cpp" else {}
  write_json(out / "output_manifest.json", output_manifest)
  measure = _load_optional(out, "measure_status.json")
  (out / "summary.md").write_text(_summary_md(manifest, profile, report, plan, primitive, timing, runner, res, measure),
                                  encoding="utf-8")
  (out / "rollback.md").write_text(_rollback_md(policy), encoding="utf-8")
  final = update_manifest(out, stage="output", artifacts=list(OUTPUT_ARTIFACTS))
  # report.html renders the same staged facts as summary.md, standalone and self-contained (no server, no CDN).
  # It renders from `final`, not the pre-update manifest, so the pipeline rail shows the output stage as run and
  # a second `boltbeam output` over an unchanged run reproduces the file byte for byte.
  # The headline reads the speed limit and measured tokens/s from screen.results: the same function the TUI reads,
  # so the page and the TUI cannot disagree. Imported here because screen imports this module.
  (out / "report.html").write_text(
    render_run_html(manifest=final, profile=profile, report=report, plan=plan, policy=policy,
                    providers=providers, primitive=primitive, timing=timing, runner=runner,
                    source_run=out.name if (res.get('loss',{}).get('replay') or {}).get('forced') else str(out), results=screen_results(out) | {"engine_update": res["engine_update"]}, measure=measure,
                    gameplan=gameplan.MD_FILE if (out / gameplan.MD_FILE).is_file() else None), encoding="utf-8")
  return final
