"""The model and the run folder as facts for a screen: `python -m boltbeam.workflow.screen <command>`.

This is the one machine-readable seam between BoltBeam's artifacts and anything that shows them (the Go TUI in
`tui/`, or an agent). Python stays the only writer of run artifacts; readers get JSON. Every command prints one
JSON object on stdout and exits 0, or prints `{"kind": "error", "error": ...}` and exits 1. `results` exits 3
when the run holds no measurement yet; the facts are still printed, so a screen can say what is missing.

    targets                                    the registered chips and which of them carry a ceiling
    detect                                     the chip this machine is, as autoscan reads it (target_id or null)
    ceiling  MODEL | --profile P  --target T   the roofline: the best tokens/s this chip allows for this model
    runs     --root DIR                        one summary per run folder under DIR
    run      --run DIR                         the stages, what is blocked, the next step, the results
    results  --run DIR                         what won per role, the timing against the ceiling, the regimes
    pipeline MODEL --run DIR --target T        load, autoscan, analyze, [measure], [ingest-probe, ingest-timing], output;
                                               `pipeline steps: N` first, then one text line per stage, for tailing.
                                               --measure auto runs the target's BoltBeam collector here when this
                                               machine can (Metal: boltbeam/collectors/metal_native.py) and writes
                                               measure_status.json either way: measured, skipped or failed, with
                                               the reason and the command to run instead.

Nothing here computes a new kind of fact. The ceiling is `model_roofline` exactly as `roofline-theoretical`
calls it, with tokens/s read off the floor (one token at context 1 for decode; the context's tokens for
prefill). The stage order, the next-step ladder and the hot-kernel pick are the report's (`report/html.py`);
the stages `pipeline` runs are the workflow's own functions. The contract is pinned by `tui/testdata`: the
fixture run folders and the expected JSON, checked by `tests/test_workflow_screen.py` here and by
`tui/internal/seam` on the Go side.
"""
from __future__ import annotations

import argparse
import html as html_text
import json
import pathlib
import re
import sys
from typing import Any

from boltbeam.cli._common import _parse_ctxs
from boltbeam.cli.roofline import resolve_peak_flops
from boltbeam.core.canonical import pretty_json
from boltbeam.kernel_analysis.theoretical_roofline import model_roofline
from boltbeam.profile.loaders import profile_from_model
from boltbeam.report.html import STAGES, next_step, roofline_kernels
from boltbeam.target.targets import get_target, load_target_registry
from boltbeam.workflow import analyze_run, autoscan_run, ingest_probe_run, ingest_timing_run, load_run, output_run
from boltbeam.workflow.autoscan import _hardware_profile
from boltbeam.workflow.common import load_manifest, read_json

SCHEMA = "boltbeam.tui.v1"
# what a run still needs, read off measurement_plan.json: (plan block, the evidence it asks for, its request file)
NEEDS = (("primitive_profile", "probe_evidence", "probe_request.json"),
         ("timing_profile", "timing_trace", "trace_request.json"))


MEASURE_STATUS = "measure_status.json"
SCHEMA_MEASURE_STATUS = "boltbeam.measure_status.v1"


class Refused(Exception):
  """A command that cannot answer: an unknown target, a folder that is not a run, a chip with no ceiling."""


def _optional(run:pathlib.Path, name:str) -> dict[str, Any]:
  p = run / name
  return read_json(p) if p.exists() else {}


# --- targets and the ceiling ------------------------------------------------------------------------------

def target_facts(target) -> dict[str, Any]:
  caps = target.capabilities or {}
  status = caps.get("fact_status", {}) or {}
  sources = caps.get("fact_sources", {}) or {}
  return {
    "id": target.target_id,
    "backend": target.backend,
    "backend_status": target.backend_status,
    "scope": (sources.get("memory_bandwidth_gbs") or {}).get("scope"),
    "memory_bandwidth_gbs": target.memory_bandwidth_gbs,
    "peak_tflops": dict(target.peak_tflops or {}),
    "matrix_tflops": dict(target.matrix_tflops or {}),
    "fact_status": {"memory_bandwidth_gbs": status.get("memory_bandwidth_gbs", "unknown"),
                    "peak_tflops": status.get("peak_tflops", "unknown")},
    "has_ceiling": bool(target.memory_bandwidth_gbs) and bool(target.matrix_tflops or target.peak_tflops),
  }


def targets() -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "targets", "targets": [target_facts(t) for t in load_target_registry().values()]}


def detect(profile:dict[str, Any] | None = None) -> dict[str, Any]:
  """The chip this machine is: autoscan's own GPU probe, cut to what a screen shows. No probe, no guess: null."""
  gpu = (profile or _hardware_profile())["gpu"]
  target_id = gpu.get("target_id")
  return {"schema": SCHEMA, "kind": "detect", "status": gpu.get("status"), "name": gpu.get("name"),
          "target_id": target_id, "target_kind": gpu.get("target_kind"),
          "registered": bool(target_id) and target_id in {t["id"] for t in targets()["targets"]}}


def _roles(report:dict[str, Any]) -> list[dict[str, Any]]:
  return [{"role": r["role"], "quant": r["quant"], "shape": r["shape"], "bytes_moved": r["bytes_moved"],
           "floor_ms": r["floor_ms"], "share": r["whole_share"], "regime": r["regime"]} for r in report["roles"]]


def _block(report:dict[str, Any], context:int) -> dict[str, Any]:
  whole = report["whole"]
  floor_ms = whole["floor_ms"]
  return {"context": context, "bytes_moved": whole["bytes_moved"], "floor_ms": floor_ms,
          "tok_s": (context * 1000.0 / floor_ms) if floor_ms > 0 else None, "regime": whole["regime"],
          "roles": _roles(report)}


def ceiling(profile:dict[str, Any], target, *, context:int = 512, dtype:str = "fp16",
            peak_gbs:float | None = None, peak_tflops:float | None = None) -> dict[str, Any]:
  """The roofline for one model on one chip: decode (one token) and prefill (the context), with tokens/s."""
  try:
    peak_flops = resolve_peak_flops(target, dtype, peak_tflops)
  except SystemExit as exc:
    raise Refused(str(exc)) from exc
  bw = target.memory_bandwidth_gbs or peak_gbs
  if not bw:
    raise Refused(f"target {target.target_id!r} carries no memory_bandwidth_gbs, so there is no memory ceiling; "
                  "pass --peak-gbs or add a measured figure to boltbeam/data/targets.json")
  decode = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=1)
  prefill = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=context)
  return {
    "schema": SCHEMA, "kind": "ceiling", "model_id": profile["model_id"], "target": target_facts(target),
    "peak_bandwidth_gbs": bw, "peak_tflops": peak_flops / 1e12, "truth_status": decode["truth_status"],
    "ridge_intensity": decode["ridge_intensity"], "assumptions": decode["assumptions"],
    "decode": _block(decode, 1), "prefill": _block(prefill, context),
  }


# --- run folders -----------------------------------------------------------------------------------------------

def stage_rows(manifest:dict[str, Any]) -> list[dict[str, Any]]:
  stages = manifest.get("stages", {}) or {}
  return [{"key": key, "label": label, "note": note, "done": key in stages,
           "artifacts": list((stages.get(key) or {}).get("artifacts", []) or [])} for key, label, note in STAGES]


def blocked(plan:dict[str, Any]) -> list[dict[str, Any]]:
  return [{"need": need, "request": request} for block, need, request in NEEDS
          if (plan.get(block) or {}).get("status") == "requested"]


def summary(run:pathlib.Path) -> dict[str, Any]:
  manifest = load_manifest(run)
  report = _optional(run, "analysis_report.json")
  plan = _optional(run, "measurement_plan.json")
  return {
    "id": run.name, "model_id": manifest.get("model_id"), "model_format": manifest.get("model_format"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "latest_stage": manifest.get("latest_stage"), "status": report.get("status", "not_analyzed"),
    "stages": stage_rows(manifest), "blocked": blocked(plan),
    "measured": {"probe": (run / "primitive_profile.json").exists(), "timing": (run / "timing_profile.json").exists()},
    "report": "report.html" if (run / "report.html").exists() else None,
    "measure": _optional(run, MEASURE_STATUS) or None,
  }


def is_run(path:pathlib.Path) -> bool:
  return path.is_dir() and (path / "run_manifest.json").exists()


def runs(root:pathlib.Path) -> dict[str, Any]:
  if not root.is_dir():
    raise Refused(f"{root} is not a folder")
  return {"schema": SCHEMA, "kind": "runs", "root": str(root),
          "runs": [summary(p) for p in sorted(root.iterdir()) if is_run(p)]}


def _measured_vs_ceiling(manifest:dict[str, Any], profile:dict[str, Any]) -> dict[str, Any]:
  """The modeled ceiling for this run's model and chip, or the reason there is none. Never a guess."""
  if not profile:
    return {"status": "absent", "reason": "no model_profile.json in the run"}
  try:
    c = ceiling(profile, get_target(manifest.get("target_id")))
  except (Refused, SystemExit) as exc:
    return {"status": "absent", "reason": str(exc)}
  block = c["decode"] if manifest.get("workload") == "decode" else c["prefill"]
  return {"status": "modeled", "context": block["context"], "tok_s": block["tok_s"], "floor_ms": block["floor_ms"],
          "peak_bandwidth_gbs": c["peak_bandwidth_gbs"]}


def results(run:pathlib.Path) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = _optional(run, "model_profile.json")
  policy = _optional(run, "route_policy.json")
  plan = _optional(run, "measurement_plan.json")
  primitive = _optional(run, "primitive_profile.json")
  timing = _optional(run, "timing_profile.json")
  kernels, context = roofline_kernels(timing) if timing else ([], None)
  chosen = next((s for s in timing.get("context_summaries", []) if s.get("context") == context), {}) if timing else {}
  return {
    "schema": SCHEMA, "kind": "results", "id": run.name, "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "measured": bool(timing) or bool(primitive),
    "ceiling": _measured_vs_ceiling(manifest, profile),
    "routes": [{"role": r.get("role"), "quant": r.get("quant"), "shape": r.get("shape"),
                "selected_route": r.get("selected_route"), "status": r.get("status"),
                "candidates": list(r.get("candidates", []) or []), "evidence_refs": list(r.get("evidence_refs", []) or [])}
               for r in policy.get("routes", []) or []],
    "timing": {
      "status": "classified" if timing else "absent",
      "dominant_bucket": timing.get("dominant_timing_bucket"),
      "next_actions": list(timing.get("next_actions", []) or []),
      "context": context, "total_us": chosen.get("total_us"), "tok_s": chosen.get("tok_s"),
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
    "blocked": blocked(plan),
    "report": "report.html" if (run / "report.html").exists() else None,
  }


def show(run:pathlib.Path) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = _optional(run, "model_profile.json")
  report = _optional(run, "analysis_report.json")
  plan = _optional(run, "measurement_plan.json")
  meta = profile.get("metadata", {}) or {}
  out = summary(run)
  out.update({
    "schema": SCHEMA, "kind": "run", "model_path": manifest.get("model_path"),
    "model": {"architecture": profile.get("architecture"), "architecture_class": profile.get("architecture_class"),
              "layer_count": profile.get("layer_count"), "hidden_size": profile.get("hidden_size"),
              "ffn_size": profile.get("ffn_size"), "vocab_size": profile.get("vocab_size"),
              "role_count": len(profile.get("roles", []) or []), "quant_types": list(meta.get("quant_types", []) or [])},
    # the report's own ladder, text only: the same sentence report.html and summary.md print
    "next_step": html_text.unescape(re.sub(r"<[^>]+>", "", next_step(report, plan))),
    "artifacts": list(manifest.get("artifacts", []) or []),
    "results": results(run),
  })
  return out


# --- the pipeline --------------------------------------------------------------------------------------------

def _write_measure_status(run:str | pathlib.Path, status:str, *, collector:str | None = None, reason:str | None = None,
                          command:str | None = None) -> None:
  from boltbeam.workflow.common import run_dir, update_manifest, write_json
  out = run_dir(run)
  write_json(out / MEASURE_STATUS, {"schema": SCHEMA_MEASURE_STATUS, "status": status, "collector": collector,
                                    "reason": reason, "command": command})
  update_manifest(out, stage="measure", artifacts=[MEASURE_STATUS])


def measure_plan(target_id:str, model:str, run:str) -> dict[str, Any]:
  """Which BoltBeam collector measures this target on this machine, or why none can. Decided before the run starts,
  so the step count is right from the first line."""
  from boltbeam.profiler.capabilities import resolve_profiler_capability
  try:
    cap = resolve_profiler_capability("boltbeam", target_id)
  except ValueError:
    return {"collector": None, "reason": f"BoltBeam has no collector of its own for {target_id}",
            "command": f"on a machine with {target_id}: boltbeam collect-hw-trace --provider llama --run {run}, "
                       "then pipeline --timing <run>/timing_trace.json"}
  from boltbeam.collectors.metal_native import CannotMeasure, preflight
  try:
    preflight(target_id, model, run=run)
  except CannotMeasure as exc:
    return {"collector": cap.collector_id, "reason": exc.reason, "command": exc.command}
  return {"collector": cap.collector_id, "reason": None, "command": None}


def _measure_steps(args, plan:dict[str, Any]) -> list[tuple[str, Any]]:
  if plan["reason"]:
    return [("measure", lambda: _write_measure_status(args.run, "skipped", collector=plan["collector"],
                                                       reason=plan["reason"], command=plan["command"]))]
  from boltbeam.collectors import metal_native
  run = pathlib.Path(args.run)
  args.probe, args.timing = str(run / "probe_evidence.json"), str(run / "timing_trace.json")

  def guarded(only:str):
    def step():
      try:
        metal_native.measure(metal_native.run_dir(args.run), only=only)
      except Exception as exc:  # the reason must outlive the log: a reopened screen reads it from the run
        _write_measure_status(args.run, "failed", collector=plan["collector"], reason=f"{only}: {exc}",
                              command=f"python -m boltbeam.collectors.metal_native --run {args.run}")
        raise
      if only == "timing":
        _write_measure_status(args.run, "measured", collector=plan["collector"])
    return step
  return [("measure_probe", guarded("probe")), ("measure_timing", guarded("timing"))]


def pipeline(args, out=sys.stdout) -> int:
  """Run the stages in order and say so, one line each. The run folder keeps whatever landed before a failure."""
  def say(text:str) -> None:
    out.write(text + "\n")
    out.flush()
  steps = [
    ("load", lambda: load_run(args.model, args.run, target_id=args.target, model_id=args.id, workload=args.workload,
                              contexts=_parse_ctxs(args.ctxs), target_source="user")),
    ("autoscan", lambda: autoscan_run(args.run, providers=())),
    ("analyze", lambda: analyze_run(args.run)),
  ]
  if getattr(args, "measure", "none") == "auto" and not (args.probe or args.timing):
    steps += _measure_steps(args, measure_plan(args.target, str(pathlib.Path(args.model).expanduser().resolve()), args.run))
  if args.probe:
    steps.append(("ingest_probe", lambda: ingest_probe_run(args.run, args.probe)))
  if args.timing:
    steps.append(("ingest_timing", lambda: ingest_timing_run(args.run, args.timing)))
  if args.probe or args.timing:
    steps.append(("analyze", lambda: analyze_run(args.run)))  # the plan and the report read the new evidence
  steps.append(("output", lambda: output_run(args.run)))
  say(f"pipeline steps: {len(steps)}")  # a screen draws n of N from this line
  for key, step in steps:
    say(f"stage {key}: start")
    try:
      step()
    except Exception as exc:  # application boundary: the stage's own message is the fact the reader needs
      say(f"stage {key}: failed: {exc}")
      return 1
    say(f"stage {key}: done")
  say(f"pipeline done: {args.run}")
  return 0


# --- the command line ------------------------------------------------------------------------------------------

def _profile_arg(args) -> dict[str, Any]:
  if args.profile:
    return read_json(args.profile)
  if not args.model:
    raise Refused("ceiling needs MODEL or --profile")
  return profile_from_model(args.model, args.id).to_json()


def _target_arg(name:str):
  try:
    return get_target(name)
  except SystemExit as exc:
    raise Refused(str(exc)) from exc


def main(argv:list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="python -m boltbeam.workflow.screen", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  sub.add_parser("targets")
  sub.add_parser("detect")
  p = sub.add_parser("ceiling")
  p.add_argument("model", nargs="?")
  p.add_argument("--profile", help="a model_profile.json instead of the model file")
  p.add_argument("--target", required=True)
  p.add_argument("--id", default=None)
  p.add_argument("--context", type=int, default=512, help="prefill token dimension (decode is always 1)")
  p.add_argument("--dtype", default="fp16")
  p.add_argument("--peak-gbs", type=float, default=None)
  p.add_argument("--peak-tflops", type=float, default=None)
  p = sub.add_parser("runs")
  p.add_argument("--root", required=True)
  for name in ("run", "results"):
    p = sub.add_parser(name)
    p.add_argument("--run", required=True)
  p = sub.add_parser("pipeline")
  p.add_argument("model")
  p.add_argument("--run", required=True)
  p.add_argument("--target", required=True)
  p.add_argument("--id", default=None)
  p.add_argument("--workload", default="decode", choices=["decode", "prefill"])
  p.add_argument("--ctxs", default="128,512")
  p.add_argument("--probe", default=None, help="boltbeam.probe_evidence.v1 JSON to ingest after analyze")
  p.add_argument("--timing", default=None, help="boltbeam.timing_trace.v1 JSON to ingest after analyze")
  p.add_argument("--measure", default="none", choices=["none", "auto"],
                 help="auto: measure with the target's BoltBeam collector when this machine can, then ingest")
  args = parser.parse_args(argv)
  if args.command == "pipeline":
    return pipeline(args)
  code = 0
  try:
    if args.command == "targets":
      out = targets()
    elif args.command == "detect":
      out = detect()
    elif args.command == "ceiling":
      out = ceiling(_profile_arg(args), _target_arg(args.target), context=args.context, dtype=args.dtype,
                    peak_gbs=args.peak_gbs, peak_tflops=args.peak_tflops)
    elif args.command == "runs":
      out = runs(pathlib.Path(args.root).expanduser())
    elif args.command == "run":
      out = show(pathlib.Path(args.run).expanduser())
    else:
      out = results(pathlib.Path(args.run).expanduser())
      code = 0 if out["measured"] else 3
  except (Refused, FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
    sys.stdout.write(pretty_json({"schema": SCHEMA, "kind": "error", "error": str(exc)}))
    return 1
  sys.stdout.write(pretty_json(out))
  return code


if __name__ == "__main__":
  raise SystemExit(main())
