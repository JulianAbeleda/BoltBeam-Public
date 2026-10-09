"""The model and the run folder as facts for a screen: `python -m boltbeam.workflow.screen <command>`.

This is the one machine-readable seam between BoltBeam's artifacts and anything that shows them (the Go TUI in
`tui/`, or an agent). Python stays the only writer of run artifacts; readers get JSON. Every command prints one
JSON object on stdout and exits 0, or prints `{"kind": "error", "error": ...}` and exits 1. `results` exits 3
when the run holds no measurement yet; the facts are still printed, so a screen can say what is missing.

    targets                                    the registered chips and which of them carry a ceiling
    chips                                      every chip in Setup's groups: this machine, measured, not measured
                                               yet, families (workflow/chips.py)
    autoscan [--remeasure]                     use the profile that fits this machine's GPU, or measure and save
                                               a new one; --remeasure refreshes a profile made here
    detect                                     the chip this machine is, as autoscan reads it (target_id or null)
    ceiling  MODEL | --profile P  --target T   the roofline: the best tokens/s this chip allows for this model
    ceilings MODEL | --profile P               the decode limit on every registered chip with a ceiling (what-if)
    runs     --root DIR [--all]                one summary per run folder under DIR; --all adds DIR/.work and the
                                               saved runs, each row marked where: runs, work or saved
    run      --run DIR | --run ID --root DIR   the stages, what is blocked, the next step, the results; an id is
                                               looked up in DIR/ID, DIR/.work/ID, then SAVED/ID (find_run)
    delete   --run DIR --root ROOT             remove one run folder; refused unless DIR is a run directly under ROOT
    save     --run DIR --to SAVED              export a run: the run, results.json, report.html, summary.txt
    saved    --root SAVED                      the saved runs, newest first, with the share of the limit reached
    clean-work --work DIR --before ISO         remove temporary runs started before ISO (never --keep ones)
    results  --run DIR | --run ID --root DIR   what won per role, the timing against the ceiling, the regimes
    pipeline MODEL --run DIR --target T        load, autoscan, analyze, [measure], [ingest-probe, ingest-timing], output;
                                               `pipeline steps: N` first, then one text line per stage, for tailing.
                                               --measure auto runs the target's BoltBeam collector here when this
                                               machine can (Metal: collectors/metal_native.py; NVIDIA: the
                                               llama-bench decode, collectors/llama_bench_decode.py) and writes
                                               measure_status.json either way: measured, skipped or failed, with
                                               the reason and the command to run instead. --provider llama.cpp
                                               (default) or tinygrad picks the runtime; a busy GPU is refused.
    providers --target T                       the runtimes that can measure here (llama.cpp, tinygrad), each with
                                               the reason when it cannot, and how step 5 times its roles
    gpu-free --target T                        is the GPU free, or which program holds it
    role-time --run DIR [--provider P]         time every role inside a real decode with the run's provider
                                               (collectors/providers.py); feeds `results.loss`
    compare-ready --run DIR [--tinygrad-root R]  can this machine compare kernels for the run: the missing piece
                                               (fork, venv, numpy, model, backend) and the one command that fixes it
    compare  --run DIR [--tinygrad-root R]     compare kernels per role on Metal (boltbeam/search/role_compare.py):
                                               `compare roles: N`, then `role ROLE QUANT: search|ab|done STATUS`,
                                               then `compare done: DIR` or `compare failed: REASON`. It writes the
                                               statuses into route_policy.json and refreshes the report.

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
import shutil
import sys
from typing import Any

from boltbeam.cli._common import _parse_ctxs
from boltbeam.cli.roofline import peak_flops_status, resolve_peak_flops
from boltbeam.core.canonical import pretty_json
from boltbeam.kernel_analysis.theoretical_roofline import model_roofline
from boltbeam.profile.loaders import profile_from_model
from boltbeam.report.html import STAGES, next_step, roofline_kernels, stage_state
from boltbeam.search import role_compare
from boltbeam.collectors import providers, tinygrad_role_time
from boltbeam.target import targets as reg
from boltbeam.target.targets import get_target
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
    # registry: the repo's reviewed rows; generated: a profile measured on this machine (workflow/chips.py)
    "source": "generated" if reg.is_local(target.target_id) else "registry",
    "backend": target.backend,
    "backend_status": target.backend_status,
    "scope": (sources.get("memory_bandwidth_gbs") or {}).get("scope"),
    "memory_bandwidth_gbs": target.memory_bandwidth_gbs,
    "peak_tflops": dict(target.peak_tflops or {}),
    "matrix_tflops": dict(target.matrix_tflops or {}),
    # peak_tflops: the status of the compute figure the ceiling uses (cli/roofline.py peak_flops_fact), so the chip
    # and the speed limit never disagree about where compute came from
    "fact_status": {"memory_bandwidth_gbs": status.get("memory_bandwidth_gbs", "unknown"),
                    "peak_tflops": peak_flops_status(target, "fp16")},
    # the scan and the date behind the memory figure; a recorded fact, not this machine read live
    "scope_observed_at": (sources.get("memory_bandwidth_gbs") or {}).get("observed_at"),
    "has_ceiling": bool(target.memory_bandwidth_gbs) and bool(target.matrix_tflops or target.peak_tflops),
  }


def targets() -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "targets", "targets": [target_facts(t) for t in reg.TARGETS.values()]}


def detect(profile:dict[str, Any] | None = None) -> dict[str, Any]:
  """The chip this machine is: autoscan's own GPU probe, cut to what a screen shows. No probe, no guess: null."""
  from boltbeam.workflow import layout as lay
  gpu = (profile or _hardware_profile())["gpu"]
  target_id = gpu.get("target_id")
  devs = lay.devices(gpu)
  return {"schema": SCHEMA, "kind": "detect", "status": gpu.get("status"), "name": gpu.get("name"),
          # every GPU the probe found; more than one is limited support (workflow/layout.py)
          "gpus": [{"index": d["index"], "name": d["name"], "target_id": d["target_id"]} for d in devs],
          "gpu_count": len(devs), "multi_gpu": lay.summary(devs),
          "target_id": target_id, "target_kind": gpu.get("target_kind"),
          # read live from nvidia-smi on every call; the registry's driver is the one its facts were measured on
          "driver_version": gpu.get("driver_version"),
          # known: a registry row or a profile made here fits; new: autoscan would measure one (workflow/chips.py)
          "profile": gpu.get("profile"),
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
            peak_gbs:float | None = None, peak_tflops:float | None = None,
            read:dict[str, Any] | None = None) -> dict[str, Any]:
  """The roofline for one model on one chip: decode (one token) and prefill (the context), with tokens/s.
  read is this machine's one-GPU read bandwidth (layout.read_bandwidth): when given it is the memory speed, so the
  limit, the tie-out and the per-role rule use the same number, and bandwidth_source says where it came from."""
  try:
    peak_flops = resolve_peak_flops(target, dtype, peak_tflops)
  except SystemExit as exc:
    raise Refused(str(exc)) from exc
  if read is None and (known := profile_read({"target_id": target.target_id})):
    from boltbeam.workflow import layout as lay
    when = ((target.capabilities.get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}).get("observed_at")
    read = {"gbs": known[0], "source": known[1], "short": lay.short_source(known[1], when) + ", chip profile"}
  bw = (read or {}).get("gbs") or target.memory_bandwidth_gbs or peak_gbs
  if not bw:
    raise Refused(f"target {target.target_id!r} carries no memory_bandwidth_gbs, so there is no memory ceiling; "
                  "pass --peak-gbs or add a measured figure to boltbeam/data/targets.json")
  decode = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=1)
  prefill = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=context)
  return {
    "schema": SCHEMA, "kind": "ceiling", "model_id": profile["model_id"], "target": target_facts(target),
    "peak_bandwidth_gbs": bw, "bandwidth_source": (read or {}).get("short"), "peak_tflops": peak_flops / 1e12, "truth_status": decode["truth_status"],
    "ridge_intensity": decode["ridge_intensity"], "assumptions": decode["assumptions"],
    "decode": _block(decode, 1), "prefill": _block(prefill, context),
  }


def ceilings(profile:dict[str, Any]) -> dict[str, Any]:
  """The decode limit of this model on every registered chip that carries a ceiling: read-only what-if facts."""
  chips = []
  for t in reg.TARGETS.values():
    if not target_facts(t)["has_ceiling"]:
      continue
    try:
      c = ceiling(profile, t)
    except (Refused, SystemExit):
      continue
    chips.append({"id": t.target_id, "tok_s": c["decode"]["tok_s"], "floor_ms": c["decode"]["floor_ms"],
                  "peak_bandwidth_gbs": c["peak_bandwidth_gbs"]})
  return {"schema": SCHEMA, "kind": "ceilings", "model_id": profile["model_id"], "chips": chips}


# --- run folders -----------------------------------------------------------------------------------------------

def stage_rows(manifest:dict[str, Any], measure:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """state: done, not_needed (state_note says why) or open (report/html.py stage_state decides)."""
  stages = manifest.get("stages", {}) or {}
  rows = []
  for key, label, note in STAGES:
    state, why = stage_state(key, stages.get(key), measure)
    rows.append({"key": key, "label": label, "note": note, "done": key in stages, "state": state, "state_note": why,
                 "artifacts": list((stages.get(key) or {}).get("artifacts", []) or [])})
  return rows


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
    "stages": stage_rows(manifest, _optional(run, MEASURE_STATUS)), "blocked": blocked(plan),
    "measured": {"probe": (run / "primitive_profile.json").exists(), "timing": (run / "timing_profile.json").exists()},
    "report": "report.html" if (run / "report.html").exists() else None,
    "measure": _optional(run, MEASURE_STATUS) or None,
  }


def is_run(path:pathlib.Path) -> bool:
  return path.is_dir() and (path / "run_manifest.json").exists()


def delete(root:pathlib.Path, run:pathlib.Path) -> dict[str, Any]:
  """Remove one run folder. Only a folder holding run_manifest.json directly under the runs root qualifies."""
  root, run = root.resolve(), run.resolve()
  if run.parent != root:
    raise Refused(f"{run} is not directly under the runs folder {root}")
  if not is_run(run):
    raise Refused(f"{run} is not a run folder (no run_manifest.json)")
  shutil.rmtree(run)
  return {"schema": SCHEMA, "kind": "deleted", "root": str(root), "run": run.name}


WORK = ".work"  # where the screens keep a run until it is saved
SAVED = "saved"  # where a saved run lives (the screens' default --saved)


def run_places(root:pathlib.Path, saved:pathlib.Path | None = None) -> list[tuple[str, pathlib.Path]]:
  """Where a run id is looked up, in this order: <root>/<id> (runs made by --json start and older layouts),
  <root>/.work/<id> (the screens' unsaved runs), <saved>/<id> (saved runs; default <root>/saved)."""
  return [("runs", root), ("work", root / WORK), ("saved", saved if saved is not None else root / SAVED)]


def find_run(root:pathlib.Path, run_id:str, saved:pathlib.Path | None = None) -> pathlib.Path:
  """The run folder for an id: the first of run_places that holds it. Refused, naming every place, when none does."""
  if not run_id or run_id != pathlib.Path(run_id).name or run_id.startswith("."):
    raise Refused(f"run id must be a folder name: {run_id!r}")
  places = run_places(root, saved)
  for _, folder in places:
    if is_run(folder / run_id):
      return folder / run_id
  raise Refused(f"no run {run_id} in " + ", ".join(str(f / run_id) for _, f in places))


def runs(root:pathlib.Path, saved:pathlib.Path | None = None, *, all_places:bool = False) -> dict[str, Any]:
  """One summary per run under root. A root that does not exist yet holds no runs: that is a fact, not an error.
  This read never creates it; the first pipeline run does (workflow/common.py run_dir). all_places also lists the
  screens' work area and the saved runs (run_places), each row marked where it lives: runs, work or saved."""
  if root.exists() and not root.is_dir():
    raise Refused(f"{root} is not a folder")
  places = run_places(root, saved) if all_places else [("runs", root)]
  rows = []
  for where, folder in places:
    if folder.is_dir():
      rows += [{**summary(p), "where": where, "dir": str(p)} for p in sorted(folder.iterdir()) if is_run(p)]
  return {"schema": SCHEMA, "kind": "runs", "root": str(root), "runs": rows}


def run_read(run:pathlib.Path, target_id:str | None) -> dict[str, Any] | None:
  """The run's one-GPU read bandwidth from its machine facts (layout.read_bandwidth), when its GPU is the run's chip."""
  from boltbeam.workflow import layout as lay
  got = lay.read_bandwidth(_optional(run, lay.MACHINE), _optional(run, MEASURE_STATUS).get("layout") or "one")
  return got if got and got.get("target_id") in (None, target_id) else None


def _measured_vs_ceiling(manifest:dict[str, Any], profile:dict[str, Any], run:pathlib.Path | None = None) -> dict[str, Any]:
  """The modeled ceiling for this run's model and chip, or the reason there is none. Never a guess. With machine
  facts in the run, the memory speed is the one measured on this GPU (or the registry's, labelled)."""
  if not profile:
    return {"status": "absent", "reason": "no model_profile.json in the run"}
  read = run_read(run, manifest.get("target_id")) if run is not None else None
  try:
    c = ceiling(profile, get_target(manifest.get("target_id")), read=read)
  except (Refused, SystemExit) as exc:
    return {"status": "absent", "reason": str(exc)}
  block = c["decode"] if manifest.get("workload") == "decode" else c["prefill"]
  return {"status": "modeled", "context": block["context"], "tok_s": block["tok_s"], "floor_ms": block["floor_ms"],
          "bytes_moved": block["bytes_moved"],
          "peak_bandwidth_gbs": c["peak_bandwidth_gbs"], "bandwidth_source": c["bandwidth_source"],
          "_roles": block["roles"]}


def run_provider(run:pathlib.Path) -> str:
  """The provider step 4 measured this run with. Runs from before providers were measured with llama-bench."""
  return (_optional(run, MEASURE_STATUS).get("provider")) or providers.DEFAULT


def _provider_table(run:pathlib.Path, provider:str, ceil:dict[str, Any], limit_ms:float) -> tuple[dict | None, dict]:
  """One provider's per-role table (or None) and the capture its trace names."""
  trace = _optional(run, providers.TRACES[provider])
  if not trace:
    return None, {}
  capture = trace.get("capture") or {"method": tinygrad_role_time.OWN_TIMING, "reason": None}
  return tinygrad_role_time.loss(ceil.get("_roles") or [], trace, limit_ms), capture


def loss_block(run:pathlib.Path, manifest:dict[str, Any], ceil:dict[str, Any], measured_tok_s:Any, *,
               beside:bool = False) -> dict[str, Any]:
  """The end result for the provider step 4 measured with: its speed against the limit, ms per token lost, and per
  role where it loses it. Another provider's per-role table, when the run holds one, is shown beside it, labelled.
  The floor rule (tinygrad_role_time.refusal) holds for every provider's table."""
  provider = run_provider(run)
  target = get_target(manifest.get("target_id"))
  # facts of the run only: what this machine could capture is `providers`, never part of the pinned results
  plan = None
  if ceil.get("status") != "modeled" or not ceil.get("floor_ms"):
    return {"status": "absent", "reason": ceil.get("reason"), "runtimes": [], "roles": [], "provider": provider,
            "capture": plan, "others": []}
  limit_ms = ceil["floor_ms"]
  runtimes = []
  if isinstance(measured_tok_s, (int, float)) and measured_tok_s > 0:
    ms = 1000.0 / measured_tok_s
    runtimes.append({"provider": provider, "tok_s": measured_tok_s, "ms": ms, "lost_ms": ms - limit_ms,
                     "per_role": False, "note": "whole step, measured untraced"})
  out = {"status": "modeled", "limit_tok_s": ceil["tok_s"], "limit_ms": limit_ms, "runtimes": runtimes, "roles": [],
         "not_attributed_ms": None, "source": None, "missing": None, "refused": None,
         "provider": provider, "capture": plan, "roles_provider": None, "provider_missing": None, "others": [], "unpaired_roles": [],
         "tie_out": None, "role_rule": None}
  from boltbeam.workflow import tie_out as tie
  profile, bw = _optional(run, "model_profile.json"), ceil.get("peak_bandwidth_gbs")
  out["layout"], out["machine"] = _layout_limit(run, profile, ceil)
  if out["layout"] and out["layout"].get("ms") and out["layout"]["gpus"] > 1:
    limit_ms = out["layout"]["ms"]  # the layout's limit is the one the measurement is compared with
    out.update(limit_ms=limit_ms, limit_tok_s=1000.0 / limit_ms)
  tables = {name: _provider_table(run, name, ceil, limit_ms) for name in providers.NAMES}
  for name in providers.NAMES:
    table, capture = tables[name]
    if table is None or table["status"] != "measured":
      continue
    runtimes.append({"provider": name, "tok_s": table["tok_s"], "ms": table["kernel_ms"],
                     "lost_ms": table["kernel_ms"] - limit_ms, "per_role": True, "capture": capture.get("method"),
                     "note": "GPU kernel time per token, " + _capture_words(capture.get("method"))})
    if name != provider:
      out["others"].append({"provider": name, "capture": capture, "tok_s": table["tok_s"], "ms": table["kernel_ms"],
                            "roles": table["roles"], "not_attributed_ms": table["not_attributed_ms"]})
  if beside:
    out["others"] += sibling_runs(run, manifest, provider)
  # the chosen provider's table; before any, an earlier run's other table keeps `roles` as it was, labelled
  order = [provider] + [n for n in providers.NAMES if n != provider]
  shown = next((name for name in order if tables[name][0] is not None), None)
  if tables[provider][0] is None:  # said beside the other provider's table, or alone when there is none
    out["provider_missing"] = _missing(provider, target, manifest)
  if shown is None:
    out["missing"] = out["provider_missing"]
    out["tie_out"] = tie.tie_out(run, provider=provider, table=None, trace=None, limit_ms=limit_ms, profile=profile,
                                 bandwidth_gbs=bw, missing=out["missing"])
    return out
  table, capture = tables[shown]
  out["roles_provider"] = shown
  if table["status"] != "measured":  # a floor broken: the reason is shown, the numbers never are
    out["refused"] = table["reason"]
    return out
  roles, rule = tie.role_why(table["roles"], bw)
  out.update(roles=roles, role_rule=rule, not_attributed_ms=table["not_attributed_ms"], capture=capture,
             unpaired_roles=table.get("unpaired_roles") or [], source=_source(shown, capture, target))
  out["tie_out"] = tie.tie_out(run, provider=shown, table=table, trace=_optional(run, providers.TRACES[shown]),
                               limit_ms=limit_ms, profile=profile, bandwidth_gbs=bw,
                               missing=None if roles else "the capture could not split the kernels by role")
  return out


def _layout_limit(run:pathlib.Path, profile:dict[str, Any], ceil:dict[str, Any]) -> tuple[dict | None, dict | None]:
  """The layout the run was measured with and its derived limit, and the machine facts behind it; None, None
  for a run without machine facts (single-GPU runs keep the registry limit either way)."""
  from boltbeam.workflow import layout as lay
  facts = _optional(run, lay.MACHINE)
  if not facts:
    return None, None
  status = _optional(run, MEASURE_STATUS)
  layout = status.get("layout") or "one"
  got = lay.limit(layout, bytes_per_token=ceil.get("bytes_moved") or 0, facts=facts,
                  hidden_size=profile.get("hidden_size"), layers=profile.get("layer_count"))
  shown = {"gpus": [{k: g.get(k) for k in ("index", "name", "target_id", "read_gbs", "read_source")} for g in facts["gpus"]],
           "pairs": facts.get("pairs", []), "measured_at": facts.get("measured_at"),
           "support": lay.LIMITED if len(facts["gpus"]) > 1 else None}
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


def not_measured(run:pathlib.Path) -> str:
  """Why a run holds no speed, from its measure status: "not measured: <reason>"."""
  status = _optional(run, MEASURE_STATUS)
  reason = status.get("reason") or (f"the measure step ended {status['status']}" if status.get("status") else
                                     "the run never reached the measure step")
  return "not measured: " + str(reason)


CAPTURE_WORDS = {tinygrad_role_time.OWN_TIMING: "tinygrad's own timing", "nsys": "captured with nsys",
                 "rocprofv3": "captured with rocprofv3", "metal-system-trace": "captured with Metal System Trace"}


def _capture_words(method:str | None) -> str:
  return CAPTURE_WORDS.get(method or "", method or "no capture")


def _missing(provider:str, target, manifest:dict[str, Any]) -> str:
  if provider == tinygrad_role_time.PROVIDER and not tinygrad_role_time.device_for(target):
    return f"{manifest.get('target_id')} names no tinygrad device, so no per-role time can be taken here"
  return f"no per-role time yet: Run with {provider} times each role"


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


def compare_ready(run:pathlib.Path, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Whether step 5 can compare kernels for this run on this machine: Metal runs only, with the fork, its venv and
  the model. Kept out of `results`, whose pinned JSON must not depend on the machine."""
  manifest, policy = load_manifest(run), _optional(run, "route_policy.json")
  out = {"schema": SCHEMA, "kind": "compare_ready", "id": run.name}
  backend = str((policy.get("target") or {}).get("backend") or "")
  try:  # the runtime role times come from, named once (tinygrad_role_time.runtime_name)
    out["runtime"] = tinygrad_role_time.runtime_name(get_target(manifest.get("target_id")))
  except SystemExit:
    out["runtime"] = None
  ready = role_compare.readiness(root, model=manifest.get("model_path"))
  if backend.lower() not in role_compare.COMPARE_BACKENDS:
    # timing each role works on any tinygrad device; the kernel search provider runs on Metal only
    return out | ready | {"applies": False,
                          "compare_message": f"Comparing kernels per role runs on Metal only. This run is for {backend or 'an unknown backend'}."}
  return out | ready | {"applies": True, "compare_message": None}


def results(run:pathlib.Path) -> dict[str, Any]:
  return results_core(run, beside=True)


def results_core(run:pathlib.Path, beside:bool = False) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = _optional(run, "model_profile.json")
  policy = _optional(run, "route_policy.json")
  plan = _optional(run, "measurement_plan.json")
  primitive = _optional(run, "primitive_profile.json")
  timing = _optional(run, "timing_profile.json")
  kernels, context = roofline_kernels(timing) if timing else ([], None)
  chosen = next((s for s in timing.get("context_summaries", []) if s.get("context") == context), {}) if timing else {}
  ceil = _measured_vs_ceiling(manifest, profile, run)
  return {
    "schema": SCHEMA, "kind": "results", "id": run.name, "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "measured": bool(timing) or bool(primitive),
    "ceiling": {k: v for k, v in ceil.items() if not k.startswith("_")},
    "loss": loss_block(run, manifest, ceil, chosen.get("tok_s"), beside=beside),
    "routes": [{"role": r.get("role"), "quant": r.get("quant"), "shape": r.get("shape"),
                "selected_route": r.get("selected_route"), "status": r.get("status"),
                "candidates": list(r.get("candidates", []) or []), "evidence_refs": list(r.get("evidence_refs", []) or []),
                "compare": _route_compare(r.get("compare"))}
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
                          command:str | None = None, probe:str | None = None, probe_reason:str | None = None,
                          provider:str | None = None, layout:str = "one", gpus:int = 1) -> None:
  """probe says whether this collector takes the building-block tests: "measured" or "absent" (with probe_reason)."""
  from boltbeam.workflow.common import run_dir, update_manifest, write_json
  out = run_dir(run)
  write_json(out / MEASURE_STATUS, {"schema": SCHEMA_MEASURE_STATUS, "status": status, "collector": collector,
                                    "reason": reason, "command": command, "probe": probe, "probe_reason": probe_reason,
                                    "provider": provider or providers.DEFAULT, "layout": layout, "gpus": gpus})
  update_manifest(out, stage="measure", artifacts=[MEASURE_STATUS])


def _collectors() -> dict[str, Any]:
  """BoltBeam's own collectors by registry collector_id (data/profiler_capabilities.json): the module that measures."""
  from boltbeam.collectors import llama_bench_decode, metal_native
  return {"metal-native": metal_native, llama_bench_decode.COLLECTOR_ID: llama_bench_decode}


def measure_plan(target_id:str, model:str, run:str, provider:str = providers.DEFAULT, *,
                 tinygrad_root:pathlib.Path | None = None, gpu=None, layout:str = "one",
                 devs:list[dict[str, Any]] | None = None) -> dict[str, Any]:
  """Which collector measures this target with this provider on this machine, or why none can. Decided before
  the run starts, so the step count is right from the first line. A GPU another program holds is refused."""
  from boltbeam.collectors.llama_bench_decode import CannotMeasure
  providers.check(provider)
  try:
    target = get_target(target_id)
  except SystemExit as exc:
    return {"collector": None, "provider": provider, "reason": str(exc), "command": None}
  from boltbeam.workflow import layout as lay
  cid = providers.collector(provider, target.backend)
  devs = devs if devs is not None else lay.devices(_hardware_profile()["gpu"])
  out = {"collector": cid, "provider": provider, "reason": None, "command": None, "layout": layout, "gpus": max(len(devs), 1)}
  offered = {l["id"]: l for l in lay.layouts(len(devs), provider)}
  if layout not in offered or not offered[layout]["available"]:
    why = offered[layout]["reason"] if layout in offered else f"{lay.LAYOUTS.get(layout, layout)} needs more than one GPU"
    return out | {"reason": f"{provider} cannot run {lay.LAYOUTS.get(layout, layout)} here: {why}", "command": "pick one GPU"}
  if provider == tinygrad_role_time.PROVIDER:
    from boltbeam.workflow.autoscan import this_machine_target
    here = this_machine_target()
    if here != target_id and not (target_id == "apple_metal" and target.backend == "Metal" and sys.platform == "darwin"):
      return out | {"reason": f"this machine is {here or 'no registered GPU'}, not {target_id}",
                    "command": f"on the machine with {target_id}: pipeline MODEL --run {run} --target {target_id} "
                               "--measure auto --provider tinygrad"}
    if why := tinygrad_role_time.available(target, tinygrad_root):
      return out | {"reason": why, "command": "set BOLTBEAM_TINYGRAD_ROOT to the tinygrad fork, then measure again"}
  else:
    try:
      _collectors()[cid].preflight(target_id, model, run=run)
    except CannotMeasure as exc:
      return out | {"reason": exc.reason, "command": exc.command}
  used = lay.used_gpus(layout, devs)
  uuids = {d["uuid"] for d in used if d.get("uuid")} if len(devs) > 1 else None
  check = gpu or providers.gpu_free
  free = check(target.backend, uuids=uuids) if uuids else check(target.backend)
  if not free["free"]:
    return out | {"reason": f"not measured: {free['reason']}", "command": "free the GPU, then measure again"}
  return out


def _measure_steps(args, plan:dict[str, Any]) -> list[tuple[str, Any]]:
  provider = plan.get("provider") or providers.DEFAULT
  layout, gpus = plan.get("layout") or "one", plan.get("gpus") or 1
  if plan["reason"]:
    return [("measure", lambda: _write_measure_status(args.run, "skipped", collector=plan["collector"],
                                                       reason=plan["reason"], command=plan["command"],
                                                       provider=provider, layout=layout, gpus=gpus))]
  from boltbeam.collectors import llama_bench_decode, metal_native
  run = pathlib.Path(args.run)
  args.timing = str(run / "timing_trace.json")
  again = f"pipeline {args.model} --run {args.run} --target {args.target} --measure auto --provider {provider}"

  def guarded(key:str, work, *, probe:str | None, probe_reason:str | None = None, last:bool = True):
    def step():
      try:
        work()
      except Exception as exc:  # the reason must outlive the log: a reopened screen reads it from the run
        _write_measure_status(args.run, "failed", collector=plan["collector"], reason=f"{key}: {exc}",
                              command=again, provider=provider, layout=layout, gpus=gpus)
        raise
      if last:
        _write_measure_status(args.run, "measured", collector=plan["collector"], probe=probe,
                              probe_reason=probe_reason, provider=provider, layout=layout, gpus=gpus)
    return (f"measure_{key}", step)

  out_dir = lambda: metal_native.run_dir(args.run)  # noqa: E731
  is_metal = get_target(args.target).backend == "Metal"
  # the building-block probes are BoltBeam's own Metal kernels: they belong to no provider, so both take them
  probe = [guarded("probe", lambda: metal_native.measure(out_dir(), only="probe"), probe="measured", last=False)] \
    if is_metal else []
  if provider == tinygrad_role_time.PROVIDER:
    root = pathlib.Path(args.tinygrad_root).expanduser() if getattr(args, "tinygrad_root", None) else None
    timing = guarded("timing", lambda: providers.measure_tinygrad(out_dir(), root=root),
                     probe="measured" if is_metal else "absent",
                     probe_reason=None if is_metal else llama_bench_decode.PROBE_ABSENT)
  elif plan["collector"] == llama_bench_decode.COLLECTOR_ID:
    timing = guarded("timing", lambda: llama_bench_decode.measure(out_dir(), layout=layout, gpus=gpus), probe="absent",
                     probe_reason=llama_bench_decode.PROBE_ABSENT)
  else:
    timing = guarded("timing", lambda: metal_native.measure(out_dir(), only="timing"), probe="measured")
  if probe:
    args.probe = str(run / "probe_evidence.json")
  return [*probe, timing]


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
  analyze_all = getattr(args, "analyze", False)
  if analyze_all:
    args.measure = "auto"
  plan = None
  if getattr(args, "measure", "none") == "auto" and not (args.probe or args.timing):
    root = pathlib.Path(args.tinygrad_root).expanduser() if getattr(args, "tinygrad_root", None) else None
    plan = measure_plan(args.target, str(pathlib.Path(args.model).expanduser().resolve()), args.run,
                        getattr(args, "provider", None) or providers.DEFAULT, tinygrad_root=root,
                        layout=getattr(args, "layout", None) or "one")
    if analyze_all and not plan["reason"]:  # the machine's measured facts, reused when this machine has them
      steps.append(("machine", lambda: machine(pathlib.Path(args.run), root=root)))
    steps += _measure_steps(args, plan)
  if args.probe:
    steps.append(("ingest_probe", lambda: ingest_probe_run(args.run, args.probe)))
  if args.timing:
    steps.append(("ingest_timing", lambda: ingest_timing_run(args.run, args.timing)))
  if args.probe or args.timing:
    steps.append(("analyze", lambda: analyze_run(args.run)))  # the plan and the report read the new evidence
  steps.append(("output", lambda: output_run(args.run)))
  can_time = plan and providers.capture_method(plan["provider"], get_target(args.target).backend)["method"]
  if analyze_all and plan and not plan["reason"] and can_time:  # one press: per-role time, same engine, report again
    root = pathlib.Path(args.tinygrad_root).expanduser() if getattr(args, "tinygrad_root", None) else None
    steps.append(("role_time", lambda: providers.role_time(pathlib.Path(args.run), plan["provider"], root=root)))
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


def role_time(args, out=sys.stdout) -> int:
  """Time every role inside a real decode in tinygrad's runtime, then refresh the report."""
  def say(text:str) -> None:
    out.write(text + "\n")
    out.flush()
  run = pathlib.Path(args.run).expanduser()
  try:
    provider = getattr(args, "provider", None) or run_provider(run)
    say(f"role-time: start {provider}")
    providers.role_time(run, provider, root=pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None)
    output_run(run)
  except Exception as exc:  # application boundary: the reason is the fact the reader needs
    say(f"role-time failed: {exc}")
    return 1
  say("role-time: done")
  return 0


def compare(args, out=sys.stdout) -> int:
  """Compare kernels per role for one run, one line per step, then refresh the report."""
  def say(text:str) -> None:
    out.write(text + "\n")
    out.flush()
  run = pathlib.Path(args.run).expanduser()
  root = pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None
  try:
    role_compare.compare_run(run, root=root, say=say)
    output_run(run)
  except Exception as exc:  # application boundary: the reason is the fact the reader needs
    say(f"compare failed: {exc}")
    return 1
  return 0


# --- the command line ------------------------------------------------------------------------------------------

def provider_list(target, root:pathlib.Path | None = None, gpu_count:int | None = None) -> dict[str, Any]:
  """The providers that can measure this target here, how step 5 would time their roles, and the GPU layouts each
  can run on this machine (one GPU unless more are found)."""
  from boltbeam.workflow import layout as lay
  n = gpu_count if gpu_count is not None else len(lay.devices(_hardware_profile()["gpu"]))
  rows = [{**p, "layouts": lay.layouts(n, p["provider"])} for p in providers.available(target, tinygrad_root=root)]
  return {"schema": SCHEMA, "kind": "providers", "target_id": target.target_id, "default": providers.DEFAULT,
          "gpu_count": n, "providers": rows}


DEVICE_PREFIX = {"CUDA": "NV", "AMD": "AMD"}  # tinygrad's device name per backend, for the per-GPU probe
_PROBE_WORDS = {"Metal": "BoltBeam's Metal read probe", "CUDA": "BoltBeam's native CUDA read probe"}


def profile_read(gpu:dict[str, Any]) -> tuple[float, str] | None:
  """A GPU's read bandwidth from the chip profile made on this machine, with its date; None for a registry chip.
  One measurement: the profile's number is the limit's number."""
  tid = gpu.get("target_id")
  if not reg.is_local(tid):
    return None
  t = get_target(tid)
  src = ((t.capabilities or {}).get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  if not t.memory_bandwidth_gbs:
    return None
  return t.memory_bandwidth_gbs, (f"measured on this GPU with {_PROBE_WORDS.get(t.backend, 'BoltBeam')}, "
                                  f"{src.get('observed_at', 'date unknown')} (chip profile {tid})")  # tinygrad's device name per backend, for the per-GPU probe


def machine(run:pathlib.Path, *, remeasure:bool = False, root:pathlib.Path | None = None) -> dict[str, Any]:
  """This machine's GPUs, their measured read bandwidth and the measured copies between them (workflow/layout.py),
  cached in the run and reused from the newest run of the same GPUs unless remeasure."""
  from boltbeam.workflow import layout as lay
  from boltbeam.workflow.common import write_json
  devs = lay.devices(_hardware_profile()["gpu"])
  cached = None if remeasure else lay.cached_machine(run, devs)
  if cached is not None:
    facts = {**cached, "reused": True}
  else:
    manifest = load_manifest(run)
    target = get_target(manifest.get("target_id"))
    probe = None
    fork = pathlib.Path(root) if root else role_compare.default_fork_root()
    if role_compare.readiness(fork)["ready"] and target.backend in DEVICE_PREFIX:
      probe = lambda args, env: lay.run_probe(str(role_compare.fork_python(fork)), str(fork), args, env)  # noqa: E731
    topology = ""
    if target.backend == "CUDA" and len(devs) > 1 and shutil.which("nvidia-smi"):
      topology = providers._run(["nvidia-smi", "topo", "-m"])
    metal_read = None
    if target.backend == "Metal" and sys.platform == "darwin":
      from boltbeam.collectors import metal_bandwidth
      metal_read = metal_bandwidth.measure_read_gbs
    cuda_read = None
    if target.backend == "CUDA":
      from boltbeam.collectors import cuda_bandwidth
      if cuda_bandwidth.find_nvcc():
        cuda_read = cuda_bandwidth.measure_read_gbs
    facts = lay.measure_machine(devs, backend=target.backend, device_prefix=DEVICE_PREFIX.get(target.backend),
                                probe=probe, topology=topology, metal_read=metal_read, cuda_read=cuda_read,
                                fallback=lambda tid: get_target(tid).memory_bandwidth_gbs if tid else None,
                                profile_read=profile_read)
    facts["reused"] = False
  write_json(run / lay.MACHINE, {k: v for k, v in facts.items() if k != "reused"})
  return {"schema": SCHEMA, "kind": "machine", **facts}


# --- saving: a run is temporary in the work area until it is saved (exported) -----------------------------------

SAVE_RECORD = "save.json"


def summary_text(res:dict[str, Any], why_no_roles:str | None = None) -> str:
  """The tie-out and the per-role table as plain text, to paste into a message. why_no_roles is this machine's
  reason per-role time cannot be taken here; it replaces the generic "Run times each role" sentence."""
  loss = res.get("loss") or {}
  lines = [f"{res.get('model_id')} on {res.get('target_id')} with {loss.get('provider')}"]
  t = loss.get("tie_out") or {}
  if t.get("refused"):
    lines.append(f"Not tied out: {t['refused']}")
  elif t.get("lines") and t.get("token_ms") is not None:
    lines.append(f"Tie-out, ms per token at context {t['context']:.0f}:")
    for i, l in enumerate(t["lines"]):
      lines.append(f"  {'  ' if i == 0 else '+ '}{l['label']:<48} {l['ms']:9.3f}  {l['how']}")
    lines.append(f"  = {'measured token':<48} {t['token_ms']:9.3f}  {t.get('token_source') or ''}")
  roles = loss.get("roles") or []
  if roles:
    lines.append(f"Per role ({loss.get('source') or ''}):")
    lines.append(f"  {'role':<12} {'quant':<5} {'ideal':>7} {'actual':>7} {'lost':>6} {'% peak':>7} {'us/call':>8}  why")
    for r in roles:
      pct = f"{r['pct_peak']:.1f}%" if r.get("pct_peak") is not None else ""
      us = f"{r['us_per_call']:.1f}" if r.get("us_per_call") is not None else ""
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {r['ideal_ms']:7.3f} {r['actual_ms']:7.3f} {r['lost_ms']:6.3f} "
                   f"{pct:>7} {us:>8}  {r.get('reason') or ''}")
    if loss.get("not_attributed_ms") is not None:
      lines.append(f"  {'not attributed':<18} {'':>7} {loss['not_attributed_ms']:7.3f}")
  elif why_no_roles:
    lines.append(f"Per role: {why_no_roles}")
  elif loss.get("missing"):
    lines.append(f"Per role: {loss['missing']}")
  return "\n".join(lines) + "\n"


def _why_no_roles(run:pathlib.Path, res:dict[str, Any]) -> str | None:
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


def save(run:pathlib.Path, to_root:pathlib.Path) -> dict[str, Any]:
  """Export a run into the saved-runs folder: the run itself (so it opens again), results.json (this seam's results),
  report.html and summary.txt (the tie-out and the per-role table as text). An existing save is replaced."""
  import datetime
  if not is_run(run):
    raise Refused(f"{run} is not a run folder")
  dest = to_root.expanduser().resolve() / run.name
  if dest == run.resolve():
    raise Refused(f"{run} is already saved in {to_root}")
  if dest.exists():
    shutil.rmtree(dest)
  shutil.copytree(run, dest)
  res = results(dest)
  (dest / "results.json").write_text(pretty_json(res))
  (dest / "summary.txt").write_text(summary_text(res, _why_no_roles(dest, res)))
  record = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "from": str(run)}
  (dest / SAVE_RECORD).write_text(json.dumps(record, indent=2) + "\n")
  return {"schema": SCHEMA, "kind": "saved_run", "id": run.name, "dir": str(dest),
          "files": ["report.html", "results.json", "summary.txt"], **record}


def saved(root:pathlib.Path) -> dict[str, Any]:
  """The saved runs, newest first: model, chip, engine, date and the share of the limit reached."""
  rows = []
  if root.is_dir():
    for d in root.iterdir():
      if not (is_run(d) and (d / SAVE_RECORD).exists()):
        continue
      rec, manifest, res = read_json(d / SAVE_RECORD), load_manifest(d), _optional(d, "results.json")
      timing, ceil = res.get("timing") or {}, res.get("ceiling") or {}
      pct = (100.0 * timing["tok_s"] / ceil["tok_s"]) if timing.get("tok_s") and ceil.get("tok_s") else None
      rows.append({"id": d.name, "dir": str(d), "model_id": manifest.get("model_id"), "target_id": manifest.get("target_id"),
                   "provider": run_provider(d), "saved_at": rec.get("saved_at"), "tok_s": timing.get("tok_s"),
                   "pct_of_limit": pct})
  rows.sort(key=lambda r: r["saved_at"] or "", reverse=True)
  return {"schema": SCHEMA, "kind": "saved", "root": str(root), "runs": rows}


def clean_work(work:pathlib.Path, *, before:str, keep:list[str]) -> dict[str, Any]:
  """Remove temporary runs started before `before` (ISO time, the session start), except those in keep. Only on an
  explicit request; never a run of this session or one still in use."""
  import datetime
  cut = datetime.datetime.fromisoformat(before).timestamp()
  removed = []
  if work.is_dir():
    for d in sorted(work.iterdir()):
      if is_run(d) and d.name not in keep and (d / "run_manifest.json").stat().st_mtime < cut:
        shutil.rmtree(d)
        removed.append(d.name)
  return {"schema": SCHEMA, "kind": "cleaned", "work": str(work), "removed": removed}


def gpu_free(target) -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "gpu_free", "target_id": target.target_id, **providers.gpu_free(target.backend)}


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


def _saved_arg(args) -> pathlib.Path | None:
  return pathlib.Path(args.saved).expanduser() if getattr(args, "saved", None) else None


def _run_arg(args) -> pathlib.Path:
  """--run as a folder; with --root, an id looked up by find_run."""
  if args.root:
    return find_run(pathlib.Path(args.root).expanduser(), args.run, _saved_arg(args))
  return pathlib.Path(args.run).expanduser()


def main(argv:list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="python -m boltbeam.workflow.screen", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  sub.add_parser("targets")
  sub.add_parser("chips")
  p = sub.add_parser("autoscan")
  p.add_argument("--remeasure", action="store_true", help="measure a profile made on this machine again")
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
  p.add_argument("--facts-root", action="append", default=[],
                 help="a runs folder: the newest machine facts there for this chip set the memory speed (one GPU)")
  p = sub.add_parser("ceilings")
  p.add_argument("model", nargs="?")
  p.add_argument("--profile", help="a model_profile.json instead of the model file")
  p.add_argument("--id", default=None)
  p = sub.add_parser("runs")
  p.add_argument("--root", required=True)
  p.add_argument("--saved", default=None, help="the saved-runs folder (default ROOT/saved)")
  p.add_argument("--all", action="store_true", help="also list ROOT/.work and the saved runs, each marked where")
  for name in ("run", "results"):
    p = sub.add_parser(name)
    p.add_argument("--run", required=True, help="a run folder, or a run id with --root")
    p.add_argument("--root", default=None, help="look the id up in ROOT, ROOT/.work, then the saved runs")
    p.add_argument("--saved", default=None, help="the saved-runs folder (default ROOT/saved)")
  p = sub.add_parser("machine")
  p.add_argument("--run", required=True)
  p.add_argument("--remeasure", action="store_true", help="measure again instead of reusing the cached facts")
  p.add_argument("--tinygrad-root", default=None)
  p = sub.add_parser("save")
  p.add_argument("--run", required=True)
  p.add_argument("--to", required=True, help="the saved-runs folder")
  p = sub.add_parser("saved")
  p.add_argument("--root", required=True, help="the saved-runs folder")
  p = sub.add_parser("clean-work")
  p.add_argument("--work", required=True)
  p.add_argument("--before", required=True, help="ISO time: runs started before it go (the session start)")
  p.add_argument("--keep", action="append", default=[])
  p = sub.add_parser("delete")
  p.add_argument("--run", required=True)
  p.add_argument("--root", required=True)
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
  p.add_argument("--provider", default=providers.DEFAULT, choices=list(providers.NAMES),
                 help="the runtime that decodes the model in step 4")
  p.add_argument("--layout", default="one", choices=["one", "layer", "row"],
                 help="how the engine uses this machine's GPUs (more than one GPU is limited support)")
  p.add_argument("--analyze", action="store_true",
                 help="one press: measure here, the machine's facts, then per-role time with the same engine")
  p.add_argument("--tinygrad-root", default=None, help="the tinygrad fork; default $BOLTBEAM_TINYGRAD_ROOT")
  for name in ("providers", "gpu-free"):
    p = sub.add_parser(name)
    p.add_argument("--target", required=True)
    p.add_argument("--tinygrad-root", default=None)
  for name in ("compare", "compare-ready", "role-time"):
    p = sub.add_parser(name)
    p.add_argument("--run", required=True)
    p.add_argument("--tinygrad-root", default=None, help="the tinygrad fork; default $BOLTBEAM_TINYGRAD_ROOT or ../tinygrad-arkey-exp")
    if name == "role-time":
      p.add_argument("--provider", default=None, choices=list(providers.NAMES),
                     help="default: the provider step 4 measured the run with")
  args = parser.parse_args(argv)
  if args.command == "pipeline":
    return pipeline(args)
  if args.command == "compare":
    return compare(args)
  if args.command == "role-time":
    return role_time(args)
  code = 0
  try:
    if args.command == "targets":
      out = targets()
    elif args.command == "chips":
      from boltbeam.workflow import chips
      out = {"schema": SCHEMA, **chips.chip_list()}
    elif args.command == "autoscan":
      from boltbeam.workflow import chips
      out = {"schema": SCHEMA, **chips.autoscan(remeasure=args.remeasure)}
    elif args.command == "detect":
      out = detect()
    elif args.command == "providers":
      out = provider_list(_target_arg(args.target), pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None)
    elif args.command == "gpu-free":
      out = gpu_free(_target_arg(args.target))
    elif args.command == "ceiling":
      target = _target_arg(args.target)
      read = None
      if args.facts_root:
        from boltbeam.workflow import layout as lay
        roots = [pathlib.Path(r).expanduser() for r in args.facts_root]
        read = lay.read_bandwidth(lay.newest_facts(roots, target.target_id))
      out = ceiling(_profile_arg(args), target, context=args.context, dtype=args.dtype,
                    peak_gbs=args.peak_gbs, peak_tflops=args.peak_tflops, read=read)
    elif args.command == "ceilings":
      out = ceilings(_profile_arg(args))
    elif args.command == "runs":
      out = runs(pathlib.Path(args.root).expanduser(), _saved_arg(args), all_places=args.all)
    elif args.command == "run":
      out = show(_run_arg(args))
    elif args.command == "compare-ready":
      out = compare_ready(pathlib.Path(args.run).expanduser(),
                          pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None)
    elif args.command == "machine":
      out = machine(pathlib.Path(args.run).expanduser(), remeasure=args.remeasure,
                    root=pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None)
    elif args.command == "save":
      out = save(pathlib.Path(args.run).expanduser(), pathlib.Path(args.to))
    elif args.command == "saved":
      out = saved(pathlib.Path(args.root).expanduser())
    elif args.command == "clean-work":
      out = clean_work(pathlib.Path(args.work).expanduser(), before=args.before, keep=args.keep)
    elif args.command == "delete":
      out = delete(pathlib.Path(args.root).expanduser(), pathlib.Path(args.run).expanduser())
    else:
      out = results(_run_arg(args))
      code = 0 if out["measured"] else 3
  except (Refused, FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
    sys.stdout.write(pretty_json({"schema": SCHEMA, "kind": "error", "error": str(exc)}))
    return 1
  sys.stdout.write(pretty_json(out))
  return code


if __name__ == "__main__":
  raise SystemExit(main())
