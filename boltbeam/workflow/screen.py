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
                                               `pipeline steps: N` (counted stages only) and `pipeline counted:`
                                               first, then one text line per stage, for tailing.
                                               --measure auto runs the target's BoltBeam collector here when this
                                               machine can (Metal: collectors/metal_native.py; NVIDIA: the
                                               llama-bench decode, collectors/llama_bench_decode.py) and writes
                                               measure_status.json either way: measured, skipped or failed, with
                                               the reason and the command to run instead. --provider llama.cpp
                                               (default), tinygrad, vllm, ollama or tensorrt-llm picks the engine;
                                               an engine that cannot read the model's weight format is refused, and
                                               so is a busy GPU. --batch 1,32 also times a decode step of 32 streams.
    providers --target T                       the engines that can measure here (collectors/providers.ENGINES),
                                               each with the reason when it cannot, the weight formats it reads,
                                               and how step 5 times its roles
    gpu-free --target T                        is the GPU free, or which program holds it
    role-time --run DIR [--provider P]         time every role inside a real decode with the run's provider
                                               (collectors/providers.py); feeds `results.loss`
    compare-ready --run DIR [--tinygrad-root R]  can this machine compare kernels for the run: the missing piece
                                               (fork, venv, numpy, model, backend) and the one command that fixes it
    compare  --run DIR [--tinygrad-root R]     compare kernels per role on Metal (boltbeam/search/role_compare.py):
                                               `compare roles: N`, then `role ROLE QUANT: search|ab|done STATUS`,
                                               then `compare done: DIR` or `compare failed: REASON`. It writes the
                                               statuses into route_policy.json and refreshes the report.
                                               Run (pipeline --analyze) runs the same search as its "search" stage
                                               after per-role time, with `stage search: progress P/Q` per role;
                                               --no-search skips it. Other engines and chips skip it with a reason
                                               (kernel_compare/search_status.json, results loss.search).

Nothing here computes a new kind of fact. The ceiling is `model_roofline` exactly as `roofline-theoretical`
calls it, with tokens/s read off the floor (one token at context 1 for decode; the context's tokens for
prefill). The stage order, the next-step ladder and the hot-kernel pick are the report's (`report/html.py`);
the stages `pipeline` runs are the workflow's own functions. The contract is pinned by `tui/testdata`: the
fixture run folders and the expected JSON, checked by `tests/test_workflow_screen.py` here and by
`tui/internal/seam` on the Go side.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time
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
from boltbeam.workflow import progress
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


def plausibility_band(target, read:dict[str, Any] | None = None) -> float:
  """The chip's one plausibility band, as a ± fraction: the largest of the read probe's spread over its launch
  shapes, its cold to sustained difference, the chip's documented run-to-run range (half its width around the
  middle), and 1%. The probe's numbers come from this machine's facts when the bandwidth is theirs, else from the
  chip's fact source; the documented range is the chip's."""
  src = ((getattr(target, "capabilities", None) or {}).get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  probe = (read or {}).get("band") if read and read.get("band") is not None else \
    max(float(src.get("band") or 0.0), float(src.get("spread") or 0.0), float(src.get("drift") or 0.0))
  rr = src.get("run_to_run") or {}
  documented = (rr["high"] - rr["low"]) / (rr["high"] + rr["low"]) if rr.get("high") and rr.get("low") else 0.0
  return round(max(float(probe or 0.0), documented, 0.01), 4)


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
    "peak_bandwidth_gbs": bw, "bandwidth_source": (read or {}).get("short"), "band": plausibility_band(target, read), "peak_tflops": peak_flops / 1e12, "truth_status": decode["truth_status"],
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

# the stage rows a running screen counts as steps (analyze is setup on its first run, so it is not counted here)
COUNTED_STAGES = frozenset({"ingest_probe", "ingest_timing", "output"})


def stage_rows(manifest:dict[str, Any], measure:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """state: done, not_needed (state_note says why) or open (report/html.py stage_state decides)."""
  stages = manifest.get("stages", {}) or {}
  rows = []
  for key, label, note in STAGES:
    state, why = stage_state(key, stages.get(key), measure)
    rows.append({"key": key, "label": label, "note": note, "counted": key in COUNTED_STAGES, "done": key in stages,
                 "state": state, "state_note": why,
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


def run_bandwidth(run:pathlib.Path, target) -> tuple[float | None, str]:
  """The one read bandwidth every number of a run uses (the limit, the tie-out, the per-role rule, every trace's
  peak_gbs and the probe's target): this GPU's measured read from the run's machine facts when it has them, else
  the chip's own figure, labelled. Collectors call this instead of reading the registry themselves."""
  got = run_read(run, target.target_id)
  if got:
    return got["gbs"], got["short"]
  known = profile_read({"target_id": target.target_id})
  return target.memory_bandwidth_gbs, (known[1] if known else f"the chip's registry figure ({target.target_id}), not measured on this GPU")


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
          "band": c.get("band"), "_roles": block["roles"]}


def run_provider(run:pathlib.Path) -> str:
  """The provider step 4 measured this run with. Runs from before providers were measured with llama-bench."""
  return (_optional(run, MEASURE_STATUS).get("provider")) or providers.DEFAULT


def _provider_table(run:pathlib.Path, provider:str, ceil:dict[str, Any], limit_ms:float) -> tuple[dict | None, dict]:
  """One provider's per-role table (or None) and the capture its trace names."""
  trace = _optional(run, providers.TRACES[provider])
  if not trace:
    return None, {}
  capture = trace.get("capture") or {"method": tinygrad_role_time.OWN_TIMING, "reason": None}
  return tinygrad_role_time.loss(ceil.get("_roles") or [], trace, limit_ms, ceil.get("band")), capture


def loss_block(run:pathlib.Path, manifest:dict[str, Any], ceil:dict[str, Any], token:dict[str, Any] | None, *,
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
  if token:
    runtimes.append({"provider": provider, "tok_s": token["tok_s"], "ms": token["ms"], "lost_ms": token["ms"] - limit_ms,
                     "per_role": False, "note": f"whole step, measured untraced, at context {token['context']}"})
  out = {"status": "modeled", "limit_tok_s": ceil["tok_s"], "limit_ms": limit_ms, "runtimes": runtimes, "roles": [],
         "not_attributed_ms": None, "source": None, "missing": None, "refused": None,
         "provider": provider, "capture": plan, "roles_provider": None, "provider_missing": None, "others": [], "unpaired_roles": [],
         "tie_out": None, "role_rule": None, "step": step_facts(token), "latency": None, "cross_check": None,
         "role_source": None, "role_source_words": None, "estimate": None}
  from boltbeam.workflow import tie_out as tie
  from boltbeam.workflow import evidence as ev
  profile, bw = _optional(run, "model_profile.json"), ceil.get("peak_bandwidth_gbs")
  measure = _optional(run, MEASURE_STATUS)
  floor_us = _optional(run, "probe_evidence.json").get("dispatch_floor_us")
  out["latency"] = {"us": floor_us or tie.ROLE_RULE["latency_us"],
                    "source": tie.LATENCY_MEASURED if floor_us else tie.LATENCY_ASSUMED,
                    "evidence": [{"file": "probe_evidence.json", "path": "dispatch_floor_us"}] if floor_us else []}
  out["layout"], out["machine"] = _layout_limit(run, profile, ceil)
  out["search"] = search_block(run, provider)  # searched, skipped with its reason, or not run: said with or without roles
  state = out["search"].pop("state")
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
    out["provider_missing"] = _missing(provider, target, manifest, measure)
  if shown is None:
    out["missing"] = out["provider_missing"]
    out["tie_out"] = tie.tie_out(run, provider=provider, table=None, trace=None, limit_ms=limit_ms, profile=profile,
                                 bandwidth_gbs=bw, missing=out["missing"])
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
  regimes = {(r["role"], r["quant"]): r.get("regime") for r in ceil.get("_roles") or []}
  roles, rule = tie.role_why(table["roles"], bw, regimes=regimes, throttled=facts.throttle() is not None, latency_us=floor_us)
  out["cross_check"] = cross_check(run, bw)
  roles = [{**r, "evidence": facts.role(r["role"], r["quant"])} for r in roles]
  if shown != provider:
    out["search"] = search_block(run, shown)
    state = out["search"].pop("state")
  by_key = {(r.get("role"), r.get("quant")): r for r in _optional(run, "route_policy.json").get("routes", []) or []}
  if shown != tinygrad_role_time.PROVIDER:  # the search compares tinygrad kernels; this table is another engine's
    state, by_key = {"status": "skipped", "reason": role_compare.search_applies("metal", shown)}, {}
  roles = [{**r, **role_compare.role_verdict(by_key.get((r["role"], r["quant"])), state)} for r in roles]
  out.update(roles=roles, role_rule=rule, not_attributed_ms=table["not_attributed_ms"], capture=capture,
             unpaired_roles=table.get("unpaired_roles") or [], source=_source(shown, capture, target))
  out["tie_out"] = tie.tie_out(run, provider=shown, table=table, trace=_optional(run, providers.TRACES[shown]),
                               limit_ms=limit_ms, profile=profile, bandwidth_gbs=bw,
                               missing=None if roles else "the capture could not split the kernels by role")
  trace = _optional(run, providers.TRACES[shown])
  out["role_source"], out["role_source_words"] = role_source(trace, capture)
  est = out["tie_out"].get("estimate")
  if out["role_source"] == tie.ROLE_SOURCE_ISOLATED and est:  # scaled from the sum the estimate was made of: less the floor when the rows carry one
    base = est["isolated_sum_less_floor_ms"] if est.get("floor_us") is not None else est["isolated_sum_ms"]
    out["roles"], _ = tie.scale_roles(out["roles"], est["weight_ms"], base)
    out["estimate"] = {**est, "method": method_line(trace, est, measurement_of(run))}
    for r in runtimes:  # the isolated sum is not a token: it carries no "lost", it is said as a sum
      if r.get("capture") == engine_kernels.METHOD:
        r.update(isolated=True, lost_ms=None, note="the kernels' times summed, each timed alone; attention, norms and gaps are not in it")
  out["findings"] = ev.items(facts, out)
  taken = {(r["role"], r["quant"]) for r in roles}
  out["evidence"] = {"whole_step": facts.whole(), "other_kernels": facts.others(taken), "common": facts.common()}
  if out["search"].get("next"):
    out["search"]["next"]["evidence"] = [p for r in roles if f"{r['role']} {r['quant']}" in out["search"]["next"]["roles"]
                                         for p in r["evidence"] if p["file"] == ev.COMPARE]
  return out


def search_block(run:pathlib.Path, provider:str) -> dict[str, Any]:
  """Run's kernel search for this run: searched, skipped (with the reason) or not run, and the next-step item for
  the roles where a faster kernel was found but not applied. Measured numbers only."""
  state = role_compare.read_status(run)
  routes = _optional(run, "route_policy.json").get("routes", []) or []
  if state is None and any(isinstance(r.get("compare"), dict) for r in routes):
    state = {"status": "searched", "reason": None, "seconds": None}  # `screen compare` ran before search was a stage
  if state is None:
    state = {"status": "not_run", "reason": "the kernel search did not run for this run", "seconds": None}
  out = {"status": state["status"], "reason": state.get("reason"), "seconds": state.get("seconds"), "next": None,
         "state": state}
  if state["status"] == "searched" and provider == tinygrad_role_time.PROVIDER and (f := role_compare.found_summary(routes)):
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


from boltbeam.collectors import engine_kernels  # noqa: E402  (its words belong in this table)

CAPTURE_WORDS = {tinygrad_role_time.OWN_TIMING: "tinygrad's own timing", "nsys": "captured with nsys",
                 "rocprofv3": "captured with rocprofv3", "metal-system-trace": "captured with Metal System Trace",
                 engine_kernels.METHOD: engine_kernels.WORDS}


IN_MODEL_TOOL = {tinygrad_role_time.OWN_TIMING: "tinygrad events", "nsys": "nsys", "rocprofv3": "rocprofv3",
                 "metal-system-trace": "xctrace"}


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


def measurement_of(run:pathlib.Path) -> dict[str, Any] | None:
  """The run's measurement choice (providers.resolve_measurement) as measure_status.json recorded it; None for a
  run from before the choice existed."""
  return ((_optional(run, MEASURE_STATUS).get("capture") or {}).get("measurement")) or None


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
  """Why this run has no per-role table for its provider. The run's own record comes first: the capture the
  measuring machine had (measure_status.json capture, written by the pipeline). Never the action just taken: a run
  that recorded "no capture here" is told what is missing, not to Run again."""
  if provider == tinygrad_role_time.PROVIDER and not tinygrad_role_time.device_for(target):
    return f"{manifest.get('target_id')} names no tinygrad device, so no per-role time can be taken here"
  cap = (measure or {}).get("capture") or {}
  if cap and cap.get("method") is None and cap.get("reason"):
    return f"no per-role time on the measuring machine: {cap['reason']}"
  return MISSING_GENERIC.format(provider=provider)


def step_facts(token:dict[str, Any] | None) -> dict[str, Any] | None:
  """THE measured token's own facts for a screen: context, speed, how it was run (the trace row's source), the
  graph state when the engine says it, the other contexts measured, and the pointer to the row."""
  if not token:
    return None
  graph_failed = "graph replay failed" in str(token.get("source") or "")
  return {"context": token["context"], "tok_s": token["tok_s"], "ms": token["ms"], "source": token.get("source"),
          "rule": token["rule"], "graph_failed": graph_failed, "graph_error": token.get("graph_error"),
          "also": token["also"], "evidence": [{"file": "timing_trace.json", "path": f"rows[{token['index']}]"}]}


def cross_check(run:pathlib.Path, bandwidth_gbs:float | None) -> dict[str, Any] | None:
  """The engine's kernels timed alone beside an in-model capture (providers.CROSS_CHECK), as rows a screen shows:
  role, quant, µs per call, GB/s and the share of peak; None when the run has none."""
  trace = _optional(run, providers.CROSS_CHECK)
  if not trace:
    return None
  cap = trace.get("capture") or {}
  rows = [{"role": r["role"], "quant": r["quant"], "us_per_call": r.get("us_per_call"), "gbs": r.get("gbs"),
           "pct_peak": (100.0 * r["gbs"] / bandwidth_gbs) if r.get("gbs") and bandwidth_gbs else None,
           "evidence": [{"file": providers.CROSS_CHECK, "path": f"rows[{i}]"}]}
          for i, r in enumerate(trace.get("rows") or []) if r.get("scope") == "kernel" and r.get("status") == "measured"]
  return {"method": cap.get("method"), "words": _capture_words(cap.get("method")), "reason": cap.get("reason"), "rows": rows}


def probe_rows(run:pathlib.Path, bandwidth_gbs:float | None) -> dict[str, Any]:
  """BoltBeam's own GEMV per role (the building-block probe, collectors/metal_native.py): GB/s and the share of
  peak, as a reference row group. Not the engine's kernel. absent names what Metal cannot report, so nobody asks
  for a probe this GPU cannot give."""
  ev = _optional(run, "probe_evidence.json")
  measure = _optional(run, MEASURE_STATUS)
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


def the_token(run:pathlib.Path) -> dict[str, Any] | None:
  """THE measured token of this run: step 4's whole-step row that tie_out.measured_step picks for the run's
  provider, near the context its per-role capture attended when it has one. Every headline reads this."""
  from boltbeam.workflow import tie_out as tie
  provider = run_provider(run)
  role_trace = _optional(run, providers.TRACES[provider])
  near = tie.attended_context(role_trace, provider) if role_trace else None
  return tie.measured_step(_optional(run, "timing_trace.json"), provider, near=near)


def results_core(run:pathlib.Path, beside:bool = False) -> dict[str, Any]:
  manifest = load_manifest(run)
  profile = _optional(run, "model_profile.json")
  policy = _optional(run, "route_policy.json")
  plan = _optional(run, "measurement_plan.json")
  primitive = _optional(run, "primitive_profile.json")
  timing = _optional(run, "timing_profile.json")
  ceil = _measured_vs_ceiling(manifest, profile, run)
  token = the_token(run)
  context = token["context"] if token else None
  kernels, context = roofline_kernels(timing, context) if timing else ([], context)
  chosen = next((s for s in timing.get("context_summaries", []) if s.get("context") == context), {}) if timing else {}
  return {
    "schema": SCHEMA, "kind": "results", "id": run.name, "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "measured": bool(timing) or bool(primitive),
    "ceiling": {k: v for k, v in ceil.items() if not k.startswith("_")},
    "loss": loss_block(run, manifest, ceil, token, beside=beside),
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
      "next_actions": list(timing.get("next_actions", []) or []),
      # THE measured token (tie_out.measured_step): the same row the tie-out uses; also lists the other contexts
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
    "blocked": blocked(plan),
    "report": "report.html" if (run / "report.html").exists() else None,
    "engine": {"provider": run_provider(run), "weight_format": _optional(run, MEASURE_STATUS).get("weight_format")
               or (providers.weight_format(profile) if profile else None)},
    "batches": batch_rows(run, manifest, profile, ceil),
  }


def batch_rows(run:pathlib.Path, manifest:dict[str, Any], profile:dict[str, Any], ceil:dict[str, Any]) -> list[dict[str, Any]]:
  """Every measured point (context, batch) of step 4 beside its own limit (tie_out.batch_limit): tokens/s per
  stream and in total, and the share of the limit reached. Empty when step 4 kept no points."""
  from boltbeam.workflow import tie_out as tie
  trace = _optional(run, "timing_trace.json")
  points = trace.get("batches") or []
  if not points or ceil.get("status") != "modeled":
    return []
  provider = trace.get("provider") or run_provider(run)
  element, _ = providers.kv_element(provider)
  params = sum(float(r.get("rows") or 0) * float(r.get("cols") or 0) * float(r.get("count") or 1)
               for r in profile.get("roles") or [])
  try:
    peak_flops = resolve_peak_flops(get_target(manifest.get("target_id")), "fp16", None)
  except SystemExit:
    peak_flops = None
  out = []
  for p in points:
    lim = tie.batch_limit(weight_ms=ceil["floor_ms"], profile=profile, context=float(p["context"]), batch=int(p["batch"]),
                          element_bytes=element, bandwidth_gbs=ceil["peak_bandwidth_gbs"], params=params or None,
                          peak_flops=peak_flops)
    out.append({"context": p["context"], "batch": p["batch"], "step_ms": p["step_ms"], "tok_s_stream": p["tok_s_stream"],
                "tok_s_total": p["tok_s_total"], "source": p.get("source"), "limit": lim,
                "pct_of_limit": 100.0 * lim["step_ms"] / p["step_ms"] if p["step_ms"] else None})
  return out


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
    # the report's own ladder, text only: the same sentence report.html and summary.md print, without the code marks
    "next_step": next_step(report, plan, results(run).get("probe"), _optional(run, MEASURE_STATUS)).replace("`", ""),
    "artifacts": list(manifest.get("artifacts", []) or []),
    "results": results(run),
  })
  return out


# --- the pipeline --------------------------------------------------------------------------------------------

def _write_measure_status(run:str | pathlib.Path, status:str, *, collector:str | None = None, reason:str | None = None,
                          command:str | None = None, probe:str | None = None, probe_reason:str | None = None,
                          provider:str | None = None, layout:str = "one", gpus:int = 1,
                          batches:list[int] | None = None, capture:dict[str, Any] | None = None) -> None:
  """probe says whether this collector takes the building-block tests: "measured" or "absent" (with probe_reason).
  Every run is labelled with its engine (provider) and its weight format, read off the run's model profile, and
  with the per-role capture the measuring machine had (capture: providers.capture_method), a fact of the run."""
  from boltbeam.workflow.common import run_dir, update_manifest, write_json
  out = run_dir(run)
  profile = _optional(out, "model_profile.json")
  write_json(out / MEASURE_STATUS, {"schema": SCHEMA_MEASURE_STATUS, "status": status, "collector": collector,
                                    "reason": reason, "command": command, "probe": probe, "probe_reason": probe_reason,
                                    "provider": provider or providers.DEFAULT, "layout": layout, "gpus": gpus,
                                    "weight_format": providers.weight_format(profile) if profile else None,
                                    "batches": sorted({1, *(batches or [])}), "capture": capture})
  update_manifest(out, stage="measure", artifacts=[MEASURE_STATUS])


def _collectors() -> dict[str, Any]:
  """BoltBeam's own collectors by registry collector_id (data/profiler_capabilities.json): the module that measures."""
  from boltbeam.collectors import llama_bench_decode, metal_native
  return {"metal-native": metal_native, llama_bench_decode.COLLECTOR_ID: llama_bench_decode}


def measure_plan(target_id:str, model:str, run:str, provider:str = providers.DEFAULT, *,
                 tinygrad_root:pathlib.Path | None = None, gpu=None, layout:str = "one",
                 devs:list[dict[str, Any]] | None = None, measurement:str = "auto") -> dict[str, Any]:
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
  out = {"collector": cid, "provider": provider, "reason": None, "command": None, "layout": layout, "gpus": max(len(devs), 1),
         # the per-role capture this machine has for the engine: recorded in the run (measure_status.json)
         # with the Setup choice of measurement (in-model, generic, or auto), resolved here and recorded with it
         "capture": {**providers.capture_method(provider, target.backend, measurement),
                     "measurement": providers.resolve_measurement(provider, target.backend, measurement)}}
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
  if why := providers.reads(provider, model):
    return out | {"reason": why, "command": f"pick a {' or '.join(providers.FORMATS[provider])} model for {provider}"}
  if provider in providers.DRIVEN:
    from boltbeam.workflow.autoscan import this_machine_target
    here = this_machine_target()
    if here != target_id:
      return out | {"reason": f"this machine is {here or 'no registered GPU'}, not {target_id}",
                    "command": f"on the machine with {target_id}: pipeline MODEL --run {run} --target {target_id} "
                               f"--measure auto --provider {provider}"}
    if why := providers.ENGINES[provider].available(target):
      return out | {"reason": why, "command": f"install {provider}, then measure again"}
  elif provider != tinygrad_role_time.PROVIDER:
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


def _parse_batches(text:str | None) -> list[int]:
  """--batch "1,32": the batch sizes to time. Batch 1 is always timed."""
  return sorted({1, *(int(x) for x in (text or "1").split(",") if x.strip())})


def _measure_steps(args, plan:dict[str, Any]) -> list[tuple[str, Any]]:
  provider = plan.get("provider") or providers.DEFAULT
  layout, gpus = plan.get("layout") or "one", plan.get("gpus") or 1
  batches = _parse_batches(getattr(args, "batch", None))
  capture = plan.get("capture")
  if plan["reason"]:
    return [("measure", lambda: _write_measure_status(args.run, "skipped", collector=plan["collector"],
                                                       reason=plan["reason"], command=plan["command"],
                                                       provider=provider, layout=layout, gpus=gpus, batches=batches,
                                                       capture=capture))]
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
                              command=again, provider=provider, layout=layout, gpus=gpus, capture=capture)
        raise
      if last:
        got = probe_state or {"probe": probe, "reason": probe_reason}
        _write_measure_status(args.run, "measured", collector=plan["collector"], probe=got["probe"],
                              probe_reason=got["reason"], provider=provider, layout=layout, gpus=gpus, batches=batches,
                              capture=capture)
    return (f"measure_{key}", step)

  out_dir = lambda: metal_native.run_dir(args.run)  # noqa: E731
  backend = get_target(args.target).backend
  is_metal = backend == "Metal"
  # the building-block probes are BoltBeam's own kernels (adapter 0, Metal and CUDA): they belong to no provider.
  # On CUDA a machine without the driver or nvcc records the probe absent with its reason; the timing still runs.
  probe_state:dict[str, Any] = {}

  def cuda_probe():
    try:
      metal_native.measure(out_dir(), only="probe")
    except llama_bench_decode.CannotMeasure as exc:
      probe_state.update(probe="absent", reason=f"no building-block probe: {exc.reason}")
      return
    probe_state.update(probe="measured", reason=None)

  if is_metal:
    probe = [guarded("probe", lambda: metal_native.measure(out_dir(), only="probe"), probe="measured", last=False)]
  elif backend == "CUDA":
    probe = [guarded("probe", cuda_probe, probe="measured", last=False)]
  else:
    probe = []
  if provider == tinygrad_role_time.PROVIDER:
    root = pathlib.Path(args.tinygrad_root).expanduser() if getattr(args, "tinygrad_root", None) else None
    timing = guarded("timing", lambda: providers.measure_tinygrad(out_dir(), root=root),
                     probe="measured" if is_metal else "absent",
                     probe_reason=None if is_metal else llama_bench_decode.PROBE_ABSENT)
  elif provider in providers.DRIVEN:
    timing = guarded("timing", lambda: providers.measure_engine(out_dir(), provider, batches=batches), probe="absent",
                     probe_reason=llama_bench_decode.PROBE_ABSENT)
  elif plan["collector"] == llama_bench_decode.COLLECTOR_ID:
    timing = guarded("timing", lambda: llama_bench_decode.measure(out_dir(), layout=layout, gpus=gpus, batches=batches),
                     probe="absent", probe_reason=llama_bench_decode.PROBE_ABSENT)
  else:
    timing = guarded("timing", lambda: metal_native.measure(out_dir(), only="timing", batches=batches), probe="measured")
  if probe:
    args.probe = str(run / "probe_evidence.json")
  return [*probe, timing]


# The quick setup stages run in under two seconds. The screen does not count them: step 1 is the first slow one.
SETUP = frozenset({"load", "autoscan", "analyze", "machine", "measure_probe", "measure"})


def counted(step_id:str) -> bool:
  """A step id from progress.step_ids. The first analyze is setup; the second (analyze#2) is counted."""
  return step_id not in SETUP


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
                        layout=getattr(args, "layout", None) or "one",
                        measurement=_role_time_arg(args))
    if analyze_all and not plan["reason"]:  # the machine's measured facts, reused when this machine has them
      steps.append(("machine", lambda: machine(pathlib.Path(args.run), root=root)))
    steps += _measure_steps(args, plan)
  if args.probe:  # a CUDA probe with no GPU on this machine wrote no evidence: nothing to ingest
    steps.append(("ingest_probe", lambda: ingest_probe_run(args.run, args.probe)
                  if pathlib.Path(args.probe).exists() else None))
  if args.timing:
    steps.append(("ingest_timing", lambda: ingest_timing_run(args.run, args.timing)))
  if args.probe or args.timing:
    steps.append(("analyze", lambda: analyze_run(args.run)))  # the plan and the report read the new evidence
  steps.append(("output", lambda: output_run(args.run)))
  # the per-role capture this machine has, as the plan recorded it (an older plan without it is asked now)
  can_time = plan and (plan.get("capture") or providers.capture_method(plan["provider"], get_target(args.target).backend)).get("method")
  if analyze_all and plan and not plan["reason"]:  # one press: per-role time, same engine, search, report again
    root = pathlib.Path(args.tinygrad_root).expanduser() if getattr(args, "tinygrad_root", None) else None
    if can_time:
      steps.append(("role_time", lambda: providers.role_time(pathlib.Path(args.run), plan["provider"], root=root, say=say,
                                                             step=progress.report, measurement=_role_time_arg(args))))
    if getattr(args, "no_search", False):
      steps.append(("search", lambda: role_compare.skip(pathlib.Path(args.run), "skipped with --no-search", say)))
    elif not can_time:
      steps.append(("search", lambda: role_compare.skip(pathlib.Path(args.run), "no per-role time on this machine, "
                                                        "so there is no kernel to compare with", say)))
    else:  # search faster kernels per role where the search exists
      backend = get_target(args.target).backend
      steps.append(("search", lambda: role_compare.search_stage(pathlib.Path(args.run), backend=backend,
                                                                provider=plan["provider"], root=root, say=say,
                                                                step=progress.report)))
    steps.append(("output", lambda: output_run(args.run)))
  ids = progress.step_ids([key for key, _ in steps])
  flags = [counted(sid) for sid in ids]
  say(f"pipeline steps: {sum(flags)}")  # a screen draws n of N from this line: counted stages only
  say("pipeline counted: " + ",".join("1" if c else "0" for c in flags))  # one flag per stage line, in order
  batch = max(_parse_batches(getattr(args, "batch", None)))
  where = f"{args.target}|{plan['provider'] if plan else 'none'}|batch {batch}"
  counted_ids = [sid for sid, c in zip(ids, flags) if c]
  if plan:  # the bar weights counted steps by these seconds; on a first run they are the defaults, an estimate
    expect, source = progress.expected(where, counted_ids)
    say("pipeline expect: " + ",".join(f"{s:.1f}" for s in expect))
    say(f"pipeline expect source: {source}")
  times:dict[str, float] = {}
  run_path = pathlib.Path(args.run)
  for (key, step), sid in zip(steps, ids):
    say(f"stage {key}: start")
    progress.listen(lambda done, total, key=key: say(f"stage {key}: progress {done}/{total}"))
    began = time.monotonic()
    try:
      step()
    except Exception as exc:  # application boundary: the stage's own message is the fact the reader needs
      say(f"stage {key}: failed: {exc}")
      return 1
    finally:
      progress.listen(lambda done, total: None)
    times[sid] = round(time.monotonic() - began, 2)
    progress.save(run_path, where, times, finished=False)
    say(f"stage {key}: done")
  progress.save(run_path, where, times, finished=plan is not None)
  say(f"pipeline done: {args.run}")
  return 0


def _role_time_arg(args) -> str:
  """--role-time in-model|generic|auto as the measurement id: in_model, generic or auto."""
  return str(getattr(args, "role_time", None) or "auto").replace("-", "_")


def role_time(args, out=sys.stdout) -> int:
  """Time every role inside a real decode in tinygrad's runtime, then refresh the report."""
  def say(text:str) -> None:
    out.write(text + "\n")
    out.flush()
  run = pathlib.Path(args.run).expanduser()
  try:
    provider = getattr(args, "provider", None) or run_provider(run)
    say(f"role-time: start {provider}")
    providers.role_time(run, provider, root=pathlib.Path(args.tinygrad_root).expanduser() if args.tinygrad_root else None,
                        batch=int(getattr(args, "batch", None) or 1), say=say, measurement=_role_time_arg(args))
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
  t = loss.get("tie_out") or {}
  m = next((r for r in loss.get("runtimes") or [] if not r.get("per_role")), None)
  batch = t.get("batch") or 1
  head = f"{res.get('model_id')} on {res.get('target_id')} with {loss.get('provider')}, batch {batch}"
  if m and loss.get("limit_tok_s"):
    pct = 100.0 * loss["limit_ms"] / m["ms"] if m.get("ms") else 0.0
    head += (f": {m['tok_s']:.1f} tok/s measured, limit {loss['limit_tok_s']:.1f} tok/s, {pct:.0f}% of roofline, "
             f"{m['lost_ms']:.1f} ms per token lost")
  else:
    head += ": not measured"
  lines = [head]
  step = loss.get("step") or {}
  for a in step.get("also") or []:
    lines.append(f"Also measured at context {a['context']}: {a['tok_s']:.1f} tok/s (not the headline: {step.get('rule')})")
  if step.get("graph_failed"):
    lines.append("Graph replay failed on this run: the token ran without graphs" + (f" ({step['graph_error']})" if step.get("graph_error") else ""))
  est = loss.get("estimate")
  if t.get("refused"):
    lines.append(f"Not tied out: {t['refused']}")
  elif est and t.get("token_ms") is not None:
    lines.append(f"Tie-out, ms per token at context {t['context']:.0f} (only what is measured):")
    lines.append(f"    {'limit at context 1 (ideal)':<48} {t['limit_ms_ctx1']:9.3f}  derived")
    for i, l in enumerate(t["lines"]):
      lines.append(f"  {'  ' if i == 0 else '+ '}{l['label']:<48} {l['ms']:9.3f}  {l['how']}")
    lines.append(f"  = {'measured token':<48} {t['token_ms']:9.3f}  {t.get('token_source') or ''}")
    lines.append(f"  estimated split (isolated): weight kernels {'up to ' if est['scaled'] else ''}{est['weight_ms']:.1f} ms, "
                 f"other and gaps {'at least ' if est['scaled'] else ''}{est['other_ms']:.1f} ms ({est['other_how']})")
  elif t.get("lines") and t.get("token_ms") is not None:
    lines.append(f"Tie-out, ms per token at context {t['context']:.0f}:")
    for i, l in enumerate(t["lines"]):
      lines.append(f"  {'  ' if i == 0 else '+ '}{l['label']:<48} {l['ms']:9.3f}  {l['how']}")
    lines.append(f"  = {'measured token':<48} {t['token_ms']:9.3f}  {t.get('token_source') or ''}")
  if t.get("band"):
    lines.append(f"Band: {t['band']}")
  roles = loss.get("roles") or []
  if roles and est:
    lines.append("Per role, estimated from isolated kernel times:")
    lines.append(f"  {est['method']}")
    lines.append(f"  est. columns {est['label']}; " + estimate_how(est))
    lines.append(f"  {'role':<12} {'quant':<5} {'MB/call':>8} {'us/call':>8} {'GB/s':>6} {'% peak':>7} {'ideal':>7} "
                 f"{'est. ms in token':>16} {'est. lost':>9}  why")
    for r in roles:
      mb = f"{r['mb_per_call']:.1f}" if r.get("mb_per_call") is not None else ""
      us = f"{r['us_per_call']:.1f}" if r.get("us_per_call") is not None else ""
      g = f"{r['gbs']:.1f}" if r.get("gbs") is not None else ""
      pct = f"{r['pct_peak']:.1f}%" if r.get("pct_peak") is not None else ""
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {mb:>8} {us:>8} {g:>6} {pct:>7} {r['ideal_ms']:7.3f} "
                   f"{r['est_ms']:16.3f} {r['est_lost_ms']:9.3f}  {r.get('reason') or ''}")
  elif roles:
    lines.append(f"Per role ({loss.get('source') or ''}):")
    lines.append(f"  {'role':<12} {'quant':<5} {'ideal':>7} {'actual':>7} {'lost':>6} {'% peak':>7} {'us/call':>8}  why")
    for r in roles:
      pct = f"{r['pct_peak']:.1f}%" if r.get("pct_peak") is not None else ""
      us = f"{r['us_per_call']:.1f}" if r.get("us_per_call") is not None else ""
      found = (r.get("best_found") or {}).get("text")
      verdict = (r.get("verdict") or "").replace("_", " ") + (f" ({found})" if found else "")
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {r['ideal_ms']:7.3f} {r['actual_ms']:7.3f} {r['lost_ms']:6.3f} "
                   f"{pct:>7} {us:>8}  {r.get('reason') or ''}" + (f"; kernel search: {verdict}" if verdict else ""))
    if loss.get("not_attributed_ms") is not None:
      lines.append(f"  {'not attributed':<18} {'':>7} {loss['not_attributed_ms']:7.3f}")
    if nxt := (loss.get("search") or {}).get("next"):
      lines.append(f"Next: {nxt['what']}. {nxt['do']}")
  elif why_no_roles:
    lines.append(f"Per role: {why_no_roles}")
  elif loss.get("missing"):
    lines.append(f"Per role: {loss['missing']}")
  if (cc := loss.get("cross_check")) and cc.get("rows"):
    lines.append(f"Cross-check ({cc.get('words')}):")
    for r in cc["rows"]:
      lines.append(f"  {r['role']:<12} {r['quant']:<5} {r['us_per_call']:8.1f} us {r['gbs']:6.1f} GB/s")
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
  p = sub.add_parser("chips")
  p.add_argument("--facts-root", action="append", default=[], help="a runs folder to read this machine's measured read from")
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
  p.add_argument("--batch", default="1", help="batch sizes to time, as 1,32; batch 1 is always timed")
  p.add_argument("--role-time", default="auto", choices=["auto", "in-model", "generic"],
                 help="how roles are timed: in-model (a capture of the real token), generic (BoltBeam's kernel timer, "
                      "each kernel alone) or auto (in-model when this machine has it, else generic)")
  p.add_argument("--no-search", action="store_true",
                 help="with --analyze: skip the per-role kernel search (a quick run)")
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
      p.add_argument("--batch", type=int, default=1, help="capture one decode step of this many streams")
      p.add_argument("--role-time", default="auto", choices=["auto", "in-model", "generic"])
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
      roots = [pathlib.Path(r).expanduser() for r in args.facts_root]
      out = {"schema": SCHEMA, **chips.chip_list(facts_roots=roots)}
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
