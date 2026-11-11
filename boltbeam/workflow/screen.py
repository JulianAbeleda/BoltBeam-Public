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
    recommend MODEL | --profile P --target T [--gpus N] [--engine E]  the engine settings for this model on this machine: context, speed, flags (derived)
    runs     --root DIR [--all]                one summary per run folder under DIR; --all adds DIR/.work and the
                                               saved runs, each row marked where: runs, work or saved
    run      --run DIR | --run ID --root DIR   the stages, what is blocked, the next step, the results; an id is
                                               looked up in DIR/ID, DIR/.work/ID, then SAVED/ID (find_run)
    delete   --run DIR --root ROOT             remove one run folder; refused unless DIR is a run directly under ROOT
    save     --run DIR --to SAVED              export a run: the run, results.json, report.html, summary.txt
    saved    --root SAVED                      the saved runs, newest first, with the share of the limit reached; a run
                                               brought home by `boltbeam fetch` says from where and when (fetched)
    clean-work --work DIR --before ISO         remove temporary runs started before ISO (never --keep ones)
    results  --run DIR | --run ID --root DIR   what won per role, the timing against the ceiling, the regimes
    plain    --run DIR | --run ID --root DIR   the plain-results block (report/plain.py), the words every screen shows:
                                               layer 1 (how fast, how close, what holds it back, the top three parts
                                               with their gains, the trust line) and layer 2 (the four-column table,
                                               How it was measured, Machine and engine, the glossary)
    words                                      the plain words (data/plain_words.json): stage names, glossary, templates
    speedup-offer MODEL                        Speed up for this model: enabled or why not, the copy's size, the stages' expected seconds
    speedup-plain LOG                          a Speed up job's log as layer 1: kept, not kept, refused or crashed, in plain words
    speedup-verify COPY                        a copy against its record (derive lighter-head --verify), pass or the plain reason
    head-to-head --a DIR --b DIR [--bands F]   two runs side by side (compare_runs): their headline figures, the better
                                               one marked, what differs; their parts too when both are of one model
    emit     --run DIR | --run ID --root DIR   Emit: write <run>/gameplan.json and gameplan.md (workflow/gameplan.py), the
                                               run's measured work turned into investigations, or per-role kernel instructions when supported,
                                               from the run's own files only; prints where, the plan and its markdown.
                                               Refused (exit 1) before a run has a per-role table. Refreshes report.html.
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
    engines                                    look for every engine on this machine now and save what was found
                                               (collectors/engine_scan.py, ~/.boltbeam/engines.json); the screen runs
                                               it at boot, and `providers` reads the saved paths after env and PATH
    engines --update [--engine E] [--into DIR]  look at each engine's upstream releases; install a newer one beside
                                               the current build, as pending (workflow/engine_update.py, the only
                                               installer). pipeline --analyze runs the previous build once, compares
                                               (RUN/engine_upgrade.json), then makes it current; --upgrade-engines first
    engines --latest                           each engine's build against upstream's newest release (a look at most
                                               once per engine_latest_max_age_h); the screen runs it after the scan
    update [--skip V]                          is a newer BoltBeam released: the installed and newest versions, how to
                                               upgrade this install; --skip V keeps V from prompting (cli/self_update.py)
    providers --target T                       the engines that can measure here (collectors/providers.ENGINES),
                                               each with the reason when it cannot, the weight formats it reads,
                                               and how step 5 times its roles
    gpu-free --target T                        is the GPU free, or which program holds it
    mtp MODEL --target T --out DIR [--ks 1,2,3] vLLM's multi-token prediction per draft count k on the shipped natural
                                               prompts, against k=0: tok/s per prompt and their spread, acceptance
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
import datetime
import json
import pathlib
import shutil
import sys
from typing import Any

from boltbeam.cli import self_update
from boltbeam.collectors import capture_settings, engine_builds, engine_families, engine_scan, providers, tinygrad_role_time, vllm_mtp
from boltbeam.collectors import replay as rp
from boltbeam.core.canonical import pretty_json
from boltbeam.profile.loaders import profile_from_model
from boltbeam.quantization import speed_up
from boltbeam.report import plain, speed_up_plain
from boltbeam.search import tinygrad_fork
from boltbeam.target.targets import adopt_run_chip, get_target
from boltbeam.target import gpu as tgpu
from boltbeam.trace import replay_trace as rt
from boltbeam.vocab import Workload
from boltbeam.roofline import limit
from boltbeam.workflow import chips, engine_update, fetch, gameplan, layout as lay, output_run, pipeline, recommend as rec
from boltbeam.workflow import repeat_bands as rb, replay as rw, results as rs, run_folder as rf
from boltbeam.target.hardware import hardware_profile
from boltbeam.artifacts.run_io import load_manifest, optional, read_json
from boltbeam.collectors.capture_scope import PIN_RUNS
from boltbeam.core.refused import Refused
from boltbeam.vocab import SCHEMA_TUI as SCHEMA
from boltbeam.workflow.text_report import summary_text, why_no_roles

def compare_ready(run:pathlib.Path, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Whether step 5 can compare kernels for this run on this machine: on a supported backend, with the fork, its venv and
  the model. Kept out of `results`, whose pinned JSON must not depend on the machine."""
  manifest, policy = load_manifest(run), optional(run, "route_policy.json")
  out = {"schema": SCHEMA, "kind": "compare_ready", "id": run.name}
  backend = str((policy.get("target") or {}).get("backend") or "")
  try:  # the runtime role times come from, named once (tinygrad_role_time.runtime_name)
    out["runtime"] = tinygrad_role_time.runtime_name(get_target(manifest.get("target_id")))
  except SystemExit:
    out["runtime"] = None
  ready = tinygrad_fork.readiness(root, model=manifest.get("model_path"))
  if tinygrad_fork.runtime_row(backend) is None:
    # a fast seam: the search runtime table is the cheap answer; the stage asks the provider and the GPU itself
    return out | ready | {"applies": False,
                          "compare_message": f"Comparing kernels per role has no runtime row for {backend or 'an unknown backend'} "
                                             "(boltbeam/data/search_runtime.json)."}
  return out | ready | {"applies": True, "compare_message": None}

def compare_runs(a:pathlib.Path, b:pathlib.Path, bands:dict[str, Any] | None = None) -> dict[str, Any]:
  """Two runs head to head (report/plain.head_to_head): their headline figures side by side and what differs, for any
  two measured runs. The parts are compared too when both runs are of the same model (plain.model_name): the replay
  diff or the engine-upgrade audit (diff), or replay-cross for two replays from different engines; a refusal there is
  one plain line (plain.parts_note), never a refusal of the whole compare."""
  from boltbeam.workflow.diff import DiffRefused
  ra, rb = rs.results(a), rs.results(b)
  head = plain.head_to_head({"id": a.name, "res": ra, "run": a}, {"id": b.name, "res": rb, "run": b})
  out = {"schema": SCHEMA, "kind": "head_to_head", "head": head, "lines": [], "plain": None, "parts_note": None}
  if head["refused"]:
    return out
  same = plain.model_name(ra, a) == plain.model_name(rb, b)
  refused = None
  if same:
    try:
      if ra.get("workload") == rb.get("workload") == "replay" and providers.run_provider(a) != providers.run_provider(b):
        c = rw.cross(a, b)
        out["lines"] = rw.cross_lines(c)
      else:
        d = rw.diff(a, b, bands=bands)
        out.update(lines=rw.diff_lines(d), plain=plain.compare(d))
    except DiffRefused as exc:
      refused = exc
  out["parts_note"] = plain.parts_note(same, refused)
  return out


def replay_command(args) -> dict[str, Any]:
  """The replay's seam commands: write a shape's trace, list the shapes, find running servers, compare two runs."""
  if args.command == "replay-shapes":
    table = rt.shapes()
    return {"schema": SCHEMA, "default": table["default"],
            "shapes": [{"name": n, "label": rt.label(rt.shape(n)), "words": s["words"]} for n, s in table["shapes"].items()]}
  if args.command == "replay-servers":
    return {"schema": SCHEMA, "servers": rp.running_servers()}
  if args.command == "replay-cross":
    c = rw.cross(pathlib.Path(args.a).expanduser(), pathlib.Path(args.b).expanduser())
    return {**c, "lines": rw.cross_lines(c)}
  if args.command == "head-to-head":
    return compare_runs(pathlib.Path(args.a).expanduser(), pathlib.Path(args.b).expanduser(),
                        bands=read_json(args.bands) if getattr(args, "bands", None) else None)
  if args.command in ("replay-diff", "diff"):
    d = rw.diff(pathlib.Path(args.before).expanduser(), pathlib.Path(args.after).expanduser(),
                bands=read_json(args.bands) if getattr(args, "bands", None) else None)
    return {**d, "lines": rw.diff_lines(d), "plain": plain.compare(d)}
  if args.command=='replay-review':
    from boltbeam.collectors.forced_replay import publish
    publish(pathlib.Path(args.run),read_json(args.inputs))
    output_run(pathlib.Path(args.run))
    return rs.results(pathlib.Path(args.run))
  trace = rt.shape(args.shape)
  rt.write(trace, pathlib.Path(args.out).expanduser())
  return {"schema": SCHEMA, "out": args.out, "label": rt.label(trace), "source": trace["source"]}

# --- the command line ------------------------------------------------------------------------------------------

def provider_list(target, root:pathlib.Path | None = None, gpu_count:int | None = None, profile:dict[str, Any] | None = None,
                  facts:dict[str, Any] | None = None) -> dict[str, Any]:
  """The providers that can measure this target here, how step 5 would time their roles, and the GPU layouts each
  can run on this machine (one GPU unless more are found). With the model's profile and this machine's facts, each
  layout carries its predicted decode limit (layout.limit), the highest marked recommended, and a note when the model
  fits one GPU; a layout whose limit cannot be derived keeps its words."""
  devs = tgpu.devices(hardware_profile()["gpu"])
  n = gpu_count if gpu_count is not None else len(devs)
  rows = [{**p, "layouts": lay.layouts(n, p["provider"], target.backend), "settings": capture_settings.offered(p["provider"])}
          for p in providers.available(target, tinygrad_root=root, cached=True)]
  note = None
  if n > 1 and profile and facts:
    try:
      model_bytes = limit.ceiling(profile, target, read=limit.read_bandwidth(facts))["decode"]["bytes_moved"]
    except (Refused, SystemExit):
      model_bytes = None
    if model_bytes:
      for r in rows:
        r["layouts"] = lay.predict(r["layouts"], r["provider"], bytes_per_token=model_bytes, facts=facts,
                                   hidden_size=profile.get("hidden_size"), layers=profile.get("layer_count"))
      note = lay.recommend([l for r in rows for l in r["layouts"]], model_bytes, devs)
  return {"schema": SCHEMA, "kind": "providers", "target_id": target.target_id, "default": providers.DEFAULT,
          "gpu_count": n, "providers": rows, "families": engine_families.rows(rows), "note": note,
          "repeat_runs": PIN_RUNS}  # Setup's Repeats offers this n; the Engine picker shows families (engine_families)


def recommended(profile:dict[str, Any], target, *, engine:str = "vllm", gpus:int | None = None, facts:dict[str, Any] | None = None,
                seqs:int = 1, context:int | None = None, check_in:list[pathlib.Path] | None = None, **offload) -> dict[str, Any]:
  """The engine settings for this model on this machine's GPUs (workflow/recommend.py): derived, nothing runs.
  check_in (run folders): also start the engine once with the recommended flags, read its own KV figures, cache its
  overhead in the newest machine facts there and add predicted against measured as `check`."""
  devs = tgpu.devices(hardware_profile()["gpu"])[:gpus or None]
  args = dict(target=target, seqs=seqs, context=context or rec.CONTEXT, **offload)
  out = {"schema": SCHEMA, "kind": "recommend", "target_id": target.target_id, **rec.setup(profile, facts, engine, devs, **args)}
  if check_in is None:
    return out
  free = gpu_free(target)
  if not free["free"]:
    raise Refused("Check needs a free GPU: " + free["reason"])
  found = lay.newest_facts_file(check_in, target.target_id)
  try:
    got = rec.check(out, profile["source"], facts_file=found[0] if found else None, facts=facts)
  except ValueError as exc:
    raise Refused(str(exc)) from exc
  after = rec.setup(profile, rec.facts_with_overhead(facts or {}, engine, got["overhead"], "", path=got["key"]), engine, devs, **args)
  return {**out, "check": {**got, "after": rec.compare(after, got["overhead"], got["layout"])}}


def engines(*, cached:bool = False) -> dict[str, Any]:
  """Look for every engine on this machine now and save what was found (collectors/engine_scan.py): one row per
  item with its path and how it was found. The screen runs this at boot; `providers` then sees the saved paths."""
  got = engine_scan.cached_or_scan() if cached else engine_scan.scan_and_save()
  return {"schema": SCHEMA, "kind": "engines", "file": got["file"], "scanned_at": got["scanned_at"],
          "seconds": got["seconds"], "engines": engine_scan.rows(got)}

def save(run:pathlib.Path, to_root:pathlib.Path) -> dict[str, Any]:
  """Export a run into the saved-runs folder: the run itself (so it opens again), results.json (this seam's results),
  report.html and summary.txt (the tie-out and the per-role table as text). An existing save is replaced."""
  if not rf.is_run(run):
    raise Refused(f"{run} is not a run folder")
  dest = to_root.expanduser().resolve() / run.name
  if dest == run.resolve():
    raise Refused(f"{run} is already saved in {to_root}")
  res = rs.results(run)  # validate before replacing a previously saved report
  if dest.exists():
    shutil.rmtree(dest)
  shutil.copytree(run, dest, symlinks=True)  # a link stays a link: a run may link a checkpoint's weights (trtllm_decode)
  # Regenerate all presentation artifacts from the same current evidence.
  # Copying an older report/validation file can otherwise contradict results.json.
  output_run(dest)
  (dest / "results.json").write_text(pretty_json(res))
  (dest / "summary.txt").write_text("\n".join(plain.layer1_lines(plain.plain_results(res, dest))) + "\n\n"
                                    + summary_text(res, why_no_roles(dest, res)))
  record = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "from": str(run)}
  (dest / rf.SAVE_RECORD).write_text(json.dumps(record, indent=2) + "\n")
  return {"schema": SCHEMA, "kind": "saved_run", "id": run.name, "dir": str(dest),
          "files": ["report.html", "results.json", "summary.txt"], **record}

def saved(root:pathlib.Path) -> dict[str, Any]:
  """The saved runs, newest first: model, chip, engine, date and the share of the limit reached."""
  rows = []
  if root.is_dir():
    for d in root.iterdir():
      if not (rf.is_run(d) and (d / rf.SAVE_RECORD).exists()):
        continue
      rec, manifest, res = read_json(d / rf.SAVE_RECORD), load_manifest(d), optional(d, "results.json")
      validation = optional(d, "report_validation.json")
      if validation.get("status") == "failed":
        res = {}  # a previously saved success cannot override the current validation failure
      timing, ceil, loss = res.get("timing") or {}, res.get("ceiling") or {}, res.get("loss") or {}
      if loss.get("session") and loss.get("runtimes"):  # the limit is the focus depth's: so is the turn set against it
        timing = {"tok_s": loss["runtimes"][0].get("tok_s")}
      pct = (100.0 * timing["tok_s"] / ceil["tok_s"]) if timing.get("tok_s") and ceil.get("tok_s") and not loss.get("pct_withheld") else None
      rows.append({"id": d.name, "dir": str(d), "model_id": manifest.get("model_id"), "target_id": manifest.get("target_id"),
                   "provider": providers.run_provider(d), "saved_at": rec.get("saved_at"), "tok_s": timing.get("tok_s"),
                   "pct_of_limit": pct, "workload": manifest.get("workload"), "measured": bool(res.get("measured")),
                   "validation_failed": validation.get("status") == "failed", "layout": rf.layout_words(d),
                   "fetched": fetch.provenance(d),
                   "roofline_pct": ((res.get("loss") or {}).get("replay") or {}).get("headline", {}).get("e_pct")})
  rows.sort(key=lambda r: r["saved_at"] or "", reverse=True)
  return {"schema": SCHEMA, "kind": "saved", "root": str(root), "runs": rows}

def clean_work(work:pathlib.Path, *, before:str, keep:list[str]) -> dict[str, Any]:
  """Remove temporary runs started before `before` (ISO time, the session start), except those in keep. Only on an
  explicit request; never a run of this session or one still in use."""
  cut = datetime.datetime.fromisoformat(before).timestamp()
  removed = []
  if work.is_dir():
    for d in sorted(work.iterdir()):
      if rf.is_run(d) and d.name not in keep and (d / "run_manifest.json").stat().st_mtime < cut:
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
  """--run as a folder; with --root, an id looked up by find_run. The run's own chip profile is adopted."""
  run = rf.find_run(pathlib.Path(args.root).expanduser(), args.run, _saved_arg(args)) if args.root else pathlib.Path(args.run).expanduser()
  adopt_run_chip(run)
  return run

def _offload_args(p) -> None:
  p.add_argument("--offload", action="store_true", help="state the offload setup and speed limit (experts in RAM, a cache on the card) even when the model fits")
  p.add_argument("--experts", type=int, help="with --offload: re-size the experts to N per layer (a file not on disk; the result is an estimate)")
  p.add_argument("--bits-per-weight", type=float, help="with --experts: the experts' bits per weight")
  p.add_argument("--draft", help="a draft model's file (an MTP head kept apart): its weights and KV count against the fit")


def _offload_kw(args) -> dict[str, Any]:
  if (args.experts is None) != (args.bits_per_weight is None):
    raise Refused("--experts and --bits-per-weight go together: an estimate needs both")
  if args.experts is not None and not args.offload:
    raise Refused("--experts is an estimate for --offload")
  return {"offload": args.offload, "moe_estimate": {"experts": args.experts, "bits_per_weight": args.bits_per_weight} if args.experts is not None else None,
          "draft": args.draft}


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
  sub.add_parser("words", help="the plain words: stage names, glossary, layer-1 templates (data/plain_words.json)")
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
  p.add_argument("--engine", default="llama.cpp", choices=list(providers.NAMES), help="with --offload: the engine whose flags are given")
  _offload_args(p)
  p = sub.add_parser("ceilings")
  p.add_argument("model", nargs="?")
  p.add_argument("--profile", help="a model_profile.json instead of the model file")
  p.add_argument("--id", default=None)
  p = sub.add_parser("runs")
  p.add_argument("--root", required=True)
  p.add_argument("--saved", default=None, help="the saved-runs folder (default ROOT/saved)")
  p.add_argument("--all", action="store_true", help="also list ROOT/.work and the saved runs, each marked where")
  for name in ("run", "results", "emit", "plain"):
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
  p.add_argument("--workload", default="decode", choices=[w.value for w in Workload])
  p.add_argument("--bands", help="repeat calibration artifact; only an exact setup match can supply bands")
  for name, (kind, _, about) in capture_settings.ASKED.items():
    p.add_argument("--" + name.replace("_", "-"), type=kind, help=about)
  p.add_argument("--repeats", type=int, default=None, help=f"run the same setup N times (N >= 2; {PIN_RUNS} before a band "
                 "may justify an engine pin), then their repeat band into RUN/repeat_bands.json (workflow/repeat_bands.py)")
  p.add_argument("--ctxs", default=None, help="decode contexts (default 128,512) or prompt lengths (default: the sweep "
                                              "to the model's own cap, prefill_work.sweep; a session: cap for the derived depths)")
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
  p.add_argument("--upgrade-engines", "--update-engines", dest="update_engines", action="store_true", help="Upgrade "
                 "engines first: `engines --update` for this engine installs a newer release beside the current build; "
                 "the run itself never installs (--update-engines is the old name)")
  p.add_argument("--forced-replay", action="store_true", help="vLLM replay on a backend whose capture records a marked window (vendor_capture.windowed): scripted production capture, exact pairing and forced native repeats")
  p.add_argument("--replay-trace", default=None, help="a replay: a boltbeam.replay_trace.v1 file or a shape's name "
                                                      "(default: the default shape in data/replay_shapes.json)")
  p.add_argument("--server-argv", default=None, help="a replay: the user's llama-server command, as one string")
  p.add_argument("--replay-pairs", type=int, default=None, help="a llama.cpp replay: native/profiled server pairs, in the order NP,NP,PN,PN (default 4)")
  p = sub.add_parser("repeat-plan", help="fix n before collecting repeats (an engine pin needs n >= 4 fixed so)")
  p.add_argument("--n", type=int, default=PIN_RUNS)
  p.add_argument("--out", required=True)
  p = sub.add_parser("repeat-bands", help="derive per-row bands from n >= 2 identical repeat runs (no GPU)")
  p.add_argument("--runs", nargs="+", required=True, help="the before runs, all of one exact setup")
  p.add_argument("--plan", help="the repeat-plan file written before these runs were collected")
  p.add_argument("--out", required=True, help="repeat_bands.json; existing calibrations are retained")
  p = sub.add_parser("replay-review", help="reconcile retained forced replay captures without GPU measurement")
  p.add_argument("--run", required=True)
  p.add_argument("--inputs", required=True, help="local capture manifest; raw tokens remain local")
  p = sub.add_parser("replay-trace", help="write a shape's counts-only replay trace")
  p.add_argument("--out", required=True)
  p.add_argument("--shape", default=None, help="a shape in data/replay_shapes.json")
  sub.add_parser("replay-shapes", help="the shipped replay shapes, each with its label")
  sub.add_parser("replay-servers", help="the llama-server commands running on this machine, to replay on")
  p = sub.add_parser("replay-cross", help="two engines' replays of one trace: the same work per turn, each against its roofline")
  p.add_argument("--a", required=True)
  p.add_argument("--b", required=True)
  p = sub.add_parser("head-to-head", help="two runs head to head: their headline figures side by side, what differs, and their "
                                     "parts when both are of the same model")
  p.add_argument("--a", required=True)
  p.add_argument("--b", required=True)
  p.add_argument("--bands", help="repeat_bands.json containing calibrations for both setups")
  for name, about in (("replay-diff", "two replay runs of one trace, turn by turn"),
                      ("diff", "two runs of one model and target: replays turn by turn, prefill or decode audits row by row "
                               "(the engine-upgrade audit, docs/engine-upgrade-audit.md)")):
    p = sub.add_parser(name, help=about)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--bands", help="repeat_bands.json containing calibrations for both setups")
  for name in ("providers", "gpu-free"):
    p = sub.add_parser(name)
    p.add_argument("--target", required=True)
    p.add_argument("--tinygrad-root", default=None)
    if name == "providers":
      p.add_argument("model", nargs="?", help="with this machine's facts, each layout shows its predicted decode limit")
      p.add_argument("--profile", help="a model_profile.json instead of the model file")
      p.add_argument("--id", default=None)
      p.add_argument("--facts-root", action="append", default=[], help="a runs folder: the newest machine facts there")
  p = sub.add_parser("recommend", help="the engine settings for this model on this machine's GPUs: context, speed and flags (derived; nothing runs)")
  p.add_argument("model", nargs="?")
  p.add_argument("--profile", help="a model_profile.json instead of the model file")
  p.add_argument("--id", default=None)
  p.add_argument("--target", required=True)
  p.add_argument("--gpus", type=int, default=None, help="use the first N GPUs (default: every GPU found)")
  p.add_argument("--engine", default="vllm", choices=list(providers.NAMES))
  p.add_argument("--seqs", type=int, default=1, help="sequences served together (1: one user)")
  p.add_argument("--context", type=int, default=None, help="the context the sequence count and the speed are stated at")
  p.add_argument("--facts-root", action="append", default=[], help="a runs folder: the newest machine facts there")
  _offload_args(p)
  p.add_argument("--check", action="store_true", help="start the engine once with these flags (a free GPU, a minute or two), "
                 "read its KV cache figures, cache its overhead in the facts and show predicted against measured")
  p = sub.add_parser("mtp", help="vLLM's multi-token prediction per draft count on natural prompts (collectors/vllm_mtp)")
  p.add_argument("model")
  p.add_argument("--target", required=True)
  p.add_argument("--out", required=True, help="the folder the measurement, the continuations and the logs go to")
  p.add_argument("--ks", default=",".join(map(str, vllm_mtp.KS)), help="draft counts to measure; k=0 always runs as the baseline")
  p.add_argument("--tokens", type=int, default=256, help="tokens generated per prompt and depth")
  p.add_argument("--long-tokens", type=int, default=2048, help="the long depth's natural continuation; 0: short only")
  p.add_argument("--facts-root", action="append", default=[], help="a runs folder: the measurement is cached in the newest "
                 "machine facts there, where the recommended setup reads it")
  p = sub.add_parser("speedup-offer", help="Speed up for this model: enabled or why not, the copy's size and time, no GPU (quantization/speed_up.offer)")
  p.add_argument("model")
  p = sub.add_parser("speedup-plain", help="a Speed up job's log as layer 1: kept, not kept, refused or crashed, in plain words (report/speed_up_plain)")
  p.add_argument("log")
  p = sub.add_parser("speedup-verify", help="check a Speed up copy against its record, in plain words (derive lighter-head --verify)")
  p.add_argument("copy")
  sub.add_parser("update", help="is a newer BoltBeam released (cli/self_update.py)").add_argument("--skip", help="skip this version")
  p = sub.add_parser("engines", help="look for every engine on this machine now and save what was found")
  p.add_argument("--cached", action="store_true", help="reuse saved discovery when settings and paths still match")
  p.add_argument("--update", action="store_true", help="look at upstream releases and install newer builds beside "
                 "these (workflow/engine_update.py); the next --analyze run compares each once, then uses it")
  p.add_argument("--latest", action="store_true", help="each engine's build against upstream's newest release; installs nothing")
  p.add_argument("--engine", action="append", choices=list(engine_builds.rows()), help="with --update or --latest: only this engine")
  p.add_argument("--target", default=None, help="with --update: the chip the builds are for (default: this machine's)")
  p.add_argument("--into", default=None, help="with --update: the folder builds go in (default ~/.boltbeam/engines; kept)")
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
  if args.command in ("repeat-bands", "repeat-plan"):
    try:
      if args.command == "repeat-plan":
        print(f"n = {rb.plan(args.n, pathlib.Path(args.out).expanduser())['n']} fixed in {args.out}")
        return 0
      fixed = read_json(pathlib.Path(args.plan).expanduser()) if args.plan else None
      got = rb.save([pathlib.Path(r).expanduser() for r in args.runs], pathlib.Path(args.out).expanduser(), fixed)
    except (ValueError, FileNotFoundError) as exc:
      parser.error(str(exc))
    print(f"measured repeat bands written to {args.out}: n = {got['n']}, false-move rate {got['false_move_rate']:.0%} per row"
          + (" (weak)" if got["weak"] else ""))
    return 0
  if args.command == "pipeline":
    return pipeline.pipeline_repeated(args) if args.repeats else pipeline.pipeline_current(args)
  if args.command == "compare":
    return pipeline.compare(args)
  if args.command == "role-time":
    return pipeline.role_time(args)
  code = 0
  try:
    if args.command == "targets":
      out = limit.targets()
    elif args.command == "chips":
      roots = [pathlib.Path(r).expanduser() for r in args.facts_root]
      out = {"schema": SCHEMA, **chips.chip_list(facts_roots=roots)}
    elif args.command == "autoscan":
      out = {"schema": SCHEMA, **chips.autoscan(remeasure=args.remeasure)}
    elif args.command == "detect":
      out = chips.detect()
    elif args.command == "providers":
      target = _target_arg(args.target)
      model = _profile_arg(args) if args.model or args.profile else None
      out = provider_list(target, pipeline.tinygrad_root(args), profile=model,
                          facts=lay.newest_facts([pathlib.Path(r).expanduser() for r in args.facts_root], target.target_id))
    elif args.command == "recommend":
      target = _target_arg(args.target)
      roots = [pathlib.Path(r).expanduser() for r in args.facts_root]
      out = recommended(_profile_arg(args), target, engine=args.engine, gpus=args.gpus, seqs=args.seqs, context=args.context,
                        facts=lay.newest_facts(roots, target.target_id), check_in=roots if args.check else None, **_offload_kw(args))
    elif args.command == "engines" and args.update:
      target = _target_arg(args.target or chips.detect().get("target_id") or "")
      out = {"schema": SCHEMA, "kind": "engines-update", "file": str(engine_builds.state_file()),
             "engines": engine_update.update(args.engine or list(engine_builds.rows()), backend=target.backend,
                                             root=pathlib.Path(args.into) if args.into else None,
                                             say=lambda line: print(line, file=sys.stderr))}
    elif args.command == "engines" and args.latest:
      out = {"schema": SCHEMA, "kind": "engines-latest", "file": str(engine_builds.state_file()),
             "engines": engine_update.latest(args.engine or list(engine_builds.rows()))}
    elif args.command == "update":
      out = {"schema": SCHEMA, **self_update.screen(args.skip)}
    elif args.command == "engines":
      out = engines(cached=args.cached)
    elif args.command == "gpu-free":
      out = gpu_free(_target_arg(args.target))
    elif args.command == "mtp":
      try:
        record = vllm_mtp.measure(str(pathlib.Path(args.model).expanduser()), [int(k) for k in args.ks.split(",") if k.strip()],
                                  backend=_target_arg(args.target).backend, out=pathlib.Path(args.out).expanduser(),
                                  tokens=args.tokens, long_tokens=args.long_tokens)
      except vllm_mtp.MtpRefused as exc:  # refused before or between engines: free the GPU or fix the request, then rerun
        sys.stdout.write(pretty_json({"schema": SCHEMA, "kind": "refused", "error": str(exc), "retry": "after the cause is fixed"}))
        return 2
      except RuntimeError as exc:  # an engine failed; its log is in --out
        sys.stdout.write(pretty_json({"schema": SCHEMA, "kind": "error", "error": str(exc), "logs": str(args.out)}))
        return 1
      out = {k: record[k] for k in ("schema", "model", "prompts", "ks", "depths", "measured_at")}
      found = lay.newest_facts_file([pathlib.Path(r).expanduser() for r in args.facts_root], _target_arg(args.target).target_id)
      if found:
        found[0].write_text(json.dumps(rec.facts_with_mtp(found[1], "vllm", record["model"], record), indent=2, sort_keys=True))
      out["cached_in"] = str(found[0]) if found else "not cached: no machine facts for this chip under --facts-root"
    elif args.command == "ceiling":
      target = _target_arg(args.target)
      read = None
      if args.facts_root:
        roots = [pathlib.Path(r).expanduser() for r in args.facts_root]
        read = limit.read_bandwidth(lay.newest_facts(roots, target.target_id))
      kw = _offload_kw(args)
      out = limit.ceiling(_profile_arg(args), target, context=args.context, dtype=args.dtype,
                    peak_gbs=args.peak_gbs, peak_tflops=args.peak_tflops, read=read)
      if args.offload:  # the offload limit beside the all-in-VRAM one: the band and its arithmetic (workflow/offload.py)
        got = recommended(_profile_arg(args), target, engine=args.engine, facts=lay.newest_facts([pathlib.Path(r).expanduser() for r in args.facts_root], target.target_id), **kw)
        out = {**out, "offload": got["offload"], "offload_command": got["command"]}
    elif args.command == "ceilings":
      out = limit.ceilings(_profile_arg(args))
    elif args.command == "runs":
      out = rf.runs(pathlib.Path(args.root).expanduser(), _saved_arg(args), all_places=args.all)
    elif args.command == "plain":
      run = _run_arg(args)
      out = plain.plain_results(rs.results(run), run)
    elif args.command == "speedup-offer":
      out = {"schema": SCHEMA, "kind": "speedup_offer", **speed_up.offer(args.model)}
    elif args.command == "speedup-plain":
      out = {"schema": SCHEMA, **speed_up_plain.log_result(args.log)}
    elif args.command == "speedup-verify":
      out = {"schema": SCHEMA, **speed_up_plain.verify(args.copy)}
    elif args.command == "words":
      out = {"schema": SCHEMA, "kind": "words", **plain.words()}
    elif args.command == "run":
      out = rs.show(_run_arg(args))
    elif args.command == "emit":
      run = _run_arg(args)
      out = gameplan.emit(run)  # Emit validates before writing and invalidates stale plans on failure
      output_run(run)  # the report links gameplan.md once it exists
    elif args.command == "compare-ready":
      out = compare_ready(pathlib.Path(args.run).expanduser(),
                          pipeline.tinygrad_root(args))
    elif args.command == "machine":
      run = pathlib.Path(args.run).expanduser()
      out = pipeline.machine(run, remeasure=args.remeasure, provider=providers.run_provider(run),
                    root=pipeline.tinygrad_root(args))
    elif args.command == "save":
      out = save(pathlib.Path(args.run).expanduser(), pathlib.Path(args.to))
    elif args.command == "saved":
      out = saved(pathlib.Path(args.root).expanduser())
    elif args.command == "clean-work":
      out = clean_work(pathlib.Path(args.work).expanduser(), before=args.before, keep=args.keep)
    elif args.command == "delete":
      out = rf.delete(pathlib.Path(args.root).expanduser(), pathlib.Path(args.run).expanduser())
    elif args.command.startswith("replay-") or args.command in ("diff", "head-to-head"):
      out = replay_command(args)
    else:
      out = rs.results(_run_arg(args))
      code = 0 if out["measured"] else 3
  except (Refused, FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
    sys.stdout.write(pretty_json({"schema": SCHEMA, "kind": "error", "error": str(exc)}))
    return 1
  sys.stdout.write(pretty_json(out))
  return code


if __name__ == "__main__":
  raise SystemExit(main())
