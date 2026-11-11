"""`screen pipeline`: the stages in order, one line each (load, autoscan, analyze, [machine, measure], [ingest],
output, and with --analyze per-role time, the kernel search and the report again); the measure step's plan, which
collector measures here or why none can; and the machine's measured facts. The run folder keeps whatever landed
before a failure. The command line that reaches these is workflow/screen.py."""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
from typing import Any

from boltbeam.cli._common import _parse_ctxs
from boltbeam.collectors import (boltbeam_gemv, capture_settings, engine_builds, gpu_telemetry, llama_bench_decode, metal_native,
                                 multi_gpu, providers, tinygrad_role_time, vendor_capture)
from boltbeam.collectors import capture_scope, engine_families, prefill as pf, replay as rp, session as ss
from boltbeam.collectors.attribution import NotSplitByRole
from boltbeam.collectors.llama_bench_decode import CannotMeasure
from boltbeam.collectors.toolchain import driver_version, record
from boltbeam.search import role_compare, search_stage, tinygrad_fork
from boltbeam.target import gpu as tgpu, vendors
from boltbeam.target.targets import get_target, keep_run_chip
from boltbeam.vocab import Workload
from boltbeam.workflow import analyze_run, autoscan_run, ingest_probe_run, ingest_timing_run, load_run, output_run
from boltbeam.artifacts import progress
from boltbeam.roofline import limit
from boltbeam.workflow import chips, engine_update, layout as lay, repeat_bands as rb, replay as rw
from boltbeam.target.hardware import hardware_profile, this_machine_target
from boltbeam.artifacts.run_io import MACHINE, MEASURE_STATUS, load_manifest, optional, read_json, run_dir, update_manifest, write_json
from boltbeam.collectors.providers import run_provider
from boltbeam.vocab import SCHEMA_TUI as SCHEMA
from boltbeam.workflow.run_folder import SCHEMA_MEASURE_STATUS

# --- the pipeline --------------------------------------------------------------------------------------------

def _write_measure_status(run:str | pathlib.Path, status:str, *, collector:str | None = None, reason:str | None = None,
                          command:str | None = None, probe:str | None = None, probe_reason:str | None = None,
                          provider:str | None = None, layout:str = "one", gpus:int = 1,
                          batches:list[int] | None = None, capture:dict[str, Any] | None = None,
                          details:str | None = None) -> None:
  """probe says whether this collector takes the building-block tests: "measured" or "absent" (with probe_reason).
  Every run is labelled with its engine (provider), the family and fork it is (engine_families.fork_of), its weight
  format, read off the run's model profile, and with the per-role capture the measuring machine had (capture:
  providers.capture_method), a fact of the run.
  A failed status keeps reason to one sentence; details is what stands behind it (an engine's traceback)."""
  out = run_dir(run)
  profile = optional(out, "model_profile.json")
  write_json(out / MEASURE_STATUS, {"schema": SCHEMA_MEASURE_STATUS, "status": status, "collector": collector,
                                    "reason": reason, "command": command, "probe": probe, "probe_reason": probe_reason,
                                    "provider": provider or providers.DEFAULT, "layout": layout, "gpus": gpus,
                                    **engine_families.fork_of(provider or providers.DEFAULT), **_bench_fork(provider),
                                    "weight_format": providers.weight_format(profile) if profile else None,
                                    "batches": sorted({1, *(batches or [])}), "capture": capture, "details": details})
  update_manifest(out, stage="measure", artifacts=[MEASURE_STATUS])


def _bench_fork(provider:str | None) -> dict[str, Any]:
  """The fork a llama.cpp run's llama-bench is the build of ($BOLTBEAM_LLAMA_BENCH at a fork's build), over the
  adapter's root; {} otherwise."""
  bench = llama_bench_decode.find() if (provider or providers.DEFAULT) == llama_bench_decode.PROVIDER else None
  got = llama_bench_decode.fork_of(bench) if bench else None
  return {"engine_family": got["engine_family"], "fork": got["fork"]} if got else {}


def _not_this_machine(here:str | None, target_id:str, run:str, provider:str) -> dict[str, Any]:
  """The refusal when this machine is not the target's: an engine measures on the GPU it runs on."""
  return {"reason": f"this machine is {here or 'no registered GPU'}, not {target_id}",
          "command": f"on the machine with {target_id}: pipeline MODEL --run {run} --target {target_id} "
                     f"--measure auto --provider {provider}"}


def _collectors() -> dict[str, Any]:
  """BoltBeam's own collectors by registry collector_id (data/profiler_capabilities.json): the module that measures."""
  return {"metal-native": metal_native, llama_bench_decode.COLLECTOR_ID: llama_bench_decode}


def measure_plan(target_id:str, model:str, run:str, provider:str = providers.DEFAULT, *,
                 tinygrad_root:pathlib.Path | None = None, gpu=None, layout:str = "one",
                 devs:list[dict[str, Any]] | None = None, measurement:str = "auto") -> dict[str, Any]:
  """Which collector measures this target with this provider on this machine, or why none can. Decided before
  the run starts, so the step count is right from the first line. A GPU another program holds is refused."""
  providers.check(provider)
  try:
    target = get_target(target_id)
  except SystemExit as exc:
    return {"collector": None, "provider": provider, "reason": str(exc), "command": None}
  cid = providers.collector(provider, target.backend)
  devs = devs if devs is not None else tgpu.devices(hardware_profile()["gpu"])
  out = {"collector": cid, "provider": provider, "reason": None, "command": None, "layout": layout, "gpus": max(len(devs), 1),
         # the per-role capture this machine has for the engine: recorded in the run (measure_status.json)
         # with the Setup choice of measurement (in-model, generic, or auto), resolved here and recorded with it
         "capture": {**providers.capture_method(provider, target.backend, measurement),
                     "measurement": providers.resolve_measurement(provider, target.backend, measurement)}}
  offered = {l["id"]: l for l in lay.layouts(len(devs), provider, target.backend)}
  if layout not in offered or not offered[layout]["available"]:
    why = offered[layout]["reason"] if layout in offered else f"{lay.LAYOUTS.get(layout, layout)} needs more than one GPU"
    return out | {"reason": f"{provider} cannot run {lay.LAYOUTS.get(layout, layout)} here: {why}", "command": "pick one GPU"}
  if provider == "vllm" and layout == "row":  # tensor parallel: as many GPUs as the model's heads divide
    if (n := multi_gpu.split_size(model, len(devs))) < 2:
      return out | {"reason": f"the model's head counts {multi_gpu.heads(model)} divide across none of this machine's "
                              f"{len(devs)} GPUs, so vLLM cannot split it", "command": "pick one GPU"}
    out["gpus"], devs = n, devs[:n]
  if provider == tinygrad_role_time.PROVIDER:
    here = this_machine_target()
    if here != target_id and not (target_id == "apple_metal" and target.backend == "Metal" and sys.platform == "darwin"):
      return out | _not_this_machine(here, target_id, run, provider)
    if why := tinygrad_role_time.available(target, tinygrad_root):
      return out | {"reason": why, "command": "set BOLTBEAM_TINYGRAD_ROOT to the tinygrad fork, then measure again"}
  if why := providers.reads(provider, model):
    return out | {"reason": why, "command": f"pick a {' or '.join(providers.FORMATS[provider])} model for {provider}"}
  if provider in providers.DRIVEN:
    here = this_machine_target()
    if here != target_id:
      return out | _not_this_machine(here, target_id, run, provider)
    if why := providers.ENGINES[provider].available(target):
      return out | {"reason": why, "command": f"install {provider}, then measure again"}
  elif provider != tinygrad_role_time.PROVIDER:
    try:
      _collectors()[cid].preflight(target_id, model, run=run)
    except CannotMeasure as exc:
      return out | {"reason": exc.reason, "command": exc.command}
  used = tgpu.used_gpus(layout, devs)
  uuids = {d["uuid"] for d in used if d.get("uuid")} if len(devs) > 1 else None
  check = gpu or providers.gpu_free
  free = check(target.backend, uuids=uuids) if uuids else check(target.backend)
  if not free["free"]:
    return out | {"reason": f"not measured: {free['reason']}", "command": "free the GPU, then measure again"}
  return out


def _parse_batches(text:str | None) -> list[int]:
  """--batch "1,32": the batch sizes to time. Batch 1 is always timed."""
  return sorted({1, *(int(x) for x in (text or "1").split(",") if x.strip())})


def _capture_batch(text:str | None) -> int:
  """The batch the per-role capture runs at: the one batch named beside 1 (the TUI's pick), else 1. A sweep
  (--batch 1,2,4,16) keeps its headline, tie-out and per-role table at batch 1; its batches are the per-batch rows."""
  named = set(_parse_batches(text)) - {1}
  return named.pop() if len(named) == 1 else 1


def _measure_steps(args, plan:dict[str, Any], say=lambda _: None) -> list[tuple[str, Any]]:
  provider = plan.get("provider") or providers.DEFAULT
  layout, gpus = plan.get("layout") or "one", plan.get("gpus") or 1
  batches = _parse_batches(getattr(args, "batch", None))
  capture = plan.get("capture")
  if layout != "one" and getattr(args, "workload", "decode") in ("prefill", "session") and not plan["reason"]:
    plan = {**plan, "reason": f"{args.workload} is measured on one GPU (its per-role capture reads one device's "
                              f"launches); {lay.LAYOUTS.get(layout, layout)} is not built for it", "command": "pick one GPU"}
  if plan["reason"]:
    return [("measure", lambda: _write_measure_status(args.run, "skipped", collector=plan["collector"],
                                                       reason=plan["reason"], command=plan["command"],
                                                       provider=provider, layout=layout, gpus=gpus, batches=batches,
                                                       capture=capture))]
  run = pathlib.Path(args.run)
  args.timing = str(run / "timing_trace.json")
  again = f"pipeline {args.model} --run {args.run} --target {args.target} --measure auto --provider {provider}"

  def guarded(key:str, work, *, probe:str | None, probe_reason:str | None = None, last:bool = True):
    def step():
      try:
        record(run, provider)
        work()
      except Exception as exc:  # the reason must outlive the log: a reopened screen reads it from the run
        _write_measure_status(args.run, "failed", collector=plan["collector"], reason=f"{key}: {exc}",
                              command=again, provider=provider, layout=layout, gpus=gpus, capture=capture,
                              details=failure_details(exc))
        raise
      if last:
        providers.record_build(run, provider)  # the build it ran, against its pin (data/engine_pins.json)
        got = probe_state or {"probe": probe, "reason": probe_reason}
        _write_measure_status(args.run, "measured", collector=plan["collector"], probe=got["probe"],
                              probe_reason=got["reason"], provider=provider, layout=layout, gpus=gpus, batches=batches,
                              capture=capture)
    return (f"measure_{key}", step)

  out_dir = lambda: run_dir(args.run)  # noqa: E731
  backend = get_target(args.target).backend
  is_metal = backend == "Metal"
  # the building-block probes are BoltBeam's own kernels (adapter 0, collectors/boltbeam_gemv.py) on any backend it
  # has a source for: they belong to no provider. A machine that cannot run them (no driver, no compiler, not the
  # run's chip) records the probe absent with its reason; the timing still runs.
  probe_state:dict[str, Any] = {}

  def gemv_probe():
    try:
      (metal_native.measure(out_dir(), only="probe") if is_metal else
       write_json(run / "probe_evidence.json", boltbeam_gemv.collect_probe_evidence(out_dir())))
    except llama_bench_decode.CannotMeasure as exc:
      probe_state.update(probe="absent", reason=f"no building-block probe: {exc.reason}")
      return
    probe_state.update(probe="measured", reason=None)

  probe = [guarded("probe", gemv_probe, probe="measured", last=False)] if backend in boltbeam_gemv.SOURCES else []
  root = tinygrad_root(args)
  if getattr(args, "workload", "decode") in ("prefill", "session", "replay"):  # collectors/prefill.py, session.py, replay.py
    mod = {"prefill": pf, "session": ss, "replay": rp}[args.workload]
    timing = guarded("timing", lambda: mod.measure(out_dir(), provider, root=root, say=say), probe="absent",
                     probe_reason=llama_bench_decode.PROBE_ABSENT)
    return [timing]
  if provider == tinygrad_role_time.PROVIDER:
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

def _engine_settings(args) -> dict | None:
  """The engine settings a run requests, from the command line; None when it requests none. A vLLM run split by rows
  requests its tensor-parallel size: the most of this machine's GPUs the model's heads divide (multi_gpu.split_size)."""
  settings = {k: getattr(args, k, None) for k in capture_settings.ASKED}
  if getattr(args, "provider", None) == "vllm" and getattr(args, "layout", None) == "row":
    settings["tensor_parallel_size"] = multi_gpu.split_size(pathlib.Path(args.model).expanduser(),
                                                            len(tgpu.devices(hardware_profile()["gpu"])))
  return {k: v for k, v in settings.items() if v is not None} or None

def _load(args) -> dict[str, Any]:
  """Step 0: the run folder. A replay also stages its trace and the server command (collectors/replay.stage), and its
  context is the prefix the trace reaches."""
  contexts = _parse_ctxs(args.ctxs) if args.ctxs else None
  if args.workload == Workload.REPLAY.value:
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    contexts = (rp.stage(run, getattr(args, "replay_trace", None), getattr(args, "server_argv", None), getattr(args, "replay_pairs", None)),)
  if getattr(args,'forced_replay',False) or (args.workload=='replay' and args.provider=='vllm'
      and getattr(args,'analyze',False) and vendor_capture.windowed(get_target(args.target).backend)):
    if args.workload!='replay' or args.provider!='vllm':raise ValueError('--forced-replay requires vLLM replay')
    (pathlib.Path(args.run)/'forced_replay_request.json').write_text('{}')
  result = load_run(args.model, args.run, target_id=args.target, model_id=args.id, workload=args.workload,
                  contexts=contexts, target_source="user", provider=getattr(args, "provider", None),
                  engine_settings=_engine_settings(args))
  keep_run_chip(pathlib.Path(args.run), args.target)
  if getattr(args, "bands", None):
    evidence = read_json(args.bands)
    if evidence.get("schema") != capture_scope.SCHEMA:
      raise ValueError("unsupported repeat band schema")
    write_json(pathlib.Path(args.run) / capture_scope.ARTIFACT, evidence)
  return result


def _say_to(out):
  """One stage line at a time to the stream a screen tails, flushed so the reader sees it as it lands."""
  def say(text:str) -> None:
    out.write(text + "\n")
    out.flush()
  return say


def tinygrad_root(args) -> pathlib.Path | None:
  """--tinygrad-root as a folder; None leaves the fork where the stages look by default."""
  root = getattr(args, "tinygrad_root", None)
  return pathlib.Path(root).expanduser() if root else None


def pipeline(args, out=sys.stdout) -> int:
  """Run the stages in order and say so, one line each. The run folder keeps whatever landed before a failure."""
  say = _say_to(out)
  steps = [
    ("load", lambda: _load(args)),
    ("autoscan", lambda: autoscan_run(args.run, providers=())),
    ("analyze", lambda: analyze_run(args.run)),
  ]
  analyze_all = getattr(args, "analyze", False)
  if analyze_all:
    args.measure = "auto"
  plan, root = None, tinygrad_root(args)
  if getattr(args, "measure", "none") == "auto" and not (args.probe or args.timing):
    plan = measure_plan(args.target, str(pathlib.Path(args.model).expanduser().resolve()), args.run,
                        getattr(args, "provider", None) or providers.DEFAULT, tinygrad_root=root,
                        layout=getattr(args, "layout", None) or "one",
                        measurement=_role_time_arg(args))
    if analyze_all and not plan["reason"]:  # the machine's measured facts, reused when this machine has them
      steps.append(("machine", lambda: machine(pathlib.Path(args.run), root=root, provider=getattr(args, "provider", None),
                                               hold=getattr(args, "hold_machine", None))))
    steps += _measure_steps(args, plan, say=say)
  if args.probe:  # a CUDA probe with no GPU on this machine wrote no evidence: nothing to ingest
    steps.append(("ingest_probe", lambda: ingest_probe_run(args.run, args.probe)
                  if pathlib.Path(args.probe).exists() else None))
  if args.timing:
    steps.append(("ingest_timing", lambda: ingest_timing_run(args.run, args.timing)))
  if args.probe or args.timing:
    steps.append(("analyze", lambda: analyze_run(args.run)))  # the plan and the report read the new evidence
  # A full audit must collect role evidence before publishing a report. The
  # strict evidence gate is expected to reject the intermediate weight-only
  # model for architectures whose executed work has not been captured yet.
  if not (analyze_all and plan and not plan['reason']):
    steps.append(("output", lambda: output_run(args.run)))
  # the per-role capture this machine has, as the plan recorded it (an older plan without it is asked now)
  can_time = plan and (plan.get("capture") or providers.capture_method(plan["provider"], get_target(args.target).backend)).get("method")
  if analyze_all and plan and not plan["reason"]:  # one press: per-role time, same engine, search, report again
    if mod := workload_role_time(getattr(args, "workload", "decode")):  # captured in the model, attributed
      steps.append(("role_time", lambda: mod.role_time(pathlib.Path(args.run), plan["provider"], root=root, say=say)))
    elif can_time:
      steps.append(("role_time", lambda: providers.role_time(pathlib.Path(args.run), plan["provider"], root=root, say=say,
                                                             batch=_capture_batch(getattr(args, "batch", None)),
                                                             step=progress.report, measurement=_role_time_arg(args))))
    if getattr(args, "no_search", False) or getattr(args, "workload", "decode") == "replay":
      why = "skipped with --no-search" if getattr(args, "no_search", False) else "a replay has no per-role kernel search"
      steps.append(("search", lambda: search_stage.skip(pathlib.Path(args.run), why, say)))
    elif not can_time:
      steps.append(("search", lambda: search_stage.skip(pathlib.Path(args.run), "no per-role time on this machine, "
                                                        "so there is no kernel to compare with", say)))
    else:  # search faster kernels per role where the search exists
      backend = get_target(args.target).backend
      steps.append(("search", lambda: search_stage.search_stage(pathlib.Path(args.run), backend=backend,
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
  run_path = pathlib.Path(args.run)
  if (code := _run_stages(steps, ids, run_path, get_target(args.target).backend, where, plan, say)) != 0:
    return code
  if (report := run_path / "report.html").is_file():  # the main report, said before the run is done
    say(f"pipeline report: {report}")
  say(f"pipeline done: {args.run}")
  return 0


class _Prefixed:
  """A writer that marks every line, so a screen reading the run's own stage lines does not read these."""
  def __init__(self, out, prefix:str) -> None:
    self.out, self.prefix = out, prefix

  def write(self, text:str) -> None:
    self.out.write("".join(self.prefix + line for line in text.splitlines(True)))

  def flush(self) -> None:
    self.out.flush()


def pipeline_current(args, out=sys.stdout) -> int:
  """The pipeline on this machine's current build of the engine (collectors/engine_builds.py). With --upgrade-engines,
  `engines --update` for this engine runs first. When a newer build waits (pending) and this is an --analyze run, the
  previous build runs once into RUN-was-<version>, then the new build into RUN; the two are compared (workflow/diff),
  the comparison is saved in RUN/engine_upgrade.json, and the new build becomes current. A failed new-build run leaves
  current as it was."""
  say = _say_to(out)
  provider = getattr(args, "provider", None) or providers.DEFAULT
  if provider not in engine_builds.rows():
    return pipeline(args, out)
  if getattr(args, "update_engines", False):
    engine_update.update([provider], backend=get_target(args.target).backend, say=say)
  e = engine_builds.entry(engine_builds.load(), provider)
  cur, new = e.get("current"), e.get("pending")
  if not (new and cur and getattr(args, "analyze", False)):
    if new:
      say(f"engines {provider}: {new['version']} waits for its comparison (an --analyze run); this run uses {(cur or {}).get('version')}")
    with engine_builds.using(cur):
      return pipeline(args, out)
  if overridden := engine_builds.env_for(new)[1]:  # the user's own build wins: no comparison of builds not used
    say(f"engines {provider}: {', '.join(overridden)} set, so {new['version']} is not compared on this run")
    with engine_builds.using(cur):
      return pipeline(args, out)
  before = argparse.Namespace(**vars(args))
  before.run = str(pathlib.Path(args.run).with_name(f"{pathlib.Path(args.run).name}-was-{cur.get('version') or 'previous'}"))
  say(f"engines {provider}: {new['version']} is new here; the previous build {cur.get('version')} runs once first, into {before.run}")
  with engine_builds.using(cur):
    failed = pipeline(before, _Prefixed(out, "previous build: "))
  with engine_builds.using(new):
    if (code := pipeline(args, out)) != 0:
      return code
  record = {"before": cur.get("version"), "after": new["version"], "before_run": pathlib.Path(before.run).name}
  try:
    if failed:
      raise ValueError("the previous build's run failed (its lines are marked 'previous build:')")
    d = rw.diff(pathlib.Path(before.run), pathlib.Path(args.run), bands=read_json(args.bands) if getattr(args, "bands", None) else None)
    record.update(diff=d, lines=rw.diff_lines(d))
  except ValueError as exc:  # DiffRefused is a ValueError: the refusal is the comparison's record
    record["refused"] = str(exc)
  write_json(pathlib.Path(args.run) / engine_builds.UPGRADE_FILE, record)
  state = engine_builds.load()
  engine_builds.compared(state, provider, {k: record.get(k) for k in ("before_run", "refused")} | {"run": str(args.run)})
  if not record.get("refused"):  # keep current and previous; nothing is deleted after a comparison that failed
    for gone in engine_update.prune(state, provider):
      say(f"engines {provider}: deleted {gone['version']} ({gone['folder']}), older than the previous build")
  engine_builds.save(state)
  output_run(args.run)  # the report and summary name the comparison
  say(f"engines {provider}: {new['version']} is current; " + (engine_builds.comparison(pathlib.Path(args.run)) or {}).get("words", ""))
  return 0


def pipeline_repeated(args, out=sys.stdout) -> int:
  """--repeats N: the same setup N times, then its repeat band (workflow/repeat_bands.py). n is fixed first, in
  RUN/repeat_plan.json; RUN is the first run and RUN-repeat-2 ... RUN-repeat-N the others, each its whole pipeline
  after a `repeat i/N: NAME` line; the band goes to RUN/repeat_bands.json, where RUN's results read it. A run that
  fails stops the repeats; a band the runs cannot make (repeat_bands.calibrate) is said and not written."""
  say = _say_to(out)
  run = pathlib.Path(args.run)
  try:
    fixed = rb.plan(args.repeats, run_dir(run) / rb.PLAN)
  except ValueError as exc:
    say(f"repeat band: failed: {exc}")
    return 2
  runs = [run, *(run.with_name(f"{run.name}-repeat-{i}") for i in range(2, args.repeats + 1))]
  for i, path in enumerate(runs, 1):
    say(f"repeat {i}/{len(runs)}: {path.name}")
    one = argparse.Namespace(**{**vars(args), "run": str(path), "update_engines": i == 1 and getattr(args, "update_engines", False),
                                "hold_machine": runs[0] if i > 1 else None})
    if (code := pipeline_current(one, out)) != 0:
      return code
  try:
    got = rb.save(runs, run / capture_scope.ARTIFACT, fixed)
  except ValueError as exc:
    say(f"repeat band: failed: {exc}")
    return 1
  output_run(args.run)  # the report reads the band too
  say(f"repeat band: done: n = {got['n']}, a row reads moved by chance {got['false_move_rate']:.0%} of the time")
  return 0


def _run_stages(steps, ids, run_path:pathlib.Path, backend:str, where:str, plan, say) -> int:
  """Run the pipeline's stages in order: 1 at the first that fails, its sentence said. A stage that refuses to split
  the run by role (attribution.NotSplitByRole) is skipped, said and recorded; the stages after it still run (search
  says it has no kernel to compare), so the run's measured step and its report stand."""
  times:dict[str, float] = {}
  not_split = None
  for (key, step), sid in zip(steps, ids):
    if key == "search" and not_split:  # no roles, so no kernel to compare: said, as with no per-role time here
      step = lambda: search_stage.skip(run_path, "the run is not split by role, so there is no kernel to compare with", say)  # noqa: E731
    say(f"stage {key}: start")
    progress.listen(lambda done, total, key=key: say(f"stage {key}: progress {done}/{total}"))
    began = time.monotonic()
    try:
      with gpu_telemetry.measurement_clock(run_path, key, backend=backend):
        step()
    except NotSplitByRole as exc:  # the run stands whole: recorded, said, and the report still written
      not_split = str(exc)
      _not_split(run_path, not_split, exc.key)
      say(f"stage {key}: skipped: {exc}")
    except Exception as exc:  # application boundary: the stage's own message is the fact the reader needs
      for line in failure_lines(key, exc):
        say(line)
      return 1
    finally:
      progress.listen(lambda done, total: None)
    times[sid] = round(time.monotonic() - began, 2)
    progress.save(run_path, where, times, finished=False)
    say(f"stage {key}: done")
  progress.save(run_path, where, times, finished=plan is not None)
  return 0


def _not_split(run:pathlib.Path, why:str, key:str = "not_split") -> None:
  """measure_status.json says the run was not split by role, and why: the report shows it beside the whole step."""
  status = run / MEASURE_STATUS
  m = json.loads(status.read_text()) if status.exists() else {}
  m.setdefault("capture", {})["reason"] = f"not split by role: {why}"
  m["not_split_by_role"] = why  # its own field: the capture's method is still recorded, so _missing must not need it absent
  m["not_split_key"] = key  # the reason's key (attribution.NotSplitByRole.key): its plain sentence is data/plain_words.json's
  write_json(status, m)


def failure_details(exc:BaseException) -> str | None:
  """What stands behind a stage's one-line failure: an engine's own traceback (tinygrad_role_time.DecodeFailed
  carries it as details). None when the exception's message is all there is."""
  details = getattr(exc, "details", None)
  return str(details) if details else None


def failure_lines(key:str, exc:BaseException) -> list[str]:
  """The log lines for a failed stage: one "failed:" line with the sentence a screen shows in its box, then one
  "detail:" line per line of what stands behind it (the traceback), which a screen keeps behind "Show details"."""
  out = [f"stage {key}: failed: {exc}"]
  out += [f"stage {key}: detail: {line}" for line in (failure_details(exc) or "").splitlines()]
  return out


def _role_time_arg(args) -> str:
  """--role-time in-model|generic|auto as the measurement id: in_model, generic or auto."""
  return str(getattr(args, "role_time", None) or "auto").replace("-", "_")


def workload_role_time(workload:str | None):
  """The collector that times a non-decode workload's roles (one prefill per length; one turn at the focus depth;
  the replay's picked turns and its deepest decode), or None for decode, which providers.role_time times."""
  return {"prefill": pf, "session": ss, "replay": rp}.get(workload or "decode")


def role_time(args, out=sys.stdout) -> int:
  """Time every role of the run's workload with the engine it was measured with, then refresh the report."""
  say = _say_to(out)
  run = pathlib.Path(args.run).expanduser()
  try:
    provider = getattr(args, "provider", None) or run_provider(run)
    say(f"role-time: start {provider}")
    root = tinygrad_root(args)
    if mod := workload_role_time(optional(run, "run_manifest.json").get("workload")):
      mod.role_time(run, provider, root=root, say=say)
    else:
      providers.role_time(run, provider, root=root, batch=int(getattr(args, "batch", None) or 1), say=say,
                          measurement=_role_time_arg(args))
    output_run(run)
  except Exception as exc:  # application boundary: the reason is the fact the reader needs
    say(f"role-time failed: {exc}")
    return 1
  say("role-time: done")
  return 0


def compare(args, out=sys.stdout) -> int:
  """Compare kernels per role for one run, one line per step, then refresh the report."""
  say = _say_to(out)
  run = pathlib.Path(args.run).expanduser()
  root = tinygrad_root(args)
  try:
    role_compare.compare_run(run, root=root, say=say)
    output_run(run)
  except Exception as exc:  # application boundary: the reason is the fact the reader needs
    say(f"compare failed: {exc}")
    return 1
  return 0

# tinygrad's device name per backend that tinygrad's per-GPU read probe runs on (target/vendors.py)
DEVICE_PREFIX = {v.backend: v.tinygrad_device for v in vendors.VENDORS.values() if v.tinygrad_per_gpu}

def machine(run:pathlib.Path, *, remeasure:bool = False, root:pathlib.Path | None = None,
            provider:str | None = None, hold:pathlib.Path | None = None) -> dict[str, Any]:
  """This machine's GPUs, their measured read bandwidth and the measured copies between them (workflow/layout.py),
  cached in the run and reused from the newest run of the same GPUs unless remeasure. An engine that splits by rows
  with its own all-reduce has that join measured under its python (collectors/multi_gpu.join)."""
  if hold is not None:  # a repeat band holds the setup fixed: its runs share the first run's facts, never re-read
    facts = {**json.loads((hold / MACHINE).read_text()), "reused": True,
             "reused_because": f"held fixed for the repeat band: the facts of {hold.name}"}
    write_json(run / MACHINE, {k: v for k, v in facts.items() if k not in ("reused", "reused_because")})
    return {"schema": SCHEMA, "kind": "machine", **facts}
  devs = tgpu.devices(hardware_profile()["gpu"])
  target = get_target(load_manifest(run).get("target_id"))
  row = vendors.BY_BACKEND.get(target.backend)
  native_read = None
  if row is not None and row.bridge is not None and row.platform in (None, platform.system()):
    native_read = lambda d: chips._probes(row.vendor, d)[0]()  # noqa: E731  the vendor's own read probe on that card
  driver = driver_version(row.vendor) if row is not None else None
  cached = None if remeasure else lay.cached_machine(run, devs)
  stale = lay.stale_reason(cached, driver=driver, fresh_read=(lambda g: native_read(g)["read_gbs"]) if native_read else None) \
    if cached is not None else None
  if stale:
    cached = None
  if cached is not None and any((now := limit.profile_read(g)) is not None and now[0] != g.get("read_gbs") for g in cached.get("gpus") or []):
    cached = None  # the chip profile was measured again since: one measurement, the profile's (profile_read)
  joins = provider in lay.ENGINE_JOINS and len(devs) > 1 and providers.ENGINES[provider].python()
  if cached is not None and joins and not (cached.get("join") or {}).get("joins"):
    cached = None  # the engine's join can be measured now and was not
  if cached is not None:
    facts = {**cached, "reused": True, "reused_because": "same driver, and a fresh read lands within the runs that made it"
             if native_read else "same driver; no native read here to verify it"}
  else:
    probe = None
    fork = pathlib.Path(root) if root else tinygrad_fork.default_fork_root()
    if tinygrad_fork.readiness(fork)["ready"] and target.backend in DEVICE_PREFIX:
      probe = lambda args, env: lay.run_probe(str(tinygrad_fork.fork_python(fork)), str(fork), args, env)  # noqa: E731
    topology = multi_gpu.topology(target.backend) if len(devs) > 1 else {}
    facts = lay.measure_machine(devs, backend=target.backend, device_prefix=DEVICE_PREFIX.get(target.backend),
                                probe=probe, topology=topology, native_read=native_read,
                                fallback=lambda tid: get_target(tid).memory_bandwidth_gbs if tid else None,
                                profile_read=limit.profile_read,
                                join=(lambda used: multi_gpu.join(joins, used)) if joins else None,
                                read_runs=chips.READ_RUNS)
    facts.update(reused=False, driver=driver, measured_because=stale or ("asked to remeasure" if remeasure else "no cached facts for these GPUs"))
  write_json(run / MACHINE, {k: v for k, v in facts.items() if k not in ("reused", "reused_because")})
  return {"schema": SCHEMA, "kind": "machine", **facts}
