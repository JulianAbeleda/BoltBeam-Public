"""What the engines that run as a separate program share: finding them, running their driver, and capturing it.

An engine here is a runtime BoltBeam does not import (vLLM, Ollama, TensorRT-LLM). Each one has a small driver
(boltbeam/runtime/*_decode_drive.py) that its own python runs by path. A driver prints one JSON line per measured
point, and in capture mode (--once N) one line {"tokens", "seconds"}. Everything after that is shared with
llama.cpp and tinygrad: the vendor capture (vendor_capture.py) and the attribution by bytes and count
(attribution.py).

One point of a decode is a context C and a batch B: B streams each decoding one token per step, at context C.

    step_ms        time of one decode step (B tokens, one per stream)
    tok_s_stream   1000 / step_ms: what one user sees
    tok_s_total    B * 1000 / step_ms: what the GPU delivers
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any, Callable

RUNTIME = pathlib.Path(__file__).resolve().parents[1] / "runtime"


def find_program(env_var:str, names:tuple[str, ...], env:dict[str, str] | None = None) -> str | None:
  """The engine's program: $env_var when set (it must exist), else the first name on PATH, else what the engine
  scan saved for that env var (collectors/engine_scan.py)."""
  from boltbeam.collectors import engine_scan
  env = os.environ if env is None else env
  if env.get(env_var):
    path = pathlib.Path(env[env_var]).expanduser()
    return str(path) if path.is_file() else None
  return next((p for p in (shutil.which(n) for n in names) if p), None) or engine_scan.saved(env_var)


def python_with(env_var:str, entry:str, env:dict[str, str] | None = None) -> str | None:
  """The python an engine is installed in: $env_var, else the python next to the engine's entry point on PATH
  (a venv's bin folder holds both), else what the engine scan saved for that env var."""
  from boltbeam.collectors import engine_scan
  env = os.environ if env is None else env
  if env.get(env_var):
    path = pathlib.Path(env[env_var]).expanduser()
    return str(path) if path.is_file() else None
  exe = shutil.which(entry)
  if not exe:
    return engine_scan.saved(env_var)
  py = pathlib.Path(exe).parent / "python"
  return str(py) if py.is_file() else engine_scan.saved(env_var)


def venv_env(python:str) -> dict[str, str]:
  """The environment an activated venv gives: its bin folder first on PATH, so the tools it installed (ninja for
  FlashInfer's run-time kernel builds) are found. Python workers inherit the build limit hook."""
  venv = pathlib.Path(python).parent.parent
  runtime = str(RUNTIME)
  existing_pythonpath = os.environ.get("PYTHONPATH", "")
  from boltbeam.runtime.cuda_device import find_nvcc
  nvcc = find_nvcc()
  bins = [str(pathlib.Path(python).parent)]
  if nvcc:
    bins.append(str(pathlib.Path(nvcc).parent))
  bins.extend(os.environ.get('PATH', '').split(os.pathsep))
  # the checkout too: an engine's own subprocesses (vLLM's engine core) import boltbeam's hooks, and an engine's venv
  # need not have boltbeam installed (a TUI run from a checkout found no module 'boltbeam')
  checkout = str(RUNTIME.parents[1])
  return {"VIRTUAL_ENV": str(venv), "PATH": os.pathsep.join(dict.fromkeys(bins)),
          "PYTHONPATH": os.pathsep.join([runtime, checkout, *([existing_pythonpath] if existing_pythonpath else [])]),
          "BOLTBEAM_BUILD_LIMIT": "1"}


def missing_headers(python:str) -> str | None:
  """The Python.h this python lacks (its full path), or None when present. Triton compiles against it at startup;
  without it vLLM failed to inspect its model class (a Vast RTX 3090, 2026-11-08)."""
  code = ("import os,sys,sysconfig; h=os.path.join(sysconfig.get_path('include'),'Python.h'); "
          "print(h, sys.version_info[0], sys.version_info[1]); sys.exit(not os.path.exists(h))")
  try:
    proc = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
  except (OSError, subprocess.SubprocessError) as exc:
    return f"{python} did not run: {exc}"
  return None if proc.returncode == 0 else proc.stdout.strip()


def has_module(python:str, module:str) -> bool:
  """True when that python can import the module (found by spec only: importing vLLM takes seconds)."""
  try:
    proc = subprocess.run([python, "-c", f"import importlib.util,sys; sys.exit(importlib.util.find_spec({module!r}) is None)"],
                          capture_output=True, timeout=60)
  except (OSError, subprocess.SubprocessError):
    return False
  return proc.returncode == 0


def json_lines(text:str) -> list[dict[str, Any]]:
  out = []
  for line in text.splitlines():
    line = line.strip()
    if line.startswith("{"):
      try:
        out.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return out


def run_driver(argv:list[str], *, env:dict[str, str] | None = None, log:pathlib.Path | None = None,
               timeout_s:float = 3600.0, cwd:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Run a driver and return its JSON lines. A failure raises with the end of what it printed.

  An engine that refuses to compile inside a measurement (build_jobs.PrebuildRequired: a new model's FlashInfer
  kernel) names the module; it is built outside the measurement, bounded (runtime/prebuild_cuda.py), and the driver
  started again, for every engine, up to PREBUILD_TRIES modules."""
  from boltbeam.runtime import prebuild_cuda as pc
  built = []
  while True:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env={**os.environ, **(env or {})}, cwd=cwd)
    need = pc.required(proc.stdout + proc.stderr) if proc.returncode else None
    if need is None or need in built or len(built) >= pc.PREBUILD_TRIES or not need.is_file():
      break
    built.append(need)
    pc.prebuild(need, report_path=need.parent / "boltbeam-build-profile.json", ninja=pc.ninja_beside(argv[0]))
  if log is not None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
  rows = json_lines(proc.stdout)
  if proc.returncode != 0 or not rows:
    raise RuntimeError(f"{pathlib.Path(argv[1] if len(argv) > 1 else argv[0]).name} exited {proc.returncode}: "
                       f"{(proc.stderr or proc.stdout).strip()[-400:]}")
  return rows


def point(context:int, batch:int, step_ms:float, **extra:Any) -> dict[str, Any]:
  """One measured point in the one shape every engine reports."""
  return {"context": int(context), "batch": int(batch), "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms,
          "tok_s_total": batch * 1000.0 / step_ms, **extra}


def captured_role_time(run:pathlib.Path, *, provider:str, trace:str, argv_for:Callable[[int], list[str]], target,
                       model_id:str, context:int, tokens:int, batch:int = 1, env:dict[str, str] | None = None,
                       cwd:pathlib.Path | None = None, flush_ms:int | None = None, profiler_range:bool = False,
                       warm:bool = True) -> dict[str, Any]:
  """An engine's kernels per role, captured from outside and attributed by bytes and count: the path llama.cpp
  takes. Two driver runs at one context and batch, N tokens and 1 (--once), subtracted per kernel: what is left
  is N - 1 decode steps. The captured run's own time per step is the difference of the two runs' printed decode
  times over N - 1. warm runs the 1-token driver once uncaptured first: an engine's start-up can fill on-disk caches
  (vLLM's Triton autotune, a JIT) in whichever capture runs first, and the subtraction needs both to start up alike.
  A batch-1 capture over the capture-overhead ceiling is repeated: tie_out.REPLICAS fresh pairs in all, each attributed
  alike, are the repeats its kernel rows stand by (tie_out.over_ceiling). Within the ceiling, one pair."""
  from boltbeam.collectors import attribution, vendor_capture
  from boltbeam.workflow.common import run_gpu
  from boltbeam.workflow.tie_out import REPLICAS, capture_over_ceiling
  raw = run / "kernel_compare" / f"{provider.replace('.', '_').replace('-', '_')}_capture"
  if warm:
    run_driver(argv_for(1), env=env, log=raw / "warm.log", cwd=cwd)
  def pair(folder:pathlib.Path) -> dict[str, Any]:
    got, secs, matrices, engine = {}, {}, None, {}
    for name, n in (("long", tokens), ("short", 1)):
      got[name] = vendor_capture.capture(target.backend, argv_for(n), folder / name, cwd=cwd, env=env, flush_ms=flush_ms, device=run_gpu(run), **({"profiler_range": True} if profiler_range else {}))
      printed = [r for r in json_lines(vendor_capture.program_output(folder / name)) if "seconds" in r]
      secs[name] = printed[-1]["seconds"] if printed else None
      matrices = matrices or (printed[-1].get("matrices") if printed else None)  # the engine's packed GEMMs, if it says
      engine.update((printed[-1].get("engine") or {}) if printed else {})
    window, overlap = attribution.shared_window(got["long"], got["short"])
    return {"window": window, "overlap": overlap, "long": len(got["long"]), "matrices": matrices, "engine": engine,
            "token_ms": ((secs["long"] - secs["short"]) * 1000.0 / (tokens - 1) if secs["long"] is not None and secs["short"] is not None else None)}
  pairs = [pair(raw)]
  attributed = lambda i, out, **kw: attribution.provider_trace(  # noqa: E731
    run, pairs[i]["window"], provider=provider, method=vendor_capture.plan(target.backend)["method"], model_id=model_id, target=target,
    context=context, tokens=tokens - 1, out=out, long_kernels=pairs[i]["long"], overlap=pairs[i]["overlap"], batch=batch,
    matrices=pairs[i]["matrices"], engine=pairs[i]["engine"], **kw)
  replicas = None
  if batch == 1 and capture_over_ceiling(run, context, provider, pairs[0]["token_ms"]):
    pairs += [pair(raw / f"pair-{i + 1}") for i in range(1, REPLICAS)]
    replicas = attribution.fresh_replicas(attributed, [raw / f"pair-{i + 1}" for i in range(REPLICAS)], [p["token_ms"] for p in pairs])
  return attributed(0, trace, token_ms=pairs[0]["token_ms"], replicas=replicas)
