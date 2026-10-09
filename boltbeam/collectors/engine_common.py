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
  """The engine's program: $env_var when set (it must exist), else the first name on PATH."""
  env = os.environ if env is None else env
  if env.get(env_var):
    path = pathlib.Path(env[env_var]).expanduser()
    return str(path) if path.is_file() else None
  return next((p for p in (shutil.which(n) for n in names) if p), None)


def python_with(env_var:str, entry:str, env:dict[str, str] | None = None) -> str | None:
  """The python an engine is installed in: $env_var, else the python next to the engine's entry point on PATH
  (a venv's bin folder holds both)."""
  env = os.environ if env is None else env
  if env.get(env_var):
    path = pathlib.Path(env[env_var]).expanduser()
    return str(path) if path.is_file() else None
  exe = shutil.which(entry)
  if not exe:
    return None
  py = pathlib.Path(exe).parent / "python"
  return str(py) if py.is_file() else None


def venv_env(python:str) -> dict[str, str]:
  """The environment an activated venv gives: its bin folder first on PATH, so the tools it installed (ninja for
  FlashInfer's run-time kernel builds) are found."""
  venv = pathlib.Path(python).parent.parent
  return {"VIRTUAL_ENV": str(venv), "PATH": f"{pathlib.Path(python).parent}{os.pathsep}{os.environ.get('PATH', '')}"}


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
               timeout_s:float = 3600.0) -> list[dict[str, Any]]:
  """Run a driver and return its JSON lines. A failure raises with the end of what it printed."""
  proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env={**os.environ, **(env or {})})
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
                       cwd:pathlib.Path | None = None, flush_ms:int | None = None) -> dict[str, Any]:
  """An engine's kernels per role, captured from outside and attributed by bytes and count: the path llama.cpp
  takes. Two driver runs at one context and batch, N tokens and 1 (--once), subtracted per kernel: what is left
  is N - 1 decode steps. The captured run's own time per step is the difference of the two runs' printed decode
  times over N - 1."""
  from boltbeam.collectors import attribution, vendor_capture
  raw = run / "kernel_compare" / f"{provider.replace('.', '_').replace('-', '_')}_capture"
  got, secs = {}, {}
  for name, n in (("long", tokens), ("short", 1)):
    got[name] = vendor_capture.capture(target.backend, argv_for(n), raw / name, cwd=cwd, env=env, flush_ms=flush_ms)
    printed = [r for r in json_lines(vendor_capture.program_output(raw / name)) if "seconds" in r]
    secs[name] = printed[-1]["seconds"] if printed else None
  window, overlap = attribution.shared_window(got["long"], got["short"])
  token_ms = ((secs["long"] - secs["short"]) * 1000.0 / (tokens - 1)
              if secs["long"] is not None and secs["short"] is not None else None)
  return attribution.provider_trace(run, window, provider=provider, method=vendor_capture.plan(target.backend)["method"],
                                    model_id=model_id, target=target, context=context, tokens=tokens - 1, out=trace,
                                    token_ms=token_ms, long_kernels=len(got["long"]), overlap=overlap, batch=batch)
