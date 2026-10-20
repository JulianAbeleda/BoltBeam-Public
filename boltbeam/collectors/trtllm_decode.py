"""TensorRT-LLM as a provider: whole-step decode and per-role time of a Hugging Face model in TensorRT-LLM.

TensorRT-LLM is not imported here. Its own python runs boltbeam/runtime/trtllm_decode_drive.py, found as
$BOLTBEAM_TRTLLM_PYTHON, else the python next to `trtllm-bench` on PATH (the venv it is installed in).

Whole step: its LLM API at a fixed context and batch, N tokens minus 1 token over N - 1, the same rule as vLLM.
Per role: the driver under nsys, N and 1 tokens, subtracted and attributed by bytes and count.

available() also asks the installed TensorRT-LLM whether it runs on this GPU (a probe in the driver), so a wheel
built without the GPU's architecture is greyed with that reason, not discovered in the middle of a run.
"""
from __future__ import annotations

import os
import json
import time
import pathlib
import subprocess
from typing import Any

from boltbeam.collectors import engine_common as ec

PROVIDER = "tensorrt-llm"
PROVIDER_ID = "tensorrt-llm/llm-api-decode"
TRACE = "trtllm_timing_trace.json"
FORMATS = ("safetensors",)
BACKENDS = ("CUDA",)
ENV = "BOLTBEAM_TRTLLM_PYTHON"
BACKEND_ENV = "BOLTBEAM_TRTLLM_BACKEND"  # pytorch (TensorRT-LLM's default) or tensorrt (a built engine)
# a TensorRT engine built for the GPU: TensorRT-LLM's PyTorch backend builds FlashInfer kernels at start, which
# fails where the system CUDA lacks curand (the 5090 box); the engine path needs no run-time build
DEFAULT_BACKEND = "tensorrt"
DRIVER = ec.RUNTIME / "trtllm_decode_drive.py"
TOKENS, REPS = 64, 3
ROLE_TOKENS = 32
KV_ELEMENT = (2, "TensorRT-LLM's default KV cache dtype (the model's bf16)")
_probe_cache: dict[str, str | None] = {}


def python(env:dict[str, str] | None = None) -> str | None:
  return ec.python_with(ENV, "trtllm-bench", env)


def probe(py:str) -> str | None:
  """Why this TensorRT-LLM cannot run on GPU 0, or None. One subprocess per python, kept for the process."""
  if py in _probe_cache:
    return _probe_cache[py]
  try:
    proc = subprocess.run([py, str(DRIVER), "--probe"], capture_output=True, text=True, timeout=300)
    rows = ec.json_lines(proc.stdout)
    why = rows[-1].get("reason") if rows else f"the probe exited {proc.returncode}: {(proc.stderr or '').strip()[-300:]}"
  except (OSError, subprocess.SubprocessError) as exc:
    why = f"the probe could not run: {exc}"
  _probe_cache[py] = why
  return why


def cached_probe(py:str) -> str | None:
  """Picker-only cache; actual runs still call the live compatibility probe."""
  from boltbeam.collectors import engine_scan
  path = engine_scan.engines_file().with_name("trtllm-probe.json")
  root = pathlib.Path(py).parent.parent
  paths = [pathlib.Path(py), DRIVER, engine_scan.engines_file(),
           *sorted(root.glob("lib/python*/site-packages/*dist-info"))]
  stamps = []
  for item in paths:
    try:
      stat = item.stat()
      stamps.append([str(item), stat.st_mtime_ns, stat.st_size])
    except OSError:
      stamps.append([str(item), None, None])
  driver = pathlib.Path('/proc/driver/nvidia/version')
  key = {"files": stamps, "driver": driver.read_text() if driver.exists() else None,
         "env": {k:v for k,v in os.environ.items() if k.startswith(('CUDA_', 'NVIDIA_', 'BOLTBEAM_TRTLLM'))}}
  try:
    saved = json.loads(path.read_text())
    if (saved.get('key') == key and 0 <= time.time()-saved['time'] < 86400
        and 'reason' in saved and (saved['reason'] is None or isinstance(saved['reason'], str))):
      return saved['reason']
  except (OSError, ValueError, KeyError, TypeError, AttributeError):
    pass
  _probe_cache.pop(py, None)
  reason = probe(py)
  # Timeouts and crashes are transient; only cache structured probe results.
  if reason is None or not reason.startswith(('the probe could not run:', 'the probe exited ')):
    try:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps({'key':key, 'time':time.time(), 'reason':reason}))
    except OSError:
      pass  # Cache failure must not prevent engine selection.
  return reason


def available(target, env:dict[str, str] | None = None, *, cached:bool = False) -> str | None:
  if target.backend not in BACKENDS:
    return f"TensorRT-LLM runs on NVIDIA GPUs only; {target.target_id} is {target.backend}"
  py = python(env)
  if py is None:
    return f"TensorRT-LLM was not found: set ${ENV} to the python of the venv it is installed in"
  if not ec.has_module(py, "tensorrt_llm"):
    return f"{py} cannot import tensorrt_llm"
  return cached_probe(py) if cached else probe(py)


def argv(model:str, contexts:list[int], batches:list[int], *, tokens:int = TOKENS, reps:int = REPS,
         once:int | None = None, context_limit:int | None = None, max_num_tokens:int | None = None, backend:str | None = None,
         worker_single_process:str | None = None) -> list[str]:
  py = python()
  if py is None:
    raise RuntimeError(f"TensorRT-LLM was not found: set ${ENV}")
  out = [py, str(DRIVER), "--model", model, "--contexts", ",".join(map(str, contexts)),
         "--batches", ",".join(map(str, batches)), "--tokens", str(tokens), "--reps", str(reps)]
  out += ["--backend", backend or os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND]
  out += ["--worker-single-process", worker_single_process or os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0")]
  if context_limit is not None:
    out += ["--model-context-limit", str(context_limit)]
  if max_num_tokens is not None:
    out += ["--max-num-tokens", str(max_num_tokens)]
  return out + (["--once", str(once), "--profile-window"] if once else [])


def whole_step(model:str, contexts:list[int], batches:list[int], *, log:pathlib.Path | None = None,
               tokens:int = TOKENS) -> list[dict[str, Any]]:
  rows = ec.run_driver(argv(model, contexts, batches, tokens=tokens), log=log, env=ec.venv_env(python()))
  return [ec.point(r["context"], r["batch"], r["step_ms"], tokens=r["tokens"], engine=r.get("engine", {}), backend=r.get("backend"),
                   source=f"TensorRT-LLM LLM API ({r.get('backend')} backend), N tokens minus 1 token") for r in rows]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int, batch:int = 1,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  from boltbeam.collectors.capture_settings import context_limit, observed
  limit = context_limit(run, context, 'max_seq_len')
  settings = observed(run, context)
  capacity, backend = settings.get("max_num_tokens"), settings.get("backend")
  if type(capacity) is not int or capacity <= 0 or backend not in ("pytorch", "tensorrt"):
    raise ValueError("TensorRT capture requires the untraced scheduler capacity and backend; rerun whole-step measurement")
  return ec.captured_role_time(run, provider=PROVIDER, trace=TRACE, target=target, model_id=model_id,
                               context=context, tokens=tokens, batch=batch, env=ec.venv_env(python()), profiler_range=True,
                               argv_for=lambda n: argv(model, [context], [batch], tokens=n, reps=1, once=n, context_limit=limit,
                                                       max_num_tokens=capacity, backend=backend, worker_single_process=settings.get("worker_single_process")))

