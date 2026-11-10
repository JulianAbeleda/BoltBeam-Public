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
import re
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
COMPILES_C = True  # Triton builds a C helper against Python.h when the engine starts
# pytorch (TensorRT-LLM's default) or tensorrt (a built engine); unset, the driver chooses per checkpoint
# (trtllm_decode_drive.choose_backend) and every row records the backend and why
BACKEND_ENV = "BOLTBEAM_TRTLLM_BACKEND"
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
  out += ["--backend", backend or os.environ.get(BACKEND_ENV) or "auto"]
  out += ["--worker-single-process", worker_single_process or os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0")]
  if context_limit is not None:
    out += ["--model-context-limit", str(context_limit)]
  if max_num_tokens is not None:
    out += ["--max-num-tokens", str(max_num_tokens)]
  return out + (["--once", str(once), "--profile-window"] if once else [])


ADAPTED = "trtllm_checkpoint"  # under the run: the user's checkpoint, linked file by file, with a config TensorRT-LLM reads


def engine_checkpoint(model:str, run:pathlib.Path | None) -> tuple[str, dict[str, Any]]:
  """The folder TensorRT-LLM is started on, and what was adapted (empty when nothing was). TensorRT-LLM 1.2.1's
  Nemotron-H reads the layer plan as hybrid_override_pattern letters; a checkpoint that writes only layers_block_type
  (Nemotron 3.5) stops at "'NemotronHConfig' object has no attribute 'hybrid_override_pattern'". Such a checkpoint is
  linked file by file into the run's own folder with a config.json holding both, the letters by vocab.HYBRID_PATTERN;
  the user's folder is never written."""
  from boltbeam.vocab import HYBRID_PATTERN
  src = pathlib.Path(model).expanduser()
  try:
    cfg = json.loads((src / "config.json").read_text())
  except (OSError, ValueError):
    return model, {}
  blocks = cfg.get("layers_block_type")
  if run is None or "hybrid_override_pattern" in cfg or not isinstance(blocks, list) or not blocks:
    return model, {}
  letter = {kind: ch for ch, kind in HYBRID_PATTERN.items()}
  if unknown := sorted({str(b) for b in blocks if str(b).lower() not in letter}):
    raise ValueError(f"config.json layers_block_type names {unknown}, which have no hybrid_override_pattern letter "
                     f"(vocab.HYBRID_PATTERN): TensorRT-LLM could not read this checkpoint's layer plan")
  dst = pathlib.Path(run) / ADAPTED
  if dst.is_dir():
    for old in dst.iterdir():  # links and the written config only: the files they point to are the user's
      old.unlink()
  dst.mkdir(parents=True, exist_ok=True)
  for f in src.iterdir():
    if f.name != "config.json":
      (dst / f.name).symlink_to(f.absolute())
  (dst / "config.json").write_text(json.dumps({**cfg, "hybrid_override_pattern": "".join(letter[str(b).lower()] for b in blocks)}, indent=1))
  return str(dst), {"checkpoint": str(dst), "checkpoint_adapted": f"{src}: hybrid_override_pattern written from layers_block_type"}


# Why TensorRT-LLM could not start a model, read from what it printed: the first row whose pattern matches gives the
# cause in BoltBeam's words, and the engine's line it matched stands beside it. Output no row matches keeps its end.
REFUSALS = (
  (r"model architecture (?P<arch>\S+) is not supported in TRT-LLM", "TensorRT-LLM's classic engine builder does not "
   "build {arch} (its PyTorch backend may: leave $" + BACKEND_ENV + " unset, or set it to pytorch)"),
  (r"Unknown architecture for AutoModelForCausalLM: (?P<arch>\S+)", "TensorRT-LLM's PyTorch backend does not know {arch}"),
  (r"curand\.h: No such file", "a kernel TensorRT-LLM builds at start needs curand.h, which the CUDA toolkit it "
   "compiles with lacks"),
  (r"(?i)nvrtc[^\n]*(error|fail)|Error building extension|JIT compil\w* fail", "TensorRT-LLM could not compile a "
   "kernel at run time (NVRTC/JIT); another TensorRT-LLM release, or the other backend, may"),
  (r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|cudaErrorMemoryAllocation", "TensorRT-LLM ran out "
   "of GPU memory loading the model: the checkpoint is {size} on disk against the GPU's {gpu}, quantization {quant}"),
)


def checkpoint_facts(model:str, target) -> dict[str, str]:
  """What a refusal says about the checkpoint and GPU: the weights' size on disk, the GPU's memory, the quantization
  config.json names (quantization_config, else a ModelOpt export's hf_quant_config.json)."""
  src = pathlib.Path(model).expanduser()
  size = sum(f.stat().st_size for f in src.glob("*.safetensors")) if src.is_dir() else 0
  vram = getattr(target, "vram_bytes", None)
  def read(name:str) -> dict[str, Any]:
    try:
      return json.loads((src / name).read_text())
    except (OSError, ValueError):
      return {}
  cfg = read("config.json")
  q = cfg.get("quantization_config") or read("hf_quant_config.json").get("quantization") or {}
  quant = q.get("quant_algo") or q.get("quant_method") or f"none named (weights {cfg.get('torch_dtype') or 'of unnamed dtype'})"
  return {"size": f"{size / 1e9:.1f} GB" if size else "of unknown size", "quant": str(quant),
          "gpu": f"{vram / 1e9:.1f} GB" if vram else "memory (not in its profile)"}


def refusal(output:str, facts:dict[str, str]) -> str | None:
  """The one-line cause of a failed start (REFUSALS), with the engine's own line beside it; None when no row knows it."""
  for pattern, cause in REFUSALS:
    if m := re.search(pattern, output):
      line = output[output.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0].strip()
      return f"{cause.format(**facts, **m.groupdict())}. TensorRT-LLM: {line[:300]}"
  return None


def started(call, model:str, target):
  """call()'s result, or a failed start said as REFUSALS knows it (the driver's own message behind it as details)."""
  try:
    return call()
  except ec.DriverFailed as exc:
    if why := refusal(exc.output, checkpoint_facts(model, target)):
      raise ec.DriverFailed(why, output=exc.output, details=str(exc)) from exc
    raise


def whole_step(model:str, contexts:list[int], batches:list[int], *, log:pathlib.Path | None = None,
               tokens:int = TOKENS, run:pathlib.Path | None = None, target=None) -> list[dict[str, Any]]:
  model, adapted = engine_checkpoint(model, run)
  rows = started(lambda: ec.run_driver(argv(model, contexts, batches, tokens=tokens), log=log, env=ec.venv_env(python())),
                 model, target)
  named = {"backend_reason": f"${BACKEND_ENV} names it"} if os.environ.get(BACKEND_ENV) else {}
  return [ec.point(r["context"], r["batch"], r["step_ms"], tokens=r["tokens"], engine={**r.get("engine", {}), **named, **adapted}, backend=r.get("backend"),
                   source=f"TensorRT-LLM LLM API ({r.get('backend')} backend), N tokens minus 1 token") for r in rows]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int, batch:int = 1,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  from boltbeam.collectors.capture_settings import context_limit, observed
  limit = context_limit(run, context, 'max_seq_len')
  settings = observed(run, context)
  capacity, backend = settings.get("max_num_tokens"), settings.get("backend")
  if type(capacity) is not int or capacity <= 0 or backend not in ("pytorch", "tensorrt"):
    raise ValueError("TensorRT capture requires the untraced scheduler capacity and backend; rerun whole-step measurement")
  model = engine_checkpoint(model, run)[0]
  return started(lambda: ec.captured_role_time(run, provider=PROVIDER, trace=TRACE, target=target, model_id=model_id,
                               context=context, tokens=tokens, batch=batch, env=ec.venv_env(python()), profiler_range=True,
                               argv_for=lambda n: argv(model, [context], [batch], tokens=n, reps=1, once=n, context_limit=limit,
                                                       max_num_tokens=capacity, backend=backend, worker_single_process=settings.get("worker_single_process"))),
                 model, target)

