"""The runtimes BoltBeam measures a model in, and the one check before any measurement: is the GPU free.

A provider (an engine on screen) is a runtime that decodes the model. Each adapter knows only how to find its
runtime, which weight files it reads, how to run N decode tokens at a context and batch, and how its per-role time
is taken. Everything after that is shared: the vendor capture (vendor_capture.py), the attribution by bytes and
count (attribution.py), and the floor rule (tinygrad_role_time.loss).

    provider      weights       whole step (step 4)                    per role (step 5)
    llama.cpp     GGUF          llama-bench decode; batch > 1 with     the vendor's capture of llama-bench
                                llama-batched-bench
    tinygrad      GGUF          the fork's decode, no profiling        the vendor's capture where it sees tinygrad
                                                                       (Metal), else tinygrad's own profile events
    vllm          safetensors   offline LLM API, N minus 1 tokens      nsys of the same driver, CUDA graphs on
    ollama        GGUF          its API's eval_count / eval_duration   nsys of `ollama serve` and its runner
    tensorrt-llm  safetensors   its LLM API, N minus 1 tokens          nsys of the same driver

ENGINES is the one table of them. A run is labelled with its engine and its weight format (weight_format).
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

from boltbeam.collectors import (llama_bench_decode, ollama_decode, tinygrad_role_time, trtllm_decode, vendor_capture,
                                 vllm_decode)

# name -> adapter module; the order is the order Setup lists them
ENGINES = {m.PROVIDER: m for m in (llama_bench_decode, tinygrad_role_time, vllm_decode, ollama_decode, trtllm_decode)}
NAMES = tuple(ENGINES)
DEFAULT = llama_bench_decode.PROVIDER  # a run measured before providers existed was measured with llama-bench
# the per-role trace each provider writes into the run folder
TRACES = {name: m.TRACE for name, m in ENGINES.items()}
# the engines that run as their own program through a driver (engine_common.py)
DRIVEN = (vllm_decode.PROVIDER, ollama_decode.PROVIDER, trtllm_decode.PROVIDER)
# the run's step-4 collector per provider and backend; "*" is any backend
COLLECTORS = {(llama_bench_decode.PROVIDER, "Metal"): "metal-native", (llama_bench_decode.PROVIDER, "*"): "llama-bench-decode",
              (tinygrad_role_time.PROVIDER, "*"): "tinygrad-decode",
              **{(name, "*"): f"{name}-decode" for name in DRIVEN}}
TINYGRAD_TOKENS = 16
FORMATS = {llama_bench_decode.PROVIDER: ("gguf",), tinygrad_role_time.PROVIDER: ("gguf",),
           **{name: ENGINES[name].FORMATS for name in DRIVEN}}


def collector(provider:str, backend:str) -> str:
  return COLLECTORS.get((provider, backend)) or COLLECTORS[(provider, "*")]


def check(provider:str) -> str:
  if provider not in NAMES:
    raise ValueError(f"unknown provider {provider!r}; one of {', '.join(NAMES)}")
  return provider


def kv_element(provider:str) -> tuple[int, str]:
  """KV cache bytes per element in this engine as it runs by default, and where that fact comes from (each
  adapter's KV_ELEMENT)."""
  m = ENGINES.get(provider)
  return getattr(m, "KV_ELEMENT", None) or (2, "f16 assumed")


def weight_format(profile:dict[str, Any]) -> str:
  """The weights as a label: the file family and the quant types by share of the weight bytes, largest first:
  "GGUF Q4_K+Q6_K", "GGUF F16", "safetensors BF16"."""
  family = str((profile.get("metadata") or {}).get("format_family") or "unknown")
  share: dict[str, float] = {}
  for r in profile.get("roles") or []:
    q = str(r.get("quant") or "?")
    share[q] = share.get(q, 0.0) + float(r.get("rows") or 0) * float(r.get("cols") or 0) * float(r.get("count") or 1)
  quants = [q for q, _ in sorted(share.items(), key=lambda kv: -kv[1])]
  return f"{'GGUF' if family == 'gguf' else family} {'+'.join(quants[:2]) if quants else '?'}"


def model_format(model:str | pathlib.Path) -> str | None:
  """The model's file family (profile/loaders.detect_model_format), or None when it is not one BoltBeam reads."""
  from boltbeam.profile.loaders import detect_model_format
  try:
    return detect_model_format(model)
  except (ValueError, OSError):
    return None


def reads(provider:str, model:str | pathlib.Path) -> str | None:
  """Why this engine cannot read this model file, or None when it can. A path that is not there is left to the
  engine's own preflight, which says the file is gone."""
  fmt = model_format(model)
  want = FORMATS.get(provider, ())
  if fmt in want or (fmt is None and not pathlib.Path(model).expanduser().exists()):
    return None
  return (f"{provider} reads {' or '.join(f.upper() if f == 'gguf' else f for f in want)} weights; "
          f"{pathlib.Path(model).name} is {fmt or 'not a model BoltBeam reads'}")


def capture_method(provider:str, backend:str) -> dict[str, Any]:
  """How step 5 times roles for this provider here: {"method", "reason"}. method None: whole step only."""
  p = vendor_capture.plan(backend)
  if provider == tinygrad_role_time.PROVIDER:
    if backend in tinygrad_role_time.CAPTURED_BACKENDS and p["tool"]:
      return {"method": p["method"], "reason": None}
    return {"method": tinygrad_role_time.OWN_TIMING, "reason": tinygrad_role_time.OWN_TIMING_REASON.get(backend)}
  if p["tool"] is None:
    return {"method": None, "reason": p["reason"]}
  return {"method": p["method"], "reason": None}


def _why(name:str, target, tinygrad_root:pathlib.Path | None) -> str | None:
  if name == llama_bench_decode.PROVIDER:
    return llama_bench_decode.available()
  if name == tinygrad_role_time.PROVIDER:
    return tinygrad_role_time.available(target, tinygrad_root)
  return ENGINES[name].available(target)


def state(name:str, target, why:str | None) -> str:
  """One word a screen shows for an engine here: available, not_compatible (it never runs on this GPU type) or
  not_here (it could, but it is not installed or not set up). The sentence stays in `reason`."""
  if why is None:
    return "available"
  if name == tinygrad_role_time.PROVIDER:
    return "not_here" if tinygrad_role_time.device_for(target) else "not_compatible"
  backends = getattr(ENGINES[name], "BACKENDS", None)
  return "not_compatible" if backends is not None and target.backend not in backends else "not_here"


def available(target, *, tinygrad_root:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Every provider, whether it can measure this target on this machine, the weights it reads, and how step 5
  would time its roles."""
  out = []
  for name in NAMES:
    why = _why(name, target, tinygrad_root)
    out.append({"provider": name, "available": why is None, "state": state(name, target, why), "reason": why, "formats": list(FORMATS[name]),
                "capture": capture_method(name, target.backend),
                "batch_over_one": name != tinygrad_role_time.PROVIDER})  # tinygrad's decode runs one stream
  return out


# --- is the GPU free ---------------------------------------------------------------------------------------

BUSY_PCT = 40  # the lowest of SAMPLES Apple GPU utilization reads at or above this: another program uses the GPU
SAMPLES = 3  # the desktop alone reads 0 to 15 percent; a model decoding holds it near 100
HEAVY = re.compile(r"llama-server|llama-cli|llama-bench|ollama|lm ?studio|mlx|tinygrad|GameTerm", re.I)


def _run(argv:list[str]) -> str:
  proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
  if proc.returncode != 0:
    raise RuntimeError(f"{argv[0]} exited {proc.returncode}")
  return proc.stdout


def nvidia_holders(text:str, uuids:set[str] | None = None) -> list[str]:
  """`nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits` as
  sentences; with uuids, only the programs on those GPUs."""
  out = []
  for line in text.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if uuids is not None and len(parts) >= 4 and parts[3] not in uuids:
      continue
    if len(parts) >= 3 and parts[0].isdigit():
      out.append(f"{pathlib.Path(parts[1]).name} (pid {parts[0]}) holds {parts[2]} MiB")
  return out


def amd_holders(text:str) -> list[str]:
  """`rocm-smi --showpids --json` as sentences."""
  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    return []
  out = []
  for key, val in (data.get("system") or {}).items():
    if key.startswith("PID") and isinstance(val, str):
      name = val.split(",")[0].strip()
      out.append(f"{name} (pid {key[3:]}) has the GPU open")
  return out


def apple_busy(ioreg:str) -> int | None:
  """GPU utilization percent from `ioreg -r -d 1 -c IOAccelerator`, or None when it is not reported."""
  m = re.search(r'"Device Utilization %"=(\d+)', ioreg)
  return int(m.group(1)) if m else None


def heavy_processes(ps:str) -> list[str]:
  """Programs from `ps -axo pid=,comm=` that are known to run models on the GPU."""
  out = []
  for line in ps.strip().splitlines():
    pid, _, comm = line.strip().partition(" ")
    if HEAVY.search(comm) and pid.isdigit():
      out.append(f"{pathlib.Path(comm.strip()).name} (pid {pid})")
  return out


def gpu_free(backend:str, *, run=_run, uuids:set[str] | None = None) -> dict[str, Any]:
  """{"free": bool, "reason": sentence}. A GPU another program holds gives slower numbers that look real. uuids:
  only the GPUs the layout uses (every GPU when None)."""
  try:
    if backend == "CUDA":
      holders = nvidia_holders(run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                                    "--format=csv,noheader,nounits"]), uuids)
    elif backend == "AMD":
      holders = amd_holders(run(["rocm-smi", "--showpids", "--json"]))
    elif backend == "Metal" and sys.platform == "darwin":
      reads = []
      for i in range(SAMPLES):
        if i:
          time.sleep(0.5)
        reads.append(apple_busy(run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])))
      pct = min((r for r in reads if r is not None), default=None)
      if pct is None or pct < BUSY_PCT:
        return {"free": True, "reason": f"the GPU is {pct if pct is not None else 'not reported as'}% busy"}
      found = heavy_processes(run(["ps", "-axo", "pid=,comm="]))
      return {"free": False, "reason": f"the GPU is {pct}% busy" + (": " + "; ".join(found) if found else
                                                                    "; quit the program using it")}
    else:
      return {"free": True, "reason": f"no GPU check for {backend} on {sys.platform}"}
  except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
    return {"free": False, "reason": f"the GPU could not be read: {exc}"}
  if holders:
    return {"free": False, "reason": "the GPU is in use: " + "; ".join(holders)}
  return {"free": True, "reason": "no other program holds the GPU"}


# --- step 4: whole step in tinygrad --------------------------------------------------------------------------

def measure_tinygrad(run:pathlib.Path, *, root:pathlib.Path | None = None) -> pathlib.Path:
  """trace_request.json answered with tinygrad decodes: one whole_step row per requested context."""
  from boltbeam.core.canonical import pretty_json
  from boltbeam.search import role_compare
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest, read_json
  root = pathlib.Path(root) if root else role_compare.default_fork_root()
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  if why := tinygrad_role_time.available(target, root):
    raise RuntimeError(why)
  request = read_json(run / "trace_request.json")
  rows = []
  for ctx in request.get("contexts") or [128]:
    got = tinygrad_role_time.whole_step(root=root, python=role_compare.fork_python(root), model=str(manifest["model_path"]),
                                        target=target, context=int(ctx), tokens=TINYGRAD_TOKENS)
    rows.append({"scope": "whole_step", "context": int(ctx), "wall_us": 1e6 / got["tok_s"], "tok_s": got["tok_s"],
                 "decode_tokens": got["tokens"], "source": f"tinygrad decode on {got['device']}, JIT with {got['jit']}, no profiling"})
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": manifest["workload"], "provider_id": tinygrad_role_time.PROVIDER, "collector_id": "tinygrad-decode",
           "timing_source": "tinygrad whole step", "contexts": [r["context"] for r in rows], "rows": rows,
           "measured": ["whole_step.tok_s (tinygrad decode, wall clock over the decode tokens)"],
           "absent": ["kernel rows: per role is step 5"]}
  if target.memory_bandwidth_gbs:
    trace["peak_gbs"] = target.memory_bandwidth_gbs
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


# --- step 4: whole step in an engine that runs as its own program (vLLM, Ollama, TensorRT-LLM) -------------

ENGINE_TOKENS = 64


def engine_trace(manifest:dict[str, Any], *, provider:str, provider_id:str, collector_id:str,
                 points:list[dict[str, Any]], weights:str, peak_gbs:float | None) -> dict[str, Any]:
  """A boltbeam.timing_trace.v1 from measured points. The batch-1 points are the whole_step rows every reader
  takes; every point, batch 1 included, is kept under "batches" with tokens/s per stream and in total."""
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  ones = sorted((p for p in points if p["batch"] == 1), key=lambda p: p["context"])
  rows = [{"scope": "whole_step", "context": p["context"], "wall_us": p["step_ms"] * 1000.0, "tok_s": p["tok_s_stream"],
           "decode_tokens": p.get("tokens"), "batch": 1, "source": p.get("source")} for p in ones]
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": manifest["workload"], "provider": provider, "provider_id": provider_id,
           "collector_id": collector_id, "weight_format": weights, "timing_source": f"{provider} whole step",
           "contexts": [r["context"] for r in rows], "rows": rows,
           "batches": sorted(points, key=lambda p: (p["context"], p["batch"])),
           "measured": [f"whole_step.tok_s ({provider} decode at each context, batch 1)",
                        "batches: one decode step of B streams, tokens/s per stream and in total"],
           "absent": ["kernel rows: per role is step 5"]}
  if peak_gbs:
    trace["peak_gbs"] = peak_gbs
  return trace


def measure_engine(run:pathlib.Path, provider:str, *, batches:list[int] | tuple[int, ...] = (1,),
                   tokens:int = ENGINE_TOKENS) -> pathlib.Path:
  """trace_request.json answered by a driven engine: one point per requested context and batch."""
  from boltbeam.core.canonical import pretty_json
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json
  m = ENGINES[check(provider)]
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  model = str(manifest["model_path"])
  if why := m.available(target) or reads(provider, model):
    raise RuntimeError(why)
  request = read_json(run / "trace_request.json")
  contexts = [int(c) for c in request.get("contexts") or [128]]
  points = m.whole_step(model, contexts, sorted({1, *batches}), tokens=tokens,
                        log=run / "kernel_compare" / f"{provider}_whole_step.log")
  profile = read_json(run / "model_profile.json") if (run / "model_profile.json").exists() else {}
  trace = engine_trace(manifest, provider=provider, provider_id=m.PROVIDER_ID, collector_id=collector(provider, target.backend),
                       points=points, weights=weight_format(profile), peak_gbs=target.memory_bandwidth_gbs)
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


# --- step 5: per role, any provider ----------------------------------------------------------------------------

def _layout(run:pathlib.Path) -> tuple[str, int]:
  status = run / "measure_status.json"
  m = json.loads(status.read_text()) if status.exists() else {}
  return m.get("layout") or "one", int(m.get("gpus") or 1)


def role_time(run:pathlib.Path, provider:str, *, root:pathlib.Path | None = None, batch:int = 1,
              context:int = 128) -> dict[str, Any]:
  """Per-role time for one provider, refused under the same floor rule for every provider. batch > 1 captures
  one decode step of that many streams (driven engines and llama.cpp's batched bench)."""
  from boltbeam.search import role_compare
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  check(provider)
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  gpu = gpu_free(target.backend)
  if not gpu["free"]:
    raise RuntimeError(f"not measured: {gpu['reason']}")
  how = capture_method(provider, target.backend)
  if provider == tinygrad_role_time.PROVIDER and how["method"] == tinygrad_role_time.OWN_TIMING:
    return role_compare.time_roles(run, root=root)
  if how["method"] is None:
    raise RuntimeError(f"{provider} per role is not possible here: {how['reason']}")
  model, model_id = str(manifest.get("model_path")), str(manifest.get("model_id"))
  if provider in DRIVEN:
    if why := ENGINES[provider].available(target) or reads(provider, model):
      raise RuntimeError(why)
    trace = ENGINES[provider].role_time(run, target=target, model=model, model_id=model_id, context=context, batch=batch)
  elif provider == tinygrad_role_time.PROVIDER:
    root = pathlib.Path(root) if root else role_compare.default_fork_root()
    if why := tinygrad_role_time.available(target, root):
      raise RuntimeError(why)
    trace = tinygrad_role_time.role_time_captured(run, root=root, python=role_compare.fork_python(root), model=model,
                                                  model_id=model_id, target=target)
  else:
    layout, gpus = _layout(run)
    trace = llama_bench_decode.role_time(run, target=target, model=model, model_id=model_id, layout=layout, gpus=gpus,
                                         context=context, batch=batch)
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  ceil = _measured_vs_ceiling(manifest, _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"), ceil.get("band"))
  if table and table["status"] != "measured":
    raise RuntimeError(table["reason"])
  return trace
