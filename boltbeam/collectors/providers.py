"""The runtimes BoltBeam measures a model in, and the one check before any measurement: is the GPU free.

A provider is a runtime that decodes the model: llama.cpp (collectors/llama_bench_decode.py) or tinygrad
(collectors/tinygrad_role_time.py). Each adapter knows only how to find its runtime, how to run N decode tokens at
a context, and how its per-role time is taken. Everything after that is shared: the vendor capture
(vendor_capture.py), the attribution by bytes and count (attribution.py), and the floor rule (tinygrad_role_time.loss).

    provider    whole step (step 4)              per role (step 5)
    llama.cpp   llama-bench decode               the vendor's capture of llama-bench, attributed by bytes
    tinygrad    the fork's decode, no profiling  the vendor's capture where it sees tinygrad (Metal with Xcode),
                                                 else tinygrad's own profile events (nsys and rocprofv3 cannot
                                                 see tinygrad's direct driver path)
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

from boltbeam.collectors import llama_bench_decode, tinygrad_role_time, vendor_capture

NAMES = (llama_bench_decode.PROVIDER, tinygrad_role_time.PROVIDER)
DEFAULT = llama_bench_decode.PROVIDER  # a run measured before providers existed was measured with llama-bench
# the per-role trace each provider writes into the run folder
TRACES = {llama_bench_decode.PROVIDER: llama_bench_decode.TRACE, tinygrad_role_time.PROVIDER: tinygrad_role_time.TRACE}
# the run's step-4 collector per provider and backend; "*" is any backend
COLLECTORS = {(llama_bench_decode.PROVIDER, "Metal"): "metal-native", (llama_bench_decode.PROVIDER, "*"): "llama-bench-decode",
              (tinygrad_role_time.PROVIDER, "*"): "tinygrad-decode"}
TINYGRAD_TOKENS = 16


def collector(provider:str, backend:str) -> str:
  return COLLECTORS.get((provider, backend)) or COLLECTORS[(provider, "*")]


def check(provider:str) -> str:
  if provider not in NAMES:
    raise ValueError(f"unknown provider {provider!r}; one of {', '.join(NAMES)}")
  return provider


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


def available(target, *, tinygrad_root:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Every provider, whether it can measure this target on this machine, and how step 5 would time its roles."""
  why = {llama_bench_decode.PROVIDER: llama_bench_decode.available(),
         tinygrad_role_time.PROVIDER: tinygrad_role_time.available(target, tinygrad_root)}
  return [{"provider": name, "available": why[name] is None, "reason": why[name],
           "capture": capture_method(name, target.backend)} for name in NAMES]


# --- is the GPU free ---------------------------------------------------------------------------------------

BUSY_PCT = 40  # the lowest of SAMPLES Apple GPU utilization reads at or above this: another program uses the GPU
SAMPLES = 3  # the desktop alone reads 0 to 15 percent; a model decoding holds it near 100
HEAVY = re.compile(r"llama-server|llama-cli|llama-bench|ollama|lm ?studio|mlx|tinygrad|GameTerm", re.I)


def _run(argv:list[str]) -> str:
  proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
  if proc.returncode != 0:
    raise RuntimeError(f"{argv[0]} exited {proc.returncode}")
  return proc.stdout


def nvidia_holders(text:str) -> list[str]:
  """`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits` as sentences."""
  out = []
  for line in text.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
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


def gpu_free(backend:str, *, run=_run) -> dict[str, Any]:
  """{"free": bool, "reason": sentence}. A GPU another program holds gives slower numbers that look real."""
  try:
    if backend == "CUDA":
      holders = nvidia_holders(run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"]))
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


# --- step 5: per role, any provider ----------------------------------------------------------------------------

def role_time(run:pathlib.Path, provider:str, *, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Per-role time for one provider, refused under the same floor rule for every provider."""
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
    raise RuntimeError(f"llama.cpp per role is not possible here: {how['reason']}")
  model, model_id = str(manifest.get("model_path")), str(manifest.get("model_id"))
  if provider == tinygrad_role_time.PROVIDER:
    root = pathlib.Path(root) if root else role_compare.default_fork_root()
    if why := tinygrad_role_time.available(target, root):
      raise RuntimeError(why)
    trace = tinygrad_role_time.role_time_captured(run, root=root, python=role_compare.fork_python(root), model=model,
                                                  model_id=model_id, target=target)
  else:
    trace = llama_bench_decode.role_time(run, target=target, model=model, model_id=model_id)
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  ceil = _measured_vs_ceiling(manifest, _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"))
  if table and table["status"] != "measured":
    raise RuntimeError(table["reason"])
  return trace
