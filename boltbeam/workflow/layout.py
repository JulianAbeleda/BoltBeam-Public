"""How a measurement uses a machine's GPUs, and the speed limit for that layout, derived from measured inputs.

    one     one GPU          the engine is pinned to GPU 0
    layer   split by layer   llama.cpp -sm layer (its default): each GPU holds a share of the layers and the layers
                             run one GPU after another
    row     split by rows    llama.cpp -sm row: each weight is split across the GPUs and read on all of them at once

Every input comes from this machine, with its source (machine_facts):
  - the GPUs, from the driver tool (nvidia-smi rows; one Apple GPU on a Mac), each mapped to a registered target
    when one matches and listed either way;
  - each GPU's READ bandwidth, measured on that GPU (runtime/gpu_probe.py, or collectors/metal_bandwidth.py on a
    Mac); the registry's figure is used only when measuring is impossible, and is labelled so;
  - the link between each pair of GPUs from `nvidia-smi topo -m`, and the measured copy bandwidth and latency
    between each used pair.

The limits (layer_limit, row_limit) say their formula and list every input with its source. Measurements are
cached in the run folder (MACHINE) and reused from the newest run of the same machine; re-measuring is an
explicit request. Multi-GPU carries LIMITED on every line: no real multi-GPU run has been made yet.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
from typing import Any, Callable

LIMITED = "limited support: no real multi-GPU run yet"
LAYOUTS = {"one": "one GPU", "layer": "split by layer", "row": "split by rows"}
DEFAULT = "one"
ENGINE_LAYOUTS = {"llama.cpp": ("one", "layer", "row"), "tinygrad": ("one",)}
NOT_OFFERED = {"tinygrad": "the tinygrad fork decodes on one device: its LLM decode has no sharded path"}
MACHINE = "machine_facts.json"
ACTIVATION_BYTES = 4  # float32 activations cross the GPU boundary (llama.cpp keeps hidden states in f32)
ROW_TRANSFERS_PER_LAYER = 2  # row split joins partial results after attention and after the feed-forward
PROBE = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "gpu_probe.py"


# --- the GPUs and their links ----------------------------------------------------------------------------------

def devices(gpu:dict[str, Any]) -> list[dict[str, Any]]:
  """Every GPU the probe found, in the driver's order, with its index. One Apple GPU on a Mac."""
  rows = gpu.get("devices") or ([gpu] if gpu.get("name") else [])
  return [{"index": i, "name": d.get("name"), "target_id": d.get("target_id"), "uuid": d.get("uuid"),
           "memory_bytes": d.get("memory_bytes"), "registered": bool(d.get("target_registered", d.get("target_id")))}
          for i, d in enumerate(rows)]


def summary(devs:list[dict[str, Any]]) -> str | None:
  """'4 x NVIDIA GeForce RTX 5090' or the mixed list, with the support label; None for one GPU."""
  if len(devs) < 2:
    return None
  names = [d["name"] for d in devs]
  shown = f"{len(devs)} x {names[0]}" if len(set(names)) == 1 else ", ".join(names)
  return f"{shown} ({LIMITED})"


def read_topology(text:str) -> dict[tuple[int, int], str]:
  """`nvidia-smi topo -m`: the link between each pair of GPUs (NV#, PIX, PXB, PHB, NODE, SYS)."""
  links: dict[tuple[int, int], str] = {}
  lines = [l for l in text.splitlines() if l.strip()]
  head = next((l for l in lines if re.search(r"\bGPU0\b", l) and not l.lstrip().startswith("GPU0 ")), None)
  if head is None:
    return links
  cols = [c for c in head.split() if re.fullmatch(r"GPU\d+", c)]
  for line in lines:
    m = re.match(r"\s*GPU(\d+)\s+(.*)", line)
    if not m:
      continue
    cells = m.group(2).split()
    for j, col in enumerate(cols):
      if j < len(cells) and cells[j] != "X":
        links[(int(m.group(1)), int(col[3:]))] = cells[j]
  return links


def link_words(code:str | None) -> str:
  if not code:
    return "unknown link"
  if code.startswith("NV"):
    return f"NVLink ({code})"
  return {"PIX": "PCIe, one switch", "PXB": "PCIe, several switches", "PHB": "PCIe through the CPU host bridge",
          "NODE": "PCIe across host bridges in one NUMA node", "SYS": "PCIe across NUMA nodes (SMP link)"}.get(code, code)


# --- layouts the engine can run ----------------------------------------------------------------------------------

def layouts(n_gpus:int, provider:str) -> list[dict[str, Any]]:
  """The layouts offered for this many GPUs and this engine; ones that cannot run say why."""
  if n_gpus < 2:
    return [{"id": "one", "label": LAYOUTS["one"], "available": True, "reason": None}]
  offered = ENGINE_LAYOUTS.get(provider, ("one",))
  return [{"id": k, "label": v, "available": k in offered,
           "reason": None if k in offered else NOT_OFFERED.get(provider, f"{provider} runs on one GPU")}
          for k, v in LAYOUTS.items()]


def used_gpus(layout:str, devs:list[dict[str, Any]]) -> list[dict[str, Any]]:
  return devs[:1] if layout == "one" else devs


def engine_args(layout:str, provider:str) -> tuple[list[str], dict[str, str]]:
  """Extra command arguments and environment that make the engine use this layout."""
  if provider == "llama.cpp":
    if layout == "one":
      return ["-sm", "none", "-mg", "0"], {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
    return ["-sm", layout], {}
  return [], {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}  # tinygrad: one GPU only


# --- measuring the machine -----------------------------------------------------------------------------------------

def device_name(prefix:str, index:int) -> str:
  """tinygrad's name for GPU `index`: "NV" for the first, "NV:1" for the next. It goes on the tensors; DEV is
  never set, because in the fork DEV=NV:1 selects a renderer named "1", not GPU 1."""
  return prefix if index == 0 else f"{prefix}:{index}"


def _today() -> str:
  return _dt.date.today().isoformat()


def run_probe(python:str, root:str, args:list[str], env:dict[str, str], timeout_s:float = 600.0) -> dict[str, Any]:
  """runtime/gpu_probe.py under the tinygrad fork's python; its last line is one JSON object."""
  proc = subprocess.run([python, str(PROBE), *args], cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", **env})
  lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError((proc.stderr or proc.stdout).strip()[-300:] or f"gpu_probe exited {proc.returncode}")
  return json.loads(lines[-1])


def measure_machine(devs:list[dict[str, Any]], *, backend:str, device_prefix:str | None,
                    probe:Callable[[list[str], dict[str, str]], dict[str, Any]] | None,
                    topology:str = "", fallback:Callable[[str | None], float | None] = lambda _: None,
                    metal_read:Callable[[], dict[str, Any]] | None = None,
                    cuda_read:Callable[[int], dict[str, Any]] | None = None,
                    profile_read:Callable[[dict[str, Any]], tuple[float, str] | None] = lambda _: None) -> dict[str, Any]:
  """Read bandwidth per GPU, and copy bandwidth and latency per GPU pair, with sources.

  Read bandwidth, in this order: BoltBeam's native probe for the backend (metal_read on a Mac; cuda_read(index) on
  NVIDIA, collectors/cuda_bandwidth.py); then the registry figure, labelled with why the probe did not run. On
  NVIDIA tinygrad's sum is no bandwidth measure (382 GB/s on a 5090 that reads about 1,700), so it is used only
  when nothing else exists, and labelled so. Elsewhere (AMD) it stays a labelled lower bound. probe(args, env) runs
  the fork's gpu_probe.py, which also measures the copies between GPUs. profile_read(gpu) comes first: a chip
  profile made on this machine already holds the GPU's measured read bandwidth and its date (workflow/chips.py), so
  the limit uses that one number instead of a second measurement."""
  links = read_topology(topology)
  gpus = []
  for d in devs:
    row = {**d, "read_gbs": None, "read_source": None}
    sum_gbs = None
    try:
      if (known := profile_read(d)) is not None:
        row.update(read_gbs=known[0], read_source=known[1])
      elif backend == "Metal" and metal_read is not None:
        got = metal_read()
        row.update(read_gbs=got["read_gbs"], read_source=f"measured on this GPU with BoltBeam's Metal read probe, {_today()}")
      elif backend == "CUDA" and cuda_read is not None:
        got = cuda_read(d["index"])
        row.update(read_gbs=got["read_gbs"], read_source=f"measured on GPU {d['index']} with BoltBeam's native CUDA read "
                   f"probe, best of {got.get('reps', 10)} per launch shape, {_today()}")
      elif backend == "CUDA" and cuda_read is None:
        raise RuntimeError("BoltBeam's CUDA read probe cannot run here: nvcc is not installed")
      elif probe is not None and device_prefix:
        got = probe(["read", "--device", device_name(device_prefix, d["index"])], {})
        row.update(read_gbs=got["read_gbs"], read_source=f"measured on GPU {d['index']} with a tinygrad read-sum, {_today()} "
                   "(a lower bound: on the M3 it read 88.0 GB/s where BoltBeam's Metal probe read 97.9)")
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
      row["probe_error"] = str(exc)[-200:]
    if row["read_gbs"] is None:
      reg = fallback(d.get("target_id"))
      if reg:
        row.update(read_gbs=reg, read_source=f"registry figure for {d.get('target_id')}, not measured on this GPU"
                   + (f" ({row['probe_error']})" if row.get("probe_error") else ""))
      elif backend == "CUDA" and probe is not None and device_prefix:  # nothing else: the sum, called what it is
        try:
          sum_gbs = probe(["read", "--device", device_name(device_prefix, d["index"])], {})["read_gbs"]
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyError, ValueError):
          sum_gbs = None
        row.update(read_gbs=sum_gbs, read_source="tinygrad sum, not a bandwidth measure (it reached 22% of a 5090's "
                   "bandwidth); no native probe and no registry figure" if sum_gbs else
                   "unknown: not measured and no registry figure")
      else:
        row["read_source"] = "unknown: not measured and no registry figure"
    gpus.append(row)
  pairs = []
  for a in devs:
    for b in devs:
      if a["index"] >= b["index"]:
        continue
      pair = {"a": a["index"], "b": b["index"], "link": links.get((a["index"], b["index"])),
              "copy_gbs": None, "latency_us": None, "source": "not measured"}
      if probe is not None and device_prefix:
        try:
          got = probe(["copy", "--src", device_name(device_prefix, a["index"]), "--dst", device_name(device_prefix, b["index"])], {})
          pair.update(copy_gbs=got["copy_gbs"], latency_us=got["latency_us"],
                      source=f"measured copy GPU {a['index']} to GPU {b['index']}, best of {got.get('reps', 5)}, {_today()}")
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
          pair["source"] = f"not measured: {str(exc)[-160:]}"
      pairs.append(pair)
  return {"schema": "boltbeam.machine_facts.v1", "measured_at": _today(), "gpus": gpus, "pairs": pairs,
          "uuids": [d.get("uuid") for d in devs]}


def cached_machine(run:pathlib.Path, devs:list[dict[str, Any]]) -> dict[str, Any] | None:
  """The machine facts in this run, or in the newest sibling run of the same GPUs (same uuids)."""
  uuids = [d.get("uuid") for d in devs]
  for p in [run / MACHINE, *sorted(run.parent.glob(f"*/{MACHINE}"), reverse=True)]:
    if p.exists():
      facts = json.loads(p.read_text())
      if facts.get("uuids") == uuids:
        return facts
  return None


# --- the limits ------------------------------------------------------------------------------------------------------

def _pair(facts:dict[str, Any], a:int, b:int) -> dict[str, Any] | None:
  return next((p for p in facts.get("pairs", []) if {p["a"], p["b"]} == {a, b}), None)


def limit(layout:str, *, bytes_per_token:float, facts:dict[str, Any], hidden_size:int | None, layers:int | None,
          shares:list[float] | None = None) -> dict[str, Any]:
  """The decode limit for a layout from the machine facts. Every input is listed with its source."""
  gpus = facts["gpus"] if layout != "one" else facts["gpus"][:1]
  n = len(gpus)
  shares = shares or [1.0 / n] * n  # llama.cpp splits evenly unless -ts says otherwise
  inputs = [{"what": f"GPU {g['index']} {g['name']} read bandwidth", "value": g["read_gbs"], "unit": "GB/s",
             "source": g["read_source"]} for g in gpus]
  inputs.append({"what": "weight bytes per token", "value": bytes_per_token, "unit": "bytes",
                 "source": "the model's roofline (decode, context 1)"})
  if any(not g["read_gbs"] for g in gpus):
    return {"layout": layout, "ms": None, "inputs": inputs, "reason": "a GPU in the layout has no read bandwidth"}
  out = {"layout": layout, "label": LAYOUTS[layout], "gpus": n, "inputs": inputs, "reason": None}
  if layout == "one" or n == 1:
    ms = bytes_per_token / (gpus[0]["read_gbs"] * 1e9) * 1e3
    out.update(ms=ms, formula="weight bytes / the GPU's read bandwidth")
    return {**out, "tok_s": 1000.0 / ms}
  act = (hidden_size or 0) * ACTIVATION_BYTES
  inputs.append({"what": "activation bytes per GPU boundary", "value": act, "unit": "bytes",
                 "source": f"hidden size {hidden_size} x {ACTIVATION_BYTES} bytes (float32)"})
  hops = []
  for i in range(n - 1 if layout == "layer" else n):
    a, b = gpus[i]["index"], gpus[(i + 1) % n]["index"]
    p = _pair(facts, a, b)
    if not p or not p.get("copy_gbs"):
      return {**out, "ms": None, "reason": f"no measured copy between GPU {a} and GPU {b}: {p and p['source']}"}
    hops.append(act / (p["copy_gbs"] * 1e9) * 1e3 + p["latency_us"] / 1e3)
    inputs.append({"what": f"copy GPU {a} to GPU {b} ({link_words(p.get('link'))})",
                   "value": f"{p['copy_gbs']:.1f} GB/s, {p['latency_us']:.1f} µs", "unit": "", "source": p["source"]})
  if layout == "layer":
    weights = sum(bytes_per_token * s / (g["read_gbs"] * 1e9) * 1e3 for s, g in zip(shares, gpus))
    transfer = sum(hops)
    formula = ("sum over GPUs of (its share of weight bytes / its read bandwidth), plus one activation copy per GPU "
               "boundary (bytes / measured copy bandwidth + measured latency)")
  else:
    weights = max(bytes_per_token * s / (g["read_gbs"] * 1e9) * 1e3 for s, g in zip(shares, gpus))
    transfer = (layers or 0) * ROW_TRANSFERS_PER_LAYER * max(hops)
    formula = (f"the slowest GPU's share of weight bytes / its read bandwidth, plus {ROW_TRANSFERS_PER_LAYER} joins per "
               "layer, each one activation copy on the slowest measured link (bytes / copy bandwidth + latency)")
  ms = weights + transfer
  return {**out, "ms": ms, "tok_s": 1000.0 / ms, "weights_ms": weights, "transfer_ms": transfer, "formula": formula,
          "support": LIMITED}


# --- the one read bandwidth a single-GPU limit uses -----------------------------------------------------------------

def short_source(source:str | None, measured_at:str | None = None) -> str:
  """The read bandwidth's source in a few words for a screen line: which probe, and when; or that it is the
  registry's figure. The full source stays in the limit's inputs."""
  s = source or ""
  when = f", {measured_at}" if measured_at else ""
  for needle, words in (("native CUDA read probe", "native CUDA probe"), ("Metal read probe", "Metal read probe"),
                        ("tinygrad read-sum", "tinygrad read-sum, a lower bound")):
    if needle in s:
      return f"measured on this GPU ({words}{when})"
  if s.startswith("registry figure"):
    return "registry figure, not measured on this GPU"
  return s or "unknown"


def read_bandwidth(facts:dict[str, Any] | None, layout:str = "one") -> dict[str, Any] | None:
  """The read bandwidth the one-GPU limit uses (the layout's first GPU), with its source in full and in short;
  None for no facts, a multi-GPU layout or a GPU with no read bandwidth. Measured or registry, it is the one number
  the speed limit, the tie-out and the per-role rule all use."""
  if not facts or not facts.get("gpus") or (layout != "one" and len(facts["gpus"]) > 1):
    return None
  g = facts["gpus"][0]
  if not g.get("read_gbs"):
    return None
  return {"gbs": g["read_gbs"], "source": g.get("read_source"), "target_id": g.get("target_id"),
          "measured": str(g.get("read_source") or "").startswith("measured"),
          "short": short_source(g.get("read_source"), facts.get("measured_at"))}


def newest_facts(folders:list[pathlib.Path], target_id:str) -> dict[str, Any] | None:
  """The newest machine facts, in any run under these folders, whose first GPU is this chip: what Setup shows
  before a run of this session has measured. Newest by when they were measured, then by file time."""
  found = []
  for folder in folders:
    if folder.is_dir():
      for p in folder.glob(f"*/{MACHINE}"):
        try:
          facts = json.loads(p.read_text())
        except (OSError, ValueError):
          continue
        if (facts.get("gpus") or [{}])[0].get("target_id") == target_id:
          found.append((str(facts.get("measured_at") or ""), p.stat().st_mtime, facts))
  return max(found, key=lambda f: (f[0], f[1]))[2] if found else None
