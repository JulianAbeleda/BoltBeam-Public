"""The GPU a run uses: the hardware profile's first device, and the run folder's own (hardware_profile.json).

A one-GPU run pins the engine to device 0, so the run's GPU is the first device in the runtime's order; each
vendor's probe records that order (target/vendors.py), and a scan from before the order was recorded refuses to
name the card. Every collector that captures on a GPU, logs its clocks or picks a vendor's reader asks here.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from typing import Any

from boltbeam.artifacts.run_io import read_json
from boltbeam.target.vendors import VENDORS, vendor

LIMITED = ("limited support: measured on two R9700s (vLLM split by rows, llama.cpp split by layer); not timing-qualified, "
           "and a vLLM split by rows is not split by role")


def first_gpu(hardware:dict[str, Any]) -> dict[str, Any] | None:
  """The GPU a one-GPU run uses: the hardware profile's first device, in its runtime's order, the device an engine
  pinned to 0 runs on (each vendor's probe owns its order, target/vendors.py; layout.used_gpus). GPUs from more than one vendor name no
  first one: refused, with the vendors."""
  gpu = hardware.get("gpu") or {}
  devs = gpu.get("devices") or []
  if len(vendors := sorted({str(d.get("vendor")) for d in devs})) > 1:
    raise RuntimeError(f"this machine has GPUs from {', '.join(vendors)}: a one-GPU run cannot tell which it uses")
  if len(devs) > 1 and "probes" not in gpu:  # scanned before the scan recorded each vendor's order
    if why := vendor(vendors[0]).unrecorded_order:
      raise RuntimeError(f"this hardware profile's {len(devs)} GPUs cannot name the run's GPU: {why}; re-run the "
                         "pipeline to scan the machine again")
  return devs[0] if devs else None


def run_gpu(run:pathlib.Path) -> dict[str, Any] | None:
  """The run's GPU from its hardware_profile.json (vendor, architecture, pci_bus_id); None when it has none."""
  return first_gpu(read_json(p)) if (p := pathlib.Path(run) / "hardware_profile.json").is_file() else None


def devices(gpu:dict[str, Any]) -> list[dict[str, Any]]:
  """Every GPU the probe found, in the driver's order, with its index. One Apple GPU on a Mac."""
  rows = gpu.get("devices") or ([gpu] if gpu.get("name") else [])
  return [{"index": i, "name": d.get("name"), "target_id": d.get("target_id"), "uuid": d.get("uuid"), "pci_bus_id": d.get("pci_bus_id"),
           "memory_bytes": d.get("memory_bytes") or d.get("unified_memory_bytes"), "working_set_bytes": d.get("working_set_bytes"), "registered": bool(d.get("target_registered", d.get("target_id")))}
          for i, d in enumerate(rows)]


def used_gpus(layout:str, devs:list[dict[str, Any]], n:int | None = None) -> list[dict[str, Any]]:
  return devs[:1] if layout == "one" else devs[:n or None]


# GPU 0 in each runtime's order, the order the scan lists them in (hardware.vendor_rows() "order"): CUDA numbers GPUs
# fastest first unless told to follow PCI addresses, as nvidia-smi does, so the pinned card is the profiled card
PIN = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HIP_VISIBLE_DEVICES": "0"}


def pin() -> dict[str, str]:
  """PIN, keeping the first GPU of a visibility list the caller already set: the scan lists that GPU first
  (hardware.shown), so it is the run's GPU."""
  shown = {name for v in VENDORS.values() for name in v.visible}
  return {k: os.environ[k].split(",")[0].strip() if k in shown and os.environ.get(k) else v for k, v in PIN.items()}


# llama.cpp's split by rows, newest first: "tensor" is its tensor-parallel split; "row" needs ggml's split buffers,
# which a CUDA build at 50569eb87 refused on 2 x RTX 3090 ("device CUDA0 does not support split buffers", 2026-11-09)
LLAMA_ROW_MODES = ("tensor", "row")
_SPLIT_MODES:dict[tuple[str, int], tuple[str, ...] | None] = {}


def llama_split_modes(bench:str | None) -> tuple[str, ...] | None:
  """The split modes this llama-bench offers, from its own --help (--split-mode <none|layer|row|tensor>); None when
  it cannot be read. Cached per binary and its modification time."""
  if not bench or not os.path.isfile(bench):
    return None
  key = (bench, os.stat(bench).st_mtime_ns)
  if key not in _SPLIT_MODES:
    try:
      text = subprocess.run([bench, "--help"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
      text = ""
    m = re.search(r"--split-mode\s+<([a-z|]+)>", text)
    _SPLIT_MODES[key] = tuple(m.group(1).split("|")) if m else None
  return _SPLIT_MODES[key]


def engine_args(layout:str, provider:str, bench:str | None = None) -> tuple[list[str], dict[str, str]]:
  """Extra command arguments and environment that make the engine use this layout. llama.cpp's split by rows takes
  the newest mode the build at `bench` offers (LLAMA_ROW_MODES); with no build to ask, "row"."""
  if provider == "llama.cpp":
    if layout == "one":
      return ["-sm", "none", "-mg", "0"], pin()
    if layout == "row":
      offered = llama_split_modes(bench)
      return ["-sm", next((m for m in LLAMA_ROW_MODES if offered and m in offered), "row")], {}
    return ["-sm", layout], {}
  return [], pin()  # tinygrad: one GPU only


def one_gpu(run:pathlib.Path, provider:str) -> tuple[list[str], dict[str, str]]:
  """A one-GPU engine run's arguments and environment: pinned to GPU 0, the run's GPU (gpu.run_gpu), when the
  run's machine has more than one; nothing on one GPU, as the decode collector does (llama_bench_decode)."""
  hw = pathlib.Path(run) / "hardware_profile.json"
  devs = devices(json.loads(hw.read_text()).get("gpu") or {}) if hw.is_file() else []
  return engine_args("one", provider) if len(devs) > 1 else ([], {})
