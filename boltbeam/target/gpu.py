"""The GPU a run uses: the hardware profile's first device, and the run folder's own (hardware_profile.json).

A one-GPU run pins the engine to device 0, so the run's GPU is the first device in the runtime's order; each
vendor's probe records that order (target/vendors.py), and a scan from before the order was recorded refuses to
name the card. Every collector that captures on a GPU, logs its clocks or picks a vendor's reader asks here.
"""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.artifacts.run_io import read_json
from boltbeam.target.vendors import vendor


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
