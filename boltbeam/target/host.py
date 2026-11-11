"""The host side of an offload limit: host RAM read bandwidth (probed once, remembered per machine in
~/.boltbeam/host.json like the engine scan) and the GPU's PCIe link (gen and width read, not measured).
The CPU's dequant rate is not probed: nothing here returns one, and offload_limit says so in its notes."""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import threading
import time
from typing import Any, Callable

from boltbeam.core.refused import Refused
from boltbeam.target.hardware import run_tool

FILE_ENV = "BOLTBEAM_HOST_FILE"
PROBE_BYTES, THREADS, REPS = 256 << 20, 2, 5  # per thread; far beyond any cache; the best repetition (a lower bound on peak)
GENS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}  # sysfs max_link_speed in GT/s -> PCIe generation


def host_file() -> pathlib.Path:
  return pathlib.Path(os.environ.get(FILE_ENV) or pathlib.Path.home() / ".boltbeam" / "host.json").expanduser()


def _sum_probe() -> tuple[float, str]:
  """Read bandwidth in GB/s: THREADS threads each summing PROBE_BYTES (numpy, which releases the GIL), best of REPS."""
  try:
    import numpy as np
  except ImportError as exc:
    raise Refused("host RAM bandwidth is measured with numpy, which is not installed here (pip install numpy)") from exc
  arrays = [np.ones(PROBE_BYTES // 4, dtype=np.float32) for _ in range(THREADS)]
  best = 0.0
  for _ in range(REPS):
    ts = [threading.Thread(target=a.sum) for a in arrays]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    best = max(best, THREADS * PROBE_BYTES / (time.perf_counter() - t0) / 1e9)
  return best, f"numpy sum, {THREADS} threads x {PROBE_BYTES >> 20} MiB, best of {REPS}: a lower bound on the DDR peak"


def ram_read(*, probe:Callable[[], tuple[float, str]] = _sum_probe, path:pathlib.Path | None = None, remeasure:bool = False) -> dict[str, Any]:
  """This machine's host RAM read bandwidth: the saved figure, else measured now and saved. A bad probe is a Refused."""
  path = path or host_file()
  try:
    saved = json.loads(path.read_text())
  except (OSError, ValueError):
    saved = {}
  if saved.get("ram_gbs") and not remeasure:
    return saved
  gbs, how = probe()
  if not isinstance(gbs, (int, float)) or not gbs > 0: raise Refused(f"the RAM probe returned {gbs!r} GB/s")
  saved = {"schema": "boltbeam.host.v1", "ram_gbs": gbs, "ram_source": how, "measured_at": datetime.date.today().isoformat()}
  try:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(saved, indent=1))
  except OSError:
    pass  # an unwritable home only means the next call measures again
  return saved


def pcie_gbs(gts:float, width:int) -> float:
  """One direction's link rate in GB/s: GT/s x the line code (8b/10b to 5 GT/s, 128b/130b to 32, FLIT after) / 8 x lanes."""
  return gts * (0.8 if gts <= 5 else 128 / 130 if gts <= 32 else 242 / 256) / 8 * width


def _link(d:pathlib.Path, which:str) -> tuple[float, int]:
  return float((d / f"{which}_link_speed").read_text().split()[0]), int((d / f"{which}_link_width").read_text())


def pcie_link(pci_bus_id:str | None, *, sysfs:str = "/sys/bus/pci/devices", run_command=run_tool) -> dict[str, Any]:
  """The GPU's PCIe link as GB/s per direction: the smaller of the GPU's and its upstream port's maximum (a gen5 GPU in
  a gen4 slot is gen4), from sysfs, else nvidia-smi's maximum for GPU and system; and what it trained to now (an idle
  GPU trains down). The limit uses the maximum. The figure is a wire rate, so an upper bound."""
  parts = (pci_bus_id or "").lower().split(":")
  if len(parts) == 3:
    d = pathlib.Path(sysfs) / f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    try:
      (gts, width), (pgts, pwidth), now = _link(d, "max"), _link(d.resolve().parent, "max"), _link(d, "current")
      gts, width = min(gts, pgts), min(width, pwidth)
      return {"gen": GENS.get(gts), "width": width, "gbs": pcie_gbs(gts, width), "now_gbs": pcie_gbs(*now),
              "source": f"sysfs {d.name}: {gts:g} GT/s x{width}, the smaller of the GPU's and its slot's maximum"}
    except (OSError, ValueError, IndexError):
      pass
  rc, out, _ = run_command(("nvidia-smi", "--query-gpu=pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"))
  try:
    gen, width = (int(x) for x in out.splitlines()[0].split(","))
    gts = {v: k for k, v in GENS.items()}[gen]
  except (ValueError, KeyError, IndexError):
    raise Refused("the PCIe link of the GPU could not be read (sysfs and nvidia-smi gave nothing)") from None
  return {"gen": gen, "width": width, "gbs": pcie_gbs(gts, width), "source": f"nvidia-smi: gen {gen} x{width}, the maximum for this GPU and system", "now_gbs": None}


def facts(pci_bus_id:str | None = None, **kw) -> dict[str, Any]:
  """The host inputs of an offload limit: ram_gbs, pcie_gbs, each with its source, and cpu_gbs None (unmeasured).
  h2d_gbs: a measured host-to-device rate per host memory kind ({"pinned": GB/s, "pageable": GB/s}, with h2d_source),
  when host.json records one (none is probed here: it needs the GPU); None otherwise, and the limit uses the link rate."""
  ram, link = ram_read(**{k: v for k, v in kw.items() if k in ("probe", "path", "remeasure")}), pcie_link(pci_bus_id, **{k: v for k, v in kw.items() if k in ("sysfs", "run_command")})
  return {"ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "ram_gbs": ram["ram_gbs"], "ram_source": f"{ram['ram_source']} ({ram['measured_at']})", "pcie_gbs": link["gbs"], "pcie_source": link["source"],
          "pcie_gen": link["gen"], "pcie_width": link["width"], "cpu_gbs": None, "cpu_source": "CPU dequant rate not measured",
          "h2d_gbs": ram.get("h2d_gbs"), "h2d_source": ram.get("h2d_source")}
