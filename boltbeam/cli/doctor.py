"""`boltbeam doctor`: what this machine has for BoltBeam, one row each, and the one thing to do for each gap.

It is a report, so it exits 0 either way. It probes nothing of its own: the GPU is read the way autoscan reads it
(target/hardware.hardware_profile, once), the engines are looked for by the engine scan and listed the way
Setup lists them (collectors/engine_scan, collectors/providers.available), the capture tool by the vendor table
(collectors/vendor_capture.plan), the GPU's health and counters by the vendor table too (target/vendors.py), each
Triton engine's Python.h (engine_common.missing_headers), the tinygrad fork by tinygrad_fork.readiness, and the Go
toolchain and the screen binary by cli/tui. --json prints the same rows as one object for an agent.

    row      found / not found / n/a, a detail, and for a gap the fix: one command or one env var
    gap      a row that says not found; n/a rows (an engine that never runs on this GPU type) are not gaps
    blocks   a gap no run gets past: Python too old, no GPU, no chip profile, a GPU that needs a reset, or no
             engine at all; any other gap costs one engine or one per-role split
    group    what the row is about, for a screen that groups them: gpu, engines, tools (what splits a step by
             role), host (RAM now, RAM reads, the PCIe link, the copies to the GPU) or boltbeam (its own Python, Go
             and screen)

The host rows read what is saved (target/host.py's host.json) and what the system states now (/proc/meminfo, sysfs);
a figure not measured yet is never a gap, its fix is `boltbeam host`, which measures the RAM reads and the copies to
the GPU (about a second on the GPU) and prints the same rows.
"""
from __future__ import annotations

import json
import platform
import sys
from typing import Any

from boltbeam.cli import tui
from boltbeam.collectors import engine_common as ec, engine_families, engine_scan, providers, tinygrad_role_time, vendor_capture
from boltbeam.search import tinygrad_fork
from boltbeam.target import targets as reg
from boltbeam.vocab import SCHEMA_DOCTOR as SCHEMA
from boltbeam.target import gpu as tgpu, hardware

FLOOR = (3, 10)
WORD = {True: "found", False: "not found", None: "n/a"}
AUTOSCAN = "python -m boltbeam.workflow.screen autoscan"


def row(item:str, found:bool | None, detail:str, fix:str | None = None, *, blocks:bool = False) -> dict[str, Any]:
  return {"item": item, "found": found, "detail": detail, "fix": fix, "blocks": blocks}


def _python() -> dict[str, Any]:
  ok = sys.version_info[:2] >= FLOOR
  return row("python", ok, f"{platform.python_version()} ({sys.executable})",
             None if ok else f"install Python {FLOOR[0]}.{FLOOR[1]} or newer and install BoltBeam with it: python3.12 -m pip install .", blocks=not ok)


def _gpu(gpu:dict[str, Any]) -> list[dict[str, Any]]:
  from boltbeam.workflow import layout as lay
  if gpu.get("status") != "detected":
    why = (gpu.get("notes") or [gpu.get("status") or "no GPU"])[0]
    return [row("GPU", False, why, "install the GPU's driver tool (nvidia-smi for NVIDIA, amd-smi for AMD): BoltBeam reads the GPU from it", blocks=True),
            row("chip profile", None, "no GPU")]
  devs = tgpu.devices(gpu)
  name = lay.summary(devs) or gpu.get("name") or "GPU"
  profile = gpu.get("profile") or {}
  out = [row("GPU", True, name + (f", driver {gpu['driver_version']}" if gpu.get("driver_version") else ""))]
  if profile.get("status") == "known":
    where = {"registry": "built in", "generated": "measured on this machine"}.get(profile.get("source") or "", profile.get("source") or "")
    if profile.get("source") == "generated" and reg.TARGETS.get(profile["id"]) and reg.fact_uncertainty(reg.TARGETS[profile["id"]]) is None:
      # a profile scanned before the scan measured its read's spread: every per-role check says not checked
      out.append(row("chip profile", False, f"{profile['id']} ({where}) has no measured bandwidth uncertainty, so the "
                     "per-role checks (no launch reads faster than the bandwidth) say not checked",
                     f"{AUTOSCAN} --remeasure   (plugged in, Low Power Mode off)"))
    else:
      out.append(row("chip profile", True, f"{profile['id']} ({where})"))
  else:
    out.append(row("chip profile", False, f"none yet for {gpu.get('name')}. A run does not measure the chip by itself, "
                   f"and without its measured bandwidth and compute peak every ceiling is refused. autoscan saves the "
                   f"profile as {profile.get('id')}",
                   f"{AUTOSCAN}   (measures this GPU once, about a minute, and saves the profile)", blocks=True))
  return out


def _engines(target, scanned:dict[str, Any], root) -> list[dict[str, Any]]:
  if target is None:
    return [row("engines", None, "not checked: no chip profile to measure on")]
  found = scanned.get("found") or {}
  avail = providers.available(target, tinygrad_root=root)
  by = {e["provider"]: e for e in avail}
  out = []
  for fam in engine_families.rows(avail):  # one row per family: the forks here; a fork not here is never a gap
    here = [f for f in fam["forks"] if f["state"] in ("available", "found")]
    missing = [f["label"] for f in fam["forks"] if f["state"] == "not_here"]
    if here:
      parts = []
      for f in here:
        if f["state"] == "available":
          hit = found.get(engine_scan.PRIMARY[f["provider"]]) or {}
          where = f"{hit['path']} ({hit['how']}); " if hit else ""
          parts.append(f"{f['label']}: {where}per-role time: {by[f['provider']]['capture']['method'] or 'whole step only'}")
        else:
          parts.append(f"{f['label']}: {f['path'] or 'found'} (BoltBeam cannot run it yet)")
      out.append({**row(fam["label"], True, " · ".join(parts)), "forks_missing": missing})
      continue
    runs = [f for f in fam["forks"] if f["provider"] in by]
    if not runs:  # no adapter of this family answered for this machine
      continue
    e = by[runs[0]["provider"]]  # the family's first fork BoltBeam runs
    name = e["provider"]
    if e["state"] == "not_compatible":
      out.append(row(fam["label"], None, e["reason"]))
    elif name == tinygrad_role_time.PROVIDER:
      ready = tinygrad_fork.readiness(root)
      out.append(row(fam["label"], False, ready["message"] or e["reason"], ready["fix"]))
    elif getattr(providers.ENGINES[name], "python", lambda: None)():
      out.append(row(fam["label"], False, e["reason"], "its python is found but it does not run here: the sentence above says why"))
    else:
      out.append(row(fam["label"], False, e["reason"], f"install it, or export {engine_scan.PRIMARY[name]}=<its path>; the next scan finds the usual places"))
  return out


def _vendor(target):
  from boltbeam.target import vendors
  return vendors.for_backend(target.backend) if target is not None else None


def _health(target) -> dict[str, Any]:
  """A GPU with a memory fault waiting for a reset runs slow, then crashes: its numbers look real (the vendor's read)."""
  from boltbeam.target.vendors import resolve
  v = _vendor(target)
  if v is None or v.health is None:
    return row("GPU health", None, "no memory-fault read for this GPU's vendor")
  got = resolve(v.health)()
  if got["healthy"] is None:
    return row("GPU health", False, f"not checked: {got['problems'][0]}", f"install {v.scan_tools[0]}")
  if got["healthy"]:
    return row("GPU health", True, "no remaps pending or failed, no uncorrectable DRAM errors since boot, every GPU opens a context")
  return row("GPU health", False, "; ".join(got["problems"]),
             "reset the GPU (reboot), or on a rented machine delete it and rent another: its numbers would not be its own", blocks=True)


def _headers(target) -> list[dict[str, Any]]:
  """Every found engine python that compiles C at startup (COMPILES_C) needs its Python.h."""
  out = []
  for name, mod in providers.ENGINES.items():
    py = getattr(mod, "COMPILES_C", False) and target is not None and target.backend in getattr(mod, "BACKENDS", ()) and mod.python()
    if not py:
      continue
    if (lack := ec.missing_headers(py)) is None:
      out.append(row(f"{name} headers", True, f"Python.h for {py}"))
    else:
      path, *ver = lack.split()
      dev = f"python{ver[0]}.{ver[1]}-dev" if len(ver) == 2 else "python3-dev"
      out.append(row(f"{name} headers", False, f"no {path}: Triton cannot compile its C helper, so {name} fails at startup",
                     f"install that Python's headers (Debian, Ubuntu: sudo apt install {dev})"))
  return out


def _counters(target) -> dict[str, Any]:
  """Closed vendor kernels (cuBLAS's nvjet) are split by role from the counters' executed SASS (instruction_path)."""
  from boltbeam.target.vendors import resolve
  v = _vendor(target)
  if v is None or v.counters is None:
    return row("GPU counters", None, "no counter permission to read for this GPU's vendor")
  read, how = v.counters
  if resolve(read)():
    return row("GPU counters", True, "open to this user: closed kernels (cuBLAS's nvjet) can be split by role")
  closed, own = how.split("; ", 1)
  return row("GPU counters", False, f"closed to this user ({closed}): a closed kernel's time stays unsplit by role",
             f"on your own machine: {own}; on a rented one (it cannot be set): run vLLM with --gemm-backends TRITON "
             "so its matrix kernels are readable")


def _kernel_source(target) -> dict[str, Any]:
  """The engine's kernel source the generic (kernel timer) measurement compiles: engine_kernels decides."""
  from boltbeam.collectors import engine_kernels, llama_bench_decode as lb
  if target is None:
    return row("kernel source", None, "not checked: no chip profile")
  why = engine_kernels.available(lb.PROVIDER, target.backend, lb.find(lb.DEFAULT))
  return row("kernel source", why is None, "llama.cpp's shipped kernels can be timed alone here" if why is None else why,
             None if why is None else "the sentence above names the env var; the engine scan also looks in the usual places")


def _capture(target) -> dict[str, Any]:
  if target is None:
    return row("capture tool", None, "not checked: no chip profile")
  p = vendor_capture.plan(target.backend)
  if p["method"] is None:
    return row("capture tool", None, p["reason"])
  if p["tool"]:
    return row("capture tool", True, f"{p['method']}: {p['tool']}")
  return row("capture tool", False, p["reason"].split(". To get it, ")[0], p["reason"].split(". To get it, ")[-1])


def _screen(facts:dict[str, Any]) -> list[dict[str, Any]]:
  need = facts["go_required"]
  if facts["go"] is None:
    go = row("Go", False, "not on PATH", tui.install_line(need))
  elif not facts["go_ok"]:
    go = row("Go", False, f"go{facts['go_version']} at {facts['go']} is older than the {need} the screen needs", tui.install_line(need))
  else:
    go = row("Go", True, f"go{facts['go_version']} at {facts['go']}")
  if facts["checkout"] is None:
    screen = row(tui.BINARY, False, "no checkout with tui/ found", f"run from the BoltBeam clone, or export {tui.REPO_ENV}=/path/to/BoltBeam")
  elif facts["binary"] is None:
    screen = row(tui.BINARY, False, "not built", "boltbeam tui   (builds it, then opens the screen)")
  elif facts["stale"]:
    screen = row(tui.BINARY, False, f"{facts['binary']} was built from older Go source", "boltbeam tui   (rebuilds it)")
  else:
    unchecked = "" if facts["stale"] is False else " (built by hand; boltbeam tui builds its own copy)"
    screen = row(tui.BINARY, True, facts["binary"] + unchecked)
  return [go, screen]


HOST_SCAN = "boltbeam host"


def _pci(gpu:dict[str, Any]) -> str | None:
  devs = tgpu.devices(gpu) if gpu.get("status") == "detected" else []
  return (devs[0].get("pci_bus_id") if devs else None) or gpu.get("pci_bus_id")


def _unified(gpu:dict[str, Any]) -> bool:
  """The GPU shares the CPU's memory: its vendor row has no profile shape of its own (target/vendors.py: unified memory)."""
  from boltbeam.target import vendors
  name = gpu.get("vendor") or next((d.get("vendor") for d in gpu.get("devices") or []), None)
  row = vendors.VENDORS.get(name or "")
  return row is not None and row.profile is None


def _host(gpu:dict[str, Any], *, measure:bool = False, remeasure:bool = False) -> list[dict[str, Any]]:
  """RAM total and available now, RAM read GB/s, the PCIe link and the copies to the GPU from pinned and from pageable
  memory. measure: run the probes for what is not saved (remeasure: all of it); else saved figures only."""
  from boltbeam.core.refused import Refused
  from boltbeam.target import host
  from boltbeam.workflow.offload import machine_rows
  pci = _pci(gpu)
  try:
    ram = host.ram_read(remeasure=remeasure) if measure else host._load(host.host_file())
    if not isinstance(ram.get("ram_gbs"), (int, float)):
      ram = {k: v for k, v in ram.items() if k != "ram_gbs"}
  except Refused as exc:  # a failed remeasure keeps the saved figure, and says the new one could not be taken
    saved = host._load(host.host_file())
    ram = {"ram_gbs": saved["ram_gbs"], "remeasure_failed": str(exc)} if isinstance(saved.get("ram_gbs"), (int, float)) else {"error": str(exc)}
  try:
    link = {"none": True} if not pci and _unified(gpu) else host.pcie_link(pci)
  except Refused as exc:
    link = {"error": str(exc)}
  why = "not measured yet"
  try:
    copies = host.h2d(pci, remeasure=remeasure) if measure else host.saved_h2d(pci)
  except Refused as exc:  # a failed (re)measure keeps the saved figure, as the RAM row does
    copies, why = host.saved_h2d(pci), str(exc)
  mem = host.memory()
  figures = {"ram_bytes": mem["total_bytes"], "ram_available_bytes": mem["available_bytes"], "ram_gbs": ram.get("ram_gbs"),
             "pcie_gbs": link.get("gbs"), "pcie_gen": link.get("gen"), "pcie_width": link.get("width"),
             "h2d_gbs": (copies or {}).get("h2d_gbs"), "h2d_measured_at": (copies or {}).get("measured_at")}
  rows = {r["label"]: r["value"] for r in machine_rows(figures)}
  out = [row("RAM", True, rows["RAM"] + f" ({mem['source']})")]
  out.append(row("RAM reads", True, rows["RAM reads"] + (f"; saved figure kept, the remeasure failed: {ram['remeasure_failed']}" if ram.get("remeasure_failed") else "")) if "RAM reads" in rows else
             row("RAM reads", True, ram.get("error") or "not measured yet", None if measure else HOST_SCAN + "   (measures RAM reads and the copies to the GPU, a few seconds)"))
  out.append(row("PCIe link", True, rows["PCIe link"]) if "PCIe link" in rows else
             row("PCIe link", True, "none: the GPU is part of the chip and shares the CPU's memory, so there is no link to read") if link.get("none") else row("PCIe link", None, link.get("error") or "no link read"))
  if "Copies to the card" in rows and (copies or {}).get("h2d_gbs"):
    out.append(row("copies to the GPU", True, rows["Copies to the card"] + ("" if why == "not measured yet" else f"; saved figure kept, the remeasure failed: {why}")))
  elif pci:
    out.append(row("copies to the GPU", True, ("not measured: " + why) if measure else "not measured yet; an offload plan uses the link's rate, an upper bound",
                   None if measure else HOST_SCAN + "   (about a second on the GPU)"))
  elif _unified(gpu):
    out.append(row("copies to the GPU", True, "unified memory: the GPU reads the same RAM, so no copy is needed"))
  else:
    out.append(row("copies to the GPU", None, "no GPU to copy to"))
  return out


def report(root=None) -> dict[str, Any]:
  """Every row with its group, then the count of gaps. One GPU read, one engine scan, nothing measured."""
  scanned = engine_scan.scan_and_save()
  gpu = hardware.hardware_profile()["gpu"]
  target = reg.TARGETS.get(gpu.get("target_id") or "")
  engines = _engines(target, scanned, root)
  if not any(r["found"] for r in engines):  # no engine at all: every engine gap stops the run
    engines = [{**r, "blocks": r["found"] is False} for r in engines]
  grouped = (("boltbeam", [_python()]), ("gpu", [*_gpu(gpu), _health(target)]), ("engines", [*engines, *_headers(target)]),
             ("tools", [_kernel_source(target), _capture(target)]), ("gpu", [_counters(target)]), ("host", _host(gpu)),
             ("boltbeam", _screen(tui.facts())))
  rows = [{**r, "group": group} for group, some in grouped for r in some]
  gaps = sum(1 for r in rows if r["found"] is False)
  return {"schema": SCHEMA, "rows": rows, "gaps": gaps, "ready": gaps == 0, "engines_file": scanned["file"],
          "engines": engine_scan.rows(scanned)}


def render(rep:dict[str, Any]) -> str:
  width = max(len(r["item"]) for r in rep["rows"]) + 2
  lines = []
  for r in rep["rows"]:
    lines.append(f"{r['item']:<{width}}{WORD[r['found']]:<11}{r['detail']}")
    if r["fix"]:
      lines.append(f"{'':<{width}}{'':<11}-> {r['fix']}")
  n = rep["gaps"]
  lines.append("ready to run" if n == 0 else f"{n} thing{'s' if n != 1 else ''} to set up")
  return "\n".join(lines) + "\n"


def cmd_doctor(args) -> int:
  rep = report()
  sys.stdout.write(json.dumps(rep, indent=2) + "\n" if args.json else render(rep))
  return 0


def cmd_host(args) -> int:
  """`boltbeam host`: measure what is not saved (or all with --remeasure) and print the host rows."""
  rows = [{**r, "group": "host"} for r in _host(hardware.hardware_profile()["gpu"], measure=True, remeasure=args.remeasure)]
  rep = {"schema": SCHEMA, "rows": rows, "gaps": 0, "ready": True}
  sys.stdout.write(json.dumps(rep, indent=2) + "\n" if args.json else render(rep).rsplit("\n", 2)[0] + "\n")
  return 0


def register(sub) -> None:
  h = sub.add_parser("host", help="this machine's RAM (total and available now), RAM reads, PCIe link and copies to the GPU from pinned and pageable memory; measures what is not saved")
  h.add_argument("--remeasure", action="store_true", help="measure the RAM reads and the copies again")
  h.add_argument("--json", action="store_true", help="the same rows as one JSON object")
  h.set_defaults(fn=cmd_host)
  p = sub.add_parser("doctor", help="what this machine has for BoltBeam, and the one thing to do for each gap (a report: exit 0)")
  p.add_argument("--json", action="store_true", help="the same rows as one JSON object")
  p.set_defaults(fn=cmd_doctor)
