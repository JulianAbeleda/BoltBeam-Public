"""`boltbeam doctor`: what this machine has for BoltBeam, one row each, and the one thing to do for each gap.

It is a report, so it exits 0 either way. It probes nothing of its own: the GPU is read the way autoscan reads it
(workflow/autoscan._hardware_profile, once), the engines are looked for by the engine scan and listed the way
Setup lists them (collectors/engine_scan, collectors/providers.available), the capture tool by the vendor table
(collectors/vendor_capture.plan), the tinygrad fork by role_compare.readiness, and the Go toolchain and the screen
binary by cli/tui. --json prints the same rows as one object for an agent.

    row      found / not found / n/a, a detail, and for a gap the fix: one command or one env var
    gap      a row that says not found; n/a rows (an engine that never runs on this GPU type) are not gaps
"""
from __future__ import annotations

import json
import platform
import sys
from typing import Any

from boltbeam.cli import tui
from boltbeam.collectors import engine_scan, providers, tinygrad_role_time, vendor_capture
from boltbeam.search import role_compare
from boltbeam.target import targets as reg
from boltbeam.vocab import SCHEMA_DOCTOR as SCHEMA
from boltbeam.workflow import autoscan

FLOOR = (3, 10)
WORD = {True: "found", False: "not found", None: "n/a"}
AUTOSCAN = "python -m boltbeam.workflow.screen autoscan"


def row(item:str, found:bool | None, detail:str, fix:str | None = None) -> dict[str, Any]:
  return {"item": item, "found": found, "detail": detail, "fix": fix}


def _python() -> dict[str, Any]:
  ok = sys.version_info[:2] >= FLOOR
  return row("python", ok, f"{platform.python_version()} ({sys.executable})",
             None if ok else f"install Python {FLOOR[0]}.{FLOOR[1]} or newer and install BoltBeam with it: python3.12 -m pip install .")


def _gpu(gpu:dict[str, Any]) -> list[dict[str, Any]]:
  from boltbeam.workflow import layout as lay
  if gpu.get("status") != "detected":
    why = (gpu.get("notes") or [gpu.get("status") or "no GPU"])[0]
    return [row("GPU", False, why, "install the GPU's driver tool (nvidia-smi for NVIDIA, rocm-smi for AMD): BoltBeam reads the GPU from it"),
            row("chip profile", None, "no GPU")]
  devs = lay.devices(gpu)
  name = lay.summary(devs) or gpu.get("name") or "GPU"
  profile = gpu.get("profile") or {}
  out = [row("GPU", True, name + (f", driver {gpu['driver_version']}" if gpu.get("driver_version") else ""))]
  if profile.get("status") == "known":
    where = {"registry": "built in", "generated": "measured on this machine"}.get(profile.get("source") or "", profile.get("source") or "")
    out.append(row("chip profile", True, f"{profile['id']} ({where})"))
  else:
    out.append(row("chip profile", False, f"none yet for {gpu.get('name')}; autoscan would save it as {profile.get('id')}",
                   f"{AUTOSCAN}   (measures this GPU once, about a minute, and saves the profile)"))
  return out


def _engines(target, scanned:dict[str, Any], root) -> list[dict[str, Any]]:
  if target is None:
    return [row("engines", None, "not checked: no chip profile to measure on")]
  found = scanned.get("found") or {}
  out = []
  for e in providers.available(target, tinygrad_root=root):
    name = e["provider"]
    hit = found.get(engine_scan.PRIMARY[name]) or {}
    if e["available"]:
      how = e["capture"]["method"] or "whole step only"
      where = f"{hit['path']} ({hit['how']}); " if hit else ""
      out.append(row(name, True, f"{where}per-role time: {how}"))
    elif e["state"] == "not_compatible":
      out.append(row(name, None, e["reason"]))
    elif name == tinygrad_role_time.PROVIDER:
      ready = role_compare.readiness(root)
      out.append(row(name, False, ready["message"] or e["reason"], ready["fix"]))
    else:
      out.append(row(name, False, e["reason"], f"install it, or export {engine_scan.PRIMARY[name]}=<its path>; the next scan finds the usual places"))
  return out


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


def report(root=None) -> dict[str, Any]:
  """Every row, then the count of gaps. One GPU read, one engine scan, nothing measured."""
  scanned = engine_scan.scan_and_save()
  gpu = autoscan._hardware_profile()["gpu"]
  target = reg.TARGETS.get(gpu.get("target_id") or "")
  rows = [_python(), *_gpu(gpu), *_engines(target, scanned, root), _kernel_source(target), _capture(target),
          *_screen(tui.facts())]
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


def register(sub) -> None:
  p = sub.add_parser("doctor", help="what this machine has for BoltBeam, and the one thing to do for each gap (a report: exit 0)")
  p.add_argument("--json", action="store_true", help="the same rows as one JSON object")
  p.set_defaults(fn=cmd_doctor)
