"""Runtime provenance. Missing tools stay unknown; installed pins are never observations."""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys

from boltbeam.collectors import providers


def _read(argv:list[str]) -> str:
  try:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return (p.stdout or p.stderr).strip() if p.returncode == 0 else ""
  except (OSError, subprocess.TimeoutExpired):
    return ""


def snapshot(provider:str | None = None) -> dict:
  """Read the driver, toolkit, engine Python's torch and profilers without launching a GPU workload."""
  adapter = providers.ENGINES.get(provider)
  python = adapter.python() if adapter and hasattr(adapter, "python") else sys.executable
  if provider == "tinygrad":
    from boltbeam.search.role_compare import default_fork_root, fork_python
    python = str(fork_python(default_fork_root()))
  torch = _read([python, "-c", "import importlib.metadata; print(importlib.metadata.version('torch'))"]) if python else ""
  commands = {"driver": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
              "cuda": ["nvcc", "--version"], "nsight_systems": ["nsys", "--version"],
              "nsight_compute": ["ncu", "--version"]}
  versions = {}
  for key in json.loads(providers.PINS.read_text())["toolchain"]:
    raw = torch if key == "torch" else _read(commands[key])
    match = re.search(r"release\s+(\d+\.\d+)", raw) if key == "cuda" else re.search(r"\d+\.\d+(?:\.\d+)?(?:\+\w+)?", raw)
    versions[key] = (raw or "unknown") if key == "torch" else (
      match.group(1) if key == "cuda" else match.group()) if match else "unknown"
  return {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "versions": versions}


def record(run, provider:str) -> None:
  from boltbeam.workflow.common import load_manifest, write_manifest
  manifest = load_manifest(run)
  manifest["toolchain"] = snapshot(provider)
  write_manifest(run, manifest)


def changes(before:dict, after:dict) -> list[dict]:
  pins = json.loads(providers.PINS.read_text())["toolchain"]
  a, b = (x.get("toolchain", {}).get("versions", {}) for x in (before, after))
  return [{"name": k, "before": a.get(k, "unknown"), "after": b.get(k, "unknown"), "pin": pin,
           "changed": a.get(k, "unknown") != b.get(k, "unknown"),
           "known": all(x.get(k, "unknown") != "unknown" for x in (a, b))} for k, pin in pins.items()]
