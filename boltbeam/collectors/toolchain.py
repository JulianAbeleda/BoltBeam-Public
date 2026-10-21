"""Runtime provenance. Missing tools stay unknown; installed pins are never observations."""
from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import sys

from boltbeam.collectors import providers
from boltbeam.runtime.cuda_device import find_nvcc
from boltbeam.target import toolkits, vendors


def _read(argv:list[str]) -> str:
  try:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return (p.stdout or p.stderr).strip() if p.returncode == 0 else ""
  except (OSError, subprocess.TimeoutExpired):
    return ""


def pinned_vendor() -> vendors.Vendor:
  """The vendor whose toolchain data/engine_pins.json pins: the row whose toolchain keys hold every pinned key."""
  pins = json.loads(providers.PINS.read_text())["toolchain"]
  return next(v for v in vendors.VENDORS.values() if set(pins) <= {k for k, _, _ in v.toolchain})


def _locate(name:str) -> str:
  """A toolchain command's program: nvcc by the CUDA bridge's own discovery; a toolkit tool off PATH under its
  toolkit's bin/; otherwise the name, run from PATH."""
  if name == "nvcc":
    return find_nvcc() or "nvcc"
  return name if shutil.which(name) else (toolkits.tool(name, lambda _: None) or name)


def vendor_args(vendor:str | None) -> dict:
  """snapshot's arguments for a run's GPU vendor: none for the pinned toolchain's vendor (snapshot's default) or an
  unknown one, so that call is the one it always was."""
  return {"vendor": vendor} if vendor and vendor in vendors.VENDORS and vendor != pinned_vendor().vendor else {}


def snapshot(provider:str | None = None, vendor:str | None = None) -> dict:
  """Read the driver, toolkit, engine Python's torch and profilers without launching a GPU workload: the vendor's
  toolchain commands (target/vendors.py), the pinned toolchain's vendor's when none is named."""
  row = vendors.vendor(vendor) if vendor else pinned_vendor()
  rt = row.torch_runtime
  adapter = providers.ENGINES.get(provider)
  python = adapter.python() if adapter and hasattr(adapter, "python") else sys.executable
  if provider == "tinygrad":
    from boltbeam.search.role_compare import default_fork_root, fork_python
    python = str(fork_python(default_fork_root()))
  field = f", '{rt}': torch.version.{rt}" if rt else ""
  raw_torch = _read([python, "-c", "import json, torch; print(json.dumps({'version': str(torch.__version__)" + field + "}))"]) if python else ""
  try:
    runtime = json.loads(raw_torch)
    torch = runtime.get("version", "")
    torch_runtime = runtime.get(rt) or "unknown"
  except (ValueError, TypeError, AttributeError):
    torch, torch_runtime = raw_torch, "unknown"
  versions, read = {}, {}
  for key, argv, pattern in row.toolchain:
    if argv is None:  # the engine Python's torch
      versions[key] = torch or "unknown"
      continue
    raw = read[argv] if argv in read else read.setdefault(argv, _read([_locate(argv[0]), *argv[1:]]))
    match = re.search(pattern or r"\d+\.\d+(?:\.\d+)?(?:\+\w+)?", raw)
    versions[key] = (match.group(1) if match.re.groups else match.group()) if match else "unknown"
  return {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "versions": versions,
          "runtime": {rt: torch_runtime} if rt else {}}


def record(run, provider:str) -> None:
  from boltbeam.workflow.common import load_manifest, write_manifest
  from boltbeam.workflow.common import run_gpu
  manifest = load_manifest(run)
  manifest["toolchain"] = snapshot(provider, **vendor_args((run_gpu(run) or {}).get("vendor")))
  write_manifest(run, manifest)


def changes(before:dict, after:dict) -> list[dict]:
  pins = json.loads(providers.PINS.read_text())["toolchain"]
  # the vendor whose toolchain both runs recorded (the pinned one when they recorded none): its keys are the rows
  seen = {k for x in (before, after) for k in x.get("toolchain", {}).get("versions", {})}
  row = next((v for v in vendors.VENDORS.values() if seen and seen <= {k for k, _, _ in v.toolchain}), pinned_vendor())
  rt = row.torch_runtime
  a, b = ({**x.get("toolchain", {}).get("versions", {}),
           **({f"{rt}_runtime": x.get("toolchain", {}).get("runtime", {}).get(rt, "unknown")} if rt else {})}
          for x in (before, after))
  pins = {**{k: pins.get(k) for k, _, _ in row.toolchain}, **({f"{rt}_runtime": None} if rt else {})}
  return [{"name": k, "before": a.get(k, "unknown"), "after": b.get(k, "unknown"), "pin": pin,
           "changed": a.get(k, "unknown") != b.get(k, "unknown"),
           "known": all(x.get(k, "unknown") != "unknown" for x in (a, b))} for k, pin in pins.items()]
