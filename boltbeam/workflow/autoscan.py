from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
from typing import Any

from boltbeam.vocab import (SCHEMA_HARDWARE_PROFILE, SCHEMA_PROVIDER_CAPABILITIES,
                            SCHEMA_RUNTIME_PROFILE, SCHEMA_SCAN_EVIDENCE)
from boltbeam.target.targets import (TARGETS, family_target, is_exact_target, match_target,
                                     target_kind)
from boltbeam.workflow.common import (load_manifest, read_json, run_dir, update_manifest, write_json,
                                      write_manifest)

AUTOSCAN_ARTIFACTS = ("hardware_profile.json", "runtime_profile.json",
                      "provider_capabilities.json", "scan_evidence.json")

_NVIDIA_QUERY = (
  "name,uuid,pci.bus_id,memory.total,compute_cap,driver_version"
)

_APPLE_SOC = re.compile(r"\bApple\s+(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", re.IGNORECASE)


def _run_command(argv:tuple[str, ...]) -> tuple[int, str, str]:
  try:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    return proc.returncode, proc.stdout, proc.stderr
  except (OSError, subprocess.TimeoutExpired) as exc:
    return 127, "", str(exc)


def _tool_path(name:str, resolver) -> str | None:
  found = resolver(name)
  if found:
    return str(found)
  if name == "nvcc":
    conventional = pathlib.Path("/usr/local/cuda/bin/nvcc")
    if conventional.is_file():
      return str(conventional)
  return None


def _as_int(value:str) -> int | None:
  try:
    return int(float(value.strip()))
  except (TypeError, ValueError):
    return None


def _nvidia_arch(compute_capability:str) -> tuple[str | None, str | None]:
  parts = compute_capability.strip().split(".", 1)
  if len(parts) != 2 or not all(p.isdigit() for p in parts):
    return None, None
  suffix = f"{int(parts[0])}{int(parts[1])}"
  return f"sm_{suffix}", f"nvidia_sm{suffix}"


def _probe_nvidia(nvidia_smi:str, run_command) -> tuple[list[dict[str, Any]], str | None]:
  argv = (nvidia_smi, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits")
  code, stdout, stderr = run_command(argv)
  if code != 0:
    return [], (stderr.strip() or f"nvidia-smi exited with status {code}")
  devices = []
  for row in csv.reader(io.StringIO(stdout)):
    if not row or all(not value.strip() for value in row):
      continue
    if len(row) != 6:
      return [], f"nvidia-smi returned {len(row)} columns; expected 6"
    name, uuid, pci_bus_id, memory_mib_raw, compute_capability, driver_version = (v.strip() for v in row)
    memory_mib = _as_int(memory_mib_raw)
    architecture, by_convention = _nvidia_arch(compute_capability)
    # A row may claim this device outright; otherwise the vendor's own architecture name is the id.
    target_id = match_target({"compute_capability": compute_capability}) or by_convention
    devices.append({
      "vendor": "nvidia",
      "name": name,
      "uuid": uuid,
      "pci_bus_id": pci_bus_id,
      "memory_bytes": memory_mib * 1024 * 1024 if memory_mib is not None else None,
      "compute_capability": compute_capability or None,
      "architecture": architecture,
      "target_id": target_id,
      "target_registered": target_id in TARGETS if target_id else False,
      "driver_version": driver_version or None,
    })
  return devices, None if devices else "nvidia-smi returned no GPU rows"


def _apple_gpu_cores(row:dict[str, Any]) -> int | None:
  """Read the display-report spelling used by supported macOS releases without guessing a value."""
  for key in ("spdisplays_cores", "sppci_cores", "gpu_cores", "cores"):
    value = row.get(key)
    if value is None: continue
    match = re.search(r"\d+", str(value))
    if match: return int(match.group())
  return None


def _apple_soc(name:str) -> str | None:
  """The SoC the display report names ("M3", "M4 Pro"), or None when the report does not say it."""
  match = _APPLE_SOC.search(name)
  return re.sub(r"\s+", " ", match.group(1)).title() if match else None


def _probe_apple_metal(system_profiler:str, run_command) -> tuple[list[dict[str, Any]], str | None]:
  code, stdout, stderr = run_command((system_profiler, "SPDisplaysDataType", "-json"))
  if code != 0:
    return [], (stderr.strip() or f"system_profiler exited with status {code}")
  try:
    report = json.loads(stdout)
  except json.JSONDecodeError as exc:
    return [], f"system_profiler returned invalid JSON: {exc.msg}"
  rows = report.get("SPDisplaysDataType")
  if not isinstance(rows, list):
    return [], "system_profiler report omitted SPDisplaysDataType"
  devices = []
  for row in rows:
    if not isinstance(row, dict): continue
    name = next((str(row[key]).strip() for key in ("sppci_model", "_name", "spdisplays_vendor")
                 if row.get(key)), "")
    if not name or "apple" not in name.lower(): continue
    gpu_cores = _apple_gpu_cores(row)
    soc = _apple_soc(name)
    # The registry says which row a device is; the scan only reports what the machine said.
    target_id = match_target({"apple_soc": soc, "gpu_cores": gpu_cores}) or family_target("Metal")
    devices.append({
      "vendor": "apple",
      "name": name,
      "apple_soc": soc,
      "gpu_cores": gpu_cores,
      "metal_support": row.get("spdisplays_metal"),
      "metal_gpu_family_support": row.get("spdisplays_mtlgpufamilysupport"),
      "target_id": target_id,
      "target_registered": target_id in TARGETS,
      "target_kind": target_kind(target_id),
      "fact_status": {
        "apple_soc": "hardware_scan" if soc else "unknown",
        "gpu_cores": "hardware_scan" if gpu_cores is not None else "unknown",
        "metal_family": "unavailable_from_system_profiler",
        "recommended_max_working_set_size": "requires_provider_probe",
        "max_threads_per_threadgroup": "requires_provider_probe",
        "max_threadgroup_memory_length": "requires_provider_probe",
      },
    })
  return devices, None if devices else "system_profiler returned no Apple GPU rows"


def _meminfo() -> dict[str, int]:
  path = pathlib.Path("/proc/meminfo")
  if not path.exists():
    return {}
  out: dict[str, int] = {}
  for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    if ":" not in line:
      continue
    key, rest = line.split(":", 1)
    parts = rest.strip().split()
    if parts and parts[0].isdigit():
      out[key] = int(parts[0]) * 1024
  return out


def _hardware_profile(*, run_command=_run_command, tool_resolver=None,
                      system_name:str | None = None) -> dict[str, Any]:
  mem = _meminfo()
  resolver = tool_resolver or shutil.which
  tool_paths = {name: _tool_path(name, resolver)
                for name in ("nvidia-smi", "rocm-smi", "rocminfo", "hipcc", "nvcc", "system_profiler")}
  gpu: dict[str, Any] = {
    "status": "not_probed",
    "notes": ["No supported GPU probe was available; provider-specific evidence may add hardware facts."],
    "tools_present": {name: path is not None for name, path in tool_paths.items()},
    "tool_paths": tool_paths,
  }
  host_system = system_name or platform.system()
  if host_system == "Darwin" and tool_paths["system_profiler"]:
    devices, error = _probe_apple_metal(tool_paths["system_profiler"], run_command)
    if devices:
      primary = devices[0]
      gpu = {
        **gpu, "status": "detected", **primary, "devices": devices,
        "notes": ["Primary GPU is the first Apple device returned by system_profiler.",
                  "Dynamic Metal limits require the tinygrad provider fact adapter."],
      }
    else:
      gpu.update(status="probe_failed", notes=[error or "system_profiler Metal probe failed"])
  elif tool_paths["nvidia-smi"]:
    devices, error = _probe_nvidia(tool_paths["nvidia-smi"], run_command)
    if devices:
      primary = devices[0]
      gpu = {
        **gpu,
        "status": "detected",
        "vendor": primary["vendor"],
        "name": primary["name"],
        "uuid": primary["uuid"],
        "pci_bus_id": primary["pci_bus_id"],
        "memory_bytes": primary["memory_bytes"],
        "compute_capability": primary["compute_capability"],
        "architecture": primary["architecture"],
        "target_id": primary["target_id"],
        "target_registered": primary["target_registered"],
        "driver_version": primary["driver_version"],
        "devices": devices,
        "notes": ["Primary GPU is the first device returned by nvidia-smi."],
      }
    else:
      gpu.update(status="probe_failed", notes=[error or "nvidia-smi probe failed"])
  return {
    "schema": SCHEMA_HARDWARE_PROFILE,
    "cpu": {
      "machine": platform.machine(),
      "processor": platform.processor(),
      "logical_count": os.cpu_count(),
    },
    "system": {
      "platform": platform.platform(),
      "python": platform.python_version(),
      "memory_bytes": mem.get("MemTotal"),
    },
    "gpu": gpu,
  }


def _target_resolution_decision(hardware:dict[str, Any], manifest:dict[str, Any]) -> dict[str, Any]:
  """The irreversible-manifest-rewrite POLICY, isolated from the writes it authorizes ("contain dangerous
  power": an unsafe operation must have a small, reviewed boundary documenting validity/callers/mutation).

  What invariant makes this valid: this function is pure. It only reads `hardware` and `manifest` — never
  mutates either argument, touches disk, or spawns a subprocess. Every field in its return value is derived
  solely from those two read-only inputs, so calling it twice with the same inputs always agrees, and calling
  it can never itself corrupt state.

  Who may call it: only `_resolve_manifest_target`, the orchestration boundary below, which is the sole
  place permitted to turn this decision into a manifest rewrite and a workload_profile.json write. Nothing
  else should reimplement or inline this policy.

  What state it mutates: none. (The caller mutates manifest/workload_profile.json based on the "status" and
  "resolved_target_*" fields returned here.)
  """
  gpu = hardware["gpu"]
  detected = gpu.get("target_id")
  current = manifest.get("target_id")
  source = manifest.get("target_source")
  registered = bool(detected and detected in TARGETS)
  exact = is_exact_target(detected)
  replaceable = source in {"default", "autoscan"} or current == "auto"
  if registered and exact and replaceable:
    status = "selected"
  elif registered and not replaceable:
    status = "explicit_target_preserved"
  elif registered and not exact:
    status = "family_descriptor_only"
  elif detected:
    status = "unregistered_target"
  else:
    status = "not_detected"
  selected = status == "selected"
  return {
    "status": status,
    "resolved_target_id": detected if selected else current,
    "resolved_target_source": "autoscan" if selected else source,
    "detected_target_id": detected,
    "detected_target_kind": target_kind(str(detected)) if detected else None,
    "previous_target_id": current,
    "previous_target_source": source,
  }


def _resolve_manifest_target(out:pathlib.Path, manifest:dict[str, Any], hardware:dict[str, Any]) -> dict[str, Any]:
  """Orchestration boundary for the irreversible manifest/workload-profile rewrite. Applies the pure
  decision from `_target_resolution_decision` — this is the only function permitted to perform that write,
  and it performs no policy logic of its own beyond routing the decision to disk."""
  decision = _target_resolution_decision(hardware, manifest)
  gpu = hardware["gpu"]
  if decision["status"] == "selected":
    manifest["target_id"] = decision["resolved_target_id"]
    manifest["target_source"] = decision["resolved_target_source"]
    workload_path = out / "workload_profile.json"
    if workload_path.exists():
      workload = read_json(workload_path)
      workload["target_id"] = decision["resolved_target_id"]
      write_json(workload_path, workload)
  gpu["target_resolution"] = {
    "status": decision["status"],
    "detected_target_id": decision["detected_target_id"],
    "detected_target_kind": decision["detected_target_kind"],
    "selected_target_id": manifest.get("target_id"),
    "previous_target_id": decision["previous_target_id"],
    "previous_target_source": decision["previous_target_source"],
  }
  return manifest


def _runtime_profile() -> dict[str, Any]:
  return {
    "schema": SCHEMA_RUNTIME_PROFILE,
    "launch_overhead": {"status": "not_measured"},
    "graph_capture": {"status": "not_measured"},
    "host_sync": {"status": "not_measured"},
    "notes": [
      "Runtime overhead is provider/workload-specific and should be supplied as normalized evidence.",
    ],
  }


def _parse_provider(raw:str) -> tuple[str, str | None]:
  if "=" in raw:
    pid, path = raw.split("=", 1)
    return pid.strip(), path.strip() or None
  return raw.strip(), None


def _provider_row(provider_id:str, path:str | None) -> dict[str, Any]:
  p = pathlib.Path(path).expanduser() if path else None
  module_available = importlib.util.find_spec(provider_id) is not None if provider_id.isidentifier() else False
  return {
    "provider_id": provider_id,
    "configured_path": str(p) if p else None,
    "path_exists": bool(p and p.exists()),
    "python_module_available": module_available,
    "status": "available" if module_available or (p and p.exists()) else "missing",
    "capabilities": {
      "can_autoscan": False,
      "can_measure": False,
      "adapter": "not_configured",
    },
  }


def _provider_capabilities(provider_args:tuple[str, ...]) -> dict[str, Any]:
  configured = [_parse_provider(p) for p in provider_args]
  seen = {pid for pid, _path in configured}
  for pid in ("triton", "cutlass", "ck", "tinygrad", "iree"):
    if pid not in seen:
      configured.append((pid, None))
  rows = [_provider_row(pid, path) for pid, path in configured if pid]
  rows.append({
    "provider_id": "external",
    "configured_path": None,
    "path_exists": False,
    "python_module_available": False,
    "status": "available",
    "capabilities": {
      "can_autoscan": False,
      "can_measure": True,
      "adapter": "normalized_evidence_json",
    },
  })
  return {
    "schema": SCHEMA_PROVIDER_CAPABILITIES,
    "providers": rows,
    "notes": [
      "Providers are evidence producers. BoltBeam core consumes normalized artifacts, not provider internals.",
    ],
  }


def _scan_evidence(hardware:dict[str, Any], providers:dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": SCHEMA_SCAN_EVIDENCE,
    "kind": "cheap_autoscan",
    "observations": [
      {"key": "cpu.logical_count", "value": hardware["cpu"]["logical_count"]},
      {"key": "system.memory_bytes", "value": hardware["system"].get("memory_bytes")},
      {"key": "gpu.status", "value": hardware["gpu"].get("status")},
      {"key": "gpu.target_id", "value": hardware["gpu"].get("target_id")},
      {"key": "gpu.selected_target_id",
       "value": hardware["gpu"].get("target_resolution", {}).get("selected_target_id")},
      {"key": "provider.available",
       "value": [p["provider_id"] for p in providers["providers"] if p["status"] == "available"]},
    ],
    "measurement_status": "no_benchmarks_run",
  }


def autoscan_run(run:str | pathlib.Path, *, providers:tuple[str, ...] = (),
                 run_command=_run_command, tool_resolver=None, system_name:str | None = None) -> dict[str, Any]:
  out = run_dir(run)
  manifest = load_manifest(out)
  hardware = _hardware_profile(run_command=run_command, tool_resolver=tool_resolver, system_name=system_name)
  manifest = _resolve_manifest_target(out, manifest, hardware)
  write_manifest(out, manifest)
  runtime = _runtime_profile()
  provider_caps = _provider_capabilities(providers)
  scan = _scan_evidence(hardware, provider_caps)
  write_json(out / "hardware_profile.json", hardware)
  write_json(out / "runtime_profile.json", runtime)
  write_json(out / "provider_capabilities.json", provider_caps)
  write_json(out / "scan_evidence.json", scan)
  return update_manifest(out, stage="autoscan", artifacts=list(AUTOSCAN_ARTIFACTS))
