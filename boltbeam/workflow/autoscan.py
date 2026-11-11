from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

from boltbeam.target.hardware import hardware_profile, run_tool
from boltbeam.vocab import SCHEMA_PROVIDER_CAPABILITIES, SCHEMA_RUNTIME_PROFILE, SCHEMA_SCAN_EVIDENCE
from boltbeam.target.targets import TARGETS, is_exact_target, target_kind
from boltbeam.artifacts.run_io import load_manifest, read_json, run_dir, update_manifest, write_json, write_manifest

AUTOSCAN_ARTIFACTS = ("hardware_profile.json", "runtime_profile.json",
                      "provider_capabilities.json", "scan_evidence.json")

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
                 run_command=run_tool, tool_resolver=None, system_name:str | None = None) -> dict[str, Any]:
  out = run_dir(run)
  manifest = load_manifest(out)
  hardware = hardware_profile(run_command=run_command, tool_resolver=tool_resolver, system_name=system_name)
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
