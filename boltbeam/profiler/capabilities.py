"""Profiler capability registry loader."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any


SCHEMA_PROFILER_CAPABILITIES = "boltbeam.profiler_capabilities.v1"
_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "profiler_capabilities.json"

_REQUIRED_FIELDS = {
  "provider_id",
  "provider_support",
  "collector_id",
  "tool",
  "tool_version",
  "operation",
  "selection_priority",
  "target_support",
  "supported_scopes",
  "requires_replay",
  "requires_root_or_privilege",
  "supports_dispatch_timing",
  "supports_kernel_resources",
  "supports_dynamic_counters",
  "supports_memory_copy",
  "supports_allocations",
  "supports_source_correlation",
  "supports_disassembly",
  "supports_thread_trace",
  "supported_metrics",
  "unsupported_metrics",
  "known_blind_spots",
  "calibration_notes",
}


@dataclass(frozen=True)
class ProfilerCapability:
  provider_id: str
  provider_support: tuple[str, ...]
  collector_id: str
  tool: str
  tool_version: str
  operation: str
  selection_priority: int
  target_support: tuple[str, ...]
  supported_scopes: tuple[str, ...]
  requires_replay: bool
  requires_root_or_privilege: bool
  supports_dispatch_timing: bool
  supports_kernel_resources: bool
  supports_dynamic_counters: bool
  supports_memory_copy: bool
  supports_allocations: bool
  supports_source_correlation: bool
  supports_disassembly: bool
  supports_thread_trace: bool
  supported_metrics: tuple[str, ...]
  unsupported_metrics: tuple[str, ...]
  known_blind_spots: tuple[str, ...]
  calibration_notes: tuple[str, ...]


def _as_str_list(items: Any, field_name: str) -> tuple[str, ...]:
  if not isinstance(items, list):
    raise ValueError(f"field {field_name!r} must be a list")
  out: list[str] = []
  for idx, item in enumerate(items):
    if not isinstance(item, str):
      raise ValueError(f"field {field_name!r}[{idx}] must be a string")
    out.append(item)
  return tuple(out)


def _as_bool(value: Any, field_name: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"field {field_name!r} must be a boolean")
  return value


def _from_row(row: dict[str, Any]) -> ProfilerCapability:
  missing = sorted(_REQUIRED_FIELDS - row.keys())
  if missing:
    raise ValueError(f"profiler capability row missing required fields: {', '.join(missing)}")

  return ProfilerCapability(
    provider_id=row["provider_id"],
    provider_support=_as_str_list(row.get("provider_support", [row["provider_id"]]), "provider_support"),
    collector_id=row["collector_id"], tool=row["tool"], tool_version=row["tool_version"],
    operation=row.get("operation", "capture"), selection_priority=int(row.get("selection_priority", 0)),
    target_support=_as_str_list(row["target_support"], "target_support"),
    supported_scopes=_as_str_list(row["supported_scopes"], "supported_scopes"),
    requires_replay=_as_bool(row["requires_replay"], "requires_replay"),
    requires_root_or_privilege=_as_bool(row["requires_root_or_privilege"], "requires_root_or_privilege"),
    supports_dispatch_timing=_as_bool(row["supports_dispatch_timing"], "supports_dispatch_timing"),
    supports_kernel_resources=_as_bool(row["supports_kernel_resources"], "supports_kernel_resources"),
    supports_dynamic_counters=_as_bool(row["supports_dynamic_counters"], "supports_dynamic_counters"),
    supports_memory_copy=_as_bool(row["supports_memory_copy"], "supports_memory_copy"),
    supports_allocations=_as_bool(row["supports_allocations"], "supports_allocations"),
    supports_source_correlation=_as_bool(row["supports_source_correlation"], "supports_source_correlation"),
    supports_disassembly=_as_bool(row["supports_disassembly"], "supports_disassembly"),
    supports_thread_trace=_as_bool(row["supports_thread_trace"], "supports_thread_trace"),
    supported_metrics=_as_str_list(row["supported_metrics"], "supported_metrics"),
    unsupported_metrics=_as_str_list(row["unsupported_metrics"], "unsupported_metrics"),
    known_blind_spots=_as_str_list(row["known_blind_spots"], "known_blind_spots"),
    calibration_notes=_as_str_list(row["calibration_notes"], "calibration_notes"),
  )


def load_profiler_capabilities(path: str | pathlib.Path | None = None) -> dict[str, ProfilerCapability]:
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  data = json.loads(p.read_text())
  if data.get("schema") != SCHEMA_PROFILER_CAPABILITIES:
    raise ValueError(f"{p} is not a {SCHEMA_PROFILER_CAPABILITIES} registry")

  caps = {}
  collectors = data.get("collectors")
  if not isinstance(collectors, list):
    raise ValueError("profiler capabilities registry requires a collectors list")

  for row in collectors:
    cap = _from_row(row)
    caps[cap.collector_id] = cap
  return caps


_CAPABILITIES = load_profiler_capabilities()


def profiler_capability(collector_id: str, path: str | pathlib.Path | None = None) -> ProfilerCapability | None:
  """Lookup a profiler capability by collector_id."""
  reg = load_profiler_capabilities(path) if path else _CAPABILITIES
  return reg.get(collector_id)


def capabilities_for_provider(provider_id: str, path: str | pathlib.Path | None = None) -> tuple[ProfilerCapability, ...]:
  reg = load_profiler_capabilities(path) if path else _CAPABILITIES
  return tuple(cap for cap in reg.values() if provider_id in cap.provider_support)


def capabilities_for_tool(tool: str, path: str | pathlib.Path | None = None) -> tuple[ProfilerCapability, ...]:
  reg = load_profiler_capabilities(path) if path else _CAPABILITIES
  return tuple(cap for cap in reg.values() if cap.tool == tool)


_PROVIDER_ALIASES = {"llama": "llama.cpp", "llama-cpp": "llama.cpp", "llama.cpp": "llama.cpp",
                     "tinygrad": "tinygrad"}
_COLLECTOR_ALIASES = {"backend-csv": "rocprofv3"}


def normalize_provider_id(provider_id:str) -> str:
  value = str(provider_id).strip()
  return _PROVIDER_ALIASES.get(value, value)


def normalize_collector_id(collector_id:str) -> str:
  value = str(collector_id).strip()
  return _COLLECTOR_ALIASES.get(value, value)


def resolve_profiler_capability(provider_id:str, target_id:str, *, collector_id:str | None = None,
                                operation:str = "capture",
                                path:str | pathlib.Path | None = None) -> ProfilerCapability:
  """Resolve one collector from orthogonal provider/target/operation facts."""
  provider = normalize_provider_id(provider_id)
  reg = load_profiler_capabilities(path) if path else _CAPABILITIES
  if collector_id is not None:
    collector_id = normalize_collector_id(collector_id)
    cap = reg.get(collector_id)
    if cap is None: raise ValueError(f"unknown profiler collector {collector_id!r}")
    candidates = [cap]
  else:
    candidates = list(reg.values())
  matches = [cap for cap in candidates if provider in cap.provider_support and target_id in cap.target_support
             and cap.operation == operation]
  if not matches:
    requested = f" collector={collector_id!r}" if collector_id else ""
    raise ValueError(f"no {operation} profiler for provider={provider!r} target={target_id!r}{requested}")
  return sorted(matches, key=lambda cap:(-cap.selection_priority, cap.collector_id))[0]
