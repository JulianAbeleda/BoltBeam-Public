"""Resolve a portable trace request into one provider/target collector profile."""
from __future__ import annotations

import pathlib
from typing import Any, Mapping

from boltbeam.core.canonical import sha256_file

from boltbeam.profiler.capabilities import normalize_provider_id, resolve_profiler_capability
from boltbeam.target.targets import TARGETS
from boltbeam.vocab import SCHEMA_TRACE_EXECUTION_PROFILE


def _sha256(path:pathlib.Path) -> str:
  return sha256_file(path)


def _model_identity(manifest:Mapping[str, Any]) -> dict[str, Any]:
  raw = manifest.get("model_path")
  path = pathlib.Path(raw).expanduser() if isinstance(raw, str) and raw else None
  exists = bool(path and path.is_file())
  return {
    "model_id": manifest.get("model_id"),
    "format": manifest.get("model_format"),
    "path": str(path) if path else None,
    "sha256": _sha256(path) if exists else None,
    "identity_status": "content_hashed" if exists else "unavailable",
  }


def _collector_coverage(cap:Any, trace_request:Mapping[str, Any]) -> dict[str, Any]:
  requirements = dict(trace_request.get("collector_requirements") or {})
  scopes = set(cap.supported_scopes)
  levels = {
    "whole_step": "whole_step" in scopes,
    "command_buffer": "command_buffer" in scopes,
    "dispatch": cap.supports_dispatch_timing,
    "kernel_resources": cap.supports_kernel_resources,
    "dynamic_counters": cap.supports_dynamic_counters,
  }
  required = [str(level) for level in requirements.get("required_levels", [])]
  optional = [str(level) for level in requirements.get("optional_levels", [])]
  missing_required = [level for level in required if not levels.get(level, False)]
  requested = {str(scope): [str(metric) for metric in metrics]
               for scope, metrics in dict(trace_request.get("requested_metrics") or {}).items()}
  supported = set(cap.supported_metrics)
  return {
    "required_levels": {level: levels.get(level, False) for level in required},
    "optional_levels": {level: levels.get(level, False) for level in optional},
    "missing_required_levels": missing_required,
    "requested_metrics": {
      scope: {
        "supported": [metric for metric in metrics if scope in scopes and metric in supported],
        "unavailable": [metric for metric in metrics if scope not in scopes or metric not in supported],
      }
      for scope, metrics in requested.items()
    },
    "scope_rule": requirements.get("scope_rule"),
  }


def resolve_trace_execution_profile(manifest:Mapping[str, Any], workload_profile:Mapping[str, Any],
                                    trace_request:Mapping[str, Any], provider:Mapping[str, Any], *,
                                    collector_id:str | None = None,
                                    capabilities_path:str | pathlib.Path | None = None) -> dict[str, Any]:
  target_id = str(manifest.get("target_id") or workload_profile.get("target_id") or "")
  target = TARGETS.get(target_id)
  if target is None: raise ValueError(f"unknown trace target {target_id!r}")
  provider_id = normalize_provider_id(str(provider.get("provider_id") or ""))
  if not provider_id: raise ValueError("trace runtime provider_id is required")
  cap = resolve_profiler_capability(provider_id, target_id, collector_id=collector_id, path=capabilities_path)
  coverage = _collector_coverage(cap, trace_request)
  if coverage["missing_required_levels"]:
    raise ValueError(f"collector {cap.collector_id!r} lacks required trace levels: "
                     f"{', '.join(coverage['missing_required_levels'])}")
  contexts = [int(x) for x in trace_request.get("contexts", workload_profile.get("contexts", []))]
  execution = dict(trace_request.get("execution") or {})
  execution.setdefault("context_mode", "fixed_decode_depth" if manifest.get("workload") == "decode" else "fixed_prefill_length")
  execution.setdefault("contexts", contexts)
  execution.setdefault("warmups", 1)
  execution.setdefault("samples", 1)
  return {
    "schema": SCHEMA_TRACE_EXECUTION_PROFILE,
    "status": "ready",
    "model": _model_identity(manifest),
    "workload": {
      "kind": manifest.get("workload") or workload_profile.get("workload"),
      "contexts": contexts,
      "execution": execution,
    },
    "target": {"target_id": target.target_id, "backend": target.backend},
    "runtime": {"provider_id": provider_id, "configured_path": provider.get("configured_path")},
    "collector": {
      "collector_id": cap.collector_id,
      "tool": cap.tool,
      "operation": cap.operation,
      "supported_metrics": list(cap.supported_metrics),
      "supported_scopes": list(cap.supported_scopes),
      "unsupported_metrics": list(cap.unsupported_metrics),
      "known_blind_spots": list(cap.known_blind_spots),
      "coverage": coverage,
    },
    "evidence_rule": "compare only equal model hash, workload context mode/depth, target, and timing scope",
  }


__all__ = ["resolve_trace_execution_profile"]
