from __future__ import annotations

import pathlib, shutil

from boltbeam.core.canonical import sha256_hex
from typing import Any

from boltbeam.runtime.execution_bridge import parse_operand_execution_request, parse_operand_execution_result
from boltbeam.target import vendors
from boltbeam.target.targets import TARGETS
from boltbeam.profiler.trace_profile import resolve_trace_execution_profile
from boltbeam.profiler.capabilities import normalize_provider_id

from boltbeam.vocab import (
  SCHEMA_EXECUTION_BRIDGE_REQUEST, SCHEMA_EXECUTION_BRIDGE_RESULT, SCHEMA_HW_TRACE,
  SCHEMA_PROBE_EVIDENCE, SCHEMA_RUNNER_PLAN, SCHEMA_TIMING_TRACE,
)
from boltbeam.artifacts.run_io import load_manifest, read_json, write_json

_REQUIRED_INPUTS = (
  "model_profile.json",
  "weight_inventory.json",
  "workload_profile.json",
  "measurement_plan.json",
  "probe_request.json",
  "trace_request.json",
)

_OPTIONAL_INPUTS = (
  "hardware_profile.json",
  "runtime_profile.json",
  "provider_capabilities.json",
  "search_space.json",
  "route_policy.json",
  "fixture_manifest.json",
  "ws_sweep_request.json",
  "operand_path_request.json",
)

def build_runner_plan(run:pathlib.Path, *, providers:tuple[str, ...] = (),
                      bundle_dir:str | pathlib.Path | None = None) -> dict[str, Any]:
  manifest = load_manifest(run)
  bundle = pathlib.Path(bundle_dir).expanduser() if bundle_dir else run / "runner_bundle"
  provider_rows = [_provider_row(p) for p in providers] or [{"provider_id": "tinygrad", "configured_path": None}]
  executor_hint = "tinygrad" if any(normalize_provider_id(p["provider_id"]) == "tinygrad" for p in provider_rows) else provider_rows[0]["provider_id"]  # arkey's recorded executor id
  inputs = [_input_row(run, bundle, name, required=True) for name in _REQUIRED_INPUTS]
  inputs += [_input_row(run, bundle, name, required=False) for name in _OPTIONAL_INPUTS]
  existing = [row for row in inputs if row["status"] == "present"]
  missing_required = [row["artifact"] for row in inputs if row["required"] and row["status"] != "present"]
  _write_bundle(bundle, existing)
  operand_request_path = run / "operand_path_request.json"
  operand_request = parse_operand_execution_request(read_json(operand_request_path)) if operand_request_path.exists() else None
  workload_profile = read_json(run / "workload_profile.json") if (run / "workload_profile.json").exists() else {}
  trace_request = read_json(run / "trace_request.json") if (run / "trace_request.json").exists() else {}
  trace_profiles = [_trace_profile_or_blocked(manifest, workload_profile, trace_request, provider)
                    for provider in provider_rows] if workload_profile and trace_request else []
  bundle_manifest = {
    "schema": "boltbeam.runner_bundle_manifest.v1",
    "source_run": str(run),
    "purpose": "audit_tracer_handoff",
    "artifact_count": len(existing),
    "artifacts": [{"artifact": row["artifact"], "fingerprint": row["fingerprint"]} for row in existing],
  }
  write_json(bundle / "runner_bundle_manifest.json", bundle_manifest)
  return {
    "schema": SCHEMA_RUNNER_PLAN,
    "purpose": "audit_tracer",
    "model_id": manifest.get("model_id"),
    "target_id": manifest.get("target_id"),
    "workload": manifest.get("workload"),
    "source_run": str(run),
    "status": "ready" if not missing_required else "incomplete",
    "providers": provider_rows,
    "runtime_boundary": {
      "boltbeam_role": "audit_tracer_contract",
      "runtime_executor": executor_hint,
      "not_runtime_executor": True,
      "requires_external_runtime_executor": True,
      "executor_responsibility": "load/run the model or kernels and produce normalized probe/timing evidence",
      "current_integration": "tinygrad should own runtime execution; BoltBeam only requests and audits traces",
      "target_runtime": (_tinygrad_target_runtime(manifest.get("target_id")) if executor_hint == "tinygrad"
                         else {"target_id": manifest.get("target_id"), "status": "provider_owned", "env": {}}),
    },
    "bundle": {
      "path": str(bundle),
      "manifest": str(bundle / "runner_bundle_manifest.json"),
      "artifact_count": len(existing),
    },
    "inputs": inputs,
    "missing_inputs": missing_required,
    "operand_path_execution": {
      "request_schema": SCHEMA_EXECUTION_BRIDGE_REQUEST,
      "result_schema": SCHEMA_EXECUTION_BRIDGE_RESULT,
      "request_artifact": "operand_path_request.json" if operand_request is not None else None,
      "request_digest": _fingerprint(operand_request_path) if operand_request is not None else None,
      "candidate_id": operand_request.get("candidate_id") if operand_request is not None else None,
      "status": "requested" if operand_request is not None else "not_requested",
      "boltbeam_dispatches": False,
    },
    "expected_outputs": [
      {
        "kind": "probe_evidence",
        "schema": SCHEMA_PROBE_EVIDENCE,
        "path": "probe_evidence.json",
        "request_artifact": "probe_request.json",
        "ingest_command": ["boltbeam", "ingest-probe", "probe_evidence.json", "--run", str(run)],
        "required": True,
    },
    {
      "kind": "timing_trace",
      "schema": SCHEMA_TIMING_TRACE,
      "path": "timing_trace.json",
      "request_artifact": "trace_request.json",
      "ingest_command": ["boltbeam", "ingest-timing", "timing_trace.json", "--run", str(run)],
      "required": True,
    },
    {
      "kind": "hw_trace",
      "schema": SCHEMA_HW_TRACE,
      "path": "hw_trace.json",
      "request_artifact": "trace_request.json",
      "ingest_command": ["boltbeam", "ingest-timing", "hw_trace.json", "--run", str(run)],
      "required": True,
    },
      {
        "kind": "runner_manifest",
        "schema": "provider.runner_manifest.v1",
        "path": "runner_manifest.json",
        "required": False,
      },
      {
        "kind": "operand_path_result",
        "schema": SCHEMA_EXECUTION_BRIDGE_RESULT,
        "path": "operand_path_result.json",
        "request_artifact": "operand_path_request.json",
        "required": operand_request is not None,
      },
    ],
    "trace_profiles": trace_profiles,
    "provider_commands": _provider_commands(manifest, provider_rows, trace_profiles, run, bundle),
    "notes": [
      "This plan is for audit tracing only. It is not a runtime executor or an inference serving path.",
      "A runtime-side executor should consume the bundle and return normalized JSON evidence.",
      "BoltBeam core validates, ingests, classifies, and reports the returned evidence.",
    ],
  }


def _provider_row(raw:str) -> dict[str, Any]:
  if "=" in raw:
    provider_id, path = raw.split("=", 1)
    p = pathlib.Path(path).expanduser() if path.strip() else None
    return {"provider_id": provider_id.strip(), "configured_path": str(p) if p else None}
  return {"provider_id": raw.strip(), "configured_path": None}


def _provider_commands(manifest:dict[str, Any], providers:list[dict[str, Any]], trace_profiles:list[dict[str, Any]],
                       run:pathlib.Path, bundle:pathlib.Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for provider, trace_profile in zip(providers, trace_profiles):
    pid = normalize_provider_id(provider["provider_id"])  # an arkey hint reads as its recorded id tinygrad
    # tinygrad has no rows: its trace harnesses and search provider left tinygrad-arkey with the kernel generator (retired 2026-11-21)
    if trace_profile.get("status") == "ready" and pid in {"llama", "llama.cpp", "llama-cpp"}:
      rows.extend(_llama_trace_commands(manifest, provider, trace_profile, run, bundle))
  return rows


def _trace_profile_or_blocked(manifest:dict[str, Any], workload_profile:dict[str, Any],
                              trace_request:dict[str, Any], provider:dict[str, Any]) -> dict[str, Any]:
  try:
    return resolve_trace_execution_profile(manifest, workload_profile, trace_request, provider)
  except ValueError as exc:
    return {
      "schema": "boltbeam.trace_execution_profile.v1",
      "status": "blocked",
      "model": {"model_id": manifest.get("model_id"), "format": manifest.get("model_format"),
                "path": manifest.get("model_path"), "sha256": None, "identity_status": "unavailable"},
      "workload": {"kind": manifest.get("workload"), "contexts": list(workload_profile.get("contexts", [])),
                   "execution": dict(trace_request.get("execution") or {})},
      "target": {"target_id": manifest.get("target_id"), "backend": None},
      "runtime": {"provider_id": provider.get("provider_id"), "configured_path": provider.get("configured_path")},
      "collector": {"collector_id": None, "tool": None, "operation": "capture", "supported_metrics": [],
                    "supported_scopes": [], "unsupported_metrics": [], "known_blind_spots": [],
                    "coverage": {"required_levels": {}, "optional_levels": {}, "missing_required_levels": [],
                                 "requested_metrics": {}, "scope_rule": None}},
      "evidence_rule": "blocked profiles emit no trace command",
      "reason": str(exc),
    }


def _tinygrad_target_runtime(target_id:str | None) -> dict[str, Any]:
  target = TARGETS.get(str(target_id)) if target_id else None
  if target is None:
    return {"target_id": target_id, "status": "unknown", "env": {}}
  caps = target.capabilities or {}
  device = caps.get("tinygrad_device")
  if not device:
    device = row.tinygrad_device if (row := vendors.BY_BACKEND.get(target.backend)) else None
  env = {"DEV": str(device)} if device else {}
  if caps.get("compiler_arch"):
    env["ARCH"] = str(caps["compiler_arch"])
  return {
    "target_id": target.target_id,
    "status": "configured" if env else "unconfigured",
    "backend": target.backend,
    "env": env,
  }


def _first_context(run:pathlib.Path, default:int = 512) -> int:
  workload_path = run / "workload_profile.json"
  if workload_path.exists():
    try:
      ctxs = list(read_json(workload_path).get("contexts", []))
      if ctxs: return int(ctxs[0])
    except Exception:
      pass
  return default


def _llama_trace_commands(manifest:dict[str, Any], provider:dict[str, Any], trace_profile:dict[str, Any],
                          run:pathlib.Path, bundle:pathlib.Path) -> list[dict[str, Any]]:
  llama_bench = provider.get("configured_path") or "llama-bench"
  model = manifest.get("model_path") or "<model.gguf>"
  ctx = _first_context(run)
  trace_dir = bundle / "llama_hw_trace"
  trace_path = run / "llama_hw_trace.json"
  timing_trace_path = run / "llama_timing_trace.json"
  collector_cmd = [
    "boltbeam", "collect-hw-trace",
    "--provider", "llama",
    "--run", str(run),
    "--model", str(model),
    "--context", str(ctx),
    "--sampler", trace_profile["collector"]["collector_id"],
    "--llama-bench", str(llama_bench),
    "--trace-dir", str(trace_dir),
    "--out", str(trace_path),
    "--timing-out", str(timing_trace_path),
  ]
  return [
    {
      "id": "llama_boltbeam_trace",
      "purpose": "run the BoltBeam llama.cpp collector and emit boltbeam.hw_trace.v1",
      "command": collector_cmd,
      "command_shell": " ".join(collector_cmd),
      "expected_artifact": str(trace_path),
      "expected_artifacts": [str(timing_trace_path), str(trace_path)],
      "aux_artifact_dir": str(trace_dir),
    },
    {
      "id": "llama_hw_trace_ingest",
      "purpose": "classify the imported hardware trace in this staged run",
      "command": [
        "boltbeam", "ingest-timing", str(trace_path), "--run", str(run),
      ],
      "expected_artifact": str(run / "timing_profile.json"),
    },
  ]


def _fingerprint(path:pathlib.Path) -> str:
  return "sha256:" + sha256_hex(path.read_bytes())


def _input_row(run:pathlib.Path, bundle:pathlib.Path, artifact:str, *, required:bool) -> dict[str, Any]:
  src = run / artifact
  present = src.exists()
  return {
    "artifact": artifact,
    "required": required,
    "status": "present" if present else "missing",
    "path": str(src),
    "bundle_path": str(bundle / artifact) if present else None,
    "schema": _schema(src) if present else None,
    "fingerprint": _fingerprint(src) if present else None,
  }


def _schema(path:pathlib.Path) -> str | None:
  try:
    obj = read_json(path)
  except Exception:
    return None
  return obj.get("schema") if isinstance(obj, dict) else None


def _write_bundle(bundle:pathlib.Path, inputs:list[dict[str, Any]]) -> None:
  bundle.mkdir(parents=True, exist_ok=True)
  for row in inputs:
    src = pathlib.Path(row["path"])
    dst = pathlib.Path(row["bundle_path"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
