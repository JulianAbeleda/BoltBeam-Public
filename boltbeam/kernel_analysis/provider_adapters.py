"""KA-3: adapters for real tinygrad artifact schemas.

These normalize existing provider artifacts into `EvidenceFragment`s that join, by identity, with an
execution fragment to form one `KernelEvidence`. They reuse the repository's own pure validators
(`search.resource_join`, `search.transfer_contract`) rather than duplicating field rules, and never
import tinygrad runtime internals. A malformed known artifact becomes a structured invalid result.
"""
from __future__ import annotations

from typing import Any, Mapping

from boltbeam.artifacts.base import EvidenceSource
from boltbeam.kernel_analysis.adapters import (
  AdaptResult, EvidenceFragment, OK, _AdapterInvalid, register_kernel_adapter,
)
from boltbeam.kernel_analysis.model import FactBlocker, KernelHash
from boltbeam.vocab import SCHEMA_EXECUTION_BRIDGE_RESULT, SCHEMA_HW_TRACE
from boltbeam.search.joins.resource_join import SCHEMA as RESOURCE_TRACE_SCHEMA, validate_resource_snapshot
from boltbeam.search.transfer_contract import (
  AMD_ISA_PROOF_MANIFEST_SCHEMA, validate_amd_isa_proof_manifest,
)

# ISA row kind -> normalized structure counter it contributes to.
_ISA_KIND_TO_STRUCTURE = {
  "global_load": "global_loads",
  "global_load_b128": "global_loads",
  "global_store": "global_stores",
  "buffer_load": "global_loads",
  "buffer_store": "global_stores",
  "scalar_load": "scalar_loads",
  "ds_load": "shared_loads",
  "ds_load_b128": "shared_loads",
  "ds_store": "shared_stores",
  "ds_store_b128": "shared_stores",
  "shared_load": "shared_loads",
  "shared_store": "shared_stores",
  "scratch_load": "scratch_loads",
  "scratch_store": "scratch_stores",
  "private_load": "scratch_loads",
  "private_store": "scratch_stores",
  "barrier": "barriers",
  "wait": "waits",
  "waitcnt": "waits",
  "wmma": "tensor_ops",
  "tensor_op": "tensor_ops",
  "branch": "branches",
  "predicate": "predicates",
  "loop": "loops",
  "kernel_exit": "exits",
  "exit": "exits",
}

_MEMORY_KINDS = {"global_load", "global_load_b128", "global_store", "buffer_load", "buffer_store", "scalar_load",
                 "ds_load", "ds_load_b128", "ds_store", "ds_store_b128", "shared_load", "shared_store",
                 "scratch_load", "scratch_store", "private_load", "private_store"}


def _hash_or_none(value: Any) -> str | None:
  if not isinstance(value, str) or not value:
    return None
  try:
    return KernelHash(value).value
  except ValueError:
    return None


def _identity_from(mapping: Mapping[str, Any]) -> dict[str, Any]:
  ident: dict[str, Any] = {}
  binary = _hash_or_none(mapping.get("binary_sha256") or mapping.get("binary_hash"))
  if binary:
    ident["binary_hash"] = binary
  for src_key, dst_key in (("experiment_id", "experiment_id"), ("system_snapshot_id", "system_snapshot_id"),
                           ("session_id", "session_id")):
    if isinstance(mapping.get(src_key), str) and mapping[src_key]:
      ident[dst_key] = mapping[src_key]
  return ident


def adapt_resource_trace(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  """Normalize a `tinygrad.kernel_resource_trace.v1` into a resource fragment."""
  result = validate_resource_snapshot(mapping)
  if not result.ok:
    fields = tuple(result.missing) + tuple(str(i) for i in result.invalid)
    raise _AdapterInvalid(f"resource trace invalid: missing={result.missing} invalid={result.invalid}", fields)
  res = dict(result.resources or {})
  resources: dict[str, Any] = {}
  for key in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "scratch_loads", "scratch_stores",
              "spilled_vgprs", "spilled_sgprs", "workgroup_threads", "occupancy"):
    if key in res:
      resources[key] = res[key]
  for key in ("workgroup", "grid"):
    if key in res:
      resources[key] = list(res[key])
  candidate_id = result.candidate_id or mapping.get("candidate_id")
  payload = {
    "candidate": {"candidate_id": candidate_id} if candidate_id else {},
    "identity": _identity_from(mapping),
    "resources": resources,
  }
  return AdaptResult(OK, fragment=EvidenceFragment(payload=payload, provenance=(source,)),
                     schema=RESOURCE_TRACE_SCHEMA)


def adapt_isa_manifest(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  """Normalize a `tinygrad.amd_isa_proof_manifest.v1` into a structure fragment."""
  validation = validate_amd_isa_proof_manifest(mapping)
  if not validation.ok:
    fields = tuple(validation.missing) + tuple(validation.invalid)
    raise _AdapterInvalid(f"ISA manifest invalid: {fields}", fields)
  rows = list(mapping.get("rows", []))
  counts: dict[str, int] = {"instructions": len(rows)}
  for row in rows:
    kind = row.get("kind")
    field = _ISA_KIND_TO_STRUCTURE.get(kind)
    if field:
      counts[field] = counts.get(field, 0) + 1
  ident = _identity_from(mapping)
  source_hash = _hash_or_none(mapping.get("source_sha256") or mapping.get("source_hash"))
  if source_hash: ident["source_hash"] = source_hash
  for key in ("semantic_schedule_digest", "abi_digest", "operand_role_map_digest", "candidate_digest"):
    if isinstance(mapping.get(key), str) and mapping[key]: ident[key] = mapping[key]
  operand_paths, path_blockers = _operand_paths_from_isa(mapping, rows, source)
  # ISA proves the compiled binary emitted these instructions -> its binary is the compiled binary.
  payload = {
    "candidate": {"candidate_id": mapping.get("candidate_id")} if mapping.get("candidate_id") else {},
    "identity": ident,
    "structure": {**counts, "extra": {"isa_rows": [dict(row) for row in rows]}},
  }
  if operand_paths: payload["operand_paths"] = operand_paths
  return AdaptResult(OK, fragment=EvidenceFragment(payload=payload, provenance=(source,), blockers=tuple(path_blockers)),
                     schema=AMD_ISA_PROOF_MANIFEST_SCHEMA)


_PHASE_STATUS = {
  "compile": {"not_attempted": "not_run", "passed": "pass", "failed": "fail", "timed_out": "timeout",
              "unsupported": "unsupported", "blocked": "no_result"},
  "execution": {"not_attempted": "not_run", "passed": "pass", "failed": "runtime_fault", "timed_out": "timeout",
                "unsupported": "no_result", "blocked": "no_result"},
  "correctness": {"not_attempted": "not_run", "passed": "pass", "failed": "fail", "timed_out": "incomplete",
                  "unsupported": "incomplete", "blocked": "incomplete"},
  "timing": {"not_attempted": "not_run", "passed": "measured", "failed": "invalid", "timed_out": "invalid",
             "unsupported": "blocked", "blocked": "blocked"},
}


def _normalized_timing(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
  samples = None
  for key, scale in (("samples_ms", 1.0), ("samples_us", 1e-3), ("samples_ns", 1e-6)):
    if key in evidence:
      values = evidence.get(key)
      if not isinstance(values, (list, tuple)): raise _AdapterInvalid(f"{key} must be a list", (key,))
      samples = [float(v) * scale for v in values]
      break
  if samples is None and "samples" in evidence:
    values, units = evidence.get("samples"), str(evidence.get("units") or "ms")
    if not isinstance(values, (list, tuple)): raise _AdapterInvalid("timing samples must be a list", ("samples",))
    scale = {"ms": 1.0, "us": 1e-3, "ns": 1e-6, "s": 1e3}.get(units)
    if scale is None: raise _AdapterInvalid(f"unsupported timing units {units!r}", ("units",))
    samples = [float(v) * scale for v in values]
  if samples is None: return None
  return {"scope": str(evidence.get("scope") or "kernel"), "samples": samples, "units": "ms",
          "warmups": int(evidence.get("warmups", 0)),
          "repetitions": int(evidence.get("repetitions", len(samples))),
          "inclusion": str(evidence.get("inclusion") or "kernel_only"),
          **({"sync": str(evidence["sync"])} if evidence.get("sync") is not None else {})}


def adapt_execution_result(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  """Normalize an execution-bridge result; optional counter failures do not replace valid timing."""
  for name in ("experiment_id", "candidate_id", "request_digest"):
    if not isinstance(mapping.get(name), str) or not mapping[name]:
      raise _AdapterInvalid(f"execution result missing {name}", (name,))
  phases = mapping.get("phases")
  if not isinstance(phases, list) or not phases: raise _AdapterInvalid("execution result phases must be non-empty", ("phases",))
  extensions = mapping.get("extensions", {})
  if not isinstance(extensions, Mapping): raise _AdapterInvalid("execution result extensions must be object", ("extensions",))
  candidate = dict(extensions.get("candidate", {})) if isinstance(extensions.get("candidate", {}), Mapping) else {}
  candidate["candidate_id"] = mapping["candidate_id"]
  identity = dict(extensions.get("identity", {})) if isinstance(extensions.get("identity", {}), Mapping) else {}
  identity["experiment_id"] = mapping["experiment_id"]
  identity_extra = dict(identity.get("extra", {})) if isinstance(identity.get("extra", {}), Mapping) else {}
  identity_extra["execution_request_digest"] = mapping["request_digest"]
  identity["extra"] = identity_extra
  blockers: list[Any] = []
  stages: dict[str, Any] = {}
  sections: dict[str, Any] = {}
  unsupported_rows: list[dict[str, Any]] = []
  seen_phases: set[str] = set()
  for raw in phases:
    if not isinstance(raw, Mapping): raise _AdapterInvalid("execution result phase must be object", ("phases",))
    phase, producer_status = raw.get("phase"), raw.get("status")
    if not isinstance(phase, str) or not phase or phase in seen_phases:
      raise _AdapterInvalid("phase names must be unique strings", ("phases",))
    seen_phases.add(phase)
    evidence = raw.get("evidence", {})
    if not isinstance(evidence, Mapping): raise _AdapterInvalid(f"{phase} evidence must be object", (phase, "evidence"))
    phase_identity = raw.get("identity", {})
    if not isinstance(phase_identity, Mapping): raise _AdapterInvalid(f"{phase} identity must be object", (phase, "identity"))
    for key, value in phase_identity.items():
      normalized_key = {"binary_sha256": "binary_hash", "executed_binary_sha256": "executed_binary_hash",
                        "source_sha256": "source_hash", "target": "target_id"}.get(key, key)
      if normalized_key in identity and identity[normalized_key] != value:
        blockers.append(FactBlocker("execution_result_identity_conflict", "identity",
          f"phase {phase} conflicts on {normalized_key}", (normalized_key,)))
      else: identity[normalized_key] = value
    unsupported = raw.get("unsupported", [])
    if not isinstance(unsupported, list): raise _AdapterInvalid(f"{phase} unsupported must be a list", (phase, "unsupported"))
    unsupported_rows.extend(dict(v) for v in unsupported if isinstance(v, Mapping))
    if phase in _PHASE_STATUS:
      normalized = _PHASE_STATUS[phase].get(str(producer_status))
      if normalized is None: raise _AdapterInvalid(f"unsupported {phase} phase status {producer_status!r}", (phase, "status"))
      error = raw.get("error") if isinstance(raw.get("error"), Mapping) else {}
      stages[phase] = {"status": normalized, "producer_status": str(producer_status),
                       **({"duration_ms": evidence.get("duration_ms")} if evidence.get("duration_ms") is not None else {}),
                       **({"error_class": str(error.get("code"))} if error.get("code") else {}),
                       **({"bounded_detail": str(error.get("phase"))} if error.get("phase") else {})}
    if phase == "compile":
      resources = evidence.get("resources", evidence.get("resource_summary"))
      if resources is not None: sections["resources"] = resources
      structure = evidence.get("structure")
      isa_summary = evidence.get("isa_structure")
      if structure is None and isinstance(isa_summary, Mapping):
        counts = isa_summary.get("counts", {}) if isinstance(isa_summary.get("counts", {}), Mapping) else {}
        structure = {"instructions": isa_summary.get("row_count"), "global_loads": counts.get("global_load"),
                     "shared_loads": counts.get("ds_load"), "shared_stores": counts.get("ds_store"),
                     "waits": counts.get("wait"), "barriers": counts.get("barrier"),
                     "tensor_ops": counts.get("wmma"), "extra": {"isa_structure": dict(isa_summary)}}
      if structure is not None: sections["structure"] = structure
      if evidence.get("operand_paths") is not None: sections["operand_paths"] = evidence["operand_paths"]
      manifest = evidence.get("final_isa_manifest")
      if isinstance(manifest, Mapping) and isinstance(manifest.get("rows"), list):
        paths, manifest_blockers = _operand_paths_from_isa(manifest, list(manifest["rows"]), source)
        blockers.extend(manifest_blockers)
        if paths: sections["operand_paths"] = paths
    elif phase == "execution" and evidence.get("health") is not None: sections["health"] = evidence["health"]
    elif phase == "correctness" and producer_status in {"passed", "failed"}: sections["correctness"] = dict(evidence)
    elif phase == "timing" and producer_status == "passed":
      timing = _normalized_timing(evidence)
      if timing is None: raise _AdapterInvalid("passed timing phase lacks timing samples", ("timing", "samples"))
      sections["timing"] = timing
    elif phase in {"counter", "counters"} and evidence.get("operand_paths") is not None:
      sections["operand_paths"] = evidence["operand_paths"]
  candidate_extra = dict(candidate.get("extra", {})) if isinstance(candidate.get("extra", {}), Mapping) else {}
  candidate_extra["execution_result_unsupported"] = unsupported_rows
  candidate["extra"] = candidate_extra
  payload = {"candidate": candidate, "identity": identity, "stages": stages, **sections}
  if isinstance(extensions.get("workload"), Mapping): payload["workload"] = dict(extensions["workload"])
  return AdaptResult(OK, fragment=EvidenceFragment(payload=payload, provenance=(source,), blockers=tuple(blockers)),
                     schema=SCHEMA_EXECUTION_BRIDGE_RESULT)


def _operand_paths_from_isa(mapping: Mapping[str, Any], rows: list[Mapping[str, Any]],
                            source: EvidenceSource) -> tuple[dict[str, Any], list[Any]]:
  """Normalize final-row ownership. Unresolved traffic stays whole-kernel and blocks attribution."""
  from boltbeam.kernel_analysis.model import FactBlocker
  ownership = mapping.get("operand_ownership", {})
  ownership = dict(ownership) if isinstance(ownership, Mapping) else {}
  abi = mapping.get("abi_operand_map", {})
  abi = dict(abi) if isinstance(abi, Mapping) else {}
  operand_ids = {str(v) for v in ownership.values() if isinstance(v, str)} | {str(k) for k in abi}
  operand_ids |= {str(r.get("operand_id")) for r in rows if isinstance(r.get("operand_id"), str)}
  sites: dict[str, list[dict[str, Any]]] = {op: [] for op in operand_ids}
  ambiguous: list[str] = []
  disagreements: set[str] = set()
  for index, row in enumerate(rows):
    op_id = row.get("operand_id")
    logical_op = str(row.get("logical_op") or f"row-{index}")
    mapped = ownership.get(logical_op)
    if op_id is None: op_id = mapped
    elif mapped is not None and mapped != op_id: disagreements.update((str(op_id), str(mapped)))
    source_op = row.get("source_operand_id")
    if source_op is not None and op_id is not None and source_op != op_id: disagreements.add(str(op_id))
    if row.get("kind") in _MEMORY_KINDS and not isinstance(op_id, str):
      ambiguous.append(logical_op); continue
    if not isinstance(op_id, str): continue
    operand_ids.add(op_id); sites.setdefault(op_id, [])
    known = {"schema", "kind", "logical_op", "emitted", "operand_id", "source_operand_id", "width_bytes",
             "vector_width_bytes", "address", "instruction_address", "fetch_group", "cache_policy"}
    sites[op_id].append({
      "operation_id": logical_op, "kind": str(row.get("kind")),
      **({"width_bytes": row.get("width_bytes", row.get("vector_width_bytes"))}
         if row.get("width_bytes", row.get("vector_width_bytes")) is not None else {}),
      **({"address": str(row.get("instruction_address", row.get("address")))}
         if row.get("instruction_address", row.get("address")) is not None else {}),
      **({"fetch_group": str(row["fetch_group"])} if row.get("fetch_group") is not None else {}),
      **({"cache_policy": str(row["cache_policy"])} if row.get("cache_policy") is not None else {}),
      "extra": {k: v for k, v in row.items() if k not in known},
    })
  blockers: list[FactBlocker] = []
  if ambiguous:
    blockers.append(FactBlocker("ambiguous_operand_ownership", "operand_path",
                                "final memory operations cannot be assigned to a semantic operand", tuple(ambiguous)))
  if disagreements:
    blockers.append(FactBlocker("source_binary_ownership_disagreement", "operand_path",
                                "source and final-program operand ownership disagree", tuple(sorted(disagreements))))
  paths: dict[str, Any] = {}
  declared_map = mapping.get("operand_transports", {})
  declared_map = dict(declared_map) if isinstance(declared_map, Mapping) else {}
  for op_id in sorted(operand_ids):
    op_sites = sites.get(op_id, [])
    declared_raw = declared_map.get(op_id, {})
    try:
      from boltbeam.kernel_analysis.model import OperandTransport
      declared = OperandTransport.from_json(declared_raw).declared_strategy
    except ValueError:
      declared = "unknown"
      blockers.append(FactBlocker("contradictory_declared_strategy", f"operand:{op_id}", fields=(op_id,)))
    attribution = ("source_binary_disagreement" if op_id in disagreements else
                   "ambiguous" if ambiguous else ("final_isa" if op_sites else "unknown"))
    fetch_groups = tuple(dict.fromkeys(s["fetch_group"] for s in op_sites if s.get("fetch_group")))
    paths[op_id] = {"operand_id": op_id, "semantic_role": str(abi.get(op_id, {}).get("semantic_role", op_id))
                    if isinstance(abi.get(op_id), Mapping) else op_id, "scope": "kernel",
                    "declared_strategy": declared,
                    "static": {"instruction_sites": op_sites, "compulsory_fetch_groups": list(fetch_groups),
                               "attribution_status": attribution},
                    "dynamic": {"scope": "per_operand"},
                    "provenance": [source.fingerprint]}
  return paths, blockers




def _workload_from_row(row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
  shape = row.get("shape")
  shape_map: dict[str, int] = {}
  if isinstance(shape, (list, tuple)):
    keys = ("m", "n", "k") if len(shape) == 3 else tuple(f"d{i}" for i in range(len(shape)))
    for key, value in zip(keys, shape):
      if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        shape_map[key] = value
  wl = {
    "operation": str(row.get("kind") or "unknown"),
    "role": str(row.get("role") or "unknown"),
    "shape": shape_map,
    "extra": {k: v for k, v in (("quant", row.get("quant")), ("context", row.get("context")),
                                ("model_id", trace.get("model_id"))) if v is not None},
  }
  if row.get("quant"):
    wl["dtypes"] = {"input": str(row["quant"])}
  return wl


def adapt_hw_trace(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  """Normalize a real `boltbeam.hw_trace.v1` per-kernel trace into one evidence fragment per kernel row.

  These traces carry measured per-kernel timing but no correctness oracle, so each candidate is
  diagnosis-eligible and timing-present but not performance-eligible until correctness is supplied —
  the eligibility authority reports exactly that rather than fabricating a verdict.
  """
  rows = mapping.get("rows")
  if not isinstance(rows, list):
    raise _AdapterInvalid("hw_trace missing 'rows' list", ("rows",))
  provider = mapping.get("provider_id")
  target = mapping.get("target_id")
  health = _health_from_aux(mapping.get("aux_sources"))
  fragments: list[EvidenceFragment] = []
  for index, row in enumerate(rows):
    if not isinstance(row, Mapping) or row.get("scope") != "kernel":
      continue
    name = row.get("kernel")
    if not name:
      continue
    calls = row.get("calls") or 1
    wall_us = row.get("wall_us")
    timing = None
    timing_status = "not_run"
    if isinstance(wall_us, (int, float)) and not isinstance(wall_us, bool) and wall_us >= 0:
      per_call_ms = (float(wall_us) / max(calls, 1)) / 1000.0
      timing = {"scope": "kernel", "samples": [per_call_ms], "units": "ms",
                "inclusion": "profile_scaled_per_call",
                "sync": str(row.get("time_source") or "")}
      timing_status = "measured"
    payload = {
      "candidate": {"candidate_id": str(name)},
      "identity": {k: v for k, v in (("provider", provider), ("target_id", target)) if v},
      "workload": _workload_from_row(row, mapping),
      "stages": {"compile": {"status": "pass"}, "execution": {"status": "pass"},
                 "timing": {"status": timing_status}},
    }
    if timing is not None:
      payload["timing"] = timing
    if health is not None:
      payload["health"] = health
    fragments.append(EvidenceFragment(payload=payload, provenance=(source,)))
  if not fragments:
    raise _AdapterInvalid("hw_trace contained no kernel-scope rows", ("rows",))
  return AdaptResult(OK, fragments=tuple(fragments), schema=SCHEMA_HW_TRACE)


def _health_from_aux(aux: Any) -> dict[str, Any] | None:
  """Surface the gpu_health already carried in a trace's aux_sources as a HealthSummary payload."""
  if not isinstance(aux, Mapping):
    return None
  gh = aux.get("gpu_health")
  if not isinstance(gh, Mapping):
    return None
  pre = gh.get("preflight") if isinstance(gh.get("preflight"), Mapping) else {}
  post = gh.get("postrun") if isinstance(gh.get("postrun"), Mapping) else {}
  out: dict[str, Any] = {}
  if "ok" in pre:
    out["preflight"] = bool(pre["ok"])
  if "ok" in post:
    out["postflight"] = bool(post["ok"])
  return out or None


def register_provider_adapters() -> None:
  register_kernel_adapter(RESOURCE_TRACE_SCHEMA, adapt_resource_trace)
  register_kernel_adapter(AMD_ISA_PROOF_MANIFEST_SCHEMA, adapt_isa_manifest)
  register_kernel_adapter(SCHEMA_HW_TRACE, adapt_hw_trace)
  register_kernel_adapter(SCHEMA_EXECUTION_BRIDGE_RESULT, adapt_execution_result)
