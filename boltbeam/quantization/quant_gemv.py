from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.core.canonical import sha256_hex

from boltbeam.profile.ir import ModelProfile, TargetProfile, TensorRole
from boltbeam.quantization.quant import quant_capability, route_families_for
from boltbeam.vocab import (SCHEMA_PRIMITIVE_PROFILE, SCHEMA_PROBE_EVIDENCE,
                            SCHEMA_PROBE_REQUEST)

_REQUIRED_METRICS = {
  "timing": ["candidate_us", "spread_pct"],
  "bytes": ["physical_weight_bytes", "metadata_bytes", "activation_bytes"],
  "throughput": ["achieved_gbs", "target_gbs"],
  "latency_concurrency": ["memory_latency_ns", "active_waves", "in_flight_groups", "occupancy_pct"],
  "isa": ["instruction_histogram", "vector_load_bits"],
  "resources": ["registers", "lds_bytes", "scratch_bytes"],
}

_BITUNPACK_TOKENS = ("bfe", "lshr", "alignbit", "and")
_CONVERT_TOKENS = ("cvt", "convert")
_DOT_TOKENS = ("dot", "mma", "wmma", "fma", "fmac")
_META_TOKENS = ("scale", "zero", "metadata")


def quant_gemv_roles(profile:ModelProfile) -> list[TensorRole]:
  roles: list[TensorRole] = []
  for role in profile.roles:
    cap = quant_capability(role.quant)
    if cap is None or cap.dequant_family == "float_passthrough":
      continue
    if role.rows <= 0 or role.cols <= 0:
      continue
    roles.append(role)
  return roles


def probe_id(role:TensorRole) -> str:
  return f"quant_gemv:{role.role}:{role.quant}:{role.rows}x{role.cols}:n{role.count}:e{role.n_expert}"


def build_probe_request(profile:ModelProfile, target:TargetProfile, workload_profile:dict[str, Any]) -> dict[str, Any]:
  probes = []
  for role in quant_gemv_roles(profile):
    cap = quant_capability(role.quant)
    probes.append({
      "probe_id": probe_id(role),
      "kind": "quant_gemv",
      "role": role.role,
      "role_class": role.role_class,
      "tensor_name": role.tensor_name,
      "shape": [role.rows, role.cols],
      "count": role.count,
      "n_expert": role.n_expert,
      "quant": role.quant,
      "dequant_family": cap.dequant_family if cap else None,
      "route_families": list(route_families_for(role.quant)),
      "target_vector_load_bits": target.vector_load_bits,
      "required_metrics": _REQUIRED_METRICS,
    })
  return {
    "schema": SCHEMA_PROBE_REQUEST,
    "model_id": profile.model_id,
    "target_id": target.target_id,
    "workload": workload_profile.get("workload", "decode"),
    "contexts": list(workload_profile.get("contexts", [])),
    "provider_neutral": True,
    "required_metrics": _REQUIRED_METRICS,
    "probes": probes,
    "notes": [
      "External runners should return boltbeam.probe_evidence.v1.",
      "Full ISA/resource evidence is preferred; missing non-identity fields classify as inconclusive.",
    ],
  }


def build_primitive_profile(evidence:dict[str, Any], *, request:dict[str, Any] | None = None,
                            source_path:str | pathlib.Path | None = None) -> dict[str, Any]:
  _validate_evidence_identity(evidence)
  request_index = {p["probe_id"]: p for p in (request or {}).get("probes", [])}
  regimes = []
  for row in evidence.get("probes", []):
    if not isinstance(row, dict):
      raise ValueError("probe_evidence probes must be JSON objects")
    for key in ("probe_id", "kind"):
      if not row.get(key):
        raise ValueError(f"probe row missing required identity field {key!r}")
    if row.get("kind") != "quant_gemv":
      continue
    regimes.append(classify_quant_gemv(row, request_index.get(row["probe_id"])))
  return {
    "schema": SCHEMA_PRIMITIVE_PROFILE,
    "model_id": evidence["model_id"],
    "target_id": evidence["target_id"],
    "workload": evidence["workload"],
    "provider_id": evidence.get("provider_id", "external"),
    "source": {
      "path": str(source_path) if source_path is not None else evidence.get("source_path"),
      "fingerprint": _fingerprint(evidence),
      "schema": evidence.get("schema"),
    },
    "request_probe_count": len((request or {}).get("probes", [])),
    "evidence_probe_count": len(evidence.get("probes", [])),
    "quant_gemv_regimes": regimes,
  }


def classify_quant_gemv(row:dict[str, Any], request_probe:dict[str, Any] | None = None) -> dict[str, Any]:
  timing = dict(row.get("timing", {}))
  bytes_ = dict(row.get("bytes", {}))
  throughput = dict(row.get("throughput", {}))
  lat = dict(row.get("latency_concurrency", {}))
  isa = dict(row.get("isa", {}))
  resources = dict(row.get("resources", {}))
  hist = dict(isa.get("instruction_histogram", {}))
  missing = _missing_fields(timing, bytes_, throughput, lat, isa, resources)
  features = _features(timing, bytes_, throughput, lat, isa, resources, hist, request_probe)
  classification, bottleneck, reasons, next_action = _classify(features, missing)
  return {
    "probe_id": row["probe_id"],
    "kind": "quant_gemv",
    "role": row.get("role") or (request_probe or {}).get("role"),
    "quant": row.get("quant") or (request_probe or {}).get("quant"),
    "shape": row.get("shape") or (request_probe or {}).get("shape"),
    "classification": classification,
    "visible_bottleneck": bottleneck,
    "next_action": next_action,
    "reasons": reasons,
    "missing_fields": missing,
    "metrics": features,
    "marlin_viability": _marlin_viability(features, classification),
  }


def _validate_evidence_identity(evidence:dict[str, Any]) -> None:
  if evidence.get("schema") != SCHEMA_PROBE_EVIDENCE:
    raise ValueError(f"expected schema {SCHEMA_PROBE_EVIDENCE}, got {evidence.get('schema')!r}")
  for key in ("model_id", "target_id", "workload"):
    if not evidence.get(key):
      raise ValueError(f"probe evidence missing required field {key!r}")
  if not isinstance(evidence.get("probes", []), list):
    raise ValueError("probe evidence field 'probes' must be a list")


def _fingerprint(obj:dict[str, Any]) -> str:
  raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
  return "sha256:" + sha256_hex(raw)


def _as_float(d:dict[str, Any], key:str) -> float | None:
  v = d.get(key)
  if v is None:
    return None
  try:
    return float(v)
  except (TypeError, ValueError):
    return None


def _hist_count(hist:dict[str, Any], tokens:tuple[str, ...]) -> float:
  total = 0.0
  for name, val in hist.items():
    n = name.lower()
    if any(tok in n for tok in tokens):
      try:
        total += float(val)
      except (TypeError, ValueError):
        pass
  return total


def _missing_fields(timing:dict[str, Any], bytes_:dict[str, Any], throughput:dict[str, Any],
                    lat:dict[str, Any], isa:dict[str, Any], resources:dict[str, Any]) -> list[str]:
  groups = {
    "timing": timing,
    "bytes": bytes_,
    "throughput": throughput,
    "latency_concurrency": lat,
    "isa": isa,
    "resources": resources,
  }
  wanted = {
    "timing": ("candidate_us", "spread_pct"),
    "bytes": ("physical_weight_bytes", "metadata_bytes", "activation_bytes"),
    "throughput": ("achieved_gbs", "target_gbs"),
    "latency_concurrency": ("memory_latency_ns", "active_waves", "in_flight_groups", "occupancy_pct"),
    "isa": ("instruction_histogram", "vector_load_bits"),
    "resources": ("registers", "lds_bytes", "scratch_bytes"),
  }
  missing: list[str] = []
  for group, keys in wanted.items():
    for key in keys:
      if groups[group].get(key) is None:
        missing.append(f"{group}.{key}")
  return missing


def _features(timing:dict[str, Any], bytes_:dict[str, Any], throughput:dict[str, Any],
              lat:dict[str, Any], isa:dict[str, Any], resources:dict[str, Any],
              hist:dict[str, Any], request_probe:dict[str, Any] | None) -> dict[str, Any]:
  weight_b = _as_float(bytes_, "physical_weight_bytes") or 0.0
  metadata_b = _as_float(bytes_, "metadata_bytes") or 0.0
  activation_b = _as_float(bytes_, "activation_bytes") or 0.0
  total_b = max(weight_b + metadata_b + activation_b, 0.0)
  achieved_gbs = _as_float(throughput, "achieved_gbs")
  target_gbs = _as_float(throughput, "target_gbs")
  memory_latency_ns = _as_float(lat, "memory_latency_ns")
  vector_bits = _as_float(isa, "vector_load_bits")
  active_waves = _as_float(lat, "active_waves")
  in_flight = _as_float(lat, "in_flight_groups")
  occupancy_pct = _as_float(lat, "occupancy_pct")
  required_bytes = memory_latency_ns * target_gbs if memory_latency_ns is not None and target_gbs is not None else None
  available_bytes = None
  if vector_bits is not None and active_waves is not None:
    available_bytes = (vector_bits / 8.0) * active_waves * max(in_flight if in_flight is not None else 1.0, 1.0)
  bitunpack = _as_float(isa, "dequant_bitunpack_instruction_count")
  convert = _as_float(isa, "dequant_convert_instruction_count")
  dequant = _as_float(isa, "dequant_instruction_count")
  if bitunpack is None:
    bitunpack = _hist_count(hist, _BITUNPACK_TOKENS)
  if convert is None:
    convert = _hist_count(hist, _CONVERT_TOKENS)
  if dequant is None:
    dequant = bitunpack + convert
  dot = _as_float(isa, "dot_instruction_count")
  if dot is None:
    dot = _hist_count(hist, _DOT_TOKENS)
  metadata_instr = _as_float(isa, "metadata_instruction_count")
  if metadata_instr is None:
    metadata_instr = _hist_count(hist, _META_TOKENS)
  route_instr = max(dequant + dot + metadata_instr, 0.0)
  return {
    "candidate_us": _as_float(timing, "candidate_us"),
    "baseline_us": _as_float(timing, "baseline_us"),
    "spread_pct": _as_float(timing, "spread_pct"),
    "physical_weight_bytes": weight_b,
    "metadata_bytes": metadata_b,
    "activation_bytes": activation_b,
    "weight_byte_fraction": (weight_b / total_b) if total_b > 0 else None,
    "metadata_byte_fraction": (metadata_b / total_b) if total_b > 0 else None,
    "achieved_gbs": achieved_gbs,
    "target_gbs": target_gbs,
    "pct_of_target_gbs": (100.0 * achieved_gbs / target_gbs) if achieved_gbs is not None and target_gbs else None,
    "memory_latency_ns": memory_latency_ns,
    "active_waves": active_waves,
    "in_flight_groups": in_flight,
    "occupancy_pct": occupancy_pct,
    "required_concurrency_bytes": required_bytes,
    "available_concurrency_bytes": available_bytes,
    "concurrency_ratio": (available_bytes / required_bytes) if available_bytes is not None and required_bytes else None,
    "vector_load_bits": vector_bits,
    "target_vector_load_bits": (request_probe or {}).get("target_vector_load_bits"),
    "registers": _as_float(resources, "registers"),
    "lds_bytes": _as_float(resources, "lds_bytes"),
    "scratch_bytes": _as_float(resources, "scratch_bytes"),
    "barrier_count": _as_float(resources, "barrier_count") or _as_float(isa, "barrier_count"),
    "waitcnt_count": _as_float(resources, "waitcnt_count") or _as_float(isa, "waitcnt_count"),
    "bitunpack_instruction_count": bitunpack,
    "convert_instruction_count": convert,
    "dequant_instruction_count": dequant,
    "dot_instruction_count": dot,
    "metadata_instruction_count": metadata_instr,
    "dequant_instruction_fraction": (dequant / route_instr) if route_instr > 0 else None,
    "metadata_instruction_fraction": (metadata_instr / route_instr) if route_instr > 0 else None,
    "reduction_us": _as_float(timing, "reduction_us"),
    "dequant_us": _as_float(timing, "dequant_us"),
    "metadata_us": _as_float(timing, "metadata_us"),
    "compute_util_pct": _as_float(throughput, "compute_util_pct"),
  }


def _classify(m:dict[str, Any], missing:list[str]) -> tuple[str, str, list[str], str]:
  if missing:
    return ("inconclusive", "missing_evidence",
            [f"missing required full-probe fields: {', '.join(missing[:6])}" + ("..." if len(missing) > 6 else "")],
            "rerun the external probe with timing, bytes, latency/concurrency, and ISA/resource rows")

  reasons: list[str] = []
  metadata_time = m.get("metadata_us")
  candidate_us = m.get("candidate_us") or 0.0
  if metadata_time is not None and candidate_us > 0 and metadata_time / candidate_us >= 0.15:
    reasons.append(f"metadata_us is {100.0 * metadata_time / candidate_us:.1f}% of candidate time")
    return "metadata_bound", "metadata_load", reasons, "change scale/zero layout or hoist metadata out of the hot loop"
  if (m.get("metadata_byte_fraction") or 0.0) >= 0.15 or (m.get("metadata_instruction_fraction") or 0.0) >= 0.20:
    reasons.append("metadata traffic/instructions are visible on the hot path")
    return "metadata_bound", "metadata_load", reasons, "use a Marlin-style metadata layout matched to thread/block access"

  dequant_time = m.get("dequant_us")
  if dequant_time is not None and candidate_us > 0 and dequant_time / candidate_us >= 0.25:
    reasons.append(f"dequant_us is {100.0 * dequant_time / candidate_us:.1f}% of candidate time")
    return "dequant_bound", "dequant", reasons, "reduce or overlap bit-unpack/convert work; dequant output layout must feed the dot path directly"
  if (m.get("dequant_instruction_fraction") or 0.0) >= 0.50 and (m.get("pct_of_target_gbs") or 0.0) < 75.0:
    reasons.append("dequant instructions dominate while achieved bandwidth remains below target")
    return "dequant_bound", "dequant", reasons, "measure convert-avoidance and direct-to-fragment dequant variants"

  ratio = m.get("concurrency_ratio")
  if ratio is not None and ratio < 1.0:
    reasons.append(f"available concurrency is {ratio:.2f}x of latency*throughput requirement")
    if (m.get("occupancy_pct") or 0.0) < 50.0:
      return "occupancy_starved", "insufficient_residency", reasons, "increase independent tiles/splits or reduce resource pressure"
    return "latency_bound", "insufficient_inflight_work", reasons, "add ILP/pipeline depth or independent groups before chasing throughput"

  if (m.get("compute_util_pct") or 0.0) >= 70.0:
    reasons.append("compute utilization is high relative to supplied target")
    return "compute_bound", "dot_or_mma", reasons, "prefer tensor/dot throughput levers and avoid extra metadata/dequant work"

  if (m.get("weight_byte_fraction") or 0.0) >= 0.80:
    reasons.append("physical weight bytes dominate and auxiliary work is not visible")
    return "streaming_bound", "weight_load", reasons, "preserve wide coalesced packed loads; only compression or layout that keeps load-bound behavior can help"

  return "inconclusive", "mixed", ["no single Volkov/Marlin bottleneck rule matched"], "collect a focused per-stage timing or ISA breakdown"


def _marlin_viability(m:dict[str, Any], classification:str) -> dict[str, Any]:
  target_bits = m.get("target_vector_load_bits")
  vector_bits = m.get("vector_load_bits")
  ratio = m.get("concurrency_ratio")
  reduction_us = m.get("reduction_us")
  candidate_us = m.get("candidate_us")
  return {
    "offline_layout_useful": bool(classification in {"metadata_bound", "dequant_bound"} or
                                  (target_bits and vector_bits and vector_bits < target_bits)),
    "packed_load_width_sufficient": (None if target_bits is None or vector_bits is None else vector_bits >= target_bits),
    "dequant_hidden": classification != "dequant_bound",
    "metadata_hidden": classification != "metadata_bound",
    "pipeline_depth_sufficient": None if ratio is None else ratio >= 1.0,
    "partitioning_overhead_acceptable": (
      None if reduction_us is None or not candidate_us else (reduction_us / candidate_us) < 0.10
    ),
  }
