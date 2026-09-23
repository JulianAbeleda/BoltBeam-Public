"""Adapt tinygrad ``decode_runtime_overhead.py`` authority JSON without executing tinygrad."""
from __future__ import annotations

from typing import Any

from boltbeam.vocab import SCHEMA_TIMING_TRACE

AUTHORITY_SCHEMA = "tinygrad.decode.fixed_depth.v2"


def _require(value: Any, label: str) -> Any:
  if value is None or value == "":
    raise ValueError(f"decode authority missing {label}")
  return value


def validate_decode_authority(authority: dict[str, Any]) -> None:
  """Fail closed unless artifact, lifecycle, model, and route identity agree."""
  if authority.get("schema") != AUTHORITY_SCHEMA:
    raise ValueError(f"expected {AUTHORITY_SCHEMA}, got {authority.get('schema')!r}")
  if authority.get("artifact_version") != 2:
    raise ValueError("unsupported decode authority artifact_version")
  _require(authority.get("model_id"), "model_id")
  model, identity = authority.get("model"), authority.get("model_identity")
  if not isinstance(model, dict) or not isinstance(identity, dict):
    raise ValueError("decode authority missing model identity")
  for key in ("identity_sha256", "size_bytes", "mtime_ns"):
    if model.get(key) != identity.get(key):
      raise ValueError(f"model identity mismatch for {key}")
  workload = authority.get("workload")
  if not isinstance(workload, dict) or workload.get("decode_tokens") != 1:
    raise ValueError("authority is not a one-token fixed-depth decode lifecycle")
  if authority.get("nmeas") != authority.get("reps") or authority.get("reps") != workload.get("reps"):
    raise ValueError("authority lifecycle repetitions disagree")
  rows = authority.get("rows")
  if not isinstance(rows, list) or not rows:
    raise ValueError("decode authority has no rows")
  for row in rows:
    ctx = _require(row.get("ctx"), "row.ctx")
    if row.get("fixed_depth") != ctx or row.get("decode_tokens") != 1:
      raise ValueError("row is not a fixed-depth one-token decode")
    route = _require(row.get("route"), "row.route")
    sequence = row.get("route_sequence")
    if not isinstance(sequence, list) or sequence != [route] or not row.get("route_sequences_identical"):
      raise ValueError("route sequence is not identical to the selected route")
    programs = row.get("programs_per_token_by_route")
    if not isinstance(programs, dict) or not isinstance(programs.get(route), int):
      raise ValueError("route program count missing for selected route")
    if not row.get("generated_reps_identical"):
      raise ValueError("generated-token lifecycle is not identical across repetitions")
    for key in ("W_reps", "D_reps"):
      repetitions = row.get(key)
      expected = 0 if key == "D_reps" and workload.get("dispatch_diagnostic") is False else authority["reps"]
      if not isinstance(repetitions, list) or len(repetitions) != expected:
        raise ValueError(f"{key} does not match declared lifecycle repetitions")


def adapt_decode_timing_trace(authority: dict[str, Any], *, target_id: str) -> dict[str, Any]:
  """Emit ``boltbeam.timing_trace.v1`` whole-step W/D rows from authoritative JSON."""
  validate_decode_authority(authority)
  rows = []
  counter_evidence = {
    "status": "unavailable",
    "reason": "decode_runtime_overhead authority JSON contains no hardware counter stream",
    "host_sync_pct": authority.get("median_host_sync_pct"),
    "host_sync_residual_ms": None,
  }
  for source in authority["rows"]:
    context, route = int(source["ctx"]), str(source["route"])
    common = {
      "scope": "whole_step", "context": context, "depth": int(source["fixed_depth"]),
      "route": route, "route_sequence": list(source["route_sequence"]),
      "programs_per_token": source["programs_per_token_by_route"][route],
      "programs_per_token_by_route": dict(source["programs_per_token_by_route"]),
      "counter_evidence": dict(counter_evidence, host_sync_residual_ms=source.get("host_sync_residual_ms")),
      "generated_token_evidence": source.get("generated_token_evidence"),
    }
    rows.append(dict(common, measurement="W", wall_us=float(source["wall_ms_W"]) * 1000.0,
                     tok_s=float(source["tok_s_W"]), time_source="production_generate_item_token"))
    if authority["workload"].get("dispatch_diagnostic") is not False:
      rows.append(dict(common, measurement="D", wall_us=float(source["dispatch_ms_D"]) * 1000.0,
                       tok_s=float(source["tok_s_D_diagnostic"]), time_source="same_model_jit_final_sync_diagnostic",
                       interpretation=source["D_interpretation"]))
  contexts = sorted({row["context"] for row in rows})
  return {
    "schema": SCHEMA_TIMING_TRACE, "model_id": authority["model_id"], "target_id": target_id,
    "workload": "decode", "provider_id": "tinygrad/decode_runtime_overhead.py",
    "timing_source": "tinygrad_decode_authority", "contexts": contexts,
    "metadata": {"authority_schema": AUTHORITY_SCHEMA, "artifact_version": 2,
                 "model_identity": authority["model_identity"], "runtime_settings": authority.get("runtime_settings"),
                 "counter_evidence": counter_evidence},
    "rows": rows,
  }
