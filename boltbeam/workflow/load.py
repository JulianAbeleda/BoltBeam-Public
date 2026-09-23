from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.profile.loaders import detect_model_format, profile_from_model
from boltbeam.quantization.quant import quant_capability
from boltbeam.vocab import (SCHEMA_WEIGHT_INVENTORY, SCHEMA_WORKLOAD_PROFILE,
                            Workload, SCHEMA_RUN_MANIFEST)
from boltbeam.workflow.common import run_dir, write_json, write_manifest

LOAD_ARTIFACTS = ("run_manifest.json", "model_profile.json", "weight_inventory.json", "workload_profile.json")


def _role_elements(role) -> int:
  depth = role.n_expert if role.n_expert else 1
  return int(role.rows) * int(role.cols) * int(role.count) * int(depth)


def _role_inventory(profile) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for role in profile.roles:
    cap = quant_capability(role.quant)
    elems = _role_elements(role)
    bytes_est = int(elems * cap.bytes_per_elem) if cap else None
    rows.append({
      "role": role.role,
      "role_class": role.role_class,
      "tensor_name": role.tensor_name,
      "shape": [role.rows, role.cols],
      "count": role.count,
      "n_expert": role.n_expert,
      "quant": role.quant,
      "elements": elems,
      "estimated_weight_bytes": bytes_est,
      "quant_known": cap is not None,
    })
  return rows


def _weight_inventory(profile) -> dict[str, Any]:
  roles = _role_inventory(profile)
  known_bytes = [r["estimated_weight_bytes"] for r in roles if r["estimated_weight_bytes"] is not None]
  return {
    "schema": SCHEMA_WEIGHT_INVENTORY,
    "model_id": profile.model_id,
    "source": profile.source,
    "architecture_class": profile.architecture_class,
    "role_count": len(roles),
    "total_known_weight_bytes": sum(known_bytes),
    "unknown_quant_roles": [r for r in roles if not r["quant_known"]],
    "roles": roles,
  }


def _workload_profile(profile, target_id:str, workload:str, contexts:tuple[int, ...]) -> dict[str, Any]:
  if workload not in {w.value for w in Workload}:
    raise ValueError(f"unknown workload {workload!r}")
  return {
    "schema": SCHEMA_WORKLOAD_PROFILE,
    "model_id": profile.model_id,
    "target_id": target_id,
    "workload": workload,
    "contexts": list(contexts),
    "constraints": {
      "correctness_required": True,
      "rollback_required_for_promotion": True,
      "provider_neutral": True,
    },
  }


def load_run(model:str | pathlib.Path, run:str | pathlib.Path, *, target_id:str | None = None,
             model_id:str | None = None, workload:str = Workload.DECODE.value,
             contexts:tuple[int, ...] = (128, 512), target_source:str | None = None) -> dict[str, Any]:
  out = run_dir(run)
  model_path = pathlib.Path(model).expanduser()
  selected_target = target_id or "amd_gfx1100"
  selected_target_source = target_source or ("user" if target_id else "default")
  fmt = detect_model_format(model_path)
  profile = profile_from_model(model_path, model_id=model_id)
  write_json(out / "model_profile.json", profile.to_json())
  write_json(out / "weight_inventory.json", _weight_inventory(profile))
  write_json(out / "workload_profile.json", _workload_profile(profile, selected_target, workload, contexts))
  manifest = {
    "schema": SCHEMA_RUN_MANIFEST,
    "model_id": profile.model_id,
    "model_path": str(model_path),
    "model_format": fmt,
    "target_id": selected_target,
    "target_source": selected_target_source,
    "workload": workload,
    "latest_stage": "load",
    "stages": {"load": {"artifacts": list(LOAD_ARTIFACTS)}},
    "artifacts": list(LOAD_ARTIFACTS),
  }
  return write_manifest(out, manifest)
