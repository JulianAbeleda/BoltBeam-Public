from __future__ import annotations

from typing import Any

from boltbeam.profile.ir import ModelProfile


def emit_fixture_manifest(profile:ModelProfile) -> dict[str, Any]:
  """Emit shape-only fixture requests for downstream codegen tests.

  These fixtures intentionally do not claim model quality or W==D validity. They
  are for quickly proving that a generated graph shape emits or removes a target
  codegen pattern.
  """
  return {
    "schema": "boltbeam.fixture_manifest.v1",
    "model_id": profile.model_id,
    "fixtures": [
      {
        "id": f"{role.role}_{role.rows}x{role.cols}_{role.quant}",
        "role": role.role,
        "shape": [role.rows, role.cols],
        "quant": role.quant,
        "purpose": "codegen_shape_probe",
      }
      for role in profile.roles
    ],
    "warning": "Synthetic fixtures are proving grounds; real model gates are still required.",
  }

