"""Pure validators for the tinygrad-to-BoltBeam transfer artifact contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


EXPECTED_TINYGRAD_ARTIFACTS: tuple[str, ...] = (
  "candidate_manifest.json",
  "amd_isa_proof_manifest.json",
  "kernel_resource_trace.json",
  "numeric_result.json",
  "timing_result.json",
)

AMD_ISA_PROOF_MANIFEST_SCHEMA = "tinygrad.amd_isa_proof_manifest.v1"
AMD_ISA_PROOF_MANIFEST_ROW_SCHEMA = "amd-isa-renderer-proof-manifest-row.v1"

_SHA256_LEN = 64


@dataclass(frozen=True)
class TransferValidation:
  missing: tuple[str, ...] = ()
  invalid: tuple[str, ...] = ()

  @property
  def ok(self) -> bool:
    return not self.missing and not self.invalid

  def to_dict(self) -> dict[str, list[str]]:
    return {
      "missing": list(self.missing),
      "invalid": list(self.invalid),
    }


def validate_expected_artifact_names(names: Iterable[str]) -> TransferValidation:
  """Validate that a tinygrad artifact bundle contains required JSON files."""
  present = set(names)
  missing = tuple(name for name in EXPECTED_TINYGRAD_ARTIFACTS if name not in present)
  return TransferValidation(missing=missing)


def validate_transfer_bundle(bundle: Mapping[str, Any]) -> TransferValidation:
  """Validate loaded tinygrad transfer artifacts without touching tinygrad."""
  presence = validate_expected_artifact_names(bundle.keys())
  missing = list(presence.missing)
  invalid = list(presence.invalid)

  if "amd_isa_proof_manifest.json" in bundle:
    proof = validate_amd_isa_proof_manifest(bundle["amd_isa_proof_manifest.json"])
    missing.extend(f"amd_isa_proof_manifest.json:{item}" for item in proof.missing)
    invalid.extend(f"amd_isa_proof_manifest.json:{item}" for item in proof.invalid)

  for name in EXPECTED_TINYGRAD_ARTIFACTS:
    if name in bundle and not isinstance(bundle[name], Mapping):
      invalid.append(f"{name}: expected object")

  return TransferValidation(missing=tuple(missing), invalid=tuple(invalid))


def validate_amd_isa_proof_manifest(manifest: Any) -> TransferValidation:
  """Validate the D3 AMD ISA proof manifest shape.

  Missing optional D3 fields such as row ``epoch_id`` are intentionally accepted.
  This validator reports normal contract failures as structured data.
  """
  if not isinstance(manifest, Mapping):
    return TransferValidation(invalid=("manifest: expected object",))

  missing: list[str] = []
  invalid: list[str] = []

  _require_string(manifest, "schema", missing, invalid)
  _require_string(manifest, "candidate_id", missing, invalid)
  _require_string(manifest, "kernel_name", missing, invalid)
  _require_string(manifest, "source_sha256", missing, invalid)
  _require_string(manifest, "binary_sha256", missing, invalid)

  if manifest.get("schema") != AMD_ISA_PROOF_MANIFEST_SCHEMA and "schema" in manifest:
    invalid.append(f"schema: expected {AMD_ISA_PROOF_MANIFEST_SCHEMA}")
  for field in ("source_sha256", "binary_sha256"):
    if field in manifest and isinstance(manifest[field], str) and not _looks_like_sha256(manifest[field]):
      invalid.append(f"{field}: expected sha256")

  if "rows" not in manifest:
    missing.append("rows")
  elif not isinstance(manifest["rows"], list):
    invalid.append("rows: expected list")
  else:
    for index, row in enumerate(manifest["rows"]):
      _validate_amd_isa_row(row, index, missing, invalid)

  return TransferValidation(missing=tuple(missing), invalid=tuple(invalid))


def _validate_amd_isa_row(row: Any, index: int, missing: list[str], invalid: list[str]) -> None:
  prefix = f"rows[{index}]"
  if not isinstance(row, Mapping):
    invalid.append(f"{prefix}: expected object")
    return

  for field in ("schema", "kind", "logical_op", "emitted"):
    _require_string(row, f"{prefix}.{field}", missing, invalid, source_key=field)

  if row.get("schema") != AMD_ISA_PROOF_MANIFEST_ROW_SCHEMA and "schema" in row:
    invalid.append(f"{prefix}.schema: expected {AMD_ISA_PROOF_MANIFEST_ROW_SCHEMA}")


def _require_string(
  value: Mapping[str, Any],
  path: str,
  missing: list[str],
  invalid: list[str],
  *,
  source_key: str | None = None,
) -> None:
  key = source_key or path
  if key not in value:
    missing.append(path)
  elif not isinstance(value[key], str) or not value[key]:
    invalid.append(f"{path}: expected non-empty string")


def _looks_like_sha256(value: str) -> bool:
  return len(value) == _SHA256_LEN and all(char in "0123456789abcdefABCDEF" for char in value)


__all__ = [
  "AMD_ISA_PROOF_MANIFEST_ROW_SCHEMA",
  "AMD_ISA_PROOF_MANIFEST_SCHEMA",
  "EXPECTED_TINYGRAD_ARTIFACTS",
  "TransferValidation",
  "validate_amd_isa_proof_manifest",
  "validate_expected_artifact_names",
  "validate_transfer_bundle",
]
