"""Static, fail-closed AMD decode resource evidence adapters.

This is a consumer of already captured code objects, ELF-note text, ISA manifests,
and KFD/rocprof traces.  It deliberately does not import tinygrad, invoke ROCm
tools, create a GPU runtime, or infer resource facts from instruction text.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Mapping, Sequence


RESOURCE_TRACE_SCHEMA = "boltbeam.decode_resource_trace.v1"
ISA_MANIFEST_SCHEMA = "boltbeam.amd_isa_manifest.v1"
REQUIRED_RESOURCES = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def artifact_sha256(value: bytes | str) -> str:
  """Return the identity digest for exact artifact bytes, without a prefix."""
  if not isinstance(value, (bytes, str)):
    raise TypeError("artifact must be bytes or str")
  return sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def parse_amdgpu_elf_notes(notes: str) -> dict[str, Any]:
  """Parse the stable AMDGPU YAML fields emitted by ``llvm-readelf --notes``.

  The caller owns obtaining the note text.  Every resource counter is required:
  an omitted SGPR or spill count is unknown, not zero.
  """
  if not isinstance(notes, str) or not notes.strip():
    raise ValueError("ELF note text must be non-empty")

  def integer(field: str) -> int:
    match = re.search(rf"^\s*\.{re.escape(field)}:\s*(\d+)\s*$", notes, re.MULTILINE)
    if match is None:
      raise ValueError(f"AMDGPU metadata missing .{field}")
    return int(match.group(1))

  target = re.search(r"^\s*amdhsa\.target:\s*(\S+)\s*$", notes, re.MULTILINE)
  symbol = re.search(r"^\s*\.symbol:\s*(\S+)\s*$", notes, re.MULTILINE)
  if target is None:
    raise ValueError("AMDGPU metadata missing amdhsa.target")
  if symbol is None:
    raise ValueError("AMDGPU metadata missing .symbol")
  out = {"target": target.group(1), "symbol": symbol.group(1)}
  out.update({"vgpr": integer("vgpr_count"), "sgpr": integer("sgpr_count"),
              "vgpr_spills": integer("vgpr_spill_count"), "sgpr_spills": integer("sgpr_spill_count"),
              "lds_bytes": integer("group_segment_fixed_size"),
              "scratch_bytes": integer("private_segment_fixed_size")})
  for field, source in (("workgroup_threads", "max_flat_workgroup_size"), ("wavefront_size", "wavefront_size")):
    out[field] = integer(source)
  return out


def build_decode_resource_trace(*, candidate_id: str, source: bytes | str, code_object: bytes,
                                elf_notes: str, workgroup: Sequence[int], grid: Sequence[int]) -> dict[str, Any]:
  """Create a resource trace bound to source and final-code-object digests."""
  if not isinstance(candidate_id, str) or not candidate_id:
    raise ValueError("candidate_id must be non-empty")
  if not isinstance(code_object, bytes) or not code_object:
    raise ValueError("code_object must be non-empty bytes")
  geometry = {"workgroup": _dim3(workgroup, "workgroup"), "grid": _dim3(grid, "grid")}
  metadata = parse_amdgpu_elf_notes(elf_notes)
  resources = {field: metadata[field] for field in REQUIRED_RESOURCES}
  return {"schema": RESOURCE_TRACE_SCHEMA, "candidate_id": candidate_id,
          "source_sha256": artifact_sha256(source), "binary_sha256": artifact_sha256(code_object),
          "kernel": metadata["symbol"], "target": metadata["target"], "resources": resources,
          **geometry, "workgroup_threads": metadata["workgroup_threads"],
          "wavefront_size": metadata["wavefront_size"]}


def join_isa_resources(isa_manifest: Mapping[str, Any], resource_trace: Mapping[str, Any]) -> dict[str, Any]:
  """Join final ISA and resources only when candidate and binary identities agree."""
  if not isinstance(isa_manifest, Mapping) or not isinstance(resource_trace, Mapping):
    raise TypeError("ISA manifest and resource trace must be mappings")
  if isa_manifest.get("schema") != ISA_MANIFEST_SCHEMA:
    raise ValueError("unsupported ISA manifest schema")
  if resource_trace.get("schema") != RESOURCE_TRACE_SCHEMA:
    raise ValueError("unsupported resource trace schema")
  candidate_id = _identity(isa_manifest, "candidate_id")
  if candidate_id != _identity(resource_trace, "candidate_id"):
    raise ValueError("candidate identity mismatch")
  binary = _digest(isa_manifest, "binary_sha256")
  if binary != _digest(resource_trace, "binary_sha256"):
    raise ValueError("binary digest mismatch")
  resources = resource_trace.get("resources")
  if not isinstance(resources, Mapping):
    raise ValueError("resource trace lacks resources")
  for field in REQUIRED_RESOURCES:
    value = resources.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise ValueError(f"resource trace missing or invalid {field}")
  isa = isa_manifest.get("isa")
  if not isinstance(isa, str) or not isa:
    raise ValueError("ISA manifest lacks final ISA")
  return {"candidate_id": candidate_id, "binary_sha256": binary, "source_sha256": _digest(resource_trace, "source_sha256"),
          "isa": isa, "resources": dict(resources), "kernel": resource_trace.get("kernel"),
          "target": resource_trace.get("target")}


def adapt_direct_kfd_trace(kfd_rows: Sequence[Mapping[str, Any]], rocprof_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
  """Preserve KFD dispatch provenance when rocprof has no matching timing row.

  Matching is by explicit dispatch id only.  A missing row produces a structured
  timing blocker rather than a zero duration or a nearest-neighbour join.
  """
  if isinstance(kfd_rows, (str, bytes)) or isinstance(rocprof_rows, (str, bytes)):
    raise TypeError("trace rows must be sequences of mappings")
  timings: dict[str, Mapping[str, Any]] = {}
  for row in rocprof_rows:
    if not isinstance(row, Mapping):
      raise ValueError("rocprof row must be a mapping")
    dispatch_id = _identity(row, "dispatch_id")
    if dispatch_id in timings:
      raise ValueError("duplicate rocprof dispatch_id")
    timings[dispatch_id] = row
  out = []
  for row in kfd_rows:
    if not isinstance(row, Mapping):
      raise ValueError("KFD row must be a mapping")
    dispatch_id = _identity(row, "dispatch_id")
    item = {"dispatch_id": dispatch_id, "kernel": row.get("kernel"), "queue_id": row.get("queue_id"), "kfd": dict(row)}
    timed = timings.get(dispatch_id)
    if timed is None:
      item["timing"] = None
      item["blockers"] = [{"missing": "rocprof_timing", "reason": "no rocprof row for direct KFD dispatch"}]
    else:
      item["timing"] = dict(timed)
      item["blockers"] = []
    out.append(item)
  return tuple(out)


def _identity(row: Mapping[str, Any], field: str) -> str:
  value = row.get(field)
  if not isinstance(value, str) or not value:
    raise ValueError(f"missing {field}")
  return value


def _digest(row: Mapping[str, Any], field: str) -> str:
  value = _identity(row, field)
  if _SHA256.fullmatch(value) is None:
    raise ValueError(f"invalid {field}")
  return value


def _dim3(value: Sequence[int], name: str) -> list[int]:
  if isinstance(value, (str, bytes)) or len(value) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in value):
    raise ValueError(f"{name} must contain three positive ints")
  return list(value)


__all__ = ["ISA_MANIFEST_SCHEMA", "RESOURCE_TRACE_SCHEMA", "REQUIRED_RESOURCES", "adapt_direct_kfd_trace",
           "artifact_sha256", "build_decode_resource_trace", "join_isa_resources", "parse_amdgpu_elf_notes"]
