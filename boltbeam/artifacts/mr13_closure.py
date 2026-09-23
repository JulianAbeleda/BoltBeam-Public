"""Deterministic, payload-free MR13 closure manifests.

This module records only named artifact identities and exact-byte hashes.  It
does not interpret traces, models, caches, or benchmark outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass
import json, pathlib

from boltbeam.core.canonical import canonical_json as _canonical, sha256_hex as _sha256_hex

SCHEMA = "boltbeam.mr13_closure_manifest.v1"
_FORBIDDEN = ("trace", "model", "cache", "gguf", "safetensors")
_BASE_REQUIRED = ("mr9_result", "replay_design", "transfer_matrix", "test_summary", "mr12_static_audit", "policy_verdict")
_WINNER_EXTRAS = ("mr10_result", "selected_plan", "route_census")

@dataclass(frozen=True)
class PacketRequirement:
  name: str
  path: pathlib.Path
  schema: str
  statuses: tuple[str, ...]

def _blocked(name: str) -> bool: return any(word in name.lower() for word in _FORBIDDEN)
def _safe_readme_link(link: object) -> bool:
  if not isinstance(link, str) or not link or link.startswith(("/", "\\")) or "://" in link: return False
  return all(part not in ("", ".", "..") for part in pathlib.PurePosixPath(link).parts)

def _read(requirement: PacketRequirement) -> dict:
  if not requirement.name or _blocked(requirement.name) or _blocked(requirement.path.name):
    raise ValueError(f"forbidden full payload artifact: {requirement.name}")
  try: raw = requirement.path.read_bytes()
  except OSError as exc: raise ValueError(f"missing artifact: {requirement.name}") from exc
  try: packet = json.loads(raw)
  except json.JSONDecodeError as exc: raise ValueError(f"artifact is not JSON: {requirement.name}") from exc
  mr9_disposition = None
  if requirement.name == "mr9_result":
    from boltbeam.search.semantic.mr9_semantic_search import SCHEMA as MR9_SCHEMA, closure_disposition
    if requirement.schema != MR9_SCHEMA or requirement.statuses != ("COMPLETE",):
      raise ValueError("MR9 closure requirement must use the canonical schema and COMPLETE status")
    mr9_disposition = closure_disposition(packet)
  if requirement.name == "mr12_static_audit":
    from boltbeam.artifacts.mr12_static_audit import SCHEMA as MR12_SCHEMA, validate as validate_mr12
    if requirement.schema != MR12_SCHEMA or requirement.statuses != ("pass",):
      raise ValueError("MR12 closure requirement must use the canonical schema and PASS status")
    validate_mr12(packet, require_pass=True)
  if packet.get("schema") != requirement.schema: raise ValueError(f"schema mismatch: {requirement.name}")
  if packet.get("status") not in requirement.statuses: raise ValueError(f"status mismatch: {requirement.name}")
  row = {"name":requirement.name, "schema":requirement.schema, "status":packet["status"],
         "sha256":_sha256_hex(raw), "bytes":len(raw)}
  if mr9_disposition is not None: row["closure_disposition"] = mr9_disposition
  return row

def build(requirements: list[PacketRequirement], *, mr9_winner: bool, omissions: dict[str, str] | None = None) -> dict:
  if len({r.name for r in requirements}) != len(requirements): raise ValueError("duplicate artifact name")
  rows = {_row["name"]:_row for _row in (_read(r) for r in requirements)}
  if missing := set(_BASE_REQUIRED) - set(rows): raise ValueError(f"missing required artifact: {sorted(missing)}")
  if rows["mr9_result"]["closure_disposition"] != ("winner" if mr9_winner else "refuted"):
    raise ValueError("MR9 decisions do not match closure disposition")
  extras = set(_WINNER_EXTRAS) & set(rows)
  if mr9_winner:
    if extras != set(_WINNER_EXTRAS): raise ValueError("MR9 winner requires MR10, selected plan, and route census")
    if omissions: raise ValueError("winner closure must not omit conditional packets")
    disposition = "winner"
  else:
    if extras: raise ValueError("refutation closure must omit winner-only packets")
    omissions = {} if omissions is None else omissions
    if set(omissions) != set(_WINNER_EXTRAS) or not all(isinstance(v, str) and v.strip() for v in omissions.values()):
      raise ValueError("refutation closure requires explicit omission reasons")
    disposition = "refuted"
  return {"schema":SCHEMA, "status":"complete", "mr9_disposition":disposition,
          "artifacts":[rows[name] for name in sorted(rows)], "omissions":{k:omissions[k] for k in sorted(omissions or {})},
          "readme_links":{"closure_protocol":"docs/mr13-closure-manifest.md"}}

def validate(manifest: dict, requirements: list[PacketRequirement]) -> None:
  if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete": raise ValueError("invalid closure manifest")
  if not all(_safe_readme_link(link) for link in manifest.get("readme_links", {}).values()): raise ValueError("unsafe README link")
  rebuilt = build(requirements, mr9_winner=manifest.get("mr9_disposition") == "winner", omissions=manifest.get("omissions"))
  if _canonical(rebuilt) != _canonical(manifest): raise ValueError("stale, tampered, or non-deterministic closure manifest")

def write_fresh(manifest: dict, destination: pathlib.Path) -> None:
  if destination.exists(): raise FileExistsError(f"closure output already exists; use a fresh path: {destination}")
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
