"""Reproducible, non-hardware MR12 duplicate-authority audit.

The packet is intentionally narrow: it proves repository structure and exact
policy-asset identity.  It never claims runtime correctness or benchmark test
coverage.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
from collections.abc import Sequence

from boltbeam.core.canonical import sha256_hex
from boltbeam.policy.route_manifest import load_authoritative_asset

SCHEMA = "boltbeam.mr12_static_audit.v1"
_HEX40 = re.compile(r"[0-9a-f]{40}")


def _sha256(path: pathlib.Path) -> str:
  return sha256_hex(path.read_bytes())


def _git_identity(root: pathlib.Path) -> dict:
  def run(*args: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, check=False)
    if result.returncode: raise ValueError(f"git identity unavailable for {root}: {result.stderr.strip()}")
    return result.stdout.strip()
  status = run("status", "--porcelain")
  return {"commit":run("rev-parse", "HEAD"), "tree":run("rev-parse", "HEAD^{tree}"), "dirty":bool(status)}


def _source_files(root: pathlib.Path, paths: Sequence[str]) -> list[pathlib.Path]:
  files: list[pathlib.Path] = []
  for relative in paths:
    base = root / relative
    if base.is_file(): files.append(base)
    elif base.is_dir(): files.extend(sorted(base.rglob("*.py")))
  return files


def _literal_route_tables(files: Sequence[pathlib.Path]) -> list[str]:
  pattern = re.compile(r"(?m)^\s*ROUTES\s*=\s*\{")
  return [str(path) for path in files if pattern.search(path.read_text(errors="replace"))]


def _research_imports(files: Sequence[pathlib.Path]) -> list[str]:
  pattern = re.compile(r"(?m)^\s*(?:from|import)\s+extra\.llm_research(?:\.|\s|$)")
  return [str(path) for path in files if pattern.search(path.read_text(errors="replace"))]


def build(*, boltbeam_root: pathlib.Path, tinygrad_root: pathlib.Path) -> dict:
  boltbeam_root, tinygrad_root = boltbeam_root.resolve(), tinygrad_root.resolve()
  asset = boltbeam_root / "boltbeam/policy/assets/route_manifest.v1.json"
  snapshot = tinygrad_root / "extra/llm_research/generated/boltbeam_route_manifest.v1.json"
  legacy = tinygrad_root / "extra/llm_research/route_manifest.json"
  checks: list[dict] = []

  def check(name: str, passed: bool, evidence: object) -> None:
    checks.append({"name":name, "status":"pass" if passed else "fail", "evidence":evidence})

  # Exercise the same verified loader used by production BoltBeam policy code.
  loaded = load_authoritative_asset(asset)
  check("canonical_route_asset", loaded.get("authority") == "BoltBeam", {"sha256":_sha256(asset), "version":loaded.get("version")})
  snapshot_exists = snapshot.is_file()
  check("exp_snapshot_exact", snapshot_exists and snapshot.read_bytes() == asset.read_bytes(),
        {"snapshot_sha256":_sha256(snapshot) if snapshot_exists else None, "authority_sha256":_sha256(asset)})
  check("legacy_exp_manifest_removed", not legacy.exists(), {"path":str(legacy)})

  route_files = _source_files(boltbeam_root, ("boltbeam",)) + _source_files(tinygrad_root, ("tinygrad", "extra/llm_research"))
  literals = _literal_route_tables(route_files)
  check("no_duplicate_literal_route_tables", not literals, {"matches":literals})
  production = _source_files(tinygrad_root, ("tinygrad",))
  imports = _research_imports(production)
  check("production_has_no_research_imports", not imports, {"matches":imports})

  repositories = {"boltbeam":_git_identity(boltbeam_root), "tinygrad_exp":_git_identity(tinygrad_root)}
  check("recorded_revisions_are_clean", not any(row["dirty"] for row in repositories.values()),
        {name:{"commit":row["commit"], "tree":row["tree"], "dirty":row["dirty"]} for name,row in repositories.items()})
  failed = [row["name"] for row in checks if row["status"] != "pass"]
  return {"schema":SCHEMA, "status":"pass" if not failed else "fail", "scope":"static_nonhardware_only",
          "repositories":repositories, "checks":checks, "failed_checks":failed,
          "claims":{"runtime_correctness":False, "hardware_executed":False, "benchmark_tests_executed":False}}


def validate(packet: dict, *, require_pass: bool = False) -> None:
  """Validate the MR12 packet shape and its pinned repository identities."""
  if packet.get("schema") != SCHEMA or packet.get("scope") != "static_nonhardware_only":
    raise ValueError("invalid MR12 static-audit schema or scope")
  if packet.get("status") not in ("pass", "fail"):
    raise ValueError("invalid MR12 static-audit status")
  checks = packet.get("checks")
  if not isinstance(checks, list) or not checks or any(not isinstance(row, dict) for row in checks):
    raise ValueError("invalid MR12 static-audit checks")
  if any(not isinstance(row.get("name"), str) or row.get("status") not in ("pass", "fail") for row in checks):
    raise ValueError("invalid MR12 static-audit check row")
  failed = [row["name"] for row in checks if row["status"] != "pass"]
  if packet.get("failed_checks") != failed or packet["status"] != ("pass" if not failed else "fail"):
    raise ValueError("inconsistent MR12 static-audit disposition")
  repositories = packet.get("repositories")
  if not isinstance(repositories, dict) or set(repositories) != {"boltbeam", "tinygrad_exp"}:
    raise ValueError("MR12 static audit requires BoltBeam and tinygrad EXP identities")
  for name, identity in repositories.items():
    if not isinstance(identity, dict) or _HEX40.fullmatch(str(identity.get("commit", ""))) is None or \
       _HEX40.fullmatch(str(identity.get("tree", ""))) is None or not isinstance(identity.get("dirty"), bool):
      raise ValueError(f"invalid MR12 repository identity: {name}")
  if packet.get("claims") != {"runtime_correctness":False, "hardware_executed":False, "benchmark_tests_executed":False}:
    raise ValueError("invalid MR12 static-audit claims")
  if require_pass and (packet["status"] != "pass" or any(identity["dirty"] for identity in repositories.values())):
    raise ValueError("MR13 requires a passing MR12 audit at clean pinned revisions")


def write_fresh(packet: dict, destination: pathlib.Path) -> None:
  validate(packet)
  if destination.exists(): raise FileExistsError(f"MR12 output already exists; use a fresh path: {destination}")
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
