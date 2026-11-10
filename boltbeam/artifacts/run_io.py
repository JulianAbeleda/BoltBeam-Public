"""The run folder's files: its directory, JSON artifacts read and written deterministically, and the run manifest
(run_manifest.json, SCHEMA_RUN_MANIFEST) that names the stages a run has passed and the artifacts they wrote.

Imports core.canonical and vocab only: every package that reads or writes a run folder (workflow, collectors, cli,
search, control, plan) imports this, so it sits below them all. boltbeam/manifest.py is the candidate manifest, a
different file.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.vocab import SCHEMA_RUN_MANIFEST
from boltbeam.core.canonical import pretty_json

RUN_MANIFEST = "run_manifest.json"


def run_dir(path:str | pathlib.Path) -> pathlib.Path:
  p = pathlib.Path(path).expanduser()
  p.mkdir(parents=True, exist_ok=True)
  return p


def read_json(path:str | pathlib.Path) -> Any:
  """Read a JSON artifact."""
  p = pathlib.Path(path)
  return json.loads(p.read_text(encoding="utf-8"))


def write_json(path:str | pathlib.Path, obj:Any) -> None:
  """Write one JSON artifact, deterministically (sorted keys)."""
  p = pathlib.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(pretty_json(obj), encoding="utf-8")


def manifest_path(run:pathlib.Path) -> pathlib.Path:
  return run / RUN_MANIFEST


def load_manifest(run:pathlib.Path) -> dict[str, Any]:
  p = manifest_path(run)
  if not p.exists():
    raise FileNotFoundError(f"{p} does not exist; run `boltbeam load` first")
  data = read_json(p)
  if data.get("schema") != SCHEMA_RUN_MANIFEST:
    raise ValueError(f"{p} is not a {SCHEMA_RUN_MANIFEST} manifest")
  return data


def write_manifest(run:pathlib.Path, manifest:dict[str, Any]) -> dict[str, Any]:
  manifest["schema"] = SCHEMA_RUN_MANIFEST
  write_json(manifest_path(run), manifest)
  return manifest


def update_manifest(run:pathlib.Path, *, stage:str, artifacts:list[str]) -> dict[str, Any]:
  manifest = load_manifest(run)
  stages = dict(manifest.get("stages", {}))
  stages[stage] = {"artifacts": artifacts}
  manifest["stages"] = stages
  manifest["latest_stage"] = stage
  known = list(manifest.get("artifacts", []))
  for artifact in artifacts:
    if artifact not in known:
      known.append(artifact)
  manifest["artifacts"] = known
  return write_manifest(run, manifest)
