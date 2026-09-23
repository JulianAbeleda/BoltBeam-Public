from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.profile.ir import ModelProfile, TensorRole
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


def profile_from_json(data:dict[str, Any]) -> ModelProfile:
  roles = tuple(TensorRole(role=r["role"], tensor_name=r["tensor_name"], rows=int(r["rows"]),
                           cols=int(r["cols"]), quant=r["quant"], ggml_type=int(r.get("ggml_type", -1)),
                           count=int(r.get("count", 1)), role_class=r.get("role_class", ""),
                           n_expert=int(r.get("n_expert", 0)))
                for r in data.get("roles", ()))
  return ModelProfile(model_id=data["model_id"], source=data["source"], architecture=data.get("architecture"),
                      hidden_size=data.get("hidden_size"), ffn_size=data.get("ffn_size"),
                      vocab_size=data.get("vocab_size"), layer_count=data.get("layer_count"),
                      roles=roles, metadata=dict(data.get("metadata", {})),
                      architecture_class=data.get("architecture_class", "unknown_transformer"))


def load_profile(run:pathlib.Path) -> ModelProfile:
  return profile_from_json(read_json(run / "model_profile.json"))
