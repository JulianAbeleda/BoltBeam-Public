from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from boltbeam.profile.loaders import profile_from_model
from boltbeam.target.targets import get_target
from boltbeam.vocab import SCHEMA_EVIDENCE_BUNDLE
from boltbeam.artifacts.base import NormalizedEvidence, read_evidence
from boltbeam.core.canonical import pretty_json


def _write(obj:dict, out:str | None) -> None:
  text = pretty_json(obj)
  if out:
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(text)
  else:
    sys.stdout.write(text)


def _profile(args) -> tuple:
  return profile_from_model(args.model, args.id), get_target(args.target)


def _parse_ctxs(raw:str) -> tuple[int, ...]:
  try:
    out = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
  except ValueError as exc:
    raise SystemExit(f"invalid --ctxs {raw!r}; expected comma-separated integers") from exc
  if not out: raise SystemExit("--ctxs must contain at least one context")
  return out


def _parse_route_flags(raw:str) -> dict[str, str]:
  flags: dict[str, str] = {}
  if not raw: return flags
  for part in raw.split(","):
    part = part.strip()
    if not part: continue
    if "=" not in part: raise SystemExit(f"invalid --route-flags entry {part!r}; expected KEY=VALUE")
    k, v = part.split("=", 1)
    flags[k.strip()] = v.strip()
  return flags


def _run_manifest_defaults(args, default_out:str) -> tuple[dict[str, Any], str | None]:
  run_manifest: dict[str, Any] = {}
  out = args.out
  if args.run:
    from boltbeam.workflow.common import load_manifest, run_dir
    run_path = run_dir(args.run)
    run_manifest = load_manifest(run_path)
    out = out or str(run_path / default_out)
    if getattr(args, "weight_inventory", None) is None and (run_path / "weight_inventory.json").exists():
      args.weight_inventory = str(run_path / "weight_inventory.json")
  return run_manifest, out


def _fail(msg:str) -> int:
  sys.stderr.write(msg.rstrip() + "\n")
  return 2


def _load_json(path:str) -> Any:
  return json.loads(pathlib.Path(path).read_text())


def _load_evidences(path:str) -> list[NormalizedEvidence]:
  """Read an evidence file that is either a single NormalizedEvidence or a directory-ingest bundle."""
  d = _load_json(path)
  if isinstance(d, dict) and d.get("schema") == SCHEMA_EVIDENCE_BUNDLE:
    return [NormalizedEvidence.from_json(e) for e in d.get("evidence", [])]
  return [read_evidence(path)]


def _ensure_ledger_file(path:str) -> None:
  pp = pathlib.Path(path)
  pp.parent.mkdir(parents=True, exist_ok=True)
  if not pp.exists():
    pp.write_text("")


def _add_common(p:argparse.ArgumentParser) -> None:
  p.add_argument("model", help="GGUF model path")
  p.add_argument("--target", default="amd_gfx1100")
  p.add_argument("--id", default=None, help="profile/model id override")
  p.add_argument("--out", default=None, help="write JSON to this path instead of stdout")
