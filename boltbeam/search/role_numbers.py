"""The numbers one role is judged by: a plan in words and the roles' own time inside a real decode (time_roles,
role_losses). The provider's search-result numbers left with the per-role search (retired 2026-11-21).
"""
from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from boltbeam.artifacts.run_io import optional
from boltbeam.role_key import role_key
from boltbeam.search import role_space, tinygrad_fork
from boltbeam.collectors.capture_scope import for_decode

DEFAULT_KIND = role_space.HEURISTIC


def plan_text(schedule:Mapping[str, Any]) -> str:
  """One plan in words: the default kernel, an Opt sequence, or an emitter with its schedule values."""
  kind = str(schedule.get("plan_kind") or "")
  if kind == DEFAULT_KIND: return "default kernel"
  if kind == role_space.EMITTER_PLAN_KIND:
    mem = schedule.get("memory") or {}
    return (f"{(schedule.get('compute') or {}).get('family')}: {(schedule.get('launch') or {}).get('threads')} threads, "
            f"{(schedule.get('tile') or {}).get('n')} rows a block, {((mem.get('b') or {}).get('vector_width'))}-byte weight loads, "
            f"activation in {(mem.get('a') or {}).get('space')} memory, {(schedule.get('pipeline') or {}).get('stage_count')} accumulators")
  return _plan_text(list(schedule.get("transforms") or []), kind)


def _plan_text(transforms:list[Mapping[str, Any]], kind:str) -> str:
  if kind == DEFAULT_KIND: return "default kernel"
  if not transforms: return "no opts"
  return " ".join(f"{t['op']} {t['arg']}" + (f" on axis {t['axis']}" if t.get("axis") else "") for t in transforms)


def time_roles(run:pathlib.Path, *, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Time every role inside a real decode in tinygrad's runtime (collectors/arkey_role_time.py)."""
  from boltbeam.collectors import arkey_role_time
  from boltbeam.target.targets import get_target
  from boltbeam.artifacts.run_io import load_manifest
  root = pathlib.Path(root) if root else tinygrad_fork.default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = tinygrad_fork.readiness(root, model)
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  from boltbeam.collectors.capture_settings import selected_context, require_supported, decode_tokens
  require_supported(run)
  trace = arkey_role_time.collect(run, root=root, python=tinygrad_fork.fork_python(root), model=model,
                                     model_id=str(manifest.get("model_id")), target=get_target(manifest.get("target_id")), context=selected_context(run),
                                     tokens=decode_tokens(run, selected_context(run)))
  from boltbeam.roofline import limit
  ceil = limit.measured_vs_ceiling(manifest, optional(run, "model_profile.json"))
  table = arkey_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"), for_decode(run, trace))
  if table and table["status"] != "measured":  # say it where the job ends, not only on the next screen
    raise RuntimeError(table["reason"])
  return trace


def role_losses(run:pathlib.Path) -> dict[tuple[Any, ...], dict[str, Any]]:
  """The per-role loss rows of a run, by role_key (role, quant, rows, k); empty before the roles were timed."""
  from boltbeam.collectors import arkey_role_time
  from boltbeam.artifacts.run_io import read_json
  from boltbeam.roofline import limit
  from boltbeam.artifacts.run_io import load_manifest
  path = run / arkey_role_time.TRACE
  if not path.is_file(): return {}
  ceil = limit.measured_vs_ceiling(load_manifest(run), optional(run, "model_profile.json"))
  table = arkey_role_time.loss(ceil.get("_roles") or [], read_json(path), ceil.get("floor_ms"), for_decode(run, read_json(path))) or {"roles": []}
  return {role_key(r): r for r in table["roles"]}  # an incomplete table has no roles
