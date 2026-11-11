"""The numbers one role is judged by: a plan in words and its id, what the provider's search result says (its
winner, the default kernel, the model's own kernel, the counts: claims), the roles' own time inside a real decode
(time_roles, role_losses), and BoltBeam's reproduced, like-for-like kernel times (kernel_numbers).
"""
from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from boltbeam.artifacts.run_io import optional
from boltbeam.role_key import role_key
from boltbeam.search import provider_check, role_space, tinygrad_fork
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


def _median(r:Mapping[str, Any] | None) -> float | None:
  return ((r or {}).get("measurement") or {}).get("median_ns")


def kernel_record(row:Mapping[str, Any] | None) -> dict[str, Any] | None:
  """The kernel the provider says it compiled for a population row: source, function, launch, buffers."""
  compiled = (((row or {}).get("worker") or {}).get("provider") or {}).get("compile") or {}
  record = (compiled.get("result") or {}).get("kernel") if isinstance(compiled, Mapping) else None
  return dict(record) if isinstance(record, Mapping) else None


def entry(row:Mapping[str, Any]) -> dict[str, Any]:
  sched = row["candidate"]["schedule"]
  generated = (row.get("worker") or {}).get("generated") or {}
  return {"candidate_hash": row["candidate_hash"], "plan_kind": sched["plan_kind"], "schedule": dict(sched),
          "transforms": list(sched.get("transforms", [])), "plan": plan_text(sched), "plan_hash": generated.get("plan_hash"),
          "median_ns": _median(row), "is_default": sched["plan_kind"] == DEFAULT_KIND}


def summarize_search(result:Mapping[str, Any], installed:Mapping[str, Any] | None = None) -> dict[str, Any]:
  """The provider's winner, the default kernel, the model's own kernel (the row the decode installs) and the counts,
  from one full-kernel search result. Ranks and times here are the provider's claims."""
  measured = [r for r in result.get("population", []) if r.get("state") == "MEASURED"]
  measured.sort(key=lambda r: (r.get("rank") or 1 << 30))
  default = next((r for r in measured if r["candidate"]["schedule"]["plan_kind"] == DEFAULT_KIND), None)
  model = next((r for r in measured if role_space.is_row(r["candidate"], installed)), None)
  out = {"status": result.get("status"), "counts": dict(result.get("counts", {})), "winner": None,
         "default_median_ns": _median(default), "model_kernel": entry(model) if model else None}
  if measured:
    out["winner"] = entry(measured[0])
    out["winner"]["is_model_kernel"] = model is not None and measured[0]["candidate_hash"] == model["candidate_hash"]
  return out


def time_roles(run:pathlib.Path, *, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Time every role inside a real decode in tinygrad's runtime (collectors/tinygrad_role_time.py)."""
  from boltbeam.collectors import tinygrad_role_time
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
  trace = tinygrad_role_time.collect(run, root=root, python=tinygrad_fork.fork_python(root), model=model,
                                     model_id=str(manifest.get("model_id")), target=get_target(manifest.get("target_id")), context=selected_context(run),
                                     tokens=decode_tokens(run, selected_context(run)))
  from boltbeam.roofline import limit
  ceil = limit.measured_vs_ceiling(manifest, optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"), for_decode(run, trace))
  if table and table["status"] != "measured":  # say it where the job ends, not only on the next screen
    raise RuntimeError(table["reason"])
  return trace


def role_losses(run:pathlib.Path) -> dict[tuple[Any, ...], dict[str, Any]]:
  """The per-role loss rows of a run, by role_key (role, quant, rows, k); empty before the roles were timed."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.artifacts.run_io import read_json
  from boltbeam.roofline import limit
  from boltbeam.artifacts.run_io import load_manifest
  path = run / tinygrad_role_time.TRACE
  if not path.is_file(): return {}
  ceil = limit.measured_vs_ceiling(load_manifest(run), optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], read_json(path), ceil.get("floor_ms"), for_decode(run, read_json(path))) or {"roles": []}
  return {role_key(r): r for r in table["roles"]}  # an incomplete table has no roles


def kernel_numbers(summary:Mapping[str, Any], in_model:Mapping[str, Any] | None = None,
                   verify:Mapping[str, Any] | None = None, tensors:int | None = None) -> dict[str, Any] | None:
  """BoltBeam's reproduced, like-for-like kernel times; in-model time is context only."""
  w = summary.get("winner")
  if not w or w.get("median_ns") is None: return None
  mine = ((verify or {}).get("winner") or {}).get("boltbeam") or {}
  model = ((verify or {}).get("model_kernel") or {}).get("boltbeam") or {}
  plan_us = mine.get("us_less_floor") if ((verify or {}).get("winner") or {}).get("verdict") == provider_check.REPRODUCED else None
  model_us = model.get("us_less_floor") if ((verify or {}).get("model_kernel") or {}).get("verdict") == provider_check.REPRODUCED else None
  band = (verify or {}).get("band")
  out = {"plan_us": plan_us,
         "plan_us_source": "BoltBeam's kernel timer, less the dispatch floor" if plan_us is not None else None,
         "provider_plan_us": w["median_ns"] / 1000.0,
         "reference_us": (summary.get("default_median_ns") or 0) / 1000.0 if summary.get("default_median_ns") is not None else None,
         "reference": "search fixture heuristic (the provider's time)",
         "model_kernel_us": model_us, "band": band,
         "model_us_per_call": None, "faster_than_model":
           plan_us < model_us * (1.0 - band) if plan_us is not None and model_us is not None and band is not None else None}
  if in_model and in_model.get("calls_per_token"):
    # the baseline: the role's own kernels inside the running model, GPU time per tensor at the same shape
    reads = tensors or in_model["calls_per_token"]
    per_call = in_model["actual_ms"] * 1000.0 / reads
    out.update(model_us_per_call=per_call, tensors_per_token=reads,
               model_calls_per_token=in_model["calls_per_token"])
    if tensors and abs(tensors - in_model["calls_per_token"]) > 1e-9:
      out["per_tensor_note"] = (f"the model reads {tensors} tensors of this role per token in {in_model['calls_per_token']:.0f} "
                                "calls; the plan reads one a call, so the model's time is divided per tensor")
  return out


def plan_id(role:str, quant:str, winner:Mapping[str, Any]) -> str:
  return f"tinygrad-plan:{role}:{quant}:{(winner.get('plan_hash') or winner['candidate_hash'])[:12]}"
