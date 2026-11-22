"""The kernel search as a stage of Run, and what the screens show per role: the stage's status file, the stage itself
(the engine paths owner tooling registers, else a skip with its reason), and each
role's best_found and verdict from its route_policy.json compare record (records written before 2026-11-21 still read).

The per-role search (search/role_compare and the flash search, asking the tinygrad fork's search provider) was retired on
2026-11-21: that provider left tinygrad-arkey with the kernel generator, and the generator is reached only through its
contract (search/generator_repo).
"""
from __future__ import annotations

import json, pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from boltbeam.role_key import role_label
from boltbeam.search import compare_record

STATUS_SCHEMA = "boltbeam.kernel_search_status.v1"
ARKEY_ONLY = "the kernel search runs in arkey only"  # an older run's skip reason; Emit reads it
RETIRED = ("the per-role kernel search was retired 2026-11-21 (its provider left tinygrad-arkey with the kernel generator); "
           "the kernel generator's contract is the only search")


def search_applies(backend:str | None = None, provider:str | None = None) -> str:
  """Why Run's per-role search does not run here: retired (RETIRED), for every engine. The engine paths still run."""
  return RETIRED


def write_status(run:pathlib.Path, status:str, reason:str | None = None, seconds:float | None = None) -> None:
  compare_record.write(run / compare_record.FOLDER / compare_record.STATUS_FILE, {"schema": STATUS_SCHEMA, "status": status, "reason": reason, "seconds": seconds})


def read_status(run:pathlib.Path) -> dict[str, Any] | None:
  path = run / compare_record.FOLDER / compare_record.STATUS_FILE
  try:
    # an older run's status says "runs in tinygrad only" (arkey's former name): read as ARKEY_ONLY, which Emit matches
    return json.loads(path.read_text().replace("runs in tinygrad only", "runs in arkey only")) if path.is_file() else None
  except (OSError, ValueError):
    return None


def skip(run:pathlib.Path, reason:str, say:Callable[[str], None] = print) -> dict[str, Any]:
  write_status(run, "skipped", reason)
  say(f"search skipped: {reason}")
  return {"status": "skipped", "reason": reason}


def search_stage(run:pathlib.Path, *, backend:str | None = None, provider:str, root:pathlib.Path | None = None,
                 say:Callable[[str], None] = print, step:Callable[[int, int], None] | None = None,
                 paths:Sequence[Any] = ()) -> dict[str, Any]:
  """Run's "Search faster kernels" stage: each of paths in turn (engine paths: owner tooling that
  workflow/pipeline.OWNER_SEARCH registers only where its module is present, none on main; each a module with
  stage(run, engine=, say=) -> {status, reason}); else a skip with the reasons."""
  import time
  why = RETIRED
  for path in paths:
    began = time.monotonic()
    got = path.stage(pathlib.Path(run), engine=provider, say=say)
    if got["status"] == "searched":
      seconds = round(time.monotonic() - began, 1)
      write_status(run, "searched", None, seconds)
      return {"status": "searched", "reason": None, "seconds": seconds}
    why = f"{why}; {got['reason']}"
  write_status(run, "skipped", why)
  say(f"search skipped: {why}")
  return {"status": "skipped", "reason": why}


def _speedup(kernel:Mapping[str, Any] | None) -> float | None:
  if not kernel or not kernel.get("model_kernel_us") or not kernel.get("plan_us"):
    return None
  return float(kernel["model_kernel_us"]) / float(kernel["plan_us"])


def role_verdict(route:Mapping[str, Any] | None, status:Mapping[str, Any] | None) -> dict[str, Any]:
  """best_found and verdict for one role, from its route_policy.json compare record and the stage's status.
  Only measured numbers: the plan alone and the model's own kernel per call, the binding count, the A/B."""
  c = (route or {}).get("compare") if isinstance((route or {}).get("compare"), dict) else None
  if c is None:
    reason = (status or {}).get("reason") or ("the search found no route for this role" if status else
                                              "the kernel search did not run for this run")
    return {"best_found": None, "verdict": "not_searched", "verdict_reason": reason}
  k = c.get("kernel") if isinstance(c.get("kernel"), dict) else None
  x = _speedup(k)
  plan = c.get("plan")
  best = None
  if k is not None and plan:
    best = {"plan": plan, "plan_us": k.get("plan_us"), "model_us": k.get("model_kernel_us"), "speedup": x,
            "calls_per_token": c.get("role_calls_per_token"),
            "text": f"{plan} · {x:.1f}x faster" if x and x > 1 else f"{plan} · not faster" if x else plan}
  searched = {"candidates": c.get("candidates"), "measured_correct": c.get("measured_correct")}
  if c.get("not_reproduced"):  # BoltBeam's own check disagreed with the provider: never a finding
    return {"best_found": None, "verdict": "not_reproduced", "verdict_reason": c.get("reason"), **searched}
  if k is None:
    return {"best_found": None, "verdict": "not_searched", "verdict_reason": c.get("reason") or "the search measured no kernel", **searched}
  if (route or {}).get("status") == "promoted":
    return {"best_found": best, "verdict": "applied", "verdict_reason": c.get("reason"), **searched}
  if k.get("faster_than_model") is None or k.get("model_kernel_us") is None or k.get("plan_us") is None:
    return {"best_found": None, "verdict": "not_reproduced", "verdict_reason": "not decided: no reproduced like-for-like baseline", **searched}
  if not k.get("faster_than_model") or plan == "default kernel":
    return {"best_found": best, "verdict": "none_faster", "verdict_reason": c.get("reason"), **searched}
  return {"best_found": best, "verdict": "found_not_applied", "verdict_reason": c.get("reason"), **searched}


def found_summary(routes:list[Mapping[str, Any]]) -> dict[str, Any] | None:
  """The roles with a faster kernel found but not applied: ms per token saved if applied, the sum over roles of
  (model µs - plan µs) x calls per token, and how many of those calls the plan reached in the whole-model A/B."""
  rows, ms, reached, calls, ab_roles = [], 0.0, 0, 0.0, 0
  for r in routes:
    v = role_verdict(r, {"status": "searched"})
    b = v["best_found"]
    if v["verdict"] != "found_not_applied" or not b or b.get("calls_per_token") is None:
      continue
    ms += (b["model_us"] - b["plan_us"]) * b["calls_per_token"] / 1000.0
    rows.append(role_label(str(r.get('role')), str(r.get('quant')), r.get('shape')))
    ab = (r.get("compare") or {}).get("ab") or {}
    binding = ab.get("binding") if isinstance(ab.get("binding"), dict) else None
    if binding is not None:
      ab_roles += 1
      reached += int(binding.get("calls_reached") or 0)
      calls += float(b["calls_per_token"])
  if not rows:
    return None
  return {"roles": rows, "ms": ms, "calls_reached": reached if ab_roles else None,
          "calls": calls if ab_roles else None}
