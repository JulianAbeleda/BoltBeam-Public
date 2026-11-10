"""The kernel search as a stage of Run, and what the screens show per role: whether the search exists for this
engine and chip (the provider's own capability and BoltBeam's own detection of the GPU, never a list of backend
names), the stage's status file, the stage itself (the per-role comparison where it exists, else a skip with its
reason), and each role's best_found and verdict from its route_policy.json compare record.
"""
from __future__ import annotations

import json, pathlib
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.role_key import role_label
from boltbeam.search import compare_record, role_compare, tinygrad_fork

STATUS_SCHEMA = "boltbeam.kernel_search_status.v1"


def detect(backend:str | None) -> str | None:
  """BoltBeam's own detection: can its bridge open a GPU of this backend here? None when it can, else why not."""
  from boltbeam.collectors import kernel_timer
  try:
    bridge = kernel_timer.bridge_for(str(backend))
  except Exception as exc:  # no driver, no device, no compiler: BoltBeam could not check the provider's claims here
    return f"BoltBeam cannot open a {backend} GPU on this machine to check the provider: {exc}"
  close = getattr(bridge, "close", None)
  if close: close()
  return None


TINYGRAD_ONLY = "the kernel search runs in tinygrad only"  # search_applies' answer for another engine; Emit reads it


def search_applies(backend:str | None, provider:str, *, root:pathlib.Path | None = None,
                   capability_fn:Callable[[pathlib.Path, list[str]], dict[str, Any]] | None = None,
                   detect_fn:Callable[[str | None], str | None] | None = None) -> str | None:
  """Why Run's kernel search does not exist for this engine and chip, or None when it does. Two answers decide it:
  the provider's own capability (the devices it can run, asked with its capability action) and BoltBeam's own
  detection of the GPU (its bridge opens one), never a list of backend names. Without the fork checked out the
  provider cannot be asked; readiness() names the fix for that."""
  if provider != "tinygrad":
    return f"{TINYGRAD_ONLY}; this run used {provider}"
  row = tinygrad_fork.runtime_row(backend)
  if row is None:
    return f"the search runtime table (data/search_runtime.json) has no row for {backend or 'an unknown backend'}"
  if why := (detect_fn or detect)(backend):
    return why
  root = pathlib.Path(root) if root else tinygrad_fork.default_fork_root()
  if capability_fn is None and not (root / tinygrad_fork.PROVIDER).is_file():
    return None
  try:
    reply = (capability_fn or tinygrad_fork.capability)(root, list(row["tinygrad_devices"]))
  except Exception as exc:
    return f"the provider could not say which devices it runs: {exc}"
  return tinygrad_fork.pick_device(backend, reply)[1]


def write_status(run:pathlib.Path, status:str, reason:str | None = None, seconds:float | None = None) -> None:
  compare_record.write(run / compare_record.FOLDER / compare_record.STATUS_FILE, {"schema": STATUS_SCHEMA, "status": status, "reason": reason, "seconds": seconds})


def read_status(run:pathlib.Path) -> dict[str, Any] | None:
  path = run / compare_record.FOLDER / compare_record.STATUS_FILE
  try:
    return json.loads(path.read_text()) if path.is_file() else None
  except (OSError, ValueError):
    return None


def skip(run:pathlib.Path, reason:str, say:Callable[[str], None] = print) -> dict[str, Any]:
  write_status(run, "skipped", reason)
  say(f"search skipped: {reason}")
  return {"status": "skipped", "reason": reason}


def search_stage(run:pathlib.Path, *, backend:str, provider:str, root:pathlib.Path | None = None,
                 say:Callable[[str], None] = print, step:Callable[[int, int], None] | None = None,
                 compare:Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
  """Run's "Search faster kernels" stage: the per-role search where it exists, else a skip with its reason. A
  search that cannot start (no fork, a dirty fork) is a skip too: the run still ends with its results."""
  import time
  if why := search_applies(backend, provider, root=root):
    write_status(run, "skipped", why)
    say(f"search skipped: {why}")
    return {"status": "skipped", "reason": why}
  root = pathlib.Path(root) if root else tinygrad_fork.default_fork_root()
  if compare is None:
    ready = tinygrad_fork.readiness(root, str((json.loads((run / "run_manifest.json").read_text())).get("model_path") or ""))
    why = (f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"]) if not ready["ready"] \
      else tinygrad_fork.clean_refusal(root)
    if why:
      write_status(run, "skipped", why)
      say(f"search skipped: {why}")
      return {"status": "skipped", "reason": why}
  began = time.monotonic()
  try:
    (compare or role_compare.compare_run)(run, root=root, say=say, ab_only_if_faster=True, step=step)
  except role_compare.ProviderRefused as exc:  # BoltBeam refused the provider's device: nothing it would say is taken
    write_status(run, "skipped", f"provider refused: {exc}")
    say(f"search skipped: provider refused: {exc}")
    return {"status": "skipped", "reason": f"provider refused: {exc}"}
  seconds = round(time.monotonic() - began, 1)
  write_status(run, "searched", None, seconds)
  return {"status": "searched", "reason": None, "seconds": seconds}


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
