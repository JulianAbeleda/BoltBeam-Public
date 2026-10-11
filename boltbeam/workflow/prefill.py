"""A prefill run's results: the untraced prefill at each of the run's prompt lengths.

One prefill run measures one engine at its prompt lengths (`--workload prefill --ctxs ...`). Its results read:

    the truth     timing_trace.json: the untraced prefill per length, the engine's chunk, the clock and any throttle

Nothing here measures. Every number is read from the run's files.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.workflow import tie_out as tie

PER = "per prefill"


def _read(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


def truths(run:pathlib.Path) -> dict[int, dict[str, Any]]:
  trace = _read(run / "timing_trace.json")
  return {int(r["context"]): r for r in trace.get("rows") or [] if r.get("scope") == "whole_step" and r.get("workload") == "prefill"}


def loss_block(run:pathlib.Path, manifest:dict[str, Any], base:dict[str, Any]) -> dict[str, Any]:
  """The results' loss block for a prefill run, on the decode block's keys (base) so every renderer reads it: the
  untraced prefill against the sum of its kernels' ceilings, and the where table at the run's first length."""
  truth = truths(run)
  out = {**base, "workload": "prefill", "per": PER, "runtimes": [], "roles": [], "tie_out": None, "where_token_goes": None,
         "prefill": {"lengths": {}}}
  if not truth:
    out.update(status="absent", reason="no prefill measured in this run")
    return out
  first = next(iter(truth))
  where: dict[int, dict[str, Any]] = {}  # the where tables come with the capture (step 5)
  trace = _read(run / "timing_trace.json")
  for length, row in truth.items():
    w = where.get(length)
    out["prefill"]["lengths"][str(length)] = {"ms": row["prefill_ms"], "tok_s": row["tok_s"], "chunk": row["chunk"],
                                              "batched": row["batched"], "telemetry": row.get("telemetry"),
                                              "throttled": row.get("throttled"), "engine": row.get("engine"),
                                              "ceiling_ms": w["limit_ms"] if w else None, "mfu_pct": w["mfu_pct"] if w else None}
  out["prefill"]["clock_policy"] = trace.get("clock_policy")
  head = truth[first]
  w = where.get(first)
  limit = w["limit_ms"] if w else None
  out.update(status="modeled" if w else "measured", limit_ms=limit, limit_tok_s=first / limit * 1e3 if limit else None,
             where_token_goes=w, step=None,
             runtimes=[{"provider": out.get("provider"), "tok_s": head["tok_s"], "ms": head["prefill_ms"],
                        "lost_ms": head["prefill_ms"] - limit if limit else None, "per_role": False,
                        "note": f"whole prefill of {first} tokens, measured untraced; {head['chunk']['words']}"
                                + (f"; {tie_words(head)}" if head.get("throttled") else "")}])
  if not w:
    out["missing"] = "no prefill captured in the model: step 5 (role time) has not run for this run"
  return out


def tie_words(row:dict[str, Any]) -> str:
  return (row.get("telemetry") or {}).get("words") or ""


