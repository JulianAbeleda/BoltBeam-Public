"""The inputs BubbleBeam proposes from for one role: the schedule seed, axis choices and coupled rows.

Nothing here judges legality or builds candidates. BubbleBeam (search/bubblebeam.py, through
futuresight_adapter.propose_request) filters these against the chip's facts, the population is exported from its
proposal (semantic_population_export), FutureSight (futuresight_adapter.assess_population) rejects and orders it,
and the semantic campaign measures the survivors. This module only says which dimensions exist for the role, from
what the provider's describe reports the model's decode binds through on the device:

  the decode emitters      tinygrad_decode_emitter.v1: one coupled row per emitter parameter combination the
                           provider lists for the role's weight format (decode_emitters)
  tinygrad's scheduler     tinygrad_opt_sequence.v1: coupled rows of UPCAST n (one thread) and LOCAL n (n threads),
                           n over the powers of two up to the chip's thread limit, which BubbleBeam then checks

Nothing here names a backend.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boltbeam.search.spec import EMITTER_PLAN_KIND

HEURISTIC, OPT_SEQUENCE = "tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1"
UPCASTS = (2, 3, 4)  # the scheduler's unroll factors tried on the output axis
SMALLEST_LOCAL = 8   # the smallest workgroup the scheduler rows try
SEED_SCHEDULE = {"plan_kind": OPT_SEQUENCE, "transforms": [], "tile": {"m": 1, "n": 1, "k": 32}, "launch": {"threads": 1},
                 "mapping": {"lane_policy": "subgroup_contiguous"},
                 "memory": {"a": {"space": "global", "vector_width": 1, "alignment": 2},
                            "b": {"space": "global", "vector_width": 4, "alignment": 16},
                            "c": {"space": "global", "vector_width": 1, "alignment": 2}},
                 "pipeline": {"stage_count": 1}, "compute": {"family": "generic_matvec"}, "numerical_mode": "fp16_acc_fp32"}


def _norm(quant:str) -> str:
  return quant.upper().replace("_", "")


def binds_through(describe:Mapping[str, Any]) -> str:
  """The plan kind the model's decode binds a role through on the described device."""
  binding = describe.get("decode_binding") if isinstance(describe.get("decode_binding"), Mapping) else {}
  return str(binding.get("binds_through") or OPT_SEQUENCE)


def emitter_family(describe:Mapping[str, Any], quant:str) -> dict[str, Any] | None:
  """The provider's decode emitter for this weight format, or None."""
  for family in describe.get("decode_emitters") or []:
    if isinstance(family, Mapping) and _norm(str(family.get("quant", ""))) == _norm(quant):
      return dict(family)
  return None


def proposal_inputs(describe:Mapping[str, Any], facts:Mapping[str, Any], quant:str) -> dict[str, Any]:
  """schedule, axis_choices, coupled_rows for BubbleBeam, plus the emitter family and the row the model's decode
  installs (to find the model's own kernel among the measured), or why the role has no space."""
  kind = binds_through(describe)
  if kind == EMITTER_PLAN_KIND:
    family = emitter_family(describe, quant)
    if family is None:
      return {"why": f"the decode binds this device's roles through the fork's emitters and the fork has no {quant} emitter",
              "binds_through": kind}
    base = {"schedule.plan_kind": EMITTER_PLAN_KIND, "schedule.compute.family": family["family"]}
    rows = [{**base, **dict(row)} for row in family.get("coupled_rows") or []]
    installed = {**base, **dict(family["installed_row"])} if isinstance(family.get("installed_row"), Mapping) else None
    return {"schedule": dict(SEED_SCHEDULE), "axis_choices": {}, "coupled_rows": rows, "family": family["family"],
            "installed_row": installed, "binds_through": kind, "why": None, "generator": f"provider emitter rows ({family['family']})"}
  limit = facts.get("max_threads_per_threadgroup") or 1 << 30
  powers, n = [], SMALLEST_LOCAL
  while n <= min(limit, 1 << 16):
    powers.append(n)
    n *= 2
  rows = [{"schedule.transforms": [{"op": "UPCAST", "axis": 0, "arg": u}], "schedule.launch.threads": 1} for u in UPCASTS]
  rows += [{"schedule.transforms": [{"op": "LOCAL", "axis": 0, "arg": t}], "schedule.launch.threads": t} for t in powers]
  return {"schedule": dict(SEED_SCHEDULE), "axis_choices": {}, "coupled_rows": rows, "family": None, "installed_row": None,
          "binds_through": kind, "why": None, "generator": "scheduler Opt rows to the chip's thread limit"}


def is_row(candidate:Mapping[str, Any], row:Mapping[str, Any] | None) -> bool:
  """Does this candidate carry every value of the row (the installed row, say)?"""
  if not row:
    return False
  for path, value in row.items():
    cur: Any = candidate
    for part in path.split("."):
      cur = cur.get(part) if isinstance(cur, Mapping) else None
    if cur != value:
      return False
  return True


__all__ = ["HEURISTIC", "OPT_SEQUENCE", "SEED_SCHEDULE", "binds_through", "emitter_family", "is_row", "proposal_inputs"]
