"""The inputs BubbleBeam proposes from for one role: the schedule seed, axis choices and coupled rows.

Nothing here judges legality or builds candidates. BubbleBeam (search/bubblebeam.py, through
futuresight_adapter.propose_request) filters these against the chip's facts, the population is exported from its
proposal (semantic_population_export), FutureSight (futuresight_adapter.assess_population) rejects and orders it,
and the semantic campaign measures the survivors. This module only says which dimensions exist for the role, from
what the provider's describe reports the model's decode binds through on the device:

  the decode emitters      tinygrad_decode_emitter.v1: one coupled row per emitter parameter combination the
                           provider lists for the role's weight format (decode_emitters); for a family in
                           data/emitter_kinds.json BoltBeam proposes the rows itself from the chip's facts and its own
                           block layouts (bubblebeam.subgroup_row_block_unit_rows), and checks the provider's declared
                           layout against its own
  tinygrad's scheduler     tinygrad_opt_sequence.v1: coupled rows of UPCAST n (one thread) and LOCAL n (n threads),
                           n over the powers of two up to the chip's thread limit, which BubbleBeam then checks

Nothing here names a backend.
"""
from __future__ import annotations

import functools
import json
import pathlib
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


_KINDS = pathlib.Path(__file__).resolve().parent.parent / "data" / "emitter_kinds.json"


@functools.lru_cache(maxsize=1)
def emitter_kinds() -> dict[str, dict[str, Any]]:
  """The emitter families BoltBeam proposes rows for itself, by family (data/emitter_kinds.json)."""
  return {k["family"]: dict(k) for k in json.loads(_KINDS.read_text(encoding="utf-8"))["kinds"]}


def _norm(quant:str) -> str:
  return quant.upper().replace("_", "")


def binds_through(describe:Mapping[str, Any]) -> str:
  """The plan kind the model's decode binds a role through on the described device."""
  binding = describe.get("decode_binding") if isinstance(describe.get("decode_binding"), Mapping) else {}
  return str(binding.get("binds_through") or OPT_SEQUENCE)


def emitter_family(describe:Mapping[str, Any], quant:str) -> dict[str, Any] | None:
  """The provider's first decode emitter for this weight format, or None."""
  found = emitter_families(describe, quant)
  return found[0] if found else None


def emitter_families(describe:Mapping[str, Any], quant:str) -> list[dict[str, Any]]:
  """Every decode emitter the provider lists for this weight format, in its order."""
  out = []
  for family in describe.get("decode_emitters") or []:
    if not isinstance(family, Mapping): continue
    quants = family.get("quants") if isinstance(family.get("quants"), list) else [family.get("quant", "")]
    if any(_norm(str(q)) == _norm(quant) for q in quants):
      out.append(dict(family))
  return out


def cta_k_sweep_space(family:Mapping[str, Any], quant:str, *, subgroup_size:int | None, max_threads:int | None) -> dict[str, Any]:
  """BoltBeam's own rows for a one-row-a-CTA, k-sweep family (bubblebeam.cta_row_k_sweep_rows), or why none."""
  from boltbeam.collectors import metal_native as native
  from boltbeam.search.bubblebeam import cta_row_k_sweep_rows
  q = _canon_quant(quant)
  lanes, block = family.get("lanes_per_block_slice"), family.get("block_elems")
  if q not in native.BLOCK_ELEMS or block != native.BLOCK_ELEMS[q] or not isinstance(lanes, int):
    return {"why": f"the provider's {q} slice layout (block {block}, {lanes} lanes) disagrees with BoltBeam's {native.BLOCK_ELEMS.get(q)}-element block"}
  if not isinstance(subgroup_size, int) or subgroup_size < 1 or subgroup_size % lanes:
    return {"why": "the target's subgroup does not hold whole block slices"}
  return {"why": None, "rows": cta_row_k_sweep_rows(block_elems=block, lanes_per_block_slice=lanes, subgroup_size=subgroup_size,
                                                    max_threads=max_threads), "installed_row": None}


def _canon_quant(quant:str) -> str:
  from boltbeam.collectors import metal_native as native
  return next((q for q in native.BLOCK_BYTES if _norm(q) == _norm(quant)), quant)


def block_unit_space(family:Mapping[str, Any], kind:Mapping[str, Any], quant:str, *, subgroup_size:int | None,
                     max_threads:int | None, shape:tuple[int, int] | None, sm_count:int | None) -> dict[str, Any]:
  """BoltBeam's own rows for a row-per-subgroup, block-code-unit family, and the row the model's decode installs
  (the provider's declared installed_rule applied to this shape and chip), or why there is no space."""
  from boltbeam.collectors import metal_native as native
  from boltbeam.search.bubblebeam import subgroup_row_block_unit_rows
  q = _canon_quant(quant)
  if q not in native.BLOCK_BYTES or q not in kind["quants"]:
    return {"why": f"BoltBeam has no block layout for {quant} in {kind['family']}"}
  unit = family.get("code_unit_bytes")
  units = (family.get("code_units_per_block") or {}).get(q)
  mine = native.code_bytes(q)
  if not isinstance(unit, int) or not isinstance(units, int) or unit * units != mine:
    return {"why": f"the provider's {q} layout ({units} units of {unit} bytes) disagrees with BoltBeam's {mine} code bytes per block"}
  if not isinstance(subgroup_size, int) or subgroup_size < 1:
    return {"why": "the target names no subgroup size, so no row-per-subgroup launch can be proposed"}
  rows = subgroup_row_block_unit_rows(code_bytes=mine, unit_bytes=unit, block_elems=native.BLOCK_ELEMS[q],
                                      subgroup_size=subgroup_size, max_threads=max_threads)
  installed = None
  rule = family.get("installed_rule")
  if isinstance(rule, Mapping) and shape is not None and isinstance(sm_count, int) and sm_count > 0:
    per_lane = int(rule["units_per_lane"])
    lanes = max(1, units // per_lane)
    while lanes > subgroup_size: lanes //= 2
    cap = int(rule["max_warps_per_cta"])
    if isinstance(max_threads, int): cap = min(cap, max_threads // subgroup_size)
    g = 1
    while g * 2 <= cap and -(-shape[0] // (g * 2)) >= int(rule["min_ctas_per_sm"]) * sm_count: g *= 2
    installed = {"schedule.launch.threads": g * subgroup_size, "schedule.tile.n": g, "schedule.tile.k": native.BLOCK_ELEMS[q],
                 "schedule.memory.b.vector_width": mine // lanes, "schedule.memory.a.space": "global", "schedule.pipeline.stage_count": 1}
  return {"why": None, "rows": rows, "installed_row": installed}


def proposal_inputs(describe:Mapping[str, Any], facts:Mapping[str, Any], quant:str, *, subgroup_size:int | None = None,
                    shape:tuple[int, int] | None = None, sm_count:int | None = None) -> dict[str, Any]:
  """sm_count is BoltBeam's own bridge reading of the GPU (provider_check.bridge_facts), never the provider's: it only
  places the model's installed row among the candidates."""
  """schedule, axis_choices, coupled_rows for BubbleBeam, plus the emitter family and the row the model's decode
  installs (to find the model's own kernel among the measured), or why the role has no space."""
  kind = binds_through(describe)
  if kind == EMITTER_PLAN_KIND:
    families = emitter_families(describe, quant)
    if not families:
      return {"why": f"the decode binds this device's roles through the fork's emitters and the fork has no {quant} emitter",
              "binds_through": kind}
    rows, installed, generators, names, whys = [], None, [], [], []
    for family in families:
      base = {"schedule.plan_kind": EMITTER_PLAN_KIND, "schedule.compute.family": family["family"]}
      own = emitter_kinds().get(family["family"])
      if own is not None:
        mt = facts.get("max_threads_per_threadgroup")
        space = (block_unit_space(family, own, quant, subgroup_size=subgroup_size, max_threads=mt, shape=shape, sm_count=sm_count)
                 if own.get("row_owner") == "subgroup" else cta_k_sweep_space(family, quant, subgroup_size=subgroup_size, max_threads=mt))
        if space["why"]:
          whys.append(f"{family['family']}: {space['why']}")
          continue
        mine = [{**base, **row} for row in space["rows"]]
        here = {**base, **space["installed_row"]} if space["installed_row"] else None
        generators.append(f"BubbleBeam rows from the chip's facts ({family['family']})")
      else:
        mine = [{**base, **dict(row)} for row in family.get("coupled_rows") or []]
        here = {**base, **dict(family["installed_row"])} if isinstance(family.get("installed_row"), Mapping) else None
        generators.append(f"provider emitter rows ({family['family']})")
      rows += mine
      names.append(family["family"])
      installed = installed or here
    if not rows: return {"why": "; ".join(whys), "binds_through": kind}
    return {"schedule": dict(SEED_SCHEDULE), "axis_choices": {}, "coupled_rows": rows, "family": "+".join(names),
            "installed_row": installed, "binds_through": kind, "why": None, "generator": "; ".join(generators),
            "refused_families": whys}
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


__all__ = ["HEURISTIC", "OPT_SEQUENCE", "SEED_SCHEDULE", "binds_through", "block_unit_space", "cta_k_sweep_space", "emitter_families",
           "emitter_family", "emitter_kinds",
           "is_row", "proposal_inputs"]
