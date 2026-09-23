"""Per-operand LDS-vs-cache staging calculator (W6).

Reads a `memory_hierarchy_profile` (as produced by `boltbeam.mem_hierarchy.classify_ws_sweep`) and
turns its tier facts into a staging recommendation for one operand's access-stream, via
`boltbeam.perf.mem_tier.decide`. Prediction, not verdict — promotion happens only via a real A/B
timing loop (W7), not here.
"""
from __future__ import annotations

from boltbeam.perf import mem_tier

# weak -> strong
_STATUS_RANK = {"unknown": 0, "assumed": 1, "imported": 2, "modeled": 3, "derived": 4, "measured": 5}

_TIER_FACT_TO_KEY = {
  "mem.tier.l2.bytes": "l2_bytes",
  "mem.tier.mall.bytes": "mall_bytes",
  "mem.tier.l2.bandwidth_gbs": "l2_gbs",
  "mem.tier.mall.bandwidth_gbs": "mall_gbs",
  "mem.tier.dram.bandwidth_gbs": "dram_gbs",
  "mem.tier.lds.bandwidth_gbs": "lds_gbs",
}

_REQUIRED_KEYS = ("l2_gbs", "dram_gbs", "lds_gbs")


def tiers_from_profile(profile: dict) -> dict:
  facts = profile.get("facts") or []
  by_name = {f.get("name"): f for f in facts if f.get("name") in _TIER_FACT_TO_KEY}

  tiers: dict = {"l2_bytes": None, "mall_bytes": None, "l2_gbs": None, "mall_gbs": None,
                 "dram_gbs": None, "lds_gbs": None}
  used_facts: list[dict] = []
  for name, key in _TIER_FACT_TO_KEY.items():
    fact = by_name.get(name)
    if fact is None: continue
    tiers[key] = fact.get("value")
    used_facts.append(fact)

  blockers = []
  for name, key in _TIER_FACT_TO_KEY.items():
    if key in _REQUIRED_KEYS and by_name.get(name) is None:
      blockers.append(f"missing tier fact: {name}")
  if blockers:
    return {"blockers": blockers}

  tiers["_used_facts"] = used_facts
  return tiers


def lds_l2_for_operand(profile: dict, *, tile_bytes: float, traffic_bytes: float, reuse: float,
                        lds_capacity_bytes: float = 65536.0, role: str = "") -> dict:
  tiers = tiers_from_profile(profile)
  if tiers.get("blockers"):
    return {"blockers": tiers["blockers"]}

  used_facts = tiers.pop("_used_facts")

  result = mem_tier.decide(tile_bytes=tile_bytes, traffic_bytes=traffic_bytes, reuse=reuse,
                            tiers=tiers, lds_capacity_bytes=lds_capacity_bytes)

  weakest_rank = min((_STATUS_RANK.get(f.get("status"), 0) for f in used_facts), default=_STATUS_RANK["measured"])
  truth_status = next(status for status, rank in _STATUS_RANK.items() if rank == weakest_rank)

  result["role"] = role
  result["tile_bytes"] = tile_bytes
  result["reuse"] = reuse
  result["truth_status"] = truth_status
  return result
