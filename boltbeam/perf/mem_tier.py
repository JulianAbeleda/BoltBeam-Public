"""Pure LDS-vs-cache crossover primitives (standalone; does not touch cycle_model)."""
from __future__ import annotations

from boltbeam.vocab import SCHEMA_LDS_L2_CANDIDATE


def cost_interval_s(*, bytes_moved: float, bandwidth_gbs: float,
                    bandwidth_relative_uncertainty: float = 0.0,
                    fixed_cost_s: float = 0.0) -> tuple[float, float]:
  """Return a conservative transfer-time interval without turning a model into a verdict."""
  if bytes_moved < 0 or bandwidth_gbs <= 0 or fixed_cost_s < 0:
    raise ValueError("bytes/fixed cost must be non-negative and bandwidth must be positive")
  if not 0 <= bandwidth_relative_uncertainty < 1:
    raise ValueError("bandwidth_relative_uncertainty must be in [0, 1)")
  fastest = bandwidth_gbs * (1.0 + bandwidth_relative_uncertainty)
  slowest = bandwidth_gbs * (1.0 - bandwidth_relative_uncertainty)
  return (fixed_cost_s + bytes_moved / (fastest * 1e9),
          fixed_cost_s + bytes_moved / (slowest * 1e9))


def effective_bandwidth_gbs(working_set_bytes: float, tiers: dict) -> float:
  """Pick the bandwidth of the smallest tier that holds `working_set_bytes`."""
  l2_bytes = tiers["l2_bytes"]
  mall_bytes = tiers.get("mall_bytes")
  if working_set_bytes <= l2_bytes:
    return tiers["l2_gbs"]
  if mall_bytes is not None and working_set_bytes <= mall_bytes:
    return tiers["mall_gbs"]
  return tiers["dram_gbs"]


def stream_cost_s(traffic_bytes: float, reuse: float, working_set_bytes: float, tiers: dict) -> float:
  """Cost of relying on the cache tier to serve `reuse` reads of `traffic_bytes`."""
  bw_gbs = effective_bandwidth_gbs(working_set_bytes, tiers)
  return (reuse * traffic_bytes) / (bw_gbs * 1e9)


def lds_cost_s(tile_bytes: float, traffic_bytes: float, reuse: float, tiers: dict, *,
               barrier_s: float = 0.0, occupancy_factor: float = 1.0) -> float:
  """Cost of staging `tile_bytes` through LDS: one DRAM fill + reuse from LDS + barrier."""
  fill_term = tile_bytes / (tiers["dram_gbs"] * 1e9)
  reuse_term = (reuse * traffic_bytes) / (tiers["lds_gbs"] * 1e9)
  return fill_term + (reuse_term / occupancy_factor) + barrier_s


def decide(*, tile_bytes: float, traffic_bytes: float, reuse: float, tiers: dict,
           lds_capacity_bytes: float, barrier_s: float = 0.0, occupancy_factor: float = 1.0) -> dict:
  """Predict whether to stage `tile_bytes` in LDS or rely on the cache. Prediction, not verdict."""
  stream = stream_cost_s(traffic_bytes, reuse, tile_bytes, tiers)
  lds = lds_cost_s(tile_bytes, traffic_bytes, reuse, tiers, barrier_s=barrier_s,
                    occupancy_factor=occupancy_factor)

  if reuse <= 1:
    reason = "reuse<=1"
    stage_lds = False
  elif tile_bytes > lds_capacity_bytes:
    reason = "tile_exceeds_lds"
    stage_lds = False
  elif lds < stream:
    reason = "lds_faster"
    stage_lds = True
  else:
    reason = "cache_faster"
    stage_lds = False

  return {"schema": SCHEMA_LDS_L2_CANDIDATE,
          "recommendation": "lds_staged" if stage_lds else "cache_streamed",
          "stream_cost_s": stream,
          "lds_cost_s": lds,
          "reason": reason,
          "prediction_only": True}
