"""Theoretical roofline as the anchor for analysis — computed from dimensions and hardware peaks alone.

No runtime, no measurement, and only shapes from the model: for each GEMM role we derive FLOPs and
bytes moved, the arithmetic intensity, the ridge point, and therefore the regime (compute- vs
memory-bound) and the theoretical floor time. The roofline is the ceiling every candidate is judged
against — a measured kernel becomes "X% of roofline", and a role's regime says what is even worth
optimizing before anything runs. Predictions stay predictions (truth_status = modeled); a measured
result would override, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from boltbeam.math.roofline import amdahl_whole_gain
from boltbeam.perf.mem_tier import effective_bandwidth_gbs
from boltbeam.profile.decode_roles import GGML_BITS_PER_WEIGHT
from boltbeam.vocab import SCHEMA_THEORETICAL_ROOFLINE
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

_GEMM_ROLES = ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head")
_ACT_BYTES = 2.0  # fp16 activations


def bits_per_weight(role: Mapping[str, Any]) -> float:
  """Bits per stored weight for a role, from its ggml type or an INTn quant label. Defaults to fp16."""
  ggml = role.get("ggml_type")
  if isinstance(ggml, int) and ggml in GGML_BITS_PER_WEIGHT:
    return GGML_BITS_PER_WEIGHT[ggml]
  quant = str(role.get("quant") or "")
  for token in quant.replace("-", "_").split("_"):
    if token.startswith("INT") and token[3:].isdigit():
      return float(token[3:])
    if token.startswith("Q") and token[1:2].isdigit():
      return float(token[1])
  return 16.0


def _execution_setting(execution: Mapping[str, Any] | None, role: str, key: str, default: Any = None) -> Any:
  if not execution: return default
  role_cfg = (execution.get("roles") or {}).get(role) or {}
  return role_cfg.get(key, execution.get(key, default))


def execution_bits_per_weight(role: Mapping[str, Any], execution: Mapping[str, Any] | None) -> float:
  """Physical weight width used by this execution route, not necessarily the model-file storage width."""
  value = _execution_setting(execution, str(role.get("role") or ""), "weight_bits", "model")
  if value in (None, "model"): return bits_per_weight(role)
  aliases = {"fp16": 16.0, "f16": 16.0, "fp32": 32.0, "f32": 32.0}
  if isinstance(value, str): value = aliases.get(value.lower(), value)
  try: bits = float(value)
  except (TypeError, ValueError) as exc: raise ValueError(f"invalid execution weight_bits {value!r}") from exc
  if bits <= 0: raise ValueError("execution weight_bits must be positive")
  return bits


@dataclass(frozen=True)
class RooflinePoint:
  """One roofline placement. `attainable_flops` is min(peak_flops, AI * peak_bw)."""
  flops: float
  bytes_moved: float
  arithmetic_intensity: float
  ridge_intensity: float
  regime: str                 # "compute" | "memory"
  attainable_flops: float
  floor_ms: float
  pct_of_peak_flops: float

  def to_json(self) -> dict[str, Any]:
    return {
      "flops": self.flops, "bytes_moved": self.bytes_moved,
      "arithmetic_intensity": self.arithmetic_intensity, "ridge_intensity": self.ridge_intensity,
      "regime": self.regime, "attainable_flops": self.attainable_flops,
      "floor_ms": self.floor_ms, "pct_of_peak_flops": self.pct_of_peak_flops,
    }


def roofline_point(flops: float, bytes_moved: float, peak_flops: float, peak_bw_bytes_s: float) -> RooflinePoint:
  """Place (flops, bytes) on the compute/memory roofline for a device with the given peaks. Pure math."""
  if flops < 0 or bytes_moved < 0:
    raise ValueError("flops and bytes must be non-negative")
  if peak_flops <= 0 or peak_bw_bytes_s <= 0:
    raise ValueError("peak_flops and peak_bw must be positive")
  ai = flops / bytes_moved if bytes_moved > 0 else float("inf")
  ridge = peak_flops / peak_bw_bytes_s
  memory_bound_flops = ai * peak_bw_bytes_s
  attainable = min(peak_flops, memory_bound_flops)
  regime = "compute" if ai >= ridge else "memory"
  floor_s = (flops / attainable) if attainable > 0 else float("inf")
  return RooflinePoint(
    flops=flops, bytes_moved=bytes_moved, arithmetic_intensity=ai, ridge_intensity=ridge,
    regime=regime, attainable_flops=attainable, floor_ms=floor_s * 1000.0,
    pct_of_peak_flops=100.0 * attainable / peak_flops,
  )


def roofline_interval_ms(*, flops: float, bytes_moved: float, peak_flops: float,
                         bandwidth_gbs: float, bandwidth_relative_uncertainty: float = 0.0,
                         fixed_cost_s: float = 0.0) -> tuple[float, float]:
  """Modeled roofline floor interval for candidate ordering, never route selection."""
  from boltbeam.perf.mem_tier import cost_interval_s
  if flops < 0 or peak_flops <= 0:
    raise ValueError("flops must be non-negative and peak_flops must be positive")
  memory = cost_interval_s(bytes_moved=bytes_moved, bandwidth_gbs=bandwidth_gbs,
                           bandwidth_relative_uncertainty=bandwidth_relative_uncertainty,
                           fixed_cost_s=fixed_cost_s)
  compute_s = flops / peak_flops
  return (max(compute_s, memory[0]) * 1000.0, max(compute_s, memory[1]) * 1000.0)


def role_flops_bytes(role: Mapping[str, Any], *, context: int, act_bytes: float = _ACT_BYTES) -> tuple[float, float]:
  """FLOPs and bytes moved for one GEMM role across all its layers, at a given prefill context (M tokens).

  Weight is [N, K] = [rows, cols]; y = x[M,K] @ W[N,K]^T. Weights stream once per layer; activations are
  M*K in and M*N out. Bytes uses the role's stored bit-width (quant), activations at act_bytes."""
  n = int(role.get("rows") or 0)
  k = int(role.get("cols") or 0)
  layers = int(role.get("count") or 1)
  m = int(context)
  if n <= 0 or k <= 0 or m <= 0:
    return 0.0, 0.0
  flops_per_layer = 2.0 * m * n * k
  bpw = bits_per_weight(role)
  weight_bytes = n * k * (bpw / 8.0)
  act_bytes_total = (m * k + m * n) * act_bytes
  bytes_per_layer = weight_bytes + act_bytes_total
  return flops_per_layer * layers, bytes_per_layer * layers


@dataclass(frozen=True)
class RoleRoofline:
  role: str
  quant: str | None
  shape: dict[str, int]
  point: RooflinePoint
  whole_share: float          # fraction of the model's total theoretical floor time
  amdahl_ceiling: float       # whole-model speedup if this role were driven to zero cost

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "quant": self.quant, "shape": self.shape,
            "whole_share": self.whole_share, "amdahl_ceiling_if_free": self.amdahl_ceiling,
            **self.point.to_json()}


def model_roofline(profile: Mapping[str, Any], *, peak_flops: float, peak_bw_bytes_s: float,
                   context: int = 512) -> dict[str, Any]:
  """Theoretical per-role + whole-model roofline for a model profile on a device. No runtime, no weights read."""
  roles = profile.get("roles") or []
  # One row per (role, shape, quant). A role stored in two formats (Q4_K_M keeps half of ffn_down in
  # Q6_K) is two rows: keying by role name alone dropped the second format and its bytes.
  variants: dict[tuple[Any, ...], dict[str, Any]] = {}
  for role in roles:
    if not isinstance(role, Mapping) or role.get("role") not in _GEMM_ROLES: continue
    key = (role["role"], int(role.get("rows") or 0), int(role.get("cols") or 0), role.get("quant"), role.get("ggml_type"))
    row = variants.setdefault(key, {**role, "count": 0})
    row["count"] += int(role.get("count") or 1)

  points: list[tuple[str, Mapping[str, Any], float, float, RooflinePoint]] = []
  total_floor_ms = 0.0
  for role in variants.values():
    name = role["role"]
    flops, bytes_moved = role_flops_bytes(role, context=context)
    if flops <= 0:
      continue
    pt = roofline_point(flops, bytes_moved, peak_flops, peak_bw_bytes_s)
    points.append((name, role, flops, bytes_moved, pt))
    total_floor_ms += pt.floor_ms

  role_reports: list[RoleRoofline] = []
  for name, role, flops, bytes_moved, pt in points:
    share = (pt.floor_ms / total_floor_ms) if total_floor_ms > 0 else 0.0
    role_reports.append(RoleRoofline(
      role=name, quant=role.get("quant"),
      shape={"m": context, "n": int(role.get("rows") or 0), "k": int(role.get("cols") or 0)},
      point=pt, whole_share=share,
      amdahl_ceiling=amdahl_whole_gain(share, float("inf")) if 0 < share < 1 else float("inf") if share >= 1 else 1.0,
    ))

  role_reports.sort(key=lambda r: r.whole_share, reverse=True)
  total_flops = sum(f for _, _, f, _, _ in points)
  total_bytes = sum(b for _, _, _, b, _ in points)
  whole = roofline_point(total_flops, total_bytes, peak_flops, peak_bw_bytes_s) if total_flops > 0 else None
  return {
    "schema": SCHEMA_THEORETICAL_ROOFLINE,
    "truth_status": "modeled",
    "context": context,
    "peak_flops": peak_flops,
    "peak_bandwidth_bytes_s": peak_bw_bytes_s,
    "ridge_intensity": peak_flops / peak_bw_bytes_s,
    "whole": (whole.to_json() | {"total_floor_ms": total_floor_ms}) if whole else None,
    "roles": [r.to_json() for r in role_reports],
    "assumptions": [
      "prefill GEMM y=x@W^T; weights streamed once per layer; fp16 activations",
      "peak flops at the matrix-engine dtype; no cache reuse credit beyond a single weight stream",
      "modeled ceiling — a measured A/B overrides it and calibrates the error",
    ],
  }


def pct_of_roofline(measured_flops_per_s: float, roofline: RooflinePoint) -> float:
  """How close a measured throughput is to the role's theoretical attainable ceiling."""
  if roofline.attainable_flops <= 0:
    return 0.0
  return 100.0 * measured_flops_per_s / roofline.attainable_flops


# --- cache-aware (tiered) roofline -------------------------------------------
#
# The flat roofline above assumes one memory ceiling (DRAM). Real hardware has a cache hierarchy, so
# the memory bandwidth a kernel *actually* gets depends on which tier holds its reused working set:
# LDS (explicit) -> L2 -> MALL (AMD Infinity Cache) -> DRAM. A working set that spills DRAM is served
# at ~960 GB/s; one that fits the 96 MB Infinity Cache is served at the (much higher) MALL bandwidth.
# This turns the single roofline line into a stepped one, and makes "fits in Infinity Cache" a real,
# targetable regime — the third rung the LDS-vs-L2 discussion was missing.


def gfx1100_tiers(dram_gbs: float = DEFAULT_PEAK_MEM_GBS) -> dict[str, float]:
  """RDNA3 / Navi 31 (RX 7900 XTX) cache hierarchy. L2 (6 MB, on GCD), MALL = Infinity Cache
  (96 MB, on the MCDs), DRAM (960 GB/s). The cache *bandwidths* are MODELED placeholders — run a
  working-set sweep (`boltbeam mem-sweep-request` / `ingest-mem-sweep`) to replace them with measured
  values. Only the capacities (6 MB L2, 96 MB Infinity Cache) are hardware facts."""
  return {
    "l2_bytes": 6 * 1024 * 1024, "l2_gbs": 3400.0,        # modeled
    "mall_bytes": 96 * 1024 * 1024, "mall_gbs": 2400.0,   # modeled — Infinity Cache
    "dram_gbs": dram_gbs, "lds_gbs": 16000.0,             # modeled
  }


def registered_memory_tiers(target, *, dram_gbs: float | None = None) -> dict[str, Any]:
  """Read provider-neutral L2/last-level-cache/DRAM/LDS facts from a target descriptor."""
  hierarchy = (target.capabilities or {}).get("memory_hierarchy") or {}
  raw = hierarchy.get("tiers") or {}
  try:
    l2, llc, dram, lds = raw["l2"], raw["last_level_cache"], raw["dram"], raw["lds"]
    return {
      "l2_bytes": float(l2["bytes"]), "l2_gbs": float(l2["bandwidth_gbs"]),
      "mall_bytes": float(llc["bytes"]), "mall_gbs": float(llc["bandwidth_gbs"]),
      "dram_gbs": float(dram_gbs if dram_gbs is not None else dram["bandwidth_gbs"]),
      "lds_gbs": float(lds["bandwidth_gbs"]),
      "_truth_status": hierarchy.get("truth_status", "unknown"),
      "_last_level_cache_name": llc.get("vendor_name") or "last_level_cache",
      "_l2_bandwidth_status": l2.get("bandwidth_status", "unknown"),
      "_mall_bandwidth_status": llc.get("bandwidth_status", "unknown"),
      "_dram_bandwidth_status": dram.get("bandwidth_status", "unknown"),
      "_lds_bandwidth_status": lds.get("bandwidth_status", "unknown"),
      "_calibration": hierarchy.get("calibration"),
    }
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"target {getattr(target, 'target_id', None)!r} has no complete memory_hierarchy") from exc


def calibrated_memory_tiers(target, profile: Mapping[str, Any], *, dram_gbs: float | None = None) -> dict[str, Any]:
  """Overlay working-set-sweep facts onto target defaults, preserving the weakest evidence status."""
  from boltbeam.memory.lds_l2 import tiers_from_profile
  observed = tiers_from_profile(dict(profile))
  if observed.get("blockers"):
    raise ValueError("memory hierarchy profile incomplete: " + "; ".join(observed["blockers"]))
  used = observed.pop("_used_facts", [])
  out = registered_memory_tiers(target, dram_gbs=dram_gbs)
  for key in ("l2_bytes", "mall_bytes", "l2_gbs", "mall_gbs", "dram_gbs", "lds_gbs"):
    if observed.get(key) is not None: out[key] = float(observed[key])
  rank = {"unknown": 0, "assumed": 1, "imported": 2, "modeled": 3, "derived": 4, "measured": 5}
  statuses = [str(f.get("status") or "unknown") for f in used]
  out["_truth_status"] = min(statuses, key=lambda s: rank.get(s, 0)) if statuses else "unknown"
  by_name = {f.get("name"): f for f in used}
  out["_l2_bandwidth_status"] = (by_name.get("mem.tier.l2.bandwidth_gbs") or {}).get("status", "unknown")
  out["_mall_bandwidth_status"] = (by_name.get("mem.tier.mall.bandwidth_gbs") or {}).get("status", "unknown")
  out["_dram_bandwidth_status"] = (by_name.get("mem.tier.dram.bandwidth_gbs") or {}).get("status", "unknown")
  out["_lds_bandwidth_status"] = (by_name.get("mem.tier.lds.bandwidth_gbs") or {}).get("status", "unknown")
  out["_calibration"] = f"working-set sweep {profile.get('system_snapshot_id') or 'unknown snapshot'}"
  return out


def cache_tier_for(working_set_bytes: float, tiers: dict) -> str:
  """Name the smallest tier that holds the working set: l2 | mall | dram."""
  if working_set_bytes <= tiers["l2_bytes"]:
    return "l2"
  if tiers.get("mall_bytes") and working_set_bytes <= tiers["mall_bytes"]:
    return "mall"
  return "dram"


def cache_tier_candidates(working_set_bytes: float, tiers: dict, *, boundary_fraction: float = 0.10) -> tuple[str, ...]:
  """Return a soft tier band near capacity boundaries instead of claiming exact physical residency."""
  l2, mall = float(tiers["l2_bytes"]), float(tiers.get("mall_bytes") or 0.0)
  if l2 * (1.0 - boundary_fraction) <= working_set_bytes <= l2 * (1.0 + boundary_fraction):
    return ("l2", "mall")
  if mall and mall * (1.0 - boundary_fraction) <= working_set_bytes <= mall * (1.0 + boundary_fraction):
    return ("mall", "dram")
  return (cache_tier_for(working_set_bytes, tiers),)


@dataclass(frozen=True)
class TieredRooflinePoint:
  flops: float
  working_set_bytes: float
  reuse: float
  cache_tier: str                 # l2 | mall | dram
  effective_bw_gbs: float         # bandwidth of the serving tier
  dram_bw_gbs: float
  regime: str                     # "compute" | "memory"
  attainable_flops: float
  floor_ms: float                 # with cache residency
  floor_ms_dram_only: float       # if the same traffic were forced to DRAM
  cache_speedup: float            # floor_ms_dram_only / floor_ms

  def to_json(self) -> dict[str, Any]:
    return {
      "flops": self.flops, "working_set_bytes": self.working_set_bytes, "reuse": self.reuse,
      "cache_tier": self.cache_tier, "effective_bw_gbs": self.effective_bw_gbs,
      "dram_bw_gbs": self.dram_bw_gbs, "regime": self.regime,
      "attainable_flops": self.attainable_flops, "floor_ms": self.floor_ms,
      "floor_ms_dram_only": self.floor_ms_dram_only, "cache_speedup": self.cache_speedup,
    }


def cache_aware_roofline_point(*, flops: float, working_set_bytes: float, reuse: float,
                               activation_bytes: float, peak_flops: float,
                               tiers: dict) -> TieredRooflinePoint:
  """Roofline where the memory ceiling is the *serving tier's* bandwidth, not a flat DRAM number.

  The reused operand (working_set_bytes, reused `reuse` times) is served at the bandwidth of the
  smallest tier that holds it; one-time activation traffic always comes from DRAM. Compute is bounded
  by peak_flops. This models the `cache_streamed` strategy: whether the working set stays resident in
  L2 / Infinity Cache / DRAM sets the achievable bandwidth."""
  if peak_flops <= 0 or tiers["dram_gbs"] <= 0:
    raise ValueError("peak_flops and dram bandwidth must be positive")
  dram_bw = tiers["dram_gbs"] * 1e9
  eff_bw = effective_bandwidth_gbs(working_set_bytes, tiers) * 1e9
  t_compute = flops / peak_flops
  # The first (fill) pass is always a compulsory DRAM miss — a cache can't serve data it doesn't yet
  # hold. Only the *reuse* (reuse-1 further passes) is served from the resident tier. So reuse<=1
  # (e.g. single-token decode) gets no cache benefit; heavy reuse (prefill) rides the cache.
  fill = working_set_bytes / dram_bw
  reuse_reads = max(reuse - 1.0, 0.0) * working_set_bytes
  t_mem = fill + reuse_reads / eff_bw + activation_bytes / dram_bw
  t_mem_dram = fill + reuse_reads / dram_bw + activation_bytes / dram_bw
  t = max(t_compute, t_mem)
  t_dram = max(t_compute, t_mem_dram)
  regime = "compute" if t_compute >= t_mem else "memory"
  floor_ms = t * 1000.0
  floor_ms_dram = t_dram * 1000.0
  return TieredRooflinePoint(
    flops=flops, working_set_bytes=working_set_bytes, reuse=reuse,
    cache_tier=cache_tier_for(working_set_bytes, tiers), effective_bw_gbs=eff_bw / 1e9,
    dram_bw_gbs=dram_bw / 1e9, regime=regime,
    attainable_flops=flops / t if t > 0 else 0.0, floor_ms=floor_ms,
    floor_ms_dram_only=floor_ms_dram, cache_speedup=(floor_ms_dram / floor_ms) if floor_ms > 0 else 1.0,
  )


def throughput_rollup(profile: Mapping[str, Any], *, peak_flops: float, tiers: dict,
                      prefill_context: int = 512, prefill_execution: Mapping[str, Any] | None = None,
                      decode_execution: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Translate the per-role roofline into end-to-end tok/s for decode and prefill.

  Decode (one token at a time): per-token time is the sum of every GEMM role's floor across all layers;
  it is memory-bound, so tok/s ~= DRAM_bw / total_weight_bytes and the Infinity Cache does not help.
  Prefill (M tokens together): tok/s = M / (time to process M tokens); only duplicate global weight-fetch
  groups implied by tile_M are cache-eligible, while within-workgroup reuse belongs to registers/LDS.
  Each role reports its share of the per-token budget — i.e. what actually eats your speed."""
  def _rollup(context: int, execution: Mapping[str, Any] | None) -> dict[str, Any]:
    report = tiered_model_roofline(profile, peak_flops=peak_flops, tiers=tiers, context=context,
                                   execution=execution)
    roles = report["roles"]
    total_ms = sum(r["floor_ms"] for r in roles)
    tokens = context
    tok_s = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
    breakdown = [{
      "role": r["role"], "ms_per_pass": r["floor_ms"],
      "pct_of_step": (100.0 * r["floor_ms"] / total_ms) if total_ms > 0 else 0.0,
      "cache_tier": r["cache_tier"], "regime": r["regime"], "cache_speedup": r["cache_speedup"],
      "execution_weight_bits": r["execution_weight_bits"], "tile_m": r["tile_m"],
      "global_weight_fetch_groups": r["global_weight_fetch_groups"], "transport": r["transport"],
      "weight_mib": r["weight_mib"], "cache_tier_candidates": r["cache_tier_candidates"],
      "cache_tier_confidence": r["cache_tier_confidence"],
      "cache_reuse_eligible": r["cache_reuse_eligible"], "serving_tier": r["serving_tier"],
    } for r in roles]
    return {"tok_s": tok_s, "step_ms": total_ms, "context": context, "tiers": report["tiers"],
            "roles": breakdown, "assumptions": report["assumptions"]}

  return {
    "schema": SCHEMA_THEORETICAL_ROOFLINE,
    "truth_status": tiers.get("_truth_status", "modeled"),
    "model_id": profile.get("model_id"),
    "decode": _rollup(1, decode_execution),
    "prefill": _rollup(prefill_context, prefill_execution),
    "note": "decode is a compulsory DRAM stream at one fetch group; prefill duplicate global fetch groups may be "
            "served by a fitting cache tier. Tier truth status is carried explicitly.",
  }


def tiered_model_roofline(profile: Mapping[str, Any], *, peak_flops: float, tiers: dict,
                          context: int = 512, execution: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Cache-aware per-role roofline using physical route bytes and global weight-fetch groups.

  A tiled GEMM does not reread the full weight once per token: workgroups along M duplicate each weight tile. The
  cache reuse count is therefore ceil(M/tile_M), supplied by execution evidence. Missing tile evidence gets no
  cache-reuse credit (one compulsory stream), rather than inventing M rereads. Mixed quant variants are preserved.
  """
  roles = profile.get("roles") or []
  grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
  for role in roles:
    if not isinstance(role, Mapping) or role.get("role") not in _GEMM_ROLES: continue
    name = str(role["role"])
    n, k = int(role.get("rows") or 0), int(role.get("cols") or 0)
    if n <= 0 or k <= 0: continue
    m = int(_execution_setting(execution, name, "context", context))
    if m <= 0: raise ValueError(f"execution context for {name} must be positive")
    bits = execution_bits_per_weight(role, execution)
    tile_m_raw = _execution_setting(execution, name, "tile_m", None)
    tile_m = m if tile_m_raw is None else int(tile_m_raw)
    if tile_m <= 0: raise ValueError(f"execution tile_m for {name} must be positive")
    transport = str(_execution_setting(execution, name, "transport", "cache_streamed"))
    key = (name, n, k, bits, m, tile_m, transport)
    row = grouped.setdefault(key, {"role": name, "rows": n, "cols": k, "bits": bits, "context": m,
                                   "tile_m": tile_m, "transport": transport, "layers": 0, "quants": set()})
    row["layers"] += int(role.get("count") or 1)
    if role.get("quant"): row["quants"].add(str(role["quant"]))
  out_roles = []
  for row in grouped.values():
    name, n, k, m = row["role"], row["rows"], row["cols"], row["context"]
    layers, bpw, tile_m = row["layers"], row["bits"], row["tile_m"]
    fetch_groups = int(math.ceil(m / tile_m))
    # Tier residency is decided per-layer: one GEMM kernel holds one layer's weight matrix. Time then
    # accumulates over all `layers` layers that share this role.
    weight_bytes = n * k * (bpw / 8.0)          # one layer's weight matrix = the reused working set
    activation_bytes = (m * k + m * n) * _ACT_BYTES
    flops = 2.0 * m * n * k
    pt = cache_aware_roofline_point(flops=flops, working_set_bytes=weight_bytes, reuse=fetch_groups,
                                    activation_bytes=activation_bytes, peak_flops=peak_flops, tiers=tiers)
    tier_candidates = cache_tier_candidates(weight_bytes, tiers)
    role_json = pt.to_json()
    role_json["floor_ms"] = pt.floor_ms * layers           # all layers of this role
    role_json["floor_ms_dram_only"] = pt.floor_ms_dram_only * layers
    role_json["floor_ms_per_layer"] = pt.floor_ms
    out_roles.append({"role": name, "quant": "+".join(sorted(row["quants"])) or None, "layers": layers,
                      "execution_weight_bits": bpw, "execution_context": m, "tile_m": tile_m,
                      "global_weight_fetch_groups": fetch_groups, "transport": row["transport"],
                      "cache_tier_candidates": list(tier_candidates),
                      "cache_tier_confidence": "low_boundary" if len(tier_candidates) > 1 else "modeled",
                      "cache_reuse_eligible": fetch_groups > 1,
                      "serving_tier": pt.cache_tier if fetch_groups > 1 else "dram_compulsory",
                      "weight_mib": weight_bytes / (1024 * 1024), **role_json})
  out_roles.sort(key=lambda r: r["floor_ms"], reverse=True)
  return {
    "schema": SCHEMA_THEORETICAL_ROOFLINE,
    "truth_status": tiers.get("_truth_status", "modeled"),
    "model_id": profile.get("model_id"),
    "context": context,
    "tiers": {"l2_mib": tiers["l2_bytes"] / (1024 * 1024), "l2_gbs": tiers["l2_gbs"],
              "mall_mib": (tiers.get("mall_bytes") or 0) / (1024 * 1024), "mall_gbs": tiers.get("mall_gbs"),
              "dram_gbs": tiers["dram_gbs"],
              "last_level_cache_name": tiers.get("_last_level_cache_name", "MALL"),
              "truth_status": tiers.get("_truth_status", "modeled"),
              "l2_bandwidth_status": tiers.get("_l2_bandwidth_status", "modeled"),
              "mall_bandwidth_status": tiers.get("_mall_bandwidth_status", "modeled"),
              "dram_bandwidth_status": tiers.get("_dram_bandwidth_status", "modeled"),
              "calibration": tiers.get("_calibration")},
    "roles": out_roles,
    "assumptions": [
      "physical weight width and tile_M come from execution evidence when provided; otherwise model storage is used and no cache-reuse credit is granted",
      "global weight fetches = ceil(execution M / tile_M); only fetches after the compulsory first may hit the smallest fitting cache tier",
      "activations are conservatively charged to DRAM",
      "cache-tier capacity and bandwidth truth statuses are carried in the tier block; modeled values may be replaced by a working-set sweep",
      "models the cache_streamed strategy; explicit LDS staging is the separate lds_l2 decision",
    ],
  }


# thresholds for the roofline verdict
_AT_CEILING_PCT = 80.0
_HEADROOM_PCT = 50.0


def assess_evidence(evidence, *, peak_flops: float, peak_bw_bytes_s: float) -> dict[str, Any]:
  """Judge one measured kernel against its own theoretical roofline: place its shape, and if timing is
  measured, report the fraction of the attainable ceiling it reached and what to do about it.

  This is the roofline-first analysis: the ceiling is derived first (from shape + peaks), then evidence
  is scored against it. A prediction never becomes a verdict — the roofline stays modeled; only the
  measured throughput is measured."""
  w = evidence.workload
  m = int(w.shape.get("m") or 0)
  n = int(w.shape.get("n") or 0)
  k = int(w.shape.get("k") or 0)
  if min(m, n, k) <= 0:
    return {"assessed": False, "reason": "workload has no concrete m/n/k shape"}
  bpw = bits_per_weight({"quant": (w.dtypes or {}).get("input")})
  flops = 2.0 * m * n * k
  bytes_moved = n * k * (bpw / 8.0) + (m * k + m * n) * _ACT_BYTES
  point = roofline_point(flops, bytes_moved, peak_flops, peak_bw_bytes_s)
  out: dict[str, Any] = {"assessed": True, "roofline": point.to_json(), "roofline_truth_status": "modeled"}

  t = evidence.timing
  if t is None or not t.samples or evidence.stages.timing.status != "measured":
    out["measured"] = False
    out["verdict"] = "no measured timing; roofline gives the ceiling and the regime only"
    return out
  median_ms = sorted(t.normalized_samples_ms)[len(t.samples) // 2]
  achieved_flops = flops / (median_ms / 1000.0) if median_ms > 0 else 0.0
  pct = pct_of_roofline(achieved_flops, point)
  out.update({"measured": True, "measured_flops_per_s": achieved_flops,
              "measured_ms": median_ms, "pct_of_roofline": pct,
              "measured_truth_status": "measured"})
  if pct >= _AT_CEILING_PCT:
    out["verdict"] = f"at the {point.regime} ceiling ({pct:.0f}% of roofline) — little headroom"
  elif pct >= _HEADROOM_PCT:
    out["verdict"] = f"{pct:.0f}% of the {point.regime} roofline — moderate headroom"
  else:
    out["verdict"] = (f"{pct:.0f}% of the {point.regime} roofline — large headroom; "
                      f"optimize the {point.regime}-bound path")
  return out
