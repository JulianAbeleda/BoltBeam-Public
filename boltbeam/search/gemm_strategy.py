"""Tile-GEMM strategy space derived from a GPU's facts and its compiler's lowering facts.

The space a kernel search walks (tile, warp grid, LDS pipeline and stage count, split-K depth) is not a hand list:
every axis bound is one fact of the GPU (``GemmMachine``: SM count, shared memory per SM and per block, register
file, thread and block limits, measured DRAM bandwidth and matrix rate -- the target registry's scan) or one fact of
the compiler that must emit it (``GemmLowering``: the tensor-core atom and fragment registers, the LDS limits, whether
async copies / native matrix fragment loads exist, which unpadded LDS rows are bank-conflict free under the XOR
swizzle, and what the lowering cannot yet do).  The lowering facts are produced by the compiler (tinygrad exports
them from its renderer and tensor-core descriptor); BoltBeam never imports the compiler.

Axes (each bound is named where it is applied):
  * tiles are powers-of-two multiples of the tensor-core atom (M, N, K);
  * rows may be padded up to the tile only while the padded product stays under the shape's memory roof (padding a
    memory-bound product is free; padding a compute-bound one is not);
  * warp grids tile the CTA tile with whole atoms, within the per-block thread limit; each thread moves whole
    ``vector_bytes`` vectors (the cooperative copy's constraint);
  * register demand (accumulators + double-buffered fragments) within the per-thread register limit;
  * pipelines: the register-staged two-buffer schedule within the static LDS limit, and -- where async copies exist --
    a ring of every stage count from 2 up to what the launch-sized LDS limit and the K loop admit.  A swizzled
    (unpadded) window is used whenever it is conflict free: a padded window is conflict free too but strictly larger;
  * split-K: every split the lowering can express whose K slice still holds two K tiles.  No wave-count bound: a
    measured winner (output, 32 rows) splits a grid that already fills two waves, so waves are the cost model's job,
    not the space's.

A roofline-with-quantization model (``estimate``) ranks candidates and names the limiting factor (occupancy source,
wave quantization, memory or compute); ``prune`` keeps candidates within a ratio of the model's best.  The model
orders and prunes only -- selection is always by measurement.

Families the facts admit but the lowering cannot yet emit are reported by ``deferred`` (ragged split-K, stream-K),
so a derivation gap is visible instead of silently absent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math, re
from typing import Any, Iterable, Mapping, NamedTuple

SCHEMA_LOWERING = "boltbeam.gemm_lowering_facts.v1"
SCHEMA_SPACE = "boltbeam.gemm_strategy_space.v1"

# Pipeline tuple, identical to tinygrad's dense search: (stages, async_copy, matrix_fragments, xor_swizzle).
SYNC2 = (2, False, False, False)


def _pow2_multiples(base:int, limit:int) -> list[int]:
  out, v = [], base
  while v <= limit: out.append(v); v *= 2
  return out


@dataclass(frozen=True)
class GemmMachine:
  """The GPU's facts a GEMM strategy depends on (all scanned or measured on the card; see targets.json)."""
  target_id: str
  sm_count: int
  warp_size: int
  max_threads_per_block: int
  max_threads_per_sm: int
  max_blocks_per_sm: int
  smem_per_sm: int
  smem_per_block_optin: int
  smem_reserved_per_block: int
  registers_per_sm: int
  max_registers_per_thread: int
  sm_subpartitions: int
  l2_bytes: int
  l2_gbs: float
  dram_gbs: float
  matrix_tflops: float
  # Optional measured facts for the calibrated cost model (gemm_calibration.py): one SM's own DRAM streaming rate and
  # the fitted term coefficients.  Absent -> estimate() is the uncalibrated roofline.
  sm_stream_gbs: float | None = None
  cost_calibration: Mapping[str, Any] | None = field(default=None, compare=False, hash=False)

  @classmethod
  def from_target(cls, target_id:str="nvidia_sm120", registry:Mapping[str, Any] | None=None) -> "GemmMachine":
    """Read the facts from the target registry's scan rows; a missing fact is an error, never a default."""
    if registry is None:
      from boltbeam.target.targets import TARGETS as registry
    profile = registry[target_id]
    sources = (profile.capabilities or {}).get("fact_sources", {})
    scan, occ = sources.get("hardware_scan", {}), sources.get("occupancy_scan", {})
    def need(where:Mapping[str, Any], key:str, label:str):
      if where.get(key) is None: raise ValueError(f"target {target_id} lacks fact {label}.{key}; scan the GPU (targets.json)")
      return where[key]
    matrix = (profile.matrix_tflops or {}).get("fp16")
    if matrix is None or profile.memory_bandwidth_gbs is None:
      raise ValueError(f"target {target_id} lacks a measured matrix rate or DRAM bandwidth")
    return cls(target_id=target_id, sm_count=need(scan, "sm_count", "hardware_scan"), warp_size=need(scan, "warp_size", "hardware_scan"),
               max_threads_per_block=need(scan, "max_threads_per_block", "hardware_scan"),
               max_threads_per_sm=need(occ, "max_threads_per_sm", "occupancy_scan"),
               max_blocks_per_sm=need(occ, "max_blocks_per_sm", "occupancy_scan"),
               smem_per_sm=need(scan, "shared_mem_per_sm_bytes", "hardware_scan"),
               smem_per_block_optin=need(occ, "shared_mem_per_block_optin_bytes", "occupancy_scan"),
               smem_reserved_per_block=need(occ, "reserved_shared_mem_per_block_bytes", "occupancy_scan"),
               registers_per_sm=need(occ, "registers_per_sm", "occupancy_scan"),
               max_registers_per_thread=need(occ, "max_registers_per_thread", "occupancy_scan"),
               sm_subpartitions=need(sources.get("sm_architecture", {}), "sm_subpartitions", "sm_architecture"),
               l2_bytes=need(occ, "l2_cache_bytes", "occupancy_scan"),
               l2_gbs=float(need(sources.get("l2_bandwidth_gbs", {}), "value_gbs", "l2_bandwidth_gbs")), dram_gbs=float(profile.memory_bandwidth_gbs),
               matrix_tflops=float(matrix),
               sm_stream_gbs=(float(sources["sm_stream_gbs"]["value_gbs"]) if "sm_stream_gbs" in sources else None),
               cost_calibration=sources.get("gemm_cost_calibration"))


@dataclass(frozen=True)
class GemmLowering:
  """What the compiler can emit for one (backend, arch, dtype) GEMM -- exported by the compiler, never guessed here."""
  backend: str
  arch: str
  mma_m: int
  mma_n: int
  mma_k: int
  a_fragment_regs: int            # 32-bit registers per lane for one A atom fragment
  b_fragment_regs: int
  accumulator_regs: int           # per lane per atom
  operand_bytes: int
  accumulator_bytes: int
  vector_bytes: int               # cooperative global->LDS vector
  static_lds_bytes: int           # register-staged schedule (static shared memory)
  runtime_lds_bytes: int | None   # launch-sized (dynamic) shared memory, None when the renderer has none
  lds_padding_bytes: int          # padded (unswizzled) window row padding
  async_copy: bool                # global->LDS async copy (cp.async class)
  matrix_fragments: bool          # native LDS matrix fragment loads (ldmatrix class) derivable from the atom
  swizzle_row_bytes: tuple[int, ...]   # unpadded LDS row widths the XOR swizzle makes conflict free
  weight_row_granule: int         # the route pads N (weight rows) to a multiple of this; tile_n must divide it
  ragged_split_k: bool = False    # a K slice that is not a whole number of K tiles
  stream_k: bool = False          # persistent / stream-K grid for dense tiles
  provenance: Mapping[str, str] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    out = asdict(self); out["swizzle_row_bytes"] = list(self.swizzle_row_bytes); out["provenance"] = dict(self.provenance)
    return {"schema": SCHEMA_LOWERING, **out}

  @classmethod
  def from_json(cls, value:Mapping[str, Any]) -> "GemmLowering":
    if value.get("schema") != SCHEMA_LOWERING: raise ValueError(f"not a {SCHEMA_LOWERING} document")
    body = {k: v for k, v in value.items() if k != "schema"}
    names = {f for f in cls.__dataclass_fields__}
    if set(body) - names: raise ValueError(f"unknown lowering facts {sorted(set(body) - names)}")
    body["swizzle_row_bytes"] = tuple(body.get("swizzle_row_bytes", ()))
    return cls(**body)


class Config(NamedTuple):
  geometry: tuple[int, int, int, int, int]   # (tile_m, tile_n, tile_k, warps_m, warps_n) -- tinygrad's order
  split_k: int
  pipeline: tuple[int, bool, bool, bool]


@dataclass(frozen=True)
class Shape:
  m: int
  n: int
  k: int

  def n_route(self, lowering:GemmLowering) -> int: return -(-self.n // lowering.weight_row_granule) * lowering.weight_row_granule


def lds_bytes(lowering:GemmLowering, geometry, pipeline) -> int:
  tm, tn, tk = geometry[:3]
  stages, _, _, swizzle = pipeline
  return stages * (tm + tn) * (tk * lowering.operand_bytes + (0 if swizzle else lowering.lds_padding_bytes))


def register_demand(lowering:GemmLowering, geometry) -> int:
  """Accumulators plus double-buffered A/B fragments per lane (the loop body holds one K atom while loading the next)."""
  tm, tn, _, wm, wn = geometry
  sub_m, sub_n = tm // wm // lowering.mma_m, tn // wn // lowering.mma_n
  return sub_m * sub_n * lowering.accumulator_regs + 2 * (sub_m * lowering.a_fragment_regs + sub_n * lowering.b_fragment_regs)


# Address/index/predicate registers on top of the demand above.  An estimate, used only for the occupancy the model
# reports -- never to admit or reject a candidate.  Calibrated on ncu: ours 64x64x64 2x2 = 122 regs (model 104+32=136
# upper), cuBLAS 128x64x64 = 158 (model 112+32=144), cuBLAS 64x256x32 = 230 (model 208+32=240).
REGISTER_OVERHEAD_ESTIMATE = 32


def occupancy(machine:GemmMachine, lowering:GemmLowering, geometry, pipeline) -> tuple[int, str]:
  """(CTAs resident per SM, the limiting resource)."""
  tm, tn, tk, wm, wn = geometry
  threads = wm * wn * machine.warp_size
  regs = min(machine.max_registers_per_thread, register_demand(lowering, geometry) + REGISTER_OVERHEAD_ESTIMATE)
  regs = -(-regs // 8) * 8
  limits = {"blocks": machine.max_blocks_per_sm, "threads": machine.max_threads_per_sm // threads,
            "lds": machine.smem_per_sm // (lds_bytes(lowering, geometry, pipeline) + machine.smem_reserved_per_block),
            "registers": machine.registers_per_sm // (regs * threads)}
  source = min(limits, key=lambda key: (limits[key], key))
  return max(0, limits[source]), source


def _families(lowering:GemmLowering) -> list[str]:
  return ["sync2"] + (["ring"] if lowering.async_copy and lowering.runtime_lds_bytes else [])


def _memory_bytes(lowering:GemmLowering, shape:Shape, m_pad:int, n_pad:int, split:int) -> int:
  ob, ab = lowering.operand_bytes, lowering.accumulator_bytes
  out = m_pad * n_pad * ab
  partials = 2 * split * out if split > 1 else 0   # written by the GEMM, read back by the reduction
  return n_pad * shape.k * ob + m_pad * shape.k * ob + out + partials


def _row_tiles(machine:GemmMachine, lowering:GemmLowering, shape:Shape) -> list[int]:
  """Tile rows: divisors of M, plus padded tiles while the padded product stays under the memory roof."""
  roof_us = _memory_bytes(lowering, shape, shape.m, shape.n_route(lowering), 1) / (machine.dram_gbs * 1e3)
  out = []
  for tm in _pow2_multiples(lowering.mma_m, max(lowering.mma_m, 4 * shape.m)):
    m_pad = -(-shape.m // tm) * tm
    if m_pad == shape.m or 2 * m_pad * shape.n_route(lowering) * shape.k / (machine.matrix_tflops * 1e6) <= roof_us: out.append(tm)
  return out


def _splits(lowering:GemmLowering, shape:Shape, tk:int) -> list[int]:
  """Splits whose K slices the lowering expresses: whole K tiles and at least two per slice (the precontract loop),
  or -- with ragged K -- any slice of at least one (predicated) tile."""
  if lowering.ragged_split_k:
    tiles = -(-shape.k // tk)
    return [s for s in range(1, tiles + 1) if -(-tiles // s) * (s - 1) < tiles]   # every slice non-empty
  return [s for s in range(1, shape.k // (2 * tk) + 1) if not shape.k % s and not (shape.k // s) % tk]


def derive_space(machine:GemmMachine, lowering:GemmLowering, shape:Shape) -> list[Config]:
  """Every (geometry, split, pipeline) the facts admit for one exact shape (see module docstring for each bound)."""
  n_route = shape.n_route(lowering)
  out: list[Config] = []
  swizzle_rows = set(lowering.swizzle_row_bytes)
  for tm in _row_tiles(machine, lowering, shape):
    m_pad = -(-shape.m // tm) * tm
    for tn in _pow2_multiples(lowering.mma_n, n_route):
      if n_route % tn: continue
      for tk in _pow2_multiples(lowering.mma_k, shape.k if lowering.ragged_split_k else shape.k // 2):
        vpr = tk * lowering.operand_bytes // lowering.vector_bytes
        if vpr < 1: continue
        splits = _splits(lowering, shape, tk)
        if not splits: continue
        for wm in _pow2_multiples(1, tm // lowering.mma_m):
          for wn in _pow2_multiples(1, tn // lowering.mma_n):
            threads = wm * wn * machine.warp_size
            if threads > machine.max_threads_per_block or (tm * vpr) % threads or (tn * vpr) % threads: continue
            geometry = (tm, tn, tk, wm, wn)
            if register_demand(lowering, geometry) > machine.max_registers_per_thread: continue
            pipes: list[tuple[int, bool, bool, bool]] = []
            if lds_bytes(lowering, geometry, SYNC2) <= lowering.static_lds_bytes: pipes.append(SYNC2)
            if "ring" in _families(lowering):
              swizzle = tk * lowering.operand_bytes in swizzle_rows
              probe = (1, True, lowering.matrix_fragments, swizzle)
              per_stage = lds_bytes(lowering, geometry, probe)
              for stages in range(2, min(lowering.runtime_lds_bytes // per_stage, -(-shape.k // tk)) + 1):
                pipes.append((stages, True, lowering.matrix_fragments, swizzle))
            for pipe in pipes:
              if occupancy(machine, lowering, geometry, pipe)[0] < 1: continue
              for s in splits:
                if pipe[0] > -(-(-(-shape.k // s)) // tk): continue   # a ring deeper than the K loop
                out.append(Config(geometry, s, pipe))
  return out


@dataclass(frozen=True)
class Estimate:
  model_us: float
  compute_us: float
  memory_us: float
  l2_us: float
  ctas: int
  ctas_per_sm: int
  occupancy_limit: str
  waves: float
  wave_efficiency: float
  lds_bytes: int
  registers_estimate: int
  bound: str
  limiter: str


def estimate(machine:GemmMachine, lowering:GemmLowering, shape:Shape, config:Config, *, calibrated:bool=True) -> Estimate:
  """Roofline with SM quantization: max of the tensor-core, DRAM and L2->SM times, the busiest SM running
  ceil(ctas / sm_count) CTAs' share of the compute, and a grid smaller than the SM count unable to pull full bandwidth.
  When the target carries a measured ``gemm_cost_calibration`` (and ``calibrated``), ``model_us`` is the calibrated
  wave-sequenced model of gemm_calibration.py instead; the roofline fields are reported either way."""
  (tm, tn, tk, wm, wn), split, pipe = config
  n_route = shape.n_route(lowering)
  m_pad = -(-shape.m // tm) * tm
  ctas = (m_pad // tm) * (n_route // tn) * split
  cps, source = occupancy(machine, lowering, config.geometry, pipe)
  per_sm = -(-ctas // machine.sm_count)
  sm_eff = ctas / (per_sm * machine.sm_count)
  # Each SM sub-partition has its own tensor core: an SM with fewer resident warps than sub-partitions idles some.
  resident_warps = min(cps, per_sm) * wm * wn
  pipe_share = min(1.0, resident_warps / machine.sm_subpartitions)
  compute_us = 2 * m_pad * n_route * shape.k / (machine.matrix_tflops * 1e6) / sm_eff / pipe_share
  # Fewer CTAs than SMs cannot pull the whole DRAM bandwidth (each SM has its own load path).
  memory_us = _memory_bytes(lowering, shape, m_pad, n_route, split) / (machine.dram_gbs * 1e3) / min(1.0, ctas / machine.sm_count)
  # Every CTA streams its A and B K-slices through L2 once: small tiles re-read operands many times.
  l2_bytes_moved = ctas * (tm + tn) * (-(-shape.k // split)) * lowering.operand_bytes
  l2_us = l2_bytes_moved / (machine.l2_gbs * 1e3) / min(1.0, ctas / machine.sm_count)
  waves = ctas / (machine.sm_count * max(cps, 1))
  wave_eff = waves / math.ceil(waves)
  bound = max((("compute", compute_us), ("memory", memory_us), ("l2", l2_us)), key=lambda kv: kv[1])[0]
  if sm_eff < 0.9: limiter = "waves" if ctas >= machine.sm_count else "grid<sms"
  elif cps <= 1 and bound != "compute": limiter = f"occupancy:{source}"
  else: limiter = bound
  model_us = max(compute_us, memory_us, l2_us)
  if calibrated and machine.cost_calibration is not None and machine.sm_stream_gbs is not None:
    from boltbeam.search.gemm_calibration import calibrated_us, cost_terms
    # a fitted sum of terms may undershoot the config's own roofline; the roofline is a hard floor
    model_us = max(model_us, calibrated_us(machine.cost_calibration, cost_terms(machine, lowering, shape, config)))
  return Estimate(round(model_us, 3), round(compute_us, 3), round(memory_us, 3), round(l2_us, 3), ctas, cps, source,
                  round(waves, 3), round(wave_eff, 3), lds_bytes(lowering, config.geometry, pipe),
                  register_demand(lowering, config.geometry) + REGISTER_OVERHEAD_ESTIMATE, bound, limiter)


def roofline_us(machine:GemmMachine, lowering:GemmLowering, shape:Shape) -> float:
  """The logical op's floor: max(2MNK / matrix rate, bytes(W + X + Y) / bandwidth), unpadded, output in operand dtype."""
  ob = lowering.operand_bytes
  return max(2 * shape.m * shape.n * shape.k / (machine.matrix_tflops * 1e6),
             (shape.n * shape.k + shape.m * shape.k + shape.m * shape.n) * ob / (machine.dram_gbs * 1e3))


# The model is a floor without the SM-internal limits (tensor-pipe issue from few warps, the per-K-tile barrier and
# fragment-load latency) -- those need facts not yet measured.  Measured winners sit up to 5.3x above the model's best
# (attn_kv at 16 rows: a 25 us kernel against a 4 us floor) and every reference (cuBLAS) pick the lowering can express
# within 2x (tests pin both), so the prune removes only configs the model puts beyond that.
PRUNE_RATIO = 6.0


def prune(machine:GemmMachine, lowering:GemmLowering, shape:Shape, configs:Iterable[Config], ratio:float=PRUNE_RATIO) -> list[Config]:
  """Configs within ``ratio`` of the model's best, model-best first.  Ties (the model does not price the LDS mechanism,
  K-tile depth or stage count) go to the pipeline using more of the lowering's mechanisms (async copy, matrix fragments,
  swizzle: each removes issue pressure the audit measured), then more warps, then the deeper K tile (fewer barriers
  per K), then fewer stages (less LDS)."""
  scored = [(estimate(machine, lowering, shape, c).model_us, -(c.pipeline[1] + c.pipeline[2] + c.pipeline[3]),
             -c.geometry[3] * c.geometry[4], -c.geometry[2], c.pipeline[0], c) for c in configs]
  if not scored: return []
  best = min(row[0] for row in scored)
  return [row[-1] for row in sorted(scored, key=lambda row: row[:5]) if row[0] <= best * ratio]


class PruneViolation(ValueError):
  """The cost model pruned a config that must survive (a promoted winner or a reference pick)."""


def check_prune_keeps(kept:Iterable[Config], required:Iterable[Config]) -> None:
  kept = set(kept)
  lost = [c for c in required if c not in kept]
  if lost: raise PruneViolation(f"cost model pruned required configs: {[config_json(c) for c in lost]}")


def deferred(machine:GemmMachine, lowering:GemmLowering, shape:Shape) -> list[dict[str, Any]]:
  """Families the GPU's facts admit that this lowering cannot emit, with what they would add."""
  out = []
  if not lowering.ragged_split_k:
    base = set(derive_space(machine, lowering, shape))
    extra = [c for c in derive_space(machine, _replace(lowering, ragged_split_k=True), shape) if c not in base]
    if extra: out.append({"family": "ragged_k", "requires": "K slices that are not whole K tiles (predicated tail)",
                          "configs": len(extra), "splits": sorted({c.split_k for c in extra})})
  if not lowering.stream_k:
    proposals = stream_k_proposals(machine, lowering, shape)
    if proposals:
      out.append({"family": "stream_k", "requires": "the tile pipeline bound to a persistent Stream-K grid "
                  "(tinygrad codegen/opt/stream_k.py StreamKSchedule + persistent_accumulator, today used by the generated "
                  "Q4/Q6 routes only)", "proposals": proposals[:4]})
  return out


# A grid whose busiest SM runs more than this share of idle slots is a Stream-K candidate.
STREAM_K_MIN_GAIN = 0.05


def stream_k_proposals(machine:GemmMachine, lowering:GemmLowering, shape:Shape, configs:Iterable[Config] | None=None,
                       top:int=16) -> list[dict[str, Any]]:
  """Family-agnostic: for the model's best distinct tile grids, a persistent Stream-K grid of one owner per resident CTA
  slot (``sm_count * ctas_per_sm``) splitting the (tile, K-block) work evenly, whenever that beats the tile grid's own
  SM quantization.  Each proposal carries the StreamKSchedule parameters and whether this lowering can emit it."""
  if configs is None: configs = prune(machine, lowering, shape, derive_space(machine, lowering, shape))
  out, seen = [], set()
  for config in configs:
    (tm, tn, tk, wm, wn), split, pipe = config
    if split != 1 or (tm, tn, tk) in seen: continue
    n_route = shape.n_route(lowering)
    if shape.m % tm or n_route % tn or shape.k % tk: continue
    seen.add((tm, tn, tk))
    e = estimate(machine, lowering, shape, config)
    tiles, k_blocks = (shape.m // tm) * (n_route // tn), shape.k // tk
    owners = min(machine.sm_count * e.ctas_per_sm, tiles * k_blocks)
    work = tiles * k_blocks
    per_sm = -(-owners // machine.sm_count)
    stream_eff = (work / owners) / -(-work // owners) * owners / (per_sm * machine.sm_count)
    tile_eff = e.ctas / (-(-e.ctas // machine.sm_count) * machine.sm_count)
    if stream_eff - tile_eff >= STREAM_K_MIN_GAIN:
      out.append({"config": config_json(config), "tiles": tiles, "tile_grid_sm_efficiency": round(tile_eff, 3),
                  "schedule": {"m": shape.m, "n": n_route, "k": shape.k, "tile_m": tm, "tile_n": tn, "tile_k": tk, "owners": owners},
                  "stream_k_sm_efficiency": round(stream_eff, 3), "emittable": lowering.stream_k})
    if len(seen) >= top: break
  return out


def _replace(lowering:GemmLowering, **changes) -> GemmLowering:
  return GemmLowering(**{**{f: getattr(lowering, f) for f in lowering.__dataclass_fields__}, **changes})


def config_json(config:Config) -> dict[str, Any]:
  return {"geometry": list(config.geometry), "split_k": config.split_k, "pipeline": list(config.pipeline)}


def config_from_row(row:Mapping[str, Any]) -> Config:
  """A tinygrad selection/search row -> Config (a row without "pipeline" is the register-staged two-buffer schedule)."""
  return Config(tuple(row["geometry"]), int(row["split_k"]), tuple(row.get("pipeline", SYNC2)))


# ---- reference (cuBLAS / CUTLASS) picks ---------------------------------------------------------------------------

_CUTLASS = re.compile(r"cutlass_\d+_(?P<wmma>wmma_)?tensorop_\w+?gemm(?:_relu)?_\w+?_(?P<nout>\d+)x(?P<mtok>\d+)_(?P<k>\d+)x(?P<stages>\d+)_tn")


def parse_cutlass_kernel(name:str, grid:Iterable[int], block:Iterable[int], warp_size:int=32) -> dict[str, Any] | None:
  """CUTLASS kernel name + launch -> the strategy it encodes, in this module's (M = tokens, N = outputs) orientation.

  cuBLAS is column-major, so a ``128x64_64x3`` kernel is 128 output rows x 64 tokens, K tile 64, 3 stages; the split-K
  depth is the grid's z.  Returns None for a kernel that is not a CUTLASS tensor-op GEMM (e.g. splitKreduce)."""
  match = _CUTLASS.search(name)
  if match is None: return None
  grid, block = list(grid), list(block)
  return {"tile_m": int(match["mtok"]), "tile_n": int(match["nout"]), "tile_k": int(match["k"]), "stages": int(match["stages"]),
          "warps": block[0] * block[1] * block[2] // warp_size, "split_k": int(grid[2]), "wmma": bool(match["wmma"])}


def reference_admitted(space:Iterable[Config], pick:Mapping[str, Any]) -> list[Config]:
  """The configs of ``space`` that realize a reference pick (same tile, stage count, warp count and split-K)."""
  return [c for c in space if c.geometry[:3] == (pick["tile_m"], pick["tile_n"], pick["tile_k"])
          and c.geometry[3] * c.geometry[4] == pick["warps"] and c.split_k == pick["split_k"] and c.pipeline[0] == pick["stages"]]


def explain_exclusion(machine:GemmMachine, lowering:GemmLowering, shape:Shape, pick:Mapping[str, Any]) -> str | None:
  """None when the derived space realizes the pick; else the missing lowering capability that excludes it:
  ``ragged_k`` (a K slice that is not a whole number of K tiles, split or not), ``n_pad_to_tile`` (the route pads
  weights to ``weight_row_granule`` rows, which the tile's N does not divide), their combination,
  ``single_stage`` (an unpipelined single LDS buffer, not a searched family), or ``not_derived``."""
  if reference_admitted(derive_space(machine, lowering, shape), pick): return None
  if pick["stages"] < 2: return "single_stage"
  pad = {"weight_row_granule": math.lcm(lowering.weight_row_granule, pick["tile_n"])}
  for reason, changes in (("ragged_k", {"ragged_split_k": True}), ("n_pad_to_tile", pad),
                          ("ragged_k+n_pad_to_tile", {"ragged_split_k": True, **pad})):
    if reference_admitted(derive_space(machine, _replace(lowering, **changes), shape), pick): return reason
  return "not_derived"


def space_document(machine:GemmMachine, lowering:GemmLowering, shape:Shape, *, ratio:float=PRUNE_RATIO, top:int=8) -> dict[str, Any]:
  """One shape's derived space as JSON: counts, the model's top candidates with their limiting factor, deferred families."""
  space = derive_space(machine, lowering, shape)
  kept = prune(machine, lowering, shape, space, ratio)
  return {"schema": SCHEMA_SPACE, "target_id": machine.target_id, "shape": asdict(shape), "n_route": shape.n_route(lowering),
          "admitted": len(space), "kept": len(kept), "prune_ratio": ratio,
          "roofline_us": round(roofline_us(machine, lowering, shape), 3),
          "top": [{**config_json(c), **asdict(estimate(machine, lowering, shape, c))} for c in kept[:top]],
          "deferred": deferred(machine, lowering, shape)}


__all__ = ["PruneViolation", "check_prune_keeps", "Config", "Estimate", "GemmLowering", "GemmMachine", "PRUNE_RATIO", "SCHEMA_LOWERING", "SCHEMA_SPACE", "SYNC2",
           "Shape", "config_from_row", "config_json", "deferred", "derive_space", "estimate", "explain_exclusion",
           "lds_bytes", "occupancy", "parse_cutlass_kernel", "prune", "reference_admitted", "register_demand",
           "roofline_us", "space_document", "stream_k_proposals"]
