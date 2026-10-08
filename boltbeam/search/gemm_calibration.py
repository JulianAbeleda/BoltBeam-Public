"""Measured calibration of the tile-GEMM cost model.

``gemm_strategy.estimate``'s roofline (max of tensor-core, DRAM and L2 time over SM quantization) orders candidates
poorly inside one shape: on 3884 measured cp.async-ring candidates over 35 Nemotron-H shapes (RTX 5090, 2026-09-26)
its median within-shape Spearman is 0.10 and its top-1 pick is 1.27x the measured best on average.  It misses what
decides a memory-bound decode GEMM: waves run one after another, a tail wave's few CTAs stream only at one SM's rate,
every K tile pays a wait+barrier, and a split-K product pays a separate reduction.

This module computes those physical terms from the GPU facts (``GemmMachine``, incl. the measured per-SM streaming
rate) and the lowering facts, and fits one non-negative coefficient per term to measured kernel times (relative error,
non-negative least squares).  ``estimate`` uses the fitted coefficients when the target carries a
``gemm_cost_calibration`` fact.  The model still only orders and prunes; selection is by measurement.
"""
from __future__ import annotations

import math, statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

TERMS = ("mem_us", "mma_issue_kslots", "iter_sync", "waves", "reduce_us", "reduce_launch", "latency_iters")
SCHEMA = "boltbeam.gemm_cost_calibration.v1"


def cost_terms(machine, lowering, shape, config) -> list[float]:
  """The model's physical terms for one candidate, in ``TERMS`` order (units: us, or counts the coefficient scales)."""
  from boltbeam.search.gemm_strategy import occupancy
  (tm, tn, tk, wm, wn), split, pipe = config
  cps = max(occupancy(machine, lowering, (tm, tn, tk, wm, wn), pipe)[0], 1)
  n_route = shape.n_route(lowering)
  m_pad = -(-shape.m // tm) * tm
  ctas = (m_pad // tm) * (n_route // tn) * split
  iters = -(-shape.k // split) // tk
  warps = wm * wn
  sub_m, sub_n = tm // wm // lowering.mma_m, tn // wn // lowering.mma_n
  k_atoms = tk // lowering.mma_k
  mma = sub_m * sub_n * k_atoms                                   # tensor-core atoms per warp per K tile
  # fragment loads per warp per K tile: one native matrix load per fragment, else one 32-bit load per register
  loads = (sub_m + sub_n) * k_atoms if pipe[2] else (sub_m * lowering.a_fragment_regs + sub_n * lowering.b_fragment_regs) * k_atoms
  tile_bytes = (tm + tn) * tk * lowering.operand_bytes
  slots = machine.sm_count * cps
  full, tail = divmod(ctas, slots)
  mem = issue = sync = 0.0
  for resident, count in ((slots, full), (tail, 1 if tail else 0)):
    if not count: continue
    per_sm = -(-resident // machine.sm_count)
    bw = min(machine.dram_gbs / resident, machine.sm_stream_gbs / per_sm) * 1e3   # bytes/us per CTA
    mem += count * iters * tile_bytes / bw
    warps_per_sub = max(1.0, per_sm * warps / machine.sm_subpartitions)
    issue += count * iters * (mma + 0.5 * loads) * warps_per_sub / 1000
    sync += count * iters * per_sm / cps
  nwaves = full + (1 if tail else 0)
  reduce_bytes = (split + 1) * m_pad * n_route * lowering.accumulator_bytes if split > 1 else 0
  latency = iters * nwaves / max(pipe[0] - 1, 1)
  return [mem, issue, sync, float(nwaves), reduce_bytes / (machine.dram_gbs * 1e3), 1.0 if split > 1 else 0.0, latency]


def calibrated_us(calibration:Mapping[str, Any], terms:Sequence[float]) -> float:
  coef = calibration["coefficients"]
  return sum(coef[name] * value for name, value in zip(TERMS, terms))


def nnls(rows:Sequence[Sequence[float]], target:Sequence[float], iters:int=20000, tol:float=1e-12) -> list[float]:
  """Non-negative least squares by projected coordinate descent on the normal equations (few variables, no deps)."""
  n = len(rows[0])
  ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
  atb = [sum(r[i] * t for r, t in zip(rows, target)) for i in range(n)]
  x = [0.0] * n
  for _ in range(iters):
    delta = 0.0
    for i in range(n):
      if ata[i][i] <= 0: continue
      g = atb[i] - sum(ata[i][j] * x[j] for j in range(n) if j != i)
      new = max(0.0, g / ata[i][i])
      delta = max(delta, abs(new - x[i])); x[i] = new
    if delta < tol: break
  return x


def fit(samples:Iterable[tuple[Sequence[float], float]]) -> dict[str, float]:
  """Coefficients minimizing sum((model - t) / t)^2 with every coefficient >= 0."""
  rows, target = [], []
  for terms, t in samples: rows.append([v / t for v in terms]); target.append(1.0)
  return dict(zip(TERMS, nnls(rows, target)))


def _spearman(a:Sequence[float], b:Sequence[float]) -> float:
  def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); out = [0] * len(xs)
    for r, i in enumerate(order): out[i] = r
    return out
  n = len(a); ra, rb = ranks(a), ranks(b)
  return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def ranking_quality(groups:Mapping[Any, Sequence[tuple[float, float]]]) -> dict[str, float]:
  """Per shape (predicted, measured) pairs -> median Spearman, top-1 regret (measured time of the model's pick over the
  measured best) mean/max, and best-of-top-5 regret mean/max."""
  sp, top1, top5 = [], [], []
  for pairs in groups.values():
    pred, meas = [p for p, _ in pairs], [t for _, t in pairs]
    order = sorted(range(len(pred)), key=lambda i: pred[i]); best = min(meas)
    sp.append(_spearman(pred, meas)); top1.append(meas[order[0]] / best); top5.append(min(meas[i] for i in order[:5]) / best)
  r = lambda xs: round(xs, 3)
  return {"shapes": len(sp), "points": sum(len(p) for p in groups.values()), "median_spearman": r(statistics.median(sp)),
          "top1_regret_mean": r(statistics.mean(top1)), "top1_regret_max": r(max(top1)),
          "top5_regret_mean": r(statistics.mean(top5)), "top5_regret_max": r(max(top5))}


def calibrate(machine, lowering, measured:Iterable[Mapping[str, Any]], *, min_points:int=8) -> dict[str, Any]:
  """Fit on every measured candidate row (role, m, n, k, geometry, split_k, pipeline, median_us) and validate by
  leaving each shape out of its own fit.  Returns a ``gemm_cost_calibration`` fact with the validation numbers,
  against the uncalibrated roofline on the same points."""
  from boltbeam.search.gemm_strategy import Config, Shape, estimate
  by: dict[tuple, dict[Config, float]] = defaultdict(dict)
  for row in measured:
    if "median_us" not in row or not row.get("finite", True) or len(row.get("pipeline") or ()) != 4: continue
    config = Config(tuple(row["geometry"]), int(row["split_k"]), tuple(row["pipeline"]))
    key = (row["m"], row["n"], row["k"])
    by[key][config] = min(row["median_us"], by[key].get(config, math.inf))
  shapes = {k: v for k, v in by.items() if len(v) >= min_points}
  terms = {(k, c): cost_terms(machine, lowering, Shape(*k), c) for k, v in shapes.items() for c in v}
  floor = {(k, c): estimate(machine, lowering, Shape(*k), c, calibrated=False).model_us for k, v in shapes.items() for c in v}
  held = {}
  for out in shapes:
    coef = fit((terms[(k, c)], t) for k, v in shapes.items() if k != out for c, t in v.items())
    held[out] = [(max(calibrated_us({"coefficients": coef}, terms[(out, c)]), floor[(out, c)]), t) for c, t in shapes[out].items()]
  roofline = {k: [(estimate(machine, lowering, Shape(*k), c, calibrated=False).model_us, t) for c, t in v.items()] for k, v in shapes.items()}
  return {"schema": SCHEMA, "kind": "fit", "terms": list(TERMS),
          "coefficients": fit((terms[(k, c)], t) for k, v in shapes.items() for c, t in v.items()),
          "validation": {"held_out_by_shape": ranking_quality(held), "uncalibrated_roofline": ranking_quality(roofline)}}


__all__ = ["TERMS", "SCHEMA", "calibrate", "calibrated_us", "cost_terms", "fit", "nnls", "ranking_quality"]
