"""Compiler pathology classifier (compiler-pathology-diagnostics-scope.md CP2).

Input:  list of EvidenceRow(kind=RowKind.COMPILER_PATHOLOGY) for one candidate.
Output: PathologyReport with classification, confidence, reason, and reopen directive.

Rules are driven by per-metric thresholds defined in THRESHOLDS. Add new classes by
editing THRESHOLDS + _CLASSIFY_RULES — not by scattering literals elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any

from boltbeam.vocab import PathologyClass, RowKind

SCHEMA = "boltbeam.compiler_pathology_report.v1"

# Default thresholds (override via classify_pathology(thresholds=...)).
THRESHOLDS: dict[str, Any] = {
  "time_ratio_huge": 10.0,      # ratio above this → suspect pathology
  "scratch_bytes_spill": 1,     # any scratch → REGALLOC_SPILL
  "barrier_static_high": 16,    # absolute barrier count cap before BARRIER_EXPLOSION
  "barrier_ratio_high": 4.0,    # barrier_count / expected_barrier_count ratio cap
  "instruction_ratio_high": 10.0,  # static_instr / valu_instr ratio cap for LOOP_LOWERING_BAD
  "waitcnt_ratio_high": 4.0,    # waitcnt_count / barrier_count ratio cap
  "lds_ratio_high": 8.0,        # lds_bytes / expected_lds_bytes ratio cap for LDS_OR_MEMORY_OVERHEAD
  "time_ratio_lds_fallback": 10.0,  # slowdown threshold for LDS fallback when lds_bytes absent
}


@dataclass
class PathologyReport:
  schema: str
  candidate_id: str
  classification: str           # PathologyClass value
  confidence: str               # "measured" | "timing_only" | "unknown"
  reason: str
  trigger_rows: list[dict]
  next_action: str
  reopen_directive: str

  def to_json(self) -> dict:
    return asdict(self)

  def to_json_str(self, indent: int = 2) -> str:
    return json.dumps(self.to_json(), indent=indent, sort_keys=True)


def _first_value(rows: list, metric: str) -> float | None:
  for r in rows:
    if r.metric == metric:
      return float(r.value)
  return None


def classify_pathology(
  candidate_id: str,
  rows: list,
  *,
  thresholds: dict[str, Any] | None = None,
) -> PathologyReport:
  """Classify compiler pathology from normalized compiler_pathology evidence rows.

  Returns NATIVE_ISA_ORACLE_NEEDED when only timing rows are present (no resource signal).
  """
  t = {**THRESHOLDS, **(thresholds or {})}

  cp_rows = [r for r in rows if r.kind == RowKind.COMPILER_PATHOLOGY]

  time_ratio   = _first_value(cp_rows, "time_ratio_vs_baseline")
  scratch      = _first_value(cp_rows, "scratch_bytes")
  barriers     = _first_value(cp_rows, "barrier_count")
  exp_barr     = _first_value(cp_rows, "expected_barrier_count")
  static_i     = _first_value(cp_rows, "static_instr")
  valu_i       = _first_value(cp_rows, "valu_instr")
  waitcnt      = _first_value(cp_rows, "waitcnt_count")
  vec_bits     = _first_value(cp_rows, "vector_load_bits")
  exp_vec      = _first_value(cp_rows, "expected_vector_load_bits")
  lds_bytes    = _first_value(cp_rows, "lds_bytes")
  exp_lds      = _first_value(cp_rows, "expected_lds_bytes")
  candidate_us = _first_value(cp_rows, "time_us_per_workgroup")
  baseline_us  = _first_value(cp_rows, "baseline_time_us_per_workgroup")

  has_resource_rows = any(v is not None for v in [scratch, barriers, static_i, waitcnt, lds_bytes])

  def _report(cls: PathologyClass, reason: str, action: str, reopen: str,
              trigger_metrics: list[str]) -> PathologyReport:
    triggers = [r.__dict__ if hasattr(r, '__dict__') else dict(r._asdict())
                for r in cp_rows if r.metric in trigger_metrics]
    return PathologyReport(
      schema=SCHEMA,
      candidate_id=candidate_id,
      classification=cls.value,
      confidence="measured" if has_resource_rows else "timing_only",
      reason=reason,
      trigger_rows=triggers,
      next_action=action,
      reopen_directive=reopen,
    )

  # No resource rows at all — only timing is available.
  if not has_resource_rows:
    ratio_str = f"~{time_ratio:.0f}×" if time_ratio else "unknown"
    candidate_str = f"{candidate_us:.0f}µs" if candidate_us else "unknown"
    baseline_str  = f"{baseline_us:.0f}µs" if baseline_us else "unknown"
    return _report(
      PathologyClass.NATIVE_ISA_ORACLE_NEEDED,
      f"Per-workgroup slowdown is {ratio_str} ({candidate_str} vs baseline {baseline_str}), "
      f"but no scratch/barrier/instruction-count rows are present to identify the specific pathology.",
      "Produce tinygrad.compiler_pathology.v1 artifact with scratch_bytes, barrier_count, "
      "static_instr, valu_instr for the candidate kernel.",
      f"reopen when tinygrad.compiler_pathology.v1 artifact is ingested for candidate {candidate_id!r}",
      ["time_us_per_workgroup", "time_ratio_vs_baseline"],
    )

  # Register spill: any scratch bytes.
  if scratch is not None and scratch >= t["scratch_bytes_spill"]:
    return _report(
      PathologyClass.REGALLOC_SPILL,
      f"scratch_bytes={scratch:.0f} — kernel spills registers to global memory.",
      "Reduce register pressure: use smaller tile size, fewer live values, or native ISA path.",
      "reopen when scratch_bytes=0 is measured on a modified tiling or native ISA implementation",
      ["scratch_bytes"],
    )

  # Barrier explosion.
  if barriers is not None:
    if exp_barr is not None and barriers > t["barrier_ratio_high"] * exp_barr:
      return _report(
        PathologyClass.BARRIER_EXPLOSION,
        f"barrier_count={barriers:.0f} is {barriers/exp_barr:.1f}× expected={exp_barr:.0f}.",
        "Inspect generated loop/barrier structure; remove redundant barriers in emitter.",
        "reopen when barrier_count ≤ expected_barrier_count × 2",
        ["barrier_count", "expected_barrier_count"],
      )
    if barriers > t["barrier_static_high"]:
      return _report(
        PathologyClass.BARRIER_EXPLOSION,
        f"barrier_count={barriers:.0f} exceeds absolute threshold {t['barrier_static_high']}.",
        "Inspect generated loop/barrier structure; remove redundant barriers in emitter.",
        f"reopen when barrier_count ≤ {t['barrier_static_high']}",
        ["barrier_count"],
      )

  # Instruction bloat / loop lowering.
  if static_i is not None and valu_i is not None and valu_i > 0:
    ratio = static_i / valu_i
    if ratio > t["instruction_ratio_high"]:
      return _report(
        PathologyClass.LOOP_LOWERING_BAD,
        f"static_instr={static_i:.0f} / valu_instr={valu_i:.0f} = {ratio:.1f}× "
        f"(threshold {t['instruction_ratio_high']}×).",
        "Inspect loop lowering and unrolling in tinygrad emitter.",
        f"reopen when static_instr / valu_instr < {t['instruction_ratio_high']}",
        ["static_instr", "valu_instr"],
      )

  # Vector load lost.
  if vec_bits is not None and exp_vec is not None and vec_bits < exp_vec:
    return _report(
      PathologyClass.VECTOR_LOAD_LOST,
      f"vector_load_bits={vec_bits:.0f} < expected={exp_vec:.0f}.",
      "Wire vectorized load lowering for the relevant dtype/alignment in the AMD renderer.",
      f"reopen when vector_load_bits ≥ {exp_vec:.0f}",
      ["vector_load_bits", "expected_vector_load_bits"],
    )

  # Waitcnt anomaly.
  if waitcnt is not None and barriers is not None and barriers > 0:
    wc_ratio = waitcnt / barriers
    if wc_ratio > t["waitcnt_ratio_high"]:
      return _report(
        PathologyClass.WAITCNT_BAD,
        f"waitcnt_count={waitcnt:.0f} / barrier_count={barriers:.0f} = {wc_ratio:.1f}× "
        f"(threshold {t['waitcnt_ratio_high']}×).",
        "Inspect waitcnt scheduler; reduce unnecessary memory fence overhead.",
        f"reopen when waitcnt/barrier ratio < {t['waitcnt_ratio_high']}",
        ["waitcnt_count", "barrier_count"],
      )

  # LDS / memory traffic overhead: scratch=0 (no spill) but LDS footprint >> baseline,
  # or high slowdown with no other specific class matched (K+V double-staging pattern).
  if scratch is not None and scratch == 0:
    if lds_bytes is not None and exp_lds is not None and exp_lds > 0:
      lds_ratio = lds_bytes / exp_lds
      if lds_ratio > t["lds_ratio_high"]:
        return _report(
          PathologyClass.LDS_OR_MEMORY_OVERHEAD,
          f"lds_bytes={lds_bytes:.0f} is {lds_ratio:.0f}× expected={exp_lds:.0f} with scratch_bytes=0; "
          f"LDS/VMEM traffic dominates without register spill.",
          "Reduce LDS footprint: stage K-only (not K+V), use smaller tile, or use native ISA path.",
          f"reopen when lds_bytes / expected_lds_bytes < {t['lds_ratio_high']:.0f} "
          f"OR when native ISA implementation achieves target throughput",
          ["lds_bytes", "expected_lds_bytes", "scratch_bytes"],
        )
    if time_ratio is not None and time_ratio > t["time_ratio_lds_fallback"]:
      return _report(
        PathologyClass.LDS_OR_MEMORY_OVERHEAD,
        f"work-adjusted slowdown {time_ratio:.1f}× with scratch_bytes=0 and no other pathology class; "
        f"likely LDS/VMEM bandwidth overhead (e.g. staging both K and V increases LDS traffic vs K-only baseline).",
        "Measure lds_bytes and expected_lds_bytes to confirm; if confirmed, stage K-only into LDS "
        "or use native ISA path to eliminate the overhead.",
        "reopen when lds_bytes row is present and confirms or refutes LDS_OR_MEMORY_OVERHEAD, "
        "or when native ISA implementation achieves target throughput",
        ["time_ratio_vs_baseline", "scratch_bytes"],
      )

  # Resource rows present but no class matched.
  return _report(
    PathologyClass.UNKNOWN,
    f"Resource rows present but no pathology class matched. "
    f"time_ratio={time_ratio}, scratch={scratch}, barriers={barriers}, "
    f"static_instr={static_i}, valu_instr={valu_i}.",
    "Manual investigation required.",
    "reopen after manual disassembly review identifies the bottleneck",
    ["time_us_per_workgroup", "scratch_bytes", "barrier_count"],
  )
