"""Post-falsification SQ residual diagnosis; never a calibration fitter."""
from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

PMC_SCHEMA = "tinygrad.amd_pmc_result.v1"


def diagnose_sq_residual(*, gated_pmc: Mapping[str, Any], direct_pmc: Mapping[str, Any],
                         gated_wall_ms: Sequence[float], direct_wall_ms: Sequence[float],
                         predicted_ratio: float) -> dict[str, Any]:
  gated = _sq_summary(gated_pmc); direct = _sq_summary(direct_pmc)
  measured_ratio = statistics.median(gated_wall_ms) / statistics.median(direct_wall_ms)
  wave_ratio = gated["wave_cycles_median"] / direct["wave_cycles_median"]
  busy_ratio = gated["busy_cycles_median"] / direct["busy_cycles_median"]
  wall_ns_per_wave_cycle = {
    "gated": statistics.median(gated_wall_ms) * 1e6 / gated["wave_cycles_median"],
    "direct": statistics.median(direct_wall_ms) * 1e6 / direct["wave_cycles_median"],
  }
  return {"schema": "boltbeam.mmq_sq_residual.v1", "status": "diagnostic_only_post_falsification",
    "may_fit_model": False, "gated": gated, "direct": direct,
    "ratios": {"predicted": predicted_ratio, "measured_wall": measured_ratio,
               "sq_wave_cycles": wave_ratio, "sq_busy_cycles": busy_ratio},
    "wall_ns_per_sq_wave_cycle": wall_ns_per_wave_cycle,
    "classification": "static_issue_and_wave_scheduling_model_missing",
    "reason": "SQ cycle ratios track wall ordering better than the frozen static prediction, but candidate SQ is holdout evidence",
    "next_training_artifact": "independent generated-microbenchmark SQ grid-tail/resource calibration"}


def _sq_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
  if artifact.get("schema") != PMC_SCHEMA or artifact.get("kind") != "candidate_pmc": raise ValueError("expected candidate PMC")
  samples = [row.get("counters", {}) for row in artifact.get("samples", []) if row.get("status") == "live"]
  required = ("SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES")
  if len(samples) < 3 or any(any(not isinstance(row.get(key), (int, float)) or row[key] <= 0 for key in required) for row in samples):
    raise ValueError("insufficient live SQ residual samples")
  waves = statistics.median(row["SQ_WAVES"] for row in samples)
  wave_cycles = statistics.median(row["SQ_WAVE_CYCLES"] for row in samples)
  return {"candidate_id": artifact.get("candidate_id"), "binary_sha256": artifact.get("binary_sha256"),
          "sample_count": len(samples), "waves_median": waves, "wave_cycles_median": wave_cycles,
          "busy_cycles_median": statistics.median(row["SQ_BUSY_CYCLES"] for row in samples),
          "cycles_per_wave": wave_cycles / waves}


__all__ = ["diagnose_sq_residual"]
