"""Pure roofline / Amdahl math for BoltBeam ceiling reports.

No model constants are baked in here: every byte count, bandwidth, wall share, and speedup is passed by the
caller. This module answers "what is the theoretical floor?" and "how much whole-model gain does optimizing one
role buy?" without touching a model, a target, or a GPU.
"""
from __future__ import annotations

from boltbeam.math.roofline import (
  floor_ms_per_token,
  ceiling_tok_s,
  amdahl_whole_gain,
  CeilingReport,
  build_ceiling_report,
  PracticalRooflinePoint,
  PracticalRooflineResult,
  practical_roofline_score,
  dual_roofline_placement,
)

__all__ = [
  "floor_ms_per_token",
  "ceiling_tok_s",
  "amdahl_whole_gain",
  "CeilingReport",
  "build_ceiling_report",
  "PracticalRooflinePoint",
  "PracticalRooflineResult",
  "practical_roofline_score",
  "dual_roofline_placement",
]
