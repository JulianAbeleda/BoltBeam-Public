"""One role's proposal facts: the correctness tolerance and the one column count the decode workload states. Its
semantic workload and the proposal pipeline left with the per-role search (retired 2026-11-21).
"""
from __future__ import annotations


def tolerances() -> dict[str, float]:
  """The provider's own correctness filter against its CPU reference (its population's MEASURED rows): screening only.
  BoltBeam decides on its own read: each candidate's output against the model's own kernel, bit for bit
  (provider_check.held, search/identity), never this tolerance."""
  from boltbeam.math.numerics import TOLERANCE
  return {"atol": TOLERANCE, "rtol": TOLERANCE}


LOGICAL_M = 1  # the one column count (tokens a forward) the decode workload states: a one-token GEMV; readers ask this, not the workload
