"""KA-6B: LDS/cache staging plugin.

A prediction is never a verdict. This plugin reconciles a per-strategy LDS-vs-cache *prediction*
against *measured* A/B evidence: when measurement exists it wins and any disagreement with the
prediction is retained as calibration evidence; without measurement the prediction stays a prediction
with its own (never "measured") truth status; with neither, the decision is blocked. Physical cache
residency is a separate observed axis — it is never inferred from a candidate's label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boltbeam.kernel_analysis.contrast import PairContrast
from boltbeam.kernel_analysis.model import OperandTransport

# All four register/LDS A/B combinations, representable with the existing model (no new harness schema).
PLACEMENT_MATRIX: tuple[tuple[tuple[str, OperandTransport], ...], ...] = (
  (("a", OperandTransport(register_resident=True)), ("b", OperandTransport(register_resident=True))),
  (("a", OperandTransport(lds_staged=True)), ("b", OperandTransport(lds_staged=True))),
  (("a", OperandTransport(register_resident=True)), ("b", OperandTransport(lds_staged=True))),
  (("a", OperandTransport(lds_staged=True)), ("b", OperandTransport(register_resident=True))),
)


@dataclass(frozen=True)
class Prediction:
  """A modeled LDS-vs-cache prediction. `winner` is a strategy label (e.g. 'lds' or 'register'),
  never a candidate name. Intervals are optional [low, high] ms bands per strategy."""
  winner: str
  truth_status: str = "modeled"
  interval_lds_ms: tuple[float, float] | None = None
  interval_register_ms: tuple[float, float] | None = None
  intervals_ms: dict[str, tuple[float, float]] | None = None
  assumptions: tuple[str, ...] = ()

  def __post_init__(self):
    if self.truth_status == "measured":
      raise ValueError("a prediction may not carry truth_status 'measured'")
    if self.intervals_ms is not None and not isinstance(self.intervals_ms, dict):
      raise ValueError("intervals_ms must map strategy names to intervals")

  def to_json(self) -> dict[str, Any]:
    out = {"winner": self.winner, "truth_status": self.truth_status,
           "assumptions": list(self.assumptions)}
    if self.interval_lds_ms is not None:
      out["interval_lds_ms"] = list(self.interval_lds_ms)
    if self.interval_register_ms is not None:
      out["interval_register_ms"] = list(self.interval_register_ms)
    if self.intervals_ms is not None:
      out["intervals_ms"] = {k: list(v) for k, v in sorted(self.intervals_ms.items())}
    return out


def _measured_winner(contrast: PairContrast) -> str | None:
  """Map a measured contrast to the winning *strategy* using the declared transport change,
  never a candidate label. Returns None when timing is not comparison-grade."""
  if not contrast.performance_eligible:
    return None
  qualified = contrast.timing.get("qualified")
  if qualified not in ("win", "regression"):
    return None
  mech = contrast.mechanism
  if mech.get("status") != "present":
    return None
  # winner is whichever side (candidate/comparator) was faster; describe by its transport.
  faster = "candidate" if qualified == "win" else "comparator"
  changed = mech.get("changed_operands", [])
  strategies = {}
  for operand in changed:
    flags = mech.get(faster, {}).get(operand, {}) or {}
    strategies[operand] = next((name for name, enabled in flags.items() if enabled), "unknown")
  return f"{faster}:" + ",".join(f"{op}={strategies[op]}" for op in sorted(strategies))


def reconcile(prediction: Prediction | None = None,
              contrast: PairContrast | None = None) -> dict[str, Any]:
  """Reconcile a prediction against measured A/B evidence. Measured always overrides; disagreement
  is calibration evidence; a prediction alone stays a prediction; neither is blocked."""
  measured = None
  if contrast is not None and contrast.performance_eligible \
      and contrast.timing.get("qualified") in ("win", "regression"):
    measured = {
      "qualified": contrast.timing["qualified"],
      "speedup": contrast.timing.get("speedup"),
      "delta_ms": contrast.timing.get("delta_ms"),
    }

  if measured is not None:
    winner = _measured_winner(contrast)
    out = {"decision": "measured", "truth_status": "measured", "measured": measured,
           "measured_winner": winner,
           "selection_confidence": "high",
           "mechanism_confidence": "unknown" if winner is None else "low"}
    if prediction is not None:
      # Did the prediction call the winning direction? Compare predicted winner strategy to observed.
      predicted_faster_is_candidate = _prediction_favors_candidate(prediction, contrast)
      observed_candidate_faster = contrast.timing["qualified"] == "win"
      agreed = predicted_faster_is_candidate is not None \
          and predicted_faster_is_candidate == observed_candidate_faster
      out["prediction"] = prediction.to_json()
      out["calibration"] = {
        "prediction_confirmed": bool(agreed),
        "prediction_disagreed": predicted_faster_is_candidate is not None and not agreed,
      }
    return out

  if prediction is not None:
    # No measurement -> prediction stays a prediction (truth status preserved, never 'measured').
    return {"decision": "predicted", "truth_status": prediction.truth_status,
            "selection_confidence": "unknown", "mechanism_confidence": "modeled",
            "prediction": prediction.to_json(),
            "note": "prediction only; a measured A/B is required before any verdict"}

  return {"decision": "blocked", "truth_status": "unknown",
          "selection_confidence": "unknown", "mechanism_confidence": "unknown",
          "missing": ["a prediction (memory-hierarchy profile) or measured A/B evidence"]}


def _prediction_favors_candidate(prediction: Prediction, contrast: PairContrast) -> bool | None:
  """Whether the prediction's winning strategy matches the candidate's declared transport.
  Returns None when it cannot be determined without guessing."""
  mech = contrast.mechanism
  if mech.get("status") != "present":
    return None
  changed = mech.get("changed_operands", [])
  if not changed:
    return None
  cand_transport = mech.get("candidate", {}).get(changed[0], {})
  key = {"lds": "lds_staged", "register": "register_resident", "cache": "cache_streamed",
         "reload": "reloaded"}.get(prediction.winner, prediction.winner)
  if key is None:
    return None
  return bool(cand_transport.get(key))


def observed_residency(evidence_cache_tiers: dict[str, Any] | None) -> dict[str, Any]:
  """Report physical L0/L1/L2/MALL/DRAM residency strictly as observed evidence. Absent input yields
  an explicit unknown — residency is never inferred from a candidate label such as 'direct_l2'."""
  if not evidence_cache_tiers:
    return {"residency": "unknown", "truth_status": "unknown",
            "note": "no measured cache-residency evidence; not inferred from any candidate label"}
  rows = list(evidence_cache_tiers.values()) if all(isinstance(v, dict) for v in evidence_cache_tiers.values()) else []
  statuses = {str(row.get("status") or row.get("truth_status") or "unknown") for row in rows}
  truth_status = "measured" if rows and statuses == {"measured"} else "unknown"
  out = {"residency": dict(evidence_cache_tiers), "truth_status": truth_status}
  if truth_status != "measured":
    out["note"] = "tier data is not uniformly measured; modeled, proxy, unsupported, or untyped values are not residency proof"
  return out
