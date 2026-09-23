"""Failure-isolated controller for dynamic full-kernel candidate admission."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from boltbeam.search.spec import FullKernelCandidate
from boltbeam.search.full_kernel.tinygrad_full_kernel import TinygradAdmissionResult


SCHEMA = "boltbeam.full_kernel_admission_report.v1"
TERMINAL_STATES = frozenset(("ADMITTED", "REJECTED_INVALID", "REJECTED_UNSUPPORTED", "WORKER_ERROR"))


@dataclass(frozen=True)
class FullKernelAdmissionRun:
  candidate_hash: str
  state: str
  history: tuple[str, ...]
  result: TinygradAdmissionResult | None = None
  error: str = ""

  def __post_init__(self):
    if self.state not in TERMINAL_STATES: raise ValueError(f"unknown full-kernel admission state {self.state!r}")
    if not self.history or self.history[0] != "PLANNED" or self.history[-1] != self.state:
      raise ValueError("full-kernel admission history must run from PLANNED to terminal state")

  def to_dict(self) -> dict[str, Any]:
    return {"candidate_hash": self.candidate_hash, "state": self.state, "history": list(self.history),
            "error": self.error or None, "result": self.result.to_dict() if self.result is not None else None}


@dataclass(frozen=True)
class FullKernelAdmissionReport:
  runs: tuple[FullKernelAdmissionRun, ...]

  def __post_init__(self):
    hashes = [run.candidate_hash for run in self.runs]
    if len(hashes) != len(set(hashes)): raise ValueError("full-kernel admission report contains duplicate candidate hashes")

  def by_hash(self) -> dict[str, FullKernelAdmissionRun]: return {run.candidate_hash: run for run in self.runs}

  def to_dict(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "runs": {run.candidate_hash: run.to_dict() for run in self.runs}}


def run_full_kernel_admission(candidates: Iterable[FullKernelCandidate],
                              adapter: Callable[[FullKernelCandidate], TinygradAdmissionResult]) -> FullKernelAdmissionReport:
  """Admit every distinct candidate, continuing after rejection or worker failure."""
  runs: list[FullKernelAdmissionRun] = []
  seen: set[str] = set()
  for candidate in candidates:
    if not isinstance(candidate, FullKernelCandidate): raise TypeError("full-kernel admission requires FullKernelCandidate values")
    identity = candidate.candidate_hash
    if identity in seen: continue
    seen.add(identity)
    try: result = adapter(candidate)
    except Exception as exc:  # worker adapters are an isolation boundary
      runs.append(_run(identity, "WORKER_ERROR", error=f"{type(exc).__name__}: {exc}"))
      continue
    if not isinstance(result, TinygradAdmissionResult):
      runs.append(_run(identity, "WORKER_ERROR", error="adapter returned a non-TinygradAdmissionResult value"))
      continue
    if result.candidate_hash != identity:
      runs.append(_run(identity, "WORKER_ERROR", result=result,
                       error=f"adapter candidate hash mismatch: expected {identity}, got {result.candidate_hash}"))
      continue
    if result.admitted:
      runs.append(_run(identity, "ADMITTED", result=result))
    elif result.status == "rejected" and result.classification == "candidate_invalid":
      runs.append(_run(identity, "REJECTED_INVALID", result=result))
    elif result.status == "rejected" and result.classification == "compiler_unsupported":
      runs.append(_run(identity, "REJECTED_UNSUPPORTED", result=result))
    else:
      detail = result.error.detail if result.error is not None else \
        f"unexpected adapter outcome status={result.status!r} classification={result.classification!r}"
      runs.append(_run(identity, "WORKER_ERROR", result=result, error=detail))
  return FullKernelAdmissionReport(tuple(runs))


def _run(identity: str, state: str, *, result: TinygradAdmissionResult | None = None,
         error: str = "") -> FullKernelAdmissionRun:
  return FullKernelAdmissionRun(identity, state, ("PLANNED", state), result, error)


__all__ = ["SCHEMA", "TERMINAL_STATES", "FullKernelAdmissionReport", "FullKernelAdmissionRun",
           "run_full_kernel_admission"]
