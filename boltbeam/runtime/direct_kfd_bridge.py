"""Capability-gated direct-launch seam with a CPU-only fake backend.

The production backend is intentionally absent.  This module validates the
launch contract, exercises allocation/submit/wait/cleanup ordering, and emits a
launch observation through an injected backend without opening KFD.
"""
from __future__ import annotations

import math
import re
import time

from boltbeam.core.canonical import sha256_hex
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA = "boltbeam.kfd_launch_request.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("candidate_id", "program_id", "target_id", "driver_id", "runtime_id", "code_object",
             "binary_sha256", "argument_layout_digest", "launch_digest", "workload_digest", "schedule_digest",
             "grid", "workgroup", "allocations", "timeout_ms")


@dataclass(frozen=True)
class LaunchResult:
  candidate_id: str
  program_id: str
  observation: Mapping[str, Any] | None
  status: str
  blockers: tuple[Mapping[str, Any], ...] = ()

  @property
  def ok(self) -> bool: return self.status == "completed" and not self.blockers and self.observation is not None


class FakeKFDBackend:
  """Deterministic backend for validation and cleanup tests; never touches KFD."""

  def __init__(self, mode: str = "success"):
    self.mode, self.calls, self.allocations = mode, [], []

  def allocate(self, descriptor: Mapping[str, Any]) -> str:
    self.calls.append(("allocate", dict(descriptor)))
    if self.mode == "allocate_error": raise RuntimeError("fake allocation failure")
    token = f"alloc-{len(self.allocations)}"
    self.allocations.append(token)
    return token

  def submit(self, request: Mapping[str, Any], allocations: Sequence[Any]) -> str:
    self.calls.append(("submit", request.get("program_id"), tuple(allocations)))
    if self.mode == "submit_error": raise RuntimeError("fake submit failure")
    return "launch-1"

  def wait(self, token: Any, timeout_ms: int) -> Mapping[str, Any]:
    self.calls.append(("wait", token, timeout_ms))
    if self.mode == "timeout": return {"status": "timeout"}
    if self.mode == "invalid_abi": return {"status": "invalid_abi"}
    if self.mode == "reject_digest": return {"status": "rejected_digest"}
    return {"status": "completed", "counters": {}}

  def release(self, allocation: Any) -> None:
    self.calls.append(("release", allocation))
    if allocation in self.allocations: self.allocations.remove(allocation)


def validate_launch_request(request: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
  if not isinstance(request, Mapping): return ({"field": "request", "reason": "expected mapping"},)
  errors: list[dict[str, Any]] = []
  if request.get("schema") != SCHEMA: errors.append({"field": "schema", "reason": f"expected {SCHEMA}"})
  for field in _REQUIRED:
    if field not in request: errors.append({"field": field, "reason": "required"})
  code_object = request.get("code_object")
  if not isinstance(code_object, bytes) or not code_object: errors.append({"field": "code_object", "reason": "expected non-empty bytes"})
  elif request.get("binary_sha256") != sha256_hex(code_object): errors.append({"field": "binary_sha256", "reason": "does not match code_object"})
  for field in ("binary_sha256", "argument_layout_digest", "launch_digest", "workload_digest", "schedule_digest"):
    value = request.get(field)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None: errors.append({"field": field, "reason": "expected 64 lowercase hex"})
  for field in ("grid", "workgroup"):
    if not _dim3(request.get(field)): errors.append({"field": field, "reason": "expected three positive ints"})
  allocations = request.get("allocations")
  if not isinstance(allocations, list) or not allocations: errors.append({"field": "allocations", "reason": "expected non-empty list"})
  elif any(not isinstance(row, Mapping) or not isinstance(row.get("size"), int) or row["size"] <= 0 for row in allocations):
    errors.append({"field": "allocations", "reason": "each allocation needs positive size"})
  timeout = request.get("timeout_ms")
  if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0: errors.append({"field": "timeout_ms", "reason": "expected positive int"})
  for field in ("candidate_id", "program_id", "target_id", "driver_id", "runtime_id"):
    if not isinstance(request.get(field), str) or not request[field]: errors.append({"field": field, "reason": "expected non-empty string"})
  return tuple(errors)


def launch(request: Mapping[str, Any], backend: Any, *, clock_ns=time.monotonic_ns) -> LaunchResult:
  # backend duck contract: allocate/submit/wait/release. The only implementation is FakeKFDBackend;
  # the Protocol that anticipated a real KFD backend was removed (LN-120).
  errors = validate_launch_request(request)
  if errors: return LaunchResult(str(request.get("candidate_id", "")), str(request.get("program_id", "")), None, "blocked", errors)
  allocations: list[Any] = []
  submit_ns = clock_ns()
  try:
    for descriptor in request["allocations"]: allocations.append(backend.allocate(descriptor))
    token = backend.submit(request, allocations)
    status = backend.wait(token, request["timeout_ms"])
    complete_ns = clock_ns()
    if status.get("status") != "completed":
      return LaunchResult(request["candidate_id"], request["program_id"], None, str(status.get("status", "blocked")),
                          ({"field": "backend", "reason": "launch did not complete", "status": status.get("status")},))
    observation = {"schema": "tinygrad.kfd_launch_sidecar.v1", "candidate_id": request["candidate_id"],
                   "program_id": request["program_id"], "source_sha256": request.get("source_sha256", ""),
                   "binary_sha256": request["binary_sha256"], "grid": list(request["grid"]),
                   "workgroup": list(request["workgroup"]), "submit_ns": submit_ns, "complete_ns": complete_ns,
                   "counters": dict(status.get("counters", {}))}
    return LaunchResult(request["candidate_id"], request["program_id"], observation, "completed")
  except BaseException as error:
    return LaunchResult(request["candidate_id"], request["program_id"], None, "blocked",
                        ({"field": "backend", "reason": str(error)},))
  finally:
    for allocation in reversed(allocations): backend.release(allocation)


def _dim3(value: Any) -> bool:
  return isinstance(value, list | tuple) and len(value) == 3 and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value)


__all__ = ["FakeKFDBackend", "LaunchResult", "SCHEMA", "launch", "validate_launch_request"]
