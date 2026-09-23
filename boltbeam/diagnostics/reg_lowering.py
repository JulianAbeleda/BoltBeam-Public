"""REG scalar / reduction-accumulator lowering classifier (TG-P10.2).

Input:  the COMPILER_LOWERING EvidenceRows from a normalized tinygrad.reg_scalar_lowering.v1 artifact.
Output: a RegLoweringReport whose `reachability` is:
  - EMITTER_BLOCKED  when a control kernel compiles+is correct but the split-preserving combine shapes fail to lower
                     (invalid reduction-accumulator vector store), and REG_STORE_DEVEC compiles but returns NaN;
  - REACHABLE        when a fixed artifact shows every generated case compiles AND is numerically correct.

This is deliberately narrow: it does NOT classify REGALLOC_SPILL (no scratch signal), PRIMITIVE_MISSING (the REG and
generated UOp exist — the lowering is wrong), or REFUTED (the route is not disproven; it is compiler-blocked).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from boltbeam.vocab import RowKind, Reachability

SCHEMA = "boltbeam.reg_lowering_report.v1"


@dataclass
class RegLoweringReport:
  schema: str
  candidate_id: str
  reachability: str            # Reachability value
  reason: str
  reopen_directive: str
  control_ok: bool
  compile_failures: list[str]
  reg_store_devec_nan: bool
  invalid_reg_vector_store: bool

  def to_json(self) -> dict[str, Any]: return asdict(self)
  def to_json_str(self, indent: int = 2) -> str: return json.dumps(self.to_json(), indent=indent, sort_keys=True)


def classify_reg_scalar_lowering(candidate_id: str, rows: list) -> RegLoweringReport:
  cl = [r for r in rows if r.kind == RowKind.COMPILER_LOWERING.value]
  if not cl:
    return RegLoweringReport(SCHEMA, candidate_id, Reachability.TARGET_INCOMPLETE.value,
                             "no compiler_lowering rows present", "supply a tinygrad.reg_scalar_lowering.v1 artifact",
                             False, [], False, False)
  # guardrail: the repro must be generated-UOp only (never a handwritten kernel masquerading as a repro)
  if any(not r.extra.get("generated_uop_only", True) or r.extra.get("uses_external_kernel", False) for r in cl):
    return RegLoweringReport(SCHEMA, candidate_id, Reachability.REFUTED_BY_LEDGER.value,
                             "repro contains a non-generated / external kernel case; not a valid compiler-lowering repro",
                             "rebuild the repro with generated UOp only", False, [], False, False)

  # the control is the shipped single-reduce combine (must compile + be numerically correct to validate the repro)
  is_control = lambda r: "shipped" in r.metric
  control_ok = any(is_control(r) and r.extra.get("numeric_ok") for r in cl)
  compile_failures = [r.metric for r in cl if not r.extra.get("compile_ok")]
  invalid_store = any(r.extra.get("error_class") == "invalid_reg_vector_store" for r in cl)
  devec_nan = any(r.extra.get("uses_reg_store_devec") and r.extra.get("compile_ok") and not r.extra.get("numeric_ok")
                  for r in cl)

  # non-control cases: the split-preserving combine shapes (weight-sharing / fused-gmax / devec)
  non_control = [r for r in cl if not is_control(r)]
  all_non_control_fixed = bool(non_control) and all(r.extra.get("numeric_ok") for r in non_control)

  if control_ok and all_non_control_fixed:
    return RegLoweringReport(
      SCHEMA, candidate_id, Reachability.REACHABLE.value,
      "every generated split-preserving combine shape now compiles and is numerically correct (lowering fix landed)",
      "reopen: re-run the split-preserving combine microgate + full generated attention W==D (TG-P10.4/5)",
      control_ok, compile_failures, devec_nan, invalid_store)

  if control_ok and (invalid_store or compile_failures or devec_nan):
    return RegLoweringReport(
      SCHEMA, candidate_id, Reachability.EMITTER_BLOCKED.value,
      "AMD lowering mis-vectorizes a reduction-accumulator REG needed by the split-preserving LSE combine "
      "(control compiles+correct; weight-sharing/fused-gmax shapes fail with invalid_reg_vector_store; "
      "REG_STORE_DEVEC compiles but returns NaN)",
      "keep the combine reduction accumulator scalar, or make REG_STORE_DEVEC numerically correct for max/LSE "
      "reduction state -- a generic AMD/UOp lowering fix, no kernel-name special casing",
      control_ok, compile_failures, devec_nan, invalid_store)

  return RegLoweringReport(
    SCHEMA, candidate_id, Reachability.TARGET_INCOMPLETE.value,
    "no passing control case; cannot attribute the failure to lowering vs the repro harness itself",
    "add a passing control (the shipped single-reduce combine) to validate the repro", control_ok,
    compile_failures, devec_nan, invalid_store)
