"""Pure provider-neutral classification of normalized operand-path evidence."""
from __future__ import annotations

from dataclasses import replace

from boltbeam.kernel_analysis.model import KernelEvidence, OperandClassification, OperandPathEvidence


RULESET_VERSION = "operand-path-structural.v2"
_GLOBAL_LOADS = {"global_load", "global_load_b128", "buffer_load", "scalar_load"}
_DS_LOADS = {"ds_load", "ds_load_b128", "shared_load"}
_DS_STORES = {"ds_store", "ds_store_b128", "shared_store"}
_IDENTITY_BLOCKERS = {"identity_mismatch", "fragment_conflict", "binary_identity_mismatch",
                      "executed_binary_mismatch"}


def classify_operand_path(path: OperandPathEvidence, *, identity_corrupt: bool = False) -> OperandClassification:
  """Classify one normalized path. Serving-tier observations never select a strategy."""
  static = path.static
  sites = static.instruction_sites
  support: list[str] = []
  contradict: list[str] = []
  missing: list[str] = []

  if identity_corrupt:
    return _unknown(("valid_identity",), ("identity:corrupt",))
  if static.attribution_status in {"ambiguous", "source_binary_disagreement", "unknown"}:
    discriminator = {"ambiguous": "unambiguous_final_load_ownership",
                     "source_binary_disagreement": "consistent_source_final_ownership",
                     "unknown": "per_operand_final_program_attribution"}[static.attribution_status]
    return _unknown((discriminator,), (f"attribution:{static.attribution_status}",))

  loads = tuple(s for s in sites if s.kind in _GLOBAL_LOADS)
  ds_loads = tuple(s for s in sites if s.kind in _DS_LOADS)
  ds_stores = tuple(s for s in sites if s.kind in _DS_STORES)
  retained = tuple(s for s in sites if s.extra.get("retained_fragment") is True)
  spill_sites = tuple(s for s in sites if s.kind in {"scratch_load", "scratch_store", "private_load", "private_store"})
  spill = bool(spill_sites) or bool(static.spill_bytes)
  excess_reload = _has_excess_reload(path, loads)
  dynamic_reload = (path.dynamic.scope == "per_operand" and
                    path.dynamic.extra.get("repeated_global_fetch") is True)
  excess_reload = excess_reload or dynamic_reload

  signatures = sum((bool(ds_loads and ds_stores), bool(retained), excess_reload))
  if signatures > 1:
    ids = tuple(s.operation_id for s in (*ds_loads, *ds_stores, *retained, *loads))
    return _unknown(("noncontradictory_transport_structure",), ids)

  strategy = "unknown"
  confidence = "medium"
  if loads and ds_stores and ds_loads:
    strategy = "lds_staged"; support.extend(s.operation_id for s in (*loads, *ds_stores, *ds_loads))
  elif loads and retained and not excess_reload:
    strategy = "register_resident"; support.extend(s.operation_id for s in (*loads, *retained))
  elif loads and excess_reload:
    strategy = "reloaded"; support.extend(s.operation_id for s in loads)
    if dynamic_reload: support.append(path.dynamic.counter_id or "dynamic:repeated_global_fetch")
  elif loads and not ds_loads and not ds_stores:
    strategy = "cache_streamed"; support.extend(s.operation_id for s in loads)
  else:
    return _unknown(("final_operand_transport_discriminator",))

  if spill:
    contradict.extend(s.operation_id for s in spill_sites)
    if static.spill_bytes: contradict.append("static:spill_bytes")
    missing.append("spill_free_operand_path")
    confidence = "low"
  if path.declared_strategy != "unknown" and path.declared_strategy != strategy:
    contradict.append(f"declaration:{path.declared_strategy}")
    missing.append("declaration_evidence_reconciliation")
    confidence = "low"
  if static.complete_control_flow is True and not contradict:
    confidence = "high"
  elif static.complete_control_flow is not True:
    missing.append("complete_relevant_control_flow")

  # Aggregate counters can corroborate a kernel, but cannot become operand proof.
  if path.dynamic.scope != "per_operand" and any(v is not None for v in
      (path.dynamic.requested_bytes, path.dynamic.transferred_bytes, path.dynamic.requests, path.dynamic.transactions)):
    missing.append("per_operand_dynamic_attribution")

  return OperandClassification(strategy, "derived", confidence, tuple(dict.fromkeys(support)),
                               tuple(dict.fromkeys(contradict)), tuple(dict.fromkeys(missing)), RULESET_VERSION)


def classify_kernel_evidence(evidence: KernelEvidence) -> KernelEvidence:
  """Return evidence with every operand classification replaced by this ruleset's result."""
  corrupt = any(b.code in _IDENTITY_BLOCKERS or
                (b.scope == "identity" and any(word in b.code for word in ("mismatch", "conflict", "corrupt")))
                for b in evidence.blockers)
  paths = tuple((key, replace(path, classification=classify_operand_path(path, identity_corrupt=corrupt)))
                for key, path in evidence.operand_paths)
  return replace(evidence, operand_paths=paths)


def _has_excess_reload(path: OperandPathEvidence, loads: tuple) -> bool:
  if path.static.reload_factor is not None and path.static.reload_factor > 1:
    return True
  groups = path.static.compulsory_fetch_groups
  return bool(groups) and len(loads) > len(groups)


def _unknown(missing: tuple[str, ...], contradict: tuple[str, ...] = ()) -> OperandClassification:
  return OperandClassification("unknown", "unknown", "unknown", (), contradict, missing, RULESET_VERSION)
