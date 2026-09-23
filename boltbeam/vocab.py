"""Centralized authority for BoltBeam vocabularies.

Per coding-principles.md ("Centralize authority"): every verdict, artifact kind, ledger status, guardrail
state, reachability class, tier, workload, and schema id lives HERE and is imported everywhere else. No module
may invent a parallel string vocabulary. Adding a value = editing this file, not scattering literals.

Enums subclass (str, Enum) so members are JSON-serializable and compare equal to their string value on Python
3.10+ (the project's floor), without needing 3.11 StrEnum.
"""
from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
  """The only verdicts an evaluator/decision may emit (audit-brain-build-scope 'Verdict Vocabulary')."""
  DIAGNOSTIC = "diagnostic"                              # explains a bottleneck, not a candidate
  CANDIDATE = "candidate"                                # worth measuring, not promoted
  PROMOTE = "promote"                                    # passes correctness/route/speed/memory/rollback
  REFUTE = "refute"                                      # failed the relevant gate with firm evidence
  DEFER = "defer"                                        # promising but blocked by a named capability
  INCONCLUSIVE = "inconclusive"                          # evidence insufficient or noisy
  SEARCH_SPACE_INCOMPLETE = "search-space-incomplete"    # needed knob not expressible yet
  ADAPTER_INCOMPLETE = "adapter-incomplete"              # artifact cannot be safely read yet


class Tier(str, Enum):
  """Speed-movement tier (thresholds live in boltbeam/eval/thresholds.py)."""
  A = "A"          # broad win, default-promotable if guardrails pass
  B = "B"          # residual win, promotable with clean rollback + no protected regression
  C = "C"          # diagnostic only unless strategic
  NONE = "none"


class OperandStrategy(str, Enum):
  """Exclusive software transport selected for one semantic kernel operand."""
  REGISTER_RESIDENT = "register_resident"
  LDS_STAGED = "lds_staged"
  CACHE_STREAMED = "cache_streamed"
  RELOADED = "reloaded"
  UNKNOWN = "unknown"


class ServingTier(str, Enum):
  """Physical tier serving operand traffic; deliberately orthogonal to OperandStrategy."""
  REGISTER = "register"
  SCRATCH = "scratch"
  LDS = "lds"
  L0 = "l0"
  L1 = "l1"
  L2 = "l2"
  LAST_LEVEL_CACHE = "last_level_cache"
  DRAM = "dram"
  HOST = "host"
  UNKNOWN = "unknown"


class EvidenceConfidence(str, Enum):
  HIGH = "high"
  MEDIUM = "medium"
  LOW = "low"
  UNKNOWN = "unknown"


class Workload(str, Enum):
  DECODE = "decode"
  PREFILL = "prefill"


class Architecture(str, Enum):
  """Derived architecture class (audit A2). Metadata-first, tensor-shape fallback second. `unknown_transformer`
  is not a crash — it yields an honest incomplete profile the downstream must not treat as dense."""
  DENSE_DECODER = "dense_decoder"
  MOE_DECODER = "moe_decoder"
  HYBRID_DECODER = "hybrid_decoder"
  HYBRID_MOE_DECODER = "hybrid_moe_decoder"
  ENCODER_DECODER = "encoder_decoder"
  UNKNOWN_TRANSFORMER = "unknown_transformer"


class RoleClass(str, Enum):
  """Fine-grained tensor role taxonomy (audit A1). This is the richest role model; the coarse grouped roles
  used by the search space are DERIVED from these via ROLE_GROUP_OF. Classification (name -> RoleClass) lives
  in profile/roles.py; the vocabulary and the fine->coarse map live here as the single authority."""
  ATTENTION_Q = "attention_q"
  ATTENTION_K = "attention_k"
  ATTENTION_V = "attention_v"
  ATTENTION_O = "attention_o"
  FFN_GATE = "ffn_gate"
  FFN_UP = "ffn_up"
  FFN_DOWN = "ffn_down"
  LM_HEAD = "lm_head"
  EMBEDDING = "embedding"
  NORM = "norm"
  MOE_ROUTER = "moe_router"
  MOE_EXPERT_GATE = "moe_expert_gate"
  MOE_EXPERT_UP = "moe_expert_up"
  MOE_EXPERT_DOWN = "moe_expert_down"
  MOE_SHARED_EXPERT_GATE = "moe_shared_expert_gate"
  MOE_SHARED_EXPERT_UP = "moe_shared_expert_up"
  MOE_SHARED_EXPERT_DOWN = "moe_shared_expert_down"
  SSM_PROJECTION = "ssm_projection"
  SSM_CONV = "ssm_conv"
  SSM_STATE = "ssm_state"
  SSM_SCAN = "ssm_scan"
  OTHER = "other"


class RoleGroup(str, Enum):
  """Coarse grouped roles — the DERIVED views the search space and candidate manifest key off. The four dense
  legacy groups (attn_qo, attn_kv, ffn_gate_up, ffn_down) are preserved byte-for-byte; the rest generalize the
  taxonomy (MoE groups added by A5)."""
  ATTN_QO = "attn_qo"
  ATTN_KV = "attn_kv"
  FFN_GATE_UP = "ffn_gate_up"
  FFN_DOWN = "ffn_down"
  LM_HEAD = "lm_head"
  EMBEDDING = "embedding"
  NORM = "norm"
  MOE_ROUTER = "moe_router"
  MOE_EXPERT_GATE_UP = "moe_expert_gate_up"
  MOE_EXPERT_DOWN = "moe_expert_down"
  MOE_SHARED_EXPERT_GATE_UP = "moe_shared_expert_gate_up"
  MOE_SHARED_EXPERT_DOWN = "moe_shared_expert_down"
  SSM_PROJECTION = "ssm_projection"
  SSM_CONV = "ssm_conv"
  SSM_STATE = "ssm_state"
  SSM_SCAN = "ssm_scan"
  OTHER = "other"


# fine RoleClass -> coarse RoleGroup (the derived grouped view). q/o collapse to attn_qo, k/v to attn_kv, and
# gate/up to *_gate_up exactly as the shipped dense pipeline did; MoE fine roles collapse to MoE groups.
ROLE_GROUP_OF: dict[str, str] = {
  RoleClass.ATTENTION_Q.value: RoleGroup.ATTN_QO.value,
  RoleClass.ATTENTION_O.value: RoleGroup.ATTN_QO.value,
  RoleClass.ATTENTION_K.value: RoleGroup.ATTN_KV.value,
  RoleClass.ATTENTION_V.value: RoleGroup.ATTN_KV.value,
  RoleClass.FFN_GATE.value: RoleGroup.FFN_GATE_UP.value,
  RoleClass.FFN_UP.value: RoleGroup.FFN_GATE_UP.value,
  RoleClass.FFN_DOWN.value: RoleGroup.FFN_DOWN.value,
  RoleClass.LM_HEAD.value: RoleGroup.LM_HEAD.value,
  RoleClass.EMBEDDING.value: RoleGroup.EMBEDDING.value,
  RoleClass.NORM.value: RoleGroup.NORM.value,
  RoleClass.MOE_ROUTER.value: RoleGroup.MOE_ROUTER.value,
  RoleClass.MOE_EXPERT_GATE.value: RoleGroup.MOE_EXPERT_GATE_UP.value,
  RoleClass.MOE_EXPERT_UP.value: RoleGroup.MOE_EXPERT_GATE_UP.value,
  RoleClass.MOE_EXPERT_DOWN.value: RoleGroup.MOE_EXPERT_DOWN.value,
  RoleClass.MOE_SHARED_EXPERT_GATE.value: RoleGroup.MOE_SHARED_EXPERT_GATE_UP.value,
  RoleClass.MOE_SHARED_EXPERT_UP.value: RoleGroup.MOE_SHARED_EXPERT_GATE_UP.value,
  RoleClass.MOE_SHARED_EXPERT_DOWN.value: RoleGroup.MOE_SHARED_EXPERT_DOWN.value,
  RoleClass.SSM_PROJECTION.value: RoleGroup.SSM_PROJECTION.value,
  RoleClass.SSM_CONV.value: RoleGroup.SSM_CONV.value,
  RoleClass.SSM_STATE.value: RoleGroup.SSM_STATE.value,
  RoleClass.SSM_SCAN.value: RoleGroup.SSM_SCAN.value,
  RoleClass.OTHER.value: RoleGroup.OTHER.value,
}


def role_group_of(role_class:str) -> str:
  """Coarse grouped role for a fine RoleClass value (defaults to 'other' for unknown input)."""
  return ROLE_GROUP_OF.get(role_class, RoleGroup.OTHER.value)


def role_group_members(group:str) -> tuple[str, ...]:
  """The fine RoleClass values that collapse into a coarse group (the derived-view membership)."""
  return tuple(rc for rc, g in ROLE_GROUP_OF.items() if g == group)


# Shared coarse-role priority table used by the two role rankers in analyze.py and workflow/analyze.py
# (_rank_by_memory_cost and _rank_by_routability). Both formerly hardcoded this same name->priority mapping;
# it is one rule (which coarse role sorts before which, all else equal) and now has one owner. Unlisted
# roles rank last (see ROLE_RANK_DEFAULT). This is orthogonal to each ranker's quant-priority axis, which is
# genuinely different between the two (memory cost vs. route support) and stays local to each caller.
ROLE_RANK_ORDER: dict[str, int] = {
  RoleGroup.ATTN_KV.value: 0,
  RoleGroup.FFN_DOWN.value: 1,
  RoleGroup.FFN_GATE_UP.value: 2,
  RoleGroup.ATTN_QO.value: 3,
  RoleGroup.LM_HEAD.value: 4,
}
ROLE_RANK_DEFAULT = 8


def role_rank_priority(role:str) -> int:
  """Shared coarse-role ordering rule: lower sorts first. Roles absent from ROLE_RANK_ORDER rank last
  (ROLE_RANK_DEFAULT). Used identically by both _rank_by_memory_cost (boltbeam/analyze.py) and
  _rank_by_routability (boltbeam/workflow/analyze.py); those two functions differ only on the quant axis."""
  return ROLE_RANK_ORDER.get(role, ROLE_RANK_DEFAULT)


def is_moe_role(role_class_or_group:str) -> bool:
  """True for any MoE fine role or MoE coarse group (router / expert / shared-expert)."""
  return role_class_or_group.startswith("moe_")


def is_ssm_role(role_class_or_group:str) -> bool:
  """True for any state-space / DeltaNet fine role or coarse group."""
  return role_class_or_group.startswith("ssm_")


def architecture_compatible(candidate_architectures:tuple[str, ...], architecture:str | None) -> bool:
  """Whether an arch-scoped candidate may apply to a model architecture.

  Hybrid decoder profiles inherit the dense decoder route surface for their attention/dense/shared roles.
  Hybrid-MoE profiles additionally inherit MoE route templates for expert/router roles. Keeping this relation
  centralized avoids copying every dense/MoE candidate row for each hybrid architecture variant.
  """
  if not architecture or not candidate_architectures:
    return True
  expanded = {architecture}
  if architecture == Architecture.HYBRID_DECODER.value:
    expanded.add(Architecture.DENSE_DECODER.value)
  if architecture == Architecture.HYBRID_MOE_DECODER.value:
    expanded.update({Architecture.DENSE_DECODER.value, Architecture.MOE_DECODER.value,
                     Architecture.HYBRID_DECODER.value})
  return bool(expanded & set(candidate_architectures))


class TinygradArtifactKind(str, Enum):
  """Raw tinygrad artifact families BoltBeam v1 can ingest (audit-brain-build-scope BB0)."""
  DECODE_ROLE_ATTRIBUTION = "decode_role_attribution"
  DECODE_REDUCE_SOURCE_TRACE = "decode_reduce_source_trace"
  DECODE_RUNTIME_OVERHEAD = "decode_runtime_overhead"
  DECODE_WD = "decode_wd"
  PREFILL_AUTHORITY = "prefill_authority"
  PROMOTION_GATE = "promotion_gate"
  CEILING_REPORT = "ceiling_report"
  COMPILER_PATHOLOGY = "compiler_pathology"   # tinygrad.compiler_pathology.v1 disasm/resource artifact
  KV_CTX_SLOPE = "kv_ctx_slope"               # tinygrad.llama_kv_ctx_slope.v1 long-context KV slope artifact
  REG_SCALAR_LOWERING = "reg_scalar_lowering" # tinygrad.reg_scalar_lowering.v1 REG/reduction-accumulator lowering repro


class RowKind(str, Enum):
  """Normalized evidence row kinds (what a single normalized measurement describes)."""
  ROLE_ATTRIBUTION = "role_attribution"       # per-role wall share / bandwidth
  REDUCE_SOURCE = "reduce_source"             # a reduce kernel attributed to a source op
  RUNTIME_OVERHEAD = "runtime_overhead"       # host-sync vs GPU wall separation
  WD_SPEED = "wd_speed"                       # whole-decode tok/s (W==D)
  PREFILL_SPEED = "prefill_speed"             # whole-prefill tok/s
  CEILING = "ceiling"                         # roofline / Amdahl derived
  COMPILER_PATHOLOGY = "compiler_pathology"   # per-kernel resource/timing diagnostic row
  KV_CTX_SLOPE = "kv_ctx_slope"               # ms/token = A + B*ctx for a KV cache dtype pair
  COMPILER_LOWERING = "compiler_lowering"     # a generated-UOp compile/lowering outcome (compile_ok/numeric_ok/accum state)


class KVPolicyClass(str, Enum):
  """Quantized KV cache policy classes (kv-cache-quantization-policy-scope.md)."""
  STORAGE_WIN_SPEED_LOSS = "KV_STORAGE_WIN_SPEED_LOSS"
  STORAGE_WIN_SPEED_WIN = "KV_STORAGE_WIN_SPEED_WIN"
  CAPACITY_ONLY = "KV_CAPACITY_ONLY"
  FASTPATH_MISSING = "KV_FASTPATH_MISSING"
  MIXED_FASTPATH_UNKNOWN = "KV_MIXED_FASTPATH_UNKNOWN"
  INCONCLUSIVE = "KV_INCONCLUSIVE"


class PracticalRooflineClass(str, Enum):
  """Practical-roofline classification for promotion/route-family closeout."""
  PROMOTABLE = "PRACTICAL_ROOFLINE_PROMOTABLE"
  CLOSEOUT = "PRACTICAL_ROOFLINE_CLOSEOUT"
  HEADROOM_REMAINS = "HEADROOM_REMAINS"
  INCONCLUSIVE = "PRACTICAL_ROOFLINE_INCONCLUSIVE"


class TargetCapability(str, Enum):
  """Target capability keys stored in data/targets.json."""
  SAME_TYPE_QUANT_KV_FLASH_FASTPATH = "same_type_quant_kv_flash_fastpath"
  MIXED_QUANT_KV_FLASH_FASTPATH = "mixed_quant_kv_flash_fastpath"
  KV_QUANT_SPEED_RESIDUAL_REQUIRED = "kv_quant_speed_residual_required"
  KV_QUANT_QUALITY_GATE_REQUIRED = "kv_quant_quality_gate_required"


class PathologyClass(str, Enum):
  """Compiler pathology classifications (compiler-pathology-diagnostics-scope.md CP2)."""
  REGALLOC_SPILL = "REGALLOC_SPILL"           # scratch_bytes > 0 → spills to global memory
  BARRIER_EXPLOSION = "BARRIER_EXPLOSION"     # barrier_count far above expected
  LDS_OR_MEMORY_OVERHEAD = "LDS_OR_MEMORY_OVERHEAD"  # LDS/VMEM traffic dominates without spill
  LOOP_LOWERING_BAD = "LOOP_LOWERING_BAD"     # static instruction ratio >> math ops
  VECTOR_LOAD_LOST = "VECTOR_LOAD_LOST"       # emitted vector_load_bits < expected
  WAITCNT_BAD = "WAITCNT_BAD"                 # waitcnt density anomalously high
  NATIVE_ISA_ORACLE_NEEDED = "NATIVE_ISA_ORACLE_NEEDED"  # timing known but no resource rows yet
  STRUCTURAL_ROUTE_REFUTED = "STRUCTURAL_ROUTE_REFUTED"  # oracle also slow; route is structurally bad
  UNKNOWN = "UNKNOWN"                         # resource rows present but no class matches


class LedgerStatus(str, Enum):
  """Durable status a route/candidate holds in the ledger."""
  PROMOTED = "promoted"
  REFUTED = "refuted"
  DEFERRED = "deferred"
  SEARCH_SPACE_INCOMPLETE = "search-space-incomplete"
  CANDIDATE = "candidate"


class Guardrail(str, Enum):
  """The promotion guardrails (audit-brain-build-scope BB5). Each resolves to a GuardrailState."""
  CORRECTNESS = "correctness"
  ROUTE_BOUND = "route_bound"
  SPEED = "speed"
  MEMORY_FIT = "memory_fit"
  FALLBACK = "fallback"
  DETERMINISM = "determinism"
  ROLLBACK = "rollback"
  # mechanism check for reduce-elimination candidates: the targeted reduce bucket must actually shrink in the
  # before/after profile, not merely a faster tok/s number (which attention timing can produce by confound).
  REDUCE_ELIMINATED = "reduce_eliminated"


class GuardrailState(str, Enum):
  PASS = "pass"
  FAIL = "fail"
  UNKNOWN = "unknown"


class Reachability(str, Enum):
  """Search-space reachability classes (audit-brain-build-scope BB10). This is the codegen-directive signal."""
  REACHABLE = "reachable"
  EMITTER_BLOCKED = "emitter-blocked"
  PRIMITIVE_MISSING = "primitive-missing"
  TARGET_INCOMPLETE = "target-incomplete"
  REFUTED_BY_LEDGER = "refuted-by-ledger"


class BackendStatus(str, Enum):
  """How far a target backend is actually built (audit A6). Only `complete` targets can promote a route;
  descriptor-only targets carry capability data but have no evaluator yet, so their routes are
  target-backend-incomplete, never promoted."""
  COMPLETE = "complete"
  DESCRIPTOR_ONLY = "descriptor_only"
  UNSUPPORTED = "unsupported"


class QuantSupport(str, Enum):
  """How far the quant capability registry (data/quants.json) can reason about a quant (audit A3)."""
  SUPPORTED = "supported"                 # in the registry AND has at least one route family
  KNOWN_NO_ROUTE = "known_no_route"       # in the registry but no route family emitted yet (pending)
  UNSUPPORTED = "unsupported_quant"       # not in the registry -> search-space-incomplete, never silent


class Confidence(str, Enum):
  MEASURED = "measured"
  DERIVED = "derived"
  MODELED = "modeled"
  IMPORTED = "imported"        # historical seed, not freshly measured by BoltBeam
  ASSUMED = "assumed"


# ---- schema ids (the durable boundary; bump the trailing version to break compatibility) -------------------
SCHEMA_MODEL_PROFILE = "boltbeam.model_profile.v1"
SCHEMA_NORMALIZED_EVIDENCE = "boltbeam.normalized_evidence.v1"
SCHEMA_CANDIDATE_DECISION = "boltbeam.candidate_decision.v1"
SCHEMA_ROUTE_LEDGER_ENTRY = "boltbeam.route_ledger_entry.v1"
SCHEMA_ROUTE_LEDGER_SUMMARY = "boltbeam.route_ledger_summary.v1"
SCHEMA_ARTIFACT_CACHE = "boltbeam.artifact_cache.v1"
SCHEMA_CEILING_REPORT = "boltbeam.ceiling_report.v1"
SCHEMA_CANDIDATE_MANIFEST = "boltbeam.candidate_manifest.v1"
SCHEMA_QUANT_REGISTRY = "boltbeam.quant_registry.v1"
SCHEMA_TARGET_REGISTRY = "boltbeam.target_registry.v1"
SCHEMA_REACHABILITY_REPORT = "boltbeam.reachability_report.v1"
SCHEMA_EVIDENCE_BUNDLE = "boltbeam.evidence_bundle.v1"
SCHEMA_INGEST_ERROR = "boltbeam.ingest_error.v1"
SCHEMA_RUN_MANIFEST = "boltbeam.run_manifest.v1"
SCHEMA_WEIGHT_INVENTORY = "boltbeam.weight_inventory.v1"
SCHEMA_WORKLOAD_PROFILE = "boltbeam.workload_profile.v1"
SCHEMA_HARDWARE_PROFILE = "boltbeam.hardware_profile.v1"
SCHEMA_RUNTIME_PROFILE = "boltbeam.runtime_profile.v1"
SCHEMA_PROVIDER_CAPABILITIES = "boltbeam.provider_capabilities.v1"
SCHEMA_SCAN_EVIDENCE = "boltbeam.scan_evidence.v1"
SCHEMA_MEASUREMENT_PLAN = "boltbeam.measurement_plan.v1"
SCHEMA_ANALYSIS_REPORT = "boltbeam.analysis_report.v1"
SCHEMA_OUTPUT_MANIFEST = "boltbeam.output_manifest.v1"
SCHEMA_PROBE_REQUEST = "boltbeam.probe_request.v1"
SCHEMA_PROBE_EVIDENCE = "boltbeam.probe_evidence.v1"
SCHEMA_PRIMITIVE_PROFILE = "boltbeam.primitive_profile.v1"
SCHEMA_TRACE_REQUEST = "boltbeam.trace_request.v1"
SCHEMA_TRACE_EXECUTION_PROFILE = "boltbeam.trace_execution_profile.v1"
SCHEMA_TIMING_TRACE = "boltbeam.timing_trace.v1"
SCHEMA_TIMING_PROFILE = "boltbeam.timing_profile.v1"
SCHEMA_HW_TRACE = "boltbeam.hw_trace.v1"
# SCHEMA_PROFILER_CAPABILITIES / SCHEMA_COUNTER_REGISTRY are owned by profiler/ (capabilities.py,
# counters.py); re-exported here for callers that import the whole schema vocabulary from one place.
from boltbeam.profiler.capabilities import SCHEMA_PROFILER_CAPABILITIES  # noqa: E402
from boltbeam.profiler.counters import SCHEMA_COUNTER_REGISTRY  # noqa: E402
SCHEMA_PROFILER_REPORT = "boltbeam.profiler_report.v1"
SCHEMA_HW_TRACE_COMPARE = "boltbeam.hw_trace_compare.v1"
SCHEMA_SUBSTRATE_COMPARE = "boltbeam.substrate_compare.v1"
SCHEMA_RUNNER_PLAN = "boltbeam.runner_plan.v1"
SCHEMA_MATCHED_CONTROL_PROTOCOL = "boltbeam.matched_control_protocol.v1"
SCHEMA_MATCHED_CONTROL_ROW = "boltbeam.matched_control_row.v1"
SCHEMA_MATCHED_CONTROL_SUMMARY = "boltbeam.matched_control_summary.v1"
SCHEMA_RUNTIME_OUTPUT_IDENTITY = "boltbeam.runtime_output_identity.v1"
SCHEMA_TINYGRAD_GRAPH_ADMISSION_CENSUS = "tinygrad.graph_admission_census.v1"
SCHEMA_REPLAY_AB_PROTOCOL = "boltbeam.replay_ab_protocol.v1"
SCHEMA_REPLAY_AB_ROW = "boltbeam.replay_ab_row.v1"
SCHEMA_REPLAY_AB_SUMMARY = "boltbeam.replay_ab_summary.v1"
SCHEMA_EXECUTION_BRIDGE_REQUEST = "execution_bridge.request.v1"
SCHEMA_EXECUTION_BRIDGE_RESULT = "execution_bridge.result.v1"
SCHEMA_TINYGRAD_KV_CTX_SLOPE = "tinygrad.llama_kv_ctx_slope.v1"
SCHEMA_TINYGRAD_REG_SCALAR_LOWERING = "tinygrad.reg_scalar_lowering.v1"
SCHEMA_FACT = "boltbeam.fact.v1"
SCHEMA_SYSTEM_SNAPSHOT = "boltbeam.system_snapshot.v1"
# LDS-vs-L2 analysis pipeline (docs/lds-vs-l2-analysis-scope-20260712.md)
SCHEMA_WS_SWEEP_REQUEST = "boltbeam.ws_sweep_request.v1"
SCHEMA_WS_SWEEP_SAMPLES = "boltbeam.ws_sweep_samples.v1"
SCHEMA_MEMORY_HIERARCHY_PROFILE = "boltbeam.memory_hierarchy_profile.v1"
SCHEMA_LDS_L2_CANDIDATE = "boltbeam.lds_l2_candidate.v1"
SCHEMA_EXPERIMENT_MANIFEST = "boltbeam.experiment_manifest.v1"
SCHEMA_EVIDENCE_REQUIREMENTS = "boltbeam.evidence_requirements.v1"
SCHEMA_KERNEL_EVIDENCE = "boltbeam.kernel_evidence.v1"
SCHEMA_KERNEL_ANALYSIS = "boltbeam.kernel_analysis.v1"
SCHEMA_KERNEL_RECOMMENDATION = "boltbeam.kernel_recommendation.v1"
SCHEMA_KERNEL_EVIDENCE_REQUEST_SET = "boltbeam.kernel_evidence_request_set.v1"
SCHEMA_THEORETICAL_ROOFLINE = "boltbeam.theoretical_roofline.v1"
# NOTE: SCHEMA_TINYGRAD_ARTIFACT_INDEX was removed — it was never referenced (no index emitter/consumer
# exists), so per "centralize authority / no dead vocabulary" it is dropped rather than kept as a stub.
# tools/ same-run promotion pipeline (outstanding-fixes-scope-20260731.md item 2): these scripts mint
# artifacts inside the boltbeam.* namespace, so the ids they claim are defined here even though the scripts
# themselves are gate/evidence checkers rather than CandidateDecision producers.
#
# These ids name a KIND of gate, never the model under test. A schema id carrying a model name cannot
# describe a second model, which is what forced a per-model copy of each script; the model, target, quant
# and context ladder are fields of the artifact instead. See docs/task_workflow/output/
# model-agnostic-same-run-tools-20260731.md.
SCHEMA_ROUTE_PROMOTION_GATE = "boltbeam.route_promotion_gate.v1"
SCHEMA_FUSED_ROUTE_PROMOTION = "boltbeam.fused_route_promotion.v1"
SCHEMA_PREFILL_ROLE_ACCOUNTING = "boltbeam.prefill_role_accounting.v1"
SCHEMA_SAME_RUN_PARITY = "boltbeam.same_run_parity.v1"
SCHEMA_QUANT_COMPARISON_PREPARATION = "boltbeam.quant_comparison_preparation.v1"
SCHEMA_SAME_RUN_MEASUREMENT = "boltbeam.same_run_measurement.v1"
SCHEMA_SAME_RUN_PREPARATION = "boltbeam.same_run_preparation.v1"


def enum_values(enum_cls:type[Enum]) -> list[str]:
  """String values of an enum, for schema `enum` blocks and validation messages."""
  return [m.value for m in enum_cls]
