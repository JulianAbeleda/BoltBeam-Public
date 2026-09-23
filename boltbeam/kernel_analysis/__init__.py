"""Provider-neutral kernel analysis package.

BoltBeam requests, ingests, validates, compares, diagnoses, recommends, and remembers
codegen performance evidence. It never compiles kernels or dispatches GPU work.
The normalized authority is :class:`KernelEvidence`; analysis and recommendation reports
are :class:`KernelAnalysis` and :class:`KernelRecommendation`.
"""
from boltbeam.kernel_analysis.model import (
  CandidateModel,
  CorrectnessMetrics,
  FactBlocker,
  HealthSummary,
  IdentityModel,
  KernelAnalysis,
  KernelEvidence,
  KernelHash,
  KernelRecommendation,
  KernelStages,
  Measurement,
  OperandClassification,
  OperandDynamicEvidence,
  OperandPathEvidence,
  OperandStaticEvidence,
  OperandTransport,
  PipelineStage,
  ResourceSummary,
  StructureSummary,
  ServingTierObservation,
  WorkloadModel,
)
from boltbeam.kernel_analysis.classify import RULESET_VERSION, classify_kernel_evidence, classify_operand_path
from boltbeam.kernel_analysis.selection import MeasuredCandidateRank, MeasuredMatrixRanking, rank_measured_matrix

__all__ = [
  "CandidateModel",
  "CorrectnessMetrics",
  "FactBlocker",
  "HealthSummary",
  "IdentityModel",
  "KernelAnalysis",
  "KernelEvidence",
  "KernelHash",
  "KernelRecommendation",
  "KernelStages",
  "Measurement",
  "OperandClassification",
  "OperandDynamicEvidence",
  "OperandPathEvidence",
  "OperandStaticEvidence",
  "OperandTransport",
  "PipelineStage",
  "ResourceSummary",
  "StructureSummary",
  "ServingTierObservation",
  "WorkloadModel",
  "RULESET_VERSION",
  "classify_kernel_evidence",
  "classify_operand_path",
  "MeasuredCandidateRank",
  "MeasuredMatrixRanking",
  "rank_measured_matrix",
]
