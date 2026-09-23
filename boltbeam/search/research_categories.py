"""Research grounding categories for GPU-kernel machine search.

These categories define the evidence BoltBeam should be able to answer before a
candidate family is considered more than a probe.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchCategory:
  id: int
  slug: str
  name: str
  question: str
  evidence: tuple[str, ...]
  reference_anchor: str

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "slug": self.slug,
      "name": self.name,
      "question": self.question,
      "evidence": list(self.evidence),
      "reference_anchor": self.reference_anchor,
    }


RESEARCH_CATEGORIES: tuple[ResearchCategory, ...] = (
  ResearchCategory(1, "correctness_contract", "Correctness contract",
                   "What exact operation, inputs, outputs, dtype, shape, and tolerance must the candidate satisfy?",
                   ("operation", "input_format", "output_format", "shape", "dtype", "tolerance"),
                   "TVM Ansor starts from a tensor computation definition before schedule/search construction."),
  ResearchCategory(2, "numeric_oracle", "Numeric oracle",
                   "What trusted implementation or reference output rejects wrong candidates before timing?",
                   ("reference_impl", "test_vectors", "tolerance", "failure_mode"),
                   "CUDA optimization workflows separate verification from profiling."),
  ResearchCategory(3, "data_layout", "Data layout",
                   "How are tensors packed, scaled, aligned, strided, and addressed?",
                   ("packed_format", "scale_layout", "stride", "alignment", "address_formula"),
                   "CUDA memory-access efficiency depends on layout and addressing."),
  ResearchCategory(4, "tile_geometry", "Tile geometry / thread-block geometry",
                   "What workgroup, wave/warp, and tile M/N/K geometry is legal?",
                   ("workgroup_shape", "wave_count", "tile_m", "tile_n", "tile_k"),
                   "CUDA defines execution through grids, blocks, threads, and per-block shared resources."),
  ResearchCategory(5, "work_ownership", "Work ownership / thread-to-output mapping",
                   "Which thread, lane, wave/warp, or fragment owns each output element?",
                   ("owner_map", "lane_map", "fragment_map", "duplicate_store_count", "missing_store_count"),
                   "GPU execution models make thread-to-work mapping the substrate for parallel kernels."),
  ResearchCategory(6, "accumulator_mapping", "Accumulator / sum mapping",
                   "How does each logical accumulator slot map to lane/register/output identity?",
                   ("sum_slot_map", "register_identity", "lane_identity", "output_identity"),
                   "Accumulator placement directly drives VGPR pressure and correctness."),
  ResearchCategory(7, "shared_memory_lifecycle", "Shared memory / LDS lifecycle",
                   "What is staged in shared memory/LDS, when is it visible, and how is it reused?",
                   ("lds_layout", "stage_events", "padding", "barriers", "reuse_count"),
                   "Shared memory/LDS trades data reuse against occupancy and synchronization cost."),
  ResearchCategory(8, "k_loop_cadence", "K-loop cadence / reuse cadence",
                   "How does the reduction loop advance through K, reload panels, and amortize work per byte?",
                   ("k_step", "panel_lifetime", "reload_count", "reuse_per_load"),
                   "Roofline operational intensity depends on work per byte moved."),
  ResearchCategory(9, "dot_primitive", "Dot primitive / tensor instruction choice",
                   "Which scalar, vector, or tensor instruction family performs the inner product?",
                   ("instruction_family", "operand_shape", "throughput_ceiling", "precision"),
                   "Instruction throughput ceilings differ from memory-bandwidth ceilings."),
  ResearchCategory(10, "resource_model", "Register, VGPR, SGPR, LDS, occupancy model",
                   "What resource usage limits resident waves and feasible tile shapes?",
                   ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "occupancy"),
                   "AMD occupancy guidance treats VGPR and LDS as primary resident-wave limiters."),
  ResearchCategory(11, "sync_cadence", "Wait / barrier / synchronization cadence",
                   "Which waits and barriers are required, and which ones are overhead?",
                   ("waitcnts", "barriers", "memory_visibility", "dependency_edges"),
                   "CUDA and AMD programming models require synchronization for shared-memory visibility and ordering."),
  ResearchCategory(12, "store_path", "Store path / memory coalescing",
                   "How are final outputs stored, coalesced, vectorized, and tied to source values?",
                   ("store_map", "coalescing", "store_width", "data_register", "address_register"),
                   "CUDA Best Practices highlights global-memory coalescing as a high-priority performance factor."),
  ResearchCategory(13, "baseline_comparator", "Baseline and comparator",
                   "What same-session baseline and external comparator define success?",
                   ("baseline", "candidate_timing", "external_timing", "measurement_protocol"),
                   "Performance models are useful only when compared against measured baselines and hardware capability."),
  ResearchCategory(14, "roofline_target", "Roofline / operational-intensity target",
                   "Is the candidate memory-bound, compute-bound, or capped by a lower practical ceiling?",
                   ("flops", "bytes_moved", "operational_intensity", "bandwidth_ceiling", "compute_ceiling"),
                   "Roofline bounds performance by peak compute and bandwidth times operational intensity."),
  ResearchCategory(15, "search_space_knobs", "Search-space knobs",
                   "Which legal dimensions can search vary without changing the problem?",
                   ("tile_knobs", "staging_knobs", "unroll_knobs", "wait_knobs", "placement_knobs"),
                   "Ansor explores a hierarchical search space guided by a cost model."),
  ResearchCategory(16, "route_binding_rules", "Route binding rules",
                   "Which modular workload facts select this candidate family?",
                   ("role", "quant", "shape_rule", "arch_rule", "resource_rule"),
                   "Workload facts should select candidates, not model-name branches."),
  ResearchCategory(17, "promotion_gates", "Promotion gates",
                   "What criteria move a research candidate into a live/default route?",
                   ("correctness_gate", "performance_gate", "resource_gate", "rollback", "evidence_refs"),
                   "Promotion requires correctness, reproducible measurement, and rollback evidence."),
  ResearchCategory(18, "stop_criteria", "Stop / blocked criteria",
                   "What missing fact, primitive, API, or hardware limit halts the search?",
                   ("blocker", "blocked_category", "missing_api", "hard_ceiling", "next_probe"),
                   "Explicit ceilings and blockers prevent search from optimizing impossible paths."),
)


def research_categories() -> tuple[ResearchCategory, ...]:
  return RESEARCH_CATEGORIES


def research_category_by_slug(slug: str) -> ResearchCategory:
  for category in RESEARCH_CATEGORIES:
    if category.slug == slug: return category
  raise ValueError(f"unknown research category {slug!r}; expected one of {research_category_slugs()}")


def research_category_slugs() -> tuple[str, ...]:
  return tuple(c.slug for c in RESEARCH_CATEGORIES)


def validate_research_category_coverage(answers: dict[str, object]) -> dict[str, list[str]]:
  """Return missing/unknown category slugs for a candidate-family answer map."""
  known = set(research_category_slugs())
  answered = {k for k, v in answers.items() if v not in (None, "", [], {}, ())}
  return {
    "missing": [c.slug for c in RESEARCH_CATEGORIES if c.slug not in answered],
    "unknown": sorted(k for k in answers if k not in known),
  }
