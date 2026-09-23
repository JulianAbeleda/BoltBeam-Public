"""Pure llama MMQ epoch oracle for bounded BoltBeam search questions.

The data here is a source-anchored expectation model, not runtime evidence. It
describes the ordered lifecycle documented in
``docs/mmq-epoch-model-exhaustive-scope-20260710.md`` for the two bounded
prefill shapes BoltBeam currently reasons about without importing tinygrad or
touching hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_SCOPE = "docs/mmq-epoch-model-exhaustive-scope-20260710.md"

EPOCH_ORDER: tuple[str, ...] = (
  "launch_ownership",
  "q4k_tile_x_load_decode",
  "q8_tile_y_stage",
  "visibility_sync",
  "dot_accumulate",
  "k_advance",
  "stage_reuse_or_overwrite",
  "writeback",
  "epilogue",
)


@dataclass(frozen=True)
class MMQWorkload:
  m: int
  n: int
  k: int

  @property
  def shape_id(self) -> str:
    return f"{self.m}x{self.n}x{self.k}"


@dataclass(frozen=True)
class LlamaMMQEpoch:
  epoch_id: str
  kind: str
  source_anchors: tuple[str, ...]
  expected_behavior: str
  ownership_claim: str | None = None
  reuse_claim: str | None = None
  sync_before: tuple[str, ...] = ()
  sync_after: tuple[str, ...] = ()
  resource_claims: tuple[str, ...] = ()
  spec: Any | None = None


@dataclass(frozen=True)
class LlamaMMQEpochSequence:
  workload: MMQWorkload
  epochs: tuple[LlamaMMQEpoch, ...]
  source_scope: str = SOURCE_SCOPE

  @property
  def epoch_ids(self) -> tuple[str, ...]:
    return tuple(epoch.epoch_id for epoch in self.epochs)

  def by_id(self, epoch_id: str) -> LlamaMMQEpoch:
    for epoch in self.epochs:
      if epoch.epoch_id == epoch_id:
        return epoch
    raise KeyError(epoch_id)


_SUPPORTED_WORKLOADS: tuple[MMQWorkload, ...] = (
  MMQWorkload(16, 16, 256),
  MMQWorkload(128, 128, 256),
)


def supported_llama_mmq_workloads() -> tuple[MMQWorkload, ...]:
  return _SUPPORTED_WORKLOADS


def _maybe_epoch_spec(epoch_id: str) -> Any | None:
  """Reuse E0 epoch specs when present, but keep this helper independently pure."""
  try:
    from boltbeam.search.epochs.epoch_model import epoch_spec_by_id  # type: ignore
  except Exception:
    return None
  try:
    return epoch_spec_by_id(epoch_id)
  except Exception:
    return None


def _normalize_workload(m: int, n: int, k: int) -> MMQWorkload:
  workload = MMQWorkload(m, n, k)
  if workload not in _SUPPORTED_WORKLOADS:
    supported = ", ".join(w.shape_id for w in _SUPPORTED_WORKLOADS)
    raise ValueError(f"unsupported llama MMQ oracle workload {workload.shape_id}; supported: {supported}")
  return workload


def _ownership_claim(workload: MMQWorkload) -> str:
  if workload.shape_id == "16x16x256":
    return (
      "One 8-wave CTA owns one 16x16 output tile; each wave owns one 16-row output stripe "
      "for a 16-column fragment, and each output element has exactly one writeback owner."
    )
  return (
    "The 128x128 tile is decomposed into 16x16 fragments; each fragment keeps the llama MMQ "
    "8-wave CTA ownership rule, so 64 fragments cover the tile with exactly one owner per output element."
  )


def _resource_claims(workload: MMQWorkload) -> tuple[str, ...]:
  fragments = (workload.m // 16) * (workload.n // 16)
  return (
    f"k_panel_depth={workload.k}",
    "n_fragment_width=16",
    "wave_count_per_cta=8",
    "output_stripe_rows_per_wave=16",
    f"output_fragments_16x16={fragments}",
    "resource formulas are source-level expectations only; no VGPR/LDS/hardware measurement is implied",
  )


def llama_mmq_epoch_sequence(m: int, n: int, k: int) -> LlamaMMQEpochSequence:
  """Return the bounded source-anchored llama MMQ epoch expectation sequence."""
  workload = _normalize_workload(m, n, k)
  ownership = _ownership_claim(workload)
  resources = _resource_claims(workload)
  rows: tuple[dict[str, Any], ...] = (
    {
      "epoch_id": "launch_ownership",
      "source_anchors": ("mul_mat_q", "mul_mat_q_process_tile"),
      "expected_behavior": "Launch maps CTA, wave, and lane identity onto the output tile before any K work.",
      "ownership_claim": ownership,
      "resource_claims": resources,
    },
    {
      "epoch_id": "q4k_tile_x_load_decode",
      "source_anchors": ("load_tiles_q4_K",),
      "expected_behavior": "Cooperative load/decode of packed Q4_K fields into shared tile_x.",
      "reuse_claim": "Decoded tile_x is staged once per K panel and consumed by dot work before overwrite.",
      "resource_claims": ("packed_q4_k_bytes", "q4_k_scales", "q4_k_mins", "shared_tile_x"),
    },
    {
      "epoch_id": "q8_tile_y_stage",
      "source_anchors": ("load_tiles_*", "block_q8_1_mmq"),
      "expected_behavior": "Q8_1/DS4 activation panel is arranged for contiguous shared-memory copies.",
      "reuse_claim": "Staged tile_y is reused across the 16-column fragment consumers in the current K panel.",
      "resource_claims": ("q8_1_activation_panel", "shared_tile_y"),
    },
    {
      "epoch_id": "visibility_sync",
      "source_anchors": ("__syncthreads()", "shared-memory barriers"),
      "expected_behavior": "Producers finish tile staging before consumers issue dot work.",
      "sync_before": ("q4k_tile_x_load_decode", "q8_tile_y_stage"),
      "sync_after": ("dot_accumulate",),
    },
    {
      "epoch_id": "dot_accumulate",
      "source_anchors": ("vec_dot_q4_K_q8_1_impl_mmq",),
      "expected_behavior": "Dot work consumes tile_x/tile_y and accumulates into per-thread sum[] slots.",
      "reuse_claim": "The staged K panel is consumed without reloading global Q4/Q8 data for each output.",
      "sync_before": ("visibility_sync",),
      "resource_claims": ("live_sum_slots", "staged_tile_x", "staged_tile_y"),
    },
    {
      "epoch_id": "k_advance",
      "source_anchors": ("for (k0...)", "MMQ_ITER_K"),
      "expected_behavior": "K advances in bounded panels while preserving live accumulator identity.",
      "reuse_claim": "sum[] remains live across panel advances until writeback.",
      "sync_before": ("dot_accumulate",),
      "sync_after": ("stage_reuse_or_overwrite",),
    },
    {
      "epoch_id": "stage_reuse_or_overwrite",
      "source_anchors": ("shared tile reuse before next panel",),
      "expected_behavior": "Staged data is reused across many outputs, then safely invalidated or overwritten.",
      "reuse_claim": "tile_x/tile_y may be overwritten only after all current-panel consumers have completed.",
      "sync_before": ("k_advance",),
      "sync_after": ("q4k_tile_x_load_decode", "q8_tile_y_stage"),
    },
    {
      "epoch_id": "writeback",
      "source_anchors": ("mmq_write_back_mma", "mmq_write_back_dp4a"),
      "expected_behavior": "One owner stores each output element from its accumulator identity.",
      "ownership_claim": "Writeback preserves the launch ownership claim: exactly one store owner per output element.",
      "sync_before": ("dot_accumulate", "k_advance"),
    },
    {
      "epoch_id": "epilogue",
      "source_anchors": ("kernel tail",),
      "expected_behavior": "Outstanding stores complete and the kernel exits.",
      "sync_before": ("writeback",),
    },
  )
  epochs = tuple(
    LlamaMMQEpoch(kind=row["epoch_id"], spec=_maybe_epoch_spec(row["epoch_id"]), **row)
    for row in rows
  )
  return LlamaMMQEpochSequence(workload=workload, epochs=epochs)


def llama_mmq_epoch_by_id(epoch_id: str, *, m: int = 16, n: int = 16, k: int = 256) -> LlamaMMQEpoch:
  return llama_mmq_epoch_sequence(m, n, k).by_id(epoch_id)


__all__ = [
  "EPOCH_ORDER",
  "LlamaMMQEpoch",
  "LlamaMMQEpochSequence",
  "MMQWorkload",
  "SOURCE_SCOPE",
  "llama_mmq_epoch_by_id",
  "llama_mmq_epoch_sequence",
  "supported_llama_mmq_workloads",
]
