from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from boltbeam.vocab import SCHEMA_MODEL_PROFILE


@dataclass(frozen=True)
class TensorRole:
  role: str                       # coarse grouped role (the derived view: attn_kv, ffn_gate_up, ...)
  tensor_name: str
  rows: int
  cols: int
  quant: str
  ggml_type: int
  count: int = 1
  role_class: str = ""            # fine RoleClass of the representative tensor (audit A1); "" = unclassified
  n_expert: int = 0               # MoE expert-stack depth (audit A5); 0 = dense (2D) weight

  @property
  def shape(self) -> tuple[int, int]:
    return (self.rows, self.cols)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class TargetProfile:
  target_id: str
  backend: str
  wave_size: int
  lds_bytes_per_cu: int | None = None
  vram_bytes: int | None = None
  # audit A6 capability data — subgroup/vector-load/LDS/dot+dequant primitives + build status
  subgroup_size: int | None = None
  vector_load_bits: int | None = None
  dot_primitives: tuple[str, ...] = ()          # the FULL hardware dot vocabulary (what the chip can do)
  dot_primitives_lowered: tuple[str, ...] = ()  # subset with a GENERATED renderer/matcher lowering (search-usable
                                                # + fusable). dot_primitives - dot_primitives_lowered = the
                                                # "lower all" backlog (each is EMITTER_BLOCKED for the search).
  dequant_primitives: tuple[str, ...] = ()
  backend_status: str = "complete"
  compute_units: int | None = None              # CU/SM/cluster count used by profiler percent-of-peak reports.
  memory_bandwidth_gbs: float | None = None     # Sustained/advertised peak used by profiler roofline reports.
  peak_tflops: dict[str, float] | None = None   # dtype/primitive -> target peak TFLOP/s, e.g. fp32/fp16/int8.
  capabilities: dict[str, Any] | None = None

  def dot_vocabulary_backlog(self) -> tuple[str, ...]:
    """Hardware dot primitives that have no generated lowering yet — the substrate work to complete the vocab."""
    return tuple(p for p in self.dot_primitives if p not in set(self.dot_primitives_lowered))

  def peak_tflops_for(self, key: str) -> float | None:
    """Peak TFLOP/s for a dtype/primitive key, if the target descriptor carries it."""
    peaks = self.peak_tflops or {}
    value = peaks.get(key.lower())
    return float(value) if value is not None else None

  @property
  def is_complete(self) -> bool:
    return self.backend_status == "complete"

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
  model_id: str
  source: str
  architecture: str | None
  hidden_size: int | None
  ffn_size: int | None
  vocab_size: int | None
  layer_count: int | None
  roles: tuple[TensorRole, ...]
  metadata: dict[str, Any]
  architecture_class: str = "unknown_transformer"    # derived Architecture value (audit A2)

  @property
  def is_complete(self) -> bool:
    """A profile is incomplete if we could not place the architecture or found no searchable roles — the
    downstream must not treat it as a promotable dense model."""
    return self.architecture_class != "unknown_transformer" and bool(self.roles)

  def to_json(self) -> dict[str, Any]:
    return {
      "schema": SCHEMA_MODEL_PROFILE,
      "model_id": self.model_id,
      "source": self.source,
      "architecture": self.architecture,
      "architecture_class": self.architecture_class,
      "hidden_size": self.hidden_size,
      "ffn_size": self.ffn_size,
      "vocab_size": self.vocab_size,
      "layer_count": self.layer_count,
      "roles": [r.to_json() for r in self.roles],
      "complete": self.is_complete,
      "metadata": self.metadata,
    }
