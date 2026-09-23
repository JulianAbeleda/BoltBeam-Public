"""KV-cache context-slope math.

This module is deliberately target/model constant-free. Callers supply the measured slope, the storage bytes,
and the baseline bandwidth/slope. The key distinction is capacity vs speed: quantized KV may shrink storage
while adding residual unpack/dequant/dot work in the attention kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

from boltbeam.vocab import KVPolicyClass


def storage_slope_ms_per_ctx(kv_bytes_per_ctx_token:float, bandwidth_bytes_per_s:float) -> float:
  """Storage-only B term in `ms/token = A + B*ctx`."""
  _require_positive("kv_bytes_per_ctx_token", kv_bytes_per_ctx_token)
  _require_positive("bandwidth_bytes_per_s", bandwidth_bytes_per_s)
  return kv_bytes_per_ctx_token / bandwidth_bytes_per_s * 1000.0


def quant_residual_ms_per_ctx(measured_b_ms_per_ctx:float, storage_b_ms_per_ctx:float) -> float:
  """Quantized-KV residual: measured slope minus storage-only slope."""
  return float(measured_b_ms_per_ctx) - float(storage_b_ms_per_ctx)


def predict_decode_tok_s(a_ms:float, b_ms_per_ctx:float, ctx:int) -> float:
  """Predicted decode tok/s from a linear context-slope fit."""
  if ctx < 0: raise ValueError(f"ctx must be >= 0, got {ctx!r}")
  denom = float(a_ms) + float(b_ms_per_ctx) * ctx
  _require_positive("predicted_ms_per_token", denom)
  return 1000.0 / denom


@dataclass(frozen=True)
class KVPolicyResult:
  policy_class: str
  storage_only_b_ms_per_ctx: float
  quant_residual_b_ms_per_ctx: float
  residual_share: float | None
  reason: str


def classify_kv_slope(*, measured_b_ms_per_ctx:float, storage_only_b_ms_per_ctx:float, r2:float,
                      kv_bytes_per_ctx_token:float, baseline_b_ms_per_ctx:float | None = None,
                      baseline_kv_bytes_per_ctx_token:float | None = None,
                      cache_type_k:str = "f16", cache_type_v:str = "f16",
                      mixed_fastpath:bool = True, whole_decode_delta_pct:float | None = None,
                      capacity_required:bool = False, min_r2:float = 0.97) -> KVPolicyResult:
  """Classify a KV cache dtype choice.

  `whole_decode_delta_pct` is required to call a quantized-KV route a speed win. A lower context-slope alone is
  not enough, because A and non-KV work can dominate whole-decode behavior.
  """
  _require_positive("measured_b_ms_per_ctx", measured_b_ms_per_ctx)
  _require_positive("storage_only_b_ms_per_ctx", storage_only_b_ms_per_ctx)
  _require_positive("kv_bytes_per_ctx_token", kv_bytes_per_ctx_token)
  if r2 < min_r2:
    return KVPolicyResult(KVPolicyClass.INCONCLUSIVE.value, storage_only_b_ms_per_ctx,
                          quant_residual_ms_per_ctx(measured_b_ms_per_ctx, storage_only_b_ms_per_ctx),
                          None, f"slope fit R2 {r2:.4f} is below {min_r2:.2f}")

  if cache_type_k != cache_type_v and not mixed_fastpath:
    return KVPolicyResult(KVPolicyClass.MIXED_FASTPATH_UNKNOWN.value, storage_only_b_ms_per_ctx,
                          quant_residual_ms_per_ctx(measured_b_ms_per_ctx, storage_only_b_ms_per_ctx),
                          None, "mixed K/V cache dtype requested but target lacks a mixed quantized FA fastpath")

  residual = quant_residual_ms_per_ctx(measured_b_ms_per_ctx, storage_only_b_ms_per_ctx)
  residual_share = residual / measured_b_ms_per_ctx if measured_b_ms_per_ctx else None

  storage_win = baseline_kv_bytes_per_ctx_token is not None and kv_bytes_per_ctx_token < baseline_kv_bytes_per_ctx_token
  speed_loss = baseline_b_ms_per_ctx is not None and measured_b_ms_per_ctx > baseline_b_ms_per_ctx
  slope_speed_win = baseline_b_ms_per_ctx is not None and measured_b_ms_per_ctx < baseline_b_ms_per_ctx

  if storage_win and speed_loss:
    if capacity_required:
      return KVPolicyResult(KVPolicyClass.CAPACITY_ONLY.value, storage_only_b_ms_per_ctx, residual,
                            residual_share, "KV storage shrinks and enables capacity, but measured slope is slower")
    return KVPolicyResult(KVPolicyClass.STORAGE_WIN_SPEED_LOSS.value, storage_only_b_ms_per_ctx, residual,
                          residual_share, "KV storage shrinks but measured context slope is slower than baseline")

  if storage_win and slope_speed_win:
    if whole_decode_delta_pct is not None and whole_decode_delta_pct > 0:
      return KVPolicyResult(KVPolicyClass.STORAGE_WIN_SPEED_WIN.value, storage_only_b_ms_per_ctx, residual,
                            residual_share, "KV storage shrinks and whole-decode speed improves")
    return KVPolicyResult(KVPolicyClass.INCONCLUSIVE.value, storage_only_b_ms_per_ctx, residual, residual_share,
                          "slope improved but whole-decode W==D speed evidence is missing")

  if capacity_required and storage_win:
    return KVPolicyResult(KVPolicyClass.CAPACITY_ONLY.value, storage_only_b_ms_per_ctx, residual, residual_share,
                          "KV storage shrinks for capacity; speed is not proven")

  return KVPolicyResult(KVPolicyClass.INCONCLUSIVE.value, storage_only_b_ms_per_ctx, residual, residual_share,
                        "KV slope does not prove a speed or capacity decision")


def _require_positive(name:str, value:float) -> None:
  if value <= 0.0:
    raise ValueError(f"{name} must be > 0, got {value!r}")

