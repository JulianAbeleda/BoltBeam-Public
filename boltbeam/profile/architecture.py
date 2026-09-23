"""Architecture classification (audit A2).

Derive an `Architecture` class from GGUF metadata first, tensor-graph facts second. This never raises: an
input it cannot confidently place returns `unknown_transformer`, which the profile carries honestly rather
than silently treating as a dense decoder.

Signals (metadata beats tensors):
  Hybrid SSM    `<arch>.ssm.*` metadata, OR SSM/DeltaNet tensor names. If it also has MoE signals, the class is
                `hybrid_moe_decoder`; otherwise `hybrid_decoder`.
  MoE           `<arch>.expert_count > 1`, OR expert-stack tensors (`*_exps`), OR a router (`ffn_gate_inp`),
                OR any 3D weight tensor (expert stacks are `[n_expert, out, in]`).
  encoder-dec   a known enc-dec arch name, OR both encoder (`enc.` / `.encoder.`) and decoder tensors.
  dense-dec     decoder-only with attention AND dense FFN tensors.
  unknown       none of the above hold with confidence.
"""
from __future__ import annotations

from typing import Any

from boltbeam.vocab import Architecture

# arch strings that are encoder-decoder regardless of tensor layout (metadata-first)
_ENCODER_DECODER_ARCHS = frozenset({"t5", "t5encoder", "mt5", "bart", "bert"})


def _to_int(v:Any) -> int | None:
  try:
    return int(v)
  except (TypeError, ValueError):
    return None


def _has_ssm_metadata(kv:dict[str, Any], arch:str | None) -> bool:
  if not arch:
    return False
  prefix = f"{arch}.ssm."
  return any(k.startswith(prefix) for k in kv)


def _name_has_ssm_signal(name:str) -> bool:
  n = name.lower()
  return any(tok in n for tok in ("ssm", "delta", "conv1d", "time_step", "dt_proj", "a_log", "d_inner"))


def _name_has_packed_quant_signal(name:str) -> bool:
  n = name.lower()
  return any(tok in n for tok in ("qweight", "qzeros", "matmulnbits", "scales", "g_idx"))


def classify_architecture(kv:dict[str, Any], tensors:list[tuple[str, tuple[int, ...], int, int]],
                          arch:str | None = None) -> tuple[str, dict[str, Any]]:
  """Return (architecture_class, signals). `signals` records what fired, for the profile metadata and for
  honest incompleteness reporting. `arch` defaults to kv['general.architecture']."""
  arch = arch if arch is not None else kv.get("general.architecture")
  names = [t[0] for t in tensors]
  expert_count = _to_int(kv.get(f"{arch}.expert_count")) if arch else None
  expert_used_count = _to_int(kv.get(f"{arch}.expert_used_count")) if arch else None
  has_exps = any("_exps" in n for n in names)
  has_router = any("ffn_gate_inp" in n for n in names)
  has_3d = any(len(dims) == 3 and not _name_has_ssm_signal(_n) and not _name_has_packed_quant_signal(_n)
               for _n, dims, _t, _o in tensors)
  has_ssm = _has_ssm_metadata(kv, arch) or any(_name_has_ssm_signal(n) for n in names)
  has_enc = any(n.startswith("enc.") or ".encoder." in n for n in names)
  has_dec = any(n.startswith("dec.") or ".decoder." in n for n in names)
  has_attn = any(("attn_" in n) or ("self_attn" in n) or ("q_proj" in n) or ("k_proj" in n) or ("v_proj" in n)
                 for n in names)
  has_ffn = any(("ffn_down" in n) or ("ffn_up" in n) or ("ffn_gate" in n) or ("down_proj" in n)
                or ("up_proj" in n) or ("gate_proj" in n) for n in names)

  signals = {
    "arch": arch, "expert_count": expert_count, "expert_used_count": expert_used_count,
    "has_expert_stacks": has_exps, "has_router": has_router,
    "has_3d_tensors": has_3d, "has_ssm": has_ssm,
    "has_encoder_tensors": has_enc, "has_decoder_tensors": has_dec,
    "has_attention": has_attn, "has_dense_ffn": has_ffn,
  }

  # metadata-first: an explicit expert count > 1, or an enc-dec arch name
  has_moe = (expert_count is not None and expert_count > 1) or has_exps or has_router or has_3d
  if has_ssm and has_moe:
    return Architecture.HYBRID_MOE_DECODER.value, signals
  if has_ssm:
    return Architecture.HYBRID_DECODER.value, signals
  if expert_count is not None and expert_count > 1:
    return Architecture.MOE_DECODER.value, signals
  if arch in _ENCODER_DECODER_ARCHS:
    return Architecture.ENCODER_DECODER.value, signals
  # tensor-shape fallbacks
  if has_exps or has_router or has_3d:
    return Architecture.MOE_DECODER.value, signals
  if has_enc and has_dec:
    return Architecture.ENCODER_DECODER.value, signals
  if has_attn and has_ffn:
    return Architecture.DENSE_DECODER.value, signals
  return Architecture.UNKNOWN_TRANSFORMER.value, signals
