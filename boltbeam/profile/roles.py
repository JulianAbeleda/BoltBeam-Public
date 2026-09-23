"""Tensor-name -> role classification (audit A1).

`profile/` owns role classification. This module turns a GGUF tensor name into a FINE `RoleClass`
(the richest role model), from which the coarse grouped role used by the search space is DERIVED via
`boltbeam.vocab.role_group_of`. The vocabulary and the fine->coarse map are centralized in vocab.py; only
the name PATTERNS live here.

Ordering matters: MoE tensor names are supersets of dense names (`ffn_gate_exps`, `ffn_gate_inp`,
`ffn_gate_shexp` all contain `ffn_gate`), so MoE patterns are matched before the dense fallbacks. Attention
patterns are anchored on `.weight` so per-head norms (`attn_q_norm.weight`) fall through to `norm`, exactly
as the shipped dense classifier did.
"""
from __future__ import annotations

from boltbeam.vocab import RoleClass, role_group_of


def _ssm_role(name:str) -> RoleClass | None:
  n = name.lower()
  if not any(tok in n for tok in ("ssm", "delta", "conv1d", "time_step", "dt_proj", "a_log", "d_inner")):
    return None
  if "norm" in n:
    return None
  if "conv" in n:
    return RoleClass.SSM_CONV
  parts = n.replace(".weight", "").replace(".bias", "").split(".")
  if any(tok in n for tok in ("a_log", "dt", "time_step", "state", "d_inner")) or parts[-1] == "ssm_a":
    return RoleClass.SSM_STATE
  if "scan" in n:
    return RoleClass.SSM_SCAN
  return RoleClass.SSM_PROJECTION

# (substring, RoleClass) checked in order; first match wins.
_PATTERNS: tuple[tuple[str, RoleClass], ...] = (
  # --- MoE experts (3D stacks) and shared experts must precede the dense ffn_* fallbacks ---
  ("ffn_gate_exps", RoleClass.MOE_EXPERT_GATE),
  ("ffn_up_exps", RoleClass.MOE_EXPERT_UP),
  ("ffn_down_exps", RoleClass.MOE_EXPERT_DOWN),
  ("ffn_gate_shexp", RoleClass.MOE_SHARED_EXPERT_GATE),
  ("ffn_up_shexp", RoleClass.MOE_SHARED_EXPERT_UP),
  ("ffn_down_shexp", RoleClass.MOE_SHARED_EXPERT_DOWN),
  ("ffn_gate_inp", RoleClass.MOE_ROUTER),          # the top-k router projection
  # --- dense FFN ---
  ("ffn_gate", RoleClass.FFN_GATE),
  ("ffn_up", RoleClass.FFN_UP),
  ("ffn_down", RoleClass.FFN_DOWN),
  ("gate_proj", RoleClass.FFN_GATE),
  ("up_proj", RoleClass.FFN_UP),
  ("down_proj", RoleClass.FFN_DOWN),
  # --- attention (anchored so *_norm falls through) ---
  ("attn_q.weight", RoleClass.ATTENTION_Q),
  ("attn_k.weight", RoleClass.ATTENTION_K),
  ("attn_v.weight", RoleClass.ATTENTION_V),
  ("attn_output", RoleClass.ATTENTION_O),
  ("q_proj.", RoleClass.ATTENTION_Q),
  ("k_proj.", RoleClass.ATTENTION_K),
  ("v_proj.", RoleClass.ATTENTION_V),
  ("o_proj.", RoleClass.ATTENTION_O),
  ("out_proj.", RoleClass.ATTENTION_O),
)


def classify_tensor_role(name:str) -> RoleClass:
  """Fine RoleClass for a GGUF tensor name."""
  ssm = _ssm_role(name)
  if ssm is not None:
    return ssm
  if name == "output.weight" or name.endswith("lm_head.weight"):
    return RoleClass.LM_HEAD
  for token, rc in _PATTERNS:
    if token in name:
      return rc
  if "token_embd" in name or "embed_tokens" in name:
    return RoleClass.EMBEDDING
  if name.endswith("_norm.weight") or ".norm" in name or "layernorm" in name or "layer_norm" in name:
    return RoleClass.NORM
  return RoleClass.OTHER


def role_from_tensor_name(name:str) -> str:
  """Coarse grouped role (the derived view) for a GGUF tensor name. Byte-compatible with the shipped dense
  classifier for all Qwen/Llama dense names."""
  return role_group_of(classify_tensor_role(name).value)


def is_expert_role(role_class_or_group:str) -> bool:
  """True for a per-expert weight role/group (excludes the router). Used by MoE search semantics (A5)."""
  return "expert" in role_class_or_group
