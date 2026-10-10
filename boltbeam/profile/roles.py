"""Tensor-name -> role classification (audit A1).

`profile/` owns role classification. This module turns a GGUF tensor name into a FINE `RoleClass`
(the richest role model), from which the coarse grouped role used by the search space is DERIVED via
`boltbeam.vocab.role_group_of`. The vocabulary and the fine->coarse map are centralized in vocab.py; only
the name PATTERNS live here.

Ordering matters: MoE tensor names are supersets of dense names (`ffn_gate_exps`, `ffn_gate_inp`,
`ffn_gate_shexp` all contain `ffn_gate`), so MoE patterns are matched before the dense fallbacks. Attention
patterns are anchored on `.weight` so per-head norms (`attn_q_norm.weight`) fall through to `norm`, exactly
as the shipped dense classifier did.

Every weight a decode reads once per token is named by what it is, never left as `other`: a fused `attn_qkv`
projection is one role with that tensor's bytes, the `attn_gate` beside it another. The ssm family is classified by
the tensor's rank and shape, not by words in its name: a rank-two matrix (both sides at least VECTOR_SIDE) is a GEMV
role, `ssm_in`, `ssm_out` or `ssm_alpha`/`ssm_beta` (one shape, grouped as k and v are) when a pattern names it,
else a role named after its tensor (`blk.0.ssm_dt.weight` -> `ssm_dt`). A vector or a thin matrix (`ssm_a`, `ssm_d`,
the `ssm_dt` bias) is state, a conv tap is conv. Two different tensors of one quant so never share a (role, quant)
row in the limit (vocab.WEIGHT_GEMV_ROLES). Without a shape, an unnamed ssm tensor stays `ssm_projection`, outside
the limit.
"""
from __future__ import annotations

from boltbeam.vocab import RoleClass, role_group_of


VECTOR_SIDE = 16  # a matrix with a side under this is vector-like (conv taps 4, norm groups 8, a (1, n) scale)

_SSM_TOKENS = ("ssm", "delta", "conv1d", "time_step", "dt_proj", "a_log", "d_inner")


def is_matrix(dims:tuple[int, ...] | None) -> bool:
  """A rank-two tensor with both sides at least VECTOR_SIDE: what a decode reads as one matrix-vector product."""
  return dims is not None and len(dims) == 2 and min(int(d) for d in dims) >= VECTOR_SIDE


def _ssm_role(name:str, dims:tuple[int, ...] | None = None) -> RoleClass | None:
  n = name.lower()
  if not any(tok in n for tok in _SSM_TOKENS) or "norm" in n:
    return None
  if "conv" in n:
    return RoleClass.SSM_CONV
  if dims is not None and not is_matrix(dims):  # a vector or thin tensor: A, D, the dt bias, a scan buffer
    return RoleClass.SSM_SCAN if "scan" in n else RoleClass.SSM_STATE
  for token, rc in _SSM_MATRIX_PATTERNS:
    if token in n:
      return rc
  return RoleClass.SSM_PROJECTION


def ssm_tensor_role(name:str) -> str:
  """The role of an ssm matrix no pattern names, after its tensor: blk.0.ssm_dt.weight -> ssm_dt."""
  parts = [p for p in name.lower().split(".") if p not in ("weight", "bias") and not p.isdigit()]
  stem = (parts[-1] if parts else name.lower()).replace("-", "_")
  return stem if stem.startswith("ssm_") else f"ssm_{stem}"


# ssm matrices named by tensor (checked in order inside _ssm_role, after conv and the vector-like state tensors)
_SSM_MATRIX_PATTERNS: tuple[tuple[str, RoleClass], ...] = (
  ("ssm_in", RoleClass.SSM_IN),
  ("ssm_out", RoleClass.SSM_OUT),
  ("ssm_alpha", RoleClass.SSM_ALPHA),
  ("ssm_beta", RoleClass.SSM_BETA),
)

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
  ("attn_qkv.weight", RoleClass.ATTENTION_QKV),
  ("attn_gate.weight", RoleClass.ATTENTION_GATE),
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


def classify_tensor_role(name:str, dims:tuple[int, ...] | None = None) -> RoleClass:
  """Fine RoleClass for a GGUF tensor name; dims (the file's shape) decide matrix from vector in the ssm family."""
  ssm = _ssm_role(name, dims)
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


def role_from_tensor_name(name:str, dims:tuple[int, ...] | None = None) -> str:
  """Coarse grouped role (the derived view) for a GGUF tensor name. Byte-compatible with the shipped dense
  classifier for all Qwen/Llama dense names. An ssm matrix of known shape that no pattern names is named after its
  tensor (ssm_tensor_role); vocab.is_weight_gemv_role counts it in the limit."""
  rc = classify_tensor_role(name, dims)
  if rc is RoleClass.SSM_PROJECTION and is_matrix(dims):
    return ssm_tensor_role(name)
  return role_group_of(rc.value)


def is_expert_role(role_class_or_group:str) -> bool:
  """True for a per-expert weight role/group (excludes the router). Used by MoE search semantics (A5)."""
  return "expert" in role_class_or_group
