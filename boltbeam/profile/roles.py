"""Tensor-name -> role classification (audit A1).

`profile/` owns role classification. This module turns a GGUF tensor name into a FINE `RoleClass`
(the richest role model), from which the coarse grouped role used by the search space is DERIVED via
`boltbeam.vocab.role_group_of`. The vocabulary and the fine->coarse map are centralized in vocab.py; only
the name PATTERNS live here.

A pattern matches whole components of the tensor's dotted name, never a substring of one: `ffn_up` is the
component `ffn_up` (`blk.0.ffn_up.weight`), not `hc_ffn_up`, and `q_proj` is `self_attn.q_proj.qweight`'s, not
`qkv_proj`. A per-head norm (`attn_q_norm`) is its own component and falls through to `norm`. Two families are named
by their own component before any pattern: a hyper-connection mixer's tensors (`hc_*`, `output_hc_*`) and a sparse
attention indexer's (`indexer.*`, `indexer_*`); their norms stay `norm`. (Flash-Next's `hc_ffn_up`/`hc_ffn_down` were
the FFN's up and down and its `indexer.q_proj`/`k_proj` attention's q and k while the patterns were substrings.)

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

from boltbeam.vocab import RoleClass, RoleClass as R, role_group_of


VECTOR_SIDE = 16  # a matrix with a side under this is vector-like (conv taps 4, norm groups 8, a (1, n) scale)

_SSM_TOKENS = ("ssm", "delta", "conv1d", "time_step", "dt_proj", "dt_bias", "a_log", "d_inner",
               "in_proj_a.", "in_proj_b.", "linear_attn.out_proj.", "mixer.in_proj.", "mixer.out_proj.")


def is_matrix(dims:tuple[int, ...] | None) -> bool:
  """A rank-two tensor with both sides at least VECTOR_SIDE: what a decode reads as one matrix-vector product."""
  return dims is not None and len(dims) == 2 and min(int(d) for d in dims) >= VECTOR_SIDE


def _component_role(parts:list[str], table:dict[str, RoleClass]) -> RoleClass | None:
  """The role of the first component of the name the table names whole (`mixer.in_proj` as one); an anchored GGUF
  component counts only before `weight`."""
  for i, p in enumerate(parts):
    for key in (".".join(parts[i:i + 2]), p):
      if key in table and (key not in _ANCHORED or parts[i + 1:i + 2] == ["weight"]):
        return table[key]
  return None


def _family(parts:list[str]) -> RoleClass | None:
  """A tensor of a family named by its own component: a hyper-connection mixer or a sparse attention indexer."""
  own = [p for p in parts if p.startswith(("hc_", "output_hc_", "indexer"))]
  if not own or any(p.endswith("_norm") for p in parts):  # its norms are norms
    return None
  return RoleClass.ATTENTION_INDEXER if own[0].startswith("indexer") else RoleClass.HYPER_CONNECTION


def _ssm_role(name:str, dims:tuple[int, ...] | None = None) -> RoleClass | None:
  n = name.lower()
  if not any(tok in n for tok in _SSM_TOKENS) or "norm" in n:
    return None
  if "conv" in n:
    return RoleClass.SSM_CONV
  if dims is not None and not is_matrix(dims):  # a vector or thin tensor: A, D, the dt bias, a scan buffer
    return RoleClass.SSM_SCAN if "scan" in n else RoleClass.SSM_STATE
  return _component_role(n.split("."), _SSM_MATRIX_ROLES) or RoleClass.SSM_PROJECTION


def ssm_tensor_role(name:str) -> str:
  """The role of an ssm matrix no pattern names, after its tensor: blk.0.ssm_dt.weight -> ssm_dt."""
  parts = [p for p in name.lower().split(".") if p not in ("weight", "bias") and not p.isdigit()]
  stem = (parts[-1] if parts else name.lower()).replace("-", "_")
  return stem if stem.startswith("ssm_") else f"ssm_{stem}"


# ssm matrices named by tensor (checked in order inside _ssm_role, after conv and the vector-like state tensors)
_SSM_MATRIX_ROLES: dict[str, RoleClass] = {
  "ssm_in": R.SSM_IN, "mixer.in_proj": R.SSM_IN, "ssm_out": R.SSM_OUT, "ssm_alpha": R.SSM_ALPHA, "ssm_beta": R.SSM_BETA,
  "in_proj_a": R.SSM_ALPHA, "in_proj_b": R.SSM_BETA, "out_proj": R.SSM_OUT}

# a name component -> its RoleClass. A component is matched whole, so `ffn_gate` never takes `ffn_gate_exps` or
# `hc_ffn_up` `ffn_up`; _ANCHORED components (GGUF attention) only before `weight`: a bias is no projection
_ROLES: dict[str, RoleClass] = {
  "ffn_gate_exps": R.MOE_EXPERT_GATE, "ffn_up_exps": R.MOE_EXPERT_UP, "ffn_down_exps": R.MOE_EXPERT_DOWN,
  "ffn_gate_up_exps": R.MOE_EXPERT_GATE,  # gate and up in one expert stack
  "ffn_gate_shexp": R.MOE_SHARED_EXPERT_GATE, "ffn_up_shexp": R.MOE_SHARED_EXPERT_UP,
  "ffn_down_shexp": R.MOE_SHARED_EXPERT_DOWN, "ffn_gate_inp": R.MOE_ROUTER,  # the top-k router projection
  "ffn_gate": R.FFN_GATE, "ffn_gate_up": R.FFN_GATE, "ffn_up": R.FFN_UP, "ffn_down": R.FFN_DOWN,
  "gate_proj": R.FFN_GATE, "up_proj": R.FFN_UP, "down_proj": R.FFN_DOWN,
  "attn_qkv": R.ATTENTION_QKV, "attn_gate": R.ATTENTION_GATE, "attn_q": R.ATTENTION_Q, "attn_k": R.ATTENTION_K,
  "attn_v": R.ATTENTION_V, "cross_attn_q": R.ATTENTION_Q, "cross_attn_k": R.ATTENTION_K, "cross_attn_v": R.ATTENTION_V,
  "attn_kv_a_mqa": R.ATTENTION_KV_A, "attn_kv_b": R.ATTENTION_KV_B,
  "in_proj_qkv": R.ATTENTION_QKV, "in_proj_z": R.ATTENTION_GATE,  # a gated delta net's q, k and v and its output gate
  "attn_output": R.ATTENTION_O, "attn_output_a": R.ATTENTION_O, "attn_output_b": R.ATTENTION_O,  # a low-rank o's factors
  "q_proj": R.ATTENTION_Q, "k_proj": R.ATTENTION_K, "v_proj": R.ATTENTION_V, "o_proj": R.ATTENTION_O,
  "out_proj": R.ATTENTION_O}
_ANCHORED = frozenset(("attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v", "cross_attn_q", "cross_attn_k",
                       "cross_attn_v", "attn_kv_a_mqa", "attn_kv_b"))


def classify_tensor_role(name:str, dims:tuple[int, ...] | None = None) -> RoleClass:
  """Fine RoleClass for a GGUF tensor name; dims (the file's shape) decide matrix from vector in the ssm family."""
  ssm = _ssm_role(name, dims)
  if ssm is not None:
    return ssm
  if name == "output.weight" or name.endswith("lm_head.weight"):
    return RoleClass.LM_HEAD
  if ".experts." in name or ".shared_experts." in name:
    shared = ".shared_experts." in name
    for projection, expert, common in (
      ("gate_proj", RoleClass.MOE_EXPERT_GATE, RoleClass.MOE_SHARED_EXPERT_GATE),
      ("up_proj", RoleClass.MOE_EXPERT_UP, RoleClass.MOE_SHARED_EXPERT_UP),
      ("down_proj", RoleClass.MOE_EXPERT_DOWN, RoleClass.MOE_SHARED_EXPERT_DOWN),
    ):
      if f".{projection}." in name:
        return common if shared else expert
  if name.endswith(".mixer.gate.weight"):
    return RoleClass.MOE_ROUTER
  if found := _family(name.split(".")) or _component_role(name.split("."), _ROLES):
    return found
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
