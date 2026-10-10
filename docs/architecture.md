# BoltBeam Architecture

BoltBeam is intentionally split into stable boundaries:

```text
profile/  artifact readers, ProfileIR, role taxonomy, architecture classifier
search/   candidate-space emission, route-family knob grammar, reachability
synth/    shape-only fixtures for codegen proving grounds
eval/     promotion/evaluation contracts
policy/   runtime route-policy documents + source-scan policy guard
ledger/   durable append-only route ledger
data/     quant registry, target registry, candidate manifest (the authorities)
vocab.py  centralized vocabularies (roles, architectures, quant/backend status)
quant.py / targets.py   capability registry loaders
```

The first implementation supports GGUF profile extraction. Later readers can add
Safetensors, ONNX, or runtime-captured tinygrad graphs without changing the
search/policy contracts.

## Model-Agnostic Authorities (audit A0–A10)

Three data files are the extension points; adding a quant, target, or candidate
is a data edit, not a code branch:

- `data/quants.json` — quant behavior (block size, dequant family, route
  families, refuted/pending routes). Unknown quant -> `unsupported_quant`.
- `data/targets.json` — target capability + `backend_status`. Only `complete`
  backends promote; `descriptor_only` targets defer.
- `data/candidates.json` — route-family templates (arch-scoped, origin-tagged).

The decision path (search emission, route families, seed policy, evaluator) is
model/quant/target-agnostic and is source-scanned by `policy/guards.py`. An
`unknown_transformer` profile fails closed: no authorized search space, all
routes blocked. See [model-agnostic-search-roadmap.md](model-agnostic-search-roadmap.md).

Hybrid SSM/MoE profiles are classified from facts, not model names. SSM metadata
(`<arch>.ssm.*`) or tensor names (`ssm`, `delta`, `conv1d`, `dt_proj`, `a_log`)
produce `hybrid_decoder`; the same signal plus MoE expert/router metadata
produces `hybrid_moe_decoder`. Hybrid profiles inherit the dense route surface
for ordinary attention/FFN roles, hybrid-MoE profiles also inherit MoE expert
families, and SSM tensors get their own `ssm_*` route families.

Every weight the decode reads once per token is a role of the roofline limit,
named by what it is (`vocab.WEIGHT_GEMV_ROLES`): the dense roles, a fused
`attn_qkv` projection, the `attn_gate` beside it, and the SSM matrices split by
tensor (`ssm_out`; `ssm_alpha` and `ssm_beta` grouped as `ssm_alpha_beta`, as k
and v are). The per-role tables key on (role, quant), so two different tensors
of one quant are two names, never one row. These SSM matrices are quant GEMVs
and take the quant route families; conv, state and scan tensors keep the
`ssm_*` families. An SSM matrix the patterns do not name (`ssm_in`) stays
`ssm_projection`, outside the limit, and the tie-out names its GEMV after it in
other kernels.

## Why Not Generate A Full Synthetic GGUF First?

Synthetic GGUFs are useful for loader and route-policy testing, but the fastest
codegen loop is synthetic shape graphs:

```text
real GGUF -> ProfileIR
ProfileIR -> synthetic graph fixture
ProfileIR -> search candidates
ProfileIR -> runtime route policy
```

Real GGUFs remain the authority for promotion.
