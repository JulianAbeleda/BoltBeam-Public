# Default-Path Handwritten Kernel Audit

Date: 2026-07-01.

## Verdict

`DEFAULT_PATH_HANDWRITTEN_KERNEL_AUDIT_FAIL`

The current tinygrad default path is **not yet free of handwritten/specialized kernels**.

The important split:

- Q4_K decode GEMV is the promoted generated G3 route. This is the healthy direction.
- 8B long-context decode attention still defaults to an owned external HIP/AMDGCN tile.
- Prefill graph GEMM uses a specialized assembly emitter when the prefill-v2 graph-GEMM route is active.
- Q6_K coop is tinygrad UOp code, not an external HIP kernel, but it is still a hand-authored route template rather
  than machine-authored from BoltBeam's quant/shape grammar.

So the strict claim:

```text
handwritten kernels are not on the default path
```

is currently false.

## Definitions Used By This Audit

| class | meaning | default policy |
|---|---|---|
| `machine_authored_generated` | emitted from a profile/grammar/search-selected template, with route-bound evidence | allowed |
| `tinygrad_scheduler_generated` | ordinary tinygrad graph lowering, no custom route kernel | allowed |
| `hand_authored_uop_template` | Python `custom_kernel` UOp factory written by humans, no external HIP/ASM | tolerated, but not final pure-search |
| `external_handwritten_kernel` | HIP/ASM/assembly/custom object or explicit instruction emitter used as a route kernel | not allowed as the final pure-search default |
| `rollback_oracle` | handwritten/specialized route retained only behind a rollback flag or diagnostic flag | allowed |

This audit is stricter than the older census: it treats "pinned state" as insufficient. A default route may be fast
and correct and still fail the pure-default policy if it is externally handwritten.

## Evidence

The tinygrad census tool was run:

```text
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py
```

Result:

```text
PMS_R0_PASS_CENSUS_PINNED
4 kernels on the default path are non-tinygrad-generated.
1 is search/codegen-generated (G3 Q4_K GEMV);
3 are hand-owned (Q6_K coop, owned attention two-kernel, prefill pipe assembly).
Everything else in the model is tinygrad_scheduler-generated.
```

The census is a state pin, not a purity pass. This audit applies the stricter policy.

## Default Route Classification

| route | default status | class | evidence | audit result |
|---|---:|---|---|---|
| `decode_q4k_g3_generated` | default | `machine_authored_generated` | `tinygrad/llm/model.py` defaults `BUBBLEBEAM_FUTURESIGHT=1`; route calls `extra/qk_gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel`; BoltBeam candidate `decode_q4k_g3_anyshape` is shipped | pass |
| `decode_q6k_coop_shipped` | default | `hand_authored_uop_template` | `tinygrad/llm/model.py` defaults `Q6K_LM_HEAD_COOP=1`, `Q6K_FFN_DOWN_COOP=1`, and `DECODE_Q6K_FFN_DOWN_LONGK=1`; route calls `extra/q6_k_gemv_primitive.py q6k_coop_partial_kernel` | tolerated, not pure |
| `decode_attention_owned_two_kernel` | retired | `external_handwritten_kernel` | the old handwritten HIP attention implementation has been pruned; 8B long-context decode now defaults to the generated live-split KV_BOTH route | resolved |
| `prefill_pipe_role_selective_default` | conditional default when `PREFILL_V2`/graph-GEMM route is active | `external_handwritten_kernel` / specialized assembly emitter | `extra/qk_prefill_graph_gemm_route.py` defaults `PREFILL_GEMM_PIPELINE=1` and `PREFILL_PIPE_ROLE_SELECTIVE=1`; route uses `build_gemm_pipe(...)` and `Tensor.custom_kernel` with assembly instruction emission | fail for pure-default, kept for performance until generated replacement passes |

## Why I Did Not Flip Defaults In This Audit

Flipping these flags would make the default path purer but knowingly slower:

| flag | purity effect | known risk |
|---|---|---|
| generated 8B live-split route disabled | leaves the promoted generated attention path | use only as rollback/debug, not as a purity strategy |
| `PREFILL_GEMM_PIPELINE=0` | disables the specialized prefill pipe | loses the promoted prefill throughput improvement |
| `Q6K_*_COOP=0` / `DECODE_Q6K_FFN_DOWN_LONGK=0` | disables hand-authored UOp Q6_K coop routes | regresses 14B/32B decode where L3 was the shipped win |

Per the project discipline, a purity replacement must pass the same route-bound/token-match/W==D gates before it
becomes default. A purity flag can exist for experiments, but the shipped default should not be changed blindly.

## Required Fix Path

### 1. Add A Default-Purity Gate To BoltBeam

BoltBeam should grow a policy check that classifies shipped/default candidates by provenance:

```text
machine_authored_generated
tinygrad_scheduler_generated
hand_authored_uop_template
external_handwritten_kernel
rollback_oracle
```

Promotion rule:

- `external_handwritten_kernel` cannot be promoted as a final pure-search default.
- `hand_authored_uop_template` can be a temporary shipped route only with a tracked replacement scope.
- handwritten routes are allowed as rollback/oracle paths.

### 2. Decode Attention Replacement

Current blocker:

- `decode_attention_owned_two_kernel` is external HIP and default-on for Qwen3-8B/gfx1100 long context.

Replacement path:

- continue the generated-ISA primitive route, not a handwritten kernel:
  `docs/g5-generated-isa-primitive-route-scope-20260701.md`;
- require generated route-bound evidence and no protected-context regression before flipping default.

### 3. Prefill GEMM Replacement

Current blocker:

- role-selective prefill pipe is specialized assembly emission.

Replacement path:

- encode the prefill GEMM schedule as a generated/search-authored schedule template rather than a fixed assembly
  route;
- keep the current pipe as rollback until the generated route matches or beats it.

### 4. Q6_K Route Generalization

Current blocker:

- Q6_K coop route is not external HIP, but it is still a hand-authored UOp route template.

Replacement path:

- extend the TG/TG7 grammar-backed quant route author to cover Q6_K coop/long-K topology;
- only replace the shipped route after W==D proves no regression.

## Current Safe Statement

The accurate statement today is:

```text
Most of the model is tinygrad-generated, Q4_K decode weight math is generated/search-promoted,
but the default path still contains handwritten/specialized hot routes for attention and prefill,
plus hand-authored UOp templates for Q6_K.
```

The target statement is:

```text
All default hot routes are either ordinary tinygrad-generated or machine-authored/generated from BoltBeam search;
handwritten kernels remain only as rollback oracles.
```

We are not there yet.
