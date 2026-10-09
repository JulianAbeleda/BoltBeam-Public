# TG-P10 Scope: REG Scalar Combine Lowering For Pure Attention

Date: 2026-07-01.

Goal: remove the final `TINYGRAD_DEFAULT_PURITY_FAIL` blocker by fixing the generic tinygrad AMD lowering gap that prevents a generated split-preserving attention combine from compiling correctly.

This is **not** a handwritten-kernel scope. The route remains:

```text
generic lowering primitive -> generated split-preserving combine -> generated 8B attention candidate -> W==D promotion gate
```

## Current State

TG-P9 split the 8B generated attention problem into two pieces:

| piece | result | evidence |
|---|---|---|
| live-context split geometry | solved | fixed `S`, runtime `per=ceildiv(Tc,S)` is expressible in generated UOp; ctx512 tile 36.6us -> 8.0us; full model 87.7% -> 96.7% of owned |
| split-preserving combine | blocked | every combine shape that shares weights or fuses gmax mis-lowers the reduction-accumulator `REG`; `REG_STORE_DEVEC=1` compiles but returns NaN |

The current full generated candidate is still below the promotion bar:

```text
ctx512: 96.7% of owned
ctx4096: 95.3% of owned
promotion bar: >=98% at both protected contexts
```

So owned HIP remains the 8B default. That is correct.

## What BoltBeam Can Do Today

BoltBeam already has most of the required audit machinery:

- candidate manifest and durable refutation/reopen rows: `boltbeam/data/candidates.json`;
- reachability classes: `REACHABLE`, `EMITTER_BLOCKED`, `PRIMITIVE_MISSING`, `REFUTED_BY_LEDGER`;
- compiler-pathology adapter for `tinygrad.compiler_pathology.v1`: `boltbeam/artifacts/tinygrad.py`;
- pathology classifier: `boltbeam/diagnostics/pathology.py`;
- conservative evaluator: multi-context protected regressions block promotion;
- policy/ledger discipline: generated candidates must prove route-bound correctness and speed before promotion.

What BoltBeam does **not** yet have first-class is a typed artifact for this exact compiler gap:

```text
REG reduction accumulator scalarization / devectorization correctness
```

Today the gap can be recorded as prose in `candidates.json`, but BoltBeam cannot yet mechanically classify:

- invalid `make_float4(...) = ...` store;
- `REG_STORE_DEVEC=1` compile-success but NaN output;
- scalar accumulator kept scalar vs mis-vectorized;
- whether a lowering fix reopens `decode_attention_g5_8b_refuted`.

TG-P10 should add that narrow typed audit path.

## Scope Boundary

BoltBeam responsibilities:

1. Normalize tinygrad compiler/lowering artifacts.
2. Classify the REG lowering failure as `EMITTER_BLOCKED`, not `REFUTED`.
3. Emit the precise reopen condition.
4. After tinygrad fixes lowering, ingest the proof artifact and mark the candidate reopened.
5. Evaluate full W==D evidence and only promote if the protected-context bar clears.

tinygrad responsibilities:

1. Build minimal generated-UOp repro kernels.
2. Fix generic AMD lowering / verifier / REG devectorization.
3. Run microgates and W==D gates.
4. Keep owned HIP as rollback/oracle until generated attention clears promotion.

Forbidden:

- new HIP/ASM/inline-ISA attention route;
- copying owned combine logic into an external kernel;
- defaulting generated attention while still below the speed bar;
- hardcoding Qwen3-8B tensor names instead of a generic lowering rule.

## Phase TG-P10.0: BoltBeam Audit Readiness

Purpose: confirm BoltBeam can represent the blocker without new conceptual machinery.

Audit:

- `boltbeam/vocab.py`
- `boltbeam/artifacts/tinygrad.py`
- `boltbeam/diagnostics/pathology.py`
- `boltbeam/search/reachability.py`
- `boltbeam/data/candidates.json`
- tests around compiler pathology and reachability

Expected answer:

```text
Enough for ledger/reachability/evaluation.
Needs a small typed compiler-lowering artifact and classifier extension for REG scalar combine lowering.
```

Verdicts:

- `TG_P10_0_PASS_BOLTBEAM_READY_WITH_SMALL_EXTENSION`
- `TG_P10_0_BLOCKED_NO_COMPILER_PATHOLOGY_PIPELINE`

## Phase TG-P10.1: Tinygrad REG Lowering Repro Artifact

Purpose: create the smallest generated-UOp repro that distinguishes the failure modes before touching full attention.

Create tinygrad artifact:

```text
schema: tinygrad.reg_scalar_lowering.v1
candidate_id: decode_attention_split_preserving_lse_combine
model_id: qwen3-8b-q4_k_m
target_id: amd_gfx1100
cases:
  - shipped_per_d_combine_compiles
  - shared_weight_combine_compile_fails
  - fused_gmax_combine_compile_fails
  - reg_store_devec_compiles_nan
```

Required fields per case:

```json
{
  "case_id": "...",
  "generated_uop_only": true,
  "uses_external_kernel": false,
  "compile_ok": false,
  "runtime_ok": false,
  "numeric_ok": false,
  "error_class": "invalid_reg_vector_store | nan_output | verifier_error | ok",
  "error_excerpt": "...",
  "reg_accumulator_expected": "scalar",
  "reg_accumulator_observed": "vectorized_make_float4 | scalar | unknown",
  "uses_reg_store_devec": false
}
```

Acceptance:

- no full model run required;
- each case is generated UOp only;
- artifact is deterministic;
- one passing control case proves the repro harness itself is valid.

Verdicts:

- `TG_P10_1_PASS_REG_REPRO_PINNED`
- `TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL`
- `TG_P10_1_BLOCKED_HANDWRITTEN_KERNEL`

## Phase TG-P10.2: BoltBeam Adapter + Classifier

Purpose: teach BoltBeam to ingest TG-P10.1 and classify the blocker mechanically.

Add:

- schema id in `boltbeam/vocab.py`, for example:
  - `tinygrad.reg_scalar_lowering.v1`;
  - row kind `compiler_lowering`;
- adapter in `boltbeam/artifacts/tinygrad.py`;
- classifier in `boltbeam/diagnostics/pathology.py` or a small sibling module if keeping pathology focused is cleaner;
- tests.

Classification target:

```text
class: EMITTER_BLOCKED
reason: AMD lowering mis-vectorizes a reduction-accumulator REG needed by split-preserving LSE combine
reopen: keep combine reduction accumulator scalar, or make REG_STORE_DEVEC numerically correct for max-reduce/LSE cases
```

Do **not** classify this as:

- `REGALLOC_SPILL`: TG-P9 did not identify spill as the failure.
- `PRIMITIVE_MISSING`: REG and generated UOp exist; the lowering is wrong.
- `REFUTED`: the route is not disproven; it is blocked by compiler capability.

Acceptance:

- BoltBeam tests pass.
- Source scan shows no new parallel vocabulary strings outside `vocab.py`.
- Candidate `decode_attention_g5_8b_refuted` remains refuted until a reopen artifact is supplied.
- A synthetic fixed artifact flips the classifier to reopened/reachable without changing speed evaluation.

Verdicts:

- `TG_P10_2_PASS_BOLTBEAM_REG_LOWERING_CLASSIFIER`
- `TG_P10_2_BLOCKED_SCHEMA_DRIFT`

## Phase TG-P10.3: Generic tinygrad Lowering Fix

Purpose: fix the compiler/backend, not the model route.

Allowed fix surfaces:

- AMD renderer / vectorization lowering;
- UOp verifier/type metadata if scalar REG identity is lost there;
- REG store/load devectorization when reducing `max`/LSE state;
- scheduler metadata that marks a loop-carried reduction accumulator as scalar.

Not allowed:

- special-case the attention combine kernel name;
- special-case Hq=32/Hkv=8/Hd=128;
- emit handwritten HIP/ASM;
- bypass the generated UOp path.

Microgate:

- rerun TG-P10.1 repro;
- expected:
  - shared-weight combine compiles;
  - fused-gmax or selected split-preserving combine compiles;
  - no NaN;
  - scalar accumulator stays scalar or devectorizes correctly.

Verdicts:

- `TG_P10_3_PASS_GENERIC_REG_SCALAR_LOWERING`
- `TG_P10_3_BLOCKED_REG_LOWERING_STILL_WRONG`
- `TG_P10_3_BLOCKED_FIX_NOT_GENERIC`

## Phase TG-P10.4: Split-Preserving Combine Microgate

Purpose: prove the actual combine math after the lowering fix, still outside full model.

Requirements:

- generated UOp only;
- preserve parallelism: no Hq-only collapse and no Hq*Hd collapse previously refuted;
- compare LSE-merged output against Python/numpy reference;
- include 8B geometry and at least one smaller synthetic shape;
- record route geometry and launch counts.

Verdicts:

- `TG_P10_4_PASS_SPLIT_PRESERVING_COMBINE_MICROGATE`
- `TG_P10_4_REFUTE_COMBINE_NUMERIC`
- `TG_P10_4_BLOCKED_PARALLELISM_COLLAPSE`

## Phase TG-P10.5: Full Generated 8B Attention Candidate

Purpose: combine TG-P9 live split tile + TG-P10 combine primitive and measure against owned.

Candidate:

```text
candidate_id: decode_attention_split_preserving_lse_combine
flag: DECODE_ATTN_SPLIT_PRESERVING_COMBINE_GENERATED=1
rollback: disable the generated live-split attention route
```

Gates:

- token/logit equivalence;
- route-bound, no hidden fallback;
- generated UOp only;
- ctx512 and ctx4096 protected;
- optional ctx128/1024/2048 sanity;
- full W==D vs owned.

Promotion bar:

```text
>=98% of owned at ctx512
>=98% of owned at ctx4096
no protected-context regression
rollback one flag away
```

Verdicts:

- `TG_P10_5_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P10_5_REFUTE_GENERATED_ATTENTION_SPEED`
- `TG_P10_5_BLOCKED_ROUTE_ATTRIBUTION`
- `TG_P10_5_BLOCKED_CORRECTNESS`

## Phase TG-P10.6: Promotion + Final Purity

Run only if TG-P10.5 passes.

tinygrad:

- make generated attention the default for the protected 8B geometry;
- keep owned HIP as rollback/oracle;
- update `extra/qk_route_manifest.py`;
- run:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

BoltBeam:

- update `candidates.json`;
- mark the candidate promoted/shipped;
- emit route policy row if applicable;
- ensure ledger says previous refutation reopened by `TG_P10_3/TG_P10_4` evidence, not ignored.

Acceptance:

```text
TINYGRAD_DEFAULT_PURITY_PASS
```

Verdicts:

- `TG_P10_6_PASS_FINAL_DEFAULT_PURITY`
- `TG_P10_6_BLOCKED_STRICT_CENSUS`

## Claude Handoff Prompt

Use this prompt for a fresh agent:

```text
You are working across /home/ubuntu/tinygrad-arkey and /home/ubuntu/BoltBeam.

Goal: TG-P10, final pure-machine-search blocker for 8B decode attention. Do not write a handwritten HIP/ASM/ISA attention kernel. Fix the generic tinygrad AMD lowering gap that prevents split-preserving generated combine from compiling correctly.

Context:
- TG-P9 solved live-context split geometry. Generated live-split attention is token-identical and moves ctx512 from 87.7% to 96.7% of owned.
- Full generated attention is still below promotion: 96.7% ctx512 / 95.3% ctx4096; owned HIP remains default.
- Remaining blocker: split-preserving LSE combine. Shapes that share softmax weights or fuse gmax mis-vectorize the reduction-accumulator REG into invalid make_float4(...) stores. REG_STORE_DEVEC=1 compiles but produces NaNs.
- BoltBeam commit 6b977b0 records this as EMITTER_BLOCKED/reopen condition.

Required sequence:
1. Build a tinygrad generated-UOp repro artifact `tinygrad.reg_scalar_lowering.v1` for the REG scalar lowering failure. Include a passing control, failing shared-weight/fused-gmax cases, and REG_STORE_DEVEC NaN case.
2. Extend BoltBeam minimally to ingest/classify that artifact. It must classify as EMITTER_BLOCKED, not REFUTED/PRIMITIVE_MISSING/REGALLOC_SPILL.
3. Fix tinygrad generically: AMD/UOp lowering should keep the combine reduction accumulator scalar or make REG_STORE_DEVEC numerically correct for max/LSE reduction state. No model-name or kernel-name special casing.
4. Re-run the repro. If it passes, build the split-preserving combine microgate.
5. Only after microgate passes, wire full generated attention and measure W==D vs owned at ctx512 and ctx4096.
6. Promote only if generated attention reaches >=98% of owned at both protected contexts and strict final purity passes. Otherwise ledger the precise blocker.

Acceptance:
- generated UOp only, no handwritten route kernel;
- token/logit equivalent;
- route-bound;
- no hidden fallback;
- BoltBeam tests pass;
- tinygrad census --check passes;
- strict final purity only passes if the generated route actually promotes.
```

## Expected Outcome

If TG-P10 passes, tinygrad reaches:

```text
TINYGRAD_DEFAULT_PURITY_PASS
```

If TG-P10 blocks, the remaining impurity is still useful because it is reduced to a specific backend lowering invariant:

```text
AMD generated UOp lowering must preserve scalar reduction-accumulator REG state for split-preserving LSE combine.
```
