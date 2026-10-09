# Pure Machine Search Default Resolution Scope

Date: 2026-07-01.

## Objective

Resolve the default-path purity failure found by
`docs/default-path-handwritten-kernel-audit-20260701.md`.

Target end state:

```text
All default hot routes are either ordinary tinygrad-generated or
machine-authored/generated from BoltBeam search. Handwritten kernels remain
only as rollback oracles, diagnostics, or historical comparators.
```

This is a purity resolution scope, not a speed-at-any-cost scope. The default
path must not be made "pure" by knowingly regressing shipped behavior. A
handwritten route leaves the default path only after a generated/search-authored
replacement passes the same correctness, route-binding, fallback, and W==D gates
as the handwritten route it replaces.

## Citations Claude Must Read

| fact | citation |
|---|---|
| Current default path fails strict handwritten-kernel audit | `docs/default-path-handwritten-kernel-audit-20260701.md` |
| BoltBeam owns audit/ledger/search decisions; tinygrad executes | `docs/audit-brain-build-scope.md`, `docs/architecture.md` |
| Coding principle: do not hand-write a one-off route and call search complete | `docs/coding-principles.md` |
| Model/quant/target agnostic authorities | `docs/model-agnostic-search-roadmap.md`, `data/quants.json`, `data/targets.json`, `data/candidates.json` |
| Generated-ISA attention primitive boundary | `docs/g5-generated-isa-primitive-route-scope-20260701.md` |
| G=5 resource oracle that motivates generated primitive work | `docs/g5-resource-oracle-and-konly-scope-20260701.md` |
| Attention-combine closure / do-not-retry ledger | `docs/attention-combine-reachability-audit-20260701.md`, `docs/attention-combine-closure-stress-test-20260701.md` |
| tinygrad default-route census producer | tinygrad `extra/pure_machine_search_default_path_census.py` |
| route manifest authority | BoltBeam `boltbeam/policy/assets/route_manifest.v1.json`; tinygrad EXP consumes the generated hashed snapshot |

## Current State

The strict audit classifies the current default path as:

| route | current class | purity status | resolution target |
|---|---|---:|---|
| `decode_q4k_g3_generated` | `machine_authored_generated` | pass | lock as the reference generated route |
| `decode_q6k_coop_shipped` | `hand_authored_uop_template` | tolerated, not final | regenerate from quant/shape/target grammar |
| `decode_attention_owned_two_kernel` | `external_handwritten_kernel` | fail | replace with generated ISA/UOp attention route |
| `prefill_pipe_role_selective_default` | `external_handwritten_kernel` / specialized assembly emitter | fail | replace with generated/search-authored GEMM schedule |

Everything else can remain ordinary tinygrad scheduler output.

## Definition Of "Pure Machine Search"

A default hot route is pure enough to ship only if it is one of:

| class | allowed as final default? | meaning |
|---|---:|---|
| `machine_authored_generated` | yes | emitted from profile facts, grammar/search candidate, and generated lowering |
| `tinygrad_scheduler_generated` | yes | produced by ordinary tinygrad graph lowering, no custom route kernel |
| `hand_authored_uop_template` | transitional only | Python UOp route factory written by humans; must have a replacement scope |
| `external_handwritten_kernel` | no | HIP/ASM/custom object/specialized instruction emitter used as route kernel |
| `rollback_oracle` | yes, behind rollback only | handwritten/specialized implementation retained for fallback, comparison, or diagnosis |

Human code may implement reusable primitives, IRs, renderers, validators,
candidate grammars, and evaluators. Human code may not encode the final default
route as a one-off schedule and then label it "search."

## Non-Goals

- Do not flip retired attention flags, `PREFILL_GEMM_PIPELINE=0`, or Q6_K
  coop flags just to make the audit pass.
- Do not add a handwritten `.hip`, `.s`, `.asm`, `.cpp`, inline instruction
  list, or fixed route body as the replacement.
- Do not promote a generated route from synthetic correctness only. Promotion
  requires real tinygrad route-bound W==D evidence.
- Do not collapse BoltBeam and tinygrad responsibilities. BoltBeam decides and
  records; tinygrad executes and measures.

## Phase PMD0: Default-Purity Contract In BoltBeam

Add a machine-enforced default-purity policy.

Required data:

```text
RouteProvenance:
  machine_authored_generated
  tinygrad_scheduler_generated
  hand_authored_uop_template
  external_handwritten_kernel
  rollback_oracle

DefaultPurityDecision:
  pass
  fail_external_handwritten_default
  transitional_hand_authored_uop
  blocked_missing_replacement_scope
```

Every shipped/promoted candidate or default route must carry:

- `route_id`;
- `workload` (`decode`, `prefill`, or both);
- `provenance`;
- `source_refs`;
- `replacement_for` if it replaces a handwritten route;
- `rollback_to` if promoted;
- `model/target/quant/role/context` applicability;
- evidence artifact refs.

Policy:

- `external_handwritten_kernel` cannot be a final pure-search default.
- `hand_authored_uop_template` can remain only with a tracked replacement scope.
- `rollback_oracle` is allowed only when not selected by default.
- Unknown provenance is a failure, not a pass.

Verdicts:

- `PMD0_PASS_PURITY_CONTRACT_ENFORCED`
- `PMD0_BLOCKED_PARALLEL_PROVENANCE_VOCAB`
- `PMD0_BLOCKED_MANIFEST_NOT_EXPRESSIVE`

Acceptance:

- BoltBeam tests pass.
- A source scan confirms no duplicate provenance strings outside `vocab.py`.
- The current default path evaluates to the same expected fail/transitional
  state as the audit, not a false pass.

## Phase PMD1: Default-Route Census Ingestion

Make the tinygrad census a first-class BoltBeam evidence input.

Input:

- tinygrad `extra/pure_machine_search_default_path_census.py`;
- tinygrad `extra/qk_route_manifest.py`.

Output schema:

```text
default_route_census.v1:
  model_id
  target_id
  timestamp
  env_flags
  routes:
    route_id
    workload
    quant
    role
    contexts
    selected_by_default
    provenance
    source_refs
    rollback_flags
```

BoltBeam ingestion should normalize this into a purity report without running a
GPU benchmark.

Verdicts:

- `PMD1_PASS_DEFAULT_CENSUS_INGESTED`
- `PMD1_BLOCKED_TINYGRAD_CENSUS_MISSING_FIELDS`
- `PMD1_BLOCKED_ROUTE_MANIFEST_DRIFT`

Acceptance:

- The report reproduces the four current default hot routes.
- The Q4_K G3 row passes.
- The Q6_K row is transitional.
- The attention and prefill rows fail strict final-default purity.

## Phase PMD2: Generated Q4_K Default Lock

Use the existing G3 route as the positive control for the whole policy.

Scope:

- prove `decode_q4k_g3_generated` is selected from profile/search data;
- assert owned Q4_K warp kernels remain rollback or historical comparators;
- add a guard that rejects reintroducing owned Q4_K warp as default while G3 is
  speed-equivalent.

Verdicts:

- `PMD2_PASS_Q4K_GENERATED_DEFAULT_LOCKED`
- `PMD2_BLOCKED_G3_PROVENANCE_REGRESSION`

Acceptance:

- G3 route remains default where already promoted.
- No speed remeasurement is required unless route policy changed.
- The guard points at the prior G3 speed-equivalence artifacts rather than
  duplicating thresholds.

## Phase PMD3: Q6_K Machine-Authored Replacement

Replace `decode_q6k_coop_shipped` as a hand-authored UOp template by making the
Q6_K route authorable from BoltBeam's quant/shape/target grammar.

This is not a new handwritten Q6_K kernel. It is the Q6_K equivalent of the TG
bridge:

```text
GGUF/ProfileIR -> QuantSpec/TargetSpec/ShapeSpec -> route-family grammar
-> generated UOp/template lowering -> tinygrad route-bound evidence
```

Required work:

1. Extend the route-family template/grammar so it can re-express the current
   Q6_K coop route from `data/quants.json` and model profile facts.
2. Add a self-audit proving no route generation branch uses model name,
   `route_id == q6k_coop`, or hardcoded Qwen dimensions.
3. Generate the Q6_K candidate into tinygrad through a generic emitter path.
4. Run isolated correctness gates for lm_head, ffn_down, and any long-K Q6_K
   role used by 14B/32B.
5. Run W==D gates on protected contexts and models.

Protected gates:

- 8B: no protected-context regression.
- 14B/32B: preserve or improve the shipped L3 behavior.
- token/logit equivalence as appropriate.
- route-bound: generated candidate fires, no hidden fallback to the old
  hand-authored UOp route.

Verdicts:

- `PMD3_PASS_Q6K_GENERATED_REPLACEMENT`
- `PMD3_REFUTE_Q6K_GENERATED_ROUTE_SLOWER`
- `PMD3_BLOCKED_GRAMMAR_CANNOT_REEXPRESS_SHIPPED`
- `PMD3_BLOCKED_EMITTER_CAPABILITY`

Default action:

- If pass: generated Q6_K becomes default; old Q6_K UOp route becomes
  `rollback_oracle`.
- If blocked: keep old default, mark purity debt with exact missing grammar or
  emitter capability.

## Phase PMD4: Decode Attention Generated Replacement

Replace `decode_attention_owned_two_kernel` with a generated attention route.

This phase must follow `docs/g5-generated-isa-primitive-route-scope-20260701.md`
and its purity boundary.

Allowed path:

```text
GQA/flash Tile IR -> profile-driven candidate grammar -> generic UOp/ISA lowering
-> AMDISARenderer -> route-bound W==D evidence
```

Forbidden path:

```text
write a fixed G=5, G=4, or Qwen-specific HIP/ASM/RDNA3 instruction kernel
```

Required sub-gates:

1. Generated route boundary proof: no external custom object and no fixed
   instruction list.
2. Primitive gap pinned: classify current bloat as IR/lowering/scheduler/search
   or structural.
3. Tile IR can express both current generated baseline and at least one
   improvement candidate.
4. Generated primitive microgate passes resource/correctness thresholds.
5. Full route W==D proves no protected-context regression before default flip.

Verdicts:

- `PMD4_PASS_ATTENTION_GENERATED_DEFAULT`
- `PMD4_REFUTE_ATTENTION_GENERATED_ROUTE_SLOWER`
- `PMD4_BLOCKED_RENDERER_OR_IR_CAPABILITY`
- `PMD4_BLOCKED_STRUCTURAL_NO_PURE_ROUTE`

Default action:

- If pass: generated attention route becomes default for the applicable
  profiles/targets; owned HIP tile becomes `rollback_oracle`.
- If blocked/refuted: keep the owned route for performance, with purity debt
  recorded and a precise reopen condition.

## Phase PMD5: Prefill GEMM Generated Schedule Replacement

Replace `prefill_pipe_role_selective_default` with a generated/search-authored
GEMM schedule.

The current prefill pipe is a performance win, but it is specialized assembly
emission. The replacement should make the schedule itself a search artifact.

Required schedule IR:

```text
PrefillGEMMScheduleSpec:
  tile_m
  tile_n
  tile_k
  tm
  tn
  pipeline_depth
  role_policy
  vector_width
  accumulator_layout
  waitcnt_policy
  target_requirements
```

Required work:

1. Convert the current role-selective pipe into a data/spec representation that
   can losslessly re-emit the current route.
2. Add a candidate grammar over tile/pipeline/role-selection knobs.
3. Generate the GEMM route from the spec, not from a handwritten fixed assembly
   body.
4. Run prefill authority gates at `pp512/1024/2048/4096/8192` where supported.
5. Preserve the current pipe as rollback.

Verdicts:

- `PMD5_PASS_PREFILL_GENERATED_SCHEDULE_DEFAULT`
- `PMD5_REFUTE_PREFILL_GENERATED_SCHEDULE_SLOWER`
- `PMD5_BLOCKED_SCHEDULE_IR_CANNOT_REEXPRESS_PIPE`
- `PMD5_BLOCKED_RENDERER_CAPABILITY`

Default action:

- If pass: generated schedule becomes default; current assembly pipe becomes
  rollback.
- If blocked: keep current prefill default and record the exact missing schedule
  or renderer capability.

## Phase PMD6: Pure Mode, But Not As A Shortcut

Add an optional diagnostic mode only after PMD0/PMD1:

```text
PURE_MACHINE_SEARCH_ONLY=1
```

Purpose:

- prove the model can run without forbidden default routes;
- expose correctness gaps early;
- measure the performance cost of purity debt.

Rules:

- Pure mode is not the shipped default.
- Pure mode must label regressions plainly.
- Pure mode must not silently fall back to forbidden routes.

Verdicts:

- `PMD6_PASS_DIAGNOSTIC_PURE_MODE`
- `PMD6_BLOCKED_HIDDEN_HANDWRITTEN_FALLBACK`
- `PMD6_BLOCKED_CORRECTNESS`

## Phase PMD7: Default Promotion Gate

Only after PMD3, PMD4, and PMD5 pass for their applicable routes, run the final
default flip.

Promotion gates:

- token/logit equivalence for protected models;
- route-bound evidence;
- no hidden fallback;
- no protected-context W==D regression;
- rollback flags documented;
- BoltBeam ledger records old route as rollback/oracle, not active default;
- default-purity audit changes from fail to pass.

Verdicts:

- `PMD7_PASS_PURE_MACHINE_SEARCH_DEFAULT`
- `PMD7_BLOCKED_ROUTE_PURITY_DEBT_REMAINING`
- `PMD7_BLOCKED_PROTECTED_CONTEXT_REGRESSION`

## Phase PMD8: CI / Hook Enforcement

After PMD0/PMD1, add continuous enforcement so drift does not return.

Checks:

- candidate manifest contains no promoted/default
  `external_handwritten_kernel`;
- default census has no forbidden provenance selected by default;
- every transitional `hand_authored_uop_template` has a replacement scope;
- rollback routes are not selected unless an explicit rollback flag is set;
- source scan rejects new route defaults pointing at `.hip`, `.s`, `.asm`, or
  specialized instruction emitters unless the candidate is tagged
  `rollback_oracle`.

Verdicts:

- `PMD8_PASS_DEFAULT_PURITY_CI`
- `PMD8_BLOCKED_FALSE_POSITIVE_SOURCE_SCAN`
- `PMD8_BLOCKED_MISSING_CENSUS_IN_CI`

## Recommended Execution Order

1. **PMD0 + PMD1**: policy and census first. This makes the failure durable and
   prevents accidental backsliding.
2. **PMD2**: lock the known-good generated Q4_K baseline.
3. **PMD3**: Q6_K generated replacement. This is likely the smallest purity
   debt because it is already UOp-based.
4. **PMD5**: prefill schedule generation. This is high leverage and must keep
   the role-selective pipe's W==D gains.
5. **PMD4**: attention generated ISA. This is the hardest route and must obey
   the G=5 generated-primitive boundary.
6. **PMD6/PMD7/PMD8**: diagnostic pure mode, final default flip, then CI guard.

## Completion Criteria

This scope is complete only when:

```text
DEFAULT_PATH_HANDWRITTEN_KERNEL_AUDIT_PASS
```

and all active default hot routes are classified as:

```text
machine_authored_generated
tinygrad_scheduler_generated
```

with any handwritten route classified only as:

```text
rollback_oracle
```

The expected status at the start of this scope remains:

```text
DEFAULT_PATH_HANDWRITTEN_KERNEL_AUDIT_FAIL
```

That is correct. The audit should not turn green until the generated Q6_K,
generated attention, and generated prefill replacements actually pass their
gates.
