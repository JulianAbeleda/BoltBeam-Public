# G=5 Resource Oracle + K-Only Variant Scope

Date: 2026-07-01.

## Goal

Decide, with measured evidence, whether the 14B G=5 block-tile failure requires:

1. a tinygrad codegen/kernel change,
2. a BoltBeam classifier/ledger improvement,
3. both, or
4. no further work because the route is structurally refuted.

The current state is **UNKNOWN**, not refuted. The latest static pass corrected the raw slowdown from
`78x` to a work-adjusted `19.4x` and ruled out two false causes:

- barrier flood: G=5 has fewer barriers than the baseline path;
- accidental layout mismatch: `WARPS=G=5` is intentional.

Remaining hypotheses:

- register pressure / scratch spill;
- LDS or memory traffic explosion from staging both K and V;
- a mixed VGPR/LDS pressure issue;
- a deeper generated-code pathology visible only in ISA/resource metadata.

## Ownership

### tinygrad owns execution evidence

tinygrad should produce `tinygrad.compiler_pathology.v1` artifacts and, conditionally, generated UOp variants.
It may compile/run kernels and capture AMD code-object metadata. It should not make policy decisions.

Required tinygrad improvements:

- harden the G=5 resource-oracle producer;
- capture both baseline and candidate descriptors in one artifact;
- emit real `vgpr`, `sgpr`, `scratch_bytes`, `lds_bytes`, `barrier_count`, and work-normalized timing;
- never treat missing metadata as zero;
- only build the K-only LDS variant after the resource oracle says it targets the measured cause.

### BoltBeam owns audit decisions

BoltBeam should ingest tinygrad artifacts, classify pathology, update the ledger, and decide whether the K-only
variant is justified. It must not run GPU work or import tinygrad.

Required BoltBeam improvements:

- extend the compiler-pathology classifier to use existing `PathologyClass.LDS_OR_MEMORY_OVERHEAD`;
- distinguish `REGALLOC_SPILL`, `LDS_OR_MEMORY_OVERHEAD`, `UNKNOWN_RESOURCE_COMPLETE`, and
  `NATIVE_ISA_ORACLE_NEEDED`;
- record reopen/stop directives in the candidate ledger;
- keep the scope target-agnostic enough that a similar GQA shape on another model can reuse it.

## Citations Claude Must Read

| claim | citation |
|---|---|
| BoltBeam is the audit brain; tinygrad is the executor | `docs/audit-brain-build-scope.md` |
| Existing compiler-pathology schema/classifier scope | `docs/compiler-pathology-diagnostics-scope.md` |
| Current G=5 result and open candidate | `docs/g5-block-tile-result-20260701.md`, `boltbeam/data/candidates.json` candidate `decode_flash_block_tile_g5_native_context` |
| Latest static UNKNOWN classification | tinygrad commit `5bb9b5555`, BoltBeam commit `7615aab` |
| LDS/prologue route correction | `docs/lds5-ledger-update-20260701.md`, `docs/lds5-prologue-range-update-20260701.md` |
| Pathology classifier implementation | `boltbeam/diagnostics/pathology.py`, `tests/test_pathology.py` |
| tinygrad AMD resource fields | `tinygrad/runtime/ops_amd.py` (`group_segment_size`, `private_segment_size`, `rsrc1`) |
| G=5 route gate and executor path | tinygrad `tinygrad/llm/model.py` `DECODE_FLASH_BLOCK_TILE_G5`, `extra/qk_flash_decode.py` |
| Audit principle: do not chase mechanism wins that regress W==D | `docs/attention-combine-reachability-audit-20260701.md`, `docs/attention-combine-closure-stress-test-20260701.md` |

## Non-Goals

- Do not write handwritten HIP/ASM for the G=5 route.
- Do not promote any attention route from a mechanism-only win.
- Do not rerun attention-combine fused/merged/wholecache paths.
- Do not add GPU execution to BoltBeam.
- Do not classify absent resource fields as `0`.
- Do not use the raw `78x` slowdown as the classifier input; use the work-adjusted ratio.

## Phase G5R0: Evidence Contract Audit

Audit the current `tinygrad.compiler_pathology.v1` adapter and make sure it can carry these metrics without a new
row kind:

| metric | unit | required | meaning |
|---|---:|:---:|---|
| `time_us_per_workgroup` | `us/workgroup` | yes | candidate normalized kernel time |
| `baseline_time_us_per_workgroup` | `us/workgroup` | yes | baseline normalized kernel time |
| `time_ratio_vs_baseline` | `ratio` | yes | raw per-workgroup ratio |
| `work_adjusted_time_ratio` | `ratio` | yes | ratio after normalizing work per workgroup |
| `work_units_per_workgroup` | `units` | yes | candidate logical context/head work units |
| `baseline_work_units_per_workgroup` | `units` | yes | baseline logical work units |
| `vgpr` | `registers` | yes | decoded next-free VGPR / allocated VGPR |
| `sgpr` | `registers` | yes | decoded SGPR count |
| `scratch_bytes` | `bytes/thread` | yes | private segment / spill bytes |
| `lds_bytes` | `bytes/workgroup` | yes | group segment size |
| `lds_staging_bytes_per_tile` | `bytes/tile` | yes | explicit K/V LDS staging bytes |
| `baseline_lds_staging_bytes_per_tile` | `bytes/tile` | yes | baseline LDS staging bytes |
| `lds_staging_ratio_vs_baseline` | `ratio` | yes | candidate/baseline LDS staging ratio |
| `barrier_count` | `instructions` | yes | static or DSL barrier count |
| `expected_barrier_count` | `instructions` | yes | baseline/expected barrier count |

Acceptance:

- BoltBeam adapter passes existing tests.
- Missing required G5 resource fields produce `AdapterIncomplete` or a classifier `NATIVE_ISA_ORACLE_NEEDED`, not `UNKNOWN` with fabricated zeros.
- No new parallel vocabulary outside `boltbeam/vocab.py`.

## Phase G5R1: tinygrad Resource Oracle

Build or harden a committed tinygrad producer:

```text
extra/qk_g5_resource_oracle.py
```

It should supersede the current local/static-only scratch script if present.

Responsibilities:

- compile the baseline `flash_partial_coop_vec` path and the G=5 block-tile path in the same process or in fresh
  deterministic subprocesses;
- intercept `AMDProgram` metadata for the exact kernels:
  - `group_segment_size`;
  - `private_segment_size`;
  - `rsrc1`, `rsrc2`, `rsrc3`;
  - decoded `vgpr`, `sgpr`;
- emit static DSL counts used by the last audit:
  - barrier count;
  - K/V LDS staging bytes;
  - warp/reduce shape;
  - work units per workgroup;
- include timing inputs from the pinned G=5 benchmark or rerun a tiny synced role-local timing if cheap;
- write `bench/g5-block-tile/resource_oracle_v1.json`;
- keep generated bench artifacts local unless the repo's bench policy explicitly allows committing them.

Pass:

```text
G5R1_PASS_RESOURCE_ORACLE
```

if the artifact contains non-null descriptor fields for both baseline and candidate.

Blocked:

```text
G5R1_BLOCKED_DESCRIPTOR_CAPTURE
```

if AMDProgram metadata cannot be captured. Include the exact failing kernel name and route flags.

## Phase G5R2: BoltBeam Classifier Upgrade

Update `boltbeam/diagnostics/pathology.py` to classify the complete G=5 oracle:

| condition | classification | decision |
|---|---|---|
| `scratch_bytes > 0` | `REGALLOC_SPILL` | do not build K-only first; reduce live state or native-ISA/regalloc path |
| `scratch_bytes == 0`, `work_adjusted_time_ratio >= 10`, `lds_staging_ratio_vs_baseline >= 8` | `LDS_OR_MEMORY_OVERHEAD` | K-only LDS variant is justified |
| `scratch_bytes == 0`, huge VGPR vs baseline, no LDS ratio explosion | `UNKNOWN` or new measured-register-pressure reason in `UNKNOWN` | do not build K-only blindly |
| descriptor fields missing | `NATIVE_ISA_ORACLE_NEEDED` | fix G5R1 first |
| all resources normal but still slow | `STRUCTURAL_ROUTE_REFUTED` or `UNKNOWN` with manual-disasm reopen | do not keep routing experiments alive without ISA proof |

Do not add a new enum unless the existing `PathologyClass` cannot represent the result. `LDS_OR_MEMORY_OVERHEAD`
already exists and should be used before adding vocabulary.

Acceptance:

- tests cover spill, LDS/memory overhead, missing-oracle, and complete-but-unknown fixtures;
- `boltbeam diagnose compiler-pathology` reports the selected class and next action;
- the G=5 candidate ledger uses the classifier result, not a hand-written prose override.

## Phase G5R3: Decision Gate Before Kernel Work

Run the tinygrad oracle artifact through BoltBeam.

Outcomes:

| verdict | next action |
|---|---|
| `REGALLOC_SPILL` | stop K-only; scope live-state reduction/native ISA resource fix |
| `LDS_OR_MEMORY_OVERHEAD` | proceed to G5R4 K-only LDS variant |
| `NATIVE_ISA_ORACLE_NEEDED` | fix oracle; no kernel work |
| `STRUCTURAL_ROUTE_REFUTED` | close candidate with reopen condition requiring new primitive/backend capability |
| `UNKNOWN` | require disasm/PMC source rows before kernel work |

This gate is mandatory. The K-only variant is not allowed to start from the current UNKNOWN state.

## Phase G5R4: K-Only LDS Variant (Conditional)

Run this only if G5R3 returns `LDS_OR_MEMORY_OVERHEAD`.

tinygrad implementation target:

```text
K-only staging should be controlled by the route implementation, not an extra env flag.
```

Shape:

- preserve the G=5 sliced-context route;
- stage K into LDS;
- read V directly from the existing warmed L2/global path;
- keep E_49152_32_3 V warming intact;
- avoid adding extra external combine kernels;
- rollback is unset flag.

Pass criteria:

- correctness: token-match and microgate vs baseline;
- resource: no scratch spill; VGPR does not exceed the G5R1 full K+V path by more than 10%;
- mechanism: `lds_staging_ratio_vs_baseline` drops materially vs full K+V staging;
- W==D: no protected-context regression; Tier-B or better is promotable under current residual policy.

Refute criteria:

- any protected-context regression greater than 1%;
- scratch spill introduced;
- mechanism works but W==D regresses, recorded as mechanism-pass/speed-refute.

## Phase G5R5: Ledger + Reopen Conditions

Update BoltBeam candidates after measurement:

- If K-only passes: mark the route candidate promotable only for the profiled G=5 shape class and keep rollback.
- If K-only refutes: close `decode_flash_block_tile_g5_native_context` for current AMD generated UOp routes.
- Reopen only on:
  - new resource oracle showing spill removed or LDS ratio reduced;
  - new target primitive enabling cross-workgroup coordination or better LDS staging;
  - new model geometry where head/split parallelism changes the Amdahl math;
  - native ISA codegen that meets the explicit per-workgroup target.

## Principle Audit Checklist

Before committing, run this checklist:

- Central vocabulary: no new classifier strings outside `boltbeam/vocab.py`.
- Single boundary: BoltBeam ingests evidence; tinygrad executes kernels.
- Tiny modules: resource math/classifier changes stay in `boltbeam/diagnostics/pathology.py` or a small helper.
- No duplicated scope truth: this doc is canonical; tinygrad docs should cite it instead of copying the full plan.
- No hidden fallback: route-bound and token-match are required for every speed claim.
- No blind kernel work: G5R4 cannot start until G5R3 selects `LDS_OR_MEMORY_OVERHEAD`.

## Expected Answer To "Do We Need To Improve tinygrad And BoltBeam?"

Yes, both, but not equally:

- **tinygrad** needs the next concrete capability: a reliable resource oracle for AMD-generated kernels. If that oracle
  selects LDS/memory overhead, tinygrad then needs a generated K-only LDS G=5 variant.
- **BoltBeam** needs a smaller classifier improvement: consume the richer oracle and stop returning generic `UNKNOWN`
  when resource rows prove `LDS_OR_MEMORY_OVERHEAD`.

If the resource oracle instead proves spill/register pressure, the tinygrad improvement changes: do not build K-only;
scope live-state reduction or native-ISA resource fixing.
