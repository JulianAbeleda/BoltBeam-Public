# MMQ Epoch Model Exhaustive Scope

Purpose: define the epoch evidence BoltBeam must answer before machine search is
allowed to optimize a 14B prefill MMQ candidate. The 18 research categories say
what questions exist; this document scopes the epoch-specific answer layer.

## Core Claim

For llama-style MMQ, speed comes from an ordered lifecycle, not from a single
dot instruction. Machine search must therefore reason over epochs:

```text
data loaded/dequantized -> staged -> made visible -> consumed/reused -> advanced -> written back
```

A candidate that only proves numeric correctness can still be structurally slow
if its epochs reload too much data, serialize waits, overuse LDS, spill VGPRs,
or lose reuse.

## Epoch Definition

An epoch is a bounded interval in the emitted kernel where all of these are
known:

```text
epoch_id
name
kind
entry_conditions
events
live_inputs
live_outputs
resources_live
sync_before
sync_after
reuse_claim
proof_rows
exit_conditions
blockers
```

Epochs are not just timestamps. They are proof objects that join source intent,
lowered instructions, resources, and correctness ownership.

## Required Epoch Kinds

| # | Epoch kind | Question it must answer |
|---:|---|---|
| 1 | launch_ownership | Which workgroup/wave/lane owns each output tile and fragment? |
| 2 | q4k_tile_x_load_decode | How are Q4_K bytes, scales, and mins loaded/dequantized/staged? |
| 3 | q8_tile_y_stage | How is the Q8_1/DS4 activation panel loaded/staged for reuse? |
| 4 | visibility_sync | Which barrier/wait proves staged data is visible to consumers? |
| 5 | dot_accumulate | Which dot primitive consumes staged data and which accumulator slot receives it? |
| 6 | k_advance | How does the kernel advance K without losing accumulator identity or reuse cadence? |
| 7 | stage_reuse_or_overwrite | When is staged data reused, invalidated, or overwritten? |
| 8 | writeback | Which output store consumes which accumulator and owner identity? |
| 9 | epilogue | Which final waits/stores/end conditions close the kernel? |

The initial 14B MMQ R4 path must cover at least epochs 1, 5, and 8. R5/R6
performance work requires all nine.

## Llama MMQ Epoch Hypothesis

For the vendored `mmq.cuh` research source, the expected epoch sequence is:

| Epoch | Llama source anchor | Expected behavior |
|---|---|---|
| launch_ownership | `mul_mat_q`, `mul_mat_q_process_tile` | 8 waves per CTA; wave owns a 16-row output stripe; N is processed in 16-col fragments. |
| q4k_tile_x_load_decode | `load_tiles_q4_K` | Cooperative load/decode of packed Q4_K fields into shared tile_x. |
| q8_tile_y_stage | `load_tiles_*`, `block_q8_1_mmq` | Activation panel is arranged for contiguous shared-memory copies and reuse. |
| visibility_sync | `__syncthreads()` / shared-memory barriers | Producers finish tile staging before consumers issue dot work. |
| dot_accumulate | `vec_dot_q4_K_q8_1_impl_mmq` | Dot work consumes tile_x/tile_y and accumulates into per-thread `sum[]`. |
| k_advance | `for (k0...)` / `MMQ_ITER_K` | K advances in bounded panels while preserving live `sum[]`. |
| stage_reuse_or_overwrite | shared tile reuse before next panel | Staged data is reused across many outputs, then safely overwritten. |
| writeback | `mmq_write_back_mma` / `mmq_write_back_dp4a` | Exactly one owner stores each output element. |
| epilogue | kernel tail | Outstanding stores complete and kernel exits. |

## BoltBeam Data Model Needed

Add a structured module, likely:

```text
boltbeam/search/epoch_model.py
```

Minimum records:

```python
EpochSpec:
  id: str
  kind: str
  required_events: tuple[str, ...]
  required_categories: tuple[str, ...]
  entry_conditions: tuple[str, ...]
  exit_conditions: tuple[str, ...]

EpochEvent:
  epoch_id: str
  event_kind: str
  source_anchor: str | None
  instruction_mnemonic: str | None
  logical_identity: dict
  physical_identity: dict
  resources: dict
  status: PASS | FAIL | BLOCKED | UNKNOWN

EpochReport:
  candidate_id: str
  workload: dict
  epochs: tuple[EpochAnswer, ...]
  missing_epochs: tuple[str, ...]
  blockers: tuple[dict, ...]
```

The module should provide:

```text
mmq_epoch_specs()
epoch_spec_by_id(id)
validate_epoch_report(report)
epoch_coverage(report)
blocked_epoch_reasons(report)
```

## Tinygrad Evidence Needed

The current `AMD_ISA_PROOF_MANIFEST` records:

```text
ACCUM_READ
ACCUM_WRITE
V_WMMA
GLOBAL_STORE
```

That is enough for initial ownership/writeback proof, but not for full epoch
proof. Extend or join with rows for:

| Needed row | Why |
|---|---|
| GLOBAL_LOAD / GLOBAL_LOAD_B128 | Prove Q4/Q8 load width, address source, and coalescing. |
| DS_STORE / DS_STORE_B128 | Prove staged data enters LDS and where. |
| DS_LOAD / DS_LOAD_B128 | Prove dot epoch consumes staged LDS data, not reloaded globals. |
| S_BARRIER | Prove visibility boundary between stage and consume. |
| S_WAITCNT | Prove required waits without over-draining. |
| V_DOT / WMMA / CUSTOMI dot | Prove dot primitive and accumulator input/output identity. |
| resource snapshot | VGPR, SGPR, LDS bytes, scratch bytes, occupancy estimate. |
| source tag / epoch tag | Join emitted rows back to intended epoch. |

Do not require all of these for R4. Require them before any performance
promotion or tile search.

## Required Joins

BoltBeam must be able to join:

```text
llama source epoch oracle
  + tinygrad source/intention tags
  + AMD ISA proof manifest rows
  + resource/counter rows
  + numeric oracle result
  + baseline/comparator timing
```

The key joins:

| Join | Required proof |
|---|---|
| ownership oracle -> store manifest | every output has exactly one emitted store owner. |
| sum-slot oracle -> accumulator manifest | each logical slot maps to the expected physical VGPR/fragment. |
| staged source -> DS store/load rows | staged data is actually consumed from LDS. |
| barrier rows -> stage/consume boundary | no consumer epoch can precede the required visibility sync. |
| resource rows -> epoch specs | claimed tile shape fits VGPR/LDS/scratch bounds. |
| timing rows -> epoch report | performance win/loss can be attributed to epoch structure. |

## Completion Phases

### E0: Taxonomy

Done when BoltBeam has immutable epoch kinds and validators.

Required:

```text
mmq_epoch_specs()
tests proving required epoch ids are complete and ordered
coverage validator reports missing/blocker rows
```

### E1: Llama Epoch Oracle

Done when BoltBeam can describe the expected llama MMQ epoch sequence for the
bounded 14B role without executing kernels.

Required:

```text
source anchors for each epoch
expected ownership and reuse claims
expected sync points
expected resource formulas where known
```

### E2: tinygrad ASM Epoch Manifest

Done when tinygrad emits enough opt-in rows to assign relevant AMD ISA
instructions to epochs.

Required for R4:

```text
ACCUM_READ/WRITE
V_WMMA or dot primitive
GLOBAL_STORE
owner/sum-slot identity
```

Required for R5/R6:

```text
GLOBAL_LOAD
DS_STORE
DS_LOAD
S_BARRIER
S_WAITCNT
resources
```

### E3: Epoch Coverage Report

Done when BoltBeam can load candidate evidence and answer:

```text
which epochs are PASS
which epochs are FAIL
which epochs are BLOCKED
which exact evidence is missing
```

### E4: Numeric + Epoch Gate

Done when a bounded candidate must pass both:

```text
numeric oracle
epoch coverage
```

No timing result counts without this.

### E5: Performance Attribution

Done when BoltBeam can explain a win/loss by epoch:

```text
too many global loads
LDS reuse missing
barrier/wait over-drain
VGPR occupancy collapse
store path uncoalesced
dot primitive underutilized
```

### E6: Search Integration

Done when machine search can vary knobs only inside epochs that have enough
evidence.

Examples:

```text
tile_m/tile_n only after ownership + resource model pass
LDS layout only after stage/load/barrier rows exist
wait policy only after wait rows exist
dot primitive only after numeric + accumulator identity pass
```

## Machine Search Rules

1. Search cannot promote a candidate with `UNKNOWN` in a required epoch.
2. Search cannot time a candidate that fails ownership, accumulator, or numeric epochs.
3. Search can continue from `BLOCKED` only if the next candidate changes the blocked primitive/API.
4. Search must report the epoch delta between baseline, llama, and candidate.
5. Search knobs must name their owning epoch.
6. A candidate without epoch coverage is a probe, not a promotion candidate.

## 14B MMQ Immediate Path

The next practical path is:

```text
1. Add BoltBeam epoch specs and coverage validator.
2. Encode the llama MMQ epoch oracle for bounded 16x16x256 and 128x128x256.
3. Extend tinygrad proof manifest to tag enough rows for R4 ownership/writeback.
4. Build one bounded R4 candidate.
5. Join manifest rows to ownership + sum-slot + epoch specs.
6. Only then add numeric Q4_K x Q8_1.
7. Only after numeric+epoch pass, start R5 tile/resource search.
```

## Stop Conditions

Stop and mark blocked if:

```text
epoch rows cannot be joined to emitted ISA
staged data cannot be distinguished from direct global reloads
barrier/wait rows cannot be located
resource rows cannot be tied to the same compiled candidate
ownership or sum-slot identity is lost after lowering
numeric correctness fails after epoch proof passes
```

The blocked report must name the missing category, epoch, and primitive/API.

## Relationship to the 18 Categories

Epoch modeling is not a replacement for the research categories. It fills the
time/lifecycle-sensitive subset:

```text
4  tile_geometry
5  work_ownership
6  accumulator_mapping
7  shared_memory_lifecycle
8  k_loop_cadence
10 resource_model
11 sync_cadence
12 store_path
14 roofline_target
15 search_space_knobs
18 stop_criteria
```

The remaining categories still gate correctness, routing, promotion, and
comparators.
