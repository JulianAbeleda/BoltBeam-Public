# G=5 Generated-ISA Primitive Route Scope

Date: 2026-07-01.

## Goal

Pursue the "primitive route" for the 14B G=5 attention tile **without hand-writing a G=5 kernel**.

The latest oracle changed the situation:

- tinygrad commit `315253d`: G=5 resource oracle measured `VGPR=91`, `scratch=0`, `LDS=8192`,
  `static/math bloat=11.4x`.
- BoltBeam commit `bb96fb0`: classified the candidate as `LDS_OR_MEMORY_OVERHEAD`.
- `REGALLOC_SPILL` and `BARRIER_FLOOD` are ruled out.

The primitive route is now:

```text
teach the machine/backend enough primitives to emit a better G=5 tile
```

not:

```text
write a one-off RDNA3/HIP G=5 kernel by hand
```

## Purity Boundary

### Forbidden

Do **not** add any of these as the solution:

- a handwritten `.s`, `.asm`, `.hip`, `.cpp`, or inline RDNA3 kernel for G=5;
- a fixed instruction list specialized to `Hq=40,Hkv=8,Hd=128`;
- a hidden external custom kernel route that bypasses the generated path;
- a route selected by string name rather than by profile/candidate evidence;
- a speed claim without route-bound/token-match/W==D evidence.

### Allowed

These are allowed because they improve the generator/substrate rather than hand-writing the kernel:

- generic `AMDISARenderer` primitive lowerings;
- generic scheduler/waitcnt/resource-allocation improvements;
- a structured Tile IR or UOp pattern for GQA flash tiles;
- a candidate grammar over GQA geometry (`G`, `Hd`, `TK`, K-only vs K+V staging, split shape);
- generated microkernels used only as correctness/resource gates;
- BoltBeam candidate/evaluator/ledger logic.

Human-written code may implement **the emitter and primitives**. It may not encode the final G=5 kernel as a hand
schedule.

## Citations Claude Must Read

| claim | citation |
|---|---|
| BoltBeam owns audit/ledger; tinygrad owns execution | `docs/audit-brain-build-scope.md` |
| Current G=5 resource-oracle scope | `docs/g5-resource-oracle-and-konly-scope-20260701.md` |
| Latest BoltBeam G=5 classification | `boltbeam/data/candidates.json` candidate `decode_flash_block_tile_g5_native_context`, commit `bb96fb0` |
| Latest tinygrad oracle | tinygrad commit `315253d`, `docs/g5-block-tile-oracle-result.md`, `bench/g5-block-tile/compiler_pathology_v1_dynamic.json` |
| AMDISARenderer substrate | tinygrad `tinygrad/renderer/isa/amd.py`, `extra/amd_isa_inc0_gate.py`, `extra/amd_isa_phase_f_primitives_gate.py` |
| Existing AMD ISA roadmap | tinygrad `docs/amd-isa-backend-e2e-roadmap-20260629.md` |
| Do not chase mechanism-only wins | `docs/attention-combine-reachability-audit-20260701.md`, `docs/attention-combine-closure-stress-test-20260701.md` |

## Phase GP0: Purity Gate

Before implementation, add a tinygrad/BoltBeam audit note proving the planned route is generated:

- source artifact type: UOp/Tile IR/grammar candidate, not `.hip` or raw `.s`;
- renderer path: `AMDISARenderer` or tinygrad UOp renderer path;
- selection path: BoltBeam candidate id + route-bound evidence;
- rollback flag documented;
- no custom external kernel object added.

Verdicts:

- `G5_GP0_PASS_GENERATED_ROUTE_BOUNDARY`
- `G5_GP0_BLOCKED_HANDWRITTEN_KERNEL`
- `G5_GP0_BLOCKED_ROUTE_BOUNDARY_UNCLEAR`

Stop if GP0 does not pass.

## Phase GP1: Generated Primitive Gap Audit

Audit what the current generated path cannot express efficiently. Start from the measured bloat:

```text
static instructions = 1610
math ops = 141
instruction bloat = 11.4x
LDS = 8192
scratch = 0
barriers = 1
```

Classify each bloat source as:

| class | meaning |
|---|---|
| `LOWERING_GAP` | primitive exists conceptually, but lowering emits too many ops |
| `SCHEDULER_GAP` | ops are right, but ordering/waitcnt hides no latency |
| `IR_GAP` | UOp/Tile IR cannot express the efficient shape |
| `SEARCH_GAP` | grammar does not expose the knob |
| `STRUCTURAL` | even ideal generated code cannot win |

Required outputs:

- top five source groups by static instruction count;
- top three emitted instruction families causing bloat;
- whether K-only staging would remove enough bloat to matter;
- whether a generic lowering/scheduler fix exists.

Verdicts:

- `G5_GP1_PASS_PRIMITIVE_GAP_PINNED`
- `G5_GP1_BLOCKED_NEEDS_DISASM_SOURCE_MAP`
- `G5_GP1_REFUTE_STRUCTURAL_NO_GENERATED_PATH`

## Phase GP2: Tile IR / Candidate Grammar

If GP1 finds a generator gap, define the smallest reusable IR/candidate surface that can author a GQA flash tile:

```text
GQAFlashTileSpec:
  G
  Hd
  TK
  stage_k: bool
  stage_v: bool
  value_path: l2_direct | lds
  reduction: wave | lds_cross_warp
  softmax_state: per_head | per_group
  split_l
```

The grammar must be profile-driven:

- derive `G = Hq / Hkv`;
- derive `Hd`;
- derive candidate `TK` from target LDS budget;
- do not hardcode `G=5` except in tests/fixtures.

Acceptance:

- can re-express the current G=5 K+V candidate;
- can express K-only as a different generated candidate;
- can express G=4/8B as a compatibility fixture;
- candidate count is bounded.

Verdicts:

- `G5_GP2_PASS_TILE_IR_READY`
- `G5_GP2_BLOCKED_SEARCH_SPACE_EXPLOSION`
- `G5_GP2_BLOCKED_IR_CANNOT_REEXPRESS_BASELINE`

## Phase GP3: Generic Lowering / Renderer Work

Implement only generic lowering needed by GP2. Examples:

- better vectorized LDS load/store lowering;
- fewer address-calculation ops for cooperative tile preloads;
- reusable online-softmax state lowering;
- waitcnt/scheduler improvement for LDS/VMEM overlap;
- optional K-only LDS path generated from the same IR.

Rules:

- no fixed G=5 instruction sequence;
- all emitted instructions come from the renderer/lowering;
- every new primitive has a standalone microgate;
- if a primitive only works for `G=5`, it is not generic enough unless the IR proves it handles arbitrary `G`.

Verdicts:

- `G5_GP3_PASS_GENERIC_LOWERING`
- `G5_GP3_BLOCKED_RENDERER_CAPABILITY`
- `G5_GP3_BLOCKED_NON_GENERIC_G5_SPECIAL_CASE`

## Phase GP4: Generated Primitive Microgate

Generate the G=5 candidate from GP2/GP3 and run it as an isolated primitive gate.

Pass thresholds:

- token/micro correctness vs reference;
- `scratch_bytes == 0`;
- resource artifact emitted;
- static/math bloat materially below `11.4x`;
- per-workgroup target:
  - stretch target: `<100us`;
  - minimum continuation target: at least `2x` faster than the current generated G=5 block tile.

If the primitive is still slow, do not route it into the model.

Verdicts:

- `G5_GP4_PASS_GENERATED_PRIMITIVE`
- `G5_GP4_REFUTE_GENERATED_PRIMITIVE_STILL_SLOW`
- `G5_GP4_BLOCKED_CORRECTNESS`
- `G5_GP4_BLOCKED_RESOURCE_REGRESSION`

## Phase GP5: Route Binding + W==D

Only if GP4 passes, bind the generated primitive behind a default-off flag:

```text
DECODE_FLASH_G5_GENERATED_ISA=0
```

Required gates:

- route-bound: generated G=5 primitive fires, no hidden fallback;
- token-match at ctx128/512/2048/4096 where feasible;
- W==D vs shipped default;
- no protected-context regression;
- BoltBeam ingest/evaluate updates the ledger.

Verdicts:

- `G5_GP5_PASS_TIER_A_OR_B`
- `G5_GP5_CORRECT_BUT_NOT_FAST`
- `G5_GP5_REFUTE_WD_REGRESSION`
- `G5_GP5_BLOCKED_ROUTE_ATTRIBUTION`

## Phase GP6: BoltBeam Promotion / Ledger

BoltBeam decides from evidence:

- promote only if W==D tier and guardrails pass;
- keep as candidate if it is capacity/diagnostic only;
- refute if mechanism passes but W==D regresses;
- record reopen condition if the generator still lacks a primitive.

The ledger must explicitly say whether this was:

- `generated_isa_primitive`;
- `handwritten_kernel` (should be impossible under this scope);
- `renderer_capability`;
- `search_space_incomplete`;
- `structural_refute`.

## Expected Outcome

This scope may still refute the route. That is acceptable.

The successful outcome is not "we wrote a faster G=5 kernel by hand." The successful outcome is:

```text
the machine-authored/generated path learned a reusable primitive that reduces the G=5 bloat
```

If it wins, it can generalize to other GQA geometries. If it fails, BoltBeam records the precise generator capability
that is missing.

## Short Answer

No, this must not be a handwritten kernel. The primitive route means:

```text
generic primitive/lowering/search improvement -> generated G=5 tile -> measured promotion gate
```

not:

```text
manual RDNA3 kernel -> route around tinygrad
```
