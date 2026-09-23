# 8B Prefill Pure Full-Kernel Machine Search Scope

Date: 2026-07-11

Status: execution scope; implementation authorized.

## 1. Objective

Build a strict-pure, full-kernel machine-search path for Qwen3-8B fp16 prefill GEMM on an AMD Radeon RX 7900 XTX
(`gfx1100`). The first anchor is:

```text
role: ffn_gate_up
operation: A[512,4096] @ Bt[4096,12288] -> C[512,12288]
dtype: fp16 inputs and output
target: AMD gfx1100, wave32
```

The selected candidate must be generated through normal Tinygrad compiler/backend lowering. Search must own every
performance-relevant decision in the declared search space. No route-local instruction list, `Ops.INS` program,
embedded source, binary, whole-kernel UOp template, or hidden handwritten fallback may execute.

The first goal is not immediate whole-model promotion. It is one exact, correct, route-bound, compiler-generated
kernel whose complete identity and evidence can be searched reproducibly. Performance is then improved toward the S9
oracle before expanding to other roles and shapes.

## 2. Why This Anchor

`ffn_gate_up` is the only current LDS/DBUF prefill family. It is selected twice per transformer block (gate and up),
has a strong handwritten oracle, and already has extracted layout, wait, cadence, lifecycle, legality, and search
artifacts. It directly exercises the missing compiler capability instead of duplicating the existing pipe Tensor path.

Expansion order after the anchor:

1. `attn_qo`: `M=512,N=4096,K=4096`; simplest pipe family.
2. `ffn_down`: `M=512,N=4096,K=12288`; long-K pipe stress.
3. `attn_kv`: `M=512,N=1024,K=4096`; last because local staging currently crosses the 64 KiB LDS limit.
4. Other prefill `M` buckets and model profiles only after exact-shape ownership and policy-domain gates pass.

## 3. Non-Negotiable Definition Of Pure

A candidate is strict-pure only when all conditions hold:

1. BoltBeam creates or selects a versioned, serializable candidate descriptor.
2. Tinygrad binds that exact descriptor to one compiled program.
3. Kernel topology originates in compiler-owned Tensor/UOp IR and shared backend primitives.
4. The normal Tinygrad renderer emits the executable.
5. No selected path contains route-local `Ops.INS`, ASM/source strings, precompiled binaries, or handwritten full-kernel
   templates.
6. No safety/resource path silently falls back to a handwritten route.
7. Candidate hash, compiler input, compiled binary hash, runtime route, and measured result form one evidence chain.
8. Static legality, compiled resources, numerical correctness, route binding, and performance all pass.
9. Applicability is explicit; an exact-shape result is never extrapolated to another shape or GPU.

Backend-owned reusable intrinsics are allowed. A shared renderer implementation of WMMA, DS operations, waitcnt, or
register constraints is compiler infrastructure. A role-local function that writes the complete instruction lifecycle
is not.

## 4. Current Truth Baselines

| Path | pp512 | pp4096 | Classification | Use |
|---|---:|---:|---|---|
| Ordinary scheduler baseline | ~1629 | ~1420 | strict pure, `tinygrad_scheduler_generated` | pure floor |
| Spec-owned composed route | ~1333 | ~1190 | `compiler_primitive_spec_owned__asm_backend_atom` | transitional evidence only |
| S9/hybrid oracle | ~4400 | ~3230 | `external_handwritten_kernel` | ceiling/oracle only |
| Pipe-only research outcome | ~192 | ~186 | strict-pure diagnostic, incomplete structure | not selectable; diagnostic evidence only |

These regimes are incomparable unless the same model, shape, clocks, timing method, and output semantics are used.

## 5. Architecture To Preserve And Improve

### 5.1 BoltBeam control plane

Reuse and extend:

- `search/spec.py`: `Constraints`, `SearchRow`, `AcceptedPolicy`, JSONL authority.
- `search/emit.py`: model/profile/role candidate-space emission.
- `search/reachability.py`: target-incomplete, primitive-missing, emitter-blocked, refuted, and reachable states.
- `search/epoch_model.py`: immutable epoch/event evidence taxonomy.
- `search/{epoch,timing,resource}_join.py`: evidence joins.
- `search/mmq_bundle.py` and `search/mmq_controller.py`: strict bundle validation and failure isolation pattern.

BoltBeam owns candidate generation, search strategy, budgets, durable results, ranking, reachability, policy selection,
and promotion decisions. It must not import Tinygrad runtime internals or emit GPU instructions.

### 5.2 Tinygrad execution plane

Reuse and improve:

- `extra/qk/prefill_schedule_spec.py`: canonical role schedule data.
- `extra/qk/wmma_lds_spec.py`: LDS register, memory, wait, cadence, lifecycle, epoch, resource, and legality contract.
- `extra/qk/wmma_pipe_spec.py`: pipe-family descriptor and generated diagnostic.
- `extra/qk/prefill_graph_gemm_route.py`: existing route integration seam.
- `tinygrad/schedule/rangeify.py`: stage/DBUF graph ownership.
- `tinygrad/codegen/opt/postrange.py`: Tensor Core staging and layout transforms.
- `tinygrad/renderer/isa/amd.py`: shared AMD instruction lowering.
- `extra/qk/runtime_specs.py` and `extra/qk/generated_candidates.py`: runtime candidate contract and registry.
- `extra/qk/pure_search_guard.py` and `extra/qk/route_manifest.py`: purity and effective-route authority.
- `extra/qk/prefill_v2_schedule_search.py` and `prefill_v2_schedule_table_gate.py`: compile/resource/correctness/timing mechanics.
- `extra/qk/prefill_whole_synced.py`: whole-model timing, reproducibility, and route binding.

Tinygrad owns typed candidate lowering, compiler hooks, static/compiler legality, execution, binary/resource capture,
correctness, ISA/provenance evidence, and route-bound measurements. It must not grow a second search-policy stack.

### 5.3 Shared identity boundary

The systems exchange a canonical candidate document and evidence bundle, not process-global anonymous flags.

```text
BoltBeam FullKernelCandidate
  -> canonical JSON + candidate_hash
  -> Tinygrad adapter validates schema/profile/shape/target
  -> PrefillGEMMScheduleSpec + WMMALDSSpec
  -> compiler-owned lowering context
  -> compiled_program_hash + binary_hash
  -> evidence bundle keyed by candidate_hash
  -> BoltBeam result index and policy decision
```

Environment variables remain compatibility/debug adapters during migration. They may not be the canonical candidate
identity.

## 6. Explicit Anti-Duplication Rules

Do not create:

- another route manifest, purity registry, or route selector;
- another standalone GEMM IR disconnected from Tinygrad Tensor/rangeify/postrange;
- another LDS or WMMA renderer beside the existing AMD backend;
- a copy of `build_gemm_pipe`, `build_gemm_lds2`, or their instruction lists;
- another whole-prefill benchmark harness;
- one search script per individual axis;
- a third schema between BoltBeam `SearchRow` and Tinygrad runtime specs;
- a policy store independent of accepted policy, runtime registry, and route manifest;
- a new SQLite service before append-only indexed artifacts prove insufficient.

Existing historical S9/S10 scripts remain evidence and adapters. They are not the architecture for the new runner.

## 7. Phase 0: Exact-Shape Mastery Dossier

This phase is blocking. No generalized search implementation begins until the dossier states every unknown and records
fresh exact-shape evidence.

### 7.1 Already known

- Exact operation, dtype, strides, bias-free semantics, and role multiplicity.
- Fast schedule: `BM=128, BN=128, BK=32`, waves `4x2`, `WM=2`, `WN=4`, 256 threads, pad 16, DBUF, PLRAB.
- Estimated LDS: 40,960 bytes.
- gfx1100 wave32 WMMA and shared AMD renderer primitives.
- Declarative `PrefillGEMMScheduleSpec` and `WMMALDSSpec` contracts.
- S9 oracle band around 73-75 TFLOPS and whole-model 4.4K pp512.
- Existing slot/window, lifecycle, stage-owner, and route provenance audits.
- Diagnosed stage-identity loss between postrange and late lowering.

### 7.2 Required fresh evidence

1. Role-isolated timing for ordinary pure, spec-owned, and S9 oracle paths.
2. Exact call multiplicity and measured whole-prefill wall share at 512/1024/2048/4096.
3. Exact-shape compiled ISA for pure baseline and oracle.
4. Lane-to-A/B fragment map and accumulator-to-C/store map.
5. Per-epoch reaching-definition graph from global tile through LDS slot, waits, barrier, fragment load, and WMMA.
6. Exact VGPR, SGPR, scratch/spill, LDS, workgroup, and occupancy data for both sides.
7. Calculated and, where available, measured LDS bank-conflict evidence.
8. Waitcnt dependency proof and pinned sensitivity measurements.
9. Full-output numerical comparison across repeated seeds and adversarial magnitude distributions.
10. Search-axis reachability: changing every declared field must change IR/ISA or be rejected as a no-op.
11. Executable provenance proving absence/presence of forbidden lowering surfaces.

### 7.3 Dossier artifact

Produce one immutable bundle:

```text
bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/
  manifest.json
  shape.json
  semantics.json
  pure-baseline.json
  oracle.json
  role-timing.json
  lane-fragment-map.json
  epoch-graph.json
  resources.json
  isa-diff.json
  correctness.json
  unknowns.json
```

Gate: `ANCHOR_MASTERY_COMPLETE` or `ANCHOR_MASTERY_BLOCKED` with named missing facts. Never infer exact-shape counters
from representative 5120x5120 studies.

## 8. Phase 1: Canonical Full-Kernel Candidate Contract

Extend `SearchRow`; do not replace it. Add a versioned full-kernel payload containing:

- workload/profile/role/shape/dtypes/layout/target;
- tile M/N/K, waves M/N, wave size, threads;
- lane ownership and cooperative global-load mapping;
- A/B global vector widths and alignment requirements;
- LDS windows, strides, padding, banks, store/load vector widths;
- buffer count, stage count, prologue/body/tail epoch graph;
- WMMA instruction family, fragment layout, accumulator ownership;
- waitcnt/barrier dependency policy;
- preload/residency/reuse policy;
- epilogue/store lane mapping and vector width;
- numerical mode;
- static constraints and applicability domain;
- schema version and canonical hash.

Extend existing `PrefillGEMMScheduleSpec`/`WMMALDSSpec` fields only where the dossier proves a missing decision. Add a
single explicit adapter from BoltBeam candidate JSON to Tinygrad runtime specs. Reject unknown versions and fields.

Gate: serialize -> hash -> adapt -> serialize round trip is stable, and every field is either compiler-reachable or
explicitly classified metadata-only.

## 9. Phase 2: Per-Kernel Compiler Context

Replace process-global search identity with a typed per-kernel context consumed by rangeify, postrange, and the AMD
renderer. Keep env adapters solely for existing tests/debug.

Required properties:

- candidate identity travels with the kernel;
- concurrent/different kernels cannot leak settings into one another;
- cache keys include candidate hash and schema version;
- generated program metadata records candidate hash;
- unsupported fields fail closed;
- default compilation is behavior-neutral.

Gate: two different candidates compiled in one process produce independently attributable programs and cache entries.

## 10. Phase 3: Fixed Single-Buffer Pure Kernel

Before DBUF, hard-code one dossier-proven legal structure through the existing compiler chain:

```text
candidate
 -> PrefillGEMMScheduleSpec/WMMALDSSpec
 -> Tensor/rangeify stage representation
 -> postrange cooperative A+B staging
 -> AMD renderer WMMA/DS/wait lowering
 -> normal compiled program
```

Required work:

- preserve source/epoch/slot identity through late lowering;
- implement real both-operand medium-shape staging;
- implement cooperative lane partition for source B;
- bind fragment layout and accumulator stores;
- bind the existing route with raw fallback prohibited;
- capture exact program/binary identity and compiled resources.

Gates, in order:

1. Static legality.
2. Compiler construction without forbidden surfaces.
3. Full-output micro correctness.
4. Route binding with `PURE_MACHINE_SEARCH_ONLY=1`.
5. Compiled-resource safety and no spill unless explicitly allowed.
6. Deterministic repeated timing.

Performance is recorded but not a blocking parity requirement in this phase.

## 11. Phase 4: Generated DBUF Lifecycle

Implement DBUF only after single-buffer correctness. Extend the existing rangeify stage contract with a generated event
graph:

- prologue produces epoch 0;
- body consumes epoch `i` and produces `i+1` into the alternate slot;
- waits distinguish global completion from LDS completion;
- barriers protect slot reuse across waves;
- tail drains the final produced epoch;
- reaching definitions include source tile, epoch, slot, and operand;
- stores may be suppressed only with a proven equivalent reaching definition.

No hard-coded physical registers or role-specific instruction list may enter the lifecycle representation.

Gates:

- two-slot identity proof;
- no read-before-produce, overwrite-before-consume, or cross-wave barrier hazard;
- epoch trace matches candidate descriptor;
- full correctness and compiled-resource gates;
- stage-count and wait-count ISA evidence;
- single-buffer versus DBUF controlled timing.

## 12. Phase 5: Unified Candidate Evaluator

Extract the reusable mechanics already present in the schedule-table gate and S9 searches into one evaluator:

```text
canonicalize/dedupe
 -> static validate and estimate
 -> compile with timeout/crash quarantine
 -> inspect compiled resources
 -> provenance/forbidden-surface gate
 -> bounded full-output correctness
 -> timed kernel measurement
 -> immutable evidence bundle
```

Correctness ladder:

1. deterministic small structural probe;
2. exact anchor full output against trusted Tensor reference;
3. repeated random/adversarial inputs with declared tolerances;
4. route-bound model chunk logits;
5. whole-model dNLL/greedy parity before promotion.

Resource authority order:

1. static estimates reject obvious invalid candidates;
2. compiled metadata/ELF is authoritative for VGPR/SGPR/LDS/scratch/workgroup;
3. runtime faults quarantine the candidate and preserve logs;
4. measured occupancy/counters inform scoring but never override illegality.

## 13. Phase 6: Durable Search And Evidence Store

Use content-addressed immutable artifacts plus an append-only result index. Key each result by:

- candidate hash and schema version;
- workload/profile/role/shape/applicability domain;
- GPU identity, firmware/driver/runtime snapshot, clocks;
- Tinygrad and BoltBeam commits;
- compiler/backend identity and relevant compatibility env;
- program and binary hashes;
- benchmark regime and harness version.

Support resume, dedupe, compile timeout, crash quarantine, retry policy, and exact replay. JSONL plus content-addressed
directories is the initial implementation. Introduce SQLite only if concurrent query/update requirements justify it.

Generalize the MMQ evidence-envelope pattern rather than creating a prefill-only bundle. Evidence joins must never join
different candidate, binary, shape, target, commit, or timing identities.

## 14. Phase 7: Search Strategy

Search is multi-fidelity:

1. Enumerate only statically legal structural families.
2. Compile and resource-filter.
3. Run bounded correctness.
4. Run short kernel timing with repeated samples.
5. Promote the Pareto set to longer timing and counters.
6. Promote finalists to route-bound model chunks.
7. Run whole-prefill and quality authority only for finalists.

Initial search axes come from existing extracted S9 facts:

- tile and wave geometry;
- BK and padding;
- lane/load partition;
- global/DS vector widths;
- single versus double buffer;
- preload/resident A/B policy;
- wait thresholds and barrier placement;
- fragment/accumulator layout choices;
- epilogue store mapping.

Do not begin with Bayesian/evolutionary machinery. Start with bounded, deterministic enumeration and measured axis
reachability. Add adaptive proposal only after the legal space and durable resume semantics are proven.

Search outcomes use explicit labels:

- `SEARCH_FOUND_PROMOTABLE`
- `SEARCH_EXHAUSTED_SPACE`
- `SEARCH_SPACE_INCOMPLETE`
- `SEARCH_BLOCKED_BY_CODEGEN`
- `SEARCH_BLOCKED_BY_RUNTIME`
- `SEARCH_BLOCKED_BY_CORRECTNESS`

## 15. Phase 8: Performance Convergence

Compare generated candidates to both the strict-pure scheduler floor and S9 oracle.

Milestones:

1. Correct strict-pure kernel, any speed.
2. Beat the current pipe-only diagnostic outcome.
3. Recover the ordinary pure scheduler baseline.
4. Beat the scheduler baseline materially under pinned repeated timing.
5. Reach 50%, 75%, 90%, then parity with the oracle kernel TFLOPS.
6. Demonstrate whole-prefill improvement without quality or provenance regression.

For every performance gap, attribute at least:

- useful WMMA density;
- global and LDS bytes/instructions per WMMA;
- wait/barrier counts and dependency reason;
- VGPR/SGPR/scratch/LDS/occupancy;
- wave/workgroup count and tail behavior;
- kernel launch and graph wall share.

A search result without a declared search space and excluded primitive list cannot establish exhaustion.

## 16. Phase 9: Role And Shape Generalization

Reuse the same candidate schema, evaluator, store, and compiler context. No role-specific runner is allowed.

For each new role:

1. create a mastery delta dossier;
2. prove semantic/layout differences;
3. enumerate applicability constraints;
4. run exact-shape search;
5. verify policy refuses unsupported shapes;
6. measure whole-model effect.

Context-length generalization is primarily an `M`/chunk-domain problem. Policies must state exact buckets or predicates;
they may not assume that a 512-row winner generalizes.

## 17. Phase 10: Promotion And Retirement

Promotion requires one atomic evidence-backed change across:

- BoltBeam accepted policy;
- Tinygrad generated candidate registry;
- route manifest/provenance;
- strict purity census;
- correctness/quality artifacts;
- whole-prefill authority artifact;
- rollback/reference declaration;
- tests and durable ledger.

Promotion gate:

- clean commits and reproducible environment;
- pinned-clock repeated band;
- exact candidate and binary identity;
- no forbidden surface or fallback;
- quality pass;
- declared threshold pass against same-regime comparator;
- applicability-domain enforcement;
- rollback tested.

Only after pure generated coverage and performance pass may raw fallback eligibility be removed. The S9/hybrid kernel may
remain an oracle fixture but must not execute on the promoted path.

## 18. Agent Execution Graph

Minimum useful agent topology:

### Parallel tranche A

- **A1 Candidate contract (BoltBeam):** extend `SearchRow` with versioned full-kernel payload and canonical hashing;
  add round-trip, rejection, and compatibility tests.
- **A2 Anchor dossier (Tinygrad):** create the dossier builder that joins existing exact-shape facts and reports named
  missing evidence without inventing values.
- **A3 Execution adapter audit/implementation (Tinygrad):** extend runtime specs and generated candidate support for the
  canonical candidate ID, shape/target enforcement, and fail-closed adapter skeleton.

Dependencies: none between A1/A2; A3 consumes the agreed A1 schema contract and may initially use fixtures.

### Sequential tranche B

- Per-kernel compiler context and cache identity.
- Fixed single-buffer route-bound pure lowering.
- Full correctness/provenance/resource capture.

Dependency: Phase 0 and Phase 1 gates pass.

### Sequential tranche C

- Generated DBUF epoch materializer.
- DBUF correctness/resource/timing gates.
- Unified evaluator extraction.

Dependency: fixed single-buffer kernel passes.

### Parallel tranche D

- BoltBeam durable result store/controller.
- Search-space reachability and deterministic enumerator.
- Tinygrad compiled evidence bundle and whole-model quality producer.

Dependency: evaluator contract stable.

### Sequential tranche E

- Anchor search and convergence.
- Role expansion.
- Whole-model promotion package.

Agents must modify existing modules unless a new module represents a genuinely new contract. Every task begins with a
duplication audit and ends with tests plus an explicit list of reused components.

## 19. Stop Conditions

Work stops only at a documented complete standstill:

- three consecutive attempts hit the same named compiler/runtime/correctness blocker;
- the blocker cannot be reduced with existing diagnostics or a bounded new experiment;
- all artifacts, failing candidates, logs, and reopen conditions are committed;
- the search-space label is `INCOMPLETE` or `BLOCKED`, never `EXHAUSTED` unless coverage is proven.

Performance regression, long compile time, or a failed candidate is not a standstill.

## 20. Immediate Definition Of Done

The first implementation tranche is complete when:

1. This scope is committed.
2. BoltBeam has a versioned canonical full-kernel candidate payload and stable hash.
3. Tinygrad has an anchor mastery-dossier builder using existing evidence and naming missing facts.
4. Tinygrad can parse/validate the candidate identity without executing handwritten code.
5. Pipe-only research is not publicly selectable.
6. Unit tests pass in both repositories.
7. The next blocker is a concrete compiler-lowering task, not schema, route, or evidence ambiguity.

## 21. Success Definition

Full pure machine search is achieved for the anchor only when BoltBeam can generate a bounded candidate family, Tinygrad
can compile and execute every legal candidate through normal compiler lowering, the evaluator can reject or measure it
without manual intervention, and a selected candidate passes strict provenance, full correctness, resources, pinned
timing, and route binding with no handwritten fallback.

Whole-route success additionally requires all four prefill roles and declared context buckets to satisfy the same
contract and pass whole-model quality and authority gates.
