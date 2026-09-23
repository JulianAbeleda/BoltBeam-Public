# BoltBeam-to-tinygrad pure pipe integration scope

## Question this answers

How far are we from using BoltBeam to search compiler-owned WMMA pipe kernels
on the AMD 7900 XTX, and what must be implemented before that search is valid?

The short answer is that BoltBeam's search and evidence control plane is mostly
in place. The missing work is the execution-plane contract inside tinygrad:
turning a candidate into a graph/compiler operation whose stage ownership,
fragment lifetime, and wait dependencies are actually lowered and executed.

## Current state by layer

| Layer | Status | What is proven |
|---|---|---|
| GPU/target identity | Complete for current target | gfx1100 target, clock-pin protocol, resource/evidence fields |
| Candidate schema | Complete | role, shape, target, canonical identity, provenance |
| Candidate persistence | Complete | full-kernel candidate-set manifests and duplicate identity checks |
| Search/evaluation policy | Complete for orchestration | candidate emission, evidence filtering, route promotion rules |
| Benchmark authority | Complete for existing routes | pinned protocol and hybrid reference artifacts |
| Generated gate/up role | Working | generated buffer2 candidate is route-bound and measured |
| Typed non-LDS pipe contract | Host-only complete | immutable IR, lifecycle validation, ABI/cache tests |
| Graph/compiler lowering | Missing | no pipe-specific stage/wait operation reaches codegen |
| Pure AMD execution | Missing for non-LDS pipe | generic LLVM WMMA exists; pipe lifecycle does not |
| Pure machine search | Not yet valid | search must wait for executable pure candidates |

The current pure implementation percentage is therefore not the percentage of
BoltBeam code. BoltBeam is roughly complete as a candidate/evidence system, but
the end-to-end pure executable route is still at the compiler integration
boundary.

## Ownership split

### BoltBeam owns

- candidate-space grammar and role-specific parameters;
- canonical candidate identity and schema versioning;
- target/model/workload applicability;
- search populations and mutation bounds;
- benchmark protocol, clock-pin status, and artifact joins;
- route census and fallback detection;
- correctness/timing/resource evidence aggregation;
- promotion, rejection, and durable ledger decisions.

BoltBeam must not emit ISA text, instruction tuples, precompiled binaries, or
backend-specific register assignments.

### tinygrad owns

- typed graph operation and verifier;
- shape/dtype/stride/target validation at graph construction;
- stage-slot lifecycle and dependency scheduling;
- load grouping, WMMA fragment construction, accumulator lifetime;
- targeted wait semantics or a proven replacement dependency;
- AMD backend lowering and final resource extraction;
- normal program/cache/runtime ABI;
- executed-binary/source identity.

BoltBeam can only search a parameter after tinygrad proves that parameter is
compiler-owned and appears in the executed candidate identity.

## Important backend fact

tinygrad has two relevant AMD paths:

1. `AMDLLVMRenderer` can lower ordinary `LOAD`, `STORE`, and `Ops.WMMA` through
   LLVM AMDGPU intrinsics. This is the safe pure-capable backend.
2. `AMDISARenderer` has explicit b128 fragment loads, WMMA instructions, and
   targeted `s_waitcnt`, but those helpers materialize `Ops.INS`/`AMDOps` and are
   therefore the existing hybrid/native path.

Generic LLVM WMMA is not equivalent to the pipe primitive. The current
postrange path ignores the pipe IR's stage ownership and wait fields. Attaching
candidate metadata to a generic matmul would produce false pure provenance.

## Required cross-repo contract

The minimum executable contract is:

```text
BoltBeam candidate
  -> canonical identity + typed pipe payload
  -> tinygrad KernelInfo.candidate_context
  -> verified typed pipe graph operation
  -> stage/fragment/wait schedule
  -> AMD LLVM lowering
  -> ProgramInfo/cache/runtime ABI
  -> executed source/binary/resource artifact
  -> BoltBeam evidence join and search result
```

Every arrow must preserve:

- schema version and canonical identity;
- role and exact `(M,N,K)` shape;
- A/B/output buffer order and dtypes;
- launch dimensions and strides;
- stage count, load-group, and wait policy;
- source hash, binary hash, and resource provenance.

## Implementation phases

### Phase 0 — Freeze the boundary

No new BoltBeam candidate knobs. Mark the current non-LDS pipe candidates as
`typed_host_only` and keep them out of pure promotion. Add a cross-repo schema
fixture consumed by both projects.

### Phase 1 — Tinygrad typed graph operation

Add the smallest first-class operation or compiler-side node that carries the
pipe lifecycle without embedding ISA. Integrate verifier, rangeify, graph key,
and ordinary-graph compatibility tests together. A fields-only enum is not
acceptable.

### Phase 2 — Lifecycle scheduler

Lower the node into stage ownership, cooperative loads, fragment lifetimes,
K-loop progression, accumulator dependencies, and typed waits. Reuse existing
stage/fragment validation helpers where possible, but bind model buffers and
contiguous fp16 strides explicitly.

### Phase 3 — Pure LLVM vertical slice

Implement `attn_qo` at `512x4096x4096` through `AMDLLVMRenderer`. Prove the
generated LLVM source contains the intended loads/WMMA/stores, and prove the
wait/dependency behavior rather than importing native `AMDOps`.

### Phase 4 — ABI/resource/binary join

Preserve candidate identity through lowering/cache/runtime/HCQ replay. Extract
final VGPR/SGPR/LDS/scratch/wave facts from the executable program. Join source
hash, binary hash, resource facts, and route census to the BoltBeam candidate.

### Phase 5 — Role expansion

Parameterize `ffn_down` and `attn_kv` only after `attn_qo` passes independent
correctness and resource gates. KV requires its own small-N occupancy and tail
proof.

### Phase 6 — Pure authority and search

Combine generated gate/up with all passing non-LDS roles. Run the pinned whole-
model authority at ctx512, then larger contexts. Only after this passes may
BoltBeam search compiler-owned pipe knobs and promote whole-model winners.

## BoltBeam changes required

BoltBeam should remain small and declarative. Required changes are limited to:

- a candidate status distinguishing `typed_host_only`, `compiled_unproven`,
  `executed_pure`, and `executed_hybrid`;
- schema validation for compiler-owned versus diagnostic-only fields;
- artifact join requirements for source/binary/resource/route census;
- evaluator rejection when execution provenance is missing or fallback occurs;
- search-space gating so unproven knobs cannot enter promotion;
- cross-repo fixture tests for candidate identity and exact shape/ABI.

No BoltBeam change can substitute for tinygrad lifecycle lowering.

## Completion gates

The integration is complete only when all are true:

1. BoltBeam emits a candidate that tinygrad accepts as a typed graph operation.
2. The operation lowers through the normal compiler without route-owned ISA.
3. Stage ownership and waits are compiler-verified and reflected in artifacts.
4. The executable source/binary/cache identity joins to the BoltBeam candidate.
5. `attn_qo`, `ffn_down`, and `attn_kv` pass independent parity/resource gates.
6. Generated gate/up plus those roles execute with strict pure provenance.
7. Pinned whole-model timing is reproducible and has no fallback.
8. Search varies only compiler-owned parameters and promotes only whole-model
   winners.

Until gate 3, BoltBeam may search schemas and diagnostics but may not claim
pure kernel search. Until gate 7, it may not claim a pure whole-model speedup.

## Current distance

The practical distance is one major compiler workstream, not a missing search
algorithm:

- BoltBeam control/evidence plane: substantially complete;
- tinygrad host contract and scope: complete;
- tinygrad executable pipe lowering: not implemented;
- first pure role authority: not implemented;
- combined pure route and machine-search promotion: blocked behind those gates.

The correct next task is Phase 1/2 in tinygrad. Adding more BoltBeam search knobs
before that work would only search metadata that the executable compiler does
not yet consume.
