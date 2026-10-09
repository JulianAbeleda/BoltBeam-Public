# MMQ Closed-System Loop Implementation Scope

Date: 2026-07-11

Repositories:

```text
BoltBeam: /home/ubuntu/BoltBeam
tinygrad: /home/ubuntu/tinygrad-arkey
```

## Objective

Build the minimum closed experimental loop that lets BoltBeam bind a kernel
candidate to an exact system, ask tinygrad to emit and execute it, join the
resulting evidence, and make only the conclusions supported by that evidence.

The first loop is deliberately narrow. It tests whether the current cooperative
MMQ atom's 256 gated stores are a material cause of its performance loss. All
numeric work, staging, geometry, K-loop behavior, inputs, and comparator behavior
remain fixed while the writeback form changes.

This scope does not promote a route, generalize machine search to every kernel
knob, or reorganize the full BoltBeam package.

## Existing Foundations To Reuse

Do not replace these surfaces:

```text
boltbeam.artifacts.base.NormalizedEvidence
boltbeam.artifacts.base.EvidenceSource
boltbeam.workflow run manifests and stage handling
boltbeam.search epoch, R4, timing, and resource joins
tinygrad MMQ correctness, timing, resource, ISA, and owner artifacts
```

New contracts extend those foundations. Existing v1 artifacts remain readable.

## Frozen Identity Contract

Backend identity:

```text
q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0
```

Initial candidate identities:

```text
mmq.wb.gated_matrix.m16.n16.k256.v1
mmq.wb.direct_owner.m16.n16.k256.v1
```

Every experiment artifact must carry the same:

```text
candidate_id
backend
shape
experiment_id
system_snapshot_id
source_sha256
binary_sha256, when a compiled binary exists
producer repository commits
```

Reject candidate, backend, shape, experiment, system, source, or binary identity
mismatches. Do not synthesize replacement candidate ids in adapters.

The existing identities `cooperative_multi_wave_tile`,
`r5_ds4_coop_tile_16x16`, and synthesized timing ids are legacy aliases only.
New experiment bundles use the canonical ids above.

## New Schemas

```text
boltbeam.fact.v1
boltbeam.system_snapshot.v1
boltbeam.experiment_manifest.v1
boltbeam.evidence_requirements.v1
boltbeam.mmq_experiment_bundle.v1
boltbeam.mmq_search_report.v1
boltbeam.mmq_diagnosis.v1
tinygrad.mmq_candidate_spec.v1
```

## Intended File Additions

BoltBeam:

```text
boltbeam/core/__init__.py
boltbeam/core/facts.py
boltbeam/core/system.py
boltbeam/core/experiment.py
boltbeam/core/requirements.py
boltbeam/characterize/__init__.py
boltbeam/characterize/system_snapshot.py
boltbeam/search/mmq_bundle.py
boltbeam/search/mmq_controller.py
boltbeam/search/mmq_diagnosis.py
tests/core/
tests/test_system_snapshot_collector.py
tests/test_mmq_bundle.py
tests/test_mmq_controller.py
tests/test_mmq_diagnosis.py
tests/fixtures/mmq_bundle/
```

Tinygrad:

```text
extra/qk/mmq_experiment.py
test/unit/test_mmq_experiment.py
```

Existing MMQ files may receive narrowly scoped changes. Avoid broad moves.

## Core Truth Contracts

### Fact

A fact records a value and how it is known:

```text
measured
derived
modeled
assumed
imported
unknown
```

Required fields and behavior:

```text
namespaced name
JSON value
unit
truth status
source artifact references and fingerprints
derivation and input fact ids where applicable
uncertainty or sample information where applicable
observation time
content-addressed fact_id
```

Invariants:

```text
measured requires a source and forbids a derivation
derived/modeled require a derivation and source or inputs
unknown requires value=null
other statuses require a value
non-finite numeric values are invalid
ids ignore paths and timestamps but retain semantic provenance
```

### SystemSnapshot

The snapshot binds an experiment to the exact closed system. It records facts
for:

```text
GPU UUID, PCI address/vendor/device/revision
all observed marketing identities
architecture, CU/SIMD counts, VRAM, caches, wave limits
VBIOS and firmware
kernel, amdgpu, ROCm, HIP, and compiler identities
tinygrad and BoltBeam commits
relevant environment
core/memory clock and power policy when observable
```

Conflicting XTX/GRE identities are preserved as separate measured facts plus a
conflict fact. Product defaults such as 960 GB/s are imported or assumed, not
measured. Unavailable probes create explicit unknown facts.

### ExperimentManifest

The immutable experiment definition includes:

```text
system snapshot id
candidate and comparator ids
model/workload contracts
input artifact hashes
argv commands and relevant environment
warmups, repetitions, order, seed, and timeout
cache, clock, and correctness policies
evidence requirements
```

Commands are argv arrays, never shell strings. The existing workflow run
manifest remains the mutable execution/output inventory.

### EvidenceRequirement

Requirements have levels `required`, `recommended`, or `optional` and may block
diagnosis, comparison, or promotion. Durable predicates name registered pure
validators; they do not contain executable lambdas.

Requirement results distinguish:

```text
satisfied
missing
invalid
mismatch
unsupported
```

## Initial Tinygrad Candidate Space

Candidate schema fields:

```text
schema
candidate_id
backend
shape M=16,N=16,K=256
knobs.writeback_mode
objective comparator_id=direct_packed, metric=median_ms
warmups=3
rounds=10
seed
```

Only two writeback modes are legal:

```text
gated_matrix_v0
direct_owner_v0
```

`gated_matrix_v0` is the current statically enumerated gated-store form.
`direct_owner_v0` uses the existing unique 16x16 output ownership and directly
stores each owner's output. No production dispatch changes are allowed.

## Atomic Experiment Bundle

Each Tinygrad experiment produces a directory containing:

```text
manifest.json
candidate.json
correctness.json
timing.json
resources.json
isa_manifest.json
ownership.json
logs/                  optional, not evidence
```

BoltBeam adds `diagnosis.json`; Tinygrad never consumes that diagnosis.

Producer protocol:

1. Parse and validate the candidate.
2. Compile the emitted atom.
3. Record source, numeric-body, writeback, and binary hashes.
4. Run bounded numeric correctness.
5. Validate ownership for the actual emitted candidate.
6. Collect static resources.
7. Export the ISA proof manifest.
8. Run candidate and direct comparator in the same session.
9. Preserve warmup and every measured raw sample.
10. Write into a temporary bundle directory.
11. Validate the bundle.
12. Rename atomically into its final location.

Failures preserve a structured partial bundle. A partial directory can never be
mistaken for an evidence-complete run.

## Required MMQ Evidence

A candidate may be ranked only when it has:

```text
matching identity chain
compile PASS
numeric PASS with comparator, tolerances, and max errors
256 expected and uniquely owned outputs
zero missing or duplicate outputs
scratch known and zero
ISA/global-store structural counts
static VGPR, SGPR, LDS, and workgroup resources
at least 3 warmups and 10 measured rounds
raw same-session candidate and direct_packed samples
identical system snapshot and clock policy
production_dispatch_changed=false
```

Unavailable dynamic counters are explicit. They block broader causal claims but
do not block the controlled writeback A/B comparison.

## Controller

Normal states:

```text
PLANNED
PRODUCING
COMPILED
CORRECTNESS_VALIDATED
EVIDENCE_COMPLETE
RANKED
```

Terminal states:

```text
REJECTED_COMPILE
REJECTED_CORRECTNESS
REJECTED_OWNERSHIP
INCOMPLETE_EVIDENCE
INCONCLUSIVE_TIMING
CORRUPT_BUNDLE
PRODUCER_ERROR
```

The controller continues after an individual candidate rejection. It never
changes route policy or promotes a candidate.

## Diagnosis Contract

A writeback-causal conclusion is legal only when:

```text
system, experiment, shape, inputs, and comparator sessions match
numeric-body hashes match
geometry, staging, synchronization, and K-loop identities match
correctness and ownership pass for both candidates
scratch is known and zero
static resource changes are reported
only the writeback form/hash differs intentionally
timing distributions are stable and decisively separated
```

The result is exactly one of:

```text
writeback_supported
writeback_refuted
inconclusive
```

This experiment can test the immediate store-form hypothesis. It cannot by
itself prove that writeback explains the full gap to llama.

## Work Packages And Scheduling

Use three implementation agents plus the integrating root agent.

```text
Root: contract owner, integration, review, GPU execution
Agent A: BoltBeam truth contracts and system snapshot
Agent B: Tinygrad candidate and atomic bundle producer
Agent C: BoltBeam bundle consumer, controller, and diagnosis
```

Schedule:

```text
Root     contract freeze -- reviews -- integration -- GPU proof -- hardening
Agent A  truth contracts -------- system snapshot -------- done
Agent B  candidate contract ----- bundle producer -------- done
Agent C  fixture preparation ---- bundle consumer -- controller/diagnosis
```

Agent A and Agent B work in parallel. Agent C may prepare fixtures and public
interfaces immediately, but completes bundle ingestion after Agent A's contracts
and Agent B's bundle format stabilize.

### WP-A: Truth And System

Deliver Fact, SystemSnapshot, ExperimentManifest, EvidenceRequirement, collector,
serialization, validation, compatibility adapters, and focused tests.

Acceptance:

```text
content ids are stable and semantic
truth statuses cannot be laundered
system conflicts and unknowns remain explicit
legacy evidence still loads
full BoltBeam tests pass
```

### WP-B: Tinygrad Producer

Deliver canonical candidate propagation, two writeback modes, experiment runner,
atomic bundle export, failure bundles, and focused tests.

Acceptance:

```text
both candidates compile and pass bounded numerics
actual ownership passes
writeback structures differ as declared
same-session raw timing is emitted
production dispatch is unchanged
focused Tinygrad tests pass
```

### WP-C: Consumer And Controller

Deliver bundle/hash/schema validation, existing evidence joins, requirement
evaluation, sequential producer orchestration, pair diagnosis, and fixtures.

Acceptance:

```text
complete fixture reaches EVIDENCE_COMPLETE
every injected mismatch fails with a named blocker
oracle cannot be ranked or promoted
controller continues after candidate rejection
causal diagnosis refuses every declared confounder
full BoltBeam tests pass
```

## Integration And GPU Proof

The root agent performs integration after each work package, not only at the end.
After unit integration:

1. Capture the live system snapshot.
2. Run gated writeback.
3. Run direct-owner writeback.
4. Repeat in reverse order.
5. Validate each bundle independently.
6. Compare raw distributions, ISA, ownership, and resources.
7. Run BoltBeam diagnosis.
8. Repeat if timing is inconclusive.
9. Commit authoritative bundles or compact evidence summaries and a handoff.

Required final gates:

```text
BoltBeam full suite passes
Tinygrad focused MMQ suite passes
real bundles validate
same-session comparator exists
system identity is bound
production dispatch remains unchanged
diagnosis does not exceed its evidence
```

## Cost Envelope

```text
BoltBeam production: 1000-1500 LOC
Tinygrad production: 350-550 LOC
Tests: 1150-1550 LOC
Fixtures/docs: 100-300 LOC
Expected effort: 4-7 engineering days plus GPU debugging
```

## Explicit Deferrals

Do not include these in the first closed loop:

```text
broad package-directory moves
universal kernel grammar
geometry, staging, unroll, and instruction search knobs
dynamic PMC implementation
database or asynchronous scheduler
full CLI split
roofline consolidation
route-manifest history migration
production promotion
```

After two real bundles succeed, consolidate the duplicated GGUF reader first.
Then introduce shared typed roofline rungs, split CLI registration, and consider
broader directory moves based on demonstrated dependency boundaries.

## Commit Strategy

Use narrow commits in dependency order:

```text
1. [docs] scope MMQ closed-system loop
2. [core] add truth and experiment contracts
3. [system] capture exact execution snapshot
4. [mmq] add canonical writeback candidates
5. [mmq] emit atomic experiment bundles
6. [search] validate MMQ experiment bundles
7. [search] drive and diagnose MMQ writeback search
8. [docs] record real GPU result and remaining blocker
```

Do not combine directory moves or unrelated cleanup with these commits.
