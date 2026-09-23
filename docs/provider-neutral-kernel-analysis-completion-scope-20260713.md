# Provider-Neutral Kernel Analysis Completion Scope

Date: 2026-07-13

Status: implementation-ready scope

Owner: BoltBeam

Execution boundary: external compiler/runtime providers compile and execute; BoltBeam requests, ingests, validates, compares, diagnoses, recommends, and remembers.

## 1. Objective

Complete BoltBeam as a provider-neutral, evidence-driven codegen performance analyst.

Given model/workload context plus artifacts from tinygrad, Triton, CK, CUDA, or another producer, BoltBeam must answer:

1. What happened to each candidate?
2. Is each candidate valid for diagnosis, correctness comparison, or performance comparison?
3. Are candidate and comparator evidence chains actually comparable?
4. What changed structurally and measurably?
5. Which mechanisms are consistent with the evidence?
6. How strong is each conclusion, and what remains unknown?
7. What single bounded experiment should the external runtime run next?

BoltBeam is complete when those answers require no manual artifact translation and never turn missing evidence into a pass or a causal claim.

This scope does **not** authorize BoltBeam to compile kernels, construct runtime executables, submit GPU work, manage queues, repair compiler code, or emit vendor ISA.

## 2. Baseline and measured distance

The repository currently provides roughly 65% of the required system:

- model/workload profiling and route-family search;
- canonical candidate payloads and admission handoffs;
- typed facts with measured/derived/modeled/assumed/imported/unknown status;
- evidence requirements and fail-closed candidate evaluation;
- durable route/candidate memory;
- timing and resource joins;
- an MMQ-specific atomic identity/correctness/resource/timing bundle;
- memory-tier and prediction-only LDS-vs-cache models;
- 679 passing tests at scope creation.

The missing work is concentrated in generic integration:

- current tinygrad full-kernel artifacts are not recognized by `boltbeam.artifacts.tinygrad.detect_kind`;
- `search.transfer_contract` requires five fixed filenames instead of schema-driven fragments;
- general full-kernel control stops at admission;
- strong MMQ bundle validation is tied to one backend, shape, and candidate pair;
- there is no provider-neutral structural contrast, diagnosis, or next-experiment report;
- the LDS/L2 model lacks complete uncertainty, occupancy, synchronization, and overlap treatment;
- no measured memory-hierarchy profile exists in the current workspace.

## 3. Non-goals

The implementation must not:

- import tinygrad runtime/compiler modules into the analysis core;
- call `get_runtime`, construct GPU programs, or dispatch GPU work;
- make filenames part of semantic identity;
- hide provider-specific fields inside the normalized model;
- require unavailable hardware counters;
- claim an exact hang cause from instruction counts alone;
- promote a modeled prediction as a measured winner;
- replace the existing evaluator or ledger with a second authority;
- create role-specific analyzers for attn_qo, MMQ, LDS2, or direct-L2;
- use model, target, provider, or candidate names as diagnostic switches;
- delete compatibility paths before callers and fixtures migrate.

`boltbeam/amd_runtime_bridge.py` is outside this architecture. Runtime construction belongs in a provider repository. Generic process isolation may remain runner/collector infrastructure, but analysis consumes only its serialized evidence.

## 4. Architectural rules

### 4.1 Evidence before interpretation

Adapters preserve claims and provenance. Eligibility validates them. Contrast computes differences. Diagnosis interprets them. Recommendation proposes the next experiment. No layer performs a later layer's job.

### 4.2 One normalized authority

All providers normalize into one `KernelEvidence` representation. MMQ, prefill, direct-L2, LDS, Triton, and future producers must not grow independent lifecycle vocabularies.

### 4.3 Decision-specific eligibility

A candidate can be usable for failure diagnosis but not correctness comparison; usable for correctness comparison but not performance comparison; or invalid for all decisions due to identity corruption. One Boolean `passed` is insufficient.

### 4.4 Truth status is mandatory

Use existing `TruthStatus`: measured, derived, modeled, assumed, imported, unknown. Missing remains unknown. Zero is an observed value.

### 4.5 Hypotheses, not invented causes

A timeout with barriers supports a synchronization/liveness hypothesis. It does not prove a divergent barrier. Every hypothesis records support, contradiction, missing evidence, confidence, and a discriminating experiment.

## 5. Reuse and consolidation

| Existing authority | Reuse |
|---|---|
| `core.experiment.ExperimentManifest` | experiment identity, comparator, protocol, inputs, timeout |
| `core.facts.Fact` / `TruthStatus` | values, provenance, uncertainty, unknowns |
| `core.requirements` | decision-specific evidence requirements |
| `artifacts.base.EvidenceSource` | source fingerprinting |
| `search.spec.FullKernelCandidate` | canonical candidate identity |
| `search.resource_join` | resource validation primitives |
| `search.timing_join` | timing validation primitives |
| `search.mmq_bundle` | identity-chain design, not hard-coded constants |
| `eval.evaluator` | fail-closed gate concepts |
| `ledger` | durable conclusions and reopen conditions |
| `perf.mem_tier` / `lds_l2` | LDS/cache analytical plugin |
| `runner_plan` | external provider materialization |

Consolidation requirements:

- `search.transfer_contract` becomes a compatibility facade over schema-driven ingestion.
- MMQ retains extra domain gates but adapts to the generic model before generic analysis.
- Route-policy evaluation remains downstream; kernel analysis cannot promote.
- Timing/resource validators expose reusable pure helpers instead of duplicated rules.

## 6. Package plan

Create:

```text
boltbeam/kernel_analysis/
  __init__.py
  model.py          # normalized evidence/report types; sole schema authority
  adapters.py       # schema registry and fragment normalization
  eligibility.py    # identity joins and decision requirements
  contrast.py       # structural, numerical, timing deltas
  diagnosis.py      # evidence-bounded hypotheses
  recommend.py      # one-variable experiment recommendations
  report.py         # orchestration and rendering
```

Edit only as needed:

```text
boltbeam/vocab.py
boltbeam/cli.py
boltbeam/search/transfer_contract.py
boltbeam/search/mmq_bundle.py
boltbeam/search/resource_join.py
boltbeam/search/timing_join.py
boltbeam/lds_l2.py
boltbeam/perf/mem_tier.py
boltbeam/runner_plan.py
README.md
tests/kernel_analysis/
tests/fixtures/kernel_analysis/
```

Do not create a provider-specific top-level package.

## 7. Public schemas and normalized model

Add to `vocab.py`:

```python
SCHEMA_KERNEL_EVIDENCE = "boltbeam.kernel_evidence.v1"
SCHEMA_KERNEL_ANALYSIS = "boltbeam.kernel_analysis.v1"
SCHEMA_KERNEL_RECOMMENDATION = "boltbeam.kernel_recommendation.v1"
```

Avoid separate public schemas for every nested object.

### 7.1 KernelEvidence

Immutable dataclasses with `to_json`/`from_json`, JSON validation, and content-derived IDs:

```text
schema
evidence_id
candidate
identity
workload
stages
correctness
resources
structure
timing
health
provenance
blockers
```

Candidate:

- required `candidate_id`;
- optional canonical `candidate_hash`;
- semantic `family`, high-level `strategy`, `operand_transports`, `knobs`, `comparator_id`;
- names are labels, never diagnostic controls.

`operand_transports` is a map keyed by semantic operand identity (`a`, `b`, or provider-normalized equivalents). Each
operand independently declares `register_resident`, `lds_staged`, `cache_streamed`, or `reloaded`. A kernel-wide
`strategy` is a derived summary only; it must not prevent hybrid candidates such as A-register/B-LDS. Physical
L0/L1/L2/MALL/DRAM residency is observed evidence and never encoded as an explicit placement claim unless the emitted
program proves a corresponding cache-control policy.

Identity:

- source/program/binary/executed-binary hashes;
- input/reference hashes when available;
- provider/compiler/version/commit;
- target/backend/architecture/device;
- system snapshot, experiment, and session IDs;
- ABI/global ordering when relevant.

Accept bare SHA-256 or `sha256:` input and serialize canonically. Missing identity produces eligibility blockers; malformed identity is invalid.

Workload:

- operation and role;
- normalized shape plus optional batch/head/context dimensions;
- input/output/accumulator dtypes;
- transpose/layout semantics;
- seed/input case and reference semantics;
- unknown provider dimensions retained under `extra`.

Central stage statuses:

```text
compile:     not_run | pass | unsupported | fail | timeout | no_result
execution:   not_run | pass | timeout | runtime_fault | health_failure | no_result
correctness: not_run | pass | fail | incomplete
timing:      not_run | measured | blocked | invalid
```

Each stage records status, duration, error class, bounded detail, producer status, and sources. BoltBeam derives overall disposition; producer `passed` is never sufficient.

Correctness fields:

- scope: full output, sampled output, token parity, quality;
- element/sample count;
- tolerances and max/mean absolute/relative error;
- correlation, zero fraction, NaN and Inf counts;
- seeds/determinism and reference identity;
- explicit missing metrics.

For full GEMM, require element count, tolerances, max error, finite-output checks, and status unless experiment policy is stricter.

Resources/structure:

- VGPR, SGPR, LDS/shared bytes, scratch, occupancy;
- grid and workgroup dimensions;
- tile dimensions and wave/warp count;
- global/shared loads and stores;
- barriers, waits, tensor operations, branches, predicates;
- loops, pipeline stages, buffering;
- operand staging and ownership;
- provider extras retained with provenance.

Missing counts remain absent, never zero.

Timing:

- kernel/role/whole scope;
- raw samples with declared units, normalized to milliseconds;
- warmups, repetitions, run order, synchronization;
- compile/setup/copy inclusion;
- recomputed median, min, spread/interval;
- clock/telemetry references and same-session relationship.

Health:

- preflight/postflight;
- timeout containment;
- device availability/recovery;
- sources.

Health can invalidate execution but cannot prove correctness or speed.

### 7.2 KernelAnalysis

One report:

```text
schema
analysis_id
question
candidates
eligibility
comparability
contrasts
hypotheses
conclusions
unknowns
recommendation
evidence_refs
```

Every conclusion records truth status, confidence, support/contradiction fact IDs, and scope.

### 7.3 KernelRecommendation

Provider-neutral fields:

- hypothesis/question;
- candidate/comparator;
- exactly one intended variable, or an explicit named factorial;
- invariants;
- required artifacts and requirements;
- workload points and seeds;
- expected result under each hypothesis;
- timeout/health/correctness policies;
- success, refutation, stop, and reopen conditions;
- optional runner hints, never execution authority.

## 8. Schema-driven ingestion

`adapters.py` exposes:

```python
register_kernel_adapter(schema, adapter)
detect_kernel_artifact(mapping)
adapt_kernel_artifact(mapping, source)
ingest_kernel_artifacts(paths)
```

Rules:

- explicit schema first; structural detection only for documented legacy formats;
- duplicate registration is an error;
- unsupported schema is structured unsupported;
- malformed known schema is structured invalid with fields;
- paths are provenance, not identity;
- contradictory fragments are retained and block affected decisions.

Initial adapters:

1. `prefill-single-buffer-execution-authority.v1` and compatible two-buffer shape.
2. `prefill-single-buffer-kernel-timing.v1`.
3. Current direct-L2 execution/correctness schemas found during fixture inventory.
4. Current process-isolated timeout/fault result.
5. `tinygrad.kernel_resource_trace.v1`.
6. `tinygrad.amd_isa_proof_manifest.v1`.
7. `tinygrad.mmq_*` through compatibility projection.
8. Native `boltbeam.kernel_evidence.v1`.
9. Legacy `prefill-nonlds-search` as diagnostic-only and insufficient for comparison.

Inspect actual schemas before mapping; never infer fields from filenames.

Fragment join priority:

1. experiment ID + candidate ID + binary hash;
2. candidate hash + binary hash + system snapshot;
3. candidate ID + exact workload + explicit session;
4. otherwise do not auto-join.

Never choose by newest timestamp. Any identity mismatch is retained as a blocker.

## 9. Eligibility authority

`eligibility.py` is the sole authority and returns requirement results for:

- diagnosis eligibility;
- correctness-comparison eligibility;
- performance-comparison eligibility;
- downstream promotion-evidence eligibility.

Diagnosis may accept a compile/runtime failure when candidate, workload, stage, and source are known. Identity corruption is not evidence about the kernel.

Correctness requires:

- compile and execution pass;
- executed binary matches compiled binary where available;
- exact workload/reference;
- required numerical metrics;
- no invalidating health failure;
- route/fallback match when relevant.

Performance requires correctness plus:

- measured timing samples;
- known timing scope and inclusion semantics;
- candidate/comparator exact workload;
- compatible system, clock policy, protocol, and session;
- explicit comparator identity;
- no timeout, fault, or postflight health failure.

Raw LDS2 timeout is diagnosis-eligible but never performance-eligible. Shipping LDS remains independently valid.

Promotion eligibility only reports whether evidence could satisfy downstream policy; this package never promotes.

## 10. Contrast engine

For each pair emit:

- eligibility and blockers;
- identity/workload equality matrix;
- stage outcome differences;
- correctness differences;
- resource absolute/percentage deltas;
- structure and topology deltas;
- timing medians, spread/interval, delta, speedup, noise qualification;
- intended mechanism present/absent/unknown;
- used and missing facts.

Statistics:

- recompute median from samples;
- min/max and median absolute deviation;
- deterministic interval using existing repository policy where possible;
- speedup and signed percent delta;
- conservative noise floor from both candidates.

Summary-only timing is diagnostic unless policy explicitly permits it.

Structural language is semantic: LDS increased, occupancy decreased, barriers increased, global loads decreased. Contrast does not claim causality.

Group multi-context evidence by exact workload. A win cannot hide a protected regression.

## 11. Diagnostic reasoning

`diagnosis.py` provides a typed rule registry. Rules consume normalized facts, never candidate/provider/target names.

Initial classes:

- admission: unsupported, invalid contract, compile timeout, identity failure;
- liveness: timeout with sync, unknown timeout, runtime fault, health failure, no-result harness failure;
- numerical: NaN/Inf, mostly zero, low correlation, localized tile/row failure, deterministic wrong, nondeterministic, reference/ABI mismatch;
- performance: resource/occupancy, scratch, synchronization, memory residency, staging cost/reuse, overlap, launch overhead, tail/geometry, noise.

Hypothesis fields:

```text
hypothesis_id
category
statement
status: supported | plausible | contradicted | unknown
truth_status
confidence: high | medium | low | unknown
supporting_fact_ids
contradicting_fact_ids
missing_evidence
scope
```

Confidence:

- high: direct measured evidence isolates mechanism;
- medium: multiple measured/derived facts agree and alternatives are bounded;
- low: structural correlation only;
- unknown: evidence cannot distinguish alternatives.

### LDS/cache plugin

Extend existing `lds_l2` inputs/results with optional:

- per-operand tile/traffic bytes and reuse;
- LDS capacity/allocation;
- occupancy factor;
- barrier/wait cost range;
- buffering overlap assumption;
- cache bandwidth range;
- predicted interval for both strategies;
- weakest truth status and assumptions.

Prediction remains prediction. Measured A/B overrides predicted winner while preserving prediction error for calibration.

The plugin and candidate generator must support the initial placement matrix without changing the harness:

```text
A register + B register
A LDS      + B LDS
A register + B LDS
A LDS      + B register
```

It may later add `cache_streamed` or `reloaded` per operand. Cache-tier observations remain a separate evidence axis.

## 12. Recommendation engine

Priority:

1. Repair experiment/harness identity.
2. Establish execution.
3. Establish correctness.
4. Establish comparability.
5. Test highest-confidence, highest-impact unresolved mechanism.
6. Expand workload only after anchor validity.

Recommendations state intended variable, fixed invariants, candidate/comparator, workload/input/seed, evidence, and expected outcomes. If staging and geometry cannot vary independently, request an explicit two-factor experiment.

Expected current attn_qo behavior:

- corrected direct-L2: correctness-valid when current evidence is supplied;
- raw LDS2: timeout, diagnosis-eligible, quarantined from timing;
- shipping LDS: independently valid comparator with same-session evidence;
- no raw-LDS2/direct-L2 speed conclusion;
- performance request: corrected direct-L2 versus shipping LDS under identical conditions;
- raw-LDS2 request: bounded barrier/wait/K-exit contrast, separate from performance search.

## 13. Report and CLI

Primary command:

```bash
boltbeam investigate-kernels PATH [PATH ...] \
  --question "is direct L2 faster than LDS for attn_qo?" \
  --out-dir runs/attn-qo-analysis
```

Outputs:

```text
kernel_evidence.json
kernel_analysis.json
kernel_analysis.md
kernel_recommendation.json
```

Markdown leads with conclusion, eligibility, confidence, unknowns, and next experiment.

Later read-only views consume `kernel_analysis.json`:

```bash
boltbeam compare-kernels ANALYSIS
boltbeam explain-kernel ANALYSIS --candidate ID
boltbeam next-experiment ANALYSIS
```

Exit codes:

- 0: analysis completed, including blocked/inconclusive;
- 2: known but invalid evidence;
- 3: no supported evidence;
- 4: internal invariant failure.

A refutation or inconclusive result is successful analysis.

## 14. Durable memory

Use existing ledger abstractions:

- timeout refutes only exact candidate/binary/workload execution;
- numerical failure refutes exact correctness policy;
- performance regression refutes exact pair/workload/protocol;
- prediction disagreement calibrates the model;
- every refutation has a reopen condition;
- changed binary/compiler/workload/mechanism may satisfy it explicitly.

Analysis never writes promotion directly.

## 15. Work cards for Codex Spark

Every agent must read `docs/coding-principles.md` and this scope completely. Every card starts with a duplication audit, stays analysis-only, runs focused and full tests, and reports reuse and deviations.

### KA-0 — Contract inventory and fixtures

Files: `tests/fixtures/kernel_analysis/`, fixture inventory test.

Tasks:

- inventory actual schema values/fields from execution, timing, resource, ISA, timeout, MMQ;
- add minimal sanitized success, wrong-output, timeout, identity-mismatch, insufficient-timing fixtures;
- manifest expected diagnostic/correctness/performance eligibility.

Acceptance:

- no absolute-path semantics;
- no invented claims;
- source provenance documented.

### KA-1 — Canonical model

Files: package `__init__.py`, `model.py`, `vocab.py`, model tests.

Tasks:

- implement section 7 types and round trips;
- canonicalize hashes, dimensions, units, statuses;
- represent transport independently per operand and derive any kernel-wide strategy summary;
- content-derived IDs exclude storage paths;
- reject non-finite/impossible values.

Acceptance:

- stable round trip and IDs after relocation;
- unknown distinct from zero;
- no tinygrad import.

### KA-2 — Adapter registry

Depends: KA-1.

Files: `adapters.py`, adapter tests.

Tasks:

- registry, detection, source fingerprint, native adapter, unsupported/invalid results, fragment join.

Acceptance:

- no filename dependence;
- mismatch cannot merge;
- contradictions remain blockers.

### KA-3A — Prefill execution/timing adapters

Depends: KA-0/1/2.

Tasks:

- adapt shipping-LDS execution and timing;
- map correctness, route/fallback, identity, workload, resources, samples, protocol, telemetry;
- recompute producer summaries.

Acceptance:

- shipping-LDS fixture becomes evaluable;
- binary mismatch blocks;
- missing session/system remains missing.

### KA-3B — Direct-L2/failure adapters

Depends: KA-0/1/2.

Tasks:

- adapt direct-L2 evidence;
- adapt timeout, runtime fault, no-result, health;
- timeout cannot carry measured timing.

Acceptance:

- corrected direct-L2 correctness-evaluable;
- mostly-zero fixture fails numerical gate;
- raw LDS2 timeout diagnosis-only.

### KA-3C — Resource/ISA/MMQ/legacy adapters

Depends: KA-0/1/2.

Tasks:

- reuse timing/resource validators;
- normalize ISA semantics;
- project MMQ without weakening its gates;
- mark legacy timing insufficient;
- route transfer contract through generic validation compatibly.

Acceptance:

- existing MMQ tests pass;
- no backend/shape constants in generic model;
- legacy timing cannot become performance-eligible.

### KA-4 — Eligibility authority

Depends: KA-1.

Files: `eligibility.py`, tests.

Acceptance matrix:

- compile fail: diagnosis yes, correctness/performance no;
- timeout: diagnosis yes, correctness/performance no;
- numerical fail: diagnosis yes, performance no;
- measured identity mismatch: performance no;
- valid same-session pair: performance yes;
- unknown never passes.

### KA-5 — Contrast engine

Depends: KA-4.

Files: `contrast.py`, tests.

Tasks: section 10 statistics, deltas, facts, grouping.

Acceptance:

- fixture medians recompute;
- noisy tie is not a win;
- semantic resource/structure deltas appear;
- invalid pair has no fabricated speedup;
- direct-L2/raw-LDS2 has no performance contrast.

### KA-6A — Core diagnosis rules

Depends: KA-4/5.

Files: `diagnosis.py`, tests.

Acceptance:

- timeout plus barriers is plausible sync/liveness, not proven divergent barrier;
- mostly-zero output produces bounded write/address hypotheses;
- scratch plus measured regression supports resource pressure;
- contradiction lowers confidence.

### KA-6B — LDS/cache plugin

Depends: KA-1/5; parallel with KA-6A.

Tasks: uncertainty, occupancy, synchronization, overlap, calibration.

Acceptance:

- existing reuse/capacity guards remain;
- all four register/LDS A/B combinations are representable without a new harness schema;
- physical cache residency is reported as evidence, not inferred from a `direct_l2` candidate label;
- unknown inputs block or produce ranges;
- prediction never becomes measured verdict;
- measured disagreement is calibration evidence.

### KA-7 — Recommendation

Depends: KA-4/6.

Files: `recommend.py`, tests.

Acceptance:

- missing correctness precedes timing;
- timeout requests liveness evidence, not timing;
- valid direct-L2/shipping-LDS gap requests same-session A/B;
- identity mismatch requests harness repair;
- no unnecessary recommendation when answered.

### KA-8 — Orchestration/report/CLI

Depends: KA-2/4/5/6/7.

Files: `report.py`, CLI, README, tests.

Acceptance:

- one command handles mixed fragments;
- blocked comparison exits 0 and leads with blocker;
- unsupported-only exits 3; invalid known evidence exits 2;
- Markdown and JSON agree;
- deterministic output.

### KA-9 — Compatibility and ledger

Depends: KA-8.

Tasks:

- remove duplicated generic checks only after migration;
- preserve compatibility APIs;
- add scoped conclusions and reopen conditions;
- prohibit direct promotion.

Acceptance:

- all old tests pass;
- MMQ/prefill retain stricter gates;
- no circular imports.

### KA-10 — Golden validation

Depends: all.

Required cases:

1. correct and faster;
2. correct but slower;
3. mostly-zero output;
4. NaN/Inf;
5. compile failure;
6. timeout;
7. runtime fault;
8. identity mismatch;
9. noisy timing;
10. resource/scratch regression;
11. prediction confirmed;
12. prediction refuted;
13. insufficient legacy evidence;
14. shipping LDS versus corrected direct-L2 when artifacts exist.

Acceptance:

- deterministic reports;
- every conclusion cites facts;
- unknowns explicit;
- recommendations distinguish hypotheses or fill required evidence;
- at least two logical provider formats use the same authority.

## 16. Spark execution graph

Use low-reasoning Codex Spark agents only after shared contracts are fixed. Do not concurrently edit `model.py`, `adapters.py`, or `vocab.py`.

```text
Tranche 0, sequential:
  KA-0 -> KA-1 -> KA-2

Tranche 1, coordinated parallel:
  KA-3A
  KA-3B
  KA-3C
  KA-4

Tranche 2:
  KA-5

Tranche 3, parallel:
  KA-6A
  KA-6B

Tranche 4, sequential:
  KA-7 -> KA-8 -> KA-9 -> KA-10
```

Agent prompt:

```text
Read docs/coding-principles.md and the provider-neutral kernel analysis completion scope completely.
Implement only card KA-X. Reuse the named authorities. Do not execute GPU workloads, import runtime internals into the
analysis core, weaken fail-closed checks, or edit unrelated files. Run focused tests and the full BoltBeam suite.
Report files changed, reuse decisions, tests, blockers, and scope deviations. Do not commit unless the coordinator
assigns integration-branch ownership.
```

The coordinator resolves registry conflicts, runs the full suite after each tranche, and commits coherent passing tranches.

## 17. Testing

Unit:

- model invariants and round trips;
- adapter mappings and malformed inputs;
- identity joins;
- eligibility matrices;
- statistics/uncertainty;
- diagnosis support/contradiction;
- recommendation priority.

Contract:

- native provider-neutral evidence;
- current tinygrad adapters;
- MMQ and transfer compatibility;
- CLI exits.

Repository:

- `pytest -q` passes;
- no name-based generic decision branches;
- package imports without tinygrad installed;
- no GPU test;
- hardware artifacts are externally captured, sanitized, immutable fixtures.

## 18. Migration

1. Land model/native adapter without changing old APIs.
2. Add adapters and eligibility alongside existing systems.
3. Route new CLI through generic analysis.
4. Project MMQ and transfer results through generic evidence while retaining extra gates.
5. Compare old/new fixture outputs.
6. Remove duplicates only after equivalence tests.
7. Deprecate filename-based requirements without upgrading incomplete old bundles.

No big-bang rewrite.

## 19. Milestones

### 75% — Usable evidence analyst

KA-0 through KA-4: current artifacts ingest, stage disposition and eligibility are automatic.

### 85% — Kernel comparison tool

KA-5: valid candidates receive structural/numerical/timing contrasts; invalid pairs get blockers.

### 93% — Actionable advisor

KA-6 and KA-7: bounded hypotheses, confidence, unknowns, and next experiments.

### 100% — End-to-end analyst

KA-8 through KA-10: one command ingests, judges, contrasts, diagnoses, recommends, and reports across providers without manual translation or GPU execution inside BoltBeam.

## 20. Final definition of done

All must be true:

1. BoltBeam imports without tinygrad installed.
2. Providers emit canonical evidence or use registered adapters.
3. Joins are identity based, not filename based.
4. Compile, execution, correctness, timing, and health failures are distinct.
5. Unknown, zero, unsupported, invalid, and mismatch are distinct.
6. Correctness gates timing; timing gates downstream promotion evidence.
7. Timeout is analyzable but never benchmark-eligible.
8. A valid shipping implementation remains comparable when a sibling oracle fails.
9. Contrasts are provider-neutral and uncertainty-aware.
10. Diagnoses cite supporting, contradicting, and missing evidence.
11. Recommendations state one variable, invariants, and expected outcomes.
12. LDS/cache predictions yield to measured A/B evidence.
13. Existing MMQ, evaluator, ledger, timing, resource, and policy tests pass.
14. Golden cases cover success, failure, contradiction, and unknown.
15. `boltbeam investigate-kernels` answers conclusion, confidence, evidence, unknowns, and next experiment in one run.

Expected conceptual output:

```text
direct_l2: valid, correct, performance-eligible
raw_lds2: execution timeout, diagnosis-eligible, quarantined
shipping_lds: valid, correct, performance-eligible

valid comparison: direct_l2 vs shipping_lds
measured result: winner/delta/interval or named missing evidence
likely mechanism: evidence-bounded hypotheses
confidence: truth status plus calibrated confidence
next experiment: one bounded provider-neutral request
```

The external provider owns compilation and hardware execution. BoltBeam owns the complete analytical answer.
