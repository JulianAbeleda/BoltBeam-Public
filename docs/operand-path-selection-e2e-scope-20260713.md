# End-to-End Operand Path Selection Scope

Date: 2026-07-13

Status: implementation-ready delta scope

Owners: BoltBeam analysis authority; provider repositories own compilation and execution

Primary proving target: AMD gfx1100 / RX 7900 XTX

Primary proving workloads: current Qwen3 8B prefill and decode routes

## 1. Outcome

BoltBeam must determine, for each semantic kernel operand, whether reuse is implemented by:

- `register_resident`;
- `lds_staged`;
- `cache_streamed`; or
- `reloaded`.

It must then generate or request the feasible alternatives, correctness-gate them, compare them under one controlled protocol, and recommend the measured winner for a specific role, shape, dtype, layout, target, and runtime context.

The final answer must separate three different facts:

1. **Transport strategy**: where the kernel intentionally retains or stages an operand.
2. **Serving tier**: where each compulsory or repeated global fetch was physically served from.
3. **Selection result**: which correct candidate was fastest under a comparable measurement.

These are related but not interchangeable. A register-resident operand still has an initial global load. An LDS-staged operand can be filled from L2, MALL, or VRAM. A cache-streamed operand can hit L0/L1/L2/MALL or miss to VRAM. Candidate names such as `direct_l2` are not evidence for any of those outcomes.

## 2. Relationship to existing scopes

This is a focused completion delta over:

- `provider-neutral-kernel-analysis-completion-scope-20260713.md`;
- `lds-vs-l2-analysis-scope-20260712.md`;
- the profiler ontology and practical roofline scopes.

Those scopes establish evidence ingestion, register/LDS prediction, memory-tier calibration, and provider-neutral reporting. This scope joins those pieces into one per-operand path-selection loop. It does not create another evaluator, profiler, runner protocol, or candidate identity system.

## 3. Definition of 100%

There are two completion levels because hardware observability is not under BoltBeam's control.

### 3.1 Product completion: 100%

The product is complete when all of the following are true:

1. Every analyzed operand has one exclusive strategy classification or `unknown`.
2. Every classification cites identity-bound static and/or dynamic evidence and lists missing evidence.
3. Register, LDS, cache, reload, spill, and serving-tier concepts have provider-neutral names.
4. Provider-specific ISA and counter names are confined to adapters and target descriptors.
5. Feasible strategy alternatives can be requested without model-, role-, or target-specific branches in the analysis core.
6. An external runner can compile and execute each requested candidate under process isolation, guard buffers, device preflight/postflight, and a hard timeout.
7. Correctness is checked over the full required output before timing is accepted.
8. Compiled and executed binary identity, ABI, workload, inputs, system, and session are joined.
9. Timing uses randomized same-session A/B order, declared warmups/rounds, stable clocks, and an uncertainty/noise rule.
10. The selector can return `promote`, `retain`, `inconclusive`, `unsupported`, or `blocked` without inventing evidence.
11. A measured winner can be selected even when cache PMCs are unavailable; mechanism attribution remains explicitly incomplete.
12. The current 8B prefill and decode routes produce complete reports and preserve their existing correctness and throughput authorities.
13. Adding a provider requires an adapter and runner implementation, not changes to the classifier or decision rules.
14. All boundaries have fail-closed tests, golden reports, and schema round trips.

### 3.2 Observability completion: 100%

Full observability additionally requires target/tool support for identity-bound, per-kernel cache hit/miss and byte evidence at every physically distinguishable tier. On gfx1100 today, most PMCs are recorded as `blocked_gfx11_pmc`; only `SQ_BUSY_CYCLES` is known working in the target registry. Therefore:

- product completion is achievable without fabricating counters;
- full gfx1100 tier attribution remains `unsupported` or `unknown` until counters become live or a calibrated equivalent exists;
- a working-set sweep can measure effective bandwidth/capacity transitions, but cannot be relabeled as a candidate's L0/L1/L2/MALL hit rate;
- measured candidate A/B timing is sufficient to choose a path, but not to claim exactly which cache supplied its bytes.

No unavailable hardware feature may hold the selector hostage. It must reduce confidence and mechanism specificity, not replace a valid measured performance result with a fake one.

## 4. Current state and gap

| Capability | Current state | Gap to end state |
|---|---|---|
| Candidate operand declaration | `CandidateModel.operand_transports` has register/LDS/cache/MALL/reload booleans | Booleans permit ambiguous combinations and are declarations, not classifications |
| Four register/LDS placements | `kernel_analysis.lds_cache.PLACEMENT_MATRIX` | No generalized eligibility/generation authority or cache/reload diagnostic variants |
| Static allocations | `ResourceSummary` carries VGPR, SGPR, LDS, scratch, occupancy, geometry | Not connected to per-operand classification in the profiler report |
| Static instruction structure | `StructureSummary` and AMD ISA adapter count global/DS/barrier/WMMA rows | Counts are kernel-wide; waits, spills, cache policy, and per-operand ownership are incomplete |
| Final-ISA identity | ISA manifest can join by candidate/binary | Needs source-to-final-ISA attribution and explicit unknown on ambiguous ownership |
| Memory hierarchy | target registry models LDS/L2/MALL/DRAM; mem sweep derives capacity/bandwidth bands | L0/L1 are not calibrated; cache fit is not observed candidate residency |
| Dynamic bytes/counters | profiler ontology and narrow calibrated MMQ GL2 request proxies exist | No generic per-kernel, per-operand tier-byte evidence; gfx11 PMCs mostly blocked |
| Prediction | `perf.mem_tier`, `lds_l2`, and cache-aware roofline model costs | Prediction is not a selector and must be calibrated against measured A/B |
| Reconciliation | `kernel_analysis.lds_cache.reconcile` keeps measured and predicted outcomes distinct | Winner representation and per-operand mechanism need generalization |
| External execution | tinygrad has typed transport plans, guarded execution, health checks, and isolation | No generic operand-path request/result contract spanning candidate matrix to report |
| Recommendation | kernel analysis recommends the next bounded experiment | It does not yet generate the feasible transport experiment set or promote a measured per-context winner |
| End-to-end reporting | profiler reports allocations and warns dynamic residency needs measurement | No per-operand static/dynamic/classification/selection table |

Directional readiness at scope creation:

- evidence and safety foundations: about 70%;
- static whole-kernel analysis: about 65%;
- per-operand attribution/classification: about 25%;
- dynamic hierarchy calibration: about 35%;
- generic candidate search/execution loop: about 30%;
- complete product path: about 40%.

The percentages are implementation estimates, not evidence quality scores.

## 5. Storage and memory taxonomy

The provider-neutral hierarchy for data operands is:

```text
architected registers
  -> spill/scratch/private memory when register allocation overflows
shared software-managed storage (LDS/shared memory)
  -> vector/scalar near caches where exposed (L0/GL0)
  -> workgroup/CU cache (L1/GL1)
  -> device cache (L2/GL2C)
  -> last-level cache (AMD MALL / Infinity Cache; provider-neutral LLC)
  -> device memory (VRAM/HBM/GDDR)
  -> host/system memory only when migration or zero-copy is in scope
```

Instruction, scalar-constant, texture, and metadata caches are separate provider capabilities. They are included only when they serve an analyzed data operand. “L3” is not the canonical name because vendors expose different organizations; use `last_level_cache`, with AMD aliases `mall` and `Infinity Cache` retained as target metadata.

### 5.1 Exclusive strategy vocabulary

Each operand receives exactly one:

```text
register_resident
lds_staged
cache_streamed
reloaded
unknown
```

Definitions:

- `register_resident`: after the planned compulsory fetch, the operand fragment required for reuse remains in registers across its reuse window.
- `lds_staged`: global data is explicitly written to shared memory and read from shared memory for reuse or exchange.
- `cache_streamed`: the kernel intentionally consumes global loads without an explicit register-retention or shared-memory staging window beyond the immediate computation.
- `reloaded`: the same semantic operand region is fetched globally more times than the declared algorithmic fetch groups require because it was not retained in registers or LDS.
- `unknown`: available evidence cannot distinguish the strategies.

The classification describes the dominant reuse strategy for the declared operand scope. A kernel may classify A and B differently.

### 5.2 Orthogonal serving-tier vocabulary

Serving tiers are independent measurements or bounded estimates:

```text
register
scratch
lds
l0
l1
l2
last_level_cache
dram
host
unknown
```

Each tier observation carries bytes, requests, hits, misses, hit rate, evidence status, unit, scope, uncertainty, and source when available. Not every provider can distinguish every tier. Unsupported tiers remain present as typed unsupported facts when the question asks for them.

## 6. Central data model

Extend `boltbeam.kernel_analysis.model`; do not introduce a parallel public evidence envelope.

### 6.1 Operand strategy declaration

Migrate `OperandTransport` compatibly toward:

```text
declared_strategy: register_resident | lds_staged | cache_streamed | reloaded | unknown
requirements: semantic candidate requirements
fallback_eligible: bool
```

Existing booleans remain readable during migration. Serialization rejects contradictory true flags after callers and fixtures migrate. `mall_resident` moves out of strategy because MALL is a serving tier, not an operand placement strategy.

### 6.2 OperandPathEvidence

Add a nested immutable object under `KernelEvidence`, keyed by semantic operand identity:

```text
operand_id
semantic_role
scope
dtype
logical_shape
layout
declared_strategy
static
dynamic
classification
provenance
blockers
```

`static` fields:

- attributed global load/store sites and vector widths;
- attributed DS/shared load/store sites and widths;
- compulsory fetch groups and expected semantic bytes;
- reload sites or estimated reload factor;
- fragment bytes retained per lane/wave/workgroup;
- LDS allocation and operand-owned LDS bytes;
- accumulator and temporary register use when distinguishable;
- spill/scratch bytes and load/store sites;
- barriers, waits, loops, pipeline stages, and buffering associated with the operand path;
- cache-control/coherency policy encoded by the final program;
- source-to-final-instruction attribution status.

`dynamic` fields:

- requested, transferred, and useful bytes;
- request/transaction counts and calibrated transaction width;
- per-tier bytes/hits/misses/hit-rate observations;
- samples, interval, repetitions, and multiplexing status;
- source counter and calibration identity;
- whether evidence is per-operand, per-kernel, or a controlled proxy.

`classification` fields:

- exclusive strategy;
- truth status: measured, derived, modeled, assumed, imported, or unknown;
- confidence: high, medium, low, or unknown;
- supporting evidence IDs;
- contradicting evidence IDs;
- missing discriminators;
- ruleset version.

Do not create `operand_path_evidence.v1` as a second top-level authority unless external transport requires it. Prefer a versioned nested object in `boltbeam.kernel_evidence.v1`; if a runner fragment needs a schema, it normalizes into this same object.

### 6.3 Identity invariants

Operand evidence is invalid for a production decision unless it joins to:

- experiment and candidate digest;
- semantic schedule digest;
- source and compiled binary hashes;
- executed binary hash for dynamic evidence;
- ABI and operand-role map;
- target and system snapshot;
- workload shape/dtypes/layout;
- session for timing and counters;
- counter calibration identity when a proxy is interpreted.

Whole-kernel counters cannot be attributed to A or B merely because their semantic sizes are known. Attribution requires an isolated operand experiment, address-space/counter discrimination, or a validated algebraic decomposition with uncertainty.

## 7. Static evidence pipeline

### 7.1 Resource normalization

Keep `ResourceSummary` as the sole allocation summary. Add only missing target-neutral fields such as spill counts if they cannot live in `extra`. Preserve the distinction between:

- allocated VGPR/SGPR count;
- estimated operand fragment register bytes;
- scratch allocation;
- occupancy and active waves;
- total LDS allocation;
- operand-owned LDS regions.

VGPR count alone never proves register residency. Nonzero LDS allocation alone never proves both operands are LDS-staged.

### 7.2 ISA normalization

Extend `adapt_isa_manifest` and the tinygrad producer manifest to recognize at least:

- global/buffer loads and stores;
- scalar loads where they carry an operand;
- DS/shared reads and writes;
- scratch/private reads and writes;
- barriers and wait instructions;
- tensor/matrix operations;
- branches, predicates, loops, exits;
- cache-control/coherency modifiers;
- instruction address or stable logical operation ID;
- operand ownership and semantic fetch group.

The current adapter counts known row kinds. It must retain unknown provider rows in `extra`, never silently drop a row needed to establish transport.

### 7.3 Per-operand attribution

The provider emits an ABI/ownership map from semantic buffer to final load operations. Preferred evidence order:

1. final-ISA row carries a semantic operand ID preserved from compiler IR;
2. relocation/argument and address-expression analysis maps a final load to an ABI argument;
3. a controlled isolated variant makes only one operand's traffic variable;
4. attribution remains unknown.

Source-level intent without final-ISA correspondence is not sufficient to classify the executed binary. Final ISA without resolvable address ownership is valid whole-kernel structure but not per-operand structure.

### 7.4 Static consistency checks

Fail or block classification on:

- claimed register residence with attributed DS traffic for the same reuse window;
- claimed LDS staging without global-to-DS and DS-to-consumer evidence;
- claimed zero spills when resource metadata or ISA shows scratch;
- claimed one fetch group when loop/control-flow evidence implies repeated global fetches;
- source and binary ownership maps that disagree;
- final program geometry/resource metadata inconsistent with the dispatched executable.

## 8. Dynamic evidence pipeline

### 8.1 Target capability resolution

`targets.json` remains the authority for tier aliases, capacities, modeled bandwidths, counter availability, and provider mappings. Extend it to declare for each tier:

- whether capacity is known;
- whether effective bandwidth is measured, modeled, or unknown;
- available hit/miss/request/byte counters;
- counter scope and known non-equivalences;
- whether simultaneous collection is possible;
- multiplexing and replay requirements;
- unsupported reason.

AMD maps `gl0/gl1/gl2c/mall/dram` into `l0/l1/l2/last_level_cache/dram`. NVIDIA and Apple map their own names without changing the normalized classifier.

### 8.2 Hierarchy calibration sweep

Finish the existing request/ingest path for controlled working sets spanning approximately 4 KiB through at least twice VRAM-resident LLC capacity, with:

- pointer-chase or bandwidth modes selected explicitly;
- warm and cold variants;
- read, write, and mixed traffic where supported;
- enough repetitions to establish stable bands;
- an LDS-resident control;
- isolated execution and GPU health evidence;
- system/clock/compiler identity.

The sweep may derive capacity-transition and effective-bandwidth ranges for L0/L1/L2/LLC/DRAM when separable. Adjacent tiers that cannot be distinguished are reported as a combined range. It may not report candidate-specific hit rates.

### 8.3 Candidate counter collection

For each live provider counter group:

- collect baseline/empty-dispatch overhead;
- collect repeated isolated candidate samples;
- reject multiplexed or wrapped results unless corrected by a documented method;
- bind samples to the executed binary and session;
- retain raw counters and normalized facts;
- convert requests to bytes only through a target- and experiment-specific calibrated mapping;
- retain useful bytes, transferred bytes, and amplification separately.

The existing MMQ GL2 request proxy is reusable design evidence, not a generic conversion rule. It remains scoped until another workload passes its own transfer calibration.

### 8.4 Per-operand dynamic attribution

Use one of:

- address-filtered counters with proven buffer ranges;
- hardware trace with resolved memory addresses;
- controlled A-fixed/B-varied and A-varied/B-fixed differential experiments;
- candidate pairs that change exactly one operand transport while geometry and semantics remain fixed;
- a bounded algebraic estimate whose assumptions and interval are explicit.

Aggregate traffic may support a kernel-level mechanism but cannot be split proportionally by operand size and labeled measured.

### 8.5 Unsupported gfx1100 counters

The gfx1100 runner must attempt only capability-declared counters. A blocked counter emits a typed result such as:

```text
status: unsupported
reason: blocked_gfx11_pmc
counter: GL2C_HIT
```

It must not retry indefinitely, crash the experiment, supply zero, or prevent correctness/timing collection. Counter discovery is a bounded preflight, not part of every timing round.

## 9. Classification authority

Create one pure classifier in `boltbeam.kernel_analysis`, consuming normalized `KernelEvidence` only.

### 9.1 Rule precedence

1. Identity corruption -> `unknown` and invalid for selection.
2. Scratch/spill evidence affecting the operand -> classify the intended strategy separately, but surface a spill override and block a strict register-resident claim.
3. Attributed global-to-DS plus DS-to-consumer reuse -> `lds_staged`.
4. Proven retained fragment lifetime with no operand DS path and no repeated fetch -> `register_resident`.
5. Repeated global fetch beyond declared compulsory groups -> `reloaded`.
6. Global consumption with no explicit staging/retention proof and no proven excess reload -> `cache_streamed`.
7. Insufficient or contradictory evidence -> `unknown`.

Dynamic evidence can strengthen or contradict static classification, but physical L2/MALL hits do not turn `cache_streamed` into a new strategy. Serving-tier evidence is rendered next to the strategy.

### 9.2 Minimum proof rules

High-confidence classification requires:

- exact candidate/binary/ABI identity;
- per-operand final-program attribution;
- complete relevant control-flow scope;
- resource consistency;
- no unresolved contradictory evidence.

Dynamic counters are not mandatory for a high-confidence **strategy** when final code proves explicit register or LDS transport. They are mandatory for high-confidence **serving-tier** attribution.

### 9.3 Classification tests

Include fixtures for:

- A-register/B-register;
- A-LDS/B-LDS;
- A-register/B-LDS;
- A-LDS/B-register;
- cache-streamed once;
- repeated reload;
- register intent with scratch spill;
- LDS allocation used for only one operand;
- ambiguous global loads;
- final-ISA/source ownership disagreement;
- aggregate counters without per-operand attribution;
- unavailable cache counters;
- hybrid strategy with independently observed L2/MALL serving tiers.

## 10. Candidate generation and pruning

BoltBeam requests semantic strategy variants. Provider runners decide how to compile those variants through typed transport adapters.

### 10.1 Initial matrix

For two reusable operands A and B, always consider the existing four explicit-placement combinations when feasible:

```text
A register / B register
A LDS      / B LDS
A register / B LDS
A LDS      / B register
```

Add `cache_streamed` and `reloaded` variants only when they represent a distinct legal schedule or a diagnostic control. `reloaded` is normally a negative control or unavoidable fallback, not a desired optimization.

### 10.2 Feasibility pruning

Before asking a provider to compile, reject or range candidates using:

- operand fragment bytes and register budget;
- predicted VGPR/SGPR pressure and occupancy floor;
- LDS bytes per workgroup and target allocation granularity;
- required barriers/waits and target synchronization support;
- matrix-fragment layout legality;
- vector alignment and ABI constraints;
- workgroup/wave geometry;
- expected reuse window and tile lifetime;
- spill risk;
- provider/compiler capability declarations.

Modeled feasibility may prevent an impossible compile request. It may not choose a winner.

### 10.3 Search factorization

Keep these axes independent where the provider permits:

- A transport;
- B transport;
- tile M/N/K;
- workgroup/wave geometry;
- pipeline stages/buffering;
- vector width;
- cache policy;
- accumulator layout.

When transport necessarily changes geometry or tile layout, encode an explicit factorial experiment. Do not describe the measured gap as transport-only. The recommendation layer already recognizes staging/geometry entanglement and should consume the generated factor declaration.

### 10.4 Prediction ordering

The practical roofline and memory-tier model may rank feasible requests using intervals that include:

- compulsory and repeated global bytes;
- expected L0/L1/L2/LLC/DRAM bandwidth band;
- LDS bytes and synchronization cost;
- register pressure, occupancy, and spill risk;
- matrix utilization and instruction count;
- overlap/pipeline assumptions.

Prediction narrows search order. Every promoted route still requires measured correctness and same-session performance evidence.

## 11. External execution protocol

BoltBeam remains analysis-only. It emits a serializable runner plan. tinygrad or another provider compiles and runs it.

### 11.1 Request contract

Extend `runner_plan` and the provider execution bridge with:

- workload and semantic schedule identity;
- semantic operand/ABI map;
- declared strategy per operand;
- target and compiler context;
- candidate knobs and fixed invariants;
- required static artifacts;
- requested counter groups, each optional when unsupported;
- correctness oracle and tolerances;
- guard/timeout/health policy;
- timing protocol and randomization seed;
- comparator and experiment identity.

No environment variable or candidate label may silently select a transport. The typed `TransportPlan` and explicit registry remain the provider-side authority.

### 11.2 Provider compile result

Return:

- source/schedule/compile/binary identity;
- final ABI and geometry;
- resource metadata;
- final ISA manifest with ownership rows;
- capability/unsupported results;
- executable metadata or a typed compile failure.

### 11.3 Guarded execution result

Reuse tinygrad's `guarded_execution` lifecycle:

- device health preflight;
- guarded allocations;
- nonconstant inputs;
- immutable input checks;
- full-output finite and numerical comparison;
- hard process timeout around dispatch;
- device health postflight and recovery evidence;
- release.

No timing sample is accepted from a candidate that has not passed the declared correctness scope with the same binary and ABI.

### 11.4 Timing result

For each candidate pair or matrix:

- use one session/system snapshot;
- pin or record clocks and power state;
- randomize candidate order per round;
- declare warmups and measured rounds;
- synchronize consistently;
- exclude compilation and allocation unless the experiment asks for end-to-end latency;
- retain raw samples;
- report median, spread, confidence/bootstrapped interval, and noise threshold;
- repeat sessions when the winner interval overlaps.

### 11.5 Counter result

Counter collection is a separate pass when replay perturbs timing. The result references the same binary and workload but does not masquerade as the timing session. Any perturbation, multiplexing, replay, or clock difference is explicit.

## 12. Decision and route promotion

The selector evaluates each workload context independently. A recommendation key includes at least:

```text
provider + target + compiler/version + role + shape/range + dtype/layout + context band
```

### 12.1 Required promotion evidence

- candidate and comparator compile and execute;
- both pass required correctness;
- identity and ABI chains are complete;
- binaries are distinct when the experiment claims distinct mechanisms;
- timing is same-session and comparison-grade;
- the win exceeds the declared noise/uncertainty threshold;
- no health fault, guard corruption, spill disqualifier, or unsupported semantic requirement;
- fallback and rollback route exist.

Live cache counters are not required to promote a measured faster path unless the policy specifically claims cache mechanism as the reason for promotion.

### 12.2 Outcomes

- `promote`: candidate is correct and measurably faster.
- `retain`: comparator is correct and candidate is slower or invalid.
- `inconclusive`: both are valid but the interval/noise does not establish a winner.
- `unsupported`: provider/target cannot materialize the strategy.
- `blocked`: required identity, correctness, execution, or comparability evidence is missing.

Mechanism confidence is reported separately from selection confidence. “WMMA-LDS won by 2.8x” can be a high-confidence measured selection while “because MALL hit rate fell” remains unknown.

### 12.3 Policy handoff

BoltBeam emits a recommendation artifact. The provider repository owns the route-policy mutation. Promotion must be scoped, reversible, and rejected automatically if the compiler, binary, target, workload contract, or correctness authority changes.

## 13. Report contract

Add an operand-path section to both machine-readable and Markdown reports.

Per operand, render:

| Field | Meaning |
|---|---|
| declared | requested strategy |
| classified | strategy proven by normalized evidence |
| confidence/status | strength and truth status |
| static | fragment bytes, LDS bytes, global/DS/scratch sites, barriers/waits, fetch groups |
| dynamic | useful/transferred bytes and tier observations with status |
| serving tiers | measured/proxy/modeled/unknown distribution |
| contradictions | evidence against the classification |
| missing | next discriminator required |

Then render:

- candidate feasibility and rejection reasons;
- prediction intervals and assumptions;
- correctness/health results;
- comparable timing table;
- measured winner or typed non-decision;
- prediction calibration;
- one bounded next experiment when unresolved.

The current profiler warning that allocations do not establish cache residency remains until dynamic evidence exists.

## 14. Implementation phases

### P0 — Freeze vocabulary and invariants

Deliverables:

- exclusive strategy and orthogonal serving-tier enums;
- compatibility migration for current boolean transports;
- operand evidence dataclasses and round-trip validation;
- blocker/error vocabulary;
- decision keys and evidence requirements.

Acceptance:

- contradictory strategies fail closed;
- MALL/Infinity Cache cannot be serialized as a transport strategy;
- unknown, unsupported, zero, and absent remain distinct;
- old fixtures load and canonical new fixtures round-trip.

### P1 — Complete whole-kernel static evidence

Deliverables:

- ISA kinds for waits, branches, loops, scratch, cache policy, and instruction totals;
- resource/spill fields;
- lossless provider extras;
- profiler static rendering.

Acceptance:

- known raw ISA rows reconcile with normalized totals;
- missing instruction classes remain absent rather than zero;
- resource and ISA binary identities must join.

### P2 — Add per-operand final-program attribution

Deliverables:

- semantic ABI/ownership map contract;
- tinygrad final-ISA ownership emission;
- adapter normalization into operand static evidence;
- ambiguity and contradiction blockers.

Acceptance:

- all four A/B register/LDS fixtures classify structurally;
- mixed placement identifies the correct operand;
- ambiguous loads remain unknown;
- attribution follows the final executed binary, not a candidate name.

### P3 — Complete hierarchy calibration

Deliverables:

- external mem-sweep runner request and result ingestion;
- L0/L1/L2/LLC/DRAM effective bands where separable;
- warm/cold and LDS controls;
- uncertainty and unsupported-tier reporting;
- target registry updates from measured facts without overwriting hardware facts.

Acceptance:

- synthetic tier transitions classify deterministically;
- real gfx1100 result records measured bands and blocked hit-rate counters honestly;
- sweep evidence is never rendered as candidate hit rate.

### P4 — Add candidate dynamic evidence

Deliverables:

- generic counter fragment adapter;
- target mapping through counter registry;
- baseline/replay/multiplex validation;
- per-operand differential/proxy attribution;
- profiler dynamic rendering.

Acceptance:

- live supported counters normalize to measured facts;
- calibrated proxies remain proxy/modeled with bounded scope;
- unavailable gfx11 counters become unsupported without blocking timing;
- aggregate counters cannot silently become per-operand measurements.

### P5 — Implement classifier

Deliverables:

- one pure, versioned rules engine;
- classification confidence and contradiction output;
- static/dynamic reconciliation;
- unit and golden tests.

Acceptance:

- complete fixture matrix passes;
- declaration/evidence disagreement is visible;
- register-with-spill cannot pass strict register residency;
- final strategy is exclusive and serving tiers remain orthogonal.

### P6 — Implement candidate matrix and feasibility pruning

Deliverables:

- generalized placement matrix;
- target/provider capability pruning;
- register/LDS/occupancy estimates;
- factor declarations for entangled geometry;
- prediction ordering with intervals.

Acceptance:

- no model or role hardcoding in the core;
- impossible variants produce typed rejection reasons;
- prediction cannot emit a promotion;
- same candidate spec is portable to another provider adapter.

### P7 — Close provider runner loop

Deliverables:

- runner-plan request/result schemas;
- tinygrad compile/result adapter using typed `TransportPlan`;
- guarded correctness and process isolation integration;
- randomized A/B timing and optional separate counter pass;
- identity-complete result ingestion.

Acceptance:

- a known hanging candidate times out without wedging the parent or becoming a timing result;
- wrong output fails before timing;
- compiled/executed hash mismatch blocks selection;
- unsupported counters do not block a valid A/B;
- BoltBeam itself performs no GPU dispatch.

### P8 — Implement selection and promotion handoff

Deliverables:

- measured matrix ranking;
- tie/noise handling;
- prediction calibration;
- route recommendation artifact;
- rollback/reopen conditions and ledger write.

Acceptance:

- measured correctness/performance overrides prediction;
- no selection occurs from naming, fit, static counts, or modeled roofline alone;
- route recommendation is context-scoped and reversible;
- mechanism and selection confidence are separate.

### P9 — Prove current 8B end to end

Deliverables:

- prefill operand-path report for generated GEMM roles;
- decode operand-path report for packed GEMV/attention roles;
- candidate matrix or typed infeasibility per important role;
- correctness-gated timing and selected route;
- raw and practical roofline comparison using classified fetch groups;
- before/after throughput authority and no-regression record.

Acceptance:

- generated prefill's A/B staging is proven from its final binary rather than its route label;
- decode packed weights and activation operands classify independently;
- current prefill and decode correctness remain intact;
- throughput authority remains the model benchmark, not extrapolated profiler timing;
- every missing cache observation is visible and does not become a false conclusion.

## 15. File-level work map

### BoltBeam

| File | Change |
|---|---|
| `boltbeam/kernel_analysis/model.py` | exclusive strategy, serving-tier observations, operand evidence, compatibility parser |
| `boltbeam/kernel_analysis/provider_adapters.py` | richer ISA rows, resource/spill rows, dynamic counter fragments, ownership normalization |
| `boltbeam/kernel_analysis/lds_cache.py` | generalize placement matrix and measured/predicted reconciliation |
| `boltbeam/kernel_analysis/classify.py` | new pure per-operand classifier; no provider branches |
| `boltbeam/kernel_analysis/contrast.py` | operand strategy, serving-tier, byte-amplification, and resource deltas |
| `boltbeam/kernel_analysis/diagnosis.py` | bounded transport/cache/spill hypotheses |
| `boltbeam/kernel_analysis/recommend.py` | matrix experiment and unresolved discriminator requests |
| `boltbeam/kernel_analysis/report.py` | operand tables, selection/mechanism split, blockers |
| `boltbeam/kernel_analysis/theoretical_roofline.py` | consume classified fetch groups and measured hierarchy bands when present |
| `boltbeam/perf/mem_tier.py` | interval costs across normalized tiers; no verdict |
| `boltbeam/mem_sweep.py` | complete portable request/result and tier-band handling |
| `boltbeam/mem_hierarchy.py` | ingest measured/unsupported tier facts without conflation |
| `boltbeam/profiler/ontology.py` | normalized strategy, serving-tier, amplification concepts |
| `boltbeam/profiler/counters.py` | tier counter semantics and scoped conversion metadata |
| `boltbeam/profiler/report.py` | join static/dynamic/classification into current trace reports |
| `boltbeam/runner_plan.py` | operand-path candidate matrix request and provider result requirements |
| `boltbeam/data/targets.json` | per-tier capability/counter/replay/unsupported metadata |
| `boltbeam/data/counter_registry.json` | normalized tier bytes/hits/misses/request mappings and non-equivalences |
| `boltbeam/vocab.py` | only top-level schema constants that are actually required |
| `schemas/profiler_report.schema.json` | operand-path report fields and truth statuses |
| `tests/kernel_analysis/` | model, adapter, classifier, contrast, recommendation, E2E golden tests |
| `tests/fixtures/kernel_analysis/` | strategy matrix, ambiguity, spill, cache, identity, and unsupported-counter artifacts |

### tinygrad provider repository

| File | Change |
|---|---|
| `tinygrad/runtime/execution_bridge_contracts.py` | compatible operand plan/result fields and typed unsupported outcomes |
| `extra/qk/prefill/transport_execution_authority.py` | register additional explicit transport adapters; no fallback inference |
| `extra/qk/prefill/guarded_execution.py` | reuse unchanged unless result contract needs provenance fields |
| provider compile/capture path | emit final-ISA rows with semantic operand ownership and fetch groups |
| provider counter collector | bounded capability preflight and raw identity-bound samples |
| provider mem-sweep worker | isolated sweep execution and health evidence |
| provider candidate worker | compile, correctness, timing, counter result envelope |

The exact tinygrad producer filename should follow the existing compile/capture owner. Do not resurrect retired hand-kernel experiment stacks or add a second execution authority merely to satisfy this scope.

## 16. Test and verification matrix

### Unit

- enum and compatibility parsing;
- invalid strategy combinations;
- tier aliases and unsupported states;
- ISA row counting and ownership;
- static consistency rules;
- classifier precedence;
- dynamic attribution scope;
- prediction intervals;
- decision/noise rules.

### Contract

- BoltBeam request -> tinygrad parser;
- tinygrad result -> BoltBeam adapter;
- hash/ABI/system/session mismatch rejection;
- unknown provider fields preserved;
- provider-specific names do not leak into core decisions.

### Safety

- compile timeout;
- dispatch timeout;
- child termination;
- preflight failure;
- postflight failure;
- guard corruption;
- input mutation;
- NaN/Inf output;
- full-output mismatch;
- unsupported counters.

### Statistical

- randomized order preserved;
- raw samples retained;
- median and interval recomputed;
- ties remain inconclusive;
- clock/session mismatch rejected;
- counter replay not mixed with timing.

### End to end

- known correct register route;
- known correct LDS route;
- known wrong-output route;
- known hanging raw LDS route;
- mixed A/B route;
- current generated prefill default;
- current decode default;
- provider adapter fixture from a non-AMD target.

## 17. Expected implementation size

Estimated gross change:

- core models, adapters, classifier, contrast, report: 700-1,050 LOC;
- hierarchy/counter/calibration integration: 350-550 LOC;
- runner contracts and tinygrad provider wiring: 400-650 LOC;
- tests and fixtures: 850-1,250 LOC;
- schemas/docs: 200-350 LOC.

Expected total: approximately 2,500-3,850 gross LOC across both repositories. Compatibility cleanup after migration should remove roughly 250-500 LOC of duplicate boolean/route-specific logic, for a net addition around 2,000-3,600 LOC.

This estimate assumes existing guarded execution, evidence joins, profiler ontology, counter registry, placement matrix, roofline model, and runner planning are reused. Reimplementing those authorities would increase both LOC and risk and is out of scope.

## 18. Delivery order and parallel work

Safe parallel tracks after P0 freezes contracts:

- Track A: static ISA/resource ownership and adapters (P1-P2).
- Track B: hierarchy calibration and dynamic counter ingestion (P3-P4).
- Track C: classifier, contrast, and reports against fixtures (P5).
- Track D: candidate feasibility and prediction ordering (P6).
- Track E: provider runner integration after request/result contracts stabilize (P7).

P8 consumes A-E. P9 is the integration proof. No agent should independently invent schema fields; P0 model types and fixtures are the merge authority.

Recommended milestone cuts:

1. **Static truth**: P0-P2; can classify explicit register/LDS paths without cache PMCs.
2. **Dynamic truth**: P3-P5; serving tiers are measured where supported and unknown where not.
3. **Closed loop**: P6-P8; request, execute externally, ingest, classify, and select.
4. **Production proof**: P9; current 8B routes and regressions covered.

## 19. Explicit anti-patterns

The implementation is unsound if it does any of the following:

- infers register, LDS, L2, or MALL use from a candidate/route name;
- treats nonzero LDS allocation as proof that every operand is LDS-staged;
- treats VGPR allocation as proof of retained operand lifetime;
- treats cache capacity fit as a measured hit;
- treats working-set sweep bandwidth as a candidate's cache hit rate;
- splits aggregate bytes between operands without a declared attribution method;
- calls MALL/Infinity Cache a transport strategy;
- treats absent or unsupported counters as zero;
- chooses a route from a roofline prediction without measured A/B;
- times a candidate before correctness and health pass;
- compares different binaries, ABIs, sessions, clocks, or workload semantics;
- attributes a staging win when geometry changed without a factorial control;
- lets BoltBeam import a provider runtime or launch GPU work;
- creates model-, role-, target-, or candidate-name conditionals in the core classifier;
- revives failed raw ISA experiments as hidden fallbacks.

## 20. Final acceptance checklist

The product-completion definition is met only when this checklist is all true:

- [ ] Exclusive per-operand strategy model and compatibility migration landed.
- [ ] Orthogonal normalized serving tiers landed.
- [ ] Static resources and final ISA join by exact binary identity.
- [ ] Final loads/DS/scratch operations attribute to semantic operands or stay unknown.
- [ ] Static classifier passes the full strategy matrix and ambiguity fixtures.
- [ ] Hierarchy sweep request/result works through an external isolated runner.
- [ ] Candidate counters normalize where supported and fail typed where unsupported.
- [ ] Per-operand dynamic attribution never exceeds its evidence scope.
- [ ] Profiler renders static, dynamic, classification, confidence, and missing evidence.
- [ ] Candidate matrix and feasibility pruning are provider/model agnostic.
- [ ] Prediction orders experiments but cannot select or promote.
- [ ] External runner uses typed transport plans, guards, health checks, and timeouts.
- [ ] Full correctness precedes timing for the exact executed binary.
- [ ] Same-session randomized timing produces measured matrix rankings or inconclusive.
- [ ] Selection confidence and mechanism confidence are separate.
- [ ] Recommendation handoff is scoped, reversible, and ledgered.
- [ ] Current Qwen3 8B prefill and decode complete end to end without regression.
- [ ] A second-provider fixture proves core provider neutrality.
- [ ] Full test suite and golden reports pass.

When these boxes are complete, BoltBeam can answer “which operand path should we use?” end to end. On targets with blocked PMCs, it will still choose from correct measured candidates while saying honestly that exact cache-tier attribution is unavailable.
