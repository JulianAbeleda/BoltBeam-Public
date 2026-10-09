# MMQ Complete-Knowledge Implementation Scope

Date: 2026-07-11

## Mission

Raise the bounded gfx1100 MMQ pair from source-level causal evidence to an
auditable closed-system account of compilation, static resources, dynamic GPU
execution, and predicted performance.

Continue until the emitted program is explained from semantic contract through
measured timing, or a repeated named blocker leaves no meaningful extraction,
profiling, differential-probe, or model-falsification path.

Missing evidence is work, not a stop condition. Unsupported evidence must be
recorded truthfully and replaced by the strongest available controlled probe.

## Complete Operational Knowledge

For one system snapshot, candidate, input, and protocol, complete knowledge
accounts for:

```text
semantic contract
-> tinygrad sink and UOps
-> lowering and rendered source
-> compiler inputs and loaded code object
-> final ISA and dependency/epoch structure
-> static resources and occupancy limits
-> dynamic scheduling, memory, synchronization, and clocks
-> numeric output
-> frozen predicted cycle/time interval
-> measured cycle/time distribution
```

The model is complete for this scope only when prediction precedes measurement,
measurement agrees within declared uncertainty, and every residual maps to a
named fact, uncertainty, or falsified rule.

## Current Truth

The existing pair establishes bounded correctness and a stable writeback effect:

```text
gated median:         17.95-18.27 ms
direct median:         6.98-7.02 ms
direct improvement:    2.56-2.62x
direct vs comparator:  1.25-1.29x
```

Current real bundles lack loaded binary and compiler resource evidence. The
existing source hash is not rendered-source identity, the short emitted hash is
not binary SHA-256, ISA rows are empty, store counts are mode-derived, and
observed ownership/store fields are structural rather than final-ISA facts.

Exploratory exact-sink compilation found:

| mode | HSACO bytes | store sites | VGPR | SGPR | LDS | scratch |
|---|---:|---:|---:|---:|---:|---:|
| gated | 14,600 | 256 | 26 | 26 | 256 | 0 |
| direct owner | 6,792 | 1 | 27 | 29 | 256 | 0 |

These facts become authoritative only after program/binary identity is bound
into the experiment bundle and independently validated.

## Agents And Dependencies

```text
Agent A: exact compiler/binary/ISA/static-resource provenance
Agent B: dynamic counter liveness, PMC collection, telemetry, probes
Agent C: ISA graph, calibration, occupancy/cycle model, validation
Root: contracts, integration, GPU runs, falsification, follow-up probes
```

Schedule:

```text
Root     freeze -- review -- integrate -- calibrate -- falsify/reprobe
Agent A  parsers -- loaded binary binding -- exact bundle evidence
Agent B  capability/liveness -- eager PMC -- telemetry/probes
Agent C  pure fixtures/contracts -- exact ISA integration -- model
```

Agents A and B start in parallel. Agent C begins with pure fixture-driven code
and consumes real ISA/resources only after Agent A stabilizes their contract.

## WP-A: Exact Compiler Provenance

Tinygrad additions and narrow edits:

```text
extra/qk/mmq_compile_evidence.py
extra/qk/mmq_q4k_q8_atom.py
extra/qk/mmq_bounded_harness.py
extra/qk/mmq_experiment.py
extra/qk/mmq_resource_snapshot.py
extra/qk/mmq_epoch_manifest_export.py
test/unit/test_mmq_compile_evidence.py
focused existing tests
```

Required APIs:

```python
build_mmq_sink(spec)
compile_mmq_program(spec, device="AMD")
capture_loaded_mmq_program(spec, device="AMD")
parse_amdgpu_metadata(binary)
disassemble_amdgpu(binary)
analyze_final_isa(disassembly)
```

Program-binding procedure:

1. Construct the candidate through one canonical sink builder.
2. Lower using the actual AMD renderer.
3. Execute through the bounded harness.
4. Locate the loaded research `AMDProgram`.
5. Read the exact loaded library/code-object bytes.
6. Assert loaded bytes equal the compiled program binary.
7. Bind program key, target, symbol, launch, source, binary, and ISA.
8. Fail closed on any mismatch.

Emit and hash:

```text
kernel.uops.txt
kernel.source.hip
kernel.hsaco
kernel.isa.txt
compile_manifest.json
```

The compile manifest records system/candidate identity, function and target,
renderer/compiler/tool versions, program key, UOp/sink/linear/source/binary/ISA
hashes, binary bytes, launch dimensions, and compiler-affecting environment.

Parse actual AMDGPU metadata and descriptors for:

```text
VGPR/SGPR allocation and spills
LDS/group-segment bytes
private/scratch bytes
max workgroup, wave size, dynamic stack
target architecture and kernel symbol
```

Parse actual disassembly for instruction/code bytes, global and DS load/store
sites, barrier/waitcnt, scratch, branches/predicates, store PCs/mnemonics,
instruction order, operand/register text, and max referenced registers.

ISA store structure joins source/UOp ownership and launch geometry. Store count
alone is not semantic ownership proof.

WP-A acceptance:

```text
real 64-character loaded-binary SHA-256
reproducible rendered-source and ISA hashes
compiler-reported static resources
256/1 store facts derived from final ISA
all program/source/binary/launch identities agree
placeholder compiler evidence removed
real bundles pass compile/static-resource requirements
production dispatch unchanged
```

## WP-B: Dynamic Execution Evidence

Available live substrates:

```text
/opt/rocm-7.2.4/bin/rocprofv3
/opt/rocm-7.2.4/bin/rocprofv3-avail
rocprofiler-compute under ROCm libexec
tinygrad KFD native PMC
tinygrad SQTT
```

Advertised metrics cover waves/cycles, VALU/SALU, waits, LDS, GL2, memory
controller requests, TA load/store, and occupancy. Advertisement is not liveness.

New schemas:

```text
tinygrad.amd_counter_capability.v1
tinygrad.amd_pmc_result.v1
tinygrad.amd_sqtt_summary.v1
tinygrad.amd_telemetry_trace.v1
tinygrad.mmq_differential_probe.v1
```

Metric status is exactly one of:

```text
advertised | live | zero_suspect | unsupported | blocked
```

Required APIs:

```python
probe_amd_counter_capabilities(device)
run_pmc_liveness_suite(counter_groups)
collect_kernel_pmc(candidate, counters, repetitions)
collect_telemetry(process_or_window)
run_controlled_probe(pair)
```

A counter is live only when positive and negative controls produce the expected
direction with repeated nondegenerate values. Zero is never automatically real.
Use multiple passes when scheduling or multiplexing is ambiguous.

Minimum controls:

```text
VALU-heavy vs empty
SALU-heavy vs empty
LDS conflict vs conflict-free
global memory vs compute-only
wait/barrier-heavy vs no-wait
high vs low wave count
large vs cache-resident working set
```

Attempt to establish waves/cycles, occupancy with formula provenance, VALU/SALU,
wait activity, LDS activity/conflicts, GL2 hit/miss, MC request counts, store
proxies, and clock/power/temperature availability. Request counts do not become
physical bytes without calibrated semantics.

Fallback ladder:

```text
1. tinygrad eager per-kernel PMC
2. rocprofv3
3. rocprofiler-compute
4. bounded SQTT summary
5. controlled differential probes
```

If profiling blocks remain zero, preserve that fact and continue with
differential probes. Stop only when no probe distinguishes remaining causes.

WP-B acceptance:

```text
tool/permission/scheduling/liveness artifact exists
live and zero-suspect cannot be confused
candidate/binary/system identity enforced
timing and attempted SQ/memory/LDS/store evidence joined
telemetry failures explicit
randomized gated/direct counter passes reproducible
unavailable never becomes zero
```

## WP-C: Predictive Cycle Accounting

The objective is to predict both timings and their ratio before reading their
candidate timing samples. Roofline remains a lower bound, not this model.

Schemas and files:

```text
boltbeam.mmq_calibration.v1
boltbeam.mmq_cycle_model.v1
boltbeam.mmq_prediction.v1
boltbeam.mmq_model_validation.v1
boltbeam/perf/calibration.py
boltbeam/perf/isa_graph.py
boltbeam/perf/occupancy.py
boltbeam/perf/cycle_model.py
boltbeam/perf/validation.py
boltbeam/search/mmq_prediction.py
tests/perf/
```

Inputs include exact system/clock policy, ordered ISA/operands/registers/PCs,
instruction classes and dependencies, epochs, resources, launch geometry, store
ownership/active lanes/coalescing, sync, and calibrated parameters.

Required instruction classes:

```text
VALU integer/float | dot/MFMA | SALU | global load/store
LDS load/store | branch/predicate | barrier | waitcnt
```

Calibrate launch, dependent latency, independent throughput, issue compatibility,
global memory, stores/masks/coalescing, LDS/conflicts, barriers/waits, allocation
granularity/residency, tail efficiency, and clock distribution. Every parameter
is a Fact with uncertainty and calibration identity.

Minimum model:

```text
dependency_floor = longest weighted dependency path
issue_floor = max(demand / issue rate by domain)
memory_completion = transaction issue + outstanding latency + waits
wave_cycles = max(dependency_floor, issue_floor, memory_completion)
              + non-overlappable synchronization
kernel_cycles = launch + resident-wave/workgroup batches + tail
predicted_ms = kernel_cycles / clock distribution
```

Do not sum overlapping floors. Epochs bound legal overlap:

```text
load/decode/stage -> visibility sync -> dot/K-loop -> writeback -> epilogue
```

The store model distinguishes static sites, executed lanes, predicates, address
depth, physical transactions/proxies, coalescing, and serialization.

Calibration matrix:

```text
launch/grid/workgroup sweeps
dependent chains and independent streams
issue-compatibility pairs
load/store width, mask, stride, and working-set sweeps
1/2/4/.../256 gated store sites
LDS conflict sweeps
barrier/wait sweeps
VGPR/SGPR/LDS pressure sweeps
CU/grid-tail boundaries
combined load/LDS/barrier/dot/store holdouts
```

Use at least 30 measured samples per point, randomized/interleaved order, clock
capture, and bootstrap intervals.

Training separation:

```text
calibration: generated microbenchmarks
validation: unrelated bounded structural variants
final holdout: gated/direct pair
```

Prediction output includes IDs, cycle/time intervals, component bounds/overlap,
critical path, issue domains, occupancy/batches, store estimate/proxy,
assumptions, unsupported facts, and parameter sensitivity. Never emit a point
prediction without an interval.

Falsify when measured median misses the 95% interval, ordering is wrong, ratio
error exceeds 10%, absolute error exceeds 15%, residual tracks a missing axis,
parameters are nonphysical/unstable, identities differ, evidence is unknown,
overlap is inconsistent, or a pair-specific correction is required.

Validation is `validated`, `falsified`, or `inconclusive`. Only validated models
may guide search ordering; prediction never replaces measured promotion evidence.

WP-C acceptance:

```text
explicit ISA dependency/issue graph
reproducible occupancy bounds
unknown facts yield inconclusive
prediction cannot read candidate timing
prediction hash frozen first
interval/falsification tests pass
synthetic exact cases pass
unrelated holdouts validate or name the model gap
final pair predicted, measured, classified
```

## Integration Waves

Wave 1, parallel:

```text
A: binary/metadata/ISA parsers and exact sink binding
B: tool/counter capability and liveness artifacts
C: calibration/ISA graph/occupancy contracts with fixtures
```

Wave 2, parallel:

```text
A: authoritative compiler artifacts in real bundles
B: eager PMC and telemetry passes
C: real ISA/resource graph integration
```

Wave 3, parallel:

```text
A: compiler-bound microbenchmark artifacts
B: differential probes and calibration execution
C: cycle, uncertainty, and validation model
```

Wave 4, root sequential:

1. Capture fresh system snapshot and calibration profile.
2. Validate unrelated bounded kernels.
3. Compile gated/direct and freeze predictions.
4. Run randomized/interleaved measurements.
5. Validate/falsify the model.
6. Use residuals and counters to choose the next probe.
7. Repeat until validated or the stop audit passes.

## Stop Audit

A genuine standstill requires all of:

```text
the same unknown blocks at least three consecutive attempts
native PMC, ROCm profiling, SQTT where applicable, and controlled probes were
attempted or proved inapplicable
compiler/runtime inspection exposes no extractable remaining state
no candidate pair isolates the competing explanations
no named calibration experiment can reduce the residual
continuing would repeat an experiment without new information
```

At standstill record the exact unknown, every attempted path, why remaining
paths are inapplicable, which causes remain indistinguishable, and the smallest
external change that reopens progress.

## Cost And Non-Goals

```text
compiler provenance: 830-1,250 total LOC, 2-4 days
dynamic P0-P2:      900-1,450 total LOC, 4-7 days
dynamic probes:   1,000-1,700 total LOC, 5-10 days
predictive model: 2,200-3,180 total LOC, 7-12 days
calibration:        850-1,350 total LOC, 3-6 days
```

Until the pair is explained, defer broad package moves, universal simulation,
cross-GPU portability, model-level scheduling, black-box regressors, production
promotion from prediction, and expansion into geometry/staging/unroll search.
