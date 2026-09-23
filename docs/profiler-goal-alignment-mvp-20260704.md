# Profiler Goal, Alignment, and MVP

Date: 2026-07-04

Status: execution alignment. This document defines the shared done condition for the GPU-agnostic profiler build.

Related docs:

- `docs/gpu-agnostic-profiler-build-scope-20260704.md`
- `docs/native-pmc-sampler-and-graph-pmc-20260704.md`

## High-Level Goal

Build BoltBeam into a GPU-agnostic kernel profiler that can answer why a model is fast or slow on a given GPU, with enough evidence to resolve the current 8B vs 14B and tinygrad vs llama questions.

The product goal is not "collect counters." The product goal is:

> Given a hot model run, identify which kernels dominate time, what hardware resource or compiler/runtime behavior limits them, whether the slowdown is proportional work or lost efficiency, and what evidence proves or disproves the claim.

For the current 14B problem, the minimum useful answer is:

> The hot 14B prefill GEMM is slower because of one or more named causes: more FLOPs/bytes, lower percent-of-peak compute, worse memory behavior, lower occupancy, worse tile/kernel selection, graph/launch overhead, or compiler/codegen pathology. If none of those differ materially from 8B or llama, BoltBeam must say the boring answer: the model is doing proportionally more work.

## Vendor Profiler Reference Rule

We should port the logic architecture of NVIDIA Nsight Compute and ROCm Compute Profiler into BoltBeam, not depend on the tools as the product.

In this context, "port" means:

- learn from their public workflow architecture;
- mirror the useful analysis categories in BoltBeam's own schema and report engine;
- ingest their exported reports as calibration/reference artifacts;
- use their metric names and formulas only through an explicit counter registry with source, unit, target, and quality metadata;
- build our own provider-neutral collection plans, derived metrics, reports, and classifiers.

It does not mean:

- copying vendor source code, UI, report text, or private definitions;
- requiring `ncu`, `rocprof-compute`, `rocprofv3`, CUPTI, or ROCProfiler for BoltBeam's normal tinygrad path;
- pretending NVIDIA and AMD counters are equivalent when the underlying hardware blocks differ;
- wrapping vendor tools and calling that a GPU-agnostic profiler.

The reference architecture to preserve is:

```text
collection plan
  -> dispatch selection and replay/pass strategy
  -> raw counters and launch/resource facts
  -> target capability and peak lookup
  -> normalized metric taxonomy
  -> report sections
  -> baseline comparison
  -> causal rules and next measurements
```

BoltBeam owns that architecture.

## Shared Mental Model

A kernel is one dispatch of compiled device code, with a concrete launch shape, resource footprint, input/output shape, and measured hardware behavior.

A kernel is fast when its work maps cleanly onto the GPU's preferred execution substrate:

- enough parallel work to fill the device;
- tile shape matches matrix/vector primitives;
- memory movement is coalesced and reused;
- tensor/MFMA/MMA pipelines are used when the workload is matrix-compute-bound;
- occupancy is high enough to hide latency without excessive register, LDS, or scratch pressure;
- graph, launch, synchronization, barriers, and wait states do not dominate;
- generated code does not spend most cycles on scalar bookkeeping, unpack, spills, or uncoalesced loads;
- achieved throughput reaches a defensible fraction of the target peak for the limiting unit.

BoltBeam must report which of these is supported by evidence, which is refuted, and which is still unknown.

## Workstream Alignment

### Runtime/Producer Workstream

Primary owner: Claude or whoever is working inside tinygrad/runtime code.

This workstream makes the GPU emit trustworthy raw evidence.

Responsibilities:

- tinygrad native-PMC sampling through tinygrad's own AMD/KFD path;
- graph-captured per-kernel PMC collection;
- stable attribution from runtime program ids to executed graph kernels;
- eager and graph microbenchmarks to isolate PMC behavior;
- raw `ProfilePMCEvent` correctness;
- no fabricated zeros when a counter was not collected;
- no BoltBeam schema churn unless the producer/consumer contract requires it.

Current state:

- `tinygrad-native-pmc` path is reported shipped and verified in the working tree.
- graph-PMC mechanism is built but partial.
- graph path currently captures per-kernel `SQ_BUSY_CYCLES`, but not the full counter set needed for occupancy, VALU, memory, and MFMA conclusions.
- `PMC_GRAPH=0` should remain the default until multi-counter graph capture works.

### BoltBeam/Consumer Workstream

Primary owner: Codex or whoever is working inside BoltBeam's profiler package.

This workstream turns raw evidence into normalized, comparable, user-facing answers.

Responsibilities:

- stable schemas: `timing_trace`, `hw_trace`, `profiler_report`;
- target registry with peak compute, memory, architecture, and counter support;
- counter registry with raw and derived metrics, formulas, units, source names, and quality labels;
- vendor importers for Nsight Compute, ROCm Compute Profiler, and rocprofv3 outputs;
- dispatch identity normalization and joins;
- role/shape/quant attribution for model kernels;
- percent-of-peak, roofline, memory, occupancy, source/ISA, graph/launch, and baseline comparison sections;
- causal classifier that distinguishes "more work" from "lost efficiency";
- markdown and JSON reports that state missing evidence explicitly.

Current state:

- committed foundation exists for schemas, target peak metadata, counter registry, capability registry, and an initial Nsight Compute importer skeleton.
- the larger report engine and causal classifier are not MVP-complete yet.

## Interface Contract Between Workstreams

The producer workstream emits raw trace evidence. BoltBeam consumes and normalizes it.

Minimum row contract for `boltbeam.hw_trace.v1`:

- provider id and sampler id;
- target id and architecture;
- stable dispatch/program identity;
- kernel name and optional demangled/canonical name;
- start/end/duration or call count and aggregate duration;
- grid/workgroup/queue/graph identity where available;
- resource facts: VGPR, SGPR, LDS/shared memory, scratch/spills, registers, occupancy hints when available;
- role/shape/quant attribution when known;
- raw counters with provider names, canonical names, units, pass id, and quality;
- derived metrics only when formulas have all required inputs;
- missing counters represented as missing/unsupported, never as zero.

The most important rule:

> A row with missing counters is valid. A row with fake zeros is corrupt.

## MVP Done Definition

The MVP is done when BoltBeam can produce an honest profiler report for the real 14B prefill problem and can prove or disprove the kernel-fast theory against at least one baseline.

Required MVP artifacts:

1. Trace collection

   - tinygrad AMD path collects a valid `boltbeam.hw_trace.v1` without requiring `rocprofv3`.
   - graph-captured hot GEMMs have per-kernel counters, not only graph-level or eager-only counters.
   - llama or reference provider path can produce a comparable trace for the same workload class.
   - trace files validate against the schema.

2. Counter coverage for the hot GEMM

   The 14B hot prefill GEMM row, currently the `ffn_gate_up Q4_K [512,17408,5120]` class of kernel, must have enough evidence to derive:

   - wall time and percent of step;
   - launch/grid/workgroup shape;
   - model role, quant, and matrix shape;
   - active/busy cycles or equivalent;
   - achieved compute throughput;
   - percent of relevant compute peak;
   - memory bytes or enough cache/memory counters to estimate effective bandwidth;
   - percent of relevant memory peak when supported;
   - occupancy or residency evidence;
   - vector/scalar/tensor instruction evidence, including MFMA/MMA/tensor-core presence or absence when supported;
   - L2/cache hit/miss or equivalent memory locality signal when supported;
   - LDS/shared-memory signal when supported.

3. Vendor-reference import

   - Nsight Compute exported data can be imported into BoltBeam's canonical trace shape for at least one fixture.
   - ROCm Compute Profiler or rocprof-compute output can be imported into BoltBeam's canonical trace shape for at least one fixture.
   - These importers exist for reference, calibration, and cross-vendor comparison; they are not required for the normal tinygrad native path.

4. Report engine

   `boltbeam profiler-report` must emit JSON and markdown with:

   - top kernels by time;
   - per-hot-kernel speed-of-light summary;
   - roofline or arithmetic-intensity placement when inputs are available;
   - memory workload summary;
   - occupancy/resource summary;
   - source/ISA/static evidence when available;
   - baseline comparison against 8B or llama;
   - explicit evidence quality and missing-field notes;
   - final classification and next measurement.

5. 8B vs 14B decision

   The report must support one of these conclusions:

   - 14B is proportionally slower because it does proportionally more work at similar efficiency.
   - 14B loses efficiency, and BoltBeam names the limiting cause with measured evidence.
   - The trace is insufficient, and BoltBeam names the exact missing counters or artifacts needed.

6. Tests and reproducibility

   - schema tests pass;
   - counter/capability registry tests pass;
   - importer fixture tests pass;
   - report fixture tests pass;
   - one documented local command sequence regenerates the MVP report from trace inputs.

The MVP is not done if the hot 14B GEMM row only has timing and names. It must contain enough hardware evidence to avoid guessing.

## Non-MVP Scope

These are useful later but must not block the first done condition:

- full Nsight Compute UI parity;
- full ROCm Compute Profiler UI parity;
- every NVIDIA, AMD, and Apple counter;
- live CUPTI/ROCProfiler process injection for all providers;
- PC sampling and per-source-line heatmaps;
- automatic kernel rewriting or tuning;
- every model architecture;
- a polished GUI.

## Phase Plan

### P0: Align and Freeze the Contract

Done when:

- this doc exists;
- the umbrella scope remains the product scope;
- the native-PMC graph note remains the current runtime status;
- all workstreams use `hw_trace` as the handoff artifact.

### P1: Commit and Harden Native-PMC

Done when:

- tinygrad default sampler is native-PMC where intended;
- `rocprofv3` remains optional/reference for tinygrad;
- pre/post GPU health checks still pass;
- relative output path and `PROFILE=1` issues are covered by tests;
- the working-tree native-PMC changes are committed separately from this alignment doc.

### P2: Fix Graph-PMC Counter Completeness

Done when:

- `PMC_GRAPH=1` captures more than `SQ_BUSY_CYCLES` for graph-dispatched kernels;
- `GRBM_GUI_ACTIVE` or equivalent active-cycle denominator is nonzero in graph path;
- VALU, MFMA/tensor, L2/cache, LDS, and memory signals are available where supported;
- eager and graph microbenchmarks explain any counter differences;
- `PMC_GRAPH` can be enabled for the trace run without corrupting attribution or adding unacceptable overhead.

### P3: Build the Vendor-Reference Architecture

Done when:

- BoltBeam has collection-plan/report-section concepts inspired by Nsight Compute and ROCm Compute Profiler;
- `ncu` fixture import maps to canonical counters and report sections;
- ROCm Compute Profiler or rocprof-compute fixture import maps to canonical counters and report sections;
- imported metrics carry source names, units, formulas, and quality labels.

### P4: Build the MVP Report Engine

Done when:

- `profiler-report` consumes `hw_trace`;
- report output is deterministic JSON plus readable markdown;
- report sections match the MVP requirements;
- missing evidence is explicit;
- no report claims percent-of-peak, occupancy, bandwidth, or tensor utilization without required inputs.

### P5: Close the 8B/14B Case

Done when:

- real 8B and 14B traces are collected under comparable conditions;
- real tinygrad and llama/reference traces are collected for the 14B case;
- BoltBeam produces one report that proves, disproves, or blocks the theory with named missing evidence;
- the conclusion is backed by measured counters, target peak data, dispatch identity, and baseline comparison.

## Immediate Critical Path

The next blocking item is graph-PMC counter completeness.

Until graph-dispatched hot GEMMs emit active-cycle, compute, memory, and occupancy-relevant counters, BoltBeam can build schemas and reports, but it cannot honestly classify the 14B hot GEMM. The report must say "insufficient evidence" rather than infer from timing alone.

Recommended next producer task:

```text
Fix tinygrad graph-PMC so inline graph pmc_read captures the full scheduled counter set, not only SQ_BUSY_CYCLES. Prove it first with a minimal graph microbenchmark, then with the real 14B prefill trace. Keep PMC_GRAPH default OFF until the graph path emits enough counters for BoltBeam to derive occupancy, VALU/MFMA, and memory metrics.
```

Recommended next BoltBeam task:

```text
Build profiler-report against fixture traces and mark missing graph-PMC counters as blocking evidence. Do not wait for perfect counters to build the reporting contract, but do not let the report make unsupported claims.
```

## Reference Sources

These sources define the reference architecture concepts we are borrowing, not runtime dependencies:

- NVIDIA Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/
- NVIDIA Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- ROCm Compute Profiler overview: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/what-is-rocprof-compute.html
- ROCm Compute Profiler basic usage: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html
- ROCm Compute Profiler CLI analysis: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/cli.html
