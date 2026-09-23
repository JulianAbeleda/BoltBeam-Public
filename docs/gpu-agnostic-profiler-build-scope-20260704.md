# GPU-Agnostic Profiler Build Scope

Date: 2026-07-04

Status: scope. No implementation implied by this document.

## Objective

Build BoltBeam into a GPU-agnostic profiler brain with the practical diagnostic value of NVIDIA Nsight Compute and
ROCm Compute Profiler, while keeping BoltBeam's existing boundary: BoltBeam observes, normalizes, compares, and
classifies runtime evidence; provider runtimes execute models and kernels.

The product should answer:

- What kernels ran, in what order, on which queue/stream/graph, with what launch geometry?
- Which kernels dominate wall time and model throughput?
- For a hot kernel, what is the limiting hardware resource: memory bandwidth, tensor-core/MFMA throughput, scalar/vector
  ALU, occupancy, launch/graph overhead, synchronization, LDS/shared memory, or compiler/codegen pathology?
- Does a model slow down because it does proportionally more work, or because its kernels fall off the hardware fast path?
- For the 8B-vs-14B story, is 14B slower because its GEMM shapes select worse tiles/kernels, lose tensor-core utilization,
  lose occupancy, repeat dequant/unpack work, or simply perform more FLOPs/bytes?
- What evidence would prove or disprove each theory, and what should be measured next?

## Kernel-Speed Theory To Encode

A kernel is fast when its algorithmic work maps cleanly onto the hardware's preferred execution substrate:

- launch geometry creates enough independent work to fill the GPU without excessive scheduling overhead;
- tile shape matches the architecture's matrix/vector primitives;
- memory movement is coalesced, reused, and close to the minimum bytes required by the algorithm;
- tensor-core/MFMA/MMA instructions fire when the workload is matrix-compute-bound;
- vector load/store width is preserved;
- occupancy is high enough to hide latency, without register/LDS pressure causing spills or lowering residency;
- synchronization, barriers, waitcnts, and graph/launch boundaries are not dominating wall time;
- generated code does not bloat scalar/vector instructions around the real math;
- measured throughput reaches a defensible fraction of the target's peak for the limiting unit.

The profiler must not collapse these into vague "bandwidth" claims. It should distinguish:

- high memory busy with low effective GB/s because loads are scalar/uncoalesced;
- low memory busy because occupancy/latency is too low to sustain bandwidth;
- high VALU with low MFMA because unpack/dequant or compiler scalarization dominates;
- high theoretical FLOPs but low achieved FLOPs because the selected kernel variant misses tensor cores;
- proportional slowdown because the model simply has more work.

## Current Inventory

BoltBeam already has useful pieces:

| Area | Current state |
|---|---|
| Public trace contract | `boltbeam.hw_trace.v1` and `boltbeam.timing_trace.v1` exist by convention and tests. |
| Collectors | `collect-hw-trace` supports `llama` and `tinygrad`; tinygrad has backend CSV, profile-events, and native-PMC paths in the working tree. |
| Counter names | `hw_trace.py` normalizes a small counter vocabulary: occupancy, memory busy/stall, L2 hit, fetch/write KB, VALU, LDS, MFMA. |
| Resource fields | llama/tinygrad adapters can carry workgroup/grid, LDS, scratch, VGPR, SGPR when present. |
| Role attribution | tinygrad prefill kernels can be mapped to role/quant/shape from names and weight inventory. |
| 14B reports | `prefill-role-trace`, `compare-substrate`, prefill roofline, and roofline ladder localize packed-prefill gaps. |
| Compiler diagnostics | `tinygrad.compiler_pathology.v1` evidence and `diagnose compiler-pathology` classify spills, barriers, LDS/memory overhead, vector-load loss, waitcnt, loop-lowering issues. |
| GPU health | `gpu_health.py` can guard AMD trace runs against render-node loss, D-state tasks, and dead GPU FD holders. |

Known current blockers:

- No formal JSON schema exists for `timing_trace.v1` or `hw_trace.v1`.
- No profiler capability registry exists.
- No counter registry exists with units, formulas, exact/derived quality, provider mappings, or target support.
- NVIDIA and Apple targets are descriptor-only; target peak data is too thin for percent-of-peak reports.
- Current `hw_trace` counter joins are by kernel name, not stable dispatch/program id.
- Tinygrad native PMC currently misses graph-captured hot GEMM counters in the real 14B prefill path.
- Existing reports are strong for Qwen-style packed prefill, but not a general profiler report engine.

## Product Boundary

BoltBeam owns:

- schemas and stable contracts;
- provider/counter/target capability registries;
- importers for vendor profiler outputs;
- low-level sampler adapters where practical;
- dispatch identity normalization;
- counter normalization and quality metadata;
- source/ISA/static metric parsing;
- baseline comparison;
- causal classification;
- reports and next-measurement plans.

Provider runtimes own:

- model execution;
- kernel compilation;
- graph capture;
- actual kernel launches;
- raw runtime hooks where vendor APIs require process-local access;
- correctness and W==D authority gates.

Vendor tools remain useful as data sources and calibration oracles. BoltBeam should not require one vendor profiler in
the normal path for every provider, but it should ingest their outputs.

Non-goals:

- Do not clone the full Nsight Compute UI.
- Do not put model serving or GPU kernel execution into BoltBeam core.
- Do not pretend AMD and NVIDIA counters are equivalent when formulas differ.
- Do not report derived counters as measured.
- Do not silently fill missing resource/counter fields with zero.

## Target User Workflows

### Workflow A: 8B vs 14B Kernel-Fast Theory

Inputs:

- 8B and 14B traces from the same provider/runtime.
- Optional baseline traces from llama.cpp or another reference implementation.
- Target peak descriptor.

Report must show:

- top hot kernels by wall time;
- GEMM shapes and selected kernel variants;
- achieved TFLOPs and bandwidth;
- percent of relevant peak;
- tensor-core/MFMA/MMA utilization;
- occupancy and register/LDS/scratch pressure;
- memory/coalescing/vector-load evidence;
- shape/tile friendliness notes;
- conclusion: same efficiency but more work, or real kernel efficiency loss.

Disproof condition:

- If 8B and 14B reach similar percent-of-peak on dominant kernels and 14B's wall time scales with FLOPs/bytes, the
  theory that 14B is pathologically slower is refuted.

Proof condition:

- If 8B reaches a high fraction of peak and 14B reaches much less on comparable dominant kernels, then BoltBeam must
  identify the diverging dispatch geometry, kernel variant, occupancy, tensor-core utilization, memory behavior, or
  source/ISA difference.

### Workflow B: tinygrad 14B Prefill vs llama

Inputs:

- llama `hw_trace` with resources from kernel trace CSV or provider hook.
- tinygrad `hw_trace` with timing, resources, source/static facts, and counters where available.
- weight inventory for exact role/shape/bytes.

Report must explain each hot packed GEMM:

- role, quant, shape, calls, wall time, percent of step;
- effective GB/s and target/reference gap;
- workgroup/grid/local/global shape;
- VGPR/SGPR/LDS/scratch;
- generated source/binary hash and schedule params;
- static instruction mix, MFMA/WMMA presence, vector-load width, barriers, waitcnts, unpack pattern;
- dynamic counters when available;
- final classification or explicit missing fields.

### Workflow C: Vendor Profiler Import

Inputs:

- NVIDIA Nsight Compute export or report file.
- AMD ROCm Compute Profiler / rocprof-compute output.
- ROCm `rocprofv3` CSV/JSON.
- Optional RGP/SQTT artifacts.

BoltBeam output:

- normalized `hw_trace`;
- raw artifact retention/fingerprints;
- capability/metric support statement;
- profiler report with quality metadata.

## Architecture

```text
provider runtime or external profiler
  -> collector/importer
       ncu import
       rocprof-compute import
       rocprofv3 import
       tinygrad profile/PMC sampler
       llama/ggml hook
       optional HIP/CUDA interposer
  -> raw event model
       dispatches
       counters
       resources
       source/binary/static artifacts
       memory copies and allocations
  -> BoltBeam normalizer
       schema validation
       target registry lookup
       counter registry mapping
       dispatch identity join
       role/shape/quant attribution
       evidence quality labeling
  -> profiler report engine
       speed-of-light
       roofline
       memory workload
       occupancy/resource limits
       scheduler/stall/instruction mix
       source/ISA
       launch/graph overhead
       baseline diff
       causal classifier
  -> next measurement / route decision
```

Suggested package layout:

```text
boltbeam/profiler/
  __init__.py
  schema.py
  capabilities.py
  counters.py
  targets.py
  dispatch.py
  normalize.py
  report.py
  sections/
    speed_of_light.py
    roofline.py
    memory.py
    occupancy.py
    scheduler.py
    source_isa.py
    launch.py
    diff.py
    causal.py
  importers/
    ncu.py
    rocm_compute.py
    rocprofv3.py
    rgp_sqtt.py
  samplers/
    tinygrad_hcq_pmc.py
    llama_ggml_hook.py
    hip_interposer.py
    cuda_cupti.py
```

Keep compatibility shims in the existing modules until `collect-hw-trace`, `compare-hw-trace`, `compare-substrate`,
and `prefill-role-trace` can delegate to the new profiler package.

## Data Contracts

### `boltbeam.hw_trace.v1`

Formalize the current contract before extending it.

Required top-level fields:

- `schema`
- `model_id`
- `target_id`
- `workload`
- `provider_id`
- `contexts`
- `rows`

Required kernel row fields:

- `scope = "kernel"`
- `context`
- `kernel`
- `kind`
- `wall_us`

Optional kernel row fields:

- `dispatch_id`
- `program_id`
- `queue_id`
- `stream_id`
- `graph_id`
- `launch_order`
- `start_ns`
- `end_ns`
- `role`
- `quant`
- `shape`
- `calls`
- `phys_bytes`
- `resources`
- `counters`
- `counter_quality`
- `sources`
- `static_metrics`
- `evidence_quality`

Compatibility rule:

- Existing `hw_trace.v1` consumers must keep working if new fields are absent.
- Bump to `hw_trace.v2` only if row shape or semantic meaning breaks compatibility.

### `boltbeam.profiler_capabilities.v1`

Purpose: describe what a collector/importer can and cannot produce.

Fields:

- `provider_id`
- `collector_id`
- `tool`
- `tool_version`
- `target_support`
- `requires_replay`
- `requires_root_or_privilege`
- `supports_dispatch_timing`
- `supports_kernel_resources`
- `supports_dynamic_counters`
- `supports_memory_copy`
- `supports_allocations`
- `supports_source_correlation`
- `supports_disassembly`
- `supports_thread_trace`
- `supported_metrics`
- `unsupported_metrics`
- `known_blind_spots`
- `calibration_notes`

Example blind spots:

- tinygrad DEV=AMD + rocprofv3 backend CSV sees zero kernels because KFD direct submission bypasses ROCr/HIP.
- tinygrad graph-captured GEMMs currently bypass eager PMC emission.
- kernel-name-only CSV cannot safely distinguish repeated dispatches with identical names.

### `boltbeam.counter_registry.v1`

Purpose: normalize vendor metrics into portable concepts without lying about equivalence.

Each counter row:

- `counter_id`: BoltBeam name, e.g. `memory_busy_pct`.
- `unit`: `%`, `bytes`, `KB`, `cycles`, `instructions`, `ratio`.
- `concept`: memory, tensor_core, occupancy, l2, lds, scheduler, launch.
- `semantic_definition`: plain-language meaning.
- `higher_is`: `better`, `worse`, `contextual`.
- `exactness`: `measured`, `derived`, `proxy`, `estimated`.
- `formula`: optional expression using provider raw metrics.
- `provider_mappings`: per provider/tool/target.
- `required_raw_metrics`.
- `valid_range`.
- `normalization_notes`.
- `known_non_equivalences`.

Initial portable concepts:

- `elapsed_us`
- `achieved_flops`
- `achieved_tflops`
- `achieved_tflops_pct_peak`
- `dram_bytes_read`
- `dram_bytes_write`
- `dram_bw_gbs`
- `dram_bw_pct_peak`
- `l2_hit_pct`
- `occupancy_pct`
- `waves_or_warps_active`
- `tensor_core_util_pct`
- `mfma_or_mma_instruction_count`
- `valu_busy_pct`
- `scalar_alu_busy_pct`
- `lds_or_shared_conflict_pct`
- `scratch_or_local_spill_bytes`
- `barrier_count`
- `wait_or_stall_pct`
- `vector_load_bits`

### `boltbeam.target_registry.v2`

Extend the existing target registry instead of inventing a parallel file.

Required additions:

- architecture family: AMD RDNA/CDNA, NVIDIA SM, Apple Metal, etc.
- chip/generation identifiers;
- CU/SM count;
- wave/warp size;
- clock assumptions and source;
- peak memory bandwidth and source;
- peak FP16/BF16/FP32/int8/int4 throughput;
- tensor-core/MFMA/MMA instruction shapes;
- L1/L2/shared/LDS characteristics;
- register file limits;
- max blocks/waves/warps per CU/SM;
- occupancy formulas;
- counter support matrix by profiler tool;
- source/disassembly tooling support.

Do not mark a target `complete` until it can produce honest profiler reports. `descriptor_only` remains non-promotable.

### `boltbeam.profiler_report.v1`

Top-level report:

- `schema`
- `model_id`
- `target_id`
- `workload`
- `providers`
- `trace_sources`
- `quality_summary`
- `top_findings`
- `sections`
- `next_measurements`
- `raw_artifacts`

Required sections:

- `summary`
- `speed_of_light`
- `roofline`
- `memory_workload`
- `occupancy`
- `scheduler_stalls`
- `instruction_mix`
- `source_isa`
- `launch_graph`
- `baseline_diff`
- `causal_classification`
- `missing_evidence`

## Collector And Importer Scope

### NVIDIA `ncu` Import

Build first as an importer, not live CUPTI.

Inputs:

- Nsight Compute CSV/JSON/text export.
- Optional source/SASS export.

Output:

- `hw_trace` rows with kernel timing, launch geometry, occupancy, memory, tensor-core, scheduler/stall, and source/ISA
  metrics where present.

Required mapping:

- kernel name and launch identity;
- duration;
- launch/block/grid;
- registers/shared/local memory;
- achieved occupancy;
- achieved SM active;
- tensor core utilization or instruction counts;
- DRAM/L2 bytes and throughput;
- warp stall breakdown when available;
- source/SASS correlation references.

Acceptance:

- Fixture with two GEMMs imports into two kernel rows.
- Report computes percent of FP16/BF16/int8/int4 peak where target supports it.
- Missing source correlation is reported as missing, not failed.

### AMD ROCm Compute Profiler Import

Inputs:

- ROCm Compute Profiler / rocprof-compute output directory or CSV.
- Optional ROCprof Compute Viewer-compatible JSON.

Output:

- `hw_trace` rows plus counter quality metadata.

Required mapping:

- kernel timing;
- dispatch geometry;
- occupancy/waves;
- VALU/SALU/MFMA;
- LDS;
- L2/GL2;
- memory read/write;
- stall/wait counters where available.

Acceptance:

- Fixture imports to normalized counters with formulas recorded.
- Derived metrics state their raw metric dependencies.

### ROCprofv3 Import

Keep existing llama/tinygrad CSV support, but move it behind the profiler importer interface.

Add:

- stronger `kernel_trace.csv` aggregation preserving resources;
- memory copy/allocation row normalization;
- warning when CSV has no stable dispatch id;
- provider capability row stating whether counters/resources are present.

### tinygrad HCQ/PMC Sampler

Goal: BoltBeam-owned tinygrad path that does not require rocprofv3.

Required tinygrad producer facts:

- `ProfileProgramEvent` with program tag, name, binary, resource descriptor;
- `ProfileRangeEvent` / `ProfileGraphEvent` with timing;
- launch/global/local sizes keyed by program or execution tag;
- `ProfilePMCEvent` keyed to graph/eager dispatches;
- optional SQTT event for deep mode;
- source text/path and schedule metadata for hot kernels.

BoltBeam responsibilities:

- load/copy raw `profile.pkl`;
- join by program tag and execution tag, not just kernel name;
- decode AMD descriptors into resources;
- decode PMC blobs into raw and normalized counters;
- mark graph/eager bypass reasons;
- retain raw artifacts.

Acceptance:

- Synthetic fixture: one program, one dispatch, one PMC blob -> one kernel row with counters.
- 8B smoke: hot GEMMs have timing/resources/counters and agree with wall trace within tolerance.
- 14B smoke: hot packed GEMM rows either have counters or carry a precise `pmc_graph_bypass`/missing-evidence reason.

### llama/ggml Hook

Preferred over an interposer for semantic attribution.

Hook around:

- graph compute boundaries;
- backend scheduler compute boundaries;
- kernel launch helpers where available;
- tensor/node metadata before role context is lost.

Output:

- BoltBeam JSON event stream with graph node, tensor role, kernel name, launch metadata, and timing if available.

Acceptance:

- llama prefill trace can attribute `mul_mat_q` rows to exact role/shape where possible.
- If only coarse attribution is possible, report says `coarse_role_only`.

### HIP/CUDA Interposers

Optional fallback for model-agnostic dispatch capture.

HIP:

- `hipModuleLaunchKernel`
- `hipLaunchKernel`
- stream synchronization and event timing if needed

CUDA:

- CUPTI activity API preferred for stable timestamps and counters;
- launch interception only for geometry/timing fallback.

Limitations:

- weak tensor-role attribution;
- high risk of runtime perturbation;
- counters may require replay or privileged APIs.

Use as fallback after provider hooks and vendor profiler imports.

## Report Engine Scope

### Speed Of Light

For each hot kernel:

- elapsed;
- achieved FLOPs or operation estimate;
- achieved TFLOPs;
- percent of target peak for active dtype/primitive;
- achieved DRAM bandwidth;
- percent of memory peak;
- achieved tensor-core/MFMA/MMA utilization;
- limiting unit guess with confidence.

### Roofline

Generalize current roofline:

- arithmetic intensity;
- physical bytes source and quality;
- operational count source and quality;
- memory floor;
- compute floor;
- measured point;
- reclaimable time;
- whether the point is memory-bound, compute-bound, latency-bound, launch-bound, or inconclusive.

### Memory Workload

Show:

- DRAM read/write bytes;
- L2 hit/miss;
- coalescing/vector width;
- global load/store instruction mix;
- dequant/unpack bytes vs logical bytes;
- transfer/copy rows separated from timed model kernels.

### Occupancy And Resources

Show:

- registers;
- shared/LDS;
- scratch/local spill;
- workgroup/block size;
- grid size;
- active waves/warps;
- theoretical occupancy limit and cause;
- achieved occupancy if measured.

### Scheduler/Stalls

Show:

- barrier counts;
- waitcnt/fence density;
- memory dependency stalls;
- tensor pipe underutilization;
- warp/wave issue efficiency;
- graph/launch gaps.

### Instruction And Source/ISA

Show:

- static instruction counts by class;
- dynamic instruction counts where available;
- MFMA/MMA/tensor-core instruction presence;
- vector load/store width;
- scalarization or unpack ALU pattern;
- source path/hash;
- binary hash;
- disassembly path/hash;
- schedule/tile params.

### Baseline Diff

Compare:

- candidate vs baseline provider;
- 8B vs 14B;
- exact shape GEMM microbench vs full model;
- padded/tile-friendly shape vs exact shape;
- tinygrad vs llama.

Diff output:

- what changed in shape/tile/kernel variant;
- what changed in percent-of-peak;
- what changed in occupancy/resources;
- what changed in static/dynamic instruction mix;
- whether the hypothesis is proved, refuted, or still missing evidence.

### Causal Classifier

Initial classes:

- `doing_more_work`
- `shape_tile_unfriendly`
- `kernel_variant_suboptimal`
- `tensor_core_not_firing`
- `low_tensor_core_utilization`
- `memory_bandwidth_saturated`
- `memory_pipeline_gap`
- `uncoalesced_or_scalar_loads`
- `dequant_unpack_alu_bound`
- `occupancy_starved`
- `register_spill`
- `lds_shared_conflict`
- `barrier_or_waitcnt_overhead`
- `launch_or_graph_overhead`
- `source_codegen_pathology`
- `counter_coverage_blocked`
- `resource_complete_unknown`

Each class needs:

- required evidence fields;
- optional corroborating fields;
- confidence levels;
- refutation conditions;
- next measurement.

## CLI Scope

Add or extend:

```bash
boltbeam profiler-report --trace hw_trace.json --out report.json   # capability/coverage info folds into this report; no separate capabilities-query command shipped
boltbeam import-hw-trace report.csv --provider tinygrad --out hw_trace.json   # ncu import
boltbeam import-hw-trace out_dir --provider tinygrad --out hw_trace.json   # rocm-compute import
boltbeam collect-hw-trace --provider tinygrad --sampler tinygrad-hcq-pmc ...
boltbeam profiler-report --trace hw_trace.json --out report.json --markdown report.md
boltbeam compare-hw-trace --baseline base_hw_trace.json --candidate cand_hw_trace.json --out diff.json
boltbeam explain-kernel --trace hw_trace.json --kernel-id <dispatch_id-or-kernel> --markdown kernel.md
boltbeam compare-substrate --baseline 8b.json --candidate 14b.json --out test.json   # closest shipped equivalent; no separate hypothesis-comparison command shipped
```

Keep current commands:

- `compare-hw-trace`
- `compare-substrate`
- `prefill-role-trace`
- `prefill-roofline`
- `diagnose compiler-pathology`

but gradually make them wrappers over the profiler report engine.

## Implementation Phases

### Phase P0: Contracts And Registries

Deliverables:

- JSON schema for `timing_trace.v1`.
- JSON schema for `hw_trace.v1`.
- `profiler_capabilities.v1`.
- `counter_registry.v1`.
- target registry extension plan.
- validation helpers.

Acceptance:

- Existing fixtures validate.
- Missing optional fields remain legal.
- Counter rows can carry exact/derived/proxy quality.
- No module invents counter names outside the registry.

### Phase P1: Stabilize Current AMD/tinygrad Evidence

Deliverables:

- move existing profile-event/native-PMC paths under one profiler abstraction;
- preserve `profile.pkl` and native PMC JSON as raw artifacts;
- add dispatch/program ids where available;
- formalize `pmc_graph_bypass` as evidence quality, not a role-report-only detail.

Acceptance:

- Current profiler-related tests pass.
- 14B hot GEMM report explicitly says timing/bytes present and counters/resources missing or blocked.

### Phase P2: Vendor Profiler Importers

Deliverables:

- `ncu` importer with fixtures.
- ROCm Compute Profiler importer with fixtures.
- upgraded rocprofv3 importer.

Acceptance:

- Each importer produces `hw_trace.v1`.
- Each importer emits capability and metric-quality metadata.
- Cross-vendor report sections work when equivalent concepts exist and mark missing/non-equivalent metrics otherwise.

### Phase P3: Target Peak And Percent-Of-Peak

Deliverables:

- target peak descriptor for local `amd_gfx1100`.
- at least one NVIDIA target with complete profiler capability data.
- peak selection by dtype/primitive.
- percent-of-peak calculations.

Acceptance:

- Report can say whether a GEMM is at 70 percent of relevant compute peak or not.
- Report can distinguish memory peak from tensor peak.

### Phase P4: Source/ISA And Static Diagnostics

Deliverables:

- source/binary/disassembly artifact retention.
- static parser for AMD ISA and NVIDIA SASS/PTX where feasible.
- static metric rows attached to `hw_trace`.
- bridge from `compiler_pathology` into profiler causal classification.

Acceptance:

- Static-only classification can detect missing tensor-core instructions, vector-load loss, instruction bloat, barriers,
  waitcnts, and unpack ALU dominance.

### Phase P5: tinygrad Graph-Safe Counters

Deliverables:

- tinygrad producer change or artifact contract for PMC reads in graph submit path.
- BoltBeam decoder for `ProfilePMCEvent`.
- normalized counters with formulas.

Acceptance:

- Hot graph-captured GEMMs have counters, or the trace contains a precise reason why that is impossible on the run.

### Phase P6: llama/ggml Semantic Hook

Deliverables:

- llama/ggml hook contract.
- importer for hook output.
- exact or quality-tagged role attribution.

Acceptance:

- llama quantized matmul rows can be matched against BoltBeam roles better than kernel-name-only coarse buckets.

### Phase P7: Profiler Report Engine

Deliverables:

- `profiler-report`.
- `compare-hw-trace`.
- `explain-kernel`.
- markdown and JSON outputs.
- causal classifier.

Acceptance:

- One command produces an actionable report for 14B prefill with timing, resources, static diagnostics, counters when
  available, final classification, and missing-evidence list.

### Phase P8: 8B/14B Theory Test

Deliverables:

- `compare-substrate` or equivalent report mode (no separate hypothesis-comparison command shipped).
- shape padding/microbench comparison hook.
- 8B-vs-14B baseline comparison fixtures.

Acceptance:

- Report can prove or refute: "14B is slow because its shapes select worse kernels / lower percent-of-peak."
- If false, report says the slowdown is proportional work.

### Phase P9: Hardening

Deliverables:

- artifact cache integration;
- deterministic fingerprints;
- versioned schemas;
- fixture corpus;
- CI test matrix for importer fixtures;
- docs and examples;
- failure-mode reports.

Acceptance:

- New importer additions require fixtures and registry entries.
- All reports state what was measured, derived, estimated, or missing.

## Test Plan

Unit tests:

- schema validation for all new contracts;
- counter registry lookup and formula validation;
- target peak lookup;
- importer parsers for `ncu`, ROCm Compute Profiler, rocprofv3;
- tinygrad profile/PMC event joins;
- source/ISA static parsers;
- causal classifier rules.

Golden fixtures:

- one NVIDIA `ncu` report with two GEMMs and one elementwise kernel;
- one AMD ROCm Compute Profiler report with MFMA/memory/L2 counters;
- one rocprofv3 kernel trace with resources;
- one tinygrad profile event stream with program/range/launch/resource rows;
- one tinygrad PMC event stream with raw blobs and expected normalized counters;
- one graph-bypass fixture;
- one source/ISA fixture with known vector-load/MFMA/barrier/waitcnt counts;
- one 8B/14B comparison fixture.

Integration tests:

- `collect-hw-trace` writes timing and hw traces;
- `profiler-report` consumes traces from every importer;
- `compare-hw-trace` handles missing counters without crashing;
- `compare-substrate`/`diagnose compiler-pathology` emit proved/refuted/inconclusive with required evidence (no separate hypothesis-comparison command shipped).

Real-run validation:

- local AMD gfx1100 tinygrad 8B prefill smoke;
- local AMD gfx1100 tinygrad 14B pp512 smoke;
- llama 14B pp512 baseline import;
- optional NVIDIA trace import from fixture if no NVIDIA hardware is local.

Quality gates:

- no fabricated zero resources;
- no cross-vendor metric comparison without registry support;
- every derived counter states formula and raw inputs;
- reports list dropped rows and unmatched counters;
- hot-kernel ranking stable across equivalent importer paths.

## Risks

- Vendor profiler exports change across versions.
- Some counters require replay and perturb execution.
- Some dynamic counters are sampled over ranges broader than one kernel.
- Graph capture can hide dispatch boundaries from normal profiler paths.
- Source/ISA correlation can be missing in stripped or JIT-only flows.
- CUPTI and ROCm APIs may need privileges or environment setup.
- Percent-of-peak can be misleading unless dtype/primitive and clock assumptions are explicit.

Mitigations:

- capability registry per tool version;
- quality metadata on every metric;
- raw artifact retention;
- calibration fixtures;
- report missing evidence explicitly;
- keep vendor importers independent from in-house samplers.

## Done Definition

The build is complete when BoltBeam can run this class of workflow:

```bash
boltbeam collect-hw-trace --provider tinygrad --sampler tinygrad-hcq-pmc --run outputs/qwen14b ...
boltbeam import-hw-trace llama_or_microbench_report --provider tinygrad --out baseline_hw_trace.json
boltbeam compare-hw-trace --baseline baseline_hw_trace.json --candidate outputs/qwen14b/hw_trace.json --markdown report.md
```

and the report answers:

- what kernels dominate;
- what fraction of hardware peak they reach;
- which hardware unit limits them;
- what source/ISA/schedule facts explain the limit;
- whether the 14B-vs-8B/llama theory is proved, refuted, or missing specific evidence;
- what exact measurement or codegen change should happen next.

For the 14B prefill case specifically, done means the hot `ffn_gate_up Q4_K [512,17408,5120]` row has timing, bytes,
launch/resources, source/static diagnostics, counters when obtainable, and a final causal classification or a precise
instrumentation blocker.
