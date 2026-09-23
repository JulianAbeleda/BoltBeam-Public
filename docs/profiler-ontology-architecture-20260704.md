# Profiler Portable Performance Ontology & Pipeline

Date: 2026-07-04. Companion to `profiler-goal-alignment-mvp-20260704.md` and
`gpu-agnostic-profiler-build-scope-20260704.md`. This fixes the *architecture* the report engine (P4)
is built on so it stays vendor-neutral instead of hard-coding AMD/tinygrad specifics.

## Pipeline (each stage is a distinct, testable layer)

```
portable performance ontology     # vendor-neutral concepts (this doc)
  -> target capability model       # what THIS gpu/arch can do & expose
  -> provider/import adapter        # tinygrad-PMC / rocprof-compute / ncu / llama -> raw evidence
  -> canonical evidence model       # boltbeam.hw_trace.v1 rows in ontology terms
  -> target-specific metric formulas# derive %-of-peak/occupancy/bw using target constants
  -> model-aware report             # roles/shapes/quant, top kernels, sections
  -> causal explanation             # more-work vs lost-efficiency, named cause + missing evidence
```

Rule that spans every stage: **`missing_evidence` is a first-class value.** A concept that a target
cannot expose, or a provider did not collect, is `missing`/`unsupported` — never zero, never inferred.

## Core concepts (generic; providers map INTO these, formulas read FROM these)

| concept | meaning | example source (varies by provider/target) |
|---|---|---|
| `dispatch` | one launch of device code: launch geometry, queue/graph identity, start/end/duration | tinygrad ProfileProgram/Graph events; ncu launch; rocprof dispatch |
| `program` | the compiled kernel behind dispatches: name, canonical/demangled name, ISA/resource footprint | tinygrad prog id + VGPR/SGPR/LDS/scratch; ncu kernel |
| `execution_unit` | a countable compute engine class the work runs on | SIMD/CU (RDNA), SM (NV) |
| `matrix_unit` | tensor/matrix pipeline presence + utilization | RDNA3 **WMMA**, CDNA **MFMA**, NV tensor cores → `tensor_core_util_pct` |
| `vector_unit` | scalar/vector ALU activity | SQ VALU/SALU busy → `valu_busy_pct` |
| `memory_level` | a level in the memory hierarchy with bytes/hit/miss/bandwidth | L1/L2 (GL2C/TCC), HBM/GDDR DRAM |
| `occupancy` | achieved residency vs the target's max, and what limits it (regs/LDS/scratch) | busy-cycles/active-waves vs peak |
| `bytes_moved` | bytes read/written per memory_level (measured or model-derived) | GL2C req counters; or weight-inventory bytes |
| `work_done` | algorithmic work: FLOPs / element ops for the dispatch's shape | shape+quant derived FLOPs/bytes |
| `baseline_efficiency` | achieved vs a reference (another model/provider/target) at matched work | 14B-vs-8B, tinygrad-vs-llama |
| `missing_evidence` | concepts required by a section but absent/unsupported, with the reason | "gfx11 PMC returns only SQ_BUSY_CYCLES" |

Naming: the canonical tensor concept is **`tensor_core_util_pct`** (in `data/counter_registry.json`),
NOT the legacy `mfma_util_pct`. gfx1100 supports **WMMA** (tinygrad lowers it: `renderer/cstyle.py:443`,
`renderer/llvmir.py:46`); CDNA supports MFMA. The target capability model records *which* matrix_unit a
target has; the report never says "MFMA absent" as if that were the only tensor path.

## Per-target measurement map (the moat: user asks in concepts, target answers in its own counters)

The user asks *"why is this 14B prefill GEMM slow?"* — never "what's my SQ_BUSY_CYCLES." Each target
declares **how each generic concept is measured on it**, so the report translates one question into
target-specific evidence. This lives in the target capability model (`data/targets.json`
`capabilities.measurement_map`), keyed by ontology concept:

| concept | NVIDIA | AMD RDNA (gfx11) | AMD CDNA | Apple/Metal (later) |
|---|---|---|---|---|
| execution_unit | SM | WGP/CU (wave32) | CU (wavefront64) | threadgroup/simdgroup |
| matrix_unit | Tensor Core | **WMMA** | MFMA | simdgroup_matrix |
| vector_unit | warp ALU | SQ VALU/SALU | SQ VALU | SIMD ALU |
| memory_level | L1/L2/DRAM | GL1/**GL2C**/DRAM | L2/**TCC**/HBM | L1/L2/unified |
| occupancy | warps active / SM max | waves active, GRBM/SQ busy | wavefronts / CU max | threads resident |
| bytes_moved | DRAM read/write ctrs | GL2C EA req counters | TCC req counters | perf counters |
| tool (provider) | Nsight/CUPTI | tinygrad-KFD-PMC / rocprof-compute | rocprof / rocprof-compute | Metal counters |

BoltBeam owns the concept vocabulary; the map is data, extended per target without touching the report
engine. When a target's map marks a concept measurable but the run didn't collect it, that's
`missing_evidence` with the target's own counter named (e.g. "need GL2C_HIT — gfx11 PMC gap"); when the
target has no such unit, it's `unsupported` (e.g. no matrix_unit).

## Stage → BoltBeam module mapping (build on what exists)

- **ontology**: new `boltbeam/profiler/ontology.py` — the concept vocabulary + `missing`/`unsupported`
  sentinels. Anchors the section engine.
- **target capability model**: extend `boltbeam/targets.py` + `data/targets.json` and
  `boltbeam/profiler/capabilities.py` + `data/profiler_capabilities.json` — per target: execution_unit
  count, matrix_unit kind (`wmma`/`mfma`/`tensor`/none), memory_levels + peaks, and which counters are
  exposable (so `missing != coverage-failure` when the target simply can't).
- **provider/import adapter**: `boltbeam/profiler/importers/{ncu,rocprof_compute}.py` +
  `boltbeam/collectors/*` → each maps raw provider names to ontology concepts via the **counter
  registry** (`boltbeam/profiler/counters.py`, `data/counter_registry.json`) carrying source name,
  unit, formula, quality. Add tinygrad-native-pmc + rocprof-compute provider_mappings (unify the two
  vocabularies AT the registry — don't rewrite producers).
- **canonical evidence model**: `boltbeam.hw_trace.v1` rows already carry role/shape/quant, resources,
  counters. Harden JSON-Schema validation into the prod path (currently test-only).
- **target-specific metric formulas**: `boltbeam/roofline_ceiling.py` + registry formulas — compute
  %-of-peak/occupancy/bw ONLY when all required ontology inputs are present.
- **model-aware report**: `boltbeam/profiler/report.py` + `sections/` (new) — reuse
  `prefill_role_trace.py`, `prefill_roofline*.py`, `substrate_compare.py`, `compare-hw-trace`.
- **causal explanation**: `boltbeam/profiler/sections/causal.py` (new) — `doing_more_work` vs
  `lost_efficiency` with named sub-causes (shape/tile, occupancy, memory-bound-scalar,
  tensor-core-underuse, graph/launch, codegen pathology via `diagnostics/pathology.py`), each citing the
  ontology evidence it used and listing `missing_evidence`.

## Immediate consequence for the 14B case
On gfx1100 today, provider evidence is: `dispatch`/`program`/resources/`work_done`/`bytes_moved`
(model-derived) + `occupancy` proxy from `SQ_BUSY_CYCLES` only. `vector_unit`/`matrix_unit`/`memory_level`
hit-rate are **`missing_evidence`** (tinygrad gfx11 multi-counter PMC bug — under investigation). The
report must render a real speed-of-light/roofline placement from what exists and a causal verdict of
either "more work" (if `work_done` scales with the slowdown) or "insufficient — need
`valu_busy_pct`/`tensor_core_util_pct`/`memory_level` counters," naming the gfx11 PMC gap.
