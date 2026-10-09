# 14B Prefill "Why" Trace Workflow

Date: 2026-07-04.

## Goal

BoltBeam must be able to answer why tinygrad 14B prefill is slower than llama, not just where it is slower.
For every hot packed prefill GEMM row, the trace should explain:

- workgroup and grid shape;
- VGPR and SGPR count;
- LDS usage;
- scratch/spill usage;
- occupancy;
- memory-busy, VALU-busy, and MFMA utilization;
- generated source and schedule parameters;
- unpack/load pattern details.

The current L1 `tinygrad-profile-events` sampler answers lifecycle timing, role, quant, shape, calls, and estimated
packed bytes. It does not yet fill resource, counter, source, or unpack diagnostics, so it localizes the 14B gap but
does not fully explain it.

## Current BoltBeam Inventory

Already implemented:

- `boltbeam.hw_trace.v1` can carry `resources` and `counters` on kernel rows.
- `boltbeam/artifacts/llama_rocprof.py::resource_fields_from_backend_row` normalizes backend resource fields:
  `workgroup`, `grid`, `lds_bytes`, `scratch_bytes`, `vgpr`, and `sgpr`.
- `boltbeam/hw_trace.py` normalizes counter names:
  `occupancy_pct`, `memory_busy_pct`, `memory_stall_pct`, `l2_hit_pct`, `fetch_kb`, `write_kb`,
  `valu_busy_pct`, `lds_conflict_pct`, `lds_stall_pct`, and `mfma_util_pct`.
- `boltbeam/substrate_compare.py` consumes resource/counter fields and reports missing evidence.
- `boltbeam/artifacts/tinygrad.py` ingests `tinygrad.compiler_pathology.v1`.
- `boltbeam/diagnostics/pathology.py` classifies compiler causes:
  `REGALLOC_SPILL`, `BARRIER_EXPLOSION`, `LDS_OR_MEMORY_OVERHEAD`, `LOOP_LOWERING_BAD`,
  `VECTOR_LOAD_LOST`, `WAITCNT_BAD`, `NATIVE_ISA_ORACLE_NEEDED`, `STRUCTURAL_ROUTE_REFUTED`, and `UNKNOWN`.
- `boltbeam diagnose compiler-pathology` exists.
- `docs/low-level-gpu-sampler-scope.md` scopes L1/L2 in-house sampling.

Current gap:

- The 14B tinygrad `tinygrad/profile-events` trace has timing/shape/bytes only.
- The hot rows have no `resources` and no `counters`.
- Existing llama `kernel_trace.csv` has resource columns, but the previous llama comparison used `kernel_stats.csv`,
  which does not carry resource fields.

## Tinygrad Producer Surfaces

The required fields are available in or near tinygrad's AMD runtime:

| required field | tinygrad surface |
|---|---|
| workgroup/grid shape | `HCQProgram.__call__(global_size=..., local_size=...)` in `tinygrad/runtime/support/hcq.py` |
| LDS bytes | `AMDProgram.group_segment_size` from AMD kernel descriptor in `tinygrad/runtime/ops_amd.py` |
| scratch bytes | `AMDProgram.private_segment_size` from AMD kernel descriptor |
| rsrc registers | `AMDProgram.rsrc1`, `rsrc2`, `rsrc3` |
| program identity | `ProfileProgramEvent(device, name, lib, base, tag)` |
| timing | `ProfileRangeEvent` and `ProfileGraphEvent` |
| PMC counters | `ProfilePMCEvent`, `pmc_start`, `pmc_read`, `PMC=1` |
| SQTT/deeper instruction trace | `ProfileSQTTEvent`, `SQTT=1` |
| generated source/binary | compiler `src` before compile and `ProfileProgramEvent.lib` after compile |

BoltBeam should not execute kernels itself. tinygrad should emit raw profile/resource/source artifacts; BoltBeam should
normalize, compare, and classify them.

## Required Workflow

### Phase PWT0: Re-import llama resources

Use the existing llama `kernel_trace.csv`, not only `kernel_stats.csv`.

Producer input:

```text
outputs/llama-prefill-qwen14b-pp512/rocprof/qwen14b_pp512_kernel_trace.csv
```

Required BoltBeam changes:

- Add an importer path that can aggregate trace CSV rows by kernel while preserving:
  `LDS_Block_Size`, `Scratch_Size`, `VGPR_Count`, `SGPR_Count`, `Workgroup_Size_X/Y/Z`, and `Grid_Size_X/Y/Z`.
- Keep the current bench-wall normalization behavior.
- Emit a llama `hw_trace.json` with `resources` populated on `mul_mat_q<`.

Acceptance:

- llama 14B `mul_mat_q<` row has non-empty `resources`.
- `compare-substrate` shows llama resource text instead of `missing`.

### Phase PWT1: Tinygrad launch/resource rows

Extend tinygrad's profile event producer so every AMD kernel execution can be joined to:

- global size;
- local size;
- `group_segment_size`;
- `private_segment_size`;
- `rsrc1`, `rsrc2`, `rsrc3`;
- decoded VGPR/SGPR if available.

Preferred tinygrad shape:

```json
{
  "schema": "tinygrad.kernel_resource_trace.v1",
  "programs": [
    {
      "tag": 0,
      "name": "prefill_q4k_direct_packed_load_direct_out_gemm_17408_5120_512_1",
      "source_path": "optional",
      "source_sha256": "sha256:...",
      "lib_sha256": "sha256:...",
      "resources": {
        "lds_bytes": 0,
        "scratch_bytes": 0,
        "rsrc1": 0,
        "rsrc2": 0,
        "rsrc3": 0,
        "vgpr": null,
        "sgpr": null
      }
    }
  ],
  "dispatches": [
    {
      "exec_tag": 1,
      "program_tag": 0,
      "global_size": [1, 1, 1],
      "local_size": [1, 1, 1]
    }
  ]
}
```

Required BoltBeam changes:

- Add adapter/decoder for `tinygrad.kernel_resource_trace.v1`.
- Join it into `tinygrad-profile-events` rows by program tag/name and execution tag where possible.
- Put launch shape and descriptors under `row.resources`.
- Never default missing descriptor fields to zero.

Acceptance:

- tinygrad 14B hot rows show `WG=...`, `grid=...`, `lds_bytes`, `scratch_bytes`, `vgpr`, and `sgpr` when present.
- Missing fields are absent, not fabricated.

### Phase PWT2: Source and schedule fingerprint

For every hot packed GEMM program, preserve:

- generated kernel source or source path;
- source hash;
- binary hash;
- rendered schedule parameters: global size, local size, opt list, tile/upcast/local axes if tinygrad exposes them;
- role/shape/quant attribution already known by BoltBeam.

Required tinygrad work:

- Capture the rendered source at compile time for selected hot kernels, or write it to a trace directory keyed by
  program tag.
- Capture schedule/opt metadata before source rendering when available.

Required BoltBeam changes:

- Add `sources` under kernel rows:
  `source_path`, `source_sha256`, `lib_sha256`, `schedule`.
- Add a markdown report section listing source/schedule fingerprints for hot rows.

Acceptance:

- The `ffn_gate_up Q4_K [512,17408,5120]` row links to the exact generated source/binary fingerprint used by the run.

### Phase PWT3: Static generated-code diagnostics

From generated source or disassembly, derive:

- static instruction count;
- VALU/SALU/VMEM/LDS/MFMA counts;
- barrier count;
- waitcnt count;
- vector load width;
- tensor-core/MFMA presence;
- unpack pattern summary.

For the 14B prefill issue, the first static diagnoses should answer:

- Does the hot Q4_K GEMM emit MFMA/WMMA instructions?
- Are packed weight loads vectorized/coalesced?
- Is unpack mostly scalar ALU?
- Are there many barriers/waitcnts?
- Is there obvious instruction bloat vs llama?

Required BoltBeam changes:

- Add `boltbeam/codegen_static.py` or similar pure parser for static source/disasm metrics.
- Emit `tinygrad.compiler_pathology.v1` rows or attach static metrics to `hw_trace` rows.
- Extend `compare-substrate` to classify:
  `packed_gemm_mfma_not_firing`, `packed_gemm_vector_load_lost`, `packed_gemm_unpack_alu_bound`,
  `packed_gemm_waitcnt_bad`, or `packed_gemm_static_unknown`.

Acceptance:

- A static-only report can classify missing MFMA or vector-load loss without PMCs.
- If static metrics are complete but inconclusive, the report says PMCs are required.

### Phase PWT4: PMC counters

Add the L2 in-house sampler path:

```bash
boltbeam collect-hw-trace \
  --provider tinygrad \
  --sampler tinygrad-hcq-pmc \
  --model /path/model.gguf \
  --context 512
```

Required tinygrad mode:

```text
PROFILE=1 PMC=1
```

Required fields:

- `memory_busy_pct`;
- `valu_busy_pct`;
- `mfma_util_pct` when counter is available;
- `l2_hit_pct`;
- `fetch_kb` / `write_kb` or quality-tagged approximations.

Required BoltBeam changes:

- Decode `ProfilePMCEvent`.
- Normalize PMC counters into `boltbeam.hw_trace.v1` counter vocabulary.
- Add `counter_quality` metadata for derived counters.
- Attach counters to kernel rows by `program_tag`/`exec_tag`.

Acceptance:

- Hot Q4_K rows have non-empty `counters`.
- `compare-hw-trace` reports weighted counters for tinygrad.
- Counter formulas are documented and marked exact/derived.

### Phase PWT5: Causal classifier

Once resource/static/counter fields exist, `compare-substrate` should move from "missing resources" to an explicit
cause:

| evidence | classification |
|---|---|
| `scratch_bytes > 0` | `packed_gemm_register_spill` |
| low occupancy or huge VGPR/SGPR | `packed_gemm_occupancy_starved` |
| no MFMA/static tensor-core instructions | `packed_gemm_mfma_not_firing` |
| vector load width below expected | `packed_gemm_vector_load_lost` |
| high VALU, low MFMA, high unpack static count | `packed_gemm_unpack_alu_bound` |
| high memory busy, low effective GB/s, poor load vectorization | `packed_gemm_memory_pipeline_gap` |
| high LDS conflicts/stalls | `packed_gemm_lds_conflict` |
| all fields complete but no rule matches | `packed_gemm_unknown_resource_complete` |

Acceptance:

- The 14B report names one of these causes for `ffn_gate_up Q4_K [512,17408,5120]`.
- If it cannot name a cause, it explicitly lists the still-missing fields.

## Execution Order

1. PWT0: enrich llama from existing `kernel_trace.csv`.
2. PWT1: add tinygrad launch/resource trace producer and BoltBeam join.
3. PWT2: preserve generated source/schedule fingerprints for hot kernels.
4. PWT3: add static source/disasm diagnostics.
5. Rerun 14B comparison. If static cause is decisive, start tinygrad fix.
6. PWT4: add PMC counters only if static/resource evidence is insufficient.
7. PWT5: upgrade substrate compare cause classifier.

This order avoids blocking on counters if descriptors/source already explain the 13x Q4_K gap.

## Deliverable Definition

The workflow is complete when a single command can produce a report with no hand inspection:

```bash
boltbeam collect-hw-trace --provider tinygrad --sampler tinygrad-why --run outputs/qwen14b ...
boltbeam compare-substrate --baseline outputs/llama14b/hw_trace.json --candidate outputs/qwen14b/hw_trace.json ...
```

The report must include, for each hot packed GEMM:

- timing, bytes, effective GB/s;
- role, quant, shape;
- workgroup/grid;
- VGPR/SGPR/LDS/scratch;
- source/schedule fingerprint;
- static codegen summary;
- counters when available;
- final causal classification or an explicit missing-field list.
