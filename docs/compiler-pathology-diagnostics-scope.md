# Compiler pathology diagnostics scope

## Goal

Add a BoltBeam diagnostic layer that can classify **why a generated kernel is pathologically slow**, without moving GPU execution out of tinygrad.

This is motivated by the G=5 block-tile result:

```text
baseline flash_partial_coop_vec: ~27 us/workgroup
G=5 block tile:                 ~2090 us/workgroup
ratio:                          ~78x slower
```

That ratio is too large to treat as ordinary route tradeoff. It should become a reusable BoltBeam diagnosis:

```text
REGALLOC_SPILL
BARRIER_EXPLOSION
LDS_OR_MEMORY_OVERHEAD
LOOP_LOWERING_BAD
VECTOR_LOAD_LOST
WAITCNT_BAD
NATIVE_ISA_ORACLE_NEEDED
STRUCTURAL_ROUTE_REFUTED
```

## Boundary

BoltBeam owns:

- normalized diagnostic schema;
- evidence ingestion;
- pathology classification;
- ledger/reopen directives;
- reports.

tinygrad owns:

- generating microkernels;
- compiling kernels;
- producing disassembly/resource/timing artifacts;
- running authority correctness and W==D gates;
- implementing any eventual codegen/backend fix.

Do not add GPU execution to BoltBeam.

## Source citations

Claude should read these before implementing.

| claim | citation |
|---|---|
| BoltBeam is the audit brain, tinygrad is the executor | `docs/audit-brain-build-scope.md` |
| Reachability classes already distinguish emitter-blocked, primitive-missing, refuted-by-ledger, target-incomplete | `boltbeam/search/reachability.py`, `boltbeam/vocab.py` |
| Normalized evidence is the only evaluator input shape | `boltbeam/artifacts/base.py`, `schemas/normalized_evidence.schema.json` |
| Target capabilities already carry LDS/vector/dot/dequant primitives | `boltbeam/profile/ir.py`, `boltbeam/data/targets.json` |
| Roofline is math-only in BoltBeam and consumes tinygrad evidence | `docs/roofline-audit-system-scope.md`, `boltbeam/math/roofline.py` |
| G=5 block tile native-context result: both routing strategies refuted, ~2090 us/workgroup vs ~27 us baseline, suspected codegen pathology | `boltbeam/data/candidates.json` candidate `decode_flash_block_tile_g5_native_context`; tinygrad commits `a22b770`, `ddb06b1`; BoltBeam commit `0d61fe6` |
| Prior LDS/prologue correction: LDS/barrier primitives exist; missing item was route/emitter shape, not a base primitive | `docs/lds5-ledger-update-20260701.md`, `docs/lds5-prologue-range-update-20260701.md` |
| Attention combine closure shows mechanism-pass/speed-refute must be recorded honestly | `docs/attention-combine-reachability-audit-20260701.md`, `docs/attention-combine-closure-stress-test-20260701.md` |

## Phase CP0: Evidence Contract

Add a new normalized row kind:

```text
compiler_pathology
```

Add it centrally:

- `boltbeam/vocab.py`
- `schemas/normalized_evidence.schema.json`
- tests that reject parallel string vocabularies

Rows should use existing `EvidenceRow` fields:

```json
{
  "kind": "compiler_pathology",
  "metric": "scratch_bytes",
  "value": 0,
  "unit": "bytes",
  "role": "attn_kv",
  "context": 512,
  "extra": {
    "kernel": "flash_block_tiled_g5_score_pv",
    "candidate_id": "decode_flash_block_tile_g5_native_context",
    "source_group": "g5_block_tile",
    "baseline_kernel": "flash_partial_coop_vec",
    "baseline_value": 27.0,
    "baseline_unit": "us/workgroup"
  }
}
```

Required metric names:

| metric | unit | meaning |
|---|---|---|
| `time_us_per_workgroup` | `us/workgroup` | normalized role-local kernel cost |
| `time_ratio_vs_baseline` | `ratio` | candidate kernel cost / baseline kernel cost |
| `vgpr` | `registers` | VGPR count |
| `sgpr` | `registers` | SGPR count |
| `scratch_bytes` | `bytes` | scratch/spill bytes |
| `lds_bytes` | `bytes` | group segment / LDS bytes |
| `static_instr` | `instructions` | total static instruction count |
| `valu_instr` | `instructions` | static VALU count |
| `salu_instr` | `instructions` | static SALU count |
| `vmem_instr` | `instructions` | static VMEM count |
| `lds_instr` | `instructions` | static LDS/DS count |
| `barrier_count` | `instructions` | static barrier count |
| `waitcnt_count` | `instructions` | static waitcnt count |
| `vector_load_bits` | `bits` | actual emitted vector load width |
| `occupancy_waves_per_cu` | `waves/cu` | estimated or measured occupancy |

Optional metric names:

| metric | unit | meaning |
|---|---|---|
| `dynamic_valu_per_wave` | `instructions/wave` | PMC-derived dynamic VALU |
| `dynamic_lds_per_wave` | `instructions/wave` | PMC-derived dynamic LDS |
| `dynamic_vmem_per_wave` | `instructions/wave` | PMC-derived dynamic VMEM |
| `wave_cycles_per_wave` | `cycles/wave` | PMC-derived wave time |

Acceptance:

- schema validates fixtures;
- row kind is defined only in `vocab.py`;
- evidence rows can be serialized/deserialized through `NormalizedEvidence`;
- no evaluator logic reads raw tinygrad fields directly.

## Phase CP1: Tinygrad Artifact Adapter

Add a tinygrad adapter for a future artifact schema:

```text
tinygrad.compiler_pathology.v1
```

Expected raw artifact shape:

```json
{
  "schema": "tinygrad.compiler_pathology.v1",
  "model_id": "qwen3-14b",
  "target_id": "amd_gfx1100",
  "workload": "decode",
  "candidate_id": "decode_flash_block_tile_g5_native_context",
  "baseline": {
    "kernel": "flash_partial_coop_vec",
    "time_us_per_workgroup": 27.0
  },
  "kernels": [
    {
      "kernel": "flash_block_tiled_g5_score_pv",
      "role": "attn_kv",
      "context": 512,
      "time_us_per_workgroup": 2090.0,
      "vgpr": 0,
      "sgpr": 0,
      "scratch_bytes": 0,
      "lds_bytes": 0,
      "static_counts": {
        "total": 0,
        "valu": 0,
        "salu": 0,
        "vmem": 0,
        "lds": 0,
        "barrier": 0,
        "waitcnt": 0
      },
      "vector_load_bits": 0,
      "occupancy_waves_per_cu": 0,
      "sources": {
        "disasm": "bench/.../disasm.txt",
        "resource": "bench/.../resource.json"
      }
    }
  ],
  "flags": {
    "route_bound": true,
    "token_match": true,
    "hidden_fallback": false
  }
}
```

Adapter behavior:

- detect by `schema == "tinygrad.compiler_pathology.v1"`;
- emit `compiler_pathology` rows;
- preserve source paths in `row.extra.sources`;
- compute `time_ratio_vs_baseline` if baseline time is present;
- fail with `AdapterIncomplete` if required fields are missing;
- never guess missing resource counts as zero.

Acceptance:

- fixture normalizes to `NormalizedEvidence`;
- incomplete artifact reports `AdapterIncomplete`;
- `target_id` override still works through `boltbeam ingest --target-id`;
- candidate id is preserved in row `extra`.

## Phase CP2: Pathology Classifier

Add a pure classifier module:

```text
boltbeam/diagnostics/pathology.py
```

Input:

```text
candidate_id
list[EvidenceRow(kind="compiler_pathology")]
optional target profile
```

Output:

```json
{
  "schema": "boltbeam.compiler_pathology_report.v1",
  "candidate_id": "...",
  "classification": "REGALLOC_SPILL",
  "confidence": "measured",
  "reason": "...",
  "trigger_rows": [...],
  "next_action": "...",
  "reopen_directive": "..."
}
```

Initial rules:

| class | condition | next action |
|---|---|---|
| `REGALLOC_SPILL` | `scratch_bytes > 0` or explicit spill row | fix regalloc/liveness before route replay |
| `BARRIER_EXPLOSION` | `barrier_count` is much higher than baseline or above configured threshold | inspect generated loop/barrier structure |
| `LDS_OR_MEMORY_OVERHEAD` | LDS/VMEM counts or dynamic LDS/VMEM ratios dominate without spill | inspect LDS staging and memory traffic |
| `LOOP_LOWERING_BAD` | static instruction or dynamic wave-cycle ratio is far above expected | inspect loop lowering/unrolling |
| `VECTOR_LOAD_LOST` | expected vector load width > actual `vector_load_bits` | wire vectorized load lowering |
| `WAITCNT_BAD` | waitcnt density high and dynamic waits high | inspect waitcnt scheduler |
| `NATIVE_ISA_ORACLE_NEEDED` | time ratio huge but no static resource signal explains it | build native ISA oracle to separate route vs lowering |
| `STRUCTURAL_ROUTE_REFUTED` | native oracle also slow or all diagnosed fixes fail W==D | keep ledger refuted |

Default thresholds should be data/config, not scattered constants:

```text
time_ratio_huge: 10x
scratch_bytes_spill: >0
barrier_static_high: target-dependent, default 16
vector_load_expected_bits: from target descriptor or row.extra
```

Acceptance:

- synthetic fixtures cover every class;
- G=5 fixture with only `time_ratio_vs_baseline=78` and no explanatory resource signal classifies as `NATIVE_ISA_ORACLE_NEEDED`;
- G=5 fixture with `scratch_bytes > 0` classifies as `REGALLOC_SPILL`;
- classifier emits a deterministic JSON report.

## Phase CP3: CLI And Report

Add CLI command:

```bash
boltbeam diagnose compiler-pathology \
  --evidence normalized_evidence.json \
  --candidate decode_flash_block_tile_g5_native_context \
  --out report.json
```

Optional markdown:

```bash
boltbeam diagnose compiler-pathology \
  --evidence normalized_evidence.json \
  --candidate decode_flash_block_tile_g5_native_context \
  --markdown report.md
```

Report should show:

- candidate;
- model/target/workload;
- top trigger metrics;
- classification;
- confidence;
- next action;
- whether tinygrad should:
  - build microkernels;
  - fix emitter/codegen;
  - build native ISA oracle;
  - leave route refuted.

Acceptance:

- CLI works on fixture evidence;
- markdown is deterministic;
- no tinygrad import in BoltBeam.

## Phase CP4: Ledger Integration

Teach evaluator/reporting to attach pathology report status to candidate decisions without turning it into a promotion gate.

Rules:

- If speed is refuted but pathology classification is `REGALLOC_SPILL`, `BARRIER_EXPLOSION`, `LOOP_LOWERING_BAD`, `VECTOR_LOAD_LOST`, or `WAITCNT_BAD`, the ledger status should stay refuted for the measured route but `do_not_retry=false` with a concrete reopen condition.
- If classification is `NATIVE_ISA_ORACLE_NEEDED`, the next action is an oracle build, not a route retry.
- If classification is `STRUCTURAL_ROUTE_REFUTED`, ledger may set `do_not_retry=true`.

Acceptance:

- no existing promoted/refuted decisions change unless pathology evidence is supplied;
- policy guard passes;
- report cites pathology evidence source.

## Phase CP5: Tinygrad Later Producer Scope

Do not implement this in BoltBeam. This is the later tinygrad producer contract.

tinygrad should add a tool such as:

```text
extra/qk_compiler_pathology_audit.py
```

It should produce:

```text
bench/compiler-pathology/<candidate>/latest.json
bench/compiler-pathology/<candidate>/summary.md
bench/compiler-pathology/<candidate>/disasm_*.txt
bench/compiler-pathology/<candidate>/resource_*.json
```

Required tinygrad phases:

| phase | work |
|---|---|
| G5P0 | static resource/disasm audit for baseline vs G=5 block tile |
| G5P1 | microkernels: K-only LDS, V-only LDS, K+V LDS, G=5 reduce-only, online-softmax-only, full tile |
| G5P2 | per-workgroup timing for every microkernel |
| G5P3 | emit `tinygrad.compiler_pathology.v1` |
| G5P4 | optional native ISA oracle only if BoltBeam says `NATIVE_ISA_ORACLE_NEEDED` |
| G5P5 | replay full route only after a codegen/pathology fix |

The tinygrad artifact should be designed so BoltBeam can ingest it without a custom one-off script per phase.

## Non-goals

- Do not add GPU profiling or disassembly execution to BoltBeam.
- Do not make native ISA the implementation path by default.
- Do not promote a route because one microkernel looks faster.
- Do not convert every historical tinygrad audit into this schema retroactively.
- Do not classify missing fields as zero.

## Expected first use

First target:

```text
candidate_id: decode_flash_block_tile_g5_native_context
model: qwen3-14b
target: amd_gfx1100
workload: decode
```

Expected first classification if only current evidence is supplied:

```text
NATIVE_ISA_ORACLE_NEEDED
```

Reason:

```text
The measured route is ~78x slower per workgroup, but current BoltBeam evidence
does not yet contain scratch/barrier/static-count rows that identify a specific
compiler pathology.
```

That is a useful outcome. It prevents a blind retry and asks tinygrad for the exact missing microscope.

