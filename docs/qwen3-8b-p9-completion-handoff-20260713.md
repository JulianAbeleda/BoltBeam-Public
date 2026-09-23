# Qwen3-8B P9 completion handoff

Date: 2026-07-13

Audience: Claude or another implementation agent continuing the operand-path work.

Status authority: [operand-path-selection-implementation-status-20260713.md](operand-path-selection-implementation-status-20260713.md)

Scope authority: [operand-path-selection-e2e-scope-20260713.md](operand-path-selection-e2e-scope-20260713.md)

This document is an execution handoff, not a second policy source. If it conflicts with the scope or status authority,
the scope and status win.

## Required outcome

Close P9 for the current Qwen3-8B Q4_K_M routes on RX 7900 XTX / gfx1100 without changing the shipped routes merely
to make evidence collection easier. Produce a production, identity-bound P9 bundle for prefill and decode in which:

1. each important semantic operand has one exclusive strategy classification or explicit `unknown`;
2. the exact shipping binary, final ISA, ABI, resources, execution, correctness, and timing identities join;
3. every important role has a measured candidate matrix or typed infeasibility;
4. full-output correctness passes before timing is admitted;
5. raw and practical rooflines plus clean model throughput are attached; and
6. `p9_complete=true` is set only by a centralized machine-enforced predicate after every production gate passes.

Unavailable gfx1100 cache PMCs may remain typed `unsupported`. They reduce mechanism confidence but do not block a
correct same-session timing selection or product completion.

## Starting checkpoint

Repositories are clean and pushed at:

- BoltBeam `main`: `421272f` (`[eval] add isolated AMD execution helpers`)
- tinygrad-arkey `master`: `93045034c` (`[prefill] add isolated operand execution bridge`)

Current trusted facts:

- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`;
- target: RX 7900 XTX / `gfx1100`;
- clean authorities: prefill pp512 `3881 tok/s`, decode ctx512 `117.1 tok/s`;
- post-change smoke: prefill pp512 `3860 tok/s`, decode ctx512 `116.9 tok/s`;
- exact shipping prefill `attn_qo` binary:
  `ad3a947d218302443523eb03b622302b2f7c90706fcf83b1a59666a61e17cf3e`;
- that binary passed all 2,097,152 fp16 outputs with `max_abs_error=0`, intact guards, and healthy pre/postflight;
- final resources: 188 metadata VGPRs, 248 descriptor VGPRs, 18 SGPRs, 40,960-byte LDS, no scratch or spills;
- final structure: 8 global loads, 48 DS loads, 8 DS stores, 29 waits, 2 barriers, and 32 WMMAs;
- the admitted prefill schedule declares separate A/B LDS windows and double buffering, but the shipping final ISA does
  not yet bind those instructions back to A or B;
- the decode adapter compiles the generated Q4_K G3 route but intentionally refuses dispatch because no exact packed
  weight, activation, and independent reference artifact is bound to it.

Existing model and roofline artifacts are under:

```text
/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/
```

The canonical promoted prefill candidate set is:

```text
/home/ubuntu/tinygrad-arkey/bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/candidate-set.json
```

## Non-negotiable evidence rules

- Compile and execute the shipping `DEV=AMD` path. `AMD:ISA:gfx1100` produces a different binary and has already
  failed numerical correctness; it is fixture/compiler evidence only.
- Never infer operand ownership from a route name, total LDS allocation, an alternate binary, or a source schedule.
- Never convert unavailable PMCs into zero hits or zero bytes.
- Keep transport strategy orthogonal to serving tier. `lds_staged` does not mean the LDS fill came from L2.
- Keep correctness, speed, route binding, and mechanism attribution as separate facts.
- All GPU dispatch remains behind process isolation, a hard timeout, guard buffers, and pre/postflight health checks.
- Do not dispatch the retired raw LDS2 oracle; it hangs. Do not revive handwritten pipe/LDS2 selectors.
- Do not mutate runtime defaults or promote a route from BoltBeam. BoltBeam emits and evaluates typed contracts;
  provider code owns compilation and execution.
- Extend existing authoritative modules. Do not add one-off benchmark scripts or a parallel evidence schema.

## Work package 1: shipping-HIP semantic operand mapping

### Objective

Make the exact shipping AMD/HIP code object emit or derive stable semantic ownership for final global-load, DS-stage,
scratch/spill, wait/barrier, and WMMA rows. At minimum the prefill A/B path must be traceable from ABI pointer through
global load, LDS store/load, and WMMA consumption. Decode must identify packed weights and activation independently.

### Existing authority to extend

tinygrad-arkey:

- `extra/qk/prefill/current_prefill_execution_adapter.py`
- `extra/qk/decode/current_decode_execution_adapter.py`
- `extra/qk/mmq_compile_evidence.py`
- `extra/qk/mmq_epoch_manifest_export.py`
- `tinygrad/renderer/cstyle.py` (`HIPRenderer`)
- `tinygrad/renderer/amd/elf.py`
- `tinygrad/renderer/isa/amd.py` for the provider-neutral row vocabulary and fixture path only

BoltBeam already consumes explicit final-row metadata in:

- `boltbeam/kernel_analysis/provider_adapters.py`
- `boltbeam/kernel_analysis/classify.py`
- `boltbeam/kernel_analysis/model.py`

Do not add classification rules to tinygrad and do not add AMD mnemonics to BoltBeam core.

### Implementation shape

Choose one exact-binary mechanism and document why it is identity-preserving:

1. Prefer compiler-owned semantic IDs carried into an exact-code-object sidecar, if the shipping HIP compilation path
   can preserve them without changing production code generation.
2. Otherwise derive ownership from the exact final disassembly with a bounded dataflow pass rooted at the ABI pointer
   loads. Track address provenance into global loads, loaded VGPRs into DS writes/reads, and WMMA source consumption.
3. If a row remains ambiguous, emit that row with explicit unknown ownership and a missing discriminator. Never guess.

The output must use the existing `tinygrad.amd_isa_proof_manifest.v1` row fields (`operand_id`,
`source_operand_id`, `fetch_group`, cache policy, widths, retained-fragment metadata) and remain joined to the exact
`binary_sha256`. Remove the current `operand_paths: []` blocker only when the exact shipping rows carry real ownership.

### Acceptance

- A/B and packed-weight/activation ABI roots are explicit and unique.
- Every attributed row has a stable instruction address or operation ID and exact binary hash.
- Ambiguous rows survive as unknown; no relevant final row is silently dropped.
- Prefill classification reports both schedule declaration and final-binary evidence without treating them as the same
  authority.
- Tests pin positive A/B flows, ambiguous aliases, reloads, spills, missing rows, and binary mismatch.
- The existing rejection of alternate `AMD:ISA` evidence remains tested.

## Work package 2: immutable decode execution authority

### Objective

Turn `CurrentDecodeExecutionAdapter` from compile-only into a guarded production adapter for at least the canonical
generated Q4_K G3 `ffn_gate_up` shape `(rows=12288, k=4096)`, then generalize by route-manifest/model-profile facts
rather than model-name branches.

### Artifact contract

Capture, without repacking or reinterpretation:

- exact packed Q4_K words consumed by the shipping kernel;
- exact contiguous fp16 activation consumed by that invocation;
- independently computed float32 reference output;
- model file identity, tensor identity/role, shape, layout, dtype, byte length, and SHA-256 for every value;
- reference producer and its source/binary or implementation identity.

The candidate kernel's output is not an independent reference. Prefer the ordinary tinygrad dequantized graph or an
existing trusted CPU formula. Compare all output rows. Store large payloads outside Git and commit only the small
content-addressed manifest if repository policy requires durable metadata.

Extend `extra/qk/decode/current_decode_execution_adapter.py`; do not create a standalone capture script. Reuse
`PreparedExecution`, `GuardPolicy`, the isolated worker, and the input/reference identity pattern already used by
`current_prefill_execution_adapter.py`.

Register the decode adapter lazily in
`extra/qk/prefill/operand_path_execution_worker.py::_PRODUCTION_ADAPTER_LOADERS`. Importing the adapter must not mutate
global state or initialize a GPU runtime.

### Acceptance

- Parent compile and child recompile produce the same source and binary hashes before dispatch.
- ABI is exactly output, packed words, activation.
- Guarded execution checks all float32 output elements, finite output, input immutability, guard integrity, and GPU
  health before and after dispatch.
- Any compile, timeout, health, correctness, or identity failure prevents timing.
- Unsupported counters remain a typed counter outcome while valid timing remains usable.
- Focused tests cover malformed artifacts, wrong hashes/layouts/shapes, reference aliasing, binary drift, timeout, and
  success. The real GPU gate records the exact artifact identities.

## Work package 3: complete current-8B provider reports

Derive important roles from the canonical model profile, route manifest, and measured profile share. Do not hardcode a
second role list in BoltBeam. For each material role:

- emit a feasible operand strategy matrix with `build_candidate_matrix` and `prune_candidates`, or
- emit typed infeasibility with the provider capability/reason.

At minimum, account for the four promoted prefill roles (`ffn_gate_up`, `ffn_down`, `attn_qo`, `attn_kv`) and the
material generated Q4_K/Q6_K decode roles exposed by the current model profile. Generalize the prefill adapter across
the canonical candidate-set entries instead of cloning it per role.

Run feasible cohorts through the existing stdin/stdout execution bridge and
`operand_path_execution_worker.execute_session`. Requirements:

- all candidates in one cohort share workload, ABI, invariants, system snapshot, clock state, and reference;
- correctness-gate every candidate before any timed round;
- randomized same-session order, declared warmups/rounds, raw samples retained;
- ranking only through `rank_measured_matrix`;
- prediction may order experiments but may not promote;
- missing cache PMCs lower mechanism confidence only.

Normalize provider results into existing `KernelEvidence`, run `classify_kernel_evidence`, render the per-operand
report, and produce a measured recommendation or typed non-decision. No route change is required merely to complete
the report.

## Work package 4: production P9 bundle

Extend `boltbeam/kernel_analysis/p9_bundle.py` with one centralized completion predicate. The current implementation
always writes `p9_complete=false`; do not flip it unconditionally.

For a non-synthetic request, the predicate must verify at least:

- exact model/quant/target/context identity on both provider halves;
- non-empty important-role coverage with matrix or typed infeasibility;
- exact final-ISA operand report and binary join;
- full-model prefill and decode correctness status `pass`;
- valid measured ranking or typed non-decision from one session;
- raw/practical roofline plus classified fetch groups;
- clean before/after model throughput authority;
- explicit observed and missing cache evidence;
- no synthetic fixture or synthetic evidence in a production claim.

Add negative tests for each gate and one production-shaped positive fixture. Keep the existing synthetic structural
golden explicitly non-production. Build the final request from real artifacts and write the result under a dated
`/home/ubuntu/boltbeam-runs/` directory; commit only durable small authorities according to repository policy.

## Verification sequence

Run CPU/contract gates first:

```bash
cd /home/ubuntu/BoltBeam
python3 -m pytest -q tests/kernel_analysis tests/test_amd_runtime_bridge.py tests/test_process_isolated.py

cd /home/ubuntu/tinygrad-arkey
python3 -m pytest -q \
  test/unit/test_amd_final_elf_capture_20260712.py \
  test/unit/test_amd_isa_extraction_fixtures.py \
  test/unit/test_mmq_epoch_manifest_export.py \
  test/unit/test_execution_bridge_contracts.py \
  test/unit/test_current_prefill_execution_adapter.py \
  test/unit/test_current_decode_execution_adapter.py \
  test/unit/test_operand_path_execution_worker.py
```

Then, with no competing GPU process:

1. run the tiny health canary;
2. compile only and inspect source/binary/ISA/resource identities;
3. execute one guarded correctness run in a hard-timeout child;
4. confirm postflight health;
5. run the full candidate cohort;
6. rerun clean model prefill pp512 and decode ctx512 throughput;
7. confirm final GPU health and idle state;
8. build and validate the production P9 bundle.

Run each repository's full test suite before final commit. Commit provider behavior in tinygrad separately from
BoltBeam evaluation/report changes, using each repository's required subsystem prefix, and push both branches.

## Definition of done

Do not report completion until all boxes are true:

- [ ] Exact shipping HIP final rows have semantic ownership or explicit per-row unknowns.
- [ ] Prefill A/B and decode packed-weight/activation classifications are rendered from exact-binary evidence.
- [ ] Decode immutable input/reference artifact exists and is content-addressed.
- [ ] Decode guarded full-output correctness passes on real gfx1100.
- [ ] Every important current-8B role has a candidate matrix or typed infeasibility.
- [ ] Same-session measured rankings or typed non-decisions exist for prefill and decode.
- [ ] Clean prefill/decode model correctness and throughput remain within the declared authority/noise rule.
- [ ] Raw and practical rooflines are joined without relabeling modeled cache fit as measured residency.
- [ ] Production P9 request contains no synthetic evidence.
- [ ] The centralized predicate emits `p9_complete=true`.
- [ ] Both full test suites pass, GPU is healthy, commits are pushed, and working trees are clean.

If exact shipping-binary ownership cannot be recovered, preserve `unknown`, record the precise missing discriminator,
and leave P9 blocked. That is an honest blocker, not permission to substitute route names, schedule intent, or the
different `AMD:ISA` binary.
