# Low-Level GPU Sampler Scope

BoltBeam's public trace contract is `boltbeam.hw_trace.v1`. The remaining dependency to remove is the external
backend profiler used to produce kernel timing/counter rows. The low-level sampler should be a BoltBeam-owned layer
that can attach to runtime dispatch boundaries and emit the same trace rows directly.

This is not a request to clone `rocprofv3`. The target is narrower and more maintainable: own the dispatch boundary
and counter normalization needed for model prefill/decode analysis, then emit BoltBeam traces without requiring an
external profiler in the normal path.

## Goal

Build a model-agnostic sampler that reports, per dispatched kernel:

- lifecycle timing: elapsed us, launch order, queue, start/end timestamps
- identity: provider, kernel name, program id, dispatch geometry, optional role/shape/quant attribution
- bytes: estimated packed bytes from BoltBeam inventory, measured read/write counters when available
- counters: normalized BoltBeam names (`memory_busy_pct`, `fetch_kb`, `mfma_util_pct`, etc.)
- evidence quality: exact/estimated, counter set, sampler backend, dropped rows, calibration notes

The sampler is not a model runner. It observes a provider run and emits `hw_trace.v1`.

## Existing Hooks

tinygrad already has the best first substrate:

- HCQ timestamps: `tinygrad/runtime/support/hcq.py` records GPU timestamps with `hcq_profile`.
- AMD PMCs: `tinygrad/runtime/ops_amd.py` has `pmc_start`, `pmc_read`, `ProfilePMCEvent`, and counter scheduling.
- AMD SQTT: same runtime has SQTT start/stop events for deeper instruction/occupancy analysis.
- Program ids: `ProfileProgramEvent` links generated program names/binaries to runtime executions.
- Existing PMC unpacker: `tinygrad/viz/serve.py:unpack_pmc` decodes PMC blobs into per-counter aggregates.

That means the first in-house path should not shell out to a profiler. It should run tinygrad with `PROFILE=1 PMC=1`,
collect `Compiled.profile_events`, decode PMC events, normalize them, and emit `hw_trace.v1`.

Concrete tinygrad anchors:

- `tinygrad/device.py`: `Compiled.profile_events`, `ProfileProgramEvent`, and `profile.pkl` export at profile finalize.
- `tinygrad/runtime/support/hcq.py`: `hcq_profile` wraps command queues and emits GPU timestamp ranges.
- `tinygrad/runtime/ops_amd.py`: `ProfilePMCEvent`, `pmc_start`, `pmc_read`, and PMC scheduling.
- `tinygrad/viz/serve.py`: `unpack_pmc` is an existing decoder/reference for PMC blobs.

## Architecture

```text
provider run
  -> sampler adapter
       tinygrad_hcq_pmc first
       llama/native later
  -> raw sample events
       dispatch timing
       counter blobs
       kernel metadata
  -> BoltBeam normalizer
       role/quant/shape attribution
       counter vocabulary mapping
       bytes/tok/s lifecycle summary
  -> boltbeam.hw_trace.v1
```

## Independence Levels

| Level | Name | External profiler? | What BoltBeam owns | Status |
|---|---|---:|---|---|
| L0 | normalized backend import | yes | schemas, vocab, comparison, artifact retention | implemented |
| L1 | tinygrad profile-event sampler | no | profile.pkl reader, event join, timing/resource rows | next |
| L2 | tinygrad PMC sampler | no | PMC decode and normalized counters | next after L1 |
| L3 | llama provider hook | no | graph/run bracket + kernel annotation export | scoped |
| L4 | HIP dispatch interposer | no | process-level dispatch timing/resource shim | optional fallback |

`rocprofv3` remains useful only as a calibration oracle while L1/L2/L3 are being validated. It should not be required
for normal tinygrad traces once L2 lands.

## MVP: Tinygrad HCQ/PMC Sampler

Add `boltbeam/collectors/tinygrad_hcq_pmc.py`.

Responsibilities:

- run the existing tinygrad prefill trace entrypoint with `PROFILE=1`, then `PROFILE=1 PMC=1` when counters are requested
- capture the generated `profile.pkl` path and copy it into the BoltBeam run artifact directory
- decode `ProfileRangeEvent`, `ProfileProgramEvent`, and, at L2, `ProfilePMCEvent`
- join events by program tag / execution tag / range name to produce kernel rows
- reuse current tinygrad kernel classifier for role/quant/shape
- normalize raw PMC names into BoltBeam counters
- write `boltbeam.hw_trace.v1`

Initial counter set:

| BoltBeam counter | first AMD source |
|---|---|
| `valu_busy_pct` | `SQ_INSTS_VALU` over active cycles |
| `lds_conflict_pct` | `SQC_LDS_BANK_CONFLICT` / `SQC_LDS_IDX_ACTIVE` |
| `l2_hit_pct` | `GL2C_HIT` / (`GL2C_HIT` + `GL2C_MISS`) |
| `memory_busy_pct` | proxy from L2 activity over active cycles until exact counter mapping is validated |
| `fetch_kb` | proxy from L2 hit/miss line counts once line size is verified |
| `mfma_util_pct` | needs MFMA instruction counter or SQTT decode; not MVP unless counter is present |

MVP command:

```bash
boltbeam collect-hw-trace \
  --provider tinygrad \
  --sampler tinygrad-profile-events \
  --run outputs/qwen14b \
  --model /path/model.gguf \
  --context 512
```

Counter mode:

```bash
boltbeam collect-hw-trace \
  --provider tinygrad \
  --sampler tinygrad-hcq-pmc \
  --run outputs/qwen14b \
  --model /path/model.gguf \
  --context 512 \
  --pmc-counters SQ_INSTS_VALU,GL2C_HIT,GL2C_MISS
```

### L1 Output Contract

L1 must produce useful timing without any backend profiler:

- `timing_trace.json`: `boltbeam.timing_trace.v1`
- `hw_trace.json`: `boltbeam.hw_trace.v1`
- `profile.pkl`: raw tinygrad profile event stream, retained as source evidence
- kernel rows with `kernel`, `kind`, `role`, `quant`, `shape`, `wall_us`, `calls`
- resource fields when present from program metadata; missing resource fields stay absent, never zeroed

L1 does not need PMCs to be useful for 14B prefill. It can already answer: launch lifecycle, hot packed GEMM ordering,
role attribution, elapsed time, tok/s, and bytes/tok/s from the inventory.

### L2 Output Contract

L2 adds counters:

- raw PMC blobs retained under `aux_sources`
- normalized `counters` dict on kernel rows
- formula metadata for derived counters, e.g. `counter_quality: "derived"`
- a `counter_summary` with matched rows, dropped rows, and unsupported counters

## Llama Path

llama.cpp cannot be observed through tinygrad HCQ. Independence requires owning a boundary inside or below llama:

1. **L3 provider hook, preferred.**
   Patch or plugin the local llama.cpp/ggml build to emit BoltBeam JSON rows around:
   - `ggml_backend_sched_graph_compute_async`
   - `ggml_backend_graph_compute_async`
   - backend kernel launch helpers where available

   This gives phase/tensor annotation and can preserve kernel names if the backend exposes them. It is less invasive
   than an interposer and better for role attribution.

2. **L4 HIP dispatch interposer, fallback.**
   Add a small `LD_PRELOAD` shim around HIP launch APIs such as `hipModuleLaunchKernel` / `hipLaunchKernel` for
   dispatch timing and geometry. This can be model-agnostic, but it has weaker semantic attribution because it sees
   kernel launches after llama has lost tensor-role context.

So the in-house low-level sampler starts with tinygrad. Llama remains importable through backend CSV only until L3 or
L4 lands. The long-term default should be:

```bash
boltbeam collect-hw-trace --provider llama --sampler llama-ggml-hook ...
```

not:

```bash
boltbeam collect-hw-trace --provider llama --backend-tool rocprofv3 ...
```

## Validation Gates

- Golden synthetic profile event fixture: one program, one range, one PMC blob -> one `hw_trace` kernel row.
- Real tinygrad 8B prefill smoke: token match, non-empty counter rows, timing agrees with existing wall trace within tolerance.
- 14B prefill run: identify top packed GEMM rows and emit normalized counters for each.
- Cross-check once against existing backend CSV: elapsed and hot-kernel ordering should match; counters may differ by definition but must be stable run-to-run.

## Risks

- PMC availability depends on target and stable profile mode; tinygrad currently errors if profile mode is not available.
- Counter formulas need validation per GPU generation; mark derived counters with `counter_source=derived`.
- Some counters are global over the sampled window, so attribution is exact only when the bracket contains one kernel.
- SQTT can be large and intrusive; keep it as an opt-in deep mode, not the default sampler.

## Work Plan

1. Add a `--sampler` option to `collect-hw-trace`.
   - default today: `backend-csv`
   - new tinygrad default after validation: `tinygrad-profile-events`
   - counter mode: `tinygrad-hcq-pmc`
2. Implement event decoding helpers in BoltBeam, not in docs/scripts:
   - `boltbeam/samplers/tinygrad_hcq_pmc.py`
   - `load_profile_events(profile_pkl) -> events`
   - `decode_profile_events(events) -> timing_trace rows`
   - `decode_pmc_events(events) -> counter rows`
3. Add tinygrad runner support to copy the generated `profile.pkl` into the requested trace directory.
4. Wire `collect-hw-trace --provider tinygrad --sampler tinygrad-profile-events` to emit both raw timing and HW traces.
5. Add normalized AMD PMC mapping and formula metadata.
6. Add `--sampler tinygrad-hcq-pmc` and validate counters.
7. Scope llama L3 hook against the local llama.cpp build; prefer ggml backend graph/launch boundaries over HIP
   interposition if kernel names and tensor roles can be retained.
8. Run 8B, then 14B. Only after rows are stable, make the in-house sampler the default for tinygrad.

## First Implementation Slice

Build L1 before counters:

1. Add `boltbeam/samplers/tinygrad_profile_events.py`.
2. Add a fixture-based unit test that decodes:
   - one `ProfileProgramEvent`
   - one AMD `ProfileRangeEvent`
   - one copy range
   into `boltbeam.timing_trace.v1` rows.
3. Add `--sampler tinygrad-profile-events` to `collect-hw-trace`.
4. Modify the tinygrad collector to run with `PROFILE=1` and locate/copy `profile.pkl`.
5. Emit `timing_trace.json`, `hw_trace.json`, and retain `profile.pkl`.
6. Run a smoke on 8B pp512, then 14B pp512.

Stop before L2 if L1 timing rows do not match the current BoltBeam profile rows within noise. Counter work depends on
the event join being correct.

## Done Definition

The external backend profiler is no longer required for tinygrad prefill:

```bash
boltbeam collect-hw-trace --provider tinygrad --sampler tinygrad-hcq-pmc ...
boltbeam ingest-timing hw_trace.json --run ...
boltbeam compare-hw-trace --baseline llama_hw_trace.json --candidate hw_trace.json
```

For 14B, the trace must show elapsed time, tok/s, per-kernel/role time, byte estimates, and normalized counters on the
hot packed GEMM rows that explain the roofline gap.
