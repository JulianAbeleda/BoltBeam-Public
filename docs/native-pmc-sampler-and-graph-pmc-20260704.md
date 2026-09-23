# Native-PMC sampler + graph-PMC attempt (2026-07-04)

Follow-up to `hwtrace-diagnostics-feature-scope-20260704.md`. Two decisions: promote the tinygrad
native-PMC path to a first-class sampler (and make it the tinygrad default), and try to make
graph-captured GEMMs emit PMC counters.

## Shipped & verified: `tinygrad-native-pmc` sampler (our own, no rocprofv3)

- New collector `boltbeam/collectors/tinygrad_native_pmc.py` runs `extra/qk/prefill_boltbeam_trace.py
  --mode full --hw-trace` (reads AMD counters through tinygrad's own KFD interface) and returns
  `boltbeam.hw_trace.v1` directly. `timing_trace_to_hw_trace` already accepts hw-trace input, so
  counters pass through.
- `collect-hw-trace` now resolves the default sampler per provider (`_default_sampler`):
  **tinygrad → tinygrad-native-pmc**, llama → backend-csv. rocprofv3 is no longer on the tinygrad
  default path. `--sampler` choices updated in both CLI entrypoints.
- Fixes found along the way: the collector must set `PROFILE=1` (the shared tinygrad env pins it to
  0, which disables `pmc_enabled`); and `--out` must be absolute (the subprocess runs with
  `cwd=tinygrad_root`, so a relative path wrote the trace under the tinygrad tree).
- Verified end-to-end under `--gpu-health fail`: `collect-hw-trace --provider tinygrad` writes a
  valid trace (`provider_id: tinygrad/native-pmc`), pre/post health OK, 35 kernel rows. Feature B
  labels it `pmc_graph_bypass` (counters still missing on GEMMs — see below).
- Tests: `tests/test_collect_hw_trace_preflight.py` (default→native-pmc, capture unit test). Suite green.

## Built but PARTIAL: graph-PMC (`PMC_GRAPH`, default OFF)

`tinygrad/runtime/graph/hcq.py`: allocate a per-program PMC slot buffer, insert `pmc_read` after each
`exec`, and `collect_pmc` emits `ProfilePMCEvent` keyed by `runtime.prof_prg_counter` (mirrors
`collect_timestamps`; fires at the next `__call__`/`__del__`, which the trace script's two-pass
profile triggers).

**Verified working mechanism:** per-kernel capture lands nonzero in all 32 slots, attribution is
correct (kern=12 = the `ffn_gate_up` GEMM, `SQ_BUSY_CYCLES`=257M).

**Why it's not enough yet:** only `SQ_BUSY_CYCLES` (perfcounter index 0) accumulates inline. The
other counters — `GRBM_GUI_ACTIVE`, `SQ_INSTS_VALU`, `GL2C_*`, `SQC_LDS_*` — read **zero** in the
graph path, though the eager `AMDProgram.__call__` path reads them fine. `_normalize_pmc_stats` needs
`GRBM_GUI_ACTIVE` (plus VALU/vmem) to derive occupancy/VALU/memory-busy, so no derived metric is
produced and rows stay empty. Likely a sampling-latch/fence or SELECT-register-persistence
difference between the eager standalone `pmc_read` submit and the inline graph read; only index-0
latches inline.

**Gating:** `PMC_GRAPH=0` by default (opt-in) until the multi-block inline-read issue is solved, so
normal native-PMC runs pay no extra per-kernel `pmc_read` overhead and carry no added wedge risk.
Non-PMC graph execution and the full BoltBeam suite are unaffected (verified).

## Net state for the hot GEMM
Per-kernel `SQ_BUSY_CYCLES` for `ffn_gate_up` is now obtainable (new signal we lacked), but
occupancy/VALU/memory-busy/MFMA are still unavailable. Next: fix multi-block inline latching in
`pmc_read`/graph, or measure the isolated GEMM through a truly eager micro-harness. GPU health
stayed clean across every run.
