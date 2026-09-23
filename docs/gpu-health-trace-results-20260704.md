# GPU Health / 14B Trace — Run Results (2026-07-04)

Counterpart to `gpu-health-trace-handoff-prompt-20260704.md`. Records what actually ran
when resuming the Qwen3 14B prefill parity work.

## 1. GPU health: CLEAN (no wedge)

Checked before, during, and after every run. None of the prior wedge symptoms recurred.

- `/dev/kfd` present; `/dev/dri/renderD128` + `card0` present.
- `rocminfo` sees `gfx1100` (RX 7900 XTX); `rocm-smi` sees GPU 0, idle at rest, 100% under load, returns to idle.
- No GPU-related `D`-state tasks. Only normal desktop holders of the render node (Xwayland/mutter/ibus).
- `assert_gpu_health('fail')` returned `ok=True, problems=[]` at preflight and postrun of both native-PMC runs.

The safety objective of the handoff is satisfied: no run turned into a reset cycle.

## 2. Guarded rocprofv3 path is a STRUCTURAL dead end on this stack

`boltbeam collect-hw-trace --provider tinygrad --sampler backend-csv --gpu-health fail`
(rocprofv3 at `/opt/rocm-7.2.4/bin/rocprofv3`, v1.1.0) ran the workload to completion but
produced **no kernel CSV** → collector raised "expected kernel stats not found". Not a
health issue, not a flag typo:

- This tinygrad build only has the `AMD` backend. `Device['HIP']`/`['GPU']` → `ModuleNotFoundError`.
- DEV=AMD submits via its own `/dev/kfd` ioctl queues, bypassing the ROCr HSA/HIP runtime that
  rocprofiler-sdk instruments. rocprofv3 sees zero dispatches — confirmed even on a trivial
  `Tensor@Tensor` op (exit 0, no output, no `-S` summary).

rocprofv3 remains valid for `--provider llama` (llama.cpp uses ROCr).

## 3. Native tinygrad PMC path works for timing, but NOT for the hot GEMM counters

`extra/qk/prefill_boltbeam_trace.py --hw-trace` (trace_source `tinygrad_internal_pmc`) is the
only counter mechanism compatible with DEV=AMD. It ran clean (health ok pre/post), but:

- `pmc_enabled = PROFILE>0 and PMC>0`, yet the `pmc_read` + `ProfilePMCEvent` append live inside
  eager `AMDProgram.__call__` (ops_amd.py ~L629). The prefill runs its GEMMs through
  `HCQGraph.__call__`, which submits the pre-baked queue and never calls per-kernel
  `AMDProgram.__call__` → no PMC event.
- `--mode full` (graph on): **zero** counters on all rows.
- `--mode profile PREFILL_GRAPH_GEMM=0` (eager attempt): counters on only 5 of 35 rows, all tiny
  `norm` kernels with a lone `lds_conflict_pct`. **Every GEMM role got zero counters** — the
  GEMMs stay JIT/graph-captured.

## Evidence quality for `ffn_gate_up Q4_K [512,17408,5120]`

Artifacts in `outputs/tinygrad-prefill-qwen14b-hwtrace-pp512/`:
`run/hw_trace_native_pmc.json` (graph, full), `run/hw_trace_pmc_eager.json` (eager, profile),
`run/role_ffn_gate_up.{json,md}`.

| Signal | Status |
|---|---|
| role identity / hot-role confirmation | PRESENT — ffn_gate_up = 40.5% of step; kernel `prefill_q4k_direct_packed_load_direct_out_gemm_17408_5120_512_1`, 80 calls |
| elapsed / wall time | PRESENT — ~3.49s aggregate (graph), 858ms kernel-sum (eager profile) |
| tok/s (whole step) | PRESENT — 178 tok/s pp512 (synced min-of-bursts) |
| bytes / phys_bytes | PRESENT (weight-inventory derived, ~4.01 GB for the role) |
| occupancy / memory_busy / valu_busy / mfma_util / l2_hit | **MISSING** — PMC not captured for GEMM (see §3) |
| VGPR/SGPR/LDS/scratch (resources) | MISSING on GEMM in these runs |

## Conclusion / next step

The hot role is confirmed by timing, but the per-role hardware signals that would prove the
dequant-per-token vs tiled-reuse hypothesis are **not obtainable** with the current tooling.
Per the handoff, do NOT implement the tiled-GEMM change on the hypothesis alone. The blocker to
clear first is instrumentation, not health or clocks:

- add a PMC read into the `HCQGraph` submit path (ops_amd.py / runtime/graph/hcq.py) so
  graph-captured GEMMs emit `ProfilePMCEvent`, **or**
- run the isolated Q4_K GEMM through a truly eager (non-JIT) micro-harness to read occupancy /
  VALU / memory-busy for that one kernel.

Until one of those lands, the only trustworthy 14B prefill evidence is timing + role-share; do
not report measured occupancy/bandwidth for the GEMM.
