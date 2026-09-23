# Qwen3 8B current dual roofline

Date: 2026-07-13. Target: RX 7900 XTX / `amd_gfx1100`. Model:
`Qwen3-8B-Q4_K_M.gguf` (5,027,783,488 bytes).

This report keeps two ceilings separate:

- **Raw hardware roofline:** the impossible-to-sustain-everywhere specification ceiling.
- **Practical roofline:** a measured same-machine substrate or same-route ceiling. This is the optimization target.

PROFILE capture time is used only for kernel attribution. Throughput placement always uses the clean tinygrad authority
run, because profiling changes wall time.

## Current placement

| workload | authority | raw hardware roofline | raw placement | practical roofline | practical placement |
|---|---:|---:|---:|---:|---:|
| prefill pp512 | 3,881 tok/s | 8,650 tok/s | 44.9% | 4,451 tok/s | 87.2% |
| decode ctx512 | 117.1 tok/s | 190.9 tok/s | 61.3% | 164.9 tok/s | 71.0% |

## Prefill basis

- Work per pp512 step: 7.268 TFLOP (standard two-FLOP-per-MAC convention), including transformer linears,
  attention score/value work, and the final-token LM head.
- Raw ceiling: `7.268 TFLOP / 122.8 TFLOP/s = 59.19 ms`, or 8,650 tok/s.
- Practical ceiling: rescale the current PROFILE kernel shares to the clean 131.925 ms authority wall, hold non-GEMM
  time fixed, and lift generated `ffn_down`, `attn_qo`, and `attn_kv` to the current generated `ffn_gate_up` oracle
  rate. This gives 115.04 ms, or 4,451 tok/s.
- Interpretation: the generated route is 87.2% of its current practical route-family ceiling. Remaining headroom is
  concentrated in the narrower generated GEMM geometries, not a missing handwritten lane.

Route-aware cache accounting uses the admitted `tile_m=128`, so pp512 creates four global weight-fetch groups rather
than 512 full-weight rereads. Promoted fp16 per-layer working sets are 96 MiB for FFN gate/up and down (the soft
MALL/DRAM boundary), 32 MiB for attention Q/O, and 8 MiB for attention KV. All four kernels explicitly allocate LDS;
cache tier describes the global-load path feeding LDS, not an alternate route. The corrected tiered raw model produces
8,759 tok/s, consistent with the independent 8,650 tok/s compute roof; the retired model's 719 tok/s was invalid.

## Decode basis

- Raw ceiling: `960 GB/s / 5,027,783,488 bytes = 190.9 tok/s`.
- Practical ceiling: `829 GB/s / 5,027,783,488 bytes = 164.9 tok/s`, using the measured sustained-copy ceiling for
  this gfx1100 regime rather than a cold burst or the raw specification.
- Interpretation: decode is 71.0% of the sustained practical bandwidth roofline. The trace identifies Q4_K/Q6_K
  GEMV as the dominant substrate; exact remaining attribution still needs live memory/occupancy counters.

Decode uses packed model-storage bytes and one global fetch group per token. Individual matrices may fit L2 or MALL,
but their first and only pass is a compulsory DRAM stream, so fit is not reported as a cache hit or speedup.

## Reproduction artifacts

- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/prefill/roofline_input.json`
- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/prefill/profiler_report.json`
- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/decode/roofline_input.json`
- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/decode/profiler_report.json`
- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/execution_profile.json`
- `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/tiered_roofline.json`

The profiler report validates `model_id` and `workload` before accepting a roofline input, preventing a stale model's
ceiling from being attached to the current trace.
