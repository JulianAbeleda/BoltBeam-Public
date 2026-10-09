# Qwen3-14B generated-prefill parity plan

Date: 2026-07-14

## Decision

The current Arkey 14B prefill route is not missing parity because of HBM bandwidth, launch count, attention, or general
framework overhead. It is missing parity because all six packed prefill GEMMs use a scalar, direct-packed dequant-dot
kernel instead of a matrix-core-quality tiled contraction. Those six kernels consume 96.4% of the profile and account
for about 98.5% of the absolute Arkey-versus-llama wall-time gap.

The shortest route to parity is:

1. Bind and finish the existing scheduler-owned generated Q4_K/Q8_1 WMMA route for every 14B Q4 role.
2. Replace Q6_K's inline scalar dequant-dot with dequant-once plus generated fp16 WMMA, retaining a fused
   dequant-to-WMMA candidate when memory pressure requires it.
3. Integrate both behind one typed candidate authority keyed by quant, role, shape, target, and memory budget.
4. Touch norm/elementwise residuals only if the two packed routes do not independently clear the whole-model gate.

The measured parity target is achievable without assuming a new hardware primitive. Q4 at 60 TFLOP/s aggregate and
Q6 at 70 TFLOP/s aggregate projects to 1,891 tok/s with Arkey's current non-GEMM residual. Q4 at the already observed
generated-WMMA class of 66 TFLOP/s and Q6 at 70 TFLOP/s projects to 2,022 tok/s.

## Measurement authority

Both traces are sequential runs of the same workload and device:

- model: Qwen3-14B Q4_K_M
- device: AMD gfx1100 / RX 7900 XTX
- workload: 512-token prefill
- llama.cpp commit: `ac4cddeb0`
- Arkey commit: `05b67146a`
- Arkey route: `prefill_v2_scheduler_matmul_default` plus `prefill_q4k_direct_tile4x4_default`

| implementation | clean wall | prefill tok/s | evidence |
|---|---:|---:|---|
| llama.cpp | 271.230 ms | 1,889.41 | five llama-bench samples, rocprof trace normalized to clean wall |
| Arkey generated | 1,397.840 ms | 366.28 | three synchronized samples; 0.15% spread |

The Arkey profile itself is 1,446.366 ms / 353.99 tok/s. Role times below are scaled by 0.96645 to its clean synchronized
wall. This preserves measured profile shares while avoiding profiler overhead in the parity projection.

The llama Q4 role split is recovered from dispatch geometry in the raw kernel trace:

- grid X 4352 selects N=17408 gate/up;
- grid X 1280 selects N=5120; the expected 160 short calls are attention q/o and 40 long calls are FFN down;
- grid X 256 selects N=1024 attention k/v.

The common normalization factor is derived from the aggregate `mul_mat_q` row. Q6 roles are identified by exact
dequant grids (`N*K/4`) and their paired Tensile dispatches. This role split is an inference from dispatch geometry and
model inventory, while aggregate llama and Arkey timings are directly measured.

## Measured loss stack

| role | quant | Arkey clean | llama | gap | Arkey TFLOP/s | llama TFLOP/s | absolute gap |
|---|---|---:|---:|---:|---:|---:|---:|
| ffn_gate_up | Q4_K | 590.039 ms | 121.887 ms | 4.84x | 12.37 | 59.90 | 468.152 ms |
| ffn_down | Q6_K | 355.552 ms | 25.999 ms | 13.68x | 5.13 | 70.21 | 329.553 ms |
| attn_qo | Q4_K | 188.182 ms | 42.909 ms | 4.39x | 11.41 | 50.05 | 145.273 ms |
| ffn_down | Q4_K | 155.187 ms | 35.130 ms | 4.42x | 11.76 | 51.96 | 120.057 ms |
| attn_kv | Q4_K | 40.843 ms | 9.950 ms | 4.10x | 7.89 | 32.38 | 30.893 ms |
| attn_kv | Q6_K | 18.113 ms | 1.998 ms | 9.07x | 5.93 | 53.75 | 16.116 ms |
| all other work | mixed | 49.924 ms | 33.358 ms | 1.50x | — | — | 16.566 ms |

Aggregate packed performance:

| family | logical work | Arkey time | Arkey rate | llama time | llama rate |
|---|---:|---:|---:|---:|---:|
| Q4_K | 11.596 TFLOP | 974.251 ms | 11.90 TFLOP/s | 209.875 ms | 55.25 TFLOP/s |
| Q6_K | 1.933 TFLOP | 373.666 ms | 5.17 TFLOP/s | 27.997 ms | 69.03 TFLOP/s |

Absolute clean-wall gap is 1,126.610 ms. The six packed rows contribute about 1,110 ms of it. Launch count is not the
cause: Arkey launches fewer kernels than llama (1,329 versus 2,327 captured dispatches) and is still 5.16x slower.

## Roofline diagnosis

The raw source-byte HBM floor is non-binding. The packed rows have arithmetic intensity above 1,000 logical
FLOP/source-byte while reaching only 4.3-6.6 effective source GB/s. A 960 GB/s HBM ceiling therefore describes an
impossible source-byte upper bound, not the active bottleneck.

For the current scalar route, use 61.4 TFLOP/s as the gfx1100 FP32-FMA ceiling. The corrected aggregate-aware roofline
multiplies shape work by each row's call count:

| role | quant | current | scalar-FMA floor | gap to floor | llama practical |
|---|---|---:|---:|---:|---:|
| ffn_gate_up | Q4_K | 610.522 ms profile | 118.916 ms | 5.13x | 132.234 ms |
| attn_qo | Q4_K | 194.714 ms | 34.975 ms | 5.57x | 38.892 ms |
| ffn_down | Q4_K | 160.574 ms | 29.729 ms | 5.40x | 24.693 ms |
| attn_kv | Q4_K | 42.261 ms | 5.246 ms | 8.06x | 6.006 ms |
| ffn_down | Q6_K | 367.895 ms | 29.729 ms | 12.37x | about 26.0 ms including dequant |
| attn_kv | Q6_K | 18.742 ms | 1.749 ms | 10.72x | about 2.0 ms including dequant |

The practical llama rung is much more useful than raw HBM. Its Q4 rate is 4.1-4.8x above Arkey by role, and its Q6
dequant-plus-WMMA route is 9.1-13.7x faster.

## Why the current generated route is slow

This is an algorithm/lowering mismatch, not merely a bad tile parameter:

1. `q4k_gen_prefill_direct_out_*` and `q6k_gen_prefill_direct_out_*` are scalar fp32 dequant-dot kernels. Their source
   expands each quantized value, casts to fp32, multiplies by fp32 activation, and accumulates through VALU. They do not
   express a WMMA contraction.
2. Q4 only reuses a decoded weight across a 4-token upcast. At M=512, the same weight decode is therefore repeated over
   many token tiles instead of being amortized by a large matrix tile.
3. Q6 follows the same scalar pattern and is worse: 97 VGPR, no LDS, only 5.2 aggregate TFLOP/s. llama instead pays a
   small dequant-once cost and sends fp16 matrices through Tensile/WMMA at about 69 aggregate TFLOP/s.
4. Q4 uses 153 VGPR, zero LDS, and a 256-thread 4x4 tile. The high register footprint is a credible occupancy limiter,
   but it is supporting static evidence, not a proven dynamic cause because gfx11 occupancy/VALU/cache counters are
   absent from this trace.
5. llama's Q8_1 activation quantization costs only about 2.94 ms, or 1.1% of its step. Avoiding that small prerequisite
   does not justify losing roughly 765 ms on Q4 contraction quality.
6. The repo already contains the correct architectural direction: a scheduler-owned Q4_K/Q8_1 generated WMMA route.
   Historical same-session evidence reports 359 -> 808 tok/s and about 66 TFLOP/s per Q4 WMMA kernel. The current
   default profile did not select it; it selected the scalar direct-packed fallback.

Dynamic L0/L1/L2/Infinity-Cache residency remains unknown. It is not required to establish the first-order diagnosis:
the measured arithmetic intensity, low achieved compute, no-LDS static path, source lowering, and 4-14x same-role
reference gaps already refute HBM and launch count as the dominant explanations.

## Projected outcomes

These projections preserve measured logical work and use the current 49.924 ms Arkey residual unless stated otherwise.

| state | Q4 target | Q6 target | residual | projected wall | projected tok/s |
|---|---:|---:|---:|---:|---:|
| current | 11.90 TFLOP/s | 5.17 TFLOP/s | 49.924 ms | 1,397.840 ms | 366.28 |
| Q4 WMMA only | 66 TFLOP/s | current | 49.924 ms | about 599 ms | about 855 |
| Q6 WMMA only | current | 70 TFLOP/s | 49.924 ms | about 1,052 ms | about 487 |
| parity gate | 60 TFLOP/s | 70 TFLOP/s | 49.924 ms | 270.808 ms | 1,890.64 |
| beyond-parity gate | 66 TFLOP/s | 70 TFLOP/s | 49.924 ms | 253.238 ms | 2,021.82 |
| optimized residual | 66 TFLOP/s | 70 TFLOP/s | 33.358 ms | 236.672 ms | 2,163.34 |

The historical 808 tok/s Q4 result is consistent with the Q4-only projection. It is evidence that the first milestone
is real, while also proving that Q4 alone cannot reach whole-model parity. Q6 must move to a matrix-core route too.

## Implementation plan

### P0 — Keep the evidence gate honest

- Preserve the clean 366.28 and 1,889.41 tok/s artifacts as the fixed comparator pair.
- Keep correctness before timing, isolated GPU execution, post-run health checks, route identity, and distinct-binary
  proof mandatory.
- Use profile attribution for role shares and synchronized clean wall for promotion numbers.
- Do not block timing decisions on unavailable gfx11 PMC counters. Mark dynamic cache/occupancy attribution unknown;
  use static ISA/resources plus controlled schedule changes to establish causality.

Exit gate: every candidate result names quant, role, M/N/K, target, route id, binary hash, correctness status, resources,
kernel time, and whole-model time where applicable.

### P1 — Recover Q4 generated WMMA for all four role shapes

Use the existing `prefill_q4k_int8_wmma_tiled_research` / `PREFILL_Q4K_Q8=wmma_tiled` scheduler-owned route. Do not
write another scalar direct-packed variant.

Sequence:

1. Compile each 14B Q4 role in isolation and prove the emitted kernel contains the intended integer WMMA contraction.
2. Prove the forbidden full `[groups,M,N]` RAW tensor is not materialized and Q8 packing remains a bounded prerequisite.
3. Run isolated full-output correctness for `attn_kv`, `attn_qo`, `ffn_down`, and `ffn_gate_up`.
4. Benchmark each role against both scalar direct-packed and the dispatch-derived llama role target.
5. Search tile/lifecycle parameters only inside the typed schedule authority. Optimize WMMA issue density, correction
   reduction ownership, decoded-weight reuse, register pressure, and writeback; do not add role-local environment logic.
6. Bind Q4 roles together and run whole-prefill only after every role clears its kernel gate.

First-pass role ceilings:

- gate/up <= 132 ms aggregate, target <= 122 ms;
- attention q/o <= 43 ms aggregate;
- FFN down <= 35 ms aggregate;
- attention k/v <= 10 ms aggregate;
- Q8 quantization <= 4 ms whole step;
- Q4 aggregate >= 60 TFLOP/s for parity, target >= 66 TFLOP/s for beyond parity.

Fail closed if the route falls back to scalar direct-packed, emits no WMMA, materializes global RAW, spills, fails
correctness, or cannot compile within the isolated timeout.

### P2 — Move Q6 from scalar inline dequant to WMMA

Evaluate two generated candidates behind the same authority:

1. **Dequant once -> fp16 Tensor GEMM.** This mirrors llama's measured route and reuses tinygrad's ordinary generated
   GEMM/WMMA machinery. It is the shortest path. A full resident Q6 overlay is about 5.3 GB; model plus overlay appears
   feasible on a 24 GB card but must pass admission using measured free-memory headroom.
2. **Fused Q6 dequant -> WMMA tile.** Use when resident or ephemeral fp16 materialization exceeds the memory budget.
   Decode each packed weight once per matrix tile, stage/register it, and issue WMMA across the M tile.

Keep scalar direct-packed as rollback, not as an eligible winner for pp512.

Gates:

- FFN down dequant plus GEMM <= 26.0 ms aggregate across 20 calls;
- attention k/v dequant plus GEMM <= 2.0 ms aggregate across 20 calls;
- Q6 aggregate >= 69 TFLOP/s, target >= 70 TFLOP/s;
- no OOM, no stale captured overlay, no unbounded graph materialization, and full-output correctness.

If full Q6 residency fails admission, try per-role residency, ephemeral per-matrix dequant, then fused tile lowering—in
that order. Do not revive the removed chunked stale-state path.

### P3 — Centralized mixed-route integration

The execution authority should select a typed candidate from:

```text
(quant, role, M, N, K, target, memory_budget, correctness_class)
```

It should not encode `14B`, `8B`, Q4, Q6, LDS, or direct-L2 policy in scattered environment branches. Candidate
descriptors own schedule requirements; admission owns memory and target checks; the route manifest owns provenance;
the whole-model harness only requests a candidate and records what actually bound.

Required mixed route:

- Q4 roles -> scheduler-owned Q4/Q8 WMMA;
- Q6 roles -> admitted dequant-once fp16 WMMA or fused Q6 WMMA;
- all other roles -> current generated scheduler until measured otherwise.

Run three sequential sessions per implementation, alternate ordering, use the same clock policy, and report median plus
spread. Require whole-model quality/token parity and exact route census.

Promotion gates:

- parity: candidate median >= 0.98 * llama median and no statistically credible regression;
- beyond parity: candidate median >= 1.05 * llama median (>= 1,984 tok/s against this baseline);
- aspirational generated target: >= 2,000 tok/s / <= 256 ms;
- no role may silently fall back to an oracle or handwritten kernel.

### P4 — Residual optimization only if needed

At Q4=60 and Q6=70 TFLOP/s, the current residual already permits parity. Therefore norm/elementwise work is not on the
critical path until both packed families are near target.

If integration lands below 1,889 tok/s after packed targets pass:

1. regenerate the profile and verify no route fallback;
2. reduce Arkey's 49.924 ms non-packed residual toward llama's 33.358 ms;
3. prioritize the 23.4 ms norm and 28.2 ms elementwise aggregates by launch-weighted time;
4. fuse only producer/consumer pairs with a correctness proof and measured whole-model translation.

Do not trade matrix throughput for fusion. A 5% packed regression costs more than eliminating the entire residual gap.

## Experiment order and stop rules

Run in this order:

1. Q4 role-level scheduler-owned WMMA compile/correctness/ISA gate.
2. Q4 role timings; stop broad Q4 work only when aggregate reaches 60 TFLOP/s or a precise compiler blocker is proven.
3. Q6 dequant-once fp16 admission and role timing.
4. Q6 fused tile only if dequant-once is memory- or lifecycle-blocked.
5. Mixed Q4+Q6 whole-model correctness, then timing.
6. Residual work only if packed targets pass but whole parity does not.

Stop and record a compiler-level blocker when one of these repeats after controlled variants:

- the schedule cannot retain packed decode inside the contraction without global RAW materialization;
- WMMA selection cannot coexist with correction reduction ownership;
- legal lifecycle ordering forces spills or destroys WMMA issue density;
- Q6 fp16 admission cannot fit and fused tile lowering cannot express bounded reuse;
- correctness requires an oracle/reference wrapper.

Do not stop merely because gfx11 hardware counters are unavailable. The decisive gates are correctness, emitted
structure, per-role throughput, and whole-model translation.

## Artifacts

Generated by this investigation under `outputs/qwen14b-prefill-parity-20260714/`:

- `hw_compare.json` / `hw_compare.md`
- `substrate_compare.json` / `substrate_compare.md`
- `prefill_roofline.json` / `prefill_roofline.md`
- `roofline_ladder.json` / `roofline_ladder.md`
- `arkey_profiler_report.json` / `arkey_profiler_report.md`
- `llama_profiler_report.json` / `llama_profiler_report.md`

Primary inputs:

- `outputs/arkey-prefill-qwen14b-pp512-20260714-fresh/hw_trace.json`
- `outputs/arkey-prefill-qwen14b-pp512-20260714-fresh/authority-unprofiled.json`
- `outputs/llama-prefill-qwen14b-pp512-20260714-fresh/hw_trace.json`
- `outputs/llama-prefill-qwen14b-pp512-20260714-fresh/rocprof/qwen14b_pp512_kernel_trace.csv`
- `outputs/llama-prefill-qwen14b-pp512/run/weight_inventory.json`
