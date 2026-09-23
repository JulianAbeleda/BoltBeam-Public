# 14B decode — roofline attribution & in-scope convergence (2026-07-02)

**Verdict: SUPERSEDED — the "converged" call was env-scoped and wrong.** It was true only within the
no-codegen envelope. When the codegen boundary was lifted (2026-07-02), pursuing v_dot2 showed the Q4_K GEMV
runs at only ~42–52% of the 960 GB/s HBM peak, so the 405 GB/s "route ceiling" is a codegen/compute ceiling,
not the bandwidth wall. v_dot2 was wired into the GEMV (`decode_q4k_gemv_vdot2`, +0.3–0.4 tok/s, dNLL +0.00013)
and the REAL bottleneck was identified as the dequant bit-unpack. See
`docs/14b-decode-vdot2-gemv-result-20260702.md`. The tables below still hold for the *fusion* levers; they were
never the roofline lever — the weight GEMV codegen is.

## The gap

| ctx | baseline tok/s | practical floor | gap |
|---|---|---|---|
| 512 | 55.3 | 66 | +19.3% |
| 2048 | 55.1 | 66 | +19.8% |

Decode is fully HBM-bound (host-sync 0%, W==D).

## Traffic breakdown per token (why the gap is one term)

| ctx | Q4_K weights read | KV cache read | **weight share** | KV share |
|---|---|---|---|---|
| 512 | ~8.0 GB | 84 MB | **99.0%** | 1.0% |
| 2048 | ~8.0 GB | 336 MB | **96.0%** | 4.0% |
| 4096 | ~8.0 GB | 671 MB | **92.3%** | 7.7% |

The Q4_K weight read is 92–99% of traffic and is already at its bandwidth route ceiling (398–405 GB/s;
G3 vs vector_load refresh in the firewall). A 19% speedup cannot come from the 1–8% that is everything else.

## Amdahl ceiling of every reachable lever (all far below 19%)

| lever | max share | status |
|---|---|---|
| aggregate elementwise fusion (silu+residual+rmsnorm+qk_norm) | 4.9% | REFUTED_NO_MOVEMENT (+0.2 tok/s) |
| KV-cache q8 (halve KV read) | 3.9% @ctx4096, 0.5% @ctx512 | EMITTER_BLOCKED `kv_quant_attention_dequant`; lossy, long-ctx only |
| fused reduce+scale (RMSNorm / reduce) | 2.8% | EMITTER_BLOCKED `fused_reduce_scale` |
| attention combine fusion | 13.6% | REFUTED (occupancy); reopen needs `cross_workgroup_atomic_float` |

## The only two levers that can close +19% — both walled

1. **Fewer weight bytes** (sub-4-bit Q3/Q2 weights) → `REFUTED_QUALITY` (dNLL fails for high-byte roles).
2. **Higher effective GEMV bandwidth** (v_dot2 dot-dequant / new codegen) → `EMITTER_BLOCKED: v_dot2`
   (`v_dot2_f32_f16` is in the target `dot_primitives`, but the renderer cannot lower to it).

Both are genuinely-new capability work (quality relaxation or renderer lowering), i.e. **stop-for-approval
frontiers**, not low-risk default-off route experiments.

## Co-improvement recorded this pass

- tinygrad → BoltBeam: implementation reality (`KV_CACHE_TYPE` / `DECODE_RMSNORM_REDUCE_FUSION` absent) and
  the measured traffic breakdown corrected BoltBeam's reachability, which had over-claimed 21 candidates as
  reachable. Now 7 are correctly `emitter-blocked` with named blockers.
- BoltBeam → tinygrad: the reachability report showed no in-scope candidate was worth a bench run, so no
  tinygrad cycles were spent on walled routes — the ledger steered the search away from dead ends.

Data artifact: `outputs/qwen3-14b-analysis/roofline_attribution.json` (local; `outputs/` is gitignored).
