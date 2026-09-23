# 14B/32B Decode Loss Stack + Reachability (post-L3, post-attention-closure)

Date: 2026-07-01. Re-baseline after the L3 Q6_K ffn_down win and the attention-combine closure. Goal: find
the largest remaining REACHABLE non-attention decode gap from the measured loss stack, not from intuition.

## Re-baseline (current defaults, gfx1100, synced W==D)

| model | ctx128 | ctx512 | ctx2048 | ctx4096 |
|-------|--------|--------|---------|---------|
| 14B   | 52.3   | 50.1   | 47.1    | 43.4    |
| 32B   | 26.9   | 24.9   | —       | —       |

llama (external sanity only, not authority): 14B parity line ~66 tok/s. 14B is ~76% of llama after L3.

## Loss stack (14B ctx512, role attribution + reduce-source trace, 100% of reduce resolved)

| bucket | % decode | route | notes |
|--------|----------|-------|-------|
| **Q4_K GEMV** (ffn_gate_up 26.9, attn_qo 8.6, ffn_down-Q4K 6.2, attn_k 1.7) | **43.3%** | generated_g3 | ffn_gate_up is the single biggest kernel |
| **attention** (combine reduce 13.6, flash partial/score ~8, elementwise E_49152 6.7) | **~27%** | flash gqa_coop_vec | combine is CLOSED (refuted) |
| **Q6_K GEMV** (ffn_down 9.2, lm_head 3.5, attn_v 2.4) | **~15%** | coop_partial (L3) | |
| **RMSNorm / reduces** (coop_partial_combine 4.1, rmsnorm 1.3) | **~5.4%** | fallback | |
| **lm_head** | **3.5%** | Q6_K coop | 706 GB/s (fine) |
| **launch/activation/other** (silu/gate/rope/residual elementwise, launch fragmentation) | **~12%** | fallback_graph | system-level |

Unknown bucket after resolution: reduce = 0% unknown (100% resolved: attention_combine 13.6, coop_combine 4.1,
rmsnorm 1.3, sampling 0.03); "other" resolves to attention elementwise + activation. Under ceiling → selection allowed.

## Reachability classification

| bucket / lever | class | evidence |
|----------------|-------|----------|
| Q4_K GEMV route (vector_load / topology / split-K) | **REFUTED_BY_LEDGER (at route ceiling)** | FRESH A/B: vector_load 398 GB/s ≈ G3 ~405 on 14B ffn_gate; G3=owned-warp (bandwidth-ceiling scope 2026-06-29); topology + split-K refuted earlier. ~42% of 960 peak is the Q4_K dequant-GEMV access-pattern ceiling |
| Attention combine | **REFUTED_BY_LEDGER** | Hq-only fused (-88%), merged (-16%), FLASH_L, wholecache — all refuted; do_not_retry |
| Attention partial (flash score/PV) | **LOW_AMDAHL / attention-scoped** | ~8%; the flash compute, not a combine; not this phase's non-attention target |
| Q6_K GEMV (ffn_down/lm_head/attn_v) | **REFUTED_BY_LEDGER / LOW_AMDAHL** | ffn_down L3-shipped (coop ~51% peak); warp in-kernel combine measured 1.09x (not worth); attn_v 2.4% low Amdahl |
| coop_partial_combine reduce (4.1%) | **REFUTED_BY_LEDGER** | the Q6_K coop external .sum; warp-fusion 1.09x (not worth), same class as above |
| RMSNorm fusion (1.3%) | **LOW_AMDAHL** | L1C deferred; needs a fused-load kernel for ~2% |
| launch/graph fragmentation (~12%) | **LOW_AMDAHL (per-piece) / scheduler frontier** | many small elementwise kernels; broad scheduler fusion, not a targeted generated route |
| v_dot2 / cross-lane compute primitive | **EMITTER_BLOCKED** | renderer cannot lower v_dot2; AND the dominant buckets are HBM-bound, so compute primitives have low expected W==D value |
| cross-workgroup LSE (atomics / grid-sync) | **PRIMITIVE_MISSING** | no atomic/grid-sync UOp on AMD |
| nvidia_sm89 / apple_metal attention combine | **TARGET_BACKEND_INCOMPLETE** | descriptor-only; native coop-groups could reopen there |

## Selection + outcome

**No Tier-B+ reachable NON-attention route remains.** Every major bucket is at its best-known generated route,
closed by the attention ledger, or LOW_AMDAHL:
- the dominant Q4_K GEMV (43%) is HBM-bound at ~400 GB/s — confirmed with a FRESH vector_load A/B (398 GB/s,
  no gain over G3), not just cited from the June 29 scope;
- Q6_K GEMV is L3-shipped and its in-kernel combine measured only 1.09x;
- the residual is system-level launch fragmentation + activation overhead (LOW_AMDAHL per-piece).

Decision: **no promotion this phase.** This is the ledger doing its job — preventing another loop through
resolved routes. The reachable route-search space for 14B/32B decode is exhausted at the route level.

## What would reopen a win (precise)

1. **System fusion (LOW_AMDAHL → could aggregate):** a generated/scheduler capability that fuses the ~12%
   launch-fragmentation + activation elementwise into far fewer kernels, raising effective bandwidth toward the
   streaming ceiling. Reachable in principle; per-piece low, so only worth it as an aggregate scheduler pass.
2. **New compute primitive:** v_dot2 renderer lowering (EMITTER_BLOCKED) — but only reopens value if a bucket
   is compute-bound, which the dominant Q4_K/Q6_K GEMVs are not (HBM-bound).
3. **New coordination primitive:** atomic-float or grid-sync UOp (PRIMITIVE_MISSING) — reopens the attention
   combine per its recorded split-preserving reopen condition.
4. **Target backend:** nvidia_sm89 / apple_metal become `complete` (native cooperative groups/atomics).

BoltBeam candidate recorded: `decode_q4k_gemv_vector_load` (refuted, at route ceiling, fresh A/B).
