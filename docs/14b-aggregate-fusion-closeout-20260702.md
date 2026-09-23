# 14B decode — aggregate elementwise fusion result + route-search closeout (2026-07-02)

**Candidate:** `decode_aggregate_elementwise_fusion` (`DECODE_AGG_SYSTEM_FUSION=1`, default-OFF)
**Verdict:** `REFUTED_NO_MOVEMENT`
**Route-class verdict:** `CLOSEOUT_ROUTE_SEARCH_PRIMITIVE_MISSING`

This was the sanctioned reopen of `decode_silu_gate_fusion` (its refute record asked for an aggregate
multi-group pass) and the SELECTED Frontier A from `recommendation_14b_20260702.md`. The recommendation
predicted this exact outcome ("the SF3-implied expectation") and pre-authorized the closeout flip.

## What was tried (smallest default-off change)

One flag removing two `.contiguous()` barriers on the decode path: the silu-gate intermediate
(`silu(gate) * up`) and the block-output residual add (`h + ffn(ffn_norm(h))`). No new ops, dtypes, or
machinery. No HIP/ASM/atomics/renderer/GEMV/attention-primitive changes. `E_49152_32_3` untouched.

## Gates

| gate | result |
|---|---|
| host health | clean — 0 D-state workers, GPU idle (27 MB used), gfx1100, smoke PASS |
| default-off purity | byte-identical (flag-off path unchanged by construction; baseline reproduced 55.3/55.1) |
| correctness | **PASS** — 48/48 token-identical, real flash `m.generate` temp=0 |
| route-bound | **fired, but only silu fused** — total kernels 872→832, `E_*` 264→224; `E_136_32_4n1 ×40` (silu, 17408) removed; `E_40_32_4` family ×40 (residual, 5120) **unchanged**; `r_*` unchanged at 227 |
| W==D speed | ctx512 55.3→55.5, ctx2048 55.1→55.3 → **+0.2 tok/s, below 0.5 noise threshold** |
| protected routes | Q4_K G3, flash block-tiled attention fire; no owned/gqa fallback |

## Roofline (practical floor 66 tok/s)

| | baseline | candidate |
|---|---|---|
| tok/s (ctx512) | 55.3 | 55.5 |
| P | 83.8% | 84.1% |
| G | 10.7 | 10.5 |
| A (vs 98%) | 14.2pp | 13.9pp |

Movement +0.3pp — below noise.

## Why it closes out the route class

Of the four target groups, only `silu_gate` and `residual_adds` are reachable by barrier removal:

- **silu_gate** fuses but is sub-noise — already `refuted_no_movement` individually (SF3, +0.4 tok/s);
  unchanged when aggregated.
- **residual_adds** cannot fuse via barrier removal: the block output fans out into the next block's
  `attn_norm` (sum-of-squares reduce + scale), which forces a realize regardless of the `.contiguous()`.
  Route-bound proves zero kernel change.
- **rmsnorm_scale / qk_norm_scale** need a dedicated fused-LOAD UOp kernel (`decode_rmsnorm_reduce_fusion`:
  "not a scheduler tweak") — new machinery, out of the low-risk scope.

Decode is fully GPU-bound (host-sync 0%) and JIT already batches the forward into ~6 program launches, so
removing eager elementwise launches does not reduce wall time. No materially-different low-risk subvariant
remains inside the safety boundaries.

## Named missing capabilities (per the recommendation doc)

Further 14B decode gains require a genuinely-new primitive, not a route experiment:

1. cross-workgroup LSE atomics / grid-sync (the ~13.6% attention-combine bucket)
2. `v_dot2` renderer lowering
3. vec-store-to-REG accumulator widening (fused-load RMSNorm)

## Disposition

tinygrad `model.py` change **reverted** (the flag was redundant with the pre-existing
`DECODE_FUSE_SILU_GATE` — its only firing part — plus an inert residual branch). Negative evidence retained
in `bench/system-fusion-sf4-aggregate/latest.json` (tinygrad repo) and the ledger candidate.
