# 14B Decode Loss Stack — Refreshed Under Current Promoted Defaults

Date: 2026-07-02. Re-measured after the G5 K-only attention promotion, to update attribution that predated it.

## Provenance / authorities

- tinygrad commit `4b406bdbc` (dirty: 4 unrelated bench/doc files); BoltBeam commit `b0832e5` (dirty: 9, roofline WIP).
- Model `qwen3-14b` / `Qwen3-14B-Q4_K_M.gguf`; target `amd_gfx1100`.
- Contexts: **512, 2048** (ctx4096 NOT run — safety rule + shares stable).
- Command: `DEV=AMD PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py --model .../Qwen3-14B-Q4_K_M.gguf --id qwen3-14b --capture --ctxs <512|2048>` under **current defaults** (G5 K-only, Q6_K L3, Q4_K G3 all default-on).
- **Attribution authority:** PROFILE=1 per-kernel GPU-compute share (attribution only — *not* a wall W==D promotion authority).
- **Wall W==D authority:** `bench/gp-track/gp4_latest.json` → 53.8 tok/s ctx512/ctx2048 (promoted G5 K-only).
- **Route-bound evidence:** attention = `flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128` + `flash_state_gmax_40_128` + `flash_state_combine_40_128`; **no** `owned_flash`/`gqa_coop_vec` → no hidden fallback; `use_flash=True`.

## Refreshed stack (stable across ctx512 ≈ ctx2048)

| bucket | ctx512 | ctx2048 | kernels | class | chaseable? |
|---|---:|---:|---:|---|---|
| q4k_gemv | 40.56% | 40.69% | 4 (6.31 GB/step) | ROUTE_CEILING (HBM ~400 GB/s, fresh A/B) | no |
| reduce_partial | 19.63% | 19.7% | 14 | attention-combine ~13.6 (refuted transforms) + coop-combine ~4.1 (refuted) + rmsnorm ~1.3 | partly (see below) |
| attention (flash tile) | 16.94% | 16.86% | 3 | COMPUTE_PRIMITIVE_GAP (scalar PV; vec-PV blocked, v_dot2 emitter-blocked) | no |
| q6k_gemv | 11.06% | 10.98% | 1 (1.0 GB/step) | ROUTE_CEILING (L3-shipped, coop ~51% peak) | no |
| other | 7.85% | 7.84% | 13 | E_49152 6.69 EMITTER_BLOCKED + **4.90 reachable elementwise** + low-Amdahl | aggregate-only |
| lm_head | 3.96% | 3.94% | 1 | AT_CEILING (706 GB/s) | no |

## Key reconciliation findings

1. **G5 K-only changed the attention TILE, not the combine.** 14B still runs the **2-kernel** `flash_state_gmax` + `flash_state_combine`. The TG-P14.8 **fused split-preserving combine** (`flash_fused_gmax_combine`, now compilable+correct) is **not wired into 14B**. This is the one *non-refuted* combine lever — but the 8B precedent for it was a ~2% near-tie.
2. **E_49152 (6.69%) is not removed by K-only — it is *used*.** K-only reads V from global, L2-warmed by `E_49152_32_3`. It stays EMITTER_BLOCKED (elementwise→flash_reduce at a global barrier).
3. **Every large bucket is route-ceiling or primitive-missing.** Q4_K (40.6) and Q6_K (11) are at their bandwidth/route ceilings; attention combine (13.6) has only refuted transforms or a missing coordination primitive; attention PV (16.9) needs vectorized-PV/v_dot2 lowering (blocked at UOp spec / emitter).
4. **The only reachable-now route lever is the 4.9% aggregate elementwise** (silu_gate 1.26, qk_norm_scale 1.46, rmsnorm_scale 1.28, residual_adds 0.90). Individually below W==D noise (SF3: silu_gate alone +0.4 < 0.5). Untested as an aggregate.

## Health observations

Host clean at start (smoke `[2.]`, 0 D-state). Each PROFILE capture transiently spiked ~16 `ttm` kworkers to D; all cleared within seconds (5 clean re-samples, no dmesg reset errors). No wedge. ctx4096 intentionally skipped.

## Limitations

- Attribution is GPU-compute *share*, not wall-time — a share can be off the critical path if overlapped (this is exactly the SF3 risk for the elementwise bucket).
- reduce_partial composition (combine/coop/rmsnorm) carried from the 2026-07-01 reduce-source trace (those reduce kernels are unchanged by G5 K-only), not re-run this pass.
- No W==D spread measured this pass; roofline uses the 2.0pp floor.
