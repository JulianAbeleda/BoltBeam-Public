# G=5 Block Tile — BoltBeam Ledger Update

Date: 2026-07-01.

## What was tried

`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` had `WARPS = 4` hardcoded and rejected
G=5 (14B: Hq=40/Hkv=8=5). Fix: parameterize `WARPS = G`. Gate in model.py:
`DECODE_FLASH_BLOCK_TILE_G5=0` routes through `flash_decode_attention_whole_cache` with
`DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1`.

## Result

- Correctness: PASS (token-identical, rel_rmse=0)
- W==D ctx512: 50.2 → 10.0 tok/s (**-80.1%**, G5_REFUTED_WD_REGRESSION)
- Individual kernel: flash_partial_coop_vec=27µs → block_tile=1915µs (71× slower)

## Root cause

`flash_decode_attention_whole_cache` targets MAXC=4608 cache. At ctx=512:
- `s_route = ceildiv(512, 96) = 6` splits → grid = `8 × 6 = 48` workgroups
- Baseline `gqa_coop_vec` at ctx=512: grid = `8 × 48 = 384` workgroups (8× more)
- GPU severely under-occupied; effective bandwidth 3 GB/s vs 596 GB/s for baseline

## Candidate updates

- `decode_flash_v_prologue_lds_stage`: **refuted** (whole_cache_path_at_partial_ctx), do_not_retry=False
- `decode_flash_block_tile_g5_native_context`: **open** — same kernel (WARPS=G, LDS staging) but
  routed through sliced-KV flash path. Needs flash_partial wrapper for G=5 block tile that uses
  actual context size (not MAXC). Projected Amdahl if achieved: 6.69% upper bound.

## BoltBeam tests

Run: `cd BoltBeam && python3 -m pytest tests/ -x -q`
