# LDS5 / PR5 — BoltBeam Ledger Update: Prologue Range LDS Staging Track

Date: 2026-07-01.

## Classification evolution

| Phase | Label | Reason |
|-------|-------|--------|
| EB5 | PRIMITIVE_MISSING | LDS-alloc UOp claimed missing (wrong) |
| LDS0 | EMITTER_BLOCKED / prologue_range_uop | Primitives exist; pattern claimed missing (partially wrong) |
| PR0/PR1 | SEARCH_SPACE_INCOMPLETE / block_tile_g5_variant_missing | UOp pattern proven to exist (block tile G=4); 14B G=5 shape has no kernel variant |

## What was learned

1. `AddrSpace.LOCAL`, `UOp.barrier`, and cooperative staging are NOT missing primitives —
   they are already used in `extra/qk_flash_decode.py` for K staging (lines 214, 253) and
   for full K+V staging in the block tile kernel (lines 954–1067).

2. The block tile kernel (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`) stages
   TK=16 tokens of both K and V into LDS per block iteration, then computes attention from LDS.
   This is exactly the "prologue range" pattern, and it works for the 8B model (G=4).

3. 14B has G=Hq/Hkv=40/8=5. The block tile enforces `G == WARPS == 4` (line 963) and raises
   `ValueError` for G=5. No G=5 variant exists.

4. Per-token V staging in flash_partial_coop_vec: NOT_BENEFICIAL. The d=LOCAL axis already
   gives coalesced V reads — no LDS staging adds value for a per-token access pattern.

## Candidate updates

- `decode_bypass_kv_slice_lds`: updated from `emitter-blocked` to `search-space-incomplete`.
  Blocker: `block_tile_g5_variant_missing` (not a missing UOp).

- `decode_flash_v_prologue_lds_stage`: new candidate added. Status `search-space-incomplete`.
  Reopen condition: `flash_block_tiled_g5_score_pv_kernel` with WARPS=5 or G=5-compatible
  warp layout, TK=16 K+V LDS staging, online softmax for G=5 GQA groups per KV head.

## Reopen condition (precise)

Build `flash_block_tiled_g5_score_pv_kernel`:
- Generated UOp, no handwritten HIP
- WARPS=5 (or equivalent G=5 warp mapping)
- TK=16 cooperative K+V staging into LDS (8KB/workgroup — 64KB budget = 8× margin)
- Online softmax for G=5 GQA groups
- Gate: `DECODE_ATTN_BLOCK_TILE_G5=0` (default-off)
- Grid: Hkv × S = 8 × S workgroups (preserves Hkv*S split parallelism, with G=5 warp parallelism inside)

Once built, PR2 (microgate correctness) → PR3 (flash prototype) → PR4 (W==D) can proceed.
Upper Amdahl: 6.69% at ctx512 (E_49152_32_3 elimination) + LDS vs L2 V-read acceleration.
