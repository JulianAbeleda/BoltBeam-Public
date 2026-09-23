# GP5 Ledger Update — 2026-07-01

## Summary

Added `decode_flash_block_tile_g5_konly` candidate (PROMOTED, TIER_A) to candidates.json
following the GP0-GP5 Generated ISA Primitive track.

## What changed

**candidates.json:**
- Updated `decode_flash_block_tile_g5_native_context.compiler_pathology.reopen_condition`
  to reflect that the K-only reopen was satisfied and produced TIER_A.
- Added new candidate: `decode_flash_block_tile_g5_konly` (origin=shipped, status=promoted, tier=TIER_A).

## Evidence chain (GP0→GP5)

| Phase | Verdict | Key artifact |
|-------|---------|-------------|
| GP0 | PASS (purity gate) | docs/gp0-purity-gate-result.md |
| GP1 | REACHABLE_NOW | docs/gp1-primitive-gap-analysis.md, bench/gp-track/gp1_latest.json |
| GP2 | IMPLEMENTED | staging param in extra/qk_flash_decode.py; flag in tinygrad/llm/model.py |
| GP3 | GP3_PASS_MICROGATE | bench/gp-track/gp3_microgate.json (rel_rmse=0) |
| GP4 | GP4_PASS_TIER_A | bench/gp-track/gp4_latest.json |
| GP5 | LEDGER_UPDATED | this doc, boltbeam/data/candidates.json |

## W==D numbers

Model: Qwen3-14B-Q4_K_M on gfx1100.

| ctx | baseline | K_ONLY | delta | % |
|-----|----------|--------|-------|---|
| 128 | 52.1 | 52.2 | +0.1 | flat |
| 512F | 49.9 | 53.8 | **+3.9** | **+7.8%** |
| 2048F | 46.9 | 53.8 | **+6.9** | **+14.7%** |

## Mechanism recap

The `decode_flash_block_tile_g5_native_context` oracle found `LDS_OR_MEMORY_OVERHEAD`
(scratch=0, LDS=8192 bytes = 8KB for K+V staging). The K_ONLY variant halves LDS
(8192→4096 bytes), removes ~780 instructions, and reads V from L2-warmed global
cache (warmed by E_49152_32_3 BENEFICIAL_CACHE_WARM). This is a pure generated-UOp
fix — no handwritten ISA, no forbidden pattern.

## Rollback

```
DECODE_FLASH_BLOCK_TILE_G5=0
```

## Open track

The native ISA path (<100us per workgroup) remains open for Phase I.
32B (Hq=64, Hkv=8, G=8) with staging=K_ONLY is the next parity check.
