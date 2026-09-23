# EB5 — BoltBeam Ingestion Report

Date: 2026-07-01. Follows EB0 boundary contract.

## Candidate ingested: decode_bypass_kv_slice

**Fragment:** E_49152_32_3 (6.69% GPU at ctx512)
**Route family:** emitter_blocked / kv_cache_view
**Flag:** DECODE_BYPASS_KV_SLICE=0 (default-off)

## Evidence summary

| phase | result |
|-------|--------|
| EB0 boundary contract | V cache [Hkv=8, MAXC=4608, Hd=128] copy; callify cannot alias [1,0]-indexed view |
| EB1 dependency class | Case A real dependency + cache-warming role (dual function) |
| EB3 correctness | PASS (rel_rmse=0.00e+00, 5 steps) |
| EB4 W==D ctx128 | 52.3 → 52.3 tok/s (0.0) |
| EB4 W==D ctx512 | 50.1 → **45.3 tok/s (-4.8, -9.6%)** |
| verdict | **EB4_REFUTED_WD_REGRESSION** |

## Key finding

E_49152_32_3 is **BENEFICIAL_CACHE_WARM**, not pure dead work. It copies V into a fresh L2-warm buffer before
flash_partial_coop_vec reads it. Removing the copy makes flash_partial read cold V from the 9.4 MB persistent
KV cache, incurring more HBM latency than the copy itself costs. Net effect: -9.6%.

The SF1 EMITTER_BLOCKED classification was correct about the scheduling mechanism but did not account for the
cache-warming role. This is now recorded as a known property of this kernel class.

## Manifest update

- `decode_bypass_kv_slice`: REFUTED (do_not_retry=False)
- `decode_bypass_kv_slice_lds`: PRIMITIVE_MISSING — would achieve equivalent warming via LDS-staged V prefetch
  in flash_partial_coop_vec; requires LDS-alloc UOp in tinygrad emitter.

## Reopen condition (precise)

`decode_bypass_kv_slice_lds` reopens when tinygrad's AMD emitter supports LDS-alloc UOp (a workgroup-scoped
shared memory buffer) AND a flash_partial_coop_vec variant with LDS-staged V prefetch demonstrates:
1. CORRECTNESS_PASS (token-identical to baseline)
2. W==D ctx512 >= baseline (52+% tok/s for 14B)

Do not retry the global-copy removal without LDS staging.
