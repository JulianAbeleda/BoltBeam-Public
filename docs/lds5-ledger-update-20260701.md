# LDS5 — BoltBeam Ledger Update

Date: 2026-07-01. Follows LDS0 audit of the LDS staging primitive track.

## Correction to EB5 classification

EB5 recorded `decode_bypass_kv_slice_lds` as **PRIMITIVE_MISSING** (LDS-alloc UOp unavailable
in tinygrad AMD emitter). **LDS0 audit finds this is incorrect.**

Evidence from LDS0:
- `Ops.DEFINE_LOCAL` (AddrSpace.LOCAL) exists in `tinygrad/uop/__init__.py:23`
- `Ops.BARRIER` exists, lowered to `__builtin_amdgcn_fence + __builtin_amdgcn_s_barrier`
- **Both are already in use** in `extra/qk_flash_decode.py` lines 214 and 253, for K staging
  in `flash_pall_lds_crosslane_fdot2_score_whole_cache_kernel`

## Updated classification

| field | EB5 | LDS0 (corrected) |
|-------|-----|-----------------|
| status | PRIMITIVE_MISSING | **EMITTER_BLOCKED** |
| blocker | LDS-alloc UOp missing | `prologue_range_uop` — a pre-stage loop before REDUCE |
| LDS primitive available | No | **Yes** (already used for K staging) |

## What the actual blocker is

Full-split V staging requires:
1. A pre-stage loop (over j_load = 0..L-1) that cooperatively loads V[kv, split, :] into LDS
2. An implicit barrier after the pre-stage completes
3. The main j REDUCE reads from LDS instead of global

The current UOp kernel DSL supports REDUCE ranges but no "prologue range" that executes before
the main reduce, writes to DEFINE_LOCAL, and has an implied post-stage barrier. This is a new
structural pattern in the kernel DSL, not a missing UOp opcode.

Three angles were evaluated in LDS0:
- Cross-kernel staging: GRAPH_LIFETIME_BLOCKED (HIP LDS is workgroup-scoped)
- Per-token V staging (inside j loop): REACHABLE_NOW but NOT_BENEFICIAL (zero data reuse)
- Full-split V staging (prologue + barrier + reduce): **EMITTER_BLOCKED** — the viable angle

## Candidate manifest update

`decode_bypass_kv_slice_lds` in `boltbeam/data/candidates.json`:
- `status`: `emitter_blocked`
- `emitter_blocker`: `prologue_range_uop`
- `corrects`: note about wrong EB5 classification
- `evidence_refs`: added `bench/lds-staging-primitive/lds0_latest.json`
- Description updated to reflect corrected analysis

## Reopen condition (precise)

A "prologue range" UOp primitive in the tinygrad kernel DSL that:
1. Runs before the main REDUCE range
2. Iterates over a specified dimension (same count as the REDUCE)
3. Writes values into a DEFINE_LOCAL buffer
4. Implies an `s_barrier` before the REDUCE begins

Once this exists, `flash_partial_coop_vec_whole_cache_kernel` can be modified to add a V
prologue stage, eliminating E_49152_32_3 (6.69% GPU at ctx512) with projected upper Amdahl
of 6.69% → expected W==D gain of +3-5 tok/s at ctx512.
