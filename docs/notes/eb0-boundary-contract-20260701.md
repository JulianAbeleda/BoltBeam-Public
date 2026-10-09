# EB0 — Emitter Blocked Boundary Contract

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M, gfx1100.

## Selected fragment

`E_49152_32_3` — **6.69% of GPU compute at ctx512**, 40 calls/step (one per layer), context-dependent
(absent at ctx128 where SDPA is used instead of flash).

Shape: 49152 × 32 × 3 = 4,718,592 elements = Hkv × MAXC × Hd = 8 × 4608 × 128 (V cache full slice).

## Dependency chain

```
rope(k, freqs_cis[start_pos:start_pos+T])   ← Q4K GEMV for attn_k + RoPE apply
     ↓ r_40_28start_pos2B129_16_8            ← RoPE reduce (symbolic start_pos)
     ↓ cache_kv[..., start_pos:start_pos+T, ...].store(stack(k_rope, v))  ← KV write
     ↓ assigned_kv = Tensor(cache_kv.uop.after(store))                    ← ordering fence
     ↓ E_49152_32_3                          ← V materialization (this boundary)
     ↓ E_1920_32_3                           ← flash init kernel
     ↓ flash_max_40 / flash_partial / ...    ← flash_partial_coop_vec reads V
```

## Producer op

`apply_rope(k, freqs_cis[start_pos:start_pos+T])` → stored via `cache_kv.uop.after(store)`.
The `assigned_kv` tensor represents `cache_kv` after the write, with an ordering dependency on the store.

## Consumer op

`flash_decode_attention(q, assigned_kv[0, 0], assigned_kv[1, 0], ...)` in `model.py:1178`.

Inside this call:
```python
vc_f = v_full.reshape(Hkv * MAXC * Hd)   # v_full = assigned_kv[1, 0]
po = Tensor.empty(...).custom_kernel(prob, vc_f, fxn=flash_partial_coop_vec_kernel(...))[0]
```

The `custom_kernel` call for `flash_partial_coop_vec` requires a concrete realized buffer for `vc_f`.
Since `v_full = assigned_kv[1, 0]` derives from `cache_kv.uop.after(store)` with a `[1, 0]` index (non-contiguous
view of the full `assigned_kv`), tinygrad's callify cannot alias the view back to the persistent `cache_kv` buffer
— it inserts a copy kernel: **E_49152_32_3**.

## Materialized tensor

| field | value |
|-------|-------|
| name | V cache full slice (copy before flash_partial) |
| shape | [Hkv=8, MAXC=4608, Hd=128] = 4,718,592 elements |
| dtype | f16 (matches cache_kv dtype for 14B on gfx1100) |
| lifetime | per-step temporary (created and consumed within one decode step) |
| persistent | NO — discarded after flash_partial reads it |

## Why materialization happens

tinygrad's `callify` mechanism for `custom_kernel` requires each input to be a directly addressable realized
buffer. `assigned_kv[1, 0]` (the V half, indexed `[1, 0]` from `assigned_kv` shape [2, 1, Hkv, MAXC, Hd]) is
a non-contiguous strided view. When this view is reshaped to `[Hkv * MAXC * Hd]` and passed to `custom_kernel`,
tinygrad cannot alias it to the underlying `cache_kv` buffer (due to the `[1, 0]` indexing creating a non-trivial
stride/offset chain). A copy kernel is inserted to produce a fresh contiguous buffer.

The current generated 8B live-split route bypasses this by reading the whole assigned KV buffer directly. For 14B,
the G5 generated route uses its shape-specific staging contract instead of the old sliced fallback.

## Correctness condition for removal

Removal is SAFE if: the persistent `cache_kv[1, 0]` buffer can be passed directly to `flash_partial_coop_vec`
without an intermediate copy, provided the KV store (writing new k/v at position start_pos) is guaranteed to
complete first. The `uop.after(store)` in `assigned_kv` provides this guarantee.

## Projected Amdahl

| bound | value |
|-------|-------|
| upper | 6.69% of GPU compute at ctx512 |
| realizable | unknown — measured -9.6% W==D regression in EB3/EB4 (see below) |

The removal was tried and regressed. See `eb5-ingestion-report-20260701.md` for the mechanism.
