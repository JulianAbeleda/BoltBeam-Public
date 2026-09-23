# 14B decode — v_dot2 wired into the Q4_K GEMV (codegen improvement) — 2026-07-02

**The "converged" verdict was wrong, as suspected.** Pursuing the codegen instead of accepting the wall
produced a working v_dot2 Q4_K GEMV, a small real speedup, and — more valuably — the correct diagnosis of
where the GEMV actually spends its time.

## What was built (tinygrad, worktree branch `vdot2-lowering`, commit `6562fa684`)

`_q4k_group_dot_vdot2` in `extra/q4_k_gemv_primitive.py` pairs consecutive dequantized Q4_K nibbles into a
packed `f16x2`, dots them against the matching `x` pair via `__builtin_amdgcn_fdot2` (f32 accumulate), and
chains the accumulator. Wired into the default G3 decode GEMV behind `DECODE_Q4K_GEMV_VDOT2=1` (default-off).
This is the **first time v_dot2 fires on the weight GEMV** — prior work (`decode-attention-a3-1-vdot2-*`)
only reached the attention *score* path, where it was cross-lane-bound and showed no transfer.

Renders `__builtin_amdgcn_fdot2(make_half2(...))`, 16 fdot2/kernel (8 groups × 2 nibble-pairs), 32→16 MACs/block.

## Gates (Qwen3-14B-Q4_K_M / gfx1100)

| gate | result |
|---|---|
| renders v_dot2 | yes — `__builtin_amdgcn_fdot2`, 16/kernel (DEBUG=4) |
| correctness (NLL) | **dNLL +0.00013** @128 tok — negligible, ≪ 0.003 free threshold; microgate l2 210.0838 vs 210.0840. f16-lossy → NLL-gated, not token-parity |
| W==D decode | 55.2→55.5 ctx512, 55.0→55.4 ctx2048 — **+0.3–0.4 tok/s, reproducible reversed-order** (below the 0.5 promotion bar) |

Verdict: `PROMOTION_CANDIDATE_DEFAULT_OFF` (tier-C: correct + reproducible but sub-0.5-tok/s).

## The two corrections this forced

1. **v_dot2 is NOT emitter-blocked.** The renderer lowers it fine; it just had never been wired into the
   GEMV. Removed from `DEFAULT_EMITTER_CAPABILITIES.blocked_knobs`. My earlier "renderer can't lower v_dot2"
   encoding was stale belief, now refuted by running code.
2. **Decode had not physically converged.** The Q4_K GEMV runs at ~42–52% of the 960 GB/s HBM peak — the
   405 GB/s "route ceiling" is a codegen/compute ceiling, not the bandwidth wall. A working codegen path to
   raise it exists.

## The real frontier (redirected search)

Halving the MAC moved whole-decode only +0.6% → **the GEMV bottleneck is the dequant bit-unpack (rshift /
and / cast per nibble), not the multiply.** The next codegen lever is reducing dequant ALU — e.g. vectorized
/ v_dot-style byte unpack, or dequant-into-f16x2 with fewer ops — not further dot fusion. `decode_q4k_gemv_vdot2`
is the beachhead; the dequant-ALU candidate is the follow-on.

## Follow-on: dequant frontier profiled + exhausted (2026-07-02)

ISA profile of the vdot2 GEMV (`extra/qk_vdot2_gemv_isa_profile.py`, tinygrad worktree) showed the kernel is
**dequant-ALU-bound**: bit-unpack (`v_bfe_u32`×30, `v_lshrrev`×19, `v_and`) + int→float convert
(`v_cvt_f32_ubyte0`×48) ≈ 120 instr vs ~63 fp-math — the dequant is ~2× the arithmetic, and the converts are
inherent to Q4_K. The v_dot2 lowering converts only **7 of 16** fdot2 to real `v_dot2` (rest → `v_fma_mix_f16`
+ 16 `v_pack_b32_f16`), raising **vgpr 58→87** — the register-pressure/occupancy cost offsets the MAC win,
which is why the gain was only +0.6%.

Two dequant-reduction sub-levers were then built and **refuted**:

| sub-lever | result |
|---|---|
| f16 dequant (mode 2) | refuted — reusing the f32 group-params forces an f16→f32→f16 round-trip that *adds* 48 `v_cvt_f16_f32` (converts 50→96, total 407→442). Reverted. |
| int-dot v_dot4 / mmvq (`Q4K_VDOT`+`AMORT`) | refuted — **49.0 vs 55.2 tok/s (−11%)**. x→q8_1 quantization + the extra pack kernel outweigh the dequant savings, and the weight HBM read is unchanged. |

**Conclusion (multi-lever, physical):** the Q4_K decode GEMV codegen is at a genuine ceiling. v_dot2/v_dot4/
f16-dequant all fail to close the +19% gap. The gap is not reachable via dot primitives — it needs a different
quant format with cheaper dequant, or an occupancy path (the GEMV runs at ~42–52% of HBM peak because the
dequant ALU + register pressure cap issue rate, not because HBM is saturated). This is now the *evidenced*
frontier, replacing the earlier assumed convergence.

## CORRECTION: int-dot was NOT refuted — it is a primitive win, vocab-blocked (traced)

The "int-dot v_dot4 refuted at −11%" above was a mis-attribution — an unclassified whole-route number. Per-kernel
time tracing (`extra/qk_vdot_time_trace.py`) decomposed it:

| Q4_K GEMV shape | role | G3 µs | int-dot µs | delta |
|---|---|---|---|---|
| 17408×5120 | FFN gate/up | 8543 | **7709** | **−834 (int-dot −10%, WINS)** |
| 5120×5120 | attn q/o | 3166 | 5101 | +1935 (loses) |
| 1024×5120 | attn KV | 647 | 2545 | +1898 (loses) |

Plus un-fused x→q8_1 pack ~813 µs and extra partial-sum reduces ~882 µs. **The int-dot primitive is genuinely
faster on the dominant FFN GEMV.** The whole-route loss came from (a) int-dot being an all-or-nothing alternate
route (G3 bypasses it), so it also ran on the narrow attn/KV shapes where its partial+reduce structure loses,
and (b) the x-quantize+pack being un-fused overhead that ~cancels the FFN win.

So the frontier is **NOT** the dead ceiling claimed above — it is **open, behind two missing vocab items**:
1. `intdot_shape_selective_routing` — apply int-dot per-linear only where it wins (large FFN), keep G3 elsewhere.
   Today int-dot is a whole-route replacement at the wrong routing layer.
2. `fused_quantize_intdot` — fuse the x→q8_1 quantize into the int-dot kernel. Un-fused, it costs ~813 µs that
   cancels the ~834 µs GEMV win (net wash); fused, FFN int-dot is a net decode gain.

This is the lesson of "classify evidence before fixing mechanisms": a whole-route −11% did not prove the primitive
slow. Tracing showed we lacked the *vocabulary* to translate the primitive into a win, not that the primitive
failed. Candidate `decode_q4k_gemv_intdot_vdot4` (status `primitive_win_vocab_blocked`).
