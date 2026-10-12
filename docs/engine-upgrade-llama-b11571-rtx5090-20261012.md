# Engine upgrade audit: llama.cpp b9592 to b11571, 27B, RTX 5090 (2026-10-12)

The pin moves from b9592 to b11571. The prefill is faster and the decode is level. This is not an isolated
change. Many rows moved, and the FFN work changed by design: the new build prunes the last layer. The audit says
where each change is. The check is [engine-upgrade-audit.md](engine-upgrade-audit.md).

## What ran

- Old: llama.cpp b9592, `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`, the earlier pin.
- New: llama.cpp b11571, `097cf33d9499b1eade67d51e16d7c75427e7b267`, the newest release tag at the time.
  Built with the old build's flags (Release, CUDA, architecture 120, nvcc, `LLAMA_CURL=OFF`).
- Model: the 27B hybrid Q4_K_M GGUF of the earlier audits.
- BoltBeam `46a7703`. One run per build and workload. No retries.
  - Prefill at 512 and 2048: old is the saved loop27-baseline (the same b9592 build); new is upg-new-prefill.
  - Decode at 512: upg-old-decode (the b9592 binaries, with their own libraries) and upg-new-decode.
- The box's own model server was stopped for the runs. The GPU had no other compute process.
- Compared with `screen diff`. Each run's manifest records its build against the pin.

## Untraced times

| Workload | b9592 | b11571 | Change |
|---|---:|---:|---:|
| prefill 512, ms | 124.546 | 116.192 | -8.353 (-6.7%) |
| prefill 2048, ms | 501.438 | 469.067 | -32.372 (-6.5%) |
| decode at 512, ms per token | 14.013 (71.36 tok/s) | 13.852 (72.19 tok/s) | -0.160 (-1.1%) |

## Prefill rows that moved or changed work

Band: max(3% of the b9592 time, 0.05 ms). Times are the captured kernel ms.

| Row | 512 | 2048 | What changed |
|---|---|---|---|
| recurrent | 18.774 -> 10.874 | 76.596 -> 44.099 | moved: the GDN kernel (four state columns per warp) |
| rms_norm_f32 | 2.687 -> 3.598 | 11.080 -> 14.889 | moved, more calls: recurrent L2 norm is now an RMS norm |
| l2_norm_f32 | 0.747 -> gone | 3.180 -> gone | replaced by the RMS norm |
| concat_f32_non_cont | 2.405 -> gone | 9.952 -> gone | renamed template |
| concat_non_cont | new, 2.392 | new, 9.972 | the renamed concat |
| k_get_rows_float | 0.269 -> 0.093 | 1.445 -> 0.480 | moved, fewer calls |
| k_get_rows_float_vec | new, 0.099 | new, 0.725 | new template |
| ffn_gate_up Q4_K | 36.863 -> 36.094 | 153.358 -> 149.002 | in the band; calls 128 -> 126, 512 -> 504 |
| ffn_down Q4_K | 19.251 -> 18.903 | 80.559 -> 78.464 | in the band; calls 64 -> 63, 256 -> 252 |
| ffn_gate_up Q4_K, last layer at output tokens | new, 0.070 | new, 0.070 | the pruned last layer, paired |
| ffn_down Q4_K, last layer at output tokens | new, 0.037 | new, 0.037 | the pruned last layer, paired |
| mul_mat_q_stream_k_fixup | 3.997 -> 3.963 | 17.009 -> 16.482 | moved at 2048; calls and bytes follow the FFN |
| quantize_mmq_q8_1 | 2.592 -> 2.512 | 10.784 -> 10.540 | moved at 512; fewer calls |
| attn_qo Q6_K, ssm_alpha_beta Q8_0, rope_multi | in the band | just beyond it (-3.0%, -3.1%, -6.4%) | no work change |
| gaps | 0.484 -> 0.362 | 5.153 -> 39.085 | moved |

Every other row stayed within its band with its work unchanged. Both FFN rows' work equals the model with the
pruned layer.

## Prefill tie-out

- Kernel changes plus gap changes equal the captured wall change at both lengths.
- 512: captured 112.1 ms against 116.2 ms untraced (-3.5%). Confirmed end to end.
- 2048: captured 499.8 ms against 469.1 ms untraced (+6.6%). **Not confirmed.** Its gaps grew to 39.1 ms, 30.8 ms
  more than the untraced time leaves room for. This repeats the earlier full-checkout capture (41.3 ms gaps at
  2048). The b9592 plus GDN capture had normal gaps (5.6 ms). The gap growth comes from another change in the new
  build, under capture. Its cause was not established. The untraced -32.4 ms at 2048 stands.

## Decode at 512

- Every row stayed within its band except three:
  - "other" fell 0.385 -> 0.241 ms;
  - attention rose 0.031 -> 0.112 ms;
  - the gaps fell 3.338 -> 0.951 ms.
- No row came or went.
- The tie-out reconciles.
- b9592's capture is not confirmed end to end: 16.9 ms captured against 14.0 ms untraced, +20.8%.
- b11571's capture is confirmed: 14.5 ms against 13.9 ms, +4.8%.

## Verdict

The upgrade is justified. Prefill is 6.5 to 6.7% faster untraced, mostly from the recurrent block. Decode is
1.1% faster. The FFN's work fell by the pruned last layer, which BoltBeam now models. The norm, concat and get-rows
changes are graph changes in the new build. The RMS norm grew 0.9 ms at 512 and 3.8 ms at 2048, and the L2 norm
it replaced took 0.7 and 3.2 ms: net +0.2 and +0.6 ms.
Before trusting a 2048 capture on b11571, check its end-to-end line.

b11571 also includes llama.cpp PR #26004 ("server : preserve context checkpoints across slot save/restore",
`033df86b6`): the hybrid slot-restore fix.

## Evidence on the box

- `~/storage/bbeng/tmp/loop27-baseline/`, `upg-new-prefill/`, `upg-new-decode/`, `upg-old-decode/` and their
  `-pipeline.log` and `-summary.txt`.
- The b9592 build backup: `~/storage/llama.cpp-build-cuda-b9592/` (BACKUP.txt records the commits).
- The b11571 build: `~/storage/llama.cpp-build-cuda-b11571/`; `~/env/llama.cpp/build-cuda` links to it.
