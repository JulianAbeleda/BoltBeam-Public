# What the small roles could recover: RTX 5090, 2026-10-10

The 5090 record (in-model-vs-generic-rtx5090-20261010.md) put the token at 3.98 ms with 3.29 ms in weight kernels,
and the loss in the small roles: the attention projections (q, k, v, o) take 0.82 ms per token at 45 to 70% of peak
because a 2 to 9 MB read cannot fill a 1.7 TB/s memory in a few microseconds. gate and up are already one launch in
llama.cpp (92% of peak). This record asks what one launch for q, k and v would recover, measures it with the
engine's own kernel at the fused shapes, and measures the other lever, a batch of 2 or 4 streams, on the same roles.
No engine was changed.

## Setup

- Machine: the same Ubuntu box, RTX 5090 (170 SMs, 96 MiB L2, compute capability 12.0), CUDA 13.2. Peak read
  bandwidth 1,693.9 GB/s (BoltBeam's CUDA read probe, the earlier run). Target `nvidia_sm120`.
- Model: Qwen3-8B-Q4_K_M.gguf. Its attention weights, read from the file: attn_q and attn_k are Q4_K in all 36
  layers; attn_v is Q4_K in 18 layers and Q6_K in the other 18. That is where the 54 attn_kv Q4_K and 18 attn_kv
  Q6_K calls per token come from.
- Engine: llama.cpp, the box's CUDA build (2026-08-20 binary; `ggml-cuda/mmvq.cu` from the 2026-10-05 source).
- BoltBeam main 69a9c34, the clean clone under the data disk. The kernel timer (`engine_kernels` CUDA adapter) timed
  ggml's own `mul_mat_vec_q` at every shape, read sweep (256 MiB), 10 warmups, 60 samples per set, 3 sets, the
  dispatch floor measured before each set (3.8 to 5.0 µs). Correctness against the dequantized q8_1 vector passed
  on every shape (max_rel_err 5.7e-8 to 2.9e-7).
- The fused shapes are real weight bytes: the q, k and v tensors of one layer laid end to end (same column count,
  so the rows concatenate), 6,144 x 4,096 Q4_K (14.16 MB) and 5,120 x 4,096 Q4_K (11.80 MB); the 2,048-row Q6_K
  shape is two layers' v tensors, a shape no layer of this model has.
- Batch: the box has no `llama-batched-bench` binary (CMake stubs only), so the whole step at 2 and 4 streams came
  from a `llama-server` started for the probe (`-np 4 -c 8192 -fa on`), B concurrent completions of a 153-token
  prompt, 64 tokens each, the step time from the server's own timings, 3 repeats, the median kept. The same roles
  at batch 2 and 4 came from the engine's `mul_mat_vec_q<type, 2|4, ...>` instantiation (the kernel it launches for
  2 to 4 columns: 4 warps, 2 rows per block) through the same timer. `llama-bench -b 2|4` was also run: 254.5 and
  254.9 tok/s, the same as `-b 2048` (253.9), because `-b` is the prefill batch and llama-bench decodes one stream.
- The resident llama-server was stopped for the GPU steps and came back both times (down 21 s and 3 min 8 s).

## Prediction, written before the runs

By Little's law with the measured 4 µs floor and the big-role bandwidth (1,520 GB/s): a fused q+k+v Q4_K launch at
13.3 µs. By the in-model fit (2.4 µs + bytes / 1,690 GB/s): 10.8 µs. Today's three launches cost 14.28 µs in-model.
Token gain 0.04 to 0.12 ms (1 to 3%, 2 to 8 tok/s), with the Q6_K k+v fusion worth 0 to 0.02 ms. Batch 2: the small
roles' per-call time +5 to 20%, their per-token cost about 0.45 ms (from 0.82); batch 4 about 0.25 ms; the whole step
4.2 to 4.4 ms at batch 2 and 4.6 to 5.0 at batch 4.

Wrong before the first run: the prediction assumed the Q6_K calls were k and v of nine layers. The file says attn_v
alone, 18 layers; the k+v Q6_K case does not exist here and q+k Q4_K (11.8 MB) does. Right in direction: the fused
launch landed on the in-model fit's figure (9.9 µs less floor against 10.8 predicted), the per-token small-role cost
at batch 2 and 4 (0.43 and 0.23 ms measured against 0.45 and 0.25). Under-predicted: the whole step at batch 2 and 4
(4.63 and 5.93 ms measured against 4.2 to 4.4 and 4.6 to 5.0).

## Table 1: the engine's kernel at the 7 roles and the fused shapes (kernel timer, read sweep, less floor)

Median of 60 samples per set, three sets; "less floor" is the median over the sets of (median less that set's
floor). GB/s and % of peak are on the less-floor time. In-model is the nsys figure from the earlier record.

| shape | bytes/call | calls/token | median µs, 3 sets | less floor µs | GB/s | % peak | spread %, 3 sets | in-model µs |
|---|---|---|---|---|---|---|---|---|
| attn_kv Q4_K 1024x4096 | 2.36 MB | 54 | 7.58 / 7.65 / 7.36 | 3.41 | 692 | 41 | 16 / 19 / 30 | 3.13 |
| attn_kv Q6_K 1024x4096 | 3.44 MB | 18 | 7.95 / 7.97 / 7.98 | 4.10 | 840 | 50 | 19 / 28 / 25 | 4.17 |
| attn_qo Q4_K 4096x4096 | 9.44 MB | 72 | 11.62 / 11.60 / 11.62 | 7.60 | 1,242 | 73 | 19 / 21 / 18 | 8.02 |
| ffn_down Q4_K 4096x12288 | 28.31 MB | 18 | 22.62 / 23.42 / 22.43 | 18.62 | 1,520 | 90 | 11 / 10 / 10 | 19.15 |
| ffn_down Q6_K 4096x12288 | 41.29 MB | 18 | 32.94 / 32.42 / 32.43 | 28.45 | 1,451 | 86 | 7 / 6 / 7 | 28.76 |
| ffn_gate_up Q4_K 12288x4096 | 28.31 MB | 72 | 22.27 / 22.34 / 22.22 | 18.06 | 1,567 | 93 | 9 / 10 / 7 | 36.27 / 2 |
| lm_head Q6_K 151936x4096 | 510.50 MB | 1 | 305.39 / 305.33 / 304.86 | 301.04 | 1,696 | 100 | 1 / 1 / 1 | 301.98 |
| fused q+k+v Q4_K 6144x4096 | 14.16 MB | (18) | 13.90 / 13.86 / 13.78 | 9.89 | 1,432 | 85 | 12 / 15 / 10 | |
| fused q+k Q4_K 5120x4096 | 11.80 MB | (18) | 13.07 / 12.27 / 12.91 | 9.01 | 1,310 | 77 | 19 / 20 / 19 | |
| hypothetical k+v Q6_K 2048x4096 | 6.88 MB | (0) | 9.81 / 9.86 / 9.70 | 5.81 | 1,185 | 70 | 23 / 17 / 13 | |

The 7 roles' less-floor sum is 3.254 ms per token against the in-model weight sum of 3.292 (0.99). The medians of the
three sets agree within 1% on every shape from 9 MB up and within 4% under it; the spread inside a set is 16 to 30%
on the 2 to 3 MB shapes, 7 to 11% on 28 to 41 MB, 1% on lm_head (the noisy rows are the ones under 3 times the floor,
the case the timer now samples 60 times and labels).

## What fusing q/k/v would recover

Per layer, today's launches as nsys sees them inside the token against the fused launch less floor (the less-floor
isolated time matched nsys within 4% on every role from 3 MB up in the earlier record; the fused shapes are 12 to
14 MB):

| layers | today, in-model | fused, less floor | gain per layer | gain per token |
|---|---|---|---|---|
| 18 with v in Q4_K: q + k + v, 8.02 + 3.13 + 3.13 | 14.28 µs | 9.89 µs (one Q4_K launch, 14.16 MB) | 4.39 µs | 0.079 ms |
| 18 with v in Q6_K: q + k, 8.02 + 3.13 (v stays its own launch) | 11.15 µs | 9.01 µs (one Q4_K launch, 11.80 MB) | 2.14 µs | 0.039 ms |
| 54 launches fewer at the measured 0.21 µs of idle per launch (0.163 ms over 762) | | | | 0.012 ms |
| total | | | | 0.129 ms |

Measured against measured (today's launches also less floor from Table 1, 14.42 and 11.01 µs) gives the same total,
0.129 ms. So:

**Fusing q/k/v would recover at most 0.13 ms of the 3.98 ms token (3.2%): 251.2 to about 259.6 tok/s, 8 tok/s.** The
attention projections' whole excess over their ideal is 0.31 ms (0.82 measured, 0.51 at peak), and one launch
recovers four tenths of it, because a 14 MB read is still at 85% of peak and an 11.8 MB read at 77%. The k+v Q6_K
fusion the brief named would have recovered 2.5 µs per layer if the model had such layers; it has none.

"At most" because the fused time is the kernel alone after a read sweep with the event floor taken off, which is
the number that matched nsys per role before; inside the token the fused launch would also follow a norm and feed
RoPE and attention, as the three do today, and nothing here says that costs less or more.

## Table 2: the same roles at a batch of 2 and 4 (the engine's ncols_dst 2 and 4 kernel, kernel timer)

The same weight bytes per call, 2 or 4 q8_1 vectors and output columns. Median of three sets; less floor in
parentheses; the last column is the per-call time less floor divided by the batch, the cost per useful token.

| role | bytes/call | ncols 1 µs (less floor) | ncols 2 | ncols 4 | per call at 2 / 4 | per useful token at 1 / 2 / 4 |
|---|---|---|---|---|---|---|
| attn_kv Q4_K | 2.36 MB | 7.58 (3.41) | 7.70 (3.71) | 7.82 (3.76) | +1% / +3% | 3.41 / 1.86 / 0.94 |
| attn_kv Q6_K | 3.44 MB | 7.97 (4.10) | 9.20 (5.31) | 8.91 (4.98) | +15% / +12% | 4.10 / 2.66 / 1.24 |
| attn_qo Q4_K | 9.44 MB | 11.62 (7.60) | 11.57 (7.78) | 12.43 (8.53) | 0% / +7% | 7.60 / 3.89 / 2.13 |
| ffn_down Q4_K | 28.31 MB | 22.62 (18.62) | 24.08 (20.10) | 26.19 (22.26) | +6% / +16% | 18.62 / 10.05 / 5.56 |
| ffn_down Q6_K | 41.29 MB | 32.43 (28.45) | 33.92 (29.74) | 35.79 (31.76) | +5% / +10% | 28.45 / 14.87 / 7.94 |
| ffn_gate_up Q4_K | 28.31 MB | 22.27 (18.06) | 23.39 (19.42) | 24.02 (20.08) | +5% / +8% | 18.06 / 9.71 / 5.02 |
| lm_head Q6_K | 510.50 MB | 305.33 (301.04) | 306.70 (302.83) | 310.85 (306.98) | 0% / +2% | 301.04 / 151.42 / 76.74 |

Sums per step, less floor: weight kernels 3.254 / 3.454 / 3.632 ms at 1 / 2 / 4 streams (+6%, +12%), which is 3.254 /
1.727 / 0.908 ms per useful token. The attention projections: 0.805 / 0.856 / 0.907 ms per step, 0.805 / 0.428 /
0.227 ms per token.

## Table 3: the whole step at 1, 2 and 4 streams (llama-server, median of 3)

| streams | ms per step (3 repeats) | tok/s per stream | tok/s in total | weight kernels per step, Table 2 | the rest of the step |
|---|---|---|---|---|---|
| 1 | 4.022 (4.025 / 4.022 / 4.022) | 248.6 | 248.6 | 3.254 | 0.77 |
| 2 | 4.628 (4.664 / 4.520 / 4.628) | 216.2 | 432.4 | 3.454 | 1.17 |
| 4 | 5.931 (6.046 / 5.815 / 5.931) | 168.8 | 675.4 | 3.632 | 2.30 |

The batch-1 step through the server (4.02 ms) is within 1% of llama-bench's token (3.98). The weight kernels grow
6% and 12% while the step grows 15% and 47%: the rest of the step (attention over 4 KV caches, norms, RoPE, the
sampler and the server's per-slot work, launches) grows from 0.77 to 2.30 ms, and that, not the weight kernels, is
what keeps batch 4 at 675 rather than 900 tok/s.

## What it means

- The small roles' loss is real but small to recover by fusion: 0.13 ms of 3.98, 8 tok/s. The lever is one engine
  change (a q+k+v projection as one tensor, as gate+up already is) for 3%.
- **Batch 2 recovers 0.38 ms of the small roles per token (0.805 to 0.428) and 1.53 ms of the weight kernels per
  token; the whole step at batch 2 is 4.63 ms for two tokens, 2.31 ms per token, 432 tok/s in total.** Batch 4 brings
  the small roles to 0.23 ms per token and the step to 1.48 ms per token (675 tok/s), with the rest of the step, not
  the weights, now the larger share.
- The 7 roles' isolated less-floor sum is 0.99 of the in-model weight sum, so the timer's generic path on this GPU
  stands as the per-role cross-check without nsys, and the fused shapes it timed are in the size range where it
  matched nsys before.

## Not measured

- A fused launch inside the token (nsys of an engine with the fusion): no engine was changed; the bound is from
  the kernel alone.
- Batch 2 and 4 per role inside the token (`--role-time in-model --batch 2` needs `llama-batched-bench`, which this
  box's build does not have); the per-role batch numbers are the kernel alone.
- The server's step includes its sampler and slot bookkeeping; a batched-bench step would be a little shorter.
- tinygrad on the 5090; `halve_iters` on this GPU (the engine's rule picks it only on the DGX Spark table; see
  kernel-timer.md).

## Sources

Prediction (with its post-run note), the probe script, every result JSON (`small-roles.json`, `batched.json`,
`llama-bench-b.json`) and the probe log are kept on the Air under the scratchpad `small-roles/` folder of this
session and on the box under `~/storage/ubuntu-cmp/b4/`. Timings: kernel timer probe 93 s of GPU (10 shapes x 3 sets
+ 7 roles x 2 widths x 3 sets, 60 samples each), llama-bench 4 s, server 7 s of requests after a 4 s load.
