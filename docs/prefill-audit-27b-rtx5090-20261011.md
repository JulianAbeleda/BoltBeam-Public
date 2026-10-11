# Prefill audit: the 27B hybrid on llama.cpp, RTX 5090, 2026-10-11

The primary model is a 27B hybrid (GGUF architecture `qwen35`, Q4_K_M weights with Q8_0 and Q6_K roles). It has 64
layers: 16 attend with a KV cache and 48 are recurrent (gated delta net). One run, llama.cpp b9592, nothing else on
the GPU.

## The method

- Prompt lengths come from the model's own cap. 512 first, then x4 per step, and the binding cap last
  (`prefill_work.sweep`).
- Truth is the untraced prefill: `llama-bench -p N -n 0 -r 3` after its warm-up, the median of 3.
- The prompt is llama-bench's own: token ids from `std::rand() % n_vocab`, never seeded, so the sequence is fixed
  and each repetition draws new ids. The KV cache is cleared before every repetition, so nothing is cached.
- Clocks and power come from NVML during each length. The user cannot set clocks, so the policy is fixed power at
  the board's 600 W limit, and the clock it chose is logged.
- One captured prefill per length (nsys), attributed by counts and operations, up to 1,000,000 launches.
- Self-check: two greedy runs of `llama-cli` with seed 1. The generated text is identical; only the speed line
  differs.

## The caps

| cap | tokens | how |
|---|---|---|
| model | 262,144 | `qwen35.context_length`; the file declares no rope scaling |
| memory | 224,297 | 33.66 GB of GPU memory less 18.96 GB of weights, at 65,536 bytes of F16 KV per token |
| binding | 224,256 | memory, cut to a multiple of 512 |

65,536 bytes per token is K and V, 4 KV heads x 256 dims x 2 bytes, in the 16 attending layers only. The 48
recurrent layers add no KV per token.

The binding cap failed. llama.cpp could not create the context at 224,256 tokens. The memory cap holds nothing back
for the engine's compute buffers or the recurrent state. The longest length read is 131,072.

## The 27B prefill per length

| prompt | ms | tok/s | MFU | throttled | SM MHz mean | power W mean |
|---|---|---|---|---|---|---|
| 512 | 127.8 | 4,006 | 19.7% | 3% of samples | 2,886 | 161 |
| 2,048 | 515.0 | 3,977 | 19.9% | 32% | 2,811 | 260 |
| 8,192 | 2,112.0 | 3,879 | 20.8% | 66% | 2,694 | 430 |
| 32,768 | 9,540.2 | 3,435 | 23.4% | 89% | 2,634 | 548 |
| 131,072 | 67,671.6 | 1,937 | 24.4% (scaled) | 98% | 2,614 | 588 |
| 224,256 | not read | | | | | |

Every row is flagged throttled: the software power cap is set while the clock is under 98% of its 2,925 MHz top.
From 8K the board sits at its 600 W limit. These are fixed-power numbers.

131,072 was not captured: about 1,263,232 launches. Its MFU is scaled from the 32,768 capture. Matrix work grows
x4 and attention x16 (its causal pairs), over the untraced time.

## Pairing

GEMMs pair with roles by launch counts and operations. The recurrent layers' projections pair like any other role:

- `attn_qkv + attn_qo Q8_0`: the recurrent layers' QKV and the attending layers' Q and O share one template.
- `attn_gate + ssm_out Q8_0`: the recurrent gate and output projections.
- `ssm_alpha_beta Q8_0`: N = 48, so it runs at 2.3% MFU. It is under 1% of the time.

The delta-net kernels keep their own rows, with no ceiling: `gated_delta_net_cuda` and `ssm_conv_long_token_f32`.
The model counts no work for them yet.

## Where the time goes as the prompt grows

Share of the captured kernel time by kind:

| prompt | GEMM | attention | recurrent | other | gaps |
|---|---|---|---|---|---|
| 512 | 71.3% | 0.9% | 15.2% | 12.2% | 0.4% |
| 2,048 | 70.9% | 1.6% | 14.8% | 11.7% | 1.0% |
| 8,192 | 68.8% | 4.2% | 14.4% | 11.5% | 1.2% |
| 32,768 | 61.0% | 14.7% | 12.8% | 10.2% | 1.3% |
| 131,072 (estimate) | 35.7% | 34 to 50% | 7.5% | 6.0% | 0.8% |

- GEMMs and the recurrent kernels grow with the tokens. Their share falls as attention grows.
- Attention grows with the square of the prompt. It is 14.7% at 32K.
- At 131K there is no capture. Two estimates bound attention. Scaling 32K attention by its causal pairs gives 23.2 s
  (34%). Taking the untraced 67.7 s less everything else scaled x4 gives 33.9 s (50%).
- Attention takes over between 32K and 131K. Its share passes the GEMMs' at 131K by the second estimate and ties
  by the first.
- The GEMMs hold at 28 to 30% of the int8 peak at every captured length. Attention's MFU rises from 19% at 512 to
  44% at 32K.

## The audit at 32,768 tokens

The longest length that captures. The untraced prefill is 9,540.2 ms; the captured rows sum to 9,895.2 ms. Every
entry carries a claim, a standard, evidence, re-performance, a tie-out, materiality and a finding, with a pointer on
each number: E1 the captured rows, E2 the untraced truth, E3 the peaks and bandwidth, E4 the raw capture. The text
summary, the HTML report and the TUI print it. The top five, worst first:

| # | row | ms now | standard (ceiling) | ms lost | of prefill | prefill if fixed | finding |
|---|---|---|---|---|---|---|---|
| 1 | ffn_down + ffn_gate_up Q4_K | 3,986.7 | 1,126.2 ms: 1,121.5 TFLOP at the 996 TFLOPS int8 peak | 2,860.5 | 30.0% | 6,679.7 ms | 28.2% MFU, 3.5x its ceiling |
| 2 | gated_delta_net_cuda | 1,225.9 | none: no work counted | 1,225.9 | 12.8% | 8,314.3 ms | no ceiling yet |
| 3 | attention | 1,427.7 | 632.0 ms: 211.1 TFLOP at 334 TFLOPS over fp16 and fp16-accumulate | 795.7 | 8.3% | 8,744.5 ms | 44.3% MFU, 2.3x its ceiling |
| 4 | attn_qkv + attn_qo Q8_0 | 823.1 | 231.9 ms: 230.9 TFLOP at int8 | 591.2 | 6.2% | 8,949.1 ms | 28.2% MFU, 3.5x |
| 5 | attn_gate + ssm_out Q8_0 | 662.4 | 198.8 ms: 197.9 TFLOP at int8 | 463.6 | 4.9% | 9,076.6 ms | 30.0% MFU, 3.3x |

Then unary_gated_op (232.0 ms), the mul_mat_q fixup (272.6 ms now, 73.1 ms at its bytes), rms_norm (182.2 ms) and
quantize_mmq_q8_1 (179.6 ms). 13 rows under 1% each are one line: 345.9 ms now, 311.1 ms lost.

Re-performance: none of these is re-performed. BoltBeam's CUDA adapter times `mul_mat_vec_q` alone (the GEMV);
there is no adapter for `mul_mat_q`, `flash_attn_ext_f16` or `gated_delta_net_cuda`. Each entry says so.

## Suspects

- **The memory cap holds back nothing for the engine.** It is the GPU memory less the weights at the KV bytes per
  token. llama.cpp also needs compute buffers and the recurrent state, so the context could not be created at
  224,256. The engine's own reserve is not modelled.
- **The recurrent kernels have no ceiling.** `gated_delta_net_cuda` is 12.8% of the prefill and counts as all lost.
  The delta rule's work per token is not in the model yet.
- **131K is an estimate.** It was not captured. Its MFU and its split assume work scales by tokens and causal pairs.
  The two attention estimates differ by 10.7 s.
- **Every length throttled.** The board holds its 600 W limit from 8K; the clock mean falls from 2,886 to 2,614 MHz.
- **The capture runs longer than the truth at 32K** (9,895 against 9,540 ms). The rows sum to the capture.
- **The int8 peak** is the target's recorded fact, not re-probed in this run.

Runs: `boltbeam-runs/.work/prefill-27b-rtx5090-20261011/` on the Air (summary, traces, telemetry, the self-check;
nsys reports left on the box's data disk).
