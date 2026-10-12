# Cross-check: vLLM against llama.cpp, the 27B hybrid, RTX 5090 (2026-10-11)

Step 2 of the vLLM cross-check. Step 1 was the 8B (prefill-vllm-crosscheck-8b-rtx5090-20261011.md). The same model
on the same GPU must give the same work per row. Only times, and ceilings that depend on a format or a path, may
differ.

## The two checkpoints

| engine | checkpoint | weights | size |
|---|---|---|---|
| llama.cpp b9592 | the 27B Q4_K_M GGUF the 10-11 audit used | Q4_K FFN; Q8_0 attention and recurrent projections; Q6_K output projection and head | 17.7 GB |
| vLLM 0.31.0 | a W4A16 AWQ checkpoint of the same base model (philbert440 on Hugging Face), compressed-tensors | INT4, groups of 128, bf16 scale and int4 zero point (4.156 bits per weight); in_proj_a/b, the head and the embedding in BF16 | 19.55 GB |

Both are the same hybrid: 64 layers, 16 attending (24 heads, 4 KV heads, head dim 256) and 48 Gated DeltaNet
(state 128, 16 key heads, 48 value heads, inner 6144, conv width 4). The AWQ config lists `linear_attention` x 3 then
`full_attention` per 4 layers. The checkpoint is kept on the box at
`~/storage/bbeng/models/` (19.55 GB, sha256 checked against Hugging Face).

The formats differ. The GGUF's attention and recurrent matrices are Q8_0, the FFN Q4_K. The AWQ checkpoint has
them all in INT4 except in_proj_a/b. So bytes differ by format; operations must not.

## What ran

One `pipeline --workload prefill --ctxs 512,2048 --analyze --measure auto --no-search` per engine, then one
`--workload decode --ctxs 512` per engine (batch 1), all under this change's code.

- llama.cpp: llama-bench's own repetitions, ubatch 512, n_batch 2048: 1 chunk at 512, 4 at 2048.
- vLLM: its own request timing, median of 3 after 2 warm-up calls. max_num_batched_tokens is length + 8, chunked
  prefill on: 1 chunk at both lengths.

Two extra runs, and why:
- A first vLLM and llama.cpp prefill run exposed the bugs below. The final runs repeat them on the fixed code.
- The final llama.cpp truth at 512 was disturbed: samples 322, 261, 245 ms. The first run under the same timing
  code gave 139, 125, 125 ms (median 124.9). The 10-11 audit gave 127.8, and this run's own capture is busy for
  121 ms. The table uses 124.9 ms. The likely cause is the GPU coming off the vLLM run just before it (mean
  clock 1,270 MHz over the run's window, including load).

## Prefill work per row: every row set matches exactly

Operations per row set, from each final run's prefill_trace.json. Row names differ by engine:
- llama.cpp groups roles by kernel time.
- vLLM packs q, k and v (qkv_proj), the delta net's qkv and z (in_proj_qkvz), b and a (in_proj_ba), and gate and
  up (gate_up_proj).

So the rows are compared as sets. "Matrix rows" is every projection and FFN row.

| length | row set | llama.cpp Q4_K_M | vLLM AWQ | verdict |
|---|---|---|---|---|
| 512 | matrix rows | 2.493497e13 | 2.493497e13 | equal |
| 512 | attention (16 layers) | 5.164027e10 | 5.164027e10 | equal |
| 512 | recurrent (48 layers) | 1.373047e11 | 1.373047e11 | equal |
| 512 | lm_head | 2.542797e9 | 2.542797e9 | equal |
| 2048 | matrix rows | 9.973988e13 | 9.973988e13 | equal |
| 2048 | attention | 8.250364e11 | 8.250364e11 | equal |
| 2048 | recurrent | 5.492189e11 | 5.492189e11 | equal |
| 2048 | lm_head | 2.542797e9 | 2.542797e9 | equal |

- Matrix rows equal the profile's every-token work in both engines: projections 7.411503e12 plus FFN 1.752347e13
  at 512.
- llama.cpp does not prune the last layer's FFN on this model, unlike the 8B. The model's last layer is a full
  attention layer, but the matrix rows show no 1-token FFN, so the engines agree here.
- The recurrent row uses one work model for both engines: 7 x S^2 x H operations per token per layer (S = 128,
  H = 48), plus the convolution (2 x 4 x 10240 per token per layer). Its kernels are:
  - llama.cpp: `gated_delta_net_cuda` and `ssm_conv_long_token_f32`, 96 launches.
  - vLLM: FlashInfer's chunked delta-rule kernels (four per layer) and `_causal_conv1d_fwd_kernel`, 240 launches.
- The recurrent work is an engine difference that the model does not count. vLLM's chunked form (WY form, chunks
  of 64 tokens, tensor cores) does more arithmetic than the 7 x S^2 x H recurrence: about 6 S^2 + 10 x 64 x S
  per token per head, against 7 S^2. BoltBeam counts the recurrence's work for both. It does not model the
  chunked algorithm.
- Attention pairs: 512 x 513 / 2 and 2048 x 2049 / 2 per head in both runs.

Bytes differ, as they should:
- Weights: Q4_K, Q8_0 and Q6_K against INT4 g128 and BF16.
- Activations: q8_1 in and fp32 out for mul_mat_q, against bf16 in and out for vLLM.
- The recurrent row's state traffic: one launch per chunk, so 4 state reads at 2048 for llama.cpp and 1 for vLLM
  (1.574e10 against 1.484e10 bytes).
- Marlin's weight re-reads (below).

## Who is faster: prefill

Measured ms is the untraced prefill. Roofline ms is the sum over the rows of max(operations / path peak, bytes /
bandwidth). It uses each capture's peaks at the clock the capture held, and 1,693.4 GB/s. The script is cmp27.py
in the evidence folder.

The paths:
- llama.cpp's mul_mat_q runs on the int8 path (996 TFLOPS at the probe clock).
- vLLM's Marlin dequantizes INT4 to bf16 and runs bf16 HMMA (251.5 TFLOPS), read from its machine code.
- Two vLLM kernels have no readable code. Since 2026-10-11 their paths come from their names, labelled "from the
  kernel name" (see the section below):
  - The cuBLAS BF16 GEMM (in_proj_ba) runs on bf16 (`tensorop_bf16_s16816gemm`).
  - FlashInfer's delta-rule kernels run on fp16/bf16 tensor cores with fp32 accumulate. FlashInfer 0.7.0.post1
    builds `warp.MmaF16BF16Op(dtype, Float32, (16, 8, 16))` in delta_rule_cp_sm120.py (lines 436, 872, 2074,
    3837). Its SIMT fixup (line 2463) has no matrix instruction. The row stays bound by its bytes.

| engine | format | length | chunk | measured ms | roofline ms | % of roofline |
|---|---|---|---|---|---|---|
| vLLM | AWQ INT4 g128 | 512 | 1 x 512 | 133.49 | 103.26 | 77.4 |
| llama.cpp | Q4_K_M | 512 | 1 x 512 (ubatch 512) | 124.86 (first run, see above) | 29.00 | 23.2 |
| vLLM | AWQ INT4 g128 | 2048 | 1 x 2048 | 503.37 | 414.25 | 82.3 |
| llama.cpp | Q4_K_M | 2048 | 4 x 512 (ubatch 512) | 503.42 | 119.51 | 23.7 |

- In wall time the two engines tie: llama.cpp is 6% faster at 512, and they are equal at 2048.
- vLLM sits much closer to its roofline. Its roofline is 3.5 times higher because its matrix path is bf16. Marlin
  turns 4-bit weights into bf16 and multiplies at the bf16 rate. llama.cpp multiplies in int8 at 4 times that
  rate.
- llama.cpp's int8 path leaves most of its ceiling unused: 23%. Its rows run at 28 to 31% of the int8 peak (the
  10-11 audit), and the recurrent row takes 18.8 ms at 512 against vLLM's 7.2 ms.
- The delta net: vLLM's chunked tensor-core kernels take 7.2 ms at 512 and 15.2 ms at 2048. llama.cpp's
  recurrence takes 18.8 and 76.8 ms. That is 2.6 and 5.0 times slower, growing with the prompt.
- Marlin runs each product of 2048 rows as two launches of 1024 (512 launches for 256 products). Each launch
  reads the weights again. That is in its bytes.
- Every run flags the software power cap in some samples over the whole process. The captures' own windows held
  2,867 to 2,940 MHz.

## Decode, batch 1, context 512

Measured tok/s is each engine's untraced decode at batch 1 after 512 tokens of context.
- llama.cpp: llama-bench -n.
- vLLM: N tokens minus 1 token, CUDA graphs on.

Roofline tok/s is the decode limit: the bytes one token reads over 1,693.4 GB/s (screen.ceiling at context 1:
weights, no KV). The 8B rows use the BF16 safetensors for vLLM. For llama.cpp they use the F16 GGUF already on
the box. Converting a new BF16 GGUF would have taken 16 GB. F16 and BF16 are both 2 bytes per weight, so the bytes
are the same.

| engine | model | format | measured tok/s | roofline tok/s | % of roofline |
|---|---|---|---|---|---|
| vLLM | 27B | AWQ INT4 g128 + BF16 head | 85.60 | 111.09 | 77.1 |
| llama.cpp | 27B | Q4_K_M | 71.38 | 92.78 | 76.9 |
| vLLM | 8B | BF16 | 100.14 | 111.83 | 89.5 |
| llama.cpp | 8B | F16 | 100.21 | 111.83 | 89.6 |

Bytes per token per row (the decode work), from each run's profile:

| row | 27B llama.cpp | 27B vLLM | 8B llama.cpp F16 | 8B vLLM BF16 |
|---|---|---|---|---|
| ffn_gate_up | 6.423e9 Q4_K | 5.933e9 INT4 | 7.250e9 | 7.250e9 |
| ffn_down | 3.212e9 Q4_K | 2.966e9 INT4 | 3.625e9 | 3.625e9 |
| attn_qkv (recurrent q, k, v) | 2.675e9 Q8_0 | 1.309e9 INT4 | | |
| attn_gate + ssm_out | 3.211e9 Q8_0 | 1.571e9 INT4 | | |
| attn_qo | 1.070e9 Q8_0 + 4.132e8 Q6_K | 5.235e8 + 2.618e8 INT4 | 2.417e9 | 2.417e9 |
| attn_kv | 1.787e8 Q8_0 | 8.756e7 INT4 | 6.047e8 | 6.047e8 |
| ssm_alpha_beta | 2.606e7 Q8_0 | 4.818e7 BF16 | | |
| lm_head | 1.043e9 Q6_K | 2.543e9 BF16 | 1.245e9 | 1.245e9 |
| total | 1.825e10 | 1.524e10 | 1.514e10 | 1.514e10 |

- The 8B rows are equal byte for byte, and so are the speeds (100.1 against 100.2 tok/s).
- The 27B rows have the same shapes and counts in both profiles. Their bytes differ only by format.
  - vLLM reads 16% fewer bytes per token: INT4 projections against Q8_0.
  - Its BF16 head reads 1.5 GB more than llama.cpp's Q6_K head.
  - It is 20% faster in tok/s, at the same 77% of its roofline.
- The vLLM 27B decode capture failed BoltBeam's tie-out guard. nsys counted 12.74 ms of GPU time per token
  against 11.47 ms of wall time per token. So it has no per-kernel decode table here. Its untraced time and the
  limit stand.
  - Cause (2026-10-11, from the saved capture): not overlapping streams. The union of kernel intervals equals
    their sum in both captures (0 µs overlap). The decode steps themselves run 11.44 ms apart with 11.25 ms busy.
    The long run's start-up runs some kernels more times than the short run's (a warm-up and graph capture sized
    by the token count). A per-kernel difference of the two captures counted those launches as decode: 52.9 ms
    over 31 tokens. The heaviest GEMM also runs at start-up at start-up clocks, so subtracting the short run's
    total carried that noise in too.
  - Fix (`attribution.difference`, `shared_window`): a kernel's extra launches are its latest launches in the long
    capture, timed as they ran. A kernel whose extra launches all ran before the step both captures made is
    start-up, and is left out (`startup_us`). The overlap is reported as its own figure (`overlap_us`). The guard
    is unchanged.
  - After: 11.03 ms of GPU time per token against 11.47 ms, so the capture is a measurement.
  - Packed matrices (2026-10-11): decode now pairs with the engine's own matrices through `prefill.engine_roles`,
    prefill's authority, and gives a packed GEMM's time back to its parts by weight bytes. The vLLM driver prints
    them in capture mode too. The saved capture predates that, so it was re-attributed with the matrices the same
    model's prefill capture printed.
  - Marlin launches every matrix on one grid, so one kernel serves several matrices. Roles left unpaired are now
    placed as prefill places roles on a shared kernel (`attribution.shared_kernels`, using `prefill_attribution.pair`
    and `split`): each kernel's launches in duration clusters, in the order of the matrices' size.
  - That left a tie: ffn_down (64 a token) and in_proj_qkvz + qkv_proj (48 + 16) fit either Marlin kernel by counts,
    bytes and rates. Launch order breaks it. `vendor_capture.sequence` keeps each capture's launches in order per
    stream. `vocab.LAYER_STAGE` ranks the fine role classes in a layer's data flow (input projections, output
    projection, FFN in, FFN out, head). Among assignments tied after every other rule, the one whose launches
    rise through the stages within a layer wins (`prefill_attribution.order_violations`). It never overrides a rule.
    A tie order cannot break stays flagged.
  - Result: every role pairs. Per layer the stream runs in_proj_qkvz or qkv_proj, out_proj or o_proj, gate_up, then
    down, and the pairing matches that order. The re-attributed prefill captures (512 tokens here, and the deep
    llama.cpp turn) had no tie, and are unchanged.

## What BoltBeam got wrong, and the fixes in this change

1. **The safetensors profile could not read this checkpoint.**
   - It ignored `text_config`, so hidden size, FFN and layers were empty.
   - It counted packed int32 words, scales and zero points as three roles each.
   - It dropped the delta net's projections or called them attention (out_proj).
   - It counted the MTP draft head.
   - It had no recurrent shape.
   - Fix (profile/loaders.py, profile/roles.py): read text_config; fold a pack-quantized weight into one role
     labelled `INT4_G128_ZP`; skip the draft head and the vision encoder; map in_proj_qkv, in_proj_z, in_proj_a/b
     and linear_attn.out_proj to the GGUF's attn_qkv, attn_gate, ssm_alpha_beta and ssm_out; read the recurrent
     shape from config.json; orient ssm tensors rows first, as the GGUF reader does.
   - The profile now equals the GGUF's row for row.
   - bits_per_weight counts a group label's scale and zero point (4 + 20 / 128 bits).
2. **The vLLM driver skipped quantized layers.** A compressed-tensors linear layer has no `weight`.
   - Fix: the driver reports such a layer by its own input and output sizes. It names each packed part under its
     parent module, so linear_attn.out_proj is not taken for self_attn.o_proj.
3. **vLLM's delta net was counted as attention.** FlashInfer names it
   `kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_...`, and "flash" made it attention (208 attention
   launches, not 16).
   - Fix: `delta_rule` and `causal_conv1d` name recurrent kernels, before "flash".
   - Recurrent templates, like attention ones, serve no matrix role.
4. **The pairing search gave up.** It found more than 2,000,000 ways to pair 8 roles with 29 templates, because
   many Triton kernels with unread code stay candidates.
   - Fix: an option a role breaks on its own is dropped before the search. Those options read its weights faster
     than the bandwidth, or exceed its path's peak. Adding roles to a template only adds bytes and operations
     over the same time, so this prunes nothing the search could accept.
5. **Step 1's one-rate rule was too strict.** It broke the llama.cpp 27B pairing that the 10-11 audit had. The Q8_0
   template's attn_kv launches (N = 1024) run at 0.235 operations per ns. The bigger roles run at 0.30 to 0.32.
   That is 1.38 times apart, over the 1.3 gap.
   - A small GEMM fills fewer SMs, so it is slower per operation, and that is physical.
   - Fix: a cluster may be slower than a cluster with more operations per launch, never faster beyond the gap.
     This keeps step 1's purpose. Step 1's tests still pass. The 8B captures were not re-attributed.
6. **Marlin cuts a product along M.** At 2048 it runs each product as two launches of 1024 rows. The template had
   twice its roles' launches and was refused.
   - Fix: a shared template's launches may be a whole multiple p of its roles' launches, where p divides the
     chunk. Each launch does M / p rows and reads the weights again.
7. **Triton kernels had no readable code.** So a Triton SiLU kernel took the BF16 in_proj_ba role by count and time,
   instead of the cuBLAS GEMM beside it.
   - Fix: the driver reports the JIT caches vLLM set in its process (TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR).
     library_paths reads each captured kernel's `<name>.ptx` there. Triton kernels now read as "no matrix
     instruction", and in_proj_ba pairs with the cuBLAS kernel.
8. **The open item from step 1: llama.cpp's clock window included model load.**
   - Fix: vendor_capture.busy_window reads nsys's session start on the monotonic clock (systemClockNs in the
     export's TARGET_INFO_SESSION_START_TIME). It spans the first to the last captured kernel. That is the
     clock time.monotonic() uses on Linux.
   - At 512 the capture now reads 2,940 MHz, where the whole process averaged 1,041 MHz.

Still open:
- (Fixed 2026-10-11, see the replay doc's Open section.) The deep llama.cpp prefill captures of the replay leave
  `mul_mat_q` unpaired, and the deep decode capture leaves the FFN gate and up roles unpaired.
- The chunked delta rule's extra arithmetic (2026-10-11): resolved, see "The chunked delta rule's executed
  arithmetic" below. The formula now matches ncu exactly.
- The vLLM 27B decode tie-out guard failure, above (fixed; the packed-matrix decode pairing is open).
- For a model folder, the decode probe says "the model file is gone" (measure_status.json probe_reason). The
  probe looks for a file, not a folder. It did not block the run.

## Kernel-name paths, re-attributed from the saved captures (2026-10-11)

No GPU ran. The saved vLLM nsys exports were re-attributed with the current code. Each capture kept the clock it
had saved.

| length | roofline ms before | after | measured ms | % of roofline before | after |
|---|---|---|---|---|---|
| 512 | 100.76 | 103.26 | 133.49 | 75.5 | 77.4 |
| 2048 | 404.85 | 414.25 | 503.37 | 80.4 | 82.3 |

- Resolved: in_proj_ba (bf16 from the kernel name) and the recurrent row (the delta-rule kernels, fp16 from the
  kernel name). No refused ceiling is left in the vLLM prefill traces.
- The "after" figures equal the table above, which had assumed these paths by hand. The recurrent row stays bound
  by its bytes, so the tensor-core path does not change its ceiling.

## The chunked delta rule's executed arithmetic (2026-10-11)

No GPU ran. The count comes from FlashInfer 0.7.0.post1's public source and the saved ncu export.

The ceiling still counts the work the model needs: 7 x S^2 x H per token per layer, the same on every engine. The
recurrent row also carries `executed_flops`: what the engine's named algorithm executes
(`prefill_work.chunked_wy_ops`). It does not change the ceiling.

The algorithm: the delta rule in chunks of C = 64 tokens with the WY / UT transform, run chunk-parallel. The prompt
is cut into spans. It runs as four kernels:
1. The UT system: K K^T per chunk, then its triangular inverse (collective_inverse_hmma.py).
2. Each span's transfer matrix and local state, chunk by chunk (delta_rule_cp_sm120.py, MN precompute).
3. The fixup: a scan over spans that fixes the state at each span's start, in tf32 (one S x S x S product a span).
4. The output pass: every chunk again, from the fixed state.

Every matrix product, per chunk per head (state S = 128, each product 2 m n k operations):

| product | shape | kernel |
|---|---|---|
| K K^T | C x C x S | 1 |
| X = T K | S x C x C | 2 |
| transfer: Z = M K^T, then M + Z X | S x C x S, S x S x C | 2 |
| state: Y = S K^T, then S + Y X | S x C x S, S x S x C | 2 |
| Q K^T | C x C x S | 4 |
| output from the state, the state against K | C x S x S, S x C x S | 4 |
| the new values | S x C x C | 4 |
| the intra-chunk output | S x C x C | 4 |
| the state update | S x S x C | 4 |

- That is five C x C x S products and seven C x S x S products per chunk.
- The inverse doubles from 8 x 8 diagonal blocks. The 8 x 8 blocks are inverted without matrix instructions. At
  size n there are C / 2n blocks, each with two n x n x n products. The 8 x 8 products run on m16n8k8 and pad 8
  rows up to 16, so that level costs twice its arithmetic. Per chunk: 16,384 + 32,768 + 131,072 = 180,224.
- Each span's first chunk starts from the identity and zero, so it skips two C x S x S products.
- Each span adds one S x S x S product in the fixup. Its 3xTF32 option is off by default, so one pass.
- Tokens round up to whole chunks. Every other dimension is a multiple of the 16 x 8 x 16 tile, so no other padding.

Per head, with B chunks and P spans:

    B (10 C^2 S + 14 C S^2 + inverse(C)) - 4 P C S^2 + 2 P S^3

Against ncu, one layer at 512 tokens (48 heads, 40 launches of each kernel):

| kernel | HMMA instructions (40 launches) | operations per layer | formula per layer |
|---|---|---|---|
| UT system and inverse | 3,932,160 + 614,400 m16n8k16, 122,880 m16n8k8 | 465,567,744 + 6,291,456 | 471,859,200 |
| span transfer and state | 33,423,360 m16n8k16 | 3,422,552,064 | 3,422,552,064 |
| fixup (tf32) | 3,932,160 m16n8k8 | 201,326,592 | 201,326,592 |
| output pass | 35,389,440 m16n8k16 | 3,623,878,656 | 3,623,878,656 |
| total | | 7,719,616,512 | 7,719,616,512 |

- The formula matches ncu exactly: the difference is 0. The tolerance is exact equality, and a dev test pins it.
- The old estimate (6 S^2 + 10 C S per token per head, 4.43e9) left out the span pass, which runs the
  recurrence a second time, and the inverse.
- The span count P comes from the engine's span choice (the device's SM count and the head count). The adapter
  may name it as `span`; without it P is 1. At 512 the source rounds spans up to 512, so P is 1, and ncu agrees:
  one skipped first chunk per head. When S = 2C, as here, the two span terms cancel, so P does not change the count.

## Evidence

On the Air: `~/env/boltbeam-runs/.work/prefill-vllm-crosscheck-27b-rtx5090-20261011/`. It holds:
- The final runs: p27-vllm, p27-llama, d27-vllm, d27-llama, d8-vllm, d8-llama. Each has its traces, profile and
  summary.
- `first-run/`: the first prefill runs, whose llama.cpp 512 truth the table uses.
- `cmp27.py`, which computed the prefill tables.
- `steps-vx27.sh`, the run script.
- `reattr.py`, which re-attributed saved captures while fixing.

The nsys reports stayed on the box, under `~/storage/bbeng/tmp/vx27-final/`.
