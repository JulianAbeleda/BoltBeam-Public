# Prefill cross-check: vLLM against llama.cpp, Qwen3-8B, RTX 5090 (2026-10-11)

Why: a second engine checks BoltBeam's math. The same model on the same GPU must give the same work per row.
Only times, and ceilings that depend on a format or a path, may differ.

## What ran

| run | engine | weights | prompt lengths | chunk the engine used |
|---|---|---|---|---|
| 2026-10-10 audit (reused, not rerun) | llama.cpp (llama-bench) | GGUF Q4_K_M | 512, 2048 | ubatch 512, n_batch 2048: 1 chunk at 512, 4 chunks at 2048 |
| this doc | llama.cpp (llama-bench) | GGUF BF16 (convert_hf_to_gguf.py --outtype bf16 of the same checkpoint) | 512, 2048 | ubatch 512, n_batch 2048: 1 chunk at 512, 4 chunks at 2048 |
| this doc | vLLM 0.31.0 (offline LLM, `--prefill`) | safetensors BF16 | 512, 2048 | max_num_batched_tokens 520 and 2056 (length + 8, its default for this max_model_len), chunked prefill on: 1 chunk at both lengths |

Each run is one `pipeline --workload prefill --ctxs 512,2048 --analyze --measure auto --no-search` on the 5090 box.
Truth is untraced:
- llama.cpp: llama-bench's own repetitions.
- vLLM: its own request timing (first token less scheduled), median of 3 prompts after 2 warm-up calls.

Captures are nsys. The vLLM capture holds the measured call alone: the driver marks it with cudaProfilerStart/Stop.

The two engines did not read the same token ids:
- llama-bench draws its own ids with `std::rand()` and takes no prompt file.
- The vLLM driver draws ids from `random.Random(seed)`.

The work does not depend on the ids. This is a dense model and every token runs every layer, so the time may vary
by a little only.

## Work per row: matches, except one engine difference

Operations per row set, from each run's prefill_trace.json. llama.cpp groups roles by kernel time, and vLLM packs
q, k and v into one GEMM and gate and up into another. So the rows are compared as sets:
- projections: attn_qo + attn_kv for llama.cpp; qkv_proj + o_proj for vLLM.
- FFN: ffn_gate_up + ffn_down.

| length | row set | llama.cpp Q4_K_M | llama.cpp BF16 | vLLM BF16 | verdict |
|---|---|---|---|---|---|
| 512 | projections | 1.54619e12 | 1.54619e12 | 1.54619e12 | equal |
| 512 | FFN | 5.41166e12 | 5.41166e12 | 5.56628e12 | differs by 1.54619e11 |
| 512 | attention | 7.74604e10 | 7.74604e10 | 7.74604e10 | equal |
| 512 | lm_head | 1.24466e9 | 1.24466e9 | 1.24466e9 | equal |
| 2048 | projections | 6.18475e12 | 6.18475e12 | 6.18475e12 | equal |
| 2048 | FFN | 2.16466e13 | 2.16466e13 | 2.22651e13 | differs by 6.18475e11 |
| 2048 | attention | 1.23755e12 | 1.23755e12 | 1.23755e12 | equal |
| 2048 | lm_head | 1.24466e9 | 1.24466e9 | 1.24466e9 | equal |

The FFN difference is work the engines really do differently. It is not a BoltBeam error:
- One layer's FFN over a 512-token chunk is 2 x 512 x (2 x 12288 x 4096 + 4096 x 12288) = 1.54619e11 operations.
- llama.cpp keeps only the output rows after the last layer's attention (`inp_out_ids`). So its last-layer FFN
  runs on 1 token per ubatch, as two GEMVs (the `mul_mat_vec` rows).
- vLLM runs the last layer's FFN on every prompt token.
- At 512 that is 1 ubatch, so 1 x 1.54619e11. At 2048 it is 4 ubatches, so 4 x 1.54619e11 = 6.18475e11. Both
  equal the measured difference exactly.

BoltBeam counts only the work an engine did, so the two engines' FFN rows must differ by this.

Bytes that do not depend on the format agree as well:
- Attention query-key pairs: 512 x 513 / 2 and 2048 x 2049 / 2 per head, in every run.
- The KV cache bytes per token: 2 x 8 heads x 128 x 36 layers x 2 bytes.

Bytes that depend on the format differ, as they should. Weights are Q4_K/Q6_K or BF16. Activations are q8_1 into
mul_mat_q, bf16 into cuBLAS for llama.cpp (fp32 out), and bf16 in and out for vLLM. The KV reads also depend on
the chunking.

## Who is faster (BF16 against BF16)

Measured ms is the untraced prefill. Roofline ms is the sum over the rows of max(operations / path peak, bytes /
bandwidth), at the peaks' measured clock (cuda_compute: bf16 251.5, fp16 251.4, fp16_f16acc 497.7, int8 995.8
TFLOPS; 1694 GB/s), recomputed from each run's rows so all three use one basis.

The weight GEMM rows of both BF16 runs use the bf16 path. cuBLAS hides that code (see the next section). Since
2026-10-11 BoltBeam reads that path from the kernel's name, labelled "from the kernel name" (item 7).

| engine | format | length | chunk | measured ms | roofline ms | % of roofline |
|---|---|---|---|---|---|---|
| vLLM | BF16 | 512 | 1 x 512 | 38.30 | 29.32 | 76.6 |
| llama.cpp | BF16 | 512 | 1 x 512 (ubatch 512) | 47.36 | 28.80 | 60.8 |
| llama.cpp | Q4_K_M | 512 | 1 x 512 (ubatch 512) | 34.58 | 7.55 | 21.8 |
| vLLM | BF16 | 2048 | 1 x 2048 | 145.69 | 118.78 | 81.5 |
| llama.cpp | BF16 | 2048 | 4 x 512 (ubatch 512) | 187.94 | 115.10 | 61.2 |
| llama.cpp | Q4_K_M | 2048 | 4 x 512 (ubatch 512) | 141.59 | 32.05 | 22.6 |

- At BF16, vLLM is faster: 19% less time at 512 and 22% less at 2048.
- vLLM's roofline is a little higher because it does the last-layer FFN that llama.cpp prunes.
- The chunk matters. vLLM runs 2048 tokens as one GEMM of M = 2048. llama.cpp runs four of M = 512.
- llama.cpp BF16 also spends 19.7 ms at 2048 (4.8 ms at 512) in `convert_unary`, which converts activations
  between fp32 and bf16 around cuBLAS. vLLM keeps bf16 throughout.
- Q4_K_M on llama.cpp is faster than both BF16 runs in wall time. Its roofline is 4 times lower, since int8 runs
  at 4 times the bf16 rate, and its weights are a third the size. It runs at 22% of that roofline.
- Throttle flags over each whole run: vLLM truth at 512 and 2048, and llama.cpp BF16 at 2048, show software power
  cap in 2 to 7% of samples. Those samples include model load.

## What BoltBeam got wrong, and the fixes in this change

1. **The safetensors profile had no attention shape.** Attention rows had no work. Its rows were also split by
   role class (k and v as two rows), unlike the GGUF profile.
   - Fix: profile/loaders.py reads heads, kv heads, head dim, the attention layers and the context from
     config.json. It groups by (role, shape, quant), as the GGUF reader does.
2. **The clock for the peaks covered vLLM's start-up.** The capture's telemetry spanned the whole process, including
   minutes of load and compile at 225 MHz. The mean was 1343 MHz, so every peak was scaled to about half, and the
   attention ceiling (10.6 ms) came out above its measured time (8.3 ms).
   - Fix: the vLLM driver reports its measured call's monotonic window. role_time reads the clock over that
     window: 2827 MHz, and the ceiling is 5.0 ms.
   - The llama.cpp capture still reads the whole llama-bench process. llama-bench is GPU-bound after its model
     load, but a BF16 load is seconds long. A window for llama-bench is open work.
3. **vLLM's packed GEMMs.** One qkv launch does q, k and v. One gate_up launch does gate and up.
   - Fix: the driver reports the model's own weight matrices and their packed parts (`packed_modules_mapping`).
     prefill.engine_roles pairs launches with those matrices. It sums the parts' work and bytes, and accepts the
     matrices only when they cover every profile role's work exactly.
4. **cuBLAS kernels have no readable code.** The cutlass GEMMs vLLM and llama.cpp BF16 run come from libcublasLt,
   which ships neither PTX nor an extractable cubin for them. cuobjdump finds the name only in the host binary. ncu
   could read them, but needs admin profiling rights (RmProfilingAdminOnly=1). Since 2026-10-11 the box's counters
   are open and ncu reads them: see [ncu-instruction-mix-27b-rtx5090-20261011.md](ncu-instruction-mix-27b-rtx5090-20261011.md).
   - Fix: such a kernel may still take a role by count and time, so its work is counted. Its ceiling is refused,
     named "kernel code not found".
   - Two physical rules keep it honest. A kernel cannot read the role's weights faster than the measured
     bandwidth. Within one kernel template, every duration cluster runs at one rate (operations per time within
     GAP_RATIO).
   - Among tied assignments, the one pairing the most kernel time wins. Without this, the SiLU elementwise kernel
     tied with the qkv GEMM at 2048.
5. **A library with machine code only.** instruction_path.sass_table reads a library's machine code for this GPU
   (cuobjdump -xelf, nvdisasm). Each matrix opcode maps to the path whose probe kernel compiles to that opcode on
   this GPU. It covers vLLM's and other libraries that ship SASS without PTX. It did not help for cuBLAS here
   (item 4).
6. **llama.cpp's formats ignored float weights.** llama_formats said q8_1 activations for every model. For float
   weights llama.cpp converts activations to the weight type for cuBLAS and writes fp32. The formats now follow
   the profile's weight types.

7. **A closed kernel's path from its name (2026-10-11).** `instruction_path.NAME_PATHS` maps documented kernel-name
   patterns to a path. CUTLASS and cuBLAS names carry the element and accumulate types: `_bf16_s16816gemm` is bf16
   in with fp32 accumulate. A path read from code always wins. The row says "from the kernel name". The guards still
   apply: a name's path that would put a kernel above its peak or past the bandwidth is refused.

One difference against the 10-10 audit is older code, not a mismatch. That audit counted llama.cpp attention
activations as fp16 (1.963e9 bytes at 2048). Current code counts fp32, as llama.cpp reads them (3.171e9 bytes).
The BF16 run here uses current code.

## Kernel-name paths, re-read from the saved traces (2026-10-11)

No GPU ran. The saved traces hold each row's operations, bytes, time and the peaks at the capture's clock. The rows
refused for "kernel code not found" were recomputed with the path from the kernel's name.

| run | length | roofline ms before | after | measured ms | % of roofline before | after |
|---|---|---|---|---|---|---|
| vLLM BF16 | 512 | 1.08 | 32.51 | 38.30 | 2.8 | 84.9 |
| vLLM BF16 | 2048 | 5.77 | 121.62 | 145.69 | 4.0 | 83.5 |
| llama.cpp BF16 | 512 | 1.14 | 9.52 | 47.36 | 2.4 | 20.1 |
| llama.cpp BF16 | 2048 | 5.53 | 37.45 | 187.94 | 2.9 | 19.9 |

- Resolved in vLLM: gate_up_proj, down_proj, qkv_proj and o_proj, all bf16 from the kernel name.
- Resolved in llama.cpp: attn_qo and attn_kv.
- Still refused in llama.cpp: the grouped `ffn_down + ffn_gate_up` row. Its bf16 ceiling would be above its
  measured time, so the guard refuses it.
  - Cause (2026-10-11, from the saved trace): the clock, not the pairing. The pairing is consistent: 420 calls are
    3 matrices x 36 layers x 4 chunks, less 12 skipped. The row does 21.65 TFLOP in 102.8 ms, 210.5 TFLOPS. The
    peaks were scaled to 2,231 MHz, the whole llama-bench process's mean with its model load (lowest sample
    225 MHz): bf16 193.8 TFLOPS. At the probe's own clock the bf16 peak is 251.5 TFLOPS. The row runs at 84% of it.
  - The 27B crosscheck's item 8 fixed that clock: a llama.cpp capture now reads the clock over its kernels only.
    This capture predates that fix and has no raw export left, so it is not re-attributed. A new 8B BF16 capture
    would show the row paired with a ceiling.
- The "before" column counts no compute ceiling for the refused rows. That is why it is far below the table above,
  which assumed bf16 by hand. vLLM's "after" figures are above that table's (76.6 at 512) because they use each
  capture's own clock.

## Evidence

- Each run's prefill_trace.json, timing_trace.json and model_profile.json, and the script that computed these
  tables (cmp.py), are in ~/env/boltbeam-runs/.work/prefill-vllm-crosscheck-8b-rtx5090-20261011/ (llama-bf16,
  vllm-bf16-3). That is outside the repo, beside the 10-10 audit.
- The runs ran from a scratch clone on the box, now removed. The two earlier vLLM runs found items 1 to 4; their
  numbers are quoted in this doc.
