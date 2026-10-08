# Lifecycle comparison: ours vs vLLM (NVIDIA-Nemotron-3-Nano-4B-BF16 (RTX 5090))

schema `boltbeam.lifecycle_comparison.v1`

| workload | side | wall ms/step | kernel-sum ms | lifecycle ms | launches/step | graphs/step |
|---|---|---|---|---|---|---|
| decode_b8 | ours | 8.022 | 7.610 | 0.412 | 737 | 5.09 |
| decode_b8 | vllm | 6.508 | 6.414 | 0.094 | 306 | 1.00 |
| decode_b32 | ours | 11.692 | 11.262 | 0.431 | 867 | 5.09 |
| decode_b32 | vllm | 8.813 | 8.654 | 0.159 | 352 | 1.00 |
| decode_b64 | ours | 18.724 | 18.282 | 0.442 | 821 | 5.09 |
| decode_b64 | vllm | 13.086 | 12.984 | 0.102 | 352 | 1.00 |
| decode_b128 | ours | 32.122 | 31.407 | 0.716 | 817 | 5.09 |
| decode_b128 | vllm | 21.914 | 21.755 | 0.159 | 348 | 1.00 |
| prefill_10k | ours | 1645.943 | 1641.799 | 4.144 | 14980 | 753.00 |
| prefill_10k | vllm | 376.630 | 369.635 | 6.994 | 1012 | 0.00 |

## decode_b8 (B=8, P ours=200 / vLLM=200)

- wall gap +1.514 ms = kernels +1.196 (GEMM -0.064, non-GEMM +1.260) + lifecycle +0.318 ms; launches +431/step, graph launches +4.09/step

| term | ours ms | vLLM ms | delta ms |
|---|---|---|---|
| ssm_in | 1.649 | 1.516 | +0.133 |
| ssm_out | 0.887 | 0.958 | -0.071 |
| attn_qkv | 0.248 | 0.130 | +0.117 |
| attn_o | 0.118 | 0.150 | -0.032 |
| ffn_up | 1.012 | 0.900 | +0.112 |
| ffn_down | 1.018 | 1.379 | -0.361 |
| output | 0.526 | 0.489 | +0.037 |
| (attention) | 0.485 | 0.085 | +0.400 |
| (mamba) | 1.226 | 0.523 | +0.703 |
| (mlp_act) | 0.000 | 0.009 | -0.009 |
| (norm_residual) | 0.199 | 0.072 | +0.127 |
| (sampling_bookkeeping) | 0.205 | 0.201 | +0.004 |
| (state_flush) | 0.037 | 0.000 | +0.037 |
| lifecycle (wall - kernel sum) | 0.412 | 0.094 | +0.318 |

## decode_b32 (B=32, P ours=200 / vLLM=200)

- wall gap +2.879 ms = kernels +2.607 (GEMM +0.751, non-GEMM +1.856) + lifecycle +0.272 ms; launches +515/step, graph launches +4.09/step

| term | ours ms | vLLM ms | delta ms |
|---|---|---|---|
| ssm_in | 1.662 | 1.556 | +0.107 |
| ssm_out | 0.892 | 0.742 | +0.150 |
| attn_qkv | 0.203 | 0.133 | +0.070 |
| attn_o | 0.133 | 0.100 | +0.032 |
| ffn_up | 1.204 | 1.026 | +0.177 |
| ffn_down | 1.151 | 0.926 | +0.225 |
| output | 0.523 | 0.533 | -0.009 |
| (attention) | 0.585 | 0.180 | +0.405 |
| (mamba) | 3.269 | 3.010 | +0.259 |
| (mlp_act) | 0.000 | 0.012 | -0.012 |
| (norm_residual) | 0.769 | 0.084 | +0.685 |
| (sampling_bookkeeping) | 0.711 | 0.352 | +0.359 |
| (state_flush) | 0.160 | 0.000 | +0.160 |
| lifecycle (wall - kernel sum) | 0.431 | 0.159 | +0.272 |

## decode_b64 (B=64, P ours=2000 / vLLM=2000)

- wall gap +5.638 ms = kernels +5.297 (GEMM +3.309, non-GEMM +1.988) + lifecycle +0.340 ms; launches +469/step, graph launches +4.09/step

| term | ours ms | vLLM ms | delta ms |
|---|---|---|---|
| ssm_in | 2.550 | 1.560 | +0.990 |
| ssm_out | 1.552 | 0.761 | +0.791 |
| attn_qkv | 0.254 | 0.136 | +0.118 |
| attn_o | 0.201 | 0.100 | +0.101 |
| ffn_up | 1.645 | 0.950 | +0.696 |
| ffn_down | 1.452 | 0.934 | +0.518 |
| output | 0.632 | 0.536 | +0.096 |
| (attention) | 1.706 | 0.936 | +0.770 |
| (mamba) | 6.167 | 6.662 | -0.496 |
| (mlp_act) | 0.000 | 0.018 | -0.018 |
| (norm_residual) | 0.655 | 0.078 | +0.577 |
| (sampling_bookkeeping) | 1.145 | 0.314 | +0.831 |
| (state_flush) | 0.325 | 0.000 | +0.325 |
| lifecycle (wall - kernel sum) | 0.442 | 0.102 | +0.340 |

## decode_b128 (B=128, P ours=2000 / vLLM=2000)

- wall gap +10.208 ms = kernels +9.651 (GEMM +6.926, non-GEMM +2.725) + lifecycle +0.557 ms; launches +469/step, graph launches +4.09/step

| term | ours ms | vLLM ms | delta ms |
|---|---|---|---|
| ssm_in | 4.368 | 1.854 | +2.514 |
| ssm_out | 1.845 | 0.944 | +0.900 |
| attn_qkv | 0.321 | 0.199 | +0.122 |
| attn_o | 0.250 | 0.109 | +0.141 |
| ffn_up | 2.530 | 1.141 | +1.389 |
| ffn_down | 2.289 | 1.087 | +1.202 |
| output | 1.223 | 0.565 | +0.658 |
| (attention) | 2.700 | 1.632 | +1.068 |
| (mamba) | 12.145 | 13.665 | -1.520 |
| (mlp_act) | 0.000 | 0.032 | -0.032 |
| (norm_residual) | 0.894 | 0.092 | +0.802 |
| (sampling_bookkeeping) | 2.339 | 0.434 | +1.904 |
| (state_flush) | 0.503 | 0.000 | +0.503 |
| lifecycle (wall - kernel sum) | 0.716 | 0.159 | +0.557 |

## prefill_10k (B=1, P ours=10000 / vLLM=10000)

- warning: wall source differs: ours = host (synchronized perf_counter around the last rep; HCQ replay timestamps are not comparable across graph launches); vLLM = GPU timeline
- wall gap +1269.313 ms = kernels +1272.164 (GEMM +868.155, non-GEMM +404.009) + lifecycle -2.850 ms; launches +13968/step, graph launches +753.00/step

| term | ours ms | vLLM ms | delta ms |
|---|---|---|---|
| ssm_in | 446.348 | 107.486 | +338.862 |
| ssm_out | 235.806 | 48.578 | +187.228 |
| attn_qkv | 31.660 | 8.658 | +23.002 |
| attn_o | 29.267 | 6.410 | +22.857 |
| ffn_up | 213.643 | 61.857 | +151.786 |
| ffn_down | 210.680 | 65.198 | +145.482 |
| output | 0.000 | 1.061 | -1.061 |
| (attention) | 71.003 | 23.753 | +47.249 |
| (mamba) | 376.361 | 37.370 | +338.991 |
| (mlp_act) | 0.000 | 4.310 | -4.310 |
| (norm_residual) | 27.032 | 4.599 | +22.432 |
| (sampling_bookkeeping) | 0.000 | 0.354 | -0.354 |
| lifecycle (wall - kernel sum) | 4.144 | 6.994 | -2.850 |

## Notes

- ours = tinygrad-self-training exp (DEV=NV, HCQ graph profile, PROFILE=1), vLLM 0.30 (nsys, cuda-graph-trace=node); ours traces 2026-09-26 via bench/spec/ours_prof.py / ours_pf_prof.py (host wall/step printed in each meta).
- vLLM B=32 trace was produced 2026-09-26 with VLLM_CACHE_ROOT pointed at a fresh dir (the default cache holds a root-owned flashinfer autotune file that makes the engine fail at warmup).
- ours relu2 is fused into ffn_down's input prep (counted in ffn_down); vLLM's relu2 kernel is mlp_act.
- norm_residual includes the kernels before the first GEMM of a step (embedding, first norm); sampling_bookkeeping = everything after the output GEMM (sampler, vLLM input prep and mamba align copies).
- ours decode steps carry the amortized state flush (state_flush) that runs every ~16 steps.
