# Real 8B Format Validation

Date: 2026-07-01

## Purpose

Validate that BoltBeam's model-format adapters work on real public 8B artifacts, not only synthetic fixtures.

This is a profile/search validation pass:

```bash
boltbeam inspect <artifact> --target amd_gfx1100
boltbeam emit-search <artifact> --target amd_gfx1100
```

It is not an inference or speed benchmark.

## Artifacts Downloaded

| format | source repo | local path |
|---|---|---|
| GGUF | existing local `Qwen3-8B-Q4_K_M.gguf` | `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` |
| Safetensors | `Qwen/Qwen3-8B` | `/home/ubuntu/format-validation-models/qwen3-8b-safetensors` |
| AWQ | `Qwen/Qwen3-8B-AWQ` | `/home/ubuntu/format-validation-models/qwen3-8b-awq` |
| GPTQ | `JunHowie/Qwen3-8B-GPTQ-Int4` | `/home/ubuntu/format-validation-models/qwen3-8b-gptq` |
| ONNX | `onnx-community/Qwen3-8B-ONNX` | `/home/ubuntu/format-validation-models/qwen3-8b-onnx/onnxruntime/cuda/cuda-int4-kld-block-128/model.onnx` |
| MLX | `Qwen/Qwen3-8B-MLX-4bit` | `/home/ubuntu/format-validation-models/qwen3-8b-mlx` |

For sharded safetensors/AWQ/GPTQ repos, this validation downloaded the real config/index plus the first
real weight shard. That is enough to validate tensor naming, dtype, shape parsing, role classification, and
search-space emission. It is not a complete inference artifact.

## Results

| format | inspect | emit-search | architecture | complete | roles | quant types | families |
|---|---:|---:|---|---:|---:|---|---|
| GGUF | pass | pass | `dense_decoder` | true | 7 | F32, Q4_K, Q6_K | `lanemap_gemv`, `q6k_route` |
| Safetensors | pass | pass | `dense_decoder` | true | 7 | BF16 | `matmul_or_attention` |
| AWQ | pass | pass | `dense_decoder` | true | 21 | AWQ_INT4 | `unsupported_quant` |
| GPTQ | pass | pass | `dense_decoder` | true | 21 | GPTQ_INT4 | `unsupported_quant` |
| ONNX | pass | pass | `dense_decoder` | true | 33 | F16, U8 | `matmul_or_attention`, `unsupported_quant` |
| MLX | pass | pass | `dense_decoder` | true | 15 | BF16, U32 | `matmul_or_attention`, `unsupported_quant` |

## Fixes Found By Real Artifacts

The first real run found two adapter bugs that the synthetic fixtures did not catch:

1. AWQ/GPTQ use names like `q_proj.qweight` and `q_proj.qzeros`, not `q_proj.weight`.
   The role classifier now matches `q_proj.`, `k_proj.`, `v_proj.`, `o_proj.`, and `out_proj.`.
2. ONNX MatMulNBits stores packed quant tensors as 3D arrays. Those are not MoE expert stacks.
   The architecture classifier and shape logic now distinguish packed quant tensors from real expert stacks.

After those fixes, all real 8B artifacts classify as `dense_decoder`.

## Honest Limits

- AWQ and GPTQ are profiled correctly, but no AWQ/GPTQ route family is registered yet. BoltBeam correctly emits
  `unsupported_quant` instead of pretending it can route them.
- ONNX profiling reads static initializer metadata. It does not execute ONNX graphs.
- MLX profiling reads the model export metadata/tensor headers. It does not execute Metal kernels.
- This pass did not run GPU inference or speed gates.

## Verification

```bash
python3 -m unittest discover -s tests
```

Result: 209 tests pass.

