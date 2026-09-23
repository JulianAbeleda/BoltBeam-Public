# KV Cache Quantization Policy Scope

Status: scoped for implementation.

Owner split:

- **BoltBeam owns the decision.** It classifies quantized KV as a speed win, capacity win, refuted speed knob, or blocked fastpath.
- **tinygrad owns producer artifacts and runtime hooks.** It measures context-slope and reports cache dtype/runtime policy. It should not bake the policy into ad hoc flags.

## Why this exists

The llama.cpp long-context stress test showed that the simple KV storage formula is incomplete for quantized KV:

```text
ms/token = A + B * ctx
```

For Qwen3-8B on the local ROCm llama.cpp path:

| KV type | KV bytes/context token | measured B ms/ctx | storage-only B at f16 BW | residual B |
|---|---:|---:|---:|---:|
| f16/f16 | 147456 | 0.000171794 | 0.000171794 | 0 |
| q8_0/q8_0 | 78336 | 0.000228343 | 0.000091 | 0.000137 |
| q4_0/q4_0 | 41472 | 0.000191623 | 0.000048 | 0.000143 |

So quantized KV is smaller, but not automatically faster. Most of the quantized slope is residual unpack/dequant/quantized-dot work inside the attention kernel.

The model BoltBeam should use:

```text
T_decode(ctx, K, V)
  = A
  + ctx * (kv_storage_bytes_per_ctx_token(K,V) / BW_attention_f16)
  + ctx * R_quant(K,V,target,backend)
```

where:

```text
R_quant = measured_B - storage_only_B
```

This keeps two separate truths:

- **VRAM planning:** use storage bytes.
- **speed planning:** use storage bytes plus residual.

## Required reading

Local:

- tinygrad result note: `/home/ubuntu/tinygrad-arkey/docs/quantized-kv-residual-term-20260701.md`
- tinygrad benchmark producer: `/home/ubuntu/tinygrad-arkey/extra/llama_kv_ctx_slope_bench.py`
- local llama.cpp dtype flags: `/home/ubuntu/env/llama.cpp/tools/llama-bench/README.md`
- local llama.cpp FlashAttention dtype dispatch: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn.cu`
- local llama.cpp quantized K/V helpers: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh`

External:

- KIVI, "A Tuning-Free Asymmetric 2bit Quantization for KV Cache": https://arxiv.org/abs/2402.02750
- KIVI proceedings page: https://proceedings.mlr.press/v235/liu24bz.html
- OScaR, "Occam's Razor for Extreme KV Cache Quantization": https://arxiv.org/abs/2605.19660
- vLLM automatic prefix caching docs: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
- TensorRT-LLM KV cache reuse: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- NVIDIA KV cache reuse blog: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/

## Principles

1. **Do not infer speed from bytes.** A quantized KV route can be a memory win and a speed loss.
2. **Keep capacity and speed decisions separate.** A capacity-only route can still be useful when it enables a longer context or prevents OOM.
3. **Target capability is data.** Mixed K/V quantized FlashAttention fastpaths must be represented as target/backend capabilities, not guessed from dtype support.
4. **No default flip without W==D.** Quantized KV defaults require whole-decode speed evidence and correctness/quality gates, not only a roofline.
5. **No parallel vocabulary.** All row kinds, verdict tags, policy classes, and schema ids go through `boltbeam/vocab.py`.

## Phase KQ0 - Evidence Schema And Adapter

Goal: make tinygrad/llama KV slope artifacts first-class BoltBeam evidence.

Add:

- `RowKind.KV_CTX_SLOPE`
- schema id: `boltbeam.kv_ctx_slope.v1` or equivalent centralized constant
- tinygrad adapter support for `tinygrad.llama_kv_ctx_slope.v1`

Normalized rows:

```json
{
  "kind": "kv_ctx_slope",
  "metric": "b_ms_per_ctx",
  "value": 0.000228343,
  "unit": "ms/context-token",
  "context": null,
  "extra": {
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "a_ms": 10.997191,
    "r2": 0.999655,
    "kv_bytes_per_ctx_token": 78336,
    "implied_kv_bandwidth_gb_s": 343.1,
    "depths": [512, 1024, 2048, 4096, 8192, 16384, 32768],
    "producer_schema": "tinygrad.llama_kv_ctx_slope.v1"
  }
}
```

Acceptance:

- adapter reads existing ignored/local artifacts when given an explicit path;
- no hardcoded `/home/ubuntu` path in normalized evidence;
- row has `model_id`, `target_id`, `workload=decode`;
- missing fit fields produce `adapter-incomplete`, not a crash.

## Phase KQ1 - KV Roofline Math

Goal: add pure math helpers, no model constants.

Add `boltbeam/math/kv_cache.py`:

```python
storage_slope_ms_per_ctx(kv_bytes_per_ctx_token, bandwidth_bytes_per_s)
quant_residual_ms_per_ctx(measured_b, storage_b)
predict_decode_tok_s(a_ms, b_ms_per_ctx, ctx)
classify_kv_slope(...)
```

Classification inputs:

- measured `A`, `B`, `R2`
- `kv_bytes_per_ctx_token`
- f16 baseline attention bandwidth
- candidate cache dtype K/V
- target capability data

Acceptance:

- f16 residual is near zero under measured f16 bandwidth;
- q8/q4 residual is positive for the measured 8B artifacts;
- low R2 returns inconclusive;
- negative residual is allowed but marked as a real speed win only if whole-decode evidence agrees.

## Phase KQ2 - Policy Classifier

Goal: give BoltBeam a policy vocabulary for quantized KV.

Add policy classes in one enum:

```text
KV_STORAGE_WIN_SPEED_LOSS
KV_STORAGE_WIN_SPEED_WIN
KV_CAPACITY_ONLY
KV_FASTPATH_MISSING
KV_MIXED_FASTPATH_UNKNOWN
KV_INCONCLUSIVE
```

Rules:

- If storage bytes shrink but measured B grows vs f16: `KV_STORAGE_WIN_SPEED_LOSS`.
- If storage bytes shrink and measured whole-decode W==D improves: `KV_STORAGE_WIN_SPEED_WIN`.
- If storage bytes shrink, speed is flat/lower, but f16 KV would exceed memory/context budget: `KV_CAPACITY_ONLY`.
- If mixed K/V dtype is requested but target lacks mixed quantized FA fastpath: `KV_MIXED_FASTPATH_UNKNOWN` or `KV_FASTPATH_MISSING`.
- If artifact lacks slope/R2: `KV_INCONCLUSIVE`.

Acceptance:

- q8/q8 and q4/q4 8B artifacts classify as storage-win/speed-loss for the measured ROCm llama.cpp path;
- capacity-only can be emitted without speed promotion;
- speed promotion requires W==D evidence, not just slope.

## Phase KQ3 - Target Capabilities

Goal: make backend support explicit.

Extend `boltbeam/data/targets.json` capabilities:

```json
{
  "same_type_quant_kv_flash_fastpath": true,
  "mixed_quant_kv_flash_fastpath": false,
  "kv_quant_speed_residual_required": true,
  "kv_quant_quality_gate_required": true
}
```

For the local llama.cpp ROCm/CUDA path, same-type `q4_0/q4_0` and `q8_0/q8_0` are supported by the default compiled cases; mixed dtype support is not assumed unless the build enables all quantized FA cases.

Acceptance:

- descriptor-only targets never promote a quantized KV speed route;
- mixed K/V benchmark requests are skipped or marked incomplete unless capability is true;
- target capability checks are reused by the search emitter and evaluator.

## Phase KQ4 - Candidate Manifest

Goal: represent quantized KV as candidates without pretending it is a kernel win.

Candidate families:

```text
decode_kv_cache_q8_capacity
decode_kv_cache_q4_capacity
decode_kv_cache_q8_speed
decode_kv_cache_q4_speed
decode_kv_cache_mixed_kv_probe
```

Required evidence:

- capacity candidates: KV bytes/context-token, max-context memory estimate, protected quality/correctness gate.
- speed candidates: `kv_ctx_slope` plus W==D decode speed evidence.
- mixed probes: target fastpath capability or explicit source/build proof.

Acceptance:

- capacity candidate can be recommended/deferred without becoming a promoted speed route;
- speed candidate cannot promote without rollback and W==D;
- q4/q8 speed candidates for current ROCm llama.cpp measurements refute/defer, not promote.

## Phase KQ5 - tinygrad Producer Updates

Goal: make the producer robust enough for BoltBeam ingestion.

Update `/home/ubuntu/tinygrad-arkey/extra/llama_kv_ctx_slope_bench.py` or add a tinygrad-native equivalent:

- emit stable schema fields for BoltBeam;
- include `target_id`, backend/build commit, cache K/V dtype, flash-attention mode;
- emit `fit_quality` and `r2`;
- include `storage_only_b_ms_per_ctx` if an f16 baseline is supplied;
- detect mixed K/V support before running mixed cases;
- support `--baseline-json` to compute residuals directly.

Do not make mixed K/V probes part of the default sweep unless the target declares a mixed fastpath.

Acceptance:

- existing f16/q8/q4 same-type artifacts still reproduce;
- producer writes an artifact BoltBeam can ingest;
- no default tinygrad runtime policy changes.

## Phase KQ6 - Runtime Policy Hooks

Goal: expose the decision to clients without hiding tradeoffs.

tinygrad runtime/status should eventually report:

```json
{
  "kv_cache": {
    "type_k": "f16",
    "type_v": "f16",
    "bytes_per_context_token": 147456,
    "policy": "speed_default",
    "estimated_b_ms_per_ctx": 0.000171794,
    "quant_residual_ms_per_ctx": 0.0,
    "reason": "f16 is faster than measured q8/q4 on this target"
  }
}
```

Possible policies:

```text
speed_default
capacity_only
oom_avoidance
long_context_required
experimental_speed
unsupported
```

Acceptance:

- runtime can explain why f16 was selected over q4/q8;
- runtime can explain when q4/q8 is selected for capacity;
- policy output is data, not stderr-only.

## Phase KQ7 - Tests And Regression Guard

Minimum tests:

- vocab guard: no schema/policy string invented outside `vocab.py`;
- adapter reads a tinygrad KV slope fixture;
- roofline helper reproduces the measured residuals:
  - q8 residual about `0.000137 ms/ctx`;
  - q4 residual about `0.000143 ms/ctx`;
- evaluator refuses to promote quantized KV speed from bytes alone;
- evaluator allows capacity-only recommendation when f16 KV does not fit;
- target capability blocks mixed K/V probes when mixed fastpath is false;
- report includes citations to evidence paths.

## Implications

1. **Quantized KV is not a free long-context speed lever.** It is primarily a memory/capacity lever unless the backend has a hardware-friendly quantized attention kernel.
2. **Long-context speed estimates need residual coefficients.** The storage formula is still right for VRAM, but wrong for tok/s without `R_quant`.
3. **BoltBeam should stop treating "smaller bytes" as a roofline win.** A candidate must show measured W==D or stay capacity-only.
4. **tinygrad should default to f16/bf16 KV for speed on this target.** q8/q4 should be chosen when they unlock context/model fit, not because they are expected to be faster.
5. **Future kernel work has a clear target.** A real quantized-KV speed win means reducing `R_quant`, not just packing bytes tighter. That points to fused quantized FlashAttention, better K-dot/V-dequant lowering, or target-specific kernels like the KIVI/OScaR-style papers emphasize.
6. **This helps multi-GPU/VRAM planning.** Capacity policy can decide when the extra context enabled by quantized KV is worth lower tok/s; speed policy can stay separate.

## Stop Conditions

Stop and report instead of forcing a pass if:

- mixed K/V support cannot be proven from target capabilities;
- q4/q8 W==D loses speed but extends context;
- measured R2 is poor;
- quality/correctness gates for quantized KV are missing;
- target/backend is descriptor-only.

