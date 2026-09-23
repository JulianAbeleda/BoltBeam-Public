# Model Format Adapters Scope + Result

Date: 2026-07-01

## Goal

Support the five model artifact families BoltBeam cares about for local/search workflows:

1. GGUF
2. Safetensors
3. AWQ / GPTQ quantized Hugging Face layouts
4. ONNX
5. MLX

This is **profile/search support**, not runtime promotion. The output is a `ModelProfile`, search space, and
route-policy skeleton. Performance claims still require downstream evaluator evidence.

## Contract

All artifact readers must adapt into the same object:

```text
artifact -> ModelProfile -> search_space -> route_policy.seed
```

The search and policy layers must not branch on file format.

## Implemented

### GGUF

Uses the existing GGUF reader. `profile_from_model()` delegates to `profile_from_gguf()` and annotates
`metadata.format_family = "gguf"`.

### Safetensors

Reads safetensors headers directly:

- no `safetensors` dependency required;
- supports single-file or directory layouts;
- uses adjacent `config.json` for architecture, hidden size, layer count, and vocab size;
- classifies Hugging Face tensor names such as `q_proj`, `k_proj`, `gate_proj`, and `down_proj`.

### AWQ / GPTQ

Detected from `config.json.quantization_config` or `quantize_config.json` / `quant_config.json`.

The profile marks role quants as:

- `AWQ_INT4`
- `GPTQ_INT4`

These may still be search-space-incomplete until a target route family exists; the key improvement is that
the artifact is now profiled honestly instead of being opaque.

### ONNX

Reads enough protobuf to extract `GraphProto.initializer` tensor names, shapes, and dtype:

- no `onnx` dependency required;
- supports adjacent `config.json`;
- fixture-backed for dense transformer weight roles.

This is a static profile adapter, not an ONNX executor.

### MLX

Supports MLX-style directories through:

- safetensors exports with `config.json` carrying an MLX marker; or
- `.npz` weights when numpy is installed.

This keeps Apple/Metal model exports visible to BoltBeam while target-specific Metal route evaluation remains
separate.

## Tests

`tests/test_model_format_loaders.py` proves every format works through the shared path:

- GGUF fixture
- Safetensors directory fixture
- AWQ directory fixture
- GPTQ directory fixture
- ONNX protobuf fixture
- MLX directory fixture
- CLI `inspect` + `emit-search` on a safetensors directory

Verification:

```bash
python3 -m unittest discover -s tests
```

Result: 209 tests pass.

## Limits

- Safetensors/AWQ/GPTQ profiling depends on `config.json` for architecture metadata.
- AWQ/GPTQ route candidates are not promoted by this work; their quant strings become visible so the search
  can report missing routes honestly.
- ONNX support reads static initializers only. It does not import execution graphs or operators yet.
- MLX `.npz` support needs numpy. MLX safetensors-style exports do not.
- None of these adapters launch GPU work or mutate tinygrad.

## Next Work

1. Add real-world fixtures from small public safetensors/AWQ/GPTQ/ONNX/MLX repos.
2. Add teacher-kernel artifact ingestion: CUDA/PTX/SASS, Metal shader source, Triton kernels.
3. Add target-specific evaluators only after profile/search support identifies a candidate worth measuring.

