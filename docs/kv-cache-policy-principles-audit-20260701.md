# KV Cache Policy Principles Audit

Status: pass for the KQ implementation surface.

Scope audited:

- `boltbeam/vocab.py`
- `boltbeam/math/kv_cache.py`
- `boltbeam/artifacts/tinygrad.py`
- `boltbeam/eval/evaluator.py`
- `boltbeam/data/targets.json`
- `boltbeam/data/candidates.json`
- `tests/test_kv_cache_policy.py`
- `tests/test_tinygrad_adapters.py`

## Findings

### Centralized Vocabulary

Pass.

New vocabulary lives in `boltbeam/vocab.py`:

- `TinygradArtifactKind.KV_CTX_SLOPE`
- `RowKind.KV_CTX_SLOPE`
- `KVPolicyClass`
- `TargetCapability`
- `SCHEMA_TINYGRAD_KV_CTX_SLOPE`

The source scan shows raw policy class strings only in `vocab.py`, tests, and docs. The evaluator uses `KVPolicyClass` and `TargetCapability`, not invented local strings.

### BoltBeam / tinygrad Boundary

Pass.

BoltBeam still does not run GPU work, import tinygrad, call llama.cpp, or mutate runtime defaults. It only ingests the tinygrad/llama slope artifact and evaluates normalized evidence.

tinygrad remains the producer of `tinygrad.llama_kv_ctx_slope.v1`; BoltBeam owns the decision.

### Capacity vs Speed Separation

Pass.

The manifest separates:

- capacity candidates: `decode_kv_cache_q8_capacity`, `decode_kv_cache_q4_capacity`
- speed candidates: `decode_kv_cache_q8_speed`, `decode_kv_cache_q4_speed`
- mixed probe: `decode_kv_cache_mixed_kv_probe`

The evaluator returns `candidate` for capacity-only cases and `refute` for storage-win/speed-loss cases. It does not promote quantized KV from byte savings alone.

### Target Capabilities As Data

Pass.

Same-type and mixed quantized KV FlashAttention fastpaths are target data in `boltbeam/data/targets.json`. The evaluator reads the mixed-fastpath capability through `target_capability()` using `TargetCapability.MIXED_QUANT_KV_FLASH_FASTPATH`.

Descriptor-only NVIDIA/Metal targets remain non-promotable through the existing backend-status guard.

### Tiny Modules / Duplication

Pass with one expected extension point.

`boltbeam/math/kv_cache.py` is a small, pure math module. The evaluator only adds a narrow `_kv_policy_result()` branch keyed by `candidate.thresholds["kv_cache_policy"]`; normal route promotion logic is unchanged.

No duplicate threshold table was introduced. The only repeated strings are manifest data values, test fixtures, and documentation examples.

### Hardcoding

Pass.

No model dimensions or Qwen-specific coefficients were added to BoltBeam code. The measured q8/q4 residual constants live only in tests/docs. Target-specific facts live in `targets.json`.

Existing adapter fallback to `amd_gfx1100` is unchanged from prior behavior and still overridable with `--target-id`; this pass did not widen that surface.

## Verification

Focused suite:

```text
pytest -q tests/test_kv_cache_policy.py tests/test_tinygrad_adapters.py tests/test_evaluator.py tests/test_roofline.py tests/test_manifest_templates.py tests/test_targets.py
51 passed
```

Full suite after the audit fix:

```text
pytest -q
235 passed
```

Source scan:

```text
rg -n "tinygrad\\.llama_kv_ctx_slope|mixed_quant_kv_flash_fastpath|same_type_quant_kv_flash_fastpath|KV_STORAGE|KV_CAPACITY|KV_FASTPATH|KV_MIXED|KV_INCONCLUSIVE|kv_cache_policy" boltbeam tests docs -S
```

Result: policy vocabulary appears in `boltbeam/vocab.py`, target data, manifest data, tests, and docs. The evaluator references the central enum/capability helpers rather than duplicating verdict strings or target facts.
