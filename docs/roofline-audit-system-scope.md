# BoltBeam Roofline Audit System Scope

Goal: turn roofline from manual math into a model-, target-, and quant-aware
BoltBeam command that can explain practical ceilings and remaining performance
gaps for Qwen and future models.

Promotion and route-family closeout use the practical-roofline extension in
`docs/practical-roofline-promotion-scope-20260702.md`: a route may be "done"
because it is close to the best same-scope practical ceiling, but that does not
let roofline evidence bypass correctness, route binding, or rollback.

BoltBeam should not run GPU benchmarks for this feature. It should consume
profiles, target descriptors, bandwidth evidence, and normalized tinygrad
artifacts, then emit a durable roofline report.

## R0: Inventory Current Math

Audit:

- `boltbeam/math/roofline.py`
- normalized evidence adapters
- report helpers
- roofline tests

Record what already exists:

- Amdahl gain math
- bandwidth floor
- tok/s ceiling
- peak-vs-achievable distinction

Record missing inputs:

- role bytes
- quant bytes
- KV bytes
- measured target bandwidth
- compute floor

Output:

- `docs/roofline-system-gap-audit.md`

## R1: Quant Byte Registry

Add a quant-format registry with:

- `Q4_K`
- `Q5_K`
- `Q6_K`
- `Q8_0`
- `F16`
- `BF16`
- `F32`

Each entry should carry:

- block size
- effective bytes per weight
- scale/min overhead
- dequant family
- support status

Unknown quant formats must become explicit `unsupported_quant` rows or
`unknown` fields. Do not fake byte counts.

Acceptance:

- Q4_K and Q6_K byte math matches manually checked fixtures.
- Unknown quant fails closed or reports unknown.
- Quant constants live in the registry, not scattered across code.

## R2: Role Byte Accounting

From `ModelProfile`, compute per-token weight bytes by role:

- `ffn_gate_up`
- `ffn_down`
- `attn_qo`
- `attn_kv`
- `lm_head`
- future MoE expert roles

The accounting must include:

- tensor rows
- tensor cols
- quant format
- layer/count multiplier

Output:

- `role_bytes.json` as part of the roofline report.

Acceptance:

- Qwen fixture totals match manual calculation.
- No model-size constants are needed in the implementation.

## R3: KV And Attention Traffic Model

Compute KV-read bytes by context when metadata is available:

- context length
- layer count
- KV head count
- head dim
- KV dtype or quant

Separate:

- weight-read floor
- KV-read floor
- attention reduce/combine overhead when evidence exists

If required metadata is missing, mark the field `unknown`; do not use zero as a
placeholder.

Acceptance:

- Known Qwen fixture emits KV bytes by context.
- Incomplete fixture reports missing metadata cleanly.

## R4: Target Bandwidth Evidence

Add a target bandwidth input schema with:

- target id
- peak bandwidth
- measured achievable bandwidth
- benchmark method
- source artifact
- confidence

Support:

- static descriptor defaults
- measured evidence override

The report must distinguish peak from achievable bandwidth.

Acceptance:

- Ceiling changes when measured achievable bandwidth is provided.
- Missing achievable bandwidth is labeled unknown or uses a clearly marked
  descriptor default, never a hidden assumption.

## R5: Compute Floor

Add optional compute roofline inputs:

- peak FLOPS
- measured GEMM/GEMV effective FLOPS
- dequant tax factor

Classify roles as one of:

- bandwidth-bound
- compute-bound
- dequant-bound
- unknown

Compute data is optional. Decode weight-path analysis must still work in
bandwidth-only mode.

Acceptance:

- Synthetic Q4_K decode case classifies as bandwidth-bound.
- Synthetic prefill case can classify as compute-bound.

## R6: Roofline Report Contract

Create schema:

- `boltbeam.roofline_report.v1`

The report should include:

- model id
- target id
- workload
- contexts
- per-role bytes
- total bytes per token
- peak ceiling tok/s
- achievable ceiling tok/s
- measured tok/s, if evidence exists
- percent of ceiling
- role share
- missing fields
- confidence tags

Acceptance:

- JSON schema validates committed fixtures.
- Missing optional fields are represented explicitly.

## R7: CLI Command

Add:

```bash
boltbeam roofline \
  --profile model_profile.json \
  --target amd_gfx1100 \
  --evidence evidence.json \
  --ctxs 512,4096 \
  --bandwidth target_bandwidth.json \
  --out roofline.json
```

Behavior:

- no GPU execution
- fail closed on malformed profile/evidence
- permit missing optional evidence but label unknowns
- never silently assume measured bandwidth

## R8: Adapter Integration

Use existing normalized evidence:

- `wd_speed` for measured decode tok/s
- `prefill_speed` for measured prefill tok/s
- `role_attribution` for measured role wall share
- `runtime_overhead` for host/runtime correction

Add missing normalization only where necessary.

Acceptance:

- Existing normalized evidence files can be used without hand conversion.
- Prefill and decode both produce roofline rows when speed evidence exists.

## R9: Markdown Summary

Generate a readable summary from the JSON report:

- hard ceiling
- practical ceiling
- measured current
- percent of ceiling
- top role floors
- Amdahl upside
- recommended next audit target
- confidence and missing inputs

Example conclusion:

```text
Decode is weight-bandwidth-bound.
Current: 103 tok/s = 67% of achievable ceiling.
Q4_K GEMV is near practical route ceiling.
Remaining gap is reduce/runtime overhead, not raw weight bandwidth.
```

## R10: Qwen Validation

Run against existing Qwen evidence:

- 8B decode
- 14B decode
- 32B decode, if artifacts are available
- prefill, if authority evidence exists

Acceptance:

- Reproduces known conclusions:
  - 8B decode near llama parity after flash crossover.
  - 14B gap was route-missed `attn_k`, not FFN topology.
  - Q4_K G3 is near local route ceiling.
  - attention is low leverage for decode wall time.
  - prefill is compute-bound / role-specific.

## R11: Model-Agnostic Fixtures

Add synthetic fixtures for:

- dense Q4_K/Q6_K
- dense fp16
- MoE with expert roles
- unknown quant
- missing KV metadata
- non-AMD target descriptor

Acceptance:

- No Qwen dimensions are required in roofline code.
- Unknowns are explicit.
- Report still renders with missing fields.

## R12: Policy Guard

Add checks:

- no model-size constants in roofline code
- no target bandwidth constants outside target/bandwidth descriptors
- no quant byte constants outside the quant registry
- no hidden fallback to peak bandwidth when achievable bandwidth is absent

## Definition Of Done

- `boltbeam roofline` exists.
- Qwen roofline report is reproducible from profile plus evidence.
- Unknown quants and missing metadata fail closed or report unknown.
- Tests cover quant bytes, role bytes, KV bytes, bandwidth ceilings, CLI, and
  markdown rendering.
- Existing tests still pass.
- The output identifies whether the next gap is weight bandwidth, KV traffic,
  compute, runtime overhead, or missing evidence.
