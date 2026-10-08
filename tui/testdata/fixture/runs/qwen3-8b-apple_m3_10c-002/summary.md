# BoltBeam Run Summary: Qwen3-8B

- Latest stage: `analyze`
- Model format: `gguf`
- Target: `apple_m3_10c`
- Workload: `decode`
- Architecture: `dense_decoder`
- Analysis status: `needs_measurement`

## Next Step

Run or translate `measurement_plan.json` through a provider adapter, then ingest normalized evidence and re-analyze.

## Timing Profile

- Dominant bucket: `elementwise_dilution`
- look for launch-bound elementwise/reduce work that can fold into neighboring kernels
- candidate timing contains a measurable win; require correctness and route-bound evidence before promotion
- `ffn_gate_up` `Q4_K` `12288x4096`: `timing_hot` 41.3%, 23900.0us
- `ffn_down` `Q6_K` `4096x12288`: `timing_hot` 15.0%, 8700.0us
- `attn_qo` `Q4_K` `4096x4096`: `timing_hot` 13.8%, 8000.0us
- `lm_head` `Q6_K` `151936x4096`: `timing_hot` 11.1%, 6400.0us
- `ffn_down` `Q4_K` `4096x12288`: `timing_hot` 10.4%, 6000.0us
- `attn_kv` `Q4_K` `1024x4096`: `timing_observed` 3.3%, 1900.0us
- `attn_kv` `Q6_K` `1024x4096`: `timing_observed` 1.7%, 1000.0us

## Quant GEMV Regimes

- `ffn_gate_up` `Q4_K` `12288x4096`: `streaming_bound` via `weight_load`; preserve wide coalesced packed loads; only compression or layout that keeps load-bound behavior can help
- `attn_kv` `Q4_K` `1024x4096`: `occupancy_starved` via `insufficient_residency`; increase independent tiles/splits or reduce resource pressure

## Artifacts

- `run_manifest.json`
- `model_profile.json`
- `weight_inventory.json`
- `workload_profile.json`
- `hardware_profile.json`
- `runtime_profile.json`
- `provider_capabilities.json`
- `scan_evidence.json`
- `search_space.json`
- `route_policy.json`
- `fixture_manifest.json`
- `measurement_plan.json`
- `analysis_report.json`
- `probe_request.json`
- `trace_request.json`
- `primitive_profile.json`
- `timing_trace.json`
- `timing_profile.json`
