# llama.cpp Hardware Trace Import

BoltBeam can ingest llama.cpp kernel timing by translating backend CSV output
into `boltbeam.hw_trace.v1`. The current AMD adapter can read `rocprofv3` CSVs,
but the public workflow is BoltBeam-owned so the backend can be replaced without
changing downstream trace consumers.

BoltBeam still does not execute the model. The runtime boundary is:

```text
backend collector + llama-bench -> kernel_stats.csv + backend metadata + llama_bench.json
boltbeam import-hw-trace --provider llama -> hw_trace.json with elapsed/tok/s/bytes/counters
boltbeam ingest-timing -> timing_profile.json
boltbeam analyze/output -> timing-aware report
```

## Staged Run

Create the run and ask for llama.cpp trace commands:

```bash
boltbeam load /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --run outputs/qwen3-14b-llama-prefill \
  --id qwen3-14b \
  --workload prefill \
  --ctxs 512

boltbeam analyze --run outputs/qwen3-14b-llama-prefill

boltbeam runner-plan \
  --run outputs/qwen3-14b-llama-prefill \
  --provider llama.cpp=/home/ubuntu/env/llama.cpp/build/bin/llama-bench
```

`runner_plan.json` then contains `provider_commands` for:

- `llama_boltbeam_trace`: run `boltbeam collect-hw-trace --provider llama`.
- `llama_hw_trace_ingest`: classify the trace into `timing_profile.json`.

After running those commands:

```bash
boltbeam analyze --run outputs/qwen3-14b-llama-prefill
boltbeam output --run outputs/qwen3-14b-llama-prefill
```

The timing breakdown appears in `timing_profile.json`, `measurement_plan.json`,
`provider_plan.json`, and `summary.md`.

## Direct Import

If you already have a backend CSV:

```bash
boltbeam import-hw-trace qwen14b_pp512_kernel_stats.csv \
  --provider llama \
  --run outputs/qwen3-14b-llama-prefill \
  --context 512 \
  --llama-bench-json qwen14b_pp512_llama_bench.json

boltbeam ingest-timing outputs/qwen3-14b-llama-prefill/hw_trace.json \
  --run outputs/qwen3-14b-llama-prefill
```

Without a staged run, pass metadata explicitly:

```bash
boltbeam import-hw-trace qwen14b_pp512_kernel_stats.csv \
  --provider llama \
  --model-id qwen3-14b \
  --target-id amd_gfx1100 \
  --workload prefill \
  --context 512 \
  --llama-bench-json qwen14b_pp512_llama_bench.json \
  --weight-inventory outputs/qwen3-14b-llama-prefill/weight_inventory.json \
  --backend-json qwen14b_pp512_results.json \
  --out hw_trace.json
```

## Reported Fields

With `--llama-bench-json`, the whole-step row uses:

- `wall_us` from llama-bench `avg_ns`.
- `tok_s` from llama-bench `avg_ts`.
- kernel lifecycle shares from the backend CSV, scaled so kernel rows sum to the
  measured llama-bench wall time. Each kernel row keeps `raw_wall_us`.

With `--weight-inventory`, the importer adds:

- `total_bytes` on the whole-step row.
- `phys_bytes` on quantized matmul rows.
- `bytes_source=estimated_weight_inventory_by_quant_time_weighted`.

These are estimated packed weight bytes for one prefill lifecycle. Backend
counter CSVs can add measured hardware counters under normalized BoltBeam names
such as `fetch_kb`, `write_kb`, `memory_busy_pct`, and `mfma_util_pct`.

With `--backend-json`, the importer adds:

- `transfer_summary` with measured host/device copy bytes and allocation counts
  when the active backend adapter provides them.
- `scope=transfer` rows for host-to-device and device-to-host copies.

Transfer rows are deliberately not included in prefill kernel elapsed/tok-s,
because process-wide copies often include model upload before the timed
prefill.

## Kernel Attribution

The importer classifies llama.cpp kernels by demangled backend kernel names:

- `mul_mat_q<ggml_type)12>` -> Q4_K quantized matmul
- `mul_mat_q<ggml_type)14>` -> Q6_K quantized matmul
- `flash_attn_*` -> attention
- `rms_norm_*` -> norm
- `rope_*` -> rope
- `quantize_*` -> activation quantize

Kernel names do not include exact tensor names, so the importer currently emits
coarse roles like `quantized_matmul`, not exact roles like `ffn_gate_up`. Exact
role attribution requires either llama.cpp ROCTX graph labels or a model-profile
join that maps dispatch order to graph nodes.

## Exhaustive Trace Scope

For a full prefill lifecycle audit, capture these layers:

| layer | source | exact now |
|---|---|---|
| wall elapsed + tok/s | llama-bench JSON | yes |
| GPU kernel lifecycle | backend kernel stats/trace | yes, by kernel name |
| packed weight bytes | BoltBeam weight inventory | estimate |
| host/device copies | backend memory copy trace | raw capture available, not folded into prefill timing rows |
| GPU allocations | backend allocation trace | imported when supplied by the adapter |
| true HBM bytes/counters | backend counter CSV normalized by BoltBeam | yes when provided |
| exact graph/tensor role | llama.cpp ROCTX or graph-node labels | not yet available |
