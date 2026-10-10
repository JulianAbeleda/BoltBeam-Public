# The kernel search on any GPU, and BoltBeam's check of the provider (2026-10-10)

## What changed

Run's kernel search (`boltbeam/search/role_compare.py`) is now the documented pipeline
(`docs/semantic-campaign.md`), per role, on whatever GPU the run targets:

1. BubbleBeam proposes from the chip's facts (`futuresight_adapter.propose_request`).
2. The population is exported (`semantic_population_export.export_population`).
3. FutureSight rejects and orders it, on the CPU (`futuresight_adapter.assess_population`).
4. The semantic campaign measures the survivors through the fork's provider (`semantic_campaign_cli.run_request`).
5. BoltBeam checks what the provider said (`boltbeam/search/provider_check.py`).

The dimensions come from what the provider's `describe` says the model's decode binds through
(`boltbeam/search/role_space.py`). On the M3 and the RTX 5090 that is the fork's decode emitters: `q4_g3_gemv.v1`
for Q4_K and `q6_coop_gemv.v1` for Q6_K. Each emitter parameter combination is one coupled row.

The device comes from the provider's `capability` action and `boltbeam/data/search_runtime.json`. That table maps
each BoltBeam backend to its tinygrad devices and the A/B environment. No backend name is left in the search path.

## Runs before this change carry a false label

Before this change, Run's search used `role_compare.ROWS`. It was a fixed 13-row list copied from one Apple M4
bench: the heuristic control, no opts, UPCAST 2/3/4, and LOCAL 8 to 1024. It was used for every role, shape and
chip, and it stamped each candidate `"generator_id": "bubblebeam_futuresight"`. Neither BubbleBeam nor FutureSight
ran.

Any `kernel_compare/*-search-request.json` written before 2026-10-10 carries that false label. The candidates are
now labelled `semantic_candidate_plan`, the module that builds them. A test refuses the old label without
BubbleBeam's proposal beside it.

Those rows were also searching the wrong kernels. They were Opt sequences on tinygrad's scheduled matmul. The
8B's decode on both the M3 and the 5090 runs `q4k_g3_lanemap_gemv` and `q6k_gen_coop` emitters, as the role-time
trace shows. That is why the old A/B reported plans reaching 0 decode calls.

## BoltBeam's check

The check has three parts:

- **Identity.** The provider's device facts are compared with BoltBeam's own bridge. On the 5090: SM count 170,
  L2 100,663,296 bytes, compute capability 12.0. The check caught the provider's first NV cut reading 10.4.
- **Correctness.** Every measured candidate is rebuilt from the source the provider returned. It runs on the role's
  own GGUF bytes and is checked against `metal_native.reference_row`, with the activation rounded to fp16 as the
  kernel reads it.
- **Time.** Every measured candidate is timed by the kernel timer: one loop, a read sweep, and the dispatch floor
  beside it.

The verdict rule:

- The provider's median is judged against BoltBeam's median less the floor, within the chip's band
  (`screen.plausibility_band`: 1.0% on the M3, 1.4% on the 5090).
- A reading the tie-out calls noisy cannot confirm a claim.
- BoltBeam's winner is its own fastest candidate. If that candidate's claim is not reproduced, the role is
  "provider claim not reproduced" (verdict `not_reproduced`). It is never promoted and never refuted.

## RTX 5090, Qwen3-8B Q4_K_M (tg-5090-b13)

The proposal comes from BubbleBeam. FutureSight rejects rows statically. A claim counts as reproduced when the
provider's time is within the band of BoltBeam's. The provider and BoltBeam times are for BoltBeam's fastest
candidate.

| role | proposed | FutureSight rejected | measured | reproduced | provider µs | BoltBeam µs less floor | in model µs | verdict |
|---|---|---|---|---|---|---|---|---|
| ffn_gate_up Q4_K | 5 | 0 | 5 | 2 of 5 | 19.8 | 19.3 | 36.1 | not reproduced (1.02x, outside ±1.4%) |
| attn_qo Q4_K | 5 | 0 | 5 | 1 of 5 | 9.2 | 7.6 | 9.0 | not reproduced (BoltBeam's reading noisy, floor 4.3 µs under 11.8 µs) |
| ffn_down Q4_K | 5 | 0 | 5 | 1 of 5 | 24.6 | 22.6 | 20.7 | not reproduced (1.09x) |
| ffn_down Q6_K | 11 | 3 threads_exceed_one_subgroup | 8 | 2 of 8 | 35.0 | 33.0 | 28.0 | not reproduced |
| attn_kv Q4_K | 5 | 0 | 5 | 0 of 5 | 5.0 | 3.3 | 5.1 | not reproduced (noisy) |
| attn_kv Q6_K | 11 | 3 threads_exceed_one_subgroup | 8 | 1 of 8 | 10.7 | 9.2 | 5.0 | not reproduced |
| lm_head Q6_K | 11 | 3 threads_exceed_one_subgroup | 8 | 5 of 8 | 320.9 | 318.5 | 310.8 | refuted: the model's own kernel (coop, row_tile 2) is the fastest |

On NV the provider's emitter times sit 2 to 9% above BoltBeam's time less the floor. That is outside the band, so
nothing is promoted.

The ffn_gate_up row compares a single 12288-row GEMV with the model's fused gate+up kernel per call. The fused
w1w3 family is not in the search space (see gaps).

## Apple M3, Qwen3-8B (m3-b13)

Every role ended "provider claim not reproduced": 0 to 2 claims reproduced per role.

The provider's Metal times are far from BoltBeam's on both passes. On the first pass, BoltBeam's attn_kv Q4_K
time agreed with the model: 33.4 µs against 34.9 µs, while the provider claimed 67 to 123 µs. On the recorded
pass, BoltBeam's own times were 3 to 4x the in-model times (attn_qo 448 µs against 126.6 µs), which looks like a
contended GPU. Both passes fail closed.

## Suspects: places BoltBeam took a provider's number, or a chip, as fact

Fixed in this change:

1. `role_compare.ROWS`: a fixed M4 row list for every role and chip.
2. The `bubblebeam_futuresight` label on candidates no BubbleBeam proposed.
3. `kernel_numbers.plan_us` was the provider's claim. It is now BoltBeam's timer.
4. The provider's rank picked the winner. BoltBeam now re-times every candidate.
5. The provider timed with a warm cache, then with a store scrub. It now uses a read sweep.
6. The provider's `exact_gguf` dequantized the tensor. It now binds the packed bytes.
7. The provider's NV compute capability read 10.4.
8. `COMPARE_BACKENDS`, `RUNTIME_ENV` and the "Metal only" refusal.
9. Every `"METAL"` literal in `search_provider.py`.
10. gameplan's fixed `row_tile=4` emitter pin.

Listed, not fixed:

1. The model's per-call time is tinygrad's own in-model timing (`tinygrad_role_time`), taken as fact.
2. Target-named routing in the fork: `Q6K_COOP_ROW_TILE_BY_TARGET {("NV","sm_120"): 2}` and
   `decode_routes._q4k_single_projection_load_style` (vector loads on NV sm_120 only).
3. Emitters pinned to the 8B's shapes, which BubbleBeam therefore cannot propose:
   - `q4_ffn_down.v1` (block_count 3, 4096x12288)
   - `q6_ffn_down.v1` (4096x12288)
   - `q6_v.v1` (1024x4096)
   - `q6_vocab_four_warp.v1` (151936x4096)
   - `q4_gate_up.v1` and the rms-affine w1w3 (12288x4096)
   - the `ffn_down_resadd` epilogues (rows 4096)
   - `shared_q8_provider.v1` (k 4096)
4. `generate_route_bound_candidate` accepts `nvidia_sm120` only.
5. `boltbeam_authority.py` reads a ledger named for sm_120.
6. Every route policy's `promoted_targets` names NV sm_120.
7. `nv_decode_seed_ledger.jsonl`: 12 refuted entries and 1 deferred. Their evidence paths point at
   `<home>/tinygrad-arkey/docs/...` and were not re-measured here, so they are stale until re-run.
8. The A/B driver binds Opt sequences only. An emitter winner is "decided by binding" until the driver installs
   the fork's route admission.
