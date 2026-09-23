# Artifact Cache Inventory

**Verdict:** `C0_PASS_CACHE_INVENTORY_PINNED`

39 artifacts: A_static=29, B_correctness=3, C_speed=7; 36 need cache-metadata wrapping.

| class | wrap phase | count | reuse rule |
|---|---|---:|---|
| A_static | C2 | 29 | reuse by hash(inputs + code); no GPU |
| B_correctness | C3 | 3 | reuse only if inputs+code+runtime fingerprints match |
| C_speed | C3 | 7 | historical by default; promotion reruns unless cached speed is explicitly accepted |

## Artifacts

| path | class | speed? | correctness? | needs wrap |
|---|---|:--:|:--:|:--:|
| `bench/qk-search-spaces/decode_attention_gfx1100_v1.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/decode_attention_loop_search_space.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/decode_attention_tile_combine_a3_4.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/default_route_manifest.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/manual_oracle_not_search_generated.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/owned_delta_taxonomy.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/pms_r3_candidate_generator_check.json` | A_static | Y |  | Y |
| `bench/qk-search-spaces/profiles/_schema.json` | A_static | Y |  | Y |
| `bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.regen.json` | A_static | Y |  | Y |
| `bench/qk-search-spaces/profiles/qwen3_8b_q5_k_m_gfx1100.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/quant_semantics.json` | A_static | Y |  | Y |
| `bench/qk-search-spaces/refuted_axes.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/search_profiles.json` | A_static | Y | Y | Y |
| `bench/qk-search-spaces/targets/amd_gfx1100.json` | A_static | Y |  | Y |
| `bench/qk-search-spaces/targets/apple_metal_m3.json` | A_static |  |  | Y |
| `bench/qk-search-spaces/targets/nvidia_sm89.json` | A_static |  |  | Y |
| `bench/qk-search-spaces/topology_grammar_v1.json` | A_static | Y | Y | Y |
| `bench/qk-lanemap-template-ir/latest.json` | A_static |  | Y | Y |
| `bench/qk-lanemap-template-audit/latest.json` | A_static |  | Y | Y |
| `bench/qk-topology-author/latest.json` | A_static | Y | Y |  |
| `bench/qk-quant-semantics-audit/latest.json` | A_static | Y |  | Y |
| `bench/qk-profile-opener/latest.json` | A_static | Y |  |  |
| `bench/qk-profile-opener/qwen3_8b_q4_k_m_gfx1100/latest.json` | A_static | Y | Y | Y |
| `bench/qk-profile-opener/qwen3_8b_q5_k_m_gfx1100/latest.json` | A_static | Y | Y | Y |
| `bench/qk-target-features/latest.json` | A_static | Y |  |  |
| `bench/qk-template-candidate-gate/latest.json` | C_speed | Y | Y | Y |
| `bench/qk-new-profile-search/qwen3_8b_q6k_ffn_down_gfx1100/latest.json` | A_static | Y | Y | Y |
| `bench/qk-candidate-evaluator/decode_q4k_g3_generated/latest.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/decode_q4k_g3_generated/ledger_update.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/decode_q4k_g3_generated/route_attribution.json` | B_correctness |  | Y | Y |
| `bench/qk-candidate-evaluator/decode_q6k_direct_refuted/latest.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/decode_q6k_direct_refuted/ledger_update.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/decode_q6k_direct_refuted/route_attribution.json` | B_correctness |  | Y | Y |
| `bench/qk-candidate-evaluator/prefill_pipe_role_selective_default/latest.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/prefill_pipe_role_selective_default/ledger_update.json` | C_speed | Y | Y | Y |
| `bench/qk-candidate-evaluator/prefill_pipe_role_selective_default/route_attribution.json` | B_correctness |  | Y | Y |
