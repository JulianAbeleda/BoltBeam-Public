# Apparatus boundary follow-up, 2026-10-11

The initial dead-code sweep retained several modules on main using an August in-flight apparatus audit.
That was insufficient: tested research code does not become runnable product code because a past audit
retained it. This follow-up checks current callers and applies the branch categories in `branch-flow.md`.
The baseline is main `ad6ab31`, dev `c84d02f`; the tinygrad fork is read-only at
`f9e245c4e809e95282431130e2848c33c85b4ee3`.

All 23 paths from the August list still present on main move to dev. Six additional dependency or test-only
paths found in the current graph move with them: 28 non-initializer modules and one package initializer.
The code and its constraining tests are preserved unchanged on dev through explicit restoration commits.
There are no global deletions in this follow-up. Main's core/cache package initializers stop exporting
apparatus; dev retains those exports. Historical findings and referenced records stay at their paths.

## Individual dispositions

Every moved module lacks a current product caller after checking the CLI registry and `CORE`, pyproject
entrypoints, module/subprocess strings, Go seam, path-based schema/policy readers and the read-only fork.
The callers below establish its development purpose, not eligibility for main. A module CLI for an oracle
or research-policy reproduction is development apparatus too; both such CLIs remain runnable on dev.

| Path under boltbeam/ | Disposition | Code lines | Current consumer or retained development purpose |
| --- | --- | ---: | --- |
| `experiment/decode_counter_evidence.py` | dev apparatus | 69 | tests/test_decode_counter_evidence.py:1 |
| `experiment/decode_decay.py` | dev apparatus | 36 | tests/test_tinygrad_decode_adapter.py:7 |
| `experiment/decode_resource_evidence.py` | dev apparatus | 117 | tests/test_decode_resource_evidence.py:3 |
| `experiment/observation_protocol.py` | dev apparatus | 50 | tests/test_observation_protocol.py:1 |
| `experiment/prefill_authority.py` | dev apparatus | 30 | tests/test_prefill_authority.py:2 |
| `experiment/reconciliation.py` | dev apparatus | 45 | tests/test_reconciliation.py:2 |
| `experiment/reduce_source.py` | dev apparatus | 118 | tests/test_reduce_eliminated_guardrail.py:7; tests/test_reduce_source.py:4 |
| `adapters/tinygrad_prefill.py` | dev apparatus | 90 | tests/test_tinygrad_prefill_adapter.py:4 |
| `search/dense_route_authority.py` | dev apparatus | 59 | tests/test_dense_route_authority.py:6 |
| `search/primitive_template.py` | dev apparatus | 96 | tests/test_primitive_template.py:3 |
| `search/reachability.py` | dev apparatus | 155 | tests/test_golden_pipeline.py:64; tests/test_manifest_templates.py:8; tests/test_nv_decode_seed_ledger.py:17; tests/test_reachability.py:8; tests/test_review_fixes.py:20 |
| `search/research_categories.py` | dev apparatus | 108 | tests/test_research_categories.py:1 |
| `search/semantic/semantic_candidate.py` | dev apparatus | 47 | tests/test_semantic_candidate.py:3 |
| `runtime/amd_runtime_bridge.py` | dev apparatus | 30 | tests/test_amd_runtime_bridge.py:9 |
| `runtime/direct_kfd_bridge.py` | dev apparatus | 93 | tests/test_direct_kfd_bridge.py:3 |
| `runtime/kfd_observation_bridge.py` | dev apparatus | 146 | tests/test_kfd_observation_bridge.py:4 |
| `characterize/system_snapshot.py` | dev apparatus | 113 | boltbeam/characterize/__init__.py:2; tests/test_system_snapshot_collector.py:1 |
| `characterize/__init__.py` | dev apparatus | 2 | package initializer for the dev system-snapshot collector |
| `kernel_analysis/handoff.py` | dev apparatus | 255 | tests/kernel_analysis/test_handoff.py:10; tests/kernel_analysis/test_handoff.py:3 |
| `kernel_analysis/candidates.py` | dev apparatus | 278 | boltbeam/kernel_analysis/handoff.py:23; boltbeam/kernel_analysis/p9_assemble.py:17; tests/kernel_analysis/test_candidates.py:5; tests/kernel_analysis/test_handoff.py:6 |
| `kernel_analysis/lds_cache.py` | dev apparatus | 109 | boltbeam/kernel_analysis/candidates.py:9; tests/kernel_analysis/test_lds_cache.py:10 |
| `core/experiment.py` | dev apparatus | 85 | boltbeam/core/__init__.py:2; boltbeam/kernel_analysis/handoff.py:20; tests/core/test_experiment.py:3; tests/kernel_analysis/test_handoff.py:2 |
| `diagnostics/reg_lowering.py` | dev apparatus | 57 | tests/test_reg_lowering.py:9 |
| `trace/trace_enrich.py` | dev apparatus | 45 | tests/test_trace_enrich.py:3 |
| `cache/fingerprint.py` | dev apparatus | 33 | boltbeam/cache/__init__.py:11; tests/test_cache.py:8 |
| `policy/guards.py` | dev apparatus | 69 | tests/test_policy_guard.py:22; tests/test_policy_guard.py:34; tests/test_policy_guard.py:6 |
| `report/policy_consistency.py` | dev apparatus | 109 | tests/test_policy_consistency_port.py:5 |
| `diagnostics/packed_quant_control.py` | dev apparatus | 97 | tests/test_packed_quant_control.py:5; standalone -m CLI (qualification oracle) |
| `policy/descriptor.py` | dev apparatus | 197 | standalone -m CLI (research-policy reproduction and diff) |
| `cli/__main__.py` | keep on main | 2 | pyproject.toml boltbeam.cli:main and python -m boltbeam.cli |
| `collectors/cubin_launch.py` | keep on main | 89 | read-only fork extra/llm_research/decode/nv_cubin_ncu_launcher.py:22-23 imports and forwards main |
| `runtime/gpu_probe.py` | keep on main | 44 | workflow/layout.py:38,126 launches PROBE by file path |
| `runtime/ollama_decode_drive.py` | keep on main | 112 | collectors/ollama_decode.py launches DRIVER by file path |
| `runtime/tinygrad_decode_ab.py` | keep on main | 94 | search/role_compare.py:49 AB_DRIVER, used for model comparison |
| `runtime/tinygrad_decode_profile.py` | keep on main | 63 | collectors/tinygrad_role_time.py launches the profiling driver |
| `runtime/tinygrad_prefill_profile.py` | keep on main | 135 | collectors/prefill.py:34 DRIVER; runtime/tinygrad_session_profile.py imports sibling driver |
| `runtime/tinygrad_session_profile.py` | keep on main | 113 | collectors/session.py:56 DRIVER |
| `runtime/trtllm_decode_drive.py` | keep on main | 75 | collectors/trtllm_decode.py:31 DRIVER |
| `runtime/vllm_decode_drive.py` | keep on main | 51 | collectors/vllm_decode.py:28 DRIVER |

The active BubbleBeam/FutureSight/semantic campaign, timer, weight ledger, tie-out, `where_token_goes`,
and prefill/session paths remain on main. The old `semantic_candidate` apparatus is separate from the
live `semantic_candidate_plan` generator. Likewise the removed experiment `prefill_authority` validator
is separate from the live artifact adapter; similar names are not call edges. The cache store remains
used by the CLI while its unused fingerprint API moves to dev. The kernel-analysis product pipeline
remains on main while the uncalled handoff/candidate-qualification dependency chain moves to dev.

`packed_quant_control` and `policy.descriptor` have standalone argparse entrypoints but no current
product caller. They are an independent quantization oracle and a research-policy reproducer/differ,
respectively. Their command availability now follows the dev apparatus category; no replacement or
compatibility wrapper is introduced on main. `python -m ... --help` is checked on dev without GPU work.

## Validation

Each family is committed separately, merged into dev, and restored there with its original tests.
Each runs the complete dev Python suite, main/dev Go vet and tests, main/dev selfcheck, and both size gates.
The guarded landing checks local/origin equality, runs the main Go gate before outward merges, runs the
branch suites, pushes main/dev/exp, then publishes the mirror from origin/dev with its full leak grep.

Every family preserves the complete bytes of `screen results --run` for the saved external run
`qwen3-8b-apple_m3_10c-005`, on both main and dev. The baseline result SHA-256 is
`47f7ac8614a4f6a5244375e552efad8142a949f5b08e103e25c76617dc65b035`.
The dev code/tests are byte-identical to the baseline after restoration. The main CLI help is also
compared against dev. No GPU, remote machine or fork development is used.

Final main has **43,660** budgeted lines, down from **46,402** (2,742 fewer); dev stays
at **47,422** because the apparatus is preserved. The 48,000 cap and six authority checks are unchanged,
and the authority allowlist stays empty. Each dev suite has **1,912 passes, 27 subtests and 10 skips**;
Go has **74 main / 76 dev tests**. No tests are removed or added for this relocation.

The final graph contains 258 main modules (287 before), with no non-initializer module outside
the checked entrypoint/dependency closure. Package initializers outside that closure were also checked;
none remains. This is a current reachability and branch-category audit, not a claim that static analysis
can prove every externally possible use or every function-level dead-code question.
