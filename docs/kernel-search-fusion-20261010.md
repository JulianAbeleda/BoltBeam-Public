# Fusion candidates in the kernel search, and BoltBeam's own timing (2026-10-10)

## What changed

Run's kernel search can now propose kernels that span several roles: a **fusion candidate** replaces the kernels of
several adjacent decode-graph nodes with one launch. It runs after the per-role search, through the same pipeline
(`docs/semantic-campaign.md`), and BoltBeam checks it with its own tools.

| step | who | what |
|---|---|---|
| families | the provider's `describe` (`decode_fusions`) | each fused emitter: the nodes it covers, its operands in parameter order, its coupled rows |
| adjacency | BoltBeam (`boltbeam/data/decode_graph.json`) | the covered nodes must be one connected piece of the decode graph with one output; the provider's operands must be exactly the values the graph says they read, and each node's op the graph's |
| the model | BoltBeam (the GGUF's own tensors) | the layers that hold every covered weight in the family's quant, at one shape; a tie-out row counts only when the fusion replaces all of it (every fine role and every tensor in it) |
| the shape | the emitter's own validate (`describe` with `shapes`) | each coupled row accepted or refused at this shape, with the emitter's words |
| FutureSight | `futuresight_adapter.assess_population` | the chip's legality rules on the population, as for a role |
| campaign | the provider | measures the survivors; its times only order them |
| correctness | BoltBeam (`fusion_space.Reference`) | the fused output against the covered nodes computed one after another by BoltBeam on the model's bytes; each node's kernel alone checked the same way |
| time | BoltBeam's kernel timer | the fused kernel, and each covered node's kernel alone, less the dispatch floor |
| verdict | `role_compare.fusion_verdict` | below |

The verdict: BoltBeam's fastest correct fused kernel, times the launches per token, must beat

1. the in-model time of the tie-out rows it replaces (the "where the token goes" rows), by more than the chip's band, and
2. BoltBeam's unfused side timed the same way: the covered nodes' kernels alone, less the floor (a lower bound when a
   node's reading was noisy).

Only then is it "found, not applied" (the whole-model A/B does not install a fused emitter yet: decided by binding).
Rule 2 was added after the first 27B run: the model already ran the same fused emitter in the model, and an isolated
time against an in-model time showed a 2.9% "gain" that was only the difference between the two ways of timing.

A covered node with no kernel of its own in the capture (silu_mul, a residual add) counts 0 ms on the model's side,
which favours the model.

Emit (`gameplan.md`) shows one block per fusion: the roles it replaces, their combined ms lost in the model, the
fused and unfused times per token, what was refused and why, and the verdict.

## Decisions use BoltBeam's own time

The provider's median now only orders the search. Every per-role and fusion decision uses BoltBeam's kernel timer
less the dispatch floor. The provider's figure is kept beside it, with the ratio and whether it is inside the chip's
band, as a note (`provider_check.judge`, `provider_note`).

The reason: on the RTX 5090 the fork's device timestamps sit 2 to 9% from BoltBeam's time less the floor, outside the
1.4% band, so the earlier rule (the provider's claim must reproduce within the band) could never promote anything
(`docs/kernel-search-generic-20261010.md`). A kernel BoltBeam cannot rebuild, finds incorrect, or reads noisily is
"not reproduced by BoltBeam" and decides nothing.

Removing the band gate exposed a false comparison in the per-role search: the model's fused gate and up read 72 tensors
in 36 calls, and one plan's single-tensor GEMV was compared with one fused call (1.83x "faster"). The model's time is
now divided per tensor read (the profile's count), so a plan is compared with the work it replaces.

## Phase 1: the fused gate/up (`q4_w1w3.v1`), RTX 5090

One search run on the 8B (Qwen3-8B Q4_K_M), the pipeline with the search on (`--provider tinygrad --role-time in-model`).
A 27B run was made before the GPU budget was cut; it is kept because it found rule 2.

| | 8B (12288x4096, 36 launches) | 27B (17408x5120, 64 launches) |
|---|---|---|
| proposed rows | 3 | 3 |
| refused by the emitter's validate | 0 | 1 (quad: `blocks_per_group == 4` required, got 5) |
| refused in the campaign | seed and control (`unsupported_plan`: a fusion runs only through its emitter) | same |
| BoltBeam fused, fastest (vector loads) | 36.7 µs less floor, 1.321 ms/token | 64.7 µs, 4.142 ms/token |
| other rows | scalar 38.3 µs; quad 55.0 µs | scalar 66.6 µs |
| provider's time for the fastest | 38.5 µs (1.05x, outside the 1.4% band; ordering only) | 64.9 µs |
| correctness, max rel err against BoltBeam's unfused reference | 1.2e-7 | 3.6e-7 |
| unfused alone (gate, up, silu_mul) | 20.0, 18.7, 2.2 µs (silu_mul noisy) ≥ 1.392 ms/token | 31.6, 31.8, 0.4 µs (noisy) ≥ 4.060 ms/token |
| in the model (ffn_gate_up row) | 1.300 ms, 0.094 ms lost | 4.265 ms, 0.471 ms lost |
| verdict | none faster: 1.321 against 1.300 ms | none faster: 4.142 ms is not under its own nodes alone (4.060 ms) |

The 27B's model already runs `q4k_g3_lanemap_gemv_w1w3vec16`; the 8B's runs `q4k_gate_up_four_warp_vec_fp16`.
Both verdicts were re-judged from the run files with the final rule, without the GPU.

Correctness in the model: one logits check on the 8B, control against the fused `q4_w1w3` route
(`TINYGRAD_Q4K_GATE_UP_FOUR_WARP_DISABLE=1`), d512, 8 decode tokens: every token and argmax equal, max |Δlogit|
0.025 (1.3e-3 of the largest logit); not bit-exact, the accumulation order differs from the four-warp kernel.

Per-role search on the same 8B run, judged on BoltBeam's own time: ffn_gate_up Q4_K 18.3 µs against the model's
18.1 µs per tensor (refuted); ffn_down Q4_K and lm_head refuted; ffn_down Q6_K and attn_kv Q6_K slower than the
model; attn_qo Q4_K and attn_kv Q4_K not reproduced (BoltBeam's readings noisy).

## Suspects

Found here:

1. `decode_emitter_space.installed_row` is treated as the model's own kernel, but the 8B on the 5090 runs other
   kernels in the model (`q4k_gate_up_four_warp_vec_fp16`, `q4k_fp16_mmvq_direct_vec_*_epi_ffnresadd`). A per-role
   "refuted: the model's own kernel is fastest" means the installed emitter row, not the in-model kernel.
2. The in-model row attn_qo Q4_K holds 54 of its 72 tensors' calls; 18 q projections run
   `q4k_warp_coop_q8_dp4a_direct_4096_4096`, which the capture tags `unknown`. Its per-tensor time is understated.
3. `Q4KGEMVEpilogue.validate` pins `rows == 4096` for `ffn_down_fused` and `ffn_down_resadd`, and
   `Q6KGEMVRouteSpec.validate` pins it for `ffn_down_resadd`: the epilogue fusions refuse every other hidden size.
4. The quad fused gate/up requires `blocks_per_group == 4`, that is k = 4096 only.
5. `installed_row` for Q6_K reads `Q6K_COOP_ROW_TILE_BY_TARGET`: a target-named tile, and on Metal a tile that
   fails the coop single-warp rule.
6. The fusion's value vectors and the residual dtype (fp32) are the fork family's declaration; BoltBeam binds what
   the record's buffers say and checks the result, but the dtype of the model's residual stream is not read from the
   model.
7. The provider's elementwise kernels alone (silu_mul, the residual add) are mostly dispatch floor: BoltBeam reads
   them noisy, so the unfused side is a lower bound.
