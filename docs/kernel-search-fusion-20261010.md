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

## Phase 2: flash decode, generic, as a fusion over a layer's attention

The fork's flash decode now takes the model's attention shape (tinygrad-arkey `flash_decode_route_for`): the live-split
kernel's own spec validate admits a shape (Hq % Hkv, Hd against the lane and pair widths, the query group, the warps),
a measured tuning (the 32- and 40-head routes) is only a preference, and a shape the kernel refuses decodes on plain
attention instead of raising. The load schedule's pins (36 blocks, 32 heads, 8 KV heads, head dim 128) are replaced by
the wide substrate's own construction check.

In the search, a flash candidate covers attention_score, attention_softmax and attention_pv: two launches (the split
tile and the combine) per layer (`boltbeam/search/flash_space.py`, `role_compare.compare_flash`):

- BubbleBeam proposes tiles for the model's attention shape (read from the profile) from the chip's facts: the lane
  width from the subgroup size, stage widths 1, 2 and 4, the query groups the head ratio allows, both reduce
  structures, the split count the decode installs. The installed geometry is the first candidate.
- FutureSight applies the flash schema's legality (`build_flash_legality`) and orders the survivors.
- The provider (`search_provider --live-flash`) compiles, checks and times each; its compile now returns both launches
  with their buffers labelled, and its describe names the geometry the decode installs for the shape, or the kernel's
  refusal in its own words.
- BoltBeam rebuilds both launches, runs the tile then the combine on its own q and KV cache, checks the output against
  attention it computes itself (softmax(q.k / sqrt(Hd)).v), and times both launches with the kernel timer.
- The verdict needs two bars: the installed geometry timed the same way, and the in-model attention per token.

Checks (the reduced GPU budget):

| check | result |
|---|---|
| a model whose attention differs from the 8B's (the 0.6B, 16 query heads), logits against plain attention, d512, 8 tokens | the generic route admitted with no environment override; every token and argmax equal; max abs logit diff 0.015 (1.0e-3 relative) |
| the 8B's route on the CPU | the same binding as before (G4: split 48, no query group, stage 1); the 40-head shape keeps G5 |
| one search run, the 8B (its earlier in-model run reused) | 36 tiles proposed, 36 pass FutureSight, 36 measured; every output within 2.1e-7 of BoltBeam's attention; every BoltBeam reading noisy (each launch a few µs over the 4 µs floor, spread 11 to 17%): not reproduced by BoltBeam, nothing decided |

In the model, attention takes 0.302 ms per token at context 128 (8.4 µs a layer); BoltBeam's fastest flash pair reads
7.4 µs less the floor, noisy. A decision needs a longer context, where each launch is well over the floor.

The first search run of this phase failed on two setup faults, both fixed: the fork's single-role families now carry
`quants` (another change on exp) and the provider needs `BOLTBEAM_ROOT` to read BoltBeam's flash schema.

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
8. The fork's wide flash substrate and the single-stage candidate still pin Hd 128, 32 heads, 8 KV heads in their own
   construction checks; the load schedule now asks them, so other shapes keep the live-split route only.
9. The 32- and 40-head tunings (`FLASH_DECODE_TUNED`) are chosen by shape; the research leases keyed on the G4 tuning
   (coarse split, adaptive split) now key on that route's identity.
10. `_SHARED_Q8_LEASE` (blocks 1 to 12, 14 to 18 and 25) is a fixed block list on the q/k/v GEMV path, not on flash.
11. The flash candidates' cache length is the next power of two at or above four times the context; the in-model
    capture uses its own rule (at least 512).
12. `tie_out.kv_ms` counts every layer as an attention layer; a hybrid model's attention row limit is overstated.
