# The first 27B run: three faults the hybrid model found, RTX 5090, 2026-10-10

A 27B hybrid GGUF in Q4_K_M (hidden 5120, 64 layers: 16 attention layers and 48 gated-delta-net layers) on llama.cpp CUDA, the
same box and build as docs/in-model-vs-generic-rtx5090-20261010.md. The 8B had passed both paths; the 27B broke both,
each time on a shape the 8B does not have. Fixed on main the same day; the tables below are the box's reruns on the
fixed code with the GPU held by nothing else (the resident llama-server stopped for the session, nvidia-smi showing
only the pipeline's own llama-bench).

## Before

In-model (nsys), per role, main 800271b:

| role | calls/token | MB/call | µs/call | GB/s | % peak | reason |
|---|---|---|---|---|---|---|
| ffn_down Q4_K | 64 | 50.2 | 66.4 | 756 | 44.6 | too small to fill memory |
| attn_qo Q6_K | 16 | 25.8 | 19.3 | 1,337 | 78.9 | too small to fill memory |
| attn_kv Q8_0 | 32 | 5.6 | 4.7 | 1,198 | 70.7 | too small to fill memory |
| attn_qo Q8_0 | 16 | 66.9 | 41.8 | 1,601 | 94.5 | at the limit |
| lm_head Q6_K | 1 | 1,043.5 | 614.3 | 1,699 | 100.3 | at the limit |
| ffn_gate_up Q4_K | | | | | | unpaired (count 128) |

Other kernels 7.605 ms of which "elementwise" 6.175 ms: the real ffn_down launch (2.146 ms) and the four Q8_0
projection GEMVs of the delta-net layers (3.923 ms) both sat under that word.

Generic (`--role-time generic`): refused, "incomplete measurement: 2 role(s) measured more than ±1.4% below their
floor (attn_kv Q8_0 0.00 < 0.11 ms, attn_qo Q8_0 0.00 < 0.63 ms), so the capture missed kernels". The two Q8_0 rows
were `not_measured` ("no block layout for Q8_0") and carried 0 µs, which the floor rule read as a time.

## After

In-model (nsys), b10 11a53b7, 71.6 tok/s, token 16.960 ms, limit at context 161 7.310 ms:

| role | calls/token | MB/call | µs/call | GB/s | % peak | reason |
|---|---|---|---|---|---|---|
| ffn_gate_up Q4_K | 64 launches, gate and up fused | 100.4 | 66.4 | 1,512 | 89.2 | at the limit |
| ffn_down Q4_K | 64 | 50.2 | 33.5 | 1,497 | 88.4 | at the limit |
| attn_qo Q6_K | 16 | 25.8 | 19.3 | 1,338 | 79.0 | too small to fill memory |
| attn_kv Q8_0 | 32 | 5.6 | 4.7 | 1,199 | 70.8 | too small to fill memory |
| attn_qo Q8_0 | 16 | 66.9 | 41.8 | 1,601 | 94.5 | at the limit |
| lm_head Q6_K | 1 | 1,043.5 | 614.2 | 1,699 | 100.3 | at the limit (within ±1.4%) |

Tie-out: limit 7.310; weight kernels above their ideal 0.850; other kernels above their ideal 5.434 (ssm projection
3.923, other 0.518, quantize 0.388, norm 0.361, elementwise 0.106, copy 0.066, rope 0.041, attention 0.032, KV cache
write 0.024); gaps 3.366; token 16.960. Not attributed 5.459 ms, all of it named.

Generic, b10, 71.6 tok/s, token 13.966 ms untraced, measured, no refusal:

| role | MB/call | µs/call | less floor | GB/s | % peak | reason |
|---|---|---|---|---|---|---|
| ffn_gate_up Q4_K | 50.2 | 36.2 | 31.9 | 1,574 | 92.9 | at the limit |
| ffn_down Q4_K | 50.2 | 36.5 | 32.2 | 1,558 | 92.0 | at the limit |
| attn_kv Q8_0 | 5.6 | 9.5 | 5.2 | 1,080 | 63.8 | too small to fill memory; noisy: ±12.8%, 4.3 µs floor under a 9.5 µs kernel |
| attn_qo Q6_K | 25.8 | 21.9 | 17.6 | 1,466 | 86.6 | at the limit |
| attn_qo Q8_0 | 66.9 | 45.0 | 40.7 | 1,644 | 97.0 | at the limit |
| lm_head Q6_K | 1,043.5 | 618.0 | 613.7 | 1,700 | 100.4 | at the limit |

Sum alone less the 4.3 µs floor 7.9 ms, fits the 14.0 ms token. The Q8_0 rows are timed because the adapter now
exists; without it they would read "not timed: no Q8_0 adapter" and the table would still be measured. The isolated
cross-check also timed the two ssm_projection Q8_0 shapes (48x5120 at 6.2 µs, 5120x6144 at 26.4 µs, 1,501 GB/s less
floor); they are not limit roles, so they are not in the table.

## The three rules

1. Attribution (`collectors/attribution.py`, `PAIRING_RULE`): every role is placed at once. A role may take a group
   that runs its count of times per token, or 1/k of them for a fused kernel moving k calls' bytes, at no more than the
   peak; within one launch count more bytes per launch is more time per launch; the assignment placing the most roles
   with the fewest fused kernels wins; a group no role's bytes fit stays unattributed. The old walk placed ffn_down
   first and, both launches being 64 per token, gave it the 100 MB one (755 GB/s, plausible on its own). The 8B
   escaped because its counts differ (36 and 18).
2. Tie-out (`workflow/tie_out.py`, `KIND_WORDS`, `UNATTRIBUTED`): a GEMV no role took ("mul_mat_vec", "kernel_mul_mv",
   "gemv") is "weight kernel, unattributed", its own line with the kernels named, never elementwise. A GEMV whose
   launches per token equal the count of a profile matrix role the limit has no bytes for is named after that role
   inside other kernels ("ssm projection"): a label, nothing attributed. The capture carries no bytes for a kernel no
   role took, so the line names the kernel, its launches and its µs per launch.
3. Not timed (`collectors/tinygrad_role_time.py` loss): an isolated row with a status other than measured (no adapter
   for its format, a failed check) is not a time. It is left out of the roles, the floor rule and the sums and listed
   as "not timed: <reason>", its own row in the report, summary and TUI. The engine adapter (`engine_kernels.py`) says
   "no <quant> adapter" in that row.

## The Q8_0 adapter

Built, both backends, within the day. `metal_native.py` holds the block (34 bytes, 32 weights, `BLOCK_ELEMS` joins
`BLOCK_BYTES`; the reference and every `cols // 256` in the timer now read it) and `dequant_q8_0_block`. CUDA:
`mul_mat_vec_q<GGML_TYPE_Q8_0, 1, false, small_k, halve_iters>` with `qi = QK8_0 / (4 x QR8_0) = 8` and
`VDR_Q8_0_Q8_1_MMVQ` read from the engine's headers; the q8_1 reference as for Q4_K; 6 of 6 roles and both ssm shapes
pass on the 5090. Metal: `kernel_mul_mv_q8_0_f32` is dispatched by the engine's second rule (ggml-metal-ops.cpp): F32,
F16, BF16 and Q8_0 share one row set per threadgroup, ceil(rows / nr0) threadgroups, 32 x 4 x nr0 bytes of
threadgroup memory; with the K-quant grid the kernel wrote only the first nr0 rows. Verified on the Air against the
reference (1024x5120 err 4.7e-8, 48x5120 err 1.0e-7); a live test skips where the ggml library is absent.

## Found, not fixed: the profile misses the delta-net in-projections

The 48 gated-delta-net layers each have `attn_qkv.weight` (5120 -> 10240, Q8_0, 54.5 MB) and `attn_gate.weight`
(5120 -> 6144, Q8_0, 33.4 MB). `profile/roles.py` has no pattern for either name, so they classify as `other` and
`profile_from_gguf` drops them: 4.2 GB per token, 2.5 ms at 1,694 GB/s, missing from the limit, which is why the
roofline says 137 tok/s against a 71.6 tok/s token that is at the limit on every weight role it names. The profile
does list `ssm_out` (5120x6144) and `ssm_alpha`/`ssm_beta` (48x5120) as `ssm_projection` with shapes, but both are
Q8_0 under one role name, and the per-role machinery keys on (role, quant), so they cannot join the limit as two rows
yet. Until the taxonomy names the in-projections and roles key on shape as well, the ssm projections stay out of the
ideal and are named in other kernels (3.9 ms per token here). That is the next piece, under `[profile]`.
