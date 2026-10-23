# Pairing kernels with roles: the layer walk (2026-10-11)

Which model role each captured GPU kernel ran for. One rule set for prefill and decode, in
`boltbeam/collectors/pairing.py`. `prefill_attribution.py` and `attribution.py` call it and build their rows from
what it returns.

## Why a redesign

The pairing read a capture as a bag of kernel templates and searched for an assignment of roles to templates. Each
engine added a rule to the search: duration clusters, a one-sided rate rule, chunk-width grouping, M-split launches, a
pre-filter, a launch-order tie-break, packed matrices, start-up trimming. Each new engine found a new bug. The search
could also tie, and a tie left roles unpaired.

## The model

A capture is a sequence of launches in the order they ran (`vendor_capture.sequence`). A transformer layer runs in a
fixed order that every engine keeps:

    input projections -> the block (attention, or the recurrent scan) -> output projection -> FFN in -> FFN out

The block's kernels are the ones `tie_out.kind_of` already names (attention, recurrent). They are the landmarks.
Everything else follows from counts:

1. **Layers.** The block launches, grouped into runs (consecutive block launches with no weight launch between them),
   are grouped into layers. The profile says how many layers of each kind the model has (`recurrent.layer_count`;
   the rest attend), so runs per layer per kind is runs over layers. A recurrent llama.cpp layer has two runs (the
   convolution, then the scan, with alpha and beta launched between them); a vLLM one has one.
2. **Sides.** Between two layers' blocks lie the earlier layer's output-side launches, then the later layer's
   input-side launches. The split is solved from the gap sizes: the launches before the first block are the first
   layer's input side; every gap is `after(kind_i) + before(kind_i+1)`; all gaps of one kind pair are equal. A chunk
   boundary (an engine that skips the last layer's FFN rows) may be short of trailing output-side launches, and a
   token boundary in decode holds the head and the next token's start between them. Those are the tail.
3. **Slots.** The j-th weight launch on one side of a layer of one kind is slot j. The same slot in every layer of
   that kind is the same role: an engine launches a layer's projections in one order. A slot may have another
   template in some layers (a mixed quant per layer), so a slot group is (slot, template) with the layers it occurs
   in. A role held in several formats keeps its slots in every layer of a kind, whatever the format; each
   (role, format) row takes those slots' launches of its own template, in the layers whose weights have that format.
   A hybrid's profile row says how many of its tensors sit in each layer kind (`layer_kinds`, read from the block
   names), since a format may run in a few recurrent and a few attention layers, not a whole number per layer of
   either. A GGUF's multi-token-prediction blocks (`nextn_predict_layers`) are no layer: the engine never runs them.
4. **Rows to slots.** A profile row has a fine class, which sets its stage (`vocab.STAGE`): input, gate, output,
   FFN in, FFN out, head. A row of several classes (attn_qo: q and o) holds one launch per class per layer. Input
   stages take input-side slots, output stages output-side slots; the gate projection may be on either side (the
   engine chooses when to compute it). On the output side the stages order the slots. On the input side, within one
   template, more operations per launch is more time per launch. A template serves one weight format.
5. **Checks, not rules.** A launch cannot read a role's weights faster than the measured bandwidth or run its
   operations past its path's measured peak. Those bound which slots a row may take. One row may hold several calls
   per launch (an engine that fuses gate and up) or several launches per call (a library that cuts a product along
   M); the ratio is whole and comes from the slot count.

Prefill and decode differ in two inputs only: the unit (a chunk of M tokens, or a token with M = 1) and the weight
launches (prefill: templates whose code holds a matrix instruction, or was not read; decode and the tail: every
kernel that is not a block kernel, bounded by the bytes check).

The output head runs once per engine call. It is paired on the tail of the chunk or token, the way decode pairs.

## What the evidence says

Every saved capture (`~/env/boltbeam-runs/.work/pairing-captures-20261011/`, pulled from the box) has this structure:

| capture | layers found | first gap | inside | gaps (after + before) | tail |
|---|---|---|---|---|---|
| 27B llama.cpp prefill 512, 2048, 8192, 32768; the session turn; the deep turn 512+283 | 64 per chunk | 1 (qkv) | recurrent 2 (alpha, beta) | rec->rec 6, rec->att 8, att->rec 5 | 4 (o, gate, up, down) |
| 27B vLLM prefill 512 (both captures) | 64 | 1 oddball + 2 (qkvz, ba) | 0 | 5, 4, 5 | 3 |
| 27B vLLM prefill 2048 (Marlin cuts M in two) | 64 | 1 + 3 | 0 | 9, 8, 9 | 6 |
| 27B llama.cpp decode at 512 and at 130,000; 8B llama.cpp decode (3 runs) | 64 or 36 per token | the layer's start | recurrent 9 | constant per kind pair, token boundary larger | head and sampling |
| 27B vLLM decode; 8B vLLM decode | 64 or 36 per token | the token's start | 0 | constant | head and sampling |

The side split is exact in all of them, and the slots line up: on the 27B's recurrent layers the input side is
[qkv, alpha, beta] and the output side [z, out, gate, up, down]; the attending layers [q, k, v] and [o, gate, up, down].
llama.cpp defers the z projection past the scan. vLLM's Marlin kernel serves every INT4 matrix, and the one tie the
old search had (ffn_down against the input projections, both 64 a token, equal bytes) is not a tie here: down is on
the output side, the projections on the input side.

## Before and after, every saved capture

The old pairing and the walk over the same 20 saved windows (no GPU). Rows: role rows in the table. Joined: rows
the old search had to join because nothing told them apart. Unpaired: roles no launch was paired with. Role ms: the
time the role rows hold.

| capture | before: rows, joined, unpaired, ambiguous | after: rows, joined, unpaired, ambiguous | role ms before / after |
|---|---|---|---|
| b26-27b-2048 | 7, 3, 0, no | 10, 0, 0, no | 367.32 / 367.32 |
| b26-27b-32768 | 7, 3, 0, no | 10, 0, 0, no | 5758.66 / 5758.66 |
| b26-27b-512 | 7, 3, 0, no | 10, 0, 0, no | 86.11 / 86.11 |
| b26-27b-8192 | 7, 3, 0, no | 10, 0, 0, no | 1456.31 / 1456.31 |
| b3-in-model-8b | 7, 0, 0, no | 7, 0, 0, no | 207.43 / 207.43 |
| d27-llama | 10, 0, 0, no | 10, 0, 0, no | 759.66 / 759.66 |
| d27-vllm | 7, 0, 2, yes | 7, 0, 5, yes | 319.78 / 252.90 |
| d8-llama | 3, 0, 2, no | 5, 0, 0, no | 351.51 / 593.44 |
| d8-vllm | 4, 0, 1, no | 4, 0, 1, no | 262.83 / 288.55 |
| ddeep-64at130000 | 10, 0, 0, no | 10, 0, 0, no | 762.04 / 762.04 |
| ll-27b-inmodel | 6, 0, 0, no | 10, 0, 0, no | 512.50 / 759.67 |
| ll-inmodel-8b | 7, 0, 0, no | 7, 0, 0, no | 207.39 / 207.39 |
| p27-llama-2048 | 7, 3, 0, no | 10, 0, 0, no | 340.03 / 340.03 |
| p27-llama-512 | 7, 3, 0, no | 10, 0, 0, no | 82.22 / 82.22 |
| p27-vllm-2048 | 5, 2, 0, no | 8, 0, 0, no | 464.47 / 464.47 |
| p27-vllm-512 | 5, 2, 0, no | 8, 0, 0, no | 117.54 / 117.54 |
| p27-vllm-ncu-512 | 5, 2, 0, no | 8, 0, 0, no | 119.10 / 119.10 |
| pdeep-795at108000 | 13, 6, 0, no | 10, 0, 0, no | 138.49 / 138.49 |
| pf0-27b-2048 | 7, 3, 0, no | 10, 0, 0, no | 338.07 / 338.07 |
| session-27b-512at16384 | 7, 3, 0, no | 10, 0, 0, no | 85.57 / 85.57 |

- Prefill (12 captures): the same time in role rows, every joined row split into its roles, nothing unpaired.
- llama.cpp decode (6 captures): the 8B in-model runs and the 27B runs pair as before. d8-llama now pairs gate+up
  and q/o; the old search had given ffn_down the gate+up kernel (274.86 ms) and left two roles unpaired.
  ll-27b-inmodel now pairs the four delta-net projections (attn_qkv, attn_gate, ssm_out, ssm_alpha_beta) the old
  search left in "other kernels".
- vLLM decode: both runs lack the engine's matrix list (their capture logs predate it), so q, k and v are separate
  rows while vLLM launches one packed qkv product. No pairing of them is right. d8-vllm leaves attn_kv unpaired
  where the old search left attn_qo; the four other rows pair. d27-vllm is refused as ambiguous before and after;
  the walk withdraws the rows it cannot tell apart (5 unpaired against 2).
- Fresh 27B vLLM decode capture (2026-10-12, d1bd988, one run): the capture now carries vLLM's matrix list, and
  every role pairs. `unpaired_roles` is empty and the run is not ambiguous. The packed products pair as
  in_proj_qkvz and qkv_proj, and their time splits into their roles by weight bytes. Decode 85.94 tok/s (85.6
  before). Tie-out: 11.025 ms of kernels plus 0.407 ms of gaps = 11.432 ms per token captured, against 11.635 ms
  untraced. The per-role capture attended context 145; the headline is at context 512. Evidence:
  ~/env/boltbeam-runs/.work/vllm-27b-decode-fresh-20261012/.

  | role | ms/token | limit ms/token |
  |---|---|---|
  | ffn_gate_up | 3.849 | 3.503 |
  | ffn_down | 1.945 | 1.751 |
  | lm_head | 1.624 | 1.502 |
  | attn_qkv | 0.869 | 0.773 |
  | ssm_out | 0.620 | 0.464 |
  | attn_qo | 0.571 | 0.464 |
  | attn_gate | 0.522 | 0.464 |
  | ssm_alpha_beta | 0.186 | 0.028 |
  | attn_kv | 0.061 | 0.052 |

## Rules the replay added

Found by replaying every capture, each a check the walk already implied:

- A template's format and launch ratio are bound from the counts, each launch counted once (a gap launch is a slot
  on both sides). Among the ways the counts balance, the one skipping the fewest calls wins.
- The data flow's order is compared within the layers two slot groups share (a unit's last layer may run extra
  launches before its FFN).
- A row of two classes on two sides (attn_qo: q before the block, o after) holds half its launches on each side.
- A unit's last layer may skip its output rows; the gap after it reserves no output launch.
- Output-side rows are placed first, in the data flow's order; then the rest, most bytes first.
- The pairing placing the most weight bytes wins (a light row fused onto a heavy row's launches never outbids it).
- Rows that could exchange their launches, or a row whose group has an equal free twin, are left unpaired and said.
- One node budget (50,000) over every split of the rows over the layer kinds.

## Nsight Compute, the oracle

ncu does not run in the pairing. It checked it once (`~/storage/bb-t3/ncu-src.csv`, the 27B vLLM prefill at 512):

- The cutlass kernel paired with `in_proj_ba` (BF16, 96 x 5120, 512 tokens) executed 162,816 HMMA.16816 per launch,
  6.67e8 operations: the role's GEMM padded to the kernel's 64-tile grid (2 x 512 x 128 x 5120 = 6.71e8). It is a
  weight GEMM of that shape and no other.
- The four delta-rule kernels, paired as the recurrent block and never as a role, executed 7,719,616,512 matrix
  operations per layer: the chunked delta rule's arithmetic to the last digit
  (prefill-vllm-crosscheck-27b-rtx5090-20261011.md). They are the block.

`tests/test_pairing_replay.py` on dev replays every saved capture from a reduced copy of its window and checks the
pairing against the one recorded here. `tests/test_pairing_ncu.py` holds the two ncu figures.

## Rules removed

Duration clusters and the cluster gap, the one-sided rate rule, `chunk_of` and per-width templates, `m_splits` with
its chunk divisibility, the assignment search with its score, ties and `MAX_ASSIGN`, the pre-filter, the launch-order
tie-break and `order_violations`, `role_groups` as a pairing step, decode's depth-first `_pairing`, `assemble`,
`split_clusters`, the count tolerance, `shared_kernels`. Start-up trimming stays: it decides which launches are the
decode's, before any pairing (`attribution.shared_window`).

Kept: the window rules (warm-up halves, the repeated turn, the long-minus-short difference), the block kinds, the
packed matrices (`prefill.engine_roles`: what the engine's matrices are, not how they pair), the engine's own
program metadata, the peak and bandwidth checks, the fused-launch allowance (now a slot ratio), and every row builder.

## What changes in the output

- A role is one row over every chunk; the deep turn's "(512-token chunks)" and "(283-token chunks)" rows are one row
  whose launches carry their own chunk width.
- `ffn_down` and `ffn_gate_up` are two rows: the FFN's output projection follows its input projections, so time can
  tell them apart. Rows the old search had to join by operations ("ffn_down + ffn_gate_up") stay joined only on the
  input side, where no order separates them.
- A row the walk cannot place says why: no slot of its side and kind fits its bytes, or the layers could not be told
  apart (no attention or recurrent kernel named). A capture with no launch times (an old export) pairs the flat way:
  counts per unit, bytes and time, one side.

Principle: centralization of authority. One module owns how a launch becomes a role; prefill and decode are its two
callers, and the data-flow table lives in `vocab`.
