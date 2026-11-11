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

The block's kernels are the ones `kernel_kind.kind_of` (in `collectors/`) already names (attention, recurrent). They are the landmarks.
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

## Unresolved pairings and the report (2026-10-23)

An unresolved pairing carries one of three kinds (`pairing.TIE`, `SEARCH_LIMIT`, `NO_FIT`), because each one needs a
different fix:

- **tie**: several pairings fit equally, and the rows that could swap launches are withdrawn. The fix is a feature
  that tells the rows apart, such as their tensors' formats.
- **search_limit**: the walk stopped at its node or launch-visit budget before it finished. The fix is pruning or the
  budget, not features.
- **no_fit**: no slot fits a row's launches. The fix is the slot model or the checks.

A prefill capture's reason can join several kinds, such as rows no slot fits beside a tie, so it records `kinds` as
well as `kind`, which is the first of them in the order search_limit, tie, no_fit. A trace written before the kinds
were recorded gets them from the reasons the search wrote, one part at a time (`pairing.ambiguity_kinds`).

Report validation (`report_validation.unresolved_attribution`) lists every unresolved pairing in a run under
`unresolved`, with its file, kind and reason, and whether the report shows it. It fails only when the report claims the
attribution without showing it:

- A decode report shows its engine's trace as `loss.ambiguous`, and the TUI names the kind. An engine trace whose
  pairing is unresolved therefore passes when the report shows it.
- Another engine's trace in the same run is not the report's table, so it is recorded and not failed.
- Prefill, session and replay reports have no place to show an unresolved per-length capture, so one fails the
  report. Before this change, session and replay captures were not checked at all.

A release's render gate compares `unresolved` against the expectations it saved, and fails on a new entry or on a
higher unpaired-role count.

Principle: disclose, not gate. Unresolved attribution is a stated property of the run, like capture overhead. Hiding it
is a failure.

## The model's own layer sequence (2026-11-09)

The walk above knows each layer kind by its count, and one slot pattern per kind. A Nemotron-H hybrid (Nemotron 3.5
Lightning 30B-A3B: 52 layers, each one of mamba, moe or attention, in the order its config names) breaks that: a
standalone expert layer has no block kernel to mark it, and one follows 17 of the 23 mamba layers and all 6 attention
layers. On the stored TensorRT-LLM H200 decode capture the walk put the expert launches on "the output side of
recurrent layers", where no slot held a launch in every layer: the five MoE rows and both attention rows were unpaired
(`no slot ... fits 1449 launches`), with a tie on attn_qo.

- **The profile records the sequence.** A config that names each layer's block (`layers_block_type`: Jamba, Nemotron
  3.5; Nemotron-H's `hybrid_override_pattern` letters M, *, -, E) is read into `metadata.layers.sequence` in the walk's
  kinds (`vocab.LAYER_KIND_OF_BLOCK`: mamba is recurrent, mlp is ffn; `profile/loaders._layers_from_config`). A block
  the table lacks, or a count the config contradicts, records nothing, and the walk goes by counts as before.
- **The walk follows it** (`pairing.sequence`, through `pair(order=)` from `pairing.layer_order(profile)`; every engine
  reaches it through `attribution.attribute` and `prefill_attribution`). A layer of a kind with no block kernel runs in
  the gap after the block layer before it, on that layer's output side after its own rows. A block layer followed by
  such layers launches differently from one that is not, so it is a kind of its own, as `signatures` already makes of
  layers that launch differently: `recurrent/moe` (17 layers), `recurrent` (6), `attention/moe` (6). The rows of the
  unmarked kind's classes (`vocab.LAYER_CLASSES`) take those kinds' layers through `layer_kinds`, as many per layer as
  their count over the sequence's layers of the kind. The output side's data flow then orders the expert rows after
  the layer's out projection, router and experts' gate_up before the experts' down. A sequence that begins with an
  unmarked layer, names a kind the walk has not, or contradicts the profile's counts is `NoStructure`, said in the
  trace's `structure.flat`. A trace walked this way carries `structure.sequence` and `SEQUENCE_RULE` in its rule.
- **Nothing changes without a sequence.** The stored llama.cpp decode captures of Qwen3.8-27B UD-Q4_K_M (H100, three
  pairs with replicas) and Qwen3-14B Q4_K_M (H200) re-attribute to byte-identical traces.

On the stored capture, 10 of 11 rows pair (before: 3): ssm_in, ssm_out, the experts' gate_up (the cutlass MoeFCGemm)
and down (the grouped GemmUniversal), both shared-expert rows, attn_qo's q (the qkv nvjet) and o, attn_kv, lm_head;
moe_router is a tie, said. What the walk still gets wrong there is not the sequence: in decode every unnamed kernel
is a candidate, and a row of 0.7 MB (the router) or 2.8 MB (k and v) fits any launch slower than its bytes need. The
router's first fitting slot is cuBLAS's `splitKreduce_kernel` of the out projection (a reduction, 1.7 us), with a
twin in the router's own fp32 reduction (the tie); attn_kv takes that reduction as two fused calls; and the shared
expert's gate_up (20 MB) takes the router's nvjet (5.7 us) ahead of its own (8.9 us), so shared down takes the gate_up's.
The fix is a row of data, not a walk rule: name cuBLAS's split-K reduction and torch's elementwise kernels for what
they are in `tie_out.KIND_WORDS`, which `projection_candidate` already honours. It is not in this change because every
stored vLLM AWQ capture holds thousands of those launches as slots, so their traces' `structure.sides` would move.

## Helpers are no slots; a packed launch; the order bytes give at M = 1 (2026-11-10)

Three things the H200 TensorRT-LLM capture needed, found by re-attributing it offline (no stored llama.cpp decode trace
changes its pairs except where the walk was wrong, below; the 37 pairing fixtures re-record one, DeepSeek-V2-Lite).

- **A library's helpers carry no weight** (`tie_out.KIND_WORDS`, data): cuBLASLt's `splitKreduce_kernel` (a split-K
  GEMM's reduction), torch's `reduce_kernel`, its `elementwise_kernel` family (fills, compares, index puts,
  scatter/gather, arange), CUB's `DeviceScan`, torch's `indexSelect` gather and sort (`Sort`: `bitonicSortKVInPlace`,
  an MoE dispatch's `fusedBuildExpertMapsSortFirstToken`), and the dispatch's row gather (`expandInputRows`) and
  stride table (`computeStrides`). `projection_candidate` reads the matrix words (gemm, gemv, matmul) in a kernel's
  own name, not in its template arguments: `expandInputRowsKernel<..., GroupedGemmInput>` gathers rows for the
  grouped GEMM and reads no weight. Every stored capture's kernel names were listed against the new words: no weight
  kernel is caught.
- **A layer's input rows that outnumber its launches run as one launch** (`pairing._packed`): TensorRT-LLM (and
  vLLM's `qkv_proj`) run q, k and v as one GEMM where the profile has two rows. A kind's input slots, less the launch
  the layer before owes its out projection, hold one launch per attention layer; the rows are then one item of that
  launch (parts), each part's calls per launch its own (`attribution.attribute` splits the launch's time by bytes and
  records each part's calls per launch as `fused`). Before, the walk gave the launch to q and said k and v fit no slot,
  and, with every row owed a launch, the preceding mamba layer's out projection lost its slot too and the search ran
  out of budget. The RTX 5090 vLLM BF16 fixture (`d8-vllm`) keeps its recorded result: its frozen candidate list still
  holds the helpers as slots, so its room is not one launch.
- **At M = 1 a launch's work is its weight bytes.** On one template, among rows of one format on one side of a kind, a
  row of more bytes per launch takes no faster launches than a placed row of fewer, and one of fewer no slower, where
  the two slots' inner spreads (`SPREAD`: the launches between the 10th and 90th percentiles; a thousand launches have
  outliers at both ends) do not overlap and the bytes differ by the policy `operations_order_ratio` — the input side's
  operations order (prefill, `nk`) carried to bytes and to the output side. Across templates a kernel's floor differs,
  so the order is a preference the walk applies when two rows could exchange their launches: the pairing that gives
  the row of more bytes the slower launches is the one shown (the data flow's order and the one-template order
  re-checked on it), instead of both rows withdrawn as a tie. The router (0.7 MB, a latency-bound split-K GEMM at 5.7
  us) and the shared expert's gate_up (20 MB at 8.9 us) are that exchange on the H200 capture.

On the stored capture every row pairs (11 of 11; before 10, with the router a tie and three rows on wrong launches):
ssm_out, the router, the shared expert's gate_up and down, and the experts' gate_up and down on their own launches in
that order after each mamba or attention block; q, k and v as one packed launch; o; lm_head. Not attributed: 1.350
ms/token of 3.498 (before 1.440). On the stored llama.cpp runs: Qwen3-14B Q4_K_M on the H200 had attn_kv's k and v on
40 of the q/o launches (grid 5120, 12.8 us) and attn_qo on 40 of the k/v launches (grid 1024, 4.0 us), inverted; the
one-template order puts each on its grid. Flash-Next Q4_K_XL on the H200 had the dense FFN's down and the shared
expert's down as a said exchange in every layer (25 rows unpaired); their bytes tell them apart, 5 rows stay
unpaired (q/o against k/v, the router's twin) and 0.35 ms/token less is unattributed. The other six llama.cpp runs
(H100, L40S, RTX 3090, 4090, 5090) re-attribute to the same pairs. The evidence runs' summary.md and results.json
re-render identically.

What this capture still cannot say: the TensorRT-LLM driver's packed matrices were not read (`trtllm_decode_drive._matrices`:
`'GenerationExecutorProxy' object has no attribute 'engine'`), so the qkv launch is read from the counts, not from
the engine's own say; and the Flash-Next shared expert's three 1.7 MB rows (gate, up, down, each 640 x 2560) have
equal bytes, so the walk cannot tell the down's launches (grid 2560) from the gate's and up's (grid 640) without the
grid, which decode does not read.

## A matrix-vector launch's grid reads the rows it writes (2026-11-10)

The section above left the Flash-Next capture with 5 roles unpaired and said only the grid could tell the shared
expert's rows apart. The capture already holds the grid, and it settles all three cases.

- **The contract.** llama.cpp's matrix-vector kernels launch grid.x = ceil(rows / rpb) over the rows they write:
  `mul_mat_vec_f` one block per row (mmvf.cu), `mul_mat_vec_q` rpb 1 at one column, 2 at 2 to 8 columns, and nwarps
  (2, 4 or 8 by device table) when its template's small_k is set (`<type, ncols, has_fusion, small_k, halve_iters>`;
  small_k is chosen when K is too short to keep the warps busy, only where nwarps > 1). Checked against the selected
  capture of all 12 stored llama.cpp decode runs (14B and 27B on H100, H200, H200 NVL, L40S, RTX 3090, 4090 and 5090,
  and Flash-Next): every one of their 158,630 mul_mat_vec launches has a profile matrix's rows at that rpb, except
  Flash-Next's F32 products of grid 4 and 1, which read two small vectors the profile excludes without recording rows
  (hc_*_inject, ffn_gate_inp_shexp). rpb is 1 throughout but on Flash-Next's small_k launches, where it is 4. A
  template of another arity with a flag set leaves rpb open.
- **Where it is not used.** mul_mat_q has its own proof (`llama_mmq.fits`, prefill). cuBLAS and TensorRT-LLM's nvjet
  GEMMs tile and split K: on the H200 Nemotron capture ssm_out and the shared expert's down, both 2688 rows, launch
  2x63 and 4x22, and lm_head (131072 rows) 2x66; cutlass's grouped GEMMs are persistent (264x1, 1x132). Their grids
  state no rows, and the walk reads nothing from them (`llama_mmq.rows_per_block` is None).
- **The walk.** `_fits` rules out a launch whose grid cannot cover the row's output rows (the profile's rows, passed as
  n by `attribution.attribute` from the role's shape); among the groups a row fits, one whose grid confirmed it comes
  before one of a kernel that states no rows, an option all of whose launches confirm it comes first at any launch
  ratio, and a free group stating no rows is no twin of one whose grid confirmed the row.

What the Flash-Next capture shows (H200, llama.cpp 50569eb87; the model is a gated-delta-net hybrid with a lightning
indexer, a 512-expert MoE with a shared expert, and mHC hyper-connections):

| role | before (6d1f2fc6) | after |
|---|---|---|
| attn_qo BF16 (indexer.q_proj, 512 x 2560) | exchange with attn_kv BF16, unpaired | mmvf grid 512 |
| attn_kv BF16 (indexer.k_proj, 128 x 2560) | exchange with attn_qo BF16, unpaired | mmvf grid 128 |
| moe_router F32 (512 x 2560) | took a 4-row F32 product, its own grid-512 launch the twin: unpaired | mmvf grid 512 |
| moe_shared_expert_gate_up (640 x 2560, two) | its fused grid-640 launch and a 10240-row hc up launch, one call each | the fused grid-640 launch, two calls |
| moe_shared_expert_down (2560 x 640) | small_k grid 640 | small_k grid 640 (rpb 4: 2560 rows) |
| ffn_gate_up (hc_ffn_up, 10240 x 320) | grid-320 launches (320 rows) | small_k grid 2560 (rpb 4: 10240 rows) |
| ffn_down (hc_ffn_down, 320 x 10240) | grid 320, and one launch a token of 2560 rows | grid 320 |

Unpaired 5 -> 0, no tie; not attributed 4.131 -> 4.158 ms/token (the roles left launches that were no role's). The
other stored captures re-attribute byte-identically; one prefill fixture (Qwen3-0.6B's pruned last layer) re-records
because ffn_gate_up no longer takes the head's 151936-row GEMV as a second launch. The diagnosis the section above gave
was half right: the shared expert's down does not launch grid 2560. It is small_k, four rows a block, so it launches
grid 640 like the fused gate and up; the grid-2560 launch beside them writes 10240 rows (an mHC up projection).

The evidence run's report (docs/evidence/gpu-sweep-20261109/runpod-h200/h200-flash-next-q4kxl-llama, re-rendered with
the re-attributed trace): 17 of 21 weight roles timed (14), weight kernels above their ideal 3.614 ms (3.703) and
unattributed 0.949 ms (0.859), the router and the indexer's q and k get rows, the shared expert's gate_up halves to its
one launch (0.22 ms), and report_validation.json lists no unresolved pairing (the tie). results.json is unchanged.

What the capture still cannot say. The profile classes the mHC tensors by substring (`profile/roles.py`: `ffn_up`,
`ffn_down` match `hc_ffn_up`, `hc_ffn_down`), so ffn_gate_up and ffn_down here are the FFN sublayer's hyper-connection
pair, and `hc_attn_down`/`hc_attn_up` are unclassified weights. The launch order is hc down (320 rows) then hc up (10240
rows), the reverse of an FFN's in-then-out, so the data flow's order puts ffn_down on the next 320-row launch after
ffn_gate_up: the next sublayer's hc_attn_down, of the same shape, format and kernel, and about as long. No grid
tells those two apart; the times attributed are the same, the launch is not. The fix is the classifier (an mHC class of
its own), not the walk.

## The hyper-connection mixers and the indexer have stages (2026-11-10)

The section above ends on the classifier: 0.3.17 gave Flash-Next's mHC tensors and its QSA indexer roles of their own
(`hyper_connection`, `attn_indexer`), with no stage and outside the limit roles, so on a capture their launches were
weight kernels no role took. Where the model runs them (llama.cpp `models/qwen4exp.cpp`): `build_hc_mix` reads a
low-rank down and up before each layer's token mixer (`hc_attn_*`), again before its FFN (`hc_ffn_*`) and once before
the head (`output_hc_*`); the indexer's k and q projections read the layer's input before attention reads the cache.

- **Data, in vocab.** The mixer is four fine classes, each its own group, since a profile row is one group, shape and
  format and so holds one stage: `hyper_connection_attn` at `mix_in` (a new input-side stage), `hyper_connection_ffn`
  at `mix_out` (output side, after the out projection, before the FFN: `OUTPUT_ORDER` is gate, out, mix_out, ffn_in,
  ffn_out), `hyper_connection_head` at `head` (the tail), `hyper_connection_inject` with no stage;
  `attention_indexer` at `in`. `vocab.STAGES` is the stage order the walk reads. No rule in the walk changed.
- **The limit.** The mixers' down and up and the indexer's projections are read once per token, each its own
  matrix-vector launch, so they are limit roles: their bytes leave the ledger's unclassified rows for the roles and
  the limit's total is the same. The inject (4 x 10240 F32, 160 KB a layer pair) stays the ledger's small_vector
  exclusion, as the shared expert's gate vector: its launch is four blocks wide and floor-bound, slower than the
  router's 5 MB launch, so the M = 1 bytes order holds it no slot of its own (as a role it took one launch for two
  calls). Its launches (grid 4, about 0.3 ms a token) stay unattributed. A profile written by 0.3.17 holds the
  mixers both as rows and as unclassified: `limit_roles` counts them once.

On the stored H200 capture (its profile rebuilt with each tree's classifier), 946de221 against this: 15 pair rows to
23, nothing unpaired, no tie, not attributed 4.682 to 3.682 ms/token. Each mixer's down launches grid 320, its up
(10240 rows, K 320: small_k, four rows a block) grid 2560; the token mixer's before the input projections, the FFN's
between the out projection and the router, the head's on the tail; the indexer on BF16 grid 128 and 512. The twelve
other stored captures re-attribute to byte-identical traces (`tests/fixtures/pairing/h200-flash-next-decode-4-mixers`).

What the report still cannot show: the llama.cpp trace's kernel rows carry no shape, so `tinygrad_role_time.loss`
cannot time a (role, format) the ceiling holds at two shapes (the mixers' down and up, the indexer's q and k, and
attn_qo's q and o here and on the 27B): their launches read as weight kernels no role took in the report although
the trace pairs them.

## A per-layer-quant MoE prefill on llama-server: chunks walked apart, layers anchored (2026-11-11)

The Flash-Next Coder IQ1_M capture on llama-server (10,000 tokens, -b 2048 -ub 2048, RTX 5090) failed its output stage
with an unresolved pairing (search_limit + no_fit). Notes: `~/storage/boltbeam-runs/pairing-anchor-20261111.md`.

- **The chunks.** The server checkpoints a model whose memory cannot roll back (a recurrent state): it ends a batch where
  min(n_batch, 4 + n_ubatch) and min(n_batch, 4) tokens remain (llama.cpp server-context.cpp), so the prompt ran as
  2,048 x 3, 1,808, 2,044 and 4, not llama-bench's 2,048 x 4 and 1,808. The record said five chunks, the capture held six
  (72 attention runs), so the layers did not read and the window fell to the flat pairing, which exhausted its budget.
  `llama_server_prefill.server_chunks` records the server's chunks.
- **A narrow chunk.** The 4-token chunk runs its projections as matrix-vector kernels (no matrix instruction, so no
  weight launch by path): they are weight launches where a window has a narrow tail (`prefill_attribution.narrow_weights`).
- **Chunks that launch otherwise.** A chunk of another width runs a layer's kernels otherwise (the shared expert's gate
  and up fused at 2,048 tokens, two launches at 1,808; one expert launch for gate and up at 4 tokens), so the j-th launch
  of a layer is not one role across chunks. When the one walk runs out of nodes or reads no layers, `pair` walks each run
  of alike chunks apart (`_segments`, `_segmented`), in order, each run on the launches the runs before it left; only the
  last holds the head and the pruned layer, and before it the last layer's pruned stages ran on no token.
- **Anchored layers.** With a format per layer every layer is its own kind (`signatures`); a failing layer made the one
  walk backtrack over all the layers before it. When that walk runs out of nodes, `_search` walks one layer kind at a time
  (`staged`), each with its share of the node budget, keeping its best pairing before the next; ties are found over the
  whole pairing as before. Within it a layer's output slots go in launch order after a format-stating template.
- **Smaller rules.** A row may serve its calls at each template's own ratio when no one ratio fits (`mixed`); a row and its
  transpose (a low-rank mixer's down and up: one role, format and count) exchanging launches is no tie; a cuBLAS gemv over
  a chunk serves no many-row weight; `mix_out` is a pruned stage (llama.cpp prunes before the last FFN mixer); the mixer's
  inject (no limit role) is paired in a prefill as one row per mixer (`vocab.PREFILL_ONLY_CLASSES`).

- **The per-layer embedding (0.3.20).** `ple_key` and `ple_value` (Flash-Next's one PLE layer) are a role of their own,
  `per_layer_embedding`, a limit role (a decode reads them once per token; they were counted before as an unclassified
  weight, so the limit's bytes are the same). They read the layer's own token lookup, not the layer before's output, so
  llama.cpp runs them anywhere before the layer (between layer 0's routed and shared experts here): their stage `ple` is
  one the gap's order neither binds nor is set by (`vocab.UNORDERED_STAGES`), and they join no row of another stage.
  The FFN mixer's inject reads after the whole FFN, routed and shared (`mix_post`, pruned with it). A role held at two
  shapes (PLE's key and value, attn_qo's q and o) has each launch's work counted from its own row's shape. A free group
  in a gap two chunk runs share is no twin (the other run may own it).
- **Straight to the anchored walk (0.3.20).** When the chunks launch their layers in more than one way and the profile's
  per-layer tensors set the layers apart (`signatures`), `pair` walks the runs apart at once (`_apart_first`) instead of
  spending the one walk's 50,000 nodes first; every other window walks as before.

Every saved capture the one walk settled pairs as recorded (the replay fixtures unchanged); the Flash-Next capture is a
new fixture (`flash-next-iq1m-server-prefill-10000`): before, search_limit with 385 rows unplaced; after, no search
limit, and (0.3.20, with the per-layer embedding classified) every row placed in all six chunks, no tie. Re-analysed
offline with the profile rebuilt, the capture's output stage passes (474 checks, no failure); the attribution takes
about 6 s, against 25 s for the failing flat search and 71 s for the one walk spent first.
