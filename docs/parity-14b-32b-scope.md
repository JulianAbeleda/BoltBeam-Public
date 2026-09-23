# Using BoltBeam to Close 14B/32B Decode Parity (Scope)

Goal: use BoltBeam — the model-agnostic audit brain (A0–A10) — to drive Qwen3
**14B and 32B** decode to llama.cpp parity, iterating **both models against one
search space and one ledger** so a fix is only "real" when it holds on both.

This is the capstone the A-phases were built for: A1 (role taxonomy), A3 (quant
registry), A4 (capability-driven search), A6 (target caps), A7 (route-family
templates), A9 (policy) are exactly the machinery that turns "hand-tune a kernel
per model" into "search one route family across two shape instances."

Execution begins **after A10**. This document is the plan, not the work.

## 1. Parity definition (the bar)

Same bar 8B already clears: decode tok/s ≥ llama.cpp at matched context and flash
setting, **token-identical** to the owned oracle, **route-bound**, with a
**rollback** knob. Measured only via the AUTHORITY harness (synced fixed-ctx
decode + matched-depth llama) — never the contaminated flash auto-bench.

**Parity is the floor, not the goal.** The search objective is the **achieved HBM
bandwidth fraction** of the XTX 960 GB/s peak, and llama only reaches ~62% of that
peak. So "beat llama" and "keep climbing toward the roofline" are the *same*
instruction — the search does not stop at llama, it stops at the roofline (or at
noise). 8B already demonstrates the headroom is real (109% of llama). See §7.

| model | now (decode) | % of llama | parity floor (~62% peak) | exceed target (illustrative, ~75% peak) |
|-------|-------------|-----------|--------------------------|------------------------------------------|
| 8B    | 107.6 tok/s | **109%** (already beyond) | met | keep as regression guard |
| 14B   | 44.5 tok/s  | ~67%      | ~66 tok/s | ~80 tok/s (~121% llama) |
| 32B   | 21.0 tok/s  | ~68%      | ~30 tok/s | ~37 tok/s (~123% llama) |

Exceed numbers are illustrative (batch-1 roofline peak ≈ 14B 107 / 32B 49 tok/s;
75% of peak shown). 8B is the regression guard: it must stay > llama.

## 2. What the gap actually is (and is NOT)

The FFN-GEMV / topology hunt is **over** — do not re-open it. Recorded results:

- **FFN GEMV is fine.** Role-local 14B: ffn_gate/up ~405 GB/s, attn_q/o 120 GB/s,
  lm_head 706 GB/s. The generated G3 lanemap GEMV generalizes to every 14B/32B
  shape (rel_rmse ~4e-4) — route-binding, not a codegen gap.
- **REFUTED — carry as walls so the search never re-chases:**
  - shape-tuned `words_per_group`/topology (KT): bg=4/wpg=8 is already the most
    parallel *legal* wave split for every 14B/32B shape (gcd(32,k_blocks)=4);
    reachable wpg 16/32 are strictly more serial.
  - split-K for **ffn_down** (SK4A): ~7× slower — that GEMV is memory-bound, not
    latency-bound; 5120 row-workgroups already hide the reduction chain.
  - sub-4-bit demotion (Q3/Q2): quality-walled (fails dNLL on high-byte roles).

The residual gap is **three pinpointed, measured levers**:

| # | lever | evidence | leverage | BoltBeam status today |
|---|-------|----------|----------|-----------------------|
| L1 | **Unfused reduce / scheduling elimination** — `r_*` reduce kernels are ~52% of 14B decode (dominant), tagged `reduce_other` because they're not yet role-resolved (RMSNorm / attention-combine / coop-partial-combine). | reduce_source_trace + role_attribution | **highest** (~52%) | candidate `reduce_elimination` already in the manifest; blocked on r_* role resolution |
| L2 | **Split-K *decode* kernel for occupancy-starved KV projections** — attn_k/attn_v 5120→1024 run at 24/76 GB/s (26% GPU occupancy). Split-K *helps* here (attn_k 24→34.9 GB/s @parts4, byte-correct) — the opposite of ffn_down. The existing partial substrate is prefill-wired; the fix is a generated g3-wave-with-split-K **decode** kernel (SK2). | wd_speed + role-local bandwidth | high (~7ms/tok combined) | `split_k` is EMITTER_BLOCKED in reachability today → this is the codegen directive |
| L3 | **Q6_K `ffn_down` long-K route** — 17408→5120 Q6_K at 253 GB/s is the biggest single wall (11.5ms/tok). | wd_speed | medium | Q6_K route family; candidate-able |

If L1+L2 land, the estimate is 14B ~22→~15 ms/tok ≈ llama parity, with L3 as
headroom to exceed.

## 3. Why BoltBeam, and what "use both to iterate both" means

The two models are **two shape instances of one route family**. BoltBeam gives:

- **One candidate set, one ledger, two scopes.** Ledger entries are scoped by
  `model_id`, so 14B and 32B are distinct scopes that share candidate ids (A7
  route-family templates). Evidence from both feeds the same evaluator.
- **Generalization as the iteration signal.** A lever *promotes generally* only
  when it clears the bar on **both** 14B and 32B evidence (evaluator multi-context
  + row-scope + roofline gate). If it wins on 14B but not 32B, it stays a
  `candidate` on 32B and the divergence tells you where a shape-specific route is
  actually needed — instead of overfitting one model and hoping.
- **Reachability = shared codegen backlog.** `classify_reachability` over both
  profiles emits one machine-readable work queue (EMITTER_BLOCKED split-K,
  PRIMITIVE_MISSING reduce-fusion, …). That queue is the directive to codegen; it
  shrinks as levers land and is honest about what remains.
- **Refuted axes stay walled.** The KT/ffn_down-split-K/sub-4-bit refutations go
  in as refuted ledger entries, so reachability marks them `refuted-by-ledger`
  and the search does not burn cycles re-deriving them.

## 4. Phased loop (P0–P5)

**P0 — Encode the objective as evidence.** For each (model, ctx), emit a
`wd_speed` row vs the llama baseline (`baseline_tok_s`) plus a roofline/`ceiling`
row (achieved GB/s vs 960 peak). Both 14B and 32B. This makes "parity" a scored
quantity, not a vibe. (Uses the roofline + evaluator already built.)

**P1 — Seed the brain with current truth.** Import into the ledger: the three
levers as candidates with correct `required_evidence_kinds` (L1: reduce_source +
role_attribution; L2: wd_speed; L3: wd_speed), and the three refutations (KT
topology, ffn_down split-K, sub-4-bit) as refuted entries with reopen
conditions. Now the brain remembers what's open vs walled.

**P2 — Reachability → directives.** Run reachability for 14B and 32B. Expected:
`reduce_elimination` PRIMITIVE_MISSING until r_* reduces are role-resolved;
`split_k` EMITTER_BLOCKED (no decode-wired split-K kernel); L3 reachable. The
report is the work queue.

**P3 — Measured iteration, per lever, both models.**
- L1: extend the reduce-source trace to map each hot `r_*` reduce to its source
  role by shape+position; ingest → evaluate; fuse/eliminate the resolved reduces;
  re-measure. Highest leverage first.
- L2: generate the g3-wave-with-split-K **decode** kernel; microbench on the
  5120→1024 KV shapes for both models; ingest → evaluate; promote only if
  byte-identical and it clears the bar on both.
- L3: bench the Q6_K long-K `ffn_down` route; ingest → evaluate.
Every candidate flows profile → search → measure → evidence → evaluator, gated by
correctness/route-bound/roofline/rollback.

**P4 — Converge.** Promote winners with rollback (per-model or generalized);
refute dead regions (recorded so they're not re-chased); the residual after L1–L3
is either **parity reached** or a single **named** search-space-incomplete hole —
honest, not mysterious. Re-run reachability to show the shrunk backlog.

**P5 — Guardrails.** (a) token-identical to the owned oracle per model/shape;
(b) **no per-model hand kernel** — every win must be a search-space knob or a
route-family template, else it's a one-off and A9 policy flags it; (c) parity is
claimed only through the AUTHORITY harness; (d) 8B re-checked every promote (must
stay > llama).

## 5. Exceeding llama, not just matching it

Because the objective is roofline fraction (not llama), exceed is the *expected*
outcome of the loop, in two tiers:

**Tier-1 — exceed falls out of L1–L3 (no new idea needed).** The generated
kernels are *already* faster than llama's per-role bandwidth; the scheduling tax
is what hides it. 14B FFN role-local is ~405 GB/s while the full-model decode is
only 243 GB/s — the difference is the unfused-reduce overhead (L1, ~52%). Remove
that tax and decode approaches the kernels' native bandwidth, which is above
llama's ~62%-of-peak on those roles. 8B is the existence proof: same route family,
overhead already low, 109% of llama. So for 14B/32B, **landing L1 (and L2 for the
occupancy-starved KV roles) is expected to cross parity and keep going**, not
stop at it. The roofline `ceiling` evidence row is what tells the evaluator a role
is near the wall and further search is wasted.

**Tier-2 — a structural lever llama does not have: AMD dual compute-ring
overlap.** A second AMD compute ring unlocks ~2.0× same-process overlap (verified
available), but no scheduler is built yet. Overlapping independent decode kernels
(or compute with the next weight's HBM read) across two rings raises *effective*
throughput past what any single-stream bandwidth ceiling allows — territory a
single-stream llama decode cannot reach. In BoltBeam this is a new route-family
candidate whose base primitive (a ring scheduler) is `PRIMITIVE_MISSING` /
`EMITTER_BLOCKED` today, so reachability emits it as an explicit codegen directive
rather than a hand hack. Highest ceiling, highest build cost — sequenced **after**
L1–L3 land and only if roofline headroom remains.

**Exceed guardrails.** Same gates as parity (token-identical, route-bound,
rollback, AUTHORITY harness). Two additions: (a) **stop at the roofline** — once a
role's achieved bandwidth is within noise of the 960 GB/s-derived ceiling, mark it
done and move on; do not chase past the wall. (b) The dual-ring lever must show
overlap in the profile (two rings actually concurrent), not just a faster number,
or it is `inconclusive`. Known non-levers stay walled: decode two-kernel combine
overlap is EXHAUSTED (+5.7%, saturates); sub-4-bit is quality-refuted (cannot buy
bytes).

## 6. Definition of done

- 14B and 32B decode ≥ llama parity at matched ctx/flash, token-identical,
  route-bound, with rollback — OR the residual is a single named emitter-capability
  gap on the reachability backlog (not an unexplained deficit).
- Each landed lever is a ledger'd candidate with rollback; each dead end is a
  refuted axis; the reachability report is the running, shrinking codegen queue.
- 8B unregressed (still > llama).
- No model-name branches and no per-model hand kernels — everything is a profile
  fact, a registry capability, or a route-family template (enforced by A9).

## 7. Dependencies on the A-phases

- **A4** capability-driven emission: the search space must express L1–L3 per
  (role, quant, shape) so both models draw candidates from the same grammar.
- **A6** target capability data: split-K reasoning (L2) needs wave/subgroup + LDS
  from the target descriptor, not hardcoded 32/64KB.
- **A7** route-family templates: 14B and 32B must share candidate ids to make the
  cross-model generalization test meaningful.
- **A9** policy guard: enforces "no per-model hand kernel" so a parity hack can't
  bypass the search.

Related: `qwen-14b-32b-truegen-state`, `model-bench-authority-findings`,
`pure-machine-search-goal`.
