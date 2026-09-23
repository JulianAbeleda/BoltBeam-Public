# 14B Frontier Recommendation — 2026-07-02

## Final audit verdict

**HEADROOM_REMAINS_AGGREGATE_SYSTEM_FUSION** — with an explicit, honest caveat.

Roofline headroom is real (P_worst=81.5%, G_worst=18.5pp, A_worst=16.5pp vs the ~66 tok/s llama parity floor; current 53.8 tok/s). But after refreshing the loss stack under the promoted G5 K-only defaults:

- every **large** bucket is **route-ceiling** (Q4_K 40.6%, Q6_K 11%) or **primitive-missing** (attention combine 13.6%, attention PV 16.9%);
- the **only reachable-now route lever** is the **4.9% aggregate elementwise** bucket;
- prior evidence (SF3: `silu_gate` alone +0.4 tok/s, below the 0.5 noise threshold) suggests the elementwise bucket is largely **overlapped/off-critical-path**, so even the aggregate may move W==D below noise.

So the honest framing: **Frontier A is the one safe experiment that will either recover real headroom or earn a route-level CLOSEOUT verdict.** It is selected not because it is high-yield, but because it is the *only low-risk reachable-now* lever and it has **never been tested as an aggregate**.

## Frontier selection (Phase 5 / IP0)

Selected: **AGGREGATE_SYSTEM_FUSION** (Frontier A).

| frontier | bucket | reachable? | risk | expected W==D | decision |
|---|---|---|---|---|---|
| **A. aggregate system fusion** | 4.9% elementwise | yes (now) | **LOW** (scheduler/graph fusion) | +2.8 ideal / likely <noise | **SELECTED** — only safe reachable lever; aggregate untested |
| B. new attention coordination primitive | 13.6% combine | no — PRIMITIVE_MISSING (atomic-float/grid-sync) | **HIGH** | +8.5 ideal | deferred — approval-gated; genuinely-new capability, not a route |
| C. closeout | — | — | — | — | premature until A is tested; becomes the verdict if A moves < noise |

Why not B now: the expressible combine transforms are refuted; the only non-refuted expressible option (TG-P14.8 fused combine) was a ~2% near-tie for 8B; a *decisive* combine win needs a genuinely-new cross-workgroup LSE/atomics primitive that AMD tinygrad does not have (PRIMITIVE_MISSING) — HIGH risk, stop-for-approval per IP3.

Why not C yet: the 4.9% aggregate has only ever been tested one-fragment-at-a-time (SF3). Running it once as a true aggregate is the cheap, safe way to either win or *earn* the closeout verdict with direct evidence rather than inference.

## Exhausted-route firewall (Phase 4) — confirmed by fresh evidence

- **Q4_K GEMV** (40.6%): ROUTE_CEILING. Fresh A/B vector_load 398 vs G3 405 GB/s. Do not chase vector_load / topology / split-K. Not disproven by refresh.
- **Attention combine topologies** (Hq-only, merged, FLASH_L, wholecache/score-broadcast): REFUTED. Fresh stack still shows the combine is the 2-kernel `flash_state_gmax+combine` at ~13.6%; no refreshed evidence contradicts the refutations. Reopen only with a **new** cross-workgroup coordination primitive that preserves Hq*S / Hq*Hd parallelism.
- **Single low-Amdahl elementwise** (silu_gate/rmsnorm/qk_norm_scale/residual_add alone): below W==D noise; only actionable **aggregated**.
- **Q6_K ffn_down long-K**: promoted (L3), firing (route-bound in fresh capture); no reopen.

## Bottom line for the next pass

Run exactly one candidate — the **aggregate multi-group elementwise/scheduler fusion** — as a default-off experiment with full gates. If its W==D at ctx512/ctx2048 exceeds noise, real headroom is recovered and it is a promotion candidate. If it does not (the SF3-implied expectation), the route-level search for 14B decode is **closed out**, and the verdict flips to **CLOSEOUT_ROUTE_SEARCH_PRIMITIVE_MISSING** with named missing capabilities: cross-workgroup LSE atomics/grid-sync, v_dot2 renderer lowering, and vec-store-to-REG accumulator widening.
