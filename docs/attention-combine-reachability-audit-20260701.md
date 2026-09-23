# Attention-Combine Reachability Audit (falsification pass 2)

Date: 2026-07-01. Follows `attention-combine-closure-stress-test-20260701.md`. Purpose: enumerate and
classify every path to reduce `flash_gmax`/`flash_den`/`flash_combine` (the `attention_combine` bucket)
WITHOUT re-running the refuted Hq-only in-workgroup fused combine, and run the smallest generated experiment
for anything REACHABLE_NOW.

Ground truth (14B gfx1100, gqa_coop_vec): the combine is 3 separate UOp reduce kernels over the S splits —
`flash_gmax` (Hq wg), `flash_den` (Hq wg), `flash_combine` (Hq*Hd wg) — reading `pm[Hq*S]` and `po[Hq*Smax*W]`.
`po` is produced by `flash_partial` over **Hq*S workgroups** (the parallel phase). The refuted fused route
collapsed the *partial* phase to Hq workgroups; that is the -88% failure and is not retried.

## The five questions

**Q1 — Any route that preserves Hq*S partials while reducing one or more of gmax/den/combine?**
Yes. The partial phase (`flash_partial`, Hq*S) is independent of the combine phase (already Hq/Hq*Hd wg). The
3 combine kernels can be MERGED into 1 generated kernel per head (compute gm, then den, then out[h,d] in one
launch), eliminating 2 kernel launches + the `gm`/`dn` global buffers, WITHOUT touching `flash_partial`. This
is split-preserving. → **REACHABLE_NOW** (experiment below).

**Q2 — Is cross-workgroup LSE coordination representable in current tinygrad UOps?**
No. In-WORKGROUP LSE is representable (the fused kernel proved it: REG/LDS placeholders + `s_barrier` + warp
reduce). Cross-WORKGROUP coordination (merging S partials that live in different workgroups, in one kernel)
needs a global barrier or atomic accumulation. tinygrad UOps expose no cross-workgroup barrier and no atomic
op (`grep -i atomic tinygrad/uop/ops.py` = none; barriers are `s_barrier`, workgroup-scoped only). The only
cross-workgroup coordination available is *a second kernel* — which is exactly the current combine. →
**PRIMITIVE_MISSING** for a single-kernel cross-workgroup LSE.

**Q3 — Atomics / cooperative groups / global barriers / multi-kernel fusion on the current AMD path?**
- Multi-kernel fusion (merge the 3 combine kernels): available → REACHABLE_NOW (Q1).
- Atomic float add: not a tinygrad UOp op → PRIMITIVE_MISSING.
- Global barrier / cooperative groups (grid sync): not a tinygrad UOp op → PRIMITIVE_MISSING.
So the only *cross-workgroup* mechanism realistic now is the multi-kernel form (which is the status quo).

**Q4 — Model geometry where Hq-only workgroups saturate the GPU?**
gfx1100 = 96 CUs. Hq-only launches = number of heads. 14B Hq=40, 32B Hq=64 — both < 96 and far below the
wave-count needed to hide latency, so Hq-only is occupancy-starved for both (14B measured -88%; 32B same
class, not separately measured — do not assume a number). A model with Hq >> ~2*CUs (many attention heads)
could saturate with Hq-only workgroups, but no such model is in the current profile set, and it must be shown
by occupancy/W==D evidence, not assumed. → **SEARCH_SPACE_INCOMPLETE** (no in-scope high-head geometry).

**Q5 — Transfer to 32B / MoE / hybrid / NVIDIA / Metal?**
- **32B (dense):** same gqa_coop_vec structure, Hq=64 still < saturation → the Hq-only fused route transfers
  as refuted-class; the split-preserving merge transfers as REACHABLE_NOW. Needs its own measurement to claim
  a number.
- **MoE / hybrid:** attention is ordinary dense attention, but decode wall is dominated by experts/SSM, so the
  `attention_combine` fraction is smaller → **LOW_AMDAHL** (not the wall; don't chase there yet).
- **NVIDIA (nvidia_sm89) / Metal (apple_metal):** descriptor-only targets, no evaluator → **TARGET_BACKEND_INCOMPLETE**.
  Note: CUDA/Metal DO have native cooperative-groups/atomics, so the PRIMITIVE_MISSING cross-workgroup LSE of
  Q2/Q3 could become reachable there once those backends are `complete` — recorded as a target-conditional reopen.

## Classification of every path

| path | class | why |
| --- | --- | --- |
| Merge flash_gmax+den+combine into 1 kernel (split-preserving) | **REACHABLE_NOW → REFUTED_BY_LEDGER** | tried (experiment below): correct + shrinks bucket 13.6->11.3% but ctx512 -16% — it collapses `flash_combine`'s OWN Hq*Hd parallelism to Hq wg |
| Single-kernel cross-workgroup LSE via atomic float accumulate | **PRIMITIVE_MISSING** | no atomic UOp on AMD |
| Single-kernel cross-workgroup LSE via global barrier / cooperative groups | **PRIMITIVE_MISSING** | no grid-sync UOp |
| Hq-only in-workgroup fused combine (collapse partial to Hq wg) | **REFUTED_BY_LEDGER** | measured -88% @ctx512; do_not_retry (`attention_combine_fusion_occupancy`) |
| FLASH_L chunk-count knob | **REFUTED_BY_LEDGER** | shrinks bucket, regresses tok/s (50.2->43.5) |
| wholecache / score-broadcast route on 14B | **REFUTED_BY_LEDGER** | token-correct, ctx512 -9.8% |
| DECODE_OUTER_B_SPLIT transform on default route | **EMITTER_BLOCKED** | declines (no 3-carry fused kernel to split); needs a fused-partial kernel to bind to |
| High-head model geometry where Hq-only saturates | **SEARCH_SPACE_INCOMPLETE** | no in-scope model; needs occupancy/W==D evidence |
| Cross-workgroup LSE on CUDA/Metal (native coop groups/atomics) | **TARGET_BACKEND_INCOMPLETE** | descriptor-only targets, no evaluator |
| Attention-combine on MoE/hybrid decode | **LOW_AMDAHL** | attention is a small fraction of expert/SSM-dominated decode |

## Reopen conditions (precise)

- The REFUTED_BY_LEDGER Hq-only path reopens ONLY via `attention_combine_fusion_occupancy` being falsified by a
  **split-preserving** or **new-primitive** mechanism (this doc's REACHABLE_NOW or PRIMITIVE_MISSING rows).
- The PRIMITIVE_MISSING rows reopen when tinygrad UOps gain an **atomic float** or **grid-sync/cooperative-group**
  primitive on the AMD path.
- The TARGET_BACKEND_INCOMPLETE row reopens when `nvidia_sm89`/`apple_metal` become `complete` targets (they
  natively have the coordination primitive AMD lacks).
- The SEARCH_SPACE_INCOMPLETE row reopens when a profiled model has Hq high enough that Hq-only workgroups
  saturate the GPU (shown by evidence, not assumed).

## Experiment run (REACHABLE_NOW): merge the 3 combine kernels — RESULT: REFUTED

Implemented `flash_combine_merged_kernel` (tinygrad `extra/qk_flash_decode.py`, generated UOp, no handwritten
kernel) and wired it behind `DECODE_ATTN_COMBINE_MERGED` (default-off) in the gqa_coop_vec path. `flash_partial`
(Hq*S) is untouched; the 3 combine kernels become 1 launch.

Measured (14B gfx1100):
- **Correctness:** 14B token-identical; merged kernel fires (`flash_combine_merged_40_128`); the 3 old kernels
  are gone.
- **Mechanism:** `attention_combine` bucket 13.57% -> 11.32% (total reduce 19.0% -> 15.8%). `reduce_eliminated`
  guardrail = PASS (shrank 2.25pp).
- **W==D:** ctx128 52.3 -> 52.0 (flat); **ctx512 50.2 -> 42.1 (-16%)**. BoltBeam verdict = **refute**
  (protected-context regression).

**Why it lost — a second occupancy level.** `flash_gmax`/`flash_den` are already tiny (Hq=40 wg). `flash_combine`
launches **Hq*Hd = 5120 workgroups** — high occupancy. Merging into one per-head kernel (d-sharded to 32 lanes)
collapses that to **Hq=40 workgroups**, so the -16% is the same occupancy failure the Hq-only fused route hit,
now at the combine's Hd axis. The 3-kernel decomposition is load-bearing at BOTH levels: partial parallelism
(Hq*S) AND combine parallelism (Hq*Hd). Merging cannot help without a coordination primitive that keeps both.

This does not weaken the closure — it strengthens the exact reopen condition: a winning attention-combine route
must preserve BOTH the Hq*S partial parallelism AND the Hq*Hd combine parallelism, which with current AMD UOps
requires the PRIMITIVE_MISSING cross-workgroup coordination (atomics / grid sync), not a kernel merge.

BoltBeam candidate: `decode_attention_combine_merged` (origin refuted, `attention_combine_merge_occupancy`).
