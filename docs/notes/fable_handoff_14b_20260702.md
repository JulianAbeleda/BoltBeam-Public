# Fable 5 Handoff — 14B Aggregate System Fusion Candidate

Date: 2026-07-02. Audit complete. **Do not implement until approved (IP5).** This packet covers IP0–IP4.

## Provenance
- tinygrad `4b406bdbc` (dirty 4: unrelated); BoltBeam `b0832e5` (dirty 9: roofline WIP).
- Model `qwen3-14b` / `Qwen3-14B-Q4_K_M.gguf`; target `amd_gfx1100`; contexts 512/2048 measured, 4096 skipped.
- Authorities: attribution = PROFILE=1 share; wall = gp4_latest.json (53.8 tok/s); route-bound = `flash_block_tiled_40_128` present, no owned/gqa fallback.

## IP0 — Frontier
**AGGREGATE_SYSTEM_FUSION.** (Rationale in `recommendation_14b_20260702.md`.) It is the only low-risk reachable-now lever; running it once as an aggregate either wins or earns CLOSEOUT.

## IP1 — Smallest candidate
- **candidate id:** `decode_aggregate_elementwise_fusion`
- **route family:** system/scheduler elementwise fusion (not a weight-GEMV route)
- **scope:** model=qwen3-14b (and 32B, same arch); shape=decode T=1; quant=agnostic (elementwise); roles = activation/norm/residual elementwise; contexts ctx512, ctx2048
- **env flag (default-off):** `DECODE_AGG_SYSTEM_FUSION=1`
- **rollback:** `DECODE_AGG_SYSTEM_FUSION=0`
- **expected affected kernels:** the 4 REACHABLE_NOW groups — `silu_gate` (model.py:1017 `.silu().contiguous()` barrier, the SF2 TODO), `qk_norm_scale` (1.46%), `rmsnorm_scale` (1.28%), `residual_adds` (0.90%). Target: fuse these into their neighbors / drop redundant `.contiguous()` barriers so ~40 small elementwise launches collapse into far fewer.
- **expected removed/reduced bucket:** the 4.90% reachable elementwise fragmentation
- **expected max W==D gain (Amdahl):** ideal 53.8→56.6 (+2.8) at ctx512; **realistic likely below noise** (SF3 prior). Stop threshold: aggregate must beat 0.5 tok/s AND exceed measured spread.
- **known refuted approaches to avoid:** single-fragment fusions (silu_gate alone = SF3 refuted_no_movement +0.4); do NOT touch `E_49152_32_3` (EMITTER_BLOCKED KV-write/RoPE); do NOT chase Q4_K/Q6_K GEMV; do NOT collapse attention parallelism.

## IP2 — Gates before implementation
- **isolated numeric/microgate:** per fused group, `rel_rmse = 0.0` / identical logits vs unfused (SF3 standard).
- **route-bound proof:** emitted kernel count drops vs default; the fused elementwise kernels present; protected routes (`flash_block_tiled_40_128`, Q4_K G3, Q6_K coop) still fire; no owned/gqa fallback.
- **correctness authority:** deterministic — prefilled NLL (`qk_nll_eval.py`) or prefilled parity (`qk_prefilled_route_parity.py`), and per-group rel_rmse. **NOT** the unprefilled `qk_decode_token_match_check.py`.
- **W==D contexts:** ctx512 + ctx2048; ctx4096 only if host clean afterward and no transient D-state persists.
- **practical roofline line:** emit P_worst, G_worst, A_worst, ceiling basis (llama 66), action.
- **health checks:** gpu-D between every run; smoke before/after; **stop on persistent D-state / reset / smoke fail**.
- **default-off regression ladder:** `pure_machine_search_default_path_census.py --check` unchanged with flag off; fix-off == default byte/NLL identical; protected routes unaffected.
- **BoltBeam ledger / reopen-check:** add `decode_aggregate_elementwise_fusion` to candidates.json; after W==D, `boltbeam reopen-check` (or evaluate) with the wd_speed evidence. If < noise → ledger `refuted_no_movement` (like `decode_silu_gate_fusion`) and flip audit verdict to **CLOSEOUT_ROUTE_SEARCH_PRIMITIVE_MISSING**.

## IP3 — Risk classification
**LOW** — scheduler/graph fusion only (drop `.contiguous()` barriers / fuse adjacent elementwise), **no new lowering semantics, no new UOp primitive, no synchronization/atomics**. Same class as the shipped `decode_silu_gate_fusion` (SF2/SF3). LOW risk does not itself require approval, but per IP5 the workflow still stops here for a go/no-go.

## IP4 — Implementation packet
- **files expected to change:** `tinygrad/llm/model.py` (the 4 elementwise fusion points, incl. the :1017 `.silu().contiguous()`); possibly a small scheduler-fusion helper in `extra/`. No codegen/renderer changes.
- **exact code paths:** decode forward elementwise sites for silu_gate, qk_norm_scale, rmsnorm_scale, residual_adds; the `DECODE_AGG_SYSTEM_FUSION` gate wraps all four.
- **tests to add/update:** per-group numeric microgate (rel_rmse=0); a W==D vs current-defaults gate at ctx512/2048; default-off census check.
- **artifacts to regenerate:** refreshed roofline line with the new candidate W==D; `bench/system-fusion-sf5/*`; candidates.json entry.
- **stop conditions:** first correctness / route-bound / host-health failure; OR aggregate W==D < max(0.5 tok/s, spread) → stop, ledger refuted, verdict CLOSEOUT.
- **commit allowlist (only if asked):** `tinygrad/llm/model.py` [nn]; new test/bench under `extra/` + `bench/` [test]; BoltBeam candidates.json [schema]. No default flip, no unrelated bench artifacts.

## STOP
Audit + planning complete. Awaiting go/no-go on implementing this single default-off candidate (IP5). If declined, the standing verdict is HEADROOM_REMAINS with the aggregate untested; if the aggregate is later run and moves < noise, the verdict becomes CLOSEOUT_ROUTE_SEARCH_PRIMITIVE_MISSING (missing capabilities: cross-workgroup LSE atomics/grid-sync, v_dot2 renderer lowering, vec-store-to-REG accumulator widening).
