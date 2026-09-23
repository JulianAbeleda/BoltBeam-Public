# System Fusion SF4 — Ledger Report (BoltBeam)

> **Artifact location, 2026-07-31.** The `bench/` measurement files cited below were
> pruned from the trunk as raw measurement (see
> `docs/task_workflow/input/bench-decouple-scope-20260731.md`). They remain on the
> `dev` and `exp` branches per `docs/branch-flow.md`; recover with
> `git show dev:<path>`. Verdict-bearing artifacts stayed on the trunk.


Date: 2026-07-01. Covers SF0-SF4 for the 14B/32B decode system-fusion track.

## Audit: ~12% elementwise bucket (SF0)

The ~12% "launch/activation/other" bucket from the decode loss stack is fully resolved into 14 named fragments at ctx512. Unknown% = 0.0% → SF0_PASS.

Key: E_49152_32_3 (6.69%) is EMITTER_BLOCKED; the REACHABLE_NOW total is 4.90%.

See: `bench/system-fusion-sf0/latest.json`.

## Classification (SF1)

| class | % | fragments |
|-------|---|-----------|
| EMITTER_BLOCKED | 6.69 | E_49152_32_3 (KV write/RoPE apply) |
| REACHABLE_NOW | 4.90 | silu_gate 1.26%, qk_norm_scale 1.46%, rmsnorm_scale 1.28%, residual_adds 0.90% |
| LOW_AMDAHL | 0.83 | E_1920_32_3, E_1187_32_4, E_40_32_4n4 |
| NOT_FUSEABLE | 0.34 | TracingKey, E_2n7, E_20_4_2_8_16_2_4_4 |

See: `bench/system-fusion-sf1/latest.json`.

## Selected candidate (SF2): decode_silu_gate_fusion

Root cause: model.py:1017 has `.silu().contiguous()` (marked TODO) splitting silu+gate_up into 2 kernels. Fix: remove `.contiguous()` behind `DECODE_FUSE_SILU_GATE` (default-off).

## Experiment result (SF3)

**Correctness: PASS** (rel_rmse=0.00e+00, identical logits).

W==D (Qwen3-14B, gfx1100):
- ctx128: 52.3 → 52.4 tok/s (+0.1)
- ctx512: 50.0 → 50.4 tok/s (+0.4)

**Verdict: SF3_LOW_AMDAHL_NO_MOVEMENT** — 0.4 tok/s < 0.5 tok/s threshold.

The Q4K GEMV (43%, HBM-bound at ~400 GB/s) dominates. Removing 40 small elementwise launches does not move the bottleneck.

See: `bench/system-fusion-sf3/latest.json`.

## BoltBeam candidate status

`decode_silu_gate_fusion` is recorded in `boltbeam/data/candidates.json`:
- status: refuted_no_movement
- do_not_retry: False
- Code change: kept in tinygrad repo (correct, default-off)
- Rollback: DECODE_FUSE_SILU_GATE=0

## Reopen condition

Individual elementwise fusions (1.26-1.46% each) are below the W==D noise floor. Reopen as an **aggregate multi-group fusion pass** covering all 4 REACHABLE_NOW groups simultaneously (~4.9% → ~2.4 tok/s at ctx512).

The EMITTER_BLOCKED E_49152_32_3 (6.69%) reopens when tinygrad UOps gain a mechanism for elementwise→flash_reduce fusion at a global barrier boundary.

## Frontier state

System-fusion track SF0 is complete. The 12.76% bucket is mapped. The individual lever is deferred; the aggregate pass is the correct next target.
