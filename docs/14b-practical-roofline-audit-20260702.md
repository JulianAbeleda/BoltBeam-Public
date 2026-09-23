# 14B Practical Roofline Audit

Date: 2026-07-02

Scope: Qwen3-14B-Q4_K_M decode on gfx1100 after the promoted G5 K-only attention tile.

## Current Actual

The latest promoted/default-on 14B attention result is the G5 K-only route:

| ctx | pre-G5 baseline | current G5 K-only | delta |
|---:|---:|---:|---:|
| 128 | 52.1 tok/s | 52.2 tok/s | +0.1 |
| 512 | 49.9 tok/s | 53.8 tok/s | +3.9 / +7.8% |
| 2048 | 46.9 tok/s | 53.8 tok/s | +6.9 / +14.7% |

Source: `/home/ubuntu/tinygrad-arkey/bench/gp-track/gp4_latest.json`.

The route is promoted/default-on in tinygrad:

- `DECODE_FLASH_BLOCK_TILE_G5=1`; K-only staging is the fixed tinygrad implementation detail for this route.

The default-path census also lists `decode_flash_block_tile_g5_konly` as a default route.

## Practical Roofline Score

Use the practical-roofline formula:

```text
P_c = 100 * candidate_tok_s_c / practical_ceiling_tok_s_c
G_c = 100 - P_c
A_c = max(0, G_c - max(2.0, measured_spread_pct_c))
```

For the current 14B ctx512/ctx2048 actual (`53.8 tok/s`):

| ceiling basis | ceiling | P | G | A | tok/s gap | speedup needed |
|---|---:|---:|---:|---:|---:|---:|
| llama parity floor | ~66 tok/s | 81.5% | 18.5pp | 16.5pp | +12.2 | 1.23x |
| illustrative 75%-peak target | ~80 tok/s | 67.3% | 32.8pp | 30.8pp | +26.2 | 1.49x |
| illustrative raw roofline peak | ~107 tok/s | 50.3% | 49.7pp | 47.7pp | +53.2 | 1.99x |

Verdict: **HEADROOM_REMAINS**.

This is not an 8B-style practical closeout. The route promotion was correct, but the model is still materially below
parity and far from the illustrative raw roofline.

## What Is Already Exhausted

Do not send Fable back through these unless new evidence changes the basis:

- **Q4_K GEMV route variants:** exhausted at the route level. Fresh A/B has vector_load at ~398 GB/s versus G3 at
  ~405 GB/s. Topology and split-K variants were already refuted. This is not the parity gap.
- **Attention combine transforms currently expressible in UOp:** refuted. Hq-only fusion, merged combine, FLASH_L,
  and 14B wholecache/score-broadcast all regressed W==D.
- **Single small elementwise fusions:** individually too low-Amdahl. `decode_silu_gate_fusion` moved ctx512 only
  +0.4 tok/s.
- **Q6_K ffn_down long-K:** already shipped as L3 and included in current defaults.

## Remaining Gap Stack

The most useful current loss-stack view is still the post-L3/post-attention-closure audit, with the caveat that the
absolute tok/s table predates the G5 K-only promotion. The bucket classes remain useful:

| bucket / lever | approximate share | current status | Fable action |
|---|---:|---|---|
| Q4_K GEMV | ~43% | at route ceiling | do not chase route variants |
| attention combine | ~13.6% combine, ~27% attention total | reachable transforms refuted | reopen only with a new cross-workgroup LSE/coordination primitive |
| launch/activation/other | ~12% pre-G5; 4.9% reachable-now aggregate after classification | low-Amdahl individually | pursue aggregate multi-group fusion, not one-offs |
| E_49152-style V path / attention elementwise | ~6.7% pre-G5 | partly addressed by G5 K-only; verify residual after current default | re-profile before targeting |
| RMSNorm / reduce fragments | ~1-5% depending grouping | low-Amdahl unless aggregated | fold into aggregate scheduler pass |

## Amdahl Reality Check

From the current `53.8 tok/s`:

| removed share | ideal tok/s | delta |
|---:|---:|---:|
| 4.9% aggregate reachable elementwise | 56.6 | +2.8 |
| 6.69% E_49152-style bucket | 57.7 | +3.9 |
| 12% system-fusion bucket | 61.1 | +7.3 |
| 13.6% attention-combine bucket | 62.3 | +8.5 |
| 27% whole attention bucket | 73.7 | +19.9 |

Parity needs ~`66 tok/s`, so no single low-Amdahl cleanup is enough. Fable should either:

1. build an **aggregate system-fusion/scheduler pass** and remeasure current 14B defaults; or
2. open a **new attention coordination primitive** that changes the combine ceiling, not another already-refuted
   combine topology.

## Recommended Fable 5 Prompt

```text
Use the 14B practical-roofline audit as the authority. Current promoted/default-on 14B is G5 K-only at 53.8 tok/s
ctx512/ctx2048. Llama parity is ~66 tok/s, so P=81.5%, G=18.5pp, A=16.5pp: HEADROOM_REMAINS.

Do not chase Q4_K GEMV route variants, attention combine topologies already refuted, or single low-Amdahl elementwise
fusions. First re-profile current defaults after G5 K-only to refresh the loss stack. Then choose only one of:

1. aggregate multi-group system fusion covering the reachable 4.9% elementwise fragments plus any remaining current
   E_49152/V-path residual; or
2. a new cross-workgroup LSE/coordination primitive for attention combine.

Any candidate must produce W==D evidence vs current 14B defaults, route-bound evidence, correctness, rollback, and a
practical-roofline line: P_worst, G_worst, A_worst, ceiling basis, action. Stop if the refreshed profile shows the
target bucket is gone or below W==D noise.
```

## Next Measurement

Before implementing a new primitive, refresh the 14B loss stack under the actual current defaults:

- G5 K-only default on;
- Q6_K long-K L3 default on;
- Q4_K G3 default on;
- same ctx set as the parity gate, preferably ctx512 and ctx2048, with ctx4096 if the host is healthy.

The audit question for Fable is not "is there roofline headroom?" There is. The question is whether the refreshed
headroom is concentrated enough for a safe aggregate fusion or requires a new attention coordination primitive.
