# Practical Roofline Promotion Scope

Goal: teach BoltBeam to answer one question before reopening more kernel work:

```text
Is this route meaningfully below the practical ceiling, or is it already close enough that further route work is not
worth pursuing?
```

This is not a raw-HBM roofline gate. Raw HBM peak is useful context, but promotion should be based on the best
measured same-target implementation that represents the practical route ceiling for the exact workload.

## Current Trigger

The immediate case is `decode_attention_g5_8b_refuted` after TG-P14:

- generated route is correct and route-bound;
- KV_BOTH staging recovers the ctx4096 miss;
- W==D reaches roughly 98.5-98.8% of owned HIP at protected contexts;
- the residual gap is small enough that it may be practical-ceiling noise, not actionable headroom;
- the next primitive, vectorized PV, is blocked in UOp lowering and may not be worth pursuing unless the ceiling
  audit proves real headroom remains.

BoltBeam needs a durable way to classify that situation.

## Principle

Promotion and search closure are separate decisions:

- `promote`: this route is safe to use as the selected/default replacement under its rollback contract.
- `practical-ceiling-closeout`: this route is close enough to the practical ceiling that BoltBeam should stop
  asking for more variants in this route family.

A practical-roofline pass may justify promotion when the candidate's purpose is purity or ownership replacement and
the route is close to the practical ceiling. It must not promote an arbitrary slower route just because the whole
system is hard to improve.

## Required Evidence

Add a normalized `ceiling` evidence row or a dedicated practical-roofline artifact consumed as `RowKind.CEILING`.
For every protected context, it must carry:

- `candidate_tok_s` and `practical_ceiling_tok_s`;
- `pct_of_practical_ceiling`;
- `ceiling_basis`: `owned_same_route_family`, `llama_same_model_target`, or `measured_best_same_scope`;
- `baseline_candidate_delta_pct`, when a normal W==D comparator exists;
- `spread_pct` or a separate run-to-run band;
- `route_bound`, `correctness`, `hidden_fallback`, and rollback evidence from the ordinary gate;
- route economics where available: `tile_us`, `combine_us`, `total_attention_us`, split count `S`,
  workgroups/CU, LDS bytes, spill/scratch bytes, KV bytes, and effective bandwidth.

Fail closed if any protected context is missing.

## Classification

Classify a candidate into one of these practical-roofline classes:

| class | meaning |
|---|---|
| `PRACTICAL_ROOFLINE_PROMOTABLE` | correctness, route, rollback, and practical-ceiling thresholds pass; promotion is allowed if the candidate reduces handwritten surface or satisfies an explicit replacement objective |
| `PRACTICAL_ROOFLINE_CLOSEOUT` | close enough to stop route-family search, but not a default replacement |
| `HEADROOM_REMAINS` | measured gap is larger than the ceiling/noise band or route economics show an actionable bottleneck |
| `PRACTICAL_ROOFLINE_INCONCLUSIVE` | missing economics, stale artifact, noisy W==D, or mismatched model/target/context |

Initial thresholds:

```text
pct_of_practical_ceiling >= 98.0% in every protected context
gap_to_practical_ceiling <= max(2.0 percentage points, measured_spread_pct)
no protected context below the ordinary minimum-speed floor
```

For a refuted candidate being audited through `reopen-check`, a practical-roofline promote should still be reported as
`candidate` until the normal authorized `evaluate` path is run. This preserves the existing reopen-check contract.

## Promotion Formula

Every promotion recommendation that uses roofline evidence should compute the same score.

For each protected context `c`:

```text
P_c = 100 * candidate_tok_s_c / practical_ceiling_tok_s_c
G_c = 100 - P_c
N_c = measured_spread_pct_c if present else 0
A_c = max(0, G_c - max(2.0, N_c))
```

Where:

- `P_c` is percent of the practical ceiling;
- `G_c` is remaining percentage-point gap;
- `N_c` is the measured noise band for that context;
- `A_c` is actionable headroom after the closeout tolerance.

Reduce across protected contexts conservatively:

```text
P_worst = min(P_c)
G_worst = max(G_c)
A_worst = max(A_c)
```

Then classify:

```text
if hard_guardrails_fail:
  REFUTE
elif any required practical-roofline input is missing:
  PRACTICAL_ROOFLINE_INCONCLUSIVE
elif P_worst >= practical_roofline_min_pct and A_worst == 0:
  PRACTICAL_ROOFLINE_PROMOTABLE if replacement_objective else PRACTICAL_ROOFLINE_CLOSEOUT
else:
  HEADROOM_REMAINS
```

The default `practical_roofline_min_pct` is `98.0`. A candidate may set a stricter value in its manifest.

The formula answers two separate questions:

- **Can this be promoted as a practical replacement?** Yes only when hard guardrails pass, `P_worst` clears the bar,
  `A_worst == 0`, and the candidate explicitly declares a replacement objective.
- **Should we keep searching this route family?** No when hard guardrails pass and `A_worst == 0`; the route is inside
  the practical closeout band even if it is not authorized for default promotion.

Promotion recommendations should include this compact line:

```text
practical_roofline: P_worst=98.5%, G_worst=1.5pp, A_worst=0.0pp, basis=owned_same_session_same_scope,
action=promote|closeout|headroom-remains|inconclusive
```

## Promotion Policy

The evaluator should apply this order:

1. Hard guardrails first: correctness, route-bound, hidden fallback, determinism, memory, rollback.
2. Ordinary speed win/equivalence next.
3. Practical-roofline closeout only after ordinary speed says "near tie" or "small regression".
4. A large protected regression still refutes, even if a ceiling artifact exists.

For generated replacements of handwritten kernels:

- if `pct_of_practical_ceiling >= 98%`, the route is correct, route-bound, rollbackable, and reduces handwritten
  surface, emit `promote` with tier `none` and reason `practical-roofline-equivalent replacement`;
- if it does not reduce handwritten surface, emit `diagnostic` or `candidate` with `next_action` saying the route
  family is saturated and further variants are not expected to pay.

For performance-only candidates:

- practical roofline is a stop rule, not a promotion rule;
- emit closeout/diagnostic unless there is an actual speed win.

## Candidate Manifest Requirements

The evaluator must not infer "replacement objective" from a candidate id or description text. A practical-roofline
promotion requires an explicit manifest or evidence signal such as:

```json
{
  "thresholds": {
    "practical_roofline_replacement": true,
    "reduces_surface": true,
    "practical_roofline_min_pct": 98.0
  }
}
```

For TG-P14, `decode_attention_g5_8b_refuted` would need this explicit replacement marker before an authorized
`evaluate` run can promote it. Without the marker, the same evidence should close out the route family but not flip
the selected route.

## Tinygrad Artifact Contract

The producer should emit a single artifact shaped like:

```json
{
  "schema": "tinygrad.practical_roofline.v1",
  "model_id": "qwen3-8b",
  "target_id": "amd_gfx1100",
  "candidate_id": "decode_attention_g5_8b_refuted",
  "workload": "decode",
  "ceiling_basis": "owned_same_route_family",
  "protected_contexts": [512, 4096],
  "per_ctx": [
    {
      "ctx": 4096,
      "candidate_tok_s": 99.8,
      "practical_ceiling_tok_s": 101.0,
      "pct_of_practical_ceiling": 98.8,
      "baseline_candidate_delta_pct": -1.2,
      "spread_pct": 0.3,
      "route_bound": true,
      "correctness": true,
      "hidden_fallback": false,
      "economics": {
        "tile_us": null,
        "combine_us": null,
        "kv_bytes_per_layer": 16777216,
        "split_count": 48,
        "workgroups_per_cu": 4.0,
        "lds_bytes": 8192,
        "scratch_bytes": 0
      }
    }
  ],
  "verdict": "PRACTICAL_ROOFLINE_CLOSEOUT"
}
```

Null economics fields are allowed only for an inconclusive draft. A promotable artifact must provide the fields needed
to explain why the remaining gap is not actionable.

## BoltBeam Implementation Phases

### PR0: Adapter

Add `TinygradArtifactKind.PRACTICAL_ROOFLINE` and normalize `tinygrad.practical_roofline.v1` into:

- one `RowKind.CEILING` row per protected context;
- `metric="pct_of_practical_ceiling"`;
- `extra` carrying tok/s, delta, spread, basis, and economics.

Acceptance:

- adapter rejects missing `model_id`, `candidate_id`, protected contexts, or per-context ceiling rows;
- no model-name branches;
- tests cover current TG-P14-shaped fixture.

### PR1: Classifier

Add a pure helper, probably in `boltbeam/math/roofline.py` or `boltbeam/eval/practical_roofline.py`, that consumes
normalized ceiling rows and returns:

- practical class;
- worst context;
- worst pct of practical ceiling;
- largest gap beyond spread;
- missing fields.

Acceptance:

- 98.8% with 0.3pp spread classifies closeout/promotable;
- 96.5% classifies `HEADROOM_REMAINS`;
- missing context classifies `PRACTICAL_ROOFLINE_INCONCLUSIVE`;
- stale/mismatched candidate id is ignored or rejected.

### PR2: Evaluator Integration

Teach `evaluate` and `reopen-check` to use practical-roofline rows only after hard guardrails pass.

Acceptance:

- ordinary tier-A/B wins behave unchanged;
- existing speed-equivalent purity replacement tests behave unchanged;
- a generated handwritten-surface replacement with a small ordinary regression but `PRACTICAL_ROOFLINE_PROMOTABLE`
  promotes as tier `none`;
- a performance-only route with the same ceiling evidence does not promote, but returns closeout/diagnostic;
- a protected regression larger than the ordinary hard floor still refutes.

### PR3: Report And Ledger

Reports should show the practical roofline as its own section:

```text
Practical roofline: 98.8% of owned_same_route_family at ctx4096.
Classification: PRACTICAL_ROOFLINE_PROMOTABLE.
Action: promote as practical-ceiling-equivalent replacement; do not pursue more route variants unless a new primitive
changes the ceiling basis.
```

Ledger entries for closeout should record:

- `ceiling_basis`;
- worst protected context;
- worst percent of practical ceiling;
- whether the decision was promotion or search closeout only;
- reopen condition: new primitive, new ceiling basis, or a measurement that shows headroom beyond the noise band.

## Current TG-P14 Expected Outcome

If the final KV_BOTH evidence is packaged with prefilled correctness and route-bound flags, BoltBeam should classify
`decode_attention_g5_8b_refuted` as:

```text
PRACTICAL_ROOFLINE_PROMOTABLE for the 98% practical-ceiling replacement bar
```

Because it is still a near tie rather than an owned-beating speed win, the reason should say:

```text
practical-roofline-equivalent generated replacement; route family saturated; further PV work is not expected to be
worth pursuing unless a new primitive changes the ceiling basis
```

For `reopen-check`, preserve authorization semantics:

```text
verdict: candidate
reopen_check_original_verdict: promote
next_action: add this candidate back to the search space and run evaluate to authorize promotion
```

## Non-Goals

- Do not use raw HBM peak alone as promotion authority.
- Do not let a roofline artifact bypass correctness, route-bound, or rollback.
- Do not promote large protected regressions.
- Do not turn microbench timing into default selection without W==D.
- Do not mark a route as saturated when the practical ceiling basis is stale, cross-model, or cross-target.
