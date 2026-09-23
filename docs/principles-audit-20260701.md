# BoltBeam Principles Audit - 2026-07-01

Scope: audit BoltBeam after BB0-BB12 against the local project principles in
`docs/coding-principles.md`, with emphasis on centralization of authority,
orthogonality, encoded invariants, error boundaries, tiny surface area, and
audit-brain safety.

Verification run:

```bash
python3 -m unittest discover -s tests -v
```

Result: 124 tests pass. The audit also ran targeted synthetic evaluator probes
for multi-context regression handling, quant/role scoping, and prefill speed
normalization.

## Executive Result

BoltBeam has the right repository shape: profile/search/policy/eval/artifacts
are separated; no tinygrad imports are present in the runtime path; schema ids
and core vocabulary are centralized; commit-prefix discipline is installed; and
the BB remediation tests cover the original review fixes.

The remaining risks are not architectural sprawl. They are decision-safety gaps:
the evaluator can still over-credit unscoped evidence. These should be fixed
before treating BoltBeam as the durable promotion/refutation authority.

## Resolution (2026-07-01)

All six findings below are fixed, plus the row-scope invariant the "Encode
Invariants"/"Orthogonalize" sections flagged. Verified by `python3 -m unittest
discover -s tests` (132 tests, no ResourceWarning):

- P0 multi-context: `evaluate` now aggregates every speed row (`_speed_points`)
  and refutes on any noise-qualified protected-context regression
  (`_first_protected_regression`) before tiering the median/best case.
- P0 cross-product: `_matches` requires a single row to satisfy the candidate's
  workload+quant+role together; no separate quant/role cross-product.
- Row-scope invariant: `evaluate` scores only `_candidate_rows` — rows whose
  (quant, role) the candidate applies to (None = candidate-agnostic, kept) — so
  a Q6_K/ffn_down row can no longer refute a Q4_K/attn_kv candidate.
- P1 search space: the CLI validates profile/search/evidence model+target
  consistency and only evaluates candidates authorized by the supplied space
  (`candidate_in_search_space`); an empty space means unconstrained.
- P1 prefill: the adapter emits `tok_s` rows with `extra["baseline_tok_s"]`
  (from `authority_ref`), so the evaluator computes a signed delta.
- P1 target id: adapters accept `target_id` (CLI `--target-id`); the default
  stays `amd_gfx1100` only when nothing is supplied or derivable.
- P2 hygiene: `read_evidence` uses `Path(path).read_text()` — no leaked handle.

Regression coverage: `tests/test_audit_fixes.py`.

## Findings

### P0 - Multi-context regressions can be hidden by the first speed row

Principle violated: audit-brain safety; performance search rules.

Evidence:

- `boltbeam/eval/evaluator.py:_speed_delta_and_spread` returns the first
  computable speed row only.
- A synthetic artifact with `ctx512 = +6%` and `ctx4096 = -8%` promoted as
  tier A because the evaluator ignored the later protected-context regression.

Impact: a route can promote while failing a measured context. This violates the
same discipline that kept the tinygrad work honest: every protected context must
pass, not only the first row in artifact order.

Fix direction:

- Aggregate all applicable speed rows for the candidate.
- Refute on any protected-context regression before tiering.
- Compute tier from the conservative candidate-level summary, not from the first
  row.

### P0 - Quant and role matching cross-products unrelated rows

Principle violated: scoped evidence; encoded invariants.

Evidence:

- `boltbeam/eval/evaluator.py:_matches` collects all row quants and all row
  roles independently, then checks every combination.
- A synthetic evidence bundle with `Q6_K + attn_kv` and `Q4_K + ffn_down`
  promoted Q4_K/attn_kv candidates even though no single row proved that pair.

Impact: one row can supply the quant, another row can supply the role, and a
third row can supply speed. That is not a valid route proof.

Fix direction:

- Match candidates against row-level or evidence-group-level scopes.
- Filter evidence passed to `evaluate()` down to rows applicable to the
  candidate.
- Require each required evidence kind to exist inside the candidate's applicable
  scope.

### P1 - `evaluate --search` loads but does not use the search space

Principle violated: search space as authority; centralization.

Evidence:

- `boltbeam/cli.py:cmd_evaluate` validates that `--search` is a JSON object, but
  candidate selection still comes from the global manifest and evidence only.

Impact: a stale, mismatched, or intentionally constrained search space cannot
  prevent a promotion. The CLI contract says profile/search/evidence are the
  evaluation authority, but only evidence materially affects the result.

Fix direction:

- Validate `profile.model_id`, `search.model_id`, evidence `model_id`, and
  target/workload consistency.
- Restrict candidate evaluation to candidates present and reachable in the
  supplied search space.
- Fail closed on mismatched model or target identity.

### P1 - Prefill authority rows are normalized but not evaluable

Principle violated: adapter output must be evaluator-ready.

Evidence:

- `boltbeam/artifacts/tinygrad.py:_adapt_prefill_authority` emits
  `prefill_speed` rows with metric `whole_prefill_tok_s`.
- `boltbeam/eval/evaluator.py:_row_delta_pct` computes deltas only for
  `delta_pct`, `lag_pct`, or `tok_s` with `baseline_tok_s`.
- Synthetic prefill authority evidence for `prefill_pipe_global` and
  `prefill_pipe_role_selective` returns `inconclusive: speed delta not
  computable`.

Impact: the prefill promotion path exists in the manifest, but authority
artifacts cannot drive a promotion/refutation decision without extra manual
translation.

Fix direction:

- Normalize prefill artifacts into signed `delta_pct` rows, or emit `tok_s`
  with `baseline_tok_s` and an explicit candidate arm.
- Preserve per-context spread and protected-context regression checks.

### P1 - Most tinygrad adapters hardcode `amd_gfx1100`

Principle violated: target identity must be explicit; design for replacement.

Evidence:

- `boltbeam/artifacts/tinygrad.py` defines `_GFX1100 = "amd_gfx1100"`.
- Role attribution, reduce-source, promotion-gate, decode-WD, and prefill
  adapters stamp `_GFX1100` directly.

Impact: non-gfx1100 artifacts will be mislabeled as AMD gfx1100 unless their
adapter happens to parse hardware. This blocks the target-agnostic direction.

Fix direction:

- Add an explicit `--target-id` ingest override.
- Derive target from artifact facts when present.
- Refuse target-unknown artifacts instead of defaulting to gfx1100.

### P2 - Evidence reads leak file handles

Principle violated: hardening / boring public surfaces.

Evidence:

- `boltbeam/artifacts/base.py:read_evidence` uses
  `json.loads(open(path).read())`.
- The full test run passes but emits repeated `ResourceWarning: unclosed file`.

Impact: low immediate risk, but it is a simple hygiene issue in a file-boundary
helper.

Fix direction:

- Replace with `pathlib.Path(path).read_text()` or a context manager.

## Principles Coverage

### Centralize Authority

Status: mostly good.

- `boltbeam/vocab.py` centralizes verdicts, schema ids, row kinds, guardrails,
  and reachability labels.
- `boltbeam/data/candidates.json` centralizes the candidate manifest.
- Remaining gap: the CLI does not yet make the supplied search-space document a
  true authority for evaluation.

### Modularize Execution

Status: good.

- Artifact parsing lives in `boltbeam/artifacts`.
- Candidate search lives in `boltbeam/search`.
- Promotion logic lives in `boltbeam/eval`.
- Ledger policy lives in `boltbeam/ledger` and `boltbeam/policy`.
- No GPU execution or tinygrad import is performed by BoltBeam.

### Abstract For Simplicity

Status: good with one gap.

- `NormalizedEvidence`, `CandidateDecision`, `LedgerEntry`, `ModelProfile`, and
  `TargetProfile` are the right stable interfaces.
- Gap: `NormalizedEvidence.rows` currently acts as an unscoped bag. The next
  abstraction should be candidate-scoped row filtering or evidence groups.

### Orthogonalize Concerns

Status: mostly good.

- Parsing, search, policy, evaluation, and reporting are separated.
- Gap: evaluation currently mixes all rows from matched evidence. Orthogonality
  is present at the module boundary but not yet at the row-scope boundary.

### Encode Invariants

Status: partial.

- `CandidateDecision` enforces no promotion without rollback and no refutation
  without a reopen condition.
- The missing invariants are candidate-scope invariants:
  - all rows used for a candidate share the candidate's model/target/workload;
  - quant/role are satisfied by the same row or explicit evidence group;
  - all protected contexts pass.

### Errors As System Information

Status: improved, but target/mismatch errors need hardening.

- Malformed schemas now return clean CLI exit code 2.
- Reduce-source requires an explicit `model_id`.
- Missing: target identity should fail closed rather than defaulting to gfx1100.
- Missing: evaluate should reject profile/search/evidence mismatch.

### Tiny Surface

Status: good.

- The repo is compact and module boundaries are understandable.
- The current fixes should be surgical: evaluator scoping/aggregation,
  prefill adapter delta rows, target-id ingest override, and file-read hygiene.
  No broad refactor is needed.

## Recommended Fix Order

1. Evaluator scoping and all-context aggregation.
2. CLI profile/search/evidence consistency and search-space restriction.
3. Prefill authority delta normalization.
4. Target-id ingest override and fail-closed target handling.
5. `read_evidence` file-handle cleanup.

## Regression Tests To Add

- A candidate with one positive context and one protected-context regression
  must refute or stay candidate, never promote.
- A candidate must not match when quant and role appear only on different rows.
- `evaluate --search` with a search space that omits a candidate must not
  promote that candidate.
- Prefill authority fixture must produce an evaluable delta and classify the
  expected pipe candidate.
- Ingesting an artifact with no target and no `--target-id` should fail closed
  once non-gfx1100 targets are supported.
