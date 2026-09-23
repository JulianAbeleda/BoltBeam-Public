# `perf/` versioned model verdict record

Status: **COMPLETE — verdicts found recorded in `bench/`, 2026-07-31**
Prepared: 2026-07-31
Purpose: satisfy the pruning bar in
`../input/organization-and-pruning-scope-20260730.md` §F.2 —
*"No ledger entry, no deletion."*

## Why this document and not a ledger entry

The route ledger (`boltbeam/ledger/`, `LedgerStatus`, `route_ledger.jsonl`)
records decisions about **route candidates** — whether a codegen route beat its
baseline. Its statuses are `promoted / refuted / deferred /
search-space-incomplete / candidate` and its unit is a `candidate_id` with a
`scope` of `{model_family, quant, target}`.

The 13 versioned `perf/` models are **cost-modelling infrastructure**, not route
candidates. They have no `candidate_id`, no quant, no target. Forcing them into
the route ledger would misuse the schema to satisfy a rule about durability.

§F.2's requirement is a *durable record of the verdict*. A written finding is a
valid form of that — `perf-canonicalisation-finding-20260731.md` is precedent.
This is that record, with the part only the owner knows left blank.

## What is established from evidence

All 13 modules, their upstream gate, and their provenance. Every schema literal
below is verbatim from the source.

| Module | Gates on upstream schema | Commit subject (all 2026-07-11) |
| --- | --- | --- |
| `invocation_v2` | `tinygrad.mmq_invocation_v1.generated_host_factorial.v1` | model MMQ-shaped host invocation |
| `invocation_v3` | `tinygrad.mmq_invocation_v2.generated_host_interaction.v1` | model MMQ host interaction extrapolation |
| `invocation_v4` | `tinygrad.mmq_invocation_v3.generated_host_topology.v1` | bound MMQ host topology contrast |
| `invocation_v5` | `tinygrad.mmq_invocation_v5.grouped_predicate.v1` | model grouped MMQ host topology |
| `invocation_v6` | `tinygrad.mmq_invocation_v5_scaffolding.exact_histogram.v1` | model exact-size MMQ host scaffolding |
| `invocation_v7` | `tinygrad.mmq_builder_event_factorial.v1` | model MMQ Python builder events |
| `invocation_v8` | `tinygrad.mmq_invocation_v7.writeback_builder.v1` | model MMQ operand interaction |
| `scheduling_v2` | `tinygrad.mmq_scheduling_calibration.v1` | model aggregate MMQ wave scheduling |
| `scheduling_v3` | `tinygrad.mmq_long_chain_calibration.v1` | model long-chain MMQ wall transfer |
| `scheduling_v4` | *(delegates to `scheduling_v3`)* | separate MMQ timing objectives |
| `scheduling_v5` | `tinygrad.mmq_residual_probe.v4` | model MMQ false-site interactions |
| `scheduling_v6` | `tinygrad.mmq_residual_grid_factorial.v5` | model MMQ grid transfer model |
| `scheduling_v7` | `tinygrad.mmq_exact_isa_residual.v6` | model exact-ISA MMQ residual |

### Established facts

1. **All 13 were authored in one burst on 2026-07-11**, within roughly 90
   minutes, every commit prefixed `[search]`. This was a single MMQ
   cost-modelling investigation, not accumulated drift.
2. **Each gates on a distinct, mutually exclusive upstream schema.** No two
   accept the same producer. Established in
   `perf-canonicalisation-finding-20260731.md`.
3. **They are unreachable from production.** Zero references from
   `boltbeam/cli/`, zero from any non-`perf` module, zero from `tools/`. The
   four production importers of `boltbeam.perf` (`kernel_analysis/contrast.py`,
   `kernel_analysis/theoretical_roofline.py`, `memory/lds_l2.py`,
   `search/mmq/mmq_prediction.py`) never touch the versioned models. Reachable
   only via `perf/__init__.py`'s re-exports and their own tests.
4. **CORRECTION: the artifacts exist, in `bench/`.** An earlier version of this
   document claimed none existed. That search covered `outputs/` only and was
   wrong. There are **52 `bench/mmq-*.json` artifacts**, dated the same day as
   the models, **20 of which carry an explicit recorded status.**
5. **Their 13 tests use synthetic fixtures.** They assert structural properties
   (`median > 0`) against hand-built rows with fabricated values. They prove the
   fitting code runs; they do **not** record that any model was validated
   against real measurement.
6. **The upstream producers are tinygrad-side.** Every gate names a
   `tinygrad.mmq_*` schema, so the evidence these models consume was generated
   by the tinygrad experiment harness, not by BoltBeam.

### The recorded verdicts

Read verbatim from the `bench/` artifacts:

| Model | Recorded status |
| --- | --- |
| `invocation_v2` | `falsified` |
| `invocation_v3` | `falsified` |
| `invocation_v4` | `falsified` |
| `invocation_v5` | `falsified` |
| `invocation_v6` | `falsified` |
| `invocation_v7` | `falsified` (plus `no_model_independent_calibration_missing`, `no_model_event_calibration_required`) |
| `invocation_v8` | `operational_total_validated_with_minor_phase_scope_gaps` |
| combined operational model v1 | `validated` — `individual_error_below_15pct: true`, `ratio_error_below_10pct: true` |
| scheduling residual v5 | `diagnostic_only_post_falsification` |

**This is a falsification sequence, not accumulated duplication.** Each version
was a hypothesis, tested against measurement, and rejected — until `v8`
survived. `mmq-combined-operational-model-v1` records the validated end state
along with a `knowledge_audit` naming both what was established (exact candidate
ISA and resources, device kernel duration, host event costs, operand fixed cost,
writeback/topology interaction) and what remained unknown (tail model for rare
host outliers, warm-cache lookup transfer, sub-2µs numpy view variability).

The result was recorded properly at the time. It was recorded in `bench/`
rather than the route ledger, which is why a ledger-shaped search did not find
it — and why the reachability analysis, which only looked for *callers*, read a
completed investigation as an abandoned one.

## The answers

The questions this document originally left blank are answered by the artifacts.

**Q1 — what did the invocation modelling establish?** That `v2` through `v7`
were each falsified against measurement, and `v8` reached
`operational_total_validated_with_minor_phase_scope_gaps`. The combined
operational model was `validated` inside its declared error gates.

**Q2 — what did the scheduling/residual modelling establish?** Less cleanly
recorded. `mmq-sq-residual-v5` carries `diagnostic_only_post_falsification`,
indicating the residual work served as diagnostics after the invocation models
were falsified rather than as a standalone validated model. The remaining
scheduling versions have no explicitly labelled status artifact — this is the
one genuine gap.

**Q3 — retention.** `invocation_v8` is the survivor of its family and the only
invocation model with a positive verdict. `v2`–`v7` are falsified: their value
is the *record that they were falsified*, which lives in `bench/`, not in the
code.

## Prune decision matrix

Once Q1–Q3 are answered, the action follows mechanically:

The recorded verdicts select the second row: **one version is the validated
result.**

- **Keep `invocation_v8`** — the only invocation model with a positive verdict.
  Add a docstring stating what it established and citing
  `bench/mmq-invocation-v8-validation-20260711.json`.
- **Falsified: `invocation_v2`–`v7`.** Their verdict is recorded in `bench/`.
  The code is the refuted hypothesis; the artifact is the refutation. Under the
  branch layout these are `exp` apparatus, not product.
- **Scheduling family: hold.** Q2 is the one unresolved question. Do not remove
  `scheduling_v2`–`v7` until a status artifact is located or the owner confirms
  the residual line was diagnostic-only.

**Removal here means moving off the trunk, not destroying.** `dev` and `exp`
retain everything (see `docs/branch-flow.md`). Nothing is lost, and §F.2's
purpose — never destroy the only copy of a result — is satisfied twice over:
the artifacts are in `bench/`, and the code remains on the outer branches.

## What must not happen

Do not delete any of these files while this document still contains a blank.
An entry that records the *absence* of a verdict does not satisfy §F.2 — it
launders it. The rule exists so that deleting code cannot destroy the only copy
of a result, and that risk is live here precisely because points 3–5 show the
result was never written down anywhere else.

If the owner cannot recall the outcome, that is itself an answer — it means the
investigation produced nothing durable, which supports the first row of the
matrix. But it has to be stated, not assumed.

## Already enforced

`tests/perf/test_no_new_versioned_files.py` pins the current 13 and fails on any
addition. Whatever is decided here, the pattern cannot grow. If files are
deleted, that test's pinned set must be updated in the same commit.
