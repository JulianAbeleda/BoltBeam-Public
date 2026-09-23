# Same-run campaign gates: model removed from code

Status: **COMPLETE — executed 2026-07-31**
Follow-on, same day: the gates described here were then moved off the trunk to
`dev` as campaign apparatus (`96dd5ec` / `b0ff907`). Every `tools/` path below
resolves on `dev` and `exp`, not on `main`. The reasoning was already implicit in
this document — see the argument for placing `same_run.py` under `tools/` rather
than `boltbeam/` — but was not carried through to the gates themselves at the
time. The verdicts stay on the trunk in `bench/`.
Trigger: owner review of the README scripts table — *"why do we have hardcoded
scripts. 14b scripts shouldn't exist. scripts should be modular in nature and if
they aren't they should be either decoupled or removed."*

## The finding, and why it was bigger than the filenames

Three scripts named a model: `promote_14b_routes.py`, `prepare_14b_same_run.py`,
`measure_14b_same_run.py`. A fourth named a quant, `prepare_q4_comparison.py`.
That much was visible from the README.

Two things found on inspection were not visible from the README.

**1. The model had reached the schema layer.** The wiring pass earlier the same
day gave these scripts `SCHEMA_QWEN3_14B_PROMOTION`,
`SCHEMA_QWEN3_14B_SAME_RUN_MEASUREMENT` and
`SCHEMA_QWEN3_14B_SAME_RUN_PREPARATION` in `boltbeam/vocab.py`. Centralising the
ids was right; keeping the model *inside* them moved the hardcoding one level
down rather than removing it. A schema id naming a model cannot describe a
second model, so the schema layer would have forced a per-model copy even after
the scripts were parameterised.

**2. Six scripts, not four, and the real blocker was duplicated knowledge.**
`compare_same_run.py`, `validate_fused_route_promotion.py` and
`account_prefill_roles.py` carried the same constants. More seriously, four of
them had each grown a private answer to *"is this artifact route-pure?"*, and the
answers had drifted:

| script | route block read from | accepted when |
| --- | --- | --- |
| `promote_14b_routes.py` | `route`, `route_provenance` | `fallback` is False |
| `measure_14b_same_run.py` | `route`, `route_provenance`, `route_attribution` | provenance present **and** `fallback` False |
| `compare_same_run.py` | `route_provenance`, `route_attribution` — **not** `route` | `fallback` False **or** `prefill_route_rolled_back` False |
| `account_prefill_roles.py` | `route_provenance`, `route_attribution`, `route` | `fallback` False, defaulting to False when absent |

An artifact `promote` accepted, `compare` rejected. `throughput_by_context` had
two implementations differing in whether a `contexts` key counted. This is the
repo's own rule — *DRY means knowledge, not lines* — violated on the knowledge
that matters most in a fail-closed gate: what counts as valid evidence.

It is also **why the model could not simply be lifted into a flag**: there was no
single place to lift it out of. The duplication was the blocker, not the literal.

## Decision: decouple, not remove

The branch layout offers three outcomes. Removal was rejected on evidence:
`bench/14b-promotion-report-20260715.json` records `status: BLOCKED` with
`next_action: "collect fresh 14B role and all-context raw timing artifacts;
rerun"`, last touched 2026-07-15. Unlike the MMQ models (verdicts recorded, work
concluded) this campaign is **paused mid-flight**, and
`docs/parity-14b-32b-scope.md` schedules 32B parity behind it — explicitly to
*"iterate both models against one search space and one ledger"*. Executing that
scope with per-model scripts means writing `promote_32b_routes.py`, which is the
sprawl the no-new-file rule exists to prevent.

So the gates stay, and the thing that made them per-model was removed.

## What changed

**`boltbeam/target/targets.py`** gains `target_aliases()` and `target_matches()`.
Every script had hand-typed `("gfx1100", "amd_gfx1100")` — an equivalence between
a registry id and a bare architecture. The registry owns target identity, so it
now owns the equivalence, derived from the registry's own vendor prefixes rather
than listed (so `sm120`/`nvidia_sm120` works without an edit).

**`boltbeam/vocab.py`** — ids renamed to name a *kind of gate*, never the model:

| was | is |
| --- | --- |
| `boltbeam.qwen3_14b_promotion.v1` | `boltbeam.route_promotion_gate.v1` |
| `boltbeam.qwen3_14b_same_run_measurement.v1` | `boltbeam.same_run_measurement.v1` |
| `boltbeam.qwen3_14b_same_run_preparation.v1` | `boltbeam.same_run_preparation.v1` |
| `boltbeam.q4_comparison_preparation.v1` | `boltbeam.quant_comparison_preparation.v1` |

**`tools/same_run.py`** is new: the single answer to where the route block lives,
what route purity means, how a throughput ladder is keyed, and what a run's scope
is (`RunScope`: model, target, quants, contexts, roles). Every artifact now
carries the scope as *fields*, which is what the schema id stopped carrying.

It sits in `tools/`, not `boltbeam/`, deliberately. Per `docs/branch-flow.md` a
module whose only callers are campaign gates is apparatus, not product; putting
it under `boltbeam/` would have created a trunk module with no product caller —
precisely the shape just pruned from `perf/`. It travels with the scripts.

**Renames:**

| was | is |
| --- | --- |
| `tools/promote_14b_routes.py` | `tools/promote_routes.py` |
| `tools/prepare_14b_same_run.py` | `tools/prepare_same_run.py` |
| `tools/measure_14b_same_run.py` | `tools/measure_same_run.py` |
| `tools/prepare_q4_comparison.py` | `tools/prepare_quant_comparison.py` |

All six gates take `--contexts`, and where meaningful `--model`, `--target`,
`--quants`, `--role-groups`, `--parity-fraction`, `--route-identity` and
`--candidates`. Defaults reproduce the previous hardcoded values, so the 14B
campaign's invocations change only by gaining `--model Qwen3-14B-Q4_K_M`.

`account_prefill_roles.py` deliberately takes no `--model`: the whole-prefill
artifact is the authority on what was measured and every role artifact must agree
with it. Pinning a model there would have added a second source of truth.

## Behaviour changes, stated rather than buried

Unifying four route-purity checks means each script's behaviour moved to the
union. These are deliberate:

1. **`promote_routes` is stricter.** It previously required only
   `fallback is False`; it now also requires a non-empty provenance string. A
   role artifact with no declared provenance no longer counts toward the gate.
2. **`compare_same_run` is more permissive in one direction.** It now reads a
   plain `route` key, which it previously ignored — an artifact using that
   spelling was silently failing provenance.
3. **`prefill_route_rolled_back: false` is accepted everywhere** as an
   alternative spelling of `fallback: false`. It asserts the same fact in the
   prefill vocabulary. Absence of all three spellings remains an error.
4. **`promote_routes` checks every evidence candidate row**, not only rows named
   `q4` and `q6`. A third candidate is no longer skipped unvalidated.
5. **`validate_fused_route_promotion` accepts `same_run` or `same_run_14b`** for
   the joined role block. The model-specific key was a *producer field name* —
   the deepest form of the hardcoding — and both spellings are read so existing
   producer artifacts still validate.
6. **`prepare_quant_comparison` requires `--route-identity`.** It previously
   hardcoded `q4_generated_geometry`; there is no defensible default, so it is
   explicit rather than defaulted.

## What was deliberately not touched

**`bench/` artifacts keep their old schema ids.** Four artifacts carry
`boltbeam.qwen3_14b_*` ids. They record what was emitted on 2026-07-15 and
rewriting them would falsify the record. The ids are retired for new output, not
retracted from history.

**`docs/task_workflow/` audits keep the old names.** Same reason: they describe
the state of the repo when they were written.

**The five 14B/Q4 runbooks under `docs/` keep their framing.** They describe one
specific campaign run, which remains accurate; only their invocations and cited
schema ids were updated so the commands still execute.

## Enforcement

`tests/test_tools_wired_to_library.py` gains
`test_no_tools_script_hardcodes_a_model_identifier`, which fails on any string
literal under `tools/` carrying a parameter-count token (`14B`, `-8b-`, `32B`).
The three deleted per-model scripts differed from their successors essentially
only in that literal, so this is the machine-enforced half of the owner's rule.
It caught a model name in this work's own `--model` help text on first run.

Suite: **1144 passed, 4 skipped** (was 1133 passed — 11 new tests, including
`test_gate_is_model_agnostic` and `test_same_gate_measures_a_different_model`,
which run the gates against a 32B model id and assert nothing 14B-shaped
survives).

## What this unblocks

`docs/parity-14b-32b-scope.md` can now be executed as written. Adding 32B is
`--model Qwen3-32B-Q4_K_M` on the existing gates, and the two models produce
artifacts under the same schema ids that a single ledger can compare — which is
what that scope asked for and what the per-model scripts made impossible.
