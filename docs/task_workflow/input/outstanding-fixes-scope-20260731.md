# Outstanding fixes scope

Status: proposed — all five items decided against the project principles
Scoped: 2026-07-31
Baseline: `5c960ee`
Follows: `../output/organization-and-pruning-result-20260731.md`

Everything the organization-and-pruning pass deliberately left undone, scoped to
the point where it can be executed.

Three items were initially recorded as blocked on an owner decision. They are
not: `coding-principles.md` and `minimization-principles.md` decide all three,
and in two cases evidence in the repo refutes the alternative outright. Each
decision below cites the rule that made it. Where a hypothesis was refuted, the
refuting evidence is given.

## 0. Summary

| # | Item | Decision | Decided by |
| ---: | --- | --- | --- |
| 1 | Divergent `_role_rank` | **Rename both, centralize the shared table** | DRY-as-knowledge; misleading names |
| 2 | `tools/` bypasses the library | **Wire up to the library** | "scripts that redefine canonical rules" |
| 3 | Environment-dependent test failures | **Make hermetic** | Test at the boundary; encode invariants |
| 4 | Twelve fused modules | **Split none, except one extraction** | Don't abstract prematurely; contain dangerous power |
| 5 | Versioned `perf/` files | **Forbid new ones; canonicalise before merging** | No-new-file rule; wrong-abstraction rule |

All five are now executable. Item 5 still requires a research step before its
final stage, but that step is itself specified.

---

## 1. Two `_role_rank` implementations that disagree

### 1.1 Evidence

```python
# boltbeam/analyze.py:26        — quant priority from registry bytes_per_elem
def _role_rank(role:TensorRole) -> tuple[float, int, int, str]:
    cap = quant_capability(role.quant)          # continuous, cheaper bytes = hotter

# boltbeam/workflow/analyze.py:20 — quant priority binary supported/unsupported
def _role_rank(role) -> tuple[float, int, int, str]:
    supported = bool(route_families_for(role.quant))
    role_rank = {"attn_kv": 0, ...}             # hardcoded name table
```

Confirmed divergence: for `Q8_0` vs `F16`, `analyze.py` ranks `Q8_0` first
(smaller bytes/elem); `workflow/analyze.py` ranks `F16` first (supported beats
unsupported).

Both are live. `boltbeam/analyze.py` is reached from `cli/workflow.py` and
`tests/test_analyze.py`; `boltbeam/workflow/analyze.py` from
`boltbeam/workflow/__init__.py`. Neither supersedes the other.

### 1.2 Why this is latent and not yet a bug

Both functions only order a list that is emitted in full. No consumer in the
tree indexes list position to drive a route or promotion decision. The
divergence is invisible until something treats position as priority — at which
point two code paths silently disagree about which role to measure first.

### 1.3 Decision: rename both, centralize the shared table

**Both are correct. They answer different questions.**

> DRY means knowledge, not lines. Do **not** eliminate code that merely *looks*
> alike but encodes *different* concepts.
>
> Duplication is cheaper than the wrong abstraction.

One ranks by memory cost ("what is hottest"), the other by routability ("what
can we act on"). Merging them would be the wrong abstraction — every future
divergence would have to fight a forced shared shape.

So the defect is **not** that there are two rankings. It is that two functions
answering different questions share one name, which is the explicit
bad-abstraction test:

> Bad abstraction means … naming that sounds generic but does not reduce real
> complexity.

There is also real duplicated knowledge here, and it is the part that must be
centralized: **both copies carry the same hardcoded role-name ordering table.**
That is one rule expressed twice, and it is what "if multiple parts of the
system need the same rule, move that rule to one explicit source of truth"
applies to. The quant axes are genuinely different and stay separate.

Note both quant rankings are already *derived* rather than hardcoded — one from
registry `bytes_per_elem`, one from `route_families_for`. That is principle 11's
data-driven pattern being followed on the axis that matters. Only the role-name
table is hand-written, in both copies.

### 1.4 Execution

1. Move the shared role-name ordering table to one owner — `vocab.py` already
   owns `RoleGroup` and is an established authority point. Both rankers import
   it. One commit, `[schema]`.
2. Rename `boltbeam/analyze.py:_role_rank` → `_rank_by_memory_cost`, and
   `boltbeam/workflow/analyze.py:_role_rank` → `_rank_by_routability`. One
   commit each, `[repo] NFC`.
3. Add a docstring to each stating the question it answers and, explicitly, that
   the other exists and answers a different one. Cross-reference by name.
4. Add a test asserting the `Q8_0` vs `F16` order **for each**, so the
   divergence is encoded as intended behaviour rather than left to be
   rediscovered as a bug.

### 1.5 Acceptance

1. Neither function is named `_role_rank`.
2. The role-name ordering table exists in exactly one place.
3. A test asserts `Q8_0` before `F16` under the memory-cost ranker and `F16`
   before `Q8_0` under the routability ranker.
4. Each docstring names the other function.

---

## 2. `tools/` bypasses the library

### 2.1 Evidence

Seven scripts import **zero** boltbeam code:
`promote_14b_routes.py`, `validate_fused_route_promotion.py`,
`account_prefill_roles.py`, `compare_same_run.py`, `prepare_q4_comparison.py`,
`measure_14b_same_run.py`, `prepare_14b_same_run.py`.

Each re-hardcodes the context ladder `(512, 1024, 2048, 4096)` and the role
taxonomy `("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv")` — which `vocab.py`
owns as `RoleGroup`. Each invents its own promotion-verdict shape rather than
using `eval` / `Verdict`.

And they mint schema ids **inside the `boltbeam.*` namespace**:

```python
tools/promote_14b_routes.py:63     "schema": "boltbeam.qwen3_14b_promotion.v1"
tools/account_prefill_roles.py:87  "schema": "boltbeam.prefill_role_accounting.v1"
```

That is the part that matters. Duplicated constants are a maintenance cost.
Artifacts asserting a `boltbeam.*` schema the library cannot validate are a
**provenance problem**: a consumer reading one of these files has no way to know
it was produced by something sharing no code with boltbeam.

### 2.2 Decision: wire them up

**The "deliberately portable" hypothesis is refuted by evidence.**

`tests/test_promote_14b_routes.py` executes the script with:

```python
subprocess.run([sys.executable, str(SCRIPT), ...], cwd=ROOT)
```

from the repository root, where `boltbeam` is importable. Nothing about how
these scripts are invoked honours a no-install constraint. They are not
standalone by design; they simply never imported.

They are also not obsolete probes — `promote_14b_routes.py` has a dedicated
test, `docs/14b-promotion-report.md` references it, and all seven were touched
on 2026-07-24. The "delete probes whose verdict is recorded" path does not apply.

With portability refuted and obsolescence refuted, the governing rule is
unambiguous and names scripts explicitly:

> Do not duplicate load-bearing logic across **scripts**, modules, or interfaces.
> If multiple parts of the system need the same rule, move that rule to one
> explicit source of truth.

And the anti-pattern list names the exact shape:

> one-off scripts that silently redefine canonical rules

### 2.3 Execution

1. One commit per script, `[repo]`: import `RoleGroup`, the context ladder and
   `Verdict` from the library; delete the local copies.
2. The self-minted `boltbeam.*` schema ids are the provenance half and must be
   resolved either way. After wire-up they should come from `vocab.py`. If a
   script genuinely emits an artifact shape the library does not model, add that
   schema id **to `vocab.py`** — the library should know about every artifact
   claiming its namespace.
3. Add a test asserting no file under `tools/` contains a hand-typed
   `"boltbeam.*.v<N>"` literal. Machine-enforced half of the rule.
4. `tools/check_commit_msg.py` and `tools/install_hooks.py` are exempt — they
   are repo infrastructure, import nothing, and mint no schema ids.
5. Each script has a test or a doc reference; verify it still passes or still
   produces the same artifact. Say explicitly if you could not verify one.

### 2.4 Acceptance

1. No file under `tools/` re-hardcodes `RoleGroup` members or the context ladder.
2. No file under `tools/` contains a hand-typed `"boltbeam.*.v<N>"` literal.
3. A test enforces both.
4. Every schema id in the `boltbeam.*` namespace is defined in `vocab.py`.

---

## 3. Environment-dependent test failures

### 3.1 Evidence

Two tests fail on this machine and have for the whole engagement:

```
tests/test_matched_control.py::test_plan_centralizes_equal_warmups_and_all_static_identities
tests/test_semantic_campaign_cli.py::test_actual_cli_runs_persistent_five_stage_campaign
```

The first resolves the machine's real GPU and compares it against an
Apple/Metal target id; this box is an RTX 5090, so preflight blocks with
`errors=['target_resolution']`. The second requires
`/home/ubuntu/tinygrad-arkey-exp`, which does not exist.

### 3.2 Why this needs fixing

A permanently red suite is a suite nobody reads. Every agent working in this
repo today had to be told the baseline was "2 failed" rather than green, and a
genuine third failure would have been invisible without that instruction. The
cost is not the two tests; it is that the signal is gone.

### 3.3 Execution

These are integration tests wearing unit-test clothing. Either:

- **Skip on absent precondition.** `pytest.mark.skipif` on the real condition —
  Apple target unavailable, tinygrad checkout absent. The test then reports
  `skipped` rather than `failed`, and the suite goes green. Preferred: it keeps
  the test running where it *can* run.
- **Inject the dependency.** `test_matched_control` already accepts a
  `run_command` fake; the GPU resolution should be injectable the same way, at
  which point it becomes a real unit test with no environment coupling.

The second is better and is roughly an hour. The first is ten minutes and
recovers the signal immediately. Doing the first now does not preclude the
second.

### 3.4 Acceptance

1. `python3 -m pytest -q` reports zero failures on a machine with no Apple
   target and no tinygrad checkout.
2. Neither test is deleted or `xfail`ed — skipped-with-a-reason or made
   hermetic, nothing else.
3. The reason is legible from the skip message without reading the test.

---

## 4. Twelve fused modules

### 4.1 Evidence

Four with genuine path + policy + state + execution fusion:
`control/matched_control.py` (585), `control/replay_ab.py` (405),
`workflow/autoscan.py` (360), `analyze.py` (267). Eight narrower cases carry one
or two mixed concerns.

The sharpest is `autoscan_run` — 15 lines doing path resolution, manifest load,
a subprocess hardware probe, an irreversible target-resolution policy decision,
and five persistence writes.

### 4.2 Decision: split none, except one extraction

> Add an abstraction only when it demonstrably removes more complexity than it
> introduces.
>
> Abstract only what has earned it. Premature generalization is itself an
> anti-pattern.
>
> Every admired codebase has long functions where logic is linear and benefits
> from being read straight through.

Fusion is a predicted cost, not an observed one. The audit measured structure;
it did not measure pain, and no incident has been cited where a change to one of
these modules forced an unrelated edit elsewhere. Splitting twelve modules
because an audit listed twelve is exactly the premature decomposition the
principles reject. **Split none.**

**One exception, and it is not about size.** `autoscan_run` buries an
irreversible policy decision — `_resolve_manifest_target`, which rewrites the
manifest — among five persistence writes and a subprocess probe. That is a
different rule:

> **Contain dangerous power.** Unsafe operations, global state, and destructive
> side effects must be isolated behind small reviewed boundaries. The boundary
> should document what invariant makes the operation valid, who may call it, and
> what state it may mutate.

An irreversible decision interleaved with I/O has no boundary and no place to
put that documentation. This one earns the extraction — not because the module
is 360 lines, but because a destructive operation is unguarded.

### 4.3 Execution, if a module is selected

Extract the policy decision, not the file. In `autoscan_run` the fused concern
worth separating is `_resolve_manifest_target` — an irreversible decision buried
among I/O. Lift it to a pure function taking hardware and manifest and returning
the resolved target, leaving the writes in the orchestrator. That is a small,
testable change that removes the actual entanglement without restructuring a
working pipeline.

### 4.4 Acceptance

1. No module is split without a named incident that splitting would have
   prevented.
2. Any split extracts a pure decision function; orchestration stays put.
3. The suite and CLI help gates hold.

---

## 5. Versioned `perf/` files

### 5.1 Position

13 files (`invocation_v2..v8`, `scheduling_v2..v7`) with 13 dedicated test
files. They look like duplication and are not:
`fit_invocation_v3(interaction, shaped)` and `fit_invocation_v8(operand,
scaffold)` consume different evidence and have different predict signatures.

Collapsing them is the wrong abstraction, and would rewrite all 13 tests.

### 5.2 What the principles actually prescribe

The *pattern* is the anti-pattern, going forward:

> **No-new-file rule:** a new experiment, variant, or case adds a *row to a
> table* or a *parameter* — not a new file.

Three steps, in order:

1. **Stop the bleeding.** There must be no `invocation_v9.py`. Add a test
   asserting no new `_v{N}.py` appears under `perf/`. Cheap, immediate, and it
   makes the rule machine-enforced rather than decorative.
2. **Canonicalise the inputs.** The blocker to merging is that v3 takes
   `interaction`/`shaped` and v8 takes `operand`/`scaffold`. Determine whether
   those are the same evidence under different names. This is a **research
   task** producing a written answer, not a refactor.
3. **Collapse only what step 2 proves identical.** Versions whose evidence
   genuinely differs stay separate. Frozen calibrations that must remain
   runnable move to `perf/archive/` with a lineage README; ones whose verdict is
   already in the ledger are deleted, citing the ledger entry.

Step 1 is executable now and is independent of the rest.

### 5.3 Acceptance

1. A test forbids new `_v{N}.py` under `perf/`.
2. Step 2 produces a written finding before any code moves.
3. No version is deleted without a ledger citation recording its verdict.

---

## 6. Recommended order

| # | Item | Effort | Risk |
| ---: | --- | --- | --- |
| 1 | §3 test failures — skipif, recover the signal | 10 min | none |
| 2 | §5.2 step 1 — test forbidding new `_v{N}.py` | 15 min | none |
| 3 | §1 `_role_rank` — centralize table, rename both | 1 hr | low |
| 4 | §2 `tools/` — wire up, one commit per script | 1–3 hrs | medium (untested paths) |
| 5 | §3 test failures — make hermetic | 1 hr | low |
| 6 | §4 — extract `_resolve_manifest_target` | 1 hr | low |
| 7 | §5.2 step 2 — canonicalisation research | half day | none (research) |

Items 1 and 2 recover signal and stop drift for almost nothing. Item 4 carries
the most risk because those scripts are largely untested — verify each against a
real artifact or state plainly that you could not.

Item 6 is the only structural change in this scope, and it is one function.

## 7. What this scope does not do

- It does not split `kernel_analysis/model.py`.
- It does not collapse the versioned perf models.
- It does not reorganize anything further; Phase A is complete.
- It does not split eleven of the twelve fused modules. The twelfth is extracted
  for containment of an irreversible operation, not for size.
- It does not merge the two rankers. They answer different questions; merging
  would be the wrong abstraction.
