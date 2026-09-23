# Organization and pruning — result

Status: complete for Phases A–E and mechanical remediation
Executed: 2026-07-30 / 2026-07-31
Input: `../input/organization-and-pruning-scope-20260730.md`
Range: `6ca3c3a..5c960ee` (40 commits)

What was executed, what was found, what was verified independently, and what was
deliberately left undone.

## 1. Outcome

| Metric | Before | After |
| --- | ---: | ---: |
| `cli.py` | 1,773 lines, one file | 7-file package, `__init__.py` at 20 lines |
| Parser-to-handler distance | ~1,000 lines | adjacent |
| Flat modules at `boltbeam/` root | 45 | 5 |
| Domain packages | — | 10 |
| `search/` flat files | 34 | 12 + 5 sub-packages |
| CLI surface | 132 commands | 132 commands, per-command help byte-identical |
| Test baseline | 2 failed, 1196 passed | 2 failed, 1197 passed |
| Schema-id literals | 195 | 187 (8 replaced by imports) |

Every commit was gated on the full suite, a 132-command per-command `--help`
diff, and the schema-literal count. The two failures are pre-existing and
environment-dependent; that set never grew.

## 2. Phase A — organization

**A.1** `cli.py` → `cli/` package. 68 handlers split across five domain modules
(analysis 7, search 13, profile 13, roofline 5, workflow 30), each parser
definition adjacent to its handler. `main()` reduced to a registrar loop.

**A.2** 45 flat root modules → 10 packages: `roofline/ control/ runtime/ trace/
memory/ quantization/ vocabulary/ plan/ experiment/ target/`. `analyze.py`,
`vocab.py`, `manifest.py` and `full_kernel_candidate_set.py` remain at root.

**A.3** `search/` → 5 sub-packages: `full_kernel/ epochs/ mmq/ joins/ semantic/`.
No sub-package exceeds 8 files.

### Latent bugs found (not part of the scope, found by reading)

Three classes of defect that no test covered, exposed by adding a directory level:

- `Path(__file__).resolve().parent / "data"` lookups in `quant.py`,
  `model_vocab.py` and `targets.py` needed `.parent.parent`.
- `parents[N]` repo-root walks in `tinygrad_root.py`, `matched_control.py`,
  `replay_ab.py`, `full_kernel_search.py` and `tinygrad_full_kernel.py` were
  off by one after the move.
- Hardcoded path strings and `monkeypatch`/`sys.modules` string targets in four
  test files that import rewriting cannot reach.

These would have shipped broken under a "mechanical move, low risk" assumption.

## 3. Phases B–E — audits

### B. Duplication

Three same-knowledge findings, all remediated. The scope's strongest hypothesis
— that the four trace importers are one parser with four column maps — was
**refuted**: `llama_rocprof` and `tinygrad_rocprof` already share their
primitives, and what remains separate (`classify_llama_kernel` vs
`classify_tinygrad_kernel`, byte-attribution by quant-name vs shape-index) is
different knowledge. `import-ncu` reads a different schema entirely.

Correctly left alone: six MR7/MR8/MR9 handlers writing JSON outside `_write`,
because four implement a no-clobber policy `_write` does not have.

### C. Centralization

The 195-literal question **deflated on inspection**: `vocab.py` defines 62 of
~176 distinct schema ids; most of the rest are legitimate per-subsystem
authority (`perf/`, `search/mmq/`, `search/semantic/` own theirs). Real
duplication was 10 sites across four ids — all remediated.

Biggest finding, **not remediated**: `tools/` (§5.2).

Also found: the `commit-msg` checker rejected five prefixes that eight commits
in this history already used. Remediated.

### D. Abstraction

Nine handlers reimplemented `_load_json` inline and two reimplemented
`_run_manifest_defaults` — a second unofficial way to do what a canonical helper
already does. Both remediated. `_VERDICT_PRIORITY` was policy living in the CLI;
moved to `eval/evaluator.py`.

`CandidateFactors` and siblings had `to_json()` but no inverse, so the CLI
invented one across 23 lines. The remedy belonged on the owning module:
`from_json()` added with a round-trip test.

`provider_adapters.py` audited **clean** — shared policy above the adapter,
backend translation inside, the one real capability gap (hw_trace has no
correctness oracle) stated in a docstring rather than faked.

### E. Orthogonality

`vocab.py` is an **authority point, not a god module**, on the test that
matters: every export is a constant, enum member or pure function. It exports
knowledge, never behaviour. ~65 importers is centralization working as designed.

Runtime use and meta-development are cleanly separated.

Twelve fused modules found, **not remediated** (§5.3).

Surfaced an extra that neither other audit caught (§5.1).

## 4. Remediation applied

| Commit | Change |
| --- | --- |
| `f1e259a` | commit-msg allowlist accepts the five prefixes already in use |
| `43dfcca` | `_VERDICT_PRIORITY` moved to `eval/evaluator.py` as `primary_decision()` |
| `b34c0dd` | `trace/timing.py:_fingerprint` replaced by `artifacts/base.sha256_json` |
| `380c6fb` | 9 inline `_load_json` bypasses replaced with the helper |
| `e00eb4f` | 2 inline `_run_manifest_defaults` reimplementations replaced |
| `81cd9c0` | 8 re-hardcoded schema ids replaced by imports; literal count 195 → 187 |
| `5c960ee` | `from_json()` added to four candidate types; CLI's 23-line reconstruction removed |

## 5. Deliberately not done

Recorded so silence is not read as completion. Scoped for execution in
`../input/outstanding-fixes-scope-20260731.md`.

### 5.1 Two divergent `_role_rank` implementations

`boltbeam/analyze.py:26` and `boltbeam/workflow/analyze.py:20` both rank tensor
roles, by **different rules**. The first derives quant priority from registry
`bytes_per_elem`; the second uses binary supported/unsupported via
`route_families_for`, plus a hardcoded name table.

Confirmed order swap: for `Q8_0` vs `F16`, `analyze.py` ranks `Q8_0` first,
`workflow/analyze.py` ranks `F16` first.

Both are live and feed independent CLI paths. **Latent, not a live bug** — no
consumer indexes list position to drive a decision today. It would become one
the moment a consumer treats position as "measure this first".

Not remediated because deciding which ranking is correct is a behaviour
question, not a refactor.

### 5.2 `tools/` scripts bypass the library entirely

Seven scripts import **zero** boltbeam code while re-hardcoding the role
taxonomy `vocab.py` owns as `RoleGroup`, and each inventing its own
promotion-verdict shape instead of using `eval`/`Verdict`.

Worse than duplication: they mint schema ids **in the `boltbeam.*` namespace**
that the library has never heard of —
`"boltbeam.qwen3_14b_promotion.v1"`, `"boltbeam.prefill_role_accounting.v1"`.
They emit artifacts claiming boltbeam schemas while sharing no code with
boltbeam.

Not remediated because whether these are meant to be standalone and portable is
a design decision the owner has not made.

### 5.3 Twelve fused modules

`control/matched_control.py` (585), `control/replay_ab.py` (405),
`workflow/autoscan.py` (360) and `analyze.py` (267) genuinely fuse path, policy,
state and execution — `autoscan_run` does path resolution, a subprocess probe,
an irreversible policy decision and five persistence writes in 15 lines. Eight
narrower cases carry one or two mixed concerns.

Not remediated: splitting these is restructuring, not cleanup, and the value is
softer than the risk.

### 5.4 Versioned `perf/` files

13 files `invocation_v2..v8` / `scheduling_v2..v7` were left intact per §6 of the
input scope. They look like duplication and are not:
`fit_invocation_v3(interaction, shaped)` and `fit_invocation_v8(operand,
scaffold)` consume different evidence.

## 6. Corrections to the input scope

The scope was wrong in two places, both found during execution and both fixed
in the document itself rather than worked around:

**Acceptance criterion 3** required `--help` byte-identical *including command
ordering*, while also requiring domain-grouped modules with one `register()`
each. Unsatisfiable: domains interleave non-contiguously in the source
(`roofline` at positions 23, 24, 43, 44, 71), so no permutation of five
whole-module calls reproduces the order. Replaced with two checks — set equality
plus per-command help byte-identity — which is *stricter* on the property that
matters, since a top-level listing cannot reveal a dropped argument.

**The A.2 package membership table** was a filename-derived hypothesis and was
wrong about `vocab.py`, which it filed under `vocabulary/`. With ~65 importers
across every subsystem it is a repo-wide authority module, not part of the
model-codegen vocabulary domain that `vocab_capture` and `model_vocab` share.
Left at root.

The A.3 semantic membership was also wrong: the real import graph has 10 files,
not the 8 listed. Two were verified as import leaves and left flat rather than
breaching the 8-file cap.
