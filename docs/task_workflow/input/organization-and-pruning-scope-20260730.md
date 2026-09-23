# BoltBeam organization and pruning scope

Status: proposed — ready to execute
Scoped: 2026-07-30
Supersedes: `organization-scope-claude-20260730.md`, `organization-scope-qwen-20260730.md`
Authority: `structure-template/structure/Development/coding-principles.md`

This document merges the two prior organization scopes, resolves where they
disagreed, and extends the work into an exhaustive pruning audit against the
project coding principles. It is written to be **executed by an agent**, so
every phase states what to look for, how to prove it mechanically, and what
counts as done.

## 0. Verdict table

| Area | Prior scopes | This scope |
| --- | --- | --- |
| `cli.py` (1,773 LOC) | both: split | **Split** — parser adjacent to handler |
| `kernel_analysis/model.py` (1,492 LOC) | both: leave (Qwen self-contradicted) | **Leave.** Single dependency tree |
| 45 flat root files | Qwen only | **Package.** Claude missed this |
| `search/` (37 files) | Qwen only | **Carve.** Claude missed this |
| 13 versioned `perf/` files | Qwen: collapse | **Do not collapse yet.** Remedy is wrong (§6.1) |
| Duplication / centralization / abstraction / orthogonality | neither | **New: phases B–E** |

Two corrections carried forward from the prior review:

- Qwen's §7 phase table proposed splitting `model.py` into 8 files while its own
  §0 and acceptance criterion 6 said leave it untouched. The "leave it" position
  is correct; the phase is dropped.
- Qwen counted 15 versioned perf files across `invocation_v1–v8`. There are
  **13** (`invocation_v2–v8`, `scheduling_v2–v7`); `invocation_v1` does not
  exist.

## 1. How to execute this document

Read this section before starting. It exists because the failure mode this
scope is guarding against is **confident, unverified structural claims** — the
defect that produced the wrong remedy in §6.1.

1. **Evidence before remedy.** No finding may be recorded without the command
   that produced it and that command's output. A finding whose evidence is
   "the filenames look similar" is not a finding.
2. **Read the code, not the names.** Every claim that two things are the same
   must cite their signatures or bodies, not their paths.
3. **Findings and fixes are separate commits.** Audit phases produce a report.
   Remediation happens after the report is reviewed. Do not fix while auditing.
4. **Stop and report on contradiction.** If evidence contradicts this scope,
   write that down and stop that phase. This document is a hypothesis, not an
   instruction to make reality match it.
5. **Every commit follows §10.** `[subsystem] NFC — summary` for pure moves.
   The repo has a `commit-msg` hook; it is the machine-enforced half of the rule
   (`coding-principles.md`, "Human-Facing And Machine-Enforced").

## 2. Principles → audit lens

The four core principles map to the four audit phases. Quoted rules are from
`coding-principles.md` and are the authority for every judgement below.

| Principle | Phase | The question |
| --- | --- | --- |
| Centralize authority | C | Is there exactly one source of truth for each rule? |
| Modularize execution | A | Are execution surfaces narrow and replaceable? |
| Abstract for simplicity | D | Is each interface simpler than its implementation? |
| Orthogonality | E | Does one change force unrelated changes? |
| DRY-as-knowledge | B | Is duplicated code duplicated *knowledge*? |

The governing constraint, quoted because it inverts the obvious approach:

> **Line count is not the metric. Knowledge duplication is.**

A phase that reduces LOC without removing duplicated knowledge has not
succeeded. A phase that removes duplicated knowledge and adds LOC has.

## 3. Baseline measurements

Reproduce these before starting; they are the before-state.

```
206 python files in boltbeam/, 34,693 LOC
409 python files repo-wide (includes tests)
median file ~129 LOC
45 flat files directly in boltbeam/
37 files in boltbeam/search/
13 versioned files in boltbeam/perf/
68 cmd_* handlers in cli.py; parser wiring 1,000+ lines away
22 dataclasses in kernel_analysis/model.py, one dependency tree
```

The median file is ~129 lines. **This codebase is not bloated.** The defects
are specific and local; do not treat this scope as licence for a general rewrite.

---

# Phase A — Organization

## A.1 `cli.py` → `cli/` package

**Defect (measured):** a command's parser definition sits ~1,000 lines from its
handler. Handlers occupy lines 57–1157; `main()` and its ~600 lines of argparse
wiring start at 1165. Answering "what does this command accept and what does it
do" requires holding two distant regions in mind. This is a locality failure,
and it is why the file only grows.

The 68 handlers are **not** a defect. They are thin adapters — parse, call a
subsystem, write JSON. Per "Keep Public Surfaces Boring", thin glue is the
correct shape for a CLI layer.

**Target:**

```
boltbeam/cli/
  __init__.py      main(); registrar dispatch only
  _common.py       _write, _fail, _load_json, _profile, _add_common,
                   _parse_ctxs, _parse_route_flags, _run_manifest_defaults
  analysis.py      kernel-analysis commands
  search.py        candidate-set, search, campaign commands
  profile.py       import-*, collect-*, trace, enrich, compare-* commands
  roofline.py      roofline*, prefill-roofline*, theoretical
  workflow.py      autoscan, plan-*, route-manifest, closure, audit
```

Domain boundaries mirror the existing package layout, so no new taxonomy is
invented ("Consistency beats cleverness").

**Pattern** — each module exposes `register(sub)` that adds its parser *and*
sets `func`, with the handler directly beneath:

```python
def register(sub):
    p = sub.add_parser("prefill-roofline-ladder", help="...")
    p.add_argument("--ctxs", ...)
    p.set_defaults(func=cmd_prefill_roofline_ladder)

def cmd_prefill_roofline_ladder(args) -> int:
    ...
```

`__init__.py` becomes a loop over the five modules. The 600-line `main()`
disappears because its contents moved next to the code they describe.

**Why a package, not a rename:** `boltbeam.cli:main` stays byte-identical, so
`pyproject.toml` and all 27 test files that import it are untouched.

**Rejected:** split by technical role (`handlers.py`/`parsers.py`) — organizing
by file type is the swamp path. One file per command — 68 files destroys
discoverability. A decorator registry framework — an abstraction that saves a
for-loop is not one that "removes more complexity than it introduces". Migrating
to Click/Typer — orthogonal to the defect; separate scope, after this, tests
green first.

**Gate**, two checks, both required:

1. **Set equality.** The sorted list of subcommands before and after must be
   identical — nothing added, dropped, or renamed.
2. **Per-command help is byte-identical.** For every subcommand (66 top-level
   plus the 5 nested ones — `ledger add/report/import`, `cache status`,
   `diagnose compiler-pathology`), `boltbeam <cmd> --help` must be byte-for-byte
   the same as before. This is the check that actually matters: it catches a
   dropped argument, a changed default, a changed metavar, or a reworded help
   string, none of which the top-level listing alone would reveal. Capture
   per-command help before and after the split and diff; the diff must be empty.

Top-level listing *order* may change and is not part of the gate: argparse
dispatches by name, not position, so reordering the subcommand list is not a
behavioural change. (An earlier draft of this criterion required the top-level
listing byte-identical *including order* — that was unsatisfiable, because
domains interleave in the original file: e.g. the roofline commands appear at
four non-contiguous positions in `cli.py`'s original `main()`, and no ordering
of five whole-module `register()` calls can reproduce a non-contiguous
interleaving. Executing this scope surfaced the error; the criterion is fixed
here rather than carried forward wrong.)

## A.2 Flat root files → domain packages

**Defect:** 45 files (22% of the package) sit directly in `boltbeam/` with no
taxonomy. These are production modules, not scripts — `matched_control.py` is
585 LOC. You cannot predict where a module lives without `grep`. This is a
discoverability failure and it is the mechanism that produces the next god file.

**Target packages** (verify each membership before moving; the list below is a
hypothesis from filenames and must be confirmed by reading imports):

```
roofline/       roofline_plan, roofline_ceiling, roofline_trace,
                prefill_roofline, prefill_roofline_ladder
control/        matched_control, replay_ab
runtime/        amd_runtime_bridge, direct_kfd_bridge, kfd_observation_bridge,
                execution_bridge, process_isolated, gpu_health
trace/          hw_trace, timing, timing_compare, schedule_trace,
                trace_enrich, substrate_compare
memory/         mem_hierarchy, mem_sweep, lds_l2
quantization/   quant, quant_gemv
vocabulary/     vocab, vocab_capture, model_vocab
plan/           runner_plan, boundary, resolved_target
experiment/     reduce_source, reconciliation, observation_protocol,
                decode_counter_evidence, decode_decay, decode_resource_evidence,
                prefill_authority, prefill_role_trace
target/         targets, tinygrad_root
```

`analyze.py` stays at root — it is the top-level bundle emitter and an authority
point ("Centralize … routing rules and system policy").

**Risk:** low. Mechanical moves plus import updates; an import error surfaces
immediately.

## A.3 `search/` → sub-packages

**Defect:** 37 files spanning search-space emission, full-kernel search, MMQ
diagnosis, semantic/machine-research campaigns, epoch modelling, and evidence
joins. Six subsystems in one namespace.

**Target:** sub-packages `semantic/`, `full_kernel/`, `mmq/`, `epochs/`,
`joins/`, with `emit.py`, `spec.py`, `families.py`, `util.py` staying flat as
the package's core surface.

**Risk:** medium — `search/` is heavily cross-referenced. One sub-package per
commit, full suite green after each. Watch for cycles at the join modules.

## A.4 `kernel_analysis/model.py` — leave it

22 dataclasses is not the same defect. The dependency graph is a single tree
converging on `KernelEvidence` → `KernelAnalysis`; `OperandPathEvidence`
references 4 siblings and `KernelEvidence` references 11. There is no seam.

Splitting would scatter a schema that currently reads top-to-bottom in one pass,
convert zero imports into a web of cross-imports, and risk cycles at both
junctions — for no locality gain, because schema changes touch parent and child
together. **Do not split it.**

Considered and rejected as not worth doing alone: moving the 8 generic
validators to a sibling `_validate.py` (~90 lines, 6%).

---

# Phase B — Duplication audit

The most dangerous phase, because the naive remedy is worse than the defect:

> **DRY means knowledge, not lines.** Eliminate duplicated *knowledge* — a rule,
> a schema, a policy. Do not eliminate code that merely *looks* alike but
> encodes *different* concepts.

> **Duplication is cheaper than the wrong abstraction.** If you are tempted to
> merge similar-looking code, first prove it represents the *same* knowledge.

## B.1 Method

1. Find candidates mechanically — near-identical function bodies, repeated
   literal constants, repeated validation blocks, copy-pasted `main()`s.
2. For each candidate, **read both bodies and compare signatures.**
3. Classify into exactly one bucket:

| Bucket | Test | Action |
| --- | --- | --- |
| **Same knowledge** | same rule, same inputs, same meaning | Centralize |
| **Same shape, different knowledge** | different inputs or meaning | **Leave. Record why.** |
| **Same knowledge, divergent inputs** | same rule, ad-hoc input shapes | Canonicalise inputs first (§B.3) |

4. Apply Rule of Three: a pattern must repeat ~3 times *and be genuinely
   identical* before abstraction. Two occurrences is not a finding.

## B.2 Required evidence format

Every duplication finding must carry:

```
files+lines:   a.py:120-160, b.py:88-130
signatures:    <both, verbatim>
verdict:       same-knowledge | same-shape-different-knowledge | divergent-inputs
proof:         why the inputs/meaning are (or are not) the same
remedy:        centralize | leave | canonicalise-then-collapse
```

A finding without `signatures` and `proof` is rejected.

## B.3 The canonicalisation move

> **If similar things should be merged, fix the inputs first.** When
> near-duplicate code is blocked from merging only because its inputs differ,
> unify the inputs into one typed/canonical shape, *then* collapse the code into
> one parameterised path.

This is the correct remedy whenever the rule is shared but the input shapes are
ad-hoc. Do not force-merge over divergent inputs.

## B.4 Known hunting grounds

Confirm or refute each; do not assume:

- repeated JSON load/validate/write blocks across the 68 CLI handlers
- repeated manifest-defaults handling (`_run_manifest_defaults` exists — is it
  used everywhere it should be?)
- `sha256` computation appearing in `timing.py`, `perf/validation.py`,
  `perf/cycle_model.py`, `kfd_observation_bridge.py` — same knowledge or
  different?
- the trace importers (`import-llama-rocprof`, `import-tinygrad-rocprof`,
  `import-hw-trace`, `import-ncu`) — four converters to two schemas. Genuinely
  different parsers, or one parser with four column maps? This is the strongest
  table-driven candidate in the tree.

---

# Phase C — Centralization audit

> Keep one clear authority point for: environment and config access; schemas and
> durable data shape; integration boundaries to external services; routing rules
> and system policy; state definitions other modules depend on.

## C.1 What to enumerate

For each of the five authority categories, produce the list of every site that
reads or defines it, and identify the intended single authority:

| Category | Find | Expected authority |
| --- | --- | --- |
| Environment/config | every `os.environ` / `getenv` read | one config module |
| Schemas | every `schema` string literal and version constant | `kernel_analysis/model.py` and siblings |
| External boundaries | every subprocess/HTTP call to rocprof, ncu, tinygrad, llama | one adapter per provider |
| Routing/policy | every promotion/eligibility/threshold rule | `policy/`, `eval/thresholds` |
| Paths | every hardcoded path or path-joining convention | one path module |

## C.2 The specific smell to hunt

> one-off scripts that silently redefine canonical rules

`tools/` and `scripts/` are the likely locations. A script that re-implements a
threshold, a schema version, or a path convention that also exists in
`boltbeam/` is a centralization violation even if the code is short.

## C.3 Evidence format

```
rule:        what knowledge this is
authority:   where it should live (or "none exists")
sites:       every file:line that defines or hardcodes it
verdict:     centralized | duplicated | no-authority
remedy:      move to <authority>, import at each site
```

---

# Phase D — Abstraction audit

> Good abstraction: a small number of clear entry points; shared utilities
> instead of repeated low-level logic; stable interfaces around backends, paths,
> runtime state; implementation detail staying inside the owning module.
>
> Bad abstraction: vague wrappers that hide where authority actually lives;
> helper layers that duplicate underlying rules; naming that sounds generic but
> does not reduce real complexity.

## D.1 Test each abstraction

For every wrapper, helper, base class, and adapter, ask:

1. Is the interface **simpler** than the implementation? (Deep, not shallow.)
2. Does it **hide** implementation, or merely **forward** to it? A pass-through
   with no added invariant is a shallow module and should be inlined.
3. Does it **duplicate** a rule that lives elsewhere?
4. Does its name reduce complexity, or just sound generic?

## D.2 Ergonomics vs semantics

> Avoid helpers that: skip validation; hide ownership of state; swallow
> meaningful errors; make a policy decision look like a transport detail; create
> a second unofficial way to perform the same operation.

Specifically hunt for **a second unofficial way** to do something the canonical
API already does — most likely between the CLI handlers and the subsystem APIs
they call.

## D.3 Provider adapters

`kernel_analysis/provider_adapters.py` (412 LOC) is the designated replacement
boundary. Verify against "Design For Replacement":

- shared policy lives *above* the adapter
- backend-specific behaviour lives *inside* it
- capability differences are explicit, not implied
- the contract is testable without every real backend

An adapter that "quietly redefines shared policy" is an explicit anti-pattern.

---

# Phase E — Orthogonality audit

> Bad orthogonality: one module carrying multiple unrelated responsibilities;
> path, policy, state, and execution logic fused together; side effects that
> leak across the system; a refactor in one subsystem forcing incidental edits
> in many others.

## E.1 Method

1. For each module >250 LOC, list its responsibilities. More than one unrelated
   responsibility is a finding.
2. Check the specific fusion the principles name: **path + policy + state +
   execution in one file.**
3. Check the required separation: **runtime use and meta-development stay
   separate.** In this repo that is the boundary between analysing a model and
   analysing BoltBeam's own search process.
4. Measure entanglement empirically: for each subsystem, how many *other*
   subsystems import it? A module imported by everything is either a legitimate
   authority point (fine) or an entangled god module (finding). The distinction
   is whether it exports **knowledge** or **behaviour**.

## E.2 Evidence format

```
module:          path, LOC
responsibilities: <enumerated>
fused concerns:   path | policy | state | execution | none
imported by:      <count and list of subsystems>
verdict:          orthogonal | fused | authority-point
```

---

# Phase F — Asset reuse and pruning

## F.1 Reuse before writing

> shared utilities instead of repeated low-level logic

Before any remediation commit introduces a helper, prove no equivalent exists.
Search the tree first; extend the existing authoritative module rather than
adding a parallel one.

> **No-new-file rule.** A new experiment, variant, or case adds a *row to a
> table* or a *parameter* — not a new file or a copy-pasted function/`main()`.

## F.2 Obsolete probe pruning

> Avoid preserving obsolete probes after their verdict is recorded.

BoltBeam records verdicts durably in the ledger. A probe, fixture, or one-off
script whose verdict is already in the ledger is a candidate for deletion — the
ledger is the durable record, the script is not.

For each candidate: cite the ledger entry recording its verdict, then delete.
**No ledger entry, no deletion.** This is the one phase that destroys
information, so it carries the strictest evidence bar.

## F.3 Authored vs generated

> Authored vs generated is a hard boundary; only authored counts. Generated
> artifacts must be marked `@generated`, reproducible from a committed recipe,
> and excluded from the budget.

Audit: is every generated artifact in this tree marked and reproducible? An
artifact that cannot be regenerated is authored code hiding as generated, and
must be reclassified.

---

# 6. Contested item: the versioned `perf/` models

This is the one place the prior scopes produced a wrong remedy, and it is worth
stating fully because it is the clearest illustration of the principles.

## 6.1 What Qwen proposed, and why it is wrong

Qwen proposed collapsing `invocation_v2–v8` and `scheduling_v2–v7` into two
parameterised models, on the stated grounds that "each file defines the same
interface with incremental changes to the parameter set."

That premise is false. Read the signatures:

```python
fit_invocation_v3(interaction, shaped)
predict_invocation_v3(model, *, candidate_id, binary_sha256, base_uops, false_sites)

fit_invocation_v8(operand, scaffold)
predict_invocation_v8(model, direct, *, candidate_id, binary_sha256, gated)
```

Different inputs, different predict signatures. These consume **different
evidence**; they are not one model with different constants. Under
"DRY means knowledge, not lines", they are not duplicated knowledge, and under
"Duplication is cheaper than the wrong abstraction", merging them is the
error, not the fix.

The proposal also violates its own acceptance criterion: there are **13
dedicated test files**, one per version, and collapsing the versions necessarily
rewrites all of them — while criterion 2 required that no test change.

## 6.2 What the principles actually prescribe

The smell is real; the remedy is different.

> **No-new-file rule:** a new variant adds a *row to a table* or a *parameter* —
> not a new file.

So the *pattern* — one new file per model revision — is the anti-pattern, going
forward. Three actions, in order:

1. **Stop the bleeding.** There must be no `invocation_v9.py`. The next
   revision is a row, not a file. Enforce with a test asserting no new
   `_v{N}.py` appears in `perf/`.
2. **Canonicalise the inputs** (§B.3). The blocker to merging is that v3 takes
   `interaction/shaped` and v8 takes `operand/scaffold`. If those are the same
   evidence under different names, unify them into one typed shape — *then* the
   duplication becomes real duplication that can be correctly removed.
3. **Collapse only what canonicalisation proves identical.** Versions whose
   evidence genuinely differs stay separate. Frozen historical calibrations that
   must remain runnable move to `perf/archive/` with a lineage README; ones
   whose verdict is in the ledger are deleted under §F.2.

Step 2 is a research task, not a refactor, and must produce evidence before step
3 is attempted.

---

# 7. Acceptance criteria

Global:

1. `pyproject.toml` unchanged; `boltbeam.cli:main` resolves.
2. All existing tests pass **without modification**. A test that needs editing
   means behaviour changed and the work is wrong.
3. Two checks, both required (see the corrected A.1 gate above): the sorted
   subcommand set is identical before and after, and every one of the 71
   per-command `--help` outputs (66 top-level + 5 nested) is byte-identical.
   Top-level listing order may change; it is not part of this criterion,
   because domains interleave in the original file and no single-register-call
   arrangement can preserve that order (measured during execution — see A.1).
4. `model.py` untouched.
5. No file in `boltbeam/` is a flat orphan after A.2, except `analyze.py`.
6. No sub-package in `search/` exceeds 8 files.
7. Every command's parser and handler are in the same file, within 40 lines.
8. `cli/__init__.py` under 60 lines; no `cli/` module imports another except
   `_common`.
9. No new `_v{N}.py` in `perf/`, enforced by a test.
10. Every audit finding carries its evidence block (§B.2, §C.3, §E.2).
11. Every deletion in §F.2 cites the ledger entry recording its verdict.
12. Total LOC may **increase**. It is not a success metric.

Criterion 12 is deliberate and follows from "Line count is not the metric."

# 8. Risks and controls

| Risk | Control |
| --- | --- |
| Command silently dropped | `--help` byte-diff (criterion 3) |
| Wrong abstraction from look-alike code | §B.1 bucket test; signatures mandatory |
| Perf numerical output changes | §6.2 forbids collapsing before canonicalisation |
| Circular imports in `search/` | one sub-package per commit, suite green each |
| Import break on package move | suite after every move |
| Deleting knowledge that is not recorded elsewhere | §F.2 requires a ledger citation |
| Audit becomes an opinion essay | every finding needs an evidence block or is rejected |
| Scope creep into the other 407 files | median file is 129 LOC; no defect measured |

# 9. Sequence

| # | Phase | Risk | Depends on |
| ---: | --- | --- | --- |
| 1 | A.1 `cli.py` → package (pure rename first) | low | — |
| 2 | A.1 extract `_common`, then one commit per domain module | low | 1 |
| 3 | A.1 reduce `main()` to registrar dispatch | low | 2 |
| 4 | A.2 package flat root files, one package per commit | low | — (parallel with 1–3) |
| 5 | A.3 carve `search/`, one sub-package per commit | medium | 4 |
| 6 | B duplication audit (report only) | none | 1–5 |
| 7 | C centralization audit (report only) | none | 1–5 |
| 8 | D abstraction audit (report only) | none | 1–5 |
| 9 | E orthogonality audit (report only) | none | 1–5 |
| 10 | Review reports; select remediations | — | 6–9 |
| 11 | Remediation commits, one finding each | varies | 10 |
| 12 | F.2 pruning, ledger-cited deletions only | medium | 10 |
| 13 | §6.2 step 2: canonicalise perf inputs (research) | — | 10 |

Phases 6–9 change no code. They exist to produce a reviewable report *before*
anything is remediated, because the failure mode being guarded against is
confident remediation of a misdiagnosed defect.

# 10. Commit discipline

Per `coding-principles.md`, enforced by the repo's `commit-msg` hook.

Prefixes in active use, by frequency: `[schema]` `[search]` `[docs]` `[eval]`
`[profile]` `[cli]` `[test]` `[trace]` `[target]` `[policy]` `[bench]` `[repo]`
`[path]` `[collectors]`. Use the subsystem that owns the behaviour.

Pure moves are non-functional and must be marked:

```
[cli] NFC — move roofline commands into cli/roofline.py
```

Do not mix NFC refactors with behaviour changes. Nearly every commit in phases
1–5 is NFC; if one is not, it is in the wrong phase.

# 11. What this scope deliberately does not do

- It does not reorganize the other 407 files. Median file ~129 LOC, no defect
  measured.
- It does not split `model.py`.
- It does not collapse the versioned perf models (§6.2).
- It does not change CLI behaviour, flags, or output formats.
- It does not migrate off argparse.
- It does not add a dependency.
- It does not target a line count.
