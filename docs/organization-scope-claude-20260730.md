# BoltBeam organization scope

Status: proposed
Scoped: 2026-07-30

Scope covers the two largest modules in the tree, `boltbeam/cli.py` (1,773 lines)
and `boltbeam/kernel_analysis/model.py` (1,492 lines). The recommendation differs
for each, and one of them is "do nothing".

## 0. Verdict first

| Module | Lines | Verdict |
| --- | ---: | --- |
| `boltbeam/cli.py` | 1,773 | **Split**, by domain, into a `cli/` package |
| `boltbeam/kernel_analysis/model.py` | 1,492 | **Leave alone.** Size is not the defect |

Size triggered this review; size is not the argument. The case for touching
`cli.py` rests on a specific measured defect, and the case for leaving
`model.py` rests on its dependency graph.

## 1. Measurements

409 Python files, 52,715 lines. Median file ~129 lines. These two files are
outliers, not evidence of a general problem — the rest of the tree does not need
reorganizing and this scope does not propose it.

`boltbeam/cli.py`:

- 68 `cmd_*` handlers, occupying roughly lines 57–1157
- 10 `_`-prefixed helpers, interleaved with the handlers
- `main()` at line 1165, containing ~600 lines of `argparse` wiring
- entry point is `boltbeam = "boltbeam.cli:main"` (`pyproject.toml`)
- 27 test files import `boltbeam.cli` or call `cli.main`
- nothing else in `boltbeam/` imports `cli` — it is a leaf

`boltbeam/kernel_analysis/model.py`:

- 22 frozen dataclasses, 8 module-level validators
- dependency graph is a single tree converging on
  `KernelEvidence` → `KernelAnalysis`
- no independent clusters (see §3)

## 2. `cli.py` — the actual defect

The problem is not that the file is long. It is that **a command's parser
definition sits roughly 1,000 lines away from its handler.** To answer "what
does `prefill-roofline-ladder` accept and what does it do", a reader holds two
distant regions of one file in their head at once.

That is a locality failure, and locality is the cheapest of the organization
goals to fix. It is also why the file grows monotonically: adding a command
means editing two far-apart places, so nobody ever tidies while passing through.

The 68 handlers are *not* a defect. They are thin adapters — parse args, call a
subsystem, write JSON. Thin glue is the correct shape for a CLI layer, and
"deep module" reasoning does not apply to adapters whose whole job is
translation.

### 2.1 Proposed structure

`boltbeam/cli.py` becomes `boltbeam/cli/`, which keeps the entry point
`boltbeam.cli:main` byte-identical and every one of the 27 test imports working
unchanged.

```
boltbeam/cli/
  __init__.py      main(); builds the parser by calling each registrar
  _common.py       _write, _fail, _load_json, _profile, _add_common,
                   _parse_ctxs, _parse_route_flags, _run_manifest_defaults
  analysis.py      kernel-analysis commands
  search.py        candidate-set, search, campaign commands
  profile.py       import-*, collect-*, trace, enrich, compare-* commands
  roofline.py      roofline, roofline-plan, prefill-roofline*, theoretical
  workflow.py      autoscan, plan-*, route-manifest, closure, audit
```

Domain boundaries mirror the existing package layout (`search/`,
`kernel_analysis/`, `profiler/`, `perf/`, `workflow/`), so a reader who knows
where a subsystem lives already knows where its commands live. No new taxonomy
is invented.

### 2.2 The registrar pattern

Each module exposes one function:

```python
def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prefill-roofline-ladder", help="...")
    p.add_argument("--ctxs", ...)
    p.set_defaults(func=cmd_prefill_roofline_ladder)
```

with `cmd_prefill_roofline_ladder` defined directly beneath it. `__init__.py`
shrinks to roughly:

```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="boltbeam", ...)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for module in (analysis, search, profile, roofline, workflow):
        module.register(sub)
    args = ap.parse_args(argv)
    return args.func(args)
```

This is the change that matters: **parser and handler become adjacent.** The
600-line `main()` disappears not because it was split but because its contents
moved next to the code they describe.

### 2.3 Explicitly rejected alternatives

**Split by technical role** (`handlers.py`, `parsers.py`, `validators.py`).
Organizing by file type is the fastest path to a swamp: every change touches
three files and none of them is about anything. Rejected.

**One file per command** (68 files). Discoverability collapses — you must
memorize a map — and the shared helpers become an import tangle. Rejected.

**A plugin/registry framework with decorators.** Adds an indirection layer to
save a for-loop over five modules. An abstraction must remove more complexity
than it introduces; this one does not. Rejected.

**Click or Typer instead of argparse.** A real improvement in ergonomics, and
entirely orthogonal to the defect. Doing both at once would make the diff
unreviewable and put a working CLI at risk for a taste preference. If it is
wanted, it is a separate scope, after this one, with tests already green.

## 3. `model.py` — leave it alone

22 dataclasses in one file looks like the same problem. It is not. The
dependency graph:

```
KernelHash ─┐
            ├→ CandidateModel ─┐
OperandTransport ─┘             │
KernelHash → IdentityModel ─────┤
WorkloadModel ──────────────────┤
PipelineStage → KernelStages ───┤
Measurement ────────────────────┼→ KernelEvidence → KernelAnalysis
CorrectnessMetrics ─────────────┤
InstructionSite → OperandStaticEvidence ─┐
ServingTierObservation → OperandDynamicEvidence ─┼→ OperandPathEvidence ─┤
OperandClassification ───────────────────┘                              │
ResourceSummary, StructureSummary, HealthSummary, FactBlocker ──────────┘
```

This is one artifact schema, not several. Every type exists to be reachable from
`KernelAnalysis`. There is no seam along which to cut that would not leave both
halves referring to each other.

Splitting it would:

- scatter a schema that is currently readable top to bottom in one pass;
- convert zero imports into a web of cross-imports between sibling modules;
- create a real risk of circular imports at the `OperandPathEvidence` and
  `KernelEvidence` junctions, which reference four and eleven siblings;
- deliver no locality benefit, because a schema change usually touches the
  parent and child types together anyway.

The measured cost of the current arrangement is one long file that a reader
reads once. The measured cost of splitting is permanent import churn. Do not
split it.

If `model.py` is ever genuinely painful, the honest fix is different: the 8
validators (`_validate_id`, `_normalize_sources`, `_json_tuple`, …) are generic
and could move to a sibling `_validate.py`, taking ~90 lines with them and
leaving the schema itself intact. That is a 6% reduction and is not worth doing
on its own — it is listed only so a future reader knows it was considered.

## 4. Acceptance criteria

1. `boltbeam.cli:main` still resolves; `pyproject.toml` is unchanged.
2. All 27 test files importing `boltbeam.cli` pass without modification. If any
   test needs editing, the refactor changed behavior and is wrong.
3. `boltbeam --help` output is byte-identical before and after, including
   command ordering. Captured to a file and diffed as a gate.
4. Every command's parser definition and its handler are in the same file,
   within 40 lines of each other.
5. `__init__.py` is under 60 lines.
6. No module in `cli/` imports another except `_common`.
7. `model.py` is untouched.

Criterion 3 is the important one: it makes the refactor provably behavior-free,
which is what allows it to land without a review of all 68 commands.

## 5. Risks and controls

| Risk | Control |
| --- | --- |
| Command silently dropped during the move | Criterion 3 — `--help` diff catches a missing subparser |
| Argument default changed by hand-copying | Move code verbatim; no edits in the same commit as the move |
| Circular import between cli modules | Criterion 6 — only `_common` is shared, and it imports nothing from siblings |
| Reviewer cannot verify a 1,700-line move | One commit per domain module; each is a pure move with a green suite |
| Refactor competes with in-flight work | The file is a leaf; nothing in `boltbeam/` imports it |

## 6. Sequence

1. `refactor: convert cli.py to a package` — pure rename, `cli.py` →
   `cli/__init__.py`, no other change. Suite green, `--help` identical.
2. `refactor: extract cli helpers to _common` — move the 10 helpers.
3. One commit per domain module, each moving its handlers *and* their parser
   definitions together.
4. `refactor: reduce main() to registrar dispatch` — the payoff commit.

Each step is independently revertable and independently green. If the effort is
abandoned after step 2, the tree is still in a better state than it started.

## 7. What this scope deliberately does not do

- It does not reorganize the other 407 files. The median file is ~129 lines and
  no defect has been measured in them.
- It does not change any CLI behavior, flag, or output format.
- It does not introduce a new dependency.
- It does not split `model.py`.
