# Maintaining BoltBeam

This page is for people who work on BoltBeam itself. A reader who wants to use it starts at the
[README](../README.md).

## Install for development

```bash
python -m pip install -e '.[dev]'
python3 tools/install_hooks.py     # the commit-message checker and the pre-commit sz.py check
boltbeam selfcheck
```

The trunk (`main`) carries no test suite. The suite lives on `dev`, which is where correctness is answered:

```bash
git checkout dev && pytest
```

`boltbeam selfcheck` is the one check that ships on `main`. The `dev` suite runs it too
(`tests/test_selfcheck.py`).

## Branches

`master` ⊂ `dev` ⊂ `exp`. They are split by content category, not by maturity.

- `main` holds the runnable product and the records. No test suite. No module without a live caller.
- `dev` adds the verification apparatus: the test suite and the campaign gates.
- `exp` holds active experimentation.

`tools/sync_branches.sh` carries the trunk outward. `dev` is hard-gated on the suite, so a trunk commit that
breaks the product cannot reach `exp`. `exp` is report-only, because it admits broken intermediate states by
design. Do not pipe `sync_branches.sh` into `head`: closing its stdout kills it before it merges.

[branch-flow.md](branch-flow.md) has the full layout. It also covers the one case that needs a person:
when the trunk prunes something the outer branches keep.

## The public mirror

[BoltBeam-Public](https://github.com/JulianAbeleda/BoltBeam-Public) is a snapshot of `main`. It differs from
`main` in one file: README.md, where the mirror adds a note on the line after the
`<!-- mirror-note: ... -->` marker. Keep that marker in the README.

The published articles link into the mirror by path. Never move or rename a file an article links to.
[README.md](README.md) in this folder lists those files.

## Scripts

| script | what it does |
|---|---|
| `sz.py` | the size budget and the duplicate-authority check (the `pre-commit` hook) |
| `tools/install_hooks.py` | installs the hooks: the commit-message checker and `pre-commit` |
| `tools/check_commit_msg.py` | enforces `[subsystem]` prefixes and `NFC` marking |
| `tools/sync_branches.sh` | carries the trunk outward to `dev` and `exp` |
| `tools/classify_bench.py` | classifies `bench/` artifacts as verdict or measurement |
| `tools/export_route_manifest_snapshot.py` | the only exporter for the tinygrad EXP snapshot |

That is the whole of `tools/` on the trunk: repo infrastructure and one exporter.

### Campaign gates live on `dev`

The same-run measurement gates are campaign tooling: `promote_routes.py`, `prepare_same_run.py`,
`measure_same_run.py`, `compare_same_run.py`, `prepare_quant_comparison.py`,
`validate_fused_route_promotion.py`, `account_prefill_roles.py` and their shared `same_run.py`.
[branch-flow.md](branch-flow.md) keeps that category off the trunk. They and their runbooks are on `dev`,
with their tests:

```bash
git checkout dev
python tools/promote_routes.py --model <model-id> \
  --evidence evidence.json --roles roles/ --run-id clean-1 --out report.json
```

They take the model, target, quant set and context ladder as arguments. A second model is a different
invocation, not a second copy of the script. Their verdicts are records, so those stay on `main`: see
`bench/` and
[task_workflow/output/model-agnostic-same-run-tools-20260731.md](task_workflow/output/model-agnostic-same-run-tools-20260731.md).

## The command list

`boltbeam --help` shows a core group first. The table is `CORE` in `boltbeam/cli/__init__.py`. Every other
command is listed after it under "Advanced and campaign commands". A command leaves the core group by
editing that table. It never stops working because of it: records and scripts call commands by name.

## Packaging

Only the `boltbeam` package ships (`[tool.setuptools.packages.find]` in `pyproject.toml`). The data files the
code reads next to its own modules (`boltbeam/data/`, `boltbeam/policy/assets/`) are package data, so a
plain `pip install .` carries them. `schemas/` is a repo folder. `selfcheck` checks required keys against it
when it is present, and checks the schema id alone when it is not.

## Route-manifest authority

`boltbeam/policy/assets/route_manifest.v1.json` and its SHA-256 sidecar are the only canonical
route-selection policy. `boltbeam.policy.route_manifest` validates the asset before exposing it. Refresh the
tinygrad EXP snapshot with the exporter. Do not copy the file:

```bash
python tools/export_route_manifest_snapshot.py \
  /path/to/tinygrad-arkey-exp/extra/llm_research/generated/boltbeam_route_manifest.v1.json
```

Missing, stale-schema, noncanonical or hash-mismatched input fails closed.

## Handing a model to an agent

```bash
boltbeam analyze /path/to/model.gguf --target amd_gfx1100 \
  --id qwen3-14b --out-dir outputs/qwen3-14b-analysis
```

The bundle is audit-first. It emits the profile, search and policy files, then states which provider commands
to run next. If a generated candidate fails because a topology knob is missing, the expected outcome is
`search-space-incomplete`. Expose the knob. Do not hand-write a one-off route.

## Size

`sz.py` at the repo root holds the size budget and the duplicate-authority check. The `pre-commit` hook runs
it, and `tools/sync_branches.sh` runs it after each merge (hard on `dev`, report-only on `exp`).

```bash
python3 sz.py              # the table, the total and the cap, then the authority check
SZ_FILES=1 python3 sz.py   # every file
```

The budget counts lines that hold code in `boltbeam/` and `tui/`. Blank lines, comments and docstrings do not
count. Tests do not count. `tools/` is reported but not budgeted. The cap is written in the header of
`sz.py`, with the reason for the number.

A commit over the cap is refused. Raising the cap is a decision, not a fix. The code has to earn the lines,
and the commit that raises the cap shows the table in its message.

Line count is not the main metric. Duplicated knowledge is. `sz.py` keeps a short list of authorities, each
with the one module that owns it: the ggml type table, the block layouts, the role vocabulary, the measured
chip peaks, the dispatch-floor and noisy constants, and the correctness tolerance. A second definition of one
of them anywhere else in `boltbeam/` or `tools/` fails the check, with its file and line. Only definitions
count: assignments, dict, tuple, list and set literals, and numeric tolerance arguments. A mention in a
comment or a doc does not.

The duplicates that existed when the check landed are listed in `ALLOWED` in `sz.py`, each with its review
finding. Review landing 6 removes them. Do not add to that list to get a commit through: import the owner
instead. A new authority is one entry in `AUTHORITIES` and one detector.

## Design principles

Measured, not asserted. Every claim in an artifact cites the evidence that produced it. A gate that cannot
verify its input fails closed. It does not guess.

Provider-neutral. Backend differences live inside adapters. Shared policy lives above them. A capability a
backend lacks is stated, never faked.

The house rules are in [coding-principles.md](coding-principles.md) and
[commit-discipline.md](commit-discipline.md).

## Further reading

- [branch-flow.md](branch-flow.md): branch layout and sync procedure
- [task_workflow/](task_workflow/): how delegated work is specified and recorded
- [exp-metal-mr0-mr13-runbook.md](exp-metal-mr0-mr13-runbook.md): the Apple Metal MR0 to MR13 workflow
- [INTEGRATION_TINYGRAD.md](../INTEGRATION_TINYGRAD.md): the tinygrad integration boundary
