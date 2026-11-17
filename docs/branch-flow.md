# Branch workflow: exp → dev → main

Develop active work on `exp`, qualify it on `dev`, and promote only finished,
qualified product changes to `main`. Commit and push coherent increments on
`exp` so progress is visible. An in-progress commit is not a release.

## Content boundaries

| Branch | Admitted content |
| --- | --- |
| `main` | Finished runnable product and concise, reviewed product documentation |
| `dev` | Product plus tests, qualification tooling, and durable development records |
| `exp` | Active implementation, experiments, probes, and unfinished audit evidence |

Tests, measurement scaffolding, research archives, and unfinished investigation
records do not belong on `main`, even when their individual checks pass.
The branch called `master` in older documentation is `main` in this checkout.

## Promotion

1. Implement, review, commit, and push increments on `exp` with the owning
   subsystem prefix from `docs/commit-discipline.md`.
2. Select a complete change for qualification on `dev`. Include its tests and
   necessary qualification apparatus. Resolve failures and complete the relevant
   end-to-end review, including Run and Emit when affected. Size and duplicate
   authority gates must pass; targeted tests alone do not establish readiness.
3. Promote the qualified product subset to `main`. Keep tests and investigation
   apparatus on the outer branches. Review the actual staged diff before
   committing; do not merge the entire experimental tree inward.
4. Push each completed stage and report its commit and branch. Unfinished work
   stays on `exp`; a failed gate is reported, not waived to complete promotion.

Evidence intended for publication must be reviewed for machine home paths and
hostnames, including screenshots. Keep raw machine-specific captures local;
publish relative artifact references and hashes where provenance is needed.
Do not substitute invented data when redacting presentation details.

## Synchronization

`tools/sync_branches.sh` merges product changes outward from `main` into `dev`
and `exp`, with qualification gates. This maintains shared product content;
it is not authorization to originate unfinished changes on `main` and is not
an inward promotion command.

Before an outward merge, use `--dry-run` and inspect removals. A product prune
may delete apparatus that an outer branch should retain. Restore that apparatus
and its constraining tests on the appropriate outer branch, then validate the
result. Do not restore excluded apparatus onto `main`.

### What a sync tests

A test can only fail because something it depends on changed, and a pass is a fact about an exact tree. So after
each merge `tools/impacted_tests.py` decides what to run on the merged tree:

- **SKIP**: this tree (`git rev-parse HEAD^{tree}`) already passed here: same interpreter and installed packages,
  and the same machine state that decides skips (`nvidia-smi -L`, the engine builds file, `go version`, the tools the
  tests look up on PATH, `BOLTBEAM_*`). Kept in `~/.cache/boltbeam/suite-pass.json` (`BOLTBEAM_SUITE_CACHE` moves it).
- **NONE / SELECT**: the paths changed since the nearest ancestor tree that passed, each mapped to the tests that
  can observe it, read from the tree itself. Followed transitively: Python imports (including bare sibling names put
  on `sys.path` at run time), modules loaded by a name written in code or data, a `.py` run by path, and data files
  named exactly. Followed one hop (the reader and what imports it directly): source read as text, names with holes
  (`f"{stem}.json"`), and directory walks, where a walk whose root is not `tmp_path` counts as reading every file its
  pattern matches. A doc only a walk can see runs those walkers; `tui/` adds the TUI's `go test ./...`.
- **FULL**: anything not mapped: a conftest, what it imports or names (a `PYTHONPATH` directory it sets up),
  `pyproject.toml`, `tests/requires.py`, `sz.py`, `tools/`, an unknown directory, a data file or module that nothing
  names, or no passed ancestor tree. Unknown is not pass. Over-selection is the price: a miss is not acceptable.

The gate keeps its per-branch meaning (dev hard, exp report-only, `sz.py` always), and a green run records the tree.
The selection reads the code, not its execution: a name assembled at run time from pieces with no literal part is not
seen. That is why a release candidate runs the whole suite (`tools/sync_branches.sh --full`,
[release-runbook.md](release-runbook.md)). Run `python3 tools/impacted_tests.py --base <rev>` to see what a change
selects and why.

## Correcting accidental publication

Preserve active work on `exp`, then remove unqualified work and excluded content
from the current `main` tree with explicit corrective commits. Verify the remote
branch after pushing. This repairs the current tree; it does not erase earlier
published history. Do not rewrite shared history without explicit authorization.
