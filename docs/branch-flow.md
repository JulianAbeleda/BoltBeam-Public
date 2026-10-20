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

## Correcting accidental publication

Preserve active work on `exp`, then remove unqualified work and excluded content
from the current `main` tree with explicit corrective commits. Verify the remote
branch after pushing. This repairs the current tree; it does not erase earlier
published history. Do not rewrite shared history without explicit authorization.
