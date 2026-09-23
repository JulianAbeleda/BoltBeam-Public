# BoltBeam tinygrad integration consolidation scope

Date: 2026-07-27

## Objective

Consolidate the useful tinygrad decode observation and analysis work from six BoltBeam research worktrees onto current
BoltBeam `main`, validate the combined capability, push it, then archive and remove the superseded worktrees and local
branches.

## Current state

- `main` is clean at `3952596c9a0559164bd677e977d465484ac607d2`.
- Five research branches and one detached worktree are not merged and have no remote upstream.
- `codex/boltbeam-decode-decay` has one uncommitted llama rocprof collector change.
- The research branches are ten current-main commits behind and overlap heavily, so branch-level merging is prohibited.

## Promotion set

Apply these capabilities in dependency order:

1. Tinygrad prefill timing adapter and fixtures.
2. Static AMD decode resource evidence adapter.
3. Fail-closed KFD observation bridge.
4. Tinygrad decode authority adapter and decay analysis.
5. Tinygrad evidence-package documentation.
6. Decode generation-control validation.
7. Observation protocol validator.
8. Fail-closed direct KFD launch seam.
9. Decode PMC/graph/HSACO evidence contracts.
10. Llama rocprof decode workload command construction and naming.

The first eight code capabilities plus documentation are owned by `boltbeam/tinygrad-integration`. The counter evidence
contract is the only patch unique to `codex/boltbeam-decode-decay`. The collector change must first be committed in its
dirty worktree so it cannot be lost during cleanup.

## Explicitly archive-only

- `boltbeam-port-resources`: exact subset of the integration branch.
- `port-timing-routes`: exact subset of the integration branch.
- Detached KFD bridge tip: exact subset of the integration branch.
- `boltbeam-port-scope`: historical planning documents superseded by the integrated evidence documentation and this
  consolidation scope.

## Integration rules

- Cherry-pick capability commits onto current `main`; do not merge old branch heads.
- Preserve current-main behavior when resolving conflicts.
- Do not add runtime dependencies on the tinygrad checkout.
- Adapters consume artifacts; they do not execute or mutate tinygrad production inference.
- Counter evidence remains fail-closed: unavailable or unattributed counters cannot become measured conclusions.
- Decode collection must use llama-bench depth mode and keep prefill command behavior unchanged.

## Validation

Run the focused tests covering:

- Tinygrad prefill and decode adapters.
- Decode resource and counter evidence.
- KFD observation and direct bridge protocols.
- Observation protocol validation.
- Llama rocprof collector command construction.

Then run the full BoltBeam test suite if the focused set passes. No GPU execution is required for this consolidation.

## Cleanup protocol

1. Commit and push consolidated `main`.
2. Create one archive branch whose parents retain every research branch and detached worktree tip.
3. Push and verify archive reachability.
4. Remove every non-main BoltBeam worktree.
5. Delete archived local research branches.
6. Move this scope to `docs/task_workflow/output/` and append the exact promotion, test, archive, and deletion ledger.

## Stop conditions

- An overlapping commit cannot be reconciled without changing current-main semantics.
- Focused tests fail after integration.
- Archive push or tip-reachability verification fails.
- Unexpected dirty state appears outside the known llama collector file or this scope document.

# Execution result

## Verdict

Consolidation and cleanup completed. The useful tinygrad integration capabilities are on BoltBeam main, the complete
focused and full CPU test suites pass, and every superseded research tip remains recoverable from a pushed archive ref.

## Main promotion

- Consolidated main range: `67d1093` through `6535b6a`.
- Pushed main tip before this ledger: `6535b6a`.
- Promoted capabilities:
  - tinygrad prefill timing adapter;
  - static AMD decode resource evidence adapter;
  - fail-closed KFD observation bridge;
  - tinygrad decode authority adapter and depth-decay analysis;
  - generation-control validation;
  - observation protocol validator;
  - fail-closed direct KFD launch seam;
  - decode counter/graph/HSACO evidence contracts;
  - llama rocprof decode workload collection.
- No old branch head was merged wholesale; selected commits were applied onto current main in dependency order.
- No integration conflict occurred.

## Validation

- Focused adapter/counter/bridge/collector tests: 36 passed.
- Full BoltBeam suite: 1037 tests passed plus 48 subtests.
- Runtime: 3.86 seconds for the full suite.
- GPU execution was neither required nor performed.
- The first test invocation used `.venv/bin/python`, which lacks pytest and therefore executed no tests; the system
  Python authority produced the passing results above.

## Archive authority

- Branch: `archive/tinygrad-integration-worktrees-20260727`.
- Commit: `2b78518034c63ef5f3f5ef159fa2b0369d6309b8`.
- Remote: `origin/archive/tinygrad-integration-worktrees-20260727`.
- Unique tips checked: 7, covering main, all five research branches, and the detached KFD worktree tip.
- Every captured tip passed ancestor verification and the remote archive SHA matched the local SHA.
- Recovery manifest: `docs/archive/tinygrad-integration-worktrees-20260727.md` on the archive branch.

## Preserved dirty state

The only dirty source was `boltbeam/collectors/llama_rocprof.py` in `codex/boltbeam-decode-decay`. It was committed as
`1fdc839` before promotion and archive construction, then promoted to main as `6535b6a`.

## Removed

- Six non-main BoltBeam worktrees.
- Five local research branches.
- Worktrees removed:
  - `/home/ubuntu/worktrees/boltbeam-decode-decay`;
  - `/home/ubuntu/worktrees/boltbeam-kfd-bridge`;
  - `/home/ubuntu/worktrees/boltbeam-port-resources`;
  - `/home/ubuntu/worktrees/boltbeam-port-scope`;
  - `/home/ubuntu/worktrees/boltbeam-port-timing-routes`;
  - `/home/ubuntu/worktrees/boltbeam-tinygrad-integration`.

## Final repository shape

- Production worktree: `/home/ubuntu/BoltBeam` on `main`.
- Local branches: `main` and `archive/tinygrad-integration-worktrees-20260727`.
- Research history is remote-recoverable; the archive branch must not be merged into main because its additional parents
  are reachability anchors, not production integration.
