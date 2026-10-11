# Handoff: fix main's Go tests, then review cleanup items 3–7 (2026-10-11)

For an agent picking up BoltBeam cleanup. Nothing here needs a GPU. Do not touch any remote machine.

## Rules (read first)

- Read docs/coding-principles.md, docs/branch-flow.md, docs/commit-discipline.md and docs/maintaining.md.
- Branches:
  - main is the runnable product, with no Python tests.
  - dev is main plus tests.
  - exp is dev plus experiments.
  - Go tests live on main. Code goes on main and tests go on dev.
  - Promote with tools/sync_branches.sh.
- Commit prefixes: [cli] [profile] [search] [collectors] [eval] [target] [test] [docs] [repo].
- Never use --no-verify. The pre-commit hook runs sz.py (a line cap, plus a duplicate-authority check).
  - Do not raise the cap or weaken the hook to get past it.
  - If the cap genuinely must rise, do it in its own [repo] commit, with the per-directory table and the reason.
- Everything is generic: no model names, model shapes or chip names in code.
- Work in worktrees, not in the main checkout:
  `git worktree add -b fix1 ../BoltBeam-fix1 origin/main` and `git worktree add -b fix1-dev ../BoltBeam-fix1-dev origin/dev`.
- Tests:
  - Python runs on dev: a venv with `pip install -e '.[dev]'`, then `python -m pytest -q tests/`.
  - Go: `cd tui && BOLTBEAM_PYTHON=python3.12 go vet ./... && go test ./...`.
- Landing uses a guarded chain:
  1. First check `git rev-parse main origin/main dev origin/dev`: main must equal origin/main, and dev must equal origin/dev.
  2. Then run ONE `&&` chain with no pipes:
     `git merge --ff-only fix1 && git update-ref refs/heads/dev fix1-dev <old dev sha> && BOLTBEAM_PYTHON=python3.12 tools/sync_branches.sh && git push origin main dev exp`
  3. If it aborts, fetch, rebase and retry.
  4. Then publish the mirror from a worktree of origin/dev with `tools/publish_mirror.sh`. Never weaken its leak grep.
  5. Remove your worktrees afterwards.

## Task 1: main's Go tests fail. Fix them, and gate on them.

On origin/main (52bab64), `cd tui && BOLTBEAM_PYTHON=python3.12 go test ./...` fails:
- `TestLiveSeamMatchesPinnedContract` (tui/internal/seam). It fails at least since d95de91.
- `TestIsolatedLossSaysHowItWasMeasured` and `TestBatchResults` (tui/internal/ui). They broke between d95de91 and 9109425.

They pass on dev, because the fixtures and runs they read are dev-only. The landing flow only ever ran the
Go tests in dev worktrees, so nothing caught this.

Do:
1. Find what each test reads that main lacks, and make main self-sufficient.
   - Preferred: move the minimal fixture files the Go tests need onto main, next to the Go tests. This is allowed, since Go tests live on main.
   - Otherwise: make the test skip, with a stated reason, when the fixture is dev-only.
   - Never delete a test.
2. Add "Go vet + Go tests on main" to the landing gate: run them on main before promoting, in tools/sync_branches.sh
   or wherever the suites run. Document it in docs/maintaining.md.
3. Prove it:
   - A clean checkout of main passes `go test ./...`.
   - The gate fails on a deliberately broken Go test. Do this in a scratch branch, and do not land it.

## Task 2: review cleanup, items 3–7

The review file is outside the repo, at ~/env/boltbeam-runs/.work/review-20261010.md. The finding ids below
(A*, C*, R*, E*, P*) refer to it.
- Verify each finding against current main before fixing it. Some may be stale; say "stale" when so.
- Land each item separately, in this order.

3. **Verdict logic** (boltbeam/search/role_compare.py):
   - A1, A4: a fusion must never pass on an in-model-only comparison when its unfused nodes did not
     reproduce. If the like-for-like check cannot run, the verdict is "not decided", not a pass.
   - A2: compare a plan timed alone against the like-for-like `model_kernel_us`, timed the same way,
     within the chip's band.
   - A3, A7, A8: never fall back to the provider's number. Never print a provider number as "faster".
     The provider's time is for ordering only.
   - A5, A6: one winner rule, in one place.
   - Add tests on dev for each, including a lying provider and an unreproducible fusion.
4. **Tie-out labels:** A13–A17 and A21, plus one noisy constant (C10) and one peak source (C11). Each must
   have exactly one owner module.
5. **One "installed kernel" authority:** C1, C2 and A9–A11.
   - The row BoltBeam calls "the model's own kernel" must be the kernel the model actually runs, from one source.
   - The block-quant launch geometry must not be re-implemented in BoltBeam with a different SM count from the fork's.
   - This may touch the tinygrad fork only as read-only metadata. Do NOT develop or optimise the fork.
     If a fork change is unavoidable, stop and describe it instead.
6. **One type and layout table:** C3, plus the other duplicated tables the review lists.
   - Keep one ggml type table and one block-layout table, each in its owner module (see the authorities list in sz.py).
   - Replace every other definition with an import.
   - Then remove the matching entries from sz.py's ALLOWED list. The check must pass with an empty allowlist
     for those authorities.
7. **Docs and records:** R1–R8, E9–E13, P1–P8.
   - R1: docs/kernel-timer.md claims timed-alone "lands within 4% of the token", but its own data gives
     0.85 (5090) and 0.82 (M3). Fix the claim.
   - The shape-pins route counts are stale: 27B 5 → 4, and 4B 3 → 2, per the fork today.
   - Fix text only where it is wrong, and cite the evidence.

## Report back

For each task and item, report:
- the findings fixed, or marked stale or not fixed, with the reason
- the tests added
- SHAs for main, dev, exp and the public mirror
- the pytest and go test counts
- the sz.py total
- a table of the worktrees added and removed
