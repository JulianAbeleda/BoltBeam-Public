#!/usr/bin/env bash
# Propagate the trunk outward: main -> dev -> exp.
#
# The branches are supersets by content category (docs/branch-flow.md): dev is
# main plus the apparatus, exp is dev plus experimentation. So the trunk must
# flow OUTWARD or the outer branches stop being supersets and become forks —
# which is what tinygrad-arkey's dev sitting 88 commits behind master means.
#
# The one case that needs a human: when the trunk PRUNES something the outer
# branches retain. The merge will delete it there too, which is the opposite of
# what the layout is for. This script does not guess — it merges, then reports
# exactly what each merge deleted, and stops so the deletions can be restored as
# an apparatus commit if they should be.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
trunk="${TRUNK_BRANCH:-main}"
dry_run=0
full=0  # --full: run the whole suite on each merged tree, whatever the selector says (a release candidate does)
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --full) full=1 ;;
    *) echo "usage: $0 [--dry-run] [--full]" >&2; exit 64 ;;
  esac
done
# sz.py and the suite run under $BOLTBEAM_PYTHON, the interpreter the TUI and its Go seam test
# also use; it needs the dev extras (pip install -e '.[dev]').
py="${BOLTBEAM_PYTHON:-python3}"

[[ -z "$(git status --porcelain)" ]] || { echo "working tree is dirty; commit or stash first" >&2; exit 1; }
start_branch="$(git rev-parse --abbrev-ref HEAD)"
restore() { cd "$repo_root"; git checkout -q "$start_branch"; }
trap restore EXIT

# git refuses to check out a branch a second worktree already holds, so a branch that has
# its own worktree is worked on there; any other branch is checked out here, as before.
worktree_of() {
  git worktree list --porcelain | awk -v ref="refs/heads/$1" '/^worktree /{w=substr($0,10)} $0=="branch " ref {print w}'
}
enter() {
  local dir; dir="$(worktree_of "$1")"
  if [[ -n "$dir" ]]; then cd "$dir"; else cd "$repo_root"; git checkout -q "$1"; fi
  [[ -z "$(git status --porcelain)" ]] || { echo "$1 in $PWD is dirty; commit or stash first" >&2; exit 1; }
}

# Files the source branch deletes that the target currently has. These are the
# ones a merge would silently remove from a branch whose job is to keep them.
would_delete() {
  local from="$1" into="$2"
  git diff --diff-filter=D --name-only "$(git merge-base "$into" "$from")" "$from" 2>/dev/null | sort
}

sync_one() {
  local from="$1" into="$2"
  echo "=== $from -> $into ==="
  local deletes; deletes="$(would_delete "$from" "$into")"
  if [[ -n "$deletes" ]]; then
    echo "  this merge DELETES from $into:"
    sed 's/^/    /' <<<"$deletes"
    echo "  if $into is meant to retain these, restore them after the merge and commit"
    echo "  as apparatus:  git checkout <pre-prune-sha> -- <paths>"
  fi
  if (( dry_run )); then echo "  (dry run, not merging)"; return; fi
  enter "$into"
  git merge -q --no-edit "$from" || { echo "  MERGE CONFLICT in $into — resolve, then rerun" >&2; exit 2; }
  # The suite gate is per-branch, because the branches admit different content.
  #   main: Go vet and tests pass before either outward merge (below).
  #   dev:  hard gate. dev is where "is the product correct" is answerable.
  #   exp:  report only. exp admits broken intermediate states by design; blocking on a
  #         red suite there would gate experimentation on a rule exp does not accept.
  # the size budget and the duplicate-authority check (sz.py): hard on dev, report-only on exp, like the suite
  if ! "$py" sz.py >/dev/null 2>&1; then
    if [[ "$into" == exp ]]; then echo "  sz.py FAILS on exp — reported only (run python3 sz.py there)"
    else echo "  sz.py FAILS on $into after the merge: run python3 sz.py there." >&2; exit 4; fi
  fi
  # Test only what this merge can affect (tools/impacted_tests.py; docs/branch-flow.md): a tree that already passed on
  # this interpreter is skipped, else the tests that can observe what changed since the last tree that passed. Anything
  # the selector does not map, or a selector that fails, runs the whole suite.
  local sel verdict go_too tests=() rc=0
  if [[ -f tools/impacted_tests.py ]] && sel="$("$py" tools/impacted_tests.py $( (( full )) && echo --full))"; then
    grep -E '^(base|changed|reason) ' <<<"$sel" | sed 's/^/  /' || true
    verdict="$(awk '$1=="verdict"{print $2}' <<<"$sel")"
    go_too="$(awk '$1=="go"{print $2}' <<<"$sel")"
    mapfile -t tests < <(awk '$1=="test"{print $2}' <<<"$sel")
    case "$verdict" in SKIP|NONE|SELECT) ;; *) verdict=FULL; go_too=yes ;; esac  # no verdict: unknown is not pass
    [[ "$verdict" != SELECT || ${#tests[@]} -gt 0 ]] || { verdict=FULL; go_too=yes; }
    go_too="${go_too:-yes}"
  else
    echo "  no selector result on $into: the whole suite runs"; verdict=FULL; go_too=yes
  fi
  case "$verdict" in
    SKIP|NONE) echo "  pytest: nothing to run ($verdict)" ;;
    SELECT) echo "  pytest: ${#tests[@]} test files"; "$py" -m pytest -q "${tests[@]}" >/dev/null 2>&1 || rc=$? ;;
    *) echo "  pytest: the whole suite"; "$py" -m pytest -q >/dev/null 2>&1 || rc=$? ;;
  esac
  if [[ "$go_too" == yes && "$rc" == 0 ]]; then
    (cd tui && go test -count=1 ./... >/dev/null 2>&1) || { echo "  go test ./... FAILS in tui on $into"; rc=1; }
  fi
  if [[ "$rc" == 0 && "$verdict" != SKIP ]]; then
    "$py" tools/impacted_tests.py --record --how "$verdict" | sed 's/^/  /' || echo "  (could not record the pass; the next sync tests this tree again)"
  fi
  case "$rc" in
    0) echo "  merged, tests green ($verdict)" ;;
    5) echo "  merged, but NO TESTS COLLECTED on $into."
       echo "  Expected only mid-prune, before the apparatus layer is restored. If $into is"
       echo "  meant to hold the suite, restore it now and commit as apparatus." ;;
    *) if [[ "$into" == exp ]]; then
         echo "  merged, suite RED on exp — allowed: exp admits broken intermediate states"
       else
         echo "  SUITE FAILS on $into after the merge." >&2
         echo "  Most likely an incomplete apparatus layer: code restored without the" >&2
         echo "  test that constrains it, or vice versa. Fix on $into, do not fix on $from." >&2
         exit 3
       fi ;;
  esac
}

# Go tests and their fixtures ship with main; gate that tree before promoting it.
# A dry run remains read-only. Python tests belong only to the outer branches.
if (( ! dry_run )); then
  "$py" -c 'import pytest' 2>/dev/null || {
    echo "$py has no pytest; set BOLTBEAM_PYTHON to an interpreter with the dev extras" >&2
    exit 6
  }
  enter "$trunk"
  echo "=== $trunk: Go vet + Go tests ==="
  (cd tui && go vet ./... && go test -count=1 ./...) || {
    echo "Go gate FAILS on $trunk; no outward merge attempted." >&2
    exit 5
  }
fi

sync_one "$trunk" dev
sync_one dev exp

echo
echo "push with:  git push origin dev exp"
