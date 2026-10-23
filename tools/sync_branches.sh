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
[[ "${1:-}" == "--dry-run" ]] && dry_run=1
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
  local rc=0
  "$py" -m pytest -q >/dev/null 2>&1 || rc=$?
  case "$rc" in
    0) echo "  merged, suite green" ;;
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
