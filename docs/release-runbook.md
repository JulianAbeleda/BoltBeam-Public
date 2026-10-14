# Release runbook: a new BoltBeam version on Homebrew

Users install with `brew install JulianAbeleda/tap/boltbeam` and upgrade with `brew update && brew upgrade boltbeam`.
The formula is `Formula/boltbeam.rb` in [homebrew-tap](https://github.com/JulianAbeleda/homebrew-tap). It downloads
a tagged tarball of [BoltBeam-Public](https://github.com/JulianAbeleda/BoltBeam-Public), builds the screen once, and
installs a `boltbeam` that opens it. A release is four steps, in this order. 0.2.0 (2026-10-13) was released this way.

## Before you start

- `gh auth status` is logged in with `repo` scope. The push goes to the mirror and the tap.
- Commits are authored with the GitHub noreply address. A personal address is refused at push time.
- `main`, `dev` and `exp` match `origin`, and the working tree is clean.
- `tools/publish_mirror.sh` is on `dev` only (tooling stays off the trunk, [branch-flow.md](branch-flow.md)), so it
  is not in a `main` checkout or in the mirror. `tools/release_brew.sh` is on `main`.

## 1. Bump the version on main and carry it outward

Set `version` in `pyproject.toml` (for example `0.3.0`) and commit it on `main` as `[repo] version 0.3.0`. Then merge
it into `dev` and `exp` (`tools/sync_branches.sh`, or merge by hand with the suite green on `dev`), and push all three:

```bash
git push origin main dev exp
```

## 2. Publish the mirror, from a worktree of origin/dev

```bash
git worktree add --detach "$TMPDIR/pubwt" origin/dev
(cd "$TMPDIR/pubwt" && tools/publish_mirror.sh --dry-run && tools/publish_mirror.sh)
git worktree remove --force "$TMPDIR/pubwt"
```

It publishes `origin/main`, refusing on a leak (see below), a failed install or a failed `selfcheck`. It ends with
`published <mirror sha> from private <main sha>`.

## 3. Release to the tap

From the `main` checkout. **Commit nothing to `main` between step 2 and this step.**

```bash
tools/release_brew.sh 0.3.0 --dry-run
tools/release_brew.sh 0.3.0
```

It tags the mirror's head `v0.3.0`, hashes that tag's tarball, and commits `boltbeam 0.3.0` to the tap with the new
`url` and `sha256`. It ends with `released boltbeam 0.3.0`. The dry run hashes the head's tarball instead of the
tag's, so its `sha256` differs from the real run's. That is expected.

## 4. Check it as a user would

```bash
brew update
brew install JulianAbeleda/tap/boltbeam      # or brew upgrade boltbeam
brew test boltbeam
boltbeam doctor                              # boltbeam-tui found under the brew prefix
```

On a machine that runs BoltBeam from a checkout, `brew uninstall boltbeam` afterwards so the checkout's `boltbeam`
is the one on PATH.

## When a step refuses

| message | cause | fix |
|---|---|---|
| `pyproject.toml says version X, not Y` | step 1 not done or not pushed | bump, push, publish the mirror again |
| `the mirror's head is ..., not the snapshot of <sha>` | `main` moved after the mirror was published | run step 2 again, then step 3 |
| `vY already exists on the mirror at ..., not the head` | the version was released from older code | bump to a new version; never move a released tag |
| `pattern '...' found in:` (publish_mirror) | private text in the tree, for example the account name in lower case | remove it on `main`; write the tap `JulianAbeleda/tap` |
| `the mirror already matches <sha>: nothing to publish` | the mirror is current | go to step 3 |
| `the formula already names vY: nothing to push` | the release is already done | nothing |

A tag on the mirror is never deleted or moved: the formula's `sha256` and anyone's installed copy depend on it.
