#!/usr/bin/env bash
# Release a version to Homebrew: tag the public mirror and point the tap's formula at the tag.
#
#   brew install JulianAbeleda/tap/boltbeam      a user's first install
#   brew update && brew upgrade boltbeam         a user's upgrade, after this script ran
#
# Before it: bump `version` in pyproject.toml on main, push main, publish the mirror. This script refuses unless
# origin/main names VERSION and the mirror's head is the snapshot of origin/main, so a tag always names the code
# the version says. It then:
#   1. tags the mirror's head vVERSION (a tag already there must be on that head);
#   2. hashes the tag's tarball, as Homebrew will download it;
#   3. rewrites url and sha256 in the tap's Formula/boltbeam.rb, commits and pushes.
#
# Usage: tools/release_brew.sh VERSION [--dry-run]
#   --dry-run  do every check and show the formula change; tag nothing, push nothing.
set -euo pipefail

version="${1:-}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: tools/release_brew.sh VERSION [--dry-run] (VERSION like 0.2.0)" >&2; exit 2; }
dry_run=0
[[ "${2:-}" == "--dry-run" ]] && dry_run=1

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mirror="${MIRROR_REPO:-JulianAbeleda/BoltBeam-Public}"
tap="${TAP_REPO:-JulianAbeleda/homebrew-tap}"
formula="Formula/boltbeam.rb"
tag="v$version"

work="$(mktemp -d "${TMPDIR:-/tmp}/release-brew.XXXXXX")"
trap 'rm -rf "$work"' EXIT
step() { echo "== $*"; }
die() { echo "$*" >&2; exit 1; }

step "origin/main names $version"
git -C "$repo_root" fetch -q origin
src="$(git -C "$repo_root" rev-parse --short origin/main)"
named="$(git -C "$repo_root" show origin/main:pyproject.toml | sed -n 's/^version = "\(.*\)"$/\1/p')"
[[ "$named" == "$version" ]] || die "origin/main's pyproject.toml says version $named, not $version: bump it on main and push first"

step "the mirror's head is the snapshot of origin/main ($src)"
head_sha="$(gh api "repos/$mirror/commits/main" --jq .sha)"
head_msg="$(gh api "repos/$mirror/commits/main" --jq .commit.message)"
[[ "$head_msg" == *"mirror private main $src"* ]] || die "the mirror's head is '$head_msg', not the snapshot of $src: publish the mirror first"

step "tag $tag on the mirror at ${head_sha:0:7}"
tagged="$(gh api "repos/$mirror/git/ref/tags/$tag" --jq .object.sha 2>/dev/null || true)"
if [[ -n "$tagged" ]]; then
  [[ "$tagged" == "$head_sha" ]] || die "$tag already exists on the mirror at ${tagged:0:7}, not the head ${head_sha:0:7}"
  echo "$tag is already there"
  url="https://github.com/$mirror/archive/refs/tags/$tag.tar.gz"
elif [[ $dry_run -eq 1 ]]; then
  echo "dry run: would create $tag; hashing the head's tarball instead (the tag's differs only in its folder name)"
  url="https://github.com/$mirror/archive/$head_sha.tar.gz"
else
  gh api -X POST "repos/$mirror/git/refs" -f ref="refs/tags/$tag" -f sha="$head_sha" --jq .ref
  url="https://github.com/$mirror/archive/refs/tags/$tag.tar.gz"
fi

step "hash the tarball Homebrew downloads"
curl -fsSL -o "$work/src.tar.gz" "$url"
sha="$(shasum -a 256 "$work/src.tar.gz" | cut -d' ' -f1)"
echo "$sha"

step "point $tap's $formula at $tag"
git clone -q "https://github.com/$tap.git" "$work/tap"
f="$work/tap/$formula"
[[ -f "$f" ]] || die "$tap has no $formula"
sed -i.bak -E \
  -e "s|^  url \".*\"$|  url \"https://github.com/$mirror/archive/refs/tags/$tag.tar.gz\"|" \
  -e "s|^  sha256 \".*\"$|  sha256 \"$sha\"|" "$f"
rm -f "$f.bak"
if git -C "$work/tap" diff --quiet; then
  echo "the formula already names $tag: nothing to push"
  exit 0
fi
git -C "$work/tap" diff
if [[ $dry_run -eq 1 ]]; then
  echo "dry run: every check passed; would commit 'boltbeam $version' to $tap and push"
  exit 0
fi
git -C "$work/tap" -c user.name="Julian Abeleda" -c user.email="29716705+JulianAbeleda@users.noreply.github.com" \
  commit -q -am "boltbeam $version"
git -C "$work/tap" push -q origin HEAD
echo "released boltbeam $version: brew update && brew upgrade boltbeam"
