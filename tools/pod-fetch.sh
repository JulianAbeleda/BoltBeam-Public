#!/usr/bin/env bash
# Fetch the report bundle tools/pod-setup.sh left on a pod and unpack it on this machine.
#
#   tools/pod-fetch.sh HOST PORT [TARBALL]
#
# HOST and PORT are the pod's SSH address (RunPod shows them as "ssh root@HOST -p PORT"). Without TARBALL the newest
# $WORK/boltbeam-*.tgz on the pod is taken (WORK defaults to /workspace/boltbeam, as in pod-setup.sh). The bundle
# lands under $DEST/<tag>-<date>/ (DEST defaults to ~/env/boltbeam-runs/.work), the same work folder the other
# measurement records keep their sources in. The user is root unless POD_USER says otherwise.
set -euo pipefail

[[ $# -ge 2 ]] || { sed -n '2,9p' "$0"; exit 2; }
host="$1" port="$2" tarball="${3:-}"
WORK="${WORK:-/workspace/boltbeam}"
DEST="${DEST:-$HOME/env/boltbeam-runs/.work}"
user="${POD_USER:-root}"

if [[ -z "$tarball" ]]; then
  tarball="$(ssh -p "$port" "$user@$host" "ls -t '$WORK'/boltbeam-*.tgz 2>/dev/null | head -1")"
  [[ -n "$tarball" ]] || { echo "no boltbeam-*.tgz under $WORK on $host: run tools/pod-setup.sh --from 11 there" >&2; exit 1; }
fi
name="$(basename "$tarball" .tgz)"       # boltbeam-h100-20261010
folder="${name#boltbeam-}"               # h100-20261010
mkdir -p "$DEST/$folder"
echo "== $user@$host:$tarball -> $DEST/$folder/"
scp -P "$port" "$user@$host:$tarball" "$DEST/$folder/"
tar -xzf "$DEST/$folder/$(basename "$tarball")" -C "$DEST/$folder"
echo "== unpacked:"
find "$DEST/$folder" -maxdepth 3 -type d | sort | sed 's/^/   /'
echo "== results:"
ls "$DEST/$folder"/runs/*/results.json "$DEST/$folder"/runs/*/report.html 2>/dev/null | sed 's/^/   /' || true
