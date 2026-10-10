#!/usr/bin/env bash
# BoltBeam on a rented NVIDIA pod: from the bare image to the first report with one command.
#
#   tools/pod-setup.sh                 run every step
#   tools/pod-setup.sh --dry-run       print every command with its resolved values; run only the checks that
#                                      need no GPU and download nothing (bash -n of this file, a HEAD request
#                                      for MODEL_URL, the ref names); exit 0
#   tools/pod-setup.sh --from 7        resume at step 7 (a number or a step name)
#   tools/pod-setup.sh --to 6          stop after step 6 (the steps that need no GPU)
#   tools/pod-setup.sh --multi         refused: "not yet"
#
# docs/h100-pod.md says what to rent and what comes back. docs/linux.md is the same setup by hand.
# Every step is idempotent: a second run finds what the first left and skips it. Each step's wall time goes to
# $WORK/logs/timings.tsv and the whole run to $WORK/logs/pod-setup-<time>.log.
#
# Settings (environment, each with a default):
#   WORK          the volume: /workspace/boltbeam
#   MODEL_URL     a GGUF: https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
#                 (a file:// URL links a model that is already on the machine)
#   MODEL_SHA256  the expected sha256 of that file, or its first hex digits; empty skips the comparison
#                 (the default model's known digest starts d98cdcbd03e17ce4)
#   BOLTBEAM_REPO, BOLTBEAM_REF   the public mirror, main
#   LLAMA_REPO, LLAMA_REF         ggml-org/llama.cpp at commit 50569eb87
#   TINYGRAD_REPO, TINYGRAD_REF   the tinygrad fork, branch exp; TINYGRAD_REF= skips it
#   CUDA_ARCH     CMAKE_CUDA_ARCHITECTURES; default: the GPU's compute capability from nvidia-smi (H100: 90)
#   RUN_TAG       the run ids: <RUN_TAG>-in-model and <RUN_TAG>-generic; default h100
#   PYTHON        the interpreter for the venv; default python3 (3.10 or newer)
#   JOBS          parallel compile jobs for llama.cpp; default: every CPU
#
# Steps:
#    1 facts        nvidia-smi, nvcc, nsys, CUDA version, free disk -> $WORK/facts.json
#    2 packages     apt only what is missing, only as root (cmake, git, build-essential, python3-venv, curl;
#                   nsys from the NVIDIA apt repo as nsight-systems-cli)
#    3 boltbeam     clone, venv under $WORK, pip install . plus numpy; the tinygrad fork beside it when public
#    4 llama.cpp    clone at LLAMA_REF, cmake -DGGML_CUDA=ON Release, CUDA arch from the GPU, build llama-bench
#    5 model        download with resume, sha256 recorded and compared
#    6 env          $WORK/env.sh (PATH, BOLTBEAM_LLAMA_BENCH, BOLTBEAM_GGML_CUDA_SRC, TMPDIR, BOLTBEAM_CUDA_CACHE)
#    7 check        boltbeam doctor when the ref has it, then boltbeam selfcheck
#    8 autoscan     this GPU's profile; its id is the --target of the runs
#    9 pipeline     --role-time in-model (nsys), then generic (BoltBeam's kernel timer)
#   10 results      results.json per run; the per-role table and the tie-out printed
#   11 bundle       $WORK/boltbeam-<RUN_TAG>-<date>.tgz: the runs' JSON, report.html, summary.md, no nsys binaries
set -euo pipefail

WORK="${WORK:-/workspace/boltbeam}"
MODEL_URL_DEFAULT="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
MODEL_URL="${MODEL_URL:-$MODEL_URL_DEFAULT}"
if [[ "$MODEL_URL" == "$MODEL_URL_DEFAULT" ]]; then MODEL_SHA256="${MODEL_SHA256:-d98cdcbd03e17ce4}"; else MODEL_SHA256="${MODEL_SHA256:-}"; fi
BOLTBEAM_REPO="${BOLTBEAM_REPO:-https://github.com/JulianAbeleda/BoltBeam-Public}"
BOLTBEAM_REF="${BOLTBEAM_REF:-main}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_REF="${LLAMA_REF:-50569eb87}"
TINYGRAD_REPO="${TINYGRAD_REPO:-https://github.com/JulianAbeleda/tinygrad-arkey}"
TINYGRAD_REF="${TINYGRAD_REF-exp}"
CUDA_ARCH="${CUDA_ARCH:-}"
RUN_TAG="${RUN_TAG:-h100}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-}"

BOLTBEAM_DIR="$WORK/BoltBeam"
VENV="$WORK/venv"
LLAMA_DIR="$WORK/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build-cuda"
TINYGRAD_DIR="$WORK/tinygrad-arkey-exp"
MODEL_NAME="$(basename "$MODEL_URL")"
MODEL="$WORK/models/$MODEL_NAME"
ENV_FILE="$WORK/env.sh"
RUNS="$WORK/runs"
STEPS=(facts packages boltbeam llama.cpp model env check autoscan pipeline results bundle)

DRY=0
FROM=1
TO=11
step_number() {  # a step name or number -> its number
  local v="$1" i
  if ! [[ "$v" =~ ^[0-9]+$ ]]; then
    for i in "${!STEPS[@]}"; do [[ "${STEPS[$i]}" == "$v" ]] && v=$((i + 1)); done
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "unknown step '$v'; one of 1..11 or ${STEPS[*]}" >&2; exit 2; }
  fi
  ((v >= 1 && v <= 11)) || { echo "step out of range: $v" >&2; exit 2; }
  echo "$v"
}
while (($#)); do
  case "$1" in
    --dry-run) DRY=1 ;;
    --from) shift; FROM="$(step_number "${1:-}")" ;;
    --from=*) FROM="$(step_number "${1#--from=}")" ;;
    --to) shift; TO="$(step_number "${1:-}")" ;;
    --to=*) TO="$(step_number "${1#--to=}")" ;;
    --multi) echo "--multi: not yet" >&2; exit 2 ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
((FROM <= TO)) || { echo "--from $FROM is after --to $TO" >&2; exit 2; }

# --- helpers -----------------------------------------------------------------------------------------------------

say() { echo "== $*"; }
note() { echo "   $*"; }
fail() { echo "FAILED: $*" >&2; exit 1; }

# run CMD...: print the command, then run it unless this is a dry run
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  ((DRY)) || "$@"
}

# run_sh 'shell text': the same for a line that needs a pipe or a redirection
run_sh() {
  echo "+ $1"
  ((DRY)) || bash -euo pipefail -c "$1"
}

# have CMD: the command is on PATH (a dry run answers for the machine it runs on and says so)
have() { command -v "$1" >/dev/null 2>&1; }

# exists TEST PATH: a file test that a dry run treats as "not there" (the pod is bare)
exists() { ((DRY)) && return 1; test "$1" "$2"; }

nsys_bin() {
  if have nsys; then command -v nsys; return; fi
  local p
  for p in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems*/*/bin/nsys /opt/nvidia/nsight-systems-cli/*/bin/nsys; do
    [[ -x "$p" ]] && { echo "$p"; return; }
  done
  return 1
}

nvcc_bin() {
  if have nvcc; then command -v nvcc; elif [[ -x /usr/local/cuda/bin/nvcc ]]; then echo /usr/local/cuda/bin/nvcc; else return 1; fi
}

gpu_compute_cap() {  # "9.0" -> "90"
  have nvidia-smi || return 1
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .'
}

cpus() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

cuda_arch() {
  if [[ -n "$CUDA_ARCH" ]]; then echo "$CUDA_ARCH"; return; fi
  local cc; cc="$(gpu_compute_cap || true)"
  [[ -n "$cc" ]] && { echo "$cc"; return; }
  ((DRY)) && { echo "90"; return; }   # a dry run with no nvidia-smi assumes an H100; the pod reads its own
  fail "no nvidia-smi and no CUDA_ARCH: set CUDA_ARCH (an H100 is 90)"
}

# head_model: "<bytes> <sha256-or-empty>" for an https MODEL_URL, from the response headers after redirects
head_model() {
  local headers
  headers="$(curl -sIL --max-time 30 "$MODEL_URL" | tr -d '\r')" || return 1
  local size sha
  size="$(awk 'tolower($1)=="content-length:"{v=$2} END{print v}' <<<"$headers")"
  sha="$(awk 'tolower($1)=="x-linked-etag:"{v=$2} END{print v}' <<<"$headers" | tr -d '"')"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || sha=""
  echo "$size $sha"
}

check_sha() {  # check_sha ACTUAL EXPECTED-OR-PREFIX
  [[ -z "$2" ]] && { note "sha256 $1 (nothing to compare with)"; return; }
  if [[ "$1" == "$2"* ]]; then note "sha256 $1 matches the expected $2..."; else fail "sha256 $1 does not start with the expected $2"; fi
}

# git_at DIR REPO REF: a clone at DIR checked out at REF (a branch, tag or commit), fetched fresh
git_at() {
  local dir="$1" repo="$2" ref="$3"
  exists -d "$dir/.git" || run git clone -q "$repo" "$dir"
  run git -C "$dir" fetch -q origin
  if ((DRY)); then
    if [[ "$ref" =~ ^[0-9a-f]{7,40}$ ]]; then echo "+ git -C $dir checkout -q --detach $ref"; else echo "+ git -C $dir checkout -q -B pod origin/$ref"; fi
    return
  fi
  if git -C "$dir" rev-parse -q --verify "origin/$ref" >/dev/null 2>&1; then
    run git -C "$dir" checkout -q -B pod "origin/$ref"
  else
    run git -C "$dir" checkout -q --detach "$ref"
  fi
}

TIMINGS=""
step_start=0
begin_step() {
  local n="$1"
  say "step $n: ${STEPS[$((n - 1))]}  ($(date '+%H:%M:%S'))"
  step_start=$(date +%s)
}
end_step() {
  local n="$1" secs=$(( $(date +%s) - step_start ))
  note "step $n done in ${secs}s"
  if [[ -n "$TIMINGS" ]]; then printf '%s\t%s\t%s\t%s\n' "$n" "${STEPS[$((n - 1))]}" "$secs" "$(date +%FT%T)" >> "$TIMINGS"; fi
}

# --- steps -------------------------------------------------------------------------------------------------------

step_facts() {
  local nvcc nsys
  nvcc="$(nvcc_bin || echo "")"; nsys="$(nsys_bin || echo "")"
  run mkdir -p "$WORK/logs" "$WORK/models" "$WORK/tmp" "$WORK/chips" "$RUNS"
  echo "+ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader"
  echo "+ $PYTHON (write $WORK/facts.json: gpu, nvcc ${nvcc:-<absent>}, nsys ${nsys:-<absent>}, cuda version, free disk under $WORK, nproc, os)"
  ((DRY)) && return
  local gpus="" cuda_ver="" nsys_ver="" ; local -i gpu_count=0
  if have nvidia-smi; then
    gpus="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"
    gpu_count=$(grep -c . <<<"$gpus" || true)
  fi
  [[ -n "$nvcc" ]] && cuda_ver="$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
  [[ -n "$nsys" ]] && nsys_ver="$("$nsys" --version 2>/dev/null | head -1)"
  GPUS="$gpus" GPU_COUNT="$gpu_count" NVCC="$nvcc" CUDA_VER="$cuda_ver" NSYS="$nsys" NSYS_VER="$nsys_ver" \
  FREE_KB="$(df -Pk "$WORK" | awk 'NR==2{print $4}')" NPROC="$(cpus)" OS="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")" \
  "$PYTHON" - "$WORK/facts.json" <<'PY'
import json, os, sys, datetime
gpus = [dict(zip(("name", "compute_cap", "memory_mib", "driver"), [c.strip() for c in line.split(",")]))
        for line in os.environ["GPUS"].splitlines() if line.strip()]
facts = {"taken_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "os": os.environ["OS"],
         "nproc": int(os.environ["NPROC"]), "gpu_count": int(os.environ["GPU_COUNT"]), "gpus": gpus,
         "nvcc": os.environ["NVCC"] or None, "cuda_version": os.environ["CUDA_VER"] or None,
         "nsys": os.environ["NSYS"] or None, "nsys_version": os.environ["NSYS_VER"] or None,
         "work_free_gib": round(int(os.environ["FREE_KB"]) / (1 << 20), 1)}
json.dump(facts, open(sys.argv[1], "w"), indent=2)
print(json.dumps(facts, indent=2))
PY
  ((gpu_count == 0)) && note "no GPU found by nvidia-smi: steps 8 to 11 will refuse; steps 1 to 7 run"
  [[ -z "$nvcc" ]] && note "nvcc is absent: step 4 needs the CUDA toolkit (the PyTorch image carries it under /usr/local/cuda)"
  [[ -z "$nsys" ]] && note "nsys is absent: step 2 installs it as root; without it step 9 runs the generic path only"
  return 0
}

step_packages() {
  local missing=() pkgs=()
  have cmake || { missing+=(cmake); pkgs+=(cmake); }
  have git || { missing+=(git); pkgs+=(git); }
  { have gcc && have g++ && have make; } || { missing+=(build-essential); pkgs+=(build-essential); }
  have curl || { missing+=(curl); pkgs+=(curl); }
  "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1 || { missing+=(python3-venv); pkgs+=(python3-venv); }
  local nsys_missing=0
  nsys_bin >/dev/null || nsys_missing=1
  if ((DRY)); then
    note "on this machine: ${missing[*]:-nothing} missing of cmake git build-essential curl python3-venv; nsys $( ((nsys_missing)) && echo absent || echo present)"
    echo "+ [root] apt-get update -q && apt-get install -y -q <the missing ones>"
    echo "+ [root, nsys absent] the NVIDIA apt repo (cuda-keyring for this Ubuntu) && apt-get install -y -q nsight-systems-cli"
    return
  fi
  if ((${#pkgs[@]} == 0)) && ((nsys_missing == 0)); then note "everything present: nothing to install"; return; fi
  if [[ "$(id -u)" != 0 ]]; then
    ((${#pkgs[@]})) && fail "not root and these are missing: ${missing[*]} (install them, then rerun --from 2)"
    note "not root and nsys is absent: step 9 runs the generic path only (install nsight-systems-cli for the in-model path)"
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  if ((${#pkgs[@]})); then
    run apt-get update -q
    run apt-get install -y -q "${pkgs[@]}"
  fi
  if ((nsys_missing)); then
    if ! apt-cache show nsight-systems-cli >/dev/null 2>&1; then
      local distro arch
      distro="$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")"   # ubuntu2204
      arch="$(dpkg --print-architecture)"; [[ "$arch" == amd64 ]] && arch=x86_64; [[ "$arch" == arm64 ]] && arch=sbsa
      run curl -fsSL -o "$WORK/tmp/cuda-keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb"
      run dpkg -i "$WORK/tmp/cuda-keyring.deb"
      run apt-get update -q
    fi
    run apt-get install -y -q nsight-systems-cli
    nsys_bin >/dev/null || fail "nsight-systems-cli installed but no nsys found; add its bin folder to PATH and rerun --from 2"
  fi
}

step_boltbeam() {
  if ! ((DRY)); then
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || fail "$PYTHON is older than 3.10; set PYTHON"
  fi
  git_at "$BOLTBEAM_DIR" "$BOLTBEAM_REPO" "$BOLTBEAM_REF"
  exists -x "$VENV/bin/python" || run "$PYTHON" -m venv "$VENV"
  local sha="" stamp
  ((DRY)) || sha="$(git -C "$BOLTBEAM_DIR" rev-parse --short HEAD)"
  stamp="$VENV/.boltbeam-${sha:-<sha>}"
  if exists -f "$stamp"; then note "BoltBeam $sha already installed in the venv"; else
    run "$VENV/bin/pip" install -q "$BOLTBEAM_DIR" numpy
    run touch "$stamp"
  fi
  if [[ -z "$TINYGRAD_REF" ]]; then note "tinygrad skipped: TINYGRAD_REF is empty"; return; fi
  if ! git ls-remote --exit-code -q "$TINYGRAD_REPO" "$TINYGRAD_REF" >/dev/null 2>&1; then
    note "tinygrad skipped: $TINYGRAD_REPO $TINYGRAD_REF is not reachable (private, or no network)"
    TINYGRAD_REF=""
    return
  fi
  git_at "$TINYGRAD_DIR" "$TINYGRAD_REPO" "$TINYGRAD_REF"
  exists -x "$TINYGRAD_DIR/.venv/bin/python" || run "$PYTHON" -m venv "$TINYGRAD_DIR/.venv"
  exists -f "$TINYGRAD_DIR/.venv/.numpy" || { run "$TINYGRAD_DIR/.venv/bin/pip" install -q numpy; run touch "$TINYGRAD_DIR/.venv/.numpy"; }
}

step_llama() {
  local arch nvcc
  arch="$(cuda_arch)"
  nvcc="$(nvcc_bin || true)"
  [[ -n "$nvcc" ]] || ((DRY)) || fail "no nvcc: install the CUDA toolkit or put /usr/local/cuda/bin on PATH"
  git_at "$LLAMA_DIR" "$LLAMA_REPO" "$LLAMA_REF"
  local sha="" stamp
  ((DRY)) || sha="$(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
  stamp="$LLAMA_BUILD/.pod-built-${sha:-<sha>}-sm$arch"
  if exists -f "$stamp" && exists -x "$LLAMA_BUILD/bin/llama-bench"; then note "llama.cpp $sha already built for sm_$arch"; return; fi
  run cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=ON "-DCMAKE_CUDA_ARCHITECTURES=$arch" "-DCMAKE_CUDA_COMPILER=${nvcc:-/usr/local/cuda/bin/nvcc}"
  run cmake --build "$LLAMA_BUILD" --config Release --parallel "${JOBS:-$(cpus)}" --target llama-bench llama-batched-bench
  run touch "$stamp"
}

step_model() {
  if [[ "$MODEL_URL" == file://* ]]; then
    local src="${MODEL_URL#file://}"
    exists -f "$src" || ((DRY)) || fail "MODEL_URL names a file that is not there: $src"
    run ln -sfn "$src" "$MODEL"
  else
    local size sha
    read -r size sha <<<"$(head_model || echo "")"
    [[ -n "$size" ]] || fail "HEAD $MODEL_URL answered no content-length: is the URL right?"
    note "HEAD: $size bytes, sha256 ${sha:-not declared by the server}"
    if ((DRY)); then
      [[ -n "$sha" ]] && check_sha "$sha" "$MODEL_SHA256"
      echo "+ curl -fL -C - --retry 5 -o $MODEL $MODEL_URL"
      echo "+ sha256sum $MODEL > $MODEL.sha256"
      return
    fi
    local have_size=0
    [[ -f "$MODEL" ]] && have_size="$(stat -Lc %s "$MODEL" 2>/dev/null || stat -Lf %z "$MODEL")"
    if [[ "$have_size" == "$size" ]]; then note "model already complete at $MODEL"; else
      run curl -fL -C - --retry 5 --retry-all-errors -o "$MODEL" "$MODEL_URL"
    fi
  fi
  ((DRY)) && { echo "+ sha256sum $MODEL > $MODEL.sha256"; return; }
  if [[ ! -f "$MODEL.sha256" ]] || [[ "$MODEL" -nt "$MODEL.sha256" ]]; then
    run_sh "sha256sum '$MODEL' | awk '{print \$1}' > '$MODEL.sha256'"
  fi
  local actual; actual="$(cat "$MODEL.sha256")"
  note "model $MODEL $(stat -Lc %s "$MODEL" 2>/dev/null || stat -Lf %z "$MODEL") bytes"
  check_sha "$actual" "$MODEL_SHA256"
  "$PYTHON" - "$WORK/facts.json" "$MODEL" "$actual" "$MODEL_URL" <<'PY'
import json, sys
p, model, sha, url = sys.argv[1:]
try: facts = json.load(open(p))
except FileNotFoundError: facts = {}
facts["model"] = {"path": model, "sha256": sha, "url": url}
json.dump(facts, open(p, "w"), indent=2)
PY
}

step_env() {
  local nsys_dir="" ; local nsys
  nsys="$(nsys_bin || true)"; [[ -n "$nsys" ]] && nsys_dir="$(dirname "$nsys")"
  local text
  text="$(cat <<EOF
# BoltBeam on this pod: written by tools/pod-setup.sh. Source it: . $ENV_FILE
export PATH=/usr/local/cuda/bin:${nsys_dir:+$nsys_dir:}$VENV/bin:\$PATH
export BOLTBEAM_LLAMA_BENCH=$LLAMA_BUILD/bin/llama-bench
export BOLTBEAM_LLAMA_BATCHED_BENCH=$LLAMA_BUILD/bin/llama-batched-bench
export BOLTBEAM_GGML_CUDA_SRC=$LLAMA_DIR/ggml/src/ggml-cuda
export BOLTBEAM_CHIPS_DIR=$WORK/chips
export TMPDIR=$WORK/tmp
export BOLTBEAM_CUDA_CACHE=$WORK/tmp/boltbeam-cuda
export BOLTBEAM_MODEL=$MODEL
${TINYGRAD_REF:+export BOLTBEAM_TINYGRAD_ROOT=$TINYGRAD_DIR}
EOF
)"
  echo "+ write $ENV_FILE:"
  sed 's/^/    /' <<<"$text"
  ((DRY)) && { echo "+ . $ENV_FILE"; return; }
  mkdir -p "$WORK/tmp/boltbeam-cuda" "$WORK/chips"
  printf '%s\n' "$text" > "$ENV_FILE"
  # shellcheck disable=SC1090
  . "$ENV_FILE"
}

load_env() {  # steps 7 to 11 need the env when the run starts there
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
  elif ! ((DRY)); then
    fail "$ENV_FILE is missing: run --from 6 first"
  fi
}

step_check() {
  load_env
  if ((DRY)); then
    echo "+ $VENV/bin/boltbeam doctor        # when the ref lists it in boltbeam --help"
    echo "+ $VENV/bin/boltbeam selfcheck"
    return
  fi
  if "$VENV/bin/boltbeam" --help 2>/dev/null | grep -qE '^\s*doctor\b'; then run "$VENV/bin/boltbeam" doctor; else note "this ref has no doctor command"; fi
  run "$VENV/bin/boltbeam" selfcheck
}

TARGET=""
step_autoscan() {
  load_env
  run_sh "$VENV/bin/python -m boltbeam.workflow.screen autoscan > '$WORK/autoscan.json'"
  echo "+ read target_id from $WORK/autoscan.json"
  ((DRY)) && { TARGET="<target_id from autoscan>"; return; }
  "$PYTHON" -m json.tool "$WORK/autoscan.json" | sed 's/^/    /'
  TARGET="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("target_id") or "")' "$WORK/autoscan.json")"
  [[ -n "$TARGET" ]] || fail "autoscan gave no target_id (status $("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status"), d.get("action"), d.get("reason") or "")' "$WORK/autoscan.json"))"
  note "target: $TARGET"
}

read_target() {
  [[ -n "$TARGET" ]] && return
  if ((DRY)); then TARGET="<target_id from autoscan>"; return; fi
  [[ -f "$WORK/autoscan.json" ]] || fail "$WORK/autoscan.json is missing: run --from 8 first"
  TARGET="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("target_id") or "")' "$WORK/autoscan.json")"
  [[ -n "$TARGET" ]] || fail "$WORK/autoscan.json has no target_id: run --from 8"
}

step_pipeline() {
  load_env; read_target
  local mode run_dir
  for mode in in-model generic; do
    run_dir="$RUNS/$RUN_TAG-$mode"
    if [[ "$mode" == in-model ]] && ! nsys_bin >/dev/null && ! ((DRY)); then
      note "in-model skipped: no nsys on this machine (step 2 installs it as root)"; continue
    fi
    if exists -f "$run_dir/.pod-done"; then note "$run_dir already measured (remove .pod-done to run it again)"; continue; fi
    run_sh "cd '$WORK/tmp' && $VENV/bin/python -m boltbeam.workflow.screen pipeline '$MODEL' --run '$run_dir' --target '$TARGET' --id '$RUN_TAG-$mode' --provider llama.cpp --analyze --measure auto --role-time $mode --no-search"
    run touch "$run_dir/.pod-done"
  done
}

step_results() {
  load_env
  local mode run_dir
  for mode in in-model generic; do
    run_dir="$RUNS/$RUN_TAG-$mode"
    if ! exists -d "$run_dir" && ! ((DRY)); then note "$run_dir: no run (in-model needs nsys)"; continue; fi
    run_sh "$VENV/bin/python -m boltbeam.workflow.screen results --run '$run_dir' > '$run_dir/results.json'"
    echo "+ $PYTHON (print the per-role table and the tie-out of $run_dir/results.json)"
    ((DRY)) && continue
    "$PYTHON" - "$run_dir/results.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); l = r.get("loss") or {}
print(f"    {r.get('id')}  target {r.get('target_id')}  measured {r.get('measured')}  loss status {l.get('status')}")
if l.get("role_source_words"): print("    " + l["role_source_words"])
est = l.get("estimate") or {}
if est.get("columns_words"): print("    " + est["columns_words"])
f = lambda v, w=8, d=1: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{str(v) if v is not None else '-':>{w}}")
print(f"    {'role':<22}{'quant':<7}{'us/call':>9}{'less floor':>11}{'GB/s':>8}{'% peak':>7}  reason")
for x in l.get("roles") or []:
    print(f"    {str(x.get('role')):<22}{str(x.get('quant')):<7}{f(x.get('us_per_call'), 9, 2)}{f(x.get('us_per_call_less_floor'), 11, 2)}"
          f"{f(x.get('gbs'), 8, 0)}{f(x.get('pct_peak'), 7, 0)}  {x.get('reason')}")
tie = l.get("tie_out") or {}
for line in tie.get("lines") or []:
    print(f"    tie-out  {f(line.get('ms'), 8, 3)} ms  {line.get('label')}  ({line.get('how')})")
if tie.get("token_ms"): print(f"    tie-out  {f(tie['token_ms'], 8, 3)} ms  the token ({tie.get('token_source')})")
for rt in l.get("runtimes") or []:
    print(f"    {rt.get('provider')}: {rt.get('tok_s')} tok/s, {rt.get('ms')} ms; limit {l.get('limit_tok_s')} tok/s")
PY
  done
}

step_bundle() {
  local date out list
  date="$(date +%Y%m%d)"
  out="$WORK/boltbeam-$RUN_TAG-$date.tgz"
  list="$WORK/tmp/bundle-files.txt"
  echo "+ list the files: runs/$RUN_TAG-*/ (*.json *.html *.md *.txt *.log *.csv; no *.nsys-rep, *.sqlite, *.qdstrm), facts.json, autoscan.json, logs/"
  run_sh "cd '$WORK' && { find runs/$RUN_TAG-in-model runs/$RUN_TAG-generic -type f \\( -name '*.json' -o -name '*.html' -o -name '*.md' -o -name '*.txt' -o -name '*.log' -o -name '*.csv' \\) 2>/dev/null; ls facts.json autoscan.json 2>/dev/null; find logs -type f; } > '$list'"
  run tar -czf "$out" -C "$WORK" -T "$list"
  ((DRY)) || note "bundle: $out ($(du -h "$out" | cut -f1))"
  local host="${RUNPOD_PUBLIC_IP:-HOST}" port="${RUNPOD_TCP_PORT_22:-PORT}"
  note "fetch it:  scp -P $port root@$host:$out ."
  note "or, from the BoltBeam checkout on your machine:  tools/pod-fetch.sh $host $port"
}

# --- main --------------------------------------------------------------------------------------------------------

current=0
say "pod-setup $( ((DRY)) && echo '(dry run) ' )WORK=$WORK"
note "MODEL_URL=$MODEL_URL"
note "MODEL_SHA256=${MODEL_SHA256:-<none>}  BOLTBEAM=$BOLTBEAM_REPO@$BOLTBEAM_REF  LLAMA=$LLAMA_REPO@$LLAMA_REF"
note "TINYGRAD=$( [[ -n "$TINYGRAD_REF" ]] && echo "$TINYGRAD_REPO@$TINYGRAD_REF" || echo skipped )"
note "CUDA_ARCH=$(cuda_arch)  RUN_TAG=$RUN_TAG  PYTHON=$PYTHON  steps $FROM to $TO"

if ((DRY)); then
  say "checks that need no GPU"
  bash -n "$0" && note "bash -n $0: ok"
  if [[ "$MODEL_URL" == file://* ]]; then
    [[ -f "${MODEL_URL#file://}" ]] && note "model file present: ${MODEL_URL#file://}" || note "model file NOT present: ${MODEL_URL#file://}"
  else
    read -r size sha <<<"$(head_model || echo "")"
    [[ -n "$size" ]] || fail "HEAD $MODEL_URL: no content-length (the URL does not resolve to a file)"
    note "MODEL_URL exists: $size bytes$( [[ -n "$sha" ]] && echo ", server-declared sha256 $sha")"
    [[ -n "$sha" ]] && check_sha "$sha" "$MODEL_SHA256"
  fi
  if git ls-remote --exit-code -q "$BOLTBEAM_REPO" "$BOLTBEAM_REF" >/dev/null 2>&1; then
    note "BOLTBEAM_REF $BOLTBEAM_REF: $(git ls-remote -q "$BOLTBEAM_REPO" "$BOLTBEAM_REF" | head -1 | cut -c1-12) at $BOLTBEAM_REPO"
  elif [[ "$BOLTBEAM_REF" =~ ^[0-9a-f]{7,40}$ ]] && [[ "$(curl -s -o /dev/null -w '%{http_code}' -I "$BOLTBEAM_REPO/commit/$BOLTBEAM_REF")" == 200 ]]; then
    note "BOLTBEAM_REF $BOLTBEAM_REF: a commit at $BOLTBEAM_REPO"
  else fail "BOLTBEAM_REF $BOLTBEAM_REF is not a branch, tag or commit of $BOLTBEAM_REPO"; fi
  if git ls-remote --exit-code -q "$LLAMA_REPO" "$LLAMA_REF" >/dev/null 2>&1; then
    note "LLAMA_REF $LLAMA_REF: a branch or tag of $LLAMA_REPO"
  elif [[ "$LLAMA_REF" =~ ^[0-9a-f]{7,40}$ ]] && [[ "$(curl -s -o /dev/null -w '%{http_code}' -I "$LLAMA_REPO/commit/$LLAMA_REF")" == 200 ]]; then
    note "LLAMA_REF $LLAMA_REF: a commit at $LLAMA_REPO"
  else fail "LLAMA_REF $LLAMA_REF is not a branch, tag or commit of $LLAMA_REPO"; fi
  if [[ -n "$TINYGRAD_REF" ]]; then
    if git ls-remote --exit-code -q "$TINYGRAD_REPO" "$TINYGRAD_REF" >/dev/null 2>&1; then
      note "TINYGRAD_REF $TINYGRAD_REF: $(git ls-remote -q "$TINYGRAD_REPO" "$TINYGRAD_REF" | head -1 | cut -c1-12) at $TINYGRAD_REPO (public: it will be cloned)"
    else note "TINYGRAD_REF $TINYGRAD_REF is not reachable at $TINYGRAD_REPO: tinygrad will be skipped"; TINYGRAD_REF=""; fi
  fi
  note "nvidia-smi $(have nvidia-smi && echo present || echo absent) here; nvcc $(nvcc_bin >/dev/null && echo present || echo absent); nsys $(nsys_bin >/dev/null && echo present || echo absent); root $( [[ "$(id -u)" == 0 ]] && echo yes || echo no)"
else
  mkdir -p "$WORK/logs"
  LOG="$WORK/logs/pod-setup-$(date +%Y%m%d-%H%M%S).log"
  TIMINGS="$WORK/logs/timings.tsv"
  exec > >(tee -a "$LOG") 2>&1
  note "log: $LOG"
  trap 'rc=$?; if ((rc && current > 0)); then echo "FAILED in step $current (${STEPS[$((current - 1))]}); fix it and rerun: $0 --from $current" >&2; fi; exit $rc' EXIT
fi

fns=(step_facts step_packages step_boltbeam step_llama step_model step_env step_check step_autoscan step_pipeline step_results step_bundle)
for i in "${!fns[@]}"; do
  current=$((i + 1))
  ((current < FROM || current > TO)) && continue
  begin_step "$current"
  "${fns[$i]}"
  end_step "$current"
done
if ((DRY)); then say "dry run complete: nothing was installed, cloned, downloaded or measured"
elif ((TO < 11)); then say "done: steps $FROM to $TO; continue with $0 --from $((TO + 1))"
else say "done: report bundle under $WORK"; fi
