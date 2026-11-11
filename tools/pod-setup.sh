#!/usr/bin/env bash
# BoltBeam on a rented NVIDIA pod: from the bare image to the first report with one command.
#
#   tools/pod-setup.sh                 run every step
#   tools/pod-setup.sh --dry-run       print every command with its resolved values; run only the checks that
#                                      need no GPU and download nothing (bash -n of this file, a HEAD request
#                                      for MODEL_URL, the ref names); exit 0
#   tools/pod-setup.sh --from 7        resume at step 7 (a number or a step name)
#   tools/pod-setup.sh --to 6          stop after step 6 (the steps that need no GPU)
#
# docs/pods.md says what to rent and what comes back. docs/linux.md is the same setup by hand.
# Every step is idempotent: a second run finds what the first left and skips it. Each step's wall time goes to
# $WORK/logs/timings.tsv and the whole run to $WORK/logs/pod-setup-<time>.log.
#
# Settings (environment, each with a default):
#   WORK          builds, venvs, runs: /root/boltbeam, the container's own disk. Engines and builds are thousands of
#                 small files; on RunPod's /workspace (a network filesystem) cloning and installing took minutes per
#                 step where the local disk took seconds (2026-11-08). The container disk is erased when the pod stops.
#   KEEP          what outlives the pod, few and large: models and report bundles; default /workspace when it exists,
#                 else WORK. Models download to KEEP/models; each bundle is copied to KEEP/bundles.
#   MODEL_URL     a GGUF: https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
#                 (a file:// URL links a model that is already on the machine)
#   MODEL_SHA256  the full expected sha256 of that file; required for a non-default URL
#                 (the default digest is pinned to the upstream GGUF object)
#   MODEL_STREAMS the byte ranges an https model downloads in at once: 8 (1 is one plain curl). Measured on a rented
#                 RTX 4090 host, 2026-11-09: one stream 13 MB/s, eight 68 MB/s from the same Hugging Face file.
#   BOLTBEAM_REPO, BOLTBEAM_REF   the public mirror, main
#   LLAMA_REPO, LLAMA_REF         ggml-org/llama.cpp at commit 097cf33d9 (b11571, the engine pin in boltbeam/data/engine_pins.json)
#   LLAMA_PREBUILT                1 (default): step 4 first fetches the prebuilt llama.cpp of the pin
#                                 (tools/build-llama-prebuilt.sh; driver 525 or newer); 0 compiles on the pod
#   LLAMA_PREBUILT_URL            the archive; default the llama-<LLAMA_REF> release asset on the public mirror
#   LLAMA_PREBUILT_SHA256         the archive's sha256; default the one recorded below for the pinned LLAMA_REF
#   TINYGRAD_REPO, TINYGRAD_REF   the in-house tinygrad engine: off by default (a rented pod measures the mature
#                                 engines); TINYGRAD_REF=exp clones the fork
#   LAYOUT        how step 9 uses the pod's GPUs, the pipeline's --layout: one (default), layer (llama.cpp split by
#                 layer) or row (split by rows); more than one GPU gives tok/s and the limit, not the per-role split
#   ENGINES       the CUDA engines step 4 installs: "llama.cpp vllm" (default); add tensorrt-llm to install it too
#                 (a multi-GB wheel set that was most of step 4 on a slow host, 2026-11-09); Ollama is not installed
#   VLLM_VERSION, TRTLLM_VERSION  default: the pins in boltbeam/data/engine_pins.json (vLLM 0.31.0, TensorRT-LLM 1.2.1)
#   VLLM_CUDA     cu130 or cu129; default: cu130 on a driver of 580 or newer, else cu129
#   TRTLLM_PYTHON the interpreter for TensorRT-LLM's venv (3.10 or 3.12 have wheels); default: PYTHON
#   CUDA_ARCH     CMAKE_CUDA_ARCHITECTURES; default: the GPU's compute capability from nvidia-smi (H100: 90)
#   RUN_TAG       the run ids: <RUN_TAG>-in-model and <RUN_TAG>-generic; default from the GPUs, such as rtx3090,
#                 h100 or 8xrtx4090 ("pod" when nvidia-smi names none)
#   PYTHON        the interpreter for the venv; default python3 (3.10 or newer)
#   JOBS          parallel compile jobs for llama.cpp; default: every CPU
#
# Steps:
#    1 facts        nvidia-smi, nvcc, nsys, CUDA version, free disk -> $WORK/facts.json
#    2 packages     apt only what is missing, only as root (cmake, git, build-essential, python3-venv, python3-dev, curl, zstd;
#                   libopenmpi-dev when ENGINES has tensorrt-llm; nsys from the NVIDIA apt repo as nsight-systems-cli)
#    3 boltbeam     clone, venv under $WORK, pip install -e . (the pipeline runs the checkout) plus numpy; the tinygrad fork beside it when public
#    4 engines      llama.cpp: clone at LLAMA_REF; the prebuilt archive when its sha256, this GPU's architecture, the
#                   driver and llama-bench --list-devices all check out, else cmake -DGGML_CUDA=ON Release for the
#                   GPU's arch (facts.json llama_cpp says which and why); vLLM and TensorRT-LLM: each in its own venv
#                   under $WORK/engines at its pinned version (they need different torch builds), installed with uv
#                   (pip if uv fails); the three at once; a failed optional engine is noted, the run goes on with llama.cpp
#    5 model        download fresh, full sha256 verified before installation; on a real run it starts in the
#                   background before step 3 ($WORK/logs/step5-model.log) and step 9 waits for it
#    6 env          $WORK/env.sh (PATH, BOLTBEAM_LLAMA_BENCH, BOLTBEAM_GGML_CUDA_SRC, TMPDIR, BOLTBEAM_CUDA_CACHE,
#                   BOLTBEAM_VLLM_PYTHON and BOLTBEAM_TRTLLM_PYTHON for the engines step 4 installed)
#    7 check        boltbeam doctor when the ref has it, then boltbeam selfcheck
#    8 autoscan     this GPU's profile; its id is the --target of the runs
#    9 pipeline     --role-time in-model (nsys), then generic (BoltBeam's kernel timer)
#   10 results      results.json per run; the per-role table and the tie-out printed
#   11 bundle       $WORK/boltbeam-<RUN_TAG>-<date>.tgz: the runs' JSON, report.html, summary.md, no nsys binaries
set -euo pipefail

WORK="${WORK:-/root/boltbeam}"
if [[ -z "${KEEP:-}" ]]; then if [[ -d /workspace && -w /workspace ]]; then KEEP=/workspace; else KEEP="$WORK"; fi; fi
MODEL_URL_DEFAULT="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
MODEL_URL="${MODEL_URL:-$MODEL_URL_DEFAULT}"
if [[ "$MODEL_URL" == "$MODEL_URL_DEFAULT" ]]; then MODEL_SHA256="${MODEL_SHA256:-d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785}"; else MODEL_SHA256="${MODEL_SHA256:-}"; fi
MODEL_STREAMS="${MODEL_STREAMS:-8}"
MODEL_RANGE_MIN="${MODEL_RANGE_MIN:-67108864}"   # a file under 64 MiB is one curl: ranges save nothing
BOLTBEAM_REPO="${BOLTBEAM_REPO:-https://github.com/JulianAbeleda/BoltBeam-Public}"
BOLTBEAM_REF="${BOLTBEAM_REF:-main}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_REF_PIN=097cf33d9
LLAMA_REF="${LLAMA_REF:-$LLAMA_REF_PIN}"
# The prebuilt llama.cpp of the pin (tools/build-llama-prebuilt.sh, CUDA 12.8): the sha256 a fetched archive must have
LLAMA_PREBUILT_SHA256_PIN=2b5e75859f32f4f88e09139038e75d8e7196a160039ea5657653ec7ca7547f8e
LLAMA_PREBUILT="${LLAMA_PREBUILT:-1}"
LLAMA_PREBUILT_URL="${LLAMA_PREBUILT_URL:-https://github.com/JulianAbeleda/BoltBeam-Public/releases/download/llama-$LLAMA_REF/llama.cpp-$LLAMA_REF-cuda-linux-x86_64.tar.gz}"
if [[ "$LLAMA_REF" == "$LLAMA_REF_PIN" ]]; then LLAMA_PREBUILT_SHA256="${LLAMA_PREBUILT_SHA256:-$LLAMA_PREBUILT_SHA256_PIN}"; else LLAMA_PREBUILT_SHA256="${LLAMA_PREBUILT_SHA256:-}"; fi
TINYGRAD_REPO="${TINYGRAD_REPO:-https://github.com/JulianAbeleda/tinygrad-arkey}"
TINYGRAD_REF="${TINYGRAD_REF-}"
CUDA_ARCH="${CUDA_ARCH:-}"
RUN_TAG="${RUN_TAG:-}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-}"
ENGINES="${ENGINES-llama.cpp vllm}"
LAYOUT="${LAYOUT:-one}"
case "$LAYOUT" in one|layer|row) ;; *) echo "LAYOUT must be one, layer or row: $LAYOUT" >&2; exit 2 ;; esac
LSUF=""; [[ "$LAYOUT" == one ]] || LSUF="-$LAYOUT"  # a split layout gets its own run folder: runs/<tag>-<mode>-<layout>
VLLM_VERSION="${VLLM_VERSION:-}"
TRTLLM_VERSION="${TRTLLM_VERSION:-}"
VLLM_CUDA="${VLLM_CUDA:-}"
TRTLLM_PYTHON="${TRTLLM_PYTHON:-}"

BOLTBEAM_DIR="$WORK/BoltBeam"
VENV="$WORK/venv"
LLAMA_DIR="$WORK/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build-cuda"
TINYGRAD_DIR="$WORK/tinygrad-arkey-exp"
MODEL_NAME="$(basename "$MODEL_URL")"
MODEL="$KEEP/models/$MODEL_NAME"
ENV_FILE="$WORK/env.sh"
RUNS="$WORK/runs"
ENGINES_DIR="$WORK/engines"
VLLM_VENV="$ENGINES_DIR/vllm"
TRTLLM_VENV="$ENGINES_DIR/tensorrt-llm"
STEPS=(facts packages boltbeam engines model env check autoscan pipeline results bundle)

DRY=0
FROM=1
TO=11
step_number() {  # a step name or number -> its number
  local v="$1" i
  if ! [[ "$v" =~ ^[0-9]+$ ]]; then
    [[ "$v" == llama.cpp ]] && v=engines   # step 4's name before it installed every engine
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
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
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
  local code="$1"; shift
  run bash -euo pipefail -c "$code" -- "$@"
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

# wants ENGINE: ENGINES names it
wants() { [[ " $ENGINES " == *" $1 "* ]]; }

driver_major() {  # "580.95.05" -> "580"; empty with no nvidia-smi
  have nvidia-smi || return 1
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -d ' '
}

# pin ENGINE: its version from the checkout's engine_pins.json (the one table of engine versions)
pin() {
  local f="$BOLTBEAM_DIR/boltbeam/data/engine_pins.json"
  [[ -f "$f" ]] || return 1
  "$PYTHON" -c 'import json,sys; e=json.load(open(sys.argv[1]))["engines"].get(sys.argv[2]) or {}; print(e.get("version") or "")' "$f" "$1"
}

cpus() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

cuda_arch() {
  if [[ -n "$CUDA_ARCH" ]]; then echo "$CUDA_ARCH"; return; fi
  local cc; cc="$(gpu_compute_cap || true)"
  [[ -n "$cc" ]] && { echo "$cc"; return; }
  ((DRY)) && { echo "90"; return; }   # a dry run with no nvidia-smi assumes an H100; the pod reads its own
  fail "no nvidia-smi and no CUDA_ARCH: set CUDA_ARCH (an H100 is 90)"
}

# gpu_tag: the default RUN_TAG, from nvidia-smi's GPU names: "NVIDIA GeForce RTX 3090" is rtx3090, eight H100s are
# 8xh100, mixed GPUs take the first one's name; "pod" when there is no nvidia-smi (a dry run off the pod)
gpu_tag() {
  local names n name
  names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  [[ -n "$names" ]] || { echo pod; return; }
  n="$(grep -c . <<<"$names")"
  name="$(head -1 <<<"$names" | tr 'A-Z' 'a-z' | sed -E 's/nvidia|geforce|tesla//g; s/[- ][0-9]+ ?gb.*$//; s/[^a-z0-9]//g')"
  [[ -n "$name" ]] || name=gpu
  ((n > 1)) && echo "${n}x$name" || echo "$name"
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

check_sha() {
  [[ "$2" =~ ^[0-9a-f]{64}$ ]] || fail "MODEL_SHA256 must be a full 64-hex digest"
  [[ "$1" == "$2" ]] || fail "sha256 $1 does not match expected $2"
  note "sha256 verified: $1"
}

[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "MODEL_SHA256 must be a full 64-hex digest for MODEL_URL"

# facts_set KEY NAME=VALUE...: one entry of $WORK/facts.json, under a lock (steps 4 and 5 write it at the same time)
facts_set() {
  "$PYTHON" - "$WORK/facts.json" "$@" <<'PY'
import fcntl, json, sys
p, key, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(p + ".lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try: facts = json.load(open(p))
    except (FileNotFoundError, ValueError): facts = {}
    facts[key] = dict(kv.split("=", 1) for kv in pairs)
    json.dump(facts, open(p, "w"), indent=2)
PY
}


# git_at DIR REPO REF: a clone at DIR checked out at REF (a branch, tag or commit), fetched fresh
git_at() {
  local dir="$1" repo="$2" ref="$3"
  exists -d "$dir/.git" || run git clone -q "$repo" "$dir"
  if (( ! DRY )); then
    [[ -z "$(git -C "$dir" status --porcelain)" ]] || fail "checkout has local edits: $dir"
  fi
  run git -C "$dir" fetch -q origin
  if ((DRY)); then
    if [[ "$ref" =~ ^[0-9a-f]{7,40}$ ]]; then echo "+ git -C $dir checkout -q --detach $ref"; else echo "+ git -C $dir checkout -q --detach origin/$ref"; fi
    return
  fi
  if git -C "$dir" rev-parse -q --verify "origin/$ref" >/dev/null 2>&1; then
    run git -C "$dir" checkout -q --detach "origin/$ref"
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

# gpu_health: every GPU must open a CUDA context and carry no memory fault awaiting a reset, before anything is installed
# or downloaded on a machine billed by the hour. A RunPod H200 passed RunPod's own checks with 88 uncorrectable DRAM
# errors and row remaps pending; every CUDA context failed (CUresult 999) only at step 8, after its setup and model
# download, and RunPod rented the same card again (2026-11-08). Driver facts only: nvidia-smi and libcuda via ctypes.
gpu_health() {
  have nvidia-smi || fail "no nvidia-smi: this is not an NVIDIA GPU machine"
  local bad="" q
  q="$(nvidia-smi -q -d ROW_REMAPPER,ECC 2>/dev/null || true)"
  grep -A3 "Remapped Rows" <<<"$q" | grep -q "Pending *: Yes" && bad+="row remaps pending a GPU reset; "
  grep -q "Remapping Failure Occurred *: Yes" <<<"$q" && bad+="a row remap failed; "
  local volatile; volatile="$(awk '/Volatile/{v=1} /Aggregate/{v=0} v && /DRAM Uncorrectable/{s+=$NF} END{print s+0}' <<<"$q")"
  ((volatile > 0)) && bad+="$volatile uncorrectable DRAM errors since boot; "
  "$PYTHON" - <<'PY' || bad+="a CUDA context does not open (see above); "
import ctypes, sys
cu = ctypes.CDLL("libcuda.so.1"); n = ctypes.c_int(); r = cu.cuInit(0)
if r: sys.exit(print(f"cuInit failed: CUresult {r}") or 1)
cu.cuDeviceGetCount(ctypes.byref(n))
for i in range(n.value):
    d, ctx = ctypes.c_int(), ctypes.c_void_p()
    cu.cuDeviceGet(ctypes.byref(d), i)
    r = cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d)
    if r: sys.exit(print(f"GPU {i}: cuDevicePrimaryCtxRetain failed: CUresult {r}") or 1)
    cu.cuDevicePrimaryCtxRelease(d)
print(f"   {n.value} GPU(s) open a CUDA context")
PY
  [[ -z "$bad" ]] || fail "unhealthy GPU: ${bad%; }. Delete this machine and rent another; nothing was installed"
  note "GPU health: no pending remaps, no uncorrectable DRAM errors since boot, contexts open"
}

step_facts() {
  ((DRY)) || gpu_health
  local nvcc nsys
  nvcc="$(nvcc_bin || echo "")"; nsys="$(nsys_bin || echo "")"
  run mkdir -p "$WORK/logs" "$KEEP/models" "$WORK/tmp" "$WORK/chips" "$RUNS"
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
  have zstd || { missing+=(zstd); pkgs+=(zstd); }  # cuBLASLt keeps 5,000+ kernels zstd-compressed in .ldata (instruction_path)
  "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1 || { missing+=(python3-venv); pkgs+=(python3-venv); }
  # Triton (vLLM, TensorRT-LLM) compiles a C helper against Python.h at startup: NVIDIA's CUDA images ship no headers,
  # and vLLM then failed to inspect its model class (a Vast RTX 3090, 2026-11-08)
  "$PYTHON" -c 'import os, sysconfig; raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_path("include"), "Python.h")))' \
    || { missing+=(python3-dev); pkgs+=(python3-dev); }
  if wants tensorrt-llm && ! { have mpicc || [[ -e /usr/include/x86_64-linux-gnu/openmpi ]] || [[ -e /usr/lib/x86_64-linux-gnu/openmpi ]]; }; then
    missing+=(libopenmpi-dev); pkgs+=(libopenmpi-dev)
  fi
  local nsys_missing=0
  nsys_bin >/dev/null || nsys_missing=1
  if ((DRY)); then
    note "on this machine: ${missing[*]:-nothing} missing of cmake git build-essential curl zstd python3-venv python3-dev$(wants tensorrt-llm && echo ' libopenmpi-dev'); nsys $( ((nsys_missing)) && echo absent || echo present)"
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
    # An image that already lists NVIDIA's CUDA repo keeps it: a second entry with another signed-by setting makes
    # apt refuse every source (seen on runpod/pytorch 2.0.0-cu1300, 2026-11-08). Only a machine without one gets the keyring.
    if grep -rqs 'developer.download.nvidia.com/compute/cuda/repos' /etc/apt/sources.list /etc/apt/sources.list.d/; then
      note "NVIDIA's CUDA apt repo is already listed: using it"
    else
      local distro arch
      distro="$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")"   # ubuntu2204
      arch="$(dpkg --print-architecture)"; [[ "$arch" == amd64 ]] && arch=x86_64; [[ "$arch" == arm64 ]] && arch=sbsa
      run curl -fsSL -o "$WORK/tmp/cuda-keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb"
      run dpkg -i "$WORK/tmp/cuda-keyring.deb"
    fi
    run apt-get update -q
    # The repo names the package nsight-systems-cli, or by version (nsight-systems-2026.5.2): take the newest of those.
    local nsys_pkg=nsight-systems-cli
    apt-cache show nsight-systems-cli >/dev/null 2>&1 || \
      nsys_pkg="$(apt-cache search --names-only '^nsight-systems-[0-9]' | cut -d' ' -f1 | sort -V | tail -1)"
    [[ -n "$nsys_pkg" ]] || fail "NVIDIA's apt repo has no Nsight Systems package; install nsys by hand and rerun --from 2"
    run apt-get install -y -q "$nsys_pkg"
    nsys_bin >/dev/null || fail "$nsys_pkg installed but no nsys found; add its bin folder to PATH and rerun --from 2"
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
    # editable: a pull or a fix copied into the checkout is what the pipeline runs, not a copy frozen at setup
    run "$VENV/bin/pip" install -q -e "$BOLTBEAM_DIR" numpy
    run touch "$stamp"
  fi
  install_go
  if [[ -z "$TINYGRAD_REF" ]]; then note "tinygrad skipped: in-house, opt in with TINYGRAD_REF=exp"; return; fi
  if ! git ls-remote --exit-code -q "$TINYGRAD_REPO" "$TINYGRAD_REF" >/dev/null 2>&1; then
    note "tinygrad skipped: $TINYGRAD_REPO $TINYGRAD_REF is not reachable (private, or no network)"
    TINYGRAD_REF=""
    return
  fi
  git_at "$TINYGRAD_DIR" "$TINYGRAD_REPO" "$TINYGRAD_REF"
  exists -x "$TINYGRAD_DIR/.venv/bin/python" || run "$PYTHON" -m venv "$TINYGRAD_DIR/.venv"
  exists -f "$TINYGRAD_DIR/.venv/.numpy" || { run "$TINYGRAD_DIR/.venv/bin/pip" install -q numpy; run touch "$TINYGRAD_DIR/.venv/.numpy"; }
}

# The engines share nothing (each its own folder and venv), so they install at the same time; each writes its own
# log under $WORK/logs. llama.cpp is needed by steps 9 and 10: its failure fails the step. A failed optional engine
# is noted and the run goes on.
# The TUI (boltbeam tui) builds from the checkout's tui/ with the Go its go.mod names; Ubuntu's golang-go is older
# (1.22 on 24.04 against go 1.26.7). The official release goes under /usr/local/go, as root, when go is absent or older.
install_go() {
  local want have arch
  want="$( ( ((DRY)) && echo "<tui/go.mod>" ) || sed -n 's/^go \([0-9.]*\)$/\1/p' "$BOLTBEAM_DIR/tui/go.mod" 2>/dev/null)"
  [[ -n "$want" ]] || { note "no tui/go.mod in the checkout: the TUI is not set up"; return 0; }
  have="$( { /usr/local/go/bin/go version || go version; } 2>/dev/null | sed -n 's/.*go\([0-9][0-9.]*\).*/\1/p' | head -1 || true)"  # no Go: empty, not an exit
  if [[ -n "$have" ]] && ! ((DRY)) && [[ "$(printf '%s\n%s\n' "$want" "$have" | sort -V | head -1)" == "$want" ]]; then
    note "Go $have is new enough for the TUI (go.mod: $want)"; return 0
  fi
  if [[ "$(id -u)" != 0 ]] && ! ((DRY)); then note "the TUI needs Go $want and this is not root: install it, then boltbeam tui"; return 0; fi
  arch="$(uname -m)"; [[ "$arch" == x86_64 ]] && arch=amd64; [[ "$arch" == aarch64 ]] && arch=arm64
  run curl -fsSL -o "$WORK/tmp/go.tgz" "https://go.dev/dl/go$want.linux-$arch.tar.gz"
  run rm -rf /usr/local/go
  run tar -C /usr/local -xzf "$WORK/tmp/go.tgz"
}

step_engines() {
  wants llama.cpp || note "llama.cpp skipped: not in ENGINES (steps 9 and 10 measure with it)"
  if ((DRY)); then
    wants llama.cpp && step_llama
    { wants vllm || wants tensorrt-llm; } && ensure_uv
    wants vllm && install_vllm
    wants tensorrt-llm && install_trtllm
    return 0
  fi
  if wants vllm || wants tensorrt-llm; then ensure_uv; fi
  local -A pid=() fn=([llama.cpp]=step_llama [vllm]=install_vllm [tensorrt-llm]=install_trtllm)
  local e rc log
  for e in llama.cpp vllm tensorrt-llm; do
    wants "$e" || continue
    log="$WORK/logs/engine-$e.log"
    ( "${fn[$e]}" ) > "$log" 2>&1 &
    pid[$e]=$!
    note "$e installing in the background: $log"
  done
  for e in llama.cpp vllm tensorrt-llm; do
    [[ -n "${pid[$e]:-}" ]] || continue
    rc=0; wait "${pid[$e]}" || rc=$?
    log="$WORK/logs/engine-$e.log"
    if ((rc == 0)); then note "$e: ready ($(grep -E '^(vllm|tensorrt_llm) [0-9]|^llama\.cpp: ' "$log" | tail -1 || true))"; continue; fi
    tail -n 15 "$log" | sed 's/^/   | /'
    [[ "$e" == llama.cpp ]] && fail "llama.cpp did not build (log above; full log $log)"
    note "$e not installed (log above; full log $log); the run goes on with llama.cpp; rerun --from 4 after fixing"
  done
  return 0
}

# uv installs the Python engines: vLLM and TensorRT-LLM are multi-GB wheel sets that pip fetches one at a time. It is
# the one on PATH, else Astral's standalone installer puts it under $WORK/uv (pip into a venv there when that fails).
# Its cache sits on the venvs' disk so installs hard-link from it; UV is empty when none of that works (pip then).
UV=""
ensure_uv() {
  export UV_CACHE_DIR="$WORK/tmp/uv-cache" UV_HTTP_TIMEOUT=300
  local c
  if have uv; then UV="$(command -v uv)"; note "uv: $UV"; return 0; fi
  ((DRY)) && { echo "+ [no uv] curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$WORK/uv/bin sh  (else pip install uv into $WORK/uv)"; UV="$WORK/uv/bin/uv"; return 0; }
  for c in "$WORK/uv/bin/uv" "$WORK/uv/bin/bin/uv"; do [[ -x "$c" ]] && { UV="$c"; note "uv: $UV"; return 0; }; done
  run_sh 'curl -LsSf --retry 3 https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$1" UV_NO_MODIFY_PATH=1 sh -s -- -q' "$WORK/uv/bin" \
    || { run "$PYTHON" -m venv "$WORK/uv" && run "$WORK/uv/bin/pip" install -q uv; } || true
  for c in "$WORK/uv/bin/uv" "$WORK/uv/bin/bin/uv"; do [[ -x "$c" ]] && { UV="$c"; note "uv: $UV"; return 0; }; done
  note "uv could not be installed: the Python engines install with pip"
}

# pip_install VENV ARG...: pip's arguments, run by uv into VENV (pip's index precedence: unsafe-best-match), then by
# the venv's own pip when uv is absent or fails
pip_install() {
  local venv="$1"; shift
  if [[ -n "$UV" ]]; then
    run "$UV" pip install -q --python "$venv/bin/python" --index-strategy unsafe-best-match "$@" && return 0
    note "uv pip install failed: retrying with pip"
  fi
  run "$venv/bin/pip" install -q --no-cache-dir --upgrade pip || return 1
  run "$venv/bin/pip" install -q --no-cache-dir "$@"
}

# vLLM at its pin in its own venv: the default wheel is built for CUDA 13 (driver 580 or newer); an older driver
# takes the cu129 build from vLLM's own index with the matching torch.
install_vllm() {
  local ver cuda drv
  ver="${VLLM_VERSION:-$(pin vllm || true)}"
  ((DRY)) && ver="${ver:-<the vllm pin>}"
  [[ -n "$ver" ]] || { note "no vLLM version: set VLLM_VERSION or run step 3 first"; return 1; }
  drv="$(driver_major || true)"
  cuda="$VLLM_CUDA"
  if [[ -z "$cuda" ]]; then
    if [[ -z "$drv" ]] && ((DRY)); then cuda=cu130; elif [[ -n "$drv" ]] && ((drv >= 580)); then cuda=cu130; else cuda=cu129; fi
  fi
  local stamp="$VLLM_VENV/.pod-vllm-$ver-$cuda"
  if exists -f "$stamp"; then note "vLLM $ver ($cuda) already installed in $VLLM_VENV"; return 0; fi
  note "vLLM $ver for $cuda (driver ${drv:-unknown})"
  exists -x "$VLLM_VENV/bin/python" || run "$PYTHON" -m venv "$VLLM_VENV" || return 1
  if [[ "$cuda" == cu130 ]]; then
    pip_install "$VLLM_VENV" "vllm==$ver" || return 1
  else
    pip_install "$VLLM_VENV" "vllm==$ver" \
      --extra-index-url "https://wheels.vllm.ai/$ver/$cuda" --extra-index-url "https://download.pytorch.org/whl/$cuda" || return 1
  fi
  run "$VLLM_VENV/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)' || return 1
  run touch "$stamp"
}

# TensorRT-LLM at its pin in its own venv. Its wheels are on NVIDIA's index (PyPI has only the source), for
# Python 3.10 and 3.12, built for CUDA 13 (driver 580 or newer), with torch 2.9 from the cu130 index (its pin is torch<=2.10.0a0).
install_trtllm() {
  local ver py drv pyv
  ver="${TRTLLM_VERSION:-$(pin tensorrt-llm || true)}"
  ((DRY)) && ver="${ver:-<the tensorrt-llm pin>}"
  [[ -n "$ver" ]] || { note "no TensorRT-LLM version: set TRTLLM_VERSION or run step 3 first"; return 1; }
  py="${TRTLLM_PYTHON:-$PYTHON}"
  drv="$(driver_major || true)"
  if ! ((DRY)); then
    [[ -n "$drv" ]] && ((drv >= 580)) || { note "TensorRT-LLM $ver needs a driver of 580 or newer (CUDA 13); this pod has ${drv:-none}: rent a host filtered to CUDA 13"; return 1; }
    pyv="$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    [[ "$pyv" == 3.10 || "$pyv" == 3.12 ]] || { note "TensorRT-LLM wheels are for Python 3.10 and 3.12; $py is $pyv: set TRTLLM_PYTHON"; return 1; }
  fi
  local stamp="$TRTLLM_VENV/.pod-trtllm-$ver"
  if exists -f "$stamp"; then note "TensorRT-LLM $ver already installed in $TRTLLM_VENV"; return 0; fi
  note "TensorRT-LLM $ver (driver ${drv:-unknown}, $py)"
  exists -x "$TRTLLM_VENV/bin/python" || run "$py" -m venv "$TRTLLM_VENV" || return 1
  pip_install "$TRTLLM_VENV" --upgrade setuptools wheel || return 1
  pip_install "$TRTLLM_VENV" "torch>=2.9.1,<2.10" torchvision --index-url https://download.pytorch.org/whl/cu130 || return 1
  run_sh '"$1" -c "import torch; print(\"torch==\" + torch.__version__)" > "$2"' "$TRTLLM_VENV/bin/python" "$TRTLLM_VENV/torch-constraint.txt" || return 1
  # pyarrow < 20: tensorrt_llm pins numpy < 2 and the newest pyarrow needs numpy 2, so importing tensorrt_llm (its
  # datasets import) failed with "pyarrow requires NumPy 2.0 or newer" (1.2.1, 2026-11-08)
  pip_install "$TRTLLM_VENV" "tensorrt_llm==$ver" "pyarrow<20" -c "$TRTLLM_VENV/torch-constraint.txt" \
    --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 || return 1
  run "$TRTLLM_VENV/bin/python" -c 'import tensorrt_llm, torch; print("tensorrt_llm", tensorrt_llm.__version__, "torch", torch.__version__)' || return 1
  run touch "$stamp"
}

# llama.cpp: the clone at the pin always (the ggml-cuda source is what role timing reads), then the prebuilt binaries
# when they fit this machine, else a compile for this GPU (the 3090 host's compile was most of a 21-minute step 4).
step_llama() {
  local arch nvcc
  arch="$(cuda_arch)"
  git_at "$LLAMA_DIR" "$LLAMA_REPO" "$LLAMA_REF"
  local sha="" commit="" built fetched
  ((DRY)) || { sha="$(git -C "$LLAMA_DIR" rev-parse --short HEAD)"; commit="$(git -C "$LLAMA_DIR" rev-parse HEAD)"; }
  built="$LLAMA_BUILD/.pod-built-${sha:-<sha>}-sm$arch"
  fetched="$LLAMA_BUILD/.pod-prebuilt-${sha:-<sha>}-sm$arch"
  if exists -x "$LLAMA_BUILD/bin/llama-bench" && exists -f "$fetched"; then llama_path prebuilt "already in place for sm_$arch"; return; fi
  if exists -x "$LLAMA_BUILD/bin/llama-bench" && exists -f "$built"; then llama_path compiled "already built for sm_$arch"; return; fi
  if llama_prebuilt "$arch" "$commit"; then
    run rm -f "$built"; run touch "$fetched"
    llama_path prebuilt "$LLAMA_PREBUILT_URL"
    return
  fi
  note "prebuilt llama.cpp not used: $LLAMA_WHY; compiling for sm_$arch"
  nvcc="$(nvcc_bin || true)"
  [[ -n "$nvcc" ]] || ((DRY)) || fail "no nvcc to compile llama.cpp (the prebuilt was refused: $LLAMA_WHY): install the CUDA toolkit or put /usr/local/cuda/bin on PATH"
  run cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=ON "-DCMAKE_CUDA_ARCHITECTURES=$arch" "-DCMAKE_CUDA_COMPILER=${nvcc:-/usr/local/cuda/bin/nvcc}"
  run cmake --build "$LLAMA_BUILD" --config Release --parallel "${JOBS:-$(cpus)}" --target llama-bench llama-batched-bench
  run rm -f "$fetched"; run touch "$built"
  llama_path compiled "$LLAMA_WHY"
}

# llama_path prebuilt|compiled WHY: say which llama.cpp step 4 left (step 4's summary greps this line) and record it
llama_path() {
  echo "llama.cpp: $1 ($2)"
  ((DRY)) || facts_set llama_cpp "path=$1" "ref=$LLAMA_REF" "why=$2"
}

# manifest KEY: a value of the unpacked archive's PREBUILT.txt (tools/build-llama-prebuilt.sh writes it)
manifest() { sed -n "s/^$1=//p" "$WORK/tmp/llama-prebuilt/PREBUILT.txt" 2>/dev/null | head -1; }

# llama_prebuilt ARCH COMMIT: fetch the prebuilt llama.cpp, check it and move its bin/ into LLAMA_BUILD. Each check
# that fails returns 1 with LLAMA_WHY saying which, and nothing in LLAMA_BUILD is touched: a tampered archive (sha256),
# one for another commit, one without this GPU's architecture, a driver older than its CUDA allows, or libraries that
# do not load here (llama-bench --list-devices must list a CUDA device).
LLAMA_WHY=""
llama_prebuilt() {
  local arch="$1" commit="$2" tgz="$WORK/tmp/llama-prebuilt.tar.gz" dir="$WORK/tmp/llama-prebuilt" got drv floor out a
  LLAMA_WHY=""
  [[ "$LLAMA_PREBUILT" != 0 ]] || { LLAMA_WHY="LLAMA_PREBUILT=0"; return 1; }
  [[ "$LLAMA_PREBUILT_SHA256" =~ ^[0-9a-f]{64}$ ]] || { LLAMA_WHY="no sha256 recorded for LLAMA_REF $LLAMA_REF (set LLAMA_PREBUILT_SHA256)"; return 1; }
  if ((DRY)); then
    echo "+ curl -fsSL --retry 5 -o $tgz $LLAMA_PREBUILT_URL"
    echo "+ check: sha256 $LLAMA_PREBUILT_SHA256; PREBUILT.txt names the checkout's commit, sm_$arch and a driver floor this machine meets;"
    echo "         bin/llama-bench --list-devices lists CUDA0; then bin/ -> $LLAMA_BUILD/bin. Any check failing: compile, as below"
    LLAMA_WHY="dry run"; return 1
  fi
  rm -rf "$dir" "$tgz"; mkdir -p "$dir"
  run curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 -o "$tgz" "$LLAMA_PREBUILT_URL" \
    || { LLAMA_WHY="no archive at $LLAMA_PREBUILT_URL"; return 1; }
  got="$(sha256sum "$tgz" | cut -d' ' -f1)"
  if [[ "$got" != "$LLAMA_PREBUILT_SHA256" ]]; then rm -f "$tgz"; LLAMA_WHY="refused: sha256 $got is not the recorded $LLAMA_PREBUILT_SHA256"; return 1; fi
  note "sha256 verified: $got"
  run tar -xzf "$tgz" -C "$dir" --strip-components=1 || { LLAMA_WHY="the archive does not unpack"; return 1; }
  rm -f "$tgz"
  [[ "$(manifest llama_commit)" == "$commit" ]] || { LLAMA_WHY="the archive is llama.cpp $(manifest llama_commit), the checkout is $commit"; return 1; }
  for a in ${arch//;/ }; do
    [[ " $(manifest archs) " == *" $a "* || " $(manifest archs) " == *" ${a}a "* ]] || { LLAMA_WHY="it covers sm $(manifest archs), not sm_$a"; return 1; }
  done
  drv="$(driver_major || true)"; floor="$(manifest driver_floor)"
  [[ -n "$drv" && -n "$floor" ]] && ((drv >= floor)) \
    || { LLAMA_WHY="CUDA $(manifest cuda) needs driver $floor or newer; this machine has ${drv:-none}"; return 1; }
  out="$("$dir/bin/llama-bench" --list-devices 2>&1)" || { LLAMA_WHY="llama-bench --list-devices failed: $(tail -n 2 <<<"$out" | tr '\n' ' ')"; return 1; }
  grep -q CUDA0 <<<"$out" || { LLAMA_WHY="llama-bench lists no CUDA device: $(tr '\n' ' ' <<<"$out")"; return 1; }
  sed 's/^/   | /' <<<"$out"
  run rm -rf "$LLAMA_BUILD/bin"
  run mkdir -p "$LLAMA_BUILD"
  run mv "$dir/bin" "$LLAMA_BUILD/bin"
  run mv "$dir/PREBUILT.txt" "$LLAMA_BUILD/PREBUILT.txt"
  rm -rf "$dir"
}

# fetch_model BYTES: MODEL_URL into $MODEL.download, in MODEL_STREAMS byte ranges at once when the file is large.
# A range that fails, or a server that answers a range with another length, falls back to one plain curl; the caller
# checks the sha256 of whatever arrived, so a wrong join cannot be installed.
fetch_model() {
  local size="$1" n="$MODEL_STREAMS" i a b ok=1 pids=() t0; t0=$(date +%s)
  if ((n > 1 && size >= MODEL_RANGE_MIN)); then
    local part=$(( (size + n - 1) / n ))
    note "downloading in $n ranges of up to $part bytes"
    for ((i = 0; i < n; i++)); do
      a=$((i * part)); b=$(( a + part - 1 )); ((b < size)) || b=$((size - 1))
      curl -fsSL --retry 5 --retry-all-errors -r "$a-$b" -o "$MODEL.part$i" "$MODEL_URL" & pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i" || ok=0; done
    : > "$MODEL.download"
    for ((i = 0; i < n && ok; i++)); do  # joined one range at a time, each removed as it is appended
      a=$((i * part)); b=$(( a + part - 1 )); ((b < size)) || b=$((size - 1))
      [[ "$(wc -c < "$MODEL.part$i" | tr -d ' ')" == "$((b - a + 1))" ]] || { ok=0; break; }
      cat "$MODEL.part$i" >> "$MODEL.download"; rm -f "$MODEL.part$i"
    done
    rm -f "$MODEL".part*
    ((ok)) || note "a range failed or came back another length: downloading in one stream"
  else ok=0; fi
  ((ok)) || run curl -fL --retry 5 --retry-all-errors -o "$MODEL.download" "$MODEL_URL"
  local s=$(( $(date +%s) - t0 )); note "downloaded $size bytes in ${s}s ($(( size / 1048576 / (s > 0 ? s : 1) )) MB/s)"
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
      echo "+ curl -fL --retry 5 -o $MODEL $MODEL_URL   # in $MODEL_STREAMS byte ranges when the file is 64 MiB or more"
      echo "+ sha256sum $MODEL > $MODEL.sha256"
      return
    fi
    local have_size=0
    [[ -f "$MODEL" ]] && have_size="$(stat -Lc %s "$MODEL" 2>/dev/null || stat -Lf %z "$MODEL")"
    if [[ "$have_size" == "$size" ]]; then note "model already complete at $MODEL"; else
      fetch_model "$size"
      local downloaded; downloaded="$(sha256sum "$MODEL.download")"
      check_sha "${downloaded%% *}" "$MODEL_SHA256"
      run mv "$MODEL.download" "$MODEL"
    fi
  fi
  ((DRY)) && { echo "+ sha256sum $MODEL > $MODEL.sha256"; return; }
  if [[ ! -f "$MODEL.sha256" ]] || [[ "$MODEL" -nt "$MODEL.sha256" ]]; then
    run_sh 'sha256sum "$1" | awk '"'"'{print $1}'"'"' > "$2"' "$MODEL" "$MODEL.sha256"
  fi
  local actual; actual="$(cat "$MODEL.sha256")"
  note "model $MODEL $(stat -Lc %s "$MODEL" 2>/dev/null || stat -Lf %z "$MODEL") bytes"
  check_sha "$actual" "$MODEL_SHA256"
  facts_set model "path=$MODEL" "sha256=$actual" "url=$MODEL_URL"
}

# Step 5 runs in the background from before step 3 (curl and python are there after step 2), so the download overlaps
# the clones, the engine installs, the checks and the chip scan; step 9, the first step that reads the model, waits.
# A lock keeps a rerun's download from writing the file a failed run left downloading: it waits for that one instead.
MODEL_PID=""
MODEL_LOG="$WORK/logs/step5-model.log"
model_start() {
  mkdir -p "$WORK/logs" "$KEEP/models"
  (
    if have flock; then exec 9>"$KEEP/models/.$MODEL_NAME.lock"; flock 9; fi
    t0=$(date +%s)
    step_model
    printf '%s\t%s\t%s\t%s\n' 5 model "$(( $(date +%s) - t0 ))" "$(date +%FT%T)" >> "$TIMINGS"
  ) > "$MODEL_LOG" 2>&1 &
  MODEL_PID=$!
  note "step 5 (model) downloads in the background: $MODEL_LOG; step 9 waits for it"
}
model_wait() {
  [[ -n "$MODEL_PID" ]] || return 0
  local pid="$MODEL_PID" rc=0 t0; MODEL_PID=""; t0=$(date +%s)
  note "waiting for the model (step 5, in the background)"
  wait "$pid" || rc=$?
  tr '\r' '\n' < "$MODEL_LOG" | grep -vE '^ *[0-9]|% Total|Dload' | tail -n 8 | sed 's/^/   | /' || true
  ((rc == 0)) || { current=5; fail "the model download failed (log above, full log $MODEL_LOG); rerun --from 5"; }
  note "model ready; waited $(( $(date +%s) - t0 ))s"
}

step_env() {
  local nsys_dir="" ; local nsys
  nsys="$(nsys_bin || true)"; [[ -n "$nsys" ]] && nsys_dir="$(dirname "$nsys")"
  local text  # BOLTBEAM_RUNS: the TUI saves where step 9 writes and boltbeam fetch looks, not under the checkout
  text="$(
    printf '# BoltBeam pod environment, written by tools/pod-setup.sh\n'
    # the venv first: nsys can live in /usr/local/bin, whose python would shadow the venv python in an interactive shell
    printf 'export PATH=%q:"$PATH"\n' "$VENV/bin:/usr/local/cuda/bin:/usr/local/go/bin${nsys_dir:+:$nsys_dir}"
    printf 'export %s=%q\n' \
      BOLTBEAM_LLAMA_BENCH "$LLAMA_BUILD/bin/llama-bench" \
      BOLTBEAM_LLAMA_BATCHED_BENCH "$LLAMA_BUILD/bin/llama-batched-bench" \
      BOLTBEAM_GGML_CUDA_SRC "$LLAMA_DIR/ggml/src/ggml-cuda" \
      BOLTBEAM_CHIPS_DIR "$WORK/chips" TMPDIR "$WORK/tmp" BOLTBEAM_CUDA_CACHE "$WORK/tmp/boltbeam-cuda" \
      BOLTBEAM_MODEL "$MODEL" BOLTBEAM_REPO "$BOLTBEAM_DIR" \
      BOLTBEAM_RUNS "$RUNS"
    [[ -z "$TINYGRAD_REF" ]] || printf 'export BOLTBEAM_TINYGRAD_ROOT=%q\n' "$TINYGRAD_DIR"
    if wants vllm && { ((DRY)) || compgen -G "$VLLM_VENV/.pod-vllm-*" >/dev/null; }; then printf 'export BOLTBEAM_VLLM_PYTHON=%q\n' "$VLLM_VENV/bin/python"; fi
    if wants tensorrt-llm && { ((DRY)) || compgen -G "$TRTLLM_VENV/.pod-trtllm-*" >/dev/null; }; then printf 'export BOLTBEAM_TRTLLM_PYTHON=%q\n' "$TRTLLM_VENV/bin/python"; fi
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
  run_sh '"$1" -m boltbeam.workflow.screen autoscan > "$2"' "$VENV/bin/python" "$WORK/autoscan.json"
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
  model_wait
  load_env; read_target
  local mode run_dir
  for mode in in-model generic; do
    run_dir="$RUNS/$RUN_TAG-$mode$LSUF"
    if [[ "$mode" == in-model ]] && ! nsys_bin >/dev/null && ! ((DRY)); then
      note "in-model skipped: no nsys on this machine (step 2 installs it as root)"; continue
    fi
    if exists -f "$run_dir/.pod-done"; then note "$run_dir already measured (remove .pod-done to run it again)"; continue; fi
    run_sh 'cd "$1" && "$2" -m boltbeam.workflow.screen pipeline "$3" --run "$4" --target "$5" --id "$6" --provider llama.cpp --analyze --measure auto --role-time "$7" --layout "$8" --no-search' "$WORK/tmp" "$VENV/bin/python" "$MODEL" "$run_dir" "$TARGET" "$RUN_TAG-$mode$LSUF" "$mode" "$LAYOUT"
    run touch "$run_dir/.pod-done"
  done
}

step_results() {
  load_env
  local mode run_dir
  for mode in in-model generic; do
    run_dir="$RUNS/$RUN_TAG-$mode$LSUF"
    if ! exists -d "$run_dir" && ! ((DRY)); then note "$run_dir: no run (in-model needs nsys)"; continue; fi
    run_sh '"$1" -m boltbeam.workflow.screen results --run "$2" > "$3"' "$VENV/bin/python" "$run_dir" "$run_dir/results.json"
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
  run_sh 'cd "$1" && { find "runs/$2-in-model" "runs/$2-generic" -type f \( -name "*.json" -o -name "*.html" -o -name "*.md" -o -name "*.txt" -o -name "*.log" -o -name "*.csv" \) 2>/dev/null; ls facts.json autoscan.json 2>/dev/null; find logs -type f; } > "$3"' "$WORK" "$RUN_TAG" "$list"
  run tar -czf "$out" -C "$WORK" -T "$list"
  ((DRY)) || note "bundle: $out ($(du -h "$out" | cut -f1))"
  if [[ "$KEEP" != "$WORK" ]]; then run mkdir -p "$KEEP/bundles"; run cp "$out" "$KEEP/bundles/"; fi
  local host="${RUNPOD_PUBLIC_IP:-HOST}" port="${RUNPOD_TCP_PORT_22:-PORT}"
  note "fetch it:  scp -P $port root@$host:$out ."
  note "or, from the BoltBeam checkout on your machine:  tools/pod-fetch.sh $host $port"
}

# --- main --------------------------------------------------------------------------------------------------------

[[ -n "$RUN_TAG" ]] || RUN_TAG="$(gpu_tag)"
current=0
say "pod-setup $( ((DRY)) && echo '(dry run) ' )WORK=$WORK  KEEP=$KEEP"
note "MODEL_URL=$MODEL_URL"
note "MODEL_SHA256=${MODEL_SHA256:-<none>}  BOLTBEAM=$BOLTBEAM_REPO@$BOLTBEAM_REF  LLAMA=$LLAMA_REPO@$LLAMA_REF"
note "TINYGRAD=$( [[ -n "$TINYGRAD_REF" ]] && echo "$TINYGRAD_REPO@$TINYGRAD_REF" || echo skipped )"
note "ENGINES=${ENGINES:-<none>}  VLLM_VERSION=${VLLM_VERSION:-<pin>}  TRTLLM_VERSION=${TRTLLM_VERSION:-<pin>}  VLLM_CUDA=${VLLM_CUDA:-<from the driver>}"
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
  trap 'rc=$?; if ((rc && current > 0)); then echo "FAILED in step $current (${STEPS[$((current - 1))]}); fix it and rerun: $0 --from $current" >&2
    [[ -z "$MODEL_PID" ]] || echo "   the model download goes on in the background ($MODEL_LOG); a rerun from step 5 or earlier waits for it" >&2; fi; exit $rc' EXIT
fi

fns=(step_facts step_packages step_boltbeam step_engines step_model step_env step_check step_autoscan step_pipeline step_results step_bundle)
model_bg=0; ((!DRY && FROM <= 5 && TO >= 5)) && model_bg=1   # step 5 in range: it goes to the background before step 3
for i in "${!fns[@]}"; do
  current=$((i + 1))
  ((current < FROM || current > TO)) && continue
  if ((model_bg == 1 && current >= 3)); then model_bg=2; model_start; fi
  if ((current == 5 && model_bg == 2)); then say "step 5: model  (in the background: $MODEL_LOG)"; continue; fi
  ((current == 5 && DRY)) && note "(a real run starts this step in the background before step 3; step 9 waits for it)"
  begin_step "$current"
  "${fns[$i]}"
  end_step "$current"
done
model_wait   # --to before step 9: the model is still finished and checked before the run ends
if ((DRY)); then say "dry run complete: nothing was installed, cloned, downloaded or measured"
elif ((TO < 11)); then say "done: steps $FROM to $TO; continue with $0 --from $((TO + 1))"
else say "done: report bundle under $WORK"; fi
