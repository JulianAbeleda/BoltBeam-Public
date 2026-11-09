# BoltBeam on rented GPUs (pods)

One command takes a rented pod (RunPod or Vast.ai; first an H100, since then H200s and RTX 3090s) from its bare
image to BoltBeam's first report: the speed limit for
Qwen3-8B, the measured token, the per-role table from the real token (nsys) and from BoltBeam's own kernel
timer, and the tie-out between them. [linux.md](linux.md) is the same setup done by hand; the script follows it
step by step.

## What to rent

- RunPod, one **H100 SXM 80 GB** (Community Cloud costs less and can be reclaimed). A pod with more GPUs runs the
  same steps; `LAYOUT=layer` or `LAYOUT=row` picks how step 9 uses them (below).
- The PyTorch template (a `-devel` CUDA image: it carries `nvcc`, which the kernel timer compiles with). The
  script checks in its first step and says what is missing.
- A **container disk of 80 GB** and a volume at `/workspace` sized for the models. The builds, venvs and runs go on
  the container's own disk (`WORK=/root/boltbeam`): `/workspace` is a network filesystem, and the engines' many
  small files took minutes per step there (RunPod, 2026-11-08). Models and the bundles go on the volume
  (`KEEP=/workspace`), so a stopped pod keeps what was slow to download; the container disk is erased when the pod
  stops, so a restart rebuilds the engines (about 10 minutes) and reuses the models.
- A host on **CUDA 13** (driver 580 or newer): in the console's GPU filter, pick CUDA version 13.0 or higher.
  vLLM 0.31.0 and TensorRT-LLM 1.2.1 are built for it. On an older driver step 4 installs vLLM's CUDA 12.9 build
  and skips TensorRT-LLM, saying why; llama.cpp is unaffected.
- **SSH on** ("SSH over exposed TCP"): the pod page then shows `ssh root@HOST -p PORT`, and `scp` works for the
  bundle.

## The one command

On the pod:

```bash
curl -fsSL https://raw.githubusercontent.com/JulianAbeleda/BoltBeam-Public/main/tools/pod-setup.sh -o pod-setup.sh
bash pod-setup.sh
```

`bash pod-setup.sh --dry-run` first prints every command it would run with its resolved values, checks the
model URL exists (a HEAD request: the size and the server's sha256), checks the three refs exist, and exits.
It installs, downloads and measures nothing.

The script runs eleven steps in order, each idempotent (a second run finds what the first left and skips it).
`--from 7` resumes at step 7, `--to 6` stops after 6, and a failure says which step to resume at.

| step | what | about |
|---|---|---|
| 1 facts | nvidia-smi, nvcc, nsys, CUDA version, free disk, written to `facts.json` | seconds |
| 2 packages | apt, only what is missing, only as root: cmake, git, build-essential, python3-venv, curl; libopenmpi-dev for TensorRT-LLM; nsys from NVIDIA's apt repo when the image has none (the image's own entry for that repo is used when it has one: a second entry with another signing key stops apt; where `nsight-systems-cli` is absent, the newest `nsight-systems-YYYY.x`) | 1 to 3 min |
| 3 boltbeam | the public mirror at `main`, a venv under `$WORK`, `pip install -e .` plus numpy (editable: a pull or a fix copied into the checkout is what the pipeline runs); Go at the version `tui/go.mod` names, for `boltbeam tui`; the in-house tinygrad engine only when asked (`TINYGRAD_REF=exp`): a rented pod measures the mature engines, and the fork's clone took minutes | 1 to 2 min |
| 4 engines | llama.cpp at the pinned commit (`LLAMA_REF`, the one the RTX 5090 record used), `cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`, `CMAKE_CUDA_ARCHITECTURES` from the GPU's compute capability (90 on an H100), `llama-bench` and `llama-batched-bench` built. Then vLLM and TensorRT-LLM at their pins in `boltbeam/data/engine_pins.json`, each in its own venv under `$WORK/engines` (they need different torch builds); TensorRT-LLM's wheels come from NVIDIA's index, with `pyarrow<20` (a newer one needs NumPy 2, which TensorRT-LLM's pins refuse). The three install at the same time, each logging to `$WORK/logs/engine-<name>.log`. `ENGINES` picks which (default: all three; Ollama is not installed). A failed llama.cpp stops the script; a failed optional engine is noted and the run goes on | 5 to 10 min for all three |
| 5 model | Qwen3-8B-Q4_K_M.gguf (5.0 GB) from Hugging Face with resume; sha256 computed and compared to the known digest | 2 to 5 min |
| 6 env | `env.sh`: PATH with the venv first, then the CUDA bin folder, Go and nsys; `BOLTBEAM_REPO` for the TUI; `BOLTBEAM_LLAMA_BENCH`, `BOLTBEAM_GGML_CUDA_SRC`, `TMPDIR` and `BOLTBEAM_CUDA_CACHE` under `$WORK`, and `BOLTBEAM_VLLM_PYTHON` and `BOLTBEAM_TRTLLM_PYTHON` for the engines step 4 installed, so `boltbeam doctor` finds them | seconds |
| 7 check | `boltbeam doctor` (when the ref has it), then `boltbeam selfcheck` | under 1 min |
| 8 autoscan | the H100 is not in the registry, so BoltBeam measures its read bandwidth and matrix rate and keeps the profile under `$WORK/chips`; the profile id is the target of the runs | about 1 min |
| 9 pipeline | `--role-time in-model` (two nsys captures of llama-bench), then `--role-time generic` (the engine's `mul_mat_vec_q` compiled by nvcc and timed alone); both with `--analyze --no-search` | 1 to 2 min each |
| 10 results | `results.json` per run; the per-role table and the tie-out printed to the log | seconds |
| 11 bundle | `boltbeam-h100-<date>.tgz`: both runs' JSON, `report.html`, `summary.md`, the facts and the logs; no nsys binaries | seconds |

Settings are environment variables with defaults (`WORK`, `MODEL_URL`, `MODEL_SHA256`, `BOLTBEAM_REF`,
`LLAMA_REF`, `TINYGRAD_REF`, `ENGINES`, `VLLM_VERSION`, `TRTLLM_VERSION`, `VLLM_CUDA`, `TRTLLM_PYTHON`, `CUDA_ARCH`,
`RUN_TAG`, `JOBS`, `LAYOUT`); the script's header lists them.

Steps 9 and 10 measure with llama.cpp on the GGUF. vLLM and TensorRT-LLM are installed so a session on the pod can
use them (they take Hugging Face checkpoints, not the GGUF); the script does not yet run them itself.

## What comes back

The script ends by printing the bundle's path and the `scp` line. From a BoltBeam checkout on your own machine:

```bash
tools/pod-fetch.sh HOST PORT
```

fetches the newest bundle and unpacks it under `~/env/boltbeam-runs/.work/h100-<date>/`:

```text
runs/h100-in-model/   results.json, report.html, summary.md, the timing and role traces, kernel_compare/
runs/h100-generic/    the same, timed by BoltBeam's kernel timer
facts.json            the GPU, nvcc, nsys, the model's sha256
autoscan.json         the chip profile the runs used
logs/                 pod-setup-<time>.log (every command and the printed tables), timings.tsv (seconds per step)
```

The per-role table in `results.json` (`loss.roles`) carries, for each weight role, µs per call, the same less the
dispatch floor, GB/s, the percent of the measured peak and the reason word; `loss.tie_out.lines` is the tie-out
against the token. [kernel-timer.md](kernel-timer.md) says how to read them, and
`in-model-vs-generic-rtx5090-20261010.md` is the same pair of runs on
the RTX 5090, the number to set the H100's beside.

## Cost

Budget about an hour of pod time: the steps above take 15 to 25 minutes end to end on a fresh pod (the
llama.cpp build and the model download are most of it), plus the pod's own start and the fetch. A second run on
the same volume skips the model download. Stop the pod when the bundle is fetched; a stopped pod bills for the
volume only ($0.20/GB a month on RunPod, twice the running rate).

## Setup time: do it in parallel (to do)

On rented and third-party GPUs the setup costs more than the measuring, so it has to run in parallel from now on:
nothing that can overlap waits for something else. Measured per step (`$WORK/logs/timings.tsv`, 2026-11-08/09):

| Step | Vast RTX 3090 | RunPod H200 | RunPod H100 |
|---|---|---|---|
| 2 packages (apt, nsys) | 340 s | 23 s | 0 s |
| 4 engines (llama.cpp build, vLLM, TensorRT-LLM) | 1,287 s | 546 s | about 300 s |
| 5 model download | 47 s | 74 s | 110 s |
| 8 autoscan (chip measurement) | 66 s | 59 s | 61 s |

Step 4 is most of it, and a cheap host makes it three times longer (slow CPU for the llama.cpp build, slow link for
the wheels). The quick wins, none of which needs a new image:

1. **Overlap the model download with everything else.** Start step 5 in the background at step 1; step 9 waits for
   it. Today it starts only after the engines are installed.
2. **Install the Python engines with `uv`, not pip.** vLLM and TensorRT-LLM are multi-GB wheel sets; uv downloads and
   installs them in parallel.
3. **Stop compiling llama.cpp on the pod.** Build it once per pin for every GPU we rent (`CMAKE_CUDA_ARCHITECTURES`
   80;86;89;90;100;120), keep the build as one archive, and have step 4 fetch it instead of compiling. Compile only
   when the archive does not cover the GPU.
4. **Rent hosts that set up fast.** In the Vast offer search, require a fast download link and enough CPU cores, not
   only the price; the 3090's slow host is why its step 4 took 21 minutes.
5. **Keep a machine between steps.** Pause it (stop, not destroy) when the next test can use it: its engines and
   models stay. Fetch results first.
6. **Big downloads on big machines:** try `HF_XET_HIGH_PERFORMANCE=1` with `hf download` (Hugging Face documents it
   for fast links and 64 GB of RAM or more; reports are mixed, so time it against the default once).

Later, if we rent often enough that the rest still matters: one prebuilt image (BoltBeam, the pinned engines,
llama.cpp for every GPU above, nsys, Python headers, OpenMPI), built on Vast's base image so hosts' cached layers
pull fast. Setup would then be the pull, the health check, autoscan and the model. The image is large (25 to 35 GB),
so it only pays on hosts with a fast link.

## Limits on a rented pod

RunPod's hosts keep the GPU's performance counters for admins (`RmProfilingAdminOnly: 1`; `ncu` says
`ERR_NVGPUCTRPERM` even as root in the container). vLLM's role timing reads which matrix path a closed kernel took
from those counters, so on such a host a model whose decode runs a kernel no library holds readable code for is
refused with "closed decode kernels require accessible instruction counters". cuBLASLt's Hopper `nvjet_*` GEMMs
are such kernels: `cuobjdump` extracts none of them from `libcublasLt`. Qwen3.8-27B-AWQ's lm_head runs one; its
tokens per second are still measured (stage measure_timing), only the per-role table is refused.

## On Vast.ai

The same script runs on a Vast.ai instance (a plain container on a community host); what differed there
(RTX 3090s, 2026-11-08):

- Rent with `vastai create instance OFFER --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk N --ssh --direct`.
  A reply of `success: false` still creates an instance, stopped, that never starts (its host was taken): destroy
  it (`vastai destroy instance ID -y`; without `-y` it asks and aborts) and rent another offer.
- Some hosts write `/root/.ssh/authorized_keys` with ownership sshd rejects ("bad ownership or modes", in `vastai
  logs ID`), and every key is refused, the proxy's too. Rent with an `--onstart-cmd` that keeps `/root` and
  `/root/.ssh` at 700 and the key file at 600, owned by root, for the first minutes.
- NVIDIA's CUDA image has no Python headers; step 2 now installs `python3-dev` (Triton compiles against `Python.h`).
- `search offers` labels China `..., CN`; Hugging Face downloads from there are slow. Filter on the label.
- There is no `/workspace`: `KEEP` falls back to `WORK`. Fetch models with `hf download` (several streams over
  Hugging Face's Xet storage; step 5's single curl ran at about 1.5 MB/s; `hf_transfer` is gone from current
  huggingface_hub) and point step 5 at one with `MODEL_URL=file://...` and its `MODEL_SHA256`.

## More than one GPU

The pipeline takes `--layout one|layer|row` (limited support): llama.cpp split by layer and vLLM split by rows are
measured with a limit, and the report is written without the per-role split. On 2x R9700: llama.cpp split by layer
26.8 tok/s; vLLM split by rows, 27B AWQ, 49.8 tok/s against a limit of 73.5. Split by rows needs a measured all-reduce
in the machine's facts before it has a limit. On a multi-GPU pod the script runs the same steps, and `LAYOUT` (one,
layer or row) is passed to step 9 as `--layout`.
