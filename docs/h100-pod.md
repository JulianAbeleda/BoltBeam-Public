# BoltBeam on a rented H100

One command takes a rented pod from its bare image to BoltBeam's first report on the H100: the speed limit for
Qwen3-8B, the measured token, the per-role table from the real token (nsys) and from BoltBeam's own kernel
timer, and the tie-out between them. [linux.md](linux.md) is the same setup done by hand; the script follows it
step by step.

## What to rent

- RunPod, Secure Cloud, one **H100 SXM 80 GB**. One GPU first: the multi-GPU session comes later (below).
- The PyTorch template (a `-devel` CUDA image: it carries `nvcc`, which the kernel timer compiles with). The
  script checks in its first step and says what is missing.
- A **40 GB volume** mounted at `/workspace`. The script keeps everything there (`/workspace/boltbeam`), so a
  stopped pod keeps its model, its llama.cpp build and its runs; a restart continues where it left off.
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
| 2 packages | apt, only what is missing, only as root: cmake, git, build-essential, python3-venv, curl; nsys from the NVIDIA apt repo (`nsight-systems-cli`) when the image has none | 1 to 3 min |
| 3 boltbeam | the public mirror at `main`, a venv under `/workspace/boltbeam`, `pip install .` plus numpy; the tinygrad fork beside it (branch `exp`) when it is public | 1 to 2 min |
| 4 llama.cpp | the pinned commit (`LLAMA_REF`, the one the RTX 5090 record used), `cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`, `CMAKE_CUDA_ARCHITECTURES` from the GPU's compute capability (90 on an H100), `llama-bench` and `llama-batched-bench` built | 5 to 10 min |
| 5 model | Qwen3-8B-Q4_K_M.gguf (5.0 GB) from Hugging Face with resume; sha256 computed and compared to the known digest | 2 to 5 min |
| 6 env | `env.sh`: PATH with the CUDA bin folder and nsys, `BOLTBEAM_LLAMA_BENCH`, `BOLTBEAM_GGML_CUDA_SRC`, `TMPDIR` and `BOLTBEAM_CUDA_CACHE` on the volume | seconds |
| 7 check | `boltbeam doctor` (when the ref has it), then `boltbeam selfcheck` | under 1 min |
| 8 autoscan | the H100 is not in the registry, so BoltBeam measures its read bandwidth and matrix rate and keeps the profile under `/workspace/boltbeam/chips`; the profile id is the target of the runs | about 1 min |
| 9 pipeline | `--role-time in-model` (two nsys captures of llama-bench), then `--role-time generic` (the engine's `mul_mat_vec_q` compiled by nvcc and timed alone); both with `--analyze --no-search` | 1 to 2 min each |
| 10 results | `results.json` per run; the per-role table and the tie-out printed to the log | seconds |
| 11 bundle | `boltbeam-h100-<date>.tgz`: both runs' JSON, `report.html`, `summary.md`, the facts and the logs; no nsys binaries | seconds |

Settings are environment variables with defaults (`WORK`, `MODEL_URL`, `MODEL_SHA256`, `BOLTBEAM_REF`,
`LLAMA_REF`, `TINYGRAD_REF`, `CUDA_ARCH`, `RUN_TAG`, `JOBS`); the script's header lists them.

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
[in-model-vs-generic-rtx5090-20261010.md](in-model-vs-generic-rtx5090-20261010.md) is the same pair of runs on
the RTX 5090, the number to set the H100's beside.

## Cost

Budget about an hour of pod time: the steps above take 15 to 25 minutes end to end on a fresh pod (the
llama.cpp build and the model download are most of it), plus the pod's own start and the fetch. A second run on
the same volume skips the build and the download and takes a few minutes. Stop the pod when the bundle is fetched;
a stopped pod bills for the volume only.

## Later: more than one GPU

`--multi` is refused today ("not yet"). The multi-GPU session will add: a pod with 2 to 8 H100s, `facts.json`
with one row per GPU and the links between them, the pipeline's `--layout layer` and `--layout row` (the model
split across the GPUs, as `measure_status.json` records it), and a bundle per layout. Nothing in the single-GPU
steps changes for it.
