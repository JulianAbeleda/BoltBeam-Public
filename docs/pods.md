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
| 4 engines | llama.cpp at the pinned commit (`LLAMA_REF`, the one the RTX 5090 record used): the clone (its ggml-cuda source is what role timing reads), then the prebuilt binaries for that commit (below) when they fit the machine, else `cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF` with `CMAKE_CUDA_ARCHITECTURES` from the GPU's compute capability (90 on an H100), `llama-bench` and `llama-batched-bench` built; `facts.json` `llama_cpp` says which and why. Then vLLM and TensorRT-LLM at their pins in `boltbeam/data/engine_pins.json`, each in its own venv under `$WORK/engines` (they need different torch builds), installed by `uv` (pip when uv fails); TensorRT-LLM's wheels come from NVIDIA's index, with `pyarrow<20` (a newer one needs NumPy 2, which TensorRT-LLM's pins refuse). The three install at the same time, each logging to `$WORK/logs/engine-<name>.log`. `ENGINES` picks which (default: llama.cpp and vLLM; `ENGINES="llama.cpp vllm tensorrt-llm"` adds TensorRT-LLM, whose wheels were most of step 4 on a slow 3090 host, 2026-11-09; Ollama is not installed). A failed llama.cpp stops the script; a failed optional engine is noted and the run goes on | 5 to 10 min for all three |
| 5 model | Qwen3-8B-Q4_K_M.gguf (5.0 GB) from Hugging Face, fresh; sha256 computed and compared to the known digest. It runs in the background from before step 3 (`$WORK/logs/step5-model.log`, its own row in `timings.tsv`); step 9 waits for it and stops with the log if it failed | 1 to 2 min, overlapped |
| 6 env | `env.sh`: PATH with the venv first, then the CUDA bin folder, Go and nsys; `BOLTBEAM_REPO` for the TUI; `BOLTBEAM_LLAMA_BENCH`, `BOLTBEAM_GGML_CUDA_SRC`, `TMPDIR` and `BOLTBEAM_CUDA_CACHE` under `$WORK`, and `BOLTBEAM_VLLM_PYTHON` and `BOLTBEAM_TRTLLM_PYTHON` for the engines step 4 installed, so `boltbeam doctor` finds them | seconds |
| 7 check | `boltbeam doctor` (when the ref has it), then `boltbeam selfcheck` | under 1 min |
| 8 autoscan | the H100 is not in the registry, so BoltBeam measures its read bandwidth and matrix rate and keeps the profile under `$WORK/chips`; the profile id is the target of the runs | about 1 min |
| 9 pipeline | `--role-time in-model` (two nsys captures of llama-bench), then `--role-time generic` (the engine's `mul_mat_vec_q` compiled by nvcc and timed alone); both with `--analyze --no-search` | 1 to 2 min each |
| 10 results | `results.json` per run; the per-role table and the tie-out printed to the log | seconds |
| 11 bundle | `boltbeam-<tag>-<date>.tgz` (the tag is `RUN_TAG`, by default from the GPUs: `h100`, `rtx3090`, `8xrtx4090`): both runs' JSON, `report.html`, `summary.md`, the facts and the logs; no nsys binaries | seconds |

Settings are environment variables with defaults (`WORK`, `MODEL_URL`, `MODEL_SHA256`, `BOLTBEAM_REF`,
`LLAMA_REF`, `LLAMA_PREBUILT`, `LLAMA_PREBUILT_URL`, `LLAMA_PREBUILT_SHA256`, `TINYGRAD_REF`, `ENGINES`, `VLLM_VERSION`, `TRTLLM_VERSION`, `VLLM_CUDA`, `TRTLLM_PYTHON`, `CUDA_ARCH`,
`RUN_TAG`, `JOBS`, `LAYOUT`); the script's header lists them.

Steps 9 and 10 measure with llama.cpp on the GGUF. vLLM is installed so a session on the pod can use it, and
TensorRT-LLM when `ENGINES` names it (they take Hugging Face checkpoints, not the GGUF); the script does not yet run
them itself.

## TensorRT-LLM on a pod

TensorRT-LLM is opt-in (`ENGINES`), needs driver 580 or newer (CUDA 13) and Python 3.10 or 3.12, and takes a Hugging
Face safetensors checkpoint. The recipe, from the bare image (on a pod already set up, add `--from 2 --to 6`: step 2
adds `libopenmpi-dev`, step 4 the engine at its pin in `$WORK/engines/tensorrt-llm`, step 6 writes
`BOLTBEAM_TRTLLM_PYTHON` into `env.sh`):

```bash
ENGINES="llama.cpp vllm tensorrt-llm" bash pod-setup.sh
source /root/boltbeam/env.sh
hf download Qwen/Qwen3-8B --local-dir /root/boltbeam/models/Qwen3-8B        # BF16 safetensors, 16 GB
TARGET="$(python3 -c 'import json; print(json.load(open("/root/boltbeam/autoscan.json"))["target_id"])')"
RUN="$BOLTBEAM_RUNS/h200-qwen3-8b-bf16-trtllm"
python -m boltbeam.workflow.screen pipeline /root/boltbeam/models/Qwen3-8B --run "$RUN" --target "$TARGET" \
  --id "$(basename "$RUN")" --provider tensorrt-llm --analyze --measure auto --role-time in-model --layout one --no-search
python -m boltbeam.workflow.screen results --run "$RUN" > "$RUN/results.json"
"$BOLTBEAM_REPO/tools/check_run_math.py" "$RUN"
```

`hf` is in vLLM's venv (`/root/boltbeam/engines/vllm/bin/hf`) when it is not on PATH. `boltbeam doctor` says whether
TensorRT-LLM was found and, if not, why (no MPI library, a wheel without this GPU's architecture). The same run from
the screen: `boltbeam`, pick the checkpoint folder, engine TensorRT-LLM, Measurement in-model, Run.

Measured with this recipe (Qwen3-8B BF16, the built-engine backend, 2026-11-09):

| GPU | tok/s | Of its limit | Per-role table |
|---|---|---|---|
| RTX 5090 (home) | 95.9 | 85.3% (limit 112.4) | 10 rows, `all figures check` |
| RTX 3090 (Vast) | 51.4 | 87% | not built: speed and limit only |

The built engine is saved under `$TMPDIR` (50b2d2a8): TensorRT-LLM's own cache root is a fixed `/tmp`, and an 8B's
engine (16 GB) filled a small root disk mid-save. Keep `/` clear of model downloads for the same reason: the H200
attempt at this row failed with "No space left on device" after a 62 GB checkpoint was put on the root disk, and has
not been rerun.

### A hybrid on TensorRT-LLM: Nemotron 3.5 Lightning

TensorRT-LLM 1.2.1 runs the NemotronH architecture (Mamba, MoE and attention layers) on its PyTorch backend only, so
the run is made with `BOLTBEAM_TRTLLM_BACKEND=pytorch`. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (62 GB)
needed three things on an H200 pod:

1. **The layer plan as a string.** TensorRT-LLM reads `hybrid_override_pattern` (M Mamba, E MoE, `*` attention, `-`
   MLP); the 3.5 checkpoint writes `layers_block_type`, a list, and the load stops at "'NemotronHConfig' object has no
   attribute 'hybrid_override_pattern'". A folder of links to the same weight files with a config that holds both:

   ```bash
   S=/root/boltbeam/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; D=/root/boltbeam/models/Nemotron-3.5-Lightning-30B-trt
   mkdir -p $D; for f in $S/*; do ln -sf $f $D/; done; rm $D/config.json
   python3 -c "
   import json; c = json.load(open('$S/config.json'))
   c['hybrid_override_pattern'] = ''.join({'mamba': 'M', 'moe': 'E', 'attention': '*', 'mlp': '-'}[t] for t in c['layers_block_type'])
   json.dump(c, open('$D/config.json', 'w'), indent=1)"
   ```

2. **Prefix reuse off.** The hybrid's recurrent-state cache refuses to start with block reuse on ("mamba hybrid cache
   requires block reuse to be disabled in KV cache config"). BoltBeam's driver starts the PyTorch backend with it off
   (d91f8a49); nothing to set.
3. **ninja and nvcc.** FlashInfer compiles its kernels at the first start: `apt-get install -y ninja-build` and
   `export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH`. The first run's two captures then differ by
   those builds ("the two captures' start-ups ran block kernels a different number of times ... a JIT or autotune
   cache filled in one capture and not the other"); the next run's captures match. Run the pipeline twice, or one
   request first.

Then the recipe above with `$D` as the model and `--provider tensorrt-llm`. Measured on the H200 (batch 1, three
runs): 264 to 269 tok/s at context 128 and 261 to 269 at 512. The per-role capture completes; the report then stops
at "Mamba2 state traffic and SSD work are not validated against engine evidence; the Gated DeltaNet formula cannot
supply this model's limit", so there is a speed and not yet a limit or a table for a Mamba hybrid. Its GGUF
(Nemotron 3.5 Bolt 74B) does not load in BoltBeam yet either (`ssm_in`/`ssm_out`: unknown block layout).

## What comes back

From your own machine, `boltbeam fetch root@HOST -p PORT` lists the pod's runs and `boltbeam fetch root@HOST -p PORT
--all` brings them into your saved runs, each file checked by sha256, with the chip profile autoscan made on the pod
([Bring results home](#bring-results-home)). The Saved runs page then shows them with their tokens per second and
share of the limit.

The script also ends by printing a bundle's path and the `scp` line. The bundle is the fallback (it also carries the
facts and the logs): from a BoltBeam checkout, `tools/pod-fetch.sh HOST PORT` fetches the newest bundle and unpacks
it under `~/env/boltbeam-runs/.work/<tag>-<date>/`:

```text
runs/<tag>-in-model/  results.json, report.html, summary.md, the timing and role traces, kernel_compare/
runs/<tag>-generic/   the same, timed by BoltBeam's kernel timer
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

Budget about an hour of pod time: the steps above took 15 to 25 minutes end to end on a fresh pod when the
llama.cpp build and the model download ran in turn (most of it); both are now off the critical path (above), plus the pod's own start and the fetch. A second run on
the same volume skips the model download. Stop the pod when the bundle is fetched; a stopped pod bills for the
volume only ($0.20/GB a month on RunPod, twice the running rate).

## Setup time: done in parallel

On rented and third-party GPUs the setup costs more than the measuring, so nothing that can overlap waits for
something else. Measured per step before the changes below (`$WORK/logs/timings.tsv`, 2026-11-08/09):

| Step | Vast RTX 3090 | RunPod H200 | RunPod H100 |
|---|---|---|---|
| 2 packages (apt, nsys) | 340 s | 23 s | 0 s |
| 4 engines (llama.cpp build, vLLM, TensorRT-LLM) | 1,287 s | 546 s | about 300 s |
| 5 model download | 47 s | 74 s | 110 s |
| 8 autoscan (chip measurement) | 66 s | 59 s | 61 s |

Step 4 was most of it, and a cheap host made it three times longer (slow CPU for the llama.cpp build, slow link for
the wheels). Built (2026-11-09, no new image needed):

1. **The model downloads in the background.** Step 5 starts before step 3 (curl and Python are there after step 2),
   logs to `$WORK/logs/step5-model.log` and writes its own `timings.tsv` row; step 9, the first step that reads the
   model, waits for it and stops with the log's tail if it failed ("rerun --from 5"). `--to` short of step 9 still
   waits before the script ends. A lock on the model file makes a rerun wait for a download a failed run left going
   rather than write the same file. Expected saving: the whole of step 5 (47 to 110 s), since steps 3, 4, 7 and 8
   take longer than the download.
2. **vLLM and TensorRT-LLM install with `uv`.** The one on PATH, else Astral's standalone installer under
   `$WORK/uv` (pip into a venv there if that fails). Same pins, same venvs (`python -m venv`, so pip is still in
   them), `uv pip install --python <venv> --index-strategy unsafe-best-match` (pip's index precedence across the
   extra indexes), cache in `$WORK/tmp/uv-cache` on the venvs' disk so files are hard-linked, not copied. A uv
   failure falls back to the venv's pip for that command. Both still install at the same time as llama.cpp.
   Expected: the wheel half of step 4 several times faster (uv fetches in parallel); not yet timed on a pod.
3. **llama.cpp is not compiled on the pod.** Step 4 clones the pin (its ggml-cuda source is what role timing reads),
   then fetches `LLAMA_PREBUILT_URL`, by default the release asset
   `https://github.com/JulianAbeleda/BoltBeam-Public/releases/download/llama-<LLAMA_REF>/llama.cpp-<LLAMA_REF>-cuda-linux-x86_64.tar.gz`,
   and uses it only if every check passes: its sha256 is the one recorded in `tools/pod-setup.sh`
   (`LLAMA_PREBUILT_SHA256_PIN`, beside `LLAMA_REF_PIN`; a tampered or different asset is refused), its
   `PREBUILT.txt` names the checkout's commit and this GPU's architecture, the driver is at or above its floor, and
   `bin/llama-bench --list-devices` lists `CUDA0` (the libraries load and see the GPU). Any failure is said
   ("prebuilt llama.cpp not used: ...") and step 4 compiles as before. `facts.json` `llama_cpp` records `prebuilt`
   or `compiled` and why; `LLAMA_PREBUILT=0` always compiles; another `LLAMA_REF` has no recorded sha256 and
   compiles unless `LLAMA_PREBUILT_SHA256` is given. Without a compile the pod needs no `nvcc` for step 4.
   Expected: the llama.cpp part of step 4 from minutes (most of the 3090's 21) to the archive's download and checks.

The archive (`tools/build-llama-prebuilt.sh`; for pin 097cf33d9, b11571, the engine pin: 797 MB, sha256 `2b5e75859f32f4f88e09139038e75d8e7196a160039ea5657653ec7ca7547f8e`, CUDA 12.8.93, built 2026-11-09 on linuxbox; until then pods ran 50569eb87, b11433, older than the pin) holds `bin/`: `llama-bench`,
`llama-batched-bench`, `llama-server`, llama.cpp's shared libraries, and the CUDA runtime, cuBLAS and cuBLASLt it was
built with, plus libgomp; every ELF has RUNPATH `$ORIGIN`, so it runs from any folder. `libcuda` is the pod's
driver. It is built with CUDA 12.8 for `CMAKE_CUDA_ARCHITECTURES` 80;86;89;90;100;120 (A100, 3090, 4090, H100/H200,
B200, 5090; llama.cpp makes 120 `120a`), every architecture as ahead-of-time code.

- **Driver floor: 525.** A CUDA 12.x build runs on any driver from 525 on (CUDA's minor-version compatibility; nothing
  needs JIT). Blackwell GPUs need 570 or newer anyway. A CUDA 13 build would need 580, and rented hosts often run 550
  to 570, so the builder refuses nothing but warns on 13. The cuBLAS libraries are bundled because a pod on a CUDA 13
  image has no `libcublas.so.12`.
- **glibc floor: 2.38** (built on Ubuntu 24.04). An image with an older glibc (Ubuntu 22.04 is 2.35) fails
  the `--list-devices` check and compiles; the images in this document are Ubuntu 24.04.
- **A new pin:** on a Linux machine with a CUDA 12.8+ toolkit (or none: `CUDA_REDIST=12.8.1` downloads nvcc, cudart,
  cccl and cuBLAS from NVIDIA's redist index into `CUDA_HOME`, no root), change `LLAMA_REF_PIN`, run
  `CUDA_HOME=... OUT=... tools/build-llama-prebuilt.sh`, put the sha256 it prints into `LLAMA_PREBUILT_SHA256_PIN`,
  and upload the archive as the asset of a release `llama-<ref>` on the public mirror.
  Checked 2026-11-09: unpacked in an empty folder on the RTX 5090 with no `LD_LIBRARY_PATH`, `llama-bench -n 64`
  on the 27B Q4_K_M read 81.2 tok/s, as the installed build does.

Still to do:

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

## The TUI on a pod: check it works, and check its numbers

The TUI is the product, so every new GPU gets the TUI pass before its numbers are used, with the result written down
as pass or fail per step. The CLI pipeline (step 9) alone is not the check. Rent one GPU at a time: finish its pass,
fetch, stop it, then rent the next (a host that never boots then costs one rental, not several).

Drive it in tmux so every screen is captured: `tmux new -d -s bb -x 110 -y 60 boltbeam`, keys with
`tmux send-keys -t bb ...`, each screen with `tmux capture-pane -p -t bb > screens/NN-name.txt`.

| Step | Pass when |
|---|---|
| 1 opens | `boltbeam` builds the TUI with the Go step 4 installed and shows the setup screen (note the build time) |
| 2 sees the machine | the GPU, its chip profile (autoscan's target) and every installed engine are listed |
| 3 finds the models | the GGUF and the AWQ folder under KEEP/models can be picked for their engines |
| 4 runs | a run starts, the progress screen moves through its stages and ends on the results screen, no traceback |
| 5 finishes | Save run writes the run; the report opens; `boltbeam fetch` brings that run home |
| 6 newer features | engine settings, repeats, Upgrade engines and Measure MTP open and finish where the engine has them |

The bench on every GPU: Qwen3-14B and Qwen3.8-27B at 4 bits, each on llama.cpp (Q4_K_M GGUF) and vLLM (AWQ), so
each GPU has two engines on the same two models. Qwen3.8-Flash-Next (111 GB at UD-Q4_K_XL) only on a GPU it fits.
Every engine runs in-model. When a slow host CPU puts a capture over 5%, one rule (`tie_out.over_ceiling`) makes
the gaps a bound (untraced token − kernels) and lets the kernel rows stand when the run's repeats agree: llama.cpp's
three captures, two more fresh captures for TensorRT-LLM and Ollama, or three groups of a vLLM capture's decode steps
(CUDA graphs keep vLLM near 0%, +0.24% on the 5090; the H200 was +5.6%). A model that does not fit a GPU is recorded as not fitting, not forced. The RTX 5090 at
home is the reference: a fast desktop CPU, the most evidence, no rental.

### The numbers, per kind of run

Each screen's figures are checked against the run's own files, and the files against each other, with
`tools/check_run_math.py` (standard library only; `--tui screens/NN-results.txt RUN` adds the screen):

| Kind of run | On screen | Must hold |
|---|---|---|
| any run | `N tok/s measured · roofline L tok/s`, `P% of the limit` | N is the engine's timing in `timing_profile.json`; P = N / L; the screen shows the run's N, L and P |
| any run, one GPU, dense | Limit from memory, GB/s | L ≤ read GB/s (`machine_facts.json`) / weight bytes (`weight_inventory.json`), and within 10% of it |
| any run with a table | MS NOW, +tok/s, CUMULATIVE | rows add to the total; cumulative tok/s never goes down and ends at the roofline |
| in-model (traced) | Tie-out, capture over native | limit + parts = measured token; capture within 5% of untraced. Above it nsys's per-launch cost (CPU-side) lands in the gaps: the table's gaps are the bound `untraced − kernels` (measured token − added by recording = untraced token), its rows sum to the untraced token, and its kernel rows stand only when the run's three fresh capture pairs agree per row (1% for rows of 3%+ of kernel time, 2.5% any; Qwen3.8-27B on two H100s at +25.5% and +11.9%: kernels 12.012 vs 12.023 ms, roles within 2.4%, gaps 3.411 vs 1.778 ms). Kernels over the untraced token: refused |
| repeats | median, range, spread over k runs | the median and range are those of the k repeat timings; spread = (max - min) / median |
| engine settings | the setting named in the header | the run's `run_manifest.json` carries that setting; the kernel names change with it (TRITON_ATTN: `kernel_unified_attention`) |
| Measure MTP | tok/s per k, the speedup, the recommended k | speedup = tok/s at k / tok/s at k=0; the recommendation is the fastest k |
| split over GPUs | the layout and its limit | layer split: one GPU's limit; row split: the limit adds the measured all-reduce |

A figure that does not add up fails the GPU's pass the same as a crash does.

## Limits on a rented pod

RunPod's hosts keep the GPU's performance counters for admins (`RmProfilingAdminOnly: 1`; `ncu` says
`ERR_NVGPUCTRPERM` even as root in the container). vLLM's role timing reads which matrix path a closed kernel took
from those counters only as a last resort. A kernel no library holds (cuBLASLt builds its Hopper `nvjet_*` GEMMs in
the process; Qwen3.8-27B-AWQ's lm_head runs one) is first read from a module-load capture: one more engine launch,
outside the timed captures, that saves every module the driver loads (no counter; needs `g++`, which step 2 installs,
and the CUPTI in vLLM's own venv). The summary then says which kernels were read that way. Only a kernel that capture
did not see either is refused, with "closed decode kernels require accessible instruction counters: ...; the
module-load capture ran and did not see them ..."; tokens per second are still measured (stage measure_timing), only
the per-role table is refused. Proven on the RTX 5090 on cuBLASLt's `nvjet_sm120_*` and on a Vast H200 on
`nvjet_sm90_*` (2026-11-09, [todo-vllm-closed-kernels.md](todo-vllm-closed-kernels.md)).

Hopper (H100, H200): vLLM's GEMMs there are the warpgroup instruction (`HGMMA.64xNx16.F32`, `.BF16` for bf16). The
chip profile a pod's step 8 makes holds its rate (RunPod H200, 2026-11-09: `fp16_wgmma` 781.5 and `bf16_wgmma` 880.8
TFLOPS sustained, `fp16_k8` 468.6), and with it the 14B AWQ (166.9 tok/s, 30.5% of its limit) and the 27B AWQ
(100.8 tok/s, 33.7%) complete their per-role tables with `all figures check`. A profile measured before 016063eb has
no such rate: a run against it reports the speed and the limit and says the roles have no measured rate, and
`python -m boltbeam.workflow.screen autoscan --remeasure` refreshes it. llama.cpp runs are unaffected (H200 NVL: 14B
Q4_K_M 135.0 tok/s, 27B UD-Q4_K_M 78.6, both with the full table and `all figures check`).

## Bring results home

From your own machine, with the pod's SSH address (`ssh root@HOST -p PORT` on the pod page):

```bash
boltbeam fetch root@HOST -p PORT                 # list the pod's runs: id, model, chip, GPU, date, size, here or not
boltbeam fetch root@HOST -p PORT h100-generic    # fetch one (or several, or --all)
```

It reads the runs folder `pod-setup.sh` writes (`/root/boltbeam/runs`, then `/workspace/boltbeam/runs`, then
`~/boltbeam-runs`; `--root` picks another) and says which it used. Each run lands in your saved runs
(`$BOLTBEAM_RUNS/saved`, or `--into DIR`), where the screen's Saved runs page lists it as "from HOST, fetched DATE".
Raw traces (`*.nsys-rep`, `*.sqlite`, `cuda_gpu_trace.csv`) stay on the pod unless `--with-traces`. Every file is
checked by size and sha256 against the pod's copy; a run that differs is not kept. A run already here is skipped,
and a different run under the same id is refused, never overwritten. Each fetched run carries `fetched.json`: the
host, port and path it came from, when, its GPU, and the BoltBeam version that fetched it.

A pod's chip (an H100, a 3090) is usually in no registry row: autoscan measured it there and kept the profile in
`$WORK/chips`. `boltbeam fetch` brings that one file too (`<chip>.json` beside the runs root, or
`~/.boltbeam/chips`), checked by sha256, and keeps it under `~/.boltbeam/chips/fetched/<host>/`: apart from this
machine's own profiles, never matched to this machine's GPU, and never in place of a registry row or a profile made
here with the same id. With it the run's `results.json` is computed here, so the limit and the share of it show.
A run with the same id in your work area (`.work`) is warned about: opening that id finds the work-area run first.

Nothing on the pod is written or deleted, and nothing outside the runs root is read but that chip profile; ssh runs
with `BatchMode=yes`, so a missing key fails at once instead of prompting.

Fetch, then pause: once `boltbeam fetch` says the results are home, stop the pod (stop, not destroy, when the next
test can use it).

## On Vast.ai

The same script runs on a Vast.ai instance (a plain container on a community host); what differed there
(RTX 3090s, 2026-11-08):

- Rent with `vastai create instance OFFER --image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 --disk N --ssh --direct`.
  A reply of `success: false` still creates an instance, stopped, that never starts (its host was taken): destroy
  it (`vastai destroy instance ID -y`; without `-y` it asks and aborts) and rent another offer.
- Disk: `--disk 120` for the bench ladder ([bench.md](bench.md): four models, 55 GB, both engines). Used at the end
  of a ladder (2026-11-09): RTX 3090 65 GB, RTX 4090 72 GB, L40S 78 GB, H100 99 GB.
  One model on one engine fits 40 GB. Disk is billed while an instance is stopped, so do not rent more than this.
- Some hosts write `/root/.ssh/authorized_keys` with ownership sshd rejects ("bad ownership or modes", in `vastai
  logs ID`), and every key is refused, the proxy's too. Rent with an `--onstart-cmd` that keeps `/root` and
  `/root/.ssh` at 700 and the key file at 600, owned by root, for the first minutes.
- NVIDIA's CUDA image has no Python headers; step 2 now installs `python3-dev` (Triton compiles against `Python.h`).
- `search offers` labels China `..., CN`; Hugging Face downloads from there are slow. Filter on the label.
- There is no `/workspace`: `KEEP` falls back to `WORK`. Step 5's single curl ran at about 1.5 MB/s there; it now
  downloads in `MODEL_STREAMS` byte ranges at once (default 8; RTX 4090 host, 2026-11-09: one stream 13 MB/s,
  eight 68 MB/s) and falls back to one stream when a range fails. A model folder (vLLM) still comes with
  `hf download` (several streams; `hf_transfer` is gone from current huggingface_hub); a file already on the
  machine is `MODEL_URL=file://...` with its `MODEL_SHA256`.

## More than one GPU

The pipeline takes `--layout one|layer|row` (limited support): llama.cpp split by layer and vLLM split by rows are
measured with a limit, and the report is written without the per-role split. On 2x R9700: llama.cpp split by layer
26.8 tok/s; vLLM split by rows, 27B AWQ, 49.8 tok/s against a limit of 73.5
([release-039-20261024.md](release-039-20261024.md)); the picker's rules are in
[multi-gpu-recommendation-checks.md](multi-gpu-recommendation-checks.md). Split by rows needs a measured all-reduce
in the machine's facts before it has a limit. On a multi-GPU pod the script runs the same steps, and `LAYOUT` (one,
layer or row) is passed to step 9 as `--layout`.
