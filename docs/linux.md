# BoltBeam on Linux with an NVIDIA GPU

From a bare Ubuntu to the first report. Every command on its own line. Paths are examples; put things where you
like. Nothing here needs sudo except the driver and the CUDA toolkit.

## 1. The driver and the CUDA toolkit

`nvidia-smi` must answer. BoltBeam reads the GPU from it.

```bash
nvidia-smi
```

The CUDA toolkit gives `nvcc` (the kernel timer compiles kernels with it) and `cu++filt` (it reads kernel names
with it). Both come with the toolkit; put its `bin` folder on PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
cu++filt --version
```

Put the `export` line in your shell profile so every terminal has it.

## 2. nsys

Nsight Systems times the real token per kernel (the in-model measurement). It comes with the CUDA toolkit;
`nsys` must be on PATH:

```bash
nsys --version
```

Without it BoltBeam still runs: the roles are then timed with its own kernel timer (the generic measurement) and the
report says so.

## 3. llama.cpp with CUDA

llama.cpp is the default engine. Build it with CUDA from a checkout; BoltBeam reads its `llama-bench` and, for the
generic measurement, its CUDA kernel source (`ggml/src/ggml-cuda`).

```bash
git clone https://github.com/ggml-org/llama.cpp ~/env/llama.cpp
cd ~/env/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j
ls build-cuda/bin/llama-bench
```

`~/env/llama.cpp` is one of the folders BoltBeam looks in (also `~/llama.cpp`, `~/src/llama.cpp`, `~/storage/*/llama.cpp`,
and any `build*/bin` under them). Elsewhere, name them:

```bash
export BOLTBEAM_LLAMA_BENCH=/path/to/llama.cpp/build-cuda/bin/llama-bench
export BOLTBEAM_LLAMA_BATCHED_BENCH=/path/to/llama.cpp/build-cuda/bin/llama-batched-bench
export BOLTBEAM_GGML_CUDA_SRC=/path/to/llama.cpp/ggml/src/ggml-cuda
```

## 4. The model

Any GGUF. Put it where you like; `~/models` is where the screen's model picker looks first.

```bash
mkdir -p ~/models
ls ~/models
```

## 5. BoltBeam

Python 3.10 or newer.

```bash
git clone https://github.com/JulianAbeleda/BoltBeam-Public.git
cd BoltBeam-Public
python3 -m venv .venv
.venv/bin/pip install .
.venv/bin/boltbeam selfcheck
```

If the root disk is small, point the compile cache and the temp folder at a bigger one:

```bash
export TMPDIR=/data/tmp
export BOLTBEAM_CUDA_CACHE=/data/tmp/boltbeam-cuda
```

## 6. Optional engines

Each one is found when it is in the usual place; set its env var only when it is somewhere else. `boltbeam doctor`
shows the path each engine was found at and how.

| engine | the usual place | the env var when elsewhere |
|---|---|---|
| tinygrad fork | `tinygrad-arkey-exp` next to the BoltBeam checkout, `~/tinygrad-arkey*`, `~/env/tinygrad*`, `~/storage/*/tinygrad*`, with a `.venv` inside it that has numpy | `BOLTBEAM_TINYGRAD_ROOT`, `BOLTBEAM_TINYGRAD_VENV` |
| vLLM | a venv under `~/env`, `~/storage`, `~/venvs`, `~/.venvs` or `~/storage/*` whose python imports `vllm` | `BOLTBEAM_VLLM_PYTHON` |
| Ollama | `ollama` on PATH, `/usr/local/bin`, `~/.ollama`, `~/storage/*/ollama/bin` | `BOLTBEAM_OLLAMA` |
| TensorRT-LLM | a venv as for vLLM whose python imports `tensorrt_llm` | `BOLTBEAM_TRTLLM_PYTHON` |

The tinygrad fork:

```bash
git clone -b exp https://github.com/JulianAbeleda/tinygrad-arkey ../tinygrad-arkey-exp
cd ../tinygrad-arkey-exp
python3 -m venv .venv
.venv/bin/pip install numpy
cd -
```

## 7. Check, then run

```bash
.venv/bin/boltbeam doctor
```

It prints one row per thing and ends with `ready to run` or `N things to set up`, each gap with its one fix. Then:

```bash
.venv/bin/boltbeam tui
```

The screen needs Go at the version `tui/go.mod` names. Without it, `boltbeam tui` prints the install line
(https://go.dev/dl/) and exits 2; it downloads nothing. For an agent, `boltbeam tui --json chips` prints one JSON
object.

## 8. The GPU must be free

A measurement on a GPU that another program holds gives slower numbers that look real. Before a run, BoltBeam asks
`nvidia-smi` which programs hold the GPU and refuses to measure while any does. Stop a resident model server (for
example one started with `systemctl --user stop <unit>`) for the measurement, and start it again after. The engine
scan never touches the GPU; `doctor` is safe to run at any time.
