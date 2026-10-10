# The kernel timer

One timing loop for every isolated kernel time BoltBeam reports. `boltbeam/collectors/kernel_timer.py` holds the
loop. Adapters hand it a `KernelSpec`. Bridges run the dispatch on the GPU.

```text
KernelSpec  (source, kernel, args, grid, block, bytes_read, check)
   -> bridge.library / pipeline / buffer / dispatch          Metal: runtime/metal_device.py   CUDA: runtime/cuda_device.py
   -> first launch, output checked against the pure-Python reference (collectors/metal_native.py)
   -> WARMUPS launches, then SAMPLES launches with a cache flush before each
   -> median µs, min, spread, GB/s = bytes_read / median
```

The loop is `kernel_timer.samples`. Nothing else in the tree warms and times a kernel. The bandwidth probes
(`metal_bandwidth.py`, `cuda_bandwidth.py`) and the cubin replay (`cubin_launch.py`) call `samples` and keep their
own statistic (best of N for a bandwidth).

## Adapters

| adapter | module | what it times | source read from |
|---|---|---|---|
| 0 BoltBeam GEMV | `collectors/boltbeam_gemv.py` | BoltBeam's own Q4_K and Q6_K GEMV, Metal and CUDA | the module (BoltBeam's kernel) |
| 1 llama.cpp Metal | `collectors/engine_kernels.py` | `kernel_mul_mv_<type>_f32` | the installed ggml library (`libggml-metal`, embedded `.metal` text) or a `ggml-metal.metal` file; `BOLTBEAM_GGML_METAL` names it |
| 1 llama.cpp CUDA | `collectors/engine_kernels.py` | `mul_mat_vec_q<type, 1, false, small_k>` | the installed llama.cpp source `ggml/src/ggml-cuda/mmvq.cu`; `BOLTBEAM_GGML_CUDA_SRC` names the folder |
| 2 tinygrad | none | | the fork generates its kernels per shape at run time inside a model graph; there is no shipped source to compile. Not built. tinygrad keeps its own timing (`tinygrad_role_time.py`). |

Adapter 0 is the building-block probe (`probe_evidence.json`). Its rows are "BoltBeam's own kernel (reference)".
They are not the engine's kernel.

Adapter 1 is the per-role time for llama.cpp on a Mac with no Xcode. The result is labelled
`isolated, timed by BoltBeam's kernel timer: llama.cpp kernel_mul_mv_q4_K_f32`. On a machine with the vendor
profiler (xctrace, nsys) the in-model capture stays the per-role table and the isolated numbers are the cross-check
beside it (`isolated_timing_trace.json`, results `loss.cross_check`).

Engine sources are read from the installed engine. They are never copied into BoltBeam. The trace records the path,
the version and the sha256 of the file it read, and for the CUDA source the one change made in memory: the word
`static` dropped from `mul_mat_vec_q` so the cubin exports the kernel.

## What an isolated time means

The kernel runs alone after a flush of a buffer larger than the last-level cache (64 MiB on Apple, 256 MiB on
NVIDIA). Its weights come from DRAM. In the model the same kernel runs between other kernels. An in-model capture
sees that; this does not.

A weight smaller than the cache (4 MB on an M-series GPU, 96 MiB on a 5090) stays resident between the warmups and
the sample. Those rows are "inconclusive, cache". Their time is a cache read, not a DRAM read.

The tie-out for an isolated table has three lines: the limit, the weight kernels above their ideal (isolated), and
one difference line named for what it holds: kernels not timed and gaps (attention, norms, KV read, idle).

## Proof on the Air (Apple M3, no Xcode), 2026-10-10

ggml 0.19.0 from `/opt/homebrew/opt/ggml/libexec/libggml-metal.so`. nsg 2, nr0 2, FC_MUL_MV 600, all read from the
embedded source. Outputs matched the reference on every role.

| role | kernel | threadgroups x (32, nsg) | µs/call | GB/s of 97.4 |
|---|---|---|---|---|
| ffn_gate_up Q4_K 12288x4096 | kernel_mul_mv_q4_K_f32 | 3072 x (32, 2) | 356 | 79 |
| ffn_down Q6_K 4096x12288 | kernel_mul_mv_q6_K_f32 | 1024 x (32, 2) | 496 | 83 |
| ffn_down Q4_K 4096x12288 | kernel_mul_mv_q4_K_f32 | 1024 x (32, 2) | 350 | 81 |
| attn_qo Q4_K 4096x4096 | kernel_mul_mv_q4_K_f32 | 1024 x (32, 2) | 149 | 63 |
| lm_head Q6_K 151936x4096 | kernel_mul_mv_q6_K_f32 | 37984 x (32, 2) | 5468 | 93 |
| attn_kv Q4_K 1024x4096 | kernel_mul_mv_q4_K_f32 | 256 x (32, 2) | 51 | 46, cache |
| attn_kv Q6_K 1024x4096 | kernel_mul_mv_q6_K_f32 | 256 x (32, 2) | 71 | 49, cache |

## NVIDIA: built, unit-tested, not yet run on the 5090

The CUDA bridge (`runtime/cuda_device.py`) compiles with `nvcc -cubin -O3 -arch=native`, loads the cubin through the
driver API (`libcuda`, ctypes) and times each launch between two CUDA events. It was written on a Mac and tested
with a fake driver and a fake compiler. The first run on the RTX 5090 is the proof. Run these on the Ubuntu box.
Stop any model server that holds the GPU first. No sudo is needed.

```bash
# 1. a clean checkout under ~/storage (never the live ~/BoltBeam, which is dirty)
cd ~/storage && git clone --branch fix2 ~/env/BoltBeam BoltBeam-kernel-timer && cd BoltBeam-kernel-timer
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. the bridge alone: BoltBeam's own GEMV on a synthetic Q4_K block, correctness and a time
.venv/bin/python - <<'EOF'
import random, struct
from boltbeam.collectors import boltbeam_gemv as gemv, kernel_timer as kt, metal_native as mn
from boltbeam.runtime.cuda_device import Cuda
rows, cols = 64, 4096
rng = random.Random(1)
weights = bytes(rng.getrandbits(8) for _ in range(rows * cols // 256 * 144))
x = gemv.vector("nv", cols)
cuda = Cuda(0)
print(cuda.name, cuda.facts())
flush = kt.Flusher(cuda, "CUDA")
got = kt.time_spec(cuda, gemv.spec("CUDA", "Q4_K", rows, cols, weights, x), flush)
print(got["correctness"], got["median_us"], got["gbs"])
cuda.close()
EOF

# 3. the read probe through the bridge (the number the 5090 profile holds today came from the old standalone program)
.venv/bin/python -m boltbeam.collectors.cuda_bandwidth --device 0 --reps 10

# 4. a run with the probe (adapter 0 on CUDA gives NV the probe evidence it lacked) and llama.cpp's own kernels
export BOLTBEAM_GGML_CUDA_SRC=~/env/llama.cpp/ggml/src/ggml-cuda
.venv/bin/python -m boltbeam.workflow.screen pipeline ~/storage/models/Qwen3-8B-Q4_K_M.gguf \
  --run /tmp/bb-nv-001 --target nvidia_geforce_rtx_5090_32g --analyze --no-search --provider llama.cpp
.venv/bin/python -m boltbeam.workflow.screen results --run /tmp/bb-nv-001 | python3 -c "
import json,sys; r=json.load(sys.stdin); l=r['loss']
print(l['source']); print(l.get('cross_check') and l['cross_check']['words'])
for x in l['roles']: print(x['role'], x['quant'], x['reason'], x.get('pct_peak'), x.get('us_per_call'))
print([(p['role'], p['quant'], round(p['gbs'])) for p in r['probe']['rows']])"
```

What may need a hand on the first run, in order of likelihood:

1. `mmvq.cu` includes `common.cuh`, which declares host functions ggml defines elsewhere. nvcc compiles the file;
   the link happens only for the kernels in the cubin, so undefined host symbols should not matter. If nvcc
   reports one, the message is in the stage's `failed:` line verbatim.
2. The kernel's template parameters: the adapter reads `nwarps`, `vdr` and `qi` from the source and instantiates
   `mul_mat_vec_q<GGML_TYPE_x, 1, false, small_k>`. A newer `mmvq.cu` with a different template is refused with
   "this ggml-cuda is not the version the adapter reads".
3. The `cu++filt` demangler is used to find the template's symbol. It ships with the toolkit.

Step 2 must pass before step 4 means anything.

## Environment

| variable | what it names |
|---|---|
| `BOLTBEAM_GGML_METAL` | a `libggml-metal` library or `ggml-metal.metal` file, when not under Homebrew |
| `BOLTBEAM_GGML_CUDA_SRC` | `llama.cpp/ggml/src/ggml-cuda`, when not at `~/env/llama.cpp` |
| `BOLTBEAM_NVCC` | the nvcc to use (default: PATH, then `/usr/local/cuda/bin/nvcc`) |
| `BOLTBEAM_CUDA_CACHE` | where compiled cubins are kept (default: the system temp folder) |
