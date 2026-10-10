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
| 1 llama.cpp CUDA | `collectors/engine_kernels.py` | `mul_mat_vec_q<type, 1, false, small_k>` (matched as a prefix when the source has more template parameters; the check reads the q8_1 vector) | the installed llama.cpp source `ggml/src/ggml-cuda/mmvq.cu`; `BOLTBEAM_GGML_CUDA_SRC` names the folder |
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

The kernel runs alone after a sweep of a buffer larger than the last-level cache (64 MiB on Apple, 256 MiB on
NVIDIA), so its weights come from DRAM whatever their size. In the model the same kernel runs between other
kernels. An in-model capture sees that; this does not.

The sweep reads on both backends. Until 2026-10-10 it stored, and a store leaves dirty lines that drain into DRAM
during the next kernel. On an M4 that was a fixed cost of about 60 µs per launch, which made a 2.4 MB kernel 2 to 4
times slower and a 28 MB kernel 20% slower than inside the model, and the isolated sum 1.24 times the token
(docs/in-model-vs-generic-m4-20261010.md). On an RTX 5090 the cost grew with the bytes the kernel pulled through
the L2: 2.2 to 4.0 µs on the 3 to 41 MB roles (7 to 12%) and 22.5 µs on lm_head
(docs/in-model-vs-generic-rtx5090-20261010.md). A read sweep evicts without that penalty: the sum lands within
about 4% of the token on the M3 and, less the dispatch floor, within 4% of nsys per role on the 5090.
`timing.flush_mode` in every row says which ran. "No flush" is not an option on either chip: a 5090's 96 MiB L2
holds every weight role but lm_head between warmup and sample, and those rows read at 117 to 149% of DRAM peak.

The dispatch floor (an empty kernel between the same two timestamps) is measured once per run and recorded beside
every row as `timing.dispatch_floor_us`: 2.4 µs on the M3, 3.9 to 4.2 µs on the 5090 (CUDA events). On the 5090 that
is 30 to 130% of the small roles' time, so the floor is never left implicit: each isolated row carries
`us_per_call_less_floor` beside the measured `us_per_call`, and the tie-out's estimate is made from the less-floor
times and says "less the N µs dispatch floor" where it does. The measured number is never replaced.

The tie-out for an isolated table has three lines: the limit, the weight kernels above their ideal (isolated), and
one difference line named for what it holds: kernels not timed and gaps (attention, norms, KV read, idle).

## Proof on the Air (Apple M3, no Xcode), 2026-10-10

ggml 0.19.0 from `/opt/homebrew/opt/ggml/libexec/libggml-metal.so`. nsg 2, nr0 2, FC_MUL_MV 600, all read from the
embedded source. Outputs matched the reference on every role. These are read-sweep numbers from 2026-10-10
(peak 97.9 GB/s by the Metal read probe, dispatch floor 2.4 µs); the store-flush numbers taken earlier that day
were 1% (lm_head) to 41% (attn_kv Q6_K) higher, and the small rows carried a "cache" tag that was wrong.

| role | kernel | threadgroups x (32, nsg) | µs/call | GB/s of 97.9 | % of peak |
|---|---|---|---|---|---|
| ffn_gate_up Q4_K 12288x4096 | kernel_mul_mv_q4_K_f32 | 3072 x (32, 2) | 324 | 87.4 | 89 |
| ffn_down Q6_K 4096x12288 | kernel_mul_mv_q6_K_f32 | 1024 x (32, 2) | 474 | 87.2 | 89 |
| ffn_down Q4_K 4096x12288 | kernel_mul_mv_q4_K_f32 | 1024 x (32, 2) | 315 | 90.0 | 92 |
| attn_qo Q4_K 4096x4096 | kernel_mul_mv_q4_K_f32 | 1024 x (32, 2) | 114 | 83.1 | 85 |
| lm_head Q6_K 151936x4096 | kernel_mul_mv_q6_K_f32 | 37984 x (32, 2) | 5408 | 94.5 | 96 |
| attn_kv Q4_K 1024x4096 | kernel_mul_mv_q4_K_f32 | 256 x (32, 2) | 41 | 58.2 | 60 |
| attn_kv Q6_K 1024x4096 | kernel_mul_mv_q6_K_f32 | 256 x (32, 2) | 50 | 68.6 | 70 |

The isolated sum was 54.3 ms against a 66.0 ms token: the kernels fit inside it, with 11.7 ms left for
attention, norms, launches and gaps. Under the store flush the sum was 62.9 ms against 59.8 ms.

## Proof on the RTX 5090 (Ubuntu, CUDA 13.2), 2026-10-10

llama.cpp's `mul_mat_vec_q` from the installed `ggml-cuda/mmvq.cu` (commit 50569eb8), 4 warps per block, one row
per block, compiled by nvcc at run time. Correctness 7 of 7 against the dequantized q8_1 vector (max_rel_err 1.0e-7
to 1.8e-7). Peak 1,693.9 GB/s by the CUDA read probe, dispatch floor 3.9 µs. The read-sweep column is the timer's
flush; less the floor it matches nsys inside the model on every role from 3 MB up
(docs/in-model-vs-generic-rtx5090-20261010.md, Tables 1 to 3).

| role | bytes/call | calls/token | read sweep µs | less floor µs | nsys in-model µs | ratio | GB/s less floor | % of peak |
|---|---|---|---|---|---|---|---|---|
| attn_kv Q4_K 1024x4096 | 2.36 MB | 54 | 8.0 | 4.1 | 3.13 | 1.30 | 575 | 34 |
| attn_kv Q6_K 1024x4096 | 3.44 MB | 18 | 7.9 | 4.0 | 4.17 | 0.96 | 860 | 51 |
| attn_qo Q4_K 4096x4096 | 9.44 MB | 72 | 11.8 | 7.9 | 8.02 | 0.99 | 1,195 | 71 |
| ffn_down Q4_K 4096x12288 | 28.3 MB | 18 | 23.9 | 20.0 | 19.15 | 1.04 | 1,416 | 84 |
| ffn_down Q6_K 4096x12288 | 41.3 MB | 18 | 32.7 | 28.8 | 28.76 | 1.00 | 1,434 | 85 |
| ffn_gate_up Q4_K 12288x4096 (x2 for the fused in-model launch) | 28.3 MB | 72 | 22.5 | 18.6 | 36.27 / 2 | 1.03 | 1,522 | 90 |
| lm_head Q6_K 151936x4096 | 510.5 MB | 1 | 306.2 | 302.3 | 301.98 | 1.00 | 1,689 | 99.7 |

The read-sweep sum less the floor is 3.38 ms against a 3.98 ms token (0.85; the in-model weight kernels are 0.83
of it). Under the store flush the sum was 4.80 ms, 1.21 times the token.

## Running on the Ubuntu box (RTX 5090)

The CUDA bridge (`runtime/cuda_device.py`) compiles with `nvcc -cubin -O3 -arch=native`, loads the cubin through the
driver API (`libcuda`, ctypes) and times each launch between two CUDA events. The box's rules, not BoltBeam's: the
root disk is full, so everything lives under `~/storage`; a resident `llama-server` holds the GPU and `gpu_free`
rightly refuses a held GPU, so it is stopped for the GPU steps and restarted after; no sudo. A fresh clone of `main`:

```bash
# 1. a clean checkout under ~/storage (never the live ~/BoltBeam)
cd ~/storage && git clone --branch main ~/env/BoltBeam BoltBeam-main && cd BoltBeam-main
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. the engine the adapter reads and the box's folders; nvcc is found by fallback, cu++filt only via PATH
export PATH=/usr/local/cuda/bin:$PATH
export BOLTBEAM_GGML_CUDA_SRC=$HOME/env/llama.cpp/ggml/src/ggml-cuda
export BOLTBEAM_LLAMA_BENCH=$HOME/env/llama.cpp/build-cuda/bin/llama-bench
export TMPDIR=$HOME/storage/tmp BOLTBEAM_CUDA_CACHE=$HOME/storage/tmp/boltbeam-cuda

# 3. the resident server: stop it for the GPU steps, restart it after, check it answers (the box's rule)
systemctl --user stop <the model server's user unit>
.venv/bin/python -m boltbeam.workflow.screen pipeline ~/storage/models/Qwen3-8B-Q4_K_M.gguf \
  --run ~/storage/runs/nv-001 --target nvidia_sm120 --id nv-001 --provider llama.cpp \
  --analyze --measure auto --role-time generic --no-search          # or --role-time in-model for the nsys capture
systemctl --user start <the model server's user unit> && sleep 20 && curl -s http://127.0.0.1:8080/v1/models | head -c 200

# 4. read it
.venv/bin/python -m boltbeam.workflow.screen results --run ~/storage/runs/nv-001 | python3 -c "
import json,sys; r=json.load(sys.stdin); l=r['loss']
print(l['role_source_words']); print(l['estimate'] and l['estimate']['method'])
for x in l['roles']: print(x['role'], x['quant'], x['reason'], x.get('pct_peak'), x.get('us_per_call'))"
```

The in-model pipeline takes about 45 s (two nsys captures and the mmvq cubin compile), the generic one 7 s with the
cubin cached. The llama.cpp build on the box is from 2026-08-20 and its `libllama.so` drops a `llama_decode.dot` on
every decode; the capture runs the program in its own capture folder, so the file lands beside `capture.log`. The
trace's `engine.source` records the source file's commit and mtime beside the traced binary's sha and mtime; a row
whose kernel was matched as a prefix of a longer template (this `mmvq.cu` has a fifth parameter, `halve_iters`)
carries a `note` saying so.

## Environment

| variable | what it names |
|---|---|
| `BOLTBEAM_GGML_METAL` | a `libggml-metal` library or `ggml-metal.metal` file, when not under Homebrew |
| `BOLTBEAM_GGML_CUDA_SRC` | `llama.cpp/ggml/src/ggml-cuda`, when not at `~/env/llama.cpp` |
| `BOLTBEAM_NVCC` | the nvcc to use (default: PATH, then `/usr/local/cuda/bin/nvcc`) |
| `BOLTBEAM_CUDA_CACHE` | where compiled cubins are kept (default: the system temp folder) |
