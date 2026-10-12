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
| 1 llama.cpp Metal | `collectors/engine_kernels.py` | `kernel_mul_mv_<type>_f32`, dispatched by the engine's two rules: the K-quants give each simdgroup its own `nr0` rows; F32, F16, BF16 and Q8_0 share one row set across the simdgroups and reduce through 32 × 4 × `nr0` bytes of threadgroup memory (`METAL_ROWS_SHARED`) | the installed ggml library (`libggml-metal`, embedded `.metal` text) or a `ggml-metal.metal` file; `BOLTBEAM_GGML_METAL` names it |
| 1 llama.cpp CUDA | `collectors/engine_kernels.py` | `mul_mat_vec_q<type, 1, false, small_k, halve_iters>`: small_k and halve_iters decided as the engine decides them for this GPU's compute capability (below); the check reads the q8_1 vector | the installed llama.cpp source `ggml/src/ggml-cuda/mmvq.cu` and `common.cuh`; `BOLTBEAM_GGML_CUDA_SRC` names the folder |
| 2 tinygrad | none | | the fork generates its kernels per shape at run time inside a model graph; there is no shipped source to compile. Not built. tinygrad keeps its own timing (`tinygrad_role_time.py`). |

Adapter 0 is the building-block probe (`probe_evidence.json`). Its rows are "BoltBeam's own kernel (reference)".
They are not the engine's kernel.

Adapter 1 times the weight formats whose block layout and pure-Python reference `metal_native.py` holds: Q4_K, Q6_K
and Q8_0 (`BLOCK_BYTES`, `BLOCK_ELEMS`, `DEQUANT`; the same figures as `data/quants.json`). A limit role in another
format is a `not_measured` row saying "no <quant> adapter"; the table shows it as "not timed: <reason>", its own row,
and leaves it out of the floor rule and the sums, so the roles that were timed are still a measured table
(`tinygrad_role_time.loss`). Before that rule the 27B's two Q8_0 roles came back as 0 µs and the floor rule refused
the whole table as "measured below their floor".

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
(docs/in-model-vs-generic-rtx5090-20261010.md). A read sweep evicts without that penalty. The tables below put the isolated sum at 54.3 / 66.0 = 0.82 of
the M3 token and, less the dispatch floor, 3.38 / 3.98 = 0.85 of the 5090 token. On the 5090 it is
within 3% of the in-model weight-kernel sum (3.29 ms), not the whole token.
`timing.flush_mode` in every row says which ran. "No flush" is not an option on either chip: a 5090's 96 MiB L2
holds every weight role but lm_head between warmup and sample, and those rows read at 117 to 149% of DRAM peak.

The dispatch floor (an empty kernel between the same two timestamps) is measured once per run and recorded beside
every row as `timing.dispatch_floor_us`: 2.4 µs on the M3, 3.9 to 4.2 µs on the 5090 (CUDA events). Since
2026-10-11 it is measured as a timed launch runs: after the warmups, 60 launches, each after a sweep. Measured cold
(20 launches, no warmup, no sweep) it read 4.2 µs in one run and 6.4 µs in the next, and every row's less-floor
time moved with it. On the 5090 that
is 30 to 130% of the small roles' time, so the floor is never left implicit: each isolated row carries
`us_per_call_less_floor` beside the measured `us_per_call`, and the tie-out's estimate is made from the less-floor
times and says "less the N µs dispatch floor" where it does. The measured number is never replaced.

The per-role rule judges an isolated row on the same less-floor time (`tie_out.role_why`): its GB/s and % of peak
are on the time less the floor, the figure that matched nsys within 5% per role on the 5090, while `us_per_call`
stays the measured time with the floor in it and `us_per_call_less_floor` sits beside it. Judged on the measured
time, the 5090's ffn_down Q4_K read 73% of peak and "too small to fill memory" while nsys inside the model had it at
87%, "at the limit"; less the floor it reads 89%. The seam says once which column is which (`estimate.columns_words`,
"µs/call is the measured time, floor included; less floor, GB/s and % of peak are less the N µs dispatch floor") and
the text summary, the HTML report and the TUI print that sentence over the table. The cross-check rows beside an
in-model table follow the same rule. In-model rows carry no floor and are untouched.

A kernel under 3 times the floor is mostly floor, and the floor's own jitter is most of its spread (attn_kv Q6_K on
the 5090 read 8.05 then 10.30 µs between two passes). The timer samples such a kernel 60 times instead of 20
(`kernel_timer.more_samples`, one rule for both backends; `timing.more_samples` on the row says when it fired) and
every isolated row carries `spread_pct` (P90 less P10 over the median). The tie-out reads it (`tie_out.noisy_words`,
one rule for both backends): a role is noisy when its spread, halved as a ± figure, is over ±10% (the gap between
"at the limit" at 85% and a clearly slow row), or when the dispatch floor is over a third of its measured time. The
reason word is followed by "; noisy: ±N%, F µs floor under a T µs kernel" (`tie_out.NOISY`; the floor part only when
the floor rule fired) and `reason_word` keeps the firm word for the readers that key on it. The chip's run-to-run
plausibility band is not the yardstick: it is a different statistic, and against the 5090's ±1.4% band six of seven
roles read "noisy", lm_head-sized ones at ±1.4 to ±5% among them. Under this rule the 5090 flags the three attention
rows (floor 36 to 56% of their time) and the M3 (floor 1.9 µs) flags attn_kv Q4_K alone (±20%). The text summary,
the HTML report and the TUI show the suffix as the seam sends it; no renderer has a rule of its own.

## Which `mul_mat_vec_q` the engine launches on this GPU

`mmvq.cu` picks a parameter table per GPU in its host `get_device_table_id(int cc)` (GENERIC, TURING, GB10, the AMD
tables) and, on the GB10 table only, doubles the warps per block when `should_halve_iters` says the K loop retires
in half the trips. The CUDA adapter mirrors both instead of defaulting them: it reads the GPU's compute capability
through the bridge (`cuda_device.ggml_compute_capability`, 100 x major + 10 x minor as ggml numbers it: an RTX 5090
is 1200, a DGX Spark 1210), takes the `if (...) { return MMVQ_PARAMETERS_X; }` branches of the source's own
`get_device_table_id` in order with `common.cuh`'s `GGML_CUDA_CC_*` values, then applies `should_halve_iters`'s guard
and its idle-tail rule with the constants of the source's return line, and the promotion list of `calc_nwarps`. The
row's `geometry` records `table`, `table_rule`, `halve_iters` and `halve_rule`, and the note names the value as
"mirrored from the engine's rule". Nothing is hard-coded from the table: a source whose rule the adapter cannot read
refuses, named. On the 5090 the rule fires as "no branch matched, the default" (GENERIC) and "should_halve_iters is
false off the MMVQ_PARAMETERS_GB10 table"; on a DGX Spark it would instantiate `halve_iters = true` with 8 warps for
Q4_K and Q6_K at 4,096 columns. The TURING table's own `calc_nwarps` switch is not mirrored (the GENERIC value is
used and the record says so).

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

## Checked against Nsight Compute (RTX 5090, 2026-10-11)

Nsight Compute is the oracle; the generic timer is what is improved against it. The timer never needs ncu.

How the oracle was read. Every ncu command BoltBeam builds runs with `--clock-control none`
(`profiler/ncu_counters.ncu_command`, `CLOCK_CONTROL`): ncu's default locks the base clock and slows every duration
it reports. The metrics are one pass, `gpu__time_duration.sum`, `dram__bytes_op_read.sum` and the grid, with
`--cache-control none`. With an L2 counter added ncu needs three passes, and the later passes find the L2 the first
pass filled; application replay mixed up two launches of the same kernel and grid. So the L2 hit share is derived:
1 less the DRAM bytes over the bytes the kernel reads.

- In the model: llama-bench decode (`-p 0 -n 4 -d 128`) under ncu, every `mul_mat_vec_q` launch. A launch is a role's
  by its ggml type and grid; a fused gate+up launch reads twice the weights and counts as two calls.
- The timer alone: the same timer run under ncu, the launches that follow a sweep.
- Models: Qwen3-8B Q4_K_M and a 27B hybrid (Q4_K_M), llama.cpp build 9592. Token: 3.98 ms (8B) and 13.97 ms (27B),
  llama-bench untraced, context 128.

### 1. The sweep does evict: the cache check

Each timed launch read its weight bytes from DRAM on every role of both models (DRAM over weights 1.000 to 1.002).
The check is now a recorded field. With ncu on PATH, open counters and `BOLTBEAM_NCU=1`, the generic run times its
kernels once more under ncu and writes `cache_check` on each row: ncu's duration, the DRAM bytes, the weight bytes,
their ratio, the derived L2 hit share, and `passed` when the DRAM bytes are within 2% of the weights
(`collectors/kernel_oracle.py`). Without counters the row says `"not checked"` and why. It costs one more timer pass
under ncu: about 3 minutes on the 8B.

### 2. The cache state inside the model

In the model every launch reads exactly its weight bytes from DRAM (8B attn_kv Q4_K: 2,359,296 bytes, the weight
size). The q8_1 vector the kernel before wrote is an L2 hit. The timer's sweep evicted the vector too: each timed
launch read the weights plus 4,608 bytes. The timer now reads the activation args back in after each sweep
(`KernelSpec.warm`, `WARM_RULE`; both llama.cpp adapters and BoltBeam's GEMV mark the vector). Under ncu the timed
launches then read exactly the weights, as in the model.

The time did not move. ncu's own duration of the timed kernel, vector cold then warm (8B): attn_kv Q4_K 3.87 / 3.84,
attn_kv Q6_K 4.83 / 4.74, attn_qo 8.22 / 8.19, ffn_down Q4_K 20.00 / 19.97, ffn_down Q6_K 29.09 / 28.72,
ffn_gate_up 19.39 / 19.55, lm_head 302.37 / 303.07 µs. The gap from the summed kernels to the token does not close
through the cache state: after both changes the 8B sum is 3.41 ms, 0.86 of the 3.98 ms token (it was 0.83 to 0.85),
and ncu's in-model sum over the same roles is 3.50 ms, 0.88. The rest of the token is the kernels the timer does not
time (attention, norms, quantize) and the gaps between launches.

### 3. The dispatch floor against ncu's duration

The timed launch's event time less ncu's duration of the same kernel is what the floor should take off: 2.9 to 4.3 µs
across the 15 roles (median 3.6). The floor as measured before the change read 4.26 µs, then 6.43 µs in a later run:
cold launches with no warmup. Measured after warmups, each after a sweep, it read 3.84 to 3.97 µs in five runs. Less
the floor against ncu's duration of the same kernel, 8B: before mean |difference| 0.85 µs, after 0.38 µs. The
floor still takes about 0.5 µs too much off the kernels over 8 µs on average; an empty kernel's own duration is 1.1 µs
by ncu and part of it every kernel pays too, which events alone cannot separate. The measured time is kept as it is.

### Error per role, generic less the floor against ncu in the model

Before: the timer on main a0fd266's rules (8B the same day; 27B the 2026-10-10 run). After: both changes. Error is
generic less oracle. `ncu alone` is ncu's duration of the timer's own launch after the change.

| model | role | before µs | after µs | ncu alone µs | oracle in model µs | error before | error after |
|---|---|---|---|---|---|---|---|
| 8B | attn_kv Q4_K 1024x4096 | 4.32 | 4.10 | 4.00 | 3.94 | +0.38 (+9.8%) | +0.16 (+4.1%) |
| 8B | attn_kv Q6_K 1024x4096 | 5.66 | 4.43 | 4.67 | 4.77 | +0.90 (+18.8%) | -0.34 (-7.0%) |
| 8B | attn_qo Q4_K 4096x4096 | 7.36 | 8.00 | 8.16 | 8.45 | -1.09 (-12.9%) | -0.45 (-5.3%) |
| 8B | ffn_down Q4_K 4096x12288 | 19.33 | 19.44 | 20.06 | 20.99 | -1.66 (-7.9%) | -1.55 (-7.4%) |
| 8B | ffn_down Q6_K 4096x12288 | 28.35 | 28.80 | 28.77 | 31.04 | -2.69 (-8.7%) | -2.24 (-7.2%) |
| 8B | ffn_gate_up Q4_K 12288x4096 | 17.90 | 18.90 | 19.58 | 18.75 | -0.85 (-4.5%) | +0.14 (+0.8%) |
| 8B | lm_head Q6_K 151936x4096 | 301.46 | 302.53 | 303.33 | 303.92 | -2.46 (-0.8%) | -1.39 (-0.5%) |
| 27B | attn_kv Q8_0 1024x5120 | 5.17 | 6.00 | 5.57 | 5.65 | -0.48 (-8.5%) | +0.35 (+6.2%) |
| 27B | attn_qo Q6_K 5120x6144 | 17.62 | 18.43 | 18.53 | 20.10 | -2.48 (-12.3%) | -1.66 (-8.3%) |
| 27B | attn_qo Q8_0 12288x5120 | 40.69 | 41.07 | 42.11 | 42.83 | -2.14 (-5.0%) | -1.76 (-4.1%) |
| 27B | ffn_down Q4_K 5120x17408 | 32.21 | 32.70 | 32.70 | 35.62 | -3.41 (-9.6%) | -2.91 (-8.2%) |
| 27B | ffn_gate_up Q4_K 17408x5120 | 31.89 | 31.20 | 32.50 | 34.40 | -2.51 (-7.3%) | -3.20 (-9.3%) |
| 27B | lm_head Q6_K 248320x5120 | 613.66 | 614.32 | 615.30 | 616.16 | -2.50 (-0.4%) | -1.84 (-0.3%) |
| 27B | ssm_projection Q8_0 48x5120 | 1.79 | 2.40 | 2.66 | 2.69 | -0.90 (-33.3%) | -0.29 (-10.7%) |
| 27B | ssm_projection Q8_0 5120x6144 | 21.92 | 22.21 | 22.67 | 22.91 | -0.99 (-4.3%) | -0.70 (-3.1%) |

| model | sum per token, before | after | oracle | mean abs error before | after |
|---|---|---|---|---|---|
| 8B | 3.314 ms | 3.408 ms | 3.497 ms | 9.0% | 4.6% |
| 27B | 9.079 ms | 9.141 ms | 9.844 ms | 10.1% | 6.3% |

What is left is not a timer rule. In the model ffn_down, the attention output and gate+up run as a fused launch
(`has_fusion` true: the residual add or the GLU in the same kernel); the adapter times the plain kernel. ncu puts
the fused launch 0.9 to 2.9 µs above the plain one on the same bytes (8B ffn_down Q6_K 28.77 alone, 31.04 in the
model). That is the largest remaining error, and it is the adapter's, not the timer's.

### Run the comparison again

On dev, with ncu and open counters (exit 2 and the reason otherwise):

```bash
BOLTBEAM_NCU=1 python -m boltbeam.workflow.screen pipeline MODEL.gguf --run RUN --provider llama.cpp \
  --analyze --measure auto --role-time generic --no-search          # rows carry cache_check
python tools/kernel_oracle.py RUN --model MODEL.gguf --out cmp.json  # error per role against ncu in the model
BOLTBEAM_ORACLE_RUN=RUN:MODEL.gguf python -m pytest -q tests/test_kernel_oracle.py   # the live test; skipped otherwise
```

GPU runs for this record: the timer three times on the 8B and once on the 27B, the timer under ncu four times on the
8B (a wrong metric name for this chip, then three-pass metrics, then application replay, then one pass) and once on
the 27B, the decode under ncu once per model, and the cache check once per model. Evidence is kept outside the repo
in the runs work folder, `.work/ncu-timer-20261011/`.

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
carries a `note` naming the template parameters past small_k and how each was set (this `mmvq.cu` has a fifth
parameter, `halve_iters`, mirrored from the engine's rule for the GPU's compute capability; see above).

## Environment

| variable | what it names |
|---|---|
| `BOLTBEAM_GGML_METAL` | a `libggml-metal` library or `ggml-metal.metal` file, when not under Homebrew |
| `BOLTBEAM_GGML_CUDA_SRC` | `llama.cpp/ggml/src/ggml-cuda`, when not at `~/env/llama.cpp` |
| `BOLTBEAM_NVCC` | the nvcc to use (default: PATH, then `/usr/local/cuda/bin/nvcc`) |
| `BOLTBEAM_CUDA_CACHE` | where compiled cubins are kept (default: the system temp folder) |
