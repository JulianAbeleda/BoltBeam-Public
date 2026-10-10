# In-model vs generic measurement, same engine, same chip: Apple M4, 2026-10-10

BoltBeam offers two ways to time a model's kernels. In-model watches the real decode (Metal System
Trace on a Mac with Xcode, nsys on NVIDIA, tinygrad's own events). Generic times each kernel alone
with BoltBeam's kernel timer, on any machine. This record asks how different the two are for the
same engine on the same chip, and why.

## Setup

- Machine: Mac mini, Apple M4, 10-core GPU. Read bandwidth 112.3 GB/s, measured by BoltBeam's
  Metal read probe in the same run. Xcode installed, so both methods could run.
- Model: Qwen3-8B-Q4_K_M.gguf, context 128, batch 1.
- llama.cpp: Homebrew llama.cpp 0.6.0_1 on ggml 0.26.0 (Metal). BoltBeam main 4de25f8.
- tinygrad: tinygrad-arkey exp eecaf40, in-model only (the kernel timer has no tinygrad adapter).
- Generic timer: BoltBeam default, which stores 64 MiB before every timed sample to flush the
  cache ("cold"). Two more variants were run by hand: the same cold run again, and "warm", no
  flush.

## Prediction, written before the runs

Generic 5 to 15% slower than in-model for the big roles (gate_up, down, lm_head), 0 to 10% for
attn_q/attn_o, and faster for attn_kv (2 to 3 MB stays in cache). Generic sum well under the token.

## What each method could produce

| | llama.cpp | tinygrad |
|---|---|---|
| in-model (xctrace) | whole step only. The trace has two unlabelled rows per token (one compute encoder, one blit). llama.cpp encodes the whole graph into about two encoders with no kernel labels, so no role can be paired. | per kernel, 7 roles paired by bytes and count, 39 kernel rows. |
| generic (kernel timer) | all 7 roles, correctness passed (max_rel_err 1.1e-7 to 2.5e-7 against a pure Python dequantize and dot). | not run: no adapter. |

So on a Mac, "in-model" for llama.cpp answers one question only: GPU busy time per token
(49.15 ms of a 48.2 ms untraced token). It cannot split the token by role. For tinygrad it can.

## Result: the two methods differ a lot, and BoltBeam's flush is the cause

llama.cpp, ggml Metal kernels, µs per call. Ideal is bytes over 112.3 GB/s.

| role | bytes/call | calls/tok | ideal µs | generic cold (GB/s, % peak) | cold again | generic warm (GB/s, % peak) |
|---|---|---|---|---|---|---|
| attn_kv Q4_K | 2,359,296 | 54 | 21.0 | 89.5 (26.4, 23%) | 89.6 | 45.0 (52.4, 47%) |
| attn_kv Q6_K | 3,440,640 | 18 | 30.6 | 72.8 (47.3, 42%) | 75.8 | 32.6 (105.5, 94%) |
| attn_qo Q4_K | 9,437,184 | 72 | 84.0 | 152.7 (61.8, 55%) | 163.0 | 93.2 (101.3, 90%) |
| ffn_down Q4_K | 28,311,552 | 18 | 252 | 327.7 (86.4, 77%) | 385.6 | 274.7 (103.1, 92%) |
| ffn_down Q6_K | 41,287,680 | 18 | 368 | 451.3 (91.5, 81%) | 451.8 | 395.3 (104.5, 93%) |
| ffn_gate_up Q4_K | 28,311,552 | 72 | 252 | 329.5 (85.9, 77%) | 328.4 | 271.5 (104.3, 93%) |
| lm_head Q6_K | 510,504,960 | 1 | 4,546 | 4,910 (104.0, 93%) | 4,759 | 4,745 (107.6, 96%) |

Sums per token: generic cold 59.79 ms (again 61.42), generic warm 46.08 ms, ideal weight read
41.64 ms. The token: 48.21 ms untraced (20.74 tok/s at context 128; 48.99 ms at context 512);
49.15 ms GPU busy under xctrace.

- Cold sum over token = 1.24. That is impossible as an in-token value: the weight kernels alone
  cannot take longer than the token that holds them plus attention, norms and RoPE.
- Warm sum over token = 0.96, leaving about 2 ms for everything else. tinygrad's non-weight
  kernels take 4.0 ms on the same chip, so 2 ms for ggml's fused versions is plausible.
- The prediction was wrong for attn_kv under the default timer (cold is 2 to 4 times slower, not
  faster) and right only without the flush.

### Why cold is slow: the flush probe

Same ggml kernel, three preconditions before each timed sample.

| kernel | no flush | after a 64 MiB store (the timer's flush) | after a 64 MiB read-only sweep |
|---|---|---|---|
| empty kernel | 3.2 µs | 3.4 µs | 3.3 µs |
| attn_kv Q4_K, 2.4 MB | 25.9 (91 GB/s, in cache) | 51.2 (46 GB/s) | 30.0 (79 GB/s) |
| attn_qo Q4_K, 9.4 MB | 93 (101 GB/s) | 153 (62 GB/s) | 93.3 (101 GB/s) |
| ffn_gate_up, 28 MB | 274.5 (103 GB/s) | 327.7 (86 GB/s) | 276.2 (102.5 GB/s) |

The read sweep evicts the weight just as the store does, yet the kernel runs at full speed after
it. The penalty is specific to the store: the flush leaves dirty lines in the system level cache,
and they drain to DRAM during the next kernel. The cost is about 50 to 60 µs per launch, roughly
8 MB at 112 GB/s. That is 20% of a 28 MB kernel and 2 to 3 times a 2.4 MB kernel. No kernel inside
the model ever follows a 64 MiB store, so this is a measurement artefact, not a property of the
kernels. One warm attn_qo sample set taken right after idle gave a median of 343 µs with 94%
spread: the GPU clock had not ramped; ten warmups of a 90 µs kernel do not ramp it.

### tinygrad in-model, same chip and weights, for scale

| role | bytes/call | calls/tok | in-model µs/call | GB/s | % peak | lost ms/tok |
|---|---|---|---|---|---|---|
| ffn_gate_up Q4_K | 28.37 MB | 72 | 318.4 | 89.0 | 79.3 | 4.73 |
| ffn_down Q6_K | 41.32 MB | 18 | 530.4 | 77.9 | 69.4 | 2.92 |
| ffn_down Q4_K | 28.34 MB | 18 | 316.6 | 89.5 | 79.7 | 1.16 |
| attn_qo Q4_K | 9.45 MB | 72 | 113.8 | 83.1 | 74.0 | 2.13 |
| attn_kv Q6_K | 3.45 MB | 18 | 91.7 | 37.6 | 33.5 | 1.10 |
| attn_kv Q4_K | 2.37 MB | 54 | 32.3 | 73.4 | 65.4 | 0.60 |
| lm_head Q6_K | 510.8 MB | 1 | 6,052 | 84.4 | 75.2 | 1.50 |

Tie-out: limit at context 140 is 41.82 ms; weight kernels above ideal 14.15 ms; other kernels
3.99 ms busy (attention 1.55, elementwise 1.25, reduce 1.02, other 0.15, copy 0.02); gaps between
kernels 11.90 ms; token 71.69 ms captured, 67.05 ms untraced (14.91 tok/s). Graph replay failed
(`Metal ICB offset exceeds 0xffffffff in partitioned replay`), so the token ran as 1021 launches.
tinygrad's in-model numbers sit between ggml alone-cold and ggml alone-warm for every big role.

## What each method answers

- Generic: is the kernel correct, and what bandwidth does this kernel reach on this chip at this
  shape, with a clean precondition. It cannot see gaps, attention, norms, or clock behaviour. With
  the store flush it overstates every ggml kernel's time, by 3% (lm_head) to 300% (attn_kv).
  Without the flush it lands within about 4% of the token for the big roles.
- In-model: where the token's time goes, including gaps (11.9 ms for tinygrad) and the non-weight
  kernels, but only when the engine labels its kernels. llama.cpp on Metal does not.

## Issues found in BoltBeam main 4de25f8

1. `engine_kernels.ggml_source` reads one NUL-delimited C string. ggml 0.26 embeds 35 separate
   libraries in `__DATA,__ggml_metallib` with no separators (llama-bench prints "loaded 35
   libraries from embedded data"), so the pipeline's cross-check compiled garbage and recorded
   "Metal shader compile failed". Workaround used for this record: `xcrun segedit libggml-metal.so
   -extract __DATA __ggml_metallib`, split on `\n#ifndef GGML_METAL_IMPL\n` (35 chunks), keep the
   chunk that holds both `kernel_mul_mv_q4_K_f32` and `q6_K` host names, point
   `BOLTBEAM_GGML_METAL` at it. The kernel argument struct layout still matched (correctness
   passed). The Air's ggml 0.19 has one library, which is why the Air never saw this.
2. Rows under 4 MB get `cache: true` and a note "stays in the last-level cache between warmup and
   sample" while `timing.cache` says flushed. The flush does evict; the note is wrong, and the real
   distortion is the dirty-line drain above.
3. The store flush adds about 60 µs of writeback to the next launch on the M4. A read-only sweep
   evicts without it.
4. `pip install -e '.[dev]'` does not install numpy; with `BOLTBEAM_TINYGRAD_VENV` pointing at that
   venv the tinygrad measure is skipped ("The fork's python has no numpy").

## Sources

Prediction, probe scripts and copies of every run file (without the raw xctrace) are kept on the Air
under the runs work folder, `boltbeam-runs/.work/in-model-vs-generic-m4-20261010/`: `ll-results.json`, `tg-results.json`, `ll-isolated-cold-rep2.json`,
`ll-isolated-warm.json`, `flush-probe.json`, `isolated_timing_trace.json`, `tinygrad_timing_trace.json`.
Timings: llama.cpp pipeline 65 s (role_time 32 s, two xctrace captures); isolated timer 1.6 s;
tinygrad pipeline 5 min 21 s (role_time 207 s); flush probe 1 s.
