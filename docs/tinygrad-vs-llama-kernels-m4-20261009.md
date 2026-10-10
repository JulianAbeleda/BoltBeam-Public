# tinygrad's emitted kernels read memory slower than llama.cpp's: Apple M4, 2026-10-09

BoltBeam labels every tinygrad role on Apple silicon "slow kernel". This record checks what the
label means: is the emitted kernel slower than llama.cpp's for the same bytes on the same chip,
or is the loss elsewhere in the token?

## Setup

- Machine: Mac mini, Apple M4, 10-core GPU. Read bandwidth 111.6 GB/s, measured by BoltBeam's
  Metal read probe the same day.
- Model: Qwen3-8B-Q4_K_M.gguf, context 128, batch 1.
- tinygrad: tinygrad-arkey exp eecaf40. Per-role times are inside the real decode, captured
  with Metal System Trace (xctrace), through BoltBeam 04fa753.
- llama.cpp: built from source at d81235049, the Homebrew build's commit. Per-role times are
  isolated: `test-export-graph-ops -m MODEL -c 1024 -ub 512 -fa on`, then
  `test-backend-ops perf -b MTL0` on the decode MUL_MAT ops only, two runs after 60 s cooldowns,
  averaged. The two runs agreed within 2.5%.
- GB/s is the matrix's weight bytes divided by the call time.

## Prediction, written before the runs

tinygrad ffn_gate_up at about 75 to 80% of peak, llama.cpp at about 90%. If the two match within
the chip's band (±6.8% on this run), the label is wrong.

## Result: the label holds

| role | weight bytes | tinygrad µs (GB/s) | llama.cpp µs (GB/s) | ratio | verdict |
|---|---|---|---|---|---|
| ffn_gate_up Q4_K 12288x4096 | 28,311,552 | 312.2 (90.7) | 274.9 (103.0) | 1.14 | kernel slower |
| ffn_down Q4_K 4096x12288 | 28,311,552 | 313.3 (90.4) | 275.7 (102.7) | 1.14 | kernel slower |
| ffn_down Q6_K | 41,287,680 | 521.1 (79.2) | 402.0 (102.7) | 1.30 | kernel slower |
| attn_qo Q4_K 4096x4096 | 9,437,184 | 112.5 (83.9) | 89.8 (105.1) | 1.25 | kernel slower |
| lm_head Q6_K 151936x4096 | 510,504,960 | 6221 (82.1) | 4802 (106.3) | 1.30 | kernel slower |
| attn_kv Q4_K 1024x4096 | 2,359,296 | 30.7 (76.8) | 20.9 (112.9) | 1.47 | inconclusive |
| attn_kv Q6_K | 3,440,640 | 91.2 (37.7) | 30.4 (113.2) | 3.0 | inconclusive |

Observed on ffn_gate_up: tinygrad 81% of peak, llama.cpp 92%. The prediction held.

lm_head is the cleanest row. Its 510 MB cannot stay in cache, and the ratio is still 1.30.

The two attn_kv tables are 2.3 to 3.4 MB. Repeated in isolation they stay in cache, and llama.cpp
reads them at or above the DRAM rate. Those rows do not decide anything.

## Whole step

| | tinygrad | llama.cpp |
|---|---|---|
| tokens per second, untraced | 14.9 | 20.6 |
| ms per token, untraced | 67.2 | 48.6 |
| GPU kernel time per token, traced | 59.2 | 48.6 |

The limit is 41.9 ms (23.9 tok/s). The gap between the engines is 18.6 ms per token.

BoltBeam's tie-outs:

| line | tinygrad | llama.cpp |
|---|---|---|
| limit (ideal) | 42.09 | 42.12 |
| weight kernels above ideal | 13.25 | 6.44 (all kernels, not split by role) |
| other kernels above ideal | 3.85 | not split |
| gaps between kernels (GPU idle) | 12.69 | 0.68 |

What explains the 18.6 ms:

- Slower kernels: tinygrad's weight kernels take 55.1 ms per token. The same calls at llama.cpp's
  isolated speed take 45.0 ms. About 10 ms.
- Idle between kernels: 12.7 ms against 0.7 ms. About 12 ms.

The parts add to about 22 ms, more than the gap, because tinygrad's traced token (71.9 ms) is longer
than its untraced one (67.2 ms). The split is approximate. The two causes are about the same size.

## Caveats

- Different methods: tinygrad in the model, llama.cpp in isolation. llama.cpp's whole step
  (48.6 ms with 0.7 ms of gaps) fits its isolated numbers, which supports the comparison.
- tinygrad isolated timing was not run. BoltBeam's kernel search refused the fresh clone as dirty
  because of an untracked .venv symlink inside the checkout.

## What this means

Both halves matter on Metal: the generated kernels and the launch gaps. BoltBeam's kernel search
finds faster variants in isolation (attn_qo 4.5x, ffn_down 1.4x on the M3 the day before), but the
plan did not bind to the decode kernels (0 of 72 calls), so no whole-model gain was measured yet.
