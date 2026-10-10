# In-model vs generic measurement, same engine, same chip: RTX 5090, 2026-10-10

The same comparison as the Apple M4 record (in-model-vs-generic-m4-20261010.md), on a different vendor's GPU, with
one more question: is BoltBeam's generic path generic in nature, that is, does the same code run on NVIDIA with no
per-chip edits and give numbers that agree with the engine's real token.

## Setup

- Machine: Ubuntu, RTX 5090, 170 SMs, 96 MiB L2, driver 595.99.02, CUDA 13.2, nsys 2026.1.3. Peak read
  bandwidth 1,693.9 GB/s, BoltBeam's native CUDA probe in the same run. Target `nvidia_sm120`.
- Model: Qwen3-8B-Q4_K_M.gguf, context 128, batch 1.
- Engine: llama.cpp, CUDA build. The traced binary was built 2026-08-20; the generic adapter compiled
  `ggml-cuda/mmvq.cu` from a source checkout at 2026-10-05. nsys names show four template arguments, the source
  has five (`halve_iters`); both launch 4 warps per block, one row per block, so the geometry compared is the same.
  Nothing in BoltBeam checks this; see the issues.
- BoltBeam main ec01234, cloned fresh. The two patches named below were applied in memory for the run, not to the
  repo.

## Prediction, written before the runs

Generic cold 1.3 to 2.5 times in-model on roles under 96 MiB, 1.05 to 1.15 on lm_head; the cold sum over the token;
the store-flush drain present and growing with bytes. Direction right (present, grows with bytes), size over-guessed:
measured 7 to 12%, not 50 to 120%. Wrong: "warm equals DRAM speed"; on a 96 MiB L2 every role but lm_head is an L2
hit when nothing is flushed. Not predicted: the 4 µs CUDA event floor and what it does to the small roles.

## Token

| source | ms per token | tok/s |
|---|---|---|
| llama-bench untraced, context 128 (two runs) | 3.981 / 3.982 | 251.2 / 251.1 |
| llama-bench untraced, context 512 | 4.034 | 247.9 |
| nsys, 63 tokens, 48,006 launches (762 per token), GPU busy | 3.818 | 261.9 |
| BoltBeam limit: 4.676 GB over 1,693.9 GB/s | 2.761 | 362.2 |

## Table 1: in-model per role (nsys, `--role-time in-model`)

| role | calls/token | bytes/call | µs/call | GB/s | % peak | reason | ms/token |
|---|---|---|---|---|---|---|---|
| attn_kv Q4_K | 54 | 2.36 MB | 3.13 | 758 | 44.7 | too small to fill memory | 0.169 |
| attn_kv Q6_K | 18 | 3.44 MB | 4.17 | 827 | 48.8 | too small to fill memory | 0.075 |
| attn_qo Q4_K | 72 | 9.44 MB | 8.02 | 1,179 | 69.6 | too small to fill memory | 0.577 |
| ffn_down Q4_K | 18 | 28.3 MB | 19.15 | 1,480 | 87.4 | at the limit | 0.345 |
| ffn_down Q6_K | 18 | 41.3 MB | 28.76 | 1,437 | 84.8 | too small to fill memory | 0.518 |
| ffn_gate_up Q4_K | 36 (gate and up fused, one launch reads both) | 56.6 MB | 36.27 | 1,563 | 92.3 | at the limit | 1.306 |
| lm_head Q6_K | 1 | 510.5 MB | 301.98 | 1,692 | 99.9 | at the limit | 0.302 |

Tie-out: limit 2.761 ms; weight kernels 3.292 ms (0.83 of the token); other kernels 0.526 ms across 12 kinds
(rms_norm, rope_neox, k_set_rows, flash_attn_ext_vec, combine, get_rows, elementwise); GPU busy 3.818 ms; idle between
kernels 0.163 ms (4%). The in-model path ran first time, 45 s end to end.

## Table 2: generic per role (BoltBeam kernel timer)

Cold is BoltBeam's default: a 256 MiB store before each sample, three sets. Warm (no flush) and read sweep (a 256 MiB
read before each sample) came from a hand probe using the same KernelSpecs and the same `time_spec`, only the flush
swapped. Ideal is bytes over 1,693.9 GB/s. Compare two generic gate_up calls to the one fused in-model launch.

| role | bytes/call | calls | ideal µs | cold store (GB/s, % peak) | cold 2 | cold 3 | warm, no flush (GB/s, % peak) | read sweep (GB/s, % peak) | in-model |
|---|---|---|---|---|---|---|---|---|---|
| attn_kv Q4_K | 2,359,296 | 54 | 1.4 | 8.3 (285, 17%) | 8.5 | 8.5 | 7.2 (327, 19%) | 8.0 (296, 17%) | 3.13 |
| attn_kv Q6_K | 3,440,640 | 18 | 2.0 | 10.0 (345, 20%) | 10.0 | 10.1 | 7.4 (462, 27%) | 7.9 (435, 26%) | 4.17 |
| attn_qo Q4_K | 9,437,184 | 72 | 5.6 | 12.9 (734, 43%) | 12.9 | 12.2 | 8.6 (1,100, 65%) | 11.8 (797, 47%) | 8.02 |
| ffn_down Q4_K | 28,311,552 | 18 | 16.7 | 25.9 (1,092, 64%) | 26.4 | 26.2 | 12.3 (2,295, 135%) | 23.9 (1,184, 70%) | 19.15 |
| ffn_down Q6_K | 41,287,680 | 18 | 24.4 | 36.4 (1,134, 67%) | 36.7 | 36.6 | 16.4 (2,522, 149%) | 32.7 (1,262, 74%) | 28.76 |
| ffn_gate_up Q4_K | 28,311,552 | 72 | 16.7 | 24.9 (1,135, 67%) | 25.2 | 25.8 | 14.3 (1,979, 117%) | 22.5 (1,258, 74%) | 36.27 / 2 = 18.1 |
| lm_head Q6_K | 510,504,960 | 1 | 301.4 | 329.4 (1,550, 91%) | 328.0 | 329.2 | 305.8 (1,670, 99%) | 306.2 (1,667, 98%) | 301.98 |

Correctness 7 of 7 in every set, max_rel_err 1.0e-7 to 1.8e-7, but only with the q8_1 reference (issue 2). Cold
spread 24, 11, 20, 9, 3, 9, 1.3%. Dispatch floor (an empty kernel between two CUDA events): 3.9 to 4.2 µs.

### Sums per token (token 3.981 ms)

| sum | ms/token | over the token | over the in-model weight sum |
|---|---|---|---|
| ideal (the limit) | 2.757 | 0.69 | |
| in-model weight kernels | 3.292 | 0.83 | 1.00 |
| generic cold store, three sets | 4.801 / 4.852 / 4.833 | 1.21 | 1.46 |
| generic read sweep, raw | 4.371 | 1.10 | 1.33 |
| generic read sweep less the 3.9 µs floor | 3.384 | 0.85 | 1.03 |
| generic cold store less the floor | 3.821 | 0.96 | 1.16 |
| generic warm (L2 hits, not DRAM) | 2.993 | 0.75 | 0.91 |

The cold-store sum is 1.21 times the token, impossible, the same shape as the M4 (1.24). Even the raw read-sweep sum
is 1.10 because 253 launches each carry about 4 µs of event floor (0.99 ms per token). With the floor taken off, the
read sweep matches nsys per role: attn_kv Q6_K 4.0 vs 4.17, attn_qo 7.9 vs 8.02, ffn_down Q4_K 20.0 vs 19.15,
ffn_down Q6_K 28.8 vs 28.76, gate_up 37.2 vs 36.27, lm_head 302.3 vs 301.98 (0.96 to 1.04); attn_kv Q4_K 4.1 vs
3.13 (1.30: a 3 µs kernel under a 4 µs floor). Store less floor: 1.05 to 1.19 on roles from 9 MB up, 1.49 on
attn_kv Q6_K.

## Table 3: flush probe on CUDA, 10 warmups, 20 samples, median µs

The flush itself: 256 MiB store 351.9 µs (763 GB/s written); 256 MiB read sweep 161.6 µs (1,661 GB/s).

| kernel | no flush | after a 256 MiB store | after a 256 MiB read sweep | store less sweep |
|---|---|---|---|---|
| empty kernel | 3.9 | 3.9 | 4.0 | -0.1 |
| attn_kv Q4_K 2.4 MB | 7.2 (327, L2) | 8.1 (291) | 8.0 (296) | 0.1 |
| attn_kv Q6_K 3.4 MB | 7.4 (462, L2) | 10.1 (340) | 7.9 (435) | 2.2 |
| attn_qo Q4_K 9.4 MB | 8.6 (1,100, L2) | 12.3 (766) | 11.8 (797) | 0.5 |
| ffn_down Q4_K 28.3 MB | 12.3 (2,295, L2) | 26.4 (1,073) | 23.9 (1,184) | 2.5 |
| ffn_down Q6_K 41.3 MB | 16.4 (2,522, L2) | 36.7 (1,124) | 32.7 (1,262) | 4.0 |
| ffn_gate_up Q4_K 28.3 MB | 14.3 (1,979, L2) | 25.5 (1,110) | 22.5 (1,258) | 3.0 |
| lm_head Q6_K 510.5 MB | 305.8 (1,670) | 328.7 (1,553) | 306.2 (1,667) | 22.5 |

## Verdict: the CUDA flush becomes a read sweep too

The same artefact, smaller. The store charges the next kernel 2.2 to 4.0 µs on the 3 to 41 MB roles (7 to 12%, 28%
on attn_kv Q6_K) and 22.5 µs on lm_head (7%). The read sweep evicts just as well (sweep vs warm differ two times on
ffn_down) and leaves the kernel at DRAM speed: lm_head after a read sweep 306.2 µs, warm 305.8, in-model 302.0;
after a store 328.7. The empty kernel is unchanged by either, so it is not launch cost; it is dirty lines the store
leaves in the L2, drained during the next kernel. Unlike the M4's fixed 50 to 60 µs, the cost grows with the bytes
the kernel pulls through the L2, about 22 µs on lm_head (roughly 38 MB of writeback at 1,694 GB/s). One rule, a read
sweep, fits both backends; the size stays per backend.

Two CUDA-only facts for the timer: "no flush" is never an option on CUDA (every role under 96 MiB becomes an L2 read,
117 to 149% of peak), and the 4 µs event floor must be subtracted or shown per row, or the small roles read 2 to 2.5
times high and no sum fits the token. The floor is already measured and recorded beside every row
(`timing.dispatch_floor_us`); it is never applied.

## Is the generic path generic in nature?

In one sentence: the timing loop, the tie-out and the pipeline ran on the 5090 unchanged and the in-model nsys path
worked first time, but the generic kernel-timer path produced no CUDA number at all on main until two patches
(symbol matcher, q8_1 reference); after them correctness passed 7 of 7 at 1e-7, read sweep less floor matched nsys
within 4% on 6 of 7 roles, the sum fit the token (0.85 vs 0.83), and the store-flush drain appeared here too, so the
method is generic and the CUDA adapter as shipped was not yet.

Manual steps the run needed, each a mark against:

1. A patch to `cuda_device.match_symbol`: without it zero roles were timed (issue 1).
2. A patch so the correctness check reads the q8_1 vector: without it every role failed (issue 2).
3. A wrapper so the two patches applied in process.
4. `PATH=/usr/local/cuda/bin:$PATH`: nvcc is found by fallback, `cu++filt` only via PATH.
5. `BOLTBEAM_LLAMA_BENCH` and `BOLTBEAM_GGML_CUDA_SRC` pointing at the engine (allowed; the default would also
   have found the source).
6. `TMPDIR` and `BOLTBEAM_CUDA_CACHE` under the data disk: the box's full root disk, not BoltBeam.
7. Stopping and restarting the resident llama-server: the box, not BoltBeam (`gpu_free` correctly refuses a held
   GPU).
8. Removing `llama_decode.dot` the pipeline dropped into the clone's working directory (issue 7).
9. The warm, read-sweep and floor-subtracted columns are a hand script, as on the M4; no flag produces them.

## Issues found in BoltBeam main ec01234

1. `runtime/cuda_device.py` `match_symbol` with `collectors/engine_kernels.py` (CUDA spec): the exact name
   `mul_mat_vec_q<(ggml_type)12, 1, false, false>` matches nothing because this `mmvq.cu` has a fifth template
   parameter (`halve_iters`); `cu++filt` prints `mul_mat_vec_q<(ggml_type)12, (int)1, (bool)0, (bool)0, (bool)0>`.
   The cubin held 395 function symbols, 271 of them `mul_mat_vec_q<...>` from the file's own launch tables. The
   error lists `sorted(...)[:8]`, eight `$__internal` helpers, hiding the real kernels. A prefix match on the
   normalised name is needed.
2. `engine_kernels._reference` with `metal_native.TOLERANCE` 1e-3: the CUDA kernel reads the vector as q8_1
   (`quantize_q8_1`), the reference uses the f32 vector. Every role fails by 2.3e-3 to 5.1e-3. Against the
   dequantized q8_1 vector the error is 1.0e-7 to 1.8e-7.
3. `CACHE_BYTES["CUDA"]` and the row cache flag: 6 of 7 roles "inconclusive, cache" while the flush evicts. Already
   removed in main 0837b15 (this run cloned ec01234).
4. `kernel_timer.FLUSH["CUDA"]` is a store: the drain above.
5. The dispatch floor is recorded beside every row and never applied; on CUDA it is 30 to 130% of the small roles'
   time.
6. `summary.md` of both NVIDIA runs says "Not reported by this GPU: MTLComputePipelineState does not report
   registers; Metal does not export the compiled AGX ISA", "Analysis status: needs_measurement",
   "timing_inconclusive", and every role "inconclusive via missing_evidence", after a run that measured them all.
7. The pipeline leaves `llama_decode.dot` (a 421 KB ggml graph dump) in the working directory.
8. Nothing checks that the compiled kernel source matches the traced binary (2026-08-20 binary, 2026-10-05 source).
   A current build may launch 8 warps via `halve_iters` while the adapter instantiates `false`.
9. `docs/kernel-timer.md`'s Ubuntu section clones a branch that no longer exists.

## Timings

In-model pipeline 45 s (machine 5.7, measure_timing 4.5, role_time 34.1 including two nsys captures and the
5.8 MB mmvq cubin compile). Generic pipeline 7 s (role_time 0.9 with the cached cubin). Flush probe 0.9 s of GPU.
The resident llama-server was down 55 s and 19 s and came back both times.

## Sources

Prediction, scripts, every JSON result, both logs and the nsys CSVs are kept on the Air under the runs work folder,
`boltbeam-runs/.work/in-model-vs-generic-rtx5090-20261010/`; the box keeps a copy under its data disk.

Not measured: tinygrad on the 5090; llama.cpp built from the current source; the generic numbers under `halve_iters`
(8 warps); batch above 1.
