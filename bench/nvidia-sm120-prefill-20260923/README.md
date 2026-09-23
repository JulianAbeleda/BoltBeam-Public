# RTX 5090 prefill against the clock, and against the wrong ceiling

Companion to `bench/nvidia-sm120-facts-20260923`, taken the same day on the same card. That entry
measured the chip's facts. This one spends them on reading a prompt, and finds that the harder
question is not how fast prefill runs but which ceiling it should be divided by.

## The run

llama.cpp, Qwen3-8B Q4_K_M, `llama-bench -p 512 -n 0 -r 10 -ngl 99`, graphics clock pinned.

| Clock sustained | Prefill | fp16 matrix peak there | Share of it |
|---:|---:|---:|---:|
| 2,040 MHz | 10,683.9 ± 555.5 tok/s | 164.1 TF | 99% |
| 2,392 MHz | 12,121.3 ± 704.7 tok/s | 192.4 TF | 95% |
| 2,662 MHz | 13,808.0 ± 881.2 tok/s | 214.1 TF | 98% |
| 2,835 MHz | 14,323.2 ± 907.6 tok/s | 228.0 TF | 95% |

The peak column is the measured 512 FLOP per SM per clock at 92.4%, the efficiency the matrix probe
reached at two independent clocks.

## Prefill tracks the clock, and decode does not

Across the same card and the same clock range:

| | Clock range | Throughput range | Share of proportional |
|---|---:|---:|---:|
| Prefill, pp512 | +39.0% | +34.1% | 87% |
| Decode, tg128 | +43.9% | +22.1% | 50% |

That is the regime showing up in a wall-clock measurement. Reading a prompt uses each weight 512
times, about 1,017 sums per byte against a ridge of 140, so it is compute work and it follows the
clock. Writing a token reads the weights once, so two thirds of it is memory, and memory speed does
not move when the core clock does: the streaming read measures 1,692.7 to 1,694.3 GB/s whether the
core runs at 2,662 MHz or 2,947.

## The ceiling is not one number

95 to 99% of peak is not a triumph, it is a warning. Real kernels carry loads, epilogues and
attention, and they do not run at 99% of a matrix unit. The denominator is wrong.

Prefill at 14,323 tok/s has to retire 7.75 TFLOP in 35.7 ms, which is 216.8 TFLOP/s of effective
work. Three instructions on this card could supply it, and they are not close together:

| Path | Measured here, 2,947 MHz pinned | Could it supply 217 TFLOP/s? |
|---|---:|---|
| `mma.sync` m16n8k16, fp16 in, fp32 accumulate | 237.1 TF | Only at 95% of peak, which no real kernel does |
| `mma.sync` m16n8k32, s8 in, s32 accumulate | 947.0 TOPS | Comfortably, at 24% of it |
| `dp4a`, the integer-pipe dot product | 7.6 INT8 TOPS | No, by a factor of 28 |

The integer tensor rate was measured after the first draft of this note said it had not been. It is
exactly 4.00 times the fp16 rate, and both probes reach the same share of their architectural rate,
92.4% of 512 ops per SM per clock for fp16 and 92.3% of 2,048 for INT8. At NVIDIA's 2.407 GHz
reference boost those architectural rates are 209.5 TFLOPS and 838.0 TOPS, which are the published
dense figures. The numbers NVIDIA leads with, 419 and 1,676, are those with sparsity.

llama.cpp's CUDA library ships PTX and is compiled at load, so the choice is made at runtime. The
PTX carries every path: 39,216 fp16 `m16n8k16` sites, 38,880 s8 `m16n8k32` sites, 26,688 s8
`m16n8k16` sites, and 13,900 `dp4a` sites.

So the honest reading is that prefill is not at 95% of its ceiling. Both ends are now measured, and
they are a factor of four apart:

| If the kernel is | Ceiling at 2,835 MHz | Prefill sits at |
|---|---:|---:|
| all fp16 matrix | 228.1 TF | 95% |
| all INT8 matrix | 911.0 TOPS | 24% |

Which mix ran is measured. One pp512 run under `nsys` (`nsys-kernel-summary.csv` here), kernel time
by family:

| Family | Share of kernel time | Instruction it issues |
|---|---:|---|
| `mul_mat_q` (MMQ: Q8_1 activations) | 81.4% | `IMMA`, the s8 matrix instruction |
| `flash_attn_ext_f16` | 5.2% | `HMMA`, fp16 |
| `quantize_mmq_q8_1` | 2.3% | none (activations to int8) |
| `mul_mat_vec` | 1.0% | vector |
| norm, rope, silu, copy | 10.0% | none |

The instruction column is not inferred from names. The fork's prefill lifecycle audit disassembled
these kernels and found `IMMA.16832` in the Q4_K `mul_mat_q` and `IMMA.16816` in the Q6_K one, with
no HMMA and no dp4a ([nv-llama-prefill-lifecycle-audit](https://github.com/JulianAbeleda/tinygrad-arkey/blob/exp/docs/task_workflow/output/nv-llama-prefill-lifecycle-audit.md)).
That audit put MMQ at 82.4% of the prompt and counted 214 Q4_K and 35 Q6_K launches; this capture
shows 81.4% and 428 and 70, exactly double because llama-bench ran a warmup prompt first.

So the INT8 line is the ceiling for 81% of the time and effectively all of the sums. At 2,835 MHz:

| | Rate | Share of the 911.0 TOPS INT8 ceiling |
|---|---:|---:|
| Whole prompt, 7.75 TFLOP in 35.7 ms | 216.8 TOPS | 23.8% |
| Inside the MMQ kernels only, 81.4% of that time | 266.7 TOPS | 29.3% |

Prefill on this card runs its matrix kernels at under a third of the unit's measured rate. The 95%
reading was off by a factor of four in the direction that says "nothing left".

## What this does not say

- One card, one model, one runtime, one context length.
- One nsys capture, one session, unpinned clock for the capture itself. The shares are stable against
  the fork's audit from a different session, which is why they are believed.
- The dp4a figure rules that path out for prefill but is itself low enough to be worth re-checking;
  1.9 dp4a per SM per clock is far below what the integer pipe should sustain.
- llama.cpp only. tinygrad's pp512 on this card is recorded elsewhere at 84.0 ms against llama's
  36.6, and that comparison is not what was run here.

## Reproducing it

```
llama-bench -m Qwen3-8B-Q4_K_M.gguf -p 512 -n 0 -r 10 -ngl 99
sudo nvidia-smi -lgc <clock> ; <run> ; sudo nvidia-smi -rgc
cuobjdump --dump-ptx libggml-cuda.so | grep -oE 'mma\.sync\.aligned\.m[0-9]+n[0-9]+k[0-9]+[a-z0-9.]*'
```


## Why the matrix kernels stop at 29%

The obvious guess was weight streaming. `mul_mat_q` tiles the prompt 128 tokens at a time, and one
Q4_K weight byte carries 128 tokens x 2 ops / 0.5625 bytes = 455 ops. The INT8 crossover on this
card is 947.0 / 1693.3 = 559 ops per byte, so a 128-token tile is on the memory side of the
roofline even though the whole 512-token prompt, at 1,017, is on the compute side. That predicted
a DRAM-bound kernel and a prefill rate flat in context length.

Half right. Prefill is flat in context, and dramatically so:

| Context | Prefill, clock pinned at 2,655 MHz |
|---:|---:|
| 128 | 8,001 ± 1,452 tok/s |
| 256 | 11,252.8 ± 769.1 |
| 512 | 13,870.1 ± 562.8 |
| 1,024 | 13,958.7 ± 5.4 |
| 2,048 | 13,828.5 ± 3.9 |

From 512 to 2,048 the rate moves 0.3% across a fourfold context. Work per token is constant, so
nothing amortises past the point where the grid fills the machine. The climb from 128 to 512 is
that filling: one token-tile does not have enough blocks for 170 multiprocessors.

The reason was not DRAM. `ncu` on the `mul_mat_q` launches:

| | Q4_K, busiest launch | Q6_K |
|---|---:|---:|
| IMMA, the matrix pipe | 31.6% | 33.3% |
| LSU, load and store | **40.0%** | 28.8% |
| L1TEX | **40.0%** | 28.8% |
| FMA, the float pipe | 29.6% | 14.1% |
| ALU | 12.1% | 10.4% |
| Issue slots active | 49.5% | 31.1% |
| DRAM throughput | 7.5% | 8.5% |
| L2 hit rate | 87.9% | 80.0% |

DRAM is idle at 7.5% and L2 catches 88% of the traffic, so the weights are not being re-streamed
per tile: they fit and they stay. The prediction was wrong about the mechanism while being right
about the shape.

What is busiest is load-store and L1, at 40%, above the matrix pipe at 31.6%. Nothing is
saturated and the machine issues on half its slots. Reading a Q4_K weight is not one load: it is
an unpack, a scale, and a staging write into shared memory before IMMA can see it, and that work
is proportional to the matrix work rather than amortised by it. The FMA pipe at 29.6% is the
scales.

So the matrix unit is idle two thirds of the time because the kernel is busy preparing its
operands, in L1 and shared memory rather than from DRAM. The 31.6% IMMA figure is an independent
confirmation of the 29.3% computed from wall clock at the top of this note, by a different
instrument.

`ncu` serialises and replays kernels, so the durations in those CSVs are not wall-clock times.
The percentages are what they are for.
