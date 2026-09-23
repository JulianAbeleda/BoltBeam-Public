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
