# RTX 5090 target facts

The phase 0 measurement behind the `nvidia_sm120` registry row, taken on 2026-09-23. Every number in that
row that says `measurement` or `hardware_scan` comes from `scan.txt` here.

## Machine

- GPU: NVIDIA GeForce RTX 5090, driver 595.84, 32,607 MiB
- compute capability 12.0, 170 SMs, warp 32, 102,400 bytes of shared memory per SM
- maximum graphics clock 3,135 MHz

## What was measured

| fact | value | how |
| --- | --- | --- |
| streaming read bandwidth | 1,693.3 GB/s | `bw_peak_cuda.cu read`, 4 GiB, 32 passes, three runs within 0.04% |
| copy bandwidth | 1,493.9 GB/s | `bw_peak_cuda.cu copy`, 4 GiB, 16 passes |
| matrix unit, fp16 in and fp32 accumulate | 237.1 TFLOP/s | `mma_peak_cuda.cu`, `mma.sync.aligned.m16n8k16`, register resident |

Both probes live in the tinygrad fork under `extra/llm_research/microbench/`.

## Two things the numbers say

**The bus figure is arithmetic, not a measurement.** The driver reports a 14.001 GHz memory clock on a
512-bit bus, which is 1,792.1 GB/s. That is where the number on the box comes from. A real streaming read
reaches 1,693.3, or 94.5% of it. The roofline uses the measured one.

**A matrix peak is only a fact with its clock.** The same kernel reads 237.1 TFLOP/s with the graphics
clock pinned, where the card sustained 2,947 MHz, and 214.5 TFLOP/s at the 2,662 MHz it chooses on its
own. The accumulator count does not matter: 4, 8, 12, 16 and 20 all land within 0.1 TFLOP/s, so the
kernel is saturating the unit rather than waiting on dependencies. Clocks were pinned for the timing
window and restored afterwards, which is the measurement policy in `clock_pin.py`.

An earlier session recorded 1,700 GB/s and 255.4 TFLOP/s for this card. The bandwidth agrees to 0.4%.
The matrix figure does not, and the clock is the reason: 255.4 implies roughly 3,175 MHz, above this
card's stated maximum.

## Reproducing it

```
nvcc -O3 -arch=sm_120 bw_peak_cuda.cu -o bw_peak
nvcc -O3 -arch=sm_120 -DNACC=8 mma_peak_cuda.cu -o mma_peak
./bw_peak read 4294967296 32
sudo nvidia-smi -lgc 3135 && ./mma_peak ; sudo nvidia-smi -rgc
```

## What the clock does to tokens per second

Decode is mostly a memory problem, and the memory speed does not move with the graphics clock: the
streaming read measures 1,692.7 to 1,694.3 GB/s whether the core runs at 2,662 MHz or 2,947. But about a
third of a decoded token is not memory. It is launches, small kernels and latency, and that part is core
clock work.

llama.cpp, Qwen3-8B Q4_K_M, `llama-bench -p 0 -n 128 -r 3 -ngl 99`, graphics clock pinned:

| Clock pinned | Clock sustained | Decode |
|---:|---:|---:|
| 2,100 MHz | 2,032 MHz | 208.14 ± 1.05 tok/s |
| 2,400 MHz | 2,340 MHz | 231.58 ± 1.31 tok/s |
| 2,700 MHz | 2,677 MHz | 251.20 ± 1.47 tok/s |
| 3,135 MHz | 2,925 MHz | 254.15 ± 1.56 tok/s |

208 to 254 tokens per second, a 22% spread, on one card with one model and one runtime. The only thing
that changed was the clock.

This is worth knowing before reading any decode figure as a property of the software. A run that drifts
between 242 and 249 tokens per second has not changed; its card has settled somewhere around 2.5 to
2.7 GHz. A decode number quoted without its clock has a few per cent of slack in it that belongs to the
hardware, not to the kernel.
