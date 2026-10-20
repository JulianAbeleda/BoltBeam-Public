# Guide: time your own kernel against the roofline

You wrote a kernel. You want to know how close it gets to the chip's limit. BoltBeam's kernel timer answers that
for one kernel on its own, with the same loop that times every isolated kernel BoltBeam reports.

This guide times a hand-written Metal GEMV on a MacBook Air (M3). The script below ran on 2026-10-12. Its output is
copied as it printed.

## What the timer does

The loop is `kernel_timer.samples` in `boltbeam/collectors/kernel_timer.py`. You describe the kernel in a
`KernelSpec`. `time_spec` runs it:

1. It compiles your source on the GPU bridge (Metal or CUDA).
2. It uploads your argument bytes.
3. It launches once and checks the output against your reference. A failed check stops here. Nothing is timed.
4. It launches 10 times to warm the clock.
5. It launches 20 times. Before each launch it sweeps a buffer larger than the last-level cache (64 MiB on Apple,
   256 MiB on NVIDIA). Your weights then come from DRAM, as they do in a model.
6. It returns the median µs, the spread, and GB/s.

A kernel whose median is under 3 times the dispatch floor is sampled 60 times, and the result says so.

## What you give it

| field | what it is |
|---|---|
| `source` | the kernel source: Metal Shading Language, or CUDA C for the CUDA bridge |
| `kernel` | the function name in the source |
| `args` | in buffer order. `bytes` is an input buffer. An `int` is a zeroed output buffer of that many bytes. `("value", bytes)` is passed by value (a scalar or a struct) |
| `grid`, `block` | threadgroups (Metal) or blocks (CUDA) in the grid, and threads per group |
| `bytes_read` | the bytes one launch must read. GB/s and the ideal time come from this number |
| `check` | the correctness reference: output indices to check, a function that gives the expected value, a relative tolerance |
| `warm` | indices of arguments to read back into the cache after the sweep. Use it for the activation vector: in a model the kernel before just wrote it |
| `label`, `adapter` | names carried into the result |

`bytes_read` is your claim about the work. Count every byte the kernel must read once: weights, activations, scales.
The roofline is that number over the chip's measured bandwidth. A wrong `bytes_read` gives a wrong percent of peak.

## The worked example

The kernel computes `y = W x` for float32 `W`. One simdgroup of 32 threads takes one row. Save this as
`time_my_gemv.py` and run it from the BoltBeam checkout.

```python
"""Time one hand-written Metal kernel in BoltBeam's kernel timer and read it against the roofline."""
import random, struct, sys
from boltbeam.collectors import kernel_timer as kt
from boltbeam.math.roofline import floor_us
from boltbeam.target import targets as reg

SOURCE = """
#include <metal_stdlib>
using namespace metal;
// y[r] = sum_c W[r, c] * x[c]: one simdgroup (32 threads) per row, a strided dot, then simd_sum
kernel void gemv_f32(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]], constant uint& cols [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float acc = 0.0f;
  for (uint c = lane; c < cols; c += 32) acc += W[row * cols + c] * x[c];
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}
"""

def spec(rows, cols):
  rnd = random.Random(0)
  W = [rnd.uniform(-1, 1) for _ in range(rows * cols)]
  x = [rnd.uniform(-1, 1) for _ in range(cols)]
  check = kt.Check(indices=[0, rows // 2, rows - 1], rel_tol=1e-4, words="pure-Python dot of the row",
                   reference=lambda r: sum(W[r * cols + c] * x[c] for c in range(cols)))
  return kt.KernelSpec(
    label=f"my gemv_f32 {rows}x{cols}", adapter="example", source=SOURCE, kernel="gemv_f32",
    args=[struct.pack(f"<{rows*cols}f", *W), struct.pack(f"<{cols}f", *x), rows * 4, ("value", struct.pack("<I", cols))],
    grid=(rows, 1, 1), block=(32, 1, 1),            # threadgroups in the grid, threads per threadgroup
    bytes_read=(rows * cols + cols) * 4, check=check, warm=(1,))   # x is the activation: kept warm, W comes from DRAM

target = reg.TARGETS[sys.argv[1] if len(sys.argv) > 1 else "apple_m3_10c"]
peak = target.memory_bandwidth_gbs
bridge = kt.bridge_for(target.backend)
flush = kt.Flusher(bridge, target.backend)
floor = flush.floor_us()
print(f"{bridge.name}: peak {peak} GB/s ({target.target_id}), dispatch floor {floor:.1f} µs")
for rows, cols in [(4096, 2048), (1024, 512)]:
  s = spec(rows, cols)
  r = kt.time_spec(bridge, s, flush, floor_us=floor)
  ideal = floor_us(s.bytes_read, peak)
  less = kt.less_floor_us(r["median_us"], floor)
  print(f"{s.label}: check {'pass' if r['correctness']['passed'] else 'FAIL'}; "
        f"{s.bytes_read/1e6:.1f} MB; median {r['median_us']:.1f} µs (spread {r['spread_pct']:.0f}%, {r['timing']['samples']} samples); "
        f"less floor {less:.1f} µs; ideal {ideal:.1f} µs; {100*ideal/less:.0f}% of peak less floor, {100*ideal/r['median_us']:.0f}% as measured"
        + (f"; {r['timing']['more_samples']}" if r['timing']['more_samples'] else ""))
```

```bash
python3.12 time_my_gemv.py              # the chip id is optional; apple_m3_10c is the default here
```

Three runs on the M3, about one second each:

```text
Apple M3: peak 97.2 GB/s (apple_m3_10c), dispatch floor 5.7 µs
my gemv_f32 4096x2048: check pass; 33.6 MB; median 454.3 µs (spread 2%, 20 samples); less floor 448.6 µs; ideal 345.3 µs; 77% of peak less floor, 76% as measured
my gemv_f32 1024x512: check pass; 2.1 MB; median 38.5 µs (spread 5%, 20 samples); less floor 32.7 µs; ideal 21.6 µs; 66% of peak less floor, 56% as measured

Apple M3: peak 97.2 GB/s (apple_m3_10c), dispatch floor 5.7 µs
my gemv_f32 4096x2048: ... median 455.8 µs (spread 4%) ... 77% of peak less floor, 76% as measured
my gemv_f32 1024x512:  ... median 37.3 µs (spread 13%) ... 68% of peak less floor, 58% as measured

Apple M3: peak 97.2 GB/s (apple_m3_10c), dispatch floor 5.8 µs
my gemv_f32 4096x2048: ... median 459.4 µs (spread 5%) ... 76% of peak less floor, 75% as measured
my gemv_f32 1024x512:  ... median 39.1 µs (spread 13%) ... 65% of peak less floor, 55% as measured
```

## Reading the result

- **check pass.** The first launch matched the pure-Python reference within 1e-4. Without a pass there is no time.
- **ideal.** `bytes_read` over the chip's bandwidth: 33.6 MB at 97.2 GB/s is 345.3 µs. No kernel that reads these
  bytes can be faster on this chip.
- **% of peak.** The ideal over the measured time. The big shape reaches 76 to 77% in all three runs. The answer is
  stable: this kernel leaves about a quarter of the bandwidth unused.
- **less floor.** The median less the dispatch floor (an empty kernel timed the same way). It is printed beside the
  measured time. It is never a replacement for it. See the limits below.
- **spread.** The P90 minus P10 of the samples, over the median. Over ±10% (a 20% spread) the row is noisy.

The small shape tells a different story in each run: 65 to 68% less floor, 55 to 58% as measured, spread 5 to 13%.
Its 38 µs is about 7 dispatch floors. Read it as "well below the limit", not as a precise percent.

The peak is the chip row's measured bandwidth (`boltbeam/data/targets.json`, or the profile `autoscan` saved for
this GPU). If the chip has no measured bandwidth, `memory_bandwidth_gbs` is empty and the ideal cannot be computed.
Measure it first with `python -m boltbeam.workflow.screen autoscan` (about a minute; `boltbeam doctor` says when it
is needed). Do not borrow another chip's number.

## The timer's measured limits

These come from the RTX 5090 accuracy work. They apply to any kernel you time this way.

- **Small kernels are good to about ±12%.** A 3 µs recurrent projection read −10.7% against Nsight Compute in one
  battery and +12.7% in the next (`accuracy-battery-rtx5090-20261012.md`,
  section 4). Three later repeats of the same kernel gave 1.8, 2.7 and 3.4 µs less floor against a steady 2.7 µs in
  Nsight (`accuracy-fixes-20261012.md`). The empty-launch subtraction is an estimate,
  not a correction. Under about 3 dispatch floors, treat the number as a rough size.
- **Large kernels are close.** lm_head (614 µs) read −0.1%. The Q8_0 projections of 22 to 41 µs read within 3%.
- **Fused routes differ.** In a model, llama.cpp runs some projections as one fused launch. The timer runs the plain
  kernel alone. On the 5090 the isolated times for fused ffn_down and gate/up were 5.2% and 7.4% under the in-model
  Nsight times. That gap is a different route, not timer noise. Compare a fused launch with the fused launch only.
- **Cold weights, warm activations.** The sweep sends the weights to DRAM. The `warm` arguments are read back in.
  This matches a decode step. It does not match a kernel whose weights stay in cache across calls.

## On NVIDIA

`bridge_for("CUDA")` returns the CUDA bridge (`boltbeam/runtime/cuda_device.py`). It compiles CUDA C. Give the
kernel `extern "C"` linkage, pass a CUDA source, and set `grid` and `block` in CUDA's terms. The same script with a
CUDA kernel and `nvidia_sm120` measures against the 5090's 1,693 GB/s. This guide's example was run on Metal only.

## Where it fits

This is the generic timer, one kernel alone. To time an engine's own kernels for every role of a model, BoltBeam
uses the same loop through an adapter (`collectors/engine_kernels.py`). [kernel-timer.md](kernel-timer.md) has the
adapters and the full measurement history. [guide-add-an-engine.md](guide-add-an-engine.md) covers a new engine.
