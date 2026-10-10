"""The one kernel timing loop: a kernel, its real buffers, a correctness check, warm launches, timed launches after a
cache flush, the median. Every isolated kernel time BoltBeam reports comes from here.

    spec = KernelSpec(...)                 what to run: source, kernel, arguments, grid, block, what the output must match
    bridge = bridge_for(backend)           Metal (runtime/metal_device.py) or CUDA (runtime/cuda_device.py), the same surface
    result = time_spec(bridge, spec)       correctness, samples, median µs, GB/s

Adapters only produce KernelSpecs, one per role: BoltBeam's own GEMVs (collectors/boltbeam_gemv.py), the engine's
shipped kernels (collectors/engine_kernels.py: llama.cpp from ggml's Metal and CUDA sources). The loop knows no
model and no engine; the spec carries its label ("llama.cpp kernel_mul_mv_q4_K_f32") into every result.

The bandwidth probes (metal_bandwidth.py, cuda_bandwidth.py) and the cubin replay (cubin_launch.py) time their
launches with `samples` too: there is no second warm-and-time loop in the tree. They keep their own statistic
(best of N for a bandwidth), stated where they use it.

Flushing: before each timed sample a kernel sweeps a buffer larger than the chip's last-level cache, so the timed
kernel reads its weights from DRAM. The sizes are per backend (FLUSH): 64 MiB on Apple (the system level cache),
256 MiB on NVIDIA (a 5090's L2 is 96 MiB; a copy-engine copy does not evict it, a kernel does). The sweep READS on
both: a store leaves dirty lines that drain into DRAM during the next kernel. On an M4 that was a fixed cost of about
60 us per launch, which made a 2.4 MB kernel 2 to 4 times slower and a 28 MB kernel 20% slower than inside the model
(docs/in-model-vs-generic-m4-20261010.md). On a 5090 the cost grew with the bytes the kernel pulled through the L2:
7 to 12% on the 3 to 41 MB roles and 22 us on lm_head (docs/in-model-vs-generic-rtx5090-20261010.md). A read evicts
without that: the sum of the read-sweep times, less the dispatch floor, lands within 4% of the engine's own token on
both chips. "No flush" is impossible by design: a 96 MiB L2 makes every weight under it an L2 hit between warmup and
sample (117 to 149% of DRAM peak on the 5090), and that is not the number a model sees.

The dispatch floor (an empty kernel between the same two timestamps) is measured beside every row and recorded as
timing.dispatch_floor_us. It is never subtracted silently: a row keeps us_per_call as measured and carries
us_per_call_less_floor (less_floor_us, the one rule) for the tie-out, which says when it uses it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

SAMPLES, WARMUPS = 20, 10  # warmups let the GPU clock ramp up before the first timed sample
FLOOR_SAMPLES = 20  # launches of an empty kernel: the fixed cost of one dispatch on this GPU
# a kernel whose median is under NOISY_FLOOR_MULTIPLE x the dispatch floor is mostly floor, and the floor's own jitter
# is most of its spread (attn_kv Q6_K on a 5090 read 8.05 then 10.30 µs between two passes): it is sampled
# NOISY_SAMPLES times in all, the spread reported, and the row says why. One rule for both backends.
NOISY_FLOOR_MULTIPLE, NOISY_SAMPLES = 3.0, 60

# the flush kernel per backend: a sweep over a buffer larger than the last-level cache, run before each timed sample;
# "mode" says what the sweep does to the cache lines: "read" leaves them clean, "store" leaves them dirty
FLUSH = {
  "Metal": {"bytes": 64 << 20, "kernel": "sweep", "mode": "read", "empty": "empty", "threads": 256, "source": """
#include <metal_stdlib>
using namespace metal;
kernel void sweep(device const float* b [[buffer(0)]], device float* o [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (b[i] == 12345.0f) o[0] = 1.0f; }
kernel void empty(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xFFFFFFFF) b[0] = 0.0f; }
"""},
  "CUDA": {"bytes": 256 << 20, "kernel": "sweep", "mode": "read", "empty": "empty", "threads": 256, "source": """
extern "C" __global__ void sweep(const float* b, float* o) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b[i] == 12345.0f) o[0] = 1.0f; }
extern "C" __global__ void empty(float* b) { if (blockIdx.x * blockDim.x + threadIdx.x == 0xFFFFFFFFu) b[0] = 0.0f; }
"""},
}


def less_floor_us(us:float, floor_us:float | None) -> float | None:
  """A launch's time less the dispatch floor measured beside it, never below 0; None without a floor. The one rule
  for every "less the floor" number: the row writes it, the tie-out reads it and says so."""
  if floor_us is None:
    return None
  return max(0.0, us - floor_us)


def bridge_for(backend:str, **kw:Any):
  """The GPU bridge for a target backend; the same surface on both: library, pipeline, buffer, read, dispatch,
  release, close, name, working_set_bytes."""
  if backend == "Metal":
    from boltbeam.runtime.metal_device import Metal
    return Metal(**kw)
  if backend == "CUDA":
    from boltbeam.runtime.cuda_device import Cuda
    return Cuda(**kw)
  raise RuntimeError(f"BoltBeam has no kernel bridge for {backend}; Metal and CUDA have one")


def samples(launch:Callable[[], float], *, warmups:int = WARMUPS, count:int = SAMPLES,
            before:Callable[[], Any] | None = None) -> list[float]:
  """THE loop: `warmups` launches not counted, then `count` timed launches, `before()` (a flush) ahead of each.
  Returns the µs of each timed launch; the caller picks its statistic (median for a kernel, best for a bandwidth)."""
  for _ in range(warmups):
    launch()
  out = []
  for _ in range(count):
    if before is not None:
      before()
    out.append(launch())
  return out


@dataclass
class Check:
  """What the first launch's output must match: reference(i) is the expected value of output element i, for the
  given indices, within rel_tol of the largest expected magnitude. Output elements are float32 ("f"), or half
  ("e") for a kernel that stores fp16."""
  indices: list[int]
  reference: Callable[[int], float]
  rel_tol: float
  words: str = "pure-Python dequantize and dot"
  out_format: str = "f"


@dataclass
class KernelSpec:
  """One kernel to time. args in order: bytes ("in": uploaded), an int ("out": a zeroed buffer of that many bytes)
  or ("value", bytes) for an argument passed by value (a struct or scalar). On Metal a value is setBytes at that
  index; on CUDA it is the kernel parameter's own bytes."""
  label: str  # "llama.cpp kernel_mul_mv_q4_K_f32": the adapter and the kernel, as results name their source
  adapter: str
  source: str
  kernel: str
  args: list[Any]
  grid: tuple[int, int, int]
  block: tuple[int, int, int]
  bytes_read: int  # the bytes one launch reads: GB/s = bytes_read / time
  check: Check | None = None
  macros: dict[str, str] = field(default_factory=dict)
  constants: dict[int, tuple[str, int]] = field(default_factory=dict)
  shared_bytes: int = 0
  record: dict[str, Any] = field(default_factory=dict)  # the geometry and its origin, kept with the result


class Flusher:
  """The backend's sweep kernel and empty kernel, compiled once on a bridge. A "read" sweep binds a 16-byte sink as
  its second buffer so the compiler cannot drop the loads."""

  def __init__(self, bridge, backend:str):
    row = FLUSH[backend]
    self.row = row
    lib = bridge.library(row["source"])
    self.flush_pso = bridge.pipeline(lib, row["kernel"])["pso"]
    self.empty_pso = bridge.pipeline(lib, row["empty"])["pso"]
    self.buf = bridge.buffer(length=row["bytes"])
    self.small = bridge.buffer(length=16)
    self.bridge = bridge
    self.groups = row["bytes"] // 4 // row["threads"]

  def __call__(self) -> float:
    bufs = [self.buf, self.small] if self.row["mode"] == "read" else [self.buf]
    return self.bridge.dispatch(self.flush_pso, bufs, (self.groups, 1, 1), (self.row["threads"], 1, 1))

  def floor_us(self, count:int = FLOOR_SAMPLES) -> float:
    """The fixed GPU cost of one dispatch, the median over `count` empty launches: reported beside every time."""
    launch = lambda: self.bridge.dispatch(self.empty_pso, [self.small], (1, 1, 1), (32, 1, 1))  # noqa: E731
    return statistics.median(samples(launch, warmups=0, count=count))


def _percentile(values:list[float], q:float) -> float:
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def more_samples(samples_us:list[float], floor_us:float | None) -> tuple[int, str | None]:
  """How many more samples a kernel needs and why: (count, words) under the floor rule, (0, None) otherwise. The
  one rule: a median under NOISY_FLOOR_MULTIPLE x the dispatch floor is sampled NOISY_SAMPLES times in all."""
  if floor_us is None or not samples_us:
    return 0, None
  med = statistics.median(samples_us)
  if med >= NOISY_FLOOR_MULTIPLE * floor_us or len(samples_us) >= NOISY_SAMPLES:
    return 0, None
  return NOISY_SAMPLES - len(samples_us), (f"median {med:.1f} µs is under {NOISY_FLOOR_MULTIPLE:.0f} x the {floor_us:.1f} µs dispatch "
                                           f"floor: sampled {NOISY_SAMPLES} times, the spread reported")


def time_spec(bridge, spec:KernelSpec, flush:Flusher | None = None, *, warmups:int = WARMUPS, count:int = SAMPLES,
              libraries:dict[str, int] | None = None, floor_us:float | None = None) -> dict[str, Any]:
  """Compile (a library per source is reused through `libraries`), bind, check the first output, then time.
  Returns {"correctness", "samples", "median_us", "min_us", "spread_pct", "gbs", "pipeline", "label"}; with a
  failed check the samples are empty and nothing was timed. With the run's dispatch floor (floor_us) a kernel
  under NOISY_FLOOR_MULTIPLE x it is sampled NOISY_SAMPLES times (more_samples); timing.more_samples says so."""
  import math
  import struct
  libraries = libraries if libraries is not None else {}
  key = spec.source
  if key not in libraries:
    libraries[key] = bridge.library(spec.source, spec.macros or None)
  pipeline = bridge.pipeline(libraries[key], spec.kernel, spec.constants or None)
  bound, owned, out_index, out_len = [], [], None, 0
  for i, arg in enumerate(spec.args):
    if isinstance(arg, tuple) and arg[0] == "value":
      bound.append(bytes(arg[1]))
    elif isinstance(arg, (bytes, bytearray)):
      buf = bridge.buffer(bytes(arg))
      owned.append(buf)
      bound.append(buf)
    else:  # an output buffer of this many bytes
      buf = bridge.buffer(length=int(arg))
      owned.append(buf)
      bound.append(buf)
      out_index, out_len = len(owned) - 1, int(arg)
  result:dict[str, Any] = {"label": spec.label, "adapter": spec.adapter, "kernel": spec.kernel, "samples": [],
                           "correctness": None, "pipeline": {k: v for k, v in pipeline.items() if k != "pso"}}
  try:
    launch = lambda: bridge.dispatch(pipeline["pso"], bound, spec.grid, spec.block, spec.shared_bytes)  # noqa: E731
    launch()
    if spec.check is not None and out_index is not None:
      size = struct.calcsize(spec.check.out_format)
      got = struct.unpack(f"<{out_len // size}{spec.check.out_format}", bridge.read(owned[out_index], out_len))
      ref = {i: spec.check.reference(i) for i in spec.check.indices}
      scale = max(abs(v) for v in ref.values()) or 1.0
      err = max(abs(got[i] - ref[i]) for i in spec.check.indices) / scale
      passed = math.isfinite(err) and err <= spec.check.rel_tol  # NaN compares False both ways: fail closed
      result["correctness"] = {"checked_rows": list(spec.check.indices), "reference": spec.check.words, "max_rel_err": err,
                               "tolerance": spec.check.rel_tol, "passed": passed}
      if not all(math.isfinite(v) for v in ref.values()):
        result["correctness"]["reason"] = "the reference is not finite: the input bytes hold inf or NaN scales"
      elif not math.isfinite(err):
        result["correctness"]["reason"] = "the kernel's output is not finite"
      if not passed:
        return result
    result["samples"] = samples(launch, warmups=warmups, count=count, before=flush)
    extra, why_more = more_samples(result["samples"], floor_us)
    if extra:
      result["samples"] += samples(launch, warmups=0, count=extra, before=flush)
  finally:
    for buf in owned:
      bridge.release(buf)
  med = statistics.median(result["samples"])
  result.update(median_us=med, min_us=min(result["samples"]),
                spread_pct=100.0 * (_percentile(result["samples"], 0.9) - _percentile(result["samples"], 0.1)) / med,
                gbs=spec.bytes_read / (med * 1e3), warmups=warmups,
                timing={"warmups": warmups, "samples": len(result["samples"]), "cache": f"swept ({flush.row['mode']})" if flush else "not flushed",
                        "flush_bytes": flush.row["bytes"] if flush else 0, "flush_mode": flush.row["mode"] if flush else None,
                        "clock": bridge.CLOCK, "more_samples": why_more})
  return result
