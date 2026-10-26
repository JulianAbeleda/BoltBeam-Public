"""The one kernel timing loop: a kernel, its real buffers, a correctness check, warm launches, timed launches after a
cache flush, the median. Every isolated kernel time BoltBeam reports comes from here.

    spec = KernelSpec(...)                 what to run: source, kernel, arguments, grid, block, what the output must match
    bridge = bridge_for(backend)           Metal, CUDA or HIP (runtime/metal_device.py, cuda_device.py, hip_device.py), one surface
    result = time_spec(bridge, spec)       correctness, samples, median µs, GB/s

Adapters only produce KernelSpecs, one per role: BoltBeam's own GEMVs (collectors/boltbeam_gemv.py), the engine's
shipped kernels (collectors/engine_kernels.py: llama.cpp from ggml's Metal and CUDA sources). The loop knows no
model and no engine; the spec carries its label ("llama.cpp kernel_mul_mv_q4_K_f32") into every result.

The bandwidth probes (metal_bandwidth.py, read_bandwidth.py) and the cubin replay (cubin_launch.py) time their
launches with `samples` too: there is no second warm-and-time loop in the tree. They keep their own statistic
(best of N for a bandwidth), stated where they use it.

Flushing: before each timed sample a kernel sweeps a buffer larger than the chip's last-level cache, so the timed
kernel reads its weights from DRAM. The sizes are per backend (FLUSH): 256 MiB on Apple (the system level cache),
256 MiB on NVIDIA (a 5090's L2 is 96 MiB; a copy-engine copy does not evict it, a kernel does). One read pass over a
buffer the size of the cache does not evict reliably: on an M5 a 64 MiB sweep left a 9.4 MB weight 94 to 151 GB/s
against a 132 GB/s DRAM read, round to round, where 256 MiB and 1 GiB both held it at 119 to 120. A cache's size is
not readable on every backend, so a size alone cannot be trusted: with the chip's read bandwidth (dram_gbs) a swept
kernel that reads faster than DRAM is a cache hit the sweep left behind, and time_spec doubles the sweep and times it
again (CACHE_RULE); the row records every pass. The sweep READS on
both: a store leaves dirty lines that drain into DRAM during the next kernel. On an M4 that was a fixed cost of about
60 us per launch, which made a 2.4 MB kernel 2 to 4 times slower and a 28 MB kernel 20% slower than inside the model
(docs/in-model-vs-generic-m4-20261010.md). On a 5090 the cost grew with the bytes the kernel pulled through the L2:
7 to 12% on the 3 to 41 MB roles and 22 us on lm_head (docs/in-model-vs-generic-rtx5090-20261010.md). A read evicts
without that: the recorded sums are 0.82 of the M3 token and, less floor, 0.85 of the 5090 token; the latter is
within 3% of its in-model weight-kernel sum (docs/kernel-timer.md). "No flush" is impossible by design: a 96 MiB L2 makes every weight under it an L2 hit between warmup and
sample (117 to 149% of DRAM peak on the 5090), and that is not the number a model sees.

Warm activations: after the sweep the args a spec marks `warm` (the activation vector) are read back in by the same
kernel (WARM_RULE). Inside the model the kernel before wrote the vector and it is still in the cache: Nsight Compute
on a 5090 decode reads exactly the weight bytes from DRAM per launch, and the timer swept the vector out too
(weights plus 4.6 KB). With the read-back the timed launch reads exactly the weights, as in the model. Its time moved
by 0.4 µs or less (docs/kernel-timer.md, "Checked against Nsight Compute").

The dispatch floor (an empty kernel between the same two timestamps, after warmups, each after a sweep as a timed
launch is) is measured once per timer run and recorded on each row as
timing.dispatch_floor_us. It is never subtracted silently: a row keeps us_per_call as measured and carries
us_per_call_less_floor (less_floor_us, the one rule) for the tie-out, which says when it uses it.
"""
from __future__ import annotations

import statistics
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from boltbeam.core.policy import value
from boltbeam.target import vendors

SAMPLES, WARMUPS = 20, 10  # warmups let the GPU clock ramp up before the first timed sample
# the dispatch floor: an empty kernel's launches, each after a sweep as every timed launch is, after WARMUPS. Measured
# with no warmups and no sweep it read 4.2 to 6.4 µs on a 5090 between runs; this way 3.8 to 4.0, and Nsight Compute
# puts what a timed kernel carries over its own duration at 2.9 to 4.3 (docs/kernel-timer.md, "Checked against
# Nsight Compute")
FLOOR_SAMPLES = 60
# a kernel whose median is under the policy noisy_floor_multiple x the dispatch floor is mostly floor, and the floor's own jitter
# is most of its spread (attn_kv Q6_K on a 5090 read 8.05 then 10.30 µs between two passes): it is sampled
# NOISY_SAMPLES times in all, the spread reported, and the row says why. One rule for both backends.
NOISY_SAMPLES = 60

# the flush kernels per source dialect (target/vendors.py kernel_dialect): a sweep over a buffer larger than the
# last-level cache, run before each timed sample, and an empty kernel for the dispatch floor
FLUSH_SOURCES = {
  "msl": """
#include <metal_stdlib>
using namespace metal;
kernel void sweep(device const float* b [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  if (i < n && b[i] == 12345.0f) o[0] = 1.0f; }
kernel void empty(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xFFFFFFFF) b[0] = 0.0f; }
""",
  "cuda": """
extern "C" __global__ void sweep(const float* b, float* o, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && b[i] == 12345.0f) o[0] = 1.0f; }
extern "C" __global__ void empty(float* b) { if (blockIdx.x * blockDim.x + threadIdx.x == 0xFFFFFFFFu) b[0] = 0.0f; }
""",
}
FLUSH_SOURCES["hip"] = FLUSH_SOURCES["cuda"]  # hipcc compiles CUDA's kernel syntax as is
# the flush per backend with a kernel bridge, its size from the vendor's row; "mode" says what the sweep does to the
# cache lines: "read" leaves them clean, "store" leaves them dirty
FLUSH = {v.backend: {"bytes": v.flush_bytes, "kernel": "sweep", "mode": "read", "empty": "empty", "threads": 256,
                     "source": FLUSH_SOURCES[v.kernel_dialect]} for v in vendors.VENDORS.values() if v.bridge}

WARM_RULE = ("the activation vector is read back in after the sweep: inside the model the kernel before wrote it and it is "
             "still in the cache, the weights are not")

# a swept kernel reading faster than the chip's DRAM read (beyond CACHE_MARGIN, the probe's own run-to-run spread) read
# cached weights: the sweep doubles, up to MAX_SWEEP_BYTES and half the working set, and the kernel is timed again
CACHE_MARGIN, MAX_SWEEP_BYTES = 0.05, 4 << 30
CACHE_RULE = ("a swept kernel read faster than this GPU's measured DRAM read: its weights were still in the cache, so the "
              "sweep was doubled and the kernel timed again")


def less_floor_us(us:float, floor_us:float | None, *, at_dram_us:float | None = None) -> float | None:
  """A launch's time less the dispatch floor measured beside it, never below 0; None without a floor. The one rule
  for every "less the floor" number: the row writes it, the tie-out reads it and says so. With at_dram_us (the
  launch's bytes at the GPU's measured DRAM read) it never goes below that either: the floor is not an additive cost,
  and what of it would make a kernel outrun DRAM ran under the kernel's own reads (an M5's 3.4 MB GEMV read 102.9% of
  peak less a 2.9 µs floor, 94% as measured)."""
  if floor_us is None:
    return None
  less = max(0.0, us - floor_us)
  return less if at_dram_us is None else min(us, max(less, at_dram_us))


def bridge_for(backend:str, **kw:Any):
  """The GPU bridge for a target backend; the same surface on both: library, pipeline, buffer, read, dispatch,
  release, close, name, working_set_bytes."""
  row = vendors.BY_BACKEND.get(backend)
  if row is None or row.bridge is None:
    have = " and ".join(v.backend for v in vendors.VENDORS.values() if v.bridge)
    raise RuntimeError(f"BoltBeam has no kernel bridge for {backend}; {have} have one")
  return vendors.resolve(row.bridge)(**kw)


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
  keep_output: bool = False  # return the first launch's output bytes (a chained launch reads them as its input)
  # the args the model's previous kernel has just written (the activation vector): read back in after each sweep, so
  # they are in the cache as in the model while the weights come from DRAM (WARM_RULE)
  warm: tuple[int, ...] = ()


class Flusher:
  """The backend's sweep kernel and empty kernel, compiled once on a bridge. The sweep binds a 16-byte sink as its
  second buffer so the compiler cannot drop the loads, and the count of floats to read as its third."""

  def __init__(self, bridge, backend:str):
    row = FLUSH[backend]
    self.row = dict(row)  # its own copy: grow() changes this flusher's size, not the backend's default
    lib = bridge.library(row["source"])
    self.flush_pso = bridge.pipeline(lib, row["kernel"])["pso"]
    self.empty_pso = bridge.pipeline(lib, row["empty"])["pso"]
    self.buf = bridge.buffer(length=row["bytes"])
    self.small = bridge.buffer(length=16)
    self.bridge = bridge

  def read(self, buf:int, nbytes:int) -> float:
    """Read nbytes of a buffer through the sweep kernel: what is read is left in the cache, clean."""
    words = nbytes // 4
    groups = max(1, -(-words // self.row["threads"]))
    return self.bridge.dispatch(self.flush_pso, [buf, self.small, struct.pack("<I", words)], (groups, 1, 1), (self.row["threads"], 1, 1))

  def grow(self, limit:int) -> bool:
    """Double the swept buffer, not past `limit` bytes; False when it cannot grow."""
    size = min(self.row["bytes"] * 2, limit)
    if size <= self.row["bytes"]:
      return False
    self.bridge.release(self.buf)
    self.buf, self.row["bytes"] = self.bridge.buffer(length=size), size
    return True

  def __call__(self, warm:list[tuple[int, int]] = ()) -> float:
    """The sweep, then each (buffer, bytes) in `warm` read back in: the weights leave the cache, the activations stay."""
    us = self.read(self.buf, self.row["bytes"])
    for buf, nbytes in warm:
      self.read(buf, nbytes)
    return us

  def floor_us(self, count:int = FLOOR_SAMPLES) -> float:
    """The median interval around an empty launch under the same protocol, reported beside every time.
    This includes the empty kernel itself and does not establish an additive correction for other kernels."""
    launch = lambda: self.bridge.dispatch(self.empty_pso, [self.small], (1, 1, 1), (32, 1, 1))  # noqa: E731
    return statistics.median(samples(launch, warmups=WARMUPS, count=count, before=self))


def _percentile(values:list[float], q:float) -> float:
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def more_samples(samples_us:list[float], floor_us:float | None) -> tuple[int, str | None]:
  """How many more samples a kernel needs and why: (count, words) under the floor rule, (0, None) otherwise. The
  one rule: a median under the policy noisy_floor_multiple x the dispatch floor is sampled NOISY_SAMPLES times in all."""
  if floor_us is None or not samples_us:
    return 0, None
  med, multiple = statistics.median(samples_us), value("noisy_floor_multiple")
  if med >= multiple * floor_us or len(samples_us) >= NOISY_SAMPLES:
    return 0, None
  return NOISY_SAMPLES - len(samples_us), (f"median {med:.1f} µs is under {multiple:.0f} x the {floor_us:.1f} µs dispatch "
                                           f"floor: sampled {NOISY_SAMPLES} times, the spread reported")


def time_spec(bridge, spec:KernelSpec, flush:Flusher | None = None, *, warmups:int = WARMUPS, count:int = SAMPLES,
              libraries:dict[str, int] | None = None, floor_us:float | None = None,
              dram_gbs:float | None = None) -> dict[str, Any]:
  """Compile (a library per source is reused through `libraries`), bind, check the first output, then time.
  Returns {"correctness", "samples", "median_us", "min_us", "spread_pct", "gbs", "pipeline", "label"}; with a
  failed check the samples are empty and nothing was timed. With the run's dispatch floor (floor_us) a kernel
  under the policy noisy_floor_multiple x it is sampled NOISY_SAMPLES times (more_samples); timing.more_samples says so.
  With the GPU's DRAM read (dram_gbs) a swept kernel faster than it is timed again behind a doubled sweep (CACHE_RULE);
  timing.cache_check lists each pass."""
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
  warm = [(bound[i], len(spec.args[i])) for i in spec.warm]
  before = (lambda: flush(warm)) if flush is not None and warm else flush
  try:
    launch = lambda: bridge.dispatch(pipeline["pso"], bound, spec.grid, spec.block, spec.shared_bytes)  # noqa: E731
    launch()
    if spec.keep_output and out_index is not None:
      result["output"] = bytes(bridge.read(owned[out_index], out_len))
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
    def timed() -> tuple[list[float], str | None]:
      got = samples(launch, warmups=warmups, count=count, before=before)
      extra, why = more_samples(got, floor_us)
      return (got + samples(launch, warmups=0, count=extra, before=before), why) if extra else (got, why)
    start = time.monotonic()  # the timed launches' window (time.monotonic, as telemetry samples are stamped)
    result["samples"], why_more = timed()
    passes, read_gbs = [], lambda: spec.bytes_read / (statistics.median(result["samples"]) * 1e3)  # noqa: E731
    while flush is not None and dram_gbs and read_gbs() > dram_gbs * (1 + CACHE_MARGIN):
      passes.append({"flush_bytes": flush.row["bytes"], "gbs": round(read_gbs(), 1)})
      if not flush.grow(min(MAX_SWEEP_BYTES, bridge.working_set_bytes // 2 - spec.bytes_read)):
        break
      result["samples"], why_more = timed()
    result["window"] = [start, time.monotonic()]
  finally:
    for buf in owned:
      bridge.release(buf)
  med = statistics.median(result["samples"])
  result.update(median_us=med, min_us=min(result["samples"]),
                spread_pct=100.0 * (_percentile(result["samples"], 0.9) - _percentile(result["samples"], 0.1)) / med,
                gbs=spec.bytes_read / (med * 1e3), warmups=warmups,
                timing={"warmups": warmups, "samples": len(result["samples"]), "cache": f"swept ({flush.row['mode']})" if flush else "not flushed",
                        "flush_bytes": flush.row["bytes"] if flush else 0, "flush_mode": flush.row["mode"] if flush else None,
                        "clock": bridge.CLOCK, "more_samples": why_more,
                        "warm": [WARM_RULE, sum(n for _, n in warm)] if flush and warm else None,
                        "cache_check": {"rule": CACHE_RULE, "dram_gbs": dram_gbs, "passes": passes,
                                        "settled": spec.bytes_read / (med * 1e3) <= dram_gbs * (1 + CACHE_MARGIN)}
                                       if passes else None})
  return result
