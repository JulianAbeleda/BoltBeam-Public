"""The achievable tensor-core rate of one NVIDIA GPU per matrix path (FP16, BF16, INT8, FP8, FP4), measured with a
small native kernel through BoltBeam's CUDA bridge, with the SM clock, power and temperature it ran at.

    python -m boltbeam.collectors.cuda_compute [--device N] [--seconds 2]

A prefill kernel's ceiling is its work over the rate of the instruction it issues. The rates differ by path (an
integer matrix instruction runs at four times the fp16 one on some chips), so a ceiling taken from the wrong path is
wrong by that factor. Each path here is one PTX matrix instruction (PATHS, data): the kernel keeps ACC independent
accumulators per warp in registers and issues the instruction in a loop, so no memory is read and the rate is the
unit's own. A path whose instruction this GPU or toolkit cannot compile is recorded as not available, with nvcc's
words; nothing is assumed about which chip has which path.

The figure kept is the plateau, not a burst: the best launch shape is found with a few short samples, then run back
to back for `seconds` while the telemetry sampler (gpu_telemetry.py) logs the clock; the rate is the median launch of
the second half of that series and the clock is the mean SM clock over the same half. ops_per_sm_per_clock divides
the rate by the SM count and that clock, so a rate can be read at another clock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

from boltbeam.collectors import gpu_telemetry
from boltbeam.collectors.kernel_timer import samples

ACC = 8  # independent accumulators per warp: enough to hide the instruction's latency
SHAPES = ((128, 4), (256, 2), (256, 4), (512, 2))  # threads per block, blocks per SM
ITERS = 4096
SECONDS = 2.0  # the plateau run per path
METHOD = ("BoltBeam cuda_compute: native CUDA, one PTX matrix instruction ({ptx}) in a loop, {acc} accumulators per warp "
          "in registers, no memory read; best launch shape of {reps} samples each, then that shape back to back for "
          "{seconds:.0f} s: the median launch of the second half, at the mean SM clock over it")


@dataclass(frozen=True)
class Path:
  """One matrix path: the PTX instruction, its m n k, its register operands and the accumulator's kind. arch_suffix
  names a PTX feature set beyond the base architecture (`a`: arch-specific instructions), compiled for this GPU's
  own compute capability with that suffix. extra is operand text after C (block scales)."""
  name: str
  ptx: str
  m: int
  n: int
  k: int
  a_regs: int = 4
  b_regs: int = 2
  acc: str = "f32"  # f32: 4 floats; s32: 4 ints; f16: 2 packed halves
  arch_suffix: str = ""
  extra: str = ""
  words: str = ""


PATHS = (
  Path("fp16", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 16, 8, 16, words="fp16 in, fp32 accumulate"),
  Path("fp16_f16acc", "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", 16, 8, 16, acc="f16",
       words="fp16 in, fp16 accumulate"),
  Path("bf16", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 16, 8, 16, words="bf16 in, fp32 accumulate"),
  Path("int8", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 16, 8, 32, acc="s32", words="int8 in, int32 accumulate"),
  Path("fp8", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", 16, 8, 32, words="fp8 e4m3 in, fp32 accumulate"),
  Path("fp4", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
       16, 8, 64, arch_suffix="a", extra=", {sa}, {{0, 0}}, {sb}, {{0, 0}}",
       words="fp4 e2m1 in with ue8m0 block scales, fp32 accumulate"),
)


def source(p:Path) -> str:
  """The probe kernel for one path."""
  acc_n = {"f32": 4, "s32": 4, "f16": 2}[p.acc]
  ctype, cons = {"f32": ("float", "f"), "s32": ("int", "r"), "f16": ("unsigned", "r")}[p.acc]
  d = ", ".join(f"%{i}" for i in range(acc_n))
  a = ", ".join(f"%{acc_n + i}" for i in range(p.a_regs))
  b = ", ".join(f"%{acc_n + p.a_regs + i}" for i in range(p.b_regs))
  n_in = acc_n + p.a_regs + p.b_regs
  extra = p.extra.format(sa=f"%{n_in}", sb=f"%{n_in + 1}") if p.extra else ""
  scales = ', "r"(sa), "r"(sb)' if p.extra else ""
  outs = ", ".join(f'"+{cons}"(c[k][{i}])' for i in range(acc_n))
  ins = ", ".join([f'"r"(a[{i}])' for i in range(p.a_regs)] + [f'"r"(b[{i}])' for i in range(p.b_regs)])
  return f"""
extern "C" __global__ void mma_loop(float* out, int iters) {{
  unsigned a[{p.a_regs}], b[{p.b_regs}];
  unsigned seed = 0x9E3779B9u * (threadIdx.x + 1);
#pragma unroll
  for (int i = 0; i < {p.a_regs}; i++) a[i] = (seed ^ (i * 0x85EBCA6Bu)) & 0x3F3F3F3Fu;
#pragma unroll
  for (int i = 0; i < {p.b_regs}; i++) b[i] = (seed ^ (i * 0xC2B2AE35u)) & 0x3F3F3F3Fu;
  unsigned sa = 0x7F7F7F7Fu, sb = 0x7F7F7F7Fu;
  {ctype} c[{ACC}][{acc_n}];
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {acc_n}; i++) c[k][i] = 0;
  for (int it = 0; it < iters; it++) {{
#pragma unroll
    for (int k = 0; k < {ACC}; k++)
      asm volatile("{p.ptx} {{{d}}}, {{{a}}}, {{{b}}}, {{{d}}}{extra};" : {outs} : {ins}{scales});
  }}
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {acc_n}; i++) s += (float)c[k][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}}
"""


def ops_per_launch(p:Path, blocks:int, threads:int, iters:int) -> float:
  return blocks * (threads // 32) * iters * ACC * 2.0 * p.m * p.n * p.k


def _library(cuda, p:Path, facts:dict[str, Any]) -> int:
  """The probe compiled for this GPU: -arch=native, or this GPU's own sm_XY with the path's suffix."""
  if not p.arch_suffix:
    return cuda.library(source(p))
  from boltbeam.runtime.cuda_device import build_cubin
  arch = f"sm_{facts['compute_capability_major']}{facts['compute_capability_minor']}{p.arch_suffix}"
  return cuda.module(build_cubin(cuda.nvcc, source(p), cache=cuda.cache, arch=arch))


def measure_path(cuda, p:Path, facts:dict[str, Any], tele:gpu_telemetry.Telemetry, *, reps:int = 5,
                 seconds:float = SECONDS, iters:int = ITERS) -> dict[str, Any]:
  """One path's plateau rate, or why it is not available here."""
  try:
    pso = cuda.pipeline(_library(cuda, p, facts), "mma_loop")["pso"]
  except RuntimeError as exc:
    return {"path": p.name, "ptx": p.ptx, "words": p.words, "available": False,
            "reason": f"does not compile for this GPU: {str(exc).strip().splitlines()[-1][:300]}"}
  sms = facts["sm_count"] or 1
  out = cuda.buffer(length=512 * 8 * sms * 4)
  arg = struct.pack("<i", iters)
  shapes = []
  for threads, per_sm in SHAPES:
    blocks = per_sm * sms
    best = min(samples(lambda: cuda.dispatch(pso, [out, arg], (blocks, 1, 1), (threads, 1, 1)), warmups=1, count=reps))
    shapes.append({"threads": threads, "blocks": blocks, "best_us": best,
                   "tops": ops_per_launch(p, blocks, threads, iters) / (best * 1e-6) / 1e12})
  top = max(shapes, key=lambda r: r["tops"])
  series, t_start = [], time.monotonic()
  while time.monotonic() - t_start < seconds or len(series) < 4:
    t0 = time.monotonic()
    us = cuda.dispatch(pso, [out, arg], (top["blocks"], 1, 1), (top["threads"], 1, 1))
    series.append((t0, us))
  half = series[len(series) // 2:]
  rate = ops_per_launch(p, top["blocks"], top["threads"], iters) / (statistics.median(u for _, u in half) * 1e-6) / 1e12
  clock = tele.summary(half[0][0], time.monotonic())
  mhz = (clock.get("sm_mhz") or {}).get("mean")
  cuda.release(out)
  return {"path": p.name, "ptx": p.ptx, "words": p.words, "available": True, "tops": round(rate, 1),
          "burst_tops": round(top["tops"], 1), "shape": {"threads": top["threads"], "blocks": top["blocks"]},
          "shapes": shapes, "plateau_launches": len(half), "clock": clock, "sm_mhz": mhz,
          "ops_per_sm_per_clock": round(rate * 1e12 / (sms * mhz * 1e6), 1) if mhz else None}


def measure_paths(device:int = 0, *, reps:int = 5, seconds:float = SECONDS, nvcc:str | None = None,
                  bridge=None, paths:tuple[Path, ...] = PATHS) -> dict[str, Any]:
  """Every path's plateau rate on one GPU, the clock policy, and the driver's device facts."""
  from boltbeam.runtime.cuda_device import Cuda
  policy = gpu_telemetry.clock_policy(device)
  cuda = bridge or Cuda(device, nvcc=nvcc)
  try:
    facts = cuda.facts()
    with gpu_telemetry.hold(policy, device) as held, gpu_telemetry.Telemetry(device) as tele:
      rows = [measure_path(cuda, p, facts, tele, reps=reps, seconds=seconds) for p in paths]
    whole = tele.summary()
  finally:
    if bridge is None:
      cuda.close()
  from boltbeam.collectors.cuda_bandwidth import FACT_KEYS
  return {"schema": "boltbeam.cuda_matrix_paths.v1", "device": device, "name": facts.get("name"),
          **{k: facts.get(k) for k in FACT_KEYS},
          "compute_capability": f"{facts.get('compute_capability_major')}.{facts.get('compute_capability_minor')}",
          "clock_policy": held, "telemetry": whole, "paths": rows,
          "method": METHOD.format(ptx="per path", acc=ACC, reps=reps, seconds=seconds)}


def matrix_tflops(result:dict[str, Any]) -> dict[str, float]:
  """The chip profile's matrix_tflops from a probe result: the plateau rate of every available path."""
  return {r["path"]: r["tops"] for r in result["paths"] if r.get("available")}


def fact_source(result:dict[str, Any], *, scope:str, today:str) -> dict[str, Any]:
  """The chip profile's fact source for matrix_tflops: per path its rate, the clock it was measured at, its
  instruction and the rate per SM per clock; the policy and whether anything throttled."""
  return {"kind": "measurement", "bound": "achievable", "scope": scope, "observed_at": today, "method": result["method"],
          "clock_policy": result["clock_policy"].get("policy"), "throttled": result["telemetry"].get("throttled"),
          "paths": {r["path"]: ({"tops": r["tops"], "sm_mhz": r["sm_mhz"], "ops_per_sm_per_clock": r["ops_per_sm_per_clock"],
                                 "instruction": r["ptx"], "words": r["words"], "throttled": r["clock"].get("throttled"),
                                 "power_w_mean": (r["clock"].get("power_w") or {}).get("mean")}
                                if r.get("available") else {"available": False, "reason": r["reason"], "instruction": r["ptx"]})
                    for r in result["paths"]}}


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.cuda_compute")
  p.add_argument("--device", type=int, default=0)
  p.add_argument("--seconds", type=float, default=SECONDS)
  p.add_argument("--reps", type=int, default=5)
  a = p.parse_args(argv)
  sys.stdout.write(json.dumps(measure_paths(a.device, reps=a.reps, seconds=a.seconds), indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
