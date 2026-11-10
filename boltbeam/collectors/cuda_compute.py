"""The achievable tensor-core rate of one NVIDIA GPU per matrix path (FP16, BF16, INT8, FP8, FP4), measured with a
small native kernel through BoltBeam's CUDA bridge, with the SM clock, power and temperature it ran at.

    python -m boltbeam.collectors.cuda_compute [--device N] [--seconds 2]

A prefill kernel's ceiling is its work over the rate of the instruction it issues. The rates differ by path (an
integer matrix instruction runs at four times the fp16 one on some chips), so a ceiling taken from the wrong path is
wrong by that factor. Each path here is one PTX matrix instruction (PATHS, data): the kernel keeps ACC independent
accumulators per warp in registers and issues the instruction in a loop, so no memory is read and the rate is the
unit's own. A path whose instruction this GPU or toolkit cannot compile is recorded as not available, with nvcc's
words; nothing is assumed about which chip has which path. One path is not a matrix instruction: `fp32_fma`, the
vector ALU's fused multiply-add, the rate of a kernel that issues no matrix instruction at all (a recurrent block's
scan); it is measured the same way, with each lane's accumulators in registers. A warpgroup path (Hopper's
`wgmma`, kind wgmma) reads its operands from tiles in shared memory, on chip, and is tried at each width of WGMMA_N:
its rate is the best width's, so a ceiling from it bounds a kernel of any width.

The figure kept is the plateau, not a burst: the best launch shape is found with a few short samples, then run back
to back for `seconds` while the telemetry sampler (gpu_telemetry.py) logs the clock; the rate is the median launch of
the second half of that series and the clock is the mean SM clock over the same half. ops_per_sm_per_clock divides
the rate by the SM count and that clock, so a rate can be read at another clock.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import statistics
import struct
import subprocess
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
          "in registers, no memory read (wgmma: one accumulator, operands in shared memory); best launch shape of {reps} samples each, then that shape back to back for "
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
  kind: str = "mma"  # mma: one warp-wide matrix instruction; fma: one fused multiply-add per lane per accumulator;
  # wgmma: one asynchronous matrix instruction per warpgroup (four warps), its operands tiles in shared memory
  scale_bits: int = 0x7F7F7F7F  # unit UE8M0 scales; NVFP4 uses unit UE4M3 instead


PATHS = (
  Path("fp16", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 16, 8, 16, words="fp16 in, fp32 accumulate"),
  Path("fp16_f16acc", "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", 16, 8, 16, acc="f16",
       words="fp16 in, fp16 accumulate"),
  Path("bf16", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 16, 8, 16, words="bf16 in, fp32 accumulate"),
  Path("tf32", "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", 16, 8, 8, words="tf32 in, fp32 accumulate"),
  Path("int8", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 16, 8, 32, acc="s32", words="int8 in, int32 accumulate"),
  Path("fp8", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", 16, 8, 32, words="fp8 e4m3 in, fp32 accumulate"),
  Path("fp4", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
       16, 8, 64, arch_suffix="a", extra=", {sa}, {{0, 0}}, {sb}, {{0, 0}}",
       words="fp4 e2m1 in with ue8m0 block scales, fp32 accumulate"),
  Path("fp32_fma", "fma.rn.f32", 1, 1, 1, kind="fma", words="fp32 vector fused multiply-add (no matrix instruction)"),
  Path("nvfp4", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
       16, 8, 64, arch_suffix="a", extra=", {sa}, {{0, 0}}, {sb}, {{0, 0}}",
       words="fp4 e2m1 in with ue4m3 block-16 scales, fp32 accumulate", scale_bits=0x38383838),
  Path("fp16_wgmma", "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16", 64, 128, 16, arch_suffix="a", kind="wgmma",
       words="fp16 in, fp32 accumulate, per warpgroup"),
  Path("bf16_wgmma", "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", 64, 128, 16, arch_suffix="a", kind="wgmma",
       words="bf16 in, fp32 accumulate, per warpgroup"),
  Path("fp16_k8", "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", 16, 8, 8, a_regs=2, b_regs=1,
       words="fp16 in, fp32 accumulate, K 8"),  # FlashInfer's Gated-DeltaNet precompute issues it beside m16n8k16
)
WGMMA_N = (8, 64, 128, 256)  # the widths a warpgroup instruction is tried at


def source(p:Path) -> str:
  """The probe kernel for one path."""
  if p.kind != "mma":
    return _fma_source(p) if p.kind == "fma" else _wgmma_source(p)
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
  unsigned sa = {p.scale_bits}u, sb = {p.scale_bits}u;
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


def _fma_source(p:Path) -> str:
  """A vector path: each lane keeps ACC x FMA_LANES independent fp32 accumulators and issues one `fma` on each per
  step, so the rate is the vector ALU's, with no memory read."""
  return f"""
extern "C" __global__ void mma_loop(float* out, int iters) {{
  float a = 1.0f + threadIdx.x * 1e-7f, b = 0.999999f;
  float c[{ACC}][{FMA_LANES}];
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {FMA_LANES}; i++) c[k][i] = (float)(k * {FMA_LANES} + i);
  for (int it = 0; it < iters; it++) {{
#pragma unroll
    for (int k = 0; k < {ACC}; k++)
#pragma unroll
      for (int i = 0; i < {FMA_LANES}; i++)
        asm volatile("{p.ptx} %0, %1, %2, %0;" : "+f"(c[k][i]) : "f"(a), "f"(b));
  }}
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {FMA_LANES}; i++) s += c[k][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}}
"""


def _wgmma_source(p:Path) -> str:
  """A warpgroup path: four warps issue one asynchronous matrix instruction on an M x K and an N x K tile in shared
  memory (written once, before the loop), ACC of them per committed group onto one accumulator of N / 2 registers a
  thread, as a GEMM's main loop does. The tile descriptor is CUTLASS's (mma_sm90_desc: K-major, 128-byte swizzle)."""
  regs = p.n // 2
  d = ", ".join(f"%{i}" for i in range(regs))
  outs = ", ".join(f'"+f"(c[{i}])' for i in range(regs))
  return f"""
static __device__ unsigned long long tile(unsigned short* t) {{  // start, 16-byte leading and 1024-byte stride offsets
  return ((__cvta_generic_to_shared(t) & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}}
extern "C" __global__ void mma_loop(float* out, int iters) {{
  __shared__ __align__(1024) unsigned short a[{p.m} * 64];
  __shared__ __align__(1024) unsigned short b[{p.n} * 64];
  for (int i = threadIdx.x; i < {max(p.m, p.n)} * 64; i += blockDim.x) {{
    if (i < {p.m} * 64) a[i] = (i * 0x9E37u) & 0x3F3Fu;
    if (i < {p.n} * 64) b[i] = (i * 0x85EBu) & 0x3F3Fu;
  }}
  __syncthreads();
  unsigned long long da = tile(a), db = tile(b);
  float c[{regs}];
#pragma unroll
  for (int i = 0; i < {regs}; i++) c[i] = 0.0f;
  for (int it = 0; it < iters; it++) {{
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < {ACC}; k++)
      asm volatile("{{ .reg .pred p; setp.ne.b32 p, %{regs + 2}, 0; {p.ptx} {{{d}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0; }}"
                   : {outs} : "l"(da), "l"(db), "r"(1));
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }}
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < {regs}; i++) {{ asm volatile("" : "+f"(c[i]) :: "memory"); s += c[i]; }}
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}}
"""


FMA_LANES = 4  # fp32 accumulators per lane per ACC group: ACC x FMA_LANES independent chains hide the FMA latency


def ops_per_launch(p:Path, blocks:int, threads:int, iters:int) -> float:
  if p.kind == "fma":
    return blocks * threads * iters * ACC * FMA_LANES * 2.0
  return blocks * (threads // (128 if p.kind == "wgmma" else 32)) * iters * ACC * 2.0 * p.m * p.n * p.k


def _library(cuda, p:Path, facts:dict[str, Any]) -> int:
  """The probe compiled for this GPU: -arch=native, or this GPU's own sm_XY with the path's suffix."""
  if not p.arch_suffix:
    return cuda.library(source(p))
  from boltbeam.runtime.cuda_device import build_cubin
  arch = f"sm_{facts['compute_capability_major']}{facts['compute_capability_minor']}{p.arch_suffix}"
  return cuda.module(build_cubin(cuda.nvcc, source(p), cache=cuda.cache, arch=arch))


def measure_path(cuda, p:Path, facts:dict[str, Any], tele:gpu_telemetry.Telemetry, *, reps:int = 5,
                 seconds:float = SECONDS, iters:int = ITERS) -> dict[str, Any]:
  """One path's plateau rate, or why it is not available here. A warpgroup path is tried at every width of WGMMA_N."""
  tried = [dataclasses.replace(p, n=n, ptx=p.ptx.replace(f"n{p.n}k", f"n{n}k")) for n in WGMMA_N] if p.kind == "wgmma" else [p]
  try:
    psos = [cuda.pipeline(_library(cuda, v, facts), "mma_loop")["pso"] for v in tried]
  except RuntimeError as exc:
    return {"path": p.name, "instruction": p.ptx, "words": p.words, "available": False,
            "reason": f"does not compile for this GPU: {str(exc).strip().splitlines()[-1][:300]}"}
  sms = facts["sm_count"] or 1
  out = cuda.buffer(length=512 * 8 * sms * 4)
  arg = struct.pack("<i", iters)
  shapes = []
  for (v, pso), (threads, per_sm) in itertools.product(zip(tried, psos), SHAPES):
    blocks = per_sm * sms
    try:  # a block the SM's register file cannot hold is refused at launch (CUresult 701: wgmma N 256 at 512 threads, H200)
      best = min(samples(lambda: cuda.dispatch(pso, [out, arg], (blocks, 1, 1), (threads, 1, 1)), warmups=1, count=reps))
    except RuntimeError as exc:
      refused = str(exc).strip().splitlines()[-1][:300]
      continue
    shapes.append({"threads": threads, "blocks": blocks, "best_us": best, **({"n": v.n} if len(tried) > 1 else {}),
                   "tops": ops_per_launch(v, blocks, threads, iters) / (best * 1e-6) / 1e12})
  if not shapes:
    return {"path": p.name, "instruction": p.ptx, "words": p.words, "available": False, "reason": f"no launch shape ran: {refused}"}
  top = max(shapes, key=lambda r: r["tops"])
  p, pso = next((v, x) for v, x in zip(tried, psos) if v.n == top.get("n", p.n))
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
  return {"path": p.name, "instruction": p.ptx, "words": p.words, "available": True, "tops": round(rate, 1),
          "burst_tops": round(top["tops"], 1), "shape": {"threads": top["threads"], "blocks": top["blocks"]},
          "shapes": shapes, "plateau_launches": len(half), "clock": clock, "sm_mhz": mhz,
          "ops_per_sm_per_clock": round(rate * 1e12 / (sms * mhz * 1e6), 1) if mhz else None}


def measure_paths(device:int = 0, *, reps:int = 5, seconds:float = SECONDS, nvcc:str | None = None,
                  bridge=None, paths:tuple[Path, ...] | None = None) -> dict[str, Any]:
  """Every path's plateau rate on one GPU, the clock policy, and the driver's device facts. With no `paths`, the
  named paths the installed toolkit's scan finds for this GPU's sm (cuda_path_scan; its unmapped ones are measured
  only when asked for), else PATHS."""
  from boltbeam.collectors import cuda_path_scan
  from boltbeam.runtime.cuda_device import Cuda
  policy = gpu_telemetry.clock_policy(device)
  cuda = bridge or Cuda(device, nvcc=nvcc)
  try:
    facts = cuda.facts()
    sm = f"sm_{facts.get('compute_capability_major')}{facts.get('compute_capability_minor')}"
    try:  # a scan that cannot run leaves PATHS
      scanned = (cuda_path_scan.scan(sm, nvcc=cuda.nvcc) if paths is None and getattr(cuda, "nvcc", None)
                 and facts.get("compute_capability_major") is not None else None)
    except (RuntimeError, OSError, subprocess.SubprocessError):
      scanned = None
    paths = paths or tuple(p for p in cuda_path_scan.paths(scanned) if p.name not in scanned["unmapped"]) or PATHS
    with gpu_telemetry.hold(policy, device) as held, gpu_telemetry.Telemetry(device) as tele:
      rows = [measure_path(cuda, p, facts, tele, reps=reps, seconds=seconds) for p in paths]
    whole = tele.summary()
  finally:
    if bridge is None:
      cuda.close()
  from boltbeam.collectors.read_bandwidth import FACT_KEYS
  return {"schema": "boltbeam.cuda_matrix_paths.v1", "device": device, "name": facts.get("name"),
          **{k: facts.get(k) for k in FACT_KEYS},
          "compute_capability": f"{facts.get('compute_capability_major')}.{facts.get('compute_capability_minor')}",
          "clock_policy": held, "telemetry": whole, "paths": rows,
          "path_scan": scanned if cuda_path_scan.paths(scanned) else None,
          "method": METHOD.format(ptx="per path", acc=ACC, reps=reps, seconds=seconds)}


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.cuda_compute")
  p.add_argument("--device", type=int, default=0)
  p.add_argument("--seconds", type=float, default=SECONDS)
  p.add_argument("--reps", type=int, default=5)
  p.add_argument("--runs", type=int, default=1, help="whole probe runs; each path's uncertainty is their spread at the best shape")
  p.add_argument("--path", action="append", choices=[x.name for x in PATHS], help="only this path (repeatable)")
  a = p.parse_args(argv)
  paths = tuple(x for x in PATHS if x.name in a.path) if a.path else None
  from boltbeam.collectors.read_bandwidth import repeated_paths
  runs = [measure_paths(a.device, reps=a.reps, seconds=a.seconds, paths=paths) for _ in range(a.runs)]
  got = repeated_paths(runs)
  sys.stdout.write(json.dumps(got, indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
