"""The achievable matrix rate of one AMD GPU per matrix path, measured with a small native kernel through BoltBeam's
HIP bridge, with the clock, power and temperature it ran at.

    python -m boltbeam.collectors.hip_compute [--device N] [--seconds 2]

The AMD counterpart of cuda_compute.py, with the same rule: a prefill kernel's ceiling is its work over the rate of
the instruction it issues, and the rates differ by path. Each path here is one compiler builtin that lowers to one
matrix instruction (FAMILIES, data per ISA family: the wave32 WMMA builtins of gfx11 and of gfx12). The kernel keeps ACC independent accumulators per
wave in registers, feeds each instruction its own result, and writes every accumulator out, so no memory is read and
nothing is optimised away. A path this GPU's compiler rejects is recorded as not available with hipcc's words;
nothing is assumed about which chip has which path. `fp32_fma` is the vector ALU's fused multiply-add, the rate of a
kernel with no matrix instruction.

The figure kept is the plateau: the best launch shape from a few short samples, then that shape back to back for
`seconds` while the amdgpu telemetry (gpu_telemetry.AmdSysfs, found by the device's PCI address) logs the clock; the
rate is the median launch of the second half and the clock its mean over the same half. Launch grids are sized by
HIP's multiprocessor count, which on RDNA counts work-group processors; ops_per_sm_per_clock is per that unit. The
record's clock keys are every vendor's (prefill_work.CLOCK_KEY): sm_mhz is the core clock the path ran at.

The paths are rows rather than per-chip code, so this does not duplicate cuda_compute's PTX table: the instructions,
operand types and the record's key ("instruction", not "ptx") are AMD's own. The rows are found by scanning the
installed compiler for the device's gfx target (hip_path_scan; the record lands in the chip profile and paths_for
reads it back). FAMILIES is the verified reference that scan reproduces and the table where there is no compiler; a
target with neither is refused, never given another family's builtins. A path keeps its name across families where
the math is the same (fp16, bf16, fp16_f16acc, int8, int4).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from boltbeam.collectors import gpu_telemetry
from boltbeam.collectors.cuda_bandwidth import parallel_units, repeated_paths
from boltbeam.collectors.kernel_timer import samples

ACC = 8  # independent accumulators per wave: enough to hide the instruction's latency
FMA_LANES = 4  # fp32 accumulators per lane per ACC group on the vector path
SHAPES = ((128, 4), (256, 2), (256, 4), (512, 2))  # threads per block, blocks per multiprocessor
ITERS = 4096
SECONDS = 2.0  # the plateau run per path
METHOD = ("BoltBeam hip_compute: native HIP, one matrix builtin per path in a loop, {acc} accumulators per wave in "
          "registers, no memory read; best launch shape of {reps} samples each, then that shape back to back for "
          "{seconds:.0f} s: the median launch of the second half, at the mean core clock over it")

# operand vector types the builtins take (clang ext_vector_type), declared once in every probe
TYPES = """
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef __bf16 b8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
"""
# gfx11's A/B operands are 16 elements wide (gfx12's are 8); its bf16 builtins take 16 shorts, int8 four ints
TYPES11 = TYPES + """typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef int i4 __attribute__((ext_vector_type(4)));
"""


@dataclass(frozen=True)
class Path:
  """One matrix path: the builtin, its m n k per wave, the input operand's type and value, the accumulator's type,
  and the call with {a} and {c} for the input and accumulator. wave is the lane count the instruction spans."""
  name: str
  builtin: str
  m: int
  n: int
  k: int
  a_type: str
  a_init: str
  acc_type: str
  call: str
  words: str
  wave: int = 32
  kind: str = "mma"  # mma: one wave-wide matrix instruction; fma: one fused multiply-add per lane per accumulator
  types: str = TYPES  # the operand vector types this path's probe declares
  lanes: int = 8  # accumulator elements summed into the output
  mnemonic: str = ""  # the disassembled instruction the builtin lowers to
  mapped: bool = True  # a weight format maps to the name (hip_path_scan.NAMES); False: measured, unmapped


def _wmma(name:str, builtin:str, k:int, a_type:str, a_init:str, acc_type:str, words:str, *, signed:bool = False,
          tail:str = "", **kw) -> Path:
  call = f"{builtin}(true, {{a}}, true, {{a}}, {{c}}, false)" if signed else f"{builtin}({{a}}, {{a}}, {{c}}{tail})"
  mnemonic = "v_" + re.sub(r"_w32(_gfx12)?$", "", builtin.removeprefix("__builtin_amdgcn_"))
  return Path(name, builtin, 16, 16, k, a_type, a_init, acc_type, call, words, mnemonic=mnemonic, **kw)


FP32_FMA = Path("fp32_fma", "__builtin_fmaf", 1, 1, 1, "float", "", "float", "", "fp32 vector fused multiply-add (no matrix instruction)",
                kind="fma")

# gfx12 wave32 WMMA: each compiles to one v_wmma_* instruction for gfx1201 (checked by disassembly when added)
GFX12 = (
  _wmma("fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12", 16, "h8", "(h8)(_Float16)0.001f", "f8",
        "fp16 in, fp32 accumulate"),
  _wmma("fp16_f16acc", "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12", 16, "h8", "(h8)(_Float16)0.001f", "h8",
        "fp16 in, fp16 accumulate"),
  _wmma("bf16", "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12", 16, "b8", "(b8)(__bf16)0.001f", "f8",
        "bf16 in, fp32 accumulate"),
  _wmma("int8", "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12", 16, "i2", "(i2)0x01010101", "i8",
        "int8 in, int32 accumulate", signed=True),
  _wmma("int4", "__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12", 32, "i2", "(i2)0x11111111", "i8",
        "int4 in, int32 accumulate", signed=True),
  _wmma("fp8", "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12", 16, "i2", "(i2)0x38383838", "f8",
        "fp8 e4m3 in, fp32 accumulate"),
  _wmma("bf8", "__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12", 16, "i2", "(i2)0x3c3c3c3c", "f8",
        "fp8 e5m2 in, fp32 accumulate"),
  FP32_FMA,
)

# gfx11 wave32 WMMA (hipcc 17, ROCm 6.x, --offload-arch=gfx1100, -S): each compiles to exactly one v_wmma_* instruction,
# checked by disassembly when added: v_wmma_f32_16x16x16_f16, v_wmma_f16_16x16x16_f16, v_wmma_f32_16x16x16_bf16,
# v_wmma_i32_16x16x16_iu8, v_wmma_i32_16x16x16_iu4. There is no fp8/bf8 builtin on gfx11, and int4 is k=16 (gfx12: 32).
GFX11 = (
  _wmma("fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32", 16, "h16", "(h16)(_Float16)0.001f", "f8",
        "fp16 in, fp32 accumulate", types=TYPES11),
  _wmma("fp16_f16acc", "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32", 16, "h16", "(h16)(_Float16)0.001f", "h16",
        "fp16 in, fp16 accumulate", tail=", false", types=TYPES11, lanes=16),  # the trailing bool is OPSEL
  _wmma("bf16", "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32", 16, "s16", "(s16)0x3a83", "f8",
        "bf16 in, fp32 accumulate", types=TYPES11),
  _wmma("int8", "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32", 16, "i4", "(i4)0x01010101", "i8",
        "int8 in, int32 accumulate", signed=True, types=TYPES11),
  _wmma("int4", "__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32", 16, "i2", "(i2)0x11111111", "i8",
        "int4 in, int32 accumulate", signed=True, types=TYPES11),
  FP32_FMA,
)

FAMILIES = {"gfx11": GFX11, "gfx12": GFX12}  # keyed by the gfx target's first five characters


def family(arch:str | None) -> str:
  """The ISA family key of a gfx target; a target of no known family is refused."""
  if (key := (arch or "")[:5]) not in FAMILIES:
    raise RuntimeError(f"no matrix-path table for AMD target {arch!r} (known families: {', '.join(FAMILIES)}); "
                       "BoltBeam will not measure it with another family's instructions")
  return key


def paths_for(arch:str | None) -> tuple[Path, ...]:
  """The matrix paths of one gfx target: the scan a chip profile keeps for it (hip_path_scan), else its ISA
  family's verified table; a target with neither is refused."""
  from boltbeam.collectors.hip_path_scan import paths
  from boltbeam.target.targets import stored_path_scan
  return paths(stored_path_scan(arch)) or FAMILIES[family(arch)]


def source(p:Path) -> str:
  """The probe kernel for one path."""
  if p.kind == "fma":
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
      for (int i = 0; i < {FMA_LANES}; i++) c[k][i] = {p.builtin}(c[k][i], a, b);
  }}
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {FMA_LANES}; i++) s += c[k][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}}
"""
  step = p.call.format(a="a", c="c[k]")
  return f"""{p.types}
extern "C" __global__ void mma_loop(float* out, int iters) {{
  {p.a_type} a = {p.a_init};
  {p.acc_type} c[{ACC}];
#pragma unroll
  for (int k = 0; k < {ACC}; k++) c[k] = ({p.acc_type})0;
  for (int it = 0; it < iters; it++) {{
#pragma unroll
    for (int k = 0; k < {ACC}; k++) c[k] = {step};
  }}
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < {ACC}; k++)
#pragma unroll
    for (int i = 0; i < {p.lanes}; i++) s += (float)c[k][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}}
"""


def ops_per_launch(p:Path, blocks:int, threads:int, iters:int) -> float:
  if p.kind == "fma":
    return blocks * threads * iters * ACC * FMA_LANES * 2.0
  return blocks * (threads // p.wave) * iters * ACC * 2.0 * p.m * p.n * p.k


def measure_path(hip, p:Path, units:int, tele:gpu_telemetry.Telemetry, *, reps:int = 5, seconds:float = SECONDS,
                 iters:int = ITERS) -> dict[str, Any]:
  """One path's plateau rate, or why it is not available here."""
  try:
    pso = hip.pipeline(hip.library(source(p)), "mma_loop")["pso"]
  except RuntimeError as exc:
    return {"path": p.name, "instruction": p.builtin, "words": p.words, "available": False,
            "reason": f"does not compile for this GPU: {str(exc).strip().splitlines()[-1][:300]}"}
  out = hip.buffer(length=512 * 8 * units * 4)
  arg = struct.pack("<i", iters)
  shapes = []
  for threads, per_mp in SHAPES:
    blocks = per_mp * units
    best = min(samples(lambda: hip.dispatch(pso, [out, arg], (blocks, 1, 1), (threads, 1, 1)), warmups=1, count=reps))
    shapes.append({"threads": threads, "blocks": blocks, "best_us": best,
                   "tops": ops_per_launch(p, blocks, threads, iters) / (best * 1e-6) / 1e12})
  top = max(shapes, key=lambda r: r["tops"])
  series, t_start = [], time.monotonic()
  while time.monotonic() - t_start < seconds or len(series) < 4:
    t0 = time.monotonic()
    us = hip.dispatch(pso, [out, arg], (top["blocks"], 1, 1), (top["threads"], 1, 1))
    series.append((t0, us))
  half = series[len(series) // 2:]
  rate = ops_per_launch(p, top["blocks"], top["threads"], iters) / (statistics.median(u for _, u in half) * 1e-6) / 1e12
  clock = tele.summary(half[0][0], time.monotonic())
  mhz = (clock.get("sm_mhz") or {}).get("mean")
  hip.release(out)
  return {"path": p.name, "instruction": p.builtin, "words": p.words, "available": True, "tops": round(rate, 1),
          "burst_tops": round(top["tops"], 1), "shape": {"threads": top["threads"], "blocks": top["blocks"]},
          "shapes": shapes, "plateau_launches": len(half), "clock": clock, "sm_mhz": mhz,
          "ops_per_sm_per_clock": round(rate * 1e12 / (units * mhz * 1e6), 1) if mhz else None}


def measure_paths(device:int = 0, *, reps:int = 5, seconds:float = SECONDS, bridge=None,
                  only:tuple[str, ...] = ()) -> dict[str, Any]:
  """Every path's plateau rate on one GPU (its family's table, or just the `only` names), the clock policy, and the
  driver's device facts."""
  from boltbeam.collectors import hip_path_scan
  from boltbeam.runtime.hip_device import Hip
  hip = bridge or Hip(device)
  try:
    facts = hip.facts()
    # the installed compiler's scan for this target when the bridge has one (cached per compiler), else paths_for
    arch, hipcc = facts.get("architecture"), getattr(hip, "hipcc", None)
    try:  # a scan that cannot run leaves the table paths_for gives
      scanned = hip_path_scan.scan(arch, hipcc=hipcc) if arch and hipcc else None
    except (RuntimeError, OSError, subprocess.SubprocessError):
      scanned = None
    paths = [p for p in hip_path_scan.paths(scanned) or paths_for(arch) if not only or p.name in only]
    source_ = gpu_telemetry.AmdSysfs(facts["pci_bus_id"])
    policy = gpu_telemetry.amd_clock_policy(source_)
    with gpu_telemetry.Telemetry(device, source=source_) as tele:
      rows = [measure_path(hip, p, parallel_units(facts), tele, reps=reps, seconds=seconds) for p in paths]
    whole = tele.summary()
  finally:
    if bridge is None:
      hip.close()
  return {"schema": "boltbeam.hip_matrix_paths.v1", "device": device, "name": facts.get("name"),
          **{k: v for k, v in facts.items() if k != "name"}, "clock_policy": policy, "telemetry": whole, "paths": rows,
          "path_scan": scanned if hip_path_scan.paths(scanned) else None, "method": METHOD.format(acc=ACC, reps=reps, seconds=seconds)}


def matrix_tflops(result:dict[str, Any]) -> dict[str, float]:
  """The chip profile's matrix_tflops from a probe result: the plateau rate of every available path."""
  return {r["path"]: r["tops"] for r in result["paths"] if r.get("available")}


def fact_source(result:dict[str, Any], *, scope:str, today:str) -> dict[str, Any]:
  """The chip profile's fact source for matrix_tflops, in cuda_compute.fact_source's shape: per path its rate, the
  clock it ran at, its instruction and the rate per multiprocessor per clock; the policy and throttling (unknown on
  amdgpu, which reports no throttle reasons)."""
  return {"kind": "measurement", "bound": "achievable", "scope": scope, "observed_at": today, "method": result["method"],
          "clock_policy": result["clock_policy"].get("policy"), "throttled": result["telemetry"].get("throttled"),
          "paths": {r["path"]: ({"tops": r["tops"], "sm_mhz": r["sm_mhz"], "ops_per_sm_per_clock": r["ops_per_sm_per_clock"],
                                 "instruction": r["instruction"], "words": r["words"], "throttled": r["clock"].get("throttled"),
                                 **{k: r[k] for k in ("uncertainty", "shape_spread") if k in r},
                                 "power_w_mean": (r["clock"].get("power_w") or {}).get("mean")}
                                if r.get("available") else {"available": False, "reason": r["reason"], "instruction": r["instruction"]})
                    for r in result["paths"]}}


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.hip_compute")
  p.add_argument("--device", type=int, default=0)
  p.add_argument("--seconds", type=float, default=SECONDS)
  p.add_argument("--reps", type=int, default=5)
  p.add_argument("--runs", type=int, default=1, help="whole probe runs; each path's uncertainty is their spread at the best shape")
  p.add_argument("--path", action="append", choices=sorted({x.name for f in FAMILIES.values() for x in f}), help="only this path (repeatable)")
  a = p.parse_args(argv)
  runs = [measure_paths(a.device, reps=a.reps, seconds=a.seconds, only=tuple(a.path or ())) for _ in range(a.runs)]
  got = repeated_paths(runs)
  sys.stdout.write(json.dumps(got, indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
