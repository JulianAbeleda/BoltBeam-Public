"""The engine's own kernels as KernelSpecs for the one kernel timer (collectors/kernel_timer.py): per-role time with
no vendor profiler, on Metal (no Xcode) and on CUDA.

An engine ships the shader source its decode runs. An adapter here reads that source from the installed engine
(never a copy in BoltBeam), picks the kernel the engine would run for a role's weight format, states the engine's
own dispatch geometry and argument layout, and hands the timer a KernelSpec on the model's real weight bytes. The
timer compiles, checks the output against the pure-Python reference (metal_native), flushes, warms, times.

ADAPTERS, one row per (engine, backend):
    ("llama.cpp", "Metal")   ggml's Metal backend: the .metal text embedded in libggml-metal (Homebrew) or a
                             ggml-metal.metal file; kernel_mul_mv_<type>_f32; nsg/nr0/FC_MUL_MV from its #defines
    ("llama.cpp", "CUDA")    ggml's CUDA backend: mmvq.cu from the llama.cpp source tree; mul_mat_vec_q<type,1,
                             false,false> with the vector quantized to q8_1 the way quantize_q8_1 does. The kernel
                             is `static` in that file: the one word is dropped from the text in memory so the cubin
                             exports it (recorded). Not yet compiled on a GPU: built against the source, unit-tested.
tinygrad has no adapter: the fork generates its kernels per shape at run time inside a model graph, so there is no
shipped source to compile; it keeps its own timing (collectors/tinygrad_role_time.py). Not cheap, so not built.

A result is labelled with the adapter and the kernel: "isolated, timed by BoltBeam's kernel timer: llama.cpp
kernel_mul_mv_q4_K_f32". An isolated kernel reads from DRAM after a flush, alone; in the model the same kernel runs
between other kernels, which an in-model capture sees and this does not. A weight under CACHE_BYTES stays in the
last-level cache between the warmups and the sample: those rows are "cache", not a DRAM number. The attention, norm
and KV cache kernels are not timed here; the tie-out names their time, with the GPU's idle time, as one difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import struct
from typing import Any, Callable

from boltbeam.collectors import kernel_timer
from boltbeam.collectors.kernel_timer import Check, KernelSpec

METHOD = "boltbeam-kernel-timer-isolated"  # the capture method id every reader keys on (providers.capture_method)
WORDS = "isolated, timed by BoltBeam's kernel timer"  # a row adds ": <engine> <kernel>"
ROLE_SOURCE = "isolated_by_shape"  # the kernel row's role_source: the role is the shape it was run at
CACHE_BYTES = {"Metal": 4 << 20, "CUDA": 96 << 20}  # under this, a weight stays in the last-level cache (M-series SLC; a 5090's L2)
GGML_LIBRARY_ENV = "BOLTBEAM_GGML_METAL"  # a ggml Metal library or ggml-metal.metal file, when not under Homebrew
GGML_CUDA_ENV = "BOLTBEAM_GGML_CUDA_SRC"  # the ggml-cuda source folder (llama.cpp/ggml/src/ggml-cuda)


class NoEngineSource(RuntimeError):
  """The engine's shader source is not on this machine, so its kernels cannot be timed here."""


def _reference(quant:str, rows:int, cols:int, weights:bytes, x:list[float]) -> Check:
  from boltbeam.collectors import metal_native as native
  return Check(indices=native.check_rows(rows), reference=lambda r: native.reference_row(weights, quant, r, cols, x),
               rel_tol=native.TOLERANCE)


# --- llama.cpp on Metal: ggml's Metal backend ------------------------------------------------------------------------

_GGML_CANDIDATES = ("opt/ggml/libexec/libggml-metal.so", "opt/ggml/lib/libggml-metal.dylib", "lib/libggml-metal.dylib",
                    "opt/llama.cpp/libexec/libggml-metal.so", "opt/llama.cpp/lib/libggml-metal.dylib",
                    "share/ggml/ggml-metal.metal", "share/llama.cpp/ggml-metal.metal")
_PREFIXES = ("/opt/homebrew", "/usr/local")


def ggml_library(bench:str | None = None) -> dict[str, Any]:
  """Where ggml's Metal shader source is: $BOLTBEAM_GGML_METAL, else the ggml or llama.cpp keg under the Homebrew
  prefix that holds llama-bench (the library embeds the .metal text; a source build ships the file). The version
  is the keg's: the folder name under Cellar."""
  tried = []
  if env := os.environ.get(GGML_LIBRARY_ENV):
    tried.append(env)
    if pathlib.Path(env).is_file():
      return _file_record(pathlib.Path(env))
  prefixes = list(_PREFIXES)
  if bench:
    real = pathlib.Path(bench).resolve()
    for parent in real.parents:  # .../Cellar/llama.cpp/V/bin/llama-bench: the prefix holds Cellar
      if (parent / "Cellar").is_dir() and str(parent) not in prefixes:
        prefixes.insert(0, str(parent))
  for prefix in prefixes:
    for rel in _GGML_CANDIDATES:
      p = pathlib.Path(prefix) / rel
      tried.append(str(p))
      if p.is_file():
        return _file_record(p)
  raise NoEngineSource("ggml's Metal shader source was not found (libggml-metal or ggml-metal.metal); looked in "
                       + ", ".join(tried[:4]) + f", ... Set {GGML_LIBRARY_ENV} to the file")


def _file_record(path:pathlib.Path) -> dict[str, Any]:
  real = path.resolve()
  version = None
  parts = real.parts
  if "Cellar" in parts:  # /opt/homebrew/Cellar/ggml/0.19.0/libexec/...: the formula and its version
    i = parts.index("Cellar")
    version = f"{parts[i + 1]} {parts[i + 2]}" if len(parts) > i + 2 else None
  return {"path": str(path), "real_path": str(real), "version": version, "bytes": real.stat().st_size,
          "sha256": hashlib.sha256(real.read_bytes()).hexdigest()}


def ggml_source(path:str | pathlib.Path) -> str:
  """The shader text: a .metal file as is; from a library, the embedded C string that holds the mul_mv kernels
  (ggml builds with GGML_METAL_EMBED_LIBRARY: the whole source, common header included, is one string)."""
  raw = pathlib.Path(path).read_bytes()
  if str(path).endswith(".metal"):
    return raw.decode()
  i = raw.find(b"kernel void kernel_mul_mv")
  if i < 0:
    raise NoEngineSource(f"{path} embeds no Metal source with mul_mv kernels")
  start, end = raw.rfind(b"\x00", 0, i) + 1, raw.find(b"\x00", i)
  return raw[start:end].decode()


def ggml_type_name(quant:str) -> str:
  """ggml's type name for a GGUF quant: Q4_K -> q4_K, Q6_K -> q6_K, Q8_0 -> q8_0 (ggml_type_name)."""
  return quant[0].lower() + quant[1:]


def ggml_kernel(text:str, quant:str) -> str | None:
  """The mul_mv kernel ggml runs for a weight type against an f32 vector, from the source's own host_name
  declarations; None when the source has none for this type."""
  wanted = f"kernel_mul_mv_{ggml_type_name(quant)}_f32"
  return wanted if re.search(r'\[\[host_name\("' + re.escape(wanted) + r'"\)\]\]', text) else None


def _define(text:str, name:str) -> int:
  m = re.search(r"#define\s+" + re.escape(name) + r"\s+(\d+)", text)
  if not m:
    raise NoEngineSource(f"the ggml source defines no {name}")
  return int(m.group(1))


# ggml_metal_kargs_mul_mv (ggml-metal-impl.h): the field order is the engine's
_KARGS = "<3i4x4Q3i4x4Q3i2h"  # ne00 ne01 ne02 | nb00..nb03 | ne10 ne11 ne12 | nb10..nb13 | ne0 ne1 nr0 | r2 r3
METAL_RULE = ("ggml-metal-ops.cpp mul_mv for K-quants: grid ((ne01 + nr0*nsg - 1)/(nr0*nsg), ne11, ne12*ne13), "
              "threadgroup (32, nsg, 1), no threadgroup memory; args at index 0, src0 1, src1 2, dst 3; function "
              "constants FC_MUL_MV+0 nsg, +2 ne12, +3 r2, +4 r3")
# the engine passes these to the runtime compiler (ggml-metal-device.m); BF16 is on for every M-series GPU
GGML_MACROS = {"GGML_METAL_EMBED_LIBRARY": "1", "GGML_METAL_HAS_BF16": "1"}


def ggml_metal_spec(text:str, quant:str, rows:int, cols:int, weights:bytes, x:list[float]) -> KernelSpec:
  """ggml's Metal mul_mv kernel for one weight of rows x cols against one vector, dispatched as ggml does: the
  argument struct, the function constants, the grid and the threadgroup. nsg and nr0 are the source's own
  N_SG_/N_R0_ defines."""
  from boltbeam.collectors import metal_native as native
  name = ggml_kernel(text, quant)
  if name is None:
    raise NoEngineSource(f"ggml's Metal source has no {quant} vector kernel")
  nsg, nr0, fc = _define(text, f"N_SG_{quant}"), _define(text, f"N_R0_{quant}"), _define(text, "FC_MUL_MV")
  block = native.BLOCK_BYTES[quant]
  nb01 = (cols // 256) * block
  args = struct.pack(_KARGS, cols, rows, 1, block, nb01, nb01 * rows, nb01 * rows,
                     cols, 1, 1, 4, cols * 4, cols * 4, cols * 4, rows, 1, nr0, 1, 1)
  per_group = nr0 * nsg
  groups = (rows + per_group - 1) // per_group
  return KernelSpec(label=f"llama.cpp {name}", adapter="llama.cpp", source=text, kernel=name, macros=dict(GGML_MACROS),
                    constants={fc: ("short", nsg), fc + 2: ("short", 1), fc + 3: ("short", 1), fc + 4: ("short", 1)},
                    args=[("value", args), weights, struct.pack(f"<{cols}f", *x), rows * 4],
                    grid=(groups, 1, 1), block=(32, nsg, 1), bytes_read=len(weights),
                    check=_reference(quant, rows, cols, weights, x),
                    record={"nsg": nsg, "nr0": nr0, "rows_per_threadgroup": per_group, "function_constant_base": fc,
                            "threadgroups": groups, "threads_per_threadgroup": [32, nsg, 1], "rule": METAL_RULE})


# --- llama.cpp on CUDA: ggml's CUDA backend ---------------------------------------------------------------------------

_CUDA_CANDIDATES = ("~/env/llama.cpp/ggml/src/ggml-cuda", "~/llama.cpp/ggml/src/ggml-cuda", "/usr/local/src/llama.cpp/ggml/src/ggml-cuda")
GGML_TYPE = {"Q4_K": 12, "Q6_K": 14}  # ggml.h enum values, the same ones metal_native.GGML_TYPES reads from GGUF
Q8_1_BLOCK, Q8_1_BYTES = 32, 36  # block_q8_1: half d, half s, 32 int8 (ggml-common.h)
MATRIX_ROW_PADDING = 512  # ggml-cuda common.cuh: the quantized vector is padded to this many columns
CUDA_RULE = ("ggml-cuda mmvq.cu mul_mat_vec_q<type, ncols_dst=1, has_fusion=false, small_k>: grid (ceil(rows / "
             "rows_per_block), 1, 1), block (warp 32, nwarps, 1); nwarps = calc_nwarps(type, 1, GENERIC), small_k when "
             "nwarps > 1 and blocks_per_row < nwarps x vdr x 32 / qi, rows_per_block = nwarps if small_k else 1; the "
             "vector quantized to q8_1 as quantize_q8_1 does, padded to MATRIX_ROW_PADDING columns")
LINKAGE_NOTE = ("mul_mat_vec_q is `static` in mmvq.cu; that one word is removed from the text in memory so the cubin exports "
                "the kernel. No other change to the engine's source.")


def ggml_cuda_source_dir() -> dict[str, Any]:
  """The installed llama.cpp's ggml-cuda source folder: $BOLTBEAM_GGML_CUDA_SRC or a known checkout."""
  tried = []
  for cand in ([os.environ[GGML_CUDA_ENV]] if os.environ.get(GGML_CUDA_ENV) else []) + list(_CUDA_CANDIDATES):
    p = pathlib.Path(cand).expanduser()
    tried.append(str(p))
    if (p / "mmvq.cu").is_file():
      rec = _file_record(p / "mmvq.cu")
      git = p.parents[2] / ".git" if len(p.parents) > 2 else None
      head = None
      if git and (git / "HEAD").is_file():
        ref = (git / "HEAD").read_text().strip()
        head = (git / ref.split(" ", 1)[1]).read_text().strip()[:12] if ref.startswith("ref:") and (git / ref.split(" ", 1)[1]).is_file() else ref[:12]
      return {**rec, "dir": str(p), "commit": head}
  raise NoEngineSource(f"llama.cpp's ggml-cuda source (mmvq.cu) was not found; looked in {', '.join(tried)}. "
                       f"Set {GGML_CUDA_ENV} to llama.cpp/ggml/src/ggml-cuda")


def ggml_cuda_text(src_dir:pathlib.Path) -> str:
  """mmvq.cu as the engine ships it, with the kernel's `static` dropped so the cubin exports it (LINKAGE_NOTE)."""
  text = (pathlib.Path(src_dir) / "mmvq.cu").read_text()
  patched = re.sub(r"static\s+__global__\s+void\s+mul_mat_vec_q\(", "__global__ void mul_mat_vec_q(", text, count=1)
  if patched == text:
    raise NoEngineSource("mmvq.cu has no `static __global__ void mul_mat_vec_q(`: this ggml-cuda is not the version the adapter reads")
  return patched


def cuda_includes(src_dir:pathlib.Path) -> list[str]:
  p = pathlib.Path(src_dir)
  return [str(p), str(p.parent), str(p.parent.parent / "include")]


def ggml_cuda_geometry(text:str, vecdotq:str, common:str, quant:str, rows:int, cols:int) -> dict[str, Any]:
  """nwarps, small_k and rows per block the way mmvq.cu decides them for one vector on an NVIDIA GPU (the GENERIC
  table), from the source's own constants: vdr from vecdotq.cuh, qi = QK_K / (4 x QR) from ggml-common.h."""
  vdr = _define(vecdotq, f"VDR_{quant}_Q8_1_MMVQ")
  qr = _define(common, f"QR{quant[1:]}")
  qi = 256 // (4 * qr)
  nwarps = 4  # calc_nwarps(type, ncols_dst=1, MMVQ_PARAMETERS_GENERIC)
  m = re.search(r"MMVQ_PARAMETERS_GENERIC\)\s*\{\s*switch\s*\(ncols_dst\)\s*\{\s*case 1:(?:\s*case \d+:)*\s*return (\d+);", text)
  if m:
    nwarps = int(m.group(1))
  blocks_per_row = cols // 256
  small_k = nwarps > 1 and blocks_per_row < nwarps * (vdr * 32 // qi)
  rows_per_block = nwarps if small_k else 1
  return {"vdr": vdr, "qi": qi, "nwarps": nwarps, "small_k": small_k, "rows_per_block": rows_per_block,
          "grid": ((rows + rows_per_block - 1) // rows_per_block, 1, 1), "block": (32, nwarps, 1)}


def fastdiv(d:int) -> bytes:
  """ggml-cuda init_fastdiv_values(d): uint3 (mp, L, d)."""
  L = 0
  while L < 32 and (1 << L) < d:
    L += 1
  mp = ((1 << 32) * ((1 << L) - d) // d + 1) & 0xFFFFFFFF
  return struct.pack("<3I", mp, L, d)


def quantize_q8_1(x:list[float]) -> bytes:
  """The vector as ggml-cuda's quantize_q8_1 writes it: per 32 values d = amax/127, q = round(x/d), s = sum(x);
  padded with zero blocks to MATRIX_ROW_PADDING columns."""
  n = (len(x) + MATRIX_ROW_PADDING - 1) // MATRIX_ROW_PADDING * MATRIX_ROW_PADDING
  xs = list(x) + [0.0] * (n - len(x))
  out = bytearray()
  for b in range(0, n, Q8_1_BLOCK):
    blk = xs[b:b + Q8_1_BLOCK]
    amax = max(abs(v) for v in blk)
    d = amax / 127.0
    qs = [0 if amax == 0.0 else int(math.copysign(math.floor(abs(v / d) + 0.5), v)) for v in blk]
    out += struct.pack("<2e", d, sum(blk)) + struct.pack("<32b", *qs)
  return bytes(out)


def ggml_cuda_spec(src_dir:pathlib.Path, quant:str, rows:int, cols:int, weights:bytes, x:list[float]) -> KernelSpec:
  """ggml's CUDA mul_mat_vec_q for one weight of rows x cols against one vector, every parameter as
  ggml_cuda_mul_mat_vec_q passes it (ids null, no fusion, one channel, one sample)."""
  from boltbeam.collectors import metal_native as native
  src_dir = pathlib.Path(src_dir)
  text = ggml_cuda_text(src_dir)
  vecdotq, common = (src_dir / "vecdotq.cuh").read_text(), (src_dir.parent / "ggml-common.h").read_text()
  geo = ggml_cuda_geometry(text, vecdotq, common, quant, rows, cols)
  gt = GGML_TYPE[quant]
  inst = f"mul_mat_vec_q<(ggml_type){gt}, 1, false, {'true' if geo['small_k'] else 'false'}>"
  keep = (f"mul_mat_vec_q<GGML_TYPE_{quant}, 1, false, {'true' if geo['small_k'] else 'false'}>")
  source = text + f"\n// BoltBeam: instantiate the kernel the engine runs for {quant}, one vector\n" \
                  f"static const void* const bb_keep_{quant.lower()} = (const void*) {keep};\n"
  y = quantize_q8_1(x)
  blocks_per_row = cols // 256
  padded_blocks = len(y) // Q8_1_BYTES
  u32 = lambda v: ("value", struct.pack("<I", v))  # noqa: E731
  args = [weights, y, ("value", bytes(8)), ("value", bytes(48)), rows * 4,  # vx, vy (q8_1), ids = null, fusion = {}, dst
          u32(cols), ("value", bytes(12)),  # ncols_x, nchannels_y (zeros: ggml passes make_uint3(0,0,0) without ids)
          u32(blocks_per_row), u32(padded_blocks), u32(rows),  # stride_row_x, stride_col_y, stride_col_dst
          ("value", fastdiv(1)), u32(blocks_per_row * rows), u32(padded_blocks), u32(rows),  # channel_ratio, stride_channel_x/y/dst
          ("value", fastdiv(1)), u32(blocks_per_row * rows), u32(padded_blocks), u32(rows),  # sample_ratio, stride_sample_x/y/dst
          u32(0)]  # ids_stride
  return KernelSpec(label=f"llama.cpp {inst}", adapter="llama.cpp", source=source, kernel=inst,
                    args=args, grid=geo["grid"], block=geo["block"], bytes_read=len(weights),
                    check=_reference(quant, rows, cols, weights, x),
                    record={**{k: geo[k] for k in ("vdr", "qi", "nwarps", "small_k", "rows_per_block")},
                            "threadgroups": geo["grid"][0], "threads_per_threadgroup": list(geo["block"]), "rule": CUDA_RULE,
                            "linkage": LINKAGE_NOTE, "includes": cuda_includes(src_dir), "bytes_q8_1_vector": len(y)})


# --- the adapters, one row per (engine, backend) --------------------------------------------------------------------------

def _metal_setup(bench:str | None) -> dict[str, Any]:
  lib = ggml_library(bench)
  return {"source": lib, "text": ggml_source(lib["path"]), "engine": "llama.cpp (ggml Metal backend)",
          "spec": lambda s, quant, rows, cols, w, x: ggml_metal_spec(s["text"], quant, rows, cols, w, x),
          "kernel": lambda s, quant: ggml_kernel(s["text"], quant), "bridge": {}}


def _cuda_setup(bench:str | None) -> dict[str, Any]:
  src = ggml_cuda_source_dir()
  return {"source": src, "engine": "llama.cpp (ggml CUDA backend)",
          "spec": lambda s, quant, rows, cols, w, x: ggml_cuda_spec(pathlib.Path(s["source"]["dir"]), quant, rows, cols, w, x),
          "kernel": lambda s, quant: f"mul_mat_vec_q<(ggml_type){GGML_TYPE[quant]}, 1, ...>" if quant in GGML_TYPE else None,
          "bridge": {"includes": cuda_includes(pathlib.Path(src["dir"]))}}


ADAPTERS:dict[tuple[str, str], Callable[[str | None], dict[str, Any]]] = {
  ("llama.cpp", "Metal"): _metal_setup,
  ("llama.cpp", "CUDA"): _cuda_setup,
}


def available(provider:str, backend:str, bench:str | None = None) -> str | None:
  """Why the engine's kernels cannot be timed here, or None when they can: an adapter row and its source on disk."""
  setup = ADAPTERS.get((provider, backend))
  if setup is None:
    return f"BoltBeam has no kernel adapter for {provider} on {backend}: its shipped source is not known there"
  try:
    setup(bench)
  except NoEngineSource as exc:
    return str(exc)
  return None


def role_rows(profile:dict[str, Any]) -> list[dict[str, Any]]:
  """The weight roles to time, from the model profile: one per (role, quant, shape) with its calls per token."""
  return [{"role": r["role"], "quant": r["quant"], "rows": int(r["rows"]), "cols": int(r["cols"]), "count": int(r.get("count") or 0),
           "tensor_name": r.get("tensor_name")} for r in profile.get("roles", []) if r.get("count") and r.get("rows") and r.get("cols")]


def collect(run:pathlib.Path, provider:str, *, say:Callable[[str], None] = lambda _: None,
            step:Callable[[int, int], None] | None = None, out:str | None = None, bridge=None) -> dict[str, Any]:
  """Time every weight role's kernel of this engine alone, on the model's real bytes, and write the per-role trace
  (boltbeam.timing_trace.v1) to run/out. The token the kernels tie out against is step 4's measured whole step."""
  from boltbeam.collectors import llama_bench_decode, metal_native as native, providers
  from boltbeam.profile.gguf import read_gguf_layout
  from boltbeam.workflow import tie_out
  from boltbeam.workflow.common import load_manifest, read_json
  from boltbeam.workflow.screen import run_bandwidth
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  backend = target.backend
  setup_fn = ADAPTERS.get((provider, backend))
  if setup_fn is None:
    raise NoEngineSource(available(provider, backend))
  bench = llama_bench_decode.find(llama_bench_decode.DEFAULT) if provider == "llama.cpp" else None
  setup = setup_fn(bench)
  model = pathlib.Path(str(manifest.get("model_path")))
  profile = read_json(run / "model_profile.json")
  roles = role_rows(profile)
  peak_gbs, peak_source = run_bandwidth(run, target)
  step4 = read_json(run / "timing_trace.json") if (run / "timing_trace.json").exists() else {}
  token = tie_out.measured_step(step4, provider)
  own_bridge = bridge is None
  try:
    bridge = bridge or kernel_timer.bridge_for(backend, **setup["bridge"])
  except RuntimeError as exc:
    raise NoEngineSource(str(exc)) from exc
  _, tensors, data_start = read_gguf_layout(model)
  rows_out = []
  cache_bytes = CACHE_BYTES.get(backend, 0)
  try:
    flusher = kernel_timer.Flusher(bridge, backend)
    floor_us = flusher.floor_us()
    say(f"compiling {setup['engine']} shaders: {setup['source']['path']}")
    libraries:dict[str, int] = {}
    for i, r in enumerate(roles, start=1):
      rows, cols, quant = r["rows"], r["cols"], r["quant"]
      say(f"role {i} of {len(roles)}: {r['role']} {quant} {rows}x{cols}")
      row:dict[str, Any] = {"scope": "kernel", "role": r["role"], "quant": quant, "shape": [rows, cols], "calls": r["count"],
                            "role_source": ROLE_SOURCE, "tensor": r.get("tensor_name"), "time_source": METHOD}
      if quant not in native.BLOCK_BYTES or cols % 256 or setup["kernel"](setup, quant) is None:
        row.update(status="not_measured", wall_us=0.0,
                   reason=f"{setup['engine']} has no {quant} vector kernel in its source" if quant in native.BLOCK_BYTES and cols % 256 == 0
                   else f"BoltBeam has no block layout for {quant} with {cols} columns")
        rows_out.append(row)
        continue
      size = rows * (cols // 256) * native.BLOCK_BYTES[quant]
      if size + kernel_timer.FLUSH[backend]["bytes"] > bridge.working_set_bytes * native.WORKING_SET_SHARE:
        row.update(status="not_measured", reason=f"{size / 2**30:.1f} GiB of weights does not fit the GPU's working set", wall_us=0.0)
        rows_out.append(row)
        continue
      tensor, offset = native.tensor_for(quant, rows, cols, tensors, r.get("tensor_name"))
      with open(model, "rb") as f:
        f.seek(data_start + offset)
        weights = f.read(size)
      from boltbeam.collectors import boltbeam_gemv
      x = boltbeam_gemv.vector(f"{r['role']}:{quant}", cols)
      spec = setup["spec"](setup, quant, rows, cols, weights, x)
      got = kernel_timer.time_spec(bridge, spec, flusher, libraries=libraries)
      row.update(kernel=spec.kernel, tensor=tensor, bytes=size, correctness=got["correctness"], timed_by=f"{WORDS}: {spec.label}",
                 geometry={**spec.record, "thread_execution_width": got["pipeline"]["thread_execution_width"],
                           "max_threads_per_threadgroup": got["pipeline"]["max_threads_per_threadgroup"]})
      if not got["samples"]:
        row.update(status="correctness_failed", wall_us=0.0)
        rows_out.append(row)
        continue
      med = got["median_us"]
      row.update(status="measured", us_per_call=med, min_us=got["min_us"], samples=len(got["samples"]), spread_pct=got["spread_pct"],
                 wall_us=med * r["count"], gbs=size / (med * 1e3), cache=size < cache_bytes,
                 cache_note=(f"{size / 2**20:.1f} MB stays in the last-level cache between warmup and sample: not a DRAM "
                             "read" if size < cache_bytes else None),
                 timing={**got["timing"], "dispatch_floor_us": floor_us})
      rows_out.append(row)
      if step:
        step(i, len(roles))
  finally:
    if own_bridge:
      bridge.close()
  measured = [r for r in rows_out if r.get("status") == "measured"]
  kernel_us = sum(r["wall_us"] for r in measured)
  whole = {"scope": "whole_step", "decode_tokens": 1, "wall_us": kernel_us, "measurement_scope": "summed_isolated_kernels",
           "time_source": METHOD, "context": token["context"] if token else None,
           "tok_s": token["tok_s"] if token else None, "token_source": "step 4, the untraced whole step" if token else None}
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest.get("model_id"), "target_id": manifest.get("target_id"),
           "workload": "decode", "provider_id": provider, "capture": {"method": METHOD, "reason": None},
           "timing_source": f"{setup['engine']} kernels timed alone by BoltBeam's kernel timer ({bridge.CLOCK})",
           "engine": {"name": setup["engine"], "source": setup["source"], "macros": GGML_MACROS if backend == "Metal" else {}},
           "peak_gbs": peak_gbs, "peak_source": peak_source, "rows": [whole, *rows_out],
           "measured": ["kernel.us_per_call (median of the timed flushed launches)", "kernel.correctness against the pure-Python reference",
                        "whole_step.wall_us (the sum of the measured kernels per token)"],
           "absent": ["attention, norm, RoPE and KV cache kernels: not timed alone; their time is the tie-out's difference line",
                      "in-model time: the kernel ran alone, not between the model's other kernels"],
           "notes": [WORDS, f"cache rows: a weight under {cache_bytes >> 20} MB is a cache read, not a DRAM read"]}
  path = run / (out or providers.TRACES[provider])
  path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  return trace
