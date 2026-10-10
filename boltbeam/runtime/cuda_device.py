"""NVIDIA CUDA through the driver API (libcuda, ctypes) and nvcc: one device, compiled kernels, event-timed launches.

The same surface as runtime/metal_device.py, so collectors/kernel_timer.py runs its one loop on either:
    library(source, macros)      nvcc -cubin -O3 -arch=native at run time, cached by source hash; cuModuleLoadData
    pipeline(lib, name)          cuModuleGetFunction; a template instantiation is found by its demangled name
    buffer(data | length)        device memory, uploaded or zeroed
    read(buf, n)                 device to host bytes
    dispatch(fn, bound, grid, block, shared)   one launch between two CUDA events; returns µs
    release, close, name, working_set_bytes, facts()

No Python CUDA package is needed: libcuda ships with the driver, nvcc with the toolkit. This is the only module that
calls libcuda or nvcc (cuda_bandwidth.py and cubin_launch.py go through it). The ctypes calls sit in _Driver, one
reviewed boundary; tests pass a fake driver and a fake compiler.

First run on an RTX 5090 on 2026-10-10 (docs/in-model-vs-generic-rtx5090-20261010.md): the bridge, the events and
the cubin load worked as built; the symbol matcher needed the prefix rule above, because that mmvq.cu carries a fifth
template parameter the adapter does not name.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any

CLOCK = "CUDA events (cuEventElapsedTime) around one launch"
NVCC_ENV = "BOLTBEAM_NVCC"
_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
_CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_CU_STREAM_NON_BLOCKING = 1
# cuDeviceGetAttribute codes (cuda.h CUdevice_attribute)
ATTRIBUTES = {"sm_count": 16, "shared_mem_per_sm_bytes": 81, "l2_cache_bytes": 38, "clock_khz": 13,
              "memory_clock_khz": 36, "memory_bus_bits": 37, "warp_size": 10, "max_threads_per_block": 1,
              "compute_capability_major": 75, "compute_capability_minor": 76}


def ggml_compute_capability(facts:dict[str, Any]) -> int | None:
  """The device's compute capability as ggml-cuda numbers it (common.cuh: 100 x major + 10 x minor, so an RTX 5090
  is 1200, a DGX Spark 1210), from the bridge's facts; None when the driver did not report it."""
  major, minor = facts.get("compute_capability_major"), facts.get("compute_capability_minor")
  if major is None or minor is None:
    return None
  return 100 * int(major) + 10 * int(minor)


class CudaUnavailable(RuntimeError):
  """This machine cannot run CUDA kernels: no libcuda, no device, or no nvcc."""


def find_nvcc(env:dict[str, str] | None = None) -> str | None:
  env = os.environ if env is None else env
  if env.get(NVCC_ENV):
    return env[NVCC_ENV]
  return shutil.which("nvcc") or next((p for p in ("/usr/local/cuda/bin/nvcc",) if pathlib.Path(p).is_file()), None)


def default_cache() -> pathlib.Path:
  return pathlib.Path(os.environ.get("BOLTBEAM_CUDA_CACHE") or pathlib.Path(tempfile.gettempdir()) / "boltbeam-cuda")


def build_cubin(nvcc:str, source:str, macros:dict[str, str] | None = None, *, cache:pathlib.Path | None = None,
                arch:str = "native", includes:list[str] | None = None) -> bytes:
  """Compile CUDA source to a cubin for the GPU in this machine (-arch=native), once per source: the file is cached
  by the hash of the source, the macros, the includes and the compiler."""
  cache = cache or default_cache()
  cache.mkdir(parents=True, exist_ok=True)
  macros = macros or {}
  key = hashlib.sha256((source + repr(sorted(macros.items())) + repr(includes or []) + nvcc + arch).encode()).hexdigest()[:16]
  cubin = cache / f"k-{key}.cubin"
  if not cubin.exists():
    src = cache / f"k-{key}.cu"
    src.write_text(source)
    argv = [nvcc, "-cubin", "-O3", f"-arch={arch}", "-std=c++17", *[f"-D{k}={v}" for k, v in macros.items()],
            *[f"-I{i}" for i in includes or []], "-o", str(cubin), str(src)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
      raise RuntimeError(f"nvcc failed: {(proc.stderr or proc.stdout).strip()[-600:]}")
  return cubin.read_bytes()


def elf_symbols(cubin:bytes) -> list[str]:
  """Function symbol names in a cubin (ELF64), for finding a template instantiation's mangled name."""
  if cubin[:4] != b"\x7fELF" or len(cubin) < 64:
    return []
  shoff, shentsize, shnum = struct.unpack_from("<Q", cubin, 0x28)[0], struct.unpack_from("<H", cubin, 0x3a)[0], struct.unpack_from("<H", cubin, 0x3c)[0]
  sections = [struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize) for i in range(shnum)]
  out = []
  for sh in sections:
    if sh[1] != 2:  # SHT_SYMTAB
      continue
    strtab = sections[sh[6]]
    for off in range(sh[4], sh[4] + sh[5], 24):
      st_name, st_info = struct.unpack_from("<IB", cubin, off)
      if st_info & 0xf == 2:  # STT_FUNC
        start = strtab[4] + st_name
        out.append(cubin[start:cubin.index(b"\x00", start)].decode())
  return out


def demangle(names:list[str], filt:str | None = None) -> dict[str, str]:
  """mangled -> demangled through cu++filt (ships with the toolkit); names without it map to themselves."""
  tool = filt or shutil.which("cu++filt") or shutil.which("c++filt")
  if not tool or not names:
    return {n: n for n in names}
  proc = subprocess.run([tool], input="\n".join(names) + "\n", capture_output=True, text=True, timeout=60)
  lines = proc.stdout.splitlines()
  return {n: (lines[i] if i < len(lines) and lines[i] else n) for i, n in enumerate(names)}


def normalise_symbol(name:str) -> str:
  """A demangled name as the matcher compares it: no spaces, no scalar casts, bools as 0 and 1. cu++filt prints a
  template's non-type arguments with casts and bools as (bool)0: `mul_mat_vec_q<(ggml_type)12, (int)1, (bool)0,
  (bool)0, (bool)0>`; the adapter writes `mul_mat_vec_q<(ggml_type)12, 1, false, false>`."""
  s = re.sub(r"\s+", "", name)
  s = re.sub(r"\((?:int|unsigned|long|short|char|bool)\)", "", s)
  return s.replace("true", "1").replace("false", "0")


def template_arity(name:str) -> int | None:
  """How many template arguments a demangled name carries at its top level; None for a plain name."""
  if "<" not in name:
    return None
  depth, count, inner = 0, 0, name[name.index("<") + 1:]
  for ch in inner:
    if ch == "<" or ch == "(":
      depth += 1
    elif ch == ">" or ch == ")":
      if depth == 0:
        break
      depth -= 1
    elif ch == "," and depth == 0:
      count += 1
  return count + 1


def match_symbol(wanted:str, demangled:dict[str, str]) -> str | None:
  """The mangled symbol for `wanted`, a plain name or a template instantiation, compared on the normalised names
  (normalise_symbol), with or without the parameter list. A template instantiation also matches as a prefix:
  `f<a, b>` matches `f<a, b, c>`, so an engine source with one more defaulted template parameter (mmvq.cu's
  `halve_iters`) is still found; the caller records which symbol was matched."""
  w = normalise_symbol(wanted)
  prefix = w[:-1] + "," if w.endswith(">") else None
  exact = next((m for m, plain in demangled.items() if m == wanted or normalise_symbol(plain) == w or w in normalise_symbol(plain)), None)
  if exact or prefix is None:
    return exact
  # several instantiations can share the prefix (mmvq.cu's halve_iters true and false): the smallest normalised
  # name is taken, the same one every run, which for bool and int tails is the all-false, all-zero one
  near = sorted(((normalise_symbol(plain), m) for m, plain in demangled.items() if prefix in normalise_symbol(plain)))
  return near[0][1] if near else None


def kernel_name(plain:str) -> str:
  """A demangled symbol without its return type and parameter list: `void f<(T)1, (bool)0>(void const*)` is
  `f<(T)1, (bool)0>`. The cut is after the template's closing `>`, not at the first `(`, which a cast inside the
  template arguments would hit."""
  name = re.sub(r"^void\s+", "", plain)
  if "<" not in name:
    return name.split("(", 1)[0]
  depth = 0
  for i, ch in enumerate(name):
    if ch == "<":
      depth += 1
    elif ch == ">":
      depth -= 1
      if depth == 0:
        return name[:i + 1]
  return name


def kernel_candidates(wanted:str, demangled:dict[str, str], limit:int = 8) -> list[str]:
  """The kernels in a module that share `wanted`'s name before any `<`, for an error message; compiler helpers
  ($__internal...) are left out. Sorted, at most `limit`."""
  stem = wanted.split("<", 1)[0].strip()
  names = sorted({kernel_name(plain) for plain in demangled.values() if "$__internal" not in plain and stem in plain})
  return names[:limit]


class _Driver:
  """The libcuda calls, one place. Every method raises RuntimeError with the CUresult on failure."""

  def __init__(self, libcuda:str = "libcuda.so.1"):
    try:
      self.cu = ctypes.CDLL(libcuda)
    except OSError as exc:
      raise CudaUnavailable(f"libcuda did not load: {exc}") from exc

  def call(self, name:str, *args) -> None:
    rc = getattr(self.cu, name)(*args)
    if rc != 0:
      raise RuntimeError(f"{name} failed with CUresult {rc}")

  def init(self, device:int) -> tuple[int, int]:
    self.call("cuInit", 0)
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    self.call("cuDeviceGet", ctypes.byref(dev), device)
    self.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)  # cuMemAlloc rejects a non-primary context
    self.call("cuCtxSetCurrent", ctx)
    return dev.value, ctx.value

  def device_name(self, dev:int) -> str:
    buf = ctypes.create_string_buffer(256)
    self.call("cuDeviceGetName", buf, 256, dev)
    return buf.value.decode()

  def total_mem(self, dev:int) -> int:
    n = ctypes.c_size_t()
    self.call("cuDeviceTotalMem_v2", ctypes.byref(n), dev)
    return n.value

  def attribute(self, code:int, dev:int) -> int | None:
    v = ctypes.c_int()
    try:
      self.call("cuDeviceGetAttribute", ctypes.byref(v), code, dev)
    except RuntimeError:
      return None
    return v.value

  def module_load(self, cubin:bytes) -> int:
    m = ctypes.c_void_p()
    self.call("cuModuleLoadData", ctypes.byref(m), ctypes.c_char_p(cubin))
    return m.value

  def module_unload(self, module:int) -> None:
    self.cu.cuModuleUnload(ctypes.c_void_p(module))

  def get_function(self, module:int, name:str) -> int | None:
    fn = ctypes.c_void_p()
    rc = self.cu.cuModuleGetFunction(ctypes.byref(fn), ctypes.c_void_p(module), name.encode())
    return fn.value if rc == 0 else None

  def func_attribute(self, fn:int, code:int) -> int:
    v = ctypes.c_int()
    self.call("cuFuncGetAttribute", ctypes.byref(v), code, ctypes.c_void_p(fn))
    return v.value

  def set_dynamic_shared(self, fn:int, nbytes:int) -> None:
    self.call("cuFuncSetAttribute", ctypes.c_void_p(fn), _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes)

  def mem_alloc(self, nbytes:int) -> int:
    p = ctypes.c_uint64()
    self.call("cuMemAlloc_v2", ctypes.byref(p), ctypes.c_size_t(nbytes))
    return p.value

  def mem_free(self, ptr:int) -> None:
    self.cu.cuMemFree_v2(ctypes.c_uint64(ptr))

  def memset(self, ptr:int, value:int, nbytes:int) -> None:
    self.call("cuMemsetD8_v2", ctypes.c_uint64(ptr), ctypes.c_ubyte(value), ctypes.c_size_t(nbytes))

  def upload(self, ptr:int, data:bytes) -> None:
    self.call("cuMemcpyHtoD_v2", ctypes.c_uint64(ptr), ctypes.c_char_p(data), ctypes.c_size_t(len(data)))

  def download(self, ptr:int, nbytes:int) -> bytes:
    buf = ctypes.create_string_buffer(nbytes)
    self.call("cuMemcpyDtoH_v2", buf, ctypes.c_uint64(ptr), ctypes.c_size_t(nbytes))
    return buf.raw

  def stream(self) -> int:
    s = ctypes.c_void_p()
    self.call("cuStreamCreate", ctypes.byref(s), _CU_STREAM_NON_BLOCKING)
    return s.value

  def stream_destroy(self, stream:int) -> None:
    self.cu.cuStreamDestroy_v2(ctypes.c_void_p(stream))

  def launch_timed(self, fn:int, grid:tuple[int, int, int], block:tuple[int, int, int], shared:int, stream:int,
                   params:list[bytes]) -> float:
    """One launch between two events on the stream, waited for; µs. params are each kernel parameter's own bytes
    (a device pointer as 8 bytes, a scalar or struct as its bytes), in order."""
    holders = [ctypes.create_string_buffer(p, len(p)) for p in params]
    array = (ctypes.c_void_p * max(len(holders), 1))(*[ctypes.cast(h, ctypes.c_void_p) for h in holders])
    begin, end, ms = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_float()
    self.call("cuEventCreate", ctypes.byref(begin), 0)
    self.call("cuEventCreate", ctypes.byref(end), 0)
    try:
      self.call("cuEventRecord", begin, ctypes.c_void_p(stream))
      self.call("cuLaunchKernel", ctypes.c_void_p(fn), *grid, *block, shared, ctypes.c_void_p(stream), array, None)
      self.call("cuEventRecord", end, ctypes.c_void_p(stream))
      self.call("cuEventSynchronize", end)
      self.call("cuEventElapsedTime", ctypes.byref(ms), begin, end)
    finally:
      self.cu.cuEventDestroy_v2(begin)
      self.cu.cuEventDestroy_v2(end)
    return ms.value * 1000.0

  def profiler(self, on:bool) -> None:
    self.call("cuProfilerStart" if on else "cuProfilerStop")

  def release_device(self, dev:int) -> None:
    self.cu.cuDevicePrimaryCtxRelease(dev)


class Cuda:
  """One CUDA device with a stream. Every object it returns is released by `close`. `driver` and `compile` are the
  two outside dependencies (libcuda, nvcc); tests pass fakes."""

  CLOCK = CLOCK

  def __init__(self, device:int = 0, *, nvcc:str | None = None, cache:pathlib.Path | None = None, driver=None,
               compile=None, includes:list[str] | None = None):
    self.driver = driver or _Driver()
    self.nvcc = nvcc or find_nvcc()
    self.cache, self.includes = cache, includes or []
    self._compile = compile or self._nvcc
    self.device_index = device
    try:
      self.dev, self.ctx = self.driver.init(device)
    except RuntimeError as exc:
      raise CudaUnavailable(f"CUDA device {device}: {exc}") from exc
    self.name = self.driver.device_name(self.dev)
    self.working_set_bytes = self.driver.total_mem(self.dev)
    self.stream = self.driver.stream()
    self._modules:dict[int, bytes] = {}
    self._buffers:list[int] = []

  def facts(self) -> dict[str, Any]:
    """The driver's device facts (cuDeviceGetAttribute), as the chip profile records them."""
    return {"name": self.name, "total_global_mem_bytes": self.working_set_bytes,
            **{k: self.driver.attribute(code, self.dev) for k, code in ATTRIBUTES.items()}}

  def _nvcc(self, source:str, macros:dict[str, str]) -> bytes:
    if not self.nvcc:
      raise CudaUnavailable(f"nvcc is not installed (set {NVCC_ENV} or put /usr/local/cuda/bin on PATH)")
    return build_cubin(self.nvcc, source, macros, cache=self.cache, includes=self.includes)

  def library(self, source:str, macros:dict[str, str] | None = None) -> int:
    """Compile the source (nvcc -cubin, cached) and load it; the module handle is the library."""
    return self.module(self._compile(source, macros or {}))

  def module(self, cubin:bytes) -> int:
    """Load a cubin someone else compiled (cubin_launch.py replays a compiler's export)."""
    module = self.driver.module_load(cubin)
    self._modules[module] = cubin
    return module

  def pipeline(self, lib:int, name:str, constants:dict[int, tuple[str, int]] | None = None) -> dict[str, Any]:
    """The kernel by name: an extern "C" name as is, or a template instantiation ("mul_mat_vec_q<(ggml_type)12, 1,
    false, false>") by its demangled symbol (match_symbol). `symbol` records what was matched: the mangled name and
    the demangled one, so a row can say which instantiation it timed. CUDA has no function constants: an adapter
    instantiates in source."""
    if constants:
      raise ValueError("CUDA kernels take no function constants; instantiate the template in the source")
    fn = self.driver.get_function(lib, name)
    symbol = {"mangled": name, "demangled": name}
    if fn is None:
      symbols = demangle(elf_symbols(self._modules.get(lib, b"")))
      mangled = match_symbol(name, symbols)
      fn = self.driver.get_function(lib, mangled) if mangled else None
      if fn is None:
        near = kernel_candidates(name, symbols)
        raise RuntimeError(f"CUDA module has no kernel {name!r}; " + (f"its kernels of that name: {near}" if near else
                                                                      f"it has {sorted(symbols.values())[:8]}"))
      symbol = {"mangled": mangled, "demangled": kernel_name(symbols[mangled])}
    return {"pso": fn, "thread_execution_width": self.driver.attribute(ATTRIBUTES["warp_size"], self.dev) or 32,
            "max_threads_per_threadgroup": self.driver.func_attribute(fn, _CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            "static_threadgroup_memory_bytes": self.driver.func_attribute(fn, _CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            "symbol": symbol}

  def buffer(self, data:bytes | None = None, *, length:int | None = None) -> int:
    n = len(data) if data is not None else int(length or 0)
    ptr = self.driver.mem_alloc(max(n, 1))
    self._buffers.append(ptr)
    if data is not None:
      self.driver.upload(ptr, data)
    else:
      self.driver.memset(ptr, 0, max(n, 1))
    return ptr

  def read(self, buf:int, length:int) -> bytes:
    return self.driver.download(buf, length)

  def dispatch(self, pso:int, bound:list[int | bytes], groups:tuple[int, int, int], threads:tuple[int, int, int],
               threadgroup_bytes:int = 0) -> float:
    """One launch between two events, waited for. A buffer handle becomes the pointer parameter (8 bytes); bytes
    are a by-value parameter's own bytes. groups is the grid in blocks, threads the block."""
    params = [bytes(a) if isinstance(a, (bytes, bytearray)) else struct.pack("<Q", int(a)) for a in bound]
    if threadgroup_bytes:
      self.driver.set_dynamic_shared(pso, threadgroup_bytes)
    return self.driver.launch_timed(pso, tuple(groups), tuple(threads), threadgroup_bytes, self.stream, params)

  def run(self, pso:int, buffers:list[int], constants:list[bytes], groups:int, threads:int) -> float:
    return self.dispatch(pso, [*buffers, *constants], (groups, 1, 1), (threads, 1, 1))

  def profiler(self, on:bool) -> None:
    self.driver.profiler(on)

  def release(self, buf:int) -> None:
    if buf in self._buffers:
      self._buffers.remove(buf)
    self.driver.mem_free(buf)

  def close(self) -> None:
    for buf in self._buffers:
      self.driver.mem_free(buf)
    self._buffers.clear()
    for module in self._modules:
      self.driver.module_unload(module)
    self._modules.clear()
    self.driver.stream_destroy(self.stream)
    self.driver.release_device(self.dev)
