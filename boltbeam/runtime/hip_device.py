"""AMD GPUs through HIP's module API (libamdhip64, ctypes) and hipcc: one device, compiled kernels, event-timed launches.

The same surface as runtime/cuda_device.py and runtime/metal_device.py, so collectors/kernel_timer.py and the chip
probes run one loop on any of them:
    library(source, macros)      hipcc --genco -O3 --offload-arch=<this device> at run time, cached by source hash
    pipeline(lib, name)          hipModuleGetFunction; a template instantiation is found by its demangled name
    buffer(data | length)        device memory, uploaded or zeroed
    read(buf, n)                 device to host bytes
    dispatch(fn, bound, grid, block, shared)   one launch between two HIP events; returns µs
    release, close, name, working_set_bytes, facts()

No Python ROCm package is needed: libamdhip64 and hipcc ship with ROCm, found through target/toolkits.py. The
numeric codes HIP's attribute queries take are not copied here: they are read from the installed hip_runtime_api.h
by enum name, so a ROCm release that renumbers them is followed, and a name it drops is reported as unknown. The
ctypes calls sit in _Driver, one reviewed boundary; tests pass a fake driver and a fake compiler.
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
from functools import lru_cache
from typing import Any

from boltbeam.runtime.cuda_device import demangle, elf_symbols, kernel_candidates, kernel_name, match_symbol
from boltbeam.target import toolkits

CLOCK = "HIP events (hipEventElapsedTime) around one launch"
HIPCC_ENV = "BOLTBEAM_HIPCC"
# device code only, as a bare ELF: a clang offload bundle (hipcc's default) loads too, but its symbols cannot be read
GENCO = ("--genco", "--no-gpu-bundle-output")
GGML_CC_OFFSET_AMD = 0x1000000  # ggml-cuda common.cuh GGML_CUDA_CC_OFFSET_AMD
# The installed headers that define the enums read below: device attributes, then function attributes.
HEADERS = ("include/hip/hip_runtime_api.h", "include/hip/driver_types.h")
_HIP_STREAM_NON_BLOCKING = 1
# facts() key -> hipDeviceAttribute_t enumerator, resolved against the installed header
# multiprocessor_count is what HIP counts, which on RDNA is work-group processors (two compute units each: a
# Radeon AI PRO R9700 reports 32 here and 64 compute units through amd-smi); it is recorded under HIP's own name.
ATTRIBUTES = {"multiprocessor_count": "hipDeviceAttributeMultiprocessorCount",
              "shared_mem_per_block_bytes": "hipDeviceAttributeMaxSharedMemoryPerBlock",
              "l2_cache_bytes": "hipDeviceAttributeL2CacheSize", "clock_khz": "hipDeviceAttributeClockRate",
              "memory_clock_khz": "hipDeviceAttributeMemoryClockRate", "memory_bus_bits": "hipDeviceAttributeMemoryBusWidth",
              "warp_size": "hipDeviceAttributeWarpSize", "max_threads_per_block": "hipDeviceAttributeMaxThreadsPerBlock",
              "compute_capability_major": "hipDeviceAttributeComputeCapabilityMajor",
              "compute_capability_minor": "hipDeviceAttributeComputeCapabilityMinor"}
_FUNC_MAX_THREADS = "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"
_FUNC_SHARED = "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"


def ggml_compute_capability(facts:dict[str, Any]) -> int | None:
  """The device's compute capability as ggml-cuda numbers an AMD GPU (ggml_cuda_parse_id on gcnArchName): the
  offset, then the arch after "gfx" with its last two hex digits as the minor, so gfx1201 is 0x1001201 and
  gfx1100 0x1001100; a "-generic" target takes major-minor. None when the driver reported no architecture."""
  arch = str(facts.get("architecture") or "").split(":")[0]
  name = arch[3:] if arch.startswith("gfx") else ""
  try:
    if name.endswith("-generic"):
      major, _, minor = name[:-len("-generic")].partition("-")
      return GGML_CC_OFFSET_AMD + int(major, 16) * 0x100 + (0x10 * int(minor, 16) if minor else 0)
    return GGML_CC_OFFSET_AMD + int(name[:-2], 16) * 0x100 + int(name[-2:], 16) if len(name) >= 3 else None
  except ValueError:
    return None


class HipUnavailable(RuntimeError):
  """This machine cannot run HIP kernels: no libamdhip64, no device, no hipcc, or no HIP header."""


def enum_values(header_text:str, enum:str) -> dict[str, int]:
  """The enumerators of one C enum in a header, with C's rules: an explicit `= N` (decimal or hex), an alias
  `= OTHER` to an earlier enumerator, and otherwise one more than the previous. Comments and preprocessor lines are
  dropped first. An enum not found gives {}."""
  text = re.sub(r"/\*.*?\*/", "", header_text, flags=re.S)
  text = re.sub(r"//[^\n]*", "", text)
  text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
  m = re.search(r"\benum\s+" + re.escape(enum) + r"\s*\{(.*?)\}", text, re.S) or \
      re.search(r"\benum\s*\{(.*?)\}\s*" + re.escape(enum) + r"\s*;", text, re.S)
  if not m:
    return {}
  values:dict[str, int] = {}
  nxt = 0
  for item in (part.strip() for part in m.group(1).split(",")):
    if not item:
      continue
    name, _, expr = (piece.strip() for piece in item.partition("="))
    if expr:
      expr = expr.strip("() ")
      nxt = values[expr] if expr in values else int(expr, 0)
    values[name] = nxt
    nxt += 1
  return values


@lru_cache(maxsize=4)
def header_codes(paths:tuple[str, ...]) -> dict[str, int]:
  """The device-attribute and function-attribute codes from the installed HIP headers, by enumerator name."""
  text = "\n".join(pathlib.Path(p).read_text(errors="ignore") for p in paths)
  return {**enum_values(text, "hipDeviceAttribute_t"), **enum_values(text, "hipFunction_attribute")}


def find_hipcc(env:dict[str, str] | None = None) -> str | None:
  env = os.environ if env is None else env
  return env.get(HIPCC_ENV) or toolkits.tool("hipcc", shutil.which, env)


def default_cache() -> pathlib.Path:
  return pathlib.Path(os.environ.get("BOLTBEAM_HIP_CACHE") or pathlib.Path(tempfile.gettempdir()) / "boltbeam-hip")


def build_code_object(hipcc:str, source:str, macros:dict[str, str] | None = None, *, arch:str,
                      cache:pathlib.Path | None = None, includes:list[str] | None = None) -> bytes:
  """Compile HIP source to a code object for one gfx target (the device's own, from the driver), once per source:
  the file is cached by the hash of the source, the macros, the includes, the compiler and the target."""
  cache = cache or default_cache()
  cache.mkdir(parents=True, exist_ok=True)
  macros = macros or {}
  key = hashlib.sha256((source + repr(sorted(macros.items())) + repr(includes or []) + hipcc + arch + repr(GENCO)).encode()).hexdigest()[:16]
  out = cache / f"k-{key}.hsaco"
  if not out.exists():
    src = cache / f"k-{key}.hip"
    # nvcc includes its runtime header itself; hipcc compiling device code only does not
    src.write_text("#include <hip/hip_runtime.h>\n" + source)
    argv = [hipcc, *GENCO, "-O3", "-fno-crash-diagnostics", f"--offload-arch={arch}", "-std=c++17", *[f"-D{k}={v}" for k, v in macros.items()],
            *[f"-I{i}" for i in includes or []], "-o", str(out), str(src)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
      raise RuntimeError(f"hipcc failed: {(proc.stderr or proc.stdout).strip()[-600:]}")
  return out.read_bytes()


class _Driver:
  """The libamdhip64 calls, one place. Every method raises RuntimeError with the hipError_t on failure."""

  def __init__(self, library:str | None = None):
    lib = library or str(toolkits.file("rocm", "lib/libamdhip64.so") or "libamdhip64.so")
    try:
      self.hip = ctypes.CDLL(lib)
    except OSError as exc:
      raise HipUnavailable(f"libamdhip64 did not load: {exc}") from exc

  def call(self, name:str, *args) -> None:
    rc = getattr(self.hip, name)(*args)
    if rc != 0:
      raise RuntimeError(f"{name} failed with hipError_t {rc}")

  def init(self, device:int) -> int:
    self.call("hipInit", 0)
    self.call("hipSetDevice", device)
    return device

  def device_name(self, dev:int) -> str:
    buf = ctypes.create_string_buffer(256)
    self.call("hipDeviceGetName", buf, 256, dev)
    return buf.value.decode()

  def arch(self, dev:int) -> str:
    """The device's gfx target as the compiler names it (gcnArchName up to its first ':' feature flag)."""
    props = ctypes.create_string_buffer(8192)  # larger than any release's hipDeviceProp_t
    self.call("hipGetDevicePropertiesR0600", props, dev)
    found = re.search(rb"gfx[0-9a-f]+", props.raw)
    if not found:
      raise RuntimeError("HIP did not report the device's gfx target")
    return found.group().decode()

  def device_count(self) -> int:
    n = ctypes.c_int()
    self.call("hipGetDeviceCount", ctypes.byref(n))
    return n.value

  def runtime_version(self) -> int | None:
    v = ctypes.c_int()
    try:
      self.call("hipRuntimeGetVersion", ctypes.byref(v))
    except RuntimeError:
      return None
    return v.value

  def pci_bus_id(self, dev:int) -> str:
    buf = ctypes.create_string_buffer(64)
    self.call("hipDeviceGetPCIBusId", buf, 64, dev)
    return buf.value.decode().lower()

  def total_mem(self, dev:int) -> int:
    n = ctypes.c_size_t()
    self.call("hipDeviceTotalMem", ctypes.byref(n), dev)
    return n.value

  def attribute(self, code:int, dev:int) -> int | None:
    v = ctypes.c_int()
    try:
      self.call("hipDeviceGetAttribute", ctypes.byref(v), code, dev)
    except RuntimeError:
      return None
    return v.value

  def module_load(self, code_object:bytes) -> int:
    m = ctypes.c_void_p()
    self.call("hipModuleLoadData", ctypes.byref(m), ctypes.c_char_p(code_object))
    return m.value

  def module_unload(self, module:int) -> None:
    self.hip.hipModuleUnload(ctypes.c_void_p(module))

  def get_function(self, module:int, name:str) -> int | None:
    fn = ctypes.c_void_p()
    rc = self.hip.hipModuleGetFunction(ctypes.byref(fn), ctypes.c_void_p(module), name.encode())
    return fn.value if rc == 0 else None

  def func_attribute(self, fn:int, code:int) -> int:
    v = ctypes.c_int()
    self.call("hipFuncGetAttribute", ctypes.byref(v), code, ctypes.c_void_p(fn))
    return v.value

  def mem_alloc(self, nbytes:int) -> int:
    p = ctypes.c_void_p()
    self.call("hipMalloc", ctypes.byref(p), ctypes.c_size_t(nbytes))
    return p.value

  def mem_free(self, ptr:int) -> None:
    self.hip.hipFree(ctypes.c_void_p(ptr))

  def memset(self, ptr:int, value:int, nbytes:int) -> None:
    self.call("hipMemset", ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(nbytes))

  def upload(self, ptr:int, data:bytes) -> None:
    self.call("hipMemcpyHtoD", ctypes.c_void_p(ptr), ctypes.c_char_p(data), ctypes.c_size_t(len(data)))

  def download(self, ptr:int, nbytes:int) -> bytes:
    buf = ctypes.create_string_buffer(nbytes)
    self.call("hipMemcpyDtoH", buf, ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes))
    return buf.raw

  def stream(self) -> int:
    s = ctypes.c_void_p()
    self.call("hipStreamCreateWithFlags", ctypes.byref(s), _HIP_STREAM_NON_BLOCKING)
    return s.value

  def stream_destroy(self, stream:int) -> None:
    self.hip.hipStreamDestroy(ctypes.c_void_p(stream))

  def launch_timed(self, fn:int, grid:tuple[int, int, int], block:tuple[int, int, int], shared:int, stream:int,
                   params:list[bytes]) -> float:
    """One launch between two events on the stream, waited for; µs. params are each kernel parameter's own bytes
    (a device pointer as 8 bytes, a scalar or struct as its bytes), in order."""
    holders = [ctypes.create_string_buffer(p, len(p)) for p in params]
    array = (ctypes.c_void_p * max(len(holders), 1))(*[ctypes.cast(h, ctypes.c_void_p) for h in holders])
    begin, end, ms = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_float()
    self.call("hipEventCreate", ctypes.byref(begin))
    self.call("hipEventCreate", ctypes.byref(end))
    try:
      self.call("hipEventRecord", begin, ctypes.c_void_p(stream))
      self.call("hipModuleLaunchKernel", ctypes.c_void_p(fn), *[ctypes.c_uint(v) for v in (*grid, *block, shared)],
                ctypes.c_void_p(stream), array, None)
      self.call("hipEventRecord", end, ctypes.c_void_p(stream))
      self.call("hipEventSynchronize", end)
      self.call("hipEventElapsedTime", ctypes.byref(ms), begin, end)
    finally:
      self.hip.hipEventDestroy(begin)
      self.hip.hipEventDestroy(end)
    return ms.value * 1000.0

  def profiler(self, on:bool) -> None:
    self.call("hipProfilerStart" if on else "hipProfilerStop")

  def release_device(self, dev:int) -> None:
    self.hip.hipDeviceSynchronize()


def index_of(pci_bus_id:str, driver=None) -> int:
  """The HIP device index of the GPU at a PCI address. HIP's order and the driver tools' (amd-smi, the kernel's
  card numbers) differ, so a GPU found by a scan is opened by its address, not by its position."""
  driver = driver or _Driver()
  driver.call("hipInit", 0)
  want = pci_bus_id.lower()
  for i in range(driver.device_count()):
    if driver.pci_bus_id(i) == want:
      return i
  raise HipUnavailable(f"no HIP device at PCI {pci_bus_id}")


class Hip:
  """One HIP device with a stream. Every object it returns is released by `close`. `driver`, `compile` and
  `codes` (the header's enum values) are the outside dependencies; tests pass fakes."""

  API = "HIP"
  CLOCK = CLOCK
  GGML_MACROS = {"GGML_USE_HIP": "1"}  # llama.cpp's HIP build compiles ggml-cuda/*.cu with it (-x hip)

  def __init__(self, device:int = 0, *, hipcc:str | None = None, cache:pathlib.Path | None = None, driver=None,
               compile=None, includes:list[str] | None = None, codes:dict[str, int] | None = None):
    self.driver = driver or _Driver()
    self.hipcc = hipcc or find_hipcc()
    self.cache, self.includes = cache, includes or []
    self._compile = compile or self._hipcc
    if codes is None:
      headers = [toolkits.file("rocm", h) for h in HEADERS]
      if None in headers:
        raise HipUnavailable(f"the HIP headers ({', '.join(HEADERS)}) are not under any ROCm root: set ROCM_PATH")
      codes = header_codes(tuple(str(h) for h in headers))
    self.codes = codes
    self.device_index = device
    try:
      self.dev = self.driver.init(device)
      self.arch = self.driver.arch(self.dev)
    except RuntimeError as exc:
      raise HipUnavailable(f"HIP device {device}: {exc}") from exc
    self.name = self.driver.device_name(self.dev)
    self.working_set_bytes = self.driver.total_mem(self.dev)
    self.stream = self.driver.stream()
    self._modules:dict[int, bytes] = {}
    self._buffers:list[int] = []

  def _attribute(self, enumerator:str) -> int | None:
    code = self.codes.get(enumerator)
    return None if code is None else self.driver.attribute(code, self.dev)

  def ggml_cc(self) -> int | None:
    """This device's compute capability as ggml-cuda numbers it (ggml_compute_capability)."""
    return ggml_compute_capability(self.facts())

  def facts(self) -> dict[str, Any]:
    """The driver's device facts (hipDeviceGetAttribute), as the chip profile records them; a fact this ROCm's
    header does not name is None, not a guess."""
    return {"name": self.name, "architecture": self.arch, "pci_bus_id": self.driver.pci_bus_id(self.dev),
            "hip_runtime_version": self.driver.runtime_version(),
            "total_global_mem_bytes": self.working_set_bytes,
            **{k: self._attribute(enumerator) for k, enumerator in ATTRIBUTES.items()}}

  def _hipcc(self, source:str, macros:dict[str, str]) -> bytes:
    if not self.hipcc:
      raise HipUnavailable(f"hipcc is not installed (set {HIPCC_ENV} or ROCM_PATH)")
    return build_code_object(self.hipcc, source, macros, arch=self.arch, cache=self.cache, includes=self.includes)

  def library(self, source:str, macros:dict[str, str] | None = None) -> int:
    """Compile the source (hipcc --genco for this device's gfx target, cached) and load it."""
    return self.module(self._compile(source, macros or {}))

  def module(self, code_object:bytes) -> int:
    module = self.driver.module_load(code_object)
    self._modules[module] = code_object
    return module

  def pipeline(self, lib:int, name:str, constants:dict[int, tuple[str, int]] | None = None) -> dict[str, Any]:
    """The kernel by name: an extern "C" name as is, or a template instantiation by its demangled symbol, matched as
    cuda_device.match_symbol does (a code object is an ELF like a cubin). HIP has no function constants."""
    if constants:
      raise ValueError("HIP kernels take no function constants; instantiate the template in the source")
    fn = self.driver.get_function(lib, name)
    symbol = {"mangled": name, "demangled": name}
    if fn is None:
      symbols = demangle(elf_symbols(self._modules.get(lib, b"")), shutil.which("llvm-cxxfilt") or shutil.which("c++filt"))
      mangled = match_symbol(name, symbols)
      fn = self.driver.get_function(lib, mangled) if mangled else None
      if fn is None:
        near = kernel_candidates(name, symbols)
        raise RuntimeError(f"HIP module has no kernel {name!r}; " + (f"its kernels of that name: {near}" if near else
                                                                     f"it has {sorted(symbols.values())[:8]}"))
      symbol = {"mangled": mangled, "demangled": kernel_name(symbols[mangled])}
    width = self._attribute(ATTRIBUTES["warp_size"])
    if not width: raise RuntimeError("HIP device did not report its wavefront size")
    max_threads, shared = (self.codes.get(n) for n in (_FUNC_MAX_THREADS, _FUNC_SHARED))
    return {"pso": fn, "thread_execution_width": width,
            "max_threads_per_threadgroup": self.driver.func_attribute(fn, max_threads) if max_threads is not None else None,
            "static_threadgroup_memory_bytes": self.driver.func_attribute(fn, shared) if shared is not None else None,
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
    are a by-value parameter's own bytes. groups is the grid in blocks, threads the block. Dynamic LDS is passed
    with the launch; HIP needs no per-function opt-in for it."""
    params = [bytes(a) if isinstance(a, (bytes, bytearray)) else struct.pack("<Q", int(a)) for a in bound]
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
