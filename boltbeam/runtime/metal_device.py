"""Apple Metal through ctypes: one device, one queue, compiled kernels, GPU-timed dispatches.

This is the only BoltBeam module that calls the Metal framework. It needs no Xcode: the runtime shader compiler
ships with macOS (`newLibraryWithSource`). Nothing here knows about models or probes; the collector in
`boltbeam/collectors/metal_native.py` decides what to run and what the numbers mean.

Timing is the command buffer's own GPU clock (`GPUStartTime`/`GPUEndTime`). One command buffer holds one
dispatch, so the interval is that dispatch plus the fixed cost of starting a command buffer on the GPU.
`dispatch_floor_us` measures that fixed cost with an empty kernel; callers report it next to every number.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
from typing import Any

_METAL = "/System/Library/Frameworks/Metal.framework/Metal"
_SHARED = 0  # MTLResourceStorageModeShared: CPU and GPU see the same bytes on unified memory
_COMPLETED = 4  # MTLCommandBufferStatusCompleted
# MTLDataType codes (Metal's MTLArgument.h) and the struct packing of each, for function constants
MTL_DATA_TYPES = {"int": (29, "<i"), "short": (37, "<h"), "bool": (53, "<?")}


class MetalUnavailable(RuntimeError):
  """This machine cannot run Metal kernels: not macOS, no framework, or no GPU."""


class _Size(ctypes.Structure):
  _fields_ = [("width", ctypes.c_ulong), ("height", ctypes.c_ulong), ("depth", ctypes.c_ulong)]


class Metal:
  """The default Metal device with a command queue. Every object it returns is released by `close`."""

  CLOCK = "command buffer GPUStartTime/GPUEndTime"

  def __init__(self) -> None:
    if sys.platform != "darwin":
      raise MetalUnavailable(f"Metal needs macOS; this machine is {sys.platform}")
    try:  # the framework lives in the dyld shared cache, not on disk, so loading is the only test
      self._objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
      metal = ctypes.cdll.LoadLibrary(_METAL)
    except OSError as exc:
      raise MetalUnavailable(f"Metal.framework did not load: {exc}") from exc
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")  # links the default device
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    self._objc.sel_registerName.restype = ctypes.c_void_p
    self._objc.objc_getClass.restype = ctypes.c_void_p
    self._objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    self._objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    self._fns: dict[tuple[Any, ...], Any] = {}
    self._owned: list[int] = []
    self.device = metal.MTLCreateSystemDefaultDevice()
    if not self.device:
      raise MetalUnavailable("Metal has no default device")
    self.queue = self._own(self.send(self.device, "newCommandQueue"))
    self.name = self.string(self.send(self.device, "name"))
    self.working_set_bytes = int(self.send(self.device, "recommendedMaxWorkingSetSize", restype=ctypes.c_uint64))

  # --- the Objective-C call --------------------------------------------------------------------------------

  def send(self, obj:int, selector:str, *args:Any, argtypes:tuple[Any, ...] = (), restype:Any = ctypes.c_void_p) -> Any:
    key = (restype, argtypes)
    fn = self._fns.get(key)
    if fn is None:
      fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", self._objc))
      self._fns[key] = fn
    return fn(obj, self._objc.sel_registerName(selector.encode()), *args)

  def string(self, nsstring:int) -> str:
    if not nsstring:
      return ""
    return ctypes.cast(self.send(nsstring, "UTF8String"), ctypes.c_char_p).value.decode()

  def nsstring(self, text:str) -> int:
    cls = self._objc.objc_getClass(b"NSString")
    return self.send(cls, "stringWithUTF8String:", text.encode(), argtypes=(ctypes.c_char_p,))

  def _error(self, err:ctypes.c_void_p) -> str:
    if not err.value:
      return "no error object"
    return self.string(self.send(err.value, "localizedDescription"))

  def _own(self, obj:int) -> int:
    self._owned.append(obj)
    return obj

  def release(self, obj:int) -> None:
    if obj in self._owned:
      self._owned.remove(obj)
    self.send(obj, "release", restype=None)

  def close(self) -> None:
    for obj in reversed(self._owned):
      self.send(obj, "release", restype=None)
    self._owned.clear()

  # --- kernels and buffers ---------------------------------------------------------------------------------

  def _new(self, class_name:str) -> int:
    return self.send(self.send(self._objc.objc_getClass(class_name.encode()), "alloc"), "init")

  def library(self, source:str, macros:dict[str, str] | None = None) -> int:
    """Compile MSL source into a library. macros are preprocessor definitions (MTLCompileOptions), the same way an
    engine passes its own build flags to the runtime compiler."""
    pool = self._objc.objc_autoreleasePoolPush()
    try:
      options = None
      if macros:
        prep = self.send(self._objc.objc_getClass(b"NSMutableDictionary"), "dictionary")
        for key, value in macros.items():
          self.send(prep, "setObject:forKey:", self.nsstring(value), self.nsstring(key),
                    argtypes=(ctypes.c_void_p, ctypes.c_void_p), restype=None)
        options = self._new("MTLCompileOptions")
        self.send(options, "setPreprocessorMacros:", prep, argtypes=(ctypes.c_void_p,), restype=None)
      err = ctypes.c_void_p(0)
      lib = self.send(self.device, "newLibraryWithSource:options:error:", self.nsstring(source), options, ctypes.byref(err),
                      argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
      if not lib:
        raise RuntimeError(f"Metal shader compile failed: {self._error(err)}")
      return self._own(lib)
    finally:
      self._objc.objc_autoreleasePoolPop(pool)

  def pipeline(self, lib:int, name:str, constants:dict[int, tuple[str, int]] | None = None) -> dict[str, Any]:
    """One compute pipeline for a kernel in the library, with what it reports about itself. constants are Metal
    function constants by index: {index: ("short" | "int" | "bool", value)}, the values a kernel's
    [[function_constant(n)]] declarations take at specialization."""
    pool = self._objc.objc_autoreleasePoolPush()
    try:
      err = ctypes.c_void_p(0)
      if constants:
        values = self._new("MTLFunctionConstantValues")
        for index, (kind, value) in constants.items():
          code, fmt = MTL_DATA_TYPES[kind]
          raw = struct.pack(fmt, value)
          self.send(values, "setConstantValue:type:atIndex:", raw, code, index,
                    argtypes=(ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong), restype=None)
        fn = self.send(lib, "newFunctionWithName:constantValues:error:", self.nsstring(name), values, ctypes.byref(err),
                       argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
      else:
        fn = self.send(lib, "newFunctionWithName:", self.nsstring(name), argtypes=(ctypes.c_void_p,))
      if not fn:
        raise RuntimeError(f"Metal library has no kernel {name!r}" + (f": {self._error(err)}" if constants else ""))
      self._own(fn)
      err = ctypes.c_void_p(0)
      pso = self.send(self.device, "newComputePipelineStateWithFunction:error:", fn, ctypes.byref(err),
                      argtypes=(ctypes.c_void_p, ctypes.c_void_p))
      if not pso:
        raise RuntimeError(f"Metal pipeline for {name!r} failed: {self._error(err)}")
      self._own(pso)
      return {
        "pso": pso,
        "thread_execution_width": int(self.send(pso, "threadExecutionWidth", restype=ctypes.c_ulong)),
        "max_threads_per_threadgroup": int(self.send(pso, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong)),
        "static_threadgroup_memory_bytes": int(self.send(pso, "staticThreadgroupMemoryLength", restype=ctypes.c_ulong)),
      }
    finally:
      self._objc.objc_autoreleasePoolPop(pool)

  def compile(self, source:str, names:list[str], *, macros:dict[str, str] | None = None,
              constants:dict[int, tuple[str, int]] | None = None) -> dict[str, dict[str, Any]]:
    """Compile MSL source and build one pipeline per kernel name (library, then pipeline)."""
    lib = self.library(source, macros)
    return {name: self.pipeline(lib, name, constants) for name in names}

  def buffer(self, data:bytes | None = None, *, length:int | None = None) -> int:
    if data is not None:
      return self._own(self.send(self.device, "newBufferWithBytes:length:options:", data, len(data), _SHARED,
                                 argtypes=(ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong)))
    return self._own(self.send(self.device, "newBufferWithLength:options:", int(length or 0), _SHARED,
                               argtypes=(ctypes.c_ulong, ctypes.c_ulong)))

  def read(self, buf:int, length:int) -> bytes:
    return ctypes.string_at(self.send(buf, "contents"), length)

  def run(self, pso:int, buffers:list[int], constants:list[bytes], groups:int, threads:int) -> float:
    """One 1-D dispatch: buffers at indices 0.., then constants as bytes after them (dispatch does the work)."""
    return self.dispatch(pso, [*buffers, *constants], (groups, 1, 1), (threads, 1, 1))

  def dispatch(self, pso:int, bound:list[int | bytes], groups:tuple[int, int, int], threads:tuple[int, int, int],
               threadgroup_bytes:int = 0) -> float:
    """One dispatch in its own command buffer, waited for. bound is the kernel's arguments by index: a buffer
    handle (int) is bound with setBuffer, raw bytes with setBytes (an argument struct). groups and threads are the
    3-D grid. Returns the GPU interval in microseconds."""
    pool = self._objc.objc_autoreleasePoolPush()
    try:
      cb = self.send(self.queue, "commandBuffer")
      enc = self.send(cb, "computeCommandEncoder")
      self.send(enc, "setComputePipelineState:", pso, argtypes=(ctypes.c_void_p,), restype=None)
      for i, arg in enumerate(bound):
        if isinstance(arg, (bytes, bytearray)):
          self.send(enc, "setBytes:length:atIndex:", bytes(arg), len(arg), i,
                    argtypes=(ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong), restype=None)
        else:
          self.send(enc, "setBuffer:offset:atIndex:", arg, 0, i, argtypes=(ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong),
                    restype=None)
      if threadgroup_bytes:
        self.send(enc, "setThreadgroupMemoryLength:atIndex:", threadgroup_bytes, 0,
                  argtypes=(ctypes.c_ulong, ctypes.c_ulong), restype=None)
      self.send(enc, "dispatchThreadgroups:threadsPerThreadgroup:", _Size(*groups), _Size(*threads),
                argtypes=(_Size, _Size), restype=None)
      self.send(enc, "endEncoding", restype=None)
      self.send(cb, "commit", restype=None)
      self.send(cb, "waitUntilCompleted", restype=None)
      status = int(self.send(cb, "status", restype=ctypes.c_ulong))
      if status != _COMPLETED:
        raise RuntimeError(f"Metal command buffer ended with status {status}: {self._error(ctypes.c_void_p(self.send(cb, 'error')))}")
      start = self.send(cb, "GPUStartTime", restype=ctypes.c_double)
      end = self.send(cb, "GPUEndTime", restype=ctypes.c_double)
      return (end - start) * 1e6
    finally:
      self._objc.objc_autoreleasePoolPop(pool)
