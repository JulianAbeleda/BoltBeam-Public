"""Host-to-device copy rate, measured on the GPU that will run: one buffer of BYTES copied to the card from pinned (page-
locked) host memory and from pageable (ordinary) host memory, the best of REPS after a warm-up copy, in GB/s.

Why both: an engine that copies weights out of ordinary memory (llama.cpp's experts in RAM: an mmap'd file) is held to
the pageable rate, which the driver stages through its own pinned bounce buffer; an engine that pins its host buffers
gets the pinned rate. On an RTX 5090 at gen 5 x16 (31.5 GB/s on the wire) they were 14 and 28 GB/s.

No new dependency: the CUDA driver API (libcuda, as runtime/cuda_device.py loads it) or HIP (libamdhip64, as
runtime/hip_device.py loads it) through ctypes, the GPU picked by its PCI bus id. The ctypes calls sit in _Cuda and _Hip;
measure() takes a fake for tests. Pinned memory is capped at PIN_CAP: BoltBeam never page-locks more than that.
"""
from __future__ import annotations

import ctypes
import time
from typing import Any

BYTES = 256 << 20  # far past any cache, and small enough to pin on any machine
REPS = 5
PIN_CAP = 512 << 20  # the most host memory this probe page-locks
_HIP_H2D = 1  # hipMemcpyHostToDevice
_CU_INTEGRATED = 18  # CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda.h (CUDA 13.2)
_HIP_INTEGRATED = 16  # hipDeviceAttributeIntegrated, hip/hip_runtime_api.h (ROCm 7.2.4)


class Unavailable(RuntimeError):
  """No GPU runtime here that can copy to the card at this PCI address (no libcuda / libamdhip64, or no device there)."""


class _Cuda:
  """The CUDA driver calls, one place."""
  API = "CUDA driver API"

  def __init__(self, pci_bus_id:str, lib:str = "libcuda.so.1"):
    try:
      self.cu = ctypes.CDLL(lib)
    except OSError as exc:
      raise Unavailable(f"libcuda did not load: {exc}") from exc
    try:
      self._call("cuInit", 0)
      dev, self.ctx = ctypes.c_int(), ctypes.c_void_p()
      self._call("cuDeviceGetByPCIBusId", ctypes.byref(dev), pci_bus_id.encode())
      self._call("cuDevicePrimaryCtxRetain", ctypes.byref(self.ctx), dev)
      try:
        self._call("cuCtxSetCurrent", self.ctx)
      except RuntimeError:
        self.cu.cuDevicePrimaryCtxRelease(dev)  # retained above: give it back before refusing
        raise
    except (RuntimeError, AttributeError) as exc:
      raise Unavailable(f"no CUDA device at PCI {pci_bus_id}: {exc}") from exc
    self.dev = dev

  def _call(self, name:str, *args) -> None:
    rc = getattr(self.cu, name)(*args)
    if rc != 0:
      raise RuntimeError(f"{name} failed with CUresult {rc}")

  def device_alloc(self, n:int) -> int:
    p = ctypes.c_uint64(); self._call("cuMemAlloc_v2", ctypes.byref(p), ctypes.c_size_t(n)); return p.value

  def device_free(self, p:int) -> None:
    self.cu.cuMemFree_v2(ctypes.c_uint64(p))

  def pinned_alloc(self, n:int) -> int:
    p = ctypes.c_void_p(); self._call("cuMemHostAlloc", ctypes.byref(p), ctypes.c_size_t(n), 0); return p.value

  def pinned_free(self, p:int) -> None:
    self.cu.cuMemFreeHost(ctypes.c_void_p(p))

  def integrated(self) -> bool:
    """CU_DEVICE_ATTRIBUTE_INTEGRATED (18, cuda.h: "Device is integrated with host memory")."""
    v = ctypes.c_int(); self._call("cuDeviceGetAttribute", ctypes.byref(v), _CU_INTEGRATED, self.dev); return bool(v.value)

  def copy(self, dst:int, src:int, n:int) -> None:
    self._call("cuMemcpyHtoD_v2", ctypes.c_uint64(dst), ctypes.c_void_p(src), ctypes.c_size_t(n))
    self._call("cuCtxSynchronize")

  def close(self) -> None:
    self.cu.cuDevicePrimaryCtxRelease(self.dev)


class _Hip:
  """The HIP calls, one place."""
  API = "HIP"

  def __init__(self, pci_bus_id:str):
    from boltbeam.runtime import hip_device
    try:
      drv = hip_device._Driver()
      self.hip = drv.hip
      self.index = hip_device.index_of(pci_bus_id, drv)
      self._call("hipSetDevice", self.index)
    except (hip_device.HipUnavailable, RuntimeError, OSError) as exc:
      raise Unavailable(f"no HIP device at PCI {pci_bus_id}: {exc}") from exc

  def _call(self, name:str, *args) -> None:
    rc = getattr(self.hip, name)(*args)
    if rc != 0:
      raise RuntimeError(f"{name} failed with hipError_t {rc}")

  def device_alloc(self, n:int) -> int:
    p = ctypes.c_void_p(); self._call("hipMalloc", ctypes.byref(p), ctypes.c_size_t(n)); return p.value

  def device_free(self, p:int) -> None:
    self.hip.hipFree(ctypes.c_void_p(p))

  def pinned_alloc(self, n:int) -> int:
    p = ctypes.c_void_p(); self._call("hipHostMalloc", ctypes.byref(p), ctypes.c_size_t(n), 0); return p.value

  def pinned_free(self, p:int) -> None:
    self.hip.hipHostFree(ctypes.c_void_p(p))

  def integrated(self) -> bool:
    """hipDeviceAttributeIntegrated (16, hip_runtime_api.h: "Device is integrated GPU")."""
    v = ctypes.c_int(); self._call("hipDeviceGetAttribute", ctypes.byref(v), _HIP_INTEGRATED, self.index); return bool(v.value)

  def copy(self, dst:int, src:int, n:int) -> None:
    self._call("hipMemcpy", ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(n), _HIP_H2D)
    self._call("hipDeviceSynchronize")

  def close(self) -> None:
    pass


def open_runtime(pci_bus_id:str):
  """CUDA first, then HIP: the first that has a device at this address. Unavailable says what each lacked."""
  why = []
  for kind in (_Cuda, _Hip):
    try:
      return kind(pci_bus_id)
    except Unavailable as exc:
      why.append(str(exc))
  raise Unavailable("; ".join(why))


def _best(rt, dst:int, src:int, n:int, reps:int) -> float:
  """One warm-up copy (it maps pages and sets up the driver's staging), then the best of `reps`: kernel_timer.samples,
  the tree's one warm-and-time loop."""
  from boltbeam.collectors.kernel_timer import samples
  def once() -> float:
    t0 = time.perf_counter()
    rt.copy(dst, src, n)
    return (time.perf_counter() - t0) * 1e6
  return n / (min(samples(once, warmups=1, count=reps)) / 1e6) / 1e9


def measure(pci_bus_id:str, *, nbytes:int = BYTES, reps:int = REPS, runtime=None) -> dict[str, Any]:
  """{"pinned": GB/s, "pageable": GB/s, "source": how}. runtime: a fake with device_alloc/free, pinned_alloc/free, copy,
  close and API (tests); else open_runtime. Unavailable when there is no runtime; ValueError past PIN_CAP."""
  if not 0 < nbytes <= PIN_CAP:
    raise ValueError(f"the probe pins at most {PIN_CAP >> 20} MiB, not {nbytes >> 20} MiB")
  rt = runtime or open_runtime(pci_bus_id)
  dst = pinned = 0
  try:
    dst = rt.device_alloc(nbytes)
    pageable = ctypes.create_string_buffer(nbytes)  # ordinary memory; zero-filled, so every page is resident
    ctypes.memset(pageable, 1, nbytes)
    pageable_gbs = _best(rt, dst, ctypes.addressof(pageable), nbytes, reps)
    del pageable
    pinned = rt.pinned_alloc(nbytes)
    ctypes.memset(pinned, 1, nbytes)
    pinned_gbs = _best(rt, dst, pinned, nbytes, reps)
  finally:
    if pinned:
      rt.pinned_free(pinned)
    if dst:
      rt.device_free(dst)
    rt.close()
  return {"pinned": pinned_gbs, "pageable": pageable_gbs,
          "source": f"{rt.API} copies of {nbytes >> 20} MiB to the GPU at {pci_bus_id}, best of {reps} after a warm-up"}


def integrated(pci_bus_id:str, *, runtime=None) -> dict[str, Any]:
  """Whether the GPU at this address shares the host's memory, as its runtime states it: {"integrated": bool, "source"}.
  Unavailable when there is no runtime. Nothing is allocated or copied."""
  rt = runtime or open_runtime(pci_bus_id)
  try:
    got = rt.integrated()
  finally:
    rt.close()
  return {"integrated": got, "source": f"{rt.API}: the device's integrated attribute at {pci_bus_id}"}
