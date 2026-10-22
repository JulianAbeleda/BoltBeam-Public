"""The GPU vendors BoltBeam knows: one row each, the one place a vendor's facts and words live.

    vendor("amd")            the row by vendor name (the hardware scan's `vendor`)
    for_backend("CUDA")      the row by target backend (data/targets.json `backend`)
    for_capture("nsys")      the row by its outside capture's method

A vendor is a row here, not a branch elsewhere: the scan's tool and probe, the facts a measured chip profile keeps, the
outside capture, the kernel bridge and its flush size, the toolchain a run records, the modules that are this vendor's
alone, and the words a user reads for its hardware (terms). A lookup with no row raises NoVendorRow, naming the key and
the rows there are: nothing is assumed in its place. A fact a vendor does not have is None, never another vendor's.

Code that lives in workflow/ or collectors/ is named by reference ("module:attribute") and imported when asked
(resolve), so this module imports neither at its top.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


class NoVendorRow(LookupError, RuntimeError):
  """A vendor, or a fact of one, that has no row here: nothing is assumed in its place. A RuntimeError, as the
  readers' other refusals are."""


def resolve(ref:str | None) -> Any:
  """"module:attribute" -> the attribute, or the module when there is no ":"; None stays None."""
  if ref is None:
    return None
  mod, _, attr = ref.partition(":")
  got = importlib.import_module(mod)
  return getattr(got, attr) if attr else got


@dataclass(frozen=True)
class Capture:
  """The outside capture that sees this vendor's kernels inside a running engine (collectors/vendor_capture.py)."""
  method: str            # the capture's id in traces and measure_status.json
  install: str           # what to install when it is missing
  words: str             # how a screen names a time it captured
  tool_word: str         # the tool, as the in-model source names it
  in_model_needs: str    # the Setup reason when it is missing
  tinygrad_reason: str   # why tinygrad's role time is its own events on this vendor, not this capture
  trace_file: str | None = None    # the kernel trace a capture folder keeps (tools/record_vendor_fixture.py)
  trace_header: str | None = None  # that trace's header's first word
  closed_observer: str | None = None  # "module:function" reading the instructions a closed library's kernels executed


@dataclass(frozen=True)
class ProfileShape:
  """What a measured chip profile of this vendor's discrete GPU keeps (workflow/chips.profile_row)."""
  scan_method: str
  scan_keys: tuple[str, ...]       # the driver-API facts the matrix probe prints
  lds_key: str                     # the scan key that sizes lds_bytes_per_cu
  compute_units: Callable[[dict[str, Any], dict[str, Any]], Any]  # (device, matrix result) -> the count
  dot_primitives: tuple[str, ...]
  dot_status: str
  compute: str                     # the matrix probe module: its matrix_tflops and fact_source read its result


@dataclass(frozen=True)
class Toolkit:
  """Where the vendor's toolkit installs (target/toolkits.py): its root's environment variables, the documented
  default, and the tools found under its bin/ when they are not on PATH."""
  name: str
  env: tuple[str, ...]
  default: str
  tools: tuple[str, ...]


@dataclass(frozen=True)
class Vendor:
  vendor: str
  backend: str                     # data/targets.json `backend`
  tinygrad_device: str             # tinygrad's DEV for this backend
  tinygrad_per_gpu: bool           # tinygrad's per-GPU read probe runs on it (workflow/layout.py)
  platform: str | None             # the OS its scan tool answers on; None for any
  scan_tools: tuple[str, ...]      # the management tools; the first is the one the scan asks
  other_tools: tuple[str, ...]     # tools only a scan step or a CLI command names
  probe: str                       # the scan: (tool path, run_command) -> (devices, error), in its runtime's order
  match_keys: tuple[str, ...]      # what a chip profile matches on beyond name, architecture and memory
  order: str                       # the device order its probe returns, in words ({KFD_NODES} filled by the scan)
  notes: tuple[str, ...]
  unrecorded_order: str | None     # why a multi-GPU scan from before its order was recorded names the wrong card
  ggml_backend: str                # the ggml backend whose llama.cpp builds run it (libggml-<ggml_backend>.so)
  telemetry: Callable[[dict[str, Any]], Any] | None  # device -> a telemetry reader; None: BoltBeam reads none
  clock_policy: Callable[[Any], Any] | str | None    # source -> the clock policy (or its reference); None: unknown
  profile: ProfileShape | None     # None: the profile shape is profile_row's own (Apple: unified memory)
  capture: Capture
  toolchain: tuple[tuple[str, tuple[str, ...] | None, str | None], ...]  # (key, argv, pattern); argv None: the engine's torch
  torch_runtime: str | None        # torch.version's field for this vendor's runtime
  kernel_dialect: str | None       # the source dialect of BoltBeam's own kernels on its bridge; None: no bridge
  bridge: str | None               # the kernel bridge (collectors/kernel_timer.bridge_for)
  flush_bytes: int | None          # the sweep before each timed sample: larger than the last-level cache
  topology: tuple[tuple[str, ...], str] | None  # (argv, reader): the link between each GPU pair
  probe_absent: Mapping[str, str] | None  # what the bridge's probe cannot report, by full-probe field
  toolkit: Toolkit | None
  owned_modules: tuple[str, ...]   # modules that serve this vendor alone, under boltbeam/
  terms: Mapping[str, str | None]  # the words a user reads for this vendor's hardware

  def term(self, key:str) -> str | None:
    """One of the vendor's words; None when it has no such thing. A key no row defines is NoVendorRow."""
    if key not in self.terms:
      raise NoVendorRow(f"no term {key!r} for {self.vendor!r} GPUs in target/vendors.py (terms: {', '.join(self.terms)})")
    return self.terms[key]

  def scan_facts(self, **where:str) -> dict[str, Any]:
    """The scan's facts and, for a discrete GPU, the measured profile's (workflow/chips._vendors()), references
    resolved; `where` fills the placeholders in `order`."""
    row = {"scan_tool": self.scan_tools[0], "platform": self.platform, "probe": resolve(self.probe),
           "match_keys": self.match_keys, "order": self.order.format(**where), "notes": self.notes,
           "unrecorded_order": self.unrecorded_order, "ggml_backend": self.ggml_backend}
    if self.telemetry is not None:
      row["telemetry"] = self.telemetry
    if self.clock_policy is not None:
      row["clock_policy"] = resolve(self.clock_policy) if isinstance(self.clock_policy, str) else self.clock_policy
    if self.profile is not None:
      p = self.profile
      row.update(backend=self.backend, tinygrad_device=self.tinygrad_device, scan_method=p.scan_method,
                 scan_keys=p.scan_keys, lds_key=p.lds_key, compute_units=p.compute_units,
                 dot_primitives=list(p.dot_primitives), dot_status=p.dot_status, compute=resolve(p.compute))
    return row


TERM_KEYS = ("compute_unit", "subgroup", "matrix_unit", "core_clock", "core_clock_mhz", "isa", "blas", "graphs",
             "graph_trace", "profiler", "management_tool", "telemetry_source", "native_read_probe")
# the fields of a full probe row its bridge may not report (collectors/metal_native._absent)
PROBE_FIELDS = ("counters", "isa.instruction_histogram", "isa.vector_load_bits", "resources.registers",
                "resources.scratch_bytes")
_FROZEN = lambda d: MappingProxyType(dict(d))  # noqa: E731


def _terms(*words:str | None) -> Mapping[str, str | None]:
  assert len(words) == len(TERM_KEYS)
  return _FROZEN(zip(TERM_KEYS, words))


_ROWS = (
  Vendor(
    vendor="nvidia", backend="CUDA", tinygrad_device="NV", tinygrad_per_gpu=True, platform=None,
    scan_tools=("nvidia-smi",), other_tools=("cuobjdump", "nvdisasm", "ncu"), probe="boltbeam.workflow.autoscan:_probe_nvidia", match_keys=("compute_capability",),
    order="nvidia-smi's, by PCI address (CUDA's own order is the same under CUDA_DEVICE_ORDER=PCI_BUS_ID)",
    notes=("Primary GPU is the first device returned by nvidia-smi.",), unrecorded_order=None,
    # telemetry: NVML, which gpu_telemetry.Telemetry opens by device index; the clock policy nvidia-smi allows
    ggml_backend="cuda", telemetry=lambda device: None,
    clock_policy=lambda source: resolve("boltbeam.collectors.gpu_telemetry:clock_policy")(0),
    profile=ProfileShape(
      scan_method="nvidia-smi and cudaDeviceGetAttribute (BoltBeam's CUDA matrix probe prints them)",
      scan_keys=("sm_count", "shared_mem_per_sm_bytes", "l2_cache_bytes", "clock_khz", "memory_clock_khz",
                 "memory_bus_bits", "total_global_mem_bytes"),
      lds_key="shared_mem_per_sm_bytes", compute_units=lambda device, matrix: matrix.get("sm_count"),
      dot_primitives=("mma_m16n8k16", "dp4a"), dot_status="hardware_family", compute="boltbeam.collectors.cuda_compute"),
    capture=Capture(method="nsys", install="install Nsight Systems (it comes with the CUDA toolkit) so nsys is on PATH",
                    words="captured with nsys", tool_word="nsys", in_model_needs="in-model needs nsys",
                    tinygrad_reason="nsys cannot see tinygrad's NV backend: it talks to the driver directly, not through CUDA",
                    trace_file="cuda_gpu_trace.csv", trace_header="Start",
                    closed_observer="boltbeam.collectors.instruction_path:observe_closed"),
    toolchain=(("driver", ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"), None),
               ("cuda", ("nvcc", "--version"), r"release\s+(\d+\.\d+)"),
               ("torch", None, None),
               ("nsight_systems", ("nsys", "--version"), None),
               ("nsight_compute", ("ncu", "--version"), None)),
    torch_runtime="cuda", kernel_dialect="cuda", bridge="boltbeam.runtime.cuda_device:Cuda", flush_bytes=256 << 20,
    topology=(("nvidia-smi", "topo", "-m"), "boltbeam.workflow.layout:read_topology"),
    probe_absent=_FROZEN({
      "counters": "the CUDA bridge reads no per-dispatch hardware counters (that is Nsight Compute's)",
      "isa.instruction_histogram": "the cubin's SASS is not disassembled by BoltBeam",
      "isa.vector_load_bits": "the compiled load width is not visible; the source loads are in kernel.source",
      "resources.registers": "the CUDA bridge does not read cuFuncGetAttribute(NUM_REGS)",
      "resources.scratch_bytes": "the CUDA bridge does not read cuFuncGetAttribute(LOCAL_SIZE_BYTES)"}),
    toolkit=Toolkit("cuda", ("CUDA_HOME", "CUDA_PATH"), "/usr/local/cuda", ("nvcc",)),
    owned_modules=("runtime/cuda_device.py", "collectors/cuda_compute.py", "collectors/instruction_path.py",
                   "collectors/cuda_memory.py", "collectors/cuda_counter_stream.py", "collectors/cuda_activity_stream.py", "collectors/cupti_hes.py",
                   "collectors/cubin_launch.py", "collectors/cublas_reduction_work.py", "collectors/kernel_oracle.py",
                   "profiler/ncu_counters.py", "profiler/ncu_audit.py", "profiler/importers/ncu.py", "cli/ncu.py",
                   "cli/gemm.py"),
    terms=_terms("SM", "warp", "Tensor Cores", "SM clock", "SM MHz", "PTX/SASS", "cuBLAS", "CUDA graphs",
                 "Nsight graph-node tracing", "Nsight Systems (nsys)", "nvidia-smi", "NVML",
                 "BoltBeam's native CUDA read probe")),
  Vendor(
    vendor="amd", backend="AMD", tinygrad_device="AMD", tinygrad_per_gpu=True, platform=None,
    scan_tools=("amd-smi", "rocm-smi"), other_tools=(), probe="boltbeam.workflow.autoscan:_probe_amd_hip_order",
    match_keys=("compute_units",), order="the HIP runtime's: amd-smi's devices in KFD topology node order ({KFD_NODES})",
    notes=("Primary GPU is HIP's device 0: amd-smi's devices in the KFD topology's order.",),
    # before 62bf26b the scan listed amd-smi's own order, which is not HIP's: such a list names the wrong card
    unrecorded_order="it lists amd-smi's order, scanned before BoltBeam put AMD GPUs in HIP's order",
    ggml_backend="hip", telemetry=lambda device: resolve("boltbeam.collectors.gpu_telemetry:AmdSysfs")(device["pci_bus_id"]),
    clock_policy="boltbeam.collectors.gpu_telemetry:amd_clock_policy",
    profile=ProfileShape(
      scan_method="amd-smi static and hipDeviceGetAttribute (BoltBeam's HIP matrix probe prints them)",
      scan_keys=("multiprocessor_count", "shared_mem_per_block_bytes", "l2_cache_bytes", "clock_khz",
                 "memory_clock_khz", "memory_bus_bits", "total_global_mem_bytes", "pci_bus_id", "hip_runtime_version"),
      # amd-smi's compute units; HIP's multiprocessor count is work-group processors on RDNA
      lds_key="shared_mem_per_block_bytes", compute_units=lambda device, matrix: device.get("compute_units"),
      # no probe derives AMD's dot instructions yet: none is claimed, so nothing is served on an assumption
      dot_primitives=(), dot_status="unknown", compute="boltbeam.collectors.hip_compute"),
    capture=Capture(method="rocprofv3", install="install ROCm's rocprofiler-sdk", words="captured with rocprofv3",
                    tool_word="rocprofv3", in_model_needs="in-model needs rocprofv3",
                    tinygrad_reason="rocprofv3 sees 0 kernels from tinygrad's AMD backend: it talks to /dev/kfd directly",
                    trace_file="capture_kernel_trace.csv", trace_header='"Kind"'),
    toolchain=(("rocm", ("amd-smi", "version"), r"ROCm version:\s*(\d+\.\d+(?:\.\d+)?)"),
               ("amdgpu", ("amd-smi", "version"), r"amdgpu version:\s*(\d+\.\d+(?:\.\d+)?)"),
               ("hip", ("hipcc", "--version"), r"HIP version:\s*(\d+\.\d+(?:\.\d+)?)"),
               ("torch", None, None),
               ("rocprofv3", ("rocprofv3", "--version"), r"version:\s*(\d+\.\d+(?:\.\d+)?)")),
    torch_runtime="hip", kernel_dialect=None, bridge=None, flush_bytes=None,
    topology=(("amd-smi", "topology"), "boltbeam.collectors.multi_gpu:read_amd_topology"),
    probe_absent=None,
    toolkit=Toolkit("rocm", ("ROCM_PATH", "HIP_PATH"), "/opt/rocm", ("amd-smi", "rocminfo", "rocprofv3", "hipcc")),
    owned_modules=("runtime/hip_device.py", "runtime/amd_runtime_bridge.py", "runtime/kfd_observation_bridge.py",
                   "runtime/direct_kfd_bridge.py", "collectors/hip_compute.py", "collectors/rocm_isa.py",
                   "collectors/rocpd_trace.py", "collectors/llama_rocprof.py", "collectors/tinygrad_rocprof.py",
                   "collectors/tinygrad_native_pmc.py", "artifacts/llama_rocprof.py", "artifacts/tinygrad_rocprof.py"),
    terms=_terms("CU", "wavefront", "WMMA matrix cores", "GPU core clock (sclk)", "sclk MHz", "amdgcn",
                 "hipBLASLt / rocBLAS", "HIP graphs", "rocprofv3 kernel tracing", "rocprofv3", "amd-smi",
                 "amdgpu sysfs", "BoltBeam's native HIP read probe")),
  Vendor(
    vendor="apple", backend="Metal", tinygrad_device="METAL", tinygrad_per_gpu=False, platform="Darwin",
    scan_tools=("system_profiler",), other_tools=(), probe="boltbeam.workflow.autoscan:_probe_apple_metal",
    match_keys=("apple_soc", "gpu_cores"), order="system_profiler's",
    notes=("Primary GPU is the first Apple device returned by system_profiler.",
           "Dynamic Metal limits require the tinygrad provider fact adapter."),
    # BoltBeam reads no telemetry and no clock policy from a Mac yet: unknown, said, never NVIDIA's
    unrecorded_order=None, ggml_backend="metal", telemetry=None, clock_policy=None,
    profile=None,  # Apple's profile shape is still profile_row's own (unified memory, no driver-API facts)
    capture=Capture(method="metal-system-trace", install="install Xcode (xctrace comes with it)",
                    words="captured with Metal System Trace", tool_word="xctrace",
                    in_model_needs="in-model needs Xcode's Metal System Trace on this Mac",
                    tinygrad_reason="no outside capture: Metal per-kernel time needs Xcode's Metal System Trace"),
    toolchain=(("xctrace", ("xcrun", "xctrace", "version"), None), ("torch", None, None)),
    torch_runtime=None, kernel_dialect="msl", bridge="boltbeam.runtime.metal_device:Metal", flush_bytes=256 << 20,
    topology=None,
    probe_absent=_FROZEN({
      "counters": "Metal reports no per-dispatch hardware counters without Instruments",
      "isa.instruction_histogram": "Metal does not export the compiled AGX ISA",
      "isa.vector_load_bits": "the compiled load width is not visible; the source loads are in kernel.source",
      "resources.registers": "MTLComputePipelineState does not report registers",
      "resources.scratch_bytes": "MTLComputePipelineState does not report scratch"}),
    toolkit=None,
    owned_modules=("runtime/metal_device.py", "collectors/metal_bandwidth.py", "collectors/metal_system_trace.py"),
    terms=_terms("GPU core", "SIMD-group", None, "GPU clock", "GPU MHz", "AIR", "MPS", "Metal command buffer reuse",
                 None, "Metal System Trace (xctrace)", "system_profiler", None, "BoltBeam's Metal read probe")),
)

VENDORS: Mapping[str, Vendor] = _FROZEN((v.vendor, v) for v in _ROWS)
BY_BACKEND: Mapping[str, Vendor] = _FROZEN((v.backend, v) for v in _ROWS)
BY_CAPTURE: Mapping[str, Vendor] = _FROZEN((v.capture.method, v) for v in _ROWS)


def _lookup(table:Mapping[str, Vendor], key:Any, what:str) -> Vendor:
  row = table.get(str(key)) if key is not None else None
  if row is None:
    raise NoVendorRow(f"no vendor row for {what} {key!r} in target/vendors.py (known: {', '.join(table)})")
  return row


def vendor(name:str | None) -> Vendor:
  """The row of a vendor by name ("nvidia", "amd", "apple"); NoVendorRow, naming the known ones, when there is none."""
  return _lookup(VENDORS, name, "vendor")


def for_backend(backend:str | None) -> Vendor:
  """The row of a target backend ("CUDA", "AMD", "Metal"); NoVendorRow, naming the known ones, when there is none."""
  return _lookup(BY_BACKEND, backend, "backend")


def for_capture(method:str | None) -> Vendor:
  """The row whose outside capture this is ("nsys", "rocprofv3", "metal-system-trace"); NoVendorRow otherwise."""
  return _lookup(BY_CAPTURE, method, "capture method")
