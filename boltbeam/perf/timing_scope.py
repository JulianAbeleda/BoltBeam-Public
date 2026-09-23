"""Non-interchangeable timing objectives for kernel and harness measurements."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimingMetric:
  name: str
  includes: tuple[str, ...]
  kernel_model_target: bool = False


TIMING_METRICS = {
  "kernel_device_ms": TimingMetric("kernel_device_ms", ("device_execution",), True),
  "enqueue_sync_ms": TimingMetric("enqueue_sync_ms", ("host_enqueue", "device_execution", "synchronization")),
  "setup_ms": TimingMetric("setup_ms", ("tensor_construction", "materialization")),
  "readback_ms": TimingMetric("readback_ms", ("synchronization", "device_to_host_copy", "numpy_materialization")),
  "invocation_wall_ms": TimingMetric("invocation_wall_ms", ("setup", "enqueue_sync", "readback", "runner_overhead")),
}


def require_kernel_target(metric: str) -> None:
  if metric not in TIMING_METRICS: raise ValueError(f"unknown timing metric {metric!r}")
  if not TIMING_METRICS[metric].kernel_model_target: raise ValueError(f"{metric} is not a kernel-model target")


__all__ = ["TIMING_METRICS", "TimingMetric", "require_kernel_target"]
