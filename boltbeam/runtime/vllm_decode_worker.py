"""Offline decode graph provenance after production autotuning.

One continuous range holds graph construction and both measured passes. The
driver stops it after labeling; model loading and tactic selection are excluded.
"""
from vllm.v1.worker.gpu_worker import Worker
from boltbeam.runtime.vllm_semantics import GraphProvenance


class DecodeCaptureWorker(GraphProvenance, Worker):
  capture_graphs = True
  stop_after_graphs = False
