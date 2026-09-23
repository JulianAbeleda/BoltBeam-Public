"""Exact Boltbeam seed for the admitted Q6 FFN-down Stream-K route."""
from __future__ import annotations

from boltbeam.search.kernel_candidate import KERNEL_CANDIDATE_SCHEMA, KERNEL_ROUTE_SCHEMA, KernelCandidate, KernelRoute


def _provenance(revision: str) -> dict[str, str]:
  return {"generator_id": "boltbeam.q6_destination_route", "generator_revision": revision,
          "schema_revision": KERNEL_CANDIDATE_SCHEMA}


def q6_destination_candidates(resolved_target_hash: str, generator_revision: str) -> tuple[KernelCandidate, KernelCandidate]:
  target = {"target_id": "nvidia_sm120", "backend": "CUDA", "arch": "sm_120", "subgroup_size": 32,
            "resolved_target_hash": resolved_target_hash}
  main = KernelCandidate({
    "schema_version": KERNEL_CANDIDATE_SCHEMA, "kernel_id": "q6_ffn_down_streamk_destination_main",
    "family": "packed_q6_q8_streamk_main.v1", "phase": "prefill", "role": "ffn_down",
    "operation": "packed_q6_q8_matmul_partial", "shape": {"m": 512, "n": 4096, "k": 12288},
    "abi": [
      {"name": "partials", "access": "write", "dtype": "fp32", "layout": "destination_major_slot"},
      {"name": "q6_weight", "access": "read", "dtype": "uint16", "layout": "canonical_q6_k"},
      {"name": "q8_activation", "access": "read", "dtype": "uint32", "layout": "q8_1_record"}],
    "parameters": {"tile_m": 128, "tile_n": 128, "tile_k": 256, "streamk_owners": 170,
      "streamk_segment": 0, "streamk_segments_in_cta": True, "segment_order": "ascending",
      "prefetch_second_panel": True, "combined_initial_publish": True, "factor_dA": False,
      "oracle_publisher": True, "weight_scale_contract": "trusted_fp16_packed",
      "partial_output_layout": "destination_major"},
    "launch": {"grid": [170, 1, 1], "block": [256, 1, 1], "dynamic_shared_memory_bytes": 58368},
    "resources": {"workspace_bytes": 22282240, "static_shared_memory_bytes": 0}, "target": target,
    "correctness": {"oracle": "llama_q6_streamk_partial_permutation", "atol": 0.0, "rtol": 0.0, "bit_exact": True},
    "provenance": _provenance(generator_revision)})
  fixup = KernelCandidate({
    "schema_version": KERNEL_CANDIDATE_SCHEMA, "kernel_id": "q6_ffn_down_destination_fixup",
    "family": "q6_destination_major_fixup.v1", "phase": "prefill", "role": "ffn_down",
    "operation": "ordered_partial_reduction", "shape": {"m": 512, "n": 4096, "contributors": 3},
    "abi": [
      {"name": "out", "access": "write", "dtype": "fp32", "layout": "row_major"},
      {"name": "partials", "access": "read", "dtype": "fp32", "layout": "destination_major_slot"},
      {"name": "slots", "access": "read", "dtype": "int32", "layout": "tile_contributor"},
      {"name": "counts", "access": "read", "dtype": "int32", "layout": "tile"}],
    "parameters": {"rows": 128, "cols": 128, "tiles_m": 4, "tiles_n": 32, "slices": 4,
      "threads": 128, "outputs_per_thread": 32, "fold_order": "slot0_slot1_slot2", "fadd": "rn"},
    "launch": {"grid": [128, 4, 1], "block": [128, 1, 1], "dynamic_shared_memory_bytes": 0},
    "resources": {"workspace_bytes": 0, "static_shared_memory_bytes": 0}, "target": target,
    "correctness": {"oracle": "llama_q6_final_output", "atol": 0.0, "rtol": 0.0, "bit_exact": True},
    "provenance": _provenance(generator_revision)})
  return main, fixup


def q6_destination_route(resolved_target_hash: str, generator_revision: str) -> KernelRoute:
  main, fixup = q6_destination_candidates(resolved_target_hash, generator_revision)
  components = [
    {"name": "main", "candidate_hash": main.candidate_hash, "candidate": main.to_dict(), "depends_on": []},
    {"name": "fixup", "candidate_hash": fixup.candidate_hash, "candidate": fixup.to_dict(), "depends_on": ["main"]}]
  return KernelRoute({"schema_version": KERNEL_ROUTE_SCHEMA, "route_id": "q6_ffn_down_streamk_destination.v1",
    "phase": "prefill", "role": "ffn_down", "components": components, "outputs": ["fixup"],
    "provenance": {"generator_id": "boltbeam.q6_destination_route", "generator_revision": generator_revision,
                   "schema_revision": KERNEL_ROUTE_SCHEMA}})


__all__ = ["q6_destination_candidates", "q6_destination_route"]
