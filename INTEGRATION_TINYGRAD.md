# Tinygrad evidence integration package

## Scope and ownership

This branch packages BoltBeam-side, CPU/static-only consumers of serialized tinygrad evidence. BoltBeam owns normalization, validation, joins, and decay analysis. Tinygrad remains the owner of runtime execution, model lifecycle, route selection, KFD dispatch observation, profiler collection, code-object capture, and writing producer artifacts. No BoltBeam code is to be placed in tinygrad's runtime paths.

This is **not a tinygrad cutover**. It imports no tinygrad or model code, opens no KFD device, attaches to no process, invokes no ROCm tooling, and performed no GPU execution. No speculative G5 candidate is included.

## Included modules and contracts

- `boltbeam.adapters.tinygrad_prefill`: normalizes prefill profile, route-attachment, and route-execution evidence into `boltbeam.timing_trace.v1`; route status is evidence, not policy. Missing/partial counters remain explicit metadata.
- `boltbeam.adapters.tinygrad_decode`: accepts only `tinygrad.decode.fixed_depth.v2` artifact version 2 and emits `boltbeam.timing_trace.v1` W/D whole-step rows. It fails closed on model identity, one-token fixed-depth lifecycle, route sequence, program-count, or repetition disagreement.
- `boltbeam.decode_decay`: produces `tinygrad_decode_depth_slope.v1` only for matched ctx512/4096 W and D pairs.
- `boltbeam.decode_resource_evidence`: consumes final static resource evidence under `boltbeam.decode_resource_trace.v1` and `boltbeam.amd_isa_manifest.v1`; it requires source/binary identity agreement and all resource counters.
- `boltbeam.kfd_observation_bridge`: validates and atomically joins producer sidecars under `tinygrad.kfd_launch_sidecar.v1` with candidate-scoped timing/resource evidence. Invalid or mismatched artifacts add blockers without augmenting the workload.

## Exact 14B fixture findings

Model: `Qwen3-14B-Q4_K_M`; each artifact has one measurement and one repetition. The matched ctx512/4096 pair kept route sequence `["flash"]` and `1133` programs/token.

| Measurement | ctx512 | ctx4096 | Decay slope |
| --- | --- | --- | --- |
| W, production generate-item/token | 15.691776003222913 ms, 63.727649425699894 tok/s | 17.59034499991685 ms, 56.84936821902737 tok/s | 0.5297357691668357 us/context token |
| D, same-model JIT plus final sync diagnostic | 37.83939099957934 ms, 26.42748663716911 tok/s | 41.87087799800793 ms, 23.882947953648753 tok/s | 1.1254154560353203 us/context token |

A ctx128 14B artifact exists (W 18.69268500013277 ms; D 40.802274001180194 ms) but is not used to establish a three-point fit. D is explicitly not an upper bound. The authority artifacts provide neither a hardware-counter stream nor a measured host-sync split; the adapter reports counter evidence as unavailable.

## Exhaustive-scope reconciliation and G=5 decision

The exhaustive scope is capture-driven and measurement-gated: retained evidence is coverage, not a candidate closure. These adapters therefore preserve bounded observations and fail closed at ownership boundaries. The decode adapter records only the selected `flash` route and programs/token; it has no G=5 candidate identity. The KFD bridge can establish candidate-scoped launch identity only when a producer supplies a valid `tinygrad.kfd_launch_sidecar.v1` sidecar and matching timing/resource artifacts. None is present in this package.

**Decision: `invalid_campaign`.** The 14B ctx512/4096 decay pair is valid only as a route-level wall-time observation. It cannot be classified `localized` or `bounded_not_localized` for G=5 because it lacks candidate binding, KFD/resource join evidence, hardware counters, host-sync attribution, and repeated samples. No GPU causality is claimed.

## Limitations and unrun tests

The slopes are pairwise single-repetition observations, not uncertainty-bounded regressions. KFD sidecars are producer observations, not an independent timestamp oracle; they do not prove packet visibility, counter provenance, clock calibration, kernel correctness, or profiler equivalence. Static resource parsing does not infer resource facts from ISA text.

The complete configured CPU/static BoltBeam suite was run: `pytest` collected 983 tests and all passed. No GPU, tinygrad runtime, profiler, ROCm tool, KFD, or model execution was run.
