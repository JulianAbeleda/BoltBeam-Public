# Tinygrad direct-KFD launch sidecar seam

`boltbeam.kfd_observation_bridge` is a CPU-only reader for
`tinygrad.kfd_launch_sidecar.v1`. It does not import tinygrad, open KFD, attach
to a process, monkey-patch a queue, or depend on rocprof visibility.

## Producer seam

In a tinygrad checkout, emit one record from the owned AMD launch path: after
`AMDProgram.__call__` has constructed/submitted its KFD dispatch, capture the
monotonic `submit_ns`; after the completion signal is observed by that same
owned call path, capture `complete_ns`. The producer must write the sidecar as
an ordinary output artifact after the run. No BoltBeam code belongs in
`tinygrad/runtime/ops_amd.py`.

Each record requires:

```json
{
  "program_id": "stable producer-local id",
  "source_sha256": "64 hex chars",
  "binary_sha256": "64 hex chars",
  "grid": [128, 1, 1],
  "workgroup": [256, 1, 1],
  "submit_ns": 1000,
  "complete_ns": 1900,
  "counters": {"optional_metric": 1.0}
}
```

Wrap records with the required candidate identity:

```json
{"schema":"tinygrad.kfd_launch_sidecar.v1","candidate_id":"candidate-kfd-1","records":[...]}
```

The source and binary hashes are identities supplied by the producer, not
claims BoltBeam can recompute from a process image. `program_id` is local to
the sidecar and is never treated as globally unique.

## Consumer seam

Call `join_kfd_launch_sidecar(sidecar, timing_result, resource_snapshot,
target)`. It invokes the existing copied timing and resource validators for the
same candidate and augments an `EpochReport` only if all three artifacts are
valid. A bad hash, invalid time interval, candidate mismatch, or invalid
timing/resource artifact yields blockers and leaves the workload untouched.

## Limits

This is direct producer observation, not an independent GPU timestamp oracle.
It proves only that the producer wrote internally consistent records. It does
not establish KFD packet visibility, hardware-counter provenance, clock-domain
calibration, kernel correctness, or profiler equivalence. Counters are optional
and are preserved without assigning a quality level. A tinygrad integration
must keep its own synchronization semantics unchanged and must not make output
writing part of the launch critical path.
