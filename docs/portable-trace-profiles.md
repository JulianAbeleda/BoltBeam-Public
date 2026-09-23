# Portable trace profiles

BoltBeam chooses a trace path from facts, not from an implicit ROCm, Metal, model, or runtime default.

The portable contract is `boltbeam.trace_execution_profile.v1`. A resolved profile joins five independent
inputs:

1. model identity: format, path, and streamed SHA-256;
2. workload identity: operation, fixed context/depth, warmups, and samples;
3. target identity: target id and backend;
4. runtime identity: provider id and configured executable or checkout;
5. collector capability: operation, supported providers, targets, scopes, metrics, and blind spots.

The core resolver does not map a runtime directly to ROCm or Metal. It filters the capability registry by
provider, target, and operation, then checks the request's required evidence scopes. Runtime-specific command
construction remains a leaf adapter because llama.cpp and tinygrad have different CLIs. Collector selection and
comparison policy do not live in those adapters.

## Evidence rules

- A comparison is matched only when model hash, workload mode/depth, target, and timing scope agree.
- Required scopes block resolution when unavailable.
- Optional scopes and requested metrics are recorded as unavailable; one scope is never inferred from another.
- Runtime benchmark output owns whole-step wall time and throughput.
- Collector intervals provide attribution. Overlapping intervals are unioned, never blindly summed.
- A target is required directly or through a run manifest. There is no implicit AMD target.

For Metal System Trace, the current export supports whole-step and command-buffer evidence. It does not expose
per-shader dispatch duration, kernel resources, physical bytes, or hardware counters through this capture mode.
Those fields remain explicitly unavailable.

## Extending to another backend

Add a target descriptor and one capability-registry row. Add a runtime command adapter only if the runtime CLI
is new. The rest of the lifecycle—profile resolution, scope validation, normalization, interval binding, and
comparison—stays unchanged.

Existing complete captures can be normalized again with `collect-hw-trace --reuse-capture`; this is useful when
normalization changes and avoids coupling evidence reproduction to another GPU run.
