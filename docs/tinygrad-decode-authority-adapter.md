# Tinygrad decode authority adapter

`boltbeam.adapters.tinygrad_decode` ingests the JSON written by tinygrad's
`extra/qk/decode/decode_runtime_overhead.py`. It is CPU/static-only: it imports neither
tinygrad nor model code and executes no GPU work.

The adapter accepts only `tinygrad.decode.fixed_depth.v2`, artifact version 2. It rejects
model-identity disagreement, non-one-token fixed-depth lifecycle rows, non-identical route
sequences, missing route program counts, and non-identical generated repetitions. Its output
is `boltbeam.timing_trace.v1`, with two whole-step rows per source row: `W` (production
generate-item/token) and `D` (same-model JIT with final sync diagnostic). `D` retains the
source interpretation and is never called an upper bound.

Each row includes context, fixed depth, route sequence, selected-route programs-per-token,
and `counter_evidence.status = unavailable`. The authority artifacts expose neither a hardware
counter stream nor a measured host-sync split, so the adapter does not manufacture counters.

`boltbeam.decode_decay.decode_depth_slope` compares only matched ctx512/4096 pairs. For the
provided fixtures, the measured wall-time slopes are approximately: 8B W `0.201 us/context`
and D `0.735 us/context`; 14B W `0.530 us/context` and D `1.125 us/context`. Both pairs retain
the `flash` route and constant programs-per-token (8B: 1021; 14B: 1133). These are pairwise,
single-repetition observations, not uncertainty-bounded regressions. A 14B ctx128 artifact is
present but is not used to prove a separate three-point fit; no 8B ctx128 artifact is supplied.

## G-control audit

The supplied authority records do **not** carry an explicit `G=4`/`G=5` identifier, selected
route variant, build setting, or compiled-program identity. `route = flash` proves only the
generic selected route named by this runner. Therefore the depth report marks the generation
control as unavailable for both 8B and 14B; it must not call the 8B pair G=4 or the 14B pair G=5
without a separate route-control authority artifact.
