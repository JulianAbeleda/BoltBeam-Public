# Metal Qwen3-8B fixed-depth-128 replay baseline

MR0 is complete. It reuses BoltBeam's fixed-depth decode authority and portable
trace contracts; it does not alter tinygrad runtime code or infer replay causes.

Run the entries in `interleaved-run-plan.json` in listed order. Each runtime gets
one unmeasured warmup, then five one-token fixed-depth-128 samples. Keep the same
model file, model hash, target, power state, and cache state for every row. Stop
and mark the bundle `INCONCLUSIVE` if output identity, model identity, target,
temperature/thermal state, or an individual command fails; never replace the
20260729 historical trace.

Five clean-process samples per runtime completed in strict serial order against
tinygrad EXP `8139f6725c2e96f66a76515ad6ae5dc50354243c` and llama.cpp
`4f0e43da6f8f6e9390d88409610098ec2d2dc5c7`. Median depth-128 decode was
11.259 tok/s (88.815 ms) for tinygrad Metal and 20.136 tok/s (49.663 ms) for
llama.cpp Metal. Tinygrad reached 55.9% of llama.cpp throughput.

Every accepted tinygrad artifact reports `tinygrad_device=METAL`,
`runtime_type=MetalDevice`, the same generated-token evidence hash, and the
same SDPA route. The word `PYTHON` seen during load refers to the GGUF/DISK file
backend, not the compute device.

Two development attempts are retained below `runs/invalid-*` and excluded from
all summaries: one overlapped both runtimes on the GPU, and one used an
uncommitted MR1 checkout. Full `.trace`/xctrace packages, model copies, and cache
directories remain local and ignored.

See `baseline-summary.json` for the compact authority and
`llama-samples.json` for the accepted llama rows.
