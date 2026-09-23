# Tinygrad prefill adapter

`boltbeam.adapters.tinygrad_prefill` ports the reusable normalization seam from tinygrad's prefill timing trace.
It accepts serialized route attachment/execution observations and profile kernel events, then emits the existing
`boltbeam.timing_trace.v1` contract. Route status is evidence only: the adapter never selects a route or interprets
policy. Kernel event time is scaled only when a synchronized authority wall time is supplied.

`context` and `depth` are retained on every whole-step and kernel row. Missing kernel profiler data is represented
in `metadata.profiler` rather than silently manufacturing attribution; incomplete counter coverage is likewise
reported as `partial`. Native PMC counters remain provider data and are passed through without GPU interaction.
