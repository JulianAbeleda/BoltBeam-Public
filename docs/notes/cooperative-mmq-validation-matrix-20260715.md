# Cooperative-MMQ validation matrix (14B)

This is the promotion gate for Q4/Q6 cooperative-MMQ work on
`Qwen3-14B-Q4_K_M` / AMD `gfx1100`. It is intentionally independent of
emitters and route selectors. The evidence producer writes one JSON document;
BoltBeam validates it with:

```bash
COOP_MMQ_MATRIX=/absolute/path/matrix.json \
  pytest -q tests/test_cooperative_mmq_matrix.py
```

The policy checks always run. With no evidence file, the result is explicitly
`BLOCKED` (the test must not turn absence into a skip or pass). When
`COOP_MMQ_MATRIX` is set, the file must exist and pass. Missing candidate rows,
missing gates, hidden fallback, missing context artifacts, or rates below the
target are failures.

## Required matrix

| Candidate | Quant | Required generated identity | Contexts | Throughput floor |
|---|---|---|---|---:|
| `q4` | Q4_K | `decode_q4k_g3_generated` | pp512, pp1024, pp2048, pp4096 | 14.0 tok/s each |
| `q6` | Q6_K | `decode_q6k_coop_generated` | pp512, pp1024, pp2048, pp4096 | 14.0 tok/s each |

Each row must include `correctness.status=PASS`, an independent `reference`,
declared `tolerance`, output hash, and compile/repeated-run determinism;
`route_binding.status=PASS`, `provenance=machine_authored_generated`, exact
executed candidate identity, generated-source hash, binary hash, and route
telemetry; and `resources.status=PASS`, `admitted=true`, a resource trace, and
declared limits. The workload identity and schema are mandatory.

For every context, `throughput_artifacts["512"|"1024"|"2048"|"4096"]`
must be a `PASS` record containing a non-empty `artifact_ref`, non-empty
`raw_samples_tok_s`, and numeric `throughput_tok_s >= 14.0`. A copied summary
number without raw timing-artifact provenance is blocked. `ppN` means the
whole-step prefill measurement at N tokens, not a kernel-only estimate.

## Acceptance before promotion

Promote only when both rows pass all four contexts in the same pinned run:

1. Q4 and Q6 numerical outputs match the independent packed reference at the
   producer's declared tolerance, with compile and repeated-run determinism.
2. The executed program is the exact generated candidate named in the row;
   route telemetry proves candidate identity and reports no fallback.
3. Resource admission passes the target's declared VGPR, SGPR, LDS, scratch,
   workgroup, and occupancy limits; an absent trace is not admission.
4. Each pp512/1024/2048/4096 whole-step measurement is at least 14.0 tok/s,
   using the pinned model, clocks, warmup/sample protocol, and same-session
   comparison recorded by the producer.
5. The artifact is reproducible and contains model/target identity, source and
   binary hashes, candidate hash, route telemetry, correctness result,
   resource trace, and raw timing samples.

Any unmet item is `BLOCKED`, not “candidate” or “not measured”. No selector or
emitter change is authorized by this matrix; a blocker must be resolved in the
producer/evidence path before rerunning promotion.
