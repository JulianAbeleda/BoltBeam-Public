# Pure Prefill Policy Benchmark Protocol

This protocol defines comparable authority runs for the Qwen3-8B pure prefill
route. Run each policy in a fresh subprocess with the same model, clock pin,
warmups, rounds, and context sweep. Do not compare DEBUG traces or untimed
profile sums against these wall authorities.

## Common command

From the Tinygrad checkout:

```sh
PYTHONPATH=. DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 \
python3 extra/qk/bench.py --prefill \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --prefill-K 8 --prefill-warmups 4 --prefill-rounds 3 \
  --prefill-whole-lengths 512,1024,2048,4096 --pin-clock
```

The emitted `prefill-whole-synced/latest.json` is the wall-time authority.
Required common fields are `K=8`, `warmups=4`, `rounds=3`, `pin_clock.ok=true`,
`measurement_regime.regime_id=generated_pure`, `route_attribution.prefill_route_pure=true`,
and `route_attribution.prefill_route_rolled_back=false`.

## Policy matrix

| Policy | Candidate-set input | Required route assertion | Expected census |
|---|---|---|---|
| gate/up-only | anchor `ffn_gate_up` candidate | generated pure route; no rollback | one selected exact identity, role `ffn_gate_up` |
| all-four | `multirole-buffer2-candidate-set-v1/candidate-set.json` | generated pure route; no rollback | exactly four selected identities: `ffn_gate_up`, `ffn_down`, `attn_qo`, `attn_kv`; no missing/unexpected/mismatch |
| S9 | S9 manifest/environment | S9 route identifier and pure/rollback fields recorded explicitly | S9 candidate identity and route family; never label as all-four |

Candidate-set runs must also record `candidate_set_route_census.passed=true`.
An absent census is not evidence that a candidate was unused. For S9, retain the
existing S9 artifact and do not merge its timing into the generated-pure series.

## Comparison rules

Compare `whole_tok_s` and `chunk_ms["0"]` only when all common fields match.
Retain route/census JSON beside the timing artifact, and report quality-gate
status separately from performance. A trace with missing quality metadata may be
used as a performance observation but cannot be promoted as a complete authority.

Typed pipe fixtures use `status=typed_host_only` and
`provenance=compiler_owned_typed_pipe_ir` until Tinygrad supplies execution,
binary, correctness, and resource evidence. BoltBeam does not infer execution
from a valid schema or host-side IR.
