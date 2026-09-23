# Full-kernel candidate-set persistence

BoltBeam can persist the admitted two-buffer `qwen3_8b` gate/up schedule as four exact
Tinygrad candidates for `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`:

```sh
boltbeam build-full-kernel-candidate-set proven-gate-up.json --out candidate-set.json
```

The input may be a raw `boltbeam.full_kernel_candidate.v1` payload, a
`canonical_identity`/`payload` wrapper, a search row containing
`full_kernel_candidate`, or the existing `rows[].search_row` manifest. When an input
identity is present, BoltBeam verifies it against the canonical payload hash.

Each output role has an independently derived candidate hash and exact profile, role,
shape, and target applicability. The set has its own content hash and stable serialized
bytes. This command only validates and persists the candidate contract. Compiler
legality, execution, numerical checking, measurement, and admission remain Tinygrad
loader/runner responsibilities.
