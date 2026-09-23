# Qwen3-8B Metal graph-admission census

This is the MR1 zero-unknown census for the fixed-depth-128 SDPA rollout JIT.
It uses tinygrad EXP `f30fedcb18d46dfcaf8a9f0dd4a817c65a4a6563`, model SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`,
and the same Apple M4 Metal workload as the repeated MR0 baseline.

The capture reconciles all 803 logical programs:

- 726 programs entered 24 Metal ICB graph batches;
- 67 programs were rejected because their shared base-buffer byte offset was
  greater than `0xffffffff`;
- 10 otherwise admitted programs became direct singleton calls between those
  rejected programs;
- zero calls have an unknown reason and zero graph constructors failed.

The first offset rejection occurs at logical call 668. The 67 rejected offsets
span 4,303,883,584 through 4,999,471,936 bytes and share one run-local base
allocation identity. This proves the uint32 Metal ICB binding boundary is the
structural cause of the observed 24-batch/77-direct decomposition. It does not
prove that replay compaction will materially reduce the 88 ms whole step.

All 803 metadata arrays are currently empty. Program-name families are useful
only for reconciling the historical trace (45 `r_`, 32 `E_` direct calls); they
must not be parsed into semantic role authority. MR5 remains required before
role ranking or generated schedule search.

Replay with:

```sh
cd /Users/julianabeleda/env/tinygrad-arkey-exp
DEV=METAL PYTHONPATH=. .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 128 --nmeas 1 --reps 1 --warmup-decode 2 \
  --out /tmp/metal-census-authority.json \
  --graph-admission-out /tmp/metal-graph-admission-census.json
```

`summary.json` is the compact conclusion. `graph-admission-census.json` is the
complete per-call authority. `authority.json` proves the matched Metal workload
and output identity.
