# Commands

All commands run from their indicated checkout. Write each compact result to
`runs/<id>/`; retain xctrace output outside this bundle.

```sh
# verify the immutable input before every row
shasum -a 256 /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf

# tinygrad fixed-depth authority (one row; substitute the listed run id)
cd /Users/julianabeleda/env/tinygrad-arkey-exp
PYTHONPATH=. DEV=METAL .venv/bin/python extra/llm_research/decode/decode_runtime_overhead.py \
  --model /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf --ckpts 128 --nmeas 1 --reps 1 --warmup-decode 2 \
  --out /Users/julianabeleda/env/BoltBeam/bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-01/authority.json

# llama fixed-depth row; preserve the compact JSON fields in llama-samples.json
cd /Users/julianabeleda/env/llama.cpp
./build/bin/llama-bench \
  -m /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf -p 0 -n 1 -d 128 -ngl 99 -r 1 -o json

# Run exactly one process at a time in interleaved-run-plan.json order. Capture
# optional system traces separately; never treat them as whole-step authority
# or commit the full package.
```
