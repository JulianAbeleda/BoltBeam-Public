# MR7 evidence pipeline

MR7 input JSON is generated, not hand-authored. The pipeline derives exact
isolated execution requests and complete graph/direct enclosure requests from a
finalized MR4 replay summary and MR5 census. It never divides a graph duration
among graph members.

The one-command hardware path, once the Tinygrad provider is ready, is:

```sh
MODEL_SHA256=$(shasum -a 256 /absolute/path/to/model.gguf | awk '{print $1}')
python -m boltbeam.cli mr7-evidence-run \
  /absolute/path/to/mr4-summary.json \
  /absolute/path/to/mr4/rows/control-01/graph-admission-census.json \
  --mr0 /absolute/path/to/mr0.json \
  --mr6 /absolute/path/to/mr6.json \
  --arm control \
  --provider-command "[\"/absolute/path/to/tinygrad/.venv/bin/python\",\"/absolute/path/to/tinygrad/extra/llm_research/mr7_provider.py\",\"--model\",\"/absolute/path/to/model.gguf\",\"--model-sha256\",\"${MODEL_SHA256}\",\"--census\",\"/absolute/path/to/mr4/rows/control-01/graph-admission-census.json\",\"--fixed-depth\",\"128\",\"--samples\",\"5\"]" \
  --out-dir /absolute/path/to/fresh-mr7-evidence
```

The output directory is required to be absent. It contains `census.json`,
`measurements.json`, `whole-step.json`, `enclosures.json`, and
`prerequisites.json`, plus the immutable plan, raw provider-result envelope,
`authority.json`, and a hash manifest. `authority.json` retains the exact MR0,
MR4, MR5, MR6, plan, and provider-response inputs needed to reproduce the
ranking. Run the manifest's `ranker_argv` without editing those files:

```sh
python -m boltbeam.search.role_cost_ranking \
  --authority /absolute/path/to/fresh-mr7-evidence/authority.json \
  --out /absolute/path/to/fresh-mr7-ranking.json
```

For offline or remote collection, the same path is separable without changing
the data authority:

```sh
python -m boltbeam.cli mr7-evidence-plan /absolute/path/to/mr4/summary.json \
  /absolute/path/to/mr4/rows/control-01/graph-admission-census.json --arm control --out /absolute/path/to/mr7-plan.json
python -m boltbeam.cli mr7-provider-run /absolute/path/to/mr7-plan.json \
  --provider-command "[\"/absolute/path/to/tinygrad/.venv/bin/python\",\"/absolute/path/to/tinygrad/extra/llm_research/mr7_provider.py\",\"--model\",\"/absolute/path/to/model.gguf\",\"--model-sha256\",\"${MODEL_SHA256}\",\"--census\",\"/absolute/path/to/mr4/rows/control-01/graph-admission-census.json\",\"--fixed-depth\",\"128\",\"--samples\",\"5\"]" \
  --out /absolute/path/to/mr7-provider-results.json
python -m boltbeam.cli mr7-evidence-bundle /absolute/path/to/mr7-plan.json /absolute/path/to/mr7-provider-results.json \
  --mr4-summary /absolute/path/to/mr4/summary.json \
  --census /absolute/path/to/mr4/rows/control-01/graph-admission-census.json \
  --mr0 /absolute/path/to/mr0/summary.json --mr6 /absolute/path/to/reviewed-mr6.json \
  --out-dir /absolute/path/to/fresh-mr7-bundle
```

The real provider is
`tinygrad-arkey-exp/extra/llm_research/mr7_provider.py`. It loads and captures
the selected GGUF once, verifies that its live graph census is byte-for-byte
equivalent to the finalized MR5 census, then serves every request from the same
prepared JIT, model buffers, graph cache, and timing session. `search_provider`
is intentionally not used here: it owns one semantic packed-matmul candidate
and cannot represent a fused production call or a complete graph enclosure.
The evidence runner also derives `METAL_HYBRID_REPLAY=0/1` from the selected
MR4 arm; it is not an ambient caller choice.

The provider receives `boltbeam.mr7_provider.v1` JSON requests on stdin and
emits exactly one response per request. Isolated results preserve the full
fused execution identity, pass correctness, reproduce source and binary hashes,
and retain at least five raw timing samples. Enclosure results must preserve the
complete outer ID and exact member-call list and retain at least five complete
outer-duration samples. Plan, request, artifact, provider command, and provider
executable/script hashes are retained and checked before ranking.
The resulting ranking embeds the authority packet. MR9 reconstructs the MR7
plan, provider-derived measurements and enclosures, prerequisites, and ranking
before it accepts the MR8 selection.
