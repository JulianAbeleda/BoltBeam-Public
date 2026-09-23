# EXP Metal replay and generated-search runbook

This is the start-to-finish Qwen3-8B Q4_K_M workflow for Apple Metal. It runs
only from tinygrad `exp` and BoltBeam. It does not use AMD/eGPU hardware and it
does not promote `exp` to `dev` or `master`. Promotion is a separate review.

The commands below consume real files from earlier stages. A missing packet is
a blocker, not an invitation to create a dummy JSON document. Benchmark values,
statuses, hashes, and policy verdicts must come from the named producer.

## 1. Pin the inputs

Use clean checkouts and a fresh evidence directory. Record full commits for
BoltBeam, tinygrad EXP, and llama.cpp, plus the exact model digest. The formal
MR0 and MR4 protocols reject dirty repositories and recheck identity between
rows.

```sh
git -C /path/to/BoltBeam status --short
git -C /path/to/BoltBeam rev-parse HEAD
git -C /path/to/tinygrad-arkey-exp status --short
git -C /path/to/tinygrad-arkey-exp rev-parse HEAD
git -C /path/to/llama.cpp status --short
git -C /path/to/llama.cpp rev-parse HEAD
shasum -a 256 /path/to/Qwen3-8B-Q4_K_M.gguf
MODEL_SHA256=$(shasum -a 256 /path/to/Qwen3-8B-Q4_K_M.gguf | awk '{print $1}')
```

Do not reuse an existing output directory. Preserve earlier evidence as
historical evidence, including inconclusive runs.

## 2. MR0: matched generic baseline

From the clean BoltBeam checkout:

```sh
.venv/bin/python -m boltbeam.cli matched-control \
  --model /path/to/Qwen3-8B-Q4_K_M.gguf \
  --tinygrad-root /path/to/tinygrad-arkey-exp \
  --llama-root /path/to/llama.cpp \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  --llama-server /path/to/llama.cpp/build/bin/llama-server \
  --output-root /path/to/fresh-evidence/mr0 \
  --target-id apple_m4_10c --depth 128 --warmups 2 --samples 5 \
  --minimum-free-memory-percent 10 --run-id metal-qwen3-8b-mr0
```

Continue only if the emitted summary is valid, prompt/prelude/generated token
identity agrees across runtimes, and every tinygrad census reconciles. MR1 and
MR5 use the emitted graph-admission census; do not reconstruct those identities
from labels or split graph duration evenly.

## 3. MR2-MR4: replay decision and paired replay measurement

MR2's safe design record and MR3's implementation record are required inputs.
After correctness tests and exact-token smoke pass on the pinned revision, run:

```sh
.venv/bin/python -m boltbeam.cli replay-ab \
  --model /path/to/Qwen3-8B-Q4_K_M.gguf \
  --tinygrad-root /path/to/tinygrad-arkey-exp \
  --output-root /path/to/fresh-evidence/mr4 \
  --target-id apple_m4_10c --depth 128 --warmups 2 --samples 5 \
  --minimum-free-memory-percent 10 --run-id metal-qwen3-8b-mr4
```

The MR4 summary is authoritative for `REPLAY_WIN`, `REPLAY_NEUTRAL`,
`REPLAY_REFUTED`, or `INCONCLUSIVE`. Do not hand-edit the classification.

## 4. MR5-MR7: semantic census and measured role ranking

MR6 is the reviewed AMD-concept transfer matrix; it transfers only portable
concepts, never AMD route IDs, ISA, wave assumptions, or defaults. With the real
MR4 summary and its exact census, run the persistent MR7 pipeline:

```sh
.venv/bin/python -m boltbeam.cli mr7-evidence-run \
  /path/to/fresh-evidence/mr4/summary.json \
  /path/to/fresh-evidence/mr4/rows/control-01/graph-admission-census.json \
  --mr0 /path/to/fresh-evidence/mr0/summary.json \
  --mr6 /path/to/reviewed-mr6-transfer-matrix.json \
  --arm control \
  --provider-command "[\"/path/to/tinygrad-arkey-exp/.venv/bin/python\",\"/path/to/tinygrad-arkey-exp/extra/llm_research/mr7_provider.py\",\"--model\",\"/path/to/Qwen3-8B-Q4_K_M.gguf\",\"--model-sha256\",\"${MODEL_SHA256}\",\"--census\",\"/path/to/fresh-evidence/mr4/rows/control-01/graph-admission-census.json\",\"--fixed-depth\",\"128\",\"--samples\",\"5\"]" \
  --out-dir /path/to/fresh-evidence/mr7
```

Use the ranker command written into the MR7 bundle; it consumes the generated
`authority.json`, not separately copied or edited measurement files. The
resulting ranking embeds that authority so MR9 can reconstruct the MR7 plan,
provider-derived measurements/enclosures, prerequisite hashes, and ranking
before search. A role is eligible only if the resulting complete packet
demonstrates its exact identity and whole-step importance. Freeze no more than
two role families.

## 5. MR8-MR9: finite portable search

Export the raw semantic request from tinygrad, export the deterministic
population from BoltBeam, assess that exact population with FutureSight, then
run the finite campaign. The lifecycle and schemas are detailed in
[semantic-campaign.md](semantic-campaign.md), and the one-shot MR9 command in
[mr9-semantic-search.md](mr9-semantic-search.md).

Join only the completed MR7 ranking to population requests whose content hashes,
model hash, compiler revision, target facts, and selected roles agree:

```sh
.venv/bin/python -m boltbeam.cli mr8-population-selection \
  /path/to/fresh-evidence/mr7/ranking.json \
  /path/to/fresh-evidence/mr8/request.json \
  --out /path/to/fresh-evidence/mr8/selection.json

.venv/bin/python -m boltbeam.cli mr9-semantic-search \
  /path/to/fresh-evidence/mr8/selection.json \
  --provider-command '["/path/to/tinygrad-arkey-exp/.venv/bin/python","/path/to/tinygrad-arkey-exp/extra/llm_research/search_provider.py","--backend","METAL"]' \
  --provider-revision "$(git -C /path/to/tinygrad-arkey-exp rev-parse HEAD)" \
  --boltbeam-revision "$(git rev-parse HEAD)" \
  --finalist-repeats 2 --minimum-win 0.03 --maximum-relative-mad 0.05 \
  --out /path/to/fresh-evidence/mr9/result.json
```

MR9 must measure the complete population and emit status `COMPLETE`. Each role
decision is either `MACHINE_WINNER` or `REFUTED_NO_MATERIAL_WIN`; any blocked or
unknown decision keeps the packet out of MR13. Partial results cannot select a
candidate.

## 6. Conditional MR10-MR11

Run MR10 and MR11 only when the real MR9 packet says `winner`. Attach the exact
immutable selected plan through the existing generated-plan boundary, verify a
complete route census, then measure depth 128 and depth 512 with at least seven
valid samples per arm. The resulting packets must include correctness, route and
binary identity, working-set facts, whole-step measurements, and the measured
roofline or residual-gap decomposition.

When MR9 says `refuted`, omit MR10, selected-plan, and route-census packets and
record explicit omission reasons at MR13. Never manufacture winner-only files.

## 7. MR12: regression and duplicate-policy audit

After the implementation and documentation commits are made, run the focused
and available regression suites and retain their real output as the test-summary
input. Then, from a clean BoltBeam checkout, create the static structural packet:

```sh
.venv/bin/python -m boltbeam.cli mr12-static-audit \
  --tinygrad-root /path/to/tinygrad-arkey-exp \
  --out /path/to/fresh-evidence/mr12/static-audit.json
```

This command verifies the canonical route asset, the byte-identical EXP
snapshot, removal of the legacy manifest, absence of duplicate literal route
tables, absence of tinygrad production imports from research code, and clean
recorded revisions. Its packet explicitly claims no hardware execution,
benchmark correctness, or broader test coverage.

## 8. MR13: hash real closure inputs

The closure manifest stores packet identities and exact-byte hashes, not model
files, caches, or full traces. Follow [mr13-closure-manifest.md](mr13-closure-manifest.md).
Pass each real packet's actual schema and accepted status to `--artifact`; do not
copy the illustrative test schemas from unit tests.

The required base inputs are:

| Name | Producer/dependency |
| --- | --- |
| `replay_design` | reviewed MR2 decision record |
| `transfer_matrix` | reviewed MR6 portable-concept matrix |
| `mr9_result` | complete `mr9-semantic-search` output |
| `test_summary` | real MR12 regression-test summary |
| `mr12_static_audit` | passing `mr12-static-audit` output at clean pinned BoltBeam and tinygrad EXP revisions |
| `policy_verdict` | result-based keep/remove/refute decision |

An MR9 winner additionally requires real `mr10_result`, `selected_plan`, and
`route_census` packets. A refutation must omit all three with explicit reasons.
For `mr9_result`, pass schema `boltbeam.mr9_semantic_search.v1` and accepted
status `COMPLETE`; MR13 derives winner versus refutation from the canonical
per-role decisions. For `mr12_static_audit`, pass schema
`boltbeam.mr12_static_audit.v1` and accepted status `pass`. Read every other
schema and status from its real producer output rather than inventing values.
Closure is complete only when the manifest validates at the recorded clean
revisions and the documented generic path can be rerun without source edits.
