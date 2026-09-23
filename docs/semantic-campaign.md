# Semantic campaign

Run the two-phase finite exact GGUF campaign with:

1. `boltbeam propose-semantic-dimensions spec.json --describe describe.json --out request.json` (BubbleBeam)
2. `boltbeam export-semantic-population request.json --out population.json`
3. `boltbeam assess-semantic-population population.json [--preferences preferences.json] --out futuresight.json` (FutureSight)
4. `boltbeam semantic-campaign request.json --futuresight-evidence futuresight.json --provider-command '["python3","extra/llm_research/search_provider.py","--backend","METAL"]' --out ledger.json`

Step 4 without `--futuresight-evidence` runs the population unordered and unfiltered.

`request.json` contains the exact semantic workload, the proposed dimensions and coupled rows, the compiler facts they were proposed against, the resolved target, GGUF path, execution regime, and finite budget. BoltBeam never imports Tinygrad; the provider process receives the original exact workload/path on every staged request. The output is the canonical full-kernel ledger, including rejected and blocked rows. Evidence hashes that do not match the exported population or coupled rows fail closed.

For the bounded MR8-to-MR9 path, including persistent finalist repetition and
the winner/refutation verdict, use [the MR9 semantic-search command](mr9-semantic-search.md).

## Proposal spec (`propose-semantic-dimensions`)

The request's fields, minus what BubbleBeam writes, plus two inputs. Missing or unknown fields fail closed.

| field | meaning |
| --- | --- |
| `axis_choices` | `{path: [values]}` to filter into `dimensions` |
| `coupled_rows` | `[{path: value}]` to split into `legal_coupled_rows` / `rejected_coupled_rows` |
| `semantic_workload`, `schedule`, `resolved_target`, `gguf_path`, `execution`, `request_id`, `run_id`, `timestamp`, `budget` | copied to the request |

The request gets `futuresight_evidence: null`; evidence arrives in step 4.

## Target facts

BubbleBeam (`boltbeam/search/bubblebeam.py`) proposes; FutureSight (`boltbeam/search/futuresight.py`)
rejects and orders. `boltbeam.search.futuresight_adapter.target_facts(target_id, describe)` builds the
`compiler_facts` mapping both read. Each key is looked up under its own name in the registry row
(`boltbeam/data/targets.json`: fields, then `capabilities`) and in the provider `describe` result (its
`target` object, then its top level). Sources that carry a key must agree. A key no source carries stays
missing; `propose-semantic-dimensions` names it on stderr.

| key | read by | registry today | describe |
| --- | --- | --- | --- |
| `subgroup_size` | flash legality and priority (required there; not ported yet) | all targets: 32 | NV `target` |
| `max_threads_per_threadgroup` | proposal and static legality | none | `target` |
| `max_threadgroup_memory_bytes` | proposal, static and flash legality, flash priority | none (`lds_bytes_per_cu` is per CU, not per threadgroup) | `target` |
| `compiler_transforms` | proposal and static legality | none | top level |
| `supported_plan_kinds` | proposal and static legality | none | top level |
| `generic_control_plan_kind` | static legality | none | none |

A missing `compiler_transforms` is not neutral: FutureSight then rejects every non-empty transform list.
A missing `generic_control_plan_kind` loses nothing, because the candidate schema already refuses a
`tinygrad_heuristic.v1` control with transforms. Without `--describe`, the provider facts are the
resolved target's recorded `observed_facts`.

## FutureSight evidence (`--futuresight-evidence`)

Schema: `schemas/futuresight_evidence.schema.json`. `assess-semantic-population` writes all five keys and
checks the result with the campaign's own binder before writing.

| key | required | consumed by |
| --- | --- | --- |
| `assessment_version` | no | `bind_futuresight_evidence`: must be `bubblebeam.futuresight.static.v1` |
| `assessments[]` `{candidate_hash, static_score, static_reason}` | yes | binder: exact keys, known hash, int score; campaign: `execution_order` by score descending, then hash; copied to the ledger's `futuresight_static_evidence` |
| `rejections[]` `{candidate_hash, reason}` | yes | binder: known hash; campaign: never executed, ledger row `REJECTED_STATIC` (stage `futuresight_static`) with the reason, ordered last; the heuristic control cannot be rejected |
| `rejections[]` `{row_hash}` | (same list) | binder: must name a legal coupled row |
| `population_hash` | no | binder: must equal `population_hash(candidates, legal_coupled_rows)` |
| `rejected_coupled_rows[]` `{row_hash, row, reason}` | no | campaign: hash must match the row and the request's rejected row and reason; copied to `futuresight_rejected_coupled_rows` |

Every candidate hash appears exactly once across `assessments` and `rejections`.
