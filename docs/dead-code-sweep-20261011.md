# Dead-code sweep, 2026-10-11

This follows the cleanup through `152e238` (main) / `f308024` (dev), after all review items landed.
The same main/dev worktree pair was reused. No GPU, remote machine or fork development was involved.
The fork was inspected read-only at `f9e245c4e809e95282431130e2848c33c85b4ee3`.

## Evidence and removals

Vulture at 60% and 80% confidence covered main and dev, including dev tests. An AST import graph covered
289 main modules and 298 dev modules; apparent orphans were checked against entrypoints, subprocess
module strings, CLI registration, docs and fork references. Coverage of the dev suite before removal
was 26,958 / 32,098 statements (83.99%); 1,912 tests and 27 subtests passed, with 10 skipped.
Coverage is supporting evidence, not proof that hardware or subprocess code is dead.

Line references below are against the pre-sweep tree. Each code batch has its own commit and full gates.

| Batch | Removed | Evidence |
| --- | --- | --- |
| collectors (`ce00663`) | `collectors/llama_ncu.py` (155 raw lines), including `LlamaNcuCapture`, `_resolve`, `_bench_args`, `_parse_bench_json`, `capture_llama_ncu`; `gpu_telemetry.py:33` `OTHER_REASONS` | No importer, CLI registration, module entrypoint, test caller, documented invocation or fork consumer; the collector has 0/102 covered statements. The telemetry map has no read. Historical audit prose is retained. |
| search (`f15cdda`) | `search/util.py` (45 lines): `GROUPINGS`, `grammar_max_candidates`, `ledger_candidate_ids`, and their path constants | No product, tool or fork caller. Only the ledger reader's own dev test imports the module; that single test and import were removed. The active campaign uses its canonical ledger implementation. |
| core (`7545aaf`) | `core/canonical.py:53` `write_artifact` (6 lines) | No call in main, dev, tools or fork; body unexecuted under coverage. `pretty_json`, hashing and canonical serialization remain live. Historical migration prose is not a live call. |
| report (`b782536`) | `report/html.py:43–53` `PLAIN_STATUS`, `PLAIN_NEED`, `PLAIN_BUCKET`, `PLAIN_REGIME`; `:301` `_code` | The page reads none of these maps or the helper. Only a dev test compared the maps with Go; those four comparisons were removed, while checks of the three rendered vocabularies remain. The helper body has no coverage. |
| final protected-file batch | `profile/gguf.py:92` `read_gguf`; `search/role_compare.py:1173` `VERDICTS`; `workflow/tie_out.py:60` `ROLE_SOURCE_IN_MODEL`; `search/role_space.py:139,143` dead `here` assignments and repeated `installed = None`; `workflow/chips.py:223` `here_words`; `workflow/timing.py:10` `TIMING_ARTIFACTS` | No callers or readers after checking both branches, Go seam, docs and fork. Both removed wrapper bodies are unexecuted under coverage. Model binding already comes from the explicit provider metadata, so the discarded family-default maps never reach a candidate or verdict. |

## Kept after checking reachability

- `boltbeam.cli:main` in `pyproject.toml`, the CLI module registry/`CORE`, `python -m boltbeam.cli`,
  `python -m boltbeam.workflow.screen`, and Go seam calls are entrypoints. The 543 argument declarations
  were scanned; all 22 without a direct attribute read are read through `getattr` or forwarded collector
  fields. No CLI flag was proven dead.
- BubbleBeam → FutureSight → semantic campaign is reached from `role_compare`. The fork re-exports
  BubbleBeam/FutureSight and consumes the flash candidate schema. These are not superseded copies.
  BubbleBeam/FutureSight have 96%/97% coverage; timer, weight ledger and tie-out have 95%/94%/98%.
- `where_token_goes` is used by screen results. Prefill/session collectors are selected by workload;
  their runtime modules are launched as subprocesses. Their low direct coverage is not deletion evidence.
  Prefill/session workflow modules have 96%/93% coverage.
- All 14 files in `boltbeam/data` were traced. Thirteen have runtime readers (including the fork's
  kernel-authority consumer). `nv_decode_seed_ledger.jsonl` is a documented historical evidence fixture,
  checked by `test_nv_decode_seed_ledger.py`; it is not an active runtime seed and remains as a record.
  Schemas and route-policy JSON/hash assets are also read by path, so absence of Python import edges
  does not make them unused.
- `policy/descriptor.py` and `diagnostics/packed_quant_control.py` retain standalone module entrypoints.
  AMD/runtime bridges and the decode/prefill/session drivers retain subprocess or external integration
  callers. Decorated diagnosis rules are registered indirectly. Dataclass fields are serialized;
  enum members and declared diagnostic vocabularies remain public contract data.
- Dev-only primitive/MMQ/P9 modules and the test-only experiment/reachability/diagnostic helpers remain
  qualification apparatus. Their retained purpose is documented in the prior apparatus audit, and their
  tests exercise the contracts. Hardware qualification was not run or used to justify a deletion.
- Accepted but unused compatibility arguments (`probe_quant`, `compare_runs`, `candidate_pct`, and the
  old installed-geometry parameters) retain their call signatures. Removing them would change callable
  interfaces, which this sweep does not authorize as an NFC change.

## Validation

The sweep reduces the `sz.py` budget from **46,584 to 46,402 on main** and **47,604 to 47,422 on dev**
(182 code lines). The cap and six duplicate-authority checks are unchanged; the allowlist is empty.
Every batch runs the full dev Python suite, Go vet/tests, and selfcheck before its guarded landing and
mirror publication. The sweep suite has 1,911 tests plus 27 subtests, with 10 skipped; main has
74 Go tests and dev has 76. One test was removed solely because its entire unused utility was removed.

The existing saved run `qwen3-8b-apple_m3_10c-005` was read before the sweep and after every batch with
`python -m boltbeam.workflow.screen results --run …`; `cmp` checks the complete JSON bytes unchanged.
The baseline run is external to the repository and was not edited. Static inventories, import graphs,
coverage JSON, byte snapshots and gate logs are kept with the external cleanup report.
