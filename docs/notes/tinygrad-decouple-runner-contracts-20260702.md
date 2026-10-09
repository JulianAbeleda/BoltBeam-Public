# Tinygrad Decouple Runner Contracts

Source audit: `/home/ubuntu/tinygrad-arkey/bench/qk-repo-principles-cleanup/boltbeam_boundary_audit.json`

Boundary plan: `docs/tinygrad-boundary-plan-20260702.md`

Scope: the `tinygrad_runner_adapter` and `split_tests` lanes only. These files are mixed-boundary files: tinygrad should keep runtime, kernel, compiler, hardware, and regression execution; BoltBeam should own policy, search, candidate generation, artifact interpretation, promotion decisions, roofline/economic attribution, and ledgers.

## Shared JSON Seam

Every migrated runner/test seam should be a JSON-only exchange with no Python import from BoltBeam into tinygrad and no tinygrad runtime import into BoltBeam.

Tinygrad runner input:

```json
{
  "schema": "boltbeam.runner_request.v1",
  "request_id": "stable-id",
  "runner": "tinygrad.extra_or_test_path",
  "model": {"id": "qwen3_8b_q4_k_m", "path_ref": "local-or-configured"},
  "candidate": {"id": "candidate-id", "route_family": "family", "params": {}},
  "runtime": {"device": "AMD", "env": {}, "contexts": [], "repeats": null},
  "artifacts": {"read": [], "write_dir": "bench/..."}
}
```

Tinygrad runner output:

```json
{
  "schema": "tinygrad.runner_evidence.v1",
  "request_id": "stable-id",
  "runner": "tinygrad.extra_or_test_path",
  "status": "ok|rejected|error",
  "runtime_fingerprint": {"git": "...", "device": "...", "driver": "...", "env": {}},
  "evidence": {"route": {}, "correctness": {}, "timing": {}, "isa": {}, "artifacts": []},
  "errors": []
}
```

BoltBeam decision output consumed by tinygrad:

```json
{
  "schema": "boltbeam.runtime_policy.v1",
  "policy_id": "stable-policy-id",
  "selected_routes": [],
  "route_params": {},
  "rollback": {},
  "promotion": {"status": "promoted|rejected|experimental", "reason": "..."},
  "provenance": {"evidence_refs": [], "ledger_refs": []}
}
```

Tinygrad may consume only the final `boltbeam.runtime_policy.v1` policy, never BoltBeam search code.

## Runner Adapter Contracts

| File | Tinygrad Keeps | BoltBeam Owns | JSON Contract Seam |
| --- | --- | --- | --- |
| `extra/amd_isa_g3_weight_promotion_gate.py` | Re-run G3 generated route under tinygrad, route-fire checks, token/W==D timing evidence, hardware/runtime capture. | Promotion contract, eligible role/shape list, pass/fail bar, ledger update, default-vs-rollback decision. | Request candidate `q4k_g3_lanemap_generated` plus eligible shapes and contexts; evidence returns per-context route counts, token match, lag, timing, and leaked-route diagnostics; decision returns promoted/rollback route params. |
| `extra/amd_isa_reg_accum_lds_reclaim_audit.py` | Compile native tile variants and expose ISA/resource measurements for accumulator/LDS behavior. | Interpret reclaim opportunity, decide whether a register/LDS search lane exists, record audit result. | Request compile groups/flags; evidence returns VGPR/SGPR/LDS/scratch and sink breakdown rows; decision returns candidate families or no-op audit verdict. |
| `extra/amd_isa_weight_path_route_attribution.py` | Eager tinygrad profiling, per-kernel GPU timestamps, runtime route capture for weight path. | Role taxonomy, wall-share attribution, route gap conclusions, follow-up candidate priorities. | Request model/context/route flags; evidence returns per-kernel timings and raw names; BoltBeam attaches roles, route classes, summaries, and decisions. |
| `extra/q8_ffn_fast_artifact_probe.py` | Build/run HIP/HCQ buffers, copyin/copyout, correctness and timing probes against tinygrad/ROCm artifacts. | Artifact promotion, comparison policy, accepted/rejected Q8 FFN route decision. | Request artifact id, tensor shape, kernel source variant, correctness tolerances; evidence returns timings, output diffs, compiler/runtime fingerprints, and produced artifact paths. |
| `extra/q8_ffn_hcq_artifact.py` | HCQ buffer allocation, handwritten Q8/Q4 kernel execution, source compilation, raw artifact production. | Artifact schema, storage policy, route acceptance, ledger ownership. | Request HCQ kernel variant and tensor metadata; evidence returns kernel binary/source hashes, launch params, timing, correctness, and memory footprint. |
| `extra/qk_amdgpu_isa_primitive_audit.py` | Locate/unbundle code objects, disassemble via ROCm tools, count ISA primitives/resources. | Primitive taxonomy, ISA rejection policy, audit verdicts and reports. | Request code-object refs and primitive patterns; evidence returns symbol, target, VGPR/SGPR/LDS/scratch, instruction counts, and flags; decision returns allowed/rejected primitive classification. |
| `extra/qk_artifact_cache.py` | Runtime fingerprint collection for tinygrad execution artifacts when runtime evidence is produced. | Cache key policy, artifact reuse rules, stale-artifact acceptance, storage/index ownership. | Evidence embeds `tinygrad.runner_evidence.v1.runtime_fingerprint`; BoltBeam emits cache metadata `{kind, inputs_hash, code_hash, runtime_hash, cache_key, reuse_status}`. |
| `extra/qk_candidate_template_gen.py` | Nothing runtime-critical except optional validation of runner binding names that still point at tinygrad runners. | Template expansion, candidate schema, policy metadata, pruning labels, generated candidate registries. | BoltBeam emits `boltbeam.candidate_set.v1` containing candidates and runner binding ids; tinygrad consumes only selected runner requests, not generator code. |
| `extra/qk_decode_attention_generated_pv_kernel_audit.py` | TinyJit capture, generated PV route execution, debug/program-name capture, raw wall timing. | PV-kernel bottleneck conclusion, promotion/nonpromotion interpretation, report. | Request generated attention mode/context/env; evidence returns captured programs, per-kernel wall rows, token/correctness status, and artifact refs. |
| `extra/qk_decode_attention_generated_wall_audit.py` | Tinygrad route/program capture for generated attention arms, raw wall timing, clean W==D artifact lookup. | Compare arms, attribute generated-route wall cost, decide next search or rollback. | Request arm matrix and contexts; evidence returns per-arm route names, timing rows, W==D refs, and failures; decision returns wall-attribution verdict. |
| current generated-route parity runners | Model load, TinyJit capture, route-fire/materialization/correctness/ISA/W==D checks. | Candidate queue, hard-gate definitions, promotion authority, result ranking. | Request one route policy and oracle refs; evidence returns ordered gate results, first reject reason, route names, ISA audit evidence, tokens, and synced W==D timing. |
| BoltBeam candidate batch runner | Spawn tinygrad runner subprocesses for bounded route-policy candidates and freeze runtime evidence. | Generate candidate list, prune/rank, remember/refute, decide promotion. | BoltBeam sends `boltbeam.candidate_set.v1`; tinygrad returns `tinygrad.search_gate_batch.v1` with per-candidate evidence; BoltBeam returns ranked decisions. |
| `extra/qk_harness_contract.py` | Tinygrad-side provenance helpers needed by runners: git/runtime/hardware stamp and child env construction. | Performance-claim contract vocabulary, required fields, comparator policy, ledger links. | Runner evidence includes a `harness_stamp` block; BoltBeam validates it against `boltbeam.harness_contract.v1` and records missing fields. |
| `extra/qk_lanemap_template_audit.py` | Tinygrad codegen/lowering reconstruction for LaneMap kernels and proof that emitted code matches tinygrad primitives. | Lane-map template schema, eligible-role policy, speed artifact interpretation, generalization verdict. | Request template params and eligible shapes; evidence returns reconstructed kernel descriptors, code hashes, static resource facts, and speed artifact refs. |
| `extra/qk_large_model_decode_route_gap_audit.py` | Faithful tinygrad model/shape route census and route classifier grounded in runtime model shapes. | Large-model gap interpretation, candidate priorities, baseline comparison, report. | Request model id/path and route policy; evidence returns per-linear role/shape/quant/selected-route rows and aggregate route counts. |
| `extra/qk_large_shape_knob_reachability_audit.py` | Probe which tinygrad/codegen knobs materially affect generated topology. | Decide which grammar axes are real search dimensions and which are decorative. | Request knob grid; evidence returns before/after topology keys and reachability rows; BoltBeam emits `boltbeam.search_space.v1` axes. |
| `extra/qk_large_shape_topology_space_audit.py` | Enumerate legal tinygrad topology instances for concrete large model shapes. | Search-space bounds, candidate ranking inputs, topology-space report. | Request shape set and grammar version; evidence returns legal topology rows and counts; BoltBeam emits bounded candidate space. |
| `extra/qk_lifecycle_search_loop.py` | Execute accepted candidates through tinygrad decode evaluator and return runtime artifacts. | Lifecycle loop orchestration, pruning rules, candidate schemas, ranking, refutations, ledger proposals. | BoltBeam sends accepted `candidate_set` plus eval binding ids; tinygrad returns eval run artifacts; BoltBeam returns `lifecycle_decision.v1`. |
| `extra/qk_pathology_artifact_g5.py` | Patch/intercept tinygrad AMD program compilation and capture compiler pathology evidence for G5 block tile. | Pathology schema interpretation, root-cause report, search/mitigation decision. | Request target kernel/model/context; evidence returns compiled kernels, source/ISA/resource hashes, static analysis, and pathology artifact path. |
| `extra/qk_prefill_pipe_role_profile.py` | Profile prefill roles under tinygrad on AMD and emit raw per-role effective TFLOPS/timing. | Role-level attribution, scheduler policy, promotion/report conclusions. | Request prefill arm/env/model/chunk; evidence returns role timings, top kernels, pipe flag, and runtime fingerprint. |
| `extra/qk_split_kv_economics_audit.py` | Only validate and normalize tinygrad-measured attribution/W==D artifact references when needed. | Economics model, split-KV promotion bar, derived bandwidth/wall-share conclusions. | BoltBeam consumes attribution evidence `{routed_per_ctx, tile_ms, combine_ms, route_fired}` and W==D refs; decision emits split-KV economic verdict and required next runner request. |
| `extra/qk_system_fusion_sf0_audit.py` | Eager ordered-event capture, per-kernel profiling, tinygrad model execution under selected contexts. | Fragmentation taxonomy, unknown-ceiling threshold, fusion opportunity ranking/report. | Request model/contexts/steps/env; evidence returns ordered events, per-kernel timings, route names, roles if tinygrad-known, and raw unknown rows. |
| `extra/qk_tg_p14_combine_reopen.py` | Numeric correctness and directional micro-timing for split-preserving combine implementation. | Reopen decision, promotion gate definition, final W==D requirement and report. | Request combine variant/synthetic dimensions; evidence returns numeric diff, microtiming, and structural notes; decision returns reopen/pass/fail with next full-run requirement. |
| `extra/qk_tg_p8_geometry_search.py` | Execute generated 8B attention geometry arms under tinygrad, token-match, route-bound, per-kernel wall split. | Geometry grid selection, promotion threshold, candidate choice, report. | BoltBeam sends L-grid/contexts/promotion bar; tinygrad returns per-L evidence with tokens, route flags, tok/s, wall split; BoltBeam returns selected geometry or rejection. |
| `extra/tinygrad_runtime_boundary_audit.py` | Static/live audit of tinygrad runtime server routes, streaming, busy policy, and JSON error behavior. | Client/runtime separation policy, leakage interpretation, boundary roadmap/report. | Request optional base URL plus static audit mode; evidence returns route capability checks, live HTTP responses, and leakage terms; BoltBeam records boundary verdict. |

## Split Test Contracts

| File | Tinygrad Keeps | BoltBeam Owns | JSON Contract Seam |
| --- | --- | --- | --- |
| `test/external/fixtures/qk_policy_min.json` | Minimal runtime-loadable fixture shape for generated policy parsing tests. | Canonical policy fixture generation and full policy schema validation. | Fixture becomes a checked `boltbeam.runtime_policy.v1` sample; tinygrad tests keep only fields needed to prove runtime parser behavior. |
| `test/external/test_qk_demote_search.py` | Runtime regression tests for applying an already accepted demotion/storage policy, with synthetic no-GPU evidence only if needed. | Demotion search orchestration, quality budget, accepted-policy artifact emission, provenance hygiene. | BoltBeam tests own `boltbeam.demote_search_result.v1`; tinygrad tests consume the resulting runtime policy fixture and assert loader/application behavior. |
| `test/external/test_qk_flash_decode_policy.py` | `should_use_flash_decode` runtime behavior, env override precedence, variant dispatch error handling, default runtime variant smoke checks. | Threshold search, accepted flash policy, portability/provenance assertions, promotion rationale. | BoltBeam emits flash threshold policy `{ctx_range, threshold_ctx, selected_variant}`; tinygrad tests assert policy consumption and route selection. |
| `test/external/test_qk_flash_search.py` | No search logic, except a small runtime-policy fixture can be loaded to guard compatibility. | Flash threshold sweep, crossover detection, accepted-policy JSON/Markdown artifacts, path/provenance hygiene. | BoltBeam owns `boltbeam.flash_search_result.v1`; tinygrad consumes only `boltbeam.runtime_policy.v1` derived from accepted threshold. |
| `test/external/test_qk_generated_policy_runtime.py` | Loading generated policy, shape/tensor indexing, storage budget enforcement, Q6K effective mode, shared packed view behavior. | Generation of policy entries, storage/search policy selection, schema evolution beyond runtime-required fields. | BoltBeam emits generated policy fixture; tinygrad parser test asserts accepted fields and rejects incompatible `kind`/version/conflicts. |
| `test/external/test_qk_search_spec.py` | No tinygrad runtime obligation except compatibility with runner request ids if needed. | Search enums, constraints, row assembly, accepted-policy IO, generated-policy conversion. | Move to BoltBeam as `boltbeam.search_spec.v1` tests; tinygrad keeps only JSON compatibility tests for `runner_request` and `runtime_policy` samples. |
| `test/unit/test_qk_route_purity.py` | Runtime route-policy loader/selectors and proof that absent policy selects nothing; tinygrad default-path regression if it is purely runtime. | Manifest validation, default purity report, route provenance taxonomy, census construction and final-default-allowed decisions. | BoltBeam emits `boltbeam.route_manifest.v1` and `boltbeam.runtime_policy.v1`; tinygrad tests load policy fixtures and assert route selection by shape/tensor only. |

## Counts

- Runner adapter contracts: 25
- Split test contracts: 7
- Total contracts: 32
