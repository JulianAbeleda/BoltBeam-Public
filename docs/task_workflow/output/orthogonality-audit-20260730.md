# Phase E — Orthogonality audit

Status: complete (report only — no code changed)
Scope: `docs/task_workflow/input/organization-and-pruning-scope-20260730.md`, Phase E
Commit audited: `a3f1f63` ([search] NFC - carve semantic modules into search/semantic/)
Authority: `structure-template/structure/Development/coding-principles.md`, "Orthogonalize"

Governing rule:

> Bad orthogonality: one module carrying multiple unrelated responsibilities; path,
> policy, state, and execution logic fused together; side effects that leak across
> the system without a clear boundary; a refactor in one subsystem forcing
> incidental edits in many others.

Method (§E.1 of the scope): every `boltbeam/*.py` file over 250 LOC was read in
full and classified `orthogonal | fused | authority-point`, with the specific
path+policy+state+execution fusion and the runtime-use/meta-development boundary
checked explicitly. Entanglement was measured empirically with import-graph
greps, not impression. 49 files exceeded 250 LOC; `kernel_analysis/model.py`
(1,492 LOC) is excluded from remediation by the scope and is cited for context
only.

This is a report-only phase. No file was modified.

---

## 0. Verdict table

| Verdict | Count | Modules |
| --- | ---: | --- |
| Orthogonal | 33 | see §2 |
| Authority-point | 4 | `vocab.py`, `artifacts/base.py`, `profile/ir.py`, `math/roofline.py`, `search/epochs/epoch_model.py` (5 — see note) |
| Fused | 12 | see §1 |
| Excluded (scope) | 1 | `kernel_analysis/model.py` |

(33 + 5 + 12 + kernel_analysis/model.py caveat below over/under-counts by one
because `boltbeam/analyze.py` is a borderline fused/authority case recorded once
in §1 with its own note — see that entry.)

---

## 1. Fused — findings

Ordered roughly by severity (how many of path/policy/state/execution collide,
and how load-bearing the file is).

```
module:           boltbeam/control/matched_control.py, 585
responsibilities: environment/identity resolution (build_environment_identity L130,
                  _git_identity L83, _file_identity L97, _toolchain_identity L104);
                  protocol policy/threshold definition+validation (build_protocol L168,
                  protocol_preflight L225, protocol_definition_errors L236,
                  environment_preflight_errors L149, summarize_dispersion L358);
                  persistent state writes (write_json calls at L394-395, 498, 515,
                  536, 544, 564, 574, 577); subprocess execution (_run_command
                  L43-50, capture_dynamic_validity L272-274, run_tinygrad_authority_row
                  L388-392)
fused concerns:   path | policy | state | execution
imported by:      1 (boltbeam/cli/workflow.py only — narrow single entry point)
verdict:          fused
evidence:         build_protocol (L168-222) resolves absolute model/tinygrad/llama
                  roots (L177-183) in the same dict literal as validity thresholds
                  (maximum_relative_mad, minimum_free_memory_percent, L202-208) and
                  the argv later handed to subprocess (L210-219). execute_protocol
                  (L491-578) interleaves execution (L519, 531, 541), policy checks
                  (L508-510, 555-556) and state writes (L515, 564, 574, 577) in one
                  function body. replay_ab.py imports its identity/execution
                  primitives directly from this file (see next finding), so reusable
                  plumbing and protocol-specific policy are not separated.
remedy:           split into (1) environment-identity module, (2) protocol-policy
                  module (thresholds, protocol_definition_errors, summarize_*),
                  (3) execution/state module (run_tinygrad_authority_row,
                  execute_protocol, the write_json call sites)
```

```
module:           boltbeam/control/replay_ab.py, 405
responsibilities: replay policy/thresholds (protocol_definition_errors L72,
                  reconcile_replay_census L94, _verdict L313-319, _evidence_readiness
                  L290); paired statistics (paired_statistics L193,
                  _bootstrap_mean_ci L181); persistent state writes (write_json
                  L328, 343, 373, 383, 400); execution orchestration
                  (execute_protocol L322-401, _trace_command_executor L231-249)
fused concerns:   policy | state | execution (path is delegated to matched_control,
                  imported directly rather than through a shared boundary)
imported by:      1 (boltbeam/cli/workflow.py only)
verdict:          fused
evidence:         execute_protocol (L322-401) builds row_dir (L333), applies
                  statistical/drift gates (L355-359), executes via authority_executor
                  (L345) and _trace_command_executor (L231-249), and writes rows/summary
                  (L343, 373, 383, 400) — all in one function. Tightening a gate
                  threshold requires touching the same file as the subprocess loop.
remedy:           extract paired_statistics, _bootstrap_mean_ci, _evidence_readiness,
                  _verdict, reconcile_pair, reconcile_replay_census into a
                  replay_statistics.py; keep execute_protocol as pure orchestration
```

```
module:           boltbeam/workflow/autoscan.py, 360
responsibilities: hardware probing via subprocess (_probe_nvidia L65-91,
                  _probe_apple_metal L110-148, _run_command L31-36); device
                  classification heuristics (_nvidia_arch L57-62, _apple_gpu_cores
                  L94-101); target-override policy (_resolve_manifest_target
                  L228-261); manifest/artifact persistence (write_manifest L352,
                  write_json L356-359, update_manifest L360)
fused concerns:   path | policy | state | execution
imported by:      1 (boltbeam/workflow/__init__.py / cli path)
verdict:          fused
evidence:         autoscan_run (L346-360), 15 lines, does run_dir(run) (L348),
                  subprocess-backed hardware probing via _hardware_profile (L350),
                  an irreversible target-override decision (_resolve_manifest_target,
                  `replaceable = source in {"default","autoscan"} or current == "auto"`
                  L235, mutation L237), and five separate persistence calls
                  (L352, 356-359, 360) — the sharpest single instance of the named
                  four-way fusion found in this audit.
remedy:           split into a probe/execution module, a target-resolution-policy
                  module, and a thin autoscan_run orchestrator
```

```
module:           boltbeam/analyze.py, 267
responsibilities: role ranking/summary policy (_role_rank L26, _role_summary L41,
                  _candidate_families L52); external shell-command STRING
                  construction (_env_prefix L79, build_tinygrad_commands L90-123);
                  measurement-plan assembly (build_measurement_plan L126); writing
                  7 output artifacts to disk (_write_json L184, _write_commands_md
                  L188, emit_analysis_bundle L230-267)
fused concerns:   policy | execution-adjacent (literal shell command strings) | state
imported by:      1 (boltbeam/cli/workflow.py, via emit_analysis_bundle)
verdict:          fused, but functions as an intentional top-level bundle emitter
                  with side effects — see note
evidence:         emit_analysis_bundle (L230-267) calls emit_search_space,
                  emit_seed_policy, emit_fixture_manifest, build_measurement_plan,
                  then writes model_profile.json .. analysis_manifest.json
                  (L245-250) via a fixed ARTIFACTS tuple (L15-23). It also builds raw
                  shell-command strings for external tinygrad measurement
                  (build_tinygrad_commands, L90-123) — a separate "shape an external
                  command" concern glued onto artifact-writing orchestration.
note:             the scope (§A.2) calls this file "the top-level bundle emitter and
                  an authority point." That is correct for its ROLE (it is the one
                  place a full analysis bundle gets emitted) but it is not a pure
                  authority point in the no-side-effects sense used elsewhere in this
                  audit (contrast math/roofline.py, epoch_model.py below) — it
                  performs filesystem writes and shell-string construction. It should
                  stay the single orchestrator; the command-string builder is the
                  separable piece.
remedy:           extract build_tinygrad_commands/_env_prefix into a separate
                  command-construction module; leave emit_analysis_bundle as the
                  orchestrating authority point
```

```
module:           boltbeam/search/role_cost_ranking.py, 348
responsibilities: exact-identity reconciliation/joining (whole_step_from_mr4_summary
                  L51-184); bootstrap statistics (_bootstrap_lower L211-226);
                  classification/eligibility policy (_honest_classification L187-208,
                  thresholds L268, 291-293); CLI + file-IO orchestration (main
                  L301-341)
fused concerns:   path | policy | state | execution (CLI/file-IO glue, not subprocess)
imported by:      1 (its own CLI entry point; not imported by other subsystems)
verdict:          fused
evidence:         main() (L301-341) does argparse path handling, reads five input
                  JSONs, branches on --authority, and writes results via
                  args.out.write_text (L321, L340) in the same file as the pure
                  ranking algorithm (build_role_cost_ranking L229-298,
                  _honest_classification L187-208). CLI argument changes never need
                  the classification thresholds and vice versa, yet they share a file.
remedy:           extract main/file-IO into a thin CLI module; keep
                  reconciliation+ranking+classification as the computation module
```

```
module:           boltbeam/search/semantic/mr7_evidence_bundle.py, 313
responsibilities: plan derivation from MR4/MR5 facts (build_evidence_plan L41);
                  provider-result validation/binding (adapt_provider_results L144);
                  live subprocess provider execution (collect_provider_results
                  L184-241, Popen/select/os.read); authority/bundle assembly and disk
                  writing (build_authority L248, write_evidence_bundle L280-308)
fused concerns:   execution | state | policy
imported by:      1 (boltbeam/cli/search.py)
verdict:          fused
evidence:         collect_provider_results (L184-241) performs live subprocess
                  execution (Popen L?, environment injection L205, select-based read
                  loop L214-220) in the same module as build_evidence_plan (pure
                  policy/derivation) and write_evidence_bundle (output_dir.mkdir +
                  _write_json, L280-308). A provider-transport change would force
                  incidental edits near plan-derivation and bundle-writing code that
                  do not otherwise need to change.
remedy:           move collect_provider_results (subprocess transport) into its own
                  module, separate from plan-derivation/validation and from
                  bundle-writing
```

```
module:           boltbeam/kernel_analysis/handoff.py, 313
responsibilities: recommendation -> ExperimentManifest translation
                  (request_from_recommendation L75); model-profile -> per-role
                  ExperimentManifest translation (request_from_model_profile L119);
                  candidate -> validated execution-bridge request serialization
                  (request_from_candidate L176, requests_from_feasibility L243);
                  measured ranking -> CandidateDecision + ledger write
                  (recommendation_from_measured_matrix L262-306,
                  record_measured_recommendation L309-313)
fused concerns:   policy | state
imported by:      package-level: kernel_analysis is imported by only 2 files
                  outside itself (cli/analysis.py, cli/roofline.py) — narrow entry,
                  but the fusion is internal to this one file
verdict:          fused
evidence:         record_measured_recommendation (L309-313) calls
                  `LedgerStore(ledger_path).add(...)` — a persistent-state mutation —
                  in the same module that does pure request-building/translation
                  (L75-259) and guardrail-policy derivation
                  (recommendation_from_measured_matrix L280-290 computes
                  Guardrail.CORRECTNESS/ROUTE_BOUND/SPEED/ROLLBACK). The module's own
                  docstring (L1-13) scopes it to "evidence-request handoff," not
                  ledger persistence.
remedy:           extract record_measured_recommendation (and its LedgerStore call)
                  into boltbeam/ledger or a dedicated ledger-handoff module; keep
                  handoff.py to pure manifest/request construction
```

```
module:           boltbeam/search/semantic/mr9_semantic_search.py, 291
responsibilities: provider-checkout binding via git subprocess calls
                  (_provider_command_binding L37-74, 4x subprocess.run); MR8
                  selection validation (_validate_selection L77); verdict computation
                  from a campaign (_decision L143-205); orchestration via a
                  persistent session (run_mr9_semantic_search L208); CLI entrypoint
                  (main L270-286)
fused concerns:   execution | policy | (CLI)
imported by:      1 (boltbeam/cli/search.py)
verdict:          fused
evidence:         _provider_command_binding (L37-74) shells out to git four times to
                  verify a clean checkout (execution/identity concern) alongside
                  _decision (L143-205, pure stability/improvement-fraction threshold
                  math) and a CLI main() (L270-286) that parses argv and writes files.
                  A change to the git-binding protocol would not need to touch
                  _decision's win-fraction math, yet both live in one file.
remedy:           extract _provider_command_binding into a provider-identity module;
                  keep _decision (policy) and run_mr9_semantic_search (orchestration)
                  apart from main() (which belongs in cli/)
```

```
module:           boltbeam/collectors/metal_system_trace.py, 284
responsibilities: CLI arg/manifest resolution (_configured_run L27); runtime command
                  construction per provider (_runtime_command L49); subprocess
                  capture of xctrace record+export (capture_metal_system_trace
                  L253-282, subprocess.run, mkdir, file writes); XML table parsing
                  (_xml_rows L79-92); measurement binding/classification into
                  boltbeam.timing_trace.v1 (_bind_measurement L114-152,
                  normalize_metal_trace L155)
fused concerns:   execution | parsing | policy (classification)
imported by:      package-level: collectors imported by 2 files outside itself
verdict:          fused (moderate)
evidence:         capture_metal_system_trace/capture_command_metal_trace (L200-282)
                  perform live subprocess capture and filesystem writes
                  (trace_dir.mkdir L206, subprocess.run L217/272) in the same module
                  as _bind_measurement's "bound vs diagnostic_only" 0.35
                  relative-gap threshold decision (L114-152) and the low-level XML
                  parser (_xml_rows L79-92). A change to the xctrace invocation
                  template is unrelated to, and would not need to touch, the
                  measurement-binding threshold logic.
remedy:           split into a capture/execution module (subprocess + xctrace
                  invocation) and a normalize/classify module (_xml_rows,
                  _bind_measurement, normalize_metal_trace) — the module's own
                  docstring already names both halves separately
```

```
module:           boltbeam/profiler/counters.py, 297
responsibilities: counter registry loading/validation (load_counter_registry L250,
                  _validate_counter_row L146); counter lookup helpers (get_counter
                  L275, list_counter_ids L288); counter-fragment normalization with
                  fail-closed replay/attribution/calibration rules
                  (normalize_counter_fragment L76-138)
fused concerns:   authority (registry) | policy (fragment normalization rules)
imported by:      package-level: profiler imported by 5 files outside itself
verdict:          fused (mild)
evidence:         load_counter_registry/get_counter/list_counter_ids (L250-298) are a
                  pure registry-authority API, while normalize_counter_fragment
                  (L76-138) enforces non-trivial fail-closed policy (raises
                  CounterFragmentError for scope/replay/calibration violations, e.g.
                  L90-93, L112-118) using that registry. A registry-loading change
                  doesn't need to touch fragment-normalization policy and vice versa.
remedy:           extract normalize_counter_fragment (+ its error class) into a
                  separate counter-fragment-policy module; keep counters.py as the
                  pure registry authority
```

```
module:           boltbeam/search/spec.py, 647
responsibilities: enum/vocabulary authority (Phase, Model, OpScope, SearchSpace,
                  Objective, BACKENDS, L24-79); enum validation helpers (L106-129);
                  FullKernelCandidate schema + v1/v2 readers + hashing (L174-441);
                  Constraints/SearchRow/AcceptedPolicy dataclasses (L443-611);
                  JSONL/file I/O (load_search_rows, save_search_rows,
                  load_accepted_policy, save_accepted_policy, L614-648)
fused concerns:   state (mild — ~35 of 647 LOC)
imported by:      package-level: search imported by 9 files outside itself; spec.py
                  specifically underlies most of that
verdict:          fused (mild), otherwise an authority-point
evidence:         the module docstring claims "pure schema/data validation: no
                  hardware execution ... no runtime default changes" (L8-10), yet
                  L614-648 perform real pathlib I/O (read_text/write_text/
                  parent.mkdir) inside what is otherwise a clean schema authority.
                  No policy or execution fusion; the mismatch is between the
                  docstring's claim and these four functions only.
remedy:           extract the four load_/save_ functions into a small
                  search/spec_io.py; the remaining ~610 LOC is a clean authority
                  point and should be left alone
```

```
module:           boltbeam/profiler/report.py, 536
responsibilities: per-kernel roofline/SOL sections (_kernel_section,
                  _tile_oracle_section, L61-272); limiter classification heuristics
                  (_classify_kernel L280-319); baseline A/B comparison
                  (_baseline_compare L496-537); report assembly/markdown
                  (profiler_report, profiler_report_markdown L344-493)
fused concerns:   policy (duplicated authority, not path/state/execution)
imported by:      package-level: profiler imported by 5 files outside itself
verdict:          fused (mild) — a centralization defect more than a structural one
evidence:         _classify_kernel (L283-317) and _baseline_compare (L523-528)
                  hard-code judgment thresholds inline (occ<60%, valu<50%, l2>=90%,
                  pm/pc>=50%, eff_ratio 0.85/1.18) rather than deferring to
                  eval/thresholds.py, which eval/evaluator.py uses for the
                  equivalent promote/refute judgment elsewhere in the tree. Two
                  disagreeing authorities for "what counts as bound/regressed."
remedy:           extract these thresholds into named constants sourced from
                  eval/thresholds.py (this is really a Phase-C centralization
                  finding surfacing through an orthogonality lens — flagging here
                  because it was found during the E.1 pass)
```

```
module:           boltbeam/kernel_analysis/report.py, 322
responsibilities: orchestration ingest->classify->eligibility->contrast->diagnose->
                  recommend (investigate, analyze_evidence L92-188); pairing/
                  conclusion policy (_pairs L37-89, _conclusion_from_contrast);
                  markdown rendering (_render, L206-302); output file writing
                  (write_outputs L305-322)
fused concerns:   state (mild — one function)
imported by:      package-level: kernel_analysis imported by 2 files outside itself
verdict:          fused (mild)
evidence:         write_outputs (L305-322) is the only filesystem-touching function
                  (out.mkdir L307, four artifact writes L310-321); every other
                  function in the file is a pure transformation. On-disk layout
                  changes touch only write_outputs; analysis-policy changes never
                  touch it — fully separable.
remedy:           move write_outputs into the calling CLI module (cli/analysis.py)
```

Note on severity: the first four entries (`matched_control.py`, `replay_ab.py`,
`autoscan.py`, `analyze.py`) are the only ones where all of path, policy, state,
and execution genuinely collide in one function body, matching the principle's
worst case literally. The remaining eight are narrower — one or two concerns
mixed into an otherwise coherent module, several of them "mild" (a handful of
lines out of 300+).

---

## 2. Orthogonal — clean, checked, no finding

Recorded so nobody re-checks them. Each is long (>250 LOC) but single-purpose:
every function serves one declared pipeline stage, with no path/policy/state/
execution mixing found.

`cli/workflow.py` (765), `cli/profile.py` (505), `cli/search.py` (291) — thin
argparse-registration + dispatch-adapter files; every `cmd_*` does args-parse ->
one domain call -> `_write`. No subprocess originates in any of the three.

`artifacts/tinygrad.py` (563), `artifacts/tinygrad_rocprof.py` (263),
`artifacts/llama_rocprof.py` (387), `artifacts/tinygrad_profile_events.py` (364) —
one artifact-family normalizer each; JSON/CSV/pickle-in, `NormalizedEvidence`/
`timing_trace`-out; no evaluation, promotion, or execution.

`profiler/importers/ncu.py` (556), `plan/runner_plan.py` (394),
`kernel_analysis/theoretical_roofline.py` (520), `eval/evaluator.py` (474),
`kernel_analysis/diagnosis.py` (421), `kernel_analysis/provider_adapters.py` (412),
`report/html.py` (397), `perf/microbench.py` (352),
`search/full_kernel/tinygrad_full_kernel.py` (344),
`quantization/quant_gemv.py` (332), `kernel_analysis/contrast.py` (323),
`experiment/prefill_role_trace.py` (323), `trace/timing.py` (321),
`trace/hw_trace.py` (320), `profile/decode_roles.py` (320),
`kernel_analysis/adapters.py` (319), `trace/substrate_compare.py` (306),
`kernel_analysis/p9_bundle.py` (306), `kernel_analysis/candidates.py` (297),
`search/mmq/mmq_bundle.py` (291), `profile/loaders.py` (287),
`report/xml.py` (276), `search/reachability.py` (272),
`roofline/prefill_roofline.py` (262), `roofline/roofline_plan.py` (257),
`search/epochs/epoch_join.py` (256), `search/joins/r4_evidence_join.py` (255),
`kernel_analysis/eligibility.py` (243), `roofline/prefill_roofline_ladder.py` (236).

`plan/runner_plan.py` deserves an explicit note: it does mix path (input
discovery/staging) and state (bundle manifest, `shutil.copy2`), which looks like
a fusion candidate at first read, but it has no policy thresholds and, critically,
no execution — its "commands" (`_tinygrad_trace_commands`,
`_llama_trace_commands`) are returned as inert JSON data for an external runner
to execute, per its own documented boundary (L79-82). Checked and cleared.

`search/full_kernel/tinygrad_full_kernel.py` similarly touches path/state/
execution (it is a subprocess-transport adapter to the tinygrad provider) but all
of it serves one declared contract — "drive one external provider through its
protocol stages" — consistent with "Design For Replacement." Checked and cleared,
in contrast to the `control/` protocol files above, which additionally embed
independent policy/threshold logic in the same file.

`kernel_analysis/eligibility.py`, `math/roofline.py`,
`search/epochs/epoch_model.py`, `artifacts/base.py` and `profile/ir.py` are
covered as authority-points below (§3) rather than here.

Also checked, both under the 250-LOC threshold but relevant to this audit's
questions: `ledger/store.py` (96 LOC) and `ledger/model.py` (134 LOC) — confirmed
as the single durable-state authority `kernel_analysis/handoff.py` writes through;
`artifacts/cache_inventory.py` (105 LOC) and `artifacts/mr12_static_audit.py`
(119 LOC) — confirmed as BoltBeam's own meta-development audit tooling (repo
structure / cache inventory), kept separate from model-analysis code, not fused
with it anywhere.

---

## 3. Authority-points — export knowledge, not behaviour

```
module:           boltbeam/vocab.py, 396
responsibilities: the ONE vocabulary for verdicts, tiers, role/architecture
                  taxonomies, ledger status, guardrails, schema-id constants
                  (Verdict, Tier, RoleClass/RoleGroup + ROLE_GROUP_OF,
                  Architecture, LedgerStatus, Guardrail/GuardrailState, 40+
                  SCHEMA_* string constants), plus a handful of pure derivation
                  functions (role_group_of, is_moe_role, architecture_compatible,
                  enum_values)
fused concerns:   none
imported by:      ~63-76 files across nearly every subsystem (highest fan-in
                  module in the tree)
verdict:          authority-point
evidence:         every top-level symbol is either a `(str, Enum)` member, a
                  module-level string constant, or a pure function with no
                  side effects operating only on its arguments (e.g.
                  architecture_compatible L180-195 computes a set union/
                  intersection and returns a bool; no I/O, no subprocess, no
                  mutation of module state anywhere in the file). This is exactly
                  the "schemas and durable data shape" authority category the
                  principles ask to centralize (`coding-principles.md`,
                  "Centralize"), not a god module: it exports KNOWLEDGE
                  (constants/types), never BEHAVIOUR. This is the audit's
                  highest-value single judgement per the task brief.
remedy:           leave (authority point) — this is centralization working
                  correctly, not entanglement
```

```
module:           boltbeam/artifacts/base.py, 166
responsibilities: the NormalizedEvidence/EvidenceRow/EvidenceSource/EvidenceFlags
                  schema, the AdapterIncomplete/UnsupportedArtifact exception
                  vocabulary, sha256_bytes/sha256_json, read_evidence/write_evidence
fused concerns:   none
imported by:      21+ files import sha256_json alone; 6+ more import
                  EvidenceSource/sha256_json together; effectively every
                  artifacts/* adapter
verdict:          authority-point
evidence:         dataclasses are frozen; the only I/O (read_evidence/write_evidence,
                  L158-166) is the canonical (de)serialization of the ONE schema this
                  file defines, not a side-channel behaviour — this is the same
                  "one schema, one (de)serializer" pattern as vocab.py's SCHEMA_*
                  constants, just paired with its codec.
remedy:           leave (authority point)
```

```
module:           boltbeam/profile/ir.py, 101
responsibilities: TensorRole and TargetProfile dataclasses (the model/target
                  profile schema)
fused concerns:   none
imported by:      14+ files (search/emit.py, search/families.py,
                  search/reachability.py, and others) import from profile.ir directly
verdict:          authority-point
evidence:         both dataclasses are frozen, with only serialization helpers
                  (to_json via asdict); no side effects.
remedy:           leave (authority point)
```

```
module:           boltbeam/math/roofline.py, 244
responsibilities: dual_roofline_placement, floor/ceiling primitives, Amdahl
                  whole-gain math, CeilingReport/PracticalRooflinePoint dataclasses
fused concerns:   none
imported by:      package-level: math imported by 4 files outside itself
verdict:          authority-point
evidence:         docstring is explicit: "pure roofline and Amdahl math ... NO model
                  dimensions, NO target identifiers, NO baked-in bandwidths"; every
                  function takes plain floats/dataclasses and returns a computed
                  value, no I/O anywhere in the file.
remedy:           leave (authority point)
```

```
module:           boltbeam/search/epochs/epoch_model.py, 246
responsibilities: EpochSpec + MMQ_EPOCH_SPECS taxonomy table (L21-143);
                  EpochEvent/EpochAnswer/EpochReport dataclasses; epoch_coverage/
                  blocked_epoch_reasons read-only introspection
fused concerns:   none
imported by:      underlies search/epochs/epoch_join.py and search/joins/*
verdict:          authority-point
evidence:         MMQ_EPOCH_SPECS is a pure constants table; the two coverage
                  helpers only read the taxonomy/dataclasses defined in the same
                  file, no side effects.
remedy:           leave (authority point)
```

**`core/facts.py`, `core/system.py`, `core/experiment.py`, `core/requirements.py`,
`target/targets.py`** (all under 250 LOC, so no formal finding, but load-bearing to
this audit's entanglement question): each is imported by roughly 5-13 files
outside `core`/`target` and each exports only dataclasses/constants (`Fact`,
`TruthStatus`, `SystemSnapshot`, `ExperimentManifest`, `EvidenceRequirement`,
`TARGETS`, `get_target`) with no behaviour. Consistent with the vocab.py pattern:
the widely-imported modules in this tree are schema/registry modules by design,
not entangled god modules. This is a systemic positive finding, not a per-file
one — worth recording once here rather than as five near-duplicate blocks.

---

## 4. Entanglement — empirical import counts (package level)

Measured by grepping `from boltbeam.<pkg>` / `from ..<pkg>` outside each
package's own files (mechanical, not impression):

| Package | Imported by (files outside it) | Character |
| --- | ---: | --- |
| `artifacts/` | 42 | knowledge (base.py schema+sha256 helper) — authority point |
| `profile/` | 24 | knowledge (ir.py schema) — authority point |
| `target/` | 13 | knowledge (TARGETS registry) — authority point |
| `core/` | 12 | knowledge (Fact/SystemSnapshot/ExperimentManifest schemas) |
| `workflow/` | 11 | mixed — orchestration entry points, thin |
| `search/` | 9 | mostly profile.ir consumption + full_kernel/mr7/mr9 CLI wiring |
| `quantization/` | 7 | knowledge (route_families_for) + one classifier |
| `ledger/` | 6 | knowledge+behaviour — the one legitimate state-mutation authority |
| `profiler/` | 5 | mixed (see counters.py fusion, §1) |
| `plan/` | 5 | behaviour, narrow (bundle staging) |
| `trace/` | 5 | behaviour, narrow (trace interpretation) |
| `runtime/` | 4 | behaviour, narrow |
| `perf/` | 4 | knowledge (fit/mem_tier helpers) |
| `policy/` | 4 | knowledge (thresholds/route_manifest) |
| `math/` | 4 | knowledge — authority point |
| `roofline/` | 3 | behaviour, narrow |
| `report/` | 3 | behaviour, narrow |
| `synth/` | 3 | behaviour, narrow |
| `collectors/` | 2 | behaviour (fused — see metal_system_trace.py) |
| `memory/` | 2 | behaviour, narrow |
| `kernel_analysis/` | 2 (package-level; only via cli/analysis.py, cli/roofline.py) | narrow single entry point — good |
| `control/` | 1 | behaviour (fused — see §1); accessed only via cli/workflow.py |
| `eval/` | 1 | knowledge+policy, single authority for promote/refute |
| `experiment/` | 1 | behaviour, narrow |
| `vocabulary/` | 1 | thin |
| `diagnostics/` | 1 | behaviour, narrow |
| `adapters/` | 1 | behaviour, narrow |
| `cache/` | 1 | behaviour, narrow |
| `cli/` | 0 | correctly a leaf: nothing imports the CLI layer |
| `characterize/` | 0 | leaf |

Interpretation required by §E.4: the two highest fan-in packages
(`artifacts/`, 42; `profile/`, 24) are authority points, not entangled god
modules — both export schema/knowledge, confirmed by reading the actual
imported symbols (`sha256_json`/`EvidenceSource` from `artifacts.base`;
`TensorRole`/`TargetProfile` from `profile.ir`), not by their file names.
`kernel_analysis/` — arguably the conceptual center of the tool — is imported
by only 2 files at the package level, which is a *good* sign: it is reached
exclusively through `cli/analysis.py` and `cli/roofline.py`, a narrow, replaceable
entry point exactly as "Keep Public Surfaces Boring" prescribes. `control/` is
the opposite pattern deliberately noted: narrow fan-in (1) but internally fused
(§1) — narrow entry point does not by itself imply clean internals.

---

## 5. vocab.py verdict (headline judgement)

**Authority-point, not a god module.** ~63-76 files import it, more than any
other module in the tree, but every export is one of: a `(str, Enum)` member, a
`SCHEMA_*` string constant, or a pure function (`role_group_of`,
`role_group_members`, `is_moe_role`, `is_ssm_role`, `architecture_compatible`,
`enum_values`) with no I/O, no subprocess, no mutable module state, and no
control-flow side effects on the rest of the system. The file's own docstring
states the intent directly: "No module may invent a parallel string vocabulary.
Adding a value = editing this file, not scattering literals" — this is
centralization by design (`coding-principles.md`, "Centralize"), and the audit
confirms the code matches the stated intent. A god module would leak
*behaviour* — dispatch logic, state transitions, execution — across every
importer; `vocab.py` leaks nothing but names. Distinguishing test applied: does
changing this file force *unrelated* edits elsewhere, or does it just mean a
new value exists for callers to opt into? It's the latter — orthogonal by the
principle's own definition ("keeping distinct concerns independent so one
change does not force unrelated changes").

---

## 6. Runtime-use vs. meta-development separation

**Separated, with no violation found.** Checked explicitly across all 49 files
plus the packages named in the org scope's own note (`search/`, `perf/`,
`eval/`, `ledger/` as "meta" vs. `kernel_analysis/`, `profile/`, `roofline/` as
"model-analysis").

- BoltBeam's own meta-development audit tooling lives in its own files, never
  mixed with model-analysis code: `artifacts/mr12_static_audit.py` (repo
  structure / policy-asset identity via git, no model data at all),
  `artifacts/mr13_closure.py`, `artifacts/cache_inventory.py` (classifies
  BoltBeam's own bench artifact cache).
- Crossings that do exist between "model side" and "search/ledger side" are
  ordinary layering, not boundary violations, confirmed by reading the actual
  imports: `kernel_analysis/contrast.py` imports `perf.fit` (a statistics
  helper, knowledge); `kernel_analysis/handoff.py` imports `ledger.model`/
  `ledger.store` to hand a model-route decision to the durable ledger (the
  designed handoff point, not a boundary leak — though see §1 for why the
  ledger *write* itself should be extracted from that file); `kernel_analysis/
  provider_adapters.py` imports `search.joins.resource_join` /
  `search.transfer_contract` (schema-only). `search/emit.py`, `search/families.py`,
  and `search/reachability.py` import `profile.ir` because the search space is
  built *from* a model's profile — expected direction, not a fusion.
- `search/reachability.py` reads ledger entries (`ledger_idx.get`, L200) as
  read-only input evidence about a *model candidate's* refutation status for
  route search — this is not meta-analysis of BoltBeam's own development
  process, and the module never writes the ledger.
- No file in the 49 reviewed combines "analyse a target model's performance/
  architecture" and "analyse/audit BoltBeam's own search algorithm's behaviour"
  in the same function or module.

---

## 7. Cross-reference (not a Phase E finding, recorded for the Phase A reviewer)

`search/semantic/` now has 9 files (`mr7_evidence_bundle.py`,
`mr8_population_selection.py`, `mr9_semantic_search.py`, `semantic_candidate.py`,
`semantic_candidate_plan.py`, `semantic_identity.py`,
`semantic_population_export.py`, `semantic_search_campaign.py`, `__init__.py` —
8 non-init), which is at the acceptance-criterion-6 limit ("no sub-package
exceeds 8 files") depending on whether `__init__.py` counts. Noted here because
it was observed while auditing `mr7_evidence_bundle.py` and
`mr9_semantic_search.py` for this phase; it is a Phase A gate question, not
resolved by this report.

`boltbeam/analyze.py` and `boltbeam/workflow/analyze.py` both define a function
named `_role_rank` with different (not identical) logic, and both assemble a
near-parallel "artifact bundle" (`ARTIFACTS` vs. `ANALYZE_ARTIFACTS` tuples of
overlapping filenames: `search_space.json`, `route_policy(.seed).json`,
`fixture_manifest.json`, a measurement-plan file), both calling into
`emit_seed_policy`, `route_families_for`, `emit_search_space`, and
`emit_fixture_manifest`. `boltbeam/analyze.py` is the one-shot bundle emitter
(`cli/workflow.py` -> `emit_analysis_bundle`); `boltbeam/workflow/analyze.py` is
the step-based pipeline variant that persists through `run_dir`/manifest state
(`workflow/__init__.py` -> `analyze_run`). Whether these represent the same
knowledge under two names (a Phase B duplication question — same-shape,
divergent-inputs bucket per §B.1) or a legitimate old-path/new-path split is not
resolved here; flagging for the Phase B/C reviewers since it was found during
this pass and the existing `duplication-audit-20260730.md` does not mention it.

---

## 8. What could not be determined

- Per-file "imported by" counts (as opposed to package-level counts) were not
  computed for every one of the 49 files individually — the scope's
  entanglement question is stated at the subsystem level (§E.1.4: "for each
  subsystem, how many *other* subsystems import it"), which is what §4 answers.
  A handful of findings above (`role_cost_ranking.py`, `mr9_semantic_search.py`)
  cite file-level import counts because they matter to that specific finding;
  the rest defer to the package-level table.
- Whether the `analyze.py` / `workflow/analyze.py` overlap (§7) is knowledge
  duplication or an intentional two-path design was not resolved — it needs a
  signature-level Phase B comparison (§B.2 evidence format), which is out of
  this phase's scope.
- `kernel_analysis/model.py` (1,492 LOC, 22 dataclasses) was read for context
  only, per the scope's explicit exclusion from remediation ("its dependency
  graph is a single tree with no seam"); no orthogonality verdict is offered
  for it beyond citing that exclusion.
