# Phase D abstraction audit

Status: complete (report only — no code changed)
Audited: 2026-07-31
Scope: `docs/task_workflow/input/organization-and-pruning-scope-20260730.md`, Phase D (§D.1–D.3)
Authority: `structure-template/structure/Development/coding-principles.md`, "Abstract" +
"Separate Ergonomics From Semantics" + "Design For Replacement"
Audited at commit: `a3f1f6387f737844f8c0018dc2b413aa1552c34d`, working tree clean

This is read-only. No source file was modified, moved, or deleted to produce it.

**Note on concurrency:** two other agents are writing different files under
`docs/task_workflow/output/` concurrently (a duplication audit and possibly a
centralization audit). This report cross-references
`docs/task_workflow/output/duplication-audit-20260730.md` where the same code
was independently found from the DRY lens (§B); Phase D looks at the same
handful of sites through the abstraction/interface lens (§D), so two findings
can legitimately point at the same lines for different reasons. Neither
report edits the other.

Per Phase A having already run (confirmed by reading the tree, not older
docs): `boltbeam/cli/` is a 7-file package (`__init__.py`, `__main__.py`,
`_common.py`, `analysis.py`, `profile.py`, `roofline.py`, `search.py`,
`workflow.py`), 10 new domain packages exist at `boltbeam/` root
(`roofline/ control/ runtime/ trace/ memory/ quantization/ vocabulary/ plan/
experiment/ target/`), and `boltbeam/search/` has 5 sub-packages
(`semantic/ full_kernel/ mmq/ epochs/ joins/`). All findings below are against
that tree.

## 0. Verdict table

| # | Module/symbol | Verdict | Remedy |
| - | --- | --- | --- |
| 1 | `boltbeam/cli/__init__.py` (20 lines, registrar loop) | deep | leave |
| 2 | `boltbeam/cli/_common.py` (8 helpers) | deep | leave |
| 3 | 10 new domain-package `__init__.py` files + 5 `search/*/__init__.py` | deep (empty, no re-export shim) | leave |
| 4 | Inline `json.loads(pathlib.Path(x).read_text())` in 9 call sites across `roofline.py`, `search.py`, `workflow.py`, `profile.py` | shallow-passthrough (a second unofficial way to do what `_load_json` already does) | inline → call `_load_json` |
| 5 | `cmd_import_llama_rocprof` / `cmd_import_tinygrad_rocprof` re-deriving run-manifest defaults inline instead of calling `_run_manifest_defaults` | shallow-passthrough / duplicates-a-rule | inline → call `_run_manifest_defaults(args, "timing_trace.json")` |
| 6 | `cmd_plan_kernel_experiment`'s `--execution-bridge` branch (`boltbeam/cli/analysis.py:143-165`) manually reconstructing `CandidateFactors`/`CandidateSpec`/`FeasibilityResult`/`RejectionReason` from raw JSON | duplicates-a-rule | absorb into `kernel_analysis/candidates.py` as `CandidateSpec.from_json` / `FeasibilityResult.from_json` |
| 7 | `_VERDICT_PRIORITY` + primary-decision tie-break (`boltbeam/cli/workflow.py:31-36,512`) | misleading-name / policy-in-CLI | absorb into `eval/evaluator.py` as e.g. `primary_decision(decisions)` |
| 8 | `boltbeam/kernel_analysis/provider_adapters.py` (412 LOC) vs `adapters.py` (its designated authority layer) | deep — correctly shaped Design-For-Replacement adapter | leave |
| 9 | `adapt_execution_result` / `adapt_hw_trace` inline field validation vs `adapt_resource_trace` / `adapt_isa_manifest` delegating to shared validators | asymmetry, not a violation (no second definition exists to contradict) | optional: extract for symmetry, not required |

Findings by bucket: **deep: 3** (#1, #2, #3, and #8 separately called out below),
**shallow-passthrough: 2** (#4, #5), **duplicates-a-rule: 2** (#6, #7),
**misleading-name: 1** (#7, double-counted — see its own section),
**no violation found: 1** (#9, recorded as a checked-and-clean asymmetry).

**"Second unofficial way" found:** yes, twice — #4 and #5 below are each a
literal second implementation of an existing canonical CLI helper, written
inline in the same file that correctly calls the canonical helper elsewhere.
No new "second unofficial way" was found between the CLI and the *subsystem*
APIs beyond these two (the 68 handlers are otherwise thin pass-throughs, which
is the correct shape per the scope's own note in §A.1).

---

## 1. `boltbeam/cli/__init__.py` — deep, checked clean

```
module/symbol:  boltbeam/cli/__init__.py:1-21
interface:      def main(argv: list[str] | None = None) -> int
implementation: builds one ArgumentParser, calls register(sub) on 5 domain
                modules (analysis, search, profile, roofline, workflow), then
                dispatches to args.fn(args).
verdict:        deep
evidence:       20 lines total; the entire 68-command surface (parsers +
                handlers) lives behind this one function. No `cli/` module
                imports another module's internals — each domain module only
                calls into `_common` and into subsystem packages
                (`boltbeam.kernel_analysis`, `boltbeam.search`, etc.), never
                into a sibling `cli/*.py`. Verified: `grep -n "from boltbeam.cli"
                boltbeam/cli/{analysis,profile,roofline,search,workflow}.py`
                returns no hits.
remedy:         leave
```

## 2. `boltbeam/cli/_common.py` — deep, checked clean

```
module/symbol:  boltbeam/cli/_common.py:1-91 (_write, _profile, _parse_ctxs,
                _parse_route_flags, _run_manifest_defaults, _fail, _load_json,
                _load_evidences, _ensure_ledger_file, _add_common)
interface:      one function per cross-cutting CLI concern (write-or-stdout,
                resolve a model+target pair, parse a comma-list, parse
                KEY=VALUE flags, resolve --run-directory defaults, print to
                stderr + return an exit code, load one JSON file, load one
                evidence file/bundle, touch-create a ledger file, attach the
                4 arguments every model command shares).
implementation: each is 3-15 lines but each also encodes a real invariant a
                caller could otherwise get wrong: `_parse_ctxs` rejects an
                empty list and a malformed int with a `SystemExit` (not a bare
                ValueError leaking a traceback); `_parse_route_flags` rejects
                a missing `=`; `_load_evidences` distinguishes a bundle
                schema from a single evidence file so callers don't have to.
verdict:        deep
evidence:       every helper is called from at least 2 of the 5 domain
                modules (confirmed: `_load_json` 4/5, `_write` 5/5, `_fail`
                4/5, `_add_common` 3/5) and none merely renames a stdlib call
                — `_load_json` is the one exception discussed in §4 below,
                where it is a deep-enough wrapper that other code should be
                using it and mostly does.
remedy:         leave
```

## 3. New domain-package `__init__.py` files — deep (empty), checked clean

```
module/symbol:  boltbeam/{roofline,control,runtime,trace,memory,quantization,
                vocabulary,plan,experiment,target}/__init__.py and
                boltbeam/search/{joins,epochs,mmq,full_kernel,semantic}/__init__.py
interface:      none — every file is 0 bytes.
implementation: none.
verdict:        deep (there is no shallow re-export layer to inline)
evidence:       `wc -l` on all 15 files returns 0 for each. Phase A's package
                carve did not introduce a `from .foo import *` shim anywhere,
                which would have been a classic "vague wrapper that hides
                where authority actually lives." It did not.
remedy:         leave
```

## 4. Inline `json.loads(pathlib.Path(x).read_text())` bypassing `_load_json`

```
module/symbol:  boltbeam/cli/roofline.py:19,32; boltbeam/cli/search.py:111,112,127,142,143;
                boltbeam/cli/profile.py:18; boltbeam/cli/workflow.py:82
interface:      boltbeam.cli._common._load_json(path: str) -> Any
                  "return json.loads(pathlib.Path(path).read_text())"
implementation: 9 call sites in files that already `from boltbeam.cli._common
                import ... _load_json` re-derive the identical expression
                inline instead of calling it (2 of them via a locally
                shadowed `import json` inside the handler body, e.g.
                cmd_roofline and cmd_schedule_trace re-import a module-level
                name that is already imported at file scope).
verdict:        shallow-passthrough (the inline code is a second, unofficial
                way to perform an operation `_load_json` already performs;
                since the wrapper and the inline code have byte-identical
                bodies, the wrapper adds no invariant over the inline form —
                which is exactly why the fix is "always call `_load_json`,"
                not "delete `_load_json`")
evidence:       `_common.py:68`: `return json.loads(pathlib.Path(path).read_text())`.
                `search.py:111`: `request = json.loads(pathlib.Path(args.request).read_text())`.
                `search.py:127`: `result = export_population(json.loads(pathlib.Path(args.request).read_text()))`.
                `roofline.py:19`: `_write(ingest_roofline(json.loads(pathlib.Path(args.trace).read_text()), ...`.
                Same file (`roofline.py`) imports and correctly uses
                `_load_json` at lines 62-66 in `cmd_prefill_roofline`, proving
                this is an omission, not an intentional different rule — a
                JSON file is a JSON file regardless of which command reads
                it.
remedy:         inline: replace each site with `_load_json(...)`.
```

Cross-reference: `docs/task_workflow/output/duplication-audit-20260730.md`
finding #3 documents the same 9 sites from the DRY/knowledge-duplication
lens. This entry records the same evidence from the interface lens: the
canonical loader is a deep, invariant-carrying wrapper (it is the one place
that would centralize "how do we read a JSON artifact from disk" if that ever
needed a policy, e.g. size limits or encoding), and these call sites are an
unofficial parallel path around it.

## 5. `_run_manifest_defaults` reimplemented inline by 2 of 4 sibling trace-import handlers

```
module/symbol:  boltbeam/cli/profile.py:cmd_import_llama_rocprof:42-50,
                cmd_import_tinygrad_rocprof:90-98
                vs. boltbeam/cli/_common.py:_run_manifest_defaults:49-59
interface:      def _run_manifest_defaults(args, default_out: str) -> tuple[dict, str | None]
                  "resolve run_manifest + --out defaults + weight_inventory
                   fallback from --run, if supplied"
implementation: cmd_import_llama_rocprof (profile.py:42-50) and
                cmd_import_tinygrad_rocprof (profile.py:90-98) each open-code
                the identical 8-line body — `run_manifest = {}; out = args.out;
                if args.run: ... run_path = run_dir(args.run); run_manifest =
                load_manifest(run_path); out = out or str(run_path /
                "timing_trace.json"); if args.weight_inventory is None and
                (run_path / "weight_inventory.json").exists(): ...` — with
                the literal string "timing_trace.json" substituted for the
                helper's `default_out` parameter. `cmd_import_hw_trace`
                (profile.py:149) and `cmd_import_ncu` (profile.py:215), two
                handlers in the *same file*, correctly call
                `_run_manifest_defaults(args, "hw_trace.json")`.
verdict:        shallow-passthrough / duplicates-a-rule — this is the
                clearest instance in the tree of "a second unofficial way to
                perform the same operation" the scope explicitly asked to
                hunt for. It sits exactly where §D.2 predicted: between a CLI
                handler and the subsystem/shared-utility layer it should be
                calling.
evidence:       side-by-side, `_common.py:49-59`:
                  run_manifest: dict[str, Any] = {}
                  out = args.out
                  if args.run:
                    from boltbeam.workflow.common import load_manifest, run_dir
                    run_path = run_dir(args.run)
                    run_manifest = load_manifest(run_path)
                    out = out or str(run_path / default_out)
                    if getattr(args, "weight_inventory", None) is None and (run_path / "weight_inventory.json").exists():
                      args.weight_inventory = str(run_path / "weight_inventory.json")
                  return run_manifest, out
                vs. `profile.py:42-50` (cmd_import_llama_rocprof body, verbatim):
                  run_manifest = {}
                  out = args.out
                  if args.run:
                    from boltbeam.workflow.common import load_manifest, run_dir
                    run_path = run_dir(args.run)
                    run_manifest = load_manifest(run_path)
                    out = out or str(run_path / "timing_trace.json")
                    if args.weight_inventory is None and (run_path / "weight_inventory.json").exists():
                      args.weight_inventory = str(run_path / "weight_inventory.json")
                `cmd_import_tinygrad_rocprof:90-98` is the same body again.
                This is not "looks alike but different knowledge" — same
                function, same import, same conditional, same literal
                fallback filename convention, only the constant string
                differs, which is exactly what `default_out` already
                parameterizes.
remedy:         inline: replace both bodies with
                `run_manifest, out = _run_manifest_defaults(args, "timing_trace.json")`.
```

Cross-reference: `duplication-audit-20260730.md` finding #2 covers the same
two sites as a DRY violation ("2 of 4 sibling handlers"). Recorded here
because it is also the single strongest example of the "second unofficial
way" pattern the scope asked Phase D to specifically hunt for.

## 6. `cmd_plan_kernel_experiment`'s manual dataclass reconstruction

```
module/symbol:  boltbeam/cli/analysis.py:cmd_plan_kernel_experiment:143-165
interface:      (no interface — inline reconstruction of
                `kernel_analysis.candidates.CandidateFactors`,
                `CandidateSpec`, `FeasibilityResult`, `RejectionReason` from a
                raw `dict` read out of `analysis.get("feasibility")`)
implementation: for each row of the analysis's `feasibility` list, the
                handler manually pulls `factors_row["tile"]`,
                `factors_row["workgroup"]`, `waves_per_workgroup`,
                `pipeline_stages`, etc. into a `CandidateFactors(...)` call;
                builds `operand_transports` via `OperandTransport.from_json`;
                constructs `CandidateSpec(...)`; and rebuilds
                `RejectionReason(RejectionCode(reason["code"]), ...)` and
                `FeasibilityResult(...)` by hand — 23 lines of schema
                knowledge embedded directly in a CLI handler.
verdict:        duplicates-a-rule
evidence:       `boltbeam/kernel_analysis/candidates.py` defines `to_json()`
                on `RejectionReason` (line 35), `CandidateFactors` (line 85),
                `CandidateSpec` (line 119), and `FeasibilityResult` (line
                160) — i.e. the module already owns the JSON shape of all
                four types — but defines no matching `from_json` on any of
                them. `OperandTransport` (in `kernel_analysis/model.py`) *is*
                given a `from_json` and is used correctly at line 158.
                Grepping the whole tree
                (`grep -rn "CandidateSpec(\|CandidateFactors(\|FeasibilityResult(\|RejectionReason(" `)
                shows exactly two call sites for these four types: their own
                construction inside `candidates.py` (the authority, which
                naturally builds them directly) and this one CLI handler,
                which is therefore the only place in the codebase inventing
                a deserialization rule for a schema `candidates.py` itself
                owns and already serializes the other way. That is knowledge
                duplication of the schema shape (to_json defines it one way,
                the CLI reconstructs the inverse ad hoc) living in the wrong
                module, and it hides ownership of that schema from its
                owner — the "hide ownership of state" anti-pattern named in
                "Separate Ergonomics From Semantics".
remedy:         absorb into <owner>: add `CandidateSpec.from_json` and
                `FeasibilityResult.from_json` (mirroring
                `OperandTransport.from_json`) to
                `boltbeam/kernel_analysis/candidates.py`; the CLI handler
                calls them instead of rebuilding the dataclasses field by
                field.
```

## 7. `_VERDICT_PRIORITY` tie-break policy embedded in the CLI layer

```
module/symbol:  boltbeam/cli/workflow.py:31-36 (`_VERDICT_PRIORITY` dict) and
                :512 (`decision = min(decisions, key=lambda d:
                _VERDICT_PRIORITY.get(d.verdict, 99))`, inside cmd_evaluate)
interface:      (no interface — a module-level constant plus an inline `min`)
implementation: defines an explicit total order over all 8 `Verdict` values
                ("preference order when picking one primary decision out of
                many: most actionable first" — PROMOTE=0 ... ADAPTER_INCOMPLETE=7)
                and uses it to pick which of several per-candidate decisions
                `evaluate` (no `--candidate` given) reports as "the" answer.
verdict:        misleading-name (a comment calling it "preference order" is
                doing real policy work, not formatting) — this is the §D.2
                anti-pattern "make a policy decision look like a
                formatting/transport detail": nothing about "which verdict
                wins when several candidates disagree" is a CLI/output
                concern, it is a ranking rule over the domain's own Verdict
                type.
evidence:       `boltbeam/eval/evaluator.py` defines `evaluate` and
                `evaluate_all` (the functions that produce the `decisions`
                list `_VERDICT_PRIORITY` ranks) but has no verdict-priority
                or "primary decision" concept anywhere in it — grepped
                (`grep -n "model_id\|target_id" boltbeam/eval/evaluator.py`)
                confirms `evaluate_all` returns the decisions unranked;
                `grep -rn "_VERDICT_PRIORITY" boltbeam/` shows it is defined
                and consumed only in `cli/workflow.py`, i.e. it is not
                duplicated anywhere else (so this is not a Phase B/C finding)
                but it is a policy authority sitting outside the module
                (`eval/evaluator.py`) that owns every other Verdict-related
                rule (the DEFER/target-incomplete rule at evaluator.py:262,
                the KV-policy result at evaluator.py:210, etc.).
remedy:         absorb into <owner>: move `_VERDICT_PRIORITY` and the
                selection into `boltbeam/eval/evaluator.py` as a small
                `primary_decision(decisions: list[CandidateDecision]) ->
                CandidateDecision` function; `cmd_evaluate` calls it. This
                also makes the tie-break rule unit-testable next to
                `evaluate_all` instead of only reachable through the CLI.
```

## 8. `kernel_analysis/provider_adapters.py` (412 LOC) — Design For Replacement check

```
module/symbol:  boltbeam/kernel_analysis/provider_adapters.py (all 4 adapters:
                adapt_resource_trace, adapt_isa_manifest, adapt_execution_result,
                adapt_hw_trace) against boltbeam/kernel_analysis/adapters.py
                (its designated authority layer)
interface:      Adapter = Callable[[Mapping[str, Any], EvidenceSource], AdaptResult]
                registered via register_kernel_adapter(schema, adapter) and
                dispatched by adapt_kernel_artifact(mapping, source) ->
                AdaptResult, where AdaptResult.status is one of
                OK / UNSUPPORTED / INVALID.
implementation: each of the 4 functions takes one raw provider JSON `Mapping`
                and returns an `AdaptResult` carrying an `EvidenceFragment`
                (or several, for `adapt_hw_trace`'s per-kernel rows) whose
                `payload` mirrors `KernelEvidence` JSON sections.
verdict:        deep — this is a correctly shaped replacement boundary.
evidence, checked against each of the 4 required properties:
  - shared policy lives ABOVE the adapter: `adapters.py` owns
    registration (`register_kernel_adapter`, raises on duplicate schema,
    adapters.py:69-77), schema detection (`detect_kernel_artifact`,
    adapters.py:83-96), the OK/UNSUPPORTED/INVALID error taxonomy
    (`adapt_kernel_artifact`, adapters.py:98-111), and the fragment-merge/
    identity-join logic (`_merge_section`, adapters.py:132+, which raises a
    `FactBlocker` on conflicting values rather than silently picking one —
    "contradictory fragments are retained as blockers," adapters.py
    module docstring). None of that is reimplemented in provider_adapters.py;
    every adapter there returns data to the shared merge, never merges
    itself.
  - backend-specific behavior stays INSIDE the adapter:
    `_ISA_KIND_TO_STRUCTURE` (provider_adapters.py:24-51, the ISA-row-kind →
    structure-counter map) and `_PHASE_STATUS` (provider_adapters.py:134-143,
    the execution-bridge phase/producer-status → normalized-status map) are
    both backend-format-specific translation tables that belong nowhere
    else, and they are the only place they appear
    (`grep -rn "_ISA_KIND_TO_STRUCTURE\|_PHASE_STATUS" boltbeam/` — one
    definition site each, both in this file).
  - capability differences are explicit, not implied:
    `adapt_hw_trace`'s docstring (provider_adapters.py:345-350) states the
    capability gap directly — "these traces carry measured per-kernel timing
    but no correctness oracle, so each candidate is diagnosis-eligible and
    timing-present but not performance-eligible until correctness is
    supplied — the eligibility authority reports exactly that rather than
    fabricating a verdict." That is the opposite of "quietly redefine shared
    policy": it names the missing capability and defers the verdict to the
    authority instead of faking one.
  - the contract is testable without every real backend:
    `tests/kernel_analysis/test_provider_adapters.py` (201 lines, 10 tests)
    exercises all 4 adapters via hand-built dict fixtures (`_resource_trace()`,
    `_src()`, etc.) with an `A._reset_adapters_for_test()` autouse fixture —
    no rocprof, tinygrad, or GPU dependency anywhere in the test file
    (grepped: no subprocess/import of a real backend).
  - two of the four adapters (`adapt_resource_trace`, `adapt_isa_manifest`)
    delegate structural validation to shared validators
    (`search.joins.resource_join.validate_resource_snapshot`,
    `search.transfer_contract.validate_amd_isa_proof_manifest`); the other
    two (`adapt_execution_result`, `adapt_hw_trace`) do their own inline
    field-presence/type checks. This is an asymmetry, not a violation:
    grepping for a canonical validator for `SCHEMA_EXECUTION_BRIDGE_RESULT`
    or `HW_TRACE_SCHEMA` outside this file
    (`grep -rn "SCHEMA_EXECUTION_BRIDGE_RESULT\|validate.*execution"
    boltbeam/ --include=*.py`) finds only a one-line schema-string check in
    `runtime/execution_bridge.py:108`, not a full validator — so there is no
    existing shared rule these two adapters are "quietly redefining." They
    are simply the sole owner of that validation today. Noted as a
    structural inconsistency worth optional cleanup (extract
    `validate_execution_bridge_result` / `validate_hw_trace` alongside the
    other two for symmetry and independent testability), not as an
    anti-pattern finding — no duplicate authority exists for them to
    contradict.
remedy:         leave (adapter shape); optional, non-blocking: extract the
                two inline validators for symmetry with the other two.
```

**Verdict on provider_adapters.py, directly:** it does not "quietly redefine
shared policy." Shared policy (registration, detection, error taxonomy,
fragment merge/identity-join with explicit conflict blockers) lives in
`adapters.py`, above it, exactly as Design For Replacement requires;
backend-specific translation tables live inside it; the one real capability
gap in the file (`hw_trace` has no correctness oracle) is stated in prose at
the point of adaptation rather than silently defaulted; and the whole file is
exercised by 10 tests against synthetic dict fixtures with zero real-backend
dependency.

---

## What was clean

- `boltbeam/cli/__init__.py` (§1) and `boltbeam/cli/_common.py` (§2): both
  deep, both under their acceptance-criteria size limits (20 and 91 lines),
  no cross-imports between `cli/*.py` modules other than through `_common`.
- The 15 new/relocated package `__init__.py` files from Phase A (§3): all
  empty, no re-export shim introduced, no "vague wrapper that hides where
  authority actually lives."
- The bulk of the 68 CLI handlers: read the full text of `analysis.py`,
  `search.py`, `roofline.py`, `profile.py`, `workflow.py`. Outside the two
  findings above (§5, §6), every handler is a genuinely thin
  parse-call-write adapter — it calls exactly one subsystem function (or a
  short, obviously-sequential chain of them) and does not reimplement
  anything the subsystem already does. This matches the scope's own
  characterization in §A.1 ("thin adapters ... the correct shape for a CLI
  layer") and I found nothing to contradict it beyond §5/§6.
- `kernel_analysis/provider_adapters.py` (§8): a correctly shaped replacement
  boundary against every one of the four Design-For-Replacement criteria,
  with real tests exercising the contract without a real backend.
- `cmd_reopen_check` (`workflow.py:530-568`): despite doing meaningfully more
  than "parse, call, write" (bypasses search-space authorization, downgrades
  a would-be PROMOTE to CANDIDATE), its docstring explains exactly why it
  differs from `evaluate` and what invariant it preserves ("the promotion
  contract is preserved... promotion still requires the authorized
  `evaluate` path") — this is "Explain Tradeoffs Close To The Code" done
  correctly, not a hidden policy fork, and the CANDIDATE-downgrade rule
  exists nowhere else to be a duplicate of. Checked and left as-is.

## What I could not determine / declined to flag without stronger evidence

- `cmd_evaluate`'s cross-file `model_id`/`target_id` agreement check
  (`workflow.py:483-489`) is real validation logic sitting in the CLI layer
  rather than in `eval/evaluator.py`. I looked for a duplicate of this rule
  elsewhere (there is none — `evaluate()` takes already-resolved
  `model_id`/`target_id` as parameters and does not itself reconcile three
  input artifacts) and could not establish either (a) that it is duplicated
  knowledge, or (b) that a second unofficial way exists elsewhere to bypass
  it. It is the single place three JSON artifacts converge, which is a
  legitimate reason for an entry point to own a consistency check. I am
  recording it as a borderline case rather than a finding: it does not meet
  this audit's evidence bar for `duplicates-a-rule` (nothing to duplicate)
  or `shallow-passthrough` (it adds a real invariant), so per the scope's own
  rule ("a finding without evidence... must be omitted") I left it out of
  the verdict table.
- I did not do a full inventory of every wrapper/helper repo-wide (409
  files); Phase A.1/A.2/A.3's own targets (`cli/`, the 10 new root packages,
  `search/`) plus the explicitly named `provider_adapters.py` were the
  scope's stated focus for this phase, and that is what this audit covers
  exhaustively. A broader sweep of `boltbeam/kernel_analysis/*.py` (22
  dataclasses, left alone by A.4) or the `perf/` versioned models (§6 of the
  scope, explicitly deferred to a separate research step) was out of scope
  for D and not attempted here.
