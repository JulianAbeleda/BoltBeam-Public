# Phase C centralization audit

Status: complete (report only — no source file modified, moved, or deleted)
Audited: 2026-07-31
Scope: `docs/task_workflow/input/organization-and-pruning-scope-20260730.md`, §Phase C (§304-339)
Authority: `docs/coding-principles.md` ("Centralize"), `docs/commit-discipline.md`
Audited at commit: `a3f1f6387f737844f8c0018dc2b413aa1552c34d` (2026-07-31 07:16:43 -0400)

This is read-only. No source file was modified, moved, or deleted to produce it.

**Note on the task brief's paths:** the brief points at
`structure-template/structure/Development/coding-principles.md`. That path does
not exist in this tree; the file lives at `docs/coding-principles.md` and is
the one actually referenced by `docs/commit-discipline.md` and by this scope's
own `Authority:` line. Read from there instead. Likewise `vocab.py` is a flat
module (`boltbeam/vocab.py`), not the `vocabulary/` package produced by Phase
A.2 (`boltbeam/vocabulary/` holds only `model_vocab.py` and
`vocab_capture.py`) — the brief's "vocab.py / kernel_analysis/model.py" hint
for schema authority is otherwise correct.

**Note on concurrent writers:** two other agents are writing different files
under `docs/task_workflow/output/`. This report only reads source and only
writes `centralization-audit-20260730.md`.

## 0. Verdict table

| # | Category | Rule | Verdict | Sites |
| - | --- | --- | --- | --- |
| 1 | schemas | `SCHEMA_HW_TRACE` / `SCHEMA_TIMING_TRACE` (`boltbeam.hw_trace.v1`, `boltbeam.timing_trace.v1`) | **duplicated** | 6 |
| 2 | schemas | `SCHEMA_PROFILER_CAPABILITIES` / `SCHEMA_COUNTER_REGISTRY` defined twice | **duplicated** | 4 |
| 3 | schemas | ~150 other schema ids used only as bare literals, no `vocab.py` entry | **no-authority** | ~85 files |
| 4 | external boundaries | rocprofv3 kernel-trace/memory-copy-trace CLI invocation | **duplicated** | 2 |
| 5 | routing/policy | 14B/Q4 promotion-gate rules (`CONTEXTS`, `ROLES`, correctness/quant checks) | **duplicated / no-authority** | 7 files in `tools/` |
| 6 | routing/policy | `PARITY_FRACTION = 0.98` promotion threshold | **no-authority** | 1 (should be in `eval/thresholds.py`) |
| 7 | routing/policy (commit prefixes) | allowed commit-msg subsystem prefixes | **duplicated-and-stale** | 3 |
| 8 | paths | canonical path to `policy/assets/route_manifest.v1.json` | **duplicated** | 2 |
| 9 | paths | tinygrad-checkout discovery | **duplicated (weaker copy)** | 1 |
| 10 | env/config | every `os.environ`/`getenv` read | **centralized (clean)** | 9 sites, no overlap |
| 11 | routing/policy | speed-movement tier thresholds (`eval/thresholds.py`) | **centralized (clean)** | 1 definer, 1 consumer |
| 12 | schemas | `kernel_analysis/model.py` schema constants | **centralized (clean)** | imports from `vocab.py` |
| 13 | external boundaries | provider adapters as a class (`collectors/`, `profiler/importers/`) | **centralized (clean)** | — |

Findings by bucket: **duplicated: 6** (#1, #2, #4, #5, #7, #8), **no-authority: 3** (#3, #5b, #6), **duplicated (weaker copy): 1** (#9), **centralized/clean: 4** (#10-13).

---

## 1. Environment / config — checked, clean

```
rule:        every os.environ / os.getenv read in the tree
authority:   none exists as a single module, but not needed — see proof
sites:       boltbeam/target/tinygrad_root.py:26        (TINYGRAD_ROOT)
             boltbeam/workflow/common.py:54              (BOLTBEAM_XML)
             boltbeam/characterize/system_snapshot.py:127 (arbitrary caller-supplied env var names)
             boltbeam/collectors/tinygrad_rocprof.py:94   (env passthrough for subprocess)
             boltbeam/search/semantic/mr7_evidence_bundle.py:205 (env passthrough for subprocess)
             boltbeam/search/full_kernel/tinygrad_full_kernel.py:127,164 (env passthrough for subprocess)
             boltbeam/adapters/llama_server_control.py:157 (env passthrough for subprocess)
             boltbeam/control/matched_control.py:385      (env passthrough + one override for METAL)
verdict:     centralized (as much as it needs to be)
proof:       Only 9 total `os.environ`/`os.getenv` sites in the whole tree
             (`boltbeam/` + `tools/`). Two are true "read a named config
             variable" sites, and each variable has exactly one read site:
             `TINYGRAD_ROOT` is read only in `tinygrad_root.py`, which
             documents itself as "One portable authority for locating a
             tinygrad checkout"; `BOLTBEAM_XML` is read only in
             `workflow/common.py`, next to the `write_json` it gates. No
             variable is read in two places with different defaults or
             different meaning. The remaining sites are `env={**os.environ,
             ...}` passthroughs for subprocess launches (forwarding the
             caller's environment, not reading a config value) plus one
             deliberately generic reader (`system_snapshot.py`, which takes a
             list of env-var *names* as a parameter and records each as a
             fact — that genericity is its job, not a duplicate authority).
remedy:      none. This category does not need a dedicated config module at
             this scale; each variable already has one owner.
```

---

## 2. Schemas

### 2.1 `SCHEMA_HW_TRACE` / `SCHEMA_TIMING_TRACE` re-hardcoded outside `vocab.py`

```
rule:        the schema-id string for a hw_trace / timing_trace artifact
authority:   boltbeam/vocab.py:355,357 (SCHEMA_TIMING_TRACE = "boltbeam.timing_trace.v1",
             SCHEMA_HW_TRACE = "boltbeam.hw_trace.v1") — correctly imported and used in
             boltbeam/trace/hw_trace.py:12
sites:       boltbeam/kernel_analysis/provider_adapters.py:321  HW_TRACE_SCHEMA = "boltbeam.hw_trace.v1"
             boltbeam/collectors/tinygrad_rocprof.py:144        "schema": "boltbeam.timing_trace.v1"
             boltbeam/collectors/llama_rocprof.py:85            "schema": "boltbeam.timing_trace.v1"
             boltbeam/collectors/tinygrad_profile_events.py:35  "schema": "boltbeam.timing_trace.v1"
             boltbeam/collectors/tinygrad_native_pmc.py:13      "schema": "boltbeam.timing_trace.v1"
             boltbeam/collectors/metal_system_trace.py:184      "schema": "boltbeam.timing_trace.v1"
verdict:     duplicated
proof:       vocab.py's own docstring states the rule directly: "every ...
             schema id lives HERE and is imported everywhere else. No module
             may invent a parallel string vocabulary." SCHEMA_TIMING_TRACE and
             SCHEMA_HW_TRACE already exist there and are correctly imported by
             trace/hw_trace.py. The 6 sites above emit the identical string
             value as a bare literal (5 of them with no named constant at
             all) instead of importing the existing constant.
             kernel_analysis/provider_adapters.py is the sharpest case: the
             same file imports SCHEMA_EXECUTION_BRIDGE_RESULT from vocab.py at
             line 17, proving the import path is already open, yet at line
             321 it defines its own private HW_TRACE_SCHEMA duplicate instead
             of importing SCHEMA_HW_TRACE.
remedy:      import SCHEMA_TIMING_TRACE / SCHEMA_HW_TRACE from boltbeam.vocab
             at all 6 sites; delete the local HW_TRACE_SCHEMA in
             provider_adapters.py.
```

### 2.2 `SCHEMA_PROFILER_CAPABILITIES` / `SCHEMA_COUNTER_REGISTRY` defined twice

```
rule:        the schema-id string for the profiler capability registry and
             the counter registry
authority:   ambiguous — defined identically in two places (see sites)
sites:       boltbeam/vocab.py:358    SCHEMA_PROFILER_CAPABILITIES = "boltbeam.profiler_capabilities.v1"
             boltbeam/vocab.py:359    SCHEMA_COUNTER_REGISTRY = "boltbeam.counter_registry.v1"
             boltbeam/profiler/capabilities.py:10  SCHEMA_PROFILER_CAPABILITIES = "boltbeam.profiler_capabilities.v1"
             boltbeam/profiler/counters.py:13      SCHEMA_COUNTER_REGISTRY = "boltbeam.counter_registry.v1"
verdict:     duplicated
proof:       Both pairs are the identical name and identical string value,
             defined independently in two files. Confirmed no import bridges
             them: `grep "from boltbeam.vocab import" for these two names`
             returns nothing — vocab.py's copies are dead. The live copies
             are the ones in profiler/capabilities.py and profiler/counters.py,
             re-exported through boltbeam/profiler/__init__.py and used
             throughout profiler/*. vocab.py's own docstring rule ("No module
             may invent a parallel string vocabulary") is violated by vocab.py
             itself here — it invented a second, unused copy of a vocabulary
             that profiler/ already owns.
remedy:      delete the two dead lines in vocab.py (358-359); if anything
             ever needs these from outside profiler/, re-export from
             boltbeam.profiler, not boltbeam.vocab.
```

### 2.3 The bulk of schema literals: no authority, not duplication

```
rule:        every other "boltbeam.<name>.v<N>" schema id in the tree
authority:   none exists — vocab.py does not claim these
sites:       ~150 distinct schema ids across ~85 files (perf/, search/,
             roofline/, plan/, experiment/, diagnostics/, artifacts/,
             workflow/, control/, math/, trace/, runtime/, vocabulary/,
             synth/); full occurrence list captured during this audit as
             /tmp/claude-1000/.../scratchpad/hardcoded_hits.txt (151 lines,
             regenerable via:
             grep -rEno '.*"boltbeam\.[a-zA-Z0-9_.]+\.v[0-9]+".*' boltbeam
             --include=*.py | grep -v __pycache__ | grep -vE
             ':\s*SCHEMA_[A-Z0-9_]+\s*=')
verdict:     no-authority (not "duplicated" — there is nothing to be
             duplicated against)
proof:       Each of these modules defines its own local SCHEMA/`*_SCHEMA`
             constant (e.g. perf/cycle_model.py: MODEL_SCHEMA, PREDICTION_SCHEMA;
             search/role_cost_ranking.py: 5 distinct *_SCHEMA constants;
             search/semantic/mr7_evidence_bundle.py: 5 more) or inlines the
             literal with no constant at all (roofline/roofline_trace.py,
             trace/schedule_trace.py, workflow/output.py, runtime/gpu_health.py,
             analyze.py). None of these ~150 ids appear in vocab.py. This is
             not the "duplicated" bucket from §2.1/2.2 (same id, defined
             twice) — it is the opposite: vocab.py's docstring claims to be
             the authority for "every ... schema id" but in practice covers
             only 62 of ~176 distinct schema ids in the tree (measured:
             `grep -c "^SCHEMA_" boltbeam/vocab.py` = 62 vs. 176 unique
             literal ids found tree-wide). The docstring's claim is aspirational,
             not descriptive.
remedy:      either (a) narrow vocab.py's docstring claim to the artifact
             kinds it actually governs (the "durable evidence chain" schemas:
             model_profile, evidence, candidate_decision, route_ledger,
             hw_trace, timing_trace, etc.), and treat each subsystem's local
             SCHEMA constant as that subsystem's own legitimate authority
             point for its own artifact kind (perf/, search/mmq/,
             search/semantic/ each own their MR7/MR8/MR9/mmq_* schemas) — this
             matches how the rest of the tree is already organized, one
             module = one artifact kind = one schema owner; or (b) if a
             single tree-wide schema registry is actually wanted, that is a
             remediation decision out of scope for this audit (§1.3 of the
             input scope: audit phases produce a report, not a fix).
             Recommend (a): the pattern already used by perf/, search/mmq/,
             search/semantic/ (module defines its own SCHEMA constant next to
             the dataclass it belongs to) is itself a legitimate,
             non-duplicated authority point per module — it just isn't
             vocab.py. Do not force these into vocab.py; that would be
             centralizing unrelated knowledge into one file, the "wrong
             abstraction" the principles warn against.
```

**On the task's "195" figure:** this audit's own count differs slightly (213
raw string occurrences / 176 unique ids / 79 files by one regex methodology;
151 non-definition "hardcoded-looking" lines by a second). The exact number is
sensitive to whether `SCHEMA_X = "..."` assignment lines are counted, whether
occurrences or lines are counted, and whether `tinygrad.*`/`execution_bridge.*`
namespaced literals are included. Whatever the precise count, the qualitative
answer is unambiguous and is the finding that matters: **most schema-id
literals are not references to vocab.py at all** — they are each subsystem's
own locally-defined artifact-kind marker, which is a legitimate pattern for
~150 of them (§2.3) and a real, narrow duplication for 6+4=10 of them (§2.1,
§2.2) where a vocab.py-owned id is redefined elsewhere.

### 2.4 `kernel_analysis/model.py` — checked, clean

```
rule:        SCHEMA_KERNEL_EVIDENCE / SCHEMA_KERNEL_ANALYSIS / SCHEMA_KERNEL_RECOMMENDATION
authority:   boltbeam/vocab.py:385-387
sites:       boltbeam/kernel_analysis/model.py:10-12 (imports all three from boltbeam.vocab)
verdict:     centralized
proof:       model.py imports these three constants from vocab.py and uses
             them at every read/write/validate site (lines 1229, 1274-1275,
             1354, 1372-1373, 1440, 1474-1475). No local redefinition. This is
             the correct pattern and the counter-example to §2.1/§2.2.
remedy:      none.
```

---

## 3. External boundaries

### 3.1 Provider adapters as a class — checked, clean

```
rule:        one subprocess/HTTP adapter per external tool provider
authority:   boltbeam/collectors/ (llama_rocprof.py, tinygrad_rocprof.py,
             tinygrad_native_pmc.py, tinygrad_profile_events.py,
             metal_system_trace.py, llama_ncu.py) plus
             boltbeam/profiler/importers/ncu.py
sites:       (all subprocess call sites in the tree — 16 total, listed via
             `grep -rl "subprocess\.(run|Popen|check_call|check_output)"
             boltbeam tools`)
verdict:     centralized (as a class)
proof:       Every subprocess call to an external measurement tool
             (rocprofv3, ncu, llama-bench, llama-server, tinygrad's own
             python entrypoints) lives inside collectors/ or
             profiler/importers/, one file per provider, matching the scope's
             "one adapter per provider" expectation. No CLI handler or search/
             module calls subprocess directly to reach one of these tools —
             they all go through a collector.
remedy:      none for the boundary structure itself. See §3.2 for a real
             duplication *within* two of these adapters.
```

### 3.2 rocprofv3 invocation flags duplicated across two collectors

```
rule:        how rocprofv3 must be invoked to capture kernel-trace +
             memory-copy-trace + stats in CSV form
authority:   none exists (should be one function, e.g.
             collectors/_rocprofv3.py:rocprofv3_command(...))
sites:       boltbeam/collectors/tinygrad_rocprof.py:137-139
             boltbeam/collectors/llama_rocprof.py:75-78
signatures:  (both are inline list literals inside capture_tinygrad_rocprof /
             capture_llama_rocprof, not functions)
verdict:     duplicated
proof:       Both files build the identical 10-token rocprofv3 prefix:
             [cfg.rocprofv3, "--kernel-trace", "--memory-copy-trace", "--stats",
              "-f", "csv", "-d", str(cfg.trace_dir), "-o", cfg.prefix, "--"]
             byte-for-byte, before appending the provider-specific tail
             command. The failure-path response dict each builds on
             `proc.returncode != 0` is also structurally identical (same 7
             keys: schema/model_id/target_id/workload/provider_id/
             timing_source/contexts/rows/aux_sources), differing only in the
             "stage" string and provider_id value. This is the same rule
             ("what flags make rocprofv3 emit CSV kernel+memory-copy stats,
             and what a collector failure artifact looks like") encoded
             twice, not two different rules that happen to look alike — the
             two collectors want exactly the same rocprofv3 behavior.
remedy:      canonicalize-then-collapse: extract a shared
             `_rocprofv3_command(rocprofv3_bin, trace_dir, prefix) -> list[str]`
             and a shared `_collector_failure(schema, model_id, target_id,
             workload, provider_id, stage, context, trace_dir, **aux) -> dict`
             into a small shared module (e.g. collectors/_rocprofv3.py), and
             have both capture_* functions call it, appending only their own
             tail command.
```

---

## 4. Routing / policy

### 4.1 `eval/thresholds.py` — checked, clean

```
rule:        speed-movement tier and regression thresholds
authority:   boltbeam/eval/thresholds.py (TIER_A_MIN_PCT, TIER_B_MIN_PCT,
             SPEED_EQUIVALENT_BAND_PCT, PROTECTED_REGRESSION_MAX_PCT,
             NOISE_QUALIFY_SPREAD_MULT, classify_tier, is_protected_regression)
sites:       boltbeam/eval/evaluator.py (sole importer)
verdict:     centralized
proof:       `grep` for every threshold name and for classify_tier /
             is_protected_regression across boltbeam/ and tools/ turns up
             exactly one definer (eval/thresholds.py) and one consumer
             (eval/evaluator.py). No script in tools/ redefines a 5%/2%/1%
             tier boundary. This is the clean counter-example the scope asked
             this audit to also report.
remedy:      none.
```

### 4.2 `tools/` 14B/Q4 promotion scripts — a parallel, disconnected promotion authority

This is the single largest finding of this audit and squarely the smell named
in §C.2 of the input scope.

```
rule:        the set of (context, role, quant) values a 14B/Q4 route-promotion
             check must see evidence for, and what counts as a passing gate
authority:   boltbeam/eval/ (evaluator.py, thresholds.py, contracts.py) and
             boltbeam/vocab.py (RoleGroup: ATTN_QO, ATTN_KV, FFN_GATE_UP, ...)
             — the intended single authority per coding-principles.md
             ("Model-Agnostic Rules": "Roles use the centralized taxonomy")
sites:       tools/promote_14b_routes.py:10-12          ROLES=(...); QUANTS=(...); CONTEXTS=(512,1024,2048,4096)
             tools/validate_fused_route_promotion.py:12,17  CONTEXTS=(...); ROLES=(...)
             tools/account_prefill_roles.py:11-12       CONTEXTS=(...); REQUIRED_ROLES=(...)
             tools/compare_same_run.py:10               CONTEXTS=(...)
             tools/prepare_q4_comparison.py:12           CONTEXTS=(...)
             tools/measure_14b_same_run.py:9-10          CONTEXTS=(...); ROLES=(...)
             tools/prepare_14b_same_run.py:8-9           CONTEXTS=(...); ROLES=(...)
verdict:     duplicated (the tuple values) and no-authority (the promotion
             logic itself: none of these 7 files import anything from
             boltbeam — confirmed by grepping every `^import`/`^from` line in
             each file; all imports are stdlib only: argparse, json,
             statistics, sys, pathlib)
proof:       `CONTEXTS = (512, 1024, 2048, 4096)` is byte-identical across all
             7 files. `ROLES`/`REQUIRED_ROLES = ("ffn_gate_up", "ffn_down",
             "attn_qo", "attn_kv")` is byte-identical across 5 of the 7
             (promote_14b_routes.py, validate_fused_route_promotion.py,
             account_prefill_roles.py, measure_14b_same_run.py,
             prepare_14b_same_run.py). These four role-name strings are not
             an ad hoc list the tools invented independently of BoltBeam's own
             taxonomy — vocab.py already defines this exact set as
             `RoleGroup` members (ATTN_QO="attn_qo", ATTN_KV="attn_kv",
             FFN_GATE_UP="ffn_gate_up" at vocab.py:113-115, plus FFN_DOWN).
             Each of the 7 tools re-derives each artifact's own promotion
             schema id ("boltbeam.qwen3_14b_promotion.v1",
             "boltbeam.fused_route_promotion.v1",
             "boltbeam.prefill_role_accounting.v1", "boltbeam.same_run_parity.v1",
             "boltbeam.q4_comparison_preparation.v1",
             "boltbeam.qwen3_14b_same_run_measurement.v1",
             "boltbeam.qwen3_14b_same_run_preparation.v1") — none registered
             in vocab.py, none built on boltbeam.eval.contracts or
             boltbeam.eval.evaluator's PASS/BLOCKED verdict machinery. Each
             tool hand-rolls its own "errors: [] -> PASS else BLOCKED"
             control flow instead of using boltbeam's Verdict vocabulary
             (Verdict.PROMOTE/REFUTE/etc. from vocab.py) or eval/evaluator.py.
             This is exactly "one-off scripts that silently redefine
             canonical rules" (§C.2): a short script, here 7 short scripts,
             each redefining the role taxonomy, the context ladder, and a
             promotion-verdict schema that BoltBeam already has an authority
             for, without importing it.
remedy:      import ROLES from boltbeam.vocab (RoleGroup members) and
             CONTEXTS from a shared constant (currently does not exist inside
             boltbeam/ either — search/spec.py or eval/contracts.py would be
             the natural home; introduce one CONTEXT_LADDER constant there,
             not one per tool). Route each tool's PASS/BLOCKED decision
             through boltbeam.eval's verdict vocabulary instead of a bespoke
             ad hoc status string, and register each of the 7 schema ids in
             vocab.py once decided they are durable contracts. This is a
             remediation-phase decision (§10 of the input scope), not
             executed here.
```

### 4.3 `PARITY_FRACTION = 0.98` — an un-centralized promotion threshold

```
rule:        the tinygrad/llama same-run parity floor used to gate promotion
             ("tinygrad_tok_s >= 0.98 * llama_tok_s at every ppN")
authority:   none exists; boltbeam/eval/thresholds.py is the intended home
             per "Keep one clear authority point for ... routing rules and
             system policy"
sites:       tools/compare_same_run.py:11  PARITY_FRACTION = 0.98
verdict:     no-authority
proof:       This is a route-promotion threshold of exactly the kind
             eval/thresholds.py exists to own (it already owns
             TIER_A_MIN_PCT, PROTECTED_REGRESSION_MAX_PCT, etc.), yet it lives
             solely in a tools/ script with zero import of boltbeam and no
             registered counterpart anywhere in boltbeam/eval or
             boltbeam/policy. Only one occurrence found (rule of three not
             met), so this is not yet a duplication — but it is a
             load-bearing promotion rule with no authority module, which is
             the narrower and arguably more dangerous case: if a second
             same-run-parity script is ever written (the tools/ directory's
             own pattern, given the 7 near-identical scripts in §4.2, makes
             this likely), it will very probably re-hardcode 0.98
             independently rather than finding this one.
remedy:      move PARITY_FRACTION into boltbeam/eval/thresholds.py now, ahead
             of a second occurrence, since the failure mode (silent
             independent re-derivation) is already demonstrated by every
             other constant in this same file (§4.2).
```

### 4.4 Commit-prefix policy — internally consistent, but stale against practice

```
rule:        the allowed set of commit-message subsystem prefixes
authority:   docs/commit-discipline.md, enforced by tools/check_commit_msg.py
             via .githooks/commit-msg (git config core.hooksPath=.githooks,
             confirmed active in this checkout)
sites:       docs/commit-discipline.md              (10 prefixes: profile, search,
                                                       policy, eval, synth, cli,
                                                       schema, docs, test, repo)
             tools/check_commit_msg.py:8-19          (identical 10-prefix tuple)
             docs/task_workflow/input/organization-and-pruning-scope-20260730.md:580-582
                                                      ("Prefixes in active use, by
                                                       frequency": 14 prefixes —
                                                       adds trace, target, bench,
                                                       path, collectors)
             git log --format=%s (this repo's actual history)
verdict:     duplicated-and-stale (the doc and the checker agree with each
             other, which is correct, but both disagree with recorded reality)
proof:       `docs/commit-discipline.md` and `tools/check_commit_msg.py` do
             not disagree with each other — they are one authority, correctly
             mirrored (doc human-facing, checker machine-enforced, per "Human-
             Facing And Machine-Enforced"). The problem is that both are
             stale relative to the commits that actually exist. `git log
             --format=%s | grep -oE '^\[[a-z_]+\]' | sort | uniq -c` on this
             checkout shows real, already-merged commits using `[trace]` (2),
             `[target]` (2), `[bench]` (2), `[path]` (1), `[collectors]` (1) —
             8 commits total, none of which the current
             ALLOWED_PREFIXES tuple accepts. Verified directly: running
             `tools/check_commit_msg.py` against the literal message
             `"[trace] add hw trace helper"` returns exit code 1
             ("need BoltBeam subsystem prefix... Allowed prefixes: [profile],
             [search], [policy], [eval], [synth], [cli], [schema], [docs],
             [test], [repo]"). Since Phase A.2 of this same scope created the
             domain packages (trace/, target/, roofline/, runtime/, memory/,
             quantization/, plan/, experiment/, control/) whose commits
             plausibly used exactly these new prefixes, the authority
             (commit-discipline.md's prefix table) was not updated when the
             tree it describes grew new subsystems — the doc and the hook
             both describe a pre-Phase-A subsystem list.
remedy:      add trace, target, bench, path, collectors (and any other
             domain-package name from Phase A.2/A.3 expected to take
             independent commits: roofline, runtime, memory, quantization,
             vocabulary, plan, experiment, control) to both
             docs/commit-discipline.md's table and
             tools/check_commit_msg.py's ALLOWED_PREFIXES in the same commit,
             so the two stay mirrored. This is the one finding in this audit
             that is not a structural duplication to remove but a single
             authority that fell behind the tree it governs — worth flagging
             because an unenforceable rule is worse than no rule: it either
             blocks legitimate commits (if the hook is live) or trains
             committers to route around it.
```

---

## 5. Paths

### 5.1 No dedicated path module exists

```
rule:        repo-root / subsystem-root path resolution
authority:   none exists as a single module (unlike vocab.py for schemas or
             eval/thresholds.py for policy)
sites:       boltbeam/characterize/system_snapshot.py:122   parents[2]
             boltbeam/cli/search.py:43                       parents[1]
             boltbeam/control/replay_ab.py:42                parents[2]
             boltbeam/profiler/counters.py:15                parents[1]
             boltbeam/search/util.py:14                      parents[2]
             boltbeam/policy/route_manifest.py:75,77         parents[2]
             boltbeam/control/matched_control.py:187         parents[2]
             tools/export_route_manifest_snapshot.py:8       parents[1]
             tools/install_hooks.py:10                       parents[1]
verdict:     no-authority
proof:       9 independent `pathlib.Path(__file__).resolve().parents[N]`
             call sites, each choosing its own N based on that file's current
             depth under boltbeam/ or tools/. None import a shared
             `repo_root()`/`boltbeam_root()` helper — there isn't one. This
             is fragile exactly where Phase A already changed file depths
             (e.g. anything moved from boltbeam/ root into a new domain
             package changed its parents[N] requirement by one level) and is
             a plausible source of the "confident, unverified structural
             claims" failure mode this scope's own §1 warns about, applied to
             paths instead of code shape.
remedy:      introduce one `boltbeam._paths.repo_root()` (or similar) that
             every one of these 9 sites calls instead of computing parents[N]
             independently. Out of scope to execute here (§1.3: audit, not
             remediation).
```

### 5.2 `tools/export_route_manifest_snapshot.py` duplicates `policy/route_manifest.py`'s own asset path

```
rule:        where the canonical route-manifest asset lives on disk
authority:   boltbeam/policy/route_manifest.py:8
             (_ASSET = pathlib.Path(__file__).with_name("assets") / "route_manifest.v1.json")
sites:       boltbeam/policy/route_manifest.py:8
             tools/export_route_manifest_snapshot.py:8
                (SOURCE = pathlib.Path(__file__).resolve().parents[1] /
                 "boltbeam/policy/assets/route_manifest.v1.json")
verdict:     duplicated
proof:       Both compute a path to the identical file
             (boltbeam/policy/assets/route_manifest.v1.json), by two
             unrelated methods: route_manifest.py derives it relative to its
             own `__file__` (`with_name("assets")`), which survives a move of
             the whole policy/ package; export_route_manifest_snapshot.py
             hardcodes the string "boltbeam/policy/assets/route_manifest.v1.json"
             from the assumed repo root, which does not survive a policy/
             package move (and did not survive the general reshuffle Phase A
             just did to 45 other files — it happens to still be correct only
             because policy/ itself was not moved). `_ASSET` is private
             (leading underscore), so this tool cannot currently import it
             without policy/route_manifest.py first exposing a public name —
             i.e. this is a duplication caused by a missing public seam, not
             carelessness.
remedy:      export a public `ASSET_PATH` (or a `asset_path()` accessor) from
             boltbeam.policy.route_manifest and have
             export_route_manifest_snapshot.py import it instead of
             re-deriving the path from repo root.
```

### 5.3 tinygrad-checkout discovery re-implemented, more narrowly, in `system_snapshot.py`

```
rule:        how to locate a tinygrad checkout when the caller did not supply
             one explicitly
authority:   boltbeam/target/tinygrad_root.py:resolve_tinygrad_root
             (checks explicit value, then TINYGRAD_ROOT_ENV, then any of 3
             recognized sibling directory names: tinygrad-arkey,
             tinygrad-arkey-exp, tinygrad)
sites:       boltbeam/characterize/system_snapshot.py:123
                (tg_repo = Path(tinygrad_repo or bb_repo.parent / "tinygrad-arkey"))
verdict:     duplicated (weaker copy)
proof:       Both encode "if not given explicitly, find the tinygrad sibling
             directory next to the boltbeam checkout." tinygrad_root.py's
             version checks the TINYGRAD_ROOT_ENV override and 3 recognized
             sibling names; system_snapshot.py's version skips the env
             override entirely and hardcodes exactly one sibling name
             ("tinygrad-arkey"), silently returning a wrong/nonexistent path
             if the actual checkout is named "tinygrad" or
             "tinygrad-arkey-exp", or if TINYGRAD_ROOT is set to point
             elsewhere. This is the same knowledge, encoded a second time,
             less completely.
remedy:      import resolve_tinygrad_root from boltbeam.target.tinygrad_root
             in system_snapshot.py instead of the inline fallback expression.
```

---

## 6. Checked and found clean — do not re-audit these

- **env/config as a whole** (§1): 9 sites, no variable read in two places, no
  duplicated meaning.
- **`eval/thresholds.py`** (§4.1): single definer, single consumer
  (`eval/evaluator.py`); no tool or script redefines a tier boundary.
- **`kernel_analysis/model.py`** (§2.4): imports its three schema constants
  from `vocab.py`; does not redefine them.
- **Provider-adapter boundary structure** (§3.1): every subprocess call to
  rocprofv3/ncu/llama-bench/llama-server/tinygrad lives inside `collectors/`
  or `profiler/importers/`, one file per provider. No CLI handler or `search/`
  module calls `subprocess` directly to reach an external tool.
- **`docs/commit-discipline.md` vs. `tools/check_commit_msg.py`** (§4.4): the
  human-facing doc and the machine-enforced checker agree with each other
  byte-for-byte on the allowed prefix set. (They are stale against practice —
  see §4.4 — but that is a currency problem, not an internal-consistency
  problem.)
- **`boltbeam/policy/` vs `boltbeam/eval/` boundary**: `policy/route_manifest.py`
  emits route-policy documents; `eval/evaluator.py` consumes evidence against
  `eval/thresholds.py`. No overlap found between what each owns.

## 7. Could not determine

- **The exact intended scope of `vocab.py`'s authority claim** (§2.3). Its
  docstring says "every ... schema id lives HERE," which is falsified by
  measurement (62 of ~176 unique schema ids), but this audit cannot tell
  whether that is a stale overclaim (fix the docstring) or a target the
  project intends to grow into (fix the code to match the doc, i.e. migrate
  the other ~114 schema ids into vocab.py over time). Both are legitimate
  remediation directions; picking between them is a product decision, not an
  audit finding — this scope's own §1.4 says to stop and report on
  contradiction rather than guess.
- **Whether the ~150 module-local `SCHEMA`/`*_SCHEMA` constants (§2.3) should
  ever be centralized.** Per "Duplication is cheaper than the wrong
  abstraction," forcing perf/'s 13 versioned models' schema constants and
  search/semantic's 5 MR7 schema constants into one shared file would create
  a single file with ~150 unrelated constants and no shared meaning beyond
  "is a schema id" — arguably the textbook wrong abstraction. This audit
  flags the pattern (module owns its own schema id) as *not itself a defect*
  for the ~150 (§2.3), while flagging the 10 cases where a vocab.py-owned id
  is *specifically re-hardcoded elsewhere* (§2.1, §2.2) as defects. Whether a
  reviewer agrees with that line is a judgment call this audit cannot fully
  resolve from evidence alone.
- **Whether tools/'s 7 promotion scripts (§4.2) are meant to eventually become
  `boltbeam` CLI commands or are meant to stay as standalone one-off
  scripts by design** (e.g. because they encode a specific historical 14B/Q4
  investigation rather than a durable, repeatable gate). The scope's own
  §F.2 (obsolete probe pruning) and this Phase-C finding pull in different
  directions: if these scripts' verdicts are already recorded in the ledger,
  §F.2 says delete them; if they are still an active gate, §C.2 says
  centralize their rules. This audit did not check the ledger for recorded
  verdicts from these scripts (out of Phase C's scope) and cannot resolve
  which remedy applies without that check.
