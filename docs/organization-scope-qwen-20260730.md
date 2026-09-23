# BoltBeam organization scope — Qwen assessment

Status: proposed
Scoped: 2026-07-30
Companion: `organization-scope-claude-20260730.md`

Claude's scope correctly identifies `cli.py` as the most urgent defect and
correctly argues to leave `model.py` alone — its dependency graph is a single
tree with no viable seams. This document does not contradict those conclusions;
it extends the scope to the next-largest structural problems that, left
untouched, will cause the same monotonic growth pattern that created `cli.py`
in the first place.

The thesis: **fixing only the CLI treats the symptom, not the source.** The
source is a pattern of versioned duplication and orphan accumulation that is
actively making the repository harder to navigate. This scope addresses it in
five phases, ordered so that each phase is independently green and the tree
is never worse than it started.

## 0. Verdict first

| Area | Files / Lines | Verdict |
| --- | --- | --- |
| `cli.py` | 1 file, 1,773 LOC | **Split** — by domain, into `cli/` package (agree with Claude) |
| `kernel_analysis/model.py` | 1 file, 1,492 LOC | **Leave alone** (agree with Claude) |
| `perf/` versioned models | 15 files, ~570 LOC | **Collapse** — into 2 parameterized models |
| Flat root files | 45 files, ~5,200 LOC | **Package** — into 10 domain packages |
| `search/` | 37 files, ~4,700 LOC | **Carve** — into 6 sub-packages |

## 1. Measurements

| Metric | Value |
| --- | --- |
| Total Python files | 206 |
| Total LOC (`boltbeam/`) | 34,693 |
| Flat root files (`boltbeam/*.py`) | 45 (27% of all files) |
| Files in packages | 141 |
| Sub-packages | 22 |
| CLI command handlers | 68 (all in one file) |
| Top 2 files by LOC | `cli.py` (1,773) + `model.py` (1,492) = 82% of top-3 |
| Versioned perf files | 15 (`invocation_v1`–`v8`, `scheduling_v2`–`v7`) |
| `search/` files | 37 (~4,700 LOC) |

The two numbers that matter most: **45 flat files** and **15 versioned files**.
These are not symptoms of growth — they are evidence of two distinct anti-patterns
that, if left to accumulate, will create the next god file.

## 2. `perf/` — versioned duplication

### 2.1 The defect

`boltbeam/perf/` contains:

```
invocation_v1.py  invocation_v2.py  invocation_v3.py  invocation_v4.py
invocation_v5.py  invocation_v6.py  invocation_v7.py  invocation_v8.py
scheduling_v2.py  scheduling_v3.py  scheduling_v4.py  scheduling_v5.py
scheduling_v6.py  scheduling_v7.py
```

Each file defines the same interface — `fit_*`, `freeze_*`, `predict_*` — with
incremental changes to the parameter set. The `__init__.py` exports all 8
invocation versions and 6 scheduling versions into the public API. A reader
cannot tell which version is current without reading the import chain.

This is not evolution; it is duplication. The model has changed 8 times, but
the interface is the same. The only thing that varies is the parameter vector
and phase list.

### 2.2 Proposed structure

```
boltbeam/perf/
  __init__.py              # exports only current version + factory
  invocation_model.py      # InvocationModel(dataclass) with version: int field
  invocation_registry.py   # Versioned fit/freeze/predict dispatch
  scheduling_model.py      # SchedulingModel(dataclass) with version: int field
  scheduling_registry.py   # Versioned dispatch
  calibration.py           # (unchanged)
  cycle_model.py           # (unchanged)
  microbench.py            # (unchanged)
  # ... rest unchanged
```

Pattern:

```python
@dataclass(frozen=True)
class InvocationModel:
    version: int
    params: dict[str, float]
    phases: list[str]

def fit_invocation(data, *, version: int = LATEST) -> InvocationModel: ...
def predict_invocation(model: InvocationModel) -> ...
```

Backward compat: the old `invocation_v8` names become aliases in `__init__.py`
pointing to the unified model with explicit version. Every existing import
continues to work.

### 2.3 Risk

**High** — behavior change. This is the only phase where a refactor could
silently alter numerical output. Control: run the full perf test suite
(`boltbeam/perf/tests/`) against each version independently before and after.

### 2.4 Explicitly rejected

**Keep the versioned files for audit trail.** Version control already provides
the audit trail. Keeping 15 files in the active codebase for historical
reference is the same as committing `lib/` — it costs more than it earns.
If historical versions need to be runnable, archive them under
`perf/archive/` with a `README.md` explaining the version lineage.

## 3. Flat root files — orphan accumulation

### 3.1 The defect

45 Python files sit directly in `boltbeam/` with no package home. Examples:

| File | LOC | Logical home |
| --- | ---: | --- |
| `matched_control.py` | 585 | `control/` |
| `replay_ab.py` | 405 | `control/` |
| `runner_plan.py` | 394 | `plan/` |
| `roofline_plan.py` | 257 | `roofline/` |
| `prefill_roofline.py` | 262 | `roofline/` |
| `quant_gemv.py` | 332 | `quantization/` |
| `hw_trace.py` | 320 | `trace/` |
| `timing.py` | 321 | `trace/` |
| `gpu_health.py` | 95 | `runtime/` |
| `amd_runtime_bridge.py` | — | `runtime/` |

These are not one-off scripts. They are production modules that happen to
share a flat namespace. The result: `boltbeam/*.py` is a directory with no
taxonomy, making it impossible to predict where a module lives without
`grep`.

### 3.2 Proposed target packages

```
boltbeam/roofline/          # roofline_plan, roofline_ceiling, roofline_trace,
                            # prefill_roofline, prefill_roofline_ladder
boltbeam/control/           # matched_control, replay_ab
boltbeam/runtime/           # amd_runtime_bridge, direct_kfd_bridge,
                            # kfd_observation_bridge, execution_bridge,
                            # process_isolated, gpu_health
boltbeam/trace/             # hw_trace, timing, timing_compare, schedule_trace,
                            # trace_enrich, substrate_compare
boltbeam/memory/            # mem_hierarchy, mem_sweep, lds_l2
boltbeam/quantization/      # quant, quant_gemv
boltbeam/vocabulary/        # vocab (schemas+verdicts), vocab_capture, model_vocab
boltbeam/plan/              # runner_plan, boundary, resolved_target
boltbeam/experiment/        # reduce_source, reconciliation, observation_protocol,
                            # decode_counter_evidence, decode_decay,
                            # decode_resource_evidence, prefill_authority,
                            # prefill_role_trace
boltbeam/target/            # targets, tinygrad_root
```

`analyze.py` (267 LOC) stays at root — it is the top-level bundle emitter and
an authority point per the coding principles.

### 3.3 Risk

**Low** — mechanical moves + import updates. Control: run the test suite after
each package move; any import error surfaces immediately.

## 4. `search/` — too broad for one package

### 4.1 The defect

37 files in `boltbeam/search/` spanning 4+ distinct subsystems:

- Search space emission (`emit.py`, `spec.py`, `families.py`)
- Full kernel search (`full_kernel_*.py`)
- MMQ diagnosis (`mmq_*.py`)
- Semantic search / machine research (`semantic_*.py`, `mr7–mr13`)
- Epoch modeling (`epoch_*.py`)
- Evidence joins (`*_join.py`)

A reader entering `search/` has no way to predict where a module lives without
reading every filename.

### 4.2 Proposed sub-packages

```
boltbeam/search/
  __init__.py
  emit.py                  # stays flat — core, small
  spec.py                  # stays flat — core, data-driven (647 LOC, acceptable)
  families.py              # stays flat — core
  util.py                  # stays flat — core
  semantic/
    campaign.py, candidate.py, candidate_plan.py, identity.py
    population.py          # merge semantic_population_export + mr8
    mr7.py, mr9.py
  full_kernel/
    candidates.py, controller.py, search.py, tinygrad.py
  mmq/
    bundle.py, controller.py, diagnosis.py, epoch_oracle.py, prediction.py
  epochs/
    model.py, join.py
  joins/
    resource.py, r4_evidence.py, timing.py
```

Stays flat (small, core): `reachability.py`, `role_cost_ranking.py`,
`primitive_template.py`, `research_categories.py`, `transfer_contract.py`.

### 4.3 Risk

**Medium** — search is heavily cross-referenced. Control: one sub-package per
commit, full test pass after each.

## 5. Acceptance criteria

Across all phases:

1. `pyproject.toml` is unchanged (entry point still resolves).
2. All existing tests pass without modification. If a test needs editing, the
   refactor changed behavior and is wrong.
3. No file in `boltbeam/` is a flat orphan after Phase 3.
4. No versioned file (`_v{N}.py`) exists in `perf/` after Phase 2.
5. No sub-package in `search/` has more than 8 files.
6. `model.py` is untouched.
7. `boltbeam --help` output is byte-identical before and after (from Phase 1).

## 6. Risks and controls

| Risk | Phase | Control |
| --- | ---: | --- |
| CLI command silently dropped | 1 | `--help` diff, criterion 7 |
| Perf numerical output changes | 2 | Full perf test suite, per-version |
| Import break on package move | 3 | Test suite after each move |
| Circular imports in search sub-packages | 5 | One sub-package per commit |
| Reviewer cannot verify large diff | All | One commit per logical unit; each independently green |

## 7. Sequence

| Phase | Description | Files changed | Risk | Days | Dependency |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Split `cli.py` into `cli/` package | 1 → 6+ | Low | 2 | None |
| 2 | Collapse versioned perf models | 15 → 4 | High | 3 | Phase 1 (stable CLI) |
| 3 | Package flat root orphans | 45 → 10 packages | Low | 3 | None (parallel with 1) |
| 4 | Split `kernel_analysis/model.py` | 1 → 8 | Low | 1–2 | Phase 3 (imports settled) |
| 5 | Carve `search/` into sub-packages | 37 → 6 sub-packages | Medium | 3 | Phase 3 (orphans settled) |

**Total: ~12–14 days** for a single developer. Each phase is independently
revertable and independently green. If effort is abandoned after Phase 1, the
tree is in a materially better state.

## 8. What this scope deliberately does not do

This is the negative space — the modules that are already well-structured and
should not be touched:

- **`artifacts/`, `collectors/`, `adapters/`** — narrow execution surfaces,
  clear ownership.
- **`profile/`** — modular with clear `ProfileIR` boundary.
- **`policy/`** — small (600 LOC), 4 files, clean separation.
- **`cache/`, `ledger/`, `math/`, `characterize/`** — appropriately small.
- **`eval/`** — contracts/evaluator/thresholds is a clean split.
- **`synth/`** — single file, appropriately scoped.
- **`core/`** — experiment/facts/requirements/system — clean abstraction.
- **`report/`** — HTML/markdown/XML/policy_consistency — fine as-is.
- **`workflow/`** — 7 run functions, clear orchestration boundary.
- **`diagnostics/`** — small, focused.
- **`pyproject.toml`, `.githooks/`, `tools/`, `docs/`** — non-source, leave
  alone.

The median file is ~129 lines and no defect has been measured in them.
Reorganizing them would be premature optimization of the directory structure.

## 9. Divergence from Claude's scope

| Area | Claude | Qwen |
| --- | --- | --- |
| `cli.py` | Split into `cli/` (6 modules) | Split into `cli/` (15 modules) — more granular domain boundaries |
| `model.py` | Leave alone | Leave alone |
| `perf/` versioned files | Not addressed | Collapse into 2 parameterized models |
| Flat root orphans | Not addressed | Package into 10 domain packages |
| `search/` breadth | Not addressed | Carve into 6 sub-packages |

The divergence is not disagreement — Claude's scope is correct for the two
files it examines. Qwen's scope extends to the remaining structural debt that,
if left unaddressed, will create the next round of god files.

## 10. Recommendation

Execute Phase 1 immediately — it has zero risk and is independently green.
Phases 2–5 can be scheduled in parallel with development work, one phase per
sprint cycle. The tree should not accumulate more orphans or versions while
these are in flight; any new file should land in its logical package.
