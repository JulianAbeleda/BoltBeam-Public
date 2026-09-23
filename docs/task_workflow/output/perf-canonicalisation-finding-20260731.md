# Perf `_v{N}.py` canonicalisation — research finding

Status: research only, no source changes
Executed: 2026-07-31
Input: `../input/outstanding-fixes-scope-20260731.md` §5.2 step 2
Scope: `boltbeam/perf/invocation_v2..v8`, `boltbeam/perf/scheduling_v2..v7` (13 files)

## Verdict

**No — the inputs are not the same evidence, for all 13 versions.** Every
`fit_*` function gates on a distinct, hand-checked upstream schema string
(`artifact.get("schema")!="tinygrad.mmq_..."`), each naming a different
generated microbenchmark/probe design. No two versions accept the same
producer. The dataclass fields differ in kind, not just in name, because each
version encodes a different statistical technique (linear interaction fit,
zero-dof exact lookup, sample-percentile pooling, contrast-of-difference,
factorial cell table). There is no canonical shape to define, and collapsing
would force a rewrite of all 13 dedicated test files, which directly violates
the acceptance criterion this scope inherits from the prior scope. The
recommendation is: leave all 13 separate, keep the no-new-file test
(`tests/perf/test_no_new_versioned_files.py`) as the enforcement mechanism for
the future, and treat the real duplication — the repeated freeze/tempfile
boilerplate — as a separate, already-legal internal refactor.

---

## A. Are the inputs the same evidence?

**No**, for all 13 versions, across both families. Evidence below is organized
per-family, per-version, per the method in the task.

### A.1 `invocation_v2..v8`

| Ver | `fit_` signature | Primary input's required `schema` literal (verbatim) | What the body reads off it |
| --- | --- | --- | --- |
| v2 | `fit_invocation_v2(artifact)` | `"tinygrad.mmq_invocation_v1.generated_host_factorial.v1"` | `artifact["host_fits"]` → per-phase `(intercept_ns, per_false_site_ns, r2)`; `artifact["rows"]` → `row["phases"][phase]["overhead_corrected_median_ns"]` |
| v3 | `fit_invocation_v3(interaction, shaped)` | `interaction`: `"tinygrad.mmq_invocation_v2.generated_host_interaction.v1"` | `interaction["interaction_fits"][phase]["coefficients_ns"]` (4-term bilinear: a,b,c,d), `["measured_domain"]["base_achieved_uops"]`, `interaction["rows"]`. `shaped["host_fits"]` (same shape as v2's own input) supplies the *non-interaction* phases' `intercept_ns` |
| v4 | `fit_invocation_v4(topology, shaped)` | `topology`: `"tinygrad.mmq_invocation_v3.generated_host_topology.v1"` | requires `topology["cells"]==4`, every `topology_interaction_fits[...]` to be `degrees_of_freedom==0`/`saturated_design==True` (i.e. **no fitting at all** — exact 4-cell lookup table); stores `topology["rows"]` verbatim. `shaped["host_fits"]` again supplies fixed phases |
| v5 | `fit_invocation_v5(artifact)` | `"tinygrad.mmq_invocation_v5.grouped_predicate.v1"` | `artifact["topology_admission"]["actual_deltas"]["grouped_255"]` must equal an exact dict `{"AND":256,"CMPNE":64,"INDEX":255,"STORE":255}`; `artifact["rows"]` keyed by `generated_id` suffix `"baseline"`/`"grouped_255"` — computes a **difference of two medians**, not a fit |
| v6 | `fit_invocation_v6(artifact)` | `"tinygrad.mmq_invocation_v5_scaffolding.exact_histogram.v1"` | `artifact["rows"][i]["phases"][phase]["samples_ns"]` — pools raw sample arrays and takes 2.5/50/97.5 percentiles (`_quantile`); no coefficients, no residual |
| v7 | `fit_invocation_v7(events, scaffold)` | `events`: `"tinygrad.mmq_builder_event_factorial.v1"` | `events["rows"][i]["quant_group_calls"]/["reduce_calls"]/["writeback_iterations"]/["channels"]["total"]["samples_ns"]` — builds a dict keyed by `f"q{..}.r{..}.w{..}"`, i.e. a **factorial cell table indexed by a 3-tuple of counts**. `scaffold["rows"][i]["phases"][p]["samples_ns"]` for `p in (schedule_creation, warmed_compile_cache_lookup)` — pooled percentiles |
| v8 | `fit_invocation_v8(operand, scaffold)` | `operand`: `"tinygrad.mmq_invocation_v7.writeback_builder.v1"` | `operand["rows"]` keyed by `(style, writeback_iterations)` pairs — `("candidate_shaped",1)`, `("simple",1)`, `("candidate_shaped",256)` — computes **two independent contrast-of-percentile differences** (`_difference`). `scaffold` is used identically to v7's `scaffold` **in usage** but is a *different artifact instance* (no shared schema literal check exists in v8 the way it does for `operand`) |

Field-by-field dataclass comparison (`InvocationModelV{2..8}`):

- v2: `fits: Mapping[str, tuple[float,float,float]]`, `max_fit_residual_ms`, `source_id` — single-source, one 3-tuple per phase (intercept, slope, r2).
- v3: `interaction: Mapping[str, tuple[float,float,float,float]]` (4-term bilinear), `fixed_ms`, `residual_ms`, `base_domain: tuple[int,int]`, `source_ids: tuple[str,str]` — two sources, extrapolation-bounded.
- v4: `rows: tuple[Mapping[str,Any],...]` (raw measured cells, **not coefficients**), `fixed_ms`, `source_ids` — zero-dof lookup, no fit at all.
- v5: `grouped_delta_ms`, `mismatch_uncertainty_ms`, `source_id` — single delta model from a two-row contrast.
- v6: `phase_ms: Mapping[str, Mapping[str,float]]` (only low/median/high bands), `source_id` — pure empirical percentile bands, no coefficients.
- v7: `cells: Mapping[str, Mapping[str,Any]]` (factorial table keyed by 3-int combination string), `gated_schedule_cache`, `source_ids` — table lookup, not a fit.
- v8: `fixed_operand_ms`, `writeback_interaction_ms`, `gated_schedule_cache`, `source_ids` — two independent contrast-difference dicts.

No two of these seven field sets are a superset/subset or a rename of each
other. v4 and v7 don't even contain fitted coefficients (they memoize raw
cells because the underlying design is saturated/zero-dof); v3 and v8 both
have "interaction"-flavored fields but v3's is a 4-coefficient bilinear
regression while v8's is a plain difference-of-medians — same word in the
docstring ("interaction"), different math.

`predict_` signatures (verbatim):
```
predict_invocation_v2(model, *, candidate_id, binary_sha256, false_sites)
predict_invocation_v3(model, *, candidate_id, binary_sha256, base_uops, false_sites)
predict_invocation_v4(model, *, candidate_id, binary_sha256, base_uops, false_sites)
predict_invocation_v5(model, direct_prediction, *, candidate_id, binary_sha256)
predict_invocation_v6(model, direct_prediction, *, candidate_id, binary_sha256)
predict_invocation_v7(model, direct, *, candidate_id, binary_sha256, writeback_iterations)
predict_invocation_v8(model, direct, *, candidate_id, binary_sha256, gated)
```
v2 has no `base_uops` (device-independent host cost only); v3/v4 have it but
v4 additionally *requires* it be within a measured 2-cell bracket
(`fit_invocation_v4` raises `ValueError` outside `bases[0]..bases[-1]`), which
v3 does not — v3 explicitly extrapolates with an uncertainty term instead
(`extrap=abs(excess*(c[1]+c[3]*false_sites))`). v5/v6/v7/v8 all take a prior
prediction dict (`direct_prediction`/`direct`) and layer a *correction* onto
one phase of it — this is real evidence they form a **correction chain on top
of an earlier direct model**, not seven interchangeable ways to answer the
same question. But the corrections applied differ in kind: v5 adds a fixed
mismatch-bounded delta, v6 overwrites three phases with percentile bands, v7
overwrites `uop_construction` with a factorial-table lookup keyed by
`writeback_iterations`, v8 overwrites `uop_construction` with a
`gated`-boolean-conditioned two-term correction. None of these four
"correction" predict signatures are pairwise identical either.

### A.2 `scheduling_v2..v7`

| Ver | `fit_` signature | Required upstream `schema` (verbatim) | What it computes |
| --- | --- | --- | --- |
| v2 | `fit_scheduling_v2(artifact, cases, *, compute_units=96)` | `"tinygrad.mmq_scheduling_calibration.v1"` | least-squares fit of `SQ_WAVE_CYCLES` on `(1, waves, static_valu*waves, static_salu*waves, resident_batches)`, plus a second least-squares of wall-ms on `(1, cycles)`. Two independent linear fits, ridge-regularized (`_least_squares`) |
| v3 | `fit_scheduling_v3(artifact)` | `"tinygrad.mmq_long_chain_calibration.v1"` | simple OLS slope/intercept of `auto_median_ms` on `per_wave_cycles`, but **only over exactly 3 points** (`chain_length` must be `[128,256,512]` — hard equality check, raises otherwise) |
| v4 | `fit_scheduling_v4(artifact)` | (delegates) | **literally calls `fit_scheduling_v3(artifact)` internally** (`from boltbeam.perf.scheduling_v3 import fit_scheduling_v3`) and repackages the same four numbers under `SchedulingModelV4`, changing only the declared `target_metric` from wall-clock to `"kernel_device_ms"` and requiring `require_kernel_target("kernel_device_ms")`. This is the **one place in the 13 where real code reuse already exists** — v4 is not an independent fit, it is v3's fit retagged for a different metric scope |
| v5 | `fit_scheduling_v5(manifest, cases)` | `"tinygrad.mmq_residual_probe.v4"`, `manifest.get("cases")==16` | reads named coefficients out of a pre-computed `manifest["fit"]["terms"]/["coefficients_ms"]` (`static_store_sites`, `branch_sites`, `lds`, `store_sites_x_lds`) — **no fitting in this function at all**, it validates and extracts. Enforces a numeric-range sanity check (`0.000015 <= false_site <= 0.000021`) that has no analog in any other version |
| v6 | `fit_scheduling_v6(manifest, cases)` | `"tinygrad.mmq_residual_grid_factorial.v5"`, `manifest.get("cells")==24` | extracts a **7-term** coefficient vector (`intercept, workgroups, false_sites, lds, workgroups_x_false_sites, false_sites_x_lds, workgroups_x_false_sites_x_lds`) plus a 95% interval per term, requires `bootstrap_replicates>=1000` |
| v7 | `fit_scheduling_v7(manifest, points)` | `"tinygrad.mmq_exact_isa_residual.v6"`, `points` keyed by exactly `{0,64,128,256}` | single slope+intercept from an **exact** 4-point admitted series (`admitted_points==4`), plus VGPR/SGPR/workgroup-thread resource-scope bookkeeping (`probe_resource_scope`) that no other version has |

Field-by-field dataclass comparison (`SchedulingModelV{2..7}`):

- v2: `system_snapshot_id, static_to_executed_valu, static_to_executed_salu, wave_coefficients(5-tuple), wall_coefficients(2-tuple), relative_error, source_id` — 7 fields, two chained regressions.
- v3: `intercept_ms, ms_per_wave_cycle, max_training_per_wave_cycles, relative_error, blocked_chain_lengths(tuple[int,...]), source_id` — 6 fields, single regression + explicit extrapolation-domain tracking.
- v4: `intercept_ms, ms_per_wave_cycle, max_training_per_wave_cycles, relative_error, source_id` — same 5 field *names* as v3 minus `blocked_chain_lengths`, confirmed by the source to be **v3's numbers verbatim** (`v3.intercept_ms`, `v3.ms_per_wave_cycle`, ... passed straight through). This is the one true duplicate-knowledge case in the whole set, and it is already handled by delegation, not by a copy-pasted file.
- v5: `false_site_ms, lds_offset_ms, false_site_lds_ms, source_id` — 4 fields, no relation to v2-v4's fields.
- v6: `coefficients_ms(7-tuple), intervals_ms(tuple of 7 (low,high) pairs), source_id` — 3 fields, but each is a 7-wide vector; a superset of v5's *conceptual* terms (false_sites, lds, interactions) but structurally a flat positional vector, not named fields — merging v5 and v6 would require the v5 code to unpack v6's positional tuple by index, which is not a rename, it's a schema change.
- v7: `per_false_site_ms, max_fit_residual_ms, source_id, probe_resource_scope(Mapping)` — 4 fields, includes a resource-scope dict with no analog elsewhere.

`predict_` signatures (verbatim):
```
predict_scheduling_v2(graph, *, workgroups, model, candidate_id, binary_sha256)
predict_scheduling_v3(v2_prediction, model, *, compute_units=96)
predict_scheduling_v4(v2_prediction, model)
predict_scheduling_v5(base_prediction, model, *, candidate_id, binary_sha256, false_sites, lds_stage)
predict_scheduling_v6(base, model, *, candidate_id, binary_sha256, false_sites, workgroups=256, lds_stage=True)
predict_scheduling_v7(v5_prediction, model)
```
v2 is the only one that takes an `ISAGraph` directly (independent base
model); v3/v4 both consume `v2_prediction` (confirming v4 is a metric-scope
variant of v3's chain, not an alternative to v2); v5/v6 take a `base`/
`base_prediction` whose producer is untyped at the signature level — nothing
in the codebase pins whether that base is v3's or v4's output, only the test
fixtures construct one by hand. v7 explicitly requires `v5_prediction`,
hard-wiring itself one step further down the chain. This is a **genealogical
pipeline of corrections layered on a base model**, not 6 independent
candidate models for the same fit — but each correction stage still requires
its own, differently-shaped, exactly-validated calibration artifact.

## B. Which versions share a canonical input shape

None of the 13 do, for their **primary** fit input — every one enforces a
distinct `schema` string via `raise ValueError` and every one is satisfied by
a differently-shaped upstream artifact (different key names, different
admitted-cell-count invariants, different required point sets). The one
partial exception is the **secondary** input:

- `invocation_v3` and `invocation_v4` both take a `shaped` argument and both
  read only `shaped["host_fits"][phase]["intercept_ns"]` — this secondary
  input genuinely is the same shape and the same producer contract in both
  places (verified: identical dict access pattern, `shaped.get(...)` never
  checked against a schema literal in either file, so nothing prevents it
  being the literal same artifact). This is the **only** shared-shape finding
  in either family.
- `invocation_v7` and `invocation_v8` both take a `scaffold` argument and both
  read only `scaffold["rows"][i]["phases"][p]["samples_ns"]` for
  `p in ("schedule_creation", "warmed_compile_cache_lookup")` — this is also
  genuinely the same shape.
- `scheduling_v3`/`scheduling_v4` share a fit body outright (v4 calls v3's
  `fit_scheduling_v3`), which is not "same input, different name" — it is
  already-collapsed code with zero duplication to remove.

None of these three cases involve the *primary* fit input (`interaction` /
`operand` / the versioned scheduling artifacts), which is where all the real
modeling difference lives. Sharing a secondary "fixed-phase" or
"schedule-cache-pool" side input does not make the two fits the same model —
`fit_invocation_v3` and `fit_invocation_v4` still require different primary
schemas (`generated_host_interaction.v1` vs `generated_host_topology.v1`) and
produce dataclasses with no common field beyond the name `fixed_ms`/`source_ids`.

## C. Canonical shape, if one exists

No canonical shape can be defined for the *primary* evidence across
versions — there is no typed shape that `interaction`, `topology`, `artifact`
(v5/v6 sense), `events`, `operand`, and the six scheduling calibration
artifacts could all satisfy, because the admission checks (exact cell counts,
exact point sets, exact schema strings, saturated-vs-fitted design) are
mutually exclusive by construction: v4's fit *requires* `degrees_of_freedom==0`
(no residual, zero-dof exact lookup) while v3's fit *requires* a genuine
residual-bearing regression; a value cannot satisfy both.

The two secondary-input cases in §B (`shaped` for invocation v3/v4;
`scaffold` for invocation v7/v8) **could** be given a shared typed shape if
that were useful in isolation — e.g. `HostFixedPhases = Mapping[str,
Mapping[str, float]]` for `shaped["host_fits"]`, and `PooledScaffold =
Mapping[str, list[Mapping[str, Any]]]` for `scaffold["rows"]`. But canonicalizing
only the side input while the primary input remains irreducibly different per
version does not let the `fit_`/`predict_`/`freeze_` bodies collapse into one
parameterized path — the parameterization would still have to branch on which
primary schema arrived, at which point the "collapse" is a dispatch table
over the existing 13 bodies, not a merge of them.

## D. Cost of collapsing

All 13 test files would have to change if any merge were attempted, because
every existing test calls the version-specific function name directly with a
schema-specific fixture:

`test_invocation_v2.py`, `test_invocation_v3.py`, `test_invocation_v4.py`,
`test_invocation_v5.py`, `test_invocation_v6.py`, `test_invocation_v7.py`,
`test_invocation_v8.py`, `test_scheduling_v2.py`, `test_scheduling_v3.py`,
`test_scheduling_v4.py`, `test_scheduling_v5.py`, `test_scheduling_v6.py`,
`test_scheduling_v7.py` — **13 of 13** import and call a version-suffixed
name (`fit_invocation_v3(...)`, `predict_scheduling_v6(...)`, etc.); a merge
into two parameterized functions eliminates those names, so every one of the
13 files needs at minimum an import rewrite, and in most cases (anywhere the
validation logic itself differs — see §A, e.g. v4's bracket-bounds check vs
v3's extrapolation, or v5's numeric-range sanity check) a rewrite of the
assertions too, since the parameterized function's error conditions would
necessarily differ from the current per-version ones.

**Acceptance criterion 2 of the prior scope required that no test need
editing.** A collapse here fails that criterion outright, for all 13 files,
not just some. This is a second, independent argument against collapsing,
on top of the evidence-difference finding in §A.

## E. Recommendation

**Leave all 13 versions separate.** The inputs are not the same evidence —
every primary fit input is gated by a distinct, mutually-exclusive schema
admission check, and the model dataclasses encode genuinely different
statistical techniques (regression vs. exact zero-dof lookup vs. percentile
pooling vs. contrast-of-difference vs. factorial table), not a shared field
set under different names. The one place real duplicate knowledge exists —
`scheduling_v4` reusing `scheduling_v3`'s fit — is already collapsed via a
direct function call, not a copy-pasted file, so there is nothing left to do
there. Forcing a merge elsewhere would violate the "collapse only what step 2
proves identical" instruction in §5.2, would fail acceptance criterion 2 by
rewriting all 13 test files, and — per the governing principle — would trade
cheap, legible duplication for "the wrong abstraction": a dispatch table that
still has to branch on which of the mutually-exclusive schemas arrived, wearing
a parameterized-function costume over the same 13 independent bodies.

## F. What to do instead

The no-new-file rule is already machine-enforced going forward:
`tests/perf/test_no_new_versioned_files.py` pins the exact 13-file
grandfathered set and fails loudly if any `_v{N}.py` file is added under
`boltbeam/perf/` beyond it (confirmed present at HEAD, added under this same
scope's step 1). No further code action is required by this scope; the
enforcement mechanism named in §5.2/§5.3 already exists and already
covers "no new file" for any 14th version.

Two smaller, low-risk opportunities exist if a *future*, separately-scoped
pass wants real cleanup without touching evidence handling — noted here only
because they surfaced during the read, not recommended for execution now:

1. All 13 `freeze_*` functions share byte-identical tempfile/`os.replace`
   boilerplate (differing only in the schema string and dict key names). This
   is genuine duplicate *code*, not duplicate *evidence* — the values they
   validate and persist differ, but the write-atomically-or-clean-up
   mechanics do not. It could be factored into one `_atomic_write_json(path,
   body)` helper without touching any `fit_`/`predict_` signature or test.
2. `scheduling_v4`'s delegation to `scheduling_v3` is a precedent: if a future
   version is a pure metric-relabeling of an existing fit (not a new
   calibration technique), it should call the existing `fit_` function the
   way v4 does, rather than duplicating the body — but this is a coding
   pattern to follow for the *next* version, not a change to make now.

Neither of these is part of this scope (§7 of the input scope: "does not
collapse the versioned perf models") and neither is executed here — this
document is research only.

## What could not be determined

- Whether `invocation_v5`/`v6`/`v7`/`v8`'s `direct_prediction`/`direct`
  argument and `scheduling_v5`/`v6`/`v7`'s `base`/`base_prediction`/
  `v5_prediction` argument are, in current production use, always sourced
  from one specific earlier version (e.g. always v3's output) or could
  legitimately come from either v3 or v4 (invocation) / any upstream stage
  (scheduling). No caller outside the dedicated test files invokes any of
  these `fit_`/`predict_` functions in production code — `boltbeam/perf/__init__.py`
  only re-exports the names — so there is no real call site to trace. The
  test fixtures construct a `direct_prediction`/`base_prediction` dict by
  hand in each case, which establishes the *shape* the correction functions
  expect but not which upstream version is the intended, wired-up producer.
  This is a separate, "is the pipeline actually connected end-to-end"
  question, out of scope for §5.2 step 2 as posed (which asks only whether
  the *fit* inputs are the same evidence).
- Whether `scheduling_v6`'s 7-term coefficient vector is, numerically, a
  refinement/superset of `scheduling_v5`'s 3 named coefficients
  (`false_site_ms`, `lds_offset_ms`, `false_site_lds_ms` vs. terms 2, 3, 5 of
  v6's `["false_sites", "lds", "false_sites_x_lds"]` subset) was not verified
  against real calibration data — only the code paths were compared. The term
  *names* overlap conceptually but v6 additionally requires `workgroups` and
  `workgroups_x_false_sites`/`workgroups_x_false_sites_x_lds`, terms v5 has no
  equivalent for, so even if this were a numeric refinement it is not a
  strict field-rename case.
