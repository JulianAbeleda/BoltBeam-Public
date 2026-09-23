# Scope: hw-trace diagnostics — rocprofv3-blind preflight (A) + PMC graph-bypass reason (B)

> **Status: IMPLEMENTED 2026-07-04.** B in `boltbeam/prefill_role_trace.py`
> (`_coverage_reason`), A in `boltbeam/collectors/hw_trace.py` (`_rocprofv3_blind_device` +
> preflight/backstop). Tests: `tests/test_prefill_role_trace.py` (4 new),
> `tests/test_collect_hw_trace_preflight.py` (5 new), fixture
> `tests/fixtures/tinygrad/qwen3_14b_pmc_graph_bypass.json`. Full suite 399 passed. Verified
> end-to-end: A fails in 0.07s with the diagnosis (was ~60s), B emits `pmc_graph_bypass` on the
> real graph-mode trace.


Date: 2026-07-04. Follows `gpu-health-trace-results-20260704.md`. Turns that session's
one-off diagnosis into two BoltBeam features so the next automated run explains itself.
Scoped with a Fable 5 consult; all facts checked against source + the real 14B outputs.

## Build order & totals
1. **Feature B first** — pure offline function, testable against fixtures already on disk, no GPU.
2. **Feature A second** — collector preflight; prevents the ~60s dead-end run.

Total: 2 files (`boltbeam/prefill_role_trace.py`, `boltbeam/collectors/hw_trace.py`), ~55–75 LOC + tests.

## Key facts that shape the design
- The collector **pins** `env["DEV"]="AMD"` for both tinygrad samplers, applying `--env`
  overrides *after* (`collectors/tinygrad_rocprof.py:68-76`). So "is this run rocprofv3-blind"
  is a **static property of `cfg.env["DEV"]`** — no runtime probe needed.
- `route_flags` lives at `trace["metadata"]["route_flags"]` (top-level is `null`); values are
  strings `"1"/"0"` or `null`. `prefill_role_trace_report` currently ignores `metadata` entirely.
- Real signal is stronger than "gemm counters empty": in `hw_trace_native_pmc.json`, PMC
  populated 6/14 eager `norm` rows but 0/6 `gemm` rows. "PMC live on non-gemm, dead on gemm" is
  the corroborator that separates `pmc_graph_bypass` from PMC-simply-off. `PREFILL_GRAPH_GEMM=0`
  does **not** change this (`hw_trace_pmc_eager.json` still 0/6 gemm).
- Precedent for typed reasons: `diagnostics/pathology.py` (`PathologyReport.reason/confidence`).
  Mirror its field shape, but it consumes `EvidenceRow`s, not traces — do **not** route through it.
  `_counter_coverage` (`prefill_role_trace.py:134`) is the right and only place to extend.

## Feature B — PMC-coverage diagnostic (`pmc_graph_bypass`)
**Where:** extend `_counter_coverage` / add `_coverage_reason(trace, kernels, coverage)` in
`prefill_role_trace.py`; wire into the `counter_requirements` block (L164-178); one markdown line
after L210. No CLI change — rides the existing `prefill-role-trace` report.

**Fire `pmc_graph_bypass` when** (only if `coverage["complete"]` is False):
`_on(rf,"PMC") and _on(rf,"PROFILE")` AND all `kind=="gemm"` role rows have empty `counters`
AND some non-gemm row in `trace["rows"]` has counters (corroborator). `_on` = `str(rf.get(k))=="1"`.

**Reason vocabulary:** `pmc_graph_bypass` | `pmc_disabled` (neither flag on, counters missing) |
`null` (complete or unclassified). Missing `metadata` (llama/older traces) → `{}` → `null`, no crash.

**Output** (added to `counter_requirements`):
```json
"reason": "pmc_graph_bypass",
"reason_detail": {"pmc": true, "profile": true,
  "gemm_rows": 6, "gemm_rows_with_counters": 0, "nongemm_rows_with_counters": 6,
  "hint": "PMC emitted only from eager AMDProgram.__call__; prefill GEMMs run via HCQGraph.__call__ which bypasses it. Counter coverage for gemm roles is unattainable on this path."}
```
Markdown: `- counter coverage reason: \`pmc_graph_bypass\` (6 gemm rows, 0 with counters; PMC live on 6 non-gemm)`.
Keep `reason` absent/`null` when `complete` is True so existing asserts don't churn. `--gpu-health` irrelevant (offline).

**Size:** 1 file, ~30-40 LOC.

## Feature A — rocprofv3-blind preflight in the collector
**Where:** `collectors/hw_trace.py`, tinygrad branch, right after `cfg = _resolve(args)` (L50),
before dispatching `backend-csv` (L54-55). Add `_preflight_backend_compat(cfg, sampler)`.

**Signal (static, no subprocess):** fire when `sampler=="backend-csv"` AND
`cfg.env.get("DEV","").upper()=="AMD"` (name it a constant to allow future KFD-bypass backends).
Raise `ValueError` (already caught at `hw_trace.py:110` / `cli.py:391` → exit 2). Independent of
`--gpu-health` — a healthy GPU is still rocprofv3-blind here. **llama branch (L30-38) untouched.**

**Why not the subprocess probe:** `cfg.env["DEV"]` already answers it; a `find_spec('ops_hip')`
probe costs a venv spawn and adds a spurious failure mode (would wrongly block a legit `DEV=HIP`
user). Backstop instead: if the empty-CSV `FileNotFoundError` (`tinygrad_rocprof.py:136`) still
fires under `sampler=="backend-csv"` and `DEV=AMD`, map it to the same diagnosed `ValueError`.

**Message** (preflight + backstop converge; note native-PMC is NOT a `--sampler`):
```
rocprofv3/backend-csv cannot trace tinygrad with DEV=AMD: the AMD backend submits via /dev/kfd
and bypasses the ROCr HSA/HIP runtime that rocprofv3 instruments (0 kernels captured).
Use --sampler tinygrad-profile-events for per-kernel timing, or the native tinygrad PMC path
(extra/qk/prefill_boltbeam_trace.py --hw-trace) for counters. To force rocprofv3 anyway, pass --env DEV=HIP.
```
**Size:** 1 file, ~20-30 LOC.

## Test plan
**B — `tests/test_prefill_role_trace.py`** (extend existing coverage class): `pmc_graph_bypass`
reason fires (gemm empty + norm-with-counters); no bypass when gemm counters present (`null`);
PMC-off is not bypass (`null`/`pmc_disabled`); missing-metadata safe. New fixture
`tests/fixtures/tinygrad/qwen3_14b_pmc_graph_bypass.json` = trimmed `hw_trace_native_pmc.json`
(2 gemm + 2 norm, one norm keeping counters). Add a markdown-line CLI assertion.

**A — `tests/test_hw_trace.py`** (or new `test_collect_hw_trace_preflight.py`): AMD+backend-csv
preflights before any subprocess (assert capture never called); `--env DEV=HIP` bypasses preflight;
llama provider unaffected; backstop maps the raw `FileNotFoundError` to the diagnosed message;
orthogonal to `--gpu-health=off`. Existing fixtures lack `metadata.route_flags` — B needs the new one.
