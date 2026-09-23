# Phase B duplication audit

Status: complete (report only — no code changed)
Audited: 2026-07-30
Scope: `docs/task_workflow/input/organization-and-pruning-scope-20260730.md`, §Phase B (§219–284)
Authority: `docs/coding-principles.md`, "Reducing Code The Right Way"
Audited at commit: `c547b9dfa2ba0650bace602ece999c85f598d8af` (2026-07-30 21:39:16 -0400), working tree clean

This is read-only. No source file was modified, moved, or deleted to produce it.

**Caveat on `boltbeam/cli/`:** a concurrent agent is actively editing `boltbeam/cli/`
(Phase A.1, `cli.py` → `cli/` package) while this audit ran. At the audited commit,
`cli.py` had already been converted into `boltbeam/cli/__init__.py` (1,707 lines,
68 `cmd_*` handlers, not yet split into the per-domain modules the scope
targets). Findings below that cite `cli/__init__.py` line numbers are pinned to
that commit; line numbers will drift as A.1 proceeds, but the duplication itself
is a property of the handler bodies, which A.1 (a pure move) is not expected to
change.

## 0. Verdict table

| # | Candidate | Verdict | Occurrences | Remedy |
| - | --- | --- | --- | --- |
| 1 | `timing.py:_fingerprint` vs `artifacts/base.py:sha256_json` | same-knowledge | 2 | centralize |
| 2 | `_run_manifest_defaults` skipped by 2 of 4 trace-import handlers | same-knowledge | 2 (of 4 sibling handlers) | centralize |
| 3 | Inline `json.loads(pathlib.Path(x).read_text())` bypassing `_load_json` | same-knowledge | 4 confirmed (+ ~80 conforming call sites for contrast) | centralize |
| 4 | MR7/MR8/MR9 raw `write_text(json.dumps(..., sort_keys=True))` vs `_write` | same-shape-different-knowledge (4 of 6) / gap (2 of 6) | 6 | leave (4), fix omission (2) — see notes |
| 5 | Four trace importers (`import-llama-rocprof`, `import-tinygrad-rocprof`, `import-hw-trace`, `import-ncu`) | **refuted as single table-driven parser** — mixed: 2 already centralized, 2 genuinely distinct | — | leave (see §5 discussion) |

Findings by bucket: **same-knowledge: 3** (#1, #2, #3), **same-shape-different-knowledge: 1** (#4, split verdict), **divergent-inputs: 0** confirmed as a standalone finding (see §6, checked-and-clean).

---

## 1. `timing.py:_fingerprint` vs `artifacts/base.py:sha256_json`

```
files+lines:   boltbeam/timing.py:111-113, boltbeam/artifacts/base.py:35-44
signatures:    def _fingerprint(obj:dict[str, Any]) -> str:                 # timing.py:112
               def sha256_json(obj:Any) -> str:                            # artifacts/base.py:43
                 (plus artifacts/base.py:35 def _canon(obj:Any) -> str:)
```

Bodies:

```python
# boltbeam/timing.py:112-114
def _fingerprint(obj:dict[str, Any]) -> str:
  raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
  return "sha256:" + hashlib.sha256(raw).hexdigest()

# boltbeam/artifacts/base.py:35-44
def _canon(obj:Any) -> str:
  return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def sha256_bytes(data:bytes) -> str:
  return "sha256:" + hashlib.sha256(data).hexdigest()

def sha256_json(obj:Any) -> str:
  return "sha256:" + hashlib.sha256(_canon(obj).encode()).hexdigest()
```

```
verdict:       same-knowledge
proof:         Identical rule: canonical-JSON fingerprint = sha256 of
               json.dumps(obj, sort_keys=True, separators=(",", ":")), prefixed
               "sha256:". Same input type (arbitrary JSON-able dict), same
               output format, same call site shape (trace["fingerprint"] =
               _fingerprint(trace), timing.py:85). artifacts/base.py does not
               import timing.py and timing.py does not import artifacts.base for
               anything else, so there is no circular-import reason for the
               duplicate. This is the "rule" (canonical fingerprinting), not
               merely similar-looking code — the two functions produce
               byte-identical output for the same input.
remedy:        centralize — timing.py should import sha256_json from
               boltbeam.artifacts.base and delete its private _fingerprint.
```

This directly answers the scope's named hunting ground (§B.4, sha256 in
`timing.py`, `perf/validation.py`, `perf/cycle_model.py`,
`kfd_observation_bridge.py`). See §6 below: three of those four sites do **not**
compute a hash at all — they only pass `binary_sha256` around as an opaque
identity string. Only `timing.py` and `perf/cycle_model.py` (via the already-
centralized `sha256_json`) compute a hash. The real duplication is narrower and
different from what the hunting ground implied: it is `timing.py` reinventing
`artifacts/base.py:sha256_json`, not four independent sha256 computations.

---

## 2. `_run_manifest_defaults` not used by 2 of 4 sibling import handlers

```
files+lines:   boltbeam/cli/__init__.py:402-412 (cmd_import_llama_rocprof),
               boltbeam/cli/__init__.py:433-443 (cmd_import_tinygrad_rocprof),
               boltbeam/cli/__init__.py:464-468 (cmd_import_hw_trace, uses helper),
               boltbeam/cli/__init__.py:513-515 (cmd_import_ncu, uses helper)
signatures:    def cmd_import_llama_rocprof(args) -> int:
               def cmd_import_tinygrad_rocprof(args) -> int:
               def cmd_import_hw_trace(args) -> int:
               def cmd_import_ncu(args) -> int:
               def _run_manifest_defaults(args, default_out:str) -> tuple[dict[str, Any], str | None]:  # cli/_common.py:47
```

Bodies (llama, lines 404-412 — tinygrad, lines 435-443, are byte-for-byte the
same shape modulo the default filename, which is identical too):

```python
run_manifest = {}
out = args.out
if args.run:
  from boltbeam.workflow.common import load_manifest, run_dir
  run_path = run_dir(args.run)
  run_manifest = load_manifest(run_path)
  out = out or str(run_path / "timing_trace.json")
  if args.weight_inventory is None and (run_path / "weight_inventory.json").exists():
    args.weight_inventory = str(run_path / "weight_inventory.json")
```

versus the two conforming handlers, which call the one-line helper that exists
precisely for this:

```python
run_manifest, out = _run_manifest_defaults(args, "hw_trace.json")
```

```
verdict:       same-knowledge
proof:         The inline block and _run_manifest_defaults implement the exact
               same rule — "if --run is given, load its manifest, default --out
               into the run directory, and default --weight-inventory from the
               run's weight_inventory.json if present" — over the same input
               shape (an argparse Namespace with .run/.out/.weight_inventory).
               The only variable between call sites is the default output
               filename ("timing_trace.json" vs "hw_trace.json"), which
               _run_manifest_defaults already parameterizes via its
               default_out argument. This is not a look-alike; it is the literal
               same 6 lines, reproduced instead of called.
remedy:        centralize — replace both inline blocks with
               _run_manifest_defaults(args, "timing_trace.json").
```

---

## 3. Inline JSON-file loads bypassing `_load_json`

```
files+lines:   boltbeam/cli/__init__.py:64  (cmd_schedule_trace)
               boltbeam/cli/__init__.py:71  (cmd_roofline)
               boltbeam/cli/__init__.py:78  (cmd_roofline_plan)
               boltbeam/cli/__init__.py:235 (cmd_vocab_capture)
signatures:    _write(ingest_schedule_trace(json.loads(pathlib.Path(args.trace).read_text())), args.out)
               _write(ingest_roofline(json.loads(pathlib.Path(args.trace).read_text()), peak_gbs=args.peak), args.out)
               p = json.loads(pathlib.Path(args.plan).read_text())
               cap = json.loads(pathlib.Path(args.capture).read_text())
               def _load_json(path:str) -> Any: return json.loads(pathlib.Path(path).read_text())  # cli/_common.py:66
```

```
verdict:       same-knowledge
proof:         `_load_json` is imported into cli/__init__.py and used at ~80
               other call sites in the same file for the identical operation:
               read a path, parse it as JSON, return the object. These four
               sites re-derive the identical expression
               (json.loads(pathlib.Path(x).read_text())) instead of calling the
               helper already in scope. Same rule, same input (a path string),
               same output (a dict/list). This satisfies the rule of three (4
               occurrences) and the "genuinely identical" bar — the bypassed
               code is character-for-character what _load_json does.
remedy:        centralize — replace each with _load_json(args.trace) /
               _load_json(args.plan) / _load_json(args.capture). Mechanical,
               behavior-preserving (same exception types propagate either way,
               since neither form catches errors locally at these 4 sites).
```

---

## 4. `_write` bypassed by MR7/MR8/MR9 handlers — mixed verdict

```
files+lines:   boltbeam/cli/__init__.py:774-780 (cmd_semantic_campaign)
               boltbeam/cli/__init__.py:782-786 (cmd_export_semantic_population)
               boltbeam/cli/__init__.py:788-797 (cmd_mr8_population_selection)
               boltbeam/cli/__init__.py:799-806 (cmd_mr7_evidence_plan)
               boltbeam/cli/__init__.py:815-824 (cmd_mr7_provider_run)
               boltbeam/cli/__init__.py:839-850 (cmd_mr9_semantic_search)
signatures:    def _write(obj:dict, out:str | None) -> None:                     # cli/_common.py:14
                 (mkdir parents, write if out else stdout, indent=2, no sort_keys)
               pathlib.Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
```

Four of the six sites (`cmd_mr8_population_selection`, `cmd_mr7_evidence_plan`,
`cmd_mr7_provider_run`, `cmd_mr9_semantic_search`) guard the write with:

```python
output = pathlib.Path(args.out)
if output.exists(): raise FileExistsError(f"... already exists: {output}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
```

```
verdict:       same-shape-different-knowledge (for these 4)
proof:         `_write` encodes an overwrite-tolerant, stdout-fallback contract
               ("if --out given, write there, creating parent dirs; otherwise
               print to stdout"). These 4 handlers encode a different, stricter
               rule: MR7/8/9 evidence-bundle outputs are append-only campaign
               artifacts and must error rather than silently clobber an
               existing ledger entry. That is a distinct policy (no-clobber),
               not an accidental copy of _write with sort_keys added. Forcing
               these through _write would delete the no-clobber guarantee.
remedy:        leave — the no-clobber policy is real and _write does not (and
               should not) implement it. If this pattern recurs a third time
               with the exact same guard, it is a rule of three candidate for a
               `_write_once(obj, path)` helper in cli/_common.py — not yet
               warranted at 4 occurrences of a 4-line block, since the value
               here is the guard clause, and that guard is already
               self-explanatory at each site.
```

The remaining two sites (`cmd_semantic_campaign` line 779,
`cmd_export_semantic_population` line 785) are the odd ones out:

```
verdict:       gap, not duplication — flag, do not remediate under Phase B
proof:         These two lack the FileExistsError no-clobber guard AND lack
               parent.mkdir(parents=True, exist_ok=True) that their 4 siblings
               have. That means they are not a case of the same knowledge
               expressed twice; they are two call sites missing a guard their
               siblings judged necessary. This is either (a) a real
               inconsistency worth fixing as a behavior change (out of scope
               for a Phase B audit, which changes no code) or (b) deliberate,
               because semantic-campaign/population-export outputs are not
               append-only ledger entries the way MR7/8/9 are. Evidence here
               does not distinguish (a) from (b) — recorded as "could not
               determine" in §7.
remedy:        none under Phase B. If (a), it belongs to Phase E (orthogonality
               — inconsistent policy application) or a plain bugfix, not a
               duplication remedy.
```

---

## 5. The four trace importers — settled with evidence

The scope calls this "the strongest table-driven candidate in the tree" and
asks whether `import-llama-rocprof`, `import-tinygrad-rocprof`,
`import-hw-trace`, `import-ncu` are genuinely different parsers or one parser
with four column maps.

**Answer: neither extreme. Two of the four already share centralized code for
the genuinely common knowledge; the remaining differences are real, and the NCU
importer is categorically different from the other three.**

### 5.1 `llama_rocprof.py` vs `tinygrad_rocprof.py` — partially centralized already

```
files+lines:   boltbeam/artifacts/llama_rocprof.py (388 lines, full file)
               boltbeam/artifacts/tinygrad_rocprof.py (264 lines, full file)
signatures:    def timing_trace_from_rocprof_csv(path, *, model_id, target_id="amd_gfx1100",
                 workload="prefill", context=None, provider_id="llama.cpp/rocprofv3",
                 peak_gbs=960.0, llama_bench_json=None, weight_inventory=None,
                 rocprof_json=None, memory_copy_csv=None) -> dict[str, Any]
               def timing_trace_from_tinygrad_rocprof_csv(path, *, model_id, target_id="amd_gfx1100",
                 workload="prefill", context=None, provider_id="tinygrad/rocprofv3",
                 peak_gbs=960.0, tinygrad_trace_json=None, weight_inventory=None,
                 memory_copy_csv=None) -> dict[str, Any]
```

`tinygrad_rocprof.py:9-12` imports directly from `llama_rocprof.py`:
`_apply_weight_bytes, _duration_us, _load_llama_bench, _normalize_to_bench,
resource_fields_from_backend_row, _rocprof_transfer_rows`. That is: CSV
duration parsing, resource-field extraction, bench-time normalization, and
memory-copy/transfer-row handling are **already centralized in one place and
imported, not copy-pasted.** That is the correct remedy already applied for
the knowledge that is genuinely shared.

What is *not* shared, and should not be:

- `classify_llama_kernel` (llama_rocprof.py:83-122) pattern-matches
  llama.cpp/ggml kernel names (`mul_mat_q<`, `dequantize_block_q6_k`,
  `Cijk_...`, `rms_norm`, ...).
- `classify_tinygrad_kernel` (tinygrad_rocprof.py:69-130) pattern-matches
  BoltBeam-generated tinygrad kernel names via 4 dedicated regexes
  (`_PREFILL_GEMM_RE`, `_PREFILL_DIRECT_PACKED_RE`,
  `_PREFILL_GENERATED_DIRECT_OUT_RE`, `_PREFILL_Q4K_WMMA_RE`) that encode
  BoltBeam's own kernel-name grammar, not llama.cpp's.
- Byte attribution differs by *rule*, not just input: `_apply_weight_bytes`
  (llama_rocprof.py:149-169) attributes bytes by **quant name**;
  `_apply_role_bytes` (tinygrad_rocprof.py:143-168) attributes bytes by
  **(N,K) shape** via a `_shape_index` built from the weight inventory
  (tinygrad_rocprof.py:36-66, with no llama-side equivalent). These are
  different attribution algorithms because llama.cpp kernel names carry a
  quant tag and tinygrad's do not; tinygrad's carry a shape instead.

```
verdict:       same-knowledge (for the imported helpers — already centralized,
               not a finding) / same-shape-different-knowledge (for
               classify_*_kernel and the byte-attribution functions)
proof:         classify_llama_kernel and classify_tinygrad_kernel take the same
               input type (a kernel name string) but decode two unrelated
               grammars (llama.cpp/ggml/rocBLAS symbol names vs
               BoltBeam-authored tinygrad kernel names matched by
               BoltBeam-specific regexes). Merging them would require an
               if-provider branch inside a single function, which is exactly
               the "vague wrapper hiding where authority lives" anti-pattern
               named in the scope's Phase D — not a table with one column map.
remedy:        leave. The already-centralized helpers are the correct extent
               of sharing; do not force the classify/attribution functions
               together.
```

### 5.2 `import-hw-trace` — a composition wrapper, not a third parser

`cmd_import_hw_trace` (cli/__init__.py:464-510) does not implement its own CSV
parsing. It calls whichever of `timing_trace_from_rocprof_csv` /
`timing_trace_from_tinygrad_rocprof_csv` matches `--provider`, then pipes the
result through `timing_trace_to_hw_trace` (`boltbeam/hw_trace.py:128-163`) to
attach `SCHEMA_HW_TRACE` framing and optional backend-counter matching. It is
not a fourth column map; it is the two `artifacts/*_rocprof.py` importers
composed with one schema-conversion step. Confirmed clean — no duplication to
report here.

### 5.3 `import-ncu` — genuinely a different importer

```
files+lines:   boltbeam/profiler/importers/ncu.py (full file, 557 lines)
signatures:    def import_profiler_ncu_csv(path, *, model_id, target_id="nvidia_unknown",
                 workload="decode", context=None, provider_id="nvidia/ncu") -> dict[str, Any]
```

NCU produces `SCHEMA_HW_TRACE` directly (never `SCHEMA_TIMING_TRACE`), and its
input shape is fundamentally different: NCU CSVs may be **long-form**
(one-metric-per-row, `_pivot_long_ncu_rows`, ncu.py:239-309) requiring a pivot
step the rocprof importers never need, and column names are free-form enough
to need fuzzy alias resolution (`_canon_key`, `_COUNTER_ALIASES`,
`_RESOURCE_SCALAR_ALIASES`, `_consume`) rather than the small fixed set of
column names rocprofv3 emits (`Name`/`Kernel_Name`/`TotalDurationNs`/etc., read
positionally in `llama_rocprof.py`/`tinygrad_rocprof.py`).

```
verdict:       same-shape-different-knowledge
proof:         Different schema target, different input topology (long-form
               pivot vs fixed-column CSV), different column-resolution strategy
               (fuzzy alias sets over arbitrary vendor headers vs a short fixed
               list of known rocprofv3 header names). "One parser, four column
               maps" does not hold: NCU's importer is structurally a different
               kind of parser (schema-normalizing + long-to-wide pivot), not a
               column-rename of the rocprof importers.
remedy:        leave.
```

**Bottom line on the named hunting ground:** refuted as stated. It is not "one
parser, four column maps." It is: two rocprof-CSV importers that already share
their common primitives via direct import (correct), one composition wrapper
over those two (not a parser at all), and one structurally distinct importer
for a different input topology and target vocabulary. No remediation is
warranted for the importer boundaries themselves.

---

## 6. Checked and found clean (negative results)

- **`sha256` across `timing.py` / `perf/validation.py` / `perf/cycle_model.py`
  / `kfd_observation_bridge.py`** (named hunting ground): refuted as stated.
  `perf/validation.py` and `kfd_observation_bridge.py` never call a hash
  function — `binary_sha256`/`source_sha256` are opaque identity strings they
  compare or require, not compute. `perf/cycle_model.py` computes hashes only
  via the already-centralized `sha256_json` (`boltbeam/artifacts/base.py`,
  imported at `perf/cycle_model.py:8`). Only `timing.py` reimplements the
  algorithm locally — see finding #1, which is the real, narrower duplication
  underneath this hunting ground.
- **`import-hw-trace`** is a composition wrapper over the two rocprof
  importers plus `boltbeam.hw_trace.timing_trace_to_hw_trace`; it introduces no
  new parsing logic and has no duplication to report (§5.2).
- **Counter/resource alias tables** (`boltbeam/hw_trace.py:_COUNTER_ALIASES`
  vs `boltbeam/profiler/importers/ncu.py:_COUNTER_ALIASES`/
  `_RESOURCE_SCALAR_ALIASES` vs `boltbeam/artifacts/llama_rocprof.py:
  resource_fields_from_backend_row`'s `scalar_fields` dict): inspected because
  all three map vendor-specific column-name synonyms onto a small canonical
  vocabulary, and `hw_trace.py`'s `NORMALIZED_COUNTERS` tuple **is** correctly
  centralized and imported by `ncu.py:15`. But the alias-*translation* tables
  themselves are two independent dicts (counters) plus two independent dicts
  (resource scalars) — 2 occurrences each, not 3. Per the rule of three this
  does not clear the bar for abstraction yet, and each pair maps a genuinely
  different vendor vocabulary (rocprofv3 column names vs NCU column names), so
  forcing them into one alias table now would be exactly the "wrong
  abstraction" the principles warn against. Recorded as watch-not-act, not a
  finding.
- **The 13 versioned `perf/` files** (`invocation_v2–v8`, `scheduling_v2–v7`):
  not re-audited per instructions — this is the scope's own worked example of
  the trap (§6 of the input scope) and is out of scope for this report.
- **68 CLI handlers as a class**: the handler bodies themselves (parse args,
  call one subsystem function, `_write` the result) are not duplicated
  knowledge — they are thin adapters over 68 different subsystem calls, which
  is the correct shape per "Keep Public Surfaces Boring" (also the scope's own
  conclusion in A.1). The duplication found here is narrower: specific helper
  bypasses (findings #2, #3), not the handler pattern in general.

## 7. Could not determine

- Whether `cmd_semantic_campaign` / `cmd_export_semantic_population` (finding
  #4, second half) omit the no-clobber guard **deliberately** (their outputs
  are not append-only ledger entries) or **accidentally** (missed when the
  MR7/8/9 guard convention was introduced). The code gives no comment or test
  distinguishing the two, and answering this requires product intent this
  audit cannot infer from the diff alone. Left as a flagged gap, not a Phase B
  finding, per §1.4 of the input scope ("stop and report on contradiction"
  rather than guess).
- Whether `cli/__init__.py` line numbers cited above will still match once the
  concurrent A.1 split lands. The duplication itself (which helper is or is
  not called) is a property of the handler bodies and should survive a pure
  move; line numbers will not.
