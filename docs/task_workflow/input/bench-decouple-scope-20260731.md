# `bench/` decouple scope

Status: proposed — executable, one decision embedded
Scoped: 2026-07-31
Baseline: `7b04c13` (main), `cd75287` (dev/exp)
Prerequisite met: `tools/sync_branches.sh` and the apparatus-commit mechanism
exist, so pruning the trunk no longer risks deleting from `dev`/`exp`.

## 0. What `bench/` actually is

Measured before scoping, because the obvious assumption was wrong twice.

| | |
| --- | ---: |
| Tracked entries under `bench/` | 96 |
| …referenced by any code in `boltbeam/`, `tests/`, `tools/` | **0** |
| Distinct `bench/` paths referenced *by* code | 29 |
| …of those, present on disk | **0** |

The two populations do not overlap at all. Everything tracked is evidence that
nothing reads. Everything the code names is absent from the repository.

### Consequence 1 — `bench/` holds no live product data

The initial concern that pruning `bench/` would break the product was wrong.
`boltbeam/search/util.py:15` names
`bench/qk-search-spaces/topology_grammar_v1.json`, but:

```python
def grammar_max_candidates(default: int = 64) -> int:
  try:
    return int(json.load(open(_GRAMMAR)).get("max_candidates", default))
  except Exception:
    return default
```

Read lazily, wrapped in a bare `except`, falling back to a literal. The file has
never been in the repo. The path is decorative.

### Consequence 2 — a defect the decouple exposes but does not fix

**29 code-referenced `bench/` paths do not exist.** Some are write-targets
(`policy/route_manifest.py` creates its output), which is legitimate. Others are
reads with silent fallbacks, which is not: a fallback that cannot be
distinguished from a successful read is a measurement lying quietly.

This is out of scope here — it is a correctness issue, not an organisation one —
but it must be recorded, because deleting `bench/` content could otherwise be
blamed for it later. See §4.

## 1. The split is by content, not by filename

| Category | Count |
| --- | ---: |
| Carries a verdict/status, or is named as one (`*refutation*`, `*blocked*`, `*promotion*`) | 35 |
| Raw measurement, no verdict | 56 |
| Non-JSON or unreadable | 5 |

Prefix does not determine category. `bench/metal-qwen3-8b-20260729/whole-model-ab-refutation.json`
is a verdict living under a measurement-run directory. Any split done by
directory name will misfile roughly a third of the tree.

This is the same failure that produced the wrong perf conclusion three times:
`invocation_v3` and `invocation_v8` look like versions of one thing and are not.
**Read the file; do not infer from the path.**

## 2. Decision: where do verdict artifacts live?

The branch layout (`docs/branch-flow.md`) says a *record of what you learned* is
product and belongs on the trunk; the *apparatus that produced it* is not.

Verdict artifacts are records. They should stay on `main`. Three supporting
reasons, one of them binding:

1. **Binding:** `docs/task_workflow/output/perf-model-verdict-record-20260731.md`
   lives on `main` and cites `bench/mmq-invocation-v{2..7}-validation-20260711.json`
   as the justification for pruning six models. Moving those artifacts off the
   trunk leaves a document on `main` justifying a deletion with evidence not on
   `main`. That is a dangling citation on the one document that authorises a
   destructive act.
2. A verdict is how the project knows what it knows. Deleting the record and
   keeping the code inverts the rule.
3. They are small. The 35 verdict-bearing files are a rounding error against
   the 161 MB of untracked `outputs/`.

Raw measurement runs are apparatus output. They go to `exp`.

## 3. Execution

### 3.1 Classify — one commit, no deletions

Write `tools/classify_bench.py`: for each tracked file under `bench/`, read it
and emit `verdict | measurement | unreadable` with the field that decided it.
Output a manifest to `docs/task_workflow/output/bench-classification-20260731.json`.

Requirements:
- Decide by **content** — a `status`/`verdict` key, or a documented filename
  convention for the ones that carry the verdict in the name.
- For the 5 unreadable entries, say why (not JSON, malformed, binary) rather
  than bucketing them silently.
- The manifest is the evidence for step 3.2. A file may not be moved without a
  row in it.

Commit `[repo] classify bench artifacts by content`.

### 3.2 Prune measurement runs from the trunk — one commit per group

For each group of files classified `measurement`:

1. Confirm `dev` and `exp` hold them (they will; both are ahead of the prune).
2. `git rm` from `main`.
3. Run the suite and the 132-command help diff. Neither should move — nothing
   references these — and if either does, **stop**: it means the
   classification was wrong.
4. Commit `[repo] prune <group> measurement runs from the trunk`, citing the
   manifest row count.

Then run `./tools/sync_branches.sh`, and add the apparatus commit on `dev`
restoring what was pruned — including any constraining test, per the mistake
recorded in `branch-flow.md`.

### 3.3 Leave verdict artifacts in place

No commit. Recorded here so the absence is deliberate rather than forgotten.

### 3.4 Update the citation surface

`perf-model-verdict-record-20260731.md` cites `bench/mmq-*` paths. After 3.2,
confirm every cited path still resolves on `main`. Any that does not was
misclassified — restore it and fix the manifest.

## 4. Explicitly out of scope

**The 29 missing referenced paths.** They are a correctness defect, not an
organisation one, and they predate this work. Recorded here so the decouple is
not later blamed for them.

Worth its own scope. The shape of the fix is probably: distinguish "this data is
absent" from "this data says 64", because `grammar_max_candidates()` currently
cannot. A `bench/` read that silently falls back is the measurement-discipline
equivalent of a test that passes when the assertion never ran.

**`outputs/`** — 161 MB, already gitignored, not in play.

**Deleting anything outright.** Every prune here is a move off the trunk. `dev`
and `exp` retain everything, per `branch-flow.md`.

## 5. Acceptance

1. Every tracked `bench/` file has a classification row citing the field that
   decided it.
2. No file is moved without such a row.
3. All 35 verdict-bearing artifacts remain on `main`.
4. Every `bench/` path cited by a document on `main` resolves on `main`.
5. Suite and 132-command help output unchanged at every commit.
6. `dev` and `exp` retain every pruned file, verified by
   `git cat-file -e origin/dev:<path>` rather than assumed.
7. `./tools/sync_branches.sh --dry-run` reports no unexpected deletions after
   the apparatus commits land.

## 6. Sequence

| # | Step | Risk |
| ---: | --- | --- |
| 1 | Classify by content, emit manifest | none — read-only |
| 2 | Review the manifest, especially the 5 unreadable | none |
| 3 | Prune `measurement` files from the trunk, per group | low — nothing references them |
| 4 | Sync outward, add apparatus commits | low — mechanism is tested |
| 5 | Verify cited paths still resolve | none |

Step 2 is a real gate, not a formality. The classification is the only thing
standing between "prune raw measurement" and "delete the evidence that justified
last week's deletion".
