# The long-context audit

`boltbeam context-audit RECORD.json --engine E` reads one session's records and reports where its speed goes as the
context grows. It exits 1 when a check fails. The code is `boltbeam/workflow/context_audit.py`. It measures nothing and
names no engine: engine facts are data rows.

## What it reports

| part | what | from |
|---|---|---|
| depth buckets | tok/s, ms per step, tokens per step, acceptance, prompt ms per turn, per quarter of the context and for the session | the session's turn rows (`prefix`, `predicted_n`, `predicted_ms`, `prompt_ms`, `draft_n`, `draft_n_accepted`) |
| host time | ms per step with no GPU work in flight, by part, at each depth; growth, slope (ms per 1k), constant | `host_points`: one per depth, with its trace and configuration |
| host functions | per engine function, CPU ms per step at each depth (constant and growth) and, where a capture covers the depth, the part in GPU-idle gaps, ranked by that part | `host_functions`: the host-function timer's records file and, optionally, captures of the same run (below) |
| memory | used at load, peak, lowest free, growth, largest rise, what the device keeps; the margin | the device's samples (time, used, free), its total, the load and warm-up marks |
| memory ledger | bytes held per role, the least each needs, the gap and its class; the unaccounted line | labelled rows from the engine's log and the model file's header; `shapes`: the model's tensor tables and the run's settings |
| output labels | identical, quality-gated or untested, per setting | `data/engine_patches.json` (`output`), `data/engine_knobs.json` (`bitwise`, `output`, `audit_flags`) |
| expert cache | covered share of the expert bytes on the host; hit rates when the engine logs them | the engine's log lines (its `audit_flags` row) or the two byte counts |

The input record holds `context`, `turns`, `configuration`, `repeats` and any of `memory`, `ledger`, `host_points`,
`host_apart`, `host_functions`, `roles`, `arms`, `expert_cache`.

## Rules

1. **One step rule.** `collectors/time_budget.decode_steps` states it: a decode's steps are its tokens, less the ones
   its prompt pass made (the engine's decode row), less the accepted drafts. The audit reads that rule and the
   engine's row. A record that states its own `tokens_from_prompt` is refused. Turns with no decode step are refused
   by name, one turn at a time. The step count is not yet checked against a trace: the record says `unknown` with the test.
2. **One rule for where a turn sits.** Its prefix plus half its reply: the mean context of its decode (`turn_depth`).
3. **A slope needs two depths, captured alike.** One depth: growth, slope and constant are `unknown`, with the test.
   Two depths: a line, labelled "2 points, no spread". Depths of different configurations, or one traced and one not,
   are refused.
4. **Tolerances come from the record's repeats.** Each repeat row names its source (`from`). A tolerance is the range of `repeats` of that quantity on the
   record's own `configuration` (the repeat-band rule: with no change, a new run lies outside the range of n runs with
   probability 2/(n+1)). Repeats of another quantity or another configuration are refused. Without repeats the check
   fails and says the test: at least two more runs of that configuration. Nobody types a tolerance.
5. **The memory margin is an estimate.** `margin = range of the peak across repeats + largest rise between two samples
   after warm-up`. The first request's growth is already inside the peak, so it is not added again. The check is
   `lowest free over the session >= margin`. The estimate's test: sample at ten times the rate; a larger rise between
   two samples refutes it. Free is what the device reports, so what it keeps for itself (`total - used - free`, read
   per sample) is already out.
6. **The ledger sums to the device's used memory.** `unaccounted = used - sum(rows)` is a line, never dropped. The
   check fails while it is beyond the tolerance, and says what settles it: a measured row for that memory (the
   engine's allocations by owner). A residual row derived to close the sum does not count.
7. **Roles sum to the step.** `remainder = step - sum(roles)` is a line. Negative means the roles overlap (summed
   durations, not wall time).
8. **Every figure is labelled.** `measured` (source), `derived` (formula and inputs), `estimate` (formula and the test
   that would refute it) or `unknown` (the test that would settle it, and no value). The label sits on the figure or
   on its row; a labelled block does not label the rows inside it. An unlabelled input row fails the audit.
9. **The least each role needs comes from the model's shapes.** See Minimum memory below. A held row and its minimum
   give the gap; a role the shapes do not give is `unknown`, with the measurement that settles it.
10. **A failed check refuses the conclusion.** The record's `refused` names the checks; the command exits 1.

## Minimum memory

`workflow/min_memory.py` states the least memory each role needs on the device, from a tensor table cut from the model
file's header (name, dims, type id), its metadata and the run's settings (context, KV and state types, ubatch, draft
depth). The formulas are rows in `data/min_memory.json`, one block per architecture, read by tune's formula reader.
Tensors are sorted by the one classifier (`profile/roles.py`) and sized as the weight ledger sizes them
(`decode_roles._physical_bytes`). No engine is named: what an engine holds is the ledger's held rows.

| class | means | rows today, with the condition each assumes |
|---|---|---|
| exact | the formula is the need | dense weights; output head; lookup tables (0: a step reads one row); KV; indexer keys (one pooled key per pool, raw keys for the open pool and the draft_depth tokens a rollback takes back; unknown when the layers' pool ratios differ); recurrent state (one state and the convolution windows, plus the inputs of the tokens a rollback takes back: assumes rollback by replay, whose time cost is derived, not measured; the 2026-11-14 plan keeps a window plane per token and frees 318 of the 333 MiB); draft input rows (one ubatch: assumes the draft takes the batch a ubatch at a time); a second file's copy of a matrix (0) |
| bounded | a floor; the gap is printed as `<= X` with what sets the rest | experts (0: a decode can run with every expert off the device; what to keep is a speed choice set by the off-device cost per byte and the touch rate); compute buffers (the largest single result of a pass: a step's logits, or one ubatch's indexer scores over every pooled key, the term that grows with context; the engine's graph sets the rest) |
| unknown | the shapes do not give it; the row names the measurement | device context, device reserved, tensors the classifier does not name, any role with no row |

A tensor that a second file (a draft head) holds with the same name and dims is only a candidate copy. It counts as
needed once only when the record cites a measurement that the two are one matrix (`same`: `tensor`, `from`); a
citation without `from` is no citation. When the two copies differ in format, sharing changes what the second file
computes, so the row carries an output label (`output`, `output_why`; untested unless the citation says).

The ledger record's `shapes` lists the cut tables with their settings, and needs `unit_bytes`; `prefix` and `beside`
name a second file loaded with the first. `outside` holds memory the device keeps beyond its used figure. A gap whose
held figure is an estimate is an estimate, with the held row's test. Two more checks fail the audit: an exact gap below
zero (a formula or a held figure is wrong), and a role the shapes need with no held row.

**At 128k** (`tests/fixtures/context_audit/ledger_128k_shapes.json`: the -lv 4 buffer lines at `-ncmoe 5`, the
header tables of the model and its draft head, eight loads of this configuration at 31,113 MiB): exact gaps sum to
1,162 MiB. Indexer keys stored unpooled 357 (+56 in the draft); recurrent state kept as 1 + k copies 333; the draft's
copy of the output head 341 (output untested); draft input rows sized by the batch 75. Bounded: compute at most 140
over its floor. Rows the shapes do not give: the device context (506, measured before any tensor) and the compute
reserve host expert layers add (357, an estimate: +147 with the first, +213 with the first host attention layer in
four load probes at 32k, `min-memory-20261114/probes`; its test is a 128k probe). About 80 MiB stays unaccounted, so
the ledger fails its check.

What it does not do: size anything. `tune` does not read it. Before it may size an expert cache, the rows the shapes
cannot give must be measured on the exact configuration (device context and reserve, growth and the largest step,
the compute reserve at the placement used, the device change per layer moved), predicted used-at-load must match
measured within the repeats on a second model, one with no indexer or recurrent layers, and a second context, and the
cache's output label must move off `untested`.

## Output labels

A setting's label lives in the row that owns it, read by one function (`tune.output_of`): identical only where the
row states `bitwise` and it holds at the value; otherwise the row's own `output`. A row that says neither, and an
environment switch no row names, is `untested`. A command flag no row judges is `unjudged`, unless it is on the
engine's `output_neutral` list (`data/engine_knobs.json`: the port, the context size, the slots, the log level, the
model path). Fail closed: a flag nobody has judged never reads identical.

| setting | label | why |
|---|---|---|
| a patch switch, off | identical | the stock engine |
| the kpool and sparse-direct switches | identical | reply text identical on every turn |
| a knob at a value where its `bitwise` holds (`-ctk`, `-fa`, `-bs` at their defaults; `-ngl` with every layer on the GPU; `--temp 0`) | identical | the row |
| `-ctk` (and `-ctv`), `-fa` at another value | quality-gated | `tune` keeps one only past its agreement gate; a value set by hand has not passed it |
| `-bs` on, `-ub`, `-b`, the checkpoint min-head switch | untested | no gate covers them; `-ub` changed replies at near-ties |
| a knob row with no label (`-t`, `--cache-ram`, the pinned-copy variables) | untested | nothing measured says otherwise |
| the expert cache, a draft KV below f16, host expert placement, speculative decoding, `--temp` above 0 | untested | changes reply text, or not checked |
| a flag on no row and not output-neutral (`--seed`, `--jinja`) | unjudged | nobody has judged it |

Where it shows:
- a run's summary labels every engine environment variable (`engine_env_outputs`), and its setting words mark any
  that is not identical;
- the offload page and the recommended setup say the expert cache's label beside its flag. The flag stays recommended;
- `tune` prints the label beside a knob that is not bitwise;
- the audit's ranking takes run folders as arms. An arm's settings are its recorded environment (every variable) and
  every recorded command flag, read by the one flag parser (`expert_placement.flag_pairs`: `--name=value` split, a
  switch present is on). An arm with an unjudged flag, an untested setting, no recorded command, or no replay turns is
  refused, with the reason. The text output lists each ranked arm with its labels, each refused arm with its reason,
  and the expert cache's coverage and hit rates.

## Host functions: CPU time, and the part that can come off the step

`collectors/host_functions.py` reads the records of the host-function timer (every call of the engine functions its
data rows list, `data/host_functions.json`, per engine and release) and folds them into exclusive time per function
per decode step: a function's own time, less the timed calls inside it, split by the context that ran it (the model's
or the draft's). Time on the decode thread inside no timed function is the `unattributed` row.

**The rule: rank by the part in GPU-idle gaps, not by CPU time; that part is an upper bound on what a fix saves.**
Host work that runs while the GPU is busy costs the step nothing; the part in a GPU-idle gap can come off it, unless the
GPU then waits on something else. GPU busy is kernels and device-side copies (DtoD, memset); a host copy (HtoD, DtoH)
is the calling function's own work. Measured at X16, 128k session, one traced window (2026-11-15): `kv_seq_rm` takes
0.53 ms of CPU a step at 10.7k and at 102.8k; 0.32 and 0.24 of it lie in GPU-idle gaps, the rest overlaps the draft's
kernels. Removing it saved 0.19 ms a step (paired, measured): **0.19 measured saving against an idle part of
0.24-0.32 (upper bound)**. The k-pool input build lies wholly in idle gaps (0.06 ms a step at 10.7k, 0.50 at 102.8k),
and so do its uploads (0.47 at 10.7k, 0.63 at 102.8k: synchronous host copies). That window's capture added 9.4% and
9.2% to the step (traced against untraced, the same turns), over the 5% ceiling, so its rank prints as unqualified.

The audit prints, per depth quarter, the captured turns at their own depth and count, ranked by their idle part with
their CPU time beside; a quarter with no captured turn prints `idle share unknown, test: one traced window`. Captured
and uncaptured turns are never mixed: their CPU rows are fitted as two series (traced, untraced).

The record's `host_functions`: `records` (the timer's file, beside the record), `release` (the rows' key), and
optionally `capture` (the same run's capture exports, a list), `capture_reference` (an untraced run's turn rows for the
same turns: the capture's overhead), `overhead` (`with_ms`, `without_ms`, `source`: the timer's own), `missing` (the
rows the build could not time, as the launch side reported them). Turns carry `span_ns` (the driver's
CLOCK_MONOTONIC); a turn without it is named as left out.

- **Exclusive time from nested calls.** One pass: each call's parent, its context (the nearest call of the row's
  `scope` label; contexts are told apart by first argument and named rarest caller first; a tie is refused), and its
  time less its children's. Records that do not nest are refused. The idle part comes by the same subtraction.
- **Covered turns.** A turn is covered when its decode window lies inside a capture's own start and stop (the
  export's `ANALYSIS_DETAILS`; without it, the capture's first and last kernel or API call) and the capture holds its
  launch calls. A turn covered in part is not covered.
- **Checks that carry weight.** The timer's steps equal the engine's step rule (`time_budget.decode_steps`) on every
  turn; every launch call of the capture in a covered window lies inside a timed record (the two clocks agree: on the
  W1 window, 145,108 of 145,108); a capture that covers no turn fails. `closes` (the rows sum to the window, the idle
  rows to the idle time) is bookkeeping: it holds by construction.
- **Reported, not dropped.** Rows the build could not time (`missing`), timed rows that never fired, turns left out
  (no `span_ns`) and turns skipped (no `start` call) are lines of the output.
- **A cut-short file is refused.** The file ends `#end records=N dropped=D`. No end line (the engine did not exit
  cleanly), dropped > 0 (a thread's buffer, `HOST_FUNCTIONS_MAX` records, filled), a count that differs from N, or a
  torn line is refused, never read as a short run.
- **Overhead.** Disclosed with `collectors/replay_overhead`'s words against the 5% ceiling: the capture's (traced against
  untraced, per covered quarter) and the timer's own (with against without). A rank under a capture whose overhead is
  unknown or over the ceiling is labelled an estimate and printed as unqualified. The hand timer of 2026-11-14 (the
  same hot path) measured +0.61% on the session step against five runs without it (their spread 1.69%); this timer:
  unknown until the paired run.
- **Launch (dev apparatus, `tools/host_functions/`).** `build.resolve(row, build.libraries(binary))` names what the
  build cannot time (a symbol it does not define, or that no library reaches through a dynamic symbol: a call across
  libraries or a virtual table; a row's `off`) and refuses when a `scope` or `start` function is missing.
  `build.build(wrap)` compiles `timer.c` (Linux x86-64; a C compiler; in the user's cache, atomically, keyed by rows,
  source, compiler and flags). `build.launch_env(so, out)` puts it first in `LD_PRELOAD`, keeping a tracer's preload
  behind it; under nsys, launch a wrapper that prepends the timer to the `LD_PRELOAD` nsys set. `LD_PRELOAD` reaches
  every child process: each writes its own `<out>.<pid>.tsv` (one that timed nothing holds only its end line); the
  engine's is the one with the `scope` calls.
- **Reproduced.** From raw records, the hand tables of 2026-11-14 to the third decimal (27 of 28 rows; the 28th and the
  README's summed rows were the hand side's errors); the hand GPU-idle join on the one run where the timer ran under
  the capture (276 steps, every function within 0.0002 ms a step, with the hand's busy, which counted host copies:
  a rule this audit does not keep). `tests/test_host_functions.py`.

## The expert cache is an audit metric only

Nothing that sets a value reads the coverage; a test checks it. It rests on a few points of one model. What would
change that:

- the output label moves off `untested`: the quality gate run with the cache on, at the context it is used at;
- coverage against speed on at least two more models and a second context, with repeats, so the curve has a slope
  and a band;
- the hit rates logged on those runs, so coverage is tied to what the cache did;
- prompt time per turn in the same runs (prompt chunks do not use the cache).

Then the flag becomes a knob row and `tune` sizes it.

## Limits

- Host time, the role table and the ledger arrive as records. The audit does the arithmetic and the refusals on them;
  it does not cut them from a raw trace. Their tests run on the outputs of one-off scripts, so they prove the
  arithmetic, not the cut.
- The step count from the step rule has not been matched to traced verify forwards.
- A decode capture with backend sampling is not cut into steps yet.
- Host functions: the timer runs on Linux x86-64 only; it sees only calls through a dynamic symbol, so a function
  inlined or called inside its own library is invisible (its time stays in the caller's row). Its own cost per call
  lands in the caller's row. The idle part comes from a traced run, so its CPU times carry the capture's cost.
