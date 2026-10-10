# boltbeam-tui

BoltBeam on a screen, or as JSON. One Go binary, two modes:

- a human runs `boltbeam-tui` and gets Bubble Tea screens;
- an agent runs `boltbeam-tui --json <command>` and gets one JSON object and an exit code.

Both modes read through the same seam, `python -m boltbeam.workflow.screen` (plus `boltbeam inspect` for the
model profile). Python owns compute and data and is the only writer of run artifacts. Go owns the screen and
process control (start, stop, tail). No code is shared across the boundary; the contract is a fixture runs
folder and its expected JSON, pinned by a test on each side.

## Quickstart (human)

```bash
cd tui && go build -o boltbeam-tui .           # Go 1.26
export BOLTBEAM_MODEL=~/models/Qwen3-8B.gguf    # optional: the screens open with it
export BOLTBEAM_TARGET=apple_m3_10c             # optional: the chip (default: this machine, from detect)
export BOLTBEAM_RUNS=/path/to/runs              # the folder that holds <run>/ folders (default: <repo>/runs)
./boltbeam-tui
```

The interpreter is `BOLTBEAM_PYTHON`, else the first of `python3.13`, `python3.12`, `python3.11`, `python3.10`,
`python3` on PATH (a stock macOS names a 3.9 `python3`, which BoltBeam refuses). The checkout is found by
walking up from the working directory or the binary; `--repo` or `BOLTBEAM_REPO` names it. No GPU is needed for
any screen: `inspect` reads the file's shape, the ceiling is arithmetic, and the pipeline runs the no-GPU
stages. Measurements come from an external runner as `probe_evidence.v1` and `timing_trace.v1` files.

Keys: `↑` `↓` (or `j` `k`) move, `enter` opens a step or picks a row inside it, `esc` goes back, `x` stops the
run started here, `q` quits. Five keys. Everything else is a row inside a step.

The look: a Charm-style palette (pink, purple, cyan, mint) as lipgloss adaptive tokens in
[`internal/ui/theme.go`](internal/ui/theme.go), rounded boxes with gradient titles, glyphs for every outcome
(`✓` done, `✗` failed, `⚠` slower than it could be, `⏸` waiting for evidence, `○` not run, `●` running, `♡` a
kept route), bars for each role's share of a token, for a kernel's share of the step and its share of the
memory peak, and for measured tokens/s against the ceiling, and a small bolt in the header whose face follows
the state: `(˘ω˘) zz` with nothing read, `(•ᴗ•)` idle, `(•̀ᴗ•́)⚡` while a pipeline runs or a file is read,
`(•-•)⏸` when a run waits for evidence, `(≧◡≦)⚡` when a run has a measurement, `(•́︿•̀)` when a stage failed.
Colours degrade through lipgloss to the terminal's profile and disappear under `NO_COLOR`; the layout fits
80x24 (long cells are cut with `…`).

One screen, a checklist. The top box is the path as five steps, each with a mark and one line of result. The
bottom box is the chosen step's summary. `enter` opens the step's full view (scrollable; its rows first). The
cursor starts on the first step that is not done. Every mark is computed from seam facts, so reopening the
screen mid run lands on the same marks.

| # | Step | Done when | The line says |
|---|---|---|---|
| 1 | Model | `inspect` read the file | file name, layers, quants |
| 2 | Chip | the chip has a speed limit | id, `this Mac` when `detect` names it, memory GB/s |
| 3 | Speed limit | `ceiling` answered | `up to N tokens per second` |
| 4 | Measure | the run's `output` stage is done | a bar `n of N` while it runs; `planned · needs a timing trace` after |
| 5 | Result | `results.measured` | `M tokens per second · P% of the limit` |

Full views: **Model** has a "type a path" row and the model files found next to the current one and in
`~/models`, then the role census. **Chip** lists every chip to pick from. **Speed limit** has the roofline per
role for one token and for a prompt, and the assumptions. **Measure** has one "Measure with" row per runtime this
machine has (llama.cpp, tinygrad; `screen providers` decides, and a runtime it lacks is named with why), or
"Stop the run", then the earlier runs to open, each labelled with its runtime, the stages, and the log. Step 4
refuses a GPU another program holds (`screen gpu-free`). **Result** is titled with the run's runtime and how its
roles were timed (nsys, rocprofv3, Metal System Trace, or tinygrad's own timing); another runtime's table for
the same run is shown beside it, labelled. It has "Time each role in RUNTIME" where this machine can capture
that runtime, "Open the full report" and "Measure again with", then what won per role, the kernels against the
peak, and the building blocks.

Main lines use plain words only (BoltBeam's vocabulary from `gui/README.md`). Full views show the plain word
with the record's own name beside it, muted (`small kernel, dead time  elementwise_dilution`). There is no
technical mode.

## Agent mode: the JSON contract

```
boltbeam-tui [--json] [--root RUNS] [--repo DIR] [--python PY] [--state DIR] <command>
```

| Command | Prints | Exit |
|---|---|---|
| `targets` | `{"kind":"targets","targets":[{id,backend,backend_status,scope,memory_bandwidth_gbs,peak_tflops,matrix_tflops,fact_status,has_ceiling}]}` | 0 |
| `inspect MODEL` | `boltbeam.model_profile.v1`, bytes unchanged from `boltbeam inspect` | 0 / 1 |
| `ceiling MODEL --target ID [--context N]` | `{"kind":"ceiling",model_id,target,peak_bandwidth_gbs,peak_tflops,truth_status,ridge_intensity,assumptions,decode:{context,bytes_moved,floor_ms,tok_s,regime,roles},prefill:{...}}` | 0, 1 when the chip has no ceiling |
| `runs` | `{"kind":"runs","root","runs":[summary]}` | 0 |
| `run <id>` | the summary plus `model_path, model, next_step, artifacts, results` | 0, 1 unknown run |
| `results <id>` | `{"kind":"results",measured,ceiling,routes,timing,regimes,blocked,report}` | 0, **3 when nothing is measured yet** |
| `start MODEL --target ID [--workload W] [--id ID] [--run NAME] [--probe FILE] [--timing FILE]` | the job: `{id,pid,alive,log_path,started_at,argv,dir}` | 0 / 1 |
| `stop <id>` | the job | 0, 1 when not running |
| `tail <id> [--lines N]` | `{"kind":"job","job":{...},"lines":[...],"stages":{key: running\|done\|failed}}` | 0 |

Every error is `{"kind":"error","error":"...","stderr":"..."}` with exit 1; a usage error exits 2 and prints
the usage on stderr. `targets`, `inspect`, `ceiling`, `runs`, `run` and `results` print Python's bytes
unchanged, so the agent reads the same facts the screens were built from.

A run summary:

```json
{"id": "qwen3-8b-apple_m3_10c-002", "model_id": "Qwen3-8B", "model_format": "gguf", "target_id": "apple_m3_10c",
 "workload": "decode", "latest_stage": "output", "status": "needs_measurement",
 "stages": [{"key": "load", "label": "load", "note": "model / weight / workload facts", "done": true, "artifacts": ["..."]}, "..."],
 "blocked": [{"need": "timing_trace", "request": "trace_request.json"}],
 "measured": {"probe": true, "timing": true}, "report": "report.html"}
```

`status` is `analysis_report.json`'s own (`not_analyzed | needs_measurement | policy_seeded | ...`); `blocked`
lists the evidence `measurement_plan.json` still requests; `results.ceiling.tok_s` is the modeled ceiling for
the run's workload and `results.timing.tok_s` the measured step, both for the same model and chip. The full
shapes are in [`testdata/expected/`](testdata/expected/) and typed in
[`internal/seam/types.go`](internal/seam/types.go).

`start` runs `python -m boltbeam.workflow.screen pipeline MODEL --run <runs>/<id> --target ID` detached, with
the log and pid under `--state` (default `~/.local/state/boltbeam-tui`), never inside the run folder. The
pipeline runs `load`, `autoscan`, `analyze`, then `ingest-probe` and `ingest-timing` when files were given (and
`analyze` again, so the plan reads them), then `output`; it prints `stage <key>: start|done|failed: …` lines,
which `tail` returns parsed under `stages`. `stop` sends SIGTERM; every stage writes its artifacts and the
manifest when it finishes, so the run folder keeps what landed. A new run always gets a new folder
(`<model>-<chip>-NNN`); an existing one is never overwritten.

## What the seam adds to BoltBeam

`boltbeam/workflow/screen.py` is the one Python-side addition. It computes no new kind of fact:

- `targets` reads `boltbeam/data/targets.json` through the registry, with each speed's `fact_status`;
- `detect` is autoscan's own GPU probe cut to `{status, name, target_id, target_kind, registered}`; `target_id`
  is null when no probe answered (the screen then says nothing about "this Mac");
- `ceiling` is `model_roofline` as `roofline-theoretical` calls it (same peak precedence: a measured matrix rate
  before the ALU sheet rate), at context 1 for decode and the given context for prefill, with tokens/s read off
  the floor; a descriptor-only chip gets the same refusal `roofline-theoretical` prints;
- `runs`, `run` and `results` read the run artifacts and reuse the report's stage table, next-step ladder and
  hot-kernel pick (`report/html.py`), so the screen, `summary.md` and `report.html` never disagree;
- `pipeline` calls the workflow's own stage functions in order and prints `pipeline steps: N` first, so a
  screen can draw `n of N`. N counts only the slow stages. The quick setup stages (load, autoscan, the first
  analyze, machine, measure_probe) are not steps. `pipeline counted: 0,0,1,…` gives one flag per stage line, in
  order. While a setup stage runs, the screen says Starting and shows no count.

Three private names in `report/html.py` and `cli/roofline.py` became public for that reuse (`STAGES`,
`next_step`, `roofline_kernels`, `resolve_peak_flops`); nothing else outside `tui/` changed.

## Tests

```bash
cd tui && go test ./...                                          # screens, contract, jobs
BOLTBEAM_PYTHON=python3.12 go test ./...                         # also the live seam against the fixture
python3.12 -m pytest tests/test_workflow_screen.py               # the Python side of the same contract (dev)
```

`testdata/fixture/runs/` is the fixture: run `001` is planned only (load, autoscan, analyze, output; both
measurements still requested), run `002` also ingested a probe evidence file and a timing trace from
`testdata/fixture/evidence/` (synthetic, plausible for an M3 at 89.9 GB/s, labelled as not a measurement).
Both were produced by the pipeline from `Qwen3-8B.gguf` on this chip; the model path was rewritten to
`/models/Qwen3-8B.gguf`. `testdata/expected/*.json` is the seam's output on it; both test suites compare
against those files. `testdata/screens/*.txt` are the rendered screens under `NO_COLOR` (the checklist in four states, the five
full views, a full view at 80x24, the path being typed), `styled/*.ansi` the
same with true colour (`go test ./internal/ui -update` rewrites them). The live Go test skips when no
interpreter imports `boltbeam`.

Go tests live beside the code (`_test.go`) because Go's toolchain treats them as part of the package; they
travel with the product on `main`. The Python test follows the branch rule and lives on `dev`.
