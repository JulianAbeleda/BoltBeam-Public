# boltbeam-tui

BoltBeam on a screen, or as JSON. One Go binary, two modes:

- a human runs `boltbeam-tui` and gets three screens: Setup, Run, Saved runs;
- an agent runs `boltbeam-tui --json <command>` and gets one JSON object and an exit code.

Both modes read through the same seam, `python -m boltbeam.workflow.screen` (plus `boltbeam inspect` for the
model profile). Python owns compute and data and is the only writer of run artifacts. Go owns the screen and
process control (start, stop, tail). No code is shared across the boundary; the contract is a fixture runs
folder and its expected JSON, pinned by a test on each side.

## Quickstart (human)

```bash
cd tui && go build -o boltbeam-tui .           # Go 1.26
export BOLTBEAM_MODEL=~/models/Qwen3-8B.gguf    # optional: Setup opens with it
export BOLTBEAM_RUNS=/path/to/runs              # the folder that holds runs (default: <repo>/runs)
./boltbeam-tui
```

The interpreter is `BOLTBEAM_PYTHON`, else the first of `python3.13`, `python3.12`, `python3.11`, `python3.10`,
`python3` on PATH (a stock macOS names a 3.9 `python3`, which BoltBeam refuses). The checkout is found by
walking up from the working directory or the binary; `--repo` or `BOLTBEAM_REPO` names it.

Keys: `↑` `↓` (or `j` `k`) move, `enter` picks, `esc` goes back to Setup, `x` stops the run started here, `l` shows
the log while a run goes, `d` (twice) deletes a saved run, `q` quits.

## The three screens

**Setup.** Three choices and a button.

```text
Model    Qwen3-8B.gguf · 36 layers · F32/Q4_K/Q6_K       ›
Chip     apple_m3_10c · this Mac · 97.2 GB/s             ›
Engine   llama.cpp                                       ›
Batch    1                                               ›

[ Run ]
Saved runs (2)                                           ›
```

Model lists the model files next to the current one and in `~/models`, or takes a typed path. Chip is this
machine, detected (autoscan: a known profile is used, a new GPU is measured into `~/.boltbeam/chips/<id>.json`,
`BOLTBEAM_CHIPS_DIR` moves that folder). Engine is llama.cpp or tinygrad; an engine this machine lacks is named
with why. `[ Run ]` is greyed until all three are set. Setup does not show the speed limit; Run does.

**Run.** One press runs the pipeline: read the model, check the machine, the speed limit, the GPU free check, the
building-block probe (BoltBeam's own kernels on the model's real bytes), the real decode with the engine, per-role
time, the kernel search (tinygrad on Metal), the report. A bar shows `n of N` for the counted (slow) stages,
weighted by the last run's seconds for this chip and engine; on a first run the weights are a default table and the
percent is marked an estimate (`~37%`). When the run ends the top box shows:

- the headline: measured tokens per second against the limit, and the percent of the limit. There is one measured
  token: the step-4 row the tie-out uses (the smallest context, or the context the per-role capture attended). Other
  contexts are listed as "Also measured at context N", never mixed into the headline;
- the tie-out: limit (ideal) + weight kernels above their ideal + other kernels (or, for isolated kernels, the one
  difference line "kernels not timed and gaps") = the measured token;
- per role: ideal, actual, lost ms, share of peak, µs per call and the reason word (at the limit, too small to fill
  memory, slow kernel, compute bound, inconclusive cache, unexplained). The reason rule states its numbers, with the
  dispatch floor the probe measured on this GPU when it has one;
- how the roles were timed: tinygrad's own timing, a vendor capture (nsys, rocprofv3, Metal System Trace), or
  "isolated, timed by BoltBeam's kernel timer" (the engine's own kernels compiled from its shipped source, no Xcode;
  [docs/kernel-timer.md](../docs/kernel-timer.md));
- the findings the run's own facts give: launch heavy, a failed graph replay the engine reported, KV dominated,
  throttled, unexplained; each with its lever and the files behind it;
- BoltBeam's own kernel (reference) per role: GB/s and share of peak from the probe, labelled as not the engine's.

The bottom box holds `[ Save run ]`, "Open the full report (report.html)" and `[ Back to setup ]`. A run is
temporary (`<runs>/.work`) until saved; Save exports the run, `results.json`, `report.html` and `summary.txt` to
`<runs>/saved`. The report has the same numbers as the screen; it is written by the same seam.

**Saved runs.** Every saved run, newest first, in the top box: model, chip, engine, date and the share of the limit
reached. Enter opens one to read. `d` twice deletes one.

## Agent mode: the JSON contract

```
boltbeam-tui [--json] [--root RUNS] [--saved DIR] [--repo DIR] [--python PY] [--state DIR]
             [--model FILE] [--target ID] [--context N] <command>
```

| Command | Prints | Exit |
|---|---|---|
| `targets` | the chips BoltBeam knows, and which carry a speed limit | 0 |
| `chips` | the chips in Setup's groups: this machine, measured, not measured yet, families | 0 |
| `autoscan [--remeasure]` | use the chip profile that fits this GPU, or measure and save a new one | 0 / 1 |
| `inspect MODEL` | the model profile (`boltbeam inspect`, bytes unchanged) | 0 / 1 |
| `ceiling MODEL --target ID [--context N]` | the roofline: best tokens/s for decode and prefill; with machine facts in the runs folders, the memory speed measured on this GPU | 0, 1 when the chip has no ceiling |
| `runs` | every run: `RUNS/<id>`, `RUNS/.work/<id>` and the saved runs, each marked where it lives | 0 |
| `run <id>` | the stages, what is blocked, the next step, the results | 0, 1 unknown run |
| `results <id>` | the limit, the loss (tie-out, per role, findings, the measured token and the other contexts), the probe rows, the routes | 0, **3 when nothing is measured yet** |
| `start MODEL --target ID [--workload W] [--id RUN] [--probe FILE] [--timing FILE] [--provider P] [--analyze [--no-search]]` | the job: `{id,pid,alive,log_path,started_at,argv,dir}` | 0 / 1 |
| `stop <id>` | the job | 0, 1 when not running |
| `delete <id>` | remove a run folder that is not running | 0 / 1 |
| `save <id>` | export a run to `--saved`: the run, results.json, report.html, summary.txt | 0 / 1 |
| `saved` | the saved runs, newest first | 0 |
| `tail <id> [--lines N]` | the pipeline's status and the last lines of its log | 0 |

Every error is `{"kind":"error","error":"...","stderr":"..."}` with exit 1; a usage error exits 2 and prints
the usage on stderr. The JSON commands print Python's bytes unchanged, so the agent reads the same facts the
screens were built from. The full shapes are in [`testdata/expected/`](testdata/expected/) and typed in
[`internal/seam/types.go`](internal/seam/types.go).

`start` runs `python -m boltbeam.workflow.screen pipeline MODEL --run <runs>/.work/<id> --target ID ...` detached,
with the log and pid under `--state` (default `~/.local/state/boltbeam-tui`), never inside the run folder. The
pipeline prints `pipeline steps: N`, `pipeline counted: 0,0,1,…`, `pipeline expect: a,b,…` and
`pipeline expect source: history|default` first, then `stage <key>: start|progress p/q|done|failed: …` lines,
which `tail` returns parsed. `stop` sends SIGTERM; every stage writes its artifacts when it finishes, so the run
folder keeps what landed.

## Environment

| variable | what it names |
|---|---|
| `BOLTBEAM_RUNS` | the runs folder (`.work` and `saved` live under it) |
| `BOLTBEAM_SAVED` | the saved-runs folder (default `RUNS/saved`) |
| `BOLTBEAM_PYTHON` | the interpreter that imports boltbeam |
| `BOLTBEAM_REPO` | the checkout (default: walk up from here) |
| `BOLTBEAM_TUI_STATE` | pids and logs |
| `BOLTBEAM_MODEL`, `BOLTBEAM_TARGET` | what Setup opens with |
| `BOLTBEAM_CHIPS_DIR` | the chip profiles measured on this machine and the stage-time history (default `~/.boltbeam`) |
| `BOLTBEAM_TINYGRAD_ROOT` | the tinygrad fork (default: `../tinygrad-arkey-exp` next to the checkout). The older name `TINYGRAD_ROOT` is still read. |
| `BOLTBEAM_TINYGRAD_VENV` | the fork's venv folder (its `bin/python` runs the fork) |
| `BOLTBEAM_TINYGRAD_PYTHON` | the fork's interpreter, when it is not in a venv under the fork |
| `BOLTBEAM_GGML_METAL`, `BOLTBEAM_GGML_CUDA_SRC` | llama.cpp's Metal library or CUDA source, when not where Homebrew or `~/env/llama.cpp` put them |

## Tests

```bash
cd tui && go test ./...                                          # screens, contract, jobs
BOLTBEAM_PYTHON=python3.12 go test ./...                         # also the live seam against the fixture
python3.12 -m pytest tests/test_workflow_screen.py               # the Python side of the same contract (dev)
```

`testdata/fixture/runs/` is the fixture: run `001` is planned only, run `002` also ingested a probe evidence file
and a timing trace from `testdata/fixture/evidence/` (synthetic, plausible for an M3, labelled as not a
measurement). `testdata/expected/*.json` is the seam's output on it; both test suites compare against those files
byte for byte. `testdata/screens/*.txt` are the rendered screens under `NO_COLOR`, `styled/*.ansi` the same with true
colour (`go test ./internal/ui -update` rewrites them). The live Go test skips when no interpreter imports
`boltbeam`.

Go tests live beside the code (`_test.go`) because Go's toolchain treats them as part of the package; they
travel with the product on `main`. The Python tests follow the branch rule and live on `dev`.
