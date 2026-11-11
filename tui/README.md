# boltbeam-tui

BoltBeam on a screen, or as JSON. One Go binary, two modes:

- a human runs `boltbeam-tui` and gets four screens: Setup, Run, Saved runs, Gameplan;
- an agent runs `boltbeam-tui --json <command>` and gets one JSON object and an exit code.

Both modes read through the same seam, `python -m boltbeam.workflow.screen` (plus `boltbeam inspect` for the
model profile). Python owns compute and data and is the only writer of run artifacts. Go owns the screen and
process control (start, stop, tail). No code is shared across the boundary; the contract is a fixture runs
folder and its expected JSON, pinned by a test on each side.

## Quickstart (human)

```bash
boltbeam tui                                   # from the checkout; builds boltbeam-tui once, then runs it
export BOLTBEAM_MODEL=~/models/Qwen3-8B.gguf    # optional: Setup opens with it
export BOLTBEAM_RUNS=/path/to/runs              # the folder that holds runs (default: <repo>/runs)
```

`boltbeam tui` (boltbeam/cli/tui.py) is the way in. It finds the checkout (`$BOLTBEAM_REPO`, else walking up from the
working folder), builds `boltbeam-tui` into `~/.cache/boltbeam/` (`$BOLTBEAM_TUI_DIR`) when it is missing or the Go
source changed (a hash of the sources sits beside the binary), then runs it with `--repo` and `--python` set and
every flag passed through: `boltbeam tui --json chips`. It needs Go at the version `tui/go.mod` names (`go 1.26.7`);
without it the one install line is printed and it exits 2. Nothing is downloaded (`GOTOOLCHAIN=local`).

By hand, the same thing is `cd tui && go build -o boltbeam-tui . && ./boltbeam-tui`.

The interpreter is `--python`, else `BOLTBEAM_PYTHON`, else the first of `python3.13`, `python3.12`, `python3.11`,
`python3.10`, `python3` on PATH (a stock macOS names a 3.9 `python3`, which BoltBeam refuses). The checkout is found
by walking up from the working directory or the binary; `--repo` or `BOLTBEAM_REPO` names it.

At start the screen looks for the engines in the background (`screen engines`: PATH, Homebrew, `~/env`, `~/storage/*`,
the fork next to the checkout, venvs; collectors/engine_scan.py) and remembers them in `~/.boltbeam/engines.json`. The
Engine line says `looking for engines…` until it answers. Opening Engine never starts a scan; `[ Scan again ]` in the
picker does. An env var always wins over what was found.

Keys: `↑` `↓` (or `j` `k`) move, `enter` picks, `esc` goes back to Setup, `l` shows the log while a run goes, `d`
(twice) deletes a saved run, `q` quits, `?` opens a page of every key the screen takes. The footer and that page are
Bubbles `help` drawn from the key bindings themselves (internal/ui/keymap.go): each screen names only the keys that do
something there. In a form the footer is the form's own keys, `esc` leaves it keeping nothing, `ctrl+c` quits.

**Speed up.** On Recommended setup, under Measure MTP: *Makes a faster copy. The original stays.* It is a button only when the
model's format has a writer (`data/speed_up.json`); otherwise a muted row says why ("This model's format has no faster copy
yet."). Enter opens a confirm step (what happens, the disk space, the time, the GPU) that opens on Not now. The job is
`boltbeam derive lighter-head MODEL --json --progress`; its stages show in plain words with the time left (an estimate), and
its end is worded by `screen speedup-plain`: the copy kept with its speeds, NOT KEPT (slower or failed the answer check, and
that the copy was deleted), a refusal with its one fix, or SPEED NOT MEASURED. What next offers Measure MTP on the copy and
Compare with the original; a copy offers Check this copy (`screen speedup-verify`). A copy is `<model> · BoltBeam` in model lists.

**Tuned for this machine.** On Recommended setup, under the recommendation: the chosen engine's knob rows for this model and
the build the Engine picker rests on (`screen tune MODEL --engine E [--build B]`, workflow/tune.py), read from facts only (no
engine starts) as soon as the model, chip and engine are known, the way Offload is. Each row is the knob's value, its source
(derived, measured or default), a tag where it changes outputs, and its reason. `[ Measure on this machine ]` runs the
sweeps as a job (`--measure`: pipeline progress lines, then the page), with a bar, the point in flight and Stop; its end
replaces the facts page. `[ Use these ]` copies the values into Engine settings: a run's settings by name, the rest as the
engine's environment, which Run passes as `--engine-env NAME=VALUE` (set for every engine the run starts, recorded in
`workload_profile.json`, said on the run's settings line). What a run cannot pass is said. The screen knows no engine's
knobs; a test scans the new files for engine names and flags.

## The three screens

**Setup.** The choices and a button.

```text
Machine            ready to run on this Mac                     ›

Model              model.gguf · 18 layers · F32/Q8_0            ›
Chip               apple_m3_10c · this Mac · 97.2 GB/s          ›
Engine             llama.cpp                                    ›

Recommended setup  context, speed and engine flags …            ›

Workload           decode                                       ›
Batch              1                                            ›
Measurement        generic (kernel timer)                       ›

[ Run ]            Measures decode speed: model, this Mac, llama.cpp
[ Emit ]           No run yet: Run first

Saved runs (2)                                                  ›
```

The lines come in groups, a blank line between; the cursor skips the blank lines. Run and Emit each say what pressing
them does with this setup, or why they are off.

Machine is `boltbeam doctor --json` (cli/doctor.py), read after the engine scan and again after Scan again: "ready to
run on this Mac", or the things to set up before a run (Python marks those rows `blocks`). Its page opens on that
verdict, then the checks by their `group`: This GPU, Engines you can run here (each family with the forks found; the
forks not found in one dimmed line, never a gap), Measuring tools, Not on this Mac (the n/a checks, one dimmed line
each) and BoltBeam itself; under a gap, its one fix. `[ Check again ]` asks once more. It never greys Run.

Model searches beside the current model and recursively under `~/models`, or takes a typed path. Checkpoint folders appear once, with the model name and format/quantization label; weight shards are not separate choices. Chip is this
machine, detected (autoscan: a known profile is used, a new GPU is measured into `~/.boltbeam/chips/<id>.json`,
`BOLTBEAM_CHIPS_DIR` moves that folder). A model BoltBeam cannot read keeps its line, with why under it (a weight
format with no block layout is named, from `inspect`'s `formats`); when only a fork reads that format (`needs`), Run
and Recommended setup name the fork, and the engine is never switched for you. Engine is one row per family
(`screen providers` `families`, data/engine_families.json): llama.cpp, vLLM, TensorRT-LLM and tinygrad. ↑↓ moves
between families, ←→ between a family's root and its forks (llama.cpp: upstream, Ollama, Strata; tinygrad:
upstream and its own fork), each with its state under the row: found, or not found with its one fix. The root is the
default; enter picks the fork in brackets when it runs here, and Setup then says it ("llama.cpp · Ollama fork"). An
engine this machine lacks or that cannot read the selected format is unavailable. On a machine with more than one GPU the
Engine picker also shows the layout, how the engine uses the GPUs (`screen providers` `layouts`, workflow/layout.py):
one GPU, split by layer, split by rows, each with the GPUs it would use and what it does to decode speed (split by layer
holds a bigger model but the GPUs take turns; split by rows reads on every GPU at once and pays a join per layer). The
layouts an engine can run here sit under it and are picked with it (the run's `--layout`); the others say why not, in
the engine's own words where it refused (a llama.cpp HIP build loads no split buffers, so no split by rows). Setup
then reads `Engine  vllm · split by rows (2 GPUs)`. On one GPU nothing about layouts is shown. Switching formats chooses a compatible engine and resets the previous engine’s measurement choice. Chip shows a simple GPU, memory-bandwidth and driver summary. Workload is decode, prefill, session or replay. Measurement is in-model
(a capture of the real token) or generic (the engine's kernels timed alone). `[ Run ]` is greyed until model, chip
and engine are set. Setup does not show the speed limit; Run does.

Recommended setup, under Engine, proposes the context, speed and engine flags; Settings, under it, is what the run
asks the engine for: the settings that engine takes (`screen providers`
`settings`, capture_settings.offered: chunk size, attention backend, matrix backends, MTP draft tokens and prompt on
vLLM; chunk size on llama.cpp), one field each in a Huh form (internal/ui/forms.go): a fixed set is chosen with `←`
`→`, a whole number is typed and refused under its field unless it is above 0, and nothing is kept until the last
field's `enter`. They are passed as the pipeline's own flags.
The MTP settings appear only when the recommendation says the checkpoint has a draft head. Repeats is 1 run, or the n
Python offers (`repeat_runs`, repeat_bands.PIN_RUNS): Run then measures the same setup n times (`pipeline --repeats`),
the bar giving each run an equal share, and keeps their band in the first run, where the Run page shows its median,
range and spread and every repeat check reads it. The Recommended setup page offers `[ Measure MTP ]` for vLLM and a
checkpoint with a draft head (`mtp_measure`): `screen mtp` as a job, kept in this machine's facts, then shown per depth
and k: median tok/s, speedup over k=0, tokens per step and the share of drafts kept.

`[ Emit ]` sits under Run. Completed captured-engine reports offer **Emit gap report**: measured timings,
modeled differences, unknown ceilings, and timing corrections, with no generic optimization recommendations.
This works across formats, engines, and supported workload breakdowns. Runs with kernel-search evidence retain
per-role implementation gameplans. Emit writes `<run>/gameplan.json` and `<run>/gameplan.md` and opens the
Gameplan screen. It changes no kernels or engine settings.

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
- BoltBeam's own kernel (reference) per role: GB/s and share of peak from the probe, labelled as not the engine's;
- the engine build against the pin table (`boltbeam/data/engine_pins.json`) and the toolchain the run recorded. A
  run from before either was recorded says "not recorded";
- the end-to-end check: the captured token (decode) or the captured prefill and replay turns against their untraced
  median, with the repeat band. Without a band for the exact setup it says "not checked". That is expected on a first
  run; the page says once how to make a band (n identical runs, then `screen repeat-bands --runs`);
- pairing flags: "Pairing is not certain" with the solver's reason, and each unpaired role with its reason.

Session headlines label the turn's rate as prompt tokens per second; decode speed is a separate column in the
session table. Progress says "Time the real run", then "Capture and attribute kernels" for every workload.
If attribution fails, Run says the report is incomplete and measurements and capture logs are kept. Details
remain behind `l`; Save and Open report are unavailable, and Setup disables Emit with "run failed; see Run for details".

**Results** of a finished run are the plain block (`screen plain`, report/plain.py): layer 1 first (how fast, how close
to the limit, what holds it back, under the labels of `screen words`, led by the headline figures in tok/s, ms and %
(each bold, with its short label); the three parts that lose the most time as bars as
long as their share of the lost time, each with its gain as a figure; the weight reads no part took, as one caveat; a
batch sweep's curve of tok/s per stream count; the one trust line), then layer 2 as sections that open and close: `t` the table (four columns, `a` all of them),
`b` Where the time went (a prefill's time budget, when its capture wrote one: kernels by family, copy time with no kernel
running per stream, GPU idle by cause, host time outside the GPU work, unattributed, adding up to the wall; a trace longer
than its wall is said even with the section closed), `m` How it was measured, `e` Machine and engine, `w` the words used,
`f` Full detail (the page described above). No
sentence on it is written in Go. While a run runs, the stage names are the words' own, the stages done are listed,
and "about N min left" shows only when the last run's step times are logged.

What next, after a run: Open the report (report.html), Run again with this setup (the run's own job command, else its
recorded model, chip, workload, engine, batch sizes, layout and measurement), Saved runs, Compare with a saved run (an
unsaved run is saved first), then `[ Save run ]` and `[ Back to setup ]`. A run is
temporary (`<runs>/.work`) until saved; Save exports the run, `results.json`, `report.html` and `summary.txt` to
`<runs>/saved`. The report has the same numbers as the screen; it is written by the same seam.

**Saved runs.** Every saved run, newest first, in the top box: model, chip, engine and workload, date, and the share
of the limit reached (a replay: of its roofline). Enter opens one to read. `d` twice deletes one. `c` (or Compare two
saved runs) picks two to compare.

**Fetch runs**, on Saved runs, is `boltbeam fetch --json` ([docs/pods.md](../docs/pods.md#bring-results-home)):
a Huh form takes the remote (`user@host` or an ssh Host alias; anything else is refused under the field) and a port for
a pod (1 to 65535, or empty for ssh's own); its `enter` lists the remote's runs (one ssh call: id, here or not, model,
chip, date, size). A second form picks some (`x`, or `ctrl+a` for all, which fetches with `--all`) and whether the raw
traces come too (left behind unless picked). The fetch
runs as a job (`fetch` in the state folder, beside a Run: it copies files only), the page shows how long it has run and
offers Stop; when it ends the page says what was fetched, checked by sha256 and where (`lines`, cli/fetch.fetched_words),
and Saved runs lists it. A refused ssh, an empty runs folder or a run that differs is said in Python's words.

**Compare.** Any two saved runs, head to head (`screen head-to-head`, report/plain.head_to_head): from Saved runs (`c`
picks the first, then the one beside it) or a run's own "Compare with a saved run" (it is the first). The page names
each run (A, B) and says what differs between them (model, GPU, engine, build, weights, settings, workload, streams), then
their headline figures side by side, the better one marked (two prefills sent in different prompt forms, token ids and
a chat request, say so and mark no winner on time), then both time budgets side by side when either has one. Two runs of one model also compare part by part: two runs
of one chip, engine and workload give the engine-upgrade audit row by row (`screen diff`,
[docs/engine-upgrade-audit.md](../docs/engine-upgrade-audit.md); a run's `repeat_bands.json` bands are used), two replays
of one trace on different engines give `screen replay-cross`, turn by turn. Otherwise one line says why the parts are not
compared. A run with no measured result is the one refusal.

**Gameplan.** Gap reports use the same bordered tables and cumulative decode accounting as Run. Unknown savings
and timing qualifications stay explicit. Kernel-specific gameplans show the search evidence described below.

The markdown Emit wrote, drawn in the screen's styles (headings and bold rendered, markers dropped): the header (model, chip, engine, batch, the measured token
against the limit), then one block per role ordered by lost ms (the estimate column for isolated runs, said so) with
the role's shape, bytes per call, calls per token, µs per call, GB/s and share of peak, its reason word and search
verdict, and the search's lineage in four layers, each read off the run's own files: BubbleBeam (the legal dimensions
proposed for this shape), FutureSight (static rejections and the measurement order), Measured (candidates measured,
the best plan against the model's own kernel, the A/B) and Promotion (the verdict; for a winner the fork emitter with
its spec, the promotion record to add with backend, architecture and shape, and the check: correctness against the
reference, then a Run whose tie-out puts the role within the chip's band of the plan's time). A layer the run does not
hold says "not recorded in this run" and names the file it would be in. Every number carries its evidence pointer. The
bottom box holds "Open the gameplan (gameplan.md)" and `[ Back to setup ]`. The report links gameplan.md in its Facts.

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
| `engines` | look for every engine in the usual folders now and save what was found (`~/.boltbeam/engines.json`); the path and how each was found | 0 |
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
| `BOLTBEAM_LLAMA_BENCH`, `BOLTBEAM_VLLM_PYTHON`, `BOLTBEAM_OLLAMA`, `BOLTBEAM_TRTLLM_PYTHON` | an engine the scan missed; each wins over the saved scan |
| `BOLTBEAM_ENGINES_FILE` | where the engine scan is saved (default `~/.boltbeam/engines.json`) |
| `BOLTBEAM_TUI_DIR` | where `boltbeam tui` builds the binary (default `~/.cache/boltbeam`) |

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

[`tapes/`](tapes/README.md) holds a VHS tape per main flow (`vhs tapes/setup.tape` from `tui/`): a visual check, and
the source of the release GIFs. They draw the screens from the same recorded data through `BOLTBEAM_TUI_DEMO`, a
demo file under `testdata/demo/` (left out of the usage: it is for the tapes), so they need no GPU, network or Python.

### Scripted vLLM replay audits

A vLLM replay with `--analyze` uses forced replay automatically where the vendor capture
records a program-marked window (nsys on CUDA, rocprofv3 on AMD; not Metal System Trace).
`--forced-replay` also selects it explicitly on the pipeline command. The server command
remains the user's production command: compilation and graphs stay enabled for timing.
The labeling pass temporarily disables graph replay while retaining compilation.

BoltBeam records the session once, keeps its output token script locally, measures
30 paired forcing-cost samples, runs five forced native reference sessions, and
captures timing plus per-turn labels. It refuses on the first changed token,
cache count, scheduler step or kernel sequence. It checks GPU occupancy before
each pass and stops only servers that it started.

Run, HTML and **Emit replay report** share the reconciled table. Timing qualification
may remain FAILED even when arithmetic and pairing pass; Emit preserves that status.
Missing memory counters or instruction paths stay explicit. Modeled partial limits
are labeled partial; they are never presented as a complete session floor. Token
forcing is a separate measured diagnostic, not added again to session time.

For a retained capture, `python -m boltbeam.workflow.screen replay-review --run RUN
--inputs INPUTS.json` performs CPU-only reconciliation. The local input manifest
names `script`, `paired`, `reference`, `graph`, and `segments` (one timing/label path
pair per turn), with optional `cost` and `calibration` (`run`, `context`,
`instruction_paths`). The public result stores script hashes and lengths, not token
content. Raw scripts, responses, worker logs and capture paths remain in the local run.
