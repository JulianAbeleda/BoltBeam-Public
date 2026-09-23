# Using BoltBeam from the command line

BoltBeam reads a model file and tells you how fast it could ever run on a chip, where the
time actually goes, and whether a change helped. It never runs the model. A compiler does
that; BoltBeam keeps the books.

Everything below was run on a MacBook Air (M3) against `Qwen3-8B.gguf`. Commands that need a
GPU or the tinygrad fork are marked.

## Getting it running

Python 3.10 or newer. No dependencies.

```
git clone https://github.com/JulianAbeleda/BoltBeam-Public.git
cd BoltBeam-Public
python3 -m boltbeam.cli --help
```

`pip install -e .` also installs a `boltbeam` command, which is the same thing.

Most commands take a GGUF model and a `--target`, and write JSON to stdout or to `--out`.

## The chips it knows

```
python3 -m boltbeam.cli route-manifest --dump-default routes.json
python3 -c "import json;print([t['target_id'] for t in json.load(open('boltbeam/data/targets.json'))['targets']])"
```

| Target | What it is | Where its numbers come from |
|---|---|---|
| `amd_gfx1100` | Radeon RX 7900 XTX | vendor sheet, plus modelled memory tiers |
| `nvidia_sm89` | Ada, descriptor only | no speeds recorded |
| `nvidia_sm120` | RTX 5090 | measured: 1,693.3 GB/s, 237.1 TFLOP/s on the matrix unit |
| `apple_metal` | any Metal GPU, family row | no speeds; pass your own |
| `apple_m3_10c` | MacBook Air M3, 10-core GPU | measured: 89.9 GB/s, 3.152 TFLOP/s |
| `apple_m4_10c` | Mac mini M4, 10-core GPU | vendor sheet: 120 GB/s |

A row records where each number came from, under `capabilities.fact_status` and
`fact_sources`: `measurement`, `hardware_scan`, `derived`, `vendor_spec`, `estimate`, or
`unknown`, each with the method and the date. Nothing is guessed. A chip with no number for
something simply has none, and the commands that need it say so instead of inventing one.

Speeds are measured, not read off the box. The RTX 5090's bus arithmetic gives 1,792 GB/s
and a streaming read reaches 1,693.3, so 1,693.3 is the ceiling the roofline uses and 1,792
is recorded beside it as the theoretical figure. A peak is only a fact with its clock: the
same matrix kernel reads 237.1 TFLOP/s with the clock pinned and 214.5 at the speed the card
picks on its own, and the row says so.

Two compute numbers are not the same number. `peak_tflops` is the vector-ALU rate, and
`matrix_tflops` is what the matrix unit actually did. The roofline prefers your own
`--peak-tflops`, then the measured matrix rate, then the ALU rate.

A row also states which machine it is, in `capabilities.match`. That is what `autoscan` reads
to recognise the GPU in front of it, so teaching BoltBeam a new chip is an edit to
`targets.json` and no code at all.

## 1. What is in this model

```
python3 -m boltbeam.cli inspect ~/models/Qwen3-8B.gguf --out profile.json
```

Layer count, hidden size, the roles (attention, FFN, vocabulary), and the quantisation each
weight table uses. This is the input to everything else.

## 2. How fast could it ever go

```
python3 -m boltbeam.cli roofline-theoretical ~/models/Qwen3-8B.gguf \
  --target apple_m3_10c --out ceiling.json
```

Per role and for the whole model: bytes moved, the floor in milliseconds, arithmetic
intensity, and whether the role is held back by memory or by sums. On the M3 above: 7.624 GB
moved, a 2,458 ms floor at the default 512-token prefill, ridge point 35.1 sums per byte.
The default is prefill; `--context N` changes the length it assumes.

If your chip's row has no speeds, pass them:

```
--peak-gbs 1693.3 --peak-tflops 237.1
```

The write-ups on research.arkey.ai quote 1,700 GB/s and 255.4 TFLOP/s for the RTX 5090, from
an earlier session. Re-measured on 2026-09-23 the same card gave 1,693.3 and 237.1, and the
registry carries the new figures with their method and date. The bandwidth agrees to 0.4%.
The matrix figure moved because the clock did.

**Tokens per second** needs a memory hierarchy on the target row, which today only
`amd_gfx1100` has:

```
python3 -m boltbeam.cli roofline-theoretical MODEL --target amd_gfx1100 --throughput
```

That adds `decode` and `prefill` blocks with `tok_s` and `step_ms` (205.3 tokens per second
for Qwen3-8B on that card, modelled). For a chip without tiers you get a clear error, not a
number. To give a chip tiers, measure them:

```
python3 -m boltbeam.cli mem-sweep-request --target nvidia_sm120 --out sweep.json
# run the sweep on the hardware, then:
python3 -m boltbeam.cli ingest-mem-sweep samples.json --snapshot SNAPSHOT_ID
```

## 3. What to try, and in what order

```
python3 -m boltbeam.cli emit-search MODEL --target apple_m3_10c --out search_space.json
python3 -m boltbeam.cli vocab MODEL --out vocab.json
python3 -m boltbeam.cli boundary-plan audit.json --out boundary.json   # takes an audit, not a model
```

Or all of it at once, which is the usual way in:

```
python3 -m boltbeam.cli analyze MODEL --target apple_m3_10c --out-dir out/
```

That writes `model_profile.json`, `search_space.json`, `route_policy.seed.json`,
`next_measurement_plan.json`, `fixture_manifest.json`, `analysis_manifest.json`, and
`tinygrad_commands.md`, which is the list of commands to run on the compiler side.

## 4. A run folder, when you are doing this properly

```
python3 -m boltbeam.cli load MODEL --run run/ --target apple_m3_10c --workload decode
python3 -m boltbeam.cli autoscan --run run/          # what is on this machine
python3 -m boltbeam.cli output --run run/            # the run manifest so far
```

`load` writes `model_profile.json`, `weight_inventory.json`, `workload_profile.json` and a
`run_manifest.json` that every later stage appends to.

## 5. Bringing measurements back

BoltBeam does not time anything itself. You measure with the vendor's profiler or the
tinygrad fork, then hand it the result.

```
python3 -m boltbeam.cli import-ncu ncu.csv --run run/                      # Nsight Compute
python3 -m boltbeam.cli import-hw-trace kernels.csv --provider llama --run run/
python3 -m boltbeam.cli compare-hw-trace --baseline before.json --candidate after.json
python3 -m boltbeam.cli profiler-report trace.json --top 20
python3 -m boltbeam.cli roofline trace.json --peak 1700                   # per kernel -> whole model
```

`roofline` takes a per-kernel trace and attributes the whole token: which kernels are near
their floor, which are not, and how much time is neither.

## 6. The verdict, and the ledger

```
python3 -m boltbeam.cli evaluate --profile model_profile.json --search search_space.json \
  --evidence evidence.json
python3 -m boltbeam.cli ledger add decision.json --ledger ledger.jsonl   # one verdict, with its reason
python3 -m boltbeam.cli ledger report --ledger ledger.jsonl              # deterministic markdown
python3 -m boltbeam.cli check-policy --ledger ledger.jsonl
```

`evaluate` judges a candidate against the model's floors and returns one of eight verdicts.
The ledger keeps every verdict with its reason, wins and losses alike, so nobody pays for the
same refuted idea twice.

## 7. The search space, cut before the GPU runs

Three stages, cheap to expensive:

```
python3 -m boltbeam.cli propose-semantic-dimensions spec.json --out request.json      # BubbleBeam
python3 -m boltbeam.cli export-semantic-population request.json --out population.json
python3 -m boltbeam.cli assess-semantic-population population.json --out evidence.json # FutureSight
python3 -m boltbeam.cli semantic-campaign request.json --futuresight-evidence evidence.json \
  --provider-command "python3 extra/llm_research/search_provider.py" --out results.json  # needs a GPU
```

BubbleBeam works out which values are legal from the chip's declared facts. FutureSight
rejects what cannot fit or cannot fill the chip and orders the rest. Only what survives costs
GPU time. The first two run on the main processor and need no hardware.

## What needs what

| You have | You can run |
|---|---|
| this repo and a model file | `inspect`, `roofline-theoretical`, `emit-search`, `vocab`, `analyze`, `load`, `boundary-plan`, `propose-semantic-dimensions`, `assess-semantic-population` |
| a profiler trace from any vendor | `import-ncu`, `import-hw-trace`, `compare-hw-trace`, `profiler-report`, `roofline`, `evaluate`, `ledger` |
| the tinygrad fork and a GPU | `semantic-campaign`, `search-full-kernel`, `collect-hw-trace`, `mr7-provider-run` |

`python3 -m boltbeam.cli --help` lists every command; each one takes `--help` too.

## Reading the output

Every command writes JSON with a `schema` field, and the schemas are in `schemas/`. A number
carries its status where one exists: `hardware` when a chip reported it, `measurement` when
someone measured it, `modeled` when it was derived. When you see `modeled`, it is arithmetic,
not an observation.
