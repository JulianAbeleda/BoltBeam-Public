# Guide: add an engine

BoltBeam measures a model in an engine and shows where the engine's time goes against the roofline. Five engines
have adapters today: llama.cpp, vLLM, Ollama, TensorRT-LLM and the tinygrad fork. This guide says what a new
adapter must return, and points at the code that does each part for the existing five.

The rule behind the layout is **modularize execution, centralize authority** ([coding-principles.md](coding-principles.md)).
An adapter knows only its own engine: how to find it, which weights it reads, how to run N decode tokens, and how
its kernels are captured. Everything after that is shared: the vendor capture, the pairing of kernels to roles, the
ceilings, the floor rule and the tie-out. One table lists the engines. Nothing else keeps a second list.

## Where the engines are listed

| table | file | what it holds |
|---|---|---|
| `ENGINES` | `boltbeam/collectors/providers.py` | name to adapter module; the order Setup lists them |
| `COLLECTORS` | same | the step-4 collector per engine and backend |
| `FORMATS` | same | the weight files each engine reads (`gguf`, `safetensors`) |
| `DRIVEN` | same | engines that run as their own program through a driver script |
| `ENGINES` | `boltbeam/collectors/prefill.py` | per engine: how to time one prefill (`truth`) and capture one (`capture`) |
| `ITEMS`, `PRIMARY` | `boltbeam/collectors/engine_scan.py` | where `doctor` and the screen look for the engine, and its env var |
| `engines` | `boltbeam/data/engine_pins.json` | the one pinned version per engine, with its history |

## What an adapter must return

Seven things. Each row names the existing code that does it.

### 1. Truth timing: the whole step, untraced

The engine's own decode time per token, measured with no profiler attached. Every ceiling percent and every
end-to-end check is judged against this number.

- Decode: `whole_step(model, contexts, batches, ...)` returns one `engine_common.point(context, batch, step_ms, ...)`
  per point. vLLM and TensorRT-LLM time N tokens and 1 token with the same prefill and divide the difference by
  N minus 1 (`runtime/vllm_decode_drive.py` docstring). Ollama reads its API's `eval_count` and `eval_duration`
  (`ollama_decode.py`). llama.cpp runs `llama-bench -p 0 -n N -d CTX` (`llama_bench_decode.bench_decode`). The fork
  runs its decode with no profiling (`tinygrad_role_time.whole_step`).
- Prefill: a `truth(run, model, length, target, root)` entry in `prefill.ENGINES`: `llama_truth`, `vllm_truth`,
  `tinygrad_truth`. Each returns the median ms, the samples, the chunk record (below), GPU telemetry and the engine
  settings it actually used.
- Record the token window. The row must carry `decode_tokens`: the matching-capture check
  (`capture_settings.decode_tokens`) refuses a capture without it.

### 2. Capture: the same run with a profiler

The per-role time comes from a vendor capture of the same engine, at the same context, batch and settings.

- Driven engines call `engine_common.captured_role_time(...)`. It captures two runs, N tokens and 1 token, with the
  backend's profiler (`vendor_capture.capture`: nsys on CUDA, Metal System Trace on a Mac), subtracts them kernel
  by kernel (`attribution.shared_window`) and writes the per-role trace (`attribution.provider_trace`).
- The adapter's `role_time(run, *, target, model, model_id, context, batch, tokens)` supplies `argv_for(n)`: the
  driver command for n tokens. See `vllm_decode.role_time`, `ollama_decode.role_time`, `trtllm_decode.role_time`.
- Capture settings must equal the untraced settings. Read them back from the run (`capture_settings.context_limit`,
  `capture_settings.vllm_flags`). Never let the capture pick its own allocation.
- llama.cpp captures `llama-bench` itself (`llama_bench_decode.role_time`). The fork uses its own GPU timestamps
  where CUDA's tools cannot see it (`tinygrad_role_time.collect`).
- `TRACE` names the per-role trace file the adapter writes (`vllm_timing_trace.json`, ...). `providers.TRACES`
  reads it from the adapter.

### 3. Kernel paths: which matrix unit each kernel uses

A kernel's compute ceiling is its operations over the measured peak of the path its instructions use (fp16, int8,
fp8, ...). The path is read from the kernel's code, never from its name (`collectors/instruction_path.py`).

- The prefill `capture` entry returns `paths`: `prefill.library_paths(libraries, launches, tools=..., jit_caches=...)`
  reads the embedded PTX of the libraries the engine loaded, then machine code for kernels with no PTX.
- vLLM's driver prints the libraries it loaded and Triton's JIT cache folders, so its JIT kernels are read too
  (`prefill.vllm_capture`). llama.cpp's libraries come from the binary (`prefill.linked_libraries`). The fork keeps
  each program's own source (`prefill.fork_capture`).
- A kernel whose code cannot be read gets no compute ceiling. The row says why. Do not guess a path from a name
  unless `instruction_path.NAME_PATHS` documents the pattern.

### 4. Formats: what the kernels read and write

The roofline counts bytes. It needs the formats the engine really uses, not the checkpoint's alone.

- `FORMATS` (module level): the weight files the engine reads. `providers.reads` refuses a model in another format.
- Prefill formats: `prefill.llama_formats` (q8_1 activations into `mul_mat_q`, fp32 out, the KV cache in `type_k`),
  `prefill.vllm_formats` (activations and cache in the weights' float type). Return them as `formats` from the
  capture.
- `KV_ELEMENT` (module level): bytes per KV element and the sentence that says why, for the decode KV read.
- Packed matrices: when the engine fuses checkpoint tensors (q, k and v as one GEMM), its driver should print
  `matrices` (name, N, K, count, parts). `prefill.engine_roles` and the decode attribution use them to pair one
  launch with several roles. Without them, pairing works from the profile's roles and may leave rows unpaired.

### 5. The chunk record

What the engine batched during prefill. Build it with `prefill.chunk_record(widths, how, outputs)`: the token width
of each forward call, the number of calls that produce an output, and a sentence naming the engine setting that
decided it. vLLM's comes from `max_num_batched_tokens` and chunked prefill (`prefill.vllm_chunk`). llama.cpp's
comes from `n_ubatch` and `n_batch`. An engine that reads the prompt one token at a time is "not batched prefill",
and the row says so.

### 6. The version, for the pin table

Each run records the build it measured. `providers.run_build` reads it from the untraced rows: `build` or
`engine.build` (llama.cpp's `build_commit`), or `vllm` / `engine.vllm` (vLLM's version). Put the engine's own
version string on every whole-step row. `providers.record_build` then writes the build and its check against
`boltbeam/data/engine_pins.json` into the run manifest. A run without a build cannot be calibrated or compared:
the engine-upgrade audit refuses it ([engine-upgrade-audit.md](engine-upgrade-audit.md)).

Add the engine to `engine_pins.json` with an empty `history`. Its first entry comes with its first measured audit.
Changing a pin is its own `[target]` or `[repo]` commit that cites that audit.

### 7. The toolchain

`collectors/toolchain.snapshot(provider)` records the driver, CUDA, torch and the Nsight versions before each
measurement. It asks the adapter's `python()` for torch, so a driven engine in its own venv reports that venv's
torch. Give the adapter a `python()` when it runs under its own interpreter. Missing tools are `unknown`, never
filled from the pin table.

## The module, as a checklist

A driven engine is about 100 lines plus its driver script. `vllm_decode.py` (82 lines) is the shortest complete one.

```python
PROVIDER = "myengine"                      # the name on screen and in every run
PROVIDER_ID = "myengine/offline-decode"    # provider + method, recorded in traces
TRACE = "myengine_timing_trace.json"       # the per-role trace this adapter writes
FORMATS = ("safetensors",)                 # the weight files it reads
BACKENDS = ("CUDA",)                       # where it runs; elsewhere it is "not compatible", not "not installed"
ENV = "BOLTBEAM_MYENGINE_PYTHON"           # the one env var that names it when the scan misses
DRIVER = ec.RUNTIME / "myengine_decode_drive.py"   # runs under the engine's own python; imports only the engine
KV_ELEMENT = (2, "myengine's default KV cache dtype, bf16")

def python(env=None): ...                  # the engine's interpreter (engine_common.python_with)
def available(target, env=None): ...       # None, or one sentence saying why it cannot measure here
def argv(model, contexts, batches, *, tokens, reps, once=None, context_limit=None): ...
def whole_step(model, contexts, batches, *, log=None, tokens=..., settings=None): ...   # 1: truth timing
def role_time(run, *, target, model, model_id, context, batch=1, tokens=...): ...      # 2: capture
```

Then:

1. Add the module to `ENGINES` in `providers.py`, and to `DRIVEN` when it runs through a driver.
2. Add its env var and search places to `engine_scan.ITEMS`, so `doctor` finds it.
3. For prefill, add `truth` and `capture` to `prefill.ENGINES` (step 3, 4, 5 above).
4. Add its pin to `engine_pins.json`.
5. Run it twice on one model and derive repeat bands (below). Until then every verdict reads "not checked".

## How to know the adapter is right

These are the checks the existing adapters pass. Run them on a real GPU; a unit test cannot show them.

- **Work matches.** The captured matrix work per token equals the model's own inventory, to the FLOP
  ([accuracy-fixes-20261012.md](accuracy-fixes-20261012.md)). A missing matrix means the capture or the pairing lost
  a kernel.
- **Every role pairs.** No `unpaired` roles, no `ambiguous` pairing. The screen shows both when they happen.
- **The arithmetic ties out.** Kernels plus gaps equal the captured wall within 0.001 ms.
- **End to end.** The captured time against the untraced median, inside a measured repeat band:

```bash
python -m boltbeam.workflow.screen pipeline MODEL --run runs/a --target T --provider myengine --workload decode --analyze --no-search
python -m boltbeam.workflow.screen pipeline MODEL --run runs/b --target T --provider myengine --workload decode --analyze --no-search
python -m boltbeam.workflow.screen repeat-bands --runs runs/a runs/b runs/c runs/d --out bands.json
python -m boltbeam.workflow.screen pipeline MODEL --run runs/c --target T --provider myengine --workload decode --analyze --no-search --bands bands.json
```

- **Cross-engine.** Run a second engine on the same model. The work and the ceilings must agree; only the times may
  differ. The vLLM and llama.cpp cross-checks are worked examples
  ([prefill-vllm-crosscheck-27b-rtx5090-20261011.md](prefill-vllm-crosscheck-27b-rtx5090-20261011.md)).

## What the adapter does not do

It does not compute ceilings, pair kernels, judge roles or write the report. Those read the trace it writes. If an
adapter needs a model name, a shape or a chip name in its code to work, the shared code is missing a fact: add the
fact there instead.
