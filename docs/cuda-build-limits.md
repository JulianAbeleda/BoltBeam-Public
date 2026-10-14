# CUDA build and benchmark memory limits

Run memory-heavy jobs through `tools/run_bounded.py`. The resumed NVFP4 campaign
also uses this runner. It requires noninteractive sudo access to the system
manager and fails if that access is unavailable. Commands execute as the calling
user, in `/boltbeam.slice/`, outside the interactive user manager.

```sh
python3 tools/run_bounded.py --log /path/to/benchmark.log -- /path/to/python benchmark.py
```

The runner reserves at least 4 GiB (or one eighth of physical RAM, whichever is
larger) from currently available RAM. The shared system slice also reserves that
headroom from physical RAM, so concurrent jobs cannot each consume the full
allowance. This replaces the earlier two-thirds cap, which left less than one
GiB above a 20.4 GiB checkpoint for its runtime and induced loading stalls.
An explicit
`--memory-limit-bytes` can only lower the job cap. These slice properties are
runtime settings reapplied by the runner, not a permanent system configuration.

`MemoryHigh` is unlimited: the old wrapper's low throttle threshold caused
reclaim pressure well before its hard cap. `MemoryMax`, `MemorySwapMax=0`,
`OOMPolicy=kill`, and `KillMode=control-group` contain failures to the work.
A watchdog stops a job after five consecutive seconds with at least 20% full
memory stalls. Timeouts and interruption also stop the service's entire process
group. The runner writes a log, memory/pressure samples, and a status JSON file.

## Prebuild before loading a model

The vLLM driver defaults to `BOLTBEAM_REQUIRE_PREBUILT=1`, inherited by spawned
workers. Its Ninja hook dry-runs the exact requested targets and raises
`Prebuild required` if any work remains or the dry-run fails. Explicit `-j` and
`MAX_JOBS` do not bypass this check. This guard covers Python launches through
`subprocess.run`; the cgroup cap also contains other compiler launch paths.
An explicit `BOLTBEAM_REQUIRE_PREBUILT=0` enables JIT for isolated development;
use it only within the bounded runner.

To calibrate all CUDA objects and complete a module without a loaded model:

```sh
python3 tools/prebuild_cuda.py /path/to/build.ninja \
  --ninja /path/to/ninja --report /path/to/build-profile.json
```

The prebuild helper uses the same isolated system runner for each CUDA object
and the final link. It measures cgroup peaks and resumes matching reports.
`--memory-limit-bytes` lowers its cap; `--include-dir` supplies CUDA headers.
`--measured-target OBJECT=BYTES` imports an earlier observed measurement.
`--enrich-report` adds build identity to a completed live-sampled report without
recompiling. Profiles cover the Ninja file, source contents, compiler, commands,
and observed header timestamps; a profile for one module does not calibrate
another module.

For development builds that allow JIT, a matching profile derives a worker
count with one extra compiler peak reserved. Unknown builds default to one
worker. This concurrency control is not a memory guarantee; use the isolated
runner even with a single worker.

## Autotuning also needs prebuilt helpers

FlashInfer's tuning timer uses `trtllm_utils` (`kernels_delayStream.cuda.o`), even
when the model's GEMM modules are already built. Prebuild its generated
`build.ninja` with the helper above before tuning. A missing helper used to raise
an ordinary `RuntimeError`; FlashInfer caught it per tactic, retried every
candidate, fell back to default kernels, and reported `Saved 0 configs`.

The prebuilt gate now raises `PrebuildRequired`, a deliberate `BaseException`
subclass. A build-policy failure must escape libraries' `except Exception`
fallbacks and abort the benchmark. The bounded runner still cleans up the whole
service. Do not catch this exception as an unsupported kernel or disable the
prebuilt gate to get through warmup.

Verify cache reuse with two fresh processes using identical model and engine
settings. Keep autotuning enabled. The first must save nonzero configurations;
the second must load them and report cache hits for compatible policies. Cache keys
include engine settings, so a different context/batch configuration can require
new tuning. Keep startup time separate from measured inference latency. Old
runs that saved zero configurations measured fallback kernels; remeasure them
after fixing the helper rather than treating them as tuned baselines.

Validation on RTX 5090 / vLLM 0.31.0 / FlashInfer 0.7.0.post1:

- A standalone CUTLASS FP4 probe changed from 0/5 successful configurations to
  5/5 after prebuilding `trtllm_utils`; the helper build peaked below 400 MiB.
- Two fresh Qwen3.8-27B NVFP4 prefill processes (512 tokens, two warmups, three
  measurements) succeeded with the prebuilt gate enabled. The first saved 55
  configurations; the second loaded 55 and reported FP8 config-file hits.
- Total process time was 49.4 s then 46.5 s, versus 321.5 s before the fix.
  These are startup-plus-measurement times, not inference latency.
- FP4 still retuned for about 3 s: this FlashInfer version deliberately ignores
  v1 disk entries during tuning when the requested replay/L2 policy differs
  from the default, because those entries lack policy metadata. Do not bypass
  that compatibility check. This fix removes the failed-tactic loop; it does
  not promise every kernel skips tuning.
- 22 focused unit tests passed, including a swallowed-build regression and
  signal termination classification. A live SIGTERM check also returned a
  nonzero benchmark status even when systemd considered the stop successful.
