# Q4_K_M versus AWQ and NVFP4 on the RTX 5090

On this Qwen3.8-27B chat replay, AWQ on vLLM finished responses fastest; NVFP4 on
vLLM delivered the first token fastest. Q4_K_M on llama.cpp was slower in the
completed matched run, which required disabling CUDA graphs. These results compare
specific checkpoints and engine configurations, not weight formats in isolation.

## What belongs to the format and what belongs to the engine

| Question | Finding |
| --- | --- |
| What does Q4 mean here? | The specific GGUF Q4_K_M checkpoint, not every four-bit quantization. GGUF is the container; Q4_K_M describes its weight quantization recipe. |
| Does four-bit imply equal memory traffic? | No. These checkpoints use different precision for different tensor roles, plus scales and metadata. |
| Is a 250K context a Q4 advantage? | No. Weight quantization and KV-cache precision are separate choices. The earlier llama.cpp run used q8_0 KV; the matched replay used BF16 KV on all engines. |
| Why might someone still choose llama.cpp with Q4? | CPU and Apple Silicon support, multiple GPU backends, CPU/GPU offloading, and native C/C++ deployment. These are engine/ecosystem capabilities, not inherent Q4 speed advantages. |
| Does this establish the best answer quality? | No. Exact completion lengths validate replay shape, not semantic accuracy or quantization quality. |

The [llama.cpp project](https://github.com/ggml-org/llama.cpp) documents its hardware
and offloading support. [vLLM's KV-cache documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
describes FP8 cache configuration and scaling considerations. FP8 KV and q8_0 KV
are different encodings; matching their nominal bit width does not make them identical.
Converting weights into GGUF Q4_K_M produces a different quantization; it does not
measure the original AWQ or NVFP4 checkpoint executing unchanged in llama.cpp.

## Matched chat replay

One 300-turn session per engine, October 14, 2026, RTX 5090, 600 W power limit,
automatic clocks. Same counts trace and seed 0, 114688 context, BF16 K/V, batch 2048,
one sequence, prefix caching, speculation off and thinking off. All engines produced
exactly 37354 requested completion tokens. Startup is excluded; the cold first turn
is included. Model replies become later history, so the histories are not byte-identical.

| Checkpoint and engine | Median first token | Median complete reply | Total response time | Median decode | Throttled turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| W4A16-AWQ, vLLM | 444.5 ms | 2146.6 ms | 658.7 s | 71.10 tok/s | 61/300 |
| NVIDIA NVFP4, vLLM | 316.6 ms | 2353.0 ms | 723.3 s | 58.91 tok/s | 30/300 |
| Q4_K_M, llama.cpp, CUDA graphs off | 491.6 ms | 2990.9 ms | 922.8 s | 48.17 tok/s | 248/300 |

For streaming, NVFP4 began about 128 ms sooner than AWQ, while AWQ generated the
rest faster. Median per-turn generation time, calculated as each turn's complete
latency minus its first-token latency before taking the median, was 1676.0 ms AWQ,
2025.5 ms NVFP4, and 2486.5 ms Q4. This is one session per configuration, not a
repeated-run confidence interval or a universal ranking.

vLLM was 0.31.0, embedded revision db9527a46; FlashInfer was 0.7.0.post1.
llama.cpp was b11571, revision 097cf33d9. The initial graph-enabled Q4 attempt
failed around turn 208 with a CUDA launch timeout and NVIDIA Xid 8, not an OOM.
The completed run used `GGML_CUDA_DISABLE_GRAPHS=1`, which changes performance.
The [failed attempt](evidence/quant-comparison-20261014/q4-timeout-attempt/driver-error.txt)
and its compressed server log are retained; an identical short retry was manually
stopped, not a second observed crash.
Different kernels, scheduling, recurrent-state handling, and substantial throttling
also prevent attributing the cross-engine gap solely to Q4_K_M.

The [evidence report](evidence/quant-comparison-20261014/report.txt) records exact
server commands, model paths, timestamps and original runs. The
[validation](evidence/quant-comparison-20261014/validation.json),
[summary](evidence/quant-comparison-20261014/summary.json), and per-engine timing
traces in that directory support the table. The run used an already modified
BoltBeam working tree: [code provenance](evidence/quant-comparison-20261014/code-provenance.json)
records its status and collector hashes. Do not assume the base commit alone
reproduces the measurement apparatus.

## Why the NVFP4 result does not prove a kernel cause

The original NVFP4 replay started at 10:34:49 EDT, after the containment fix
1bd1969 at 10:24:54. It saved 80 tuning configurations, not zero. The matched rerun
and a second fresh process confirmed cache use, but the output head still logged
uncovered FP4 shapes for M=1 and M=2. A nonzero saved count is not complete coverage.
See [tuning evidence](evidence/quant-comparison-20261014/tuning-evidence.txt).

The checkpoint's attention is FP8 while its MLP and output head are FP4. Header
accounting gives 17.61 GB for attention + MLP + output head versus AWQ's 15.23 GB,
about 15.6% more stored tensor bytes. This invalidates an explanation based only on
uniform 4.5 versus 4.25 bits per weight. Stored bytes are not measured GPU traffic;
cache, repacking, activations and recurrent state also matter. Neither the bytes nor
the output-head fallback isolates the cause of the observed decode gap. See
[checkpoint accounting](evidence/quant-comparison-20261014/checkpoint-bytes.json).

## Earlier Q4 numbers are a separate run

The 490 ms first token and 774 s session came from the preserved October 13 run,
using 262144 context and q8_0 KV. It was not reconstructed from the October 11
[replay record](replay-27b-rtx5090-20261011.md), which reports 511 ms and about 782 s.
The October 13 command and measurements are in the evidence report. Neither old
run matches the BF16-cache configuration of the new table.

## Long context with FP8 KV

The separate capacity experiment uses existing vLLM flags, with no engine source
patches: `--max-model-len 262144 --kv-cache-dtype fp8`, BF16 compute dtype,
`--gpu-memory-utilization 0.95 --max-num-seqs 1 --max-num-batched-tokens 2048`,
seed 0, prefix caching and speculation off. Missing FlashInfer prefill and decode
modules were prebuilt with the bounded build helper before loading the model.
The prebuilt guard stays enabled. See [CUDA build limits](cuda-build-limits.md).

Results and scope are recorded in the capacity evidence and table below.
Reported cache slots describe allocated capacity, not permission to exceed the
configured or supported model context. A four-code retrieval check over repetitive
filler establishes basic retrieval at those lengths, not broad long-context quality.
Its full-document latency is a different workload from chat replay latency.

The capacity test uses identical token-ID fixtures for both checkpoints, four
planted codes near 10%, 35%, 65% and 90% of each synthetic document, temperature 0,
and a 128-token generation budget. No additional KV-scale calibration was requested.
The maximum input of 262016 leaves 128 tokens within the configured 262144 window.
Prefix caching was enabled, so these single measurements are not a controlled
cold-cache throughput benchmark. AWQ ran first, NVFP4 second, without randomized
order or controlled starting temperature. Both experienced throttling; NVFP4 also
reported thermal throttling on long prompts. Quality equivalence across KV formats
requires a broader evaluation than this retrieval smoke test.

Both jobs exited successfully. AWQ reported **370617 KV-cache token slots** and
NVFP4 **295288**, each with a configured **262144-token context window**. These
figures are specific to the checkpoint, engine allocator and launch settings.

| Input tokens | AWQ four-code retrieval | NVFP4 four-code retrieval | AWQ first token | NVFP4 first token |
| ---: | --- | --- | ---: | ---: |
| 16384 | 4/4 in order | 4/4 in order | 4.44 s | 1.61 s |
| 65536 | 4/4 in order | 4/4 in order | 21.99 s | 10.53 s |
| 131072 | 4/4 in order | 4/4 in order | 55.23 s | 32.57 s |
| 250000 | 4/4 in order | 4/4 in order | 144.56 s | 106.48 s |
| 262016 | 4/4 in order | 4/4 in order | 159.57 s | 119.89 s |

All ten replies contained 39 generated tokens. The [AWQ results](evidence/quant-comparison-20261014/capacity/awq/results.jsonl)
and [NVFP4 results](evidence/quant-comparison-20261014/capacity/nvfp4/results.jsonl)
include usage, prompt hashes, answers, timing and telemetry. Their directories
preserve exact launch commands, timestamps, exit status and compressed server logs;
compressed input-token fixtures sit in the parent directory. These successful
requests demonstrate 250K-plus operation for both weight formats on this GPU.
They do not establish a 295K or 371K supported context window.

Prebuild profiles `capacity/p.json` and `capacity/d.json` record the FP8 prefill
and XQA decode builds. Both final links succeeded, but the helper reported a missing
positive peak-memory sample for the very short link steps. Subsequent Ninja dry runs
reported no work, and both guarded server launches succeeded. Earlier missing-module
attempts were stopped by the prebuild guard before the successful campaign.

Evidence integrity is recorded in [SHA256SUMS.json](evidence/quant-comparison-20261014/SHA256SUMS.json).
Compressed logs can be read with `gzip -cd path/to/file.log.gz`. Original absolute
run paths in the records identify the measurement machine; the relative evidence
links above remain available from a repository clone.


## Headless CUDA graph retry

A subsequent check found `display_active=Disabled` on the RTX 5090. The current
boot's kernel log initially contained only the earlier NVIDIA Xid 8 at 12:27:12 EDT.
With graphs enabled on b11571 (`097cf33d9`), a deep-context retry completed turns
200 through 228, then failed during turn 229. The kernel logged another seven-second
watchdog and Xid 8 at 13:45:14 EDT. The server log confirmed graph reuse before the
failure. This reproduction with the display disabled weakens the attached-display
explanation; it does not isolate a kernel, driver or CUDA-graph root cause.

The quick retry seeded source-document user messages and synthetic assistant replies
with the earlier reply-token counts for turns 1 through 199. It then generated real
responses for turns 200 through 300. Its first request had 74717 prompt tokens,
versus 74651 at turn 200 of the earlier completed replay. This approximates the
context depth; it neither restores the original conversation nor reproduces its
KV/recurrent state and elapsed GPU execution history. The same starting messages
and subsequent user inputs are used for the newer-build check.

The latest upstream revision fetched for the comparison, **b11636 (`529ac893e`,
October 14)**, also failed: turns 200 through 257 completed, then turn 258 lost its
response timings/usage as the server aborted with a CUDA launch timeout. The kernel
recorded the same watchdog/Xid 8 at **14:00:32 EDT**. Graph reuse was confirmed in
the new server's log too. Both jobs exited with errors and cleaned up their GPU
processes; neither failure was an OOM.

| Build with graphs enabled | Completed turns | Failure | Display active |
| --- | --- | --- | --- |
| b11571, 097cf33d9 | 200–228 (29) | Turn 229, watchdog/Xid 8 | Disabled |
| b11636, 529ac893e | 200–257 (58) | Turn 258, watchdog/Xid 8 | Disabled |

The later failure does not establish a stability improvement: these are individual
runs of an intermittent failure, and generated replies can diverge. Updating to this
revision and running headless did not resolve it. Disabling graphs remains the
workaround that completed the earlier full replay. The exact fault still needs
isolation; these observations do not distinguish a llama.cpp bug from CUDA/driver
behavior.

[Retry evidence](evidence/quant-comparison-20261014/graph-retry/comparison-notes.txt)
includes exact commands, binary versions and hashes, per-turn rows, compressed server
logs, identical starting-message fixtures, display state and
[kernel events](evidence/quant-comparison-20261014/graph-retry/kernel-events.txt).
The new binary was built separately, preserving the existing installation. Compilation
used a 16 GiB host-memory cap; after a deliberate stop to increase parallelism, the
incremental build succeeded with a measured 3.07 GB peak. No engine source patches
were applied.
