# Qwen3.8-27B short-context comparison on RTX 5090

Measured 2026-10-14. With CUDA graphs enabled, llama.cpp Q4_K_M did not have the
lowest median complete-response latency at any of the three tested input lengths.
AWQ led at 512 tokens; AWQ and NVFP4 were effectively tied at 2,048; NVFP4 led at
4,096. These are fixed-output synthetic requests, not the accumulated chat-300 replay.

## Method and evidence

The [evidence directory](evidence/short-context-20261014/) contains exact server commands,
start/end times, token fixtures, individual timings and telemetry, compressed server logs,
and a SHA-256 manifest. Source run directory on the box:
`~/storage/bbeng/tmp/short-context-20261014`.

All checkpoints are the local Qwen3.8-27B checkpoints used by the
[replay comparison](quant-comparison-rtx5090-20261014.md). Internal architecture names
`qwen35` / `qwen3_5` do not mean that an older checkpoint was substituted.

- llama.cpp b11636, commit `529ac893e754e95a50469b2122f5efe28dfc08a0`;
  vLLM 0.31.0, embedded commit `db9527a46`; FlashInfer 0.7.0.post1.
- RTX 5090, 600 W power limit, automatic clocks, single sequence, 8,192 context,
  BF16 KV cache, batch budget 2,048 (llama batch and microbatch both 2,048).
- Speculation off; llama CUDA graphs on; vLLM graph capture on.
- Exactly matching input token IDs, temperature 0, seed 0, thinking closed in the
  explicit prompt, exactly 128 output tokens with ignore_eos. Prefix reuse disabled.
- Repetitive library-record text, with a request to summarize it. Outputs were inspected
  for coherence; this is not an accuracy evaluation or a representative prompt corpus.
- Each process interleaved 512, 2,048 and 4,096-token requests for ten repetitions.
  The first three repetitions at each length were warmups; seven were measured.
- Client streaming timings exclude server startup. Streaming rate is
  `127 / (response end - first text chunk)` and includes client overhead.
- Bounded model processes and the prebuilt-helper guard remained enabled. NVFP4 saved
  80 tuning configurations; the previously documented output-head M1/M2 fallback remains.

## Results

Medians over seven measured requests per cell. Complete-response time includes the first token.

| Input tokens | llama Q4 complete | vLLM AWQ complete | vLLM NVFP4 complete |
|---:|---:|---:|---:|
| 512 | 1.981 s | 1.626 s | 1.970 s |
| 2,048 | 2.314 s | 2.003 s | 2.001 s |
| 4,096 | 2.817 s | 2.532 s | 2.155 s |

| Input tokens | llama Q4 TTFT / stream | vLLM AWQ TTFT / stream | vLLM NVFP4 TTFT / stream |
|---:|---:|---:|---:|
| 512 | 181 ms / 70.6 tok/s | 141 ms / 85.5 tok/s | 126 ms / 68.8 tok/s |
| 2,048 | 497 ms / 69.9 tok/s | 514 ms / 85.3 tok/s | 149 ms / 68.6 tok/s |
| 4,096 | 978 ms / 69.0 tok/s | 1,032 ms / 84.7 tok/s | 303 ms / 68.6 tok/s |

Requests with sampled power throttling, out of seven:

| Input tokens | llama Q4 | vLLM AWQ | vLLM NVFP4 |
|---:|---:|---:|---:|
| 512 | 0 | 0 | 0 |
| 2,048 | 5 | 3 | 1 |
| 4,096 | 7 | 7 | 3 |

The 11 ms Q4/NVFP4 difference at 512 tokens and 2 ms AWQ/NVFP4 difference at
2,048 tokens should not be treated as meaningful wins. Each engine had one process
launch, in Q4 → AWQ → NVFP4 order; engine order was not randomized. Power throttling,
temperatures and engine cache allocation differed. Repetition within a single launch
does not establish cross-launch confidence intervals. None of these three short runs
reproduced the long-context CUDA graph failure.

## Separating engine and format

The inspected llama.cpp revision does not natively execute this packed AWQ checkpoint:
there is no AWQ branch in its HF quantization converter and no AWQ GGML runtime type.
The older [AWQ discussion](https://github.com/ggml-org/llama.cpp/issues/4701) concerns
conversion/calibration workflows, not a native AWQ execution path. Converting AWQ into
Q4_K_M changes the representation and cannot isolate engine performance.

AWQ versus NVFP4 within vLLM compares checkpoint plus kernel paths within one engine.
It does not establish that either representation is intrinsically faster independent
of its implementation. Comparing llama Q4 with vLLM AWQ also changes the engine.

For a more direct engine control, the same Q4_K_M GGUF file was loaded into vLLM using
its [out-of-tree GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin), commit
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` (0.0.5). The
[vLLM documentation](https://docs.vllm.ai/en/latest/features/quantization/gguf/) labels
GGUF experimental and underoptimized. This is a comparison of available execution
paths, not a theoretical limit for either engine.

The unmodified plugin first demanded a vision projector for this text-only test.
A flattened text configuration then exposed a routing error: `qwen3_5_text` was mapped
to the conditional-generation vision class. One line in an isolated installed plugin
copy was changed to select `Qwen3_5ForCausalLM`. The patch, both failed startup logs,
and the exact configuration are preserved. Model weights and the normal vLLM environment
were unchanged. This control therefore requires a loader correction and is not an
out-of-the-box compatibility claim. Coherent output alone does not prove complete weight
mapping or numerical equivalence.

### Same-Q4-file control result

All 30 requests completed after the loader correction, with matching input token hashes
and exactly 128 output tokens. No new NVIDIA Xid was recorded during the four short runs.

| Input tokens | vLLM GGUF TTFT | Complete response | Stream | Throttled / 7 |
|---:|---:|---:|---:|---:|
| 512 | 1163 ms | 2.949 s | 71.1 tok/s | 7 |
| 2,048 | 4484 ms | 6.276 s | 70.9 tok/s | 7 |
| 4,096 | 9013 ms | 10.815 s | 70.5 tok/s | 7 |

llama.cpp processed prompts substantially faster than this experimental vLLM GGUF path
while streaming rates were similar. The large prompt-processing penalty is therefore
implementation-dependent even when the Q4 file is held fixed. AWQ’s lead over llama Q4
in the 512-token test cannot be explained as a general vLLM engine advantage. These
measurements still do not isolate a format’s theoretical capability from its kernels.
The control ran last, reached 89 C, and needs independent-launch replication before close differences are interpreted.
