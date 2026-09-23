# Qwen3-8B Metal cross-runtime trace

This is the compact record of a matched llama.cpp versus tinygrad fixed-depth decode trace on the Apple M4
10-core GPU. Full local xctrace packages and generated run artifacts are intentionally excluded from Git.

## Matched profile

- model: `Qwen3-8B-Q4_K_M.gguf`
- model SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- workload: one decode token at fixed depth 128, one warmup, one measured sample
- target: `apple_m4_10c` / Metal
- collector: `metal-system-trace`, xctrace 16.0 (17F42)
- llama.cpp revision: `4f0e43da6f8f6e9390d88409610098ec2d2dc5c7`
- tinygrad EXP revision: `1f9a3c48ddfe9a095573c51457dfc76651e045c6`

Both runtime profiles resolved through the same provider/target/operation capability path. Only the runtime
adapter and configured executable differed.

## Result

| metric | llama.cpp | tinygrad |
|---|---:|---:|
| tok/s | 19.688 | 11.210 |
| whole-step wall | 50.792 ms | 89.240 ms |
| selected command buffers | 3 | 101 |
| selected GPU interval union | 49.108 ms | 87.737 ms |
| host or unattributed | 1.684 ms | 1.503 ms |

Tinygrad reached 56.9% of llama.cpp throughput and took 1.757x its wall time. Both runs spent roughly 97–98% of
the measured token inside bound GPU intervals, so the first-order gap is not Python or host launch latency.
The visible difference is device work and its decomposition: llama.cpp submitted a compact three-buffer token;
tinygrad's runtime-reported 101 JIT calls bound to 101 buffers. Tinygrad's selected labels split into 24 batched
buffers (76.817 ms summed), 45 generated `r_` buffers (10.848 ms), and 32 generated `E_` buffers (0.072 ms).

This trace does not prove which shader instruction, tensor role, or physical-byte path owns the excess work.
Metal System Trace does not export per-shader duration or kernel resources here. The supported conclusion is:
the current tinygrad token performs materially more GPU work in a much more fragmented command-buffer sequence;
the next trace needs a finer Metal/runtime label bridge before assigning the gap to a specific kernel family.

See [portable trace profiles](../../docs/portable-trace-profiles.md) for the reusable architecture and evidence
rules. Exact compact values are in [evidence-summary.json](evidence-summary.json).
