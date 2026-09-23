# Qwen3-14B authority-aligned prefill roofline — bounded capture

Date: 2026-07-15  
Status: bounded measured + scaled; not a promotion or full-authority result.

## Capture

The current BoltBeam profile-events collector ran the authority code path after
Russell’s authority-mode fix, with:

```text
model:          /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf
target:         AMD gfx1100
K/warmups:      1 / 1
rounds:         1
start_pos:      0
chunk:          512 tokens
whole_lengths:  512, 2048
route:          direct-packed, GRAPH_GEMM=False
```

The exact provider command and trace paths are recorded in
`outputs/qwen3-14b-authority-bounded-20260715.report.json`.

Two timing sources must remain separate:

| source | pp512 | pp2048 | status |
|---|---:|---:|---|
| authority synchronized stdout | 104 tok/s (4,944.1 ms/chunk) | 104 tok/s | measured pp512; pp2048 scaled by the authority report |
| BoltBeam profile-event sum | 353.01 tok/s (1,450.381 ms) | not independently present | measured profile attribution, not whole-wall authority |

The profile trace contains 7,918,387,200 estimated physical bytes and 97.0167%
packed-step attribution. Its weighted trace is
`outputs/qwen3-14b-authority-bounded-20260715.measured-weighted.timing_trace.json`.

## Fresh llama pp512 comparator

Command:

```bash
/home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 512 -n 128 -b 2048 -ub 512 -r 3 -o json
```

Observed prompt row:

```text
n_prompt=512  n_gen=0  n_batch=2048  n_ubatch=512
avg=1834.3067 tok/s  stddev=82.1417  build=ac4cddeb0
GPU=AMD Radeon RX 7900 XTX
```

For wall-rate comparison, this is 279.03 tok/s (`512 / 1.834306702 s`).
The authority stdout’s 103.55 tok/s is 37.1% of that wall-normalized llama
rate. The profile-event sum’s 353.01 tok/s is not a valid whole-wall
comparison and is reported only for attribution.

## Roofline result

BoltBeam’s `prefill-roofline` report is
`outputs/qwen3-14b-authority-bounded-20260715.measured.roofline.json`.
It uses the existing llama pp512 kernel trace for the trusted Q4 effective
rate (25.9989 GB/s), while the candidate wall/role attribution comes from the
new bounded profile capture.

Measured-profile-derived ceilings:

```text
raw source-byte HBM ceiling:       62,073.25 tok/s (sanity bound; non-binding)
Q4 roles at trusted llama rate:       765.65 tok/s (scaled ceiling)
all packed roles at Q4 rate:        1,471.96 tok/s (scaled ceiling)
```

These are scaled/model ceilings, not candidate measurements. The report’s
largest measured packed rows were `ffn_gate_up` Q4_K at 618,347.84 us,
`ffn_down` Q6_K at 370,897.88 us, `attn_qo` Q4_K at 194,527.28 us, and
`ffn_down` Q4_K at 161,916.76 us.

## Limitations and next authority run

This is intentionally bounded: it does not satisfy the normal K=8,
warmups=4, rounds=3 authority completeness protocol, and pp2048 is not an
independent start-position capture. The existing llama kernel trace used for
Q4 calibration is also separate from the fresh llama wall run.

For a promotion-quality refresh, rerun the same collector with
`K=8`, `warmups=4`, `rounds=3`, start positions
`0,512,1024,2048,3584`, and whole lengths `512,1024,2048,4096`, then collect a
matching llama kernel trace in the same session family. Keep the synchronized
authority wall, profile attribution, and any roofline scaling in separate
fields.
