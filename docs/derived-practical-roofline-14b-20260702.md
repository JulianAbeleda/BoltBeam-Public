# 14B Derived Practical Roofline

> **Artifact location, 2026-07-31.** The `bench/` measurement files cited below were
> pruned from the trunk as raw measurement (see
> `docs/task_workflow/input/bench-decouple-scope-20260731.md`). They remain on the
> `dev` and `exp` branches per `docs/branch-flow.md`; recover with
> `git show dev:<path>`. Verdict-bearing artifacts stayed on the trunk.


Date: 2026-07-02.

This audit separates four targets that were previously too easy to conflate:

- **Current-route closeout ceiling:** what the current tinygrad route family can credibly reach from measured or
  route-expressible changes.
- **llama parity:** an external implementation comparison target.
- **Practical hardware/workload roofline:** a conservative target beyond llama, derived as a fraction of raw file
  bandwidth roofline.
- **Raw roofline:** hardware context only, not a promotion target by itself.

## Inputs

- Current tinygrad 14B decode: **53.8 tok/s** = **18.59 ms/token**.
- Practical roofline formula:
  `practical_tok_s = 960 GB/s * 0.75 / GGUF_file_bytes`.
  This makes the roofline model-size-derived, not hardcoded. The 75% factor is the conservative practical-efficiency
  line used in the prior audit; raw roofline remains `960 GB/s / GGUF_file_bytes`.
- Source artifacts:
  - `bench/practical_roofline_14b_refreshed_20260702.json`
  - `bench/refreshed_loss_stack_14b_20260702.json`
  - `/home/ubuntu/tinygrad-arkey/bench/system-fusion-sf4-aggregate/latest.json`
- Contexts in the source audit: ctx512 and ctx2048. ctx4096 was intentionally not run.

## Route-Family Floors

The conservative rule is: only book a saving if the current route has measured evidence or an already expressible
default-off candidate. Primitive-missing buckets stay at current time for the **route-family closeout** line. This is
not the practical roofline; it is the line that says whether ordinary route tuning is exhausted.

| bucket | current share | current ms/token | booked floor | basis |
|---|---:|---:|---:|---|
| Q4_K GEMV | 40.6% | 7.54 | 7.54 | route/bandwidth ceiling |
| reduce / combine | 19.6% | 3.65 | 3.28 | only measured fused-combine route saving is booked |
| attention tile / PV | 16.9% | 3.15 | 3.15 | primitive-missing, unpriced |
| Q6_K GEMV | 11.1% | 2.06 | 2.06 | route ceiling |
| other / elementwise | 7.9% | 1.46 | 1.46 | aggregate elementwise was tested and refuted below noise |
| lm_head | 4.0% | 0.74 | 0.74 | at ceiling |

## Separate Ceiling Lines

| basis | tok/s | ms/token | current % | gap |
|---|---:|---:|---:|---:|
| current default | 53.8 | 18.59 | 100.0% | 0.0 tok/s |
| current route + measured fused combine | 54.9 | 18.21 | 98.0% | 1.1 tok/s |
| current route + ideal aggregate elementwise | 56.6 | 17.68 | 95.1% | 2.8 tok/s, **refuted projection** |
| **confirmed route-family closeout ceiling** | **54.9** | **18.21** | **98.0%** | **1.1 tok/s** |
| llama parity | 66.0 | 15.15 | 81.5% | 12.2 tok/s |
| **conservative practical roofline** | **80.0** | **12.50** | **67.3%** | **26.2 tok/s** |
| illustrative raw roofline | 107.0 | 9.35 | 50.3% | 53.2 tok/s |

## Cross-Model Consistency Check

Using the same formula across Qwen3 Q4_K_M sizes:

| model | GGUF bytes | raw tok/s @ 960 GB/s | practical tok/s @ 75% | representative tinygrad | tinygrad % practical | llama % practical |
|---|---:|---:|---:|---:|---:|---:|
| 8B | 5,027,783,488 | 190.9 | 143.2 | 103.5 | 72.3% | 68.9% |
| 14B | 9,001,752,960 | 106.6 | 80.0 | 53.8 | 67.3% | 82.5% |
| 32B | 19,762,149,024 | 48.6 | 36.4 | 24.8 | 68.1% | 84.5% |

This is the consistency check that prevents the 14B practical roofline from being a hardcoded `80 tok/s` target. It is
`960 GB/s * 0.75 / 9,001,752,960 bytes = 79.98 tok/s`, rounded to 80.0.

## Conclusion

The confirmed current-route closeout ceiling is about **54.9 tok/s**. Against that line, current tinygrad is
**98.0%**, with about **1.1 tok/s** or **0.37 ms/token** left. That remaining route-level delta is the measured
default-off fused-combine candidate.

That is **not** the practical roofline. The practical roofline should be beyond llama. Using the formula above, the
14B raw file-bandwidth ceiling is **106.6 tok/s**, and the conservative 75%-of-raw practical roofline is
**80.0 tok/s**. Against that target, current tinygrad is **67.3%** of practical roofline, with about
**26.2 tok/s** or **6.09 ms/token** left.

The earlier **56.6-57.8 tok/s** route projection depended on ideal aggregate elementwise fusion. That projection is
now refuted by the SF4 aggregate artifact: correctness passed and the route fired, but only the silu-gate fragment
fused, residual/norm fragments did not, and W==D moved only **+0.2 tok/s**, below the 0.5 tok/s threshold.

The 66 tok/s llama line is still useful, but it is an **external parity target below practical roofline**, not the
roofline itself. Reaching llama from the confirmed route closeout ceiling requires another **3.06 ms/token** of savings.
Reaching the 80 tok/s practical roofline requires another **6.09 ms/token** from current. Those missing slices must be
priced by primitive-specific ceilings, mainly:

- attention PV / vectorized-PV or v_dot2 lowering;
- attention combine coordination beyond the refuted expressible transforms;
- any non-overlapped system overhead that survives aggregate elementwise fusion.

## Next Audit Step

The approved aggregate elementwise candidate has now landed below noise, so the route-level practical roofline falls
back to the measured fused-combine closeout line. The remaining gap to llama, and then to practical roofline, should be
classified as primitive-missing or bandwidth-efficiency work, not route-tunable.

If the goal is to keep climbing beyond the derived route roofline, the next BoltBeam scope should be a primitive-ceiling
audit for attention PV and combine. That audit should measure ceilings for those primitive classes directly instead of
borrowing the llama parity number.
