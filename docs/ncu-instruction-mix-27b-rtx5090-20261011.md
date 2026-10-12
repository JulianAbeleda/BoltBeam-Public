# What ncu showed the closed kernels execute: the 27B vLLM prefill at 512, RTX 5090, 2026-10-11

BoltBeam reads a kernel's matrix path from its code. For a closed kernel (no readable PTX or machine code) it falls
back to a documented name pattern (`instruction_path.NAME_PATHS`). This run used Nsight Compute once to see what
those kernels really executed, and folded what it saw back into the name rules. BoltBeam does not need ncu: the
name rules are what run on a box without it.

## The run

- One vLLM prefill of 512 tokens of the 27B (AWQ INT4 weights, bf16 activations), the measured call only
  (`--profile-from-start off`).
- ncu `SourceCounters` on the kernels whose code was not read: the first 200 such launches, 40 of each of 5 kernels
  (40 of the 48 recurrent layers). It took about 54 minutes.
- Each SASS matrix opcode is mapped to a path by the probe's own opcodes (`instruction_path.sass_signatures`).
- Counters: the driver parameter is `NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/nvidia-profiling.conf` (docs/linux.md).

## What ncu saw, against the name route

| kernel | executed (warp instructions, 40 launches) | name route said | verdict |
|---|---|---|---|
| CUTLASS wmma bf16 GEMM (in_proj_ba) | HMMA.16816.F32.BF16: 6.51M | bf16 | right |
| delta rule TPrecompute | HMMA.16816.F32.BF16 3.93M, HMMA.16816.F32 (fp16) 0.61M, HMMA.1688.F32 (fp16) 0.12M | fp16 | mostly bf16; same measured rate |
| delta rule MNPrecompute | HMMA.16816.F32.BF16: 33.4M | fp16 | bf16; same measured rate |
| delta rule Prefill | HMMA.16816.F32.BF16: 35.4M | fp16 | bf16; same measured rate |
| delta rule FixupHmma | HMMA.1688.F32.TF32: 3.93M | fp16 | wrong: tf32 |

- The kernel source the old rule cited says `MmaF16BF16Op(dtype, ...)`: the type follows the model's activations.
  On this bf16 model that is bf16. bf16 and fp16 run at one measured rate here, so the old rule's ceilings were not
  off for those three.
- The fixup kernel runs tf32, which is a different unit and rate. The old rule put it on the fp16 peak.

## What changed in the generic path

- `NAME_PATHS`: TPrecompute, MNPrecompute and Prefill now map to bf16, and FixupHmma maps to tf32. Each rule
  cites this run.
- A tf32 probe path now exists (`cuda_compute.PATHS`, `mma.sync.aligned.m16n8k8...tf32`). On the 5090 it measured
  125.7 TF at 2,898 MHz, 255.1 operations per SM per clock: half of bf16 (251.4 in the same run). So the recurrent
  row holding the tf32 fixup has a compute ceiling again, on the right unit ("bf16 + tf32"). It stays bound by its
  bytes: 2.32 ms against 7.25 ms measured.
- `instruction_path.ncu_observe` stays as an optional check (`BOLTBEAM_NCU=1`, counters open). It profiles only
  kernels with no readable code. Its path ranks below code and above a name, and the row says "observed by ncu".

## The recurrent work, checked

- At 512 tokens per layer: the model's work is 2.82e9 operations (7 x 128^2 x 48 heads x 512). The chunked WY
  estimate is 4.43e9 (6 x 128^2 + 10 x 64 x 128 per token per head).
- ncu counted 7.72e9 matrix operations per layer in the four delta-rule kernels (m16n8k16 = 4,096 operations a
  warp instruction, m16n8k8 = 2,048). That is 1.74 times the estimate. The estimate is a floor.
- The ceiling keeps the model's work. The executed figure is a separate, labelled field (`executed_flops`).

## Evidence

On the box: `~/storage/bb-t3/runs/p27-vllm-ncu/kernel_compare/prefill_capture/512/ncu/` (the report, 157 MB, and
its log), and `~/storage/bb-t3/ncu-src.csv` (the SASS export).
