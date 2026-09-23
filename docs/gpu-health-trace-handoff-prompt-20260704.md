# GPU Health / 14B Trace Handoff Prompt

Date: 2026-07-04

Read this first when resuming the 14B prefill parity work.

## Prompt To Self

You are resuming BoltBeam + tinygrad work on Qwen3 14B prefill parity against llama.cpp on AMD gfx1100. Do not start by
running another long GPU experiment. First check the host health and use BoltBeam's guarded trace path.

The previous session wedged the AMD GPU stack while moving from timing traces to hardware/counter traces. This was not
identified as a normal code crash or permissions issue. The observed state was:

- PCI still saw `08:00.0 Navi 31 RX 7900...`
- `/dev/kfd` existed
- `/dev/dri` was missing
- `rocminfo` saw only CPU
- `rocm-smi` saw no GPU
- a dead `python3` still held `/dev/kfd`, deleted `/dev/dri/renderD128`, PCI config, and BAR/resource handles
- `amdgpu` bind failed with duplicate sysfs objects and kernel oopses
- PCI reset hung and produced unkillable `D`-state tasks

Conclusion: the system was below user-space recovery. It needed reboot or physical power-cycle. More sysfs reset/bind
attempts were likely to create more stuck tasks.

## Relevant Commits

- `09812e3 [cli] guard gpu trace collection health`
  - adds `boltbeam/gpu_health.py`
  - adds `collect-hw-trace --gpu-health off|warn|fail`
  - detects missing render nodes, GPU-related `D`-state tasks, and dead/blocked GPU FD holders
- `acc4057 [cli] add prefill role counter coverage`
  - reports required counter coverage for prefill roles
- `347ad7003 [trace] derive prefill memory counters` in tinygrad
  - derives some memory counter fields from tinygrad PMC stats where present

## Rules For The Next Run

1. After reboot/power-cycle, run a health check before any benchmark:

```bash
ls -l /dev/kfd /dev/dri
rocminfo | rg -A5 'Agent|Name:|Device Type'
rocm-smi
```

2. Use BoltBeam's guarded collector for real traces:

```bash
boltbeam collect-hw-trace ... --gpu-health fail
```

If this fails, stop. Do not try to push through with profiling.

3. Keep the first run narrow:

- model: Qwen3 14B Q4_K_M
- workload: prefill
- context: 512
- first role: `ffn_gate_up Q4_K [512,17408,5120]`

4. Do not use broad long experiments until the guarded 14B pp512 trace can run and post-run health remains clean.

## Why This Matters

The main technical objective is still to explain tinygrad's 14B prefill gap against llama and the roofline. Timing-only
traces showed the hot role, but did not prove the cause. The missing evidence is per-role hardware signals:

- elapsed time
- tok/s
- bytes / effective GB/s
- workgroup/grid shape
- VGPR/SGPR
- LDS/scratch
- occupancy
- memory-busy
- VALU-busy
- MFMA utilization, if available

Do not collapse these into one vague "bandwidth" diagnosis. Use BoltBeam role traces and counter coverage to say what is
present, what is missing, and what the evidence quality is.

## Current Working Theory

For 14B prefill, the top tinygrad gap is `ffn_gate_up Q4_K [512,17408,5120]`. The strongest hypothesis from the prior
analysis is that tinygrad is effectively not amortizing Q4_K unpack/dequant across the token tile like llama does. The
important distinction:

- bad: 512 independent GEMVs, dequant repeated per token
- good: tiled GEMM/MMQ-style path, dequant once per weight tile and reuse across token tile

But do not implement based only on this hypothesis. First collect the guarded BoltBeam hardware trace so the next
tinygrad change is tied to measured occupancy / memory / VALU / MFMA behavior.

## Stop Conditions

Stop immediately if:

- `/dev/dri/renderD*` disappears
- `collect-hw-trace --gpu-health fail` reports GPU-related `D` tasks
- a dead or blocked process holds `/dev/kfd` or deleted render nodes
- `rocminfo` stops seeing the GPU
- dmesg shows amdgpu reset/bind/oops symptoms

The goal is not to avoid all failures. The goal is to avoid turning one failed trace into a full GPU reset cycle.
