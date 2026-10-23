# Matrix-path scan

A prefill kernel's ceiling is its work over the rate of the matrix instruction it issues, so BoltBeam measures one
rate per matrix path (`hip_compute`, `cuda_compute`). Which paths a chip has used to be a table written per ISA
family. It is now found by asking the installed toolchain for the GPU's own target. The hand tables
(`hip_compute.FAMILIES`, `cuda_compute.PATHS`) are the verified reference the scan has to reproduce, and the
fallback where no toolchain is installed.

    python -m boltbeam.collectors.hip_path_scan gfx1201 [--hipcc PATH] [--refresh]
    python -m boltbeam.collectors.cuda_path_scan sm_120 [--nvcc PATH] [--nvdisasm PATH] [--refresh]

## How it works

**AMD** (`collectors/hip_path_scan.py`)

1. **Candidates.** `HIPCC_VERBOSE=1 hipcc --version` names the clang that hipcc runs. Every
   `__builtin_amdgcn_*wmma*` name is read from that binary and from the clang libraries it links: a distro clang
   keeps its builtins in `libclang-cpp`. There is no hand list of builtin names.
2. **Signature.** All candidates are compiled together in one file with `-fsyntax-only -ferror-limit=0` for the
   target, one call per line. A call with no arguments makes clang state the parameter count. Passing a struct in
   position *i* makes clang state that parameter's type, and assigning the call to a struct states the result type.
   That is one compile per parameter position, about ten in all.
3. **Driveable.** A bool before the accumulator is a signed-operand flag (`true`). A bool after it is OPSEL or clamp
   (`false`). The probe drives two equal inputs and one accumulator of the result type. Anything else is put in
   `excluded` with the reason: sparse `swmmac` (its index operand needs a 2:4 pattern the probe does not build), and
   builtins that take modifier operands.
4. **On this chip.** `hip_compute.source()` is compiled for the target through `build_code_object`, using
   BoltBeam's HIP cache, in parallel. If the compiler rejects it, the builtin is not on this chip; the compiler's
   words go in `rejected`.
5. **One instruction.** `llvm-objdump` must show exactly one distinct `v_wmma_*` mnemonic, and its fields must be
   the builtin's own. When several builtins lower to the same mnemonic (`_tied`, for example), the shortest name is
   kept.
6. **Name.** The name comes from the mnemonic `v_wmma_<acc>_<MxNxK>_<in>` through `NAMES`:

   | acc, in | name |
   |---|---|
   | f32, f16 | fp16 |
   | f16, f16 | fp16_f16acc |
   | f32, bf16 | bf16 |
   | i32, iu8 | int8 |
   | i32, iu4 | int4 |
   | f32, fp8_fp8 | fp8 |
   | f32, bf8_bf8 | bf8 |

   When two paths would get one name, the one with the largest K keeps it. Any other path is named after its
   mnemonic (`wmma_bf16_16x16x16_bf16`), marked `mapped: False`, measured, and listed in `unmapped`. No weight format
   maps to an unmapped path: `roofline/prefill_work` owns that rule, and a person adds to it. The operand typedefs,
   values and words are derived from the diagnostics and the mnemonic.

**NVIDIA** (`collectors/cuda_path_scan.py`)

The toolkit ships no PTX grammar (there is none under `include/`, and the one in ptxas is not readable). So
`GRAMMAR` writes out the PTX ISA's own `mma.sync.aligned` dimensions for the m16n8 family, unfiltered: 5 shapes,
the operand types, f32/f16/s32 accumulators, and the kinds (none, `kind::f8f6f4`, and the `block_scale` kinds with
every `scale_vec` and scale type). The A/B fragment sizes follow from each element width, and a sub-byte type is
also tried as an 8-bit container. That gives 5112 forms.

ptxas checks all of them in one PTX module for `sm_XY`, then the remainder for `sm_XYa`. ptxas stops reporting after
some errors, so the module is re-run without the lines it named until it passes. Each surviving form's
`cuda_compute.source()` probe is compiled with nvcc and disassembled with nvdisasm (as `sass_signatures` does), and
kept when it issues exactly one distinct matrix opcode. When several forms lower to one opcode, the named one is
kept. Names come from `NAMES`, keyed on (kind, accumulator, input, scale type), with the largest K winning.
`cuda_compute.measure_paths` measures the named paths. The unmapped ones are measured only when passed in `paths`:
on sm_120 there are 94 of them, which would add about 15 minutes to a chip profile.

**Kept and reused.** A scan record is cached per toolchain identity (binary path, size, mtime) and target, in the
vendor cache. The matrix probe's result carries it as `path_scan`, and `workflow/chips.profile_row` keeps it as
`capabilities.matrix_path_scan`. `targets.stored_path_scan(arch)` reads it back. `hip_compute.paths_for(arch)`
returns a profile's stored scan, otherwise the family table. A target with neither is refused, as before.
`measure_paths` runs (or reuses) the scan whenever its bridge has a compiler. `rocm_isa` maps mnemonics through the
same `paths_for`, so an unknown family that has been scanned is readable, and an unscanned one is still refused.

## What it verified (2026-10-23, this box)

| scan | candidates | result | time |
|---|---|---|---|
| gfx1100, hipcc 5.7 / clang 17 | 12 | GFX11 **exactly** (every field, every probe source byte-identical); +1 unmapped `wmma_bf16_16x16x16_bf16` | 1.5 s |
| gfx1100, ROCm 7.2.4 clang 22 | 103 | GFX11 **exactly**, same +1 unmapped | 3.4 s |
| gfx1201, ROCm 7.2.4 clang 22 | 103 | GFX12 names, builtins, mnemonics, K, accumulators, calls, words, lanes **exact**; see below; +4 unmapped (`wmma_bf16_16x16x16_bf16`, `wmma_f32_16x16x16_bf8_fp8`, `wmma_f32_16x16x16_fp8_bf8`, `wmma_i32_16x16x16_iu4`) | 2.8 s |
| gfx1201, clang 17 | — | no gfx1201 target: a record with that reason, no paths; GFX12 stays the fallback | 0.2 s |
| sm_120, CUDA 13.2 ptxas + nvdisasm (cu13 wheel) | 5112 | all 9 `cuda_compute.PATHS` **exactly** (fp4/nvfp4 block-scale forms with arch suffix `a`, scale bits, words); +94 unmapped | 6.4 s |

A cached record reads back in milliseconds.

**The gfx12 difference.** clang 22 declares the gfx12 bf16 builtin on `short` vectors. So the scan's bf16 probe uses
`s8 a = (s8)0x3a83` (bf16 0.000999) where the hand table has `b8 a = (b8)(__bf16)0.001f`. The hand form compiles too,
through lax vector conversion, and both lower to `v_wmma_f32_16x16x16_bf16`. Because `s8` joins the shared typedef
preamble, every gfx12 probe source gains one `typedef short s8` line. The instructions, rates and names are
unchanged.

The gfx12 scan needs `-isystem /opt/rocm-7.2.4/include` on this box, because two ROCm installs share
`/usr/include/hip`. The test wraps each `/opt/rocm*/bin/hipcc` that way.

Tests: `tests/test_matrix_path_scan.py`. The real-compile tests are skipped without a toolchain. The sm_120 test
needs nvdisasm, which `/usr/local/cuda` on this box does not have; set `BOLTBEAM_NVDISASM` to an engine's.

## Limits

- **gfx1201 on a real R9700 compiler is unverified.** The gfx12 check ran with ROCm 7.2.4's clang 22 here, and no
  scan has been measured on a GPU yet. Only the compile, disassembly and naming are verified.
- **NVIDIA scope.** Only the m16n8 `mma.sync` forms are scanned. m8n8 shapes, f64, b1, sparse `mma.sp`,
  `.satfinite`, wgmma and tcgen05 are not. `sass_signatures` still compiles `PATHS`, not the scan.
- **AMD builtins with modifier operands** (gfx1250's neg/reuse/clamp-mod forms) and sparse `swmmac` are excluded,
  not probed.
- **Backend crash dumps.** A builtin of another family crashes clang's backend, and that is read as "not on this
  chip". Each crash leaves clang's crash files in `/tmp`.
- **Unmapped paths have no weight-format mapping.** That rule needs a person.
- **No staleness check on a stored profile scan.** `stored_path_scan` uses the first profile for the target,
  whichever compiler made it. `measure_paths` always rescans with the installed compiler (cached per compiler).
  Only `paths_for`/`rocm_isa` read the stored record. A scan that finds no matrix path, or cannot run (a missing
  tool, a timeout), is not kept, and the hand table is used.
