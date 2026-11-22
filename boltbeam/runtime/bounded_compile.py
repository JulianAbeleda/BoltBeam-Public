"""runtime/bounded-compile: one compiler run (g++, clang++, nvcc, hipcc, ptxas) on source BoltBeam wrote or was handed
(a search candidate, a generated kernel, an engine kernel), under a hard heap limit and a wall-clock limit, so that one
source cannot take the machine. 2026-11-21: a generated kernel whose macros expanded 31-deep nested fmaxf into ~2^31
copies drove the host emulator's cc1plus to 29 GB, and the global OOM killer took the whole session with it.

The limit is RLIMIT_DATA, set in the compiler's process before it starts and inherited by what it runs (g++ -> cc1plus,
as; nvcc -> cicc, ptxas; hipcc -> clang). Since Linux 4.7 it bounds brk and every private writable mapping, which is
where a compiler's heap lives (glibc malloc, GCC's GC pages, LLVM's allocators), and not shared-library text or
read-only file mappings. RLIMIT_AS would count those too and every reserved-but-untouched range: measured on the
worst generated kernel, clang++ needs 27 MiB of data but 249 MiB of address space, nvcc 81 vs 159 MiB, so an AS cap
tracks the toolchain's layout, not the source. Each process gets its own limit (the stages run one after another).
Linux only: macOS does not enforce RLIMIT_DATA on mmap'd memory and refuses setting it (current limit exceeds maximum
limit), and Windows has no rlimits; there the wall-clock limit is the bound.

The limit is set by a tiny exec shim (python -I -S: setrlimit, then exec), not preexec_fn: build_cubin runs on a
thread pool (cuda_path_scan), and preexec_fn is documented unsafe with threads. The compiler runs in its own session;
on timeout the whole group is killed, not only the driver (a killed g++ leaves its cc1plus running). The shim also
sets RLIMIT_CPU to the time limit, a backstop that holds if BoltBeam itself dies first. TMPDIR is a directory removed
after the run: clang writes a crash reproducer (megabytes of preprocessed source) there when the limit stops it.
The cap is per process: the path scans compile fixed probes on a thread pool, each far below it; a pool of runaway
sources could hold one cap each. Not routed here: engine builds (workflow/engine_update, runtime/prebuild_cuda: the
engine's own build under bounded_job's systemd MemoryMax and build_jobs' measured job count) and Metal, which
compiles in-process (newLibraryWithSource) where no child limit can reach.

The cap, derived (2026-11-21, RTX 5090 box, g++ 13.3, clang++ 18, nvcc 13; /usr/bin/time -v and a bisected least
passing RLIMIT_DATA per compiler):
  generated kernels (gen-qlower work/shipped-new + work/cand, 15 .cu, 3-53 KB) through cuda_host_emulator's two
  builds with g++ and clang++ and through build_cubin's nvcc -cubin: worst peak RSS 122 MiB (clang++), worst least
  RLIMIT_DATA 85 MiB (g++ -O2 -S, glu-iq4xs-q4k-c3), worst wall 0.74 s;
  engine sources region_timer / kernel_timer compile through build_cubin (llama.cpp ggml-cuda arange, concat, cpy,
  mmf, mmvf, scale, set-rows, unary, mmvq, whole files, with the engine's flags and includes; region_timer compiles
  only the launched kernels of a file; these are every file data/region.json's stock launches and kernel_timer name,
  fattn and mmq are compiled by no caller): worst peak RSS 451 MiB, least RLIMIT_DATA 617 MiB (mmvq.cu), worst wall 22.4 s.
  MEASURED_BYTES = 617 MiB, the largest of these; cap = MEASURED_BYTES x MARGIN (4: 2468 MiB), bounded by MemTotal x
  MEMORY_SHARE (1/8: 3.9 GiB on this 31 GiB box, so 2468 MiB here; 2 GiB on a 16 GiB machine, still 3x the worst; never below
  MEASURED_BYTES). hipcc was not measured (no AMD GPU on this box): the margin covers it until the AMD box measures.
  A runaway compile is stopped at 2.4 GiB, a twelfth of the 29 GB that took the box down. Time: the measured worst is 22.4 s
  (engine) and 0.74 s (generated, the emulator's own COMPILE_TIMEOUT_S); TIMEOUT_S keeps build_cubin's 1200 s."""
from __future__ import annotations

import math, os, signal, subprocess, sys, tempfile
from pathlib import Path

MEASURED_BYTES = 617 << 20  # the least RLIMIT_DATA llama.cpp's mmvq.cu compiles under (nvcc -cubin): the worst measured
MARGIN = 4           # an engine file or kernel four times today's worst still compiles
MEMORY_SHARE = 1 / 8  # of MemTotal: a runaway compiler leaves 7/8 of the machine to the session and its jobs
TIMEOUT_S = 1200.0   # the compile limit build_cubin and build_code_object already had; the emulator's is its own
_SHIM = ("import os,resource as r,sys\n"
         "for k,n in ((r.RLIMIT_DATA,int(sys.argv[1])),(r.RLIMIT_CPU,int(sys.argv[2]))):\n"
         " h=r.getrlimit(k)[1];n=n if h==r.RLIM_INFINITY else min(n,h);r.setrlimit(k,(n,h))\n"
         "try:os.execvp(sys.argv[3],sys.argv[3:])\n"
         "except OSError as e:sys.stderr.write(f'{sys.argv[3]}: {e.strerror}\\n');os._exit(127)")
_OOM = ("out of memory", "bad_alloc", "cannot allocate memory", "virtual memory exhausted", "memory exhausted",
        "memory allocation failure", "allocation failed")


class Exceeded(RuntimeError):
  """The compile hit its memory or time limit: that source is refused, the caller goes on."""


def mem_total() -> int | None:
  try:
    return int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal:"))) * 1024
  except (OSError, StopIteration, ValueError):
    return None


def cap_bytes(total:int | None = None) -> int:
  total = mem_total() if total is None else total
  return max(MEASURED_BYTES, int(min(MEASURED_BYTES * MARGIN, total * MEMORY_SHARE))) if total else MEASURED_BYTES * MARGIN


def run(argv:list[str], *, timeout_s:float = TIMEOUT_S, cap:int | None = None, cwd=None, env=None,
        input:str | None = None) -> subprocess.CompletedProcess:
  """argv run to the end with text output captured, its heap capped at cap (cap_bytes() by default). Returns the
  CompletedProcess as subprocess.run would, a nonzero status included; raises Exceeded when the time limit was hit,
  or when it failed on the memory limit (an out-of-memory message on its output)."""
  cap = cap_bytes() if cap is None else cap
  limited = sys.platform.startswith("linux")  # elsewhere the time limit alone (the docstring)
  full = [sys.executable, "-I", "-S", "-c", _SHIM, str(cap), str(math.ceil(timeout_s)), *map(str, argv)] if limited else [*map(str, argv)]
  with tempfile.TemporaryDirectory(prefix="boltbeam-compile-") as tmp:
    return _run(full, argv, timeout_s, cap, cwd, {**(os.environ if env is None else env), "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp}, input)


def _run(full, argv, timeout_s, cap, cwd, env, input) -> subprocess.CompletedProcess:
  try:  # no shim (not Linux): a missing compiler is said as the shim says it, status 127
    proc = subprocess.Popen(full, cwd=cwd, env=env, text=True, start_new_session=True, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except OSError as exc:
    return subprocess.CompletedProcess(list(argv), 127, "", f"{argv[0]}: {exc.strerror}\n")
  try:
    out, err = proc.communicate(input, timeout=timeout_s)
  except subprocess.TimeoutExpired:
    _kill(proc)
    raise Exceeded(f"{Path(str(argv[0])).name} ran past its {timeout_s:.0f} s limit; stopped") from None
  except BaseException:
    _kill(proc)
    raise
  if proc.returncode and any(w in (out + err).lower() for w in _OOM):
    raise Exceeded(f"{Path(str(argv[0])).name} hit its {cap >> 20} MiB memory limit; stopped")
  return subprocess.CompletedProcess(list(argv), proc.returncode, out, err)


def _kill(proc:subprocess.Popen) -> None:
  try:
    os.killpg(proc.pid, signal.SIGKILL)
  except (ProcessLookupError, PermissionError, AttributeError):
    proc.kill()
  try:
    proc.communicate(timeout=30)
  except subprocess.TimeoutExpired:  # Windows: an orphaned compiler stage holds the pipes; give up on its output
    pass
