"""Choose Ninja workers from measured compiler peaks and current resources."""
from __future__ import annotations

import math
import json
import hashlib
import functools
import os
from pathlib import Path
import shlex
import shutil
import subprocess


class PrebuildRequired(BaseException):
  """Abort the job, including libraries that catch Exception to try a fallback.

  FlashInfer catches ordinary profiling exceptions per tactic. A missing build
  is a job-policy failure, not an unsupported tactic: swallowing it retries
  every candidate and silently saves an empty tuning cache.
  """


@functools.lru_cache(maxsize=16)
def _compiler_digest(path: str, size: int, mtime_ns: int) -> str:
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_int(path: Path) -> int | None:
  try:
    value = path.read_text().strip()
    return None if value == "max" else int(value)
  except (OSError, ValueError):
    return None


def available_memory() -> int:
  """Host available RAM, capped by remaining cgroup v2 memory when present."""
  try:
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))
    available = int(line.split()[1]) * 1024
  except (OSError, StopIteration, ValueError, IndexError):
    return 0
  try:
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
  except (OSError, StopIteration):
    return available
  root = Path("/sys/fs/cgroup")
  directory = root / group.lstrip("/")
  while directory == root or root in directory.parents:
    limit = _read_int(directory / "memory.max")
    used = _read_int(directory / "memory.current")
    if limit is not None and used is not None:
      available = min(available, max(0, limit - used))
    if directory == root:
      break
    directory = directory.parent
  return available


def available_cpus() -> int:
  """CPU affinity capped by cgroup v2 CPU quota, rounded up for partial CPUs."""
  try:
    cpus = len(os.sched_getaffinity(0))
  except (AttributeError, OSError):
    cpus = os.cpu_count() or 1
  try:
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
  except (OSError, StopIteration):
    return max(1, cpus)
  root = Path("/sys/fs/cgroup")
  directory = root / group.lstrip("/")
  while directory == root or root in directory.parents:
    try:
      quota, period = (directory / "cpu.max").read_text().split()[:2]
      if quota != "max":
        cpus = min(cpus, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError, IndexError):
      pass
    if directory == root:
      break
    directory = directory.parent
  return max(1, cpus)


def cuda_build_records(build_file: str | Path, ninja: str = "ninja") -> list[dict]:
  """CUDA compilation entries from Ninja, independent of generated rule names."""
  build_file = Path(build_file).resolve()
  result = subprocess.run([ninja, "-C", str(build_file.parent), "-f", str(build_file), "-t", "compdb"],
                          capture_output=True, text=True, check=True)
  records = json.loads(result.stdout)
  def is_cuda(record: dict) -> bool:
    words = shlex.split(str(record.get("command", "")))
    compiling = any(w in ("-c", "--compile") for w in words)
    cuda_source = Path(str(record.get("file", ""))).suffix.lower() == ".cu"
    cuda_compiler = any(Path(w).name.lower() in ("nvcc", "nvcc.exe") for w in words)
    cuda_language = any(w in ("-xcuda", "-xcu") or
                        (w == "-x" and i + 1 < len(words) and words[i + 1] in ("cuda", "cu"))
                        for i, w in enumerate(words))
    return compiling and (cuda_source or cuda_compiler or cuda_language)
  return [r for r in records if is_cuda(r)]


def build_identity(build_file: str | Path, ninja: str = "ninja", *, include_deps: bool = True) -> str:
  """Fingerprint build plan, CUDA sources, compiler binaries and observed header dependencies."""
  build_file = Path(build_file).resolve()
  records = cuda_build_records(build_file, ninja)
  if not records:
    raise ValueError(f"{build_file} has no CUDA compilation entries")
  compiler_hashes: dict[str, str] = {}
  entries = []
  for record in records:
    directory = Path(record.get("directory") or build_file.parent)
    source = Path(record["file"])
    if not source.is_absolute():
      source = directory / source
    words = shlex.split(record["command"])
    compilers = [word for word in words if Path(word).name.lower() in ("nvcc", "nvcc.exe", "clang++", "clang")]
    compiler = compilers[-1] if compilers else words[0]
    executable = Path(compiler) if os.path.sep in compiler else Path(shutil.which(compiler) or compiler)
    executable = executable.resolve()
    if str(executable) not in compiler_hashes:
      stat = executable.stat()
      compiler_hashes[str(executable)] = _compiler_digest(str(executable), stat.st_size, stat.st_mtime_ns)
    entries.append((str(source.resolve()), hashlib.sha256(source.read_bytes()).hexdigest(),
                    record["command"], str(executable), compiler_hashes[str(executable)]))
  observed = {}
  if include_deps:
    deps = subprocess.run([ninja, "-C", str(build_file.parent), "-f", str(build_file), "-t", "deps"],
                          capture_output=True, text=True, check=True)
    for line in deps.stdout.splitlines():
      if not line.startswith("    "):
        continue
      path = Path(line.strip())
      if not path.is_absolute():
        path = build_file.parent / path
      try:
        stat = path.stat()
        observed[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
      except OSError:
        observed[str(path.resolve())] = None
  payload = (hashlib.sha256(build_file.read_bytes()).hexdigest(), sorted(entries), sorted(observed.items()))
  return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def derived_jobs(*, build_file: str | Path | None = None, ninja: str = "ninja",
                 memory_bytes: int | None = None, cpus: int | None = None,
                 env: dict[str, str] | None = None) -> int | None:
  """Return a safe worker count, or None when MAX_JOBS was explicitly supplied.

  A serial calibration records the largest measured compiler peak. Reserve one
  such peak beyond the running workers. Until calibrated, compile serially.
  """
  env = os.environ if env is None else env
  if "MAX_JOBS" in env:
    return None
  profile = env.get("BOLTBEAM_BUILD_PROFILE")
  try:
    data = json.loads(Path(profile).read_text()) if profile else {}
    matching = (build_file is None or
                (Path(data["ninja_file"]).resolve() == Path(build_file).resolve() and
                 data.get("build_identity") == build_identity(build_file, ninja)))
    peak = int(data["compiler_peak_bytes"]) if matching else 0
  except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
    peak = 0
  memory = available_memory() if memory_bytes is None else memory_bytes
  cpu_count = available_cpus() if cpus is None else cpus
  if peak > 0 and memory < peak:
    raise RuntimeError(f"Insufficient available memory for one measured compiler: {memory} < {peak} bytes")
  jobs_by_memory = max(1, memory // peak - 1) if peak > 0 else 1
  return max(1, min(max(1, cpu_count), jobs_by_memory))


def install_ninja_limit() -> None:
  """Bound Ninja when Python launches it, including in spawned engine workers.

  This operates at process launch, after a model may have consumed host RAM.
  It also covers aliases of FlashInfer's run_ninja imported before installation.
  """
  if getattr(subprocess.run, "_boltbeam_ninja_limit", False):
    return
  original = subprocess.run

  def bounded_run(command, *args, **kwargs):
    if (isinstance(command, (list, tuple)) and command and
        not any(str(item) in ("-t", "--help", "-h", "--version") for item in command[1:]) and
        Path(os.fspath(command[0])).name in ("ninja", "ninja.exe")):
      cmd = list(command)
      directory = Path(kwargs.get("cwd") or os.getcwd())
      ninja_file = "build.ninja"
      for index, item in enumerate(cmd):
        if item == "-C" and index + 1 < len(cmd):
          directory = directory / cmd[index + 1]
        if item == "-f" and index + 1 < len(cmd):
          ninja_file = cmd[index + 1]
      path = Path(ninja_file)
      if not path.is_absolute():
        path = directory / path
      child_env = kwargs.get("env") or os.environ
      if child_env.get("BOLTBEAM_REQUIRE_PREBUILT") == "1":
        # Dry-run the exact targets, even when callers supplied their own -j.
        # No compiler may start inside a benchmark, including unknown modules.
        check_kwargs = {k: v for k, v in kwargs.items() if k in ("cwd", "env")}
        dry = original([*cmd, "-n"], capture_output=True, text=True, **check_kwargs)
        lines = [line.strip() for line in dry.stdout.splitlines()
                 if line.strip() and not line.startswith("ninja: Entering directory ")]
        if dry.returncode or lines != ["ninja: no work to do."]:
          raise PrebuildRequired(f"Prebuild required: {path}. Run tools/prebuild_cuda.py "
                             "in isolation before starting the model.")
        # A cached build needs no compiler reservation. Keep the invocation dry
        # so late cache invalidation cannot launch a compiler inside a benchmark.
        return original([*cmd, "-n"], *args, **kwargs)
      explicit_jobs = any(str(item) in ("-j", "--jobs") or str(item).startswith(("-j", "--jobs="))
                          for item in command[1:])
      if path.is_file() and not explicit_jobs:
        jobs = derived_jobs(build_file=path, ninja=os.fspath(cmd[0]), env=child_env)
        if jobs is None:
          jobs = child_env["MAX_JOBS"]
        command = [*cmd, "-j", str(jobs)]
    return original(command, *args, **kwargs)

  bounded_run._boltbeam_ninja_limit = True
  subprocess.run = bounded_run
