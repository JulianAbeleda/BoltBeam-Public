"""Prefill, measured faithfully: the untraced time to read a prompt, what the engine batched, and where that time goes.

    pipeline MODEL --run RUN --target T --provider P --workload prefill --ctxs 2048 --analyze --measure auto --no-search

Step 4 (`measure`) answers trace_request.json with one whole_step row per prompt length: the prefill's wall time as
the engine runs it, untraced, at the engine's own chunk (llama.cpp's ubatch, the fork's prefill chunk), with the SM
clock, power and temperature logged through the run (gpu_telemetry) and a throttle flagged. What the engine batched
is recorded from the engine itself: llama-bench's own n_ubatch and n_batch, the fork's forward calls and their token
widths. An engine whose chunk is one token is "not batched prefill", said on the row.

Step 5 (`role_time`) captures one prefill per length in the model (nsys for an engine CUDA sees; the fork's own
GPU timestamps for the fork, which CUDA's tools cannot see), reads every kernel's matrix path from its code
(instruction_path), and pairs kernels with roles by launch counts and operations (prefill_attribution). The result
is prefill_trace.json; the where table is built from it (workflow/prefill.py).

ENGINES holds what differs per engine: how to time one prefill, how to capture one, where its kernels' code is.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable

from boltbeam.collectors import gpu_telemetry
from boltbeam.roofline.prefill_work import peaks_at
from boltbeam.workflow.common import run_gpu

TRACE = "prefill_trace.json"
RAW = "kernel_compare/prefill_capture"
NATIVE_RAW = "kernel_compare/prefill_native"  # the untraced llama-bench commands and stdout, by prompt name
REPS = 3
NOT_BATCHED = "not batched prefill: the engine reads the prompt one token at a time"
DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_prefill_profile.py"


def chunk_record(widths:list[Any], how:str, outputs:int | None = None) -> dict[str, Any]:
  """What the engine batched: the token width of each chunk its matrices run over, whether that is batching, and
  how many calls produce an output (the work for a call's last token, the output head, runs once per call: the fork
  calls once per chunk, llama.cpp once per n_batch tokens and splits that into ubatch chunks)."""
  ints = [w for w in widths if isinstance(w, int)]
  batched = bool(ints) and max(ints) > 1
  return {"widths": widths, "calls": len(widths), "outputs": outputs if outputs is not None else len(widths),
          "size": max(ints) if ints else None, "batched": batched,
          "words": (f"{len(widths)} forward call(s) of up to {max(ints)} tokens ({how})" if batched else
                    f"{NOT_BATCHED} ({how})") if ints else f"widths not readable ({how})"}


def split_chunks(length:int, size:int) -> list[int]:
  return [min(size, length - i) for i in range(0, length, size)]


def llama_chunks(length:int, batch:int, ubatch:int) -> list[int]:
  """llama.cpp's chunks of a prompt: llama-bench decodes it in calls of n_batch tokens, and each call splits into
  n_ubatch chunks of its own (2,198 tokens at 2,048 and 384: 384 x 5, 128, then 150)."""
  if batch <= 0 or ubatch <= 0:
    return split_chunks(length, ubatch) if ubatch > 0 else []
  return [c for call in split_chunks(length, batch) for c in split_chunks(call, ubatch)]


def call_ends(length:int, batch:int, ubatch:int) -> list[int]:
  """1 for the chunk (llama_chunks) that ends each n_batch call, where llama-bench reads its output token; else 0."""
  return [1 if i + 1 == len(chunks) else 0 for call in split_chunks(length, batch)
          for chunks in [split_chunks(call, ubatch)] for i in range(len(chunks))]


# --- llama.cpp -------------------------------------------------------------------------------------------------

def timing_window(length:int, depth:int = 0) -> dict:
  """The host interval compared across native and profiled llama-bench runs."""
  return {"method": "last_repetition", "repetitions": REPS, "warmup": True, "prompt_tokens": length, "depth": depth}


def qualification_sample(row:dict, length:int, depth:int = 0) -> float | None:
  samples = row.get("samples_ns")
  if not isinstance(samples, list) or len(samples) != REPS or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
    return None
  if row.get("n_prompt") != length or (row.get("n_depth") or 0) != depth or (row.get("n_gen") or 0) != 0:
    return None
  return samples[-1] / 1e6

def _bench(run:pathlib.Path) -> str:
  from boltbeam.collectors.llama_bench_decode import ENV, CannotMeasure, find
  bench = find()
  if bench is None:
    raise CannotMeasure(f"llama-bench was not found in ${ENV} or on PATH", f"export {ENV}=/path/to/llama-bench")
  return bench


def llama_argv(bench:str, model:str, length:int, reps:int, *, warmup:bool = True, depth:int = 0, gen:int = 0,
               extra:list[str] | None = None) -> list[str]:
  """One prompt of `length` tokens, no generation, the engine's own ubatch and batch (its defaults, recorded back).
  depth: the prompt goes after that many tokens already in the cache (llama-bench -d, which fills them untimed);
  gen: also that many generated tokens at the depth, its own row; extra: flags a caller matches to another
  configuration (a replay's cache types, chunk and flash attention, read from the user's server)."""
  return [bench, "-m", str(pathlib.Path(model).absolute()), "-p", str(length), "-n", str(gen), "-ngl", "99", "-r", str(reps),
          *(["-d", str(depth)] if depth else []), "-o", "json", *([] if warmup else ["--no-warmup"]), *(extra or [])]


def llama_row(text:str, field:str = "n_prompt") -> dict[str, Any]:
  """llama-bench's JSON row for the prompt (or, with field n_gen, for the generation), from its output (a capture's
  log has the tool's own lines after it)."""
  start = text.find("[")
  while start >= 0:
    try:
      rows = json.JSONDecoder().raw_decode(text[start:])[0]
      row = next((r for r in rows if isinstance(r, dict) and int(r.get(field, 0)) > 0), None)
      if row:
        return row
    except (ValueError, TypeError):
      pass
    start = text.find("[", start + 1)
  raise RuntimeError(f"llama-bench printed no row with {field}")


def retain_native(run:pathlib.Path, name:str, cmd:list[str], stdout:str) -> None:
  """Keep the untraced llama-bench command and its own output, so the samples can be recomputed from them."""
  folder = run / NATIVE_RAW
  folder.mkdir(parents=True, exist_ok=True)
  (folder / f"{name}.command.json").write_text(json.dumps(cmd))
  (folder / f"{name}.stdout").write_text(stdout)


def llama_truth(run:pathlib.Path, model:str, length:int, *, device:int = 0) -> dict[str, Any]:
  bench = _bench(run)
  from boltbeam.collectors.capture_settings import llama_flags
  from boltbeam.workflow.layout import one_gpu
  pin, env = one_gpu(run, "llama.cpp")
  cmd = llama_argv(bench, model, length, REPS, extra=[*pin, *llama_flags(run, length)])
  with gpu_telemetry.run_telemetry(run, device) as tele:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=run, env={**os.environ, **env})
  retain_native(run, str(length), cmd, proc.stdout)
  if proc.returncode != 0:
    raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  row = llama_row(proc.stdout)
  samples = [float(x) / 1e6 for x in row.get("samples_ns") or [row["avg_ns"]]]
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  qualified = qualification_sample(row, length)
  chunk = chunk_record(llama_chunks(length, b, ub) if ub else [], f"llama-bench n_ubatch {ub}, n_batch {b}",
                       outputs=-(-length // b) if b else None)
  return {"ms": statistics.median(samples), "qualification_ms": qualified, "samples_ms": samples, "chunk": chunk, "telemetry": tele.summary(),
          "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": f"llama-bench -p {length} -n 0, {len(samples)} repetitions after its warm-up, untraced",
          "engine": {"timing_window": timing_window(length) if qualified is not None else None, "build": row.get("build_commit"), "build_number": row.get("build_number"), "flash_attn": row.get("flash_attn"), "n_ubatch": ub, "n_batch": b,
                     "type_k": row.get("type_k"), "type_v": row.get("type_v"), "command": cmd,
                     "n_ctx": int(row.get("n_prompt") or length) + int(row.get("n_gen") or 0),
                     "n_ctx_source": "llama-bench sizes its context to n_prompt + n_gen + n_depth",
                     "prompt": PROMPT_WORDS["llama.cpp"]}}


LLAMA_TILE_ROWS = 128  # mmq_y: the weight rows of one mul_mat_q output tile on a CUDA matrix-instruction build
LLAMA_K_ITER = 256  # MMQ_ITER_K: the K elements one stream-k step covers


def llama_stream_k(tmpl:str) -> dict[str, int] | None:
  """llama.cpp's stream-k layout for a mul_mat_q fixup template: its second template argument is the tile's token
  columns (mmq_x); None for any other fixup (flash attention's has its own layout, not modelled)."""
  import re
  m = re.match(r"mul_mat_q_stream_k_fixup<[^,]+,(\d+),", tmpl.replace(" ", ""))
  return {"tile_m": int(m.group(1)), "tile_n": LLAMA_TILE_ROWS, "k_iter": LLAMA_K_ITER} if m else None


def llama_formats(type_k:str | None, weights:set[str] | None = None) -> dict[str, Any]:
  """What llama.cpp's prefill kernels read and write: for block-quantized weights mul_mat_q reads activations
  quantized to q8_1 and writes fp32; for float weights (all of one float type) cuBLAS reads activations converted to
  the weight's type and writes fp32 (ggml_cuda_mul_mat_cublas, f32 output on NVIDIA). Flash attention reads fp32 Q
  and the cache in its type_k and writes fp32, stream-k partials are fp32. Without flash attention (a head size its
  kernels lack) attention is KQ, soft_max and KQV (unfused_attention: soft_max names it)."""
  from boltbeam.collectors.llama_bench_decode import UNFUSED_ATTENTION
  from boltbeam.quantization.quant import load_quant_registry
  reg = load_quant_registry()
  floats = {w for w in weights or () if w in reg and reg[w].dequant_family == "float_passthrough"}
  act = next(iter(floats)) if len(floats) == 1 and floats == set(weights or ()) else None
  return {"activations": {"in": act or "Q8_1", "out": "F32"}, "attention_activations": {"in": "F32", "out": "F32"},
          "kv": str(type_k or "f16").upper(), "partials": "F32", "stream_k": llama_stream_k, "unfused_attention": UNFUSED_ATTENTION,
          "words": (f"cuBLAS: {act.lower()} in, fp32 out" if act else "mul_mat_q: q8_1 in, fp32 out")
                   + "; flash attention: fp32 Q and output, the cache in type_k; fp32 partials"}


PROMPT_WORDS = {"llama.cpp": "llama-bench's own prompt: token ids drawn by std::rand() % n_vocab (never seeded: one fixed "
                            "sequence; each repetition draws new ids), the KV cache cleared before every repetition, so "
                            "nothing is cached between prompts"}
COPIES = REPS + 1  # warm-up followed by the same repetition count as the native run
MAX_CAPTURE_LAUNCHES = 1_000_000  # a capture estimated to hold more launches than this is not taken: its time stands alone


def llama_capture(run:pathlib.Path, model:str, length:int, target, depth:int = 0, extra:list[str] | None = None,
                  raw:pathlib.Path | None = None) -> dict[str, Any]:
  """The prompt captured with the native run's repetition count after warm-up; attribute the final repetition.
  Compare its host timer to the native final repetition, separately from the headline median. Cache restoration
  can rebuild the graph on repetition two, so choosing another repetition changes the lifecycle. extra: llama_argv's;
  raw: the capture's folder, when a run holds more than one capture of one length at one depth."""
  from boltbeam.collectors import vendor_capture
  from boltbeam.collectors.prefill_attribution import REPEAT, UnnamedCapture, measured_window
  bench = _bench(run)
  raw = raw or run / RAW / (f"{length}-at-{depth}" if depth else str(length))
  from boltbeam.collectors.capture_settings import llama_flags
  settings = llama_flags(run, length, capture=True) if not depth else []
  from boltbeam.workflow.layout import one_gpu
  pin, env = one_gpu(run, "llama.cpp")
  argv = llama_argv(bench, model, length, REPS, depth=depth, extra=[*pin, *settings, *(extra or [])])
  launches = vendor_capture.capture(target.backend, argv, raw, env=env, device=run_gpu(run))
  row = llama_row(vendor_capture.program_output(raw))
  qualified = qualification_sample(row, length, depth)
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  try:
    measured_window(launches, False)  # refuses a capture that names no kernel, before anything reads its roles
  except UnnamedCapture as exc:  # the whole prefill still stands: the engine's timer under the capture, GPU busy beside it
    calls = COPIES + (1 if depth else 0)  # the prompts the capture holds (and the one cached depth fill): no buffer marks their edges
    raise UnnamedCapture(str(exc), {"method": vendor_capture.plan(target.backend)["method"], "token_ms": qualified, "prefills": calls,
                                    "busy_ms": sum(l["wall_us"] for l in launches) / 1e3 / calls,
                                    "timing_window": timing_window(length, depth)} if qualified else None) from exc
  formats = llama_formats(row.get("type_k"), weight_quants(run))
  if target.backend == "CUDA" and target.compute_units:
    from boltbeam.collectors import engine_kernels, llama_mmq
    try:
      source = engine_kernels.mmvq_source_dir()
      commit = str(row.get("build_commit") or "")
      if commit:
        formats["matrix_grid_constraint"] = llama_mmq.constraints(pathlib.Path(source["dir"]), target.compute_units, commit)
    except engine_kernels.NoEngineSource:
      pass
  return {"launches": launches, "chunk": llama_chunks(length, b, ub), "outputs": -(-length // b) if b else 1,
          "warmup": REPEAT, "copies": REPS if depth else COPIES, "formats": formats, "timing_window": timing_window(length, depth) if qualified is not None else None,
          "method": vendor_capture.plan(target.backend)["method"],
          "paths": library_paths(linked_libraries(pathlib.Path(bench).resolve()), launches, device=run_gpu(run)),
          "program": {"argv": argv, "env": env, "raw": raw, "profiler_range": False, "arch": (run_gpu(run) or {}).get("architecture")},
          "captured_ms": float(row["avg_ns"]) / 1e6, "window": vendor_capture.busy_window(raw, launches),
          "measured_ms": qualified,  # never substitute an average for the final repetition
          "prune": llama_prunes(launches, llama_chunks(length, b, ub), -(-length // b) if b else 1, copies=REPS if depth else COPIES,
                               ends=call_ends(length, b, ub) if b else None)}


LLAMA_PRUNED_STAGES = ("ffn_in", "ffn_out")  # llama.cpp keeps only the output rows (inp_out_ids) before the last FFN


def llama_prunes(launches:list[dict[str, Any]], chunk:list[int], outputs:int, copies:int = 2,
                 ends:list[int] | None = None) -> dict[str, Any] | None:
  """Whether this llama.cpp build ran its last layer's FFN on the output tokens only, from the capture: a prefill whose
  GEMV launches (one per output call is the head) outnumber its output calls ran the pruned layer as GEMVs. llama-bench
  asks for one output token per call, at the end of the call's last chunk. copies: the prefills the capture holds.
  ends: 1 for each chunk that ends a call (call_ends: each n_batch call splits into chunks of its own, so 2,198 tokens
  at 2,048 per call and 512 per chunk end calls after chunks 4 and 5, not 3 and 5); without it the calls share the
  chunks evenly. None when the build does not prune."""
  from boltbeam.collectors.instruction_path import template
  from boltbeam.workflow.tie_out import GEMV_WORDS
  found = [template(l["key"]) for l in launches if any(w in str(l["key"]) for w in GEMV_WORDS)]
  gemv = len(found) // copies
  if gemv <= outputs:
    return None
  if ends is not None:
    out = list(ends)
    if len(out) != len(chunk) or sum(out) != outputs:
      raise ValueError(f"call ends {out} do not mark {outputs} calls over the {len(chunk)} chunks {chunk}")
  else:
    per_call = -(-len(chunk) // max(outputs, 1))
    out = [1 if (u + 1) % per_call == 0 or u + 1 == len(chunk) else 0 for u in range(len(chunk))]
  return {"stages": list(LLAMA_PRUNED_STAGES), "outputs": out, "kernels": sorted(set(found)),
          "words": (f"{gemv} GEMV launches per prefill against {outputs} output call(s): the last layer's FFN runs on "
                    "the output tokens only (llama.cpp's inp_out_ids), in the chunk that ends each call")}


def weight_quants(run:pathlib.Path) -> set[str]:
  p = run / "model_profile.json"
  return {str(r.get("quant")).upper() for r in (json.loads(p.read_text()).get("roles") or [])} if p.exists() else set()


def library_paths(libraries:list[pathlib.Path], launches:list[dict[str, Any]] | None = None, *,
                  tools:pathlib.Path | None = None, jit_caches:list[str] | None = None,
                  capture:pathlib.Path | None = None, device:dict[str, Any] | None = None) -> dict[str, Any]:
  """The matrix paths of the kernels in the libraries the engine loads (a library without device code is skipped):
  their embedded PTX, then, for captured kernels no PTX holds (a library that ships machine code only), their machine
  code for this GPU (instruction_path.sass_table). tools: a folder with the engine's own cuobjdump/nvdisasm.
  jit_caches: folders where the engine's compiler (Triton) wrote each kernel it built as <kernel>.ptx.
  device: the run's GPU (workflow.common.run_gpu), whose vendor picks the reader (INSTRUCTIONS); none named (a run
  with no hardware profile) reads NVIDIA's, as before the profile named the GPU. A vendor with no reader refuses."""
  from boltbeam.collectors import rocm_isa
  if capture is not None and (dbs := sorted(capture.glob("*_results.db"))):  # AMD: the code objects rocprofv3 saw load
    return rocm_isa.table(dbs, jit_caches=jit_caches, libraries=libraries)
  vendor = (device or {}).get("vendor") or "nvidia"
  if vendor not in INSTRUCTIONS:
    from boltbeam.workflow.chips import NoVendorRow
    raise NoVendorRow(f"no instruction reader for {vendor} GPUs: their kernels' matrix paths cannot be read")
  return INSTRUCTIONS[vendor](libraries, launches, tools=tools, jit_caches=jit_caches, device=device or {})


def _gfx_paths(libraries:list[pathlib.Path], launches, *, device:dict[str, Any], **_) -> dict[str, Any]:
  """An AMD GPU with no capture: the libraries' fat binaries for its gfx target."""
  from boltbeam.collectors import rocm_isa
  return rocm_isa.library_table(libraries, device.get("architecture"))


def _ptx_paths(libraries:list[pathlib.Path], launches, *, tools, jit_caches, device:dict[str, Any]) -> dict[str, Any]:
  """An NVIDIA GPU: embedded PTX, the JIT caches' PTX, then machine code for its sm_XY target."""
  from boltbeam.collectors.instruction_path import library_table, paths_in_text, sass_signatures, sass_table, template
  kernels: dict[str, dict[str, int]] = {}
  read, seen = [], set()
  from boltbeam.runtime.cuda_device import find_cuda_tool
  find = lambda name: find_cuda_tool(name,tools)  # noqa: E731
  for lib in libraries:
    lib = pathlib.Path(lib).resolve()
    if lib in seen or not lib.is_file():
      continue
    seen.add(lib)
    try:
      table = library_table(lib,cuobjdump=find('cuobjdump'))
    except (RuntimeError, OSError):
      continue
    if table["kernels"]:
      kernels.update(table["kernels"])
      read.append({"library": str(lib), "sha256": table["sha256"], "kernels": len(table["kernels"]), "code": "ptx"})
  wanted = {template(l["key"]) for l in launches or []} - set(kernels)
  for d in jit_caches or []:
    found: dict[str, dict[str, int]] = {}
    for f in sorted(pathlib.Path(d).rglob("*.ptx")) if pathlib.Path(d).is_dir() else []:
      if template(f.stem) in wanted:  # one kernel built more than once: its variants' instructions added up
        got = found.setdefault(template(f.stem), {})
        for path, n in paths_in_text(f.read_text(errors="replace")).items():
          got[path] = got.get(path, 0) + n
    if found:
      kernels.update(found)
      wanted -= set(found)
      read.append({"library": d, "sha256": None, "kernels": len(found), "code": "ptx (the engine's JIT cache)"})
  arch = device.get("architecture") or gpu_arch()
  if wanted and arch and find("nvdisasm"):
    sigs = sass_signatures(arch, nvdisasm=find("nvdisasm"))
    for lib in sorted(seen):
      if not wanted:
        break
      try:
        table = sass_table(lib, wanted, arch, sigs, cuobjdump=find("cuobjdump"), nvdisasm=find("nvdisasm"))
      except (RuntimeError, OSError):
        continue
      if table["kernels"]:
        kernels.update(table["kernels"])
        wanted -= set(table["kernels"])
        read.append({"library": str(lib), "sha256": table["sha256"], "kernels": len(table["kernels"]), "code": table["how"]})
  return {"kernels": kernels, "libraries": read,
          "how": "the engine's loaded libraries: embedded PTX (cuobjdump --dump-ptx), else machine code (nvdisasm) "
                 "with each matrix opcode mapped to the path whose probe compiles to it"}


INSTRUCTIONS = {"nvidia": _ptx_paths, "amd": _gfx_paths}  # a GPU vendor's kernel-code reader, by the run's GPU's vendor


def gpu_arch(device:int = 0) -> str | None:
  """This GPU's machine-code architecture, sm_XY, from nvidia-smi's compute capability; None without it."""
  try:
    proc = subprocess.run(["nvidia-smi", "-i", str(device), "--query-gpu=compute_cap", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=30)
  except (OSError, subprocess.SubprocessError):
    return None
  cap = proc.stdout.strip()
  return f"sm_{cap.replace('.', '')}" if proc.returncode == 0 and cap.replace(".", "").isdigit() else None


def linked_libraries(binary:pathlib.Path) -> list[pathlib.Path]:
  """The shared libraries beside a binary, then those it links (ldd): where an engine binary's kernels live."""
  out = [p for p in sorted(binary.parent.glob("*.so*")) if not p.is_symlink()]
  try:
    proc = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, timeout=60)
    out += [pathlib.Path(l.split("=>", 1)[1].split("(")[0].strip()) for l in proc.stdout.splitlines()
            if "=>" in l and "/" in l.split("=>", 1)[1]]
  except (OSError, subprocess.SubprocessError):
    pass
  return out


# --- the tinygrad fork -------------------------------------------------------------------------------------------

def _fork(root:pathlib.Path | None):
  from boltbeam.search import role_compare
  root = pathlib.Path(root) if root else role_compare.default_fork_root()
  return root, role_compare.fork_python(root)


EAGER_SAMPLES = 1  # an eager prefill schedules its graph again: minutes at long prompts, so one is timed after the warm one


def _fork_run(root, python, target, model, length, mode, out:pathlib.Path | None = None) -> dict[str, Any]:
  """The driver, replaying the captured graph; when the captured graphs do not fit in GPU memory, again in the
  fork's default eager mode, and the result says so (replay False, why)."""
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed
  try:
    return _fork_once(root, python, target, model, length, mode, out, eager=False)
  except DecodeFailed as exc:
    if "MemoryError" not in str(exc) and "Allocation" not in str(exc):
      raise
    got = _fork_once(root, python, target, model, length, mode, out, eager=True)
    got["replay_failed"] = str(exc)[:300]
    return got


def _fork_once(root, python, target, model, length, mode, out, *, eager:bool) -> dict[str, Any]:
  argv = ["--model", model, "--length", str(length), "--mode", mode]
  if out is not None:
    argv += ["--out", str(out)]
  if eager:
    argv += ["--eager", "--samples", str(EAGER_SAMPLES)]
  return fork_exec(root, python, target, DRIVER, argv, mode, "prefill")


def fork_exec(root, python, target, driver:pathlib.Path, argv:list[str], mode:str, what:str) -> dict[str, Any]:
  """A fork driver run by the fork's python in the fork's folder: untraced (truth) or with every launch timed
  (capture); its last JSON line."""
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed, device_for
  env = {**os.environ, "PYTHONPATH": ".", "DEV": device_for(target), "VIZ": "0",
         **({"JIT": "1", "PROFILE": "0"} if mode == "truth" else {"JIT": "2", "PROFILE": "1"})}
  proc = subprocess.run([str(python), str(driver), *argv], cwd=root, capture_output=True, text=True, timeout=7200, env=env)
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise DecodeFailed(f"tinygrad {what} ({mode}) failed", proc.stderr or proc.stdout, root)
  return json.loads(lines[-1])


def tinygrad_truth(run:pathlib.Path, model:str, length:int, *, target, root=None, device:int = 0) -> dict[str, Any]:
  root, python = _fork(root)
  with gpu_telemetry.run_telemetry(run, device) as tele:
    got = _fork_run(root, python, target, model, length, "truth")
  return {"ms": got["wall_s"] * 1e3, "samples_ms": [s * 1e3 for s in got["samples_s"]],
          "chunk": chunk_record(got["chunks"], "the fork's forward calls during the prompt"), "telemetry": tele.summary(),
          "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": (f"the fork's generate() from its first forward call to the first token, {len(got['samples_s'])} "
                     + ("prefill(s) replayed from the captured graph, untraced" if got.get("replay", True) else
                        "eager prefill(s), the fork's default (its graph scheduled again: the replay ran out of GPU memory), "
                        "untraced")),
          "replay": got.get("replay", True), "replay_failed": got.get("replay_failed"),
          "engine": {"eager_first_prompt_ms": got["warm_s"] * 1e3,
                     "capture_ms": got["capture_s"] * 1e3 if got.get("capture_s") is not None else None,
                     "workload_reuse_default": got.get("workload_reuse_default"),
                     "eager_words": ("the fork's default for an independent prompt is eager: its graph is scheduled "
                                     "again on every prompt (workload reuse not admitted); eager_first_prompt_ms is that, "
                                     "with compiles")}}


def tinygrad_capture(run:pathlib.Path, model:str, length:int, target, *, root=None) -> dict[str, Any]:
  root, python = _fork(root)
  raw = run / RAW / str(length)
  raw.mkdir(parents=True, exist_ok=True)
  got = _fork_run(root, python, target, model, length, "capture", raw / "launches.json")
  return fork_capture(raw, got)


def fork_capture(raw:pathlib.Path, got:dict[str, Any]) -> dict[str, Any]:
  """A fork driver's capture (launches.json in raw, its JSON line got) as the attribution reads it."""
  from boltbeam.collectors.instruction_path import paths_in_text, template
  (raw / "driver.json").write_text(json.dumps(got))
  data = json.loads((raw / "launches.json").read_text())
  # a source with no kernel body in it (a precompiled binary's stub) does not say the program's path: left out, so
  # its ceiling is refused, named, rather than read as "no matrix instruction"
  kernels = {template(name): paths_in_text(src) for name, src in (data.get("sources") or {}).items() if is_kernel_code(src)}
  from boltbeam.artifacts.tinygrad_profile_events import _census_program_info
  meta = {template(name): info for (name, binary), info in _census_program_info(data.get("census")).items()
          if binary is None and info.get("role") not in (None, "mixed") and info.get("shape")}
  widths = [w for w in data["chunks"] if isinstance(w, int)]
  return {"launches": data["launches"], "chunk": widths, "outputs": len(widths), "warmup": False,
          "method": "tinygrad-profile-events", "captured_ms": got["wall_s"] * 1e3, "program_roles": meta, "own_timing": True,
          "paths": {"kernels": kernels, "libraries": [], "how": "each program's own source, as the fork's JIT captured it"}}


def is_kernel_code(src:str) -> bool:
  """Whether a program's source text holds its kernel: a CUDA C kernel or a PTX entry."""
  return "__global__" in src or ".entry" in src


# --- vLLM ----------------------------------------------------------------------------------------------------------

VLLM_WARM = 2  # warm calls before the timed or captured one (the first call after load runs slower)


def _vllm(model:str, length:int, *extra:str) -> tuple[list[str], dict[str, str]]:
  from boltbeam.collectors import engine_common as ec, vllm_decode as vd
  py = vd.python()
  if py is None:
    raise RuntimeError(f"vLLM was not found: set ${vd.ENV} to the python of the venv vLLM is installed in")
  return ([py, str(vd.DRIVER), "--model", str(pathlib.Path(model).absolute()), "--prefill", str(length),
           "--warm", str(VLLM_WARM), *extra], ec.venv_env(py))


def vllm_chunk(length:int, line:dict[str, Any]) -> dict[str, Any]:
  """vLLM's chunk: the prompt split at max_num_batched_tokens when chunked prefill is on (else one chunk); every
  step computes the logits of the request's last scheduled token, so each chunk is one output."""
  size = int(line["max_num_batched_tokens"]) if line.get("chunked_prefill") else length
  return chunk_record(split_chunks(length, size), f"vLLM max_num_batched_tokens {line['max_num_batched_tokens']}, "
                      f"chunked prefill {'on' if line.get('chunked_prefill') else 'off'}")


def vllm_truth(run:pathlib.Path, model:str, length:int, *, device:int = 0) -> dict[str, Any]:
  from boltbeam.collectors import engine_common as ec, vllm_decode as vd
  from boltbeam.collectors.capture_settings import vllm_flags
  argv, env = _vllm(model, length, "--reps", str(REPS), *vllm_flags(run, length))
  with gpu_telemetry.run_telemetry(run, device) as tele:
    line = ec.run_driver(argv, env={**env, **vd.RUN_ENV})[-1]
  engine = [s for s in line["engine_samples_s"] if s]
  own = len(engine) == len(line["engine_samples_s"])
  samples = [s * 1e3 for s in (engine if own else line["wall_samples_s"])]
  return {"ms": statistics.median(samples), "samples_ms": samples, "chunk": vllm_chunk(length, line),
          "telemetry": tele.summary(), "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": (f"vLLM's own request timing (first token less scheduled), {len(samples)} prompts after {VLLM_WARM} "
                     "warm-up calls, untraced" if own else
                     f"the wall time of LLM.generate (vLLM reported no request timing), {len(samples)} prompts after "
                     f"{VLLM_WARM} warm-up calls, untraced"),
          "engine": {**{k: line.get(k) for k in ("activation_dtype", "kv_cache_dtype", "model_type", "max_model_len")}, "vllm": line.get("vllm"), "speculative_config": line.get("speculative_config"), "max_num_batched_tokens": line["max_num_batched_tokens"],
                     "chunked_prefill": line.get("chunked_prefill"), "wall_samples_ms": [s * 1e3 for s in line["wall_samples_s"]],
                     "command": argv, "prompt": PROMPT_WORDS["vllm"]}}


def vllm_formats(weights:set[str], resolved:dict[str, Any] | None = None, *, require_resolved:bool = False) -> dict[str, Any]:
  """vLLM on a float checkpoint: activations, attention and the cache in the weights' type (the KV cache dtype
  auto), GEMMs writing that type; no stream-k fixups."""
  fmt = next(iter(weights)) if len(weights) == 1 and weights <= {"F16", "BF16"} else "BF16"
  resolved = resolved or {}
  names = {"bfloat16": "BF16", "float16": "F16", "fp8_e4m3": "FP8_E4M3", "fp8": "FP8_E4M3"}
  if require_resolved and (resolved.get('activation_dtype') not in names or
                          resolved.get('kv_cache_dtype') not in (*names, 'auto')):
    raise ValueError('attention work requires recorded supported activation and KV cache formats')
  fmt = names.get(resolved.get('activation_dtype'), fmt)
  kv = names.get(resolved.get('kv_cache_dtype'), fmt)
  return {"activations": {"in": fmt, "out": fmt}, "attention_activations": {"in": fmt, "out": fmt}, "kv": kv,
          "recurrent_form": VLLM_RECURRENT_FORM if resolved.get('model_type') in (None, 'qwen3_5', 'qwen3_next') else None,
          "activations_by_quant": {q: {"in": q, "out": fmt} for q in weights if q in {"NVFP4_G16", "FP8_E4M3"}},
          "words": f"{fmt.lower()} activations in and out; KV cache {kv.lower()} " + ("(resolved by engine)" if resolved else "(dtype auto assumed)")}


VLLM_RECURRENT_FORM = {"name": "chunked_wy", "chunk": 64}  # vLLM's prefill runs the gated delta rule chunked (WY), chunk 64


def vllm_capture(run:pathlib.Path, model:str, length:int, target) -> dict[str, Any]:
  """The measured call alone (the driver marks it for nsys), the engine in this process so nsys sees it."""
  from boltbeam.collectors import vendor_capture, vllm_decode as vd
  from boltbeam.collectors.capture_settings import vllm_flags
  profile = json.loads((run/'model_profile.json').read_text())
  # labels the launches by module where geometry cannot pair them: Mamba2's kernels, and ROCm's, whose matrices share
  # grids and order (two FFN launches after a recurrent block fit one role equally on an R9700)
  semantic = ((profile.get('metadata') or {}).get('recurrent') or {}).get('kind') == 'mamba2' or target.backend == 'AMD'
  argv, env = _vllm(model, length, "--capture", *(['--semantic-dispatch'] if semantic else []), *vllm_flags(run, length, capture=True))
  raw = run / RAW / str(length)
  if semantic:
    # Reuse the measured autotuner choice across the timing/counter processes.
    # Otherwise each process can choose a different tile, invalidating traffic.
    env = {**env, 'TRITON_CACHE_AUTOTUNING':'1'}
  launches = vendor_capture.capture(target.backend, argv, raw, env={**env, **vd.CAPTURE_ENV}, profiler_range=True,
                                    device=run_gpu(run))
  line = json.loads([x for x in vendor_capture.program_output(raw).splitlines() if x.startswith('{"length"')][-1])
  proof = None
  if semantic:
    import hashlib
    from boltbeam.collectors import semantic_capture, vllm_semantic_work, cuda_memory
    launches = vllm_semantic_work.attach(semantic_capture.read(raw/'capture.sqlite'),pathlib.Path(model),profile,
                                         ssd_chunk=line.get('mamba_chunk_size'),routing=line.get('routing'))
    # AOT compilation may set a private Triton cache only in the timing process.
    # Reuse that observed cache, rather than letting the counter process retune.
    counter_env={**env,**vd.CAPTURE_ENV,'TRITON_CACHE_DIR':line['triton_cache_dir']}
    launches, memory_file = cuda_memory.collect(argv,counter_env,raw,launches,line)
    facts = raw/'semantic_work.json'
    facts.write_text(json.dumps({'engine':line,'launches':launches},indent=1))
    proof={'schema':'boltbeam.semantic_work.v1','execution_mode':line['execution_mode'],
           'model_profile_sha256':hashlib.sha256((run/'model_profile.json').read_bytes()).hexdigest(),
           'sources':[{'file':str(f.relative_to(run)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
                      for f in (facts,memory_file) if f is not None]}
  chunk = vllm_chunk(length, line)
  tools = pathlib.Path(vd.python() or "").parent.parent
  bins = sorted(tools.glob("lib/python*/site-packages/nvidia/*/bin"))
  paths = library_paths([pathlib.Path(p) for p in line.get("libraries") or []], launches,
                        tools=bins[0] if bins else None, jit_caches=line.get("jit_caches"), capture=raw, device=run_gpu(run))
  return {"launches": launches, "chunk": chunk["widths"], "outputs": chunk["outputs"], "warmup": False,
          "formats": vllm_formats(weight_quants(run), line), "method": vendor_capture.plan(target.backend)["method"],
          "matrices": line.get("matrices"), "captured_ms": line["wall_s"] * 1e3, "window": line.get("window_monotonic"),
          "work_evidence":proof, "paths": paths,
          "program": {"argv": argv, "env": {**env, **vd.CAPTURE_ENV}, "raw": raw, "profiler_range": True,
                      "tools": bins[0] if bins else None, "arch": (run_gpu(run) or {}).get("architecture")}}


def observe_closed(cap:dict[str, Any], backend:str) -> dict[str, Any]:
  """The capture's paths with what the vendor's observer (target/vendors.py closed_observer) saw the kernels whose code
  was not read execute (a closed library such as cuBLAS), for any engine whose capture names the program it ran
  (program: argv, env, raw; profiler_range: whether it opens one; tools: a folder of the vendor's tools; arch: the
  run's GPU target): "observed", "observed_how" and "observed_by" (the observer's label), or "observed_how" alone saying why nothing was observed; paths unchanged with no observer."""
  from boltbeam.collectors.instruction_path import mixed, template
  from boltbeam.target import vendors
  row, prog, paths = vendors.BY_BACKEND.get(backend), cap.get("program"), cap["paths"]
  observer = vendors.resolve(row.capture.closed_observer) if row else None
  ran = [template(l["key"]) for l in sorted(cap["launches"], key=lambda l: l["start_us"])]
  # closed: no code read, or code holding several matrix paths (which one ran is the observer's to say)
  closed = {t for t in ran if t not in paths["kernels"] or len(mixed(paths["kernels"][t])) > 1}
  if observer is None or not prog or not closed:
    return paths
  return {**paths, **observer(prog, closed, [t for t in ran if t in closed])}


def engine_roles(roles:list[dict[str, Any]], matrices:Any) -> tuple[list[dict[str, Any]], str]:
  """The weight matrices the engine runs, as roles to pair launches with: an engine that packs several checkpoint
  tensors into one GEMM (q, k and v as one N) runs one launch for their work. Each matrix's parts are named by the
  role classifier and matched to the profile's roles by K; its weight bytes are its parts'. The matrices replace
  the profile's roles only when they cover every role's work (count x N x K) exactly; else the roles stand, said."""
  from boltbeam.profile.roles import classify_tensor_role, role_from_tensor_name
  if not isinstance(matrices, list) or not matrices:
    return roles, f"the profile's roles (the engine's matrices: {matrices or 'not reported'})"
  left: dict[tuple[str, str, int], int] = {}
  for r in roles:  # rows of one role and shape (k and v) add up
    left[(r["role"], r["quant"], r["k"])] = left.get((r["role"], r["quant"], r["k"]), 0) + r["count"] * r["n"]
  out = []
  for m in matrices:
    parts = [role_from_tensor_name(f"{p}.weight", (1 << 20, int(m["k"]))) for p in m["parts"]]
    found = [next((r for r in roles if r["role"] == p and r["k"] == int(m["k"])), None) for p in parts]
    if not all(found):
      continue
    if sum(r["n"] for r in found) != int(m["n"]):
      return roles, f"the profile's roles ({m['name']}: N {m['n']} is not the sum of its parts' rows)"
    for r in found:
      left[(r["role"], r["quant"], r["k"])] -= int(m["count"]) * r["n"]
    quants = sorted({r["quant"] for r in found})
    out.append({"role": m["name"] if len(found) == 1 and m["name"] == found[0]["role"] else
                f"{m['name']} ({' + '.join(sorted({r['role'] for r in found}))})",
                "quant": "+".join(quants), "n": int(m["n"]), "k": int(m["k"]), "count": int(m["count"]),
                "weight_bytes": sum(r["weight_bytes"] for r in found), "parts": [r["role"] for r in found],
                "part_roles": [{"role": r["role"], "quant": r["quant"], "weight_bytes": r["weight_bytes"], "shape": [r["n"], r["k"]]}
                               for r in found],
                "fine": [classify_tensor_role(f"{p}.weight", (1 << 20, int(m["k"]))).value for p in m["parts"]]})
  if any(left.values()):
    return roles, "the profile's roles (the engine's matrices do not cover " + ", ".join(
      f"{k[0]} {k[1]}" for k, v in left.items() if v) + " exactly)"
  return out, "the engine's own weight matrices (packed parts' work and bytes summed; every role's work covered once)"


PROMPT_WORDS["vllm"] = ("the driver's own prompt: token ids drawn by random.Random(seed).randrange(vocab), each call new ids, "
                        "prefix caching off, so nothing is cached between prompts")


ENGINES: dict[str, dict[str, Callable[..., dict[str, Any]]]] = {
  "llama.cpp": {"truth": lambda run, model, length, target, root: llama_truth(run, model, length),
                "capture": lambda run, model, length, target, root: llama_capture(run, model, length, target)},
  "tinygrad": {"truth": lambda run, model, length, target, root: tinygrad_truth(run, model, length, target=target, root=root),
               "capture": lambda run, model, length, target, root: tinygrad_capture(run, model, length, target, root=root)},
  "vllm": {"truth": lambda run, model, length, target, root: vllm_truth(run, model, length),
           "capture": lambda run, model, length, target, root: vllm_capture(run, model, length, target)},
}


def _engine(provider:str) -> dict[str, Callable[..., dict[str, Any]]]:
  if provider not in ENGINES:
    raise RuntimeError(f"prefill is measured for {', '.join(ENGINES)}; {provider} has no prefill adapter")
  return ENGINES[provider]


# --- step 4 and step 5 -----------------------------------------------------------------------------------------

def measure(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> pathlib.Path:
  """trace_request.json answered with one untraced prefill per requested prompt length."""
  from boltbeam.core.canonical import pretty_json
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest, read_json
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  rows = []
  from boltbeam.collectors.capture_settings import contexts, require_supported
  if provider not in ("llama.cpp", "vllm"):
    require_supported(run)
  for length in contexts(run):
    say(f"prefill of {length} tokens: {provider}")
    settled, began = gpu_telemetry.settle(run, say), time.monotonic()
    try:
      got = _engine(provider)["truth"](run, str(manifest["model_path"]), int(length), target, root)
    except RuntimeError as exc:  # a length the engine could not read (out of memory at the cap): said, the rest go on
      say(f"prefill of {length} tokens failed: {str(exc)[-200:]}")
      rows.append({"scope": "whole_step", "workload": "prefill", "context": int(length), "prompt_tokens": int(length),
                   "failed": str(exc)[-600:], "seconds": round(time.monotonic() - began, 1)})
      continue
    tele = got["telemetry"]
    rows.append({"scope": "whole_step", "workload": "prefill", "context": int(length), "prompt_tokens": int(length),
                 "wall_us": got["ms"] * 1e3, "prefill_ms": got["ms"], "tok_s": int(length) / got["ms"] * 1e3,
                 "samples_ms": got["samples_ms"], "qualification_ms": got.get("qualification_ms"), "chunk": got["chunk"], "batched": got["chunk"]["batched"],
                 **({"replay": got["replay"], "replay_failed": got.get("replay_failed")} if "replay" in got else {}),
                 "telemetry": tele, "throttled": tele.get("throttled"), "telemetry_samples": got.get("telemetry_samples"),
                 "source": got["source"], "engine": got["engine"], **({"settled_s": settled} if settled else {}),
                 "seconds": round(time.monotonic() - began, 1)})
  if not any(r.get("prefill_ms") for r in rows):
    raise RuntimeError("no prompt length was read: " + "; ".join(f"{r['context']}: {r['failed'][-160:]}" for r in rows))
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": "prefill", "provider": provider, "provider_id": provider, "collector_id": "prefill",
           "timing_source": "untraced prefill wall time", "contexts": [r["context"] for r in rows], "rows": rows,
           "clock_policy": gpu_telemetry.run_clock_policy(run),
           "measured": ["whole_step.prefill_ms (the prompt read untraced at the engine's own chunk)"],
           **({"sweep": read_json(run / "prefill_sweep.json")} if (run / "prefill_sweep.json").exists() else {})}
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


def last_level_cache(target) -> dict[str, Any] | None:
  """The chip's last-level cache: its size and measured read bandwidth, from the target's facts; None if either is
  not recorded."""
  facts = (target.capabilities or {}).get("fact_sources") or {}
  size = next((v.get("l2_cache_bytes") for v in facts.values() if isinstance(v, dict) and v.get("l2_cache_bytes")), None)
  rate = facts.get("l2_bandwidth_gbs") or {}
  gbs = rate.get("value_gbs")
  return {"bytes": float(size), "gbs": float(gbs),
          **({'calibration':rate['calibration']} if rate.get('calibration') else {})} if size and gbs else None


def prefill_roles(profile:dict[str, Any], ceil_roles:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """The profile's weight roles with their shape, launches per chunk, weight bytes per launch (the limit's own
  bytes over the role's count, so a role's bytes are the decode table's) and fine class (its place in the layer).
  An expert stack (n_expert) runs every token through experts_read experts (routed: products per token); its bytes
  stay the decode's experts_read experts: exact for one token; a chunk's ceiling is refused (prefill_attribution.ROUTED_REFUSED).
  last_layer: the row's calls in the model's last layer, when the profile read it (the layer an engine may prune);
  layer_kinds: a hybrid's row's calls per layer kind and fine class, when the profile read them."""
  from boltbeam.role_key import role_key
  by = {role_key(r): r for r in ceil_roles}
  out = []
  for r in profile.get("roles") or []:
    c = by.get(role_key(r))
    if c and r.get("count") and r.get("rows") and r.get("cols"):
      moe = {"n_expert": int(r["n_expert"]), **({"routed": int(r["experts_read"])} if r.get("experts_read") else {})} if r.get("n_expert") else {}
      out.append({"role": r["role"], "quant": r["quant"], "n": int(r["rows"]), "k": int(r["cols"]), "count": int(r["count"]),
                  "weight_bytes": float(c["bytes_moved"]) / int(r["count"]), "role_class": r.get("role_class"), **moe,
                  **({"last_layer": int(r["last_layer"])} if r.get("last_layer") is not None else {}),
                  **({"layer_kinds": r["layer_kinds"]} if r.get("layer_kinds") else {}),
                  **({"per_layer": r["per_layer"]} if r.get("per_layer") else {})})
  return out


def capture_peaks(peaks:dict[str, float], source:dict[str, Any] | None, seen:dict[str, Any],
                  window:list[float] | None) -> tuple[dict[str, float], str]:
  """The peaks at the clock the measured prefill ran at: seen is the telemetry over the capture's window (the samples
  next to it), not over the engine's start-up, whose low clocks would scale every peak down. With no window, the whole
  run's samples stand for it only when the driver marks the idle ones (NVML); amdgpu marks none, so they do not."""
  busy = window or seen.get("throttled") is not None
  got = peaks_at(peaks, source, (seen.get("sm_mhz") or {}).get("mean") if busy else None)
  if not busy and seen.get("sm_mhz"):
    return got[0], ("peaks at the clock they were measured at: the capture marks no busy window and the driver no "
                    "idle samples, so the run's mean clock would include the engine's start-up")
  return got


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None, depth:int = 0,
              lengths:list[int] | None = None, truth_ms:dict[int, float] | None = None,
              capture:Callable[..., dict[str, Any]] | None = None, trace:str = TRACE, rest:bool = True) -> dict[str, Any]:
  """One captured prefill per prompt length, attributed: prefill_trace.json. A session's turn is the same at a depth:
  its lengths, its untraced turn times, its own capture and trace file (collectors/session.py). rest: the GPU idles
  first (gpu_telemetry.settle), as the untraced arm did; False where that arm did not (a replay's server turns)."""
  from boltbeam.collectors.prefill_attribution import UnnamedCapture, attribute_prefill
  from boltbeam.target.targets import get_target
  from boltbeam.workflow import tie_out
  from boltbeam.workflow.common import load_manifest, read_json
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional, run_bandwidth
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  profile = _optional(run, "model_profile.json")
  ceil = _measured_vs_ceiling({**manifest, "workload": "decode"}, profile, run)
  bw, bw_source = run_bandwidth(run, target)
  roles = prefill_roles(profile, ceil.get("_roles") or [])
  # the engine's own timing slows its run (one command buffer per kernel): its idle is profiling, not the engine's,
  # so its gaps are the untraced prefill less the busy time, as the decode tie-out treats its own timing
  truth_rows = [r for r in read_json(run / "timing_trace.json").get("rows") or [] if r.get("prefill_ms")]
  if truth_ms is None:
    truth_ms = {int(r["context"]): r["prefill_ms"] for r in truth_rows}
  truth_chunk = {int(r["context"]): r.get("chunk") or {} for r in truth_rows}
  out = {"schema": "boltbeam.prefill_trace.v1", "provider": provider, "model_id": manifest["model_id"],
         "target_id": target.target_id, "bandwidth_gbs": bw, "bandwidth_source": bw_source,
         "peaks": dict(target.matrix_tflops or {}), "peaks_source": ((target.capabilities or {}).get("fact_sources") or {}).get("matrix_tflops"),
         "lengths": {}, **({"depth": depth} if depth else {})}
  from boltbeam.target.targets import fact_uncertainty  # each figure's own measured uncertainty, for the guards
  out["uncertainty"] = {"bandwidth": ceil.get("uncertainty"),
                        "peaks": {p: fact_uncertainty(target, "matrix_tflops", p) for p in out["peaks"]}}
  per_chunk = None  # launches per engine chunk, from the last capture: the next capture's size is estimated from it
  from boltbeam.collectors.capture_settings import contexts, require_supported
  if provider not in ("llama.cpp", "vllm"):
    require_supported(run)
  for length in lengths if lengths is not None else contexts(run):
    if int(length) not in truth_ms:
      out["lengths"][str(length)] = {"not_captured": "the untraced prefill failed at this length (timing_trace.json)"}
      continue
    chunks = len(split_chunks(int(length), (truth_chunk.get(int(length)) or {}).get("size") or int(length)))
    if per_chunk and COPIES * per_chunk * chunks > MAX_CAPTURE_LAUNCHES:
      say(f"prefill of {length} tokens: not captured (about {COPIES * per_chunk * chunks:,.0f} launches)")
      out["lengths"][str(length)] = {"not_captured": f"about {COPIES * per_chunk * chunks:,.0f} launches with the warm-up, over "
                                                     f"{MAX_CAPTURE_LAUNCHES:,}: the untraced time, MFU and clock stand alone"}
      continue
    say(f"prefill of {length} tokens in the model{f' after {depth} in the cache' if depth else ''}: {provider}")
    settled = {"settled_s": s} if rest and (s := gpu_telemetry.settle(run, say)) else {}
    unnamed = None
    with gpu_telemetry.run_telemetry(run, 0) as tele:
      try:
        cap = (capture or _engine(provider)["capture"])(run, str(manifest["model_path"]), int(length), target, root)
      except UnnamedCapture as exc:  # the generic kernel timer (engine_kernels) times a decode token's kernels only
        unnamed = exc
    if unnamed is not None:
      what = "turn" if depth else "prefill"  # a session's turn is a prefill at a depth
      clock = gpu_telemetry.witness_clock(tele.summary().get("gpu_witness"))  # the captured arm's own clock
      out["lengths"][str(length)] = {**settled, **({"clock_mhz": [round(x, 1) for x in clock]} if clock else {}),
                                     "captured": unnamed.token, "not_captured": f"in-model not available here: {str(unnamed).replace('prefill', what)}; "
                                     f"the generic kernel timer times one decode token's kernels, not a {what}, so the untraced time stands alone"}
      continue
    cap["paths"] = observe_closed(cap, target.backend)  # every engine's closed kernels, outside the timed window
    w = cap.get("window")
    seen = tele.summary(w[0] - tele.interval_s, w[1] + tele.interval_s) if w else tele.summary()
    peaks, peaks_clock = capture_peaks(dict(target.matrix_tflops or {}), out["peaks_source"], seen, w)
    pairing, pairing_source = engine_roles(roles, cap["matrices"]) if "matrices" in cap else (roles, "the profile's roles")
    attr = attribute_prefill(cap["launches"], [dict(r) for r in pairing], chunk_widths=cap["chunk"], outputs=cap["outputs"],
                             layers=int(profile.get("layer_count") or 0), kernel_paths=cap["paths"]["kernels"],
                             peaks=peaks, bandwidth_gbs=bw, profile=profile, length=int(length),
                             program_roles=cap.get("program_roles"), warmup=cap["warmup"],
                             untraced_ms=(truth_ms.get(int(length)) if cap.get("own_timing") else None),
                             formats=cap.get("formats"), cache=last_level_cache(target), depth=depth,
                             observed=cap["paths"].get("observed"), observed_by=cap["paths"].get("observed_by"), prune=cap.get("prune"), unc=out["uncertainty"],
                             measured_ms=cap.get("measured_ms"))
    from boltbeam.workflow.repeat_bands import find
    attr["measured_ms"] = cap.get("measured_ms")
    attr["timing_window"] = cap.get("timing_window")
    attr["end_to_end"] = tie_out.end_to_end(attr, truth_ms.get(int(length)), band=find(run, f"prefill {length}"))
    per_chunk = len(cap["launches"]) / cap.get("copies", 2 if cap["warmup"] else 1) / max(len(cap["chunk"]), 1)
    out["lengths"][str(length)] = {"formats": (cap.get("formats") or {}).get("words") or "fp16 activations in and out (the adapter names none)",
                                   "work_evidence":cap.get('work_evidence'),
                                   "method": cap["method"], "captured_ms": cap.get("captured_ms"), "telemetry": seen, **settled,
                                   **({"measured_ms": cap["measured_ms"]} if cap.get("measured_ms") else {}),
                                   "peaks_at": peaks, "peaks_clock": peaks_clock, "pairing_roles": pairing_source,
                                   "paths_how": cap["paths"]["how"], "libraries": cap["paths"]["libraries"], **attr}
    from boltbeam.collectors.semantic_attribution import bind_cache_rate
    bind_cache_rate(run,run/RAW/str(length),out['lengths'][str(length)],last_level_cache(target))
    # Preserve completed lengths if a later capture is stopped or rejected.
    (run / trace).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
  (run / trace).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
  return out
