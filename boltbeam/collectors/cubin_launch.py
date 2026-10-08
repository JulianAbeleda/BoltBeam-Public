#!/usr/bin/env python3
"""Replay a compiler-exported cubin through the CUDA driver API, so Nsight Compute can see it.

A compiler that drives the GPU without the CUDA driver (tinygrad's NV backend submits its own queues) is invisible
to ncu.  Its producer contract is a cubin plus the launch spec -- ``tinygrad.nv_cubin_capture.v1``: per captured
kernel ``name``, ``cubin_path``, ``cubin_sha256``, ``shared_mem`` (the launch-sized shared memory; older
captures carry only ``shmem_usage``) and ``calls`` (``global_size`` = grid in blocks,
``local_size`` = block, ``vals`` = 32-bit scalar parameters after the buffer pointers, ``n_bufs`` and ``buf_meta``
sizes).  This module loads that cubin with ``cuModuleLoadData`` and launches it with ``cuLaunchKernel`` in an ordinary
context -- through ``libcuda`` directly, no compiler import -- so ``boltbeam ncu-collect`` can profile it.  The
explicit flags (``--cubin --symbol --grid --block ...``) keep the older launcher's interface.

  python3 -m boltbeam.collectors.cubin_launch --capture cap.json --kernel E_2_137_... --label ssm_in:128 --out launch.json

With ``--label ROLE:M`` exactly one launch is bracketed by cuProfilerStart/Stop and ``SHAPE ROLE M 1`` is printed
(the collector's labelling protocol; ncu runs with --profile-from-start off).
``--condition-mib N`` streams N MiB (``cuMemsetD8``) between a reheat and each timed launch to evict L2.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, sys
from typing import Any, Mapping

CAPTURE_SCHEMA = "tinygrad.nv_cubin_capture.v1"
SCHEMA = "boltbeam.cubin_launch.v1"
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_CU_STREAM_NON_BLOCKING = 1


def launch_spec_from_capture(capture:Mapping[str, Any], kernel:str, call:int=0) -> dict[str, Any]:
  """One launch spec from the producer contract; refuses anything malformed or not matching its sha256."""
  if capture.get("schema") != CAPTURE_SCHEMA: raise ValueError(f"not a {CAPTURE_SCHEMA} document")
  rows = [r for r in capture.get("captured", ()) if r.get("name") == kernel or r.get("name", "").startswith(kernel)]
  if len(rows) != 1: raise ValueError(f"{len(rows)} captured kernels match {kernel!r}")
  row = rows[0]
  calls = row.get("calls") or []
  if not 0 <= call < len(calls): raise ValueError(f"kernel {row['name']} has {len(calls)} captured calls")
  c = calls[call]
  sizes = [int(b["size"]) for b in c.get("buf_meta", ())]
  if len(sizes) != int(c["n_bufs"]) or any(s <= 0 for s in sizes): raise ValueError("buf_meta sizes do not cover n_bufs")
  grid, block = [int(x) for x in c["global_size"]], [int(x) for x in c["local_size"]]
  if len(grid) != 3 or len(block) != 3: raise ValueError("grid and block must be 3-dimensional")
  return {"cubin_path": row["cubin_path"], "cubin_sha256": row["cubin_sha256"], "symbol": row["name"], "grid": grid,
          "block": block, "shared_mem": int(row["shared_mem"] if "shared_mem" in row else row.get("shmem_usage") or 0), "buf_sizes": sizes,
          "vals": [int(v) for v in c.get("vals", ())], "val_groups": [1] * len(c.get("vals", ()))}


def _check(lib, name:str, *args) -> None:
  rc = getattr(lib, name)(*args)
  if rc != 0: raise RuntimeError(f"{name} failed with CUresult {rc}")


def launch(spec:Mapping[str, Any], *, reps:int=3, warmup:int=20, condition_mib:int=0, profile_one:bool=False,
           libcuda:str="libcuda.so.1") -> dict[str, Any]:
  """Warm up, time ``reps`` launches with events; with ``profile_one`` also bracket exactly one launch in
  cuProfilerStart/Stop (the collector runs ncu with --profile-from-start off)."""
  blob = pathlib.Path(spec["cubin_path"]).read_bytes()
  sha = hashlib.sha256(blob).hexdigest()
  if spec.get("cubin_sha256") and spec["cubin_sha256"] != sha: raise ValueError("cubin bytes do not match the captured sha256")
  cu = ctypes.CDLL(libcuda)
  ptr, handle = ctypes.c_uint64, ctypes.c_void_p
  _check(cu, "cuInit", 0)
  dev, ctx, module, fn, stream = ctypes.c_int(), handle(), handle(), handle(), handle()
  _check(cu, "cuDeviceGet", ctypes.byref(dev), 0)
  _check(cu, "cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)   # cuMemAlloc rejects a non-primary context here
  bufs: list[ctypes.c_uint64] = []
  try:
    _check(cu, "cuCtxSetCurrent", ctx)
    _check(cu, "cuModuleLoadData", ctypes.byref(module), ctypes.c_char_p(blob))
    _check(cu, "cuModuleGetFunction", ctypes.byref(fn), module, spec["symbol"].encode())
    if spec["shared_mem"]: _check(cu, "cuFuncSetAttribute", fn, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, spec["shared_mem"])
    for size in spec["buf_sizes"]:
      p = ptr(); _check(cu, "cuMemAlloc_v2", ctypes.byref(p), ctypes.c_size_t(size)); bufs.append(p)
      _check(cu, "cuMemsetD8_v2", p, ctypes.c_ubyte(0), ctypes.c_size_t(size))
    vals, groups = spec["vals"], spec.get("val_groups") or [1] * len(spec["vals"])
    if sum(groups) != len(vals): raise ValueError("val_groups do not cover vals")
    holders, cursor = [], 0
    for width in groups:
      holders.append((ctypes.c_int32 * width)(*vals[cursor:cursor + width])); cursor += width
    params = (ctypes.c_void_p * (len(bufs) + len(holders)))(
      *[ctypes.cast(ctypes.pointer(b), ctypes.c_void_p) for b in bufs], *[ctypes.cast(h, ctypes.c_void_p) for h in holders])
    _check(cu, "cuStreamCreate", ctypes.byref(stream), _CU_STREAM_NON_BLOCKING)
    flush = ptr()
    if condition_mib: _check(cu, "cuMemAlloc_v2", ctypes.byref(flush), ctypes.c_size_t(condition_mib << 20)); bufs.append(flush)
    (gx, gy, gz), (bx, by, bz) = spec["grid"], spec["block"]
    def once():
      _check(cu, "cuLaunchKernel", fn, gx, gy, gz, bx, by, bz, spec["shared_mem"], stream, params, None)
      if condition_mib:
        _check(cu, "cuMemsetD8Async", flush, ctypes.c_ubyte(1), ctypes.c_size_t(condition_mib << 20), stream)
        _check(cu, "cuLaunchKernel", fn, gx, gy, gz, bx, by, bz, spec["shared_mem"], stream, params, None)
    for _ in range(warmup): once()
    _check(cu, "cuStreamSynchronize", stream)
    if profile_one:
      _check(cu, "cuProfilerStart")
      _check(cu, "cuLaunchKernel", fn, gx, gy, gz, bx, by, bz, spec["shared_mem"], stream, params, None)
      _check(cu, "cuStreamSynchronize", stream)
      _check(cu, "cuProfilerStop")
    begin, end, ms = handle(), handle(), ctypes.c_float()
    _check(cu, "cuEventCreate", ctypes.byref(begin), 0); _check(cu, "cuEventCreate", ctypes.byref(end), 0)
    _check(cu, "cuEventRecord", begin, stream)
    for _ in range(reps): once()
    _check(cu, "cuEventRecord", end, stream); _check(cu, "cuEventSynchronize", end)
    _check(cu, "cuEventElapsedTime", ctypes.byref(ms), begin, end)
    cu.cuEventDestroy_v2(begin); cu.cuEventDestroy_v2(end); cu.cuStreamDestroy_v2(stream)
  finally:
    for b in bufs: cu.cuMemFree_v2(b)
    if module.value: cu.cuModuleUnload(module)
    cu.cuDevicePrimaryCtxRelease(dev)
  return {"schema": SCHEMA, **{k: spec[k] for k in ("symbol", "grid", "block", "shared_mem", "buf_sizes", "vals", "val_groups")},
          "cubin": str(spec["cubin_path"]), "cubin_sha256": sha, "reps": reps, "warmup": warmup, "condition_mib": condition_mib,
          "event_us_per_launch": 1000.0 * ms.value / reps, "verdict": "CUDA_LAUNCH_OK"}


def _ints(text:str | None) -> list[int]: return [int(x) for x in text.split(",")] if text else []


def spec_from_args(args:argparse.Namespace) -> dict[str, Any]:
  if args.capture:
    return launch_spec_from_capture(json.loads(pathlib.Path(args.capture).read_text()), args.kernel, args.call)
  if not (args.cubin and args.symbol): raise SystemExit("either --capture/--kernel or --cubin/--symbol")
  grid = _ints(args.grid) or [args.grid_x, 1, 1]
  block = _ints(args.block) or [args.block_x, 1, 1]
  sizes = _ints(args.buf_sizes) or [args.buf_bytes] * args.n_bufs
  if len(sizes) != args.n_bufs: raise SystemExit(f"--buf-sizes has {len(sizes)} entries but --n-bufs is {args.n_bufs}")
  vals = _ints(args.vals)
  return {"cubin_path": str(args.cubin), "cubin_sha256": None, "symbol": args.symbol, "grid": grid, "block": block,
          "shared_mem": args.shared_mem, "buf_sizes": sizes, "vals": vals, "val_groups": _ints(args.val_groups) or [1] * len(vals)}


def main(argv:list[str] | None=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--capture", help=f"{CAPTURE_SCHEMA} JSON"); ap.add_argument("--kernel"); ap.add_argument("--call", type=int, default=0)
  ap.add_argument("--cubin"); ap.add_argument("--symbol")
  ap.add_argument("--grid-x", type=int, default=12288); ap.add_argument("--block-x", type=int, default=32)
  ap.add_argument("--grid"); ap.add_argument("--block")
  ap.add_argument("--shared-mem", type=int, default=0)
  ap.add_argument("--n-bufs", type=int, default=4); ap.add_argument("--buf-bytes", type=int, default=64 << 20)
  ap.add_argument("--buf-sizes"); ap.add_argument("--vals"); ap.add_argument("--val-groups")
  ap.add_argument("--reps", type=int, default=3); ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--condition-mib", type=int, default=0)
  ap.add_argument("--label", help="ROLE:M -- print 'SHAPE ROLE M <launches>' for the ncu collector")
  ap.add_argument("--out", type=pathlib.Path)
  args = ap.parse_args(argv)
  result = launch(spec_from_args(args), reps=args.reps, warmup=args.warmup, condition_mib=args.condition_mib,
                  profile_one=bool(args.label))
  if args.label:
    role, m = args.label.split(":")
    print(f"SHAPE {role} {m} 1", flush=True)
  text = json.dumps(result, indent=2, sort_keys=True)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(text + "\n")
  print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
