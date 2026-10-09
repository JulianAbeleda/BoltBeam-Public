#!/usr/bin/env python3
"""Read bandwidth of one GPU, or copy bandwidth and latency between two, in tinygrad, for the speed limit.

The tinygrad fork's python runs this file by path, with the fork as the working directory. It imports only
tinygrad and the standard library and prints one JSON line.

    read --device NV:1                    a read-only sum over 1 GiB on that GPU, best of 5

The device goes on the tensors, never in DEV: in this fork DEV=NV:1 means "device NV, renderer 1"
(tinygrad/device.py _select_renderer), which fails with "NV has no renderer '1'".
    copy --src NV:0 --dst NV:1            256 MiB copied between the two, best of 5; latency is the median
                                          of 20 copies of 4 KiB
"""
from __future__ import annotations

import argparse, json, statistics, sys, time


def _sync(*devs):
  from tinygrad.device import Device
  for d in devs: Device[d].synchronize()


def read(dev:str, nbytes:int, reps:int) -> dict:
  from tinygrad import Tensor, dtypes
  a = Tensor.ones(nbytes // 4, dtype=dtypes.float32, device=dev).contiguous().realize()
  best = None
  for _ in range(reps + 1):  # the first run compiles
    _sync(dev); t0 = time.perf_counter()
    a.sum().realize(); _sync(dev)
    dt = time.perf_counter() - t0
    best = dt if best is None or dt < best else best
  return {"read_gbs": nbytes / best / 1e9, "bytes": nbytes, "reps": reps, "device": dev}


def copy(src:str, dst:str, nbytes:int, reps:int) -> dict:
  from tinygrad import Tensor, dtypes
  a = Tensor.ones(nbytes // 4, dtype=dtypes.float32, device=src).contiguous().realize()
  small = Tensor.ones(1024, dtype=dtypes.float32, device=src).contiguous().realize()
  best = None
  for _ in range(reps + 1):
    _sync(src, dst); t0 = time.perf_counter()
    a.to(dst).realize(); _sync(src, dst)
    dt = time.perf_counter() - t0
    best = dt if best is None or dt < best else best
  lat = []
  for _ in range(21):
    _sync(src, dst); t0 = time.perf_counter()
    small.to(dst).realize(); _sync(src, dst)
    lat.append(time.perf_counter() - t0)
  return {"copy_gbs": nbytes / best / 1e9, "latency_us": statistics.median(lat[1:]) * 1e6, "bytes": nbytes,
          "reps": reps, "src": src, "dst": dst}


def main(argv=None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("mode", choices=["read", "copy"])
  p.add_argument("--src"); p.add_argument("--dst"); p.add_argument("--device")
  p.add_argument("--bytes", type=int, default=None)
  p.add_argument("--reps", type=int, default=5)
  a = p.parse_args(argv)
  out = read(a.device, a.bytes or (1 << 30), a.reps) if a.mode == "read" else copy(a.src, a.dst, a.bytes or (256 << 20), a.reps)
  print(json.dumps(out))
  return 0


if __name__ == "__main__":
  sys.exit(main())
