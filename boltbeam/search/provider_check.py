"""BoltBeam's own check of what a search provider claims. The provider is not trusted.

Three claims come back from the provider, and each is checked here with BoltBeam's tools, never the provider's:

  device identity   the facts the provider read from its GPU against the facts BoltBeam's own bridge reads from the
                    same machine (runtime/cuda_device.py, runtime/metal_device.py): every fact both report must agree,
                    and at least one must be shared, or the identity is refused
  correctness       the kernel the provider compiled, compiled again by BoltBeam's bridge from the source the provider
                    returned, run on the model's own weight bytes, its output checked against BoltBeam's reference
                    (metal_native.reference_row) with the activation as the kernel reads it (fp16, the same rule as the
                    q8_1 vector of the CUDA engine adapter)
  timing            that kernel timed by BoltBeam's kernel timer (collectors/kernel_timer.py: one loop, a read sweep
                    of the last-level cache before each sample, the dispatch floor beside it); the provider's median
                    against BoltBeam's median less the floor, judged by the chip's band

A provider time outside the band is "provider claim not reproduced". The caller never promotes such a candidate.
"""
from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import Any

from boltbeam.collectors import kernel_timer
from boltbeam.collectors.kernel_timer import Check, KernelSpec

NOT_REPRODUCED = "provider claim not reproduced"
REPRODUCED = "reproduced"
# a provider fact and the key BoltBeam's bridge reports the same quantity under
IDENTITY_KEYS = (("sm_count", "sm_count"), ("l2_bytes", "l2_cache_bytes"), ("compute_capability", "compute_capability"),
                 ("name", "name"))
DTYPES = {"float": ("f", 4), "float32": ("f", 4), "half": ("e", 2), "float16": ("e", 2)}


def bridge_facts(bridge) -> dict[str, Any]:
  """What BoltBeam's bridge reads from the GPU: the driver's facts where the bridge has them, else its name."""
  facts = dict(bridge.facts()) if hasattr(bridge, "facts") else {"name": getattr(bridge, "name", None)}
  major, minor = facts.get("compute_capability_major"), facts.get("compute_capability_minor")
  if major is not None and minor is not None:
    facts["compute_capability"] = [int(major), int(minor)]
  return {k: v for k, v in facts.items() if v is not None}


def identity(provider_facts:Mapping[str, Any], mine:Mapping[str, Any]) -> dict[str, Any]:
  """Compare the provider's device facts with BoltBeam's. The provider's name is compared only when it is a device
  name (an NVIDIA device opened through the resource manager reports none; the fork's device label is not one)."""
  compared, mismatched = [], []
  for theirs, ours in IDENTITY_KEYS:
    a, b = provider_facts.get(theirs), mine.get(ours)
    if a is None or b is None:
      continue
    if theirs == "name" and a == provider_facts.get("tinygrad_device"):
      continue
    same = list(a) == list(b) if isinstance(a, (list, tuple)) else a == b
    compared.append({"fact": theirs, "provider": a, "boltbeam": b, "same": same})
    if not same:
      mismatched.append(theirs)
  if mismatched:
    reason = "the provider's device is not the GPU BoltBeam reads: " + ", ".join(
      f"{c['fact']} {c['provider']} against {c['boltbeam']}" for c in compared if not c["same"])
  elif not compared:
    reason = "the provider reported no fact BoltBeam can check against its own reading of the GPU"
  else:
    reason = None
  return {"passed": reason is None, "compared": compared, "reason": reason}


def half_round(x:list[float]) -> list[float]:
  """The activation as an fp16 kernel reads it."""
  return list(struct.unpack(f"<{len(x)}e", struct.pack(f"<{len(x)}e", *x)))


def spec_from_record(record:Mapping[str, Any], *, quant:str, rows:int, cols:int, weights:bytes, x:list[float],
                     label:str) -> KernelSpec:
  """A KernelSpec for the kernel the provider described, bound to BoltBeam's own bytes. Every buffer the provider
  lists must be one of the operands at the size BoltBeam expects for this role, or the record is refused."""
  from boltbeam.collectors import metal_native as native
  args: list[Any] = []
  out_format = "f"
  x_bytes = struct.pack(f"<{cols}e", *x)
  for buf in record.get("buffers") or []:
    role, nbytes = buf.get("role"), int(buf.get("nbytes") or 0)
    if role == "weight":
      if nbytes != len(weights):
        raise ValueError(f"the provider's weight buffer is {nbytes} bytes; the role's {quant} weights are {len(weights)}")
      args.append(weights)
    elif role == "x":
      if nbytes != len(x_bytes):
        raise ValueError(f"the provider's activation buffer is {nbytes} bytes; {cols} fp16 values are {len(x_bytes)}")
      args.append(x_bytes)
    elif role == "out":
      fmt, size = DTYPES.get(str(buf.get("dtype")), (None, None))
      if fmt is None or nbytes != rows * size:
        raise ValueError(f"the provider's output buffer ({buf.get('dtype')}, {nbytes} bytes) is not {rows} values")
      out_format = fmt
      args.append(nbytes)
    else:
      raise ValueError(f"the provider listed a buffer of unknown role {role!r}")
  if sum(1 for a in args if isinstance(a, int)) != 1:
    raise ValueError("the provider's kernel must write exactly one output buffer")
  block = (list(record.get("local_size") or []) + [1, 1, 1])[:3]
  grid = (list(record.get("global_size") or []) + [1, 1, 1])[:3]
  check = Check(indices=native.check_rows(rows), reference=lambda r: native.reference_row(weights, quant, r, cols, x),
                rel_tol=native.TOLERANCE, words="pure-Python dequantize and dot, x rounded to fp16 as the kernel reads it",
                out_format=out_format)
  return KernelSpec(label=label, adapter="provider kernel, rebuilt by BoltBeam", source=str(record["source"]),
                    kernel=str(record["function"]), args=args, grid=tuple(int(v) for v in grid), block=tuple(int(v) for v in block),
                    bytes_read=len(weights) + len(x_bytes), check=check, record={"function": record["function"]})


def retime(bridge, flusher, floor_us:float | None, record:Mapping[str, Any], *, quant:str, rows:int, cols:int,
           weights:bytes, x:list[float], label:str, libraries:dict[str, int] | None = None) -> dict[str, Any]:
  """Correctness and time of the provider's kernel, by BoltBeam alone. The measured median keeps the floor; the
  judged figure is the median less the floor (kernel_timer.less_floor_us)."""
  xs = half_round(x)
  try:
    spec = spec_from_record(record, quant=quant, rows=rows, cols=cols, weights=weights, x=xs, label=label)
    got = kernel_timer.time_spec(bridge, spec, flusher, libraries=libraries, floor_us=floor_us)
  except Exception as exc:  # the provider's description or source: BoltBeam could not rebuild it
    return {"status": "not_rebuilt", "reason": f"{type(exc).__name__}: {exc}"[:400]}
  if not got["samples"]:
    return {"status": "incorrect", "correctness": got["correctness"],
            "reason": "BoltBeam's reference disagrees with the kernel's output"}
  med = got["median_us"]
  return {"status": "measured", "us": med, "us_less_floor": kernel_timer.less_floor_us(med, floor_us), "min_us": got["min_us"],
          "spread_pct": got["spread_pct"], "samples": len(got["samples"]), "correctness": got["correctness"],
          "timing": {**got["timing"], "dispatch_floor_us": floor_us}, "timed_by": f"BoltBeam's kernel timer: {label}"}


def judge(provider_ns:float | None, mine:Mapping[str, Any], band:float) -> dict[str, Any]:
  """The provider's median against BoltBeam's median less the dispatch floor (kernel_timer.less_floor_us), within the
  chip's band. The less-floor figure is the kernel's own time: the fork's device timestamps bracket the kernel, a
  CUDA event pair or a command buffer adds the launch (on the RTX 5090 the provider's emitter claims sat about one
  4 µs floor under BoltBeam's measured times). A BoltBeam reading the tie-out calls noisy (tie_out.noisy_words, the
  one rule) cannot confirm anything, so the claim is not reproduced: unknown is not pass."""
  from boltbeam.workflow.tie_out import noisy_words
  if mine.get("status") != "measured":
    return {"verdict": NOT_REPRODUCED, "reason": f"BoltBeam could not reproduce the kernel: {mine.get('reason')}"}
  ours = mine.get("us_less_floor") if mine.get("us_less_floor") is not None else mine["us"]
  if provider_ns is None or not ours:
    return {"verdict": NOT_REPRODUCED, "reason": "the provider reported no time to compare"}
  theirs = provider_ns / 1000.0
  ratio = theirs / ours
  noisy, _half, why = noisy_words(mine.get("spread_pct"), mine["us"], (mine.get("timing") or {}).get("dispatch_floor_us"))
  ok = abs(ratio - 1.0) <= band and not noisy
  reason = None
  if noisy:
    reason = f"BoltBeam's own reading of this kernel is noisy ({why}), so the provider's {theirs:.1f} µs cannot be confirmed"
  elif not ok:
    reason = (f"the provider's {theirs:.1f} µs is {ratio:.2f}x BoltBeam's {ours:.1f} µs (less the floor), "
              f"outside the chip's ±{100 * band:.1f}% band")
  return {"verdict": REPRODUCED if ok else NOT_REPRODUCED, "provider_us": theirs, "boltbeam_us": mine["us"],
          "boltbeam_us_less_floor": ours, "ratio": ratio, "band": band, "noisy": noisy, "reason": reason}


__all__ = ["NOT_REPRODUCED", "REPRODUCED", "bridge_facts", "half_round", "identity", "judge", "retime", "spec_from_record"]
