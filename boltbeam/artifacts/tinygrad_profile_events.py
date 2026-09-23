from __future__ import annotations

import pathlib
import re
import pickle
import sys
from typing import Any

from boltbeam.core.canonical import sha256_hex

from boltbeam.artifacts.tinygrad_rocprof import (
  _apply_role_bytes, _apply_weight_bytes, _shape_index, _short_name, classify_tinygrad_kernel,
)
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


_WHOLE_PREFILL_RE = re.compile(r"^\s*WHOLE-PREFILL@(?P<context>\d+):\s*(?P<tok_s>[0-9]+(?:\.[0-9]+)?)\s+tok/s\s*$", re.MULTILINE)


def reconcile_authority_wall(trace:dict[str, Any], stdout:str, *, context:int) -> dict[str, Any]:
  """Replace PROFILE event-sum wall with the synchronized authority wall."""
  match = next((m for m in _WHOLE_PREFILL_RE.finditer(stdout) if int(m["context"]) == context), None)
  if match is None:
    raise ValueError(f"authority output is missing WHOLE-PREFILL@{context} tok/s")
  tok_s = float(match["tok_s"])
  if tok_s <= 0:
    raise ValueError("authority whole-prefill tok/s must be positive")
  rows = trace.get("rows", [])
  kernels = [row for row in rows if row.get("scope") == "kernel"]
  raw_sum = sum(float(row.get("wall_us", 0.0)) for row in kernels)
  if raw_sum <= 0:
    raise ValueError("cannot reconcile authority wall without profile kernel timing")
  authority_us = 1_000_000.0 * context / tok_s
  scale = authority_us / raw_sum
  for row in kernels:
    row["wall_us"] = float(row.get("wall_us", 0.0)) * scale
  whole = next((row for row in rows if row.get("scope") == "whole_step"), None)
  if whole is None:
    raise ValueError("profile trace is missing whole_step row")
  whole["raw_wall_us"] = whole.get("wall_us", raw_sum)
  whole["wall_us"] = authority_us
  whole["tok_s"] = tok_s
  whole["tok_s_source"] = "synchronized_authority_stdout"
  trace["timing_source"] = "profile_events_reconciled_authority"
  trace.setdefault("aux_sources", {})["authority_wall"] = {
    "context": context, "tok_s": tok_s, "wall_us": authority_us,
    "scale": scale, "raw_profile_kernel_wall_us": raw_sum,
    "source": "bench.py stdout WHOLE-PREFILL line",
  }
  return trace


def load_profile_events(path:str | pathlib.Path, *, tinygrad_root:str | pathlib.Path | None = None) -> list[Any]:
  if tinygrad_root:
    root = str(pathlib.Path(tinygrad_root).expanduser())
    if root not in sys.path:
      sys.path.insert(0, root)
  with pathlib.Path(path).expanduser().open("rb") as f:
    return pickle.load(f)


def _event_type(event:Any) -> str:
  return type(event).__name__


def _name(value:Any) -> str:
  named = getattr(value, "name", None)
  if isinstance(named, str) and named:
    return named
  return str(value)


def _range_delta_us(event:Any) -> float | None:
  try:
    return max(0.0, float(getattr(event, "en")) - float(getattr(event, "st")))
  except (TypeError, ValueError):
    return None


def _graph_delta_us(graph:Any, entry:Any) -> float | None:
  try:
    sigs = getattr(graph, "sigs")
    return max(0.0, float(sigs[int(getattr(entry, "en_id"))]) - float(sigs[int(getattr(entry, "st_id"))]))
  except (IndexError, TypeError, ValueError):
    return None


def _decode_rsrc1_counts(rsrc1:int) -> dict[str, int]:
  vgpr_gran = rsrc1 & 0x3f
  sgpr_gran = (rsrc1 >> 6) & 0xf
  return {
    "vgpr": (vgpr_gran + 1) * 8 - 7,
    "sgpr": (sgpr_gran + 1) * 16,
  }


def _program_resources_from_lib(lib:bytes | None) -> dict[str, Any]:
  if not lib:
    return {}
  try:
    import ctypes
    from tinygrad.runtime.support.elf import elf_loader
    from tinygrad.runtime.autogen import amdgpu_kd
    image, sections, _relocs = elf_loader(lib)
    rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
    if rodata_entry < 0:
      return {}
    desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
    rsrc1, rsrc2, rsrc3 = int(desc.compute_pgm_rsrc1), int(desc.compute_pgm_rsrc2), int(desc.compute_pgm_rsrc3)
    resources: dict[str, Any] = {
      "lds_bytes": int(desc.group_segment_fixed_size),
      "scratch_bytes": int(desc.private_segment_fixed_size),
      "kernarg_size": int(desc.kernarg_size),
      "rsrc1": rsrc1,
      "rsrc2": rsrc2,
      "rsrc3": rsrc3,
    }
    resources.update(_decode_rsrc1_counts(rsrc1))
    return resources
  except Exception:
    return {}


def _program_rows(events:list[Any]) -> dict[Any, dict[str, Any]]:
  rows = {}
  for event in events:
    if _event_type(event) != "ProfileProgramEvent":
      continue
    tag = getattr(event, "tag", None)
    lib = getattr(event, "lib", None)
    resources = _program_resources_from_lib(lib)
    rows[tag] = {
      "device": getattr(event, "device", None),
      "kernel": _short_name(_name(getattr(event, "name", ""))),
      "base": getattr(event, "base", None),
      "tag": tag,
    }
    if resources:
      rows[tag]["resources"] = resources
    if lib:
      rows[tag]["sources"] = {"lib_sha256": "sha256:" + sha256_hex(lib)}
  return rows


def _launch_resources(events:list[Any], programs:dict[Any, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  by_kernel: dict[str, dict[str, Any]] = {}
  for event in events:
    if _event_type(event) != "ProfilePointEvent" or getattr(event, "name", None) != "launch":
      continue
    prog = programs.get(getattr(event, "key", None))
    if not prog or not prog.get("kernel"):
      continue
    arg = getattr(event, "arg", None) or {}
    try:
      global_size = [int(x) for x in arg.get("global_size", [])]
      local_size = [int(x) for x in arg.get("local_size", [])]
    except (TypeError, ValueError):
      continue
    if not global_size or not local_size:
      continue
    res: dict[str, Any] = {"global_size": global_size, "local_size": local_size}
    threads = 1
    for x in local_size:
      threads *= x
    res["workgroup_threads"] = threads
    by_kernel[str(prog["kernel"])] = res
  return by_kernel


def _freeze(value:Any) -> Any:
  if isinstance(value, dict):
    return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
  if isinstance(value, list):
    return tuple(_freeze(v) for v in value)
  return value


def _range_key(row:dict[str, Any]) -> tuple[Any, ...]:
  return (
    row.get("context"), row.get("kernel"), row.get("kind"), row.get("role"), row.get("quant"),
    tuple(row.get("shape") or ()), _freeze(row.get("resources")),
  )


def decode_profile_events(events:list[Any], *, model_id:str, target_id:str = "amd_gfx1100",
                          workload:str = "prefill", context:int | None = None,
                          provider_id:str = "tinygrad/profile-events", peak_gbs:float = DEFAULT_PEAK_MEM_GBS,
                          weight_inventory:str | pathlib.Path | None = None,
                          source_path:str | pathlib.Path | None = None) -> dict[str, Any]:
  if not model_id:
    raise ValueError("model_id is required; pass --model-id or --run")
  shape_index = _shape_index(weight_inventory)
  programs = _program_rows(events)
  programs_by_kernel = {str(v["kernel"]): v for v in programs.values() if v.get("kernel")}
  launch_by_kernel = _launch_resources(events, programs)
  grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
  raw_events = 0
  range_kernel_events = 0
  graph_entry_events = 0

  def add_kernel(kernel:str, delta_us:float | None, *, device:Any = None) -> None:
    nonlocal raw_events
    if delta_us is None:
      return
    if device is not None and not str(device).startswith("AMD"):
      return
    raw_events += 1
    info = classify_tinygrad_kernel(kernel, shape_index=shape_index)
    row = {
      "scope": "kernel",
      "kernel": kernel,
      "kind": info["kind"],
      "role": info["role"],
      "wall_us": delta_us,
      "raw_wall_us": delta_us,
      "calls": 1,
      "time_source": "tinygrad_profile_events",
    }
    if device is not None:
      row["device"] = str(device)
    if context is not None:
      row["context"] = context
    if info.get("quant"):
      row["quant"] = info["quant"]
    if info.get("shape"):
      row["shape"] = info["shape"]
    prog = programs_by_kernel.get(kernel)
    if prog and prog.get("resources"):
      row["resources"] = dict(prog["resources"])
    if kernel in launch_by_kernel:
      row.setdefault("resources", {}).update(launch_by_kernel[kernel])
    if prog and prog.get("sources"):
      row["sources"] = dict(prog["sources"])
    key = _range_key(row)
    if key in grouped:
      grouped[key]["wall_us"] += row["wall_us"]
      grouped[key]["raw_wall_us"] += row["raw_wall_us"]
      grouped[key]["calls"] += 1
    else:
      grouped[key] = row

  range_events = []
  graph_entries = []
  for event in events:
    typ = _event_type(event)
    if typ == "ProfileRangeEvent":
      raw_name = getattr(event, "name", "")
      # Device-copy/profile scopes can be attributed to AMD while carrying a TracingKey object. They are not kernels
      # and can dwarf actual device time, so accept only concrete program names from range events.
      if isinstance(raw_name, str) and str(getattr(event, "device", "")).startswith("AMD"):
        range_events.append(event)
        range_kernel_events += 1
    elif typ == "ProfileGraphEvent":
      for entry in getattr(event, "ents", []):
        if str(getattr(entry, "device", "")).startswith("AMD"):
          graph_entries.append((event, entry))
          graph_entry_events += 1

  # A profiled TinyJit capture contains two views of one lifecycle: eager ProfileRangeEvents from capture/warmup and
  # ProfileGraphEvents from the captured graph replay. Summing both counts the model step twice. Prefer graph entries
  # when present because they describe the steady captured route; range events remain the fallback for eager workloads.
  dispatch_source = "graph_entries" if graph_entry_events else "range_events"
  if graph_entry_events:
    for event, entry in graph_entries:
      add_kernel(
        _short_name(_name(getattr(entry, "name", ""))),
        _graph_delta_us(event, entry),
        device=getattr(entry, "device", None),
      )
  else:
    for event in range_events:
      raw_name = getattr(event, "name", "")
      kernel = _short_name(raw_name)
      prog = programs.get(getattr(event, "tag", None))
      if prog and prog.get("kernel"):
        kernel = prog["kernel"]
      add_kernel(kernel, _range_delta_us(event), device=getattr(event, "device", None))

  rows = list(grouped.values())
  rows.sort(key=lambda r: float(r.get("wall_us", 0.0)), reverse=True)
  total_us = sum(float(r.get("wall_us", 0.0)) for r in rows)
  total_bytes = _apply_role_bytes(rows, shape_index)
  bytes_source = "estimated_weight_inventory_by_shape_time_weighted" if total_bytes is not None else None
  if total_bytes is None:
    total_bytes = _apply_weight_bytes(rows, {})
    bytes_source = "estimated_from_kernel_roles" if total_bytes is not None else None

  whole = {
    "scope": "whole_step",
    "wall_us": total_us,
    "launch_count": sum(int(r.get("calls", 1)) for r in rows),
    "time_source": "tinygrad_profile_event_sum",
  }
  if context is not None:
    whole["context"] = context
    if total_us > 0:
      whole["tok_s"] = 1_000_000.0 * context / total_us
      whole["tok_s_source"] = "tinygrad_profile_event_sum"
  if total_bytes is not None:
    whole["total_bytes"] = total_bytes
  if bytes_source:
    whole["bytes_source"] = bytes_source

  semantic_roles = {"attn_qo", "attn_kv", "ffn_down", "ffn_gate_up", "lm_head", "attention_tile", "attention_combine"}
  role_rows = [r for r in rows if r.get("role") in semantic_roles]
  attribution = {
    "status": "attributable" if raw_events > 0 and role_rows else "non_attributive",
    "usable_for_aggregate": bool(total_us > 0),
    "reason": None if raw_events > 0 and role_rows else (
      "no_profile_kernel_events" if raw_events == 0 else "no_per_role_kernel_rows"),
    "profile_enabled": True,
    "profile_kernel_row_count": len(rows),
    "per_role_kernel_row_count": len(role_rows),
  }

  return {
    "schema": SCHEMA_TIMING_TRACE,
    "model_id": model_id,
    "target_id": target_id,
    "workload": workload,
    "provider_id": provider_id,
    "timing_source": "profile_events",
    "peak_gbs": peak_gbs,
    "contexts": [context] if context is not None else [],
    "source_path": str(source_path) if source_path else None,
    "aux_sources": {
      "weight_inventory": str(weight_inventory) if weight_inventory else None,
      "profile_event_count": len(events),
      "profile_range_event_count": range_kernel_events,
      "profile_graph_entry_count": graph_entry_events,
      "profile_dispatch_source": dispatch_source,
      "profile_selected_kernel_event_count": raw_events,
      "profile_program_event_count": len(programs),
      "attribution": attribution,
    },
    "notes": [
      "Generated from tinygrad PROFILE profile.pkl events without an external backend profiler.",
      "Kernel rows are aggregated by kernel/role/shape; exact hardware counters require a lower-level sampler.",
      "Exact tensor role attribution is shape-derived from BoltBeam weight inventory where possible.",
    ],
    "rows": [whole] + rows,
  }


def timing_trace_from_tinygrad_profile_events(path:str | pathlib.Path, *, model_id:str,
                                              target_id:str = "amd_gfx1100", workload:str = "prefill",
                                              context:int | None = None,
                                              provider_id:str = "tinygrad/profile-events",
                                              peak_gbs:float = DEFAULT_PEAK_MEM_GBS,
                                              tinygrad_root:str | pathlib.Path | None = None,
                                              weight_inventory:str | pathlib.Path | None = None) -> dict[str, Any]:
  src = pathlib.Path(path).expanduser()
  events = load_profile_events(src, tinygrad_root=tinygrad_root)
  return decode_profile_events(
    events,
    model_id=model_id,
    target_id=target_id,
    workload=workload,
    context=context,
    provider_id=provider_id,
    peak_gbs=peak_gbs,
    weight_inventory=weight_inventory,
    source_path=src,
  )
