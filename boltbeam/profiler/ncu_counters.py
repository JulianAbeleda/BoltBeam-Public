"""Nsight Compute kernel counters: one collector for any target command, one importer, one evidence schema.

BoltBeam owns NCU measurement. A compiler only has to produce something ncu can see:
  * a program that launches its kernels through the CUDA driver (tinygrad: the DEV=CUDA proxy of its NV routes, or
    ``collectors/cubin_launch.py`` replaying a ``tinygrad.nv_cubin_capture.v1`` cubin + launch spec), or
  * a reference program (vLLM / cuBLAS ``F.linear``).
``ncu_command`` builds the one ncu command line (sections, cache and clock control, the memory-capped sudo scope --
a root profiler once OOM-crashed the dev box); ``export_raw_csv`` exports ``--page raw``; ``import_raw_csv`` parses it
into ``boltbeam.ncu_kernel_counters.v1`` rows labelled with the (role, M) each kernel belongs to, from the NVTX
push/pop range (``role/M``) or from the target's ``SHAPE role M kernels`` stdout lines, and marks each shape's GEMM
kernel.  Parsing is pure and CPU-testable.  ``to_hw_trace`` feeds the rows to the kernel-evidence adapters.

The column mapping was first written for the 2026-09-26 Nemotron kernel audit (tinygrad-arkey
docs/nemotron-vllm-parity/bench/audit/ncu_kernel_counters.py, schema ``tinygrad.nv_ncu_kernel_counters.v1``); that
schema is accepted by ``load_counters`` so the committed audit evidence imports unchanged.
"""
from __future__ import annotations

import csv, io, json, pathlib, re, subprocess
from typing import Any, Iterable, Mapping, Sequence

from boltbeam.vocab import SCHEMA_HW_TRACE

SCHEMA = "boltbeam.ncu_kernel_counters.v1"
LEGACY_SCHEMAS = ("tinygrad.nv_ncu_kernel_counters.v1",)
NCU_SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "WarpStateStats", "MemoryWorkloadAnalysis",
                "ComputeWorkloadAnalysis", "InstructionStats")
SIDES = ("ours", "reference")
_SIDE_ALIASES = {"tinygrad": "ours", "ours": "ours", "reference": "reference"}


# ---- collection ---------------------------------------------------------------------------------------------------

def ncu_command(target_cmd:Sequence[str], *, out:str, ncu:str="ncu", sections:Iterable[str]=NCU_SECTIONS,
                cache_control:str="all", clock_control:str="none", profile_from_start:str="off", nvtx:bool=False,
                kernel_name:str | None=None, launch_count:int | None=None, sudo_mem_max:str | None=None,
                env:Mapping[str, str] | None=None) -> list[str]:
  """The ncu argv for one target command; with ``sudo_mem_max`` wrapped in a memory-capped root scope."""
  cmd = [ncu, "--profile-from-start", profile_from_start, "--target-processes", "all"]
  for section in sections: cmd += ["--section", section]
  if nvtx: cmd.append("--nvtx")
  if kernel_name: cmd += ["--kernel-name", kernel_name]
  if launch_count is not None: cmd += ["--launch-count", str(launch_count)]
  cmd += ["--cache-control", cache_control, "--clock-control", clock_control, "-f", "-o", str(out), *target_cmd]
  env_args = [f"{k}={v}" for k, v in (env or {}).items()]
  if sudo_mem_max: return ["sudo", "systemd-run", "--scope", "-q", "-p", f"MemoryMax={sudo_mem_max}", "env", *env_args, *cmd]
  return (["env", *env_args] if env_args else []) + cmd


def export_raw_csv(report:str | pathlib.Path, *, ncu:str="ncu", instances:bool=False) -> str:
  """The raw page.  Scalars come from the plain export (instance exports round them); ``instances`` adds per-instance
  values, which carry the per-opcode SASS census (sass__inst_executed_per_opcode)."""
  extra = ["--print-metric-instances", "details"] if instances else []
  return subprocess.run([ncu, "--import", str(report), "--page", "raw", "--csv", *extra], check=True, capture_output=True, text=True).stdout


def collect(target_cmd:Sequence[str], *, out:str | pathlib.Path, side:str, ncu:str="ncu", sudo_mem_max:str | None="20G",
            env:Mapping[str, str] | None=None, nvtx:bool=False, owner:str | None="ubuntu:ubuntu") -> dict[str, Any]:
  """Profile ``target_cmd`` under ncu, export and import: returns the counters document (and writes the raw CSV and
  target stdout next to the report)."""
  out = pathlib.Path(out)
  cmd = ncu_command(target_cmd, out=str(out), ncu=ncu, nvtx=nvtx, sudo_mem_max=sudo_mem_max, env=env)
  proc = subprocess.run(cmd, capture_output=True, text=True)
  out.with_suffix(".stdout.log").write_text(proc.stdout)
  out.with_suffix(".ncu.log").write_text(proc.stderr)
  report = out.with_suffix(".ncu-rep") if out.suffix != ".ncu-rep" else out
  if sudo_mem_max and owner and report.exists(): subprocess.run(["sudo", "chown", owner, str(report)], check=False)
  if proc.returncode: raise RuntimeError(f"ncu exited {proc.returncode}; see {out.with_suffix('.ncu.log')}")
  raw, instances = export_raw_csv(report, ncu=ncu), export_raw_csv(report, ncu=ncu, instances=True)
  out.with_suffix(".raw.csv").write_text(raw)
  out.with_suffix(".instances.csv").write_text(instances)
  doc = import_raw_csv(raw, side=side, target_stdout=proc.stdout, source_report=str(report), instances_text=instances)
  doc["provenance"]["command"] = cmd
  return doc


# ---- import -------------------------------------------------------------------------------------------------------

_STALL_RE = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
_TIME_TO_US = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "msecond": 1e3, "ms": 1e3, "nsecond": 1e-3, "s": 1e6, "second": 1e6}
_BYTES_PER_SEC = {"byte/s": 1.0, "kbyte/s": 1e3, "mbyte/s": 1e6, "gbyte/s": 1e9, "tbyte/s": 1e12}
# ncu's byte prefixes are decimal: cuBLAS's 16896-byte arena reads "16.896 Kbyte/block".
_SHARED_BYTES = {"byte/block": 1.0, "kbyte/block": 1e3, "mbyte/block": 1e6}
_NVTX_COLUMN = "thread Domain:Push/Pop_Range"
_CENSUS = re.compile(r"([A-Z][A-Z0-9_]*):\s*([0-9]+)")
_NVTX_RANGE = re.compile(r":(?P<role>[A-Za-z_]\w*)/(?P<m>\d+):")


def _f(value:Any, default:float | None=0.0) -> float | None:
  """A scalar cell; an instance-exported cell ("158 (0: 158)") contributes its leading aggregate."""
  try: return float(str(value).split(" (", 1)[0].replace(",", ""))
  except (TypeError, ValueError): return default


def read_raw_csv(text:str) -> tuple[list[dict[str, str]], dict[str, str]]:
  """``ncu --page raw --csv``: header row, unit row, then one row per profiled kernel launch."""
  rows = list(csv.reader(io.StringIO(text)))
  if len(rows) < 2: raise ValueError("ncu raw CSV needs a header and a unit row")
  header = rows[0]
  return [dict(zip(header, r)) for r in rows[2:] if r], dict(zip(header, rows[1]))


def _scaled(raw:Mapping[str, str], units:Mapping[str, str], column:str, table:Mapping[str, float], what:str) -> float:
  value = _f(raw.get(column))
  unit = (units.get(column) or "").strip().lower()
  if value is None or column not in raw: return 0.0
  if unit not in table: raise ValueError(f"unknown {what} unit {unit!r} for {column}")
  return value * table[unit]


def top_stall_reasons(raw:Mapping[str, str], top:int=3) -> list[dict[str, Any]]:
  """Each stall reason's share (%) of the stalls this kernel reports, largest first."""
  stalls = sorted(((_f(v), m.group(1)) for c, v in raw.items() if (m := _STALL_RE.match(c)) and v not in ("", None)
                   and "selected" not in c), reverse=True)
  total = sum(v for v, _ in stalls) or 1.0
  return [{"reason": name, "pct": round(value / total * 100.0, 3)} for value, name in stalls[:top]]


def parse_kernel_row(raw:Mapping[str, str], units:Mapping[str, str]) -> dict[str, Any]:
  nvtx = next((v for k, v in raw.items() if k.startswith(_NVTX_COLUMN)), "")
  row = {"kernel": raw.get("Kernel Name"), "grid": raw.get("Grid Size"), "block": raw.get("Block Size"),
         "registers_per_thread": int(_f(raw.get("launch__registers_per_thread")) or 0),
         "static_shared_bytes": _scaled(raw, units, "launch__shared_mem_per_block_static", _SHARED_BYTES, "shared memory"),
         "dynamic_shared_bytes": _scaled(raw, units, "launch__shared_mem_per_block_dynamic", _SHARED_BYTES, "shared memory"),
         "duration_us": _scaled(raw, units, "gpu__time_duration.sum", _TIME_TO_US, "time"),
         "achieved_occupancy_pct": _f(raw.get("sm__warps_active.avg.pct_of_peak_sustained_active")),
         "tensor_pipe_util_pct": _f(raw.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")),
         "dram_throughput_bytes_per_sec": _scaled(raw, units, "dram__bytes.sum.per_second", _BYTES_PER_SEC, "throughput"),
         "sm_throughput_pct": _f(raw.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
         "top_stall_reasons": top_stall_reasons(raw)}
  conflicts = {k: _f(v) for k, v in raw.items() if "bank_conflict" in k.lower() and _f(v, None) is not None}
  if conflicts: row["bank_conflicts"] = conflicts
  census = {op: int(n) for op, n in _CENSUS.findall(raw.get("sass__inst_executed_per_opcode") or "")}
  if census: row["opcode_census"] = census
  if (m := _NVTX_RANGE.search(nvtx)): row["_nvtx"] = (m["role"], int(m["m"]))
  return row


def mark_gemm(group:list[dict[str, Any]]) -> None:
  """In one shape's launches, the longest kernel that uses the tensor pipe (not a split-K reduction) is the GEMM.
  Every launch of that same kernel is GEMM too: a row-chunked op (ours prefill runs a 1024-token piece as 4 launches
  of the 512-row kernel) is one GEMM made of several launches, not one GEMM plus aux."""
  if not group: return
  cands = [r for r in group if "splitKreduce" not in (r["kernel"] or "") and (r["tensor_pipe_util_pct"] or 0) > 0.5]
  gemm = max(cands or group, key=lambda r: r["duration_us"])
  for r in group: r["aux"] = r["kernel"] != gemm["kernel"]


def _shape_lines(stdout:str) -> list[tuple[str, int, int]]:
  return [(p[1], int(p[2]), int(p[3])) for line in stdout.splitlines() if (p := line.split()) and p[0] == "SHAPE" and len(p) == 4]


def import_raw_csv(text:str, *, side:str, target_stdout:str | None=None, source_report:str | None=None,
                   provenance:Mapping[str, Any] | None=None, instances_text:str | None=None) -> dict[str, Any]:
  """Raw CSV -> counters document.  Launches are labelled by NVTX range when present, else by the target's
  ``SHAPE role M kernels`` lines (launch order); a mismatch between the two counts is an error."""
  side = _SIDE_ALIASES.get(side, side)
  if side not in SIDES: raise ValueError(f"side must be one of {SIDES}")
  raws, units = read_raw_csv(text)
  if instances_text is not None:   # the per-opcode census, joined by launch ID
    census_by_id = {r.get("ID"): r.get("sass__inst_executed_per_opcode") for r in read_raw_csv(instances_text)[0]}
    raws = [{**r, "sass__inst_executed_per_opcode": census_by_id.get(r.get("ID")) or r.get("sass__inst_executed_per_opcode", "")} for r in raws]
  rows = [parse_kernel_row(r, units) for r in raws]
  groups: list[list[dict[str, Any]]] = []
  if rows and all("_nvtx" in r for r in rows):
    for r in rows:
      role, m = r.pop("_nvtx")
      if not groups or (groups[-1][0]["role"], groups[-1][0]["m"]) != (role, m): groups.append([])
      groups[-1].append({"side": side, "role": role, "m": m, **r})
  elif target_stdout is not None:
    i = 0
    for role, m, count in _shape_lines(target_stdout):
      groups.append([{"side": side, "role": role, "m": m, **{k: v for k, v in r.items() if k != "_nvtx"}} for r in rows[i:i + count]])
      i += count
    if i != len(rows): raise ValueError(f"SHAPE lines cover {i} launches, the report has {len(rows)}")
  else:
    groups = [[{"side": side, "role": None, "m": None, **{k: v for k, v in r.items() if k != "_nvtx"}}] for r in rows]
  for g in groups: mark_gemm(g)
  first = raws[0] if raws else {}
  prov = {"device_name": first.get("device__attribute_display_name"), "compute_capability": first.get("CC"),
          "cache_control": "all", "clock_control": "none", "source_report": source_report, **dict(provenance or {})}
  return {"schema": SCHEMA, "provenance": prov, "rows": [r for g in groups for r in g]}


def load_counters(doc:Mapping[str, Any]) -> dict[str, Any]:
  """A counters document in this schema, or the audit's legacy tinygrad one (``side`` tinygrad -> ours,
  ``shape`` -> ``m``)."""
  schema = doc.get("schema")
  if schema == SCHEMA: return dict(doc)
  if schema not in LEGACY_SCHEMAS: raise ValueError(f"not an ncu counters document: {schema!r}")
  # That importer scaled shared memory by assumed units (static Kbyte, dynamic byte) where ncu reported static in byte
  # and dynamic in Kbyte; those two fields are dropped rather than carried wrong -- re-import the raw CSV for them.
  rows = []
  for r in doc["rows"]:
    r = dict(r); r["side"] = _SIDE_ALIASES[r["side"]]; r["m"] = r.pop("shape")
    r["static_shared_bytes"] = r["dynamic_shared_bytes"] = None
    rows.append(r)
  return {"schema": SCHEMA, "provenance": {**doc.get("provenance", {}), "imported_from": schema,
          "dropped": ["static_shared_bytes", "dynamic_shared_bytes"]}, "rows": rows}


def to_hw_trace(doc:Mapping[str, Any], *, model_id:str, target_id:str, workload:str, provider_id:str) -> dict[str, Any]:
  """Counters -> ``boltbeam.hw_trace.v1`` kernel rows (so the kernel-evidence adapters can read them)."""
  rows = []
  for r in load_counters(doc)["rows"]:
    counters = {k: float(r[k]) for k in ("achieved_occupancy_pct", "tensor_pipe_util_pct", "sm_throughput_pct") if r.get(k) is not None}
    counters["dram_bytes_per_sec"] = float(r["dram_throughput_bytes_per_sec"])
    counters.update({f"stall_{s['reason']}_pct": float(s["pct"]) for s in r.get("top_stall_reasons", ())})
    rows.append({"scope": "kernel", "kernel": r["kernel"], "role": r.get("role") or "unknown", "wall_us": float(r["duration_us"]),
                 "calls": 1, "kind": "gemm" if not r.get("aux") else "aux", "counters": counters,
                 "resources": {"registers": r["registers_per_thread"],
                               **({"lds_bytes": r["static_shared_bytes"] + r["dynamic_shared_bytes"]} if r.get("static_shared_bytes") is not None else {})},
                 "sources": {"side": r["side"], "m": r.get("m"), "grid": r.get("grid"), "block": r.get("block")}})
  return {"schema": SCHEMA_HW_TRACE, "model_id": model_id, "target_id": target_id, "workload": workload,
          "provider_id": provider_id, "source_schema": SCHEMA, "trace_source": "ncu_raw_csv", "rows": rows}


__all__ = ["LEGACY_SCHEMAS", "NCU_SECTIONS", "SCHEMA", "collect", "export_raw_csv", "import_raw_csv", "load_counters",
           "mark_gemm", "ncu_command", "parse_kernel_row", "read_raw_csv", "to_hw_trace", "top_stall_reasons"]
