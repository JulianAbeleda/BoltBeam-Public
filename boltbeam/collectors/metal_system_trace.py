"""Provider-neutral Apple Metal System Trace capture and normalization."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from boltbeam.target.tinygrad_root import resolve_tinygrad_root
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, run_dir


GPU_INTERVALS = "metal-gpu-intervals"
COMMAND_BUFFERS = "metal-application-command-buffer-submissions"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _provider_id(value:str) -> str:
  return "llama.cpp" if value in {"llama", "llama.cpp", "llama-cpp"} else value


def _configured_run(args:argparse.Namespace) -> tuple[dict[str, Any], pathlib.Path, str, str, str, int]:
  manifest = load_manifest(run_dir(args.run)) if args.run else {}
  out = pathlib.Path(args.out).expanduser() if args.out else None
  if out is None: raise ValueError("--out is required unless --run supplies one")
  model = args.model or manifest.get("model_path")
  if not model: raise ValueError("--model is required unless --run supplies one")
  model_id = args.model_id or manifest.get("model_id") or pathlib.Path(model).stem
  target_id = args.target_id or manifest.get("target_id")
  if not target_id: raise ValueError("--target-id is required unless --run supplies one")
  workload = args.workload or manifest.get("workload") or "decode"
  return manifest, out, str(model), str(model_id), str(target_id), int(args.context)


def _executable(value:str, *, cwd:pathlib.Path | None = None) -> str:
  path = pathlib.Path(value).expanduser()
  if not path.is_absolute() and cwd is not None and (cwd / path).exists(): path = cwd / path
  found = shutil.which(str(path))
  if found: return found
  if path.exists(): return str(path.absolute())
  raise FileNotFoundError(f"runtime executable not found: {value}")


def _runtime_command(args:argparse.Namespace, model:str, context:int, workload:str) -> tuple[list[str], pathlib.Path | None]:
  if workload != "decode":
    raise ValueError("metal-system-trace currently supports fixed-depth decode; prefill requires a matched runtime adapter")
  provider = _provider_id(args.provider)
  if provider == "llama.cpp":
    return [_executable(args.llama_bench), "-m", model, "-p", "0", "-n", "1", "-d", str(context),
            "-ngl", "99", "-r", "1", "-o", "json"], None
  if provider == "tinygrad":
    root = resolve_tinygrad_root(args.tinygrad_root, require_checkout=True)
    python = _executable(args.python, cwd=root)
    return [python, "-m", "tinygrad.llm.cli", "--model", model, "--max_context", str(context + 4),
            "--stream", "off", "--warmup", "--benchmark", "1", "--benchmark-context", str(context)], root
  raise ValueError(f"metal-system-trace runtime adapter is unavailable for provider {provider!r}")


def _env_overrides(raw:list[str] | None) -> list[str]:
  out = []
  for item in raw or []:
    if "=" not in item: raise ValueError(f"invalid env override {item!r}; expected KEY=VALUE")
    out.extend(["--env", item])
  return out


def _export_table(xcrun:str, trace:pathlib.Path, schema:str, out:pathlib.Path) -> None:
  query = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
  proc = subprocess.run([xcrun, "xctrace", "export", "--input", str(trace), "--xpath", query, "--output", str(out)],
                        capture_output=True, text=True)
  if proc.returncode != 0: raise RuntimeError(f"xctrace export {schema} failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _xml_rows(path:str | pathlib.Path) -> list[dict[str, dict[str, Any]]]:
  root = ET.parse(path).getroot()
  ids = {elem.attrib["id"]: elem for elem in root.iter() if "id" in elem.attrib}
  schema = next(root.iter("schema"), None)
  if schema is None: return []
  columns = [next(col.iter("mnemonic")).text for col in schema.findall("col")]

  def value(elem:ET.Element) -> dict[str, Any]:
    while "ref" in elem.attrib: elem = ids[elem.attrib["ref"]]
    raw = "".join(elem.itertext()).strip()
    return {"raw": raw, "formatted": elem.attrib.get("fmt", raw), "type": elem.tag}

  return [{name: value(elem) for name, elem in zip(columns, list(row))}
          for row in root.iter("row")]


def _number(cell:dict[str, Any] | None) -> int | None:
  if not cell: return None
  try: return int(cell["raw"].replace(",", ""))
  except (KeyError, TypeError, ValueError): return None


def _text(cell:dict[str, Any] | None) -> str | None:
  return str(cell.get("formatted")) if cell else None


def _interval_union_us(rows:list[dict[str, Any]]) -> float:
  intervals = sorted((float(row["start_us"]), float(row["start_us"]) + float(row["wall_us"])) for row in rows)
  merged: list[list[float]] = []
  for start, end in intervals:
    if not merged or start > merged[-1][1]: merged.append([start, end])
    else: merged[-1][1] = max(merged[-1][1], end)
  return sum(end - start for start, end in merged)


def _bind_measurement(command_rows:list[dict[str, Any]], measured:dict[str, Any], *,
                      trailing_idle_us:float = 2_000.0) -> dict[str, Any]:
  """Bind runtime-owned measurement evidence to the matching command-buffer interval set."""
  chronological = sorted(command_rows, key=lambda row:(float(row["start_us"]), str(row.get("command_buffer_id"))))
  expected = measured.get("runtime_expected_command_buffers")
  if expected is not None:
    count = int(expected)
    selected = chronological[-count:] if 0 < count <= len(chronological) else []
    rule = "runtime_reported_trailing_command_buffer_count"
  else:
    start = len(chronological) - 1
    while start > 0:
      previous_end = float(chronological[start - 1]["start_us"]) + float(chronological[start - 1]["wall_us"])
      if float(chronological[start]["start_us"]) - previous_end > trailing_idle_us: break
      start -= 1
    selected = chronological[start:] if chronological else []
    rule = "trailing_activity_cluster"
  selected_ids = {id(row) for row in selected}
  for row in command_rows:
    if id(row) in selected_ids:
      row["phase"] = "measured_decode_candidate"
      row["measurement_selected"] = True
  if not selected:
    return {"measurement_binding_status": "unbound", "measurement_binding_rule": rule,
            "selected_command_buffer_count": 0}
  first = min(float(row["start_us"]) for row in selected)
  last = max(float(row["start_us"]) + float(row["wall_us"]) for row in selected)
  union = _interval_union_us(selected)
  wall = float(measured["wall_us"])
  relative_gap = abs(union - wall) / wall if wall else None
  return {
    "measurement_binding_status": "bound" if relative_gap is not None and relative_gap <= 0.35 else "diagnostic_only",
    "measurement_binding_rule": rule,
    "selected_command_buffer_count": len(selected),
    "selected_gpu_union_us": union,
    "selected_gpu_span_us": last - first,
    "selected_gpu_sum_us": sum(float(row["wall_us"]) for row in selected),
    "selected_gpu_vs_wall_relative_gap": relative_gap,
  }


def normalize_metal_trace(gpu_xml:str | pathlib.Path, submissions_xml:str | pathlib.Path, *,
                          stdout:str, model_id:str, target_id:str, workload:str, provider_id:str,
                          context:int, trace_path:str | pathlib.Path,
                          measured:Mapping[str, Any] | None = None) -> dict[str, Any]:
  submissions = _xml_rows(submissions_xml)
  by_cb = {_text(row.get("cmdbuffer-id")): row for row in submissions if _text(row.get("cmdbuffer-id"))}
  gpu = [row for row in _xml_rows(gpu_xml) if _text(row.get("cmdbuffer-id")) in by_cb]
  command_rows = []
  for row in gpu:
    cb = _text(row.get("cmdbuffer-id"))
    app = by_cb[cb]
    duration_ns, start_ns = _number(row.get("duration")), _number(row.get("start"))
    if duration_ns is None or start_ns is None: continue
    label = _text(row.get("event-label")) or _text(row.get("channel-name")) or "Metal command buffer"
    command_rows.append({
      "scope": "command_buffer", "context": context, "phase": "unclassified",
      "command_buffer_id": cb, "frame": _text(row.get("frame-number")), "label": label,
      "channel": _text(row.get("channel-name")), "wall_us": duration_ns / 1000.0,
      "start_us": start_ns / 1000.0,
      "cpu_to_gpu_latency_us": (_number(row.get("start-latency")) or 0) / 1000.0,
      "encoder_count": _number(app.get("num-encoders")),
      "encoder_cpu_us": (_number(app.get("encoder-time")) or 0) / 1000.0,
      "process": _text(row.get("process")),
    })
  measurement = dict(measured) if measured is not None else _runtime_measurement(stdout, provider_id)
  binding = _bind_measurement(command_rows, measurement)
  whole = {"scope": "whole_step", "context": context, "phase": "measured_decode",
           "command_buffer_count": len(command_rows), **measurement, **binding}
  return {
    "schema": SCHEMA_TIMING_TRACE, "model_id": model_id, "target_id": target_id,
    "workload": workload, "provider_id": provider_id,
    "timing_source": "xctrace_metal_system_trace+runtime_stdout", "contexts": [context],
    "rows": [whole, *command_rows],
    "aux_sources": {
      "metal_trace": str(trace_path),
      "timing_scope": {
        "whole_step": "runtime benchmark wall time",
        "command_buffer": "Metal System Trace GPU active intervals",
        "per_dispatch": "unavailable",
      },
      "stdout": stdout,
    },
  }


def capture_command_metal_trace(*, command:Sequence[str], cwd:str | pathlib.Path | None,
                                environ:Mapping[str, str], trace_dir:str | pathlib.Path,
                                model_id:str, target_id:str, context:int,
                                measurement_loader:Callable[[], Mapping[str, Any]],
                                xcrun:str = "xcrun", time_limit_s:int = 1800) -> tuple[int, str, str, dict[str, Any] | None]:
  """Capture one explicit runtime command through the existing Metal trace authority."""
  root = pathlib.Path(trace_dir); root.mkdir(parents=True, exist_ok=True)
  trace, stdout_path, stderr_path = root / "capture.trace", root / "target.stdout.txt", root / "target.stderr.txt"
  gpu_xml, submissions_xml = root / "gpu.xml", root / "submissions.xml"
  executable = _executable(xcrun)
  forwarded = {key:str(environ[key]) for key in ("DEV", "PYTHONPATH", "PYTHONHASHSEED", "METAL_HYBRID_REPLAY") if key in environ}
  env_args = [part for key,value in sorted(forwarded.items()) for part in ("--env", f"{key}={value}")]
  # xctrace record redirects only stdin/stdout; it has no --target-stderr. An unredirected target
  # inherits xctrace's stderr, so the launched process's stderr arrives through proc.stderr.
  record = [executable, "xctrace", "record", "--template", "Metal System Trace", "--output", str(trace),
            "--time-limit", f"{time_limit_s}s", "--no-prompt", "--target-stdout", str(stdout_path),
            *env_args, "--launch", "--", *map(str, command)]
  proc = subprocess.run(record, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
  stdout = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
  stderr = stderr_path.read_text(errors="replace") if stderr_path.is_file() else proc.stderr
  if proc.returncode != 0: return proc.returncode, stdout, stderr or proc.stderr, None
  _export_table(executable, trace, GPU_INTERVALS, gpu_xml)
  _export_table(executable, trace, COMMAND_BUFFERS, submissions_xml)
  measured = dict(measurement_loader())
  normalized = normalize_metal_trace(gpu_xml, submissions_xml, stdout=stdout, model_id=model_id,
    target_id=target_id, workload="decode", provider_id="tinygrad", context=context,
    trace_path=trace, measured=measured)
  normalized["aux_sources"].update({"capture_command":list(command), "capture_record_command":record,
    "target_stderr":stderr, "capture_artifacts":{"trace":str(trace), "gpu_xml":str(gpu_xml),
      "submissions_xml":str(submissions_xml), "stdout":str(stdout_path), "stderr":str(stderr_path)}})
  return 0, stdout, stderr, normalized


def _runtime_measurement(stdout:str, provider_id:str) -> dict[str, Any]:
  clean = _ANSI.sub("", stdout)
  if provider_id == "llama.cpp":
    start, end = clean.find("["), clean.rfind("]")
    if start >= 0 and end > start:
      value = json.loads(clean[start:end+1])
      row = value[0] if isinstance(value, list) and value else {}
      if row.get("avg_ns") and row.get("avg_ts"):
        return {"wall_us": float(row["avg_ns"]) / 1000.0, "tok_s": float(row["avg_ts"]),
                "samples_us": [float(x) / 1000.0 for x in row.get("samples_ns", [])]}
  else:
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*ms,\s*([0-9]+(?:\.[0-9]+)?)\s*tok/s", clean)
    if matches:
      ms, tok_s = matches[-1]
      calls = re.findall(r"jit execs\s+(\d+)\s+calls", clean, flags=re.IGNORECASE)
      return {"wall_us": float(ms) * 1000.0, "tok_s": float(tok_s), "samples_us": [float(ms) * 1000.0],
              **({"runtime_expected_command_buffers": int(calls[-1])} if calls else {})}
  raise ValueError(f"unable to parse {provider_id} fixed-depth benchmark output")


def capture_metal_system_trace(args:argparse.Namespace) -> tuple[dict[str, Any], pathlib.Path]:
  manifest, out, model, model_id, target_id, context = _configured_run(args)
  workload = args.workload or manifest.get("workload") or "decode"
  provider = _provider_id(args.provider)
  command, cwd = _runtime_command(args, model, context, workload)
  trace_dir = (pathlib.Path(args.trace_dir).expanduser() if args.trace_dir else out.with_suffix(".metal.d")).absolute()
  trace_dir.mkdir(parents=True, exist_ok=True)
  stem = f"{provider.replace('.', '-')}-{model_id}-ctx{context}"
  trace, stdout_path = trace_dir / f"{stem}.trace", trace_dir / f"{stem}.stdout.txt"
  gpu_xml, submissions_xml = trace_dir / f"{stem}.gpu.xml", trace_dir / f"{stem}.submissions.xml"
  xcrun = _executable(getattr(args, "xcrun", "xcrun"))
  reuse = bool(getattr(args, "reuse_capture", False))
  if reuse:
    missing = [path for path in (trace, stdout_path, gpu_xml, submissions_xml) if not path.exists()]
    if missing: raise FileNotFoundError(f"cannot reuse incomplete Metal capture; missing: {', '.join(map(str, missing))}")
  else:
    record = [xcrun, "xctrace", "record", "--template", "Metal System Trace", "--output", str(trace),
              "--time-limit", f"{int(getattr(args, 'trace_time_limit', 120))}s", "--no-prompt",
              "--target-stdout", str(stdout_path), *_env_overrides(args.env), "--launch", "--", *command]
    proc = subprocess.run(record, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if proc.returncode != 0: raise RuntimeError(f"Metal System Trace capture failed: {proc.stderr.strip() or proc.stdout.strip()}")
    _export_table(xcrun, trace, GPU_INTERVALS, gpu_xml)
    _export_table(xcrun, trace, COMMAND_BUFFERS, submissions_xml)
  result = normalize_metal_trace(gpu_xml, submissions_xml, stdout=stdout_path.read_text(), model_id=model_id,
                                 target_id=target_id, workload=workload, provider_id=provider,
                                 context=context, trace_path=trace)
  result["aux_sources"]["capture_command"] = command
  result["aux_sources"]["capture_reused"] = reuse
  return result, out


__all__ = ["capture_command_metal_trace", "capture_metal_system_trace", "normalize_metal_trace"]
