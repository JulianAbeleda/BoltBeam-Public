"""Where the time goes, per role, in tinygrad's own runtime, and how much of it is lost against the roofline.

`collect` runs boltbeam/runtime/tinygrad_decode_profile.py under the tinygrad fork's python, on the target's own
tinygrad device (METAL, NV, ...) with the measured JIT mode and PROFILE=1, so every kernel carries its own GPU timestamps. BoltBeam's
existing normalizer (artifacts/tinygrad_profile_events.decode_profile_events) turns the events into a
boltbeam.timing_trace.v1, joining each program to the role the model itself attached to the call (the program
census). Kernels with no such role are "not attributed": no role is guessed from a kernel name.

`loss` puts that next to the roofline's own per-role floor (the ceiling's decode roles: weight bytes per token over
the target's measured bandwidth): ideal ms, actual ms, lost ms = actual - ideal.

A floor is a lower bound. A role measured below its floor, or a token measured below the whole limit, means the
capture missed kernels or counted tokens wrong. `loss` refuses such a table as an incomplete measurement, with
the reason; it never shows it as a result.
"""
from __future__ import annotations

import json, os, pathlib, re, subprocess
from typing import Any

from boltbeam.artifacts.tinygrad_profile_events import decode_profile_events, load_profile_events

DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_decode_profile.py"
TRACE = "tinygrad_timing_trace.json"
RAW = "kernel_compare/role_time"
ROLE_SOURCE = "captured_program_semantics"
# the role sources loss() accepts: the model's own metadata (tinygrad) and bytes-and-count attribution (any provider)
ROLE_SOURCES = (ROLE_SOURCE, "attributed_by_bytes_and_count", "isolated_by_shape")  # the last: engine_kernels.py


PROVIDER = "tinygrad"  # the provider adapter's name (collectors/providers.py)
KV_ELEMENT = (2, "tinygrad's fp16 KV cache on its generated decode path")
OWN_TIMING = "tinygrad-profile-events"
# backends whose vendor capture sees tinygrad's kernels: on Metal each command buffer is labelled with its kernel
# when graphs are off (JIT=2, tinygrad/runtime/ops_metal.py), so Metal System Trace splits time per kernel
CAPTURED_BACKENDS = ("Metal",)
# why tinygrad's role time is its own timing and not a vendor capture, per backend
OWN_TIMING_REASON = {
  "Metal": "no outside capture: Metal per-kernel time needs Xcode's Metal System Trace",
  "CUDA": "nsys cannot see tinygrad's NV backend: it talks to the driver directly, not through CUDA",
  "AMD": "rocprofv3 sees 0 kernels from tinygrad's AMD backend: it talks to /dev/kfd directly",
}


def available(target, root:pathlib.Path | None = None) -> str | None:
  """Why tinygrad cannot measure this target here, or None when it can."""
  from boltbeam.search import role_compare
  if not device_for(target):
    return f"{target.target_id} names no tinygrad device"
  ready = role_compare.readiness(root)
  return None if ready["ready"] else ready["message"]


def whole_step(*, root:pathlib.Path, python:pathlib.Path, model:str, target, context:int, tokens:int = 16,
               timeout_s:float = 1800.0) -> dict[str, Any]:
  """Tokens per second of a real decode in tinygrad, as a user runs it: JIT with graphs, no profiling."""
  device = device_for(target)
  if not device:
    raise RuntimeError(f"target {target.target_id} names no tinygrad device")
  import tempfile
  # JIT=1 replays graphs, as a user runs it. The fork's Metal graph cannot hold an 8B model's buffers
  # ("Metal ICB offset exceeds 0xffffffff"); then the same decode runs without graphs (JIT=2) and says so.
  graph_error = None
  for jit in ("1", "2"):
    with tempfile.TemporaryDirectory() as tmp:
      argv = [str(python), str(DRIVER), "--model", model, "--context", str(context), "--tokens", str(tokens),
              "--events", f"{tmp}/events.pkl", "--census", f"{tmp}/census.json"]
      proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                            env={**os.environ, "PYTHONPATH": ".", "DEV": device, "JIT": jit, "PROFILE": "0", "VIZ": "0"})
    lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
    if proc.returncode == 0 and lines:
      out = json.loads(lines[-1])
      return {"tok_s": out["tok_s"], "tokens": out["tokens"], "wall_s": out["wall_s"], "device": device,
              "jit": "graphs" if jit == "1" else "no graphs (graph replay failed)", "graph_error": graph_error,
              "engine": {k: out[k] for k in ("jit_mode", "warm_tokens", "context_start", "context_end", "max_context", "build")}}
    if "GraphException" not in (proc.stderr or ""):
      break
    # the engine's own words for why the graph did not replay (the last GraphException line), kept with the row
    graph_error = next((l.strip()[-240:] for l in reversed(proc.stderr.splitlines()) if "GraphException" in l), None)
  raise DecodeFailed("tinygrad decode failed", proc.stderr or proc.stdout, root)


class DecodeFailed(RuntimeError):
  """The fork's decode did not run. str() is one sentence: the engine's own last line (the exception's message)
  and where inside the fork it was raised, e.g. "tinygrad decode failed: ffn_down_resadd epilogue requires
  rows=4096, got rows=5120 (ValueError inside the tinygrad fork at tinygrad/llm/decode_kernels.py:246)".
  details is the traceback, the fork's stderr tail, for a reader who asks for it; a screen keeps it behind
  "Show details" and the run's measure_status.json carries it."""
  DETAIL_LINES = 40

  def __init__(self, what:str, output:str | None, root:pathlib.Path | None = None):
    lines = [l.rstrip() for l in (output or "").splitlines() if l.strip()]
    self.details = "\n".join(lines[-self.DETAIL_LINES:])
    last = lines[-1].strip() if lines else "no output"
    kind, sep, message = last.partition(": ")
    # a python error's last line is "Kind: message"; anything else (a signal, a shell line) is the fact itself
    message = message if sep and kind.replace(".", "").isidentifier() else last
    where = ""
    if frame := next((l for l in reversed(lines) if l.lstrip().startswith('File "')), None):
      m = re.match(r'\s*File "(.*?)", line (\d+)', frame)
      if m:
        path, inside = m.group(1), ""
        if root is not None:
          try:  # a frame inside the fork is named by its path in the fork; any other frame by its own path
            path = str(pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve()))
            inside = "inside the tinygrad fork "
          except ValueError:
            pass
        where = f" ({kind if sep else 'error'} {inside}at {path}:{m.group(2)})"
    super().__init__(f"{what}: {message}{where}")


def decode_argv(python:pathlib.Path, model:str, context:int, tokens:int, raw:pathlib.Path) -> list[str]:
  return [str(python), str(DRIVER), "--model", model, "--context", str(context), "--tokens", str(tokens),
          "--events", str(raw / "events.pkl"), "--census", str(raw / "census.json")]


def role_time_captured(run:pathlib.Path, *, root:pathlib.Path, python:pathlib.Path, model:str, model_id:str, target,
                       context:int = 128, tokens:int = 16) -> dict[str, Any]:
  """tinygrad's kernels per role, captured from outside by the vendor tool and attributed by bytes and count,
  the same path llama.cpp takes. Two decodes at one context, `tokens` and 1 token, subtracted per kernel."""
  from boltbeam.collectors import attribution, vendor_capture
  device = device_for(target)
  raw = run / "kernel_compare" / "tinygrad_capture"
  from boltbeam.collectors.capture_settings import observed
  settings = observed(run, context)
  if settings.get("jit_mode") != 2 or type(settings.get("max_context")) is not int:
    raise ValueError("outside tinygrad capture requires an untraced non-graph run with recorded allocation")
  env = {"PYTHONPATH": ".", "DEV": device, "JIT": "2", "PROFILE": "0", "VIZ": "0"}
  got = {}
  for name, n in (("long", tokens), ("short", 1)):
    (raw / name).mkdir(parents=True, exist_ok=True)
    got[name] = vendor_capture.capture(target.backend, decode_argv(python, model, context, n, raw / name) +
                                       ["--max-context", str(settings["max_context"])],
                                       raw / name, cwd=root, env=env)
  window, overlap = attribution.shared_window(got["long"], got["short"])
  printed = [json.loads(x) for x in vendor_capture.program_output(raw / "long").splitlines() if x.startswith('{"tokens"')]
  token_ms = 1000.0 / printed[-1]["tok_s"] if printed else None
  return attribution.provider_trace(run, window, provider=PROVIDER, method=vendor_capture.plan(target.backend)["method"],
                                    model_id=model_id, target=target, context=context, tokens=tokens - 1, out=TRACE,
                                    token_ms=token_ms, long_kernels=len(got["long"]), overlap=overlap)


def runtime_name(target) -> str:
  """The runtime every role time comes from, as a screen names it. The one place this sentence is built."""
  return f"tinygrad's {target.backend} runtime"


def device_for(target) -> str | None:
  """The tinygrad device a registry target runs on, or None when the registry names none."""
  return (getattr(target, "capabilities", None) or {}).get("tinygrad_device")


def collect(run:pathlib.Path, *, root:pathlib.Path, python:pathlib.Path, model:str, model_id:str, target,
            context:int = 128, tokens:int = 10, timeout_s:float = 1800.0) -> dict[str, Any]:
  device = device_for(target)
  if not device:
    raise RuntimeError(f"target {target.target_id} names no tinygrad device, so its roles cannot be timed in tinygrad")
  raw = run / RAW
  raw.mkdir(parents=True, exist_ok=True)
  events, census = raw / "events.pkl", raw / "census.json"
  argv = [str(python), str(DRIVER), "--model", model, "--context", str(context), "--tokens", str(tokens),
          "--events", str(events), "--census", str(census)]
  from boltbeam.collectors.capture_settings import observed
  settings = observed(run, context)
  jit = settings.get("jit_mode")
  if jit not in (1, 2):
    raise ValueError("tinygrad capture requires the untraced jit_mode; rerun whole-step measurement")
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", "DEV": device, "JIT": str(jit), "PROFILE": "1", "VIZ": "0"})
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise DecodeFailed("in-model profile failed", proc.stderr or proc.stdout, root)
  summary = json.loads(lines[-1])
  for key in ("jit_mode", "warm_tokens", "context_start", "context_end", "max_context", "build"):
    if summary.get(key) != settings.get(key):
      raise ValueError(f"tinygrad captured {key} differs from the untraced run")
  # the peak is the run's one read bandwidth (workflow/screen.py run_bandwidth): the limit's number, one source
  from boltbeam.workflow.screen import run_bandwidth
  peak_gbs, peak_source = run_bandwidth(run, target)
  peak = {"peak_gbs": peak_gbs} if peak_gbs else {}
  trace = decode_profile_events(load_profile_events(events, tinygrad_root=root), model_id=model_id,
                                target_id=target.target_id, workload="decode", context=context,
                                decode_tokens=summary["tokens"], program_census=json.loads(census.read_text()),
                                model_profile=json.loads((run / "model_profile.json").read_text()) if (run / "model_profile.json").exists() else None,
                                source_path=str(events.relative_to(run)), **peak)
  trace.setdefault("notes", []).append(f"in-model decode on {device} with JIT={jit} PROFILE=1; "
                                       f"{summary['tokens']} tokens after a {context}-token prefill")
  for row in trace.get("rows", []):  # the profiled run's own wall time per token: the tie-out's measured token
    if row.get("scope") == "whole_step" and summary.get("wall_s"):
      row["token_ms"] = summary["wall_s"] * 1000.0 / summary["tokens"]
      row["engine"] = settings
      row["wall_tok_s"] = summary["tok_s"]
  trace["capture"] = {"method": OWN_TIMING, "reason": OWN_TIMING_REASON.get(target.backend)}
  trace["peak_source"] = peak_source
  (run / TRACE).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  return trace


INCOMPLETE = "incomplete measurement"
OVERCOUNTED = "overcounted measurement"
AT_LIMIT_NOISE = "at the limit (within measured repeat spread ±{ms:.6f} ms)"

def noise_band(band:dict | None) -> dict[str, Any]:
  """Only a scoped repeat calibration supplies an uncertainty band."""
  from boltbeam.workflow.repeat_bands import end_to_end_width
  measured = band if isinstance(band, dict) else None
  return {"calibration": measured, "width_ms": end_to_end_width(measured),
          "words": "per-row measured repeat spread" if measured else "not checked: no repeat band for this setup"}


def refusal(table:dict[str, Any], limit_ms:float | None) -> str | None:
  """Why a loss table cannot be a result, or None. A floor is a lower bound: nothing real runs below it, beyond
  measurement noise. A role or a token below its floor by no more than its noise band is accepted (loss() labels
  it); further below is refused. The captured run's own token time is an upper bound: kernels that run one after
  another fit inside it."""
  token_ms = table.get("token_ms")
  from boltbeam.workflow.repeat_bands import width
  total = table.get("band") or noise_band(None)
  spread = total["width_ms"]
  if spread is not None and token_ms and table["kernel_ms"] - token_ms > spread:
    return f"{OVERCOUNTED}: kernel time exceeds captured token by more than its measured repeat spread"
  if spread is not None and limit_ms and not table.get("isolated") and limit_ms - table["kernel_ms"] > spread:
    return f"{INCOMPLETE}: kernel time is below the floor beyond its measured repeat spread"
  from boltbeam.role_key import role_label
  below = []
  for row in table["roles"]:
    key = role_label(row["role"], row["quant"], row.get("shape"))
    delta = width(total["calibration"], key, work={"calls": row.get("calls_per_token"), "limit_ms": row["ideal_ms"]})
    if delta is not None and row["ideal_ms"] - row["actual_ms"] > delta:
      below.append(key)
  if below:
    return f"{INCOMPLETE}: roles below their floor beyond measured repeat spread: {', '.join(below)}"
  return None


def loss(ceiling_roles:list[dict[str, Any]], trace:dict[str, Any] | None,
         limit_ms:float | None = None, band:dict | None = None) -> dict[str, Any] | None:
  """Per role: ideal ms (roofline), actual ms (in model), lost ms; sorted by lost ms. None without a trace.
  A table that breaks a floor beyond its row's measured repeat spread comes back as {"status": "incomplete", "reason":
  ...} with no numbers. A role inside the band is labelled at the limit and loses 0 ms. A kernel row with a status
  other than measured (an isolated row the timer had no adapter for) is not a time: it is left out of the roles,
  the floor rule and the sums, and listed in not_timed with its reason, so a 0 µs row never reads as below its floor
  (the 27B's Q8_0 roles on the 5090 refused the whole table that way)."""
  chip = noise_band(band)
  if not trace: return None
  rows = trace.get("rows", [])
  whole = next((r for r in rows if r.get("scope") == "whole_step"), None)
  if not whole or not whole.get("tok_s") or not whole.get("wall_us"): return None
  tokens = whole.get("decode_tokens") or max(1, round(whole["wall_us"] * whole["tok_s"] / 1e6))
  from boltbeam.role_key import role_key, shape_nk
  # A fused launch with different logical role keys has one observed duration. Preserve that joint
  # duration and its work; never allocate it by bytes or compare a partial role time to the whole role's floor.
  joint = [r for r in rows if len({role_key(p) for p in r.get("logical_parts", [])}) > 1]
  joint_keys = {role_key(p) for r in joint for p in r["logical_parts"]}
  attribution = trace.get("aux_sources", {}).get("attribution") or {}
  incomplete_keys = {role_key(p) for p in attribution.get("matrix_work", {}).get("mismatches", [])}
  actual: dict[tuple[Any, ...], dict[str, float]] = {}
  floor_us = None  # an isolated row's dispatch floor (kernel_timer.less_floor_us): one per trace, read off the rows
  not_timed = []  # rows the timer could not time (no adapter for the quant, a failed check): no time, never a 0 ms
  for r in rows:
    if r.get("scope") == "kernel" and r.get("status") not in (None, "measured"):
      not_timed.append({"role": r.get("role"), "quant": r.get("quant"), **({"shape": list(shape_nk(r))} if shape_nk(r) else {}),
                        "status": r["status"], "reason": r.get("reason")})
      continue
    if r.get("scope") == "kernel" and r.get("role_source") in ROLE_SOURCES and r.get("role") and r.get("quant"):
      slot = actual.setdefault(role_key(r), {"us": 0.0, "us_less_floor": 0.0, "calls": 0, "spread_pct": None})
      slot["us"] += float(r["wall_us"]); slot["calls"] += int(r.get("calls", 1))
      slot["us_less_floor"] += float(r.get("wall_us_less_floor", r["wall_us"]))
      if r.get("spread_pct") is not None:  # an isolated row's sample spread (P90 - P10 over the median, %): the widest kept
        slot["spread_pct"] = max(float(r["spread_pct"]), slot["spread_pct"] or 0.0)
      if r.get("wall_us_less_floor") is not None:
        floor_us = (r.get("timing") or {}).get("dispatch_floor_us", floor_us)
  from collections import Counter
  shape_counts = Counter((c["role"], c["quant"]) for c in ceiling_roles)
  out = []
  for c in ceiling_roles:
    got = actual.get(role_key(c))
    if got is None and shape_nk(c) is not None and shape_counts[(c["role"], c["quant"])] == 1 and not any(k[:2] == (c["role"], c["quant"]) and len(k) > 2 for k in actual):
      got = actual.get((c["role"], c["quant"]))  # a trace without shapes keys by role and quant
    if got is None or role_key(c) in joint_keys | incomplete_keys: continue
    ms = got["us"] / tokens / 1000.0
    from boltbeam.workflow.repeat_bands import width
    from boltbeam.role_key import role_label
    delta = width(chip["calibration"], role_label(c["role"], c["quant"], shape_nk(c)),
                  work={"calls": got["calls"] / tokens, "limit_ms": c["floor_ms"]})
    noise = delta is not None and 0 < c["floor_ms"] - ms <= delta
    out.append({"role": c["role"], "quant": c["quant"], "shape": list(shape_nk(c) or []) or None, "ideal_ms": c["floor_ms"], "actual_ms": ms,
                "lost_ms": ms - c["floor_ms"], "calls_per_token": got["calls"] / tokens,
                "less_floor_ms": got["us_less_floor"] / tokens / 1000.0 if floor_us is not None else None,
                "spread_pct": got["spread_pct"],
                "within_noise": noise, "check_status": "checked" if delta is not None else "not checked",
                "band_ms": delta, "label": AT_LIMIT_NOISE.format(ms=delta) if noise else None})
  total_lost = sum(max(r["lost_ms"], 0.0) for r in out)
  for r in out: r["share"] = (max(r["lost_ms"], 0.0) / total_lost) if total_lost > 0 else 0.0
  out.sort(key=lambda r: -r["lost_ms"])
  whole_ms = whole["wall_us"] / tokens / 1000.0
  table = {"status": "measured", "source": "measured in tinygrad's runtime", "device": trace.get("target_id"),
           "tokens": tokens, "kernel_ms": whole_ms, "tok_s": whole.get("wall_tok_s", whole["tok_s"]), "roles": out,
           "isolated": whole.get("measurement_scope") == "summed_isolated_kernels", "floor_us": floor_us,
           "not_attributed_ms": whole_ms - sum(r["actual_ms"] for r in out), "token_ms": whole.get("token_ms"),
           "unpaired_roles": list(trace.get("unpaired_roles") or []) + [
             {"role": c["role"], "quant": c["quant"], "shape": list(shape_nk(c) or []) or None,
              "reason": "time shared with other logical parts in a fused launch"} for c in ceiling_roles if role_key(c) in joint_keys],
           "joint_role_groups": [{"kernel": r["kernel"], "actual_ms": r["wall_us"] / tokens / 1000,
                                  "calls_per_token": r["calls"] / tokens, "logical_parts": r["logical_parts"],
                                  "logical_flops_per_token": r["logical_flops"] / tokens} for r in joint], "band": chip,
           "attribution": trace.get("aux_sources", {}).get("attribution"),
           "unpaired_launches": trace.get("aux_sources", {}).get("attribution", {}).get("unresolved_launches", []),
           "check_status": "checked" if chip["width_ms"] is not None and not joint_keys and (not attribution or attribution.get("complete")) and
             all(r["check_status"] == "checked" for r in out) else "not checked",
           # a limit role the timer had no adapter for: its own row with the reason, outside the floor rule and the sums
           "not_timed": [n for n in not_timed if role_key(n) in {role_key(c) for c in ceiling_roles}
                         or (n["role"], n["quant"]) in {(c["role"], c["quant"]) for c in ceiling_roles}]}
  if reason := refusal(table, limit_ms):
    return {"status": "incomplete", "reason": reason, "tokens": tokens,
            "events": whole.get("launch_count"), "roles": []}
  return table
