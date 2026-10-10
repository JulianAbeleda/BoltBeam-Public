"""Where the time goes, per role, in tinygrad's own runtime, and how much of it is lost against the roofline.

`collect` runs boltbeam/runtime/tinygrad_decode_profile.py under the tinygrad fork's python, on the target's own
tinygrad device (METAL, NV, ...) with JIT=2 PROFILE=1, so every kernel carries its own GPU timestamps. BoltBeam's
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

import json, os, pathlib, subprocess
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
              "jit": "graphs" if jit == "1" else "no graphs (graph replay failed)", "graph_error": graph_error}
    if "GraphException" not in (proc.stderr or ""):
      break
    # the engine's own words for why the graph did not replay (the last GraphException line), kept with the row
    graph_error = next((l.strip()[-240:] for l in reversed(proc.stderr.splitlines()) if "GraphException" in l), None)
  raise RuntimeError(f"tinygrad decode failed: {(proc.stderr or proc.stdout).strip()[-400:]}")


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
  env = {"PYTHONPATH": ".", "DEV": device, "JIT": "2", "PROFILE": "0", "VIZ": "0"}
  got = {}
  for name, n in (("long", tokens), ("short", 1)):
    (raw / name).mkdir(parents=True, exist_ok=True)
    got[name] = vendor_capture.capture(target.backend, decode_argv(python, model, context, n, raw / name),
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
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", "DEV": device, "JIT": "2", "PROFILE": "1", "VIZ": "0"})
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError(f"in-model profile failed: {(proc.stderr or proc.stdout).strip()[-400:]}")
  summary = json.loads(lines[-1])
  # the peak is the run's one read bandwidth (workflow/screen.py run_bandwidth): the limit's number, one source
  from boltbeam.workflow.screen import run_bandwidth
  peak_gbs, peak_source = run_bandwidth(run, target)
  peak = {"peak_gbs": peak_gbs} if peak_gbs else {}
  trace = decode_profile_events(load_profile_events(events, tinygrad_root=root), model_id=model_id,
                                target_id=target.target_id, workload="decode", context=context,
                                decode_tokens=summary["tokens"], program_census=json.loads(census.read_text()),
                                source_path=str(events.relative_to(run)), **peak)
  trace.setdefault("notes", []).append(f"in-model decode on {device} with JIT=2 PROFILE=1: one command buffer per "
                                       f"kernel; {summary['tokens']} tokens after a {context}-token prefill")
  for row in trace.get("rows", []):  # the profiled run's own wall time per token: the tie-out's measured token
    if row.get("scope") == "whole_step" and summary.get("wall_s"):
      row["token_ms"] = summary["wall_s"] * 1000.0 / summary["tokens"]
  trace["capture"] = {"method": OWN_TIMING, "reason": OWN_TIMING_REASON.get(target.backend)}
  trace["peak_source"] = peak_source
  (run / TRACE).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  return trace


INCOMPLETE = "incomplete measurement"
OVERCOUNTED = "overcounted measurement"
DEFAULT_BAND = 0.01  # the band when the chip has none recorded
BAND_RULE = ("within ±{pct:.1f}% of the floor counts as at the limit: the largest of this chip's read probe spread, "
             "its cold to sustained difference and its documented run-to-run range, at least 1%")
AT_LIMIT_NOISE = "at the limit (within ±{pct:.1f}%)"


def noise_band(band:float | None) -> dict[str, Any]:
  """The chip's one plausibility band (screen.plausibility_band), at least 1%, with the sentence that explains it."""
  b = max(float(band or 0.0), DEFAULT_BAND)
  return {"band": b, "words": BAND_RULE.format(pct=100 * b)}


def refusal(table:dict[str, Any], limit_ms:float | None) -> str | None:
  """Why a loss table cannot be a result, or None. A floor is a lower bound: nothing real runs below it, beyond
  measurement noise. A role or a token below its floor by no more than its noise band is accepted (loss() labels
  it); further below is refused. The captured run's own token time is an upper bound: kernels that run one after
  another fit inside it."""
  token_ms = table.get("token_ms")
  total = table.get("band") or noise_band(None)
  if token_ms and table["kernel_ms"] > token_ms * (1 + total["band"]):
    return (f"{OVERCOUNTED}: {table['kernel_ms']:.2f} ms of GPU time per token is more than the "
            f"{token_ms:.2f} ms the captured run took per token by more than ±{100 * total['band']:.1f}%, so the "
            "capture counted work outside the decode tokens or kernels that overlap")
  if limit_ms and not table.get("isolated") and table["kernel_ms"] < limit_ms * (1 - total["band"]):
    # (an isolated table sums only the weight kernels it timed, so the whole-token floor does not bound it)
    return (f"{INCOMPLETE}: {table['kernel_ms']:.2f} ms of GPU time per token is below the limit's floor of "
            f"{limit_ms:.2f} ms by more than ±{100 * total['band']:.1f}%, so the "
            "capture missed kernels or counted tokens wrong")
  below = [r for r in table["roles"] if r["actual_ms"] < r["ideal_ms"] * (1 - total["band"])]
  if below:
    names = ", ".join(f"{r['role']} {r['quant']} {r['actual_ms']:.2f} < {r['ideal_ms']:.2f} ms "
                      for r in below)
    return (f"{INCOMPLETE}: {len(below)} role(s) measured more than ±{100 * total['band']:.1f}% below their floor ({names}), "
            "so the capture missed kernels")
  return None


def loss(ceiling_roles:list[dict[str, Any]], trace:dict[str, Any] | None,
         limit_ms:float | None = None, band:float | None = None) -> dict[str, Any] | None:
  """Per role: ideal ms (roofline), actual ms (in model), lost ms; sorted by lost ms. None without a trace.
  A table that breaks a floor beyond the chip's band (noise_band) comes back as {"status": "incomplete", "reason":
  ...} with no numbers. A role inside the band is labelled at the limit and loses 0 ms."""
  chip = noise_band(band)
  if not trace: return None
  rows = trace.get("rows", [])
  whole = next((r for r in rows if r.get("scope") == "whole_step"), None)
  if not whole or not whole.get("tok_s") or not whole.get("wall_us"): return None
  tokens = whole.get("decode_tokens") or max(1, round(whole["wall_us"] * whole["tok_s"] / 1e6))
  actual: dict[tuple[str, str], dict[str, float]] = {}
  floor_us = None  # an isolated row's dispatch floor (kernel_timer.less_floor_us): one per trace, read off the rows
  for r in rows:
    if r.get("scope") == "kernel" and r.get("role_source") in ROLE_SOURCES and r.get("role") and r.get("quant"):
      slot = actual.setdefault((r["role"], r["quant"]), {"us": 0.0, "us_less_floor": 0.0, "calls": 0})
      slot["us"] += float(r["wall_us"]); slot["calls"] += int(r.get("calls", 1))
      slot["us_less_floor"] += float(r.get("wall_us_less_floor", r["wall_us"]))
      if r.get("wall_us_less_floor") is not None:
        floor_us = (r.get("timing") or {}).get("dispatch_floor_us", floor_us)
  out = []
  for c in ceiling_roles:
    got = actual.get((c["role"], c["quant"]))
    if got is None: continue
    ms = got["us"] / tokens / 1000.0
    noise = ms < c["floor_ms"] and ms >= c["floor_ms"] * (1 - chip["band"])
    out.append({"role": c["role"], "quant": c["quant"], "ideal_ms": c["floor_ms"], "actual_ms": ms,
                "lost_ms": 0.0 if noise else ms - c["floor_ms"], "calls_per_token": got["calls"] / tokens,
                "less_floor_ms": got["us_less_floor"] / tokens / 1000.0 if floor_us is not None else None,
                "within_noise": noise, "label": AT_LIMIT_NOISE.format(pct=100 * chip["band"]) if noise else None})
  total_lost = sum(max(r["lost_ms"], 0.0) for r in out)
  for r in out: r["share"] = (max(r["lost_ms"], 0.0) / total_lost) if total_lost > 0 else 0.0
  out.sort(key=lambda r: -r["lost_ms"])
  whole_ms = whole["wall_us"] / tokens / 1000.0
  table = {"status": "measured", "source": "measured in tinygrad's runtime", "device": trace.get("target_id"),
           "tokens": tokens, "kernel_ms": whole_ms, "tok_s": whole["tok_s"], "roles": out,
           "isolated": whole.get("measurement_scope") == "summed_isolated_kernels", "floor_us": floor_us,
           "not_attributed_ms": whole_ms - sum(r["actual_ms"] for r in out), "token_ms": whole.get("token_ms"),
           "unpaired_roles": list(trace.get("unpaired_roles") or []), "band": chip}
  if reason := refusal(table, limit_ms):
    return {"status": "incomplete", "reason": reason, "tokens": tokens,
            "events": whole.get("launch_count"), "roles": []}
  return table
