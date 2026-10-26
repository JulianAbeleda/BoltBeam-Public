"""BoltBeam's own Metal measurement: probe evidence and a timing trace with no Xcode, no Instruments, no tinygrad.

    python -m boltbeam.collectors.metal_native --run RUN [--probe-out P] [--timing-out T] [--only probe|timing]

It answers the two requests `analyze` writes into a run folder:

  probe_request.json  -> probe_evidence.json   (boltbeam.probe_evidence.v1)
    The building-block probe, the same on every GPU (collectors/boltbeam_gemv.py collect_probe_evidence), run here
    once this Mac is shown to be the run's chip.

  trace_request.json  -> timing_trace.json     (boltbeam.timing_trace.v1)
    whole_step rows: llama-bench decode at each requested depth (`-p 0 -n N -d CTX`). This is the measured
    tokens per second of a real decode on this GPU.
    candidate rows: BoltBeam's flushed GEMV time per role times that role's count per token. These are measured
    BoltBeam kernels, not the kernels llama.cpp ran, so they are not kernel rows: they cannot split llama.cpp's
    step. llama.cpp's per-kernel time stays absent (Metal shows per-dispatch time only to Instruments).

What Metal on Apple GPUs does not expose without Instruments stays absent: registers, scratch, occupancy,
memory latency, in-flight groups, the compiled ISA. Those probes classify as inconclusive (missing evidence).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable

from boltbeam.collectors import boltbeam_gemv, llama_bench_decode
from boltbeam.collectors.llama_bench_decode import CannotMeasure, bench_decode  # one refusal type for every collector
from boltbeam.core.canonical import pretty_json
from boltbeam.target.targets import get_target
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, read_json, run_dir

COLLECTOR_ID = "metal-native"
PROVIDER_ID = "boltbeam/metal-native"


# --- where measuring is possible ---------------------------------------------------------------------------

def this_machine_target() -> str | None:
  from boltbeam.workflow.autoscan import this_machine_target as here
  return here()


def preflight(target_id:str, model:str | pathlib.Path, *, run:str | pathlib.Path = "RUN", need_bench:bool = True,
              llama_bench:str = "llama-bench", here:Callable[[], str | None] = this_machine_target) -> dict[str, Any]:
  """What the measurement will use, or CannotMeasure with the reason and the command to run instead."""
  target = get_target(target_id)
  if target.backend != "Metal":
    raise CannotMeasure(f"{target_id} is a {target.backend} chip; BoltBeam measures by itself on Metal only",
                        f"on a machine with {target_id}: boltbeam collect-hw-trace --provider llama --run {run}")
  if sys.platform != "darwin":
    raise CannotMeasure(f"this machine is {sys.platform}, not a Mac", f"on a Mac with {target_id}: boltbeam metal-measure --run {run}")
  mine = here()
  if mine != target_id and target_id != "apple_metal":
    raise CannotMeasure(f"this Mac is {mine or 'an unknown chip'}, not {target_id}",
                        f"pick {mine} in step 2, or on a Mac with {target_id}: boltbeam metal-measure --run {run}")
  path = llama_bench_decode.model_file(model)
  bench = llama_bench_decode.find(llama_bench)
  if need_bench and bench is None:
    raise CannotMeasure("llama-bench is not installed, so there is no whole-step decode time",
                        f"brew install llama.cpp (or export {llama_bench_decode.ENV}=/path/to/llama-bench), "
                        "then measure again")
  return {"target": target, "model": path, "llama_bench": bench}


def _run_facts(run:pathlib.Path, **kw:Any) -> dict[str, Any]:
  manifest = load_manifest(run)
  facts = preflight(manifest.get("target_id"), manifest.get("model_path") or "", run=run, **kw)
  facts["manifest"] = manifest
  return facts


# --- timing trace ------------------------------------------------------------------------------------------

def build_timing_trace(manifest:dict[str, Any], request:dict[str, Any], evidence:dict[str, Any],
                       bench:dict[int, dict[str, Any]], peak_gbs:float | None) -> dict[str, Any]:
  """The trace from the measured pieces. Pure: the tests feed it numbers."""
  by_key = {(p["role"], p["quant"], tuple(p["shape"])): p for p in evidence.get("probes", []) if p.get("status") == "measured"}
  rows: list[dict[str, Any]] = []
  for ctx, b in sorted(bench.items()):
    rows.append({"scope": "whole_step", "context": ctx, "wall_us": 1e6 / b["tok_s"], "tok_s": b["tok_s"],
                 "stddev_tok_s": b["stddev_tok_s"], "source": "llama-bench", "engine": b.get("engine", {}),
                 # the same row llama_bench_decode writes on CUDA: the token window and the build the run measured
                 **({"decode_tokens": b["decode_tokens"]} if b.get("decode_tokens") else {}), **llama_bench_decode.build_facts(b)})
    for role in request.get("priority_roles", []):
      probe = by_key.get((role["role"], role["quant"], tuple(role["shape"])))
      if probe is None:
        continue
      count = int(role.get("count") or 1)
      bytes_one = sum(probe["bytes"].values())
      us = probe["timing"]["candidate_us"] * count
      # a candidate row, not a kernel row: BoltBeam's kernel is a different program from the llama.cpp step above,
      # so it cannot be a share of that step. baseline_us stays absent: llama.cpp's own kernel time is not visible.
      rows.append({"scope": "candidate", "context": ctx, "candidate_id": f"boltbeam_{probe['kernel']['name']}",
                   "kernel": f"boltbeam_{probe['kernel']['name']}_{role['shape'][0]}x{role['shape'][1]}",
                   "role": role["role"], "quant": role["quant"], "shape": list(role["shape"]), "count": count,
                   "candidate_us": us, "spread_pct": probe["timing"]["spread_pct"], "phys_bytes": bytes_one * count,
                   "gbs": bytes_one * count / (us * 1e3), "token_match": None, "route_bound": None,
                   "source": "boltbeam metal-native probe, flushed, times count per token"})
  trace = {
    "schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "timing_source": "llama-bench whole step; boltbeam metal-native GEMVs as role candidates",
    "contexts": sorted(bench), "rows": rows,
    "measured": ["whole_step.tok_s (llama-bench decode at each depth)", "candidate.candidate_us (BoltBeam GEMV, flushed, times count)"],
    "absent": ["kernel rows: llama.cpp's per-kernel time needs per-dispatch timing, which Metal gives only to Instruments",
               "attention, norm and RoPE: BoltBeam has no Metal kernel for them yet",
               "candidate baseline_us: there is no llama.cpp kernel time to compare against"],
    "notes": ["Candidate rows are BoltBeam's GEMVs timed alone on the real weights, one row per role per context.",
              "The GEMV does not depend on context, so the same measurement appears at every context."],
  }
  if peak_gbs:
    trace["peak_gbs"] = peak_gbs
  return trace


def collect_timing_trace(run:pathlib.Path, evidence:dict[str, Any], *, llama_bench:str = "llama-bench",
                         say:Callable[[str], None] = lambda _: None, batches=(1,)) -> dict[str, Any]:
  from boltbeam.workflow import progress
  facts = _run_facts(run, llama_bench=llama_bench)
  request = read_json(run / "trace_request.json")
  if request.get("workload") != "decode":
    raise CannotMeasure("metal-native times decode only; this run is prefill", "plan the run with workload decode")
  bench = {}
  contexts = request.get("contexts") or [0]
  parts = len(contexts) * (2 if any(int(b) > 1 for b in batches) else 1)
  progress.report(0, parts)
  for ctx in contexts:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx), cwd=run)
    progress.report(len(bench), parts)
  from boltbeam.workflow.screen import run_bandwidth
  peak_gbs, peak_source = run_bandwidth(run, facts["target"])
  trace = build_timing_trace(facts["manifest"], request, evidence, bench, peak_gbs)
  trace["peak_source"] = peak_source
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  trace["batches"] = llama_bench_decode.batch_points(bench, facts["model"], batches, say=say)
  return trace


# --- the command line --------------------------------------------------------------------------------------

def _write(obj:dict[str, Any], path:pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(pretty_json(obj))


def measure(run:pathlib.Path, *, only:str | None = None, probe_out:pathlib.Path | None = None,
            timing_out:pathlib.Path | None = None, llama_bench:str = "llama-bench",
            say:Callable[[str], None] = lambda _: None, batches=(1,)) -> dict[str, pathlib.Path]:
  probe_out = probe_out or run / "probe_evidence.json"
  timing_out = timing_out or run / "timing_trace.json"
  written = {}
  if only in (None, "probe"):
    _run_facts(run, need_bench=False)  # this Mac, and the run's chip, before the probe runs on it
    _write(boltbeam_gemv.collect_probe_evidence(run, say), probe_out)
    written["probe"] = probe_out
  if only in (None, "timing"):
    evidence = read_json(probe_out) if probe_out.exists() else {"probes": []}
    _write(collect_timing_trace(run, evidence, llama_bench=llama_bench, say=say, batches=batches), timing_out)
    written["timing"] = timing_out
  return written


def add_arguments(p:argparse.ArgumentParser) -> None:
  p.add_argument("--run", required=True, help="run folder holding probe_request.json and trace_request.json")
  p.add_argument("--only", choices=["probe", "timing"], default=None)
  p.add_argument("--probe-out", default=None, help="default: <run>/probe_evidence.json")
  p.add_argument("--timing-out", default=None, help="default: <run>/timing_trace.json")
  p.add_argument("--llama-bench", default="llama-bench")


def run_cli(args:argparse.Namespace) -> int:
  run = run_dir(args.run)
  try:
    written = measure(run, only=args.only, llama_bench=args.llama_bench,
                      probe_out=pathlib.Path(args.probe_out).expanduser() if args.probe_out else None,
                      timing_out=pathlib.Path(args.timing_out).expanduser() if args.timing_out else None,
                      say=lambda line: print(line, flush=True))
  except CannotMeasure as exc:
    sys.stderr.write(f"metal-measure: cannot measure here: {exc.reason}\n  instead: {exc.command}\n")
    return 2
  except (FileNotFoundError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"metal-measure: {exc}\n")
    return 1
  for kind, path in written.items():
    print(f"{kind}: {path}")
  return 0


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.metal_native", description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  add_arguments(p)
  return run_cli(p.parse_args(argv))


if __name__ == "__main__":
  raise SystemExit(main())
