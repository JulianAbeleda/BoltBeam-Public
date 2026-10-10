"""Pointers from every per-role row and every next-step item to the raw facts in the run folder, and the verdicts
read off facts the run already holds. No new measurement happens here.

A pointer is {"file": <path relative to the run>, "path": <a.b[3].c inside it, or absent for the whole file>}.
resolve() opens the file and follows the path; the tests resolve every pointer the seam gives.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

# Stated thresholds for the item verdicts. Each is a rule with its number, not a fit.
LAUNCH_HEAVY = 500  # kernel launches per token above this: launch_heavy on the gaps line
KV_SHARE = 0.25  # KV cache read above this share of the ideal token: kv_dominated
THROTTLE_SHARE = 0.05  # cold read above sustained by more than this: throttled
COMPARE = "kernel_compare/compare.json"
MACHINE = "machine_facts.json"
MEASURE = "measure_status.json"
PROFILE = "model_profile.json"
STEP4 = "timing_trace.json"  # step 4's whole step: THE measured token (tie_out.measured_step)

LEVERS = {
  "launch_heavy": "Launch fewer kernels: run the token as one graph or fuse kernels.",
  # the engine said its graph did not replay: the lever is the graph path, not a new kernel
  "graph_failed": "Fix the graph path first: graph replay failed on this run, so the token ran as single launches.",
  "kv_dominated": "Read less KV cache: a smaller KV type (8 or 4 bit) or a shorter context.",
  "throttled": "Measure again with the chip cool and on power; the bandwidth limit moves while it throttles.",
  "cannot_split": "Split them first: a capture that names each kernel's role (tinygrad's own timing does).",
  "compute_bound": "Use the chip's matrix units for it, or fewer flops per byte.",
  "unexplained": "Look at the kernel rows in the evidence: no rule here assigns this time.",
  # counted in the limit already (profile/weight_ledger.py); the lever is a name, so its kernel gets a per-role row
  "unclassified_weights": "Name them: add their tensor name patterns to profile/roles.py as limit roles.",
}


def _step(obj:Any, part:str) -> Any:
  name, *idx = re.findall(r"[^\[\]]+", part)
  if name and not name.isdigit():
    obj = obj[name]
  elif name.isdigit():
    idx = [name, *idx]
  for i in idx:
    obj = obj[int(i)]
  return obj


def resolve(run:pathlib.Path, ptr:dict[str, Any]) -> Any:
  """The value a pointer names. Raises KeyError, IndexError or OSError when it does not exist."""
  obj = json.loads((run / ptr["file"]).read_text())
  for part in (ptr.get("path") or "").split("."):
    if part:
      obj = _step(obj, part)
  return obj


def _read(run:pathlib.Path, name:str) -> dict[str, Any]:
  try:
    return json.loads((run / name).read_text())
  except (OSError, ValueError):
    return {}


def _ptr(file:str, path:str | None = None) -> dict[str, Any]:
  return {"file": file, "path": path} if path else {"file": file}


class Facts:
  """The run's raw files, read once, with the pointers into them."""

  def __init__(self, run:pathlib.Path, trace_file:str | None):
    self.run, self.trace_file = run, trace_file
    self.trace = _read(run, trace_file) if trace_file else {}
    self.compare = _read(run, COMPARE)
    self.machine = _read(run, MACHINE)
    self.measure = _read(run, MEASURE)
    self.profile = _read(run, PROFILE)
    self.trace4 = _read(run, STEP4)

  def step4_row(self, provider:str | None = None) -> tuple[int, dict[str, Any]] | None:
    """THE measured token's row in step 4's trace, with its index: the same pick as tie_out.measured_step."""
    from boltbeam.workflow.tie_out import measured_step
    prov = provider or (self.measure.get("provider") or "llama.cpp")
    token = measured_step(self.trace4, prov)
    return (token["index"], self.trace4["rows"][token["index"]]) if token else None

  def step4(self) -> list[dict[str, Any]]:
    got = self.step4_row()
    return [_ptr(STEP4, f"rows[{got[0]}]")] if got else []

  def graph_failure(self) -> dict[str, Any] | None:
    """The engine's own account that its graph did not replay in step 4 (tinygrad whole_step source / graph_error),
    with the pointer; None when the run says nothing of the kind."""
    got = self.step4_row()
    if not got:
      return None
    i, row = got
    if "graph replay failed" not in str(row.get("source") or "") and not row.get("graph_error"):
      return None
    return {"source": row.get("source"), "error": row.get("graph_error"), "evidence": [_ptr(STEP4, f"rows[{i}]")]}

  def common(self) -> list[dict[str, Any]]:
    out = []
    if self.machine.get("gpus"):
      out.append(_ptr(MACHINE, "gpus[0]"))
    if self.measure:
      out.append(_ptr(MEASURE))
    return out

  def role(self, role:str, quant:str, shape:Any = None) -> list[dict[str, Any]]:
    from boltbeam.role_key import shape_nk
    want = shape_nk({"shape": shape})
    same = lambda r: r.get("role") == role and r.get("quant") == quant and (want is None or shape_nk(r) in (None, want))  # noqa: E731
    out = [_ptr(self.trace_file, f"rows[{i}]") for i, r in enumerate(self.trace.get("rows") or [])
           if r.get("scope") == "kernel" and same(r)]
    out += [_ptr(PROFILE, f"roles[{i}]") for i, r in enumerate(self.profile.get("roles") or []) if same(r)]
    out += [_ptr(COMPARE, f"roles[{i}]") for i, r in enumerate(self.compare.get("roles") or []) if same(r)]
    return out + self.common()

  def whole(self) -> list[dict[str, Any]]:
    if not self.trace_file:
      return self.step4()
    return [_ptr(self.trace_file, f"rows[{i}]") for i, r in enumerate(self.trace.get("rows") or [])
            if r.get("scope") == "whole_step"][:1]

  def others(self, taken:set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    from boltbeam.role_key import role_key
    return [_ptr(self.trace_file, f"rows[{i}]") for i, r in enumerate(self.trace.get("rows") or [])
            if r.get("scope") == "kernel" and role_key(r) not in taken and role_key(r)[:2] not in taken]

  def launches_per_token(self) -> float | None:
    w = next((r for r in self.trace.get("rows") or [] if r.get("scope") == "whole_step"), {})
    if w.get("launch_count") and w.get("decode_tokens"):
      return float(w["launch_count"]) / float(w["decode_tokens"])
    return None

  def throttle(self) -> str | None:
    """Why the chip counts as throttled, from the machine facts, or None (also None when the facts lack it)."""
    g = (self.machine.get("gpus") or [{}])[0]
    if g.get("regime") == "sustained_throttle":
      return "the read probe's sustained regime is sustained_throttle"
    cold, sus = g.get("cold_gbs"), g.get("sustained_gbs")
    if cold and sus and (cold - sus) / sus > THROTTLE_SHARE:
      return f"cold read {cold:.1f} GB/s, sustained {sus:.1f} GB/s: {100 * (cold - sus) / sus:.0f}% apart, over {THROTTLE_SHARE:.0%}"
    return None


def items(facts:Facts, loss:dict[str, Any]) -> list[dict[str, Any]]:
  """The item verdicts that come from facts already in the run, each with one lever and its pointers."""
  out = []
  t = loss.get("tie_out") or {}
  token = t.get("token_ms")
  roles = loss.get("roles") or []
  launches = facts.launches_per_token()
  gaps = next((l for l in t.get("lines") or [] if l.get("how") == "difference"), None)
  graph = facts.graph_failure()
  if launches is not None and launches > LAUNCH_HEAVY and gaps:
    what, do, ev = f"Launch heavy: {launches:.0f} kernel launches per token", LEVERS["launch_heavy"], facts.whole()
    if graph:  # the engine said so itself: the lever is the graph path, never "run it as one graph" again
      why = f" ({graph['error']})" if graph.get("error") else ""
      what = f"Launch heavy: graph replay failed on this run{why}, so the token ran as {launches:.0f} launches"
      do, ev = LEVERS["graph_failed"], graph["evidence"] + facts.whole()
    out.append({"verdict": "launch_heavy", "what": what, "ms": max(gaps["ms"], 0.0), "do": do,
                "rule": f"launches per token above {LAUNCH_HEAVY}", "evidence": ev})
  elif graph and gaps:  # no launch count, but the engine still says its graph failed: said, with its row
    out.append({"verdict": "graph_failed", "what": "Graph replay failed on this run" + (f" ({graph['error']})" if graph.get("error") else ""),
                "ms": max(gaps["ms"], 0.0), "do": LEVERS["graph_failed"], "rule": "the engine's own trace row says so",
                "evidence": graph["evidence"]})
  if t.get("kv_ms") and t.get("limit_ms") and t["kv_ms"] / t["limit_ms"] > KV_SHARE:
    out.append({"verdict": "kv_dominated", "what": f"KV dominated: KV read is {100 * t['kv_ms'] / t['limit_ms']:.0f}% "
                f"of the ideal token at context {t.get('context', 0):.0f}", "ms": t["kv_ms"], "do": LEVERS["kv_dominated"],
                "rule": f"KV read above {KV_SHARE:.0%} of the ideal", "evidence": [_ptr(PROFILE, "metadata.attention"), *facts.whole()]})
  if why := facts.throttle():
    out.append({"verdict": "throttled", "what": f"Throttled: {why}", "ms": 0.0, "do": LEVERS["throttled"],
                "rule": f"cold above sustained by more than {THROTTLE_SHARE:.0%}, or sustained_throttle",
                "evidence": [_ptr(MACHINE, "gpus[0]")]})
  unpaired = loss.get("unpaired_roles") or []
  if unpaired:
    names = ", ".join(f"{u.get('role')} {u.get('quant') or ''}".strip() for u in unpaired)
    ev = [_ptr(facts.trace_file, "unpaired_roles")] if facts.trace.get("unpaired_roles") else facts.whole()
    out.append({"verdict": "cannot_split", "what": f"Cannot split: {names}. No kernel group matched their calls and bytes",
                "ms": float(loss.get("not_attributed_ms") or 0.0), "do": LEVERS["cannot_split"],
                "rule": "roles the capture could not separate", "evidence": ev})
  for verdict in ("compute_bound", "unexplained"):
    rs = [r for r in roles if r.get("reason") == WORD[verdict] and r["lost_ms"] > 0]
    if not rs:
      continue
    ms = sum(r["lost_ms"] for r in rs)
    stats = ", ".join(f"{r['role']} {r['quant']} {r['lost_ms']:.2f} ms" + (f" at {r['us_per_call']:.0f} µs/call"
                      if r.get("us_per_call") else "") for r in rs)
    band = (t.get("band") or "").strip()
    what = (f"Unexplained: {ms:.2f} ms ({stats})" + (f"; band {band}" if band else "")) if verdict == "unexplained" \
      else f"Compute bound: {stats}"
    out.append({"verdict": verdict, "what": what, "ms": ms, "do": LEVERS[verdict],
                "rule": "no rule assigns it" if verdict == "unexplained" else "the roofline regime is compute",
                "evidence": [p for r in rs for p in r.get("evidence") or []]})
  ledger = (facts.profile.get("metadata") or {}).get("weights") or {}
  if ledger.get("unclassified_tensors"):  # counted in the limit, never dropped: said so, with what they are
    from boltbeam.profile.weight_ledger import summary_line
    share = ledger["unclassified_bytes"] / ledger["counted_bytes"] if ledger.get("counted_bytes") else 0.0
    names = ", ".join(f"{r['pattern']} {r['quant']}" for r in ledger["unclassified"])
    out.append({"verdict": "unclassified_weights", "what": f"Not classified: {names}. "
                + summary_line(ledger).split("; ", 1)[1], "ms": share * float(loss.get("limit_ms") or 0.0),
                "do": LEVERS["unclassified_weights"], "rule": "a weight tensor no limit role names, counted by its bytes",
                "evidence": [_ptr(PROFILE, "metadata.weights.unclassified")]})
  for x in out:
    x["share"] = x["ms"] / token if token else 0.0
  return out


# role reason words (workflow/tie_out.py REASONS) for the two role verdicts here
WORD = {"compute_bound": "compute bound", "unexplained": "unexplained"}
