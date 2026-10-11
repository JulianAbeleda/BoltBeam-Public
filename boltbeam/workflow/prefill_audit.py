"""A prefill run's where table as an audit: one entry per row, worst first by materiality.

Each entry says, in words every renderer prints as they are (the text summary, the HTML report, the TUI):

    claim            what the kernel or role took in the captured prefill
    standard         its ceiling and how it was derived: max(operations / the measured peak of its path, bytes / bandwidth)
    evidence         where the claim's numbers are read from
    re-performance   the kernel timed alone by BoltBeam's kernel timer where an adapter exists for it; otherwise why not
    tie-out          the row against the captured prefill and the untraced one
    materiality      ms lost, and the prefill (time to first token) if this row alone were fixed
    finding          what the numbers say

Every number carries a pointer [E1]..[E4] to the file it is read from, listed once at the head. Rows that lose under
MATERIAL of the untraced prefill each are collapsed into one line. Nothing here measures: it reads the where table
(tie_out.where_prefill_goes) and the run's files.
"""
from __future__ import annotations

import pathlib
from typing import Any

MATERIAL = 0.01  # a row losing less than this share of the untraced prefill is collapsed into the one line
TITLE = "Prefill audit"
FIELDS = ("claim", "standard", "evidence", "re-performance", "tie-out", "materiality", "finding")


def sources(run:pathlib.Path, length:int, trace:dict[str, Any]) -> list[dict[str, str]]:
  """The evidence pointers, E1..E4: the captured rows, the untraced truth, the peaks and bandwidth, the raw capture."""
  from boltbeam.collectors.prefill import RAW, TRACE
  peaks = trace.get("peaks_source")
  peak_from = peaks.get("source") if isinstance(peaks, dict) else peaks
  return [{"id": "E1", "file": f"{TRACE} lengths.{length}.rows", "what": "the captured prefill, kernel by kernel, attributed"},
          {"id": "E2", "file": f"timing_trace.json rows[context={length}]", "what": "the untraced prefill, clock and power"},
          {"id": "E3", "file": f"{TRACE} peaks, bandwidth_gbs", "what": f"the measured peaks ({peak_from or 'target facts'}) and "
                                                                       f"bandwidth ({trace.get('bandwidth_source') or 'target facts'})"},
          {"id": "E4", "file": f"{RAW}/{length}/", "what": "the engine's own launches as the vendor profiler read them"}]


def reperformance(row:dict[str, Any], provider:str, backend:str | None) -> str:
  """Whether the row's kernel could be timed alone: the adapter's kernel against the row's template."""
  from boltbeam.collectors.engine_kernels import ADAPTED_KERNEL
  if row["what"] == "gaps":
    return "not applicable: the gaps are no kernel"
  tmpl = str(row.get("template") or row["name"])
  base = tmpl.split("<", 1)[0]
  adapted = ADAPTED_KERNEL.get((provider, backend or ""))
  if adapted and base.startswith(adapted):
    return f"not re-performed in this run: the {provider} {backend} adapter times {adapted} alone (--role-time generic on a decode run)"
  return f"not re-performed: no adapter for {base} ({provider}{' on ' + backend if backend else ''})"


def entry(row:dict[str, Any], *, length:int, wall:float, base:float, bw:float | None, provider:str,
          backend:str | None) -> dict[str, Any]:
  now, limit, lost = row["now_ms"], row.get("limit_ms"), row["lost_ms"]
  calls = row.get("calls")
  launches = f"{calls} launches, " if calls else ""
  claim = f"{row['name']} took {now:.2f} ms [E1] of the {length:,}-token prefill ({launches}{now / wall:.1%} of the captured {wall:.1f} ms [E1])"
  if row["what"] == "gaps":
    standard = "0 ms: a GPU idle between kernels does no work"
  elif limit is None:
    standard = "none: the model counts no operations or bytes for this kernel" + (f" ({row['refused']})" if row.get("refused") else "")
  else:
    parts = []
    if row.get("flops") and row.get("peak_tflops"):
      parts.append(f"{row['flops'] / 1e12:.2f} TFLOP [E1] / {row['peak_tflops']:.0f} TFLOPS [E3] {row.get('path') or ''} peak = {row.get('compute_ms') or 0:.2f} ms")
    if row.get("bytes") and bw:
      parts.append(f"{row['bytes'] / 1e9:.2f} GB [E1] / {bw:,.0f} GB/s [E3] = {row.get('memory_ms') or 0:.2f} ms")
    standard = f"{limit:.2f} ms = max({'; '.join(parts) or 'its work'}), {row.get('bound') or 'its'}-bound"
    if " + " in str(row.get("path") or ""):
      standard += "; its operations split over its paths by instruction sites, each at its own peak"
  evidence = "E1 row '" + row["name"] + "' from E4" + ("; the untraced prefill E2" if base != wall else "")
  tie = (f"{now:.2f} ms of the rows' {wall:.2f} ms [E1], first kernel to last; the untraced prefill is {base:.1f} ms [E2]"
         + (f" (the capture differs by {wall - base:+.1f} ms)" if abs(wall - base) > 0.05 else ""))
  share = lost / base if base else 0.0
  material = (f"{lost:.2f} ms lost [{'E1' if limit is None else 'E1, E3'}], {share:.1%} of the untraced prefill; the prefill, time to first token, "
              f"if this row alone were fixed: {base - lost:.1f} ms from {base:.1f} ms [E2]")
  if row["what"] == "gaps":
    finding = f"the GPU sat idle {now:.2f} ms between kernels"
  elif limit is None and str(row.get("refused") or "").startswith("no ceiling: a matrix kernel"):
    finding = "a matrix kernel no role could be paired with: its time is real, its ceiling is not known"
  elif limit is None:
    finding = "no ceiling: every ms of it counts as lost until the model counts its work"
  else:
    mfu = f"{row['mfu_pct']:.1f}% MFU on {row.get('path')}; " if row.get("mfu_pct") is not None else ""
    finding = mfu + (f"{now / limit:.1f}x its ceiling" if limit else "at its ceiling")
  return {"name": row["name"], "lost_ms": lost, "share": share, "fields": [
    ["claim", claim], ["standard", standard], ["evidence", evidence],
    ["re-performance", reperformance(row, provider, backend)], ["tie-out", tie], ["materiality", material], ["finding", finding]]}


def audit(run:pathlib.Path, where:dict[str, Any] | None, length:int, trace:dict[str, Any], *,
          backend:str | None = None) -> dict[str, Any] | None:
  """The where table at `length` as an audit: entries worst first, rows under MATERIAL collapsed into one line."""
  if not where or not where.get("rows"):
    return None
  wall, base = float(where["token_ms"]), float(where["base_ms"])
  provider = str(trace.get("provider") or "")
  bw = trace.get("bandwidth_gbs")
  rows = sorted(where["rows"], key=lambda r: (-r["lost_ms"], r["name"]))
  big = [r for r in rows if base and r["lost_ms"] / base >= MATERIAL]
  small = [r for r in rows if r not in big]
  out = {"title": f"{TITLE} at {length:,} tokens", "length": length, "sources": sources(run, length, trace),
         "words": (f"Every row of the {length:,}-token prefill, worst first by ms lost. The untraced prefill is {base:.1f} ms [E2]; "
                   f"the captured rows sum to {wall:.1f} ms [E1]. Rows that each lose under {MATERIAL:.0%} of it are one line."),
         "entries": [entry(r, length=length, wall=wall, base=base, bw=bw, provider=provider, backend=backend)
                     for r in big]}
  if small:
    out["collapsed"] = (f"{len(small)} rows under {MATERIAL:.0%} each: {sum(r['now_ms'] for r in small):.2f} ms now, "
                        f"{sum(r['lost_ms'] for r in small):.2f} ms lost [E1] ({', '.join(r['name'] for r in small)})")
  return out


def lines(a:dict[str, Any] | None) -> list[str]:
  """The audit as plain text: the summary and the gameplan print these lines."""
  if not a:
    return []
  out = [a["title"] + ":", f"  {a['words']}"]
  out += [f"  [{s['id']}] {s['file']}: {s['what']}" for s in a["sources"]]
  for i, e in enumerate(a["entries"], 1):
    out.append(f"  {i}. {e['name']}")
    out += [f"     {k:<15} {v}" for k, v in e["fields"]]
  if a.get("collapsed"):
    out.append(f"  {a['collapsed']}")
  return out
