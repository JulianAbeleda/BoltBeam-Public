#!/usr/bin/env python3
"""Recompute a run's numbers from its own files and say whether what BoltBeam showed adds up (standard library only).

    tools/check_run_math.py RUN_DIR...        a run folder (it holds run_manifest.json), or a folder of them
    tools/check_run_math.py --tui PANE RUN    also check a captured TUI results screen (tmux capture-pane) against RUN

For each run with a headline ("N tok/s measured, limit L tok/s, P% of roofline") in its summary.md:
  speed     N is the engine's own timing in timing_profile.json at one of the measured contexts
  share     P is N / L
  limit     one GPU, dense, no speculation: L is at most the weights-only bound (the GPU's measured read bandwidth in
            machine_facts.json over the weight bytes in weight_inventory.json) and within 10% of it (the KV read and
            the activations make up the rest)
  tie-out   limit + the measured parts + the gaps = the measured token (each line shown to 0.001 ms); over the ceiling,
            the measured token − what recording added = the untraced token, and its "gaps at most" bound is the untraced
            token less the kernels (the measured token less its gaps line)
  table     the "Where the token goes" rows add up to its total (each row shown to 0.01 ms)
  capture   the traced token is within the 5% capture-overhead ceiling of the untraced one; above it the per-kernel
            table is a warning, not a failure (the headline is the untraced speed)
  batch     a "batch B:" headline (B > 1) is per stream: N is timing_trace.json's step of B streams and its total is
            B x N; each "Per batch" row's per-stream speed and step are timing_trace.json's point, its total is B x the
            per-stream speed, its limit total is B / the limit step, its % is total / limit total, and at each context
            the limit step is affine in B (the weights once, B times each stream's KV and state); a B > 1 headline's
            limit is B / one row's limit total at that batch
With --tui: the screen's "N tok/s measured · roofline L tok/s" and "P% of the limit" are the run's, and the table's
last cumulative tok/s reaches the roofline (the table reconciles it).
Exit 1 when any figure does not add up.
"""
import json, pathlib, re, sys

HEAD = re.compile(r"batch (\d+): ([\d.]+) tok/s(?: per stream \(([\d.]+) in total\))? measured, limit ([\d.]+) tok/s(?: per stream)?, "
                  r"(\d+)% of roofline")
BATCH = re.compile(r"^  context ([\d.]+), batch (\d+): ([\d.]+) tok/s per stream, ([\d.]+) tok/s in total, step ([\d.]+) ms; "
                   r"limit ([\d.]+) ms, ([\d.]+) tok/s in total; ([\d.]+)% of limit$", re.M)
ROW = re.compile(r"^\s{2}(\S.*?)\s{2,}(-?\d+\.\d+)\s")
TIE = re.compile(r"^\s*([+=−])?\s*(.+?)\s{2,}(-?\d+\.\d{3})\s")
BOUND = re.compile(r"gaps at most (-?\d+\.\d{3}) ms")
CAPTURE = re.compile(r"capture ([+-]?[\d.]+)% (over|under) native")
TUI_HEAD = re.compile(r"([\d.]+) tok/s measured · roofline ([\d.]+) tok/s")
TUI_PCT = re.compile(r"(\d+)% of the limit")
TUI_CUM = re.compile(r"│\s*([\d.]+)\s*│\s*│?\s*$")


def load(run, name):
  p = run / name
  return json.loads(p.read_text()) if p.is_file() else None


def table_sum(text):
  lines = text.splitlines()
  start = next((i for i, l in enumerate(lines) if l.strip().startswith("row") and "ms now" in l), None)
  if start is None:
    return None
  rows = []
  for l in lines[start + 1:]:
    if l.strip().startswith("= total"):
      return sum(rows), float(re.search(r"-?\d+\.\d+", l).group()), len(rows)
    m = ROW.match(l)
    if m and "throughput" not in m.group(1):
      rows.append(float(m.group(2)))
  return None


def tie_out(text):
  """The Tie-out block: [(what, the parts' sum, the stated total, the number of parts)] for each '=' line, the next
  starting from that total ('−' subtracts: over the ceiling, what recording added to the gaps), and for a "gaps at most"
  bound, the untraced token less the kernels (the measured token less its gaps line); [] without one."""
  lines = text.splitlines()
  start = next((i for i, l in enumerate(lines) if l.startswith("Tie-out")), None)
  if start is None:
    return []
  base, parts, gaps, out = 0.0, [], None, []
  for l in lines[start + 1:]:
    if l.startswith("      "):  # a kernel listed under a part, already inside it
      continue
    m = TIE.match(l)
    if not m:
      break
    value = float(m.group(3))
    if m.group(1) == "=":
      out.append(("tie-out", base + sum(parts), value, len(parts) + 1))
      if (b := BOUND.search(l)) and gaps is not None:  # the bound: this total (the untraced token) less the kernels
        out.append(("gaps bound", value - (base - gaps), float(b.group(1)), 3))
      base, parts = value, []
    elif m.group(1):
      parts.append(-value if m.group(1) == "−" else value)
      gaps = value if m.group(2).startswith("gaps between kernels") else gaps
    else:  # an unsigned line is the starting limit; a later one (another context) replaces it
      base = value
  return out


def check(run):
  notes, ok = [], True
  text = (run / "summary.md").read_text() if (run / "summary.md").is_file() else ""
  if text.startswith("FAILED"):  # the report refused its own evidence: no figure of it stands
    return "report FAILED", [text.splitlines()[1][:200] if len(text.splitlines()) > 1 else text[:200]], False, None
  m = HEAD.search(text)
  if not m:
    tp = load(run, "timing_profile.json")
    if not tp:
      return "never measured (no headline, no timing_profile.json)", ["see summary.md and stage-log.txt"], True, None
    speeds = [round(c["tok_s"], 1) for c in tp.get("context_summaries", []) if c.get("tok_s")]
    return f"no limit in the summary; measured {speeds} tok/s", [], True, None
  b, n, lim, pct = int(m.group(1)), float(m.group(2)), float(m.group(4)), int(m.group(5))
  points = (load(run, "timing_trace.json") or {}).get("batches") or []
  speeds = [c["tok_s"] for c in (load(run, "timing_profile.json") or {}).get("context_summaries", []) if c.get("tok_s")] \
    if b == 1 else [p["tok_s_stream"] for p in points if p.get("batch") == b]
  if not any(abs(s - n) <= 0.05 + 1e-9 for s in speeds):
    notes.append(f"BAD speed: {n} is not batch {b}'s in the run {[round(s, 2) for s in speeds]}"); ok = False
  if b > 1 and (m.group(3) is None or abs(float(m.group(3)) - b * n) > 0.05 * b + 0.05):
    notes.append(f"BAD total: batch {b} x {n} tok/s per stream is not the stated total {m.group(3)}"); ok = False
  rows = [tuple(float(x) for x in r.groups()) for r in BATCH.finditer(text)]
  ok = check_batches(rows, points, b, lim, notes) and ok
  if round(100 * n / lim) != pct:
    notes.append(f"BAD share: {n}/{lim} is {100 * n / lim:.1f}%, the summary says {pct}%"); ok = False
  else:
    notes.append(f"share: {n}/{lim} = {100 * n / lim:.1f}%")
  facts, inv = load(run, "machine_facts.json"), load(run, "weight_inventory.json")
  spec = (run / "mtp_context.json").is_file() or ("speculat" in text.lower() and "MTP" in text)
  split = ("split by" in text and "GPUs" in text) or "tensor_parallel" in json.dumps(load(run, "trace_request.json") or {})
  moe = "moe" in str((inv or {}).get("architecture_class"))
  if facts and inv and inv.get("total_known_weight_bytes") and not (spec or split or moe or b > 1):
    gbs = facts["gpus"][0]["read_gbs"]
    bound = gbs * 1e9 / inv["total_known_weight_bytes"]
    if not (0.90 * bound <= lim <= bound * 1.001):
      notes.append(f"BAD limit: {lim} outside [{0.9 * bound:.1f}, {bound:.1f}] (weights-only bound)"); ok = False
    else:
      notes.append(f"limit: {lim} <= {gbs:.0f} GB/s / {inv['total_known_weight_bytes'] / 1e9:.2f} GB = {bound:.1f}")
  else:
    notes.append("limit bound not rechecked: " + ("speculative (per produced token)" if spec else
                 f"batch {b} (per stream, below the weights-only bound)" if b > 1 else
                 "MoE (a token reads only its active experts)" if moe else
                 "split over GPUs" if split else "no bandwidth or weight bytes"))
  for what, s, total, k in tie_out(text):
    if abs(s - total) > 0.0005 * k + 0.001:
      notes.append(f"BAD {what}: parts give {s:.3f} ms, the line says {total:.3f}"); ok = False
    else:
      notes.append(f"{what}: {k} parts give {s:.3f} ms = {total:.3f}")
  t = table_sum(text)
  if t:
    s, total, k = t
    if abs(s - total) > 0.005 * k + 0.01:
      notes.append(f"BAD table: {k} rows sum to {s:.2f} ms, its total {total:.2f}"); ok = False
    else:
      notes.append(f"table: {k} rows sum to {s:.2f} ms against {total:.2f}")
  c = CAPTURE.search(text)
  if c:
    over = float(c.group(1)) * (1 if c.group(2) == "over" else -1)
    replicated = "Kernel rows qualified by replication" in text
    notes.append(("capture: " if abs(over) <= 5 else "capture: kernel rows qualified by replication, gaps bounded, " if replicated
                  else "WARN capture: per-kernel table unqualified, ") + f"{over:+.2f}% against the 5% ceiling")
  return f"{n} tok/s, limit {lim}, {pct}%", notes, ok, (n, lim, pct)


def check_batches(rows, points, head_batch, head_lim, notes):
  """The "Per batch" rows (BATCH) against timing_trace.json's points and themselves; see the docstring's `batch`."""
  ok, limits = True, {}
  for ctx, b, each, total, step, lim_ms, lim_total, pct in rows:
    p = next((p for p in points if p.get("context") == ctx and p.get("batch") == b), None)
    bad = [] if p and abs(p["tok_s_stream"] - each) <= 0.005 + 1e-9 and abs(p["step_ms"] - step) <= 0.0005 + 1e-9 else ["not the run's point"]
    bad += [] if abs(total - b * each) <= 0.05 + 0.005 * b else [f"total {total} is not {b:g} x {each}"]
    bad += [] if abs(lim_total - b * 1000 / lim_ms) <= 0.05 + 0.0005 * lim_total / lim_ms else [f"limit total {lim_total} is not {b:g} / {lim_ms} ms"]
    bad += [] if abs(pct - 100 * total / lim_total) <= 0.05 + pct * (0.05 / total + 0.05 / lim_total) else [f"{pct}% is not {total} / {lim_total}"]
    notes.append(f"{'BAD ' if bad else ''}batch {b:g} at {ctx:g}: " + ("; ".join(bad) if bad else
                 f"{b:g} x {each} = {total}, {total} / {lim_total} = {pct}%")); ok = ok and not bad
    limits.setdefault(ctx, []).append((b, lim_ms))
  for ctx, ls in limits.items():  # the weights once and B streams' own reads: limit(B) = limit(1) + (B - 1) x per stream
    (b0, l0), (b1, l1) = min(ls), max(ls)
    off = [b for b, l in ls if b1 > b0 and abs(l - (l0 + (b - b0) * (l1 - l0) / (b1 - b0))) > 0.002]
    notes.append(f"BAD batch limits at {ctx:g} not affine in the batch: {off}" if off else f"batch limits at {ctx:g} affine in the batch")
    ok = ok and not off
  if head_batch > 1 and rows and not any(b == head_batch and abs(head_lim - lim_total / b) <= 0.05 + 1e-9 for _, b, _, _, _, _, lim_total, _ in rows):
    notes.append(f"BAD headline limit: {head_lim} tok/s per stream is no batch {head_batch} row's limit"); ok = False
  return ok


def check_tui(pane_text, head):
  """A captured results screen against the run's headline."""
  notes, ok = [], True
  n, lim, pct = head
  m = TUI_HEAD.search(pane_text)
  if not m:
    return ["BAD tui: no 'tok/s measured · roofline' line on the screen"], False
  if abs(float(m.group(1)) - n) > 0.05 or abs(float(m.group(2)) - lim) > 0.05:
    notes.append(f"BAD tui headline: screen {m.group(1)} / {m.group(2)}, run {n} / {lim}"); ok = False
  else:
    notes.append(f"tui headline: {m.group(1)} tok/s, roofline {m.group(2)}, as the run")
  p = TUI_PCT.search(pane_text)
  if p and int(p.group(1)) != pct:
    notes.append(f"BAD tui share: screen {p.group(1)}%, run {pct}%"); ok = False
  cums = [float(x.group(1)) for x in map(TUI_CUM.search, pane_text.splitlines()) if x]
  if cums:
    if any(b + 1e-9 < a for a, b in zip(cums, cums[1:])):
      notes.append("BAD tui table: cumulative tok/s goes down"); ok = False
    elif abs(cums[-1] - lim) > 0.2 * 1 + 0.005 * lim:
      notes.append(f"tui table: the cumulative column on screen ends at {cums[-1]} (roofline {lim}): scroll for the rest")
    else:
      notes.append(f"tui table: cumulative reaches {cums[-1]} against the roofline {lim}")
  return notes, ok


def runs(paths):
  for p in map(pathlib.Path, paths):
    if (p / "run_manifest.json").is_file():
      yield p
    else:
      yield from sorted(m.parent for m in p.rglob("run_manifest.json"))


def main(argv):
  pane = None
  if argv[:1] == ["--tui"]:
    pane, argv = pathlib.Path(argv[1]).read_text(), argv[2:]
  bad = 0
  for run in runs(argv or ["."]):
    head, notes, ok, figures = check(run)
    if pane is not None and figures:
      more, tui_ok = check_tui(pane, figures)
      notes += more; ok = ok and tui_ok
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {run}: {head}")
    for x in notes:
      print(f"      {x}")
  print("all figures check" if not bad else f"{bad} run(s) do not check")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
