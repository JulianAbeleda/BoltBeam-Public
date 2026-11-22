"""The kernel ledger: every kernel a decode step launches, grouped by shape, with its launches, time, bytes, limit and
distance from it, in two rankings (bytes known; bound unknown), and whether the search can propose a candidate.

    capture     an nsys sqlite export (collectors/time_budget.timeline_from_nsys, with each launch's grid) or an extract
                of one (JSON, gzip allowed: names, kernel_cols, kernels, copy_cols, copies, counters); one per depth. With
                the session's traced turns, only their decode windows are read (read: a capture that holds prefills).
    steps       the output layer is the matmul launch whose output rows (gridX x rows_per_block, the engine's `shapes`
                rows in data/kernel_families.json) are the header's vocabulary. Launches up to one are a forward (a
                cut by result reads, time_budget.forwards, also splits a forward at every small read, such as an expert
                cache's lookup); one with at least half the most per-layer launches is a verify, the rest draft passes.
                A forward starts at the first idle gap after the output layer before it that holds an input copy (else
                the longest); a step runs from one verify's output layer end to the next. The draft passes a step ran are
                checked against the engine's counters (decode_steps over the request's turns, the decode row's
                draft_forwards): a cut that disagrees fails its check.
    time        exclusive: launches in start order, each counts what it adds to the union (first-started wins, so a
                downstream PDL wait is charged to the launch upstream); overlap across streams is measured and checked.
                Every time is the traced run's; the untraced step, given, discloses the trace's overhead.
    join        a matmul group is tied to the header's tensors by launches per forward (the most common count a step, so
                a step a cut shortens does not unjoin it; cells by class, format, K, N; 1-D tensors are N = 1): N = gridX x
                rows_per_block (or the row's own n: an installed manifest's kernel, manifest_rows), the row's formats, K >= min_k, columns >= min_cols,
                counts that sum to the launches (to tensors_per_launch x launches for a fused launch), the one subset that
                does, repeated until nothing changes. Unmatched is "unattributed". Chip facts are the row's by_target.
    bytes       matmul: the joined weights + activations read and written (derived). Experts: weights / experts x
                distinct experts a launch reads: under a measured routing profile, the experts a forward of its columns
                reaches (derived); else between experts_used and experts_used x columns (estimate, at the high end). Attention: KV cells read x bytes a cell (estimate: cells = the selection when given, else the
                depth). Else unknown. limit = bytes / the target's measured bandwidth; distance = time - limit.
    floor       launches x the cheapest mean duration of launches no other launch overlaps: an estimate and a lower
                bound, never a limit; a row with no bytes ranks by time - floor, an upper bound of what it can give.
    coverage    per depth: search_can_propose (the region search covers the row: data/region.json has the engine's row, a
                generator row with a generate action, and the kernel among the engine's graph-op kernels; then
                formats.refusal, decode_graph.json kinds), check (a reference), install_path (engine_patches.json).

Engine-blind: an engine is a key into data rows; nothing here names one or a kernel.
Labels: measured (source), derived (formula, inputs), estimate (formula, test), unknown (test, no value)."""
from __future__ import annotations

import bisect, collections, gzip, itertools, json, math, pathlib, re, statistics
from typing import Any

from boltbeam import formats
from boltbeam.collectors import time_budget as tb
from boltbeam.collectors.kernel_kind import kind_of
from boltbeam.profile.decode_roles import _physical_bytes
from boltbeam.profile.roles import role_from_tensor_name
from boltbeam.workflow.context_audit import derived, measured, unknown, unlabelled, within

SCHEMA = "boltbeam.kernel_ledger.v1"
MATMUL = ("matmul", "expert_matmul")
SUBSET_MAX = 16  # the most cells a subset search tries (2^16 subsets); more is unmatched, never guessed
EXCLUSIVE = "first-started wins: a downstream PDL wait is charged to the launch upstream"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def estimate(value:Any, formula:str, test:str, **kw) -> dict[str, Any]:
  return {"value": value, "label": "estimate", "formula": formula, "test": test, **kw}


def read(path:str | pathlib.Path, turns:list[dict[str, Any]] | None = None, part:str = "decode", tl:tb.Timeline | None = None) -> dict[str, Any]:
  """Kernels (start, end, name, grid, stream) and copies (start, end, bytes, kind) of a capture, ns, and its counters.
  turns (an nsys export only: traced turns, time_budget.turn_windows): only each turn's `part` window is kept, the turns
  joined end to end with the time between them taken out (the next turn's prefill and the reply's way back), so a capture
  that holds prefills is cut as a decode; the step across a join is shortened by that time (one of each turn's steps).
  Its counters are the turns' (tokens, accepted, drafted)."""
  p = pathlib.Path(path)
  if p.suffix == ".sqlite":
    if not (tl := tl or tb.timeline_from_nsys(p)).grids and tl.kernels:
      raise ValueError(f"{p.name}: the export has no launch grids (gridX): the join needs them")
    ks = sorted((a, e, n, g, st) for (a, e, n, st), g in zip(tl.kernels, tl.grids))
    cs = sorted((c["start_ns"], c["end_ns"], c["bytes"], c["kind"]) for c in tl.copies)
    if turns is None:
      return {"source": p.name, "kernels": ks, "copies": cs}
    wins = [w[part] for w in tb.turn_windows(tl, turns) if w]
    if len(wins) < len(turns):
      raise ValueError(f"{p.name}: {len(turns) - len(wins)} of {len(turns)} turns are not in the trace (no clock origin, or no GPU work in the span)")
    out_k, out_c, shift, prev = [], [], 0, None
    for a, b in sorted(wins):
      shift += a - prev if prev is not None else 0
      keep = lambda xs: [(x[0] - shift, x[1] - shift, *x[2:]) for x in xs[bisect.bisect_left(xs, (a,)):bisect.bisect_left(xs, (b,))]]  # noqa: E731
      out_k, out_c, prev = out_k + keep(ks), out_c + keep(cs), b
    return {"source": f"{p.name}, {part} windows of {len(wins)} turns", "kernels": out_k, "copies": out_c,
            "counters": [{"tokens": t["tokens"], "accepted": t["accepted"], "drafted": t["drafted"]} for t in turns]}
  d = json.loads((gzip.open(p, "rt") if p.suffix == ".gz" else p.open()).read())
  kc, cc = {c: i for i, c in enumerate(d["kernel_cols"])}, {c: i for i, c in enumerate(d["copy_cols"])}
  ks = sorted((r[kc["start"]], r[kc["start"]] + r[kc["dur"]], d["names"][r[kc["name"]]], (r[kc["gx"]], r[kc["gy"]], r[kc["gz"]]), r[kc["stream"]])
              for r in d["kernels"])
  cs = [(r[cc["start"]], r[cc["start"]] + r[cc["dur"]], r[cc["bytes"]], r[cc["kind"]]) for r in d["copies"]]
  return {"source": p.name, "from": d.get("from"), "counters": d.get("counters"), "kernels": ks, "copies": cs}


# --- the model header ----------------------------------------------------------------------------------------------

def cells(table:dict[str, Any], file:str) -> list[dict[str, Any]]:
  """The header's matrices by (class, format, K, N): count and bytes (a 1-D tensor is K x 1; the roles no matmul reads
  are left out, kernel_families.json not_matmul_roles)."""
  out: dict[tuple, dict[str, Any]] = {}
  skip = tb._table("kernel_families.json")["not_matmul_roles"]["roles"]
  for name, dims, typ in table["tensors"]:
    if not 1 <= len(dims) <= 3 or (role := role_from_tensor_name(name, tuple(dims))) in skip:
      continue
    d = [*dims, 1]
    key = (re.sub(r"^blk\.\d+\.", "", name).removesuffix(".weight"), formats.header_type(typ), d[0], d[1], len(dims))
    c = out.setdefault(key, {"class": key[0], "format": key[1], "k": d[0], "n": d[1], "experts": dims[2] if len(dims) == 3 else 0,
                             "count": 0, "bytes": 0, "role": role, "file": file})
    c["count"] += 1
    c["bytes"] += _physical_bytes(math.prod(dims), typ, tensor_name=name)
  return list(out.values())


def fact(table:dict[str, Any] | None, name:str) -> Any:
  """A metadata fact by data/min_memory.json's name for it under the header's architecture (None when not given)."""
  from boltbeam.workflow.min_memory import block, one
  arch = ((table or {}).get("meta") or {}).get("general.architecture")
  key = (block((table or {}).get("meta"), (table or {}).get("tensors")) or {}).get("meta", {}).get(name)
  return one(table["meta"].get(f"{arch}.{key}")) if key else None


shape_rows, shape_of, per_columns = tb.shape_rows, tb.shape_of, tb.per_columns


def manifest_rows(manifest:dict[str, Any] | None) -> list[dict[str, Any]]:
  """Shapes rows for the kernels an installed manifest (boltbeam.kernel_manifest.v1, the engine's external-kernel hook)
  puts in place of the engine's: a kernel whose match holds a matmul is that matmul (its weight's format, its output rows);
  any other is a region of elementwise nodes. The manifest is the engine's own statement of what each kernel computes."""
  out = []
  for e in (manifest or {}).get("entries") or []:
    mm = [o for o in e["match"]["ops"] if o["op"].endswith("MUL_MAT")]
    syms = [q["kernel"]["symbol"] for q in e["launches"] if "kernel" in q] if "launches" in e else [e["kernel"]["symbol"]]  # hook R5: one a launch
    row = {"pattern": "^(" + "|".join(map(re.escape, syms)) + ")$", "manifest": e["id"], "why": f"the installed manifest's entry {e['id']}"}
    if any(o["op"].endswith("FLASH_ATTN_EXT") for o in e["match"]["ops"]):
      out.append({**row, "op": "attention", "family": "attention"})
    elif len(mm) == 1 and len(mm[0].get("src_ne") or []) == 2:
      fmt = mm[0]["src_types"][0].upper()
      out.append({**row, "op": "matmul", "family": "dense_gemm", "act": mm[0]["src_types"][1].upper(), "n": [mm[0]["src_ne"][0][1]],
                  "format_families": [(formats.row(fmt) or {}).get("family")]})
    else:
      out.append({**row, "op": "region", "family": "elementwise"})
  return out


# --- steps and groups -----------------------------------------------------------------------------------------------

def cut(ks:list[tuple], fams:list[str], rows:list[dict[str, Any]], vocab:int, layer:set[str], ups:list[int]) -> dict[str, Any]:
  """Each launch's forward, which forwards are verifies, and the steps (see the module)."""
  heads = tb.output_layers([(n, g) for _, _, n, g, _ in ks], rows, vocab)
  if len(heads) < 3:
    return {"refused": f"{len(heads)} output-layer launches (a matmul of the vocabulary's {vocab} rows): a step needs two verifies after a first"}
  ends = list(itertools.accumulate((e for _, e, *_ in ks), max))
  starts = [0]
  for h0, h1 in zip(heads, heads[1:]):
    end, best, at = ends[h0], -1, h0 + 1
    for i in range(h0 + 1, h1 + 1):
      if ks[i][0] > end and bisect.bisect_left(ups, ks[i][0]) > bisect.bisect_left(ups, end):
        at = i  # the first idle gap that holds an input copy
        break
      if ks[i][0] - end > best and fams[i] not in layer:
        best, at = ks[i][0] - end, i
      end = max(end, ks[i][1])
    starts.append(at)
  seg = [bisect.bisect_right(starts, i) - 1 for i in range(len(ks))]
  layers = collections.Counter(seg[i] for i, f in enumerate(fams) if f in layer)
  main = [j for j in range(len(heads)) if layers.get(j, 0) * 2 >= max(layers.values())]
  return {"seg": seg, "main": set(main), "steps": [(heads[a], heads[b]) for a, b in zip(main, main[1:])],
          "drafts": [b - a - 1 for a, b in zip(main, main[1:])]}


def groups(cap:dict[str, Any], *, engine:str | list[str], vendor:str | None, vocab:int, rows:list[dict[str, Any]]) -> dict[str, Any]:
  """Per group (phase, name, grid) and step: launches, exclusive ms, summed ms, launch gaps, durations of lone launches;
  per step: wall, union, idle, overlap across streams."""
  ks, rule = cap["kernels"], tb.forward_rule()
  fams = tb.launch_families([(a, e, n, st) for a, e, n, _, st in ks], tb.families_for(engine, vendor, decode=True))
  fams = [(shape_of(n, rows) or {}).get("family", f) for (_, _, n, _, _), f in zip(ks, fams)]  # a manifest kernel's own family
  c = cut(ks, fams, rows, vocab, set(rule["layer_families"]), sorted(a for a, _, _, kind in cap["copies"] if kind == rule["input_copy"]))
  if "refused" in c:
    return c
  g: dict[tuple, dict[str, Any]] = {}
  n = len(c["steps"])
  per = {q: [0.0] * n for q in ("wall", "union", "idle", "cross")}
  for s, (h0, h1) in enumerate(c["steps"]):
    a, b = ks[h0][1], ks[h1][1]
    cur, held, busy = a, None, 0
    for i in range(h0 + 1, h1 + 1):
      st, en, name, grid, stream = ks[i]
      r = g.setdefault(("verify" if c["seg"][i] in c["main"] else "draft", name, *grid),
                       {"launches": 0, "per_step": [0] * n, "excl": [0.0] * n, "dur": 0.0, "gap": 0.0, "lone": [], "family": fams[i]})
      ex = max(0, min(en, b) - max(st, cur))
      r["launches"] += 1
      r["per_step"][s] += 1
      r["excl"][s] += ex / 1e6
      r["dur"] += (en - st) / 1e6
      r["gap"] += max(0, st - cur) / 1e6
      if st >= cur and (i + 1 >= len(ks) or en <= ks[i + 1][0]):
        r["lone"].append((en - st) / 1e3)
      if st < cur and held is not None and held != stream:
        per["cross"][s] += (min(en, cur) - st) / 1e6
      busy += ex
      held = stream if en > cur else held
      cur = max(cur, en)
    per["wall"][s], per["idle"][s] = (b - a) / 1e6, (b - a - busy) / 1e6
    per["union"][s] = busy / 1e6
  return {"groups": g, "steps": n, **per, "drafts": c["drafts"]}



def join(gs:dict[tuple, dict[str, Any]], by_phase:dict[str, list[dict[str, Any]]], rows:list[dict[str, Any]], steps:int,
         forwards:dict[str, list[tuple[int, list[int]]]], cols:dict[str, int]) -> dict[tuple, tuple[list[dict[str, Any]], int]]:
  """Each matmul group's cells and launches per forward (see the module). forwards: per phase, (columns, its forwards each step) for
  each column regime it runs: the verify at k + 1; the draft head's passes at one column and its catch-up at more (decode row)."""
  taken: set[tuple[int, int]] = set()
  out: dict[tuple, tuple[list[dict[str, Any]], int]] = {}
  fam = lambda f: (formats.row(f) or {}).get("family")  # noqa: E731
  while True:
    done = False
    for key in sorted(k for k in gs if (sr := shape_of(k[1], rows)) and sr["op"] in MATMUL and k not in out):
      sr, phase, tpl, hits = shape_of(key[1], rows), key[0], max(shape_of(key[1], rows).get("tensors_per_launch", [1])), []
      for c, f in forwards[phase]:
        cand = [x for x in by_phase[phase] if (id(x), c) not in taken and (x["experts"] > 0) == (sr["op"] == "expert_matmul")
                and x["n"] in tb.out_rows(sr, key[2], c) and fam(x["format"]) in sr["format_families"]
                and x["k"] >= sr.get("min_k", 0) and c >= (sr.get("min_cols") or {}).get(x["format"], 0 if "min_cols" not in sr else math.inf)]
        per = [x / y for x, y in zip(gs[key]["per_step"], f) if y]  # launches a forward, each step (f: its forwards of this regime)
        w = statistics.mode(per) if per else 0  # the most common: a step the cut shortens does not unjoin it
        if abs(w - round(w)) > 0.05 or round(w) < 1 or not cand or len(cand) > SUBSET_MAX:
          continue
        w = round(w)  # a fused launch: the whole set, its count between w and tpl x w; else the subsets that sum to w
        hits += [(cand, w, c)] if tpl > 1 and w <= sum(x["count"] for x in cand) <= tpl * w else [] if tpl > 1 else \
          [(list(h), w, c) for m in range(1, len(cand) + 1) for h in itertools.combinations(cand, m) if sum(x["count"] for x in h) == w]
      if len(hits) == 1:
        out[key] = hits[0]
        taken |= {(id(x), hits[0][2]) for x in hits[0][0]}  # a tensor one group reads at a column count
        done = True
    if not done:
      return out


# --- bytes, limit, coverage -------------------------------------------------------------------------------------------

def bytes_of(op:str, joined:tuple | None, sr:dict[str, Any] | None, launches:float, cols:int, depth:int | None, kv:dict[str, Any]) -> dict[str, Any]:
  """Bytes a step's launches of this group move (see the module), labelled."""
  if op in MATMUL and joined:
    cs, w, cols = joined  # the columns the join found it at
    fwd = launches / w  # forwards a step
    act = formats.elem_bytes(sr["act"]) if sr.get("act") else None
    io = None if act is None else sum(x["count"] * (x["k"] + x["n"]) for x in cs) * cols * act * fwd
    if op == "matmul":
      return derived(sum(x["bytes"] for x in cs) * fwd + (io or 0), "(joined weight bytes + (K + N) x columns x act bytes each) x forwards a step",
                     {"tensors": [x["class"] for x in cs], "columns": cols, "act": sr.get("act"), "forwards_a_step": fwd})
    used, n_exp = kv.get("experts_used"), max(x["experts"] for x in cs)
    if not used:
      return unknown("the header's expert_used_count")
    hi, lo = min(n_exp, used * cols), used
    per = sum(x["bytes"] / x["experts"] for x in cs) * fwd
    if (reach := (kv.get("reached") or {}).get(cols)) is not None:
      return derived(per * reach + (io or 0), "weight bytes / experts x the distinct experts a forward of these columns reaches under the measured routing "
                     "profile (routing_profile.reached, the layers' mean) + activations", {"reached": reach, "columns": cols, "forwards_a_step": fwd})
    return estimate(per * hi + (io or 0), "weight bytes / experts x distinct experts a forward (at most experts_used x columns) + activations",
                    "the routing counts (distinct experts a launch reads per step)", range=[per * lo + (io or 0), per * hi + (io or 0)])
  if op == "attention" and kv.get("cell_bytes") and depth:
    cells_read = min(kv["selection"], depth) if kv.get("selection") else depth
    return estimate(cells_read * kv["cell_bytes"] * launches, "KV cells read (the selection, else the depth) x bytes a cell (kv_heads x (key + value) x KV format) x launches",
                    "the cells each launch reads (the engine's sparse selection at this depth)", inputs={"cells": cells_read, "cell_bytes": kv["cell_bytes"]})
  return unknown("a join of this group to the header's tensors (a shapes row for its kernel)" if op in MATMUL else
                 "the shapes this kernel reads and writes (the engine's graph), not in the trace or the header")


def region_covers(engine:str, name:str) -> str | None:
  """Why the region search (the only search since 2026-11-21) does not cover a kernel of this engine, or None. From
  data/region.json alone: the engine's row, a generator row with a generate action, the kernel among the engine's
  graph-op kernels (engines.<E>.graph_ops.kernels)."""
  path = DATA / "region.json"
  if not path.is_file():
    return "region search: not in this build (no data/region.json)"
  rows = json.loads(path.read_text())
  row = (rows.get("engines") or {}).get(engine)
  if not row:
    return f"region search: data/region.json has no engines row for {engine}"
  if not any(g.get("generate") for g in (rows.get("region_generators") or {}).values()):
    return "region search: no generator row with a generate action (data/region.json region_generators)"
  kernels: set[str] = set()
  for v in ((row.get("graph_ops") or {}).get("kernels") or {}).values():
    kernels |= {v} if isinstance(v, str) else {k for ks in (v.values() if isinstance(v, dict) else [v]) for k in ([ks] if isinstance(ks, str) else ks)}
  if not any(name == k or name.startswith(k + "<") for k in kernels):
    return f"region search: {name} is no graph op's kernel in data/region.json engines.{engine}.graph_ops"
  return None


def coverage(op:str, name:str, joined:tuple | None, m:int, engine:str, ctx:dict[str, Any]) -> dict[str, Any]:
  """search_can_propose (whether the region search covers the row: region_covers, for the engine that ran), check (a
  reference) and install_path (data/engine_patches.json). m is the columns a launch takes (the row's pass)."""
  from boltbeam.search import fusion_space
  cs = joined[0] if joined else []
  fmts, check = sorted({x["format"] for x in cs}), []
  miss = [why] if (why := region_covers(ctx.get("provider", engine), name)) else []
  if op in MATMUL:
    miss += [f"format: {r}" for f in fmts if (r := formats.refusal("search", f))] or ([] if fmts else ["format: unknown (not joined to the header)"])
    check += [f"reference: {r}" for f in fmts if (r := formats.refusal("reference", f))]
  elif op != "attention":
    kinds = fusion_space.graph().get("kinds") or {}
    ops = kinds.get(kind := kind_of({"kernel": name})) or []
    miss += [f"graph node: no data/decode_graph.json node for kind '{kind}'"] if not ops else \
      [f"check: no reference for {', '.join(ops)}"] if not set(ops) & set(fusion_space.REFERENCE_OPS) else []
  from boltbeam.workflow import mods
  inst = (mods.pointer().get("install_paths") or {}).get(engine)
  return {"search_can_propose": not miss, "missing": miss, "check": check or ["a reference exists" if op in MATMUL and fmts else "n/a"],
          "install_path": {"status": inst["status"], "mod": inst["mod"], "switch": next((mods.switch(m) for m in mods.named(engine, inst["mod"])), None),
                           "missing": inst["missing"], "evidence": inst["evidence"]} if inst else
          {"status": "none", "missing": f"an install path for {engine} (data/engine_patches.json install_paths)"}}


# --- the ledger ---------------------------------------------------------------------------------------------------

def ledger(captures:list[dict[str, Any]], *, main:dict[str, Any], draft:dict[str, Any] | None, engine:str | list[str], vendor:str | None,
           target:dict[str, Any], k:int, kv_type:str | None = None, selection:dict[str, int] | None = None, lowering:Any = None,
           manifest:dict[str, Any] | None = None, routing:dict[str, Any] | None = None) -> dict[str, Any]:
  """The record (boltbeam.kernel_ledger.v1). captures: {cap, depth, untraced_ms?, counters?}, shallowest first; ranked at the last.
  selection: the KV cells a phase's attention launch reads when it is sparse ({"verify": cells}); a phase not named reads all.
  manifest: the kernel manifest the engine ran with (its kernels joined by what each entry computes, manifest_rows).
  routing: the model's measured routing profile (roofline/routing_profile): expert bytes from the experts a forward
  reaches under it (derived), not the experts_used x columns upper estimate."""
  eng, rows_t = tb.engine_keys(engine)[-1], shape_rows(engine, target.get("id"), manifest_rows(manifest))
  vocab = next((x["n"] for x in cells(main, "main") if x["role"] == "lm_head"), None)
  if not vocab:
    return {"schema": SCHEMA, "refused": "the main header has no output layer (lm_head) to find the steps by"}
  bw = measured(target["memory_bandwidth_gbs"], f"{target['id']} memory_bandwidth_gbs (roofline/limit.targets)") if target.get("memory_bandwidth_gbs") else \
    unknown("the target's measured read bandwidth (cuda_bandwidth)")
  cols, row_d = {"verify": k + 1, "draft": 1}, tb.decode_row(engine) or {}
  by_phase = {"verify": cells(main, "main"), "draft": cells(draft, "draft") if draft else []}
  if draft and not any(x["role"] == "lm_head" for x in by_phase["draft"]):  # a draft file with no output layer reads the target's
    by_phase["draft"] += [{**x, "file": "main"} for x in by_phase["verify"] if x["role"] == "lm_head"]
  kvb = formats.elem_bytes(kv_type) if kv_type else None
  hk = [fact(main, x) for x in ("kv_heads", "key_len", "value_len")]
  kv = {"experts_used": fact(main, "experts_used"), "cell_bytes": hk[0] * (hk[1] + hk[2]) * kvb if kvb and all(hk) else None}
  if routing:
    from boltbeam.roofline import routing_profile as rp
    kv["reached"] = {c: statistics.fmean(rp.reached(routing, c)) for c in sorted({k + 1, 1})}
  selection = selection or {}
  ctx = {"kv_type": kv_type, "provider": tb.engine_keys(engine)[0]}   # the engine that ran: a fork, not its root
  depths, rows = [], {}
  for c in captures:
    cap, depth = c["cap"], c.get("depth")
    got = groups(cap, engine=engine, vendor=vendor, vocab=vocab, rows=rows_t)
    if "refused" in got:
      return {"schema": SCHEMA, "refused": f"{cap['source']}: {got['refused']}"}
    n, gs, d = got["steps"], got["groups"], str(depth)
    src = f"{cap['source']}: {n} steps, traced"
    catchups = (row_d.get("draft_forwards") or {}).get("per_main_forward", 0)  # a catch-up pass a verify, with no output layer of its own
    tied = join(gs, by_phase, rows_t, n, {"verify": [(k + 1, [1] * n)], "draft": [(1, got["drafts"]), (k + 1, [catchups] * n)]}, cols)
    lone = [(statistics.fmean(r["lone"]), key) for key, r in gs.items() if r["lone"] and r["launches"] >= n]
    fl = estimate(min(lone)[0], "the cheapest group's mean duration over its launches no other launch overlaps, us", "an empty kernel launched in the same graph",
                  inputs={"group": list(min(lone)[1])}) if lone else unknown("a group of a launch a step no other launch overlaps")
    for key, r in gs.items():
      sr = shape_of(key[1], rows_t) or {}
      op = "expert_matmul" if r["family"] == "expert_mmq" else sr.get("op")
      ms, launches, sd = sum(r["excl"]) / n, r["launches"] / n, statistics.stdev(r["excl"]) if n > 1 else None
      by = bytes_of(op, tied.get(key), sr, launches, cols[key[0]], depth, {**kv, "selection": selection.get(key[0]), "reached": kv.get("reached") if key[0] == "verify" else None})
      lim = (estimate if by["label"] == "estimate" else derived)(by["value"] / (bw["value"] * 1e9) * 1e3, "bytes / bandwidth",
                                                                   **({"test": by["test"]} if by["label"] == "estimate" else {"inputs": ["bytes", "bandwidth"]})) \
        if by["value"] is not None and bw["value"] else unknown(by.get("test") or "the target's bandwidth")
      dist = {**lim, "value": None if lim["value"] is None else ms - lim["value"], "formula": "time - limit"}
      floor = estimate(None if fl["value"] is None else launches * fl["value"] / 1e3, "launches x launch floor (a lower bound)", fl["test"])
      cs = (tied.get(key) or ([],))[0]
      row = rows.setdefault(key, {"phase": key[0], "kernel": key[1], "grid": list(key[2:]), "family": r["family"], "op": op or "none (no shapes row)", "depths": {}})
      row["serves"] = {"tensors": [{"class": x["class"], "format": x["format"], "count": x["count"], "file": x["file"]} for x in cs], "role": sorted({x["role"] for x in cs})} \
        if cs else ("unattributed" if op in MATMUL or r["family"] == "other" else {"family": r["family"]})
      row["depths"][d] = {"launches": measured(launches, src), "time_ms": {**measured(ms, src), "step_sd": sd}, "summed_ms": measured(r["dur"] / n, src),
                          "launch_gap_ms": measured(r["gap"] / n, src), "bytes": by, "limit_ms": lim, "distance_ms": dist, "floor_ms": floor,
                          "above_floor_ms": {**floor, "value": None if floor["value"] is None else ms - floor["value"], "formula": "time - floor: an upper bound"},
                          "coverage": coverage(op, key[1], tied.get(key), (tied.get(key) or (0, 0, cols[key[0]]))[2], eng, {**ctx, "selection": selection.get(key[0])})}
    depths.append(one_depth(c, got, rows, d, fl, bw, src, row_d, k, by_phase["verify"]))
  return finish(depths, rows, captures, eng, vendor, target, cols)


def one_depth(c:dict[str, Any], got:dict[str, Any], rows:dict[tuple, Any], d:str, fl:dict[str, Any], bw:dict[str, Any], src:str,
              row_d:dict[str, Any], k:int, main_cells:list[dict[str, Any]]) -> dict[str, Any]:
  """A depth's totals and checks."""
  n, cap = got["steps"], c["cap"]
  here = [r for r in rows.values() if d in r["depths"]]
  t = lambda pick, q="time_ms": sum(r["depths"][d][q]["value"] or 0.0 for r in here if pick(r))  # noqa: E731
  se = lambda xs: statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None  # noqa: E731
  band = lambda xs, what: {**(measured(v, f"{src}: standard error of {what}") if (v := se(xs)) is not None else unknown("a capture of at least two steps")), "n": n}  # noqa: E731
  step, union, idle = (statistics.fmean(got[q]) for q in ("wall", "union", "idle"))
  mat = [sum(gr["excl"][s] for key, gr in got["groups"].items() if (rows[key]["op"] in MATMUL)) for s in range(n)]
  loose = t(lambda r: r["op"] in MATMUL and r["serves"] == "unattributed")
  never = sorted({x["class"] for x in main_cells if x["n"] > 1} - {s["class"] for r in here if isinstance(r["serves"], dict) for s in r["serves"].get("tensors", []) if s["file"] == "main"})
  below = [f"{r['phase']} {r['kernel']} {r['grid'][0]}: {x['time_ms']['value']:.3f} < {x['limit_ms']['value']:.3f}" for r in here
           if (x := r["depths"][d])["limit_ms"]["label"] == "derived" and x["time_ms"]["step_sd"] is not None
           and x["time_ms"]["value"] < x["limit_ms"]["value"] - 3 * x["time_ms"]["step_sd"] / math.sqrt(n)]
  names = ", ".join(f"{r['phase']} {r['kernel']} {r['grid'][0]}" for r in here if r["op"] in MATMUL and r["serves"] == "unattributed")
  checks = [within("join coverage: matmul time no row joined", loose, {**band(mat, "the matmul time a step"), "value": 3 * (se(mat) or 0) if n > 1 else None},
                   f"{loose:.3f} of {statistics.fmean(mat):.3f} ms unjoined, 3 standard errors allowed" + (f" ({names})" if names else ""), " ms"),
            {"check": "join coverage: every main-file matrix joined", "ok": not never, "why": "each of the header's matrix classes (N > 1; a vector is a norm's or a gate's) serves a group" if not never else "never joined: " + ", ".join(never)},
            {"check": "no row below its limit", "ok": not below, "why": "every derived limit is at or under its time (3 standard errors)" if not below else "; ".join(below)},
            within("overlap across streams", statistics.fmean(got["cross"]), band(got["wall"], "the step"), f"{statistics.fmean(got['cross']):.4f} ms a step", " ms"),
            counts_check(c.get("counters") or cap.get("counters"), got["drafts"], row_d, k), traced_check(c.get("untraced_ms"), step, band(got["wall"], "the step"))]
  totals = {"step_ms": measured(step, src), "union_ms": measured(union, src), "idle_ms": measured(idle, f"{src}: these steps, not the session"),
            "limit_ms": derived(t(lambda r: r["depths"][d]["limit_ms"]["label"] == "derived", "limit_ms"), "sum of the derived limits", "rows"),
            "limit_estimate_ms": estimate(t(lambda r: r["depths"][d]["limit_ms"]["label"] == "estimate", "limit_ms"), "sum of the estimated limits", "their rows' tests"),
            "unattributed_ms": measured(t(lambda r: r["serves"] == "unattributed"), src)}
  return {"depth": c.get("depth"), "source": cap["source"], "from": cap.get("from"), "steps": measured(n, src), "launch_floor_us": fl, "bandwidth_gbs": bw,
          "exclusive": EXCLUSIVE, "totals": totals, "checks": checks}


def counts_check(counters:Any, drafts:list[int], row:dict[str, Any], k:int) -> dict[str, Any]:
  """Draft passes a step in the cut against the engine's counters ([{tokens, accepted, drafted}] over the capture's turns):
  their mean lies between drafted / steps x per_draft_token less 3 standard errors and that plus uncounted_per_step, no
  step runs more than k x per_draft_token + uncounted_per_step."""
  per = row.get("draft_forwards") or {}
  most = k * per.get("per_draft_token", 1) + per.get("uncounted_per_step", 0)
  if max(drafts) > most:
    return {"check": "draft passes against the counters", "ok": False, "why": f"a step ran {max(drafts)} draft passes, more than k x per_draft_token + uncounted ({most}): a wrong --k or cut"}
  if not counters:
    return {"check": "draft passes against the counters", "ok": False, "why": "not judged: no counters (test: the request's tokens, accepted and drafted)"}
  steps = sum(tb.decode_steps(c["tokens"], c["accepted"], row) for c in counters)
  rate = sum(c["drafted"] for c in counters) / steps * per.get("per_draft_token", 1)
  mean, se = statistics.fmean(drafts), statistics.stdev(drafts) / math.sqrt(len(drafts)) if len(drafts) > 1 else 0.0
  ok = rate - 3 * se <= mean <= rate + per.get("uncounted_per_step", 0)
  return {"check": "draft passes against the counters", "ok": ok, "why": f"cut {mean:.2f} a step against drafted / decode_steps(tokens, accepted) = {rate:.2f} "
          f"(+ up to {per.get('uncounted_per_step', 0)} uncounted)"}


def traced_check(untraced:float | None, step:float, band:dict[str, Any]) -> dict[str, Any]:
  if untraced is None:
    return {"check": "traced against untraced", "ok": False, "why": "not judged: no untraced step (test: the same turns untraced); every time here is traced"}
  over = 100.0 * (step / untraced - 1)
  return {"check": "traced against untraced", "ok": step >= untraced - 3 * (band["value"] or 0), "overhead_pct": over,
          "why": f"traced {step:.3f} against untraced {untraced:.3f} ms: {over:+.1f}% (a trace may not run faster than untraced); every time here is traced"}


def finish(depths:list[dict[str, Any]], rows:dict[tuple, Any], captures:list[dict[str, Any]], eng:str, vendor:str | None, target:dict[str, Any],
           cols:dict[str, int]) -> dict[str, Any]:
  """Slopes, the two rankings, the family roll-up, the missing pieces and the checks."""
  last, first = str(captures[-1].get("depth")), str(captures[0].get("depth"))
  for row in rows.values():
    if len(captures) > 1 and first in row["depths"] and last in row["depths"] and None not in (captures[0].get("depth"), captures[-1].get("depth")):
      a, b = (row["depths"][q]["time_ms"]["value"] for q in (first, last))
      row["slope_ms_per_1k"] = derived((b - a) / ((int(last) - int(first)) / 1e3), "(time at the deepest - at the shallowest) / depths apart, per 1k tokens", [first, last])
    else:
      row["slope_ms_per_1k"] = unknown("captures at two depths with the group in both")
  here = [r for r in rows.values() if last in r["depths"]]
  known = sorted((r for r in here if r["depths"][last]["distance_ms"]["value"] is not None), key=lambda r: -r["depths"][last]["distance_ms"]["value"])
  bound = sorted((r for r in here if r["depths"][last]["distance_ms"]["value"] is None), key=lambda r: -(r["depths"][last]["above_floor_ms"]["value"] or -math.inf))
  fams: dict[tuple, dict[str, Any]] = {}
  miss: dict[str, list[float]] = {}
  for r in here:
    x = r["depths"][last]
    f = fams.setdefault((r["phase"], r["family"]), {"phase": r["phase"], "family": r["family"], **estimate(None, "sums of the family's rows at the deepest", "its rows'"),
                                                    "launches": 0.0, "time_ms": 0.0, "distance_ms": 0.0, "above_floor_ms": 0.0, "rows_bound_unknown": 0})
    f["launches"] += x["launches"]["value"]
    f["time_ms"] += x["time_ms"]["value"]
    f["distance_ms"] += x["distance_ms"]["value"] or 0.0
    f["above_floor_ms"] += (x["above_floor_ms"]["value"] or 0.0) if x["distance_ms"]["value"] is None else 0.0
    f["rows_bound_unknown"] += x["distance_ms"]["value"] is None
    for m in x["coverage"]["missing"] + ([] if x["coverage"]["install_path"]["status"] != "none" else ["install path: none"]):
      got = miss.setdefault(m.split(" (")[0], [0, 0.0, 0.0])
      got[0] += 1
      got[1 if x["distance_ms"]["value"] is not None else 2] += x["distance_ms"]["value"] if x["distance_ms"]["value"] is not None else max(0.0, x["above_floor_ms"]["value"] or 0.0)
  out = {"schema": SCHEMA, "engine": eng, "vendor": vendor, "target": target.get("id"), "columns": cols, "depths": depths,
         "byte_limited": known, "bound_unknown": bound, "families": sorted(fams.values(), key=lambda f: -f["distance_ms"]),
         "missing": [{"piece": m, **estimate(ms, "sum of the rows' distance (bytes known)", "each row's", rows=c, bound_unknown_ms=ub)}
                     for m, (c, ms, ub) in sorted(miss.items(), key=lambda x: -(x[1][1] + x[1][2]))]}
  bad = unlabelled({"totals": [d["totals"] for d in depths], "rows": [{q: v for q, v in x.items() if q != "coverage"} for r in rows.values() for x in r["depths"].values()]})
  out["checks"] = [c for d in depths for c in d["checks"]] + [{"check": "every figure labelled", "ok": not bad, "why": "; ".join(bad[:5]) or "each figure carries its label"}]
  out["ok"] = all(c["ok"] for c in out["checks"])
  return out


def lines(rec:dict[str, Any], top:int = 20) -> list[str]:
  if rec.get("refused"):
    return [f"kernel-ledger refused: {rec['refused']}"]
  d = rec["depths"][-1]
  t, last = d["totals"], str(d["depth"])
  v = lambda f: "?" if f is None or f.get("value") is None else f"{f['value']:.3f}"  # noqa: E731
  out = [f"kernel ledger, {rec['engine']} on {rec['target']}, depth {d['depth']}: {d['steps']['value']} steps of {d['source']} (traced; {d['exclusive']})",
         f"step {v(t['step_ms'])} ms = kernels {v(t['union_ms'])} + idle {v(t['idle_ms'])} (these steps); limits {v(t['limit_ms'])} derived + {v(t['limit_estimate_ms'])} estimated; "
         f"unattributed {v(t['unattributed_ms'])}; launch floor {v(d['launch_floor_us'])} us (estimate)"]

  def row(i:int, r:dict[str, Any], q:str) -> str:
    x = r["depths"][last]
    serves = r["serves"] if isinstance(r["serves"], str) else ",".join(f"{s['class']}:{s['format']}x{s['count']}" for s in r["serves"].get("tensors", [])) or r["serves"]["family"]
    cov = x["coverage"]
    return (f"{i:3d} {r['phase'][0]} {r['kernel'][:30]} {'x'.join(map(str, r['grid']))} {x['launches']['value']:.0f} {v(x['time_ms'])} {v(x['limit_ms'])}{x['limit_ms']['label'][0]} "
            f"{v(x[q])} {v(r['slope_ms_per_1k'])} {serves[:50]} search:{'yes' if cov['search_can_propose'] else 'no'} install:{cov['install_path']['status']}")
  out += ["byte-limited (distance = time - bytes / bandwidth): rank phase kernel grid launches ms limit distance slope/1k serves"]
  out += [row(i, r, "distance_ms") for i, r in enumerate(rec["byte_limited"][:top], 1)]
  out += ["bound unknown (time - launch floor: an upper bound):"] + [row(i, r, "above_floor_ms") for i, r in enumerate(rec["bound_unknown"][:top], 1)]
  out += [f"  {f['phase']} {f['family']}: {f['launches']:.0f} launches, {f['time_ms']:.3f} ms, distance {f['distance_ms']:.3f}"
          + (f" + up to {f['above_floor_ms']:.3f} in {f['rows_bound_unknown']} rows of unknown bound" if f["rows_bound_unknown"] else "") for f in rec["families"]]
  out += ["missing pieces: " + "; ".join(f"{m['piece']} ({m['rows']} rows, {m['value']:.2f} ms + up to {m['bound_unknown_ms']:.2f})" for m in rec["missing"])]
  return out + [f"{'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['why']}" for c in rec["checks"]]
