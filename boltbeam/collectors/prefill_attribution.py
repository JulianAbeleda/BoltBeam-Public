"""Which role each kernel of a captured prefill ran for, and each kernel's ceiling.

A capture from outside the engine (nsys) sees kernels, not roles, and a prefill GEMM's launch grid need not show its
shape (a stream-k kernel launches one block per SM whatever the matrix). The pairing is the layer walk
(collectors/pairing.py, docs/pairing-layer-walk-20261011.md): the block kernels mark the layers, the launches around
them are the layer's projections, the profile's counts and fine classes say which. This module owns what is prefill's:

  the window     a capture that holds the engine's warm-up prefill and the measured one (the adapter says which) is
                 cut to its second half, when the two halves launch the same kernels the same number of times (an
                 engine may idle on the host inside a prefill, before its last token's tail, so the longest idle
                 stretch is not the edge); a session's turn is the last of two identical turns (REPEAT)
  templates      launches are grouped by kernel template (instruction_path.template), and a template's matrix path
                 is read from its code (instruction_path.kernel_path); a weight launch is one whose template holds a
                 matrix instruction or whose code was not read, and is not a block kernel
  the rows       a role's launches summed: operations 2 M N K per launch at the launch's chunk width, bytes from the
                 engine's formats (prefill_work.gemm), and the ceiling max(operations / the path's measured peak,
                 bytes / bandwidth); attention and the recurrent block are rows with the model's work; a stream-k
                 fixup has its bytes ceiling; every other kernel keeps its time, named by kind

A kernel whose code could not be read (a library that hides it) may still take a role, its work counted and its
ceiling refused. A weight kernel no role took keeps its time and says so.
"""
from __future__ import annotations

import collections
from typing import Any

from boltbeam.collectors import pairing
from boltbeam.collectors.attribution import share_overlap
from boltbeam.collectors.instruction_path import FROM_NAME, kernel_path, template
from boltbeam.collectors.pairing import PEAK_SLACK
from boltbeam.roofline import prefill_work as work
from boltbeam.workflow.tie_out import kind_of

STEM_CHARS = 40  # a row named by its template is cut to this many characters; the trace keeps the whole name
REPEAT = "repeat"  # measured_window: the capture ends in two identical turns (a session's turn, captured twice)
UNPAIRED = "no ceiling: a matrix kernel no role could be paired with, so its work is not known"


def measured_window(launches:list[dict[str, Any]], warmup:bool | str = True) -> tuple[list[dict[str, Any]], str]:
  """The measured prefill. With warmup (the engine ran a warm-up prefill inside the capture): the second half of the
  launches, when the first half launched the same kernels the same number of times, else refused. Without: every
  launch (a prefill of identical chunks has identical halves too, so halves alone cannot say there was a warm-up).
  REPEAT (the capture holds other work, then the measured prefill twice): the shortest tail of launches that the
  launches just before it repeat kernel for kernel, in order (equal counts alone would also match a window that
  straddles the two turns); a once-per-call kernel (the output head) makes that tail one whole call."""
  timed = sorted((l for l in launches if "start_us" in l), key=lambda l: l["start_us"])
  if not warmup:
    return timed, f"one prefill of {len(timed)} launches (the capture holds the measured prefill only)"
  count = lambda ls: sorted(collections.Counter(template(l["key"]) for l in ls).items())  # noqa: E731
  if warmup == REPEAT:
    keys = [template(l["key"]) for l in timed]
    for n in range(1, len(timed) // 2 + 1):
      if keys[-n:] == keys[-2 * n:-n]:
        return timed[-n:], (f"the last of two identical turns at the end of the capture ({n} launches each); the "
                            f"{len(timed) - n} launches before it fill the cache and repeat the turn")
    raise ValueError(f"the capture's {len(timed)} launches do not end in two identical turns")
  half = len(timed) // 2
  if len(timed) % 2 == 0 and half and count(timed[:half]) == count(timed[half:]):
    gap = timed[half]["start_us"] - (timed[half - 1]["start_us"] + timed[half - 1]["wall_us"])
    return timed[half:], (f"the second of two identical prefills ({half} launches each, {gap / 1000:.1f} ms apart); the "
                          "first is the engine's warm-up")
  raise ValueError(f"the capture's {len(timed)} launches are not a warm-up prefill and a measured one of the same kernels")


def _peak(t:dict[str, Any], peaks:dict[str, float]) -> float | None:
  """The template's path peak; for a mixed template the peak its paths make together for its instruction mix
  (prefill_work.effective_peak: each path's share of the operations at its own peak)."""
  p = t["path"]
  if p["status"] == "one":
    return peaks.get(p["path"])
  if p["status"] == "mixed":
    return work.effective_peak(p["paths"], peaks)
  return None


def attribute_prefill(launches:list[dict[str, Any]], roles:list[dict[str, Any]], *, chunk_widths:list[int], outputs:int = 1,
                      layers:int, kernel_paths:dict[str, dict[str, int]], peaks:dict[str, float],
                      bandwidth_gbs:float | None, profile:dict[str, Any] | None = None, length:int | None = None,
                      program_roles:dict[str, dict[str, Any]] | None = None, untraced_ms:float | None = None,
                      warmup:bool | str = True, formats:dict[str, Any] | None = None,
                      cache:dict[str, float] | None = None, depth:int = 0,
                      observed:dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
  """The measured prefill split by role, attention and other kernels, each with operations, bytes, path and ceiling.
  roles: {role, quant, n, k, count (per chunk), weight_bytes (per launch), role_class or fine}. kernel_paths: template
  -> {path: sites}. outputs: the engine calls that each ended in an output (its last token's work, the output head, once
  per call). program_roles: template -> {role, quant, shape [M, N, K]} where the engine names each program's role and
  logical shape itself (the fork's own metadata): those launches are that role at that M, N, K, and the role is not
  paired again. untraced_ms: for an engine's own timing, whose run is slowed by profiling, the prefill the rows sum to:
  the gaps are it less the busy time, never the profiled run's idle. formats: the engine adapter's {activations: {in,
  out}, kv, stream_k: template -> layout or None} (prefill.ENGINES); cache: the chip's last-level cache {bytes, gbs}.
  depth: tokens already in the cache (a session's turn), so attention's work counts every key the turn's tokens read.
  observed: template -> {path: instructions} ncu saw a closed kernel execute (instruction_path.ncu_observe)."""
  formats = formats or {}
  acts = formats.get("activations")
  gemm = lambda m, n, k, wb: work.gemm(m, n, k, wb, acts)["bytes"]  # noqa: E731
  window, window_words = measured_window(launches, warmup)
  shared, overlap = share_overlap(window)
  ordered = sorted(shared, key=lambda l: l["start_us"])  # the launches in stream order: a fixup follows its GEMM
  begin = min(l["start_us"] for l in window)
  end = max(l["start_us"] + l["wall_us"] for l in window)
  mean_m = sum(chunk_widths) / max(len(chunk_widths), 1)
  by_t: dict[str, dict[str, Any]] = {}
  for l in shared:
    t = by_t.setdefault(template(l["key"]), {"template": template(l["key"]), "name": l["key"].split(" grid=")[0], "launches": [],
                                              "calls": 0, "wall_us": 0.0})
    t["launches"].append(l)
    t["calls"] += 1
    t["wall_us"] += l["wall_us"]
  templates = sorted(by_t.values(), key=lambda t: -t["wall_us"])
  for t in templates:
    t["path"] = kernel_path(t["name"], kernel_paths, observed)
    t["attention"] = kind_of({"kernel": t["name"]}) == "attention"  # attention's work is its own row, never a role's
    t["recurrent"] = kind_of({"kernel": t["name"]}) == "recurrent"  # so is a recurrent block's scan and convolution
  rows: list[dict[str, Any]] = []
  paired: set[int] = set()  # launch ids a row holds
  shapes: dict[int, list[tuple[dict[str, Any], float]]] = {}  # a paired GEMM launch -> its roles, weighted by count
  named: dict[tuple[str, str], dict[str, Any]] = {}
  for t in templates:  # the engine's own role and (M, N, K) per program
    info = (program_roles or {}).get(t["template"])
    if not info or t.get("attention"):
      continue
    m, n, k = info["shape"]
    r = next((x for x in roles if (x["role"], x["quant"]) == (info["role"], info.get("quant"))), None)
    wb = r["weight_bytes"] if r else 0.0
    slot = named.setdefault((info["role"], str(info.get("quant"))), {"calls": 0, "us": 0.0, "flops": 0.0, "bytes": 0.0, "t": []})
    slot["calls"] += t["calls"]
    slot["us"] += t["wall_us"]
    slot["flops"] += t["calls"] * 2.0 * m * n * k
    slot["bytes"] += t["calls"] * gemm(m, n, k, wb)
    slot["t"].append(t)
    paired.update(id(l) for l in t["launches"])
  for (role, quant), slot in named.items():
    ts = slot["t"]
    t = {"template": " + ".join(x["template"] for x in ts), "path": _merge_paths([x["path"] for x in ts])}
    rows.append(_row(f"{role} {quant}", "role", t, slot["calls"], slot["us"], slot["flops"], slot["bytes"], peaks,
                     bandwidth_gbs, roles=[(role, quant)], role_source="the engine's own program metadata (role, M, N, K)"))
  roles = [{**r, "nk": r["n"] * r["k"]} for r in roles if (r["role"], str(r["quant"])) not in named]
  kinds = pairing.layer_kinds(profile or {}) or ({"attention": layers} if layers else {})
  weight = {t["template"] for t in templates if t["path"]["status"] in ("one", "mixed", "not_found")
            and not t.get("attention") and not t.get("recurrent") and not any(id(l) in paired for l in t["launches"])}
  got = pairing.pair(shared, roles, kinds=kinds, widths=chunk_widths, outputs=outputs, candidate=weight.__contains__,
                     bandwidth_gbs=bandwidth_gbs, peak_of=lambda t: _peak(by_t[t], peaks) if t in by_t else None)
  per_item: dict[tuple, dict[str, Any]] = {}  # one row per role set, over every layer kind it ran in
  for p in got["placed"]:
    key = tuple(sorted((x["role"], str(x["quant"])) for x in p["item"]["parts"]))
    it = per_item.setdefault(key, {"item": p["item"], "launches": [], "templates": set(), "need": 0, "kinds": set()})
    it["launches"] += [(l, m / p["split"], p["fused"], p["split"]) for l, m, _, _ in p["launches"]]
    it["templates"].add(p["group"]["template"])
    if p["item"]["kind"] not in it["kinds"]:
      it["kinds"].add(p["item"]["kind"])
      it["need"] += p["item"]["need"]
  for it in per_item.values():
    item = it["item"]
    parts = item["parts"]
    for l, _, _, _ in it["launches"]:
      paired.add(id(l))
      shapes[id(l)] = [(r, r["count"] / item["count"]) for r in parts]
    calls = len(it["launches"])
    flops = sum(2.0 * m * item["nk"] * k for _, m, k, _ in it["launches"])
    nbytes = sum(sum(gemm(m, r["n"], r["k"], r["weight_bytes"]) * r["count"] for r in parts) / item["count"] * k for _, m, k, _ in it["launches"])
    us = sum(l["wall_us"] for l, _, _, _ in it["launches"])
    t = {"template": " + ".join(sorted(it["templates"])), "path": _merge_paths([by_t[x]["path"] for x in sorted(it["templates"])])}
    _, _, k0, r0 = it["launches"][0]
    rows.append(_row(f"{item['role']} {item['quant']}", "role", t, calls, us, flops, nbytes, peaks, bandwidth_gbs,
                     roles=[(r["role"], r["quant"]) for r in parts], skipped=max(it["need"] * r0 // k0 - calls, 0),
                     tokens_per_call=sorted(m for _, m, _, _ in it["launches"])[calls // 2],
                     **({"fused": k0} if k0 > 1 else {}), **({"split": r0} if r0 > 1 else {})))
  unpaired = [f"{u['item']['role']} {u['item']['quant']}: {u['reason']}" for u in got["unplaced"]]
  ambiguous = ({"reason": "; ".join(unpaired)} if unpaired else None) or got.get("ambiguous")
  others: dict[str, dict[str, Any]] = {}
  att = work.attention(profile or {}, length or int(sum(chunk_widths)), chunk_widths, formats.get("kv") or "F16",
                       formats.get("attention_activations") or acts, depth=depth) if profile else None
  rec = work.recurrent(profile or {}, length or int(sum(chunk_widths)), chunk_widths) if profile else None
  for t in templates:
    for l in t["launches"]:
      if id(l) in paired:
        continue
      # attention's kernels are one row with attention's work, except one with no matrix instruction (its fixup):
      # that, like every other kernel, is its own row, named by its template's stem
      kind = ("attention" if t.get("attention") and t["path"]["status"] != "none" else
              "recurrent" if t.get("recurrent") and rec else stem(t["template"]))
      o = others.setdefault(kind, {"calls": 0, "wall_us": 0.0, "templates": {}})
      o["calls"] += 1
      o["wall_us"] += l["wall_us"]
      o["templates"][t["template"]] = t["path"]
  for kind, o in others.items():
    paths = {p["words"] for p in o["templates"].values()}
    if kind == "attention" and att:
      t = {"path": _merge_paths(list(o["templates"].values())), "template": " + ".join(sorted(o["templates"]))}
      rows.append(_row("attention", "attention", t, o["calls"], o["wall_us"], att["flops"], att["bytes"], peaks, bandwidth_gbs,
                       note=att["words"]))
    elif kind == "recurrent" and rec:
      rows.append(_recurrent_row(o, rec, peaks, bandwidth_gbs, work.recurrent_executed(
        profile or {}, length or int(sum(chunk_widths)), formats.get("recurrent_form"))))
    elif (fix := _fixup_row(kind, o, ordered, shapes, formats, mean_m, bandwidth_gbs, cache)) is not None:
      rows.append(fix)
    else:
      matrix = any(p.get("status") in ("one", "mixed") for p in o["templates"].values())
      rows.append({"name": kind, "what": "kernels", "calls": o["calls"], "ms": o["wall_us"] / 1000.0, "flops": None,
                   "bytes": None, "path": " | ".join(sorted(paths)), "ceiling_ms": None, "mfu_pct": None,
                   "templates": sorted(o["templates"]), "refused": UNPAIRED if matrix else None})
  busy = sum(r["ms"] for r in rows)
  wall = (end - begin) / 1000.0 if untraced_ms is None else float(untraced_ms)
  return {"window": window_words, "launches": len(window), "wall_ms": wall, "profiled_wall_ms": (end - begin) / 1000.0,
          "wall_source": "the untraced prefill (the engine's own timing slows its run)" if untraced_ms is not None else
                         "the captured prefill, first kernel to last", "busy_ms": busy, "gaps_ms": wall - busy,
          "overlap": overlap, "chunks": chunk_widths, "rows": sorted(rows, key=lambda r: -r["ms"]),
          "ambiguous": ambiguous, "rule": got["words"], "structure": got["structure"],
          "templates": [{"template": t["template"], "calls": t["calls"], "ms": t["wall_us"] / 1000.0, "path": t["path"]}
                        for t in templates]}


def stem(tmpl:str) -> str:
  """A template's name without its arguments or a trailing content hash: `mul_mat_q_stream_k_fixup<...>` is
  `mul_mat_q_stream_k_fixup`, `E_64_64_8_16_4_292d...16a14` is `E_64_64_8_16_4`."""
  import re
  name = re.sub(r"_[0-9a-f]{32,}$", "", tmpl.split("<", 1)[0])
  return name if len(name) <= STEM_CHARS else name[:STEM_CHARS - 1] + "…"


def _merge_paths(paths:list[dict[str, Any]]) -> dict[str, Any]:
  known: dict[str, int] = {}
  for p in paths:
    for k, n in (p.get("paths") or {}).items():
      known[k] = known.get(k, 0) + n
  if not known:
    return paths[0] if paths else {"status": "not_found", "paths": {}, "path": None, "words": "no code"}
  named = [p for p in paths if p.get("from_name") and p.get("paths")]
  tag = {"from_name": True, "name_source": named[0]["name_source"]} if named else {}
  label = f" ({FROM_NAME})" if named else ""
  if len(known) == 1:
    k = next(iter(known))
    return {"status": "one", "paths": known, "path": k, "words": k + label, **tag}
  return {"status": "mixed", "paths": known, "path": None, "words": " + ".join(sorted(known)) + label, **tag}


def _row(name:str, what:str, t:dict[str, Any], calls:int, wall_us:float, flops:float, nbytes:float,
         peaks:dict[str, float], bandwidth_gbs:float | None, **extra:Any) -> dict[str, Any]:
  """One row: time, operations, bytes, the path its instructions use, and its ceiling. A row whose kernel code was
  not found has no compute ceiling and says why (refused); a kernel with no matrix instruction is bound by bytes."""
  path = t["path"]
  peak = _peak(t, peaks)
  ms = wall_us / 1000.0
  if path["status"] == "not_found" or (path["status"] in ("one", "mixed") and peak is None):
    c = {"ms": None, "compute_ms": None, "memory_ms": None, "bound": None}
    refused = f"no ceiling: {path['words'] if path['status'] == 'not_found' else 'no measured peak for ' + path['words']}"
  else:
    c = work.ceiling(flops if peak else 0.0, nbytes, peak, bandwidth_gbs,
                     work.compute_ms(flops, path["paths"], peaks) if path["status"] == "mixed" else None)
    refused = None
    if path.get("from_name") and c["ms"] and c["ms"] > ms * PEAK_SLACK:  # a name's path may not beat the guards
      refused = (f"no ceiling: the path {path['words']} would put the kernel {c['ms']:.2f} ms over its {ms:.2f} ms "
                 f"({'above the peak' if c['bound'] == 'compute' else 'faster than the bandwidth'})")
      c = {"ms": None, "compute_ms": None, "memory_ms": None, "bound": None}
  return {"name": name, "what": what, "calls": calls, "ms": ms, "flops": flops, "bytes": nbytes, "template": t["template"],
          "path": path["words"], "path_status": path["status"], "peak_tflops": peak,
          "ceiling_ms": c["ms"], "compute_ms": c["compute_ms"], "memory_ms": c["memory_ms"], "bound": c["bound"],
          "mfu_pct": work.mfu(flops, ms, peak) if path["status"] in ("one", "mixed") and not refused else None,
          "refused": refused, "mixed": path["status"] == "mixed", **({"name_source": path["name_source"]} if path.get("from_name") else {}),
          **({"path_shares": work.path_shares(path["paths"])} if path["status"] == "mixed" else {}), **extra}


def _recurrent_row(o:dict[str, Any], rec:dict[str, Any], peaks:dict[str, float], bandwidth_gbs:float | None,
                   executed:dict[str, Any] | None = None) -> dict[str, Any]:
  """The recurrent kernels as one row with the recurrent block's work (prefill_work.recurrent). Their code issues no
  matrix instruction, so the compute side is the measured fp32 vector rate; a kernel whose code was found to use a
  matrix path keeps that path. With no measured fp32 rate the row is bound by its bytes alone and says so."""
  merged = _merge_paths(list(o["templates"].values()))
  if merged["status"] in ("one", "mixed"):
    path = merged
  elif merged["status"] == "not_found":
    path = merged  # the code was not read: no compute side is claimed
  elif rec["path"] in peaks:
    path = {"status": "one", "paths": {rec["path"]: 1}, "path": rec["path"], "words": "fp32 vector (no matrix instruction)"}
  else:
    path = {"status": "none", "paths": {}, "path": None, "words": "no matrix instruction; no measured fp32 vector rate"}
  t = {"path": path, "template": " + ".join(sorted(o["templates"]))}
  row = _row("recurrent", "recurrent", t, o["calls"], o["wall_us"], rec["flops"], rec["bytes"], peaks, bandwidth_gbs,
             note=rec["words"])
  if executed:  # what the engine's algorithm executes, beside the model's work; the ceiling is not changed by it
    row.update(executed_flops=executed["flops"], executed_words=executed["words"])
  return row


FIXUP_WORD = "stream_k_fixup"  # a kernel that combines the partial tiles a stream-k GEMM left (kind word, in its name)


def _fixup_row(kind:str, o:dict[str, Any], window:list[dict[str, Any]], shapes:dict[int, list[tuple[dict[str, Any], float]]],
               formats:dict[str, Any], mean_m:float, bandwidth_gbs:float | None,
               cache:dict[str, float] | None) -> dict[str, Any] | None:
  """A stream-k fixup's row with its bytes ceiling: each launch follows its GEMM launch on the stream; that GEMM's
  roles (N, K, weight block) and launch grid give the partial tiles it left (prefill_work.stream_k_partials), read
  once, and the tiles fixed, read and written, in the engine's partial format. Partials that fit the last-level cache,
  written by the GEMM just before, are read at its bandwidth; else at the memory's. None when this is not a fixup,
  or the adapter names no stream-k layout for it (its row then says so)."""
  if FIXUP_WORD not in kind:
    return None
  layout_of = formats.get("stream_k")
  tmpls = sorted(o["templates"])
  base = {"name": kind, "what": "kernels", "calls": o["calls"], "ms": o["wall_us"] / 1000.0, "flops": None, "path": "no matrix instruction",
          "templates": tmpls, "mfu_pct": None, "compute_ms": None, "bound": "memory"}
  layouts = {t: layout_of(t) if layout_of else None for t in tmpls}
  if not all(layouts.values()):
    return {**base, "bytes": None, "ceiling_ms": None,
            "refused": f"no ceiling: the engine adapter names no stream-k layout for {', '.join(t for t, v in layouts.items() if not v)}"}
  from boltbeam.quantization.quant import load_quant_registry
  quants = load_quant_registry()
  per_elem = work.elem_bytes(formats.get("partials") or "F32")
  nbytes, cached, unshaped, launches = 0.0, 0.0, 0, 0
  for i, l in enumerate(window):
    t = template(l["key"])
    if t not in layouts:
      continue
    launches += 1
    prev = window[i - 1] if i else None
    roles = shapes.get(id(prev)) if prev is not None else None
    if not roles:
      unshaped += 1
      continue
    lay = layouts[t]
    blocks = int(str(prev["key"]).split(" grid=", 1)[1].split()[0].split("x")[0]) if " grid=" in str(prev["key"]) else 0
    for r, wt in roles:
      q = quants.get(str(r["quant"]).upper())
      kb = q.block_elems if q else 1
      got = work.stream_k_partials(int(round(mean_m)), r["n"], r["k"], blocks=blocks, tile_m=lay["tile_m"], tile_n=lay["tile_n"],
                                   k_block=kb, k_iter=max(lay["k_iter"] // kb, 1))
      tile = lay["tile_m"] * lay["tile_n"] * per_elem
      b = (got["partials"] + 2 * got["tiles"]) * tile * wt
      nbytes += b
      if cache and blocks * tile <= cache.get("bytes", 0):
        cached += b
  bw_c, bw = (cache or {}).get("gbs"), bandwidth_gbs
  ms = (cached / (bw_c * 1e9) if bw_c else 0.0) * 1e3 + ((nbytes - cached) / (bw * 1e9) * 1e3 if bw else 0.0)
  return {**base, "bytes": nbytes, "ceiling_ms": ms if unshaped == 0 else None, "memory_ms": ms,
          "cached_bytes": cached,
          "note": (f"{launches} launches, each after its GEMM: partial tiles read once and tiles fixed read and written "
                   f"({formats.get('partials') or 'F32'}); {cached / max(nbytes, 1) * 100:.0f}% of the bytes fit the "
                   "last-level cache, read at its bandwidth"),
          "refused": None if unshaped == 0 else f"no ceiling: {unshaped} of its launches follow a GEMM no role was paired with"}
