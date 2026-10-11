"""A replay run's results: per turn and for the session, where the time goes against the roofline.

Every number is read from the run's files (collectors/replay.py wrote them); nothing here measures. The standard is
the sum of the ceilings of the work each turn needed, at the clock it ran at:

  prefill needed   prefill_work.turn_ceiling of the tokens the turn added (needed) after the rest of its prompt
  prefill done     the same of what the server processed (prompt_n after cache_n); done less needed is the
                   recompute after a cache miss, lost under the needed view however well the GPU ran it
  decode           (predicted_n - 1) tokens, each reading the ledger's limit bytes, the KV cache at the turn's mean
                   prefix and the recurrent state in and out, at the measured bandwidth (the memory side; at batch 1
                   the compute side is far under it). The first token is the prefill's
  turn roofline    R = prefill needed + decode; the turn's time T is the client's, send to last token

Peaks are scaled to the turn's observed SM clock (prefill_work.peaks_at; the run's mean when the turn holds fewer
than MIN_SAMPLES samples); bandwidth is not scaled. Session efficiency E = sum R / sum T over the warm turns (the cold
first turn is a row, shown, and left out of the headline). T splits into prefill (prompt_ms), decode (predicted_ms)
and overhead (the rest); T - R splits into prefill above its done ceiling, recompute, decode above its ceiling and
overhead. Both splits add up.

The table is built once, as cells, and every renderer (the text summary, the HTML report, the TUI) prints those cells.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

from boltbeam.workflow import tie_out as tie

PER = "per session"
TITLE = "The replay: one row per turn"
COLUMNS = ("turn", "prefix", "cache", "new", "recomputed", "TTFT ms", "prompt ms", "overhead ms", "reply", "decode tok/s",
           "ceiling ms", "% of roofline", "clock", "flags")
MIN_SAMPLES = 5  # a turn's own clock needs this many telemetry samples; fewer: the run's mean, said
LOST = (("prefill_above", "prefill above its ceiling"), ("recompute", "recompute after a cache miss"),
        ("decode_above", "decode above its ceiling"), ("overhead", "overhead (HTTP, template, sampling, streaming)"))
DIFF_SCHEMA = "boltbeam.replay_diff.v1"


def _read(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


# --- the standard -------------------------------------------------------------------------------------------------

def standard(run:pathlib.Path, trace:dict[str, Any]) -> dict[str, Any] | None:
  """What every turn's ceiling is built from: the profile's roles and ledger, the engine's formats and paths, the
  measured peaks and bandwidth. None when the run lacks the profile or the ledger, with no guess."""
  from boltbeam.collectors import prefill as pf
  from boltbeam.collectors import replay as rp
  from boltbeam.roofline.prefill_work import kv_bytes_per_token, recurrent_state_bytes
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  from boltbeam.workflow.screen import _measured_vs_ceiling, run_bandwidth
  manifest = load_manifest(run)
  profile = _read(run / "model_profile.json")
  server = next((r for r in trace.get("rows") or [] if r.get("scope") == "server"), None)
  ceil = _measured_vs_ceiling({**manifest, "workload": "decode"}, profile, run)
  if not (profile and server) or ceil.get("status") != "modeled":
    return None
  target = get_target(manifest.get("target_id"))
  eng = rp.engine(trace.get("provider") or "llama.cpp")
  settings = server.get("settings") or {}
  bw, bw_source = run_bandwidth(run, target)
  type_k = str(settings.get("type_k") or "f16").upper()
  return {"profile": profile, "roles": pf.prefill_roles(profile, ceil.get("_roles") or []),
          "formats": eng["formats"](settings, pf.weight_quants(run)), "weight_path": eng["weight_path"],
          "attention_paths": server.get("attention_paths") or {}, "ubatch": int(settings.get("n_ubatch") or 512),
          "batch": int(settings.get("n_batch") or 2048), "bandwidth_gbs": bw, "bandwidth_source": bw_source,
          "peaks": dict(target.matrix_tflops or {}),
          "peaks_source": ((target.capabilities or {}).get("fact_sources") or {}).get("matrix_tflops"),
          "limit_bytes": float(ceil["bytes_moved"]), "kv_per_token": kv_bytes_per_token(profile, type_k) or 0.0,
          "kv_type": type_k, "state_bytes": recurrent_state_bytes(profile) or 0.0}


def decode_ms_per_token(s:dict[str, Any], prefix:float) -> float:
  return (s["limit_bytes"] + s["kv_per_token"] * prefix + 2 * s["state_bytes"]) / (s["bandwidth_gbs"] * 1e9) * 1e3


def turn_ceilings(s:dict[str, Any], r:dict[str, Any], run_mhz:float | None) -> dict[str, Any]:
  """One turn's ceilings at its clock: prefill done, prefill needed, decode, and the turn's roofline."""
  from boltbeam.roofline.prefill_work import peaks_at, turn_ceiling
  tele = r.get("telemetry") or {}
  own = (tele.get("sm_mhz") or {}).get("mean") if (tele.get("samples") or 0) >= MIN_SAMPLES else None
  mhz = own or run_mhz
  peaks, words = peaks_at(s["peaks"], s["peaks_source"], mhz)
  def pre(n:int, depth:int) -> dict[str, Any]:
    return turn_ceiling(s["profile"], s["roles"], n, depth, ubatch=s["ubatch"], batch=s["batch"], formats=s["formats"],
                        weight_path=s["weight_path"], attention_paths=s["attention_paths"], peaks=peaks,
                        bandwidth_gbs=s["bandwidth_gbs"])
  done, needed = pre(r["prompt_n"], r["cache_n"]), pre(r["needed"], r["prefix"] - r["needed"])
  steps = max(r["predicted_n"] - 1, 0)
  decode = steps * decode_ms_per_token(s, r["prefix"] + r["predicted_n"] / 2)
  return {"done_ms": done["ms"], "needed_ms": needed["ms"], "decode_ms": decode, "roofline_ms": needed["ms"] + decode,
          "clock_mhz": mhz, "clock_from": "the turn" if own else "the run's mean", "peaks_clock": words,
          "unmodelled": done["unmodelled"]}


def lost_parts(r:dict[str, Any], c:dict[str, Any]) -> dict[str, float]:
  return {"prefill_above": r["prompt_ms"] - c["done_ms"], "recompute": c["done_ms"] - c["needed_ms"],
          "decode_above": r["predicted_ms"] - c["decode_ms"], "overhead": r["wall_ms"] - r["prompt_ms"] - r["predicted_ms"]}


# --- the table and the headline ------------------------------------------------------------------------------------

def _pct(xs:list[float], q:float) -> float:
  xs = sorted(xs)
  return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def table(run:pathlib.Path, trace:dict[str, Any]) -> dict[str, Any] | None:
  """The replay table from timing_trace.json: a row per turn with its ceilings, a totals row, the headline."""
  rows = [r for r in trace.get("rows") or [] if r.get("workload") == "replay"]
  turns = [r for r in rows if r["scope"] == "turn"]
  if not turns:
    return None
  s = standard(run, trace)
  run_mhz = ((trace.get("telemetry") or {}).get("sm_mhz") or {}).get("mean")
  captured = {t["i"] for t in (_read(run / "replay_capture.json").get("turns") or []) if not t.get("failed")}
  out_rows, unmodelled = [], set()
  for r in turns:
    c = turn_ceilings(s, r, run_mhz) if s else None
    unmodelled.update((c or {}).get("unmodelled") or [])
    flags = [f for f, on in (("cold", r["cold"]), ("captured", r["i"] in captured), ("THROTTLED", r.get("throttled")),
                             ("compaction", r["compaction"])) if on]
    pct = 100.0 * c["roofline_ms"] / r["wall_ms"] if c and r["wall_ms"] else None
    out_rows.append({"i": r["i"], "turn": r, "ceiling": c, "lost": lost_parts(r, c) if c else None, "flags": flags,
                     "cells": [str(r["i"]), f"{r['prefix']:,}", f"{r['cache_n']:,}", f"{r['needed']:,}", f"{r['recomputed']:,}",
                               f"{r['ttft_ms']:.0f}", f"{r['prompt_ms']:.0f}", f"{r['overhead_ms']:.0f}", str(r["predicted_n"]),
                               f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") else "-",
                               f"{c['roofline_ms']:.0f}" if c else "-", f"{pct:.0f}%" if pct is not None else "-",
                               f"{c['clock_mhz']:.0f}" if c and c.get("clock_mhz") else "-", " ".join(flags)]})
  head = headline(out_rows, rows, trace, s)
  warm = [x for x in out_rows if not x["turn"]["cold"]] or out_rows
  tot = lambda k: sum(x["turn"][k] for x in warm)  # noqa: E731
  total = ["warm", "", f"{sum(x['turn']['cache_n'] for x in warm):,}", f"{tot('needed'):,}", f"{tot('recomputed'):,}",
           f"{tot('ttft_ms'):,.0f}", f"{tot('prompt_ms'):,.0f}", f"{tot('overhead_ms'):,.0f}", f"{tot('predicted_n'):,}", "",
           f"{head['roofline_ms']:,.0f}" if head.get("roofline_ms") is not None else "-",
           f"{head['e_pct']:.0f}%" if head.get("e_pct") is not None else "-", "", ""]
  stopped = next((r for r in rows if r["scope"] == "stopped"), None)
  src = (trace.get("replay_trace") or {}).get("source") or {}
  words = [f"{len(turns)} turns of {src.get('name') or 'the trace'} ({src.get('words', '')}) replayed on the user's own "
           f"server command, prompt cache on: {next((r.get('request') for r in rows if r['scope'] == 'server'), '')}. "
           "prefix: every token the turn's prompt holds; cache: what the cache supplied; new: the tokens the turn added; "
           "recomputed: tokens the cache held last turn and the server processed again; overhead: TTFT less prompt ms; "
           "ceiling: the turn's roofline at its clock (prefill of the new tokens plus decode); % of roofline: ceiling over "
           "the turn's time, send to last token."]
  if stopped:
    words.append(f"Stopped at turn {stopped['i']}: {stopped['reason']}.")
  if not s:
    words.append("No ceiling: the run has no model profile or no decode ledger.")
  return {"title": TITLE, "columns": list(COLUMNS), "rows": [{"cells": x["cells"], "i": x["i"], "flags": x["flags"]} for x in out_rows],
          "total": total, "words": " ".join(words), "headline": head, "headline_lines": headline_lines(head),
          "unmodelled": sorted(unmodelled),
          "turns": [{"i": x["i"], "prefix": x["turn"]["prefix"], "ttft_ms": x["turn"]["ttft_ms"],
                     "decode_tok_s": x["turn"].get("decode_tok_s"), "wall_ms": x["turn"]["wall_ms"],
                     "prompt_ms": x["turn"]["prompt_ms"], "predicted_ms": x["turn"]["predicted_ms"],
                     "overhead_ms": x["turn"]["wall_ms"] - x["turn"]["prompt_ms"] - x["turn"]["predicted_ms"],
                     "recomputed": x["turn"]["recomputed"], "cache_n": x["turn"]["cache_n"],
                     "clock_mhz": (x["ceiling"] or {}).get("clock_mhz"), "roofline_ms": (x["ceiling"] or {}).get("roofline_ms"),
                     "cold": x["turn"]["cold"]} for x in out_rows]}


def headline(out_rows:list[dict[str, Any]], rows:list[dict[str, Any]], trace:dict[str, Any], s:dict[str, Any] | None) -> dict[str, Any]:
  """The session's first lines: TTFT, decode rate, E, the time and loss splits, cache reuse, load, clock."""
  every = [x for x in out_rows]
  warm = [x for x in every if not x["turn"]["cold"]] or every
  t = [x["turn"] for x in warm]
  T = sum(r["wall_ms"] for r in t)
  pre, dec = sum(r["prompt_ms"] for r in t), sum(r["predicted_ms"] for r in t)
  rates = [r["decode_tok_s"] for r in t if r.get("decode_tok_s")]
  deep = max(t, key=lambda r: r["prefix"])
  out: dict[str, Any] = {
    "turns": len(every), "warm_turns": len(warm), "ttft_median_ms": statistics.median(r["ttft_ms"] for r in t),
    "ttft_p95_ms": _pct([r["ttft_ms"] for r in t], 0.95), "decode_tok_s_median": statistics.median(rates) if rates else None,
    "decode_tok_s_deepest": deep.get("decode_tok_s"), "deepest_prefix": deep["prefix"], "time_ms": T,
    "split": {"prefill": pre, "decode": dec, "overhead": T - pre - dec},
    "cache_reuse": sum(r["cache_n"] for r in t) / max(sum(r["prefix"] for r in t), 1),
    "load_s": next((r["seconds"] for r in rows if r["scope"] == "load"), None),
    "clock_policy": (trace.get("clock_policy") or {}).get("words") or (trace.get("clock_policy") or {}).get("policy"),
    "throttled_turns": sum(1 for r in t if r.get("throttled")), "roofline_ms": None, "e_pct": None, "lost": None,
    "cold": next(({"ttft_ms": x["turn"]["ttft_ms"], "wall_ms": x["turn"]["wall_ms"]} for x in every if x["turn"]["cold"]), None)}
  if s and all(x["ceiling"] for x in warm):
    R = sum(x["ceiling"]["roofline_ms"] for x in warm)
    lost = {k: sum(x["lost"][k] for x in warm) for k, _ in LOST}
    with_cold = sum(x["ceiling"]["roofline_ms"] for x in every) / max(sum(x["turn"]["wall_ms"] for x in every), 1e-9)
    out.update(roofline_ms=R, e_pct=100.0 * R / T if T else None, lost=lost, e_with_cold_pct=100.0 * with_cold,
               basis=(f"the profile's weight ledger ({s['limit_bytes'] / 1e9:.2f} GB per token), KV {s['kv_type']} at "
                      f"{s['kv_per_token']:,.0f} B per token, the recurrent state {s['state_bytes'] / 2**20:.1f} MiB in and out "
                      f"per token, the measured matrix peaks scaled to each turn's SM clock, bandwidth "
                      f"{s['bandwidth_gbs']:,.1f} GB/s ({s['bandwidth_source']}), not scaled; chunk {s['ubatch']} tokens, "
                      f"{s['batch']} per output call, as the server logged them"))
  return out


def headline_lines(h:dict[str, Any]) -> list[str]:
  ms = lambda v: f"{v / 1e3:,.1f} s" if v >= 1e4 else f"{v:,.0f} ms"  # noqa: E731
  T = h["time_ms"] or 1.0
  out = [f"TTFT median {h['ttft_median_ms']:,.0f} ms, p95 {h['ttft_p95_ms']:,.0f} ms over {h['warm_turns']} warm turns."]
  if h.get("decode_tok_s_median"):
    out.append(f"Decode {h['decode_tok_s_median']:.1f} tok/s median; {h['decode_tok_s_deepest'] or 0:.1f} tok/s at the deepest "
               f"turn ({h['deepest_prefix']:,} tokens).")
  if h.get("e_pct") is not None:
    out.append(f"Session at {h['e_pct']:.1f}% of its roofline: {ms(h['roofline_ms'])} of ceilings in {ms(T)} "
               f"({h['e_with_cold_pct']:.1f}% with the cold first turn).")
  sp = h["split"]
  out.append("Time: " + ", ".join(f"{k} {ms(v)} ({100 * v / T:.0f}%)" for k, v in sp.items()) + ".")
  if h.get("lost"):
    out.append("Lost: " + ", ".join(f"{w} {ms(h['lost'][k])}" for k, w in LOST) + ".")
  out.append(f"Cache reuse {100 * h['cache_reuse']:.1f}% of prompt tokens. Model load {h['load_s'] or 0:.1f} s. "
             f"Turns throttled: {h['throttled_turns']} of {h['warm_turns']}.")
  if h.get("cold"):
    out.append(f"Cold first turn (out of the headline): TTFT {h['cold']['ttft_ms']:,.0f} ms, {ms(h['cold']['wall_ms'])} in all.")
  if h.get("basis"):
    out.append(f"Standard: {h['basis']}.")
  return out


# --- where underneath ----------------------------------------------------------------------------------------------

def wheres(run:pathlib.Path, trace:dict[str, Any]) -> list[dict[str, Any]]:
  """The captured turns' where tables, tied out to the server's prompt_ms; a turn whose llama-bench time is off the
  server's by more than the tie-out share is `unmeasured`, with both numbers."""
  from boltbeam.collectors import replay as rp
  cap = _read(run / rp.CAPTURE)
  turns = {r["i"]: r for r in trace.get("rows") or [] if r.get("scope") == "turn"}
  out = []
  for t in cap.get("turns") or []:
    head = f"Turn {t['i']} ({t['name']})"
    if t.get("failed"):
      out.append({"title": head, "status": "failed", "words": f"{head}: not captured: {t['failed'][-200:]}"})
      continue
    attr = (_read(run / t["file"]).get("lengths") or {}).get(str(t["prompt_n"]))
    if not t["ties_out"]:
      out.append({"title": head, "status": "unmeasured",
                  "words": (f"{head}: unmeasured. llama-bench's turn of {t['prompt_n']:,} tokens after {t['cache_n']:,} took "
                            f"{t['bench_ms'] or 0:,.1f} ms; the server's prompt_ms was {t['server_ms']:,.1f} ms, more than "
                            f"{100 * cap.get('tie_out', rp.TIE_OUT):.0f}% apart, so the capture does not stand for the server's turn.")})
      continue
    w = tie.where_prefill_goes(attr, untraced_ms=t["server_ms"], length=int(t["prompt_n"]), depth=int(t["cache_n"]))
    if not w:
      out.append({"title": head, "status": "unmeasured", "words": f"{head}: unmeasured: the capture holds no attributed rows."})
      continue
    above = t["server_ms"] - (attr.get("wall_ms") or 0.0)
    w["title"] = f"{head}: where {t['prompt_n']:,} tokens after {t['cache_n']:,} go"
    w["status"] = "measured"
    w["server_line"] = (f"The server's prompt path above the bench's: {above:+,.1f} ms (server prompt_ms {t['server_ms']:,.1f}, "
                        f"llama-bench {attr.get('wall_ms') or 0:,.1f}).")
    w["turn"] = turns.get(t["i"], {}).get("i")
    out.append(w)
  return out


def decode_where(run:pathlib.Path, trace:dict[str, Any]) -> dict[str, Any] | None:
  """The deepest turn's decode, captured, per role against each role's limit, and the KV and state read at that
  depth beside what no role took."""
  from boltbeam.collectors import replay as rp
  from boltbeam.collectors.tinygrad_role_time import loss
  from boltbeam.workflow.common import load_manifest
  from boltbeam.workflow.screen import _measured_vs_ceiling
  d = _read(run / rp.CAPTURE).get("decode")
  if not d:
    return None
  if d.get("failed"):
    return {"status": "failed", "words": f"Decode at {d['context']:,}: not captured: {d['failed'][-200:]}"}
  s = standard(run, trace)
  manifest = load_manifest(run)
  ceil = _measured_vs_ceiling({**manifest, "workload": "decode"}, _read(run / "model_profile.json"), run)
  if not s:
    return None
  limit = decode_ms_per_token(s, d["context"])
  got = loss(ceil.get("_roles") or [], _read(run / d["file"]), limit, ceil.get("band"))
  if not got or got.get("status") != "measured":
    return {"status": "unmeasured", "words": f"Decode at {d['context']:,}: unmeasured: {(got or {}).get('reason') or 'no attributed rows'}"}
  kv_ms = (s["kv_per_token"] * d["context"] + 2 * s["state_bytes"]) / (s["bandwidth_gbs"] * 1e9) * 1e3
  rows = [[f"{r['role']} {r['quant']}", f"{r['actual_ms']:.3f}", f"{r['ideal_ms']:.3f}", f"{r['lost_ms']:+.3f}"] for r in got["roles"]]
  rows.append(["attention, state and the rest (no role)", f"{got['not_attributed_ms']:.3f}", f"{kv_ms:.3f}",
               f"{got['not_attributed_ms'] - kv_ms:+.3f}"])
  server = 1e3 / d["server_tok_s"] if d.get("server_tok_s") else None
  words = (f"One decode token at {d['context']:,} tokens, captured with llama-bench and the server's cache types: "
           f"{got['kernel_ms']:.2f} ms of kernels against a {limit:.2f} ms limit (the ledger, the KV cache and the state at "
           f"this depth). ms at limit for a role is its bytes over the bandwidth; the last row's limit is the KV and state read.")
  if server:
    words += f" The server's own decode at that turn: {server:.2f} ms per token."
  return {"status": "measured", "title": f"Where a decode token goes at {d['context']:,}", "columns": ["row", "ms now", "ms at limit", "ms lost"],
          "rows": rows, "words": words}


# --- the results block ---------------------------------------------------------------------------------------------

def loss_block(run:pathlib.Path, manifest:dict[str, Any], base:dict[str, Any]) -> dict[str, Any]:
  """The results' loss block for a replay run, on the decode block's keys (base) so every renderer reads it."""
  trace = _read(run / "timing_trace.json")
  t = table(run, trace)
  out = {**base, "workload": "replay", "per": PER, "runtimes": [], "roles": [], "tie_out": None, "where_token_goes": None,
         "replay": t}
  if not t:
    out.update(status="absent", reason="no turn replayed in this run")
    return out
  h = t["headline"]
  t["wheres"] = wheres(run, trace)
  t["decode_where"] = decode_where(run, trace)
  out.update(status="modeled" if h.get("roofline_ms") else "measured", limit_ms=h.get("roofline_ms"),
             runtimes=[{"provider": out.get("provider"), "tok_s": h.get("decode_tok_s_median"), "ms": h["time_ms"],
                        "lost_ms": h["time_ms"] - h["roofline_ms"] if h.get("roofline_ms") else None, "per_role": False,
                        "note": f"{h['warm_turns']} warm turns replayed on the user's server, client clock"}])
  if not (run / "replay_capture.json").exists():
    out["missing"] = "no turn captured: step 5 (role time) has not run for this run"
  return out


def lines(t:dict[str, Any] | None) -> list[str]:
  """The replay as plain text: the headline, the table (each column as wide as its widest cell), the where tables."""
  if not t:
    return []
  out = [t["title"] + ":"] + [f"  {l}" for l in t["headline_lines"]] + [f"  {t['words']}"]
  cells = [list(t["columns"])] + [r["cells"] for r in t["rows"]] + [t["total"]]
  width = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
  out += [("  " + " ".join(c.rjust(width[i]) for i, c in enumerate(row))).rstrip() for row in cells]
  if t.get("unmodelled"):
    out.append("  Not in the ceiling: " + "; ".join(t["unmodelled"]) + ".")
  for w in t.get("wheres") or []:
    if w.get("status") != "measured":
      out.append(f"  {w['words']}")
      continue
    out += tie_lines(w)
  d = t.get("decode_where")
  if d:
    out.append(f"  {d['words']}" if d.get("status") != "measured" else f"{d['title']}:")
    if d.get("status") == "measured":
      out.append(f"  {d['words']}")
      out += [f"  {r[0]:<42} {r[1]:>9} {r[2]:>9} {r[3]:>9}" for r in [d["columns"]] + d["rows"]]
  return out


def tie_lines(w:dict[str, Any]) -> list[str]:
  from boltbeam.workflow.screen import where_lines
  return where_lines(w) + [f"  {w['server_line']}"]


# --- before and after ----------------------------------------------------------------------------------------------

class DiffRefused(ValueError):
  """Two replays that cannot be compared: the field that differs."""


def diff(a:pathlib.Path, b:pathlib.Path) -> dict[str, Any]:
  """Two replay runs of the same trace, model and target, turn by turn (aligned by i) and in total: boltbeam.replay_diff.v1.
  Refused, naming the field, when the trace's sha256, the model or the target differ."""
  from boltbeam.workflow.common import load_manifest
  ta, tb = _read(a / "timing_trace.json"), _read(b / "timing_trace.json")
  ma, mb = load_manifest(a), load_manifest(b)
  for name, x, y in (("workload", ta.get("workload"), tb.get("workload")),
                     ("replay trace sha256", (ta.get("replay_trace") or {}).get("sha256"), (tb.get("replay_trace") or {}).get("sha256")),
                     ("model", ma.get("model_id"), mb.get("model_id")), ("target", ma.get("target_id"), mb.get("target_id"))):
    if x != y or x is None:
      raise DiffRefused(f"the runs differ in {name}: {x!r} and {y!r}")
  A, B = table(a, ta), table(b, tb)
  if not (A and B):
    raise DiffRefused("a run holds no replayed turn")
  by = {r["i"]: r for r in A["turns"]}
  keys = ("ttft_ms", "prompt_ms", "decode_tok_s", "overhead_ms", "recomputed", "cache_n", "clock_mhz")
  turns = [{"i": r["i"], **{k: (r[k] - by[r["i"]][k]) if r.get(k) is not None and by[r["i"]].get(k) is not None else None
                            for k in keys}} for r in B["turns"] if r["i"] in by]
  ha, hb = A["headline"], B["headline"]
  total = {"e_pct": (hb["e_pct"] - ha["e_pct"]) if hb.get("e_pct") is not None and ha.get("e_pct") is not None else None,
           "split": {k: hb["split"][k] - ha["split"][k] for k in ha["split"]},
           "lost": {k: hb["lost"][k] - ha["lost"][k] for k, _ in LOST} if ha.get("lost") and hb.get("lost") else None}
  rows_a = {(w.get("turn"), r["name"]): r for w in A.get("wheres") or wheres(a, ta) if w.get("status") == "measured" for r in w["rows"]}
  rows_b = {(w.get("turn"), r["name"]): r for w in B.get("wheres") or wheres(b, tb) if w.get("status") == "measured" for r in w["rows"]}
  where = [{"turn": k[0], "row": k[1], "now_ms": (rows_b[k]["now_ms"] - rows_a[k]["now_ms"]) if k in rows_a and k in rows_b else None,
            "lost_ms": (rows_b[k].get("lost_ms", 0) - rows_a[k].get("lost_ms", 0)) if k in rows_a and k in rows_b else None,
            "only_in": None if k in rows_a and k in rows_b else ("before" if k in rows_a else "after")}
           for k in sorted(set(rows_a) | set(rows_b), key=lambda k: (k[0] or 0, k[1]))]
  return {"schema": DIFF_SCHEMA, "before": a.name, "after": b.name, "turns": turns, "total": total, "where": where}


def diff_lines(d:dict[str, Any]) -> list[str]:
  t = d["total"]
  out = [f"Replay {d['after']} against {d['before']} (after less before):"]
  if t.get("e_pct") is not None:
    out.append(f"  Session % of roofline {t['e_pct']:+.1f} points.")
  out.append("  Time: " + ", ".join(f"{k} {v:+,.0f} ms" for k, v in t["split"].items()) + ".")
  if t.get("lost"):
    out.append("  Lost: " + ", ".join(f"{w} {t['lost'][k]:+,.0f} ms" for k, w in LOST) + ".")
  fmt = lambda v, f: "-" if v is None else f.format(v)  # noqa: E731
  out.append(f"  {'turn':>5} {'TTFT ms':>9} {'prompt ms':>10} {'tok/s':>7} {'overhead':>9} {'recomp':>7} {'cache':>8} {'MHz':>6}")
  for r in d["turns"]:
    out.append(f"  {r['i']:>5} {fmt(r['ttft_ms'], '{:+.0f}'):>9} {fmt(r['prompt_ms'], '{:+.0f}'):>10} "
               f"{fmt(r['decode_tok_s'], '{:+.1f}'):>7} {fmt(r['overhead_ms'], '{:+.0f}'):>9} {fmt(r['recomputed'], '{:+d}'):>7} "
               f"{fmt(r['cache_n'], '{:+d}'):>8} {fmt(r['clock_mhz'], '{:+.0f}'):>6}")
  for w in d["where"]:
    out.append(f"  turn {w['turn']} {w['row']}: " + (f"only {w['only_in']}" if w["only_in"] else
                                                       f"{w['now_ms']:+.2f} ms now, {w['lost_ms']:+.2f} ms lost"))
  return out
