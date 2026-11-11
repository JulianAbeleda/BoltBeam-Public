"""A run's results in plain words, as data: the one authority the TUI, report.html, summary.md and summary.txt read for
their wording, so no screen writes its own. The words are data/plain_words.json; this module only fills them.

Layer 1 answers, in order, what the user came for: how fast; how close to what this GPU allows; what holds it back.
Then the three parts that lose the most time, each with its gain, and one trust line. Layer 2 is the where table at four
columns (part, time, share, gain; the rest under "all columns"), "How it was measured", "Machine and engine" and the
glossary of every term layer 2 uses. A template per run type (TEMPLATES) shapes layer 1.

Reads a `results` document (workflow/results.results) and computes nothing new: every number is one the results hold,
formatted as the report's tables format it, or picked from them (the largest, the first). The checks behind the trust
line are the report's own (the tie-out, replication, the capture-overhead rule, the pairing, the estimate)."""
from __future__ import annotations

import functools
import json
import pathlib
import re
from typing import Any

SCHEMA = "boltbeam.plain_results.v1"
COMPARE_SCHEMA = "boltbeam.plain_compare.v1"
WORDS_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "plain_words.json"
RUN_TYPES = ("decode", "batch_sweep", "prefill", "chat_turn", "replay", "mtp", "compare", "engine_upgrade")
TOP = 3  # the parts layer 1 names
LAYER2 = ("How it was measured", "Machine and engine")


@functools.cache
def words() -> dict[str, Any]:
  return json.loads(WORDS_FILE.read_text())


def stage_names() -> dict[str, str]:
  """The plain name of every pipeline stage key, the report's and the measuring ones; "{here}" is this machine."""
  return dict(words()["stages"])


def _say(group:str, key:str, **kw:Any) -> str:
  return words()[group][key].format(**kw)


def _unit(key:str) -> str:
  return words()["units"][key]


def _f(v:Any, fmt:str) -> str:
  return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else ""


# --- what a run is, and what its rows are called ------------------------------------------------------------------

def run_type(res:dict[str, Any]) -> str:
  """decode, batch_sweep, prefill, chat_turn, replay, mtp or engine_upgrade, from what the results hold. compare is two
  runs (compare())."""
  return "engine_upgrade" if (res.get("engine") or {}).get("comparison") else speed_type(res)


def speed_type(res:dict[str, Any]) -> str:
  loss = res.get("loss") or {}
  workload = res.get("workload") or loss.get("workload")
  if workload in ("replay", "prefill"):
    return workload
  if workload == "session":
    return "chat_turn"
  if loss.get("speculation") or (res.get("engine") or {}).get("speculation_words"):
    return "mtp"
  if any((b.get("batch") or 1) > 1 for b in res.get("batches") or []):
    return "batch_sweep"
  return "decode"


def role_words(label:str) -> str:
  """A role row's name without its format (tie_out.role_label) in plain words: "attention query", "layer mixing
  before attention, narrowing step"."""
  name, _, part = label.partition(".")
  return words()["roles"].get(name, name) + (f", {words()['role_parts'].get(part, part)}" if part else "")


def part_name(row:dict[str, Any], *, short:bool = False) -> str:
  """A where-table row in plain words: "feed-forward in (Q4_K)", "idle time between GPU steps". A kernel no part names
  says only its kind (collectors/kernel_kind.py), in layer 1 and 2 alike: its own name stays in Full detail. short is
  kept for callers; both layers name a row the same way."""
  parts, roles = words()["parts"], words()["roles"]
  name, what = str(row.get("name") or ""), row.get("what")
  if what == "role" and " " in name:
    names, quant = name.rsplit(" ", 1)  # a part of a role at two shapes is "role.part" (tie_out.role_name)
    return " + ".join(role_words(n) for n in names.split(" + ")) + f" ({quant})"
  if name in parts:
    return parts[name]
  if what in ("gaps", "recurrent", "attention"):
    return parts[what]
  if what != "kernels":
    return name
  if role := _outside_role(row):
    return parts["kernels_role"].format(role=roles[role])
  kind = _kernel_kind(row)
  return parts["kernels_short"].format(kind=parts[kind]) if kind in parts and kind != "other" else parts["kernels_unknown"]


def _kernel_kind(row:dict[str, Any]) -> str:
  """A kernels row's kind: its name when the row is already a kind (the parts table names it), else the kind of the
  kernel it names (collectors/kernel_kind.py)."""
  from boltbeam.collectors.kernel_kind import kind_of
  name = str(row.get("name") or "")
  return name if name in words()["parts"] else kind_of({"kernel": name})


def _outside_role(row:dict[str, Any]) -> str | None:
  """The role a kernels row is labelled with when it is weight reads no part took that run as often as a role the limit
  has no bytes for (tie_out.roles_outside_limit names the row by the role, spaced): the role's key, else None. A row
  named by a kind of supporting work (the parts table) is that work, never a role."""
  name = str(row.get("name") or "")
  key = name.replace(" ", "_")
  return key if row.get("what") == "kernels" and name not in words()["parts"] and key in words()["roles"] else None


def part_kind(row:dict[str, Any]) -> str:
  """What a where-table row is, from not_parts: "part" (a part of the model's work, or named supporting work such as
  copying, or the idle time), "unassigned" (weight reads no part took: said once as a caveat with their time, never as a
  part) or "none" (BoltBeam's own balance of the limit, the rest not split, kernels no kind names). A kernels row is
  judged by its kind (_kernel_kind), so one named by its own kernel is what its kind is."""
  np = words()["not_parts"]
  if row.get("what") in np["whats"]:
    return "none"
  if row.get("what") != "kernels":
    return "part"
  if _outside_role(row) or _kernel_kind(row) in np["unassigned"]:
    return "unassigned"
  return "none" if _kernel_kind(row) in np["names"] else "part"


def _measured(loss:dict[str, Any]) -> dict[str, Any] | None:
  """The engine's untraced whole step; the per-role GPU time only when it is all there is (report/html.py's rule)."""
  runs = loss.get("runtimes") or []
  return next((r for r in runs if not r.get("per_role")), runs[0] if runs else None)


def _ms_table(w:dict[str, Any] | None) -> bool:
  """A where table whose gain is time (prefill and chat turns: ms if fixed), not tokens a second."""
  return bool(w) and (w.get("workload") == "prefill" or any("ms_if_fixed" in r for r in w.get("rows") or []))


def gain(row:dict[str, Any], w:dict[str, Any]) -> dict[str, Any]:
  """What a row gives back at its limit: the where table's own allocation (tie_out.decode_progress tok_s_gain), or for a
  prefill its ms lost when the table projects; None with the reason when the table does not."""
  if _ms_table(w):
    if row.get("ms_if_fixed") is None:
      return {"value": None, "unit": "ms", "cell": "", "figure": "", "text": None}  # said once: gains_note
    cell = f"{row['lost_ms']:.2f}"
    return {"value": row["lost_ms"], "unit": "ms", "cell": cell + " ms", "figure": f"{cell} {_unit('ms')}",
            "text": _say("gain", "ms", gain=cell)}
  from boltbeam.workflow.tie_out import GAIN_COLUMN
  g = row.get("tok_s_gain")
  if g is None or GAIN_COLUMN not in w.get("columns", []):
    return {"value": None, "unit": "tok_s", "cell": "", "figure": "", "text": None}  # said once: gains_note
  cell = row["cells"][w["columns"].index(GAIN_COLUMN) - 1]  # the table's own cell: its tenths add up to the limit
  return {"value": g, "unit": "tok_s", "cell": cell, "figure": f"{cell} {_unit('tok_s')}", "text": _say("gain", "tok_s", gain=cell)}


def _row(row:dict[str, Any], w:dict[str, Any], *, short:bool = False) -> dict[str, Any]:
  share = row.get("share")
  return {"part": part_name(row, short=short), "name": row.get("name"), "what": row.get("what"), "ms": row.get("now_ms"),
          "time": f"{row['now_ms']:.2f} ms", "lost_ms": row.get("lost_ms"), "share": share,
          "share_cell": "" if share is None else f"{100 * share:.0f}%", "gain": gain(row, w)}


def top_parts(res:dict[str, Any], n:int = TOP) -> list[dict[str, Any]]:
  """The n parts that lose the most time, worst first, each with its gain: the where table's rows by ms lost (a row
  that only balances the limit, or loses nothing, is not a part that holds the run back). A replay's are its losses."""
  loss = res.get("loss") or {}
  if (h := (loss.get("replay") or {}).get("headline")) and h.get("lost"):
    lost = sorted(((k, v) for k, v in h["lost"].items() if v > 0), key=lambda kv: -kv[1])[:n]
    total = sum(v for v in h["lost"].values() if v > 0)
    return [{"part": words()["parts"].get(f"replay_{k}", k), "name": k, "what": "replay", "ms": v, "time": f"{v / 1e3:.1f} s",
             "lost_ms": v, "share": v / total if total else None, "share_cell": f"{100 * v / total:.0f}%" if total else "",
             "gain": {"value": v / 1e3, "unit": "s", "cell": f"{v / 1e3:.1f} s", "figure": f"{v / 1e3:.1f} {_unit('s')}",
                      "text": _say("gain", "s", gain=f"{v / 1e3:.1f}")}}
            for k, v in lost]
  w = loss.get("where_token_goes")
  if not w:
    return []
  rows = sorted((r for r in w["rows"] if part_kind(r) == "part" and (r.get("lost_ms") or 0) > 0),
                key=lambda r: (-r["lost_ms"], str(r["name"])))
  return [_row(r, w, short=True) for r in rows[:n]]


def unassigned_note(res:dict[str, Any]) -> str | None:
  """The weight reads no part took, as one caveat with their time (the where table's own rows, summed at its 2
  decimals); None when every weight read is matched to a part."""
  w = (res.get("loss") or {}).get("where_token_goes") or {}
  ms = [r["now_ms"] for r in w.get("rows") or [] if part_kind(r) == "unassigned" and (r.get("now_ms") or 0) > 0]
  return _say("gain", "unassigned", ms=f"{sum(ms):.2f}") if ms else None


def gains_known(w:dict[str, Any] | None) -> bool:
  return bool(w) and any(gain(r, w)["value"] is not None for r in w.get("rows") or [])


def gains_note(res:dict[str, Any]) -> str | None:
  """Why a split table shows no gain per part, once, by the reason tie_out.where_token_goes allocates none: recording
  slowed the run past the overhead rule; the recording kept no timing window that matches the untraced run's
  (timing_match); or its timing was otherwise not checked against the untraced run. None when gains are shown, or the
  table is not split."""
  from boltbeam.workflow.tie_out import CEILING_PCT
  loss = res.get("loss") or {}
  w = loss.get("where_token_goes")
  if not _split(w) or gains_known(w):
    return None
  if _e2e(loss).get("confirms") is False:
    return _say("gain", "none_over", ceiling=f"{CEILING_PCT:g}")
  return _say("gain", "none_unmatched" if _unmatched(loss) else "none_unqualified")


def _unmatched(loss:dict[str, Any]) -> bool:
  """The recording's timing window does not match the untraced run's (tie_out timing_match), so it was not checked."""
  return ((loss.get("tie_out") or {}).get("timing_match") or {}).get("matches") is False


def _split(w:dict[str, Any] | None) -> bool:
  """A where table split into parts: a row other than the unsplit rest, the gaps and the limit's balance."""
  return any(r.get("what") not in ("rest", "gaps", "limit") for r in (w or {}).get("rows") or [])


# --- the trust line --------------------------------------------------------------------------------------------------

def _e2e(loss:dict[str, Any]) -> dict[str, Any]:
  return (loss.get("tie_out") or {}).get("end_to_end") or (loss.get("where_token_goes") or {}).get("end_to_end") or {}


def trust(res:dict[str, Any]) -> dict[str, Any]:
  """"Checked ✓" with what was checked, or the one caveat that matters most, from the report's own checks."""
  from boltbeam.workflow.tie_out import CEILING_PCT
  loss = res.get("loss") or {}
  t, w = loss.get("tie_out") or {}, loss.get("where_token_goes") or {}
  say = lambda key, **kw: {"checked": False, "text": _say("trust", key, **kw), "key": key}  # noqa: E731
  e2e, rep = _e2e(loss), t.get("replication") or w.get("replication") or {}
  if _measured(loss) is None and not (loss.get("replay") or {}).get("headline"):
    return say("not_measured")
  if loss.get("refused") or t.get("refused"):  # the floor rule's words stay in Full detail
    return say("refused")
  if not _split(w) and not (loss.get("replay") or {}).get("headline"):  # no parts: nothing adds up, so nothing is claimed
    return say("whole_only")
  if a := loss.get("ambiguous"):  # one sentence per kind of unresolved pairing; the pairs themselves are in Full detail
    return {"checked": False, "key": "unsure_split", "text": words()["unsure"].get(a.get("kind"), words()["unsure"]["other"])}
  if loss.get("estimate") or w.get("estimate"):
    return say("estimate")
  if not gains_known(w):  # a split whose gains the table did not allocate
    if e2e.get("confirms") is False:
      return say("over_overhead", pct=f"{e2e['diff_pct']:.1f}", ceiling=f"{CEILING_PCT:g}")
    return say("unmatched" if _unmatched(loss) else "unqualified")
  if loss.get("unpaired_launches") or loss.get("unpaired_roles") or (
      (loss.get("attribution") or {}).get("matrix_work") or {"complete": True}).get("complete") is False:
    return say("unpaired")
  if refused := loss.get("ceiling_refused"):
    return say("left_out", parts=", ".join(part_name({"name": f"{n['role']} {n['quant']}", "what": "role"}) for n in refused))
  if (loss.get("step") or {}).get("graph_failed"):
    return say("no_graph")
  how = [_say("trust", "checked_how")]
  if rep.get("qualifies"):
    how.append(_say("trust", "replicated", captures=rep["captures"], worst=f"{rep['worst_pct']:.1f}"))
  if e2e.get("confirms") is True:
    how.append(_say("trust", "within_overhead", pct=f"{e2e['diff_pct']:.1f}", ceiling=f"{CEILING_PCT:g}"))
  return {"checked": True, "text": _say("trust", "checked") + ": " + "; ".join(how) + ".", "key": "checked"}


# --- layer 1 ---------------------------------------------------------------------------------------------------------

def _fig(value:str, unit:str, label:str, key:str) -> dict[str, str]:
  """One headline figure: its value and unit (tok/s, ms, s or %), the two joined as layer 1 prints them, a short label."""
  return {"key": key, "value": value, "unit": unit, "figure": value + ("" if unit == "%" else " ") + unit, "label": label}


def _speed(res:dict[str, Any], kind:str) -> dict[str, Any]:
  out = _speed_core(res, kind)
  if res.get("placement") and out.get("close"):  # the engine arguments were not recorded: say so beside the limit (workflow/offload.placement)
    out["close"] += " " + words()["offload"]["unrecorded"]
  return out


def _speed_core(res:dict[str, Any], kind:str) -> dict[str, Any]:
  """How fast and how close, in the run type's template, with the numbers they hold and the headline figures."""
  loss, ceil = res.get("loss") or {}, res.get("ceiling") or {}
  tpl, fw, u = words()["templates"], words()["figures"], _unit
  if kind == "replay" and (h := (loss.get("replay") or {}).get("headline")):
    t = tpl["replay"]
    nums = {"seconds": h["time_ms"] / 1e3, "pct": h.get("e_pct"), "limit_s": (h.get("roofline_ms") or 0) / 1e3 or None}
    figs = [_fig(f"{h['time_ms'] / 1e3:.1f}", u("s"), fw["replay"].format(turns=h["warm_turns"]), "time")]
    close = None
    if h.get("e_pct") is not None:
      close = t["close"].format(pct=f"{h['e_pct']:.1f}", limit_s=f"{h['roofline_ms'] / 1e3:.1f}")
      figs.append(_fig(f"{h['e_pct']:.1f}", u("pct"), fw["limit_s"].format(limit_s=f"{h['roofline_ms'] / 1e3:.1f}"), "pct"))
    return {"fast": t["fast"].format(turns=h["warm_turns"], seconds=f"{h['time_ms'] / 1e3:.1f}"), "close": close,
            "numbers": nums, "figures": figs}
  m = _measured(loss)
  t = tpl.get(kind) or tpl["decode"]
  if m is None:
    return {"fast": tpl["decode"]["not_measured"], "close": None, "numbers": {}, "figures": []}
  batch = loss.get("batch") or 1
  w = loss.get("where_token_goes") or {}
  length = w.get("length") or ceil.get("context")
  nums = {"tok_s": m["tok_s"], "ms": m["ms"], "batch": batch}
  tok, ms = f"{m['tok_s']:.1f}", f"{m['ms']:.1f}"
  if kind in ("prefill", "chat_turn"):
    fast = t["fast"].format(tok_s=tok, length=length, ms=ms, depth=f"{w.get('depth') or 0:,}")
    figs = ([_fig(tok, u("tok_s"), fw["prompt"].format(length=length), "speed"), _fig(ms, u("ms"), fw["prompt_ms"], "ms")]
            if kind == "prefill" else
            [_fig(ms, u("ms"), fw["turn_ms"].format(length=length), "ms"), _fig(tok, u("tok_s"), fw["turn_tok_s"], "speed")])
  elif batch > 1:
    total = f"{batch * m['tok_s']:.1f}"
    fast = tpl["decode"]["fast_batch"].format(tok_s=tok, batch=batch, total=total)
    figs = [_fig(total, u("tok_s"), fw["total"].format(batch=batch), "total"), _fig(tok, u("tok_s"), fw["per_stream"], "speed")]
  else:
    fast = (t.get("fast") or tpl["decode"]["fast"]).format(tok_s=tok)
    figs = [_fig(tok, u("tok_s"), fw["speed_mtp" if kind == "mtp" else "speed"], "speed")]
  if kind == "batch_sweep" and (pts := res.get("batches")):  # the sweep's best point: How it was measured says it
    best = max(pts, key=lambda b: (b["tok_s_total"], -b["batch"]))
    nums.update(best_tok_s_total=best["tok_s_total"], best_batch=best["batch"], best_context=best["context"])
  if o := res.get("offload"):  # the experts sat in RAM: the all-in-GPU-memory limit below does not apply (workflow/offload.run_block)
    return {"fast": fast, "numbers": nums, **offload_close(o, figs)}
  limit_ms = loss.get("limit_ms")
  if not limit_ms or not m.get("ms"):
    # one sentence per reason key (results: the ceiling's reason_key, or the missing split's); the run's words stay in Full detail
    key = loss.get("reason_key") or ceil.get("reason_key") or loss.get("missing_key")
    return {"fast": fast, "close": words()["no_limit"].get(key, tpl["decode"]["no_limit"]), "numbers": nums, "figures": figs}
  pct = 100.0 * limit_ms / m["ms"]
  limit_tok = loss.get("limit_tok_s") or ceil.get("tok_s")
  nums.update(pct=pct, limit_ms=limit_ms, limit_tok_s=limit_tok)
  ms_limit = kind in ("prefill", "chat_turn")
  figs.append(_fig(f"{pct:.0f}", u("pct"), fw["limit_ms"].format(limit_ms=f"{limit_ms:.1f}") if ms_limit or not limit_tok
                   else fw["limit"].format(limit=_f(limit_tok, "{:.1f}")), "pct"))
  return {"fast": fast, "close": t["close"].format(pct=f"{pct:.0f}", limit=_f(limit_tok, "{:.1f}"), limit_ms=f"{limit_ms:.1f}"),
          "numbers": nums, "figures": figs}


def offload_close(o:dict[str, Any], figs:list[dict[str, str]]) -> dict[str, Any]:
  """Layer 1's 'how close' for a run that offloaded its experts: its speed as a % of the offload limit, a range when the hit
  rate is unknown; a sentence for a run it cannot limit; 'estimate' when the shapes were not read from a header."""
  w = words()["offload"]
  if o["status"] != "modeled":
    return {"close": w["none"].format(reason=o["reason"]), "figures": figs}
  lim, pct = o["tok_s"], o["pct"]
  if pct["low"] > 100:
    return {"close": w["over"].format(limit=_range(lim["low"], lim["high"], "{:.0f}")), "figures": figs}
  limit = _range(lim["low"], lim["high"], "{:.0f}")
  close = w["close_one" if o["hit_rate_known"] or round(lim["low"]) == round(lim["high"]) else "close_band"].format(pct=_range(pct["low"], pct["high"], "{:.0f}"), limit=limit)
  return {"close": close + (w["estimate"] if o["estimated"] else ""), "figures": figs + [_fig(_range(pct["low"], pct["high"], "{:.0f}"), _unit("pct"), w["figure"].format(limit=limit), "pct")]}


def _range(lo:float, hi:float, fmt:str) -> str:
  a, b = fmt.format(lo), fmt.format(hi)
  return a if a == b else f"{a}\u2013{b}"


def _gain_figure(top:list[dict[str, Any]]) -> list[dict[str, str]]:
  """The biggest gain on offer: the first named part's (the top three are worst first, and each gain is its share of
  the lost time), when the table allocates one."""
  g = top[0]["gain"] if top else {}
  if g.get("value") is None or not g.get("figure") or not re.search(r"[1-9]", g["cell"]):  # "+0.0" is no gain to lead with
    return []
  value, unit = g["figure"].rsplit(" ", 1)
  return [_fig(value, unit, words()["figures"]["gain_" + g["unit"] if g["unit"] != "tok_s" else "gain"], "gain")]


def layer1(res:dict[str, Any], run:pathlib.Path | None = None) -> dict[str, Any]:
  kind, loss = run_type(res), res.get("loss") or {}
  speed = _speed(res, speed_type(res))
  top = top_parts(res)
  if loss.get("missing") and not _split(loss.get("where_token_goes")):
    top = []  # one row "kernels and gaps, not split" is not a part: the run says why it is not split instead
  tpl = words()["templates"]
  holds_tpl = (tpl.get(speed_type(res)) or tpl["decode"]).get("holds") or tpl["decode"]["holds"]
  if top and speed_type(res) == "replay":
    holds = holds_tpl.format(part=top[0]["part"], lost_s=f"{top[0]['lost_ms'] / 1e3:.1f}")
  elif top:
    holds = holds_tpl.format(part=top[0]["part"], share=top[0]["share_cell"] or "most")
  elif _measured(loss) is not None and (loss.get("missing") or loss.get("refused")):
    holds = tpl["decode"]["no_split"]
  else:
    holds = None
  t = trust(res)
  note = gains_note(res) if top and t["key"] not in ("over_overhead", "unqualified", "unmatched") else None  # said once, never twice
  out = {"run_type": kind, "title": title(res, run), "figures": speed["figures"] + _gain_figure(top), "fast": speed["fast"],
         "close": speed["close"], "holds": holds, "numbers": speed["numbers"], "top": top, "gains_note": note,
         "unassigned": unassigned_note(res) if top else None, "trust": t}
  if kind == "engine_upgrade":
    cmp = res["engine"]["comparison"]
    out["upgrade"] = [tpl["engine_upgrade"]["upgrade"].format(engine=(res.get("engine") or {}).get("provider") or "engine",
                                                              before=cmp.get("before_run") or "previous run")]
    record = _read(run / cmp["file"]) if run is not None and cmp.get("file") else {}
    if record.get("diff"):
      c = compare(record["diff"])["layer1"]
      out["upgrade"] += [c["fast"], *c.get("lines", [])]
  return out


MODEL_SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".pth")  # a model file's extension, never part of its name


def model_name(res:dict[str, Any], run:pathlib.Path | None = None) -> str | None:
  """The model's name as the run recorded it: the file or folder name of its model_path (a GGUF split's first file
  named without its "-00001-of-00004"), else the results' model_id. Nothing is made up: no recorded name, None."""
  path = str(_read(run / "run_manifest.json").get("model_path") or "") if run is not None else ""
  if path:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    name = next((name[: -len(x)] for x in MODEL_SUFFIXES if name.lower().endswith(x)), name)
    return re.sub(r"-\d{5}-of-\d{5}$", "", name) or None
  return res.get("model_id")


def chip_name(res:dict[str, Any], run:pathlib.Path | None = None) -> str | None:
  """The GPU's own name as the run recorded it (nvidia-smi's, amd-smi's or system_profiler's, through the machine facts,
  else the hardware profile) for the run's chip id; else the chip's plain name in its target data (targets.json `name`,
  a made profile's scanned name); the id itself when none is known."""
  target = res.get("target_id")
  machine = ((res.get("loss") or {}).get("machine") or {}).get("gpus") or []
  scanned = ((_read(run / "hardware_profile.json").get("gpu") or {}).get("devices")) or [] if run is not None else []
  for gpus in (machine, scanned):
    names = {str(g["name"]) for g in gpus if g.get("name") and g.get("target_id") == target}
    if names: return next(iter(names)) if len(names) == 1 else target
  from boltbeam.target.targets import TARGETS
  return getattr(TARGETS.get(str(target)), "name", None) or target


def title(res:dict[str, Any], run:pathlib.Path | None = None) -> str:
  engine = (res.get("engine") or {}).get("provider") or (res.get("loss") or {}).get("provider")
  engine = words()["engines"].get(engine, engine) if engine else "an unknown engine"
  return f"{model_name(res, run) or 'unknown model'} on {chip_name(res, run) or 'an unknown GPU'} with {engine}"


def _read(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


# --- layer 2 ---------------------------------------------------------------------------------------------------------

def table(res:dict[str, Any]) -> dict[str, Any] | None:
  """The where table at four columns (part, time, share, gain), in the table's own order; all_columns is every column
  the report has always printed, each row named by its part in plain words (Full detail keeps the record's names)."""
  loss = res.get("loss") or {}
  w = loss.get("where_token_goes")
  if not w:
    return None
  per = loss.get("per") or "per token"
  rows = [_row(r, w) for r in w["rows"]]
  known = gains_known(w)  # no gain anywhere: the column goes, and the note says why once
  return {"title": w["title"], "columns": ["part", f"time ({per})", words()["labels"]["share"], *(["gain"] if known else [])],
          "note": None if known else gains_note(res),
          "rows": [{**r, "section": src.get("section"),
                    "cells": [r["part"], r["time"], r["share_cell"], *([r["gain"]["cell"]] if known else [])]}
                   for r, src in zip(rows, w["rows"])],
          "all_columns": {"columns": list(w["columns"]), "rows": [[r["part"], *src["cells"]] for r, src in zip(rows, w["rows"])],
                          "baseline": w.get("baseline_cells"), "sum": w.get("sum_cells"), "words": w.get("words")}}


def measured_lines(res:dict[str, Any]) -> list[str]:
  """How it was measured, as plain sentences from the run's own records."""
  from boltbeam.workflow.tie_out import CEILING_PCT
  say = lambda key, **kw: _say("measured", key, **kw)  # noqa: E731
  loss, ceil, engine = res.get("loss") or {}, res.get("ceiling") or {}, res.get("engine") or {}
  t, w = loss.get("tie_out") or {}, loss.get("where_token_goes") or {}
  out = []
  m = _measured(loss)
  ctx = (loss.get("step") or {}).get("context") or loss.get("limit_context")
  if m is not None:
    out.append(say("speed", how=f"at a context of {ctx:g} tokens" if isinstance(ctx, (int, float)) else "on the run's own prompt"))
  nums = _speed(res, speed_type(res))["numbers"]
  if nums.get("best_tok_s_total") is not None:  # the sweep's own best, a second measurement: said here, not in layer 1
    out.append(say("sweep", best=f"{nums['best_tok_s_total']:.1f}", best_batch=nums["best_batch"], best_context=f"{nums['best_context']:g}"))
  if loss.get("estimate"):
    out.append(say("isolated"))
  elif loss.get("semantic_capture"):
    out.append(say("captured"))
  elif loss.get("role_source") == "in_model":
    tool = re.search(r"\(([^)]*)\)", loss.get("role_source_words") or "")
    out.append(say("in_model", tool=tool.group(1) if tool else "a capture tool"))
  elif m is not None and loss.get("missing"):  # one sentence per reason key (results.missing_reason); its words: Full detail
    out.append(words()["not_split"].get(loss.get("missing_key") or "not_run", words()["not_split"]["not_split"]))
  rep = t.get("replication") or w.get("replication") or {}
  if rep:
    key = "replicated" if rep.get("qualifies") else "not_replicated"
    out.append(say(key, captures=rep["captures"], worst=f"{rep['worst_pct']:.1f}", tolerance=f"{rep['tolerance_pct']:g}"))
  e2e = _e2e(loss)
  if e2e.get("status") == "checked":
    out.append(say("overhead", pct=f"{e2e['diff_pct']:.1f}", ceiling=f"{CEILING_PCT:g}"))
    if e2e.get("confirms") is False and rep.get("qualifies") and (t.get("gaps_bound") or w.get("gaps_bound")):
      out.append(say("overhead_rows"))
  elif cc := loss.get("capture_comparison"):
    rows = {r["name"]: r["now_ms"] for r in cc.get("rows") or []}
    diff = next((v for k, v in rows.items() if k != "untraced median"), None)
    if "untraced median" in rows and diff is not None:
      out.append(say("overhead_ms", untraced=f"{rows['untraced median']:.3f}", diff=f"{diff:+.3f}"))
  if loss.get("module_load"):
    out.append(say("module_load"))
  if loss.get("token_forcing"):
    out.append(say("forced"))
  step = loss.get("step") or {}
  if step.get("graph_failed"):
    out.append(say("graph"))
  if loss.get("timing_scope"):
    out.append(say("clock"))
  if loss.get("speculation") or engine.get("speculation_words"):
    out.append(say("speculation"))
  if loss.get("limit_ms") or ceil.get("status") == "modeled":
    if str(ceil.get("basis") or "").startswith(("sum of captured operation", "the sum of the prefill")) or res.get("workload") in ("prefill", "session", "replay"):
      out.append(say("limit_ops"))
    elif ceil.get("peak_bandwidth_gbs"):
      out.append(say("limit", gbs=f"{ceil['peak_bandwidth_gbs']:.1f}"))
  rows = w.get("rows") or []
  if any(gain(r, w)["value"] is not None for r in rows) and not _ms_table(w):
    out.append(say("gains"))
  if any(r.get("limit_ms") is None for r in rows):
    out.append(say("no_model"))
  if loss.get("check_status") == "not checked":
    out.append(say("repeats"))
  elif (b := t.get("band") or res.get("repeats")) and isinstance(b, str):
    out.append(say("band"))
  if engine.get("comparison"):
    out.append(say("upgrade"))
  return out


def machine_lines(res:dict[str, Any], run:pathlib.Path | None = None) -> list[str]:
  """Machine and engine: the GPU (its recorded name) and its memory speed, the GPUs used, the engine build against its
  pin and upstream, the weights, the settings asked for, the toolchain by its tools' names and how parts were timed."""
  say = lambda key, **kw: _say("machine", key, **kw)  # noqa: E731
  loss, ceil, engine = res.get("loss") or {}, res.get("ceiling") or {}, res.get("engine") or {}
  out = []
  if ceil.get("peak_bandwidth_gbs"):
    out.append(say("chip", chip=chip_name(res, run), gbs=f"{ceil['peak_bandwidth_gbs']:.1f}",
                   source=ceil.get("bandwidth_source") or "the chip registry"))
  gpus = (loss.get("machine") or {}).get("gpus") or []
  if len(gpus) > 1:
    out.append(say("gpus", gpus=", ".join(str(g.get("name") or g.get("target_id")) for g in gpus)))
  build = engine.get("build") or {}
  out.append(say("engine", words=build.get("words") or engine.get("provider") or "unknown"))
  out += [say("upstream", line=line) for line in engine.get("build_lines") or []]
  if engine.get("weight_format"):
    out.append(say("weights", format=engine["weight_format"]))
  if engine.get("settings_words"):
    out.append(say("settings", words="; ".join(engine["settings_words"])))
  versions = ((engine.get("toolchain") or {}).get("versions") or {})
  if versions:
    tools = words()["tools"]
    out.append(say("toolchain", words=", ".join(f"{tools.get(k, k)} {v}" for k, v in sorted(versions.items()))))
  if (res.get("measurement") or {}).get("label"):
    out.append(say("measurement", label=res["measurement"]["label"]))
  return out


def glossary(texts:list[str]) -> list[dict[str, str]]:
  """The glossary terms the given texts use, in the glossary's order: a term is used when one of its match words is."""
  text = "\n".join(texts)
  return [{"term": g["term"], "plain": g["plain"]} for g in words()["glossary"]
          if any(re.search(r"(?<![A-Za-z])" + re.escape(m) + r"(?![A-Za-z])", text, re.I) for m in g["match"])]


def layer2(res:dict[str, Any], run:pathlib.Path | None = None) -> dict[str, Any]:
  tab = table(res)
  sections = [{"title": LAYER2[0], "lines": measured_lines(res)}, {"title": LAYER2[1], "lines": machine_lines(res, run)}]
  texts = [line for s in sections for line in s["lines"]]
  if tab:
    texts += tab["columns"] + tab["all_columns"]["columns"] + [r["part"] for r in tab["rows"]]
  return {"table": tab, "sections": sections, "glossary": glossary(texts)}


def plain_results(res:dict[str, Any], run:pathlib.Path | None = None) -> dict[str, Any]:
  """The plain-results block of one run: layer 1 and layer 2, every reader's wording."""
  return {"schema": SCHEMA, "kind": "plain", "id": res.get("id"), "layer1": layer1(res, run), "layer2": layer2(res, run)}


# --- compare: two runs ------------------------------------------------------------------------------------------------

def compare(d:dict[str, Any]) -> dict[str, Any]:
  """Layer 1 of two runs compared (workflow/diff.audit, the engine-upgrade audit, or replay.diff): what changed, the
  three parts that moved most, and whether the audit can justify a pin."""
  t = words()["templates"]["compare"]
  if d.get("schema") == "boltbeam.engine_diff.v1":
    builds = d.get("builds") or [None, None]
    lines = [t["audit_part"].format(part=p["part"], untraced=_f(p.get("untraced_ms"), "{:+.3f}") or "-",
                                    verdict=t["not_checked"] if p.get("unchanged") is None else t["unchanged"] if p["unchanged"] else t["changed"])
             for p in d.get("parts") or []]
    moved = sorted((r for p in d.get("parts") or [] for r in p.get("rows") or [] if r.get("now_ms") is not None),
                   key=lambda r: -abs(r["now_ms"]))[:TOP]
    refused = sorted({p["pin_refused"] for p in d.get("parts") or [] if p.get("pin_refused")})
    ok = bool(d.get("justifies_pin"))
    return {"schema": COMPARE_SCHEMA, "kind": "plain_compare", "layer1": {
      "run_type": "compare", "fast": t["audit"].format(after=d["after"], before=d["before"], engine=d.get("provider"),
                                                       build_after=builds[1], build_before=builds[0], workload=d.get("workload")),
      "lines": lines, "top": [{"part": str(r["key"]), "ms": r["now_ms"], "text": t["moved"].format(row=r["key"], change=f"{r['now_ms']:+.3f}")} for r in moved],
      "trust": {"checked": ok, "text": _say("trust", "checked") if ok else _say("trust", "refused", reason="; ".join(refused) or "no repeat runs")}}}
  total = d.get("total") or {}
  where = sorted((r for r in d.get("where") or [] if r.get("lost_ms") is not None), key=lambda r: -abs(r["lost_ms"]))[:TOP]
  lines = ([t["replay_close"].format(points=f"{total['e_pct']:+.1f}")] if total.get("e_pct") is not None else []) + (
    [t["replay_time"].format(split=", ".join(f"{k} {v:+,.0f} ms" for k, v in total["split"].items()))] if total.get("split") else [])
  return {"schema": COMPARE_SCHEMA, "kind": "plain_compare", "layer1": {
    "run_type": "compare", "fast": t["replay"].format(after=d.get("after"), before=d.get("before")), "lines": lines,
    "top": [{"part": f"turn {r['turn']} {r['row']}", "ms": r["lost_ms"],
             "text": t["moved"].format(row=f"turn {r['turn']} {r['row']}", change=f"{r['lost_ms']:+.2f}")} for r in where],
    "trust": None}}


HEAD_TO_HEAD_SCHEMA = "boltbeam.plain_head_to_head.v1"


def _record(res:dict[str, Any], run:pathlib.Path | None) -> dict[str, str]:
  """What a run is, as the head-to-head line names it, each from the run's own record."""
  engine = res.get("engine") or {}
  provider = engine.get("provider") or (res.get("loss") or {}).get("provider") or ""
  return {"model": model_name(res, run) or "", "gpu": chip_name(res, run) or "",
          "engine": words()["engines"].get(provider, provider), "build": str((engine.get("build") or {}).get("build") or ""),
          "weights": engine.get("weight_format") or "", "settings": "; ".join(engine.get("settings_words") or []),
          "workload": str(res.get("workload") or ""), "batch": str((res.get("loss") or {}).get("batch") or 1)}


def head_to_head(a:dict[str, Any], b:dict[str, Any]) -> dict[str, Any]:
  """Two runs side by side: each run's own layer-1 figures in one row per kind (speed, total, % of its GPU's limit, the
  biggest gain), the better figure marked (higher or lower is better by the row's kind; the gain is marked neither
  way), and one line saying what differs between them. a and b are {"id", "res", "run"}. Refused, as one plain line,
  when a run has no measured result: there is nothing of it to set beside the other."""
  hw = words()["head_to_head"]
  sides = [{"id": x["id"], "title": title(x["res"], x.get("run")), "measured": bool(x["res"].get("measured")),
            "figures": {f["key"]: f for f in layer1(x["res"], x.get("run"))["figures"]}} for x in (a, b)]
  out = {"schema": HEAD_TO_HEAD_SCHEMA, "kind": "head_to_head", "runs": [{k: s[k] for k in ("id", "title")} for s in sides],
         "differs": None, "rows": [], "refused": None}
  if missing := next((s for s in sides if not s["measured"] or not s["figures"]), None):
    out["refused"] = hw["not_measured"].format(run=missing["id"])
    return out
  ra, rb = _record(a["res"], a.get("run")), _record(b["res"], b.get("run"))
  differ = [hw["field"].format(name=hw["fields"][k], a=ra[k] or hw["none"], b=rb[k] or hw["none"]) for k in hw["fields"] if ra[k] != rb[k]]
  out["differs"] = hw["differs"].format(fields="; ".join(differ)) if differ else hw["same"]
  for key, label in hw["rows"].items():
    figs = [s["figures"].get(key) for s in sides]
    if not any(figs) or any(r["key"] == key for r in out["rows"]):
      continue
    nums = [float(f["value"]) if f else None for f in figs]
    better = None
    if None not in nums and nums[0] != nums[1] and key in hw["higher"] + hw["lower"]:
      better = int((nums[1] > nums[0]) == (key in hw["higher"]))
    out["rows"].append({"key": key, "label": label, "cells": [f["figure"] if f else "" for f in figs], "better": better})
  return out


def parts_note(model_same:bool, refused:Exception | None) -> str | None:
  """Why the per-part comparison is not shown, in one line: two models' parts never line up; a refusal on a field
  names that field in plain words; any other refusal says its own reason. None when the parts are compared."""
  hw = words()["head_to_head"]
  if not model_same:
    return hw["parts_other_model"]
  if refused is None:
    return None
  field = getattr(refused, "field", None)
  if field:
    return hw["parts_differ"].format(field=hw["diff_fields"].get(field, field))
  return hw["parts_refused"].format(reason=str(refused).rstrip("."))


# --- the plain-text copy -----------------------------------------------------------------------------------------------

def layer1_lines(block:dict[str, Any], *, bold:bool = False) -> list[str]:
  """Layer 1 as plain text: the headline figures, the answers, the top three with their gains, the caveats, the trust
  line. bold (Markdown) sets every figure in bold and the figures as a list."""
  l1 = block["layer1"]
  out = [l1["title"], ""]
  label = words()["labels"]
  if figs := l1.get("figures"):
    out += [f"- **{f['figure']}** {f['label']}" if bold else f"{f['figure']} — {f['label']}" for f in figs] + [""]
  out += [f"{label[k]}: {l1[k]}" for k in ("fast", "close", "holds") if l1.get(k)]
  out += l1.get("upgrade") or []
  if l1["top"]:
    out += ["", label["top"] + ":"]
    out += [f"  {i + 1}. {p['part']}: {top_words(p, bold=bold)}" for i, p in enumerate(l1["top"])]
    out += [f"  {l1[k]}" for k in ("gains_note", "unassigned") if l1.get(k)]
  out += ["", l1["trust"]["text"]]
  return out


def top_words(p:dict[str, Any], *, bold:bool = False) -> str:
  """One of the top three after its name: its gain first when the table has one (its figure, bold in Markdown), then
  its time and share of the lost time."""
  said = words()["labels"]["top_line"].format(time=p["time"], share=p["share_cell"])
  g = p["gain"]
  if not g.get("text"):
    return said
  text = g["text"].replace(g["figure"], f"**{g['figure']}**", 1) if bold and g.get("figure") else g["text"]
  return f"{text}; {said}"


def _md_table(columns:list[str], rows:list[list[str]]) -> list[str]:
  esc = lambda c: str(c if c is not None else "").replace("|", "\\|")  # noqa: E731
  return ["| " + " | ".join(esc(c) for c in columns) + " |", "|" + "---|" * len(columns),
          *("| " + " | ".join(esc(c) for c in r) + " |" for r in rows)]


def markdown(block:dict[str, Any]) -> list[str]:
  """Layer 1 and layer 2 as Markdown: the same layers report.html shows, as text."""
  out = ["## The answer", "", *layer1_lines(block, bold=True)]
  l2 = block["layer2"]
  if tab := l2["table"]:
    out += ["", f"## {tab['title']}", "", *_md_table(tab["columns"], [r["cells"] for r in tab["rows"]])]
    out += ["", tab["note"]] if tab.get("note") else []
    a = tab["all_columns"]
    out += ["", words()["labels"]["all_columns"] + ":", "", *_md_table(a["columns"], a["rows"] + [x for x in (a["baseline"], a["sum"]) if x])]
    if a.get("words"):
      out += ["", a["words"]]
  for s in l2["sections"]:
    out += ["", f"## {s['title']}", "", *(f"- {line}" for line in s["lines"])]
  if l2["glossary"]:
    out += ["", "## Words used here", "", *(f"- **{g['term']}**: {g['plain']}" for g in l2["glossary"])]
  return out
