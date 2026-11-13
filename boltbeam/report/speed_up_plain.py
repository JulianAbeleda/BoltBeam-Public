"""Speed up in plain words (layer 1): a derive log, or a refusal, as the lines a screen shows. Every sentence is
data/plain_words.json speed_up; this reads it through report/plain.py and decides which sentences a record earns."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.report import plain


def _say() -> dict[str, Any]:
  return plain.words()["speed_up"]


def refusal(text:str) -> dict[str, Any]:
  """A refusal's plain reason and the one fix, if there is one. A refusal no row matches is shown as it was said (never swallowed), with no fix."""
  for row in _say()["refused"]["rows"]:
    if any(m in text for m in row["match"]):
      return {"reason": row["reason"], "fix": row.get("fix"), "said": text}
  return {"reason": text, "fix": None, "said": text}


def _fix(fix:str | None) -> str | None:
  return _say()["refused"]["fix"].format(fix=fix) if fix else None


def answer(log:str) -> dict[str, Any] | None:
  """The one JSON object a derive --json run printed after the job's start line, or None if the run ended without one."""
  lines = log.splitlines()
  start = max([i + 1 for i, l in enumerate(lines) if l.startswith("=== ")], default=0)
  for i in range(start, len(lines)):
    if lines[i].startswith("{"):
      try:
        return json.JSONDecoder().raw_decode("\n".join(lines[i:]))[0]
      except ValueError:
        continue
  return None


def _change(pct:float) -> str:
  return f"{pct:+.0f}%"


def _gate(rec:dict[str, Any]) -> dict[str, Any]:
  g = str(1)
  return {int(k): v[g] for e in rec["speed"][:1] for k, v in e["contexts"].items() if g in v}


def result(log:str) -> dict[str, Any]:
  """kept, not_kept, refused or crashed, with the lines in the order layer 1 reads them."""
  say, rec = _say(), answer(log)
  out:dict[str, Any] = {"kind": "speedup_plain", "outcome": "crashed", "head": None, "lines": [], "fix": None,
                        "copy": None, "name": None, "speed_measured": None}
  if rec is None:
    return {**out, "head": say["refused"]["crashed"], "fix": _fix(say["refused"]["crashed_fix"])}
  if rec.get("kind") == "error":
    r = refusal(rec["error"])
    return {**out, "outcome": "refused", "head": say["refused"]["head"], "lines": [r["reason"]], "fix": _fix(r["fix"]), "said": r["said"]}
  name, th, a = rec["name"], rec["threshold"], rec["accuracy"]
  out["name"] = name
  from boltbeam.quantization import speed_up
  valid = [speed_up.mtp_depths_valid(e) for e in rec.get("mtp_speed", [])[:1]]
  drafted = {round(v["context"]): v for depths, _ in valid for v in depths.values()}  # the MTP gate's depths by context, a retired one left out
  measured = bool(speed_up.measured(rec))
  if not (rec["kept"] and rec["passed"] and not speed_up.slower(rec)):
    k = say["not_kept"]
    why = (k["answers"].format(pct=f"{a[th['metric']] * 100:.1f}", need=f"{th['min'] * 100:.0f}") if not rec["passed"] else
           ([k["slower"].format(context=c, before=f"{v['original']['tok_s']:.1f}", after=f"{v['copy']['tok_s']:.1f}")
                 for c, v in sorted(_gate(rec).items()) if not v["faster"]] +
                [k["mtp_slower"].format(base=pathlib.Path(rec["source"]).name, context=c, before=f"{v['source']['tok_s']:.1f}", after=f"{v['copy']['tok_s']:.1f}")
                 for c, v in sorted(drafted.items()) if not v["faster"]])[0])
    kept_at = [k["logs_kept"].format(where=rec["logs_kept"])] if rec.get("logs_kept") else []
    return {**out, "outcome": "not_kept", "head": k["head"], "lines": [why, *kept_at, k["original"]], "speed_measured": measured}
  k, lines = say["kept"], []
  for key, rows, was in (("line", sorted(_gate(rec).items()), "original"), ("mtp_line", sorted(drafted.items()), "source")):
    for n, (c, v) in enumerate(rows):
      label = k["short" if n == 0 else "long"].format(context=c)
      lines.append(k[key].format(label=label, before=f"{v[was]['tok_s']:.1f}", after=f"{v['copy']['tok_s']:.1f}", change=_change(v["change_pct"])))
  lines += [k["mtp_retired"]] if any(gone for _, gone in valid) else []
  if not measured:
    lines.append(k["not_measured"].format(reason=k["not_measured_off"] if "switched off" in (rec["speed_note"] or "") else k["not_measured_why"]))
  lines += [k["answers"] if a else k["draft_answers"], k["listed"].format(name=name)]
  return {**out, "outcome": "kept", "head": k["head"].format(name=name), "lines": lines, "copy": rec["copy"], "speed_measured": measured}


def verify(copy:str) -> dict[str, Any]:
  """--verify in plain words: pass, or the plain reason it does not match."""
  from boltbeam.quantization import speed_up
  say = _say()["verify"]
  try:
    rec = speed_up.verify(copy)
  except Refused as exc:
    r = refusal(str(exc))
    return {"kind": "speedup_verify", "ok": False, "head": say["fail"], "lines": [r["reason"]], "fix": _fix(r["fix"]), "said": r["said"]}
  lines = [say["pass"]] + ([] if speed_up.measured(rec) else [say["no_speed"]])
  return {"kind": "speedup_verify", "ok": True, "head": lines[0], "lines": lines[1:], "fix": None, "said": None}


def log_result(path:str) -> dict[str, Any]:
  return result(pathlib.Path(path).expanduser().read_text())
