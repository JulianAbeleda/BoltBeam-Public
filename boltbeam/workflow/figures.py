"""Figures measured on one machine, read off keyed rows (data/machine_figures.json), never off a constant.

A figure's row is keyed by what it depends on that a run can read (the chip, the host's L3, a built mod, an engine on a
model shape). resolve() gives a run its figure in one of three states, and the state is always said:

    measured    a row's key is this run's: its value
    borrowed    no such row: the nearest row's value (the most key fields equal), widened against the run by a margin
                from that row's own readings (one more reading's one-sided 95% bound, t x sd x sqrt(1 + 1/n)) and the
                spread of the rows' values; a cost is charged more, a saving credited less
    unmeasured  nothing to borrow honestly (the row has fewer than two readings, or the figure is a choice or a constant):
                no value, so the formula that reads it keeps its default, and the command that measures it is named

Engine-blind: an adapter passes its engine's name and its run's key (facts)."""
from __future__ import annotations

import functools
import json
import pathlib
import statistics
from typing import Any

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "machine_figures.json"


@functools.lru_cache(maxsize=1)
def table() -> dict[str, Any]:
  return json.loads(TABLE.read_text())


def resolve(name:str, key:dict[str, Any], t:list[float] | tuple = ()) -> dict[str, Any]:
  """The figure for a run whose key is `key`: {name, status, value, margin, row, says}. t: Student's t, one-sided
  95%, by degrees of freedom (the margin's; none given: nothing is borrowed)."""
  from boltbeam.report.plain import words
  fig, w = table()["figures"][name], words()["machine_figures"]
  here = ", ".join(f"{k} {key.get(k) if key.get(k) is not None else w['unread']}" for k in fig["key"])
  hit = next((r for r in fig["rows"] if all(key.get(k) == v for k, v in r["key"].items())), None)
  if hit:
    return {"name": name, "status": "measured", "value": hit["value"], "margin": 0.0, "row": hit, "says": None}
  near = max(fig["rows"], key=lambda r: sum(key.get(k) == v for k, v in r["key"].items()))
  got, n = near["readings"], len(near["readings"])
  told = {"what": fig["words"], "unit": fig["unit"], "here": here, "how": fig["how_to_measure"], "n": n, "value": near["value"],
          "row": ", ".join(f"{k} {v}" for k, v in near["key"].items()), "machine": near["machine"], "date": near["date"], "run": near["run"]}
  if not t or n < 2 or fig.get("borrow") is False or fig["kind"] not in ("cost", "saving"):
    return {"name": name, "status": "unmeasured", "value": None, "margin": None, "row": near, "says": w["unmeasured"].format(**told)}
  values = [r["value"] for r in fig["rows"]]
  margin = max(t[min(n - 1, len(t)) - 1] * statistics.stdev(got) * (1 + 1 / n) ** 0.5, max(values) - min(values))
  value = near["value"] + margin if fig["kind"] == "cost" else max(0.0, near["value"] - margin)
  return {"name": name, "status": "borrowed", "value": value, "margin": margin, "row": near,
          "says": w["borrowed"].format(**told, used=value, margin=margin, readings=", ".join(f"{x:g}" for x in got))}


def facts(engine:str, key:dict[str, Any], t:list[float] | tuple = ()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """An engine's figures as the facts its formulas read ({fact: value in the fact's unit, None when unmeasured}) and each
  figure's resolution, for the run to say what it borrowed or did not use. An unmeasured saving frees nothing (0)."""
  got = [(f, resolve(n, key, t)) for n, f in table()["figures"].items() if f.get("engine") == engine and f.get("fact")]
  value = lambda f, r: (0 if f["kind"] == "saving" else None) if r["value"] is None else r["value"] * f.get("unit_bytes", 1)  # noqa: E731
  return {f["fact"]: value(f, r) for f, r in got}, [r for _, r in got]


def said(resolved:list[dict[str, Any]], features, holds) -> list[str]:
  """What a run says of its figures: one line per figure not measured on its key, when the build reads the figure
  (its requires are features) and its `when` holds (holds: a formula over the run's facts -> bool), and one line for
  the figures read on their key that also hang on something the run does not read (unkeyed)."""
  from boltbeam.report.plain import words
  figs = table()["figures"]
  used = [r for r in resolved if all(x in (features or ()) for x in figs[r["name"]].get("requires") or []) and (not figs[r["name"]].get("when") or holds(figs[r["name"]]["when"]))]
  also = [f"{r['name']} ({figs[r['name']]['unkeyed']})" for r in used if r["status"] == "measured" and figs[r["name"]].get("unkeyed")]
  return [r["says"] for r in used if r["says"]] + ([words()["machine_figures"]["unkeyed"].format(figures="; ".join(also))] if also else [])
