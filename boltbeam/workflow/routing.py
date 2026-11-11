"""`boltbeam routing MODEL --engine E`: a routing profile of the model on real text (roofline/routing_profile.py), from
the engine's own record through its adapter (data/routing_adapters.json), saved where the prompt limit, the offload
plan and tune find it (~/.boltbeam/routing/<model id>.json). Engine-blind: an engine is a row and its adapter module."""
from __future__ import annotations

import importlib
import json
import pathlib
import struct
from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.roofline import routing_profile as rp

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "routing_adapters.json"
CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "gate_corpus.json"
TOKENS = 10000


def table() -> dict[str, Any]:
  return json.loads(TABLE.read_text())["engines"]


def engines() -> list[str]:
  return sorted(table())


def run(model:str, engine:str, *, tokens:int = TOKENS, source:str | None = None, out:pathlib.Path | None = None,
        build:str | None = None, port:int = 0, work:pathlib.Path | None = None, corpus:str | None = None) -> dict[str, Any]:
  """The profile, saved; {profile, path}. source: the engine's own record to read (--from), whose text is `corpus` (what
  the run that wrote it read; unnamed by default); else the adapter captures one on data/gate_corpus.json when its row
  says it can. A Refused says why there is none."""
  row = table().get(engine)
  if row is None:
    raise Refused(f"no routing adapter for {engine}; the engines with one: {', '.join(engines())}")
  from boltbeam.roofline.offload_shapes import moe_split
  sh = moe_split(model)  # a Refused for a model with no expert tensors
  ident = rp.identity(model)
  mod = importlib.import_module(row["adapter"])
  try:
    if source:
      p = mod.profile(source, ident, n_expert=sh["experts"], corpus=corpus or "the text the source run read", tokens=tokens)
    elif row["captures"]:
      p = mod.capture(model, work or pathlib.Path.home() / ".boltbeam" / "routing" / f"capture-{ident['id']}", tokens=tokens,
                      ident=ident, corpus=CORPUS, build=build, port=port)
    else:
      raise Refused(f"{engine}: BoltBeam reads its record but does not start it; pass --from ({row['from']})")
  except (OSError, ValueError, struct.error) as exc:
    why = Refused(f"the {engine} routing record could not be read: it is missing, damaged or not the kind this engine writes")
    why.detail = f"{engine}: {exc}"  # the raw reader text, for a log; not the reason shown
    raise why from exc
  if p["layers"] and (len(p["layers"]) + len(p["source"].get("layers_not_measured", [])) != sh["layers"]):
    raise Refused(f"the profile has {len(p['layers'])} layers, the model {sh['layers']}: a profile of another model")
  return {"profile": p, "path": str(rp.save(p, out))}


def text(res:dict[str, Any], other:dict[str, Any] | None = None) -> str:
  p = res["profile"]
  lines = [f"routing profile: {res['path']}", f"  source: {p['source'].get('how')} ({p['source'].get('file', 'captured')})",
           f"  measured on: {rp.phase_words(p)}"]
  if not p["layers"]:
    return "\n".join(lines + [f"  a ranking of {len(p['ranking'])} (layer, expert) pairs, no counts: a cross-check only (no reach, no hit rate)"])
  d = p["derived"]
  lines.append(f"  {d['tokens']} tokens, {d['layers']} layers measured" + (f" (not measured: {p['source']['layers_not_measured']})"
                                                                         if p["source"].get("layers_not_measured") else "") + f", top {d['top_k']} of {p['n_expert']}")
  lines.append(f"  the hottest 10% of experts take {d['top10_share']:.0%} of selections (uniform: 10%)")
  lines.append(f"  a hit rate from this profile is {rp.IN_SAMPLE}")
  for w in rp.CHUNKS:
    r = d["reached"][str(w)]
    lines.append(f"  experts a {w}-token chunk reaches per layer: mean {sum(r) / len(r):.1f}, range {min(r):.1f}-{max(r):.1f} (uniform {d['uniform'][str(w)]:.1f})")
  if other:
    lines.append(f"  rank correlation with {other['source'].get('how')}: {rp.spearman(rp.ranks(p), rp.ranks(other))['rho']:.3f}")
  return "\n".join(lines)
