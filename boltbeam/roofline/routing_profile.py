"""A routing profile: which experts a mixture-of-experts model's router picks, per layer, on real text
(boltbeam.routing_profile.v1, schemas/routing_profile.schema.json). Its layers are the record prefill_work.experts_reached
already takes (roofline/moe's form: {"layer", "shape": [tokens, top_k], "expert_counts": {expert: count}}), so the
prompt limit reads a profile as it is. Derived from the counts, never stored apart from them: the hotness ranking (every
(layer, expert) by its share of the layer's tokens), the experts a chunk of N tokens reaches, and the hit rate of a cache
holding the hottest experts. A source with a ranking and no counts keeps it in `ranking` (a cross-check only: no reach,
no hit rate). Engine-blind: an engine's adapter (data/routing_adapters.json) turns its own record into one of these."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any

SCHEMA = "boltbeam.routing_profile.v1"
CHUNKS = tuple(256 << i for i in range(6))  # 256 .. 8192 tokens
HEAD_BYTES = 16 << 20  # a model is known by its first file's first 16 MiB (its header) and the bytes of all its files
MEASURED, UNIFORM = "a measured routing profile", "uniform routing (assumed: no measured routing profile)"


def layers_from(tokens:dict[int, list[list[int]]]) -> list[dict[str, Any]]:
  """The per-layer records from each layer's selections, one list of expert ids per token."""
  out = []
  for il in sorted(tokens):
    rows = tokens[il]
    counts: dict[str, int] = {}
    for row in rows:
      for e in row:
        counts[str(int(e))] = counts.get(str(int(e)), 0) + 1
    out.append({"layer": int(il), "shape": [len(rows), len(rows[0]) if rows else 0], "expert_counts": counts})
  return out


def identity(model:str | pathlib.Path) -> dict[str, Any]:
  """The model's id: the sha256 of its first file's first 16 MiB and the bytes of every file of a split model."""
  p = pathlib.Path(model).expanduser().resolve()
  parts = sorted(p.parent.glob(p.name.replace("00001-of-", "*-of-"))) if "-00001-of-" in p.name else [p]
  with open(p, "rb") as f:
    head = hashlib.sha256(f.read(HEAD_BYTES)).hexdigest()
  return {"file": p.name, "bytes": sum(x.stat().st_size for x in parts), "files": len(parts), "head_sha256": head, "id": head[:16]}


def record(layers:list[dict[str, Any]] | None, *, model:dict[str, Any], source:dict[str, Any], n_expert:int,
           ranking:list[list[int]] | None = None) -> dict[str, Any]:
  """A profile from per-layer counts (or a ranking alone), with what is derived from them for a reader. A layer the source
  saw for under half the most tokens any layer saw is left out and named in source.layers_not_measured (a reader takes
  it as uniform): its few tokens are not its routing."""
  most = max((r["shape"][0] for r in layers or []), default=0)
  short = [r["layer"] for r in layers or [] if r["shape"][0] * 2 < most]
  if short:
    layers, source = [r for r in layers if r["layer"] not in short], {**source, "layers_not_measured": short}
  p = {"schema": SCHEMA, "model": model, "source": source, "n_expert": n_expert, "layers": layers or [], **({"ranking": ranking} if ranking else {})}
  check(p)
  if layers:
    p["derived"] = summary(p)
  return p


def check(p:dict[str, Any]) -> None:
  """ValueError when the record is not a profile: the schema, the model id, and each layer's counts covering its tokens."""
  from boltbeam.roofline.moe import selected_experts
  if p.get("schema") != SCHEMA or not (p.get("model") or {}).get("id") or type(p.get("n_expert")) is not int:
    raise ValueError(f"not a {SCHEMA} record (schema, model.id and n_expert are required)")
  if not p.get("layers") and not p.get("ranking"):
    raise ValueError("a routing profile needs per-layer counts or a ranking")
  for r in p.get("layers") or []:
    selected_experts(r, tokens=r["shape"][0], top_k=r["shape"][1], experts=p["n_expert"])
  if len({r["layer"] for r in p.get("layers") or []}) != len(p.get("layers") or []):
    raise ValueError("a layer appears twice in the routing profile")


def reached(p:dict[str, Any], w:int) -> list[float]:
  """Per layer, the expected distinct experts a chunk of w tokens reaches (prefill_work.experts_reached of the counts)."""
  from boltbeam.roofline.prefill_work import experts_reached
  ls = p["layers"]
  return experts_reached(w, p["n_expert"], ls[0]["shape"][1], len(ls), ls)


def hottest(p:dict[str, Any]) -> list[tuple[int, int, float]]:
  """(layer, expert, share of the layer's tokens that select it), hottest first; ties by layer then expert."""
  rows = [(r["layer"], int(e), c / r["shape"][0]) for r in p["layers"] for e, c in r["expert_counts"].items()]
  return sorted(rows, key=lambda x: (-x[2], x[0], x[1]))


def hit_rate(p:dict[str, Any], slots:int) -> float:
  """The share of all selections a cache holding the `slots` hottest (layer, expert) pairs serves (every layer reads
  top_k experts a token, so a selection share is a byte share when the experts are one size)."""
  hot = hottest(p)
  return sum(s for *_, s in hot[:max(slots, 0)]) / sum(s for *_, s in hot) if hot else 0.0


def concentration(p:dict[str, Any], top:float = 0.1) -> float:
  """The share of selections the hottest `top` of all (layer, expert) pairs take (uniform: `top`)."""
  return hit_rate(p, round(top * len(p["layers"]) * p["n_expert"]))


def summary(p:dict[str, Any]) -> dict[str, Any]:
  """What a reader reads off the counts: tokens, per chunk the experts reached per layer against uniform routing, and
  the top-10% concentration."""
  k, n = p["layers"][0]["shape"][1], p["n_expert"]
  return {"tokens": min(r["shape"][0] for r in p["layers"]), "top_k": k, "layers": len(p["layers"]),
          "reached": {str(w): [round(x, 2) for x in reached(p, w)] for w in CHUNKS},
          "uniform": {str(w): round(n * (1 - (1 - k / n) ** w), 2) for w in CHUNKS},
          "top10_share": round(concentration(p), 4)}


def ranks(p:dict[str, Any]) -> list[tuple[int, int]]:
  """The (layer, expert) order: the counts' hottest first, else the source's own ranking."""
  return [(l, e) for l, e, _ in hottest(p)] if p.get("layers") else [tuple(x) for x in p["ranking"]]


def spearman(a:list[tuple[int, int]], b:list[tuple[int, int]]) -> dict[str, Any]:
  """Spearman's rank correlation of two orders over the pairs both rank (ties broken by order)."""
  ra, rb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
  both = [x for x in a if x in rb]
  n = len(both)
  if n < 2:
    return {"rho": None, "pairs": n}
  sa, sb = {x: i for i, x in enumerate(sorted(both, key=ra.get))}, {x: i for i, x in enumerate(sorted(both, key=rb.get))}
  return {"rho": 1 - 6 * sum((sa[x] - sb[x]) ** 2 for x in both) / (n * (n * n - 1)), "pairs": n}


def home() -> pathlib.Path:
  return pathlib.Path.home() / ".boltbeam" / "routing"


def engine_name(p:dict[str, Any]) -> str:
  """The engine the profile was measured on, as a file-name part ("unknown" when the source names none)."""
  return re.sub(r"[^A-Za-z0-9_-]+", "-", str(p.get("source", {}).get("engine") or "")).strip("-") or "unknown"


def path_for(p:dict[str, Any], where:pathlib.Path | None = None) -> pathlib.Path:
  """<model id>.<engine>.json (a ranking alone: <model id>.<engine>.ranking.json) under where (default home())."""
  return (where or home()) / f"{p['model']['id']}.{engine_name(p)}{'' if p.get('layers') else '.ranking'}.json"


def save(p:dict[str, Any], out:pathlib.Path | None = None) -> pathlib.Path:
  """Where every command looks for it (find): ~/.boltbeam/routing/<model id>.<engine>.json, unless out is given."""
  out = out or path_for(p)
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_suffix(".tmp")
  tmp.write_text(json.dumps(p, indent=1))
  tmp.replace(out)
  return out


def find(model:str | pathlib.Path | None, where:pathlib.Path | None = None, engine:str | None = None) -> dict[str, Any] | None:
  """The saved profile with counts of this model (by its id), or None: no model file, none saved, or a ranking only.
  engine: the one asking; the profile measured on it is preferred, else another engine's (or the older <id>.json) is
  returned, and words() says it was measured elsewhere."""
  try:
    ident = identity(model) if model else None
  except (OSError, TypeError):
    return None
  if not ident:
    return None
  d = where or home()
  mine = f"{ident['id']}.{re.sub(r'[^A-Za-z0-9_-]+', '-', engine or '').strip('-')}.json" if engine else None
  others = sorted(x for x in d.glob(f"{ident['id']}.*.json") if not x.name.endswith(".ranking.json") and x.name != mine) if d.is_dir() else []
  for f in ([d / mine] if mine else []) + others + [d / f"{ident['id']}.json"]:
    if not f.is_file():
      continue
    try:
      p = json.loads(f.read_text())
      check(p)
      same = bool(p["layers"]) and p["model"]["head_sha256"] == ident["head_sha256"]
    except (OSError, ValueError, KeyError, TypeError):  # a damaged file is said, and the reader tries the next, else uniform routing
      import warnings
      warnings.warn(f"the saved routing profile {f.name} cannot be read; the next one, or uniform routing, is used", RuntimeWarning)
      continue
    if same:
      return p
  return None


def phase_words(p:dict[str, Any]) -> str:
  """Whether the profile was measured while reading a prompt or writing replies, in words ("unknown" when not recorded)."""
  return {"prefill": "prompt reading (prefill)", "decode": "reply writing (decode)"}.get(p["source"].get("phase"), "prompts or replies: not recorded")


def words(p:dict[str, Any] | None, asking:str | None = None) -> str:
  """Which routing a number used, in words. asking: the engine the number is for; a profile measured on another one says so."""
  if not p:
    return UNIFORM
  eng = p["source"].get("engine")
  other = f"; measured on another engine ({eng})" if asking and eng and eng != asking else ""
  return f"{MEASURED} ({eng}, {p['derived']['tokens']} tokens of {p['source'].get('corpus')}, {"measured on " if p["source"].get("phase") else ""}{phase_words(p)}{other})"


IN_SAMPLE = "in-sample, hottest-first: an upper estimate"


def hit_words(p:dict[str, Any] | None) -> str:
  """What a hit rate from a profile is: measured on the text it is ranked on, for a cache filled hottest-first."""
  return IN_SAMPLE + " (measured on the same text the experts are ranked on; a cache that keeps the recently used experts (LRU) hits less)" if p else \
    "the cache's share of the experts, under uniform routing"


