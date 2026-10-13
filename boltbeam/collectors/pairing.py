"""Which model role each captured kernel ran for: the layer walk (docs/pairing-layer-walk-20261011.md).

One rule set for prefill and decode. A capture is a sequence of launches in the order they ran. Every engine runs a
layer in one order: the projections that read the layer's input, the block (attention or the recurrent scan), the
output projection, the FFN. The block's kernels are the ones tie_out.kind_of names, and they mark the layers:

  layers   the block launches, in runs (consecutive block launches with no weight launch between them), grouped by
           how many layers of each kind the profile has (runs per layer per kind is a whole number)
  sides    between two blocks lie the earlier layer's output-side launches, then the later layer's input-side ones;
           the launches before the first block are the first layer's input side, and every gap of one kind pair has
           one size, so the split is solved from the sizes
  slots    the j-th weight launch on one side of every layer of one kind is the same role (an engine launches a
           layer's projections in one order); a slot group is (slot, template), with the layers it occurs in
  rows     a profile row's fine class sets its stage (vocab.STAGE): input stages take input-side slots, output stages
           output-side slots in the data flow's order (vocab.OUTPUT_ORDER), the gate projection either side, the
           head the tail of the unit; a template serves one weight format; within one template on the input side
           more operations per launch is more time per launch, and two rows within GAP_RATIO are one row
  checks   a launch cannot read a role's weights faster than the measured bandwidth or run its operations past its
           path's measured peak; a launch may serve several calls (a fused kernel) or a call several launches (a
           product cut along M), by a whole ratio the slot count gives

Prefill and decode differ in the unit (a chunk of M tokens; a token, M = 1) and in which launches are weight launches
(the caller's `candidate`). The tail of a unit (after its last layer: the head, sampling) pairs at M = 1 by bytes.
A window with no launch times, or no block kernel named, pairs the flat way: counts per unit, bytes and time, no sides.

An engine may prune its last layer (the engine adapter says so, `prune`): the rows of the pruned stages run in every
layer but the unit's last at the unit's width, and in the last layer at its output tokens only, as kernels the other
layers never launch (a GEMV). A pruned row's work is so split: its layers less one at every token, and one layer at
output width, in the units that had output tokens; those small launches are that row at output width (side PRUNED).
"""
from __future__ import annotations

import collections
import itertools
import statistics
from typing import Any, Callable

from boltbeam.collectors.instruction_path import template
from boltbeam.vocab import OUTPUT_ORDER, ROLE_GROUP_OF, STAGE, STAGE_SIDE
from boltbeam.workflow.tie_out import UNATTRIBUTED, kind_of

PEAK_SLACK = 1.05  # a measured peak or bandwidth may be beaten by measurement noise only
GAP_RATIO = 1.3  # input-side rows of one format whose operations per launch are within this cannot be told apart by time
MAX_FUSED = 3  # calls one launch may serve (gate and up fused: 2)
MAX_SPLIT = 4  # launches one call may take (a library that cuts a product along M)
MAX_NODES = 50_000
MAX_SUM_STEPS = 20_000  # subsets one row's choice may visit
RECUR = 0.9  # a slot's kernel runs in at least this share of a kind's layers in a unit, or of the units' last layers
BLOCKS = ("attention", "recurrent")  # kernels tie_out.kind_of names so are the block, never a weight role
RULE = ("the block kernels (attention, recurrent) mark the layers; the launches before a layer's block are its input "
        "projections and those after it its output projection and FFN, split by the gap sizes; the j-th weight launch on a "
        "side is one role in every layer of that kind; a row's fine class puts it on a side and, on the output side, in the "
        "data flow's order; a template serves one weight format; on the input side more operations per launch is more "
        "time, and two rows within the gap are one row; no launch reads a role's weights faster than the bandwidth or runs "
        f"past its path's peak; a launch may serve up to {MAX_FUSED} calls or a call take up to {MAX_SPLIT} launches, by a "
        "whole ratio; the head pairs on the unit's tail by bytes; the pairing placing the most weight bytes wins, and rows "
        "that could exchange their launches are left unpaired and said")
FLAT_RULE = ("no layer structure (no launch times, or no attention or recurrent kernel named): a role takes kernel groups "
             "whose launches per unit are its count, or 1/k of them for a fused kernel, within the bandwidth; more bytes per "
             "launch is more time per launch; rows that could exchange their groups are left unpaired and said")
PRUNED = "pruned"  # the side and kind of a pruned last layer's launches (`prune`)
_RANK = {s: i for i, s in enumerate(("in", "gate", "out", "ffn_in", "ffn_out", "head"))}


def projection_candidate(name:str) -> bool:
  """A known standalone non-matrix operation cannot read a projection's weights. Fused matrix names win."""
  if any(w in name.lower() for w in ("gemm", "gemv", "matmul", "mul_mat", "mul_mv")):
    return True
  return kind_of({"kernel": name}) in ("other", UNATTRIBUTED)


def layer_kinds(profile:dict[str, Any]) -> dict[str, int]:
  """Layers per block kind: the profile's recurrent layers, and the rest attend."""
  md = profile.get("metadata") or {}
  total = int(profile.get("layer_count") or 0)
  rec = int((md.get("recurrent") or {}).get("layer_count") or 0)
  return {**({"recurrent": rec} if rec else {}), **({"attention": total - rec} if total > rec else {})}


def stages(row:dict[str, Any], per_layer:float | None = None) -> set[str]:
  """The stages a row's launches may be at. A packed matrix (fine classes) runs one launch at its earliest class's
  stage. A row of one class, or one launch per layer, is at its class's stage. A row of a group with several classes
  and several launches per layer (attn_qo: q and o) holds one launch per class."""
  if row.get("fine"):
    got = {STAGE[c] for c in row["fine"] if c in STAGE}
    return {min(got, key=_RANK.__getitem__)} if got else set()
  cls = row.get("role_class")
  group = ROLE_GROUP_OF.get(cls, row["role"]) if cls else row["role"]
  members = [c for c, g in ROLE_GROUP_OF.items() if g == group]
  if cls and (len(members) < 2 or per_layer == 1):
    return {STAGE[cls]} if cls in STAGE else set()
  return {STAGE[c] for c in members if c in STAGE}


def kinds_allowed(row:dict[str, Any], kinds:dict[str, int]) -> list[str]:
  """The layer kinds a row may run in: a recurrent class (ssm_*) in recurrent layers, q/k/v/o in attending ones, every
  other class (the loaders name a delta net's in_proj by attention classes) in any."""
  cls = row.get("role_class") or ""
  if cls.startswith("ssm_") and "recurrent" in kinds:
    return ["recurrent"]
  if cls in ("attention_q", "attention_k", "attention_v", "attention_o") and "attention" in kinds:
    return ["attention"]
  return list(kinds)


# --- the structure: layers, sides, slots -------------------------------------------------------------------------------


class NoStructure(ValueError):
  """The layers could not be told apart. The message says why."""


def structure(launches:list[dict[str, Any]], *, kinds:dict[str, int], units:int,
              candidate:Callable[[str], bool]) -> dict[str, Any]:
  """The layer walk of a window: launches in stream order, their block runs grouped into layers, the sides solved.
  Returns {"seq", "layers": [{kind, unit, input: [idx], output: [idx]}], "tail": [[idx] per unit], "sides": {kind:
  {input, inside, output}}}; NoStructure when the window does not read as layers."""
  if any(l.get("start_us") is None for l in launches):
    raise NoStructure("launches without a start time")
  seq = sorted(launches, key=lambda l: l["start_us"])
  kind = [kind_of({"kernel": l["key"]}) for l in seq]
  kind = [k if k in BLOCKS else ("C" if candidate(template(seq[i]["key"])) else "O") for i, k in enumerate(kind)]
  runs: list[dict[str, Any]] = []  # block launches of one kind with no weight launch between them
  for i, k in enumerate(kind):
    if k not in BLOCKS:
      continue
    if runs and runs[-1]["kind"] == k and not any(kind[j] == "C" for j in range(runs[-1]["last"] + 1, i)):
      runs[-1]["last"] = i
    else:
      runs.append({"kind": k, "first": i, "last": i})
  per_unit = sum(kinds.values())
  if not runs or not per_unit:
    raise NoStructure("no attention or recurrent kernel is named, so the layers cannot be told apart")
  count = collections.Counter(r["kind"] for r in runs)
  rpl = {k: count[k] / (units * n) for k, n in kinds.items() if n}
  if any(k not in rpl for k in count) or any(v < 1 or v != int(v) for v in rpl.values()):
    raise NoStructure(f"{dict(count)} block runs do not group into {units} x {kinds} layers")
  layers: list[dict[str, Any]] = []
  i = 0
  while i < len(runs):
    k, n = runs[i]["kind"], int(rpl[runs[i]["kind"]])
    if any(r["kind"] != k for r in runs[i:i + n]):
      raise NoStructure("block runs of two kinds in one layer")
    layers.append({"kind": k, "first": runs[i]["first"], "last": runs[i + n - 1]["last"], "unit": len(layers) // per_unit})
    i += n
  cands = lambda a, b: [j for j in range(a, b) if kind[j] == "C"]  # noqa: E731
  # a layer's launches recur: a kernel that is a slot of kind K runs in (nearly) every kind-K layer of a unit, or in
  # every unit's last layer (an engine's last layer may run other variants); one that does not (a set-up kernel, a
  # kernel some layers run and others skip) is no slot, and would shift every slot after it
  seen: dict[tuple[str, str, str], set] = collections.defaultdict(set)  # (template, kind, before/after) -> layers
  last_of_unit: dict[tuple[str, str], set] = collections.defaultdict(set)  # (template, kind) -> units whose last layer runs it after
  for i, l in enumerate(layers):
    lo = layers[i - 1]["last"] + 1 if i else 0
    hi = layers[i + 1]["first"] if i + 1 < len(layers) else len(seq)
    for j in cands(lo, l["first"]) + cands(l["first"], l["last"]):
      seen[(template(seq[j]["key"]), l["kind"], "before")].add(i)
    for j in cands(l["last"] + 1, hi):
      seen[(template(seq[j]["key"]), l["kind"], "after")].add(i)
      if (i + 1) % per_unit == 0:
        last_of_unit[(template(seq[j]["key"]), l["kind"])].add(l["unit"])
  recurs: set[tuple[str, str]] = set()
  for (t, k, side), ls in seen.items():
    per = collections.Counter(layers[i]["unit"] for i in ls)
    if any(n >= RECUR * kinds[k] for n in per.values()) or (side == "after" and len(last_of_unit.get((t, k), ())) >= RECUR * units):
      recurs.add((t, k))
  # a weight format some layers use and others do not (a mixed quant per layer) takes the same slot in each. When the
  # recurring templates leave the gaps between layers uneven and every candidate makes them even (a unit's last gap
  # aside), the templates in those gaps are slots: dropping them is what would shift the slots
  between, every, kept = set(), collections.defaultdict(set), collections.defaultdict(set)
  for i, l in enumerate(layers[:-1]):
    if (i + 1) % per_unit:
      pair, gap = (l["kind"], layers[i + 1]["kind"]), cands(l["last"] + 1, layers[i + 1]["first"])
      between |= {template(seq[j]["key"]) for j in gap}
      every[pair].add(len(gap))
      kept[pair].add(sum(1 for j in gap if {(template(seq[j]["key"]), k) for k in pair} & recurs))
  if every and all(len(v) == 1 for v in every.values()) and any(len(v) > 1 for v in kept.values()):
    recurs |= {(t, k) for t, k, _ in seen if t in between}
  for i, l in enumerate(layers):
    lo = layers[i - 1]["last"] + 1 if i else 0
    hi = layers[i + 1]["first"] if i + 1 < len(layers) else len(seq)
    for j in range(lo, hi):
      if kind[j] == "C":
        t = template(seq[j]["key"])
        own = (t, l["kind"]) in recurs or (j > l["last"] and i + 1 < len(layers) and (t, layers[i + 1]["kind"]) in recurs)
        own = own or (j < l["first"] and i and (t, layers[i - 1]["kind"]) in recurs)
        if not own:
          kind[j] = "O"
  # the gap between two blocks holds the earlier layer's output launches, then the later layer's input launches: a
  # launch there is a slot on both sides, counted from its own block, and the walk decides which side takes it
  # (never both, and never an input launch before an output one); launches inside a block are input launches
  tail: list[list[int]] = [[] for _ in range(units)]
  for i, l in enumerate(layers):
    lo = layers[i - 1]["last"] + 1 if i else 0
    hi = layers[i + 1]["first"] if i + 1 < len(layers) else len(seq)
    l["input"] = cands(lo, l["first"]) + cands(l["first"], l["last"])
    l["inside"] = len(cands(l["first"], l["last"]))
    l["output"] = cands(l["last"] + 1, hi)
    l["gap"] = (i, len(l["output"]), l["kind"], layers[i + 1]["kind"] if i + 1 < len(layers) else None,
                (i + 1) % per_unit == 0)  # the gap after this layer; a unit's last layer may skip its output rows
    if (i + 1) % per_unit == 0:  # the unit's tail: every non-block launch after its last layer's block
      tail[l["unit"]] = [j for j in range(l["last"] + 1, hi) if kind[j] not in BLOCKS]
  inside = {k: sorted({l["inside"] for l in layers if l["kind"] == k}) for k in kinds if count.get(k)}
  gaps = {k: sorted({len(l["output"]) for l in layers if l["kind"] == k}) for k in kinds if count.get(k)}
  return {"seq": seq, "layers": layers, "tail": tail, "units": units,
          "sides": {k: {"inside": inside[k], "gaps_after": gaps[k]} for k in inside}}


# --- rows to slots -----------------------------------------------------------------------------------------------------


def _groups(st:dict[str, Any], widths:list[int]) -> list[dict[str, Any]]:
  """Slot groups: (kind, side, position, template) with their launches and each launch's M. Output positions count
  from the layer's block forward; input positions count from the layer's block backward (-1 is the launch just
  before it). A gap launch is in both its layer's output group and the next layer's input group. The tail's groups
  are by key at M = 1. Each launch carries its gap, for the one order rule a gap has."""
  seq = st["seq"]
  out: dict[tuple, dict[str, Any]] = {}
  for n, l in enumerate(st["layers"]):
    for q, idx in enumerate(l["input"]):
      t = template(seq[idx]["key"])
      pos = q - len(l["input"])
      g = out.setdefault((l["kind"], "input", pos, t), {"kind": l["kind"], "side": "input", "slot": pos, "template": t, "launches": [],
                                                     "inside": q >= len(l["input"]) - l["inside"]})
      g["launches"].append((seq[idx], widths[l["unit"]], st["layers"][n - 1]["gap"] if n and q < len(l["input"]) - l["inside"] else None, q))
    for p, idx in enumerate(l["output"]):
      t = template(seq[idx]["key"])
      g = out.setdefault((l["kind"], "output", p, t), {"kind": l["kind"], "side": "output", "slot": p, "template": t, "launches": [],
                                                      "layers": set()})
      g["launches"].append((seq[idx], widths[l["unit"]], l["gap"], p))
      g["layers"].add(n)
  for idxs in st["tail"]:
    for idx in idxs:
      key = seq[idx]["key"]
      g = out.setdefault(("tail", "tail", key, None), {"kind": "tail", "side": "tail", "slot": key, "template": template(key), "launches": []})
      g["launches"].append((seq[idx], 1, None, None))
  return [_finish(g) for g in out.values()]


def _finish(g:dict[str, Any]) -> dict[str, Any]:
  g["launches"] = [(x[0], x[1], x[2] if len(x) > 2 else None, x[3] if len(x) > 3 else None) for x in g["launches"]]
  g["n"] = len(g["launches"])
  g["median_us"] = statistics.median(l.get("raw_wall_us", l["wall_us"]) for l, _, _, _ in g["launches"])
  g["m"] = statistics.median(m for _, m, _, _ in g["launches"])
  return g


def flat_groups(groups:list[dict[str, Any]], m:float = 1) -> list[dict[str, Any]]:
  """Kernel groups ({key, calls, wall_us, launches: [(raw_us, share_us)] or [launch dicts]}) as slot groups of one
  side, no order, each launch at M = m (the units' mean width); a block kernel (attention, recurrent) serves no role
  and is left out."""
  out = []
  for g in groups:
    if kind_of({"kernel": g["key"]}) in BLOCKS:
      continue
    raw = g.get("launches") or []
    launches = [((l if isinstance(l, dict) else {"key": g["key"], "wall_us": l[1], "raw_wall_us": l[0]}), m) for l in raw]
    if len(launches) != g["calls"]:
      launches = [({"key": g["key"], "wall_us": g["wall_us"] / g["calls"], "raw_wall_us": g["wall_us"] / g["calls"]}, m)] * g["calls"]
    out.append(_finish({"kind": "flat", "side": "flat", "slot": g["key"], "template": template(g["key"]), "launches": [(l, m, None, None) for l, m in launches], "key": g["key"]}))
  return out


def _fits(item:dict[str, Any], g:dict[str, Any], k:int, r:int, bandwidth_gbs:float | None, peak:float | None) -> bool:
  """One launch of the group, serving k calls of the item at M / r rows: within the bandwidth and the peak."""
  # Explicit call semantics constrain candidates before timing bounds. Missing dimensions do not prove a
  # match; they simply leave the candidate unresolved for the ambiguity check.
  if not projection_candidate(g.get("template", "")) and not all(
      launch.get("logical_parts") for launch, *_ in g.get("launches", [])):
    return False
  for launch, m, _, _ in g.get("launches", []):
    if proof := launch.get("matrix_grid_constraint"):
      from boltbeam.collectors.llama_mmq import fits
      if not fits(proof, launch, item.get("parts", [item]), m, k, r): return False
    for part in launch.get("logical_parts") or []:
      if not any((part.get("role") in (None, row["role"]) and part.get("quant") in (None, row["quant"]) and
                  (not part.get("shape") or part["shape"] == [m / r, row.get("n"), row.get("k")]))
                 for row in item.get("parts", [item])):
        return False
    if launch.get("matrix_calls") is not None and launch["matrix_calls"] != k:
      return False
  us = g["median_us"]
  if bandwidth_gbs and item["weight_bytes"] * k / (us * 1e-6) > bandwidth_gbs * 1e9 * PEAK_SLACK:
    return False
  if peak and item.get("nk") and 2.0 * (g["m"] / r) * item["nk"] * k / (us * 1e-6) > peak * 1e12 * PEAK_SLACK:
    return False
  return True


def assign(groups:list[dict[str, Any]], rows:list[dict[str, Any]], *, units:int, outputs:int | None = None,
           kinds:dict[str, int] | None = None, bandwidth_gbs:float | None = None,
           peak_of:Callable[[str], float | None] = lambda t: None, prune:dict[str, Any] | None = None) -> dict[str, Any]:
  """Rows to slot groups (_groups or flat_groups). prune: {stages, kind, units} (pair's): the pruned layer's rows. rows: {role, quant, count (per unit), weight_bytes (per call),
  nk (n x k; None at M = 1), role_class or fine}. outputs: the calls that ended in an output (the head's launches per
  window; the units when None). Returns {"placed": [{item, group, fused, split}], "unplaced": [{item, reason}],
  "nodes", "ambiguous", "words"}. The assignment placing the most weight bytes wins; among
  equals the one pairing the most time."""
  kinds = kinds or {}
  flat = not kinds or not any(g["kind"] not in ("flat", "tail") for g in groups)
  splits = [{}] if flat else _kind_splits(rows, kinds, units)
  best: dict[str, Any] = {"score": (-1, 0.0), "placed": [], "unplaced": [], "nodes": 0, "ambiguous": None}
  for split, carry in itertools.product(splits, _carriers(rows, prune)):  # one node budget over every split
    items = _items(rows, kinds, units, outputs if outputs is not None else units, split, flat, prune, carry)
    got = _search(items, groups, bandwidth_gbs, peak_of, best["nodes"])
    best["nodes"] = got["nodes"]
    if got["score"] > best["score"]:
      best.update({k: v for k, v in got.items() if k != "nodes"})
    if got["nodes"] > MAX_NODES:
      best["ambiguous"] = got["ambiguous"]  # the walk ran out of nodes: what it found is said as unsure
      break
    if not got["unplaced"]:
      break
  best["words"] = FLAT_RULE if flat else RULE
  return best


def _kind_splits(rows:list[dict[str, Any]], kinds:dict[str, int], units:int) -> list[dict[tuple, int]]:
  """Ways to spread each row's calls per unit over the layer kinds it may run in: whole calls per layer when the
  model has several kinds (a hybrid), else the row's count. Keyed (role, quant, kind) -> launches in the window; the
  evenest spread first."""
  options: list[list[dict[tuple, int]]] = []
  for r in rows:
    if "head" in stages(r):
      continue
    ks = kinds_allowed(r, kinds)
    if len(kinds) == 1:
      options.append([{(r["role"], r["quant"], ks[0]): r["count"] * units}])
      continue
    ways: list[dict[tuple, int]] = []

    def spread(i, left, acc):
      if i == len(ks):
        if left == 0:
          ways.append(dict(acc))
        return
      for c in range(left // kinds[ks[i]], -1, -1):
        spread(i + 1, left - c * kinds[ks[i]], {**acc, (r["role"], r["quant"], ks[i]): c * kinds[ks[i]] * units})
    spread(0, r["count"], {})
    mean = r["count"] / sum(kinds[k] for k in ks)
    ways.sort(key=lambda w: sum(abs(v / units / kinds[k[2]] - mean) for k, v in w.items()))
    options.append(ways or [{}])
  out: list[dict[tuple, int]] = [{}]
  for ways in options:
    out = [{**o, **w} for o in out for w in ways][:256]
  return out or [{}]


def _carriers(rows:list[dict[str, Any]], prune:dict[str, Any] | None) -> list[dict[str, str]]:
  """Which format carries the pruned last layer, per role of the pruned stages held in several formats (a mixed quant
  per layer: each format runs in some layers only, and the last layer is one of them); every way, for the walk to try."""
  formats: dict[str, list[str]] = collections.defaultdict(list)
  for r in rows if prune else []:
    if "head" not in stages(r) and stages(r) & set(prune["stages"]):
      formats[r["role"]].append(r["quant"])
  roles = sorted(x for x, q in formats.items() if len(q) > 1)
  return [dict(zip(roles, combo)) for combo in itertools.product(*(formats[x] for x in roles))]


def _items(rows, kinds, units, outputs, split, flat, prune=None, carry=None) -> list[dict[str, Any]]:
  """One item per (row, kind) with its launches in the window and its sides; head rows on the tail; input-side rows
  of one format and kind whose operations are within GAP_RATIO joined. carry: {role: format} (_carriers), the format
  whose row runs the pruned layer when the role's formats each run in some layers only."""
  items: list[dict[str, Any]] = []
  per_role: dict[tuple[str, str], float] = collections.Counter()  # (role, kind) -> launches per layer over its formats
  for (role, _, k), n in split.items():
    per_role[(role, k)] += n / (kinds[k] * units)
  for r in rows:
    if "head" in stages(r):
      items.append({"row": r, "parts": [r], "need": r["count"] * outputs, "sides": {"flat"} if flat else {"tail"},
                    "stages": {"head"}, "kind": None, **_facts(r)})
    elif flat:
      items.append({"row": r, "parts": [r], "need": r["count"] * units, "sides": {"flat"}, "stages": stages(r), "kind": None, **_facts(r)})
    else:
      for k in kinds_allowed(r, kinds):
        need = split.get((r["role"], r["quant"], k), 0)
        if not need:
          continue
        stg = stages(r, need / (kinds[k] * units))
        sides = {STAGE_SIDE.get(s) for s in stg} or {None}
        per_layer = need / (kinds[k] * units)
        if (carry or {}).get(r["role"]) == r["quant"]:  # the role's formats share the layers: the last is this one's
          per_layer = per_role[(r["role"], k)]
        if prune and k == prune["kind"] and stg and stg <= set(prune["stages"]) and per_layer == int(per_layer):
          need -= int(per_layer) * units  # the unit's last layer runs this row at output width: its own item
          items.append({"row": r, "parts": [r], "need": int(per_layer) * prune["units"], "kind": PRUNED, "stages": stg,
                        **_facts(r), "slack": 0, "sides": {PRUNED}})
        items.append({"row": r, "parts": [r], "need": need, "kind": k, "stages": stg, **_facts(r), "per_layer": per_layer,
                      "slack": min(int(per_layer) * units, need - 1) if kinds[k] > 1 else 0,
                      "sides": {s for s in ("input", "output") if None in sides or s in sides}})
  out: list[dict[str, Any]] = []
  for it in sorted(items, key=lambda i: (str(i["kind"]), str(i["quant"]), -(i.get("nk") or 0))):
    last = out[-1] if out else None
    if (last and it["kind"] and last["kind"] == it["kind"] and last["quant"] == it["quant"] and it["sides"] == last["sides"] == {"input"}
        and it.get("nk") and last.get("nk") and last["nk"] < it["nk"] * GAP_RATIO
        and kinds_allowed(last["row"], kinds) == kinds_allowed(it["row"], kinds)):
      total = last["need"] + it["need"]
      last.update(role=" + ".join(sorted(set(last["role"].split(" + ")) | {it["role"]})), need=total, parts=last["parts"] + [it["row"]],
                  slack=last.get("slack", 0) + it.get("slack", 0), per_layer=last.get("per_layer", 1) + it.get("per_layer", 1),
                  weight_bytes=(last["weight_bytes"] * last["need"] + it["weight_bytes"] * it["need"]) / total,
                  nk=(last["nk"] * last["need"] + it["nk"] * it["need"]) / total, count=last["count"] + it["count"])
      continue
    out.append(it)
  def order(i):  # output-side rows in the data flow's order, then the rest most bytes first (they bind their templates'
    stage = _output_stage(i)  # formats for the lighter rows after them); the head last
    rank = 2 if "head" in i["stages"] else 1.5 if i["kind"] == PRUNED else 0 if stage else 1
    return (rank, OUTPUT_ORDER.index(stage) if stage else 0, -(i["weight_bytes"] * i["need"]), i["role"])
  out.sort(key=order)
  return out


def _facts(r:dict[str, Any]) -> dict[str, Any]:
  return {"role": r["role"], "quant": r["quant"], "count": r["count"], "weight_bytes": float(r["weight_bytes"]), "nk": r.get("nk")}


def _bind(items:list[dict[str, Any]], groups:list[dict[str, Any]]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], tuple[int, int]]]:
  """Which weight format each template serves and at what launch ratio, from the counts: a template serves one format,
  and a format's rows fill its templates' launches (less what a unit's last layer skips), each template at one whole
  ratio (r launches per call for a library that cuts a product along M; one launch per k calls for a fused kernel).
  Counted over the layers' weight launches, each once (a gap launch is a slot on both sides). Keyed (kind, template) for
  every kind the template runs in. Among the ways the counts balance, the one skipping the fewest calls; none, or
  two such, bind nothing: the walk then binds a
  template when a row takes it (decode, where every kernel is a candidate)."""
  fmt: dict[tuple[str, str], str] = {}
  ratios: dict[tuple[str, str], tuple[int, int]] = {}
  for scope in ({g["kind"] for g in groups if g["kind"] == "flat"}, {g["kind"] for g in groups if g["kind"] not in ("flat", "tail", PRUNED)}):
    if not scope:
      continue
    launches: dict[str, set] = collections.defaultdict(set)
    kinds_of: dict[str, set] = collections.defaultdict(set)
    for g in groups:
      if g["kind"] in scope and g.get("weight", True):
        launches[g["template"]].update(id(x[0]) for x in g["launches"])
        kinds_of[g["template"]].add(g["kind"])
    slots = {t: len(ids) for t, ids in launches.items()}
    mine = [i for i in items if i["kind"] in scope or ("flat" in scope and i["kind"] is None and "head" not in i["stages"])]
    need = collections.Counter()
    slack = collections.Counter()
    for i in mine:
      need[i["quant"]] += i["need"]
      slack[i["quant"]] += i.get("slack", 0)
    tmpls, formats = sorted(slots), sorted(need)
    if not tmpls or not formats or len(formats) ** len(tmpls) > 4096:
      continue
    one = [(1, 1)]
    more = one + [(1, r) for r in range(2, MAX_SPLIT + 1)] + [(k, 1) for k in range(2, MAX_FUSED + 1)]
    for opts in (one, more):
      if (len(formats) * len(opts)) ** len(tmpls) > 4096:
        break
      found: dict[int, list] = collections.defaultdict(list)  # calls the combo leaves skipped -> combos
      for combo in itertools.product([(f, kr) for f in formats for kr in opts], repeat=len(tmpls)):
        if any(slots[t] * k % r for t, (_, (k, r)) in zip(tmpls, combo)):
          continue
        total = collections.Counter()
        for t, (f, (k, r)) in zip(tmpls, combo):
          total[f] += slots[t] * k // r
        if all(need[f] - slack[f] <= total[f] <= need[f] for f in formats):
          found[sum(need[f] - total[f] for f in formats)].append(combo)
      if found:
        least = found[min(found)]  # a last layer's skip is the exception: the fewest calls skipped
        if len(least) == 1:
          for t, (f, kr) in zip(tmpls, least[0]):
            for kind in kinds_of[t]:
              fmt[(kind, t)] = f
              ratios[(kind, t)] = kr
        break
  return fmt, ratios


def _search(items:list[dict[str, Any]], groups:list[dict[str, Any]], bandwidth_gbs:float | None,
            peak_of:Callable[[str], float | None], nodes:int) -> dict[str, Any]:
  """Depth first over the items (_items' order): each takes the launches of slot groups of its sides and templates,
  in the output side's order, within the checks, preferring a template already serving its format and the slowest
  slots; then goes unplaced. A launch is taken once (a gap launch is a slot on both sides). In a gap, a layer's output
  launches come before the next layer's input launches. The first assignment placing every item wins; else the one
  placing the most weight bytes (calls times bytes per call: a light row fused onto a heavy row's launches never
  outbids the heavy row)."""
  best: dict[str, Any] = {"score": (-1, 0.0), "placed": [], "unplaced": [], "ambiguous": None}
  bound, ratios = _bind(items, groups)
  taken: set[int] = set()  # launch ids
  fmt: dict[tuple[str, str], str] = dict(bound)  # (kind, template) -> the weight format it serves
  used: dict[tuple[str, str], list] = collections.defaultdict(list)  # (kind, stage) -> output slot groups taken
  out_max: dict[int, int] = {}  # gap -> highest position an output row took
  in_min: dict[int, int] = {}  # gap -> lowest position an input row took
  placed: list[dict[str, Any]] = []
  state = {"nodes": nodes}
  # each row runs at least one launch per layer on its side: a gap's first launches are the earlier layer's output
  # rows', its last launches the later layer's input rows' (less the launches inside the block)
  # A row whose format runs in only some layers (a mixed quant per layer) is no launch in every layer: such rows
  # count together per role and side, once when their layers cover every layer.
  min_out: dict[str, int] = collections.Counter()
  min_in: dict[str, int] = collections.Counter()
  partial: dict[tuple[str, str, str], float] = collections.Counter()
  for it in items:
    if it["kind"] in (None, PRUNED):
      continue
    both = it["stages"] >= {"in", "out"}
    for side, mins in (("output", min_out), ("input", min_in)):
      if it["sides"] == {side} or both:
        share = it.get("per_layer", 1) / (2 if both else 1)
        if share >= 1:
          mins[it["kind"]] += 1
        else:
          partial[(it["kind"], side, it["role"])] += share
  for (kind, side, _), share in partial.items():
    if share >= 1:
      (min_out if side == "output" else min_in)[kind] += 1
  inside = collections.Counter()  # input launches inside a block, per kind
  for g in groups:
    if g["side"] == "input" and g.get("inside"):
      inside[g["kind"]] += 1
  for k in list(min_in):
    min_in[k] = max(min_in[k] - inside[k], 0)

  def record(unplaced):
    score = (sum(len(p["launches"]) * p["fused"] / p["split"] * p["item"]["weight_bytes"] for p in placed),
             sum(len(p["launches"]) * p["group"]["median_us"] for p in placed))
    if score > best["score"]:
      tie, out = swaps()  # rows that could exchange their launches are withdrawn: no pairing is shown that the walk cannot stand behind
      gone = [{"item": it, "reason": why} for it, why in out.values()]
      best.update(score=score, placed=[p for p in placed if id(p["item"]) not in out], unplaced=list(unplaced) + gone,
                  ambiguous={"reason": "; ".join(tie)} if tie else None)

  def _avail(g, side):
    """The group's launches not yet taken that the gap's order allows this side to take."""
    out = []
    for x in g["launches"]:
      l, _, gap, pos = x
      if id(l) in taken:
        continue
      if gap is not None and pos is not None:
        gi, glen, prev_kind, next_kind, unit_end = gap
        if side == "output" and (pos >= in_min.get(gi, 1 << 30) or pos >= glen - min_in.get(next_kind, 0)):
          continue
        if side == "input" and (pos <= out_max.get(gi, -1) or pos < (0 if unit_end else min_out.get(prev_kind, 0))):
          continue
      out.append(x)
    return out

  def ordered(item, g):
    stage = _output_stage(item)
    if g["side"] != "output" or not stage:
      return True
    i = OUTPUT_ORDER.index(stage)  # order holds within a layer: only groups that share layers are compared
    return (all(u["slot"] < g["slot"] for e in OUTPUT_ORDER[:i] for u in used[(g["kind"], e)] if u["layers"] & g["layers"])
            and all(u["slot"] > g["slot"] for e in OUTPUT_ORDER[i + 1:] for u in used[(g["kind"], e)] if u["layers"] & g["layers"]))

  def choices(item):
    """(k, r, templates, [(group, launches)]) in preference order: slot groups the row may take, of templates serving
    its format (bound ones first), on the output side in the data flow's order, elsewhere slowest first; a row may
    span templates (a kernel variant per chunk width) at one launch ratio."""
    key = lambda g: (g["kind"] if g["kind"] != "tail" else None, g["template"])  # noqa: E731
    cand = []
    for g in groups:
      if fmt.get(key(g), item["quant"]) != item["quant"] or not (g.get("weight", True) or item["stages"] == {"head"}):
        continue
      if not (g["side"] in item["sides"] and (item["kind"] is None or g["kind"] == item["kind"])):
        continue
      got = _avail(g, g["side"])
      if got:
        cand.append((g, got))
    cand.sort(key=lambda c: (fmt.get(key(c[0])) != item["quant"], c[0]["kind"] == "tail",
                             c[0]["slot"] if c[0]["side"] == "output" else -c[0]["slot"] if c[0]["side"] == "input" else 0, -c[0]["median_us"]))
    classes = {ratios[key(g)] for g, _ in cand if key(g) in ratios} or {(1, 1)}
    if any(key(g) not in ratios for g, _ in cand):
      classes |= {(k, 1) for k in range(1, MAX_FUSED + 1)}
    sides = [STAGE_SIDE.get(st) for st in item["stages"]]  # a row of classes on both sides: one launch per class per layer
    split_in = (item["need"] * sides.count("input") // len(sides)
                if "input" in sides and "output" in sides and item["sides"] >= {"input", "output"} else None)
    out = []
    for k, r in sorted(classes):
      if item["need"] * r % k:
        continue
      fit = [(g, ls) for g, ls in cand if ratios.get(key(g), (k, r)) == (k, r)
             and _fits(item, g, k, r, bandwidth_gbs, peak_of(g["template"])) and ordered(item, g)]
      opts = [{"n": len(ls), "pick": (g, ls)} for g, ls in fit]
      want, slack = item["need"] * r // k, item.get("slack", 0) * r // k
      if split_in is None:
        subsets = _sums(opts, want, slack)
      else:  # each side its share; a last layer's skip is on the output side
        ins = _sums([o for o in opts if o["pick"][0]["side"] == "input"], split_in * r // k)
        outs = _sums([o for o in opts if o["pick"][0]["side"] == "output"], want - split_in * r // k, slack)
        subsets = [x + y for x in ins for y in outs][:8]
      for subset in subsets:
        out.append((k, r, {key(x["pick"][0]) for x in subset}, [x["pick"] for x in subset]))
    return out

  def swaps():
    """Rows that could exchange their slots (equal launches, each fitting the other's and its format), and a slot group no row took
    that is the same as one a row took: the walk cannot tell them apart, and says so. Returns (words, {item id: (item,
    why)} for the rows withdrawn)."""
    by_item: dict[int, list] = collections.defaultdict(list)
    for p in placed:
      by_item[id(p["item"])].append(p)
    found, out = [], {}
    seen = list(by_item.values())
    for i, a in enumerate(seen):
      for b in seen[i + 1:]:
        ia, ib = a[0]["item"], b[0]["item"]
        if (ia["kind"], ia["sides"], ia["need"]) != (ib["kind"], ib["sides"], ib["need"]) or (ia["stages"] != ib["stages"] and ia["sides"] != {"flat"}):
          continue
        if any(bound.get((p["group"]["kind"], p["group"]["template"]), q) != q for q, ps in ((ia["quant"], b), (ib["quant"], a)) for p in ps):
          continue  # a template the counts bound to one format cannot serve the other
        identity = lambda it: sorted((r["role"], r["quant"], r.get("n"), r.get("k"), r["count"]) for r in it["parts"])
        if identity(ia) == identity(ib):
          continue  # exchanging identical logical rows changes no attribution
        if (all(_fits(ia, p["group"], p["fused"], p["split"], bandwidth_gbs, peak_of(p["group"]["template"])) for p in b)
            and all(_fits(ib, p["group"], p["fused"], p["split"], bandwidth_gbs, peak_of(p["group"]["template"])) for p in a)):
          found.append(f"{ia['role']} {ia['quant']} and {ib['role']} {ib['quant']} could exchange their launches")
          out[id(ia)] = (ia, found[-1])
          out[id(ib)] = (ib, found[-1])
    free = [g for g in groups if len(_avail(g, g["side"])) == g["n"]]
    key = lambda g: (g["kind"] if g["kind"] != "tail" else None, g["template"])  # noqa: E731
    for p in placed:
      g, q = p["group"], p["item"]["quant"]
      # the walk prefers a template already serving the row's format (bound by the counts, or by another row): a
      # free group is a twin only when the walk had no such reason to prefer the one taken, and the row fits it
      serves = key(g) in bound or any(o is not p and key(o["group"]) == key(g) for o in placed)
      twin = next((f for f in free if f["kind"] == g["kind"] and f["side"] == g["side"] and f["n"] == len(p["launches"])
                   and ordered(p["item"], f)
                   and (f["template"] == g["template"] or (fmt.get(key(f)) == q) == serves)
                   and _fits(p["item"], f, p["fused"], p["split"], bandwidth_gbs, peak_of(f["template"]))), None)
      if twin is not None:
        found.append(f"{p['item']['role']} {p['item']['quant']} could have taken {twin['slot']} for {g['slot']}: equal eligible launches without distinguishing semantic evidence")
        out[id(p["item"])] = (p["item"], found[-1])
        break
    return found, out

  def walk(i, unplaced, strict):
    state["nodes"] += 1
    if state["nodes"] > MAX_NODES:
      raise _TooMany
    if i == len(items):
      record(unplaced)
      return not unplaced
    item = items[i]
    stage = _output_stage(item)
    options = choices(item)
    if strict and not options:
      return False
    for k, r, ts, picks in options:
      saved_out, saved_in = dict(out_max), dict(in_min)
      for g, ls in picks:
        for l, _, gap, pos in ls:
          taken.add(id(l))
          if gap is not None and pos is not None:
            if g["side"] == "output":
              out_max[gap[0]] = max(out_max.get(gap[0], -1), pos)
            elif g["side"] == "input":
              in_min[gap[0]] = min(in_min.get(gap[0], 1 << 30), pos)
        placed.append({"item": item, "group": g, "launches": ls, "fused": k, "split": r})
        if g["side"] == "output" and stage:
          used[(g["kind"], stage)].append(g)
      new = [t for t in ts if t not in fmt]
      for t in ts:
        fmt.setdefault(t, item["quant"])
      done = walk(i + 1, unplaced, strict)
      for t in new:
        del fmt[t]
      for g, ls in picks:
        for l, _, _, _ in ls:
          taken.discard(id(l))
        placed.pop()
        if g["side"] == "output" and stage:
          used[(g["kind"], stage)].pop()
      out_max.clear(); out_max.update(saved_out)
      in_min.clear(); in_min.update(saved_in)
      if done:
        return True
    return False if strict else walk(i + 1, unplaced + [{"item": item, "reason": _why(item, bandwidth_gbs)}], strict)

  try:
    if not walk(0, [], True):
      walk(0, [], False)
  except _TooMany:
    best["ambiguous"] = {"reason": f"more than {MAX_NODES} ways to walk {len(items)} rows over {len(groups)} slot groups"}
  best["nodes"] = state["nodes"]
  return best


def _why(item:dict[str, Any], bandwidth_gbs:float | None) -> str:
  where = (f"the {'/'.join(sorted(item['sides']))} side of {item['kind']} layers" if item["kind"]
           else "the unit's tail" if "tail" in item["sides"] else "the window")
  us = item["weight_bytes"] / (bandwidth_gbs * 1e9) * 1e6 if bandwidth_gbs else None
  return (f"no slot on {where} fits {item['need']} launches of {item['role']} {item['quant']}"
          + (f" (its weights need {us:.1f} us at the bandwidth)" if us else ""))


def _output_stage(item:dict[str, Any]) -> str | None:
  got = [s for s in item["stages"] if s in OUTPUT_ORDER]
  return min(got, key=OUTPUT_ORDER.index) if got else None


def _sums(gs:list[dict[str, Any]], want:int, slack:int = 0) -> list[list[dict[str, Any]]]:
  """Subsets of the groups whose launches sum to want, or to less by at most slack (a unit's last layer may skip
  trailing launches: its rows are not needed), in the groups' order of preference: each group taken before it is
  skipped, so the first subset holds the most preferred groups. A few at most, within MAX_SUM_STEPS."""
  out: list[list[dict[str, Any]]] = []
  rest = [0] * (len(gs) + 1)  # launches in the groups from i on: a branch that cannot reach want - slack stops
  for i in range(len(gs) - 1, -1, -1):
    rest[i] = rest[i + 1] + gs[i]["n"]
  steps = [0]

  def rec(i, left, acc):
    steps[0] += 1
    if len(out) >= 8 or steps[0] > MAX_SUM_STEPS or left - slack > rest[i]:
      return
    if left == 0 or i == len(gs):
      if left <= slack and acc:
        out.append(list(acc))
      return
    if 0 < gs[i]["n"] <= left:
      rec(i + 1, left - gs[i]["n"], acc + [gs[i]])
    rec(i + 1, left, acc)
  rec(0, want, [])
  return out


class _TooMany(Exception):
  pass


def pair(launches:list[dict[str, Any]], rows:list[dict[str, Any]], *, kinds:dict[str, int], widths:list[int],
         outputs:int | None = None, candidate:Callable[[str], bool] = projection_candidate, bandwidth_gbs:float | None = None,
         peak_of:Callable[[str], float | None] = lambda t: None,
         flat:list[dict[str, Any]] | None = None, prune:dict[str, Any] | None = None) -> dict[str, Any]:
  """The whole walk: structure, then assign; the flat way when the window has no structure (said in "structure").
  widths: each unit's M (a chunk's tokens; 1 per decode token). flat: kernel groups for a window with no launches.
  prune: the engine adapter's {stages, outputs: output tokens per unit, kernels} when it prunes the last layer (_pruned)."""
  units = len(widths)
  try:
    if flat is not None:
      raise NoStructure("kernel groups without launches")
    st = structure(launches, kinds=kinds, units=units, candidate=candidate)
    cut, small = _pruned(st, prune) if prune else (None, [])
    groups, words = _groups(st, widths) + small, {k: dict(v) for k, v in st["sides"].items()}
    got = assign(groups, rows, units=units, outputs=outputs, kinds=kinds, bandwidth_gbs=bandwidth_gbs, peak_of=peak_of, prune=cut)
    got["structure"] = {"layers": len(st["layers"]), "units": units, "sides": words, "flat": None,
                        **({"pruned": {"stages": list(cut["stages"]), "offered": sum(g["n"] for g in small)}} if cut else {})}
  except NoStructure as exc:
    m = sum(widths) / max(units, 1)
    groups = flat_groups(flat, m) if flat is not None else flat_groups(_by_key(launches), m)
    for g in groups:  # a layer's row takes weight launches only; the head, on no side, may take any kernel
      g["weight"] = candidate(g["template"])
    got = assign(groups, rows, units=units, outputs=outputs, kinds={}, bandwidth_gbs=bandwidth_gbs, peak_of=peak_of)
    got["structure"] = {"layers": 0, "units": units, "sides": {}, "flat": str(exc)}
  return got


def _pruned(st:dict[str, Any], prune:dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
  """The pruned last layer's small launches: in the tail of each unit with output tokens (every launch after its last
  layer's block), the launches of the kernels the adapter counted (prune's kernels) that no other part of the window
  launches. They become slot groups of side
  PRUNED, one per position, at the unit's output tokens; the tail keeps them too, so the head still pairs there (a
  launch is taken once, and the bandwidth and peak checks keep a row off a launch that cannot be its). None when there
  are none."""
  seq, outs = st["seq"], prune.get("outputs") or []
  mine = {j for u, t in enumerate(st["tail"]) if u < len(outs) and outs[u] for j in t}
  others = {template(l["key"]) for j, l in enumerate(seq) if j not in mine}
  kernels = set(prune.get("kernels") or {template(seq[j]["key"]) for j in mine})
  out: dict[int, dict[str, Any]] = {}
  for u, width in enumerate(outs):
    small = [j for j in (st["tail"][u] if width and u < len(st["tail"]) else []) if template(seq[j]["key"]) not in others
             and template(seq[j]["key"]) in kernels]
    for p, j in enumerate(small):
      g = out.setdefault(p, {"kind": PRUNED, "side": PRUNED, "slot": p, "template": template(seq[j]["key"]), "launches": []})
      if g["template"] == template(seq[j]["key"]):
        g["launches"].append((seq[j], width, None, None))
  if not out:
    return None, []
  per = len(st["layers"]) // max(st["units"], 1)
  cut = {"stages": tuple(prune["stages"]), "kind": st["layers"][per - 1]["kind"], "units": sum(1 for w in outs if w)}
  return cut, [_finish(g) for g in out.values()]


def _by_key(launches:list[dict[str, Any]]) -> list[dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for l in launches:
    g = out.setdefault(str(l["key"]), {"key": str(l["key"]), "calls": 0, "wall_us": 0.0, "launches": []})
    g["calls"] += 1
    g["wall_us"] += float(l["wall_us"])
    g["launches"].append(l)
  return list(out.values())
