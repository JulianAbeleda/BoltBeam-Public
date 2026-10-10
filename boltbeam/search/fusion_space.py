"""Fusion candidates: one kernel launch that replaces the kernels of several adjacent decode-graph nodes.

A fusion family comes from the provider's describe (`decode_fusions`): the nodes it covers, the operands it reads,
its coupled schedule rows. Nothing it says is taken as fact:

  adjacency      BoltBeam's own decode graph (data/decode_graph.json) says whether the covered nodes are connected,
                 which one is the output, and which values they read from outside. The provider's operand list must
                 be exactly that, and each covered node's op must be the graph's.
  the model      the GGUF's own tensors (blk.N.<role>) say which layers hold every covered weight in the family's
                 quant and at one shape. A tie-out row (a coarse role and quant) is counted only when the fusion
                 replaces all of it: every fine role in it and every one of its tensors.
  the shape      the emitter's own validate (asked through describe's `shapes`) accepts or refuses each row.
  correctness    BoltBeam computes the covered nodes one after another itself (this module's `reference`) on the
                 model's own bytes: a fused kernel must match that, and so must each node's kernel alone.
  time           the fused kernel and each node's kernel alone are timed by BoltBeam's kernel timer, less the floor.

BubbleBeam's part is `propose`: the fusion instances whose nodes are adjacent and whose emitter accepts their shape.
FutureSight then applies the chip's legality rules to the population, as for any role. The judge is `verdict`: the
fused kernel per token against the in-model time of the rows it replaces (the tie-out's "where the token goes"
rows), with BoltBeam's unfused time alone beside it.

Nothing here names a backend, a chip or a model shape.
"""
from __future__ import annotations

import functools, json, math, pathlib, re, struct
from collections.abc import Mapping
from typing import Any

GRAPH_TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "decode_graph.json"
# the ops BoltBeam computes itself (reference): a node with any other op cannot be inside a fusion BoltBeam checks
REFERENCE_OPS = ("gemv", "silu_mul", "add", "cast_fp16")
LAYER = re.compile(r"^blk\.(\d+)\.")  # GGUF's per-layer tensor names (blk.N.<role>.weight)
DTYPES = {"float": ("f", 4), "float32": ("f", 4), "half": ("e", 2), "float16": ("e", 2)}
NOT_IN_MODEL = ("has no kernel of its own in the model's capture: counted as 0 ms, which favours the model's "
                "side of the comparison")


@functools.lru_cache(maxsize=1)
def graph() -> dict[str, Any]:
  return json.loads(GRAPH_TABLE.read_text())


def fusion_name(covers:list[str] | tuple[str, ...]) -> str:
  return "+".join(covers)


# --- adjacency: BoltBeam's own graph ---------------------------------------------------------------------------

def adjacency(covers:list[str] | tuple[str, ...]) -> dict[str, Any]:
  """Are these nodes one connected piece of the decode graph with one output? Returns {"why"} on a refusal, else
  {"sink", "inputs", "weights", "order"}: the output node, the values read from outside (in first-read order), the
  weight nodes (in graph order) and the covered nodes in graph order."""
  nodes = graph()["nodes"]
  covers = list(covers)
  if len(covers) < 2 or len(set(covers)) != len(covers):
    return {"why": "a fusion covers two or more distinct nodes"}
  unknown = [c for c in covers if c not in nodes]
  if unknown:
    return {"why": f"not in the decode graph: {', '.join(unknown)}"}
  unchecked = [f"{c} ({nodes[c]['op']})" for c in covers if nodes[c]["op"] not in REFERENCE_OPS]
  if unchecked:
    return {"why": f"BoltBeam has no reference for {', '.join(unchecked)}"}
  inside = set(covers)
  # connected: an edge where one node reads another, or two weight nodes read the same value
  edges = {c: set() for c in covers}
  for a in covers:
    for b in covers:
      if a != b and (b in nodes[a]["reads"] or (nodes[a]["op"] == nodes[b]["op"] == "gemv" and
                                                set(nodes[a]["reads"]) & set(nodes[b]["reads"]))):
        edges[a].add(b); edges[b].add(a)
  seen, todo = set(), [covers[0]]
  while todo:
    n = todo.pop()
    if n not in seen:
      seen.add(n); todo += list(edges[n])
  if seen != inside:
    return {"why": f"not adjacent in the decode graph: {', '.join(sorted(inside - seen))} apart from {', '.join(sorted(seen))}"}
  consumers = {c: [n for n, row in nodes.items() if c in row["reads"]] for c in covers}
  leaving = [c for c in covers if not consumers[c] or any(n not in inside for n in consumers[c])]
  if len(leaving) != 1:
    return {"why": f"{len(leaving)} of the covered nodes are read outside the fusion ({', '.join(leaving)}); one launch writes one output"}
  order = [n for n in nodes if n in inside]
  inputs: list[str] = []
  for c in order:
    for v in nodes[c]["reads"]:
      if v not in inside and v not in inputs:
        inputs.append(v)
  return {"sink": leaving[0], "inputs": inputs, "weights": [c for c in order if nodes[c]["op"] == "gemv"], "order": order}


# --- the model's own tensors -------------------------------------------------------------------------------------

def layers(tensors:list[tuple[str, tuple[int, ...], int, int]]) -> dict[int, dict[str, dict[str, Any]]]:
  """The GGUF's weight matrices by layer and fine role: {layer: {role: {name, quant, rows, cols, group}}}."""
  from boltbeam.profile.gguf import GGML_TYPE_NAMES
  from boltbeam.profile.roles import classify_tensor_role
  from boltbeam.vocab import role_group_of
  out: dict[int, dict[str, dict[str, Any]]] = {}
  for name, dims, typ, _off in tensors:
    m = LAYER.match(name)
    if not m or len(dims) != 2:
      continue
    role = classify_tensor_role(name, tuple(dims)).value
    out.setdefault(int(m.group(1)), {})[role] = {"name": name, "quant": GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"),
                                                 "rows": int(dims[1]), "cols": int(dims[0]), "group": role_group_of(role)}
  return out


def _verdicts(describe_shapes:list[Mapping[str, Any]], family:str, shape:tuple[int, int]) -> list[dict[str, Any]] | None:
  return next((list(v["rows"]) for v in describe_shapes or [] if v.get("family") == family
               and (v.get("shape") or {}).get("n") == shape[0] and (v.get("shape") or {}).get("k") == shape[1]), None)


def propose(families:list[Mapping[str, Any]], by_layer:Mapping[int, Mapping[str, Mapping[str, Any]]],
            rows_in_model:set[tuple[str, str]], shape_verdicts:list[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
  """BubbleBeam's fusion proposal: per provider family, the instance (its layers, anchor tensor and shape) whose
  nodes are adjacent in BoltBeam's graph and whose tie-out rows it replaces whole, with the emitter's validate verdict
  per row. A refused family keeps its reason ("why"); an instance with no accepted row is refused too."""
  out = []
  graph_nodes = graph()["nodes"]
  for fam in families:
    covers = [str(c) for c in fam.get("covers") or []]
    base = {"family": fam.get("family"), "covers": covers, "name": fusion_name(covers), "quant": fam.get("quant"),
            "why": None, "rows": [], "rejected_rows": []}
    adj = adjacency(covers)
    if adj.get("why"):
      out.append(base | {"why": adj["why"]}); continue
    base |= {"sink": adj["sink"], "inputs": adj["inputs"], "weights": adj["weights"]}
    expected = sorted([f"weight:{w}" for w in adj["weights"]] + [f"value:{v}" for v in adj["inputs"]])
    if sorted(fam.get("operands") or []) != expected:
      out.append(base | {"why": f"the provider's operands {sorted(fam.get('operands') or [])} are not what the decode graph says "
                                f"these nodes read: {expected}"}); continue
    said = {str(n.get("node")): str(n.get("op")) for n in fam.get("nodes") or [] if isinstance(n, Mapping)}
    wrong = [f"{c} ({said.get(c)} against {graph_nodes[c]['op']})" for c in covers if said.get(c) != graph_nodes[c]["op"]]
    if wrong:
      out.append(base | {"why": f"the provider's ops disagree with the decode graph: {', '.join(wrong)}"}); continue
    quant, weights = str(fam.get("quant")), adj["weights"]
    held = sorted(l for l, roles in by_layer.items() if all(w in roles for w in weights))
    mine = [l for l in held if all(by_layer[l][w]["quant"] == quant for w in weights)]
    if not mine:
      why = (f"no layer holds every weight of {base['name']} in {quant}" if held else f"no layer holds {', '.join(weights)}")
      out.append(base | {"why": why}); continue
    first = by_layer[mine[0]][weights[0]]
    shape = (first["rows"], first["cols"])
    odd = sorted({f"{w} {by_layer[l][w]['rows']}x{by_layer[l][w]['cols']}" for l in mine for w in weights
                  if (by_layer[l][w]["rows"], by_layer[l][w]["cols"]) != shape})
    if odd:
      out.append(base | {"why": f"the weights are not all {shape[0]}x{shape[1]}: {', '.join(odd)}"}); continue
    # the tie-out rows it replaces: a row counts only when the fusion replaces every fine role and tensor in it
    rows, why = [], None
    for w in weights:
      group = by_layer[mine[0]][w]["group"]
      row = (group, quant, shape[0], shape[1])
      # the tie-out keys a row by role, quant and shape when it knows the shape, else by role and quant
      key = row if row in rows_in_model else row[:2]
      if key in rows: continue
      same = (lambda t: (t["group"], t["quant"], t["rows"], t["cols"]) == row) if key == row else (lambda t: (t["group"], t["quant"]) == row[:2])
      members = {(l, r) for l, roles in by_layer.items() for r, t in roles.items() if same(t)}
      ours = {(l, w2) for l in mine for w2 in weights if same(by_layer[l][w2])}
      if members != ours:
        others = sorted({r for _l, r in members - ours})
        why = (f"the tie-out row {group} {quant} also holds {', '.join(others)}, which this fusion does not replace"
               if others and not set(others) <= set(weights) else
               f"the tie-out row {group} {quant} has {len(members)} tensors; this fusion replaces {len(ours)} of them")
        break
      if key not in rows_in_model:
        why = f"the tie-out has no in-model row for {group} {quant}"
        break
      rows.append(key)
    instance = {"layers": mine, "calls_per_token": len(mine), "anchor": first["name"],
                "tensors": {w: by_layer[mine[0]][w]["name"] for w in weights}, "shape": list(shape),
                "tie_out_rows": [list(r) for r in rows]}
    if why:
      out.append(base | {"instance": instance, "why": why}); continue
    verdicts = _verdicts(shape_verdicts or [], str(fam.get("family")), shape)
    accepted, refused = [], []
    for row in fam.get("coupled_rows") or []:
      said_row = next((v for v in verdicts or [] if v.get("row") == row), None)
      if verdicts is None or said_row is None:
        refused.append({"row": dict(row), "reason": "emitter_validate: the provider gave no verdict for this shape"})
      elif said_row.get("refused"):
        refused.append({"row": dict(row), "reason": f"emitter_validate: {said_row['refused']}"})
      else:
        accepted.append(dict(row))
    out.append(base | {"instance": instance, "rows": accepted, "rejected_rows": refused,
                       "why": None if accepted else "the emitter's validate refused every row at this shape"})
  return out


def workload(proposal:Mapping[str, Any], *, model_sha:str, target:Mapping[str, Any], tolerance:Mapping[str, float]) -> dict[str, Any]:
  """The exact semantic workload of one fusion: the anchor tensor's matmul identity, the covered nodes as its role, and
  the fusion (family, nodes, every weight's tensor) the provider binds."""
  inst = proposal["instance"]
  rows, k = (int(x) for x in inst["shape"])
  anchor = str(inst["anchor"])
  module = anchor[:-len(".weight")] if anchor.endswith(".weight") else anchor
  quant = str(proposal["quant"])
  identity = {"phase": "decode", "tensor_name": anchor, "module_path": module, "role": proposal["name"], "logical_m": 1,
              "logical_n": rows, "logical_k": k, "source_quant_storage": quant, "source_layout": "transposed_row_major",
              "module_representation": "gguf_packed", "input_dtype": "fp16", "output_dtype": "fp16", "accumulator_dtype": "fp32"}
  return {"schema": "tinygrad.semantic_provider_workload.v1", "model_hash": model_sha, "target": dict(target),
          "semantic_identity": identity, "operation": "matmul", "shape": {"m": 1, "n": rows, "k": k},
          "operands": {"a": {"dtype": "fp16"}, "b": {"quantization": quant, "layout": "transposed_row_major"}, "c": {"dtype": "fp16"}},
          "tolerance": dict(tolerance), "fixture_shape_substitution": "forbidden",
          "fusion": {"family": proposal["family"], "covers": list(proposal["covers"]), "tensors": dict(inst["tensors"])}}


# --- BoltBeam's own reference and binding ------------------------------------------------------------------------

def _round(values:list[float], dtype:str) -> list[float]:
  fmt = DTYPES[dtype][0]
  return list(struct.unpack(f"<{len(values)}{fmt}", struct.pack(f"<{len(values)}{fmt}", *values)))


def _silu(a:float) -> float:
  return a / (1.0 + math.exp(-a)) if a > -80 else 0.0


class Reference:
  """The covered nodes computed one after another by BoltBeam alone: a weight node by pure-Python dequantize and dot
  (metal_native.reference_row) on the model's bytes, an elementwise node by its op. Values from outside are the ones
  BoltBeam bound, rounded to the dtype the kernel reads them in."""

  def __init__(self, nodes:list[str], quant:str, rows:int, cols:int, weights:Mapping[str, bytes], values:Mapping[str, list[float]]):
    self.nodes, self.quant, self.rows, self.cols = list(nodes), quant, rows, cols
    self.weights, self.values = dict(weights), dict(values)
    self._full: dict[str, list[float]] = {}
    self._lengths = value_lengths(self.nodes, rows, cols)

  def full(self, value:str) -> list[float]:
    """A whole value vector: one bound from outside, or a covered node computed for every index."""
    if value in self.values:
      return self.values[value]
    if value not in self._full:
      self._full[value] = [self.at(value, i) for i in range(self._lengths.get(value, self.rows))]
    return self._full[value]

  def at(self, node:str, i:int) -> float:
    """The value of `node` at output index i."""
    if node in self.values and node not in self.nodes:
      return self.values[node][i]
    from boltbeam.collectors import metal_native as native
    row = graph()["nodes"][node]
    op = row["op"]
    if op == "gemv":
      x = self.full(row["reads"][0])
      return native.reference_row(self.weights[node], self.quant, i, len(x), x)
    a = [self.at(v, i) for v in row["reads"]]
    if op == "silu_mul":
      return _silu(a[0]) * a[1]
    if op == "add":
      return a[0] + a[1]
    if op == "cast_fp16":
      return _round([a[0]], "half")[0]
    raise ValueError(f"BoltBeam has no reference for {node} ({op})")


def bind(record:Mapping[str, Any], *, weights:Mapping[str, bytes], value_len:Mapping[str, int], rows:int,
         seed:str) -> tuple[list[Any], dict[str, list[float]], str]:
  """The kernel's arguments in its parameter order, bound to BoltBeam's own data: a "weight:<node>" buffer to that
  node's GGUF bytes, a "value:<name>" buffer to a deterministic vector rounded to the buffer's dtype, "out" to a zeroed
  buffer of `rows` values. Every buffer must be one of these at the size BoltBeam expects, or the record is refused.
  Returns (args, the values bound, the output's struct format)."""
  from boltbeam.collectors import boltbeam_gemv
  args: list[Any] = []
  bound: dict[str, list[float]] = {}
  out_format = None
  for buf in record.get("buffers") or []:
    label, nbytes, dtype = str(buf.get("role")), int(buf.get("nbytes") or 0), str(buf.get("dtype"))
    kind, _, name = label.partition(":")
    if label == "out":
      if dtype not in DTYPES or nbytes != rows * DTYPES[dtype][1]:
        raise ValueError(f"the provider's output buffer ({dtype}, {nbytes} bytes) is not {rows} values")
      if out_format is not None:
        raise ValueError("the provider's kernel must write exactly one output buffer")
      out_format = DTYPES[dtype][0]
      args.append(nbytes)
    elif kind == "weight" and name in weights:
      if nbytes != len(weights[name]):
        raise ValueError(f"the provider's {label} buffer is {nbytes} bytes; the model's tensor is {len(weights[name])}")
      args.append(weights[name])
    elif kind == "value" and name in value_len:
      if dtype not in DTYPES or nbytes != value_len[name] * DTYPES[dtype][1]:
        raise ValueError(f"the provider's {label} buffer ({dtype}, {nbytes} bytes) is not {value_len[name]} values")
      vals = _round(boltbeam_gemv.vector(f"{seed}:{name}", value_len[name]), dtype)
      bound[name] = vals
      args.append(struct.pack(f"<{len(vals)}{DTYPES[dtype][0]}", *vals))
    else:
      raise ValueError(f"the provider listed a buffer BoltBeam does not expect: {label!r}")
  if out_format is None:
    raise ValueError("the provider's kernel writes no output buffer")
  return args, bound, out_format


def value_lengths(nodes:list[str], rows:int, cols:int) -> dict[str, int]:
  """How long each value the nodes read is: a weight node reads `cols` values; the output node writes one per row; an
  elementwise node reads values as long as its own output, which is as long as what its consumer inside reads."""
  g = graph()["nodes"]
  order = [n for n in g if n in nodes]
  length = {order[-1] if len(order) == 1 else adjacency(order)["sink"]: rows}
  out: dict[str, int] = {}
  for n in reversed(order):
    for v in g[n]["reads"]:
      out[v] = cols if g[n]["op"] == "gemv" else length.get(n, rows)
      length.setdefault(v, out[v])
  return out


__all__ = ["GRAPH_TABLE", "NOT_IN_MODEL", "REFERENCE_OPS", "Reference", "adjacency", "bind", "fusion_name", "graph",
           "layers", "propose", "value_lengths", "workload"]
