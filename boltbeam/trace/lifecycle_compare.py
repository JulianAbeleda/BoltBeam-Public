"""Workload-lifecycle comparison: ours (tinygrad HCQ graph profile) vs vLLM (nsys sqlite), per workload.

Target state (Nemotron-H 4B BF16, RTX 5090): every kernel at or better than vLLM's on the same shape, so the only
remaining difference is lifecycle (launch count, graph replay, gaps, scheduling, fusion). This module puts a number
on both halves per workload:

  wall/step = kernel_sum/step + lifecycle/step          (lifecycle = wall - kernel_sum; negative => overlap)
  kernel_sum = sum(GEMM roles) + sum(non-GEMM categories)

so  wall gap (ours - vLLM) = sum(role deltas) + sum(category deltas) + lifecycle delta, exactly.

Provenance of the step segmentation:
- vLLM decode: `docs/nemotron-vllm-parity/bench/vllm-bench/analyze.py` (tinygrad-arkey tree): a step is the interval
  between the first kernels of consecutive launches of the largest CUDA graph (the FULL decode graph; one per step),
  restricted to the middle 60% of the captured GPU timeline. `classify_vllm_name` is analyze.py's `classify`.
- ours decode: `bench/spec/ours_prof.py` + `ours_an.py`: a measured window of chained sampler steps in the
  HCQ_GRAPH_PROFILE_JSON jsonl (one line per graph launch, `meta["lines"]` = window). A step is the interval between
  consecutive launches of the step's first graph (the graph whose first kernel is the step list's first kernel).
  ours_an.py divided by `nsteps` although ours_prof.py records nsteps+1 steps in the window; anchoring on graph
  launches here avoids that off-by-one.
- prefill (both sides): the whole capture window is one "step" (wall = GPU span first start .. last end).

Role attribution is positional: GEMM ops (a GEMM kernel plus its attached helpers: vLLM `splitKreduce`; ours hi/lo
split before and split-K/hi+lo sum after, names from the audit's ours.json) are matched in order against the model's
layer pattern (Nemotron-H `hybrid_override_pattern`: M -> ssm_in, ssm_out; * -> attention in/out; - -> ffn_up,
ffn_down; then output). Non-GEMM kernels take the category of the segment they sit in (between ssm_in and ssm_out ->
mamba; between the attention input projection(s) and attn_o -> attention; between ffn_up and ffn_down -> mlp_act;
between layers -> norm_residual; after output -> sampling_bookkeeping). Pure functions; no GPU, no tinygrad, no torch.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import sqlite3
import statistics
from typing import Any, Iterable

from boltbeam.target.tinygrad_root import resolve_tinygrad_root

SCHEMA = "boltbeam.lifecycle_comparison.v1"

# NVIDIA-Nemotron-3-Nano-4B config.json hybrid_override_pattern (42 layers: 21 M, 4 *, 17 -)
NEMOTRON_H_4B_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"

GEMM_ROLES = ("ssm_in", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_kv", "attn_qkv", "attn_o", "ffn_up", "ffn_down",
              "output")
CATEGORIES = ("mamba", "attention", "mlp_act", "norm_residual", "sampling_bookkeeping", "state_flush",
              "unassigned_gemm", "other")
# attention input-projection layouts: vLLM fuses q/k/v (QKVParallelLinear); ours runs q, k, v separately
ATTN_LAYOUTS = {"fused_qkv": ("attn_qkv",), "q_k_v": ("attn_q", "attn_k", "attn_v"), "q_kv": ("attn_q", "attn_kv")}
ATTN_IN_ROLES = frozenset(r for v in ATTN_LAYOUTS.values() for r in v)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_HASH = re.compile(r"_[0-9a-f]{64}$")
FULL_KERNEL_CANDIDATE_SCHEMA = "boltbeam.full_kernel_candidate.v1"


def strip_tinygrad_name(name:str) -> str:
  return _HASH.sub("", _ANSI.sub("", name))


def layer_roles(pattern:str, attn_layout:str) -> list[str]:
  """Expected GEMM role sequence of one forward pass, without the output projection."""
  attn = ATTN_LAYOUTS[attn_layout]
  out: list[str] = []
  for ch in pattern:
    if ch == "M": out += ["ssm_in", "ssm_out"]
    elif ch == "*": out += [*attn, "attn_o"]
    elif ch == "-": out += ["ffn_up", "ffn_down"]
    else: raise ValueError(f"unknown layer char {ch!r} in pattern")
  return out


# ---- vLLM kernel-name classifier: verbatim logic of vllm-bench/analyze.py `classify` ----
def classify_vllm_name(n:str) -> str:
  l = n.lower()
  if "selective_scan_update" in l or "selective_state_update" in l or "ssu" in l.split("_") or "replayssm" in l or "_state_update" in l: return "ssm_state_update"
  if "causal_conv1d" in l or "conv1d" in l: return "mamba_conv"
  if "chunk_scan" in l or "chunk_state" in l or "state_passing" in l or "bmm_chunk" in l or "chunk_cumsum" in l: return "mamba_ssd_prefill"
  if "layer_norm_fwd" in l or ("gated" in l and "norm" in l) or ("silu" in l and "rsqrt" in l): return "gated_rmsnorm(+silu gate)"
  if "mamba_align" in l or "postprocess_mamba" in l: return "mamba_align_bookkeeping"
  if "pow_relu" in l: return "relu2"
  if "kernel2<cutlass" in l or "wmma_tensorop" in l: return "GEMM"
  if "rms" in l or "norm" in l: return "norm/fused"
  if any(x in l for x in ["flash", "fmha", "attn", "attention", "paged", "decode_kernel", "batchdecode", "batchprefill"]): return "attention"
  if any(x in l for x in ["gemm", "gemv", "cutlass", "sm80_xmma", "sm90", "sm100", "sm120", "cublas", "matmul", "ampere", "xmma", "splitk", "nvjet"]): return "GEMM"
  if any(x in l for x in ["softmax", "sample", "mbtopk", "_ranks_kernel", "topk", "top_k", "argmax", "exponential", "gumbel", "logprob", "sort", "radix", "scatter_gather", "gather", "multinomial", "cumsum", "max_", "reduce"]): return "sampling/reduce"
  if "triton" in l: return "triton-other"
  return "other(elementwise/copy)"


def _k(name:str, start:float, end:float, **extra) -> dict[str, Any]:
  return {"name": name, "s": float(start), "e": float(end), **extra}


# ---------------------------------------------------------------- positional attribution
def attribute_step(seq:list[dict[str, Any]], expected:list[str], *, is_gemm, pre_helpers:frozenset[str] = frozenset(),
                   post_helpers:frozenset[str] = frozenset(), merge_same_name_runs:bool = False,
                   output_policy:str = "auto") -> tuple[list[str], dict[str, Any]]:
  """Label every kernel of an ordered step sequence with a GEMM role or a non-GEMM category.

  `expected` = layer_roles(...) for one forward. The output projection is optional per forward:
  output_policy "auto" recognises it as the op in the slot after the last layer GEMM whose (name, grid) signature
  differs from the forward's first op; "last_only" accepts it only as the final op of the sequence (prefill pieces of
  different sizes have different kernel names, so "auto" would misread a smaller piece's first op as an output).
  merge_same_name_runs: consecutive GEMM kernels with the same name are one op (row-chunked launches, e.g. ours
  prefill runs a 1024-token piece as 4 launches of the 512-row kernel); non-GEMM kernels between them are that op's
  helpers. Returns (labels, info) with info = {"gemm_ops", "forwards", "role_assignment_ok"}."""
  if output_policy not in ("auto", "last_only"): raise ValueError(f"unknown output_policy {output_policy!r}")
  n = len(seq)
  gemm = [bool(is_gemm(k)) for k in seq]
  op_of = [-1] * n
  ops: list[list[int]] = []
  last_g = -1
  for i in range(n):
    if not gemm[i]: continue
    if merge_same_name_runs and last_g >= 0 and seq[last_g]["name"] == seq[i]["name"]:
      oi = op_of[last_g]
      for j in range(last_g + 1, i):  # chunk helpers between two launches of the same op
        if op_of[j] < 0: op_of[j] = oi; ops[oi].append(j)
      op_of[i] = oi; ops[oi].append(i)
    else:
      op_of[i] = len(ops); ops.append([i])
    last_g = i
  if merge_same_name_runs:  # the chunk helpers' first instance runs just before the op's first chunk
    for oi, members in enumerate(ops):
      inner = {seq[j]["name"] for j in members if not gemm[j]}
      j = members[0] - 1
      while j >= 0 and op_of[j] < 0 and not gemm[j] and seq[j]["name"] in inner:
        op_of[j] = oi; members.append(j); j -= 1
  for i in range(n):
    if op_of[i] >= 0: continue
    nm = seq[i]["name"]
    # post helpers may chain (split-K sum, then hi+lo sum); a name that is also a pre helper only attaches forward
    if nm in post_helpers and nm not in pre_helpers and i > 0 and op_of[i - 1] >= 0:
      op_of[i] = op_of[i - 1]; ops[op_of[i]].append(i)
    elif nm in pre_helpers and i + 1 < n and gemm[i + 1]:
      op_of[i] = op_of[i + 1]; ops[op_of[i]].append(i)
    elif nm in post_helpers and i > 0 and gemm[i - 1]:
      op_of[i] = op_of[i - 1]; ops[op_of[i]].append(i)
  # map ops -> roles, cycling over forwards with an optional output op per forward
  sig = lambda oi: (seq[ops[oi][0]]["name"], seq[ops[oi][0]].get("grid"))
  roles: list[str] = []
  slot, forwards, first_sig = 0, 0, None
  for oi in range(len(ops)):
    if slot == 0: first_sig = sig(oi); forwards += 1
    if slot < len(expected):
      roles.append(expected[slot]); slot += 1
    elif (oi == len(ops) - 1) if output_policy == "last_only" else (sig(oi) != first_sig):
      roles.append("output"); slot = 0
    else:  # output skipped this forward: this op opens the next forward
      roles.append(expected[0]); slot = 1; forwards += 1; first_sig = sig(oi)
  ok = len(ops) > 0 and (slot == 0 or slot == len(expected))
  if not ok: roles = ["unassigned_gemm"] * len(ops)
  labels: list[str] = []
  prev_role: str | None = None
  for i in range(n):
    if op_of[i] >= 0:
      prev_role = roles[op_of[i]]; labels.append(prev_role); continue
    if not ok: labels.append("other")
    elif prev_role == "ssm_in": labels.append("mamba")
    elif prev_role in ATTN_IN_ROLES: labels.append("attention")
    elif prev_role == "ffn_up": labels.append("mlp_act")
    elif prev_role == "output": labels.append("sampling_bookkeeping")
    else: labels.append("norm_residual")
  return labels, {"gemm_ops": len(ops), "forwards": forwards, "role_assignment_ok": ok,
                  "gemm_ops_per_forward_expected": len(expected) + 1}


# ---------------------------------------------------------------- vLLM (nsys sqlite)
def load_nsys_sqlite(path:str | pathlib.Path) -> tuple[list[dict[str, Any]], set[int]]:
  """Kernel rows (ordered by start) + the correlation ids of cudaGraphLaunch runtime calls."""
  db = sqlite3.connect(str(path))
  try:
    strs = dict(db.execute("select id, value from StringIds"))
    rows = db.execute("select start, end, shortName, demangledName, correlationId, gridX, gridY, gridZ "
                      "from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    graph = {c for (c,) in db.execute("select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
                                      "on r.nameId = s.id where s.value like 'cudaGraphLaunch%'")}
  finally:
    db.close()
  K = [_k(strs.get(sn, str(sn)), s, e, dem=strs.get(dn, str(dn))[:200], corr=c, grid=[gx, gy, gz])
       for s, e, sn, dn, c, gx, gy, gz in rows]
  return K, graph


def _vllm_is_gemm(k:dict[str, Any]) -> bool:
  return classify_vllm_name(k["name"] + " " + k.get("dem", "")) == "GEMM" and "splitkreduce" not in k["name"].lower()


VLLM_POST_HELPERS = frozenset({"splitKreduce_kernel"})


def vllm_decode_steps(K:list[dict[str, Any]], graph_corrs:set[int], window:tuple[float, float] = (0.2, 0.8)
                      ) -> dict[str, Any]:
  """analyze.py step segmentation: anchors = first kernel of each launch of the largest-per-step CUDA graph."""
  if not K: raise ValueError("no kernels in trace")
  t0, t1 = K[0]["s"], K[-1]["e"]
  ws, we = t0 + window[0] * (t1 - t0), t0 + window[1] * (t1 - t0)
  per_launch = collections.Counter(k["corr"] for k in K if k["corr"] in graph_corrs)
  sizes = collections.Counter(per_launch.values())
  if not sizes: raise ValueError("no CUDA-graph kernels: cannot segment decode steps")
  big = max(sizes, key=lambda s: s * sizes[s])
  first: dict[int, float] = {}
  for k in K:
    if k["corr"] in graph_corrs: first.setdefault(k["corr"], k["s"])
  anchors = sorted(first[c] for c, n in per_launch.items() if n == big and ws <= first[c] < we)
  if len(anchors) < 2: raise ValueError("fewer than two steady decode steps in window")
  steps: list[list[dict[str, Any]]] = []
  j = 0
  for a, b in zip(anchors, anchors[1:]):
    while j < len(K) and K[j]["s"] < a: j += 1
    step = []
    while j < len(K) and K[j]["s"] < b:
      step.append(dict(K[j], in_graph=K[j]["corr"] in graph_corrs)); j += 1
    steps.append(step)
  return {"steps": steps, "periods": [(b - a) for a, b in zip(anchors, anchors[1:])], "unit": "ns",
          "graph_kernels_per_step": big, "graph_launch_corrs": graph_corrs}


def _summarize(steps:list[list[dict[str, Any]]], periods:list[float], labels:list[list[str]], *, unit_to_ms:float,
               graph_launches:float, extra:dict[str, Any] | None = None) -> dict[str, Any]:
  ns = len(steps)
  roles: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
  cats: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
  ksum, nk, busy = 0.0, 0, 0.0
  for st, lab in zip(steps, labels):
    for k, l in zip(st, lab):
      d = k["e"] - k["s"]; ksum += d; nk += 1
      a = roles[l] if l in GEMM_ROLES else cats[l]
      a[0] += 1; a[1] += d
    iv = sorted((k["s"], k["e"]) for k in st)
    if iv:
      cs, ce = iv[0]
      for s, e in iv[1:]:
        if s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
      busy += ce - cs
  wall = sum(periods) / ns * unit_to_ms
  ksum_ms = ksum / ns * unit_to_ms
  out = {
    "status": "ok", "steps": ns,
    "wall_ms_per_step": round(wall, 4),
    "wall_ms_per_step_median": round(statistics.median(periods) * unit_to_ms, 4),
    "kernel_sum_ms_per_step": round(ksum_ms, 4),
    "busy_union_ms_per_step": round(busy / ns * unit_to_ms, 4),
    "lifecycle_ms_per_step": round(wall - ksum_ms, 4),
    "kernel_launches_per_step": round(nk / ns, 2),
    "graph_launches_per_step": round(graph_launches, 3),
    "roles": {r: {"ms": round(v[1] / ns * unit_to_ms, 4), "kernels": round(v[0] / ns, 2)} for r, v in sorted(roles.items())},
    "categories": {c: {"ms": round(v[1] / ns * unit_to_ms, 4), "kernels": round(v[0] / ns, 2)} for c, v in sorted(cats.items())},
  }
  if extra: out.update(extra)
  return out


def vllm_side(K:list[dict[str, Any]], graph_corrs:set[int], *, kind:str, pattern:str = NEMOTRON_H_4B_PATTERN,
              window:tuple[float, float] = (0.2, 0.8),
              source:dict[str, Any] | None = None) -> dict[str, Any]:
  expected = layer_roles(pattern, "fused_qkv")
  if kind == "decode":
    seg = vllm_decode_steps(K, graph_corrs, window)
    steps, periods = seg["steps"], seg["periods"]
    glaunch = sum(len({k["corr"] for k in st if k["in_graph"]}) for st in steps) / len(steps)
  else:
    steps, periods = [K], [K[-1]["e"] - K[0]["s"]]
    glaunch = float(len({k["corr"] for k in K if k["corr"] in graph_corrs}))
  labels, infos = [], []
  for st in steps:
    lab, info = attribute_step(st, expected, is_gemm=_vllm_is_gemm, post_helpers=VLLM_POST_HELPERS)
    labels.append(lab); infos.append(info)
  sub: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
  for st, lab in zip(steps, labels):
    for k, l in zip(st, lab):
      if l not in GEMM_ROLES:
        a = sub[f"{l}/{classify_vllm_name(k['name'] + ' ' + k.get('dem', ''))}"]; a[0] += 1; a[1] += k["e"] - k["s"]
  ns = len(steps)
  extra = {"side": "vllm", "attention_layout": "fused_qkv", "attribution": _info_summary(infos),
           "category_detail": {k: {"ms": round(v[1] / ns / 1e6, 4), "kernels": round(v[0] / ns, 2)}
                               for k, v in sorted(sub.items(), key=lambda x: -x[1][1])},
           "in_graph_kernels_per_step": round(sum(1 for st in steps for k in st if k.get("in_graph")) / ns, 2),
           "source": source or {}}
  return _summarize(steps, periods, labels, unit_to_ms=1e-6, graph_launches=glaunch, extra=extra)


def _info_summary(infos:list[dict[str, Any]]) -> dict[str, Any]:
  ok = sum(1 for i in infos if i["role_assignment_ok"])
  return {"steps_role_assignment_ok": ok, "steps": len(infos),
          "gemm_ops_per_step": sorted({i["gemm_ops"] for i in infos}),
          "forwards_per_step": sorted({i["forwards"] for i in infos}),
          "gemm_ops_per_forward_expected": infos[0]["gemm_ops_per_forward_expected"] if infos else None}


# ---------------------------------------------------------------- ours (HCQ graph profile jsonl)
def ours_gemm_names(ours_gemm_rows:Iterable[dict[str, Any]]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
  """(gemm, pre_helper, post_helper) kernel names from the audit's ours.json (bench/audit/ours_gemm.py output).
  The GEMM is the kernel flagged gemm, else the slowest kernel of the op; kernels before it are pre helpers
  (hi/lo split), after it post helpers (split-K / hi+lo sums)."""
  g, pre, post = set(), set(), set()
  for row in ours_gemm_rows:
    if row.get("mode", "prod") != "prod": continue
    ks = row.get("kernels", [])
    if not ks: continue
    idx = next((i for i, k in enumerate(ks) if k.get("gemm")), max(range(len(ks)), key=lambda i: ks[i].get("us_per_call", 0)))
    g.add(ks[idx]["name"])
    pre.update(k["name"] for k in ks[:idx]); post.update(k["name"] for k in ks[idx + 1:])
  return frozenset(g), frozenset(pre - g), frozenset(post - g)


def tinygrad_shape_signature(name:str) -> tuple[str, int, tuple[int, ...]] | None:
  """(kind letter, product of all dims, last 4 dims) of a tinygrad kernel name such as r_640_16_8_8_49_4.
  The same matvec re-split along its global dim (r_5_128_16_8_8_49_4 at B=8 attn_q) keeps the signature."""
  parts = name.split("_")
  if len(parts) < 5 or parts[0] not in ("r", "E") or not all(x.isdigit() for x in parts[1:]): return None
  dims = [int(x) for x in parts[1:]]
  prod = 1
  for d in dims: prod *= d
  return parts[0], prod, tuple(dims[-4:])


def _ours_is_gemm_factory(gemm_names:frozenset[str]):
  sigs = {sg for sg in (tinygrad_shape_signature(n) for n in gemm_names) if sg is not None and sg[1] >= 1 << 20}
  def f(k:dict[str, Any]) -> bool:
    md = k.get("metadata") or {}
    if md.get("schema_version") == FULL_KERNEL_CANDIDATE_SCHEMA or k["name"] in gemm_names: return True
    sg = tinygrad_shape_signature(k["name"])
    return sg is not None and sg in sigs
  return f


def load_hcq_profile_lines(jsonl:str | pathlib.Path, lines:tuple[int, int] | None = None) -> list[list[dict[str, Any]]]:
  """HCQ_GRAPH_PROFILE_JSON jsonl -> one kernel list per graph launch (names ANSI/hash-stripped; times in us)."""
  out = []
  with open(jsonl) as f:
    for i, line in enumerate(f):
      if lines is not None and not (lines[0] <= i < lines[1]): continue
      row = json.loads(line)
      out.append([_k(strip_tinygrad_name(e["name"]), e["start"], e["end"], metadata=e.get("metadata") or {})
                  for e in row["entries"]])
  return out


def ours_decode_steps(graphs:list[list[dict[str, Any]]], *, anchor_name:str | None = None,
                      flush_name:str | None = None) -> dict[str, Any]:
  graphs = sorted((g for g in graphs if g), key=lambda g: min(k["s"] for k in g))
  if anchor_name is None:
    anchor_name = collections.Counter(g[0]["name"] for g in graphs).most_common(1)[0][0]
  idx = [i for i, g in enumerate(graphs) if g[0]["name"] == anchor_name]
  if len(idx) < 2: raise ValueError(f"fewer than two step anchors ({anchor_name!r})")
  steps, periods, glaunch = [], [], 0
  for a, b in zip(idx, idx[1:]):
    st = []
    for g in graphs[a:b]:
      flush = flush_name is not None and g[0]["name"] == flush_name
      st += [dict(k, flush=flush) for k in g]
    st.sort(key=lambda k: k["s"])
    steps.append(st); glaunch += b - a
    periods.append(min(k["s"] for k in graphs[b]) - min(k["s"] for k in graphs[a]))
  return {"steps": steps, "periods": periods, "graph_launches_per_step": glaunch / len(steps), "anchor": anchor_name}


def ours_side(graphs:list[list[dict[str, Any]]], ours_gemm_rows:list[dict[str, Any]], *, kind:str,
              meta:dict[str, Any] | None = None, pattern:str = NEMOTRON_H_4B_PATTERN,
              attn_layout:str = "q_k_v", source:dict[str, Any] | None = None) -> dict[str, Any]:
  meta = meta or {}
  gemm_names, pre, post = ours_gemm_names(ours_gemm_rows)
  is_gemm = _ours_is_gemm_factory(gemm_names)
  expected = layer_roles(pattern, attn_layout)
  if kind == "decode":
    step_lists = list(meta.get("step", {}).values())
    anchor = strip_tinygrad_name(step_lists[-1][0]["name"]) if step_lists and step_lists[-1] else None
    flush = strip_tinygrad_name(meta["flush"][0]["name"]) if meta.get("flush") else None
    seg = ours_decode_steps(graphs, anchor_name=anchor, flush_name=flush)
    steps, periods, glaunch = seg["steps"], seg["periods"], seg["graph_launches_per_step"]
    merge, out_policy = False, "auto"
  else:
    # HCQ replay timestamps are not comparable across graph launches of a multi-graph prefill (the capture shows
    # 150-350 ms gaps inside a 1.6 s rep), so order = launch (line) order, and wall = the synchronized host wall.
    allk = [dict(k, flush=False) for g in graphs for k in sorted(g, key=lambda k: k["s"])]
    if not meta.get("walls"): raise ValueError("prefill meta has no host walls")
    steps, periods, glaunch = [allk], [float(meta["walls"][-1]) * 1e6], float(len(graphs))
    merge, out_policy = True, "last_only"
    expected = layer_roles(pattern, "q_kv")  # k and v run the same kernel back to back: one merged op
    attn_layout = "q_kv(merged k+v)"
  labels, infos = [], []
  for st in steps:
    body = [k for k in st if not k.get("flush")]
    lab_body, info = attribute_step(body, expected, is_gemm=is_gemm, pre_helpers=pre, post_helpers=post,
                                    merge_same_name_runs=merge, output_policy=out_policy)
    it = iter(lab_body)
    labels.append(["state_flush" if k.get("flush") else next(it) for k in st]); infos.append(info)
  extra = {"side": "ours", "attention_layout": attn_layout, "attribution": _info_summary(infos), "source": source or {}}
  if kind == "decode" and "wall_ms" in meta:
    extra["host_wall_ms_per_step"] = round(float(meta["wall_ms"]), 4)
  if kind == "prefill":
    extra["wall_source"] = "host (synchronized perf_counter around the last rep; HCQ replay timestamps are not " \
                           "comparable across graph launches)"
    extra["gpu_span_ms_unreliable"] = round((max(k["e"] for k in steps[0]) - min(k["s"] for k in steps[0])) / 1e3, 3)
  return _summarize(steps, periods, labels, unit_to_ms=1e-3, graph_launches=glaunch, extra=extra)


# ---------------------------------------------------------------- comparison
def _attn_in(roles:dict[str, Any]) -> float:
  return sum(v["ms"] for r, v in roles.items() if r in ATTN_IN_ROLES)


def compare_sides(ours:dict[str, Any], vllm:dict[str, Any]) -> dict[str, Any]:
  if ours.get("status") != "ok" or vllm.get("status") != "ok":
    return {"status": "incomplete", "reason": "at least one side missing"}
  role_delta: dict[str, dict[str, float]] = {}
  for r in ("ssm_in", "ssm_out", "attn_qkv", "attn_o", "ffn_up", "ffn_down", "output"):
    o = _attn_in(ours["roles"]) if r == "attn_qkv" else ours["roles"].get(r, {}).get("ms", 0.0)
    v = _attn_in(vllm["roles"]) if r == "attn_qkv" else vllm["roles"].get(r, {}).get("ms", 0.0)
    role_delta[r] = {"ours_ms": round(o, 4), "vllm_ms": round(v, 4), "delta_ms": round(o - v, 4)}
  cat_delta = {}
  for c in sorted(set(ours["categories"]) | set(vllm["categories"])):
    o = ours["categories"].get(c, {}).get("ms", 0.0); v = vllm["categories"].get(c, {}).get("ms", 0.0)
    cat_delta[c] = {"ours_ms": round(o, 4), "vllm_ms": round(v, 4), "delta_ms": round(o - v, 4)}
  unassigned = {r: v for r, v in ours["roles"].items() if r == "unassigned_gemm"}
  kernel_delta = ours["kernel_sum_ms_per_step"] - vllm["kernel_sum_ms_per_step"]
  lifecycle_delta = ours["lifecycle_ms_per_step"] - vllm["lifecycle_ms_per_step"]
  wall_delta = ours["wall_ms_per_step"] - vllm["wall_ms_per_step"]
  gemm_delta = sum(v["delta_ms"] for v in role_delta.values())
  ranked = sorted([(f"role:{k}", v["delta_ms"]) for k, v in role_delta.items()] +
                  [(f"category:{k}", v["delta_ms"]) for k, v in cat_delta.items()] +
                  [("lifecycle", round(lifecycle_delta, 4))], key=lambda x: -abs(x[1]))
  return {
    "status": "ok",
    "wall_delta_ms": round(wall_delta, 4),
    "kernel_sum_delta_ms": round(kernel_delta, 4),
    "gemm_delta_ms": round(gemm_delta, 4),
    "non_gemm_delta_ms": round(kernel_delta - gemm_delta, 4),
    "role_deltas": role_delta,
    "category_deltas": cat_delta,
    "lifecycle": {
      "delta_ms": round(lifecycle_delta, 4),
      "ours_ms": ours["lifecycle_ms_per_step"], "vllm_ms": vllm["lifecycle_ms_per_step"],
      "launch_delta": round(ours["kernel_launches_per_step"] - vllm["kernel_launches_per_step"], 2),
      "graph_launch_delta": round(ours["graph_launches_per_step"] - vllm["graph_launches_per_step"], 3),
    },
    "ranked_deltas": [{"term": t, "delta_ms": d} for t, d in ranked],
    "share_of_wall_gap": {
      "kernels": round(kernel_delta / wall_delta, 4) if wall_delta else None,
      "lifecycle": round(lifecycle_delta / wall_delta, 4) if wall_delta else None,
    },
    **({"unassigned": unassigned} if unassigned else {}),
  }


def missing(reason:str, command:str | None = None, **extra) -> dict[str, Any]:
  out: dict[str, Any] = {"status": "missing", "reason": reason}
  if command: out["command"] = command
  out.update(extra)
  return out


def build_document(workloads:list[dict[str, Any]], *, model:str, pattern:str, notes:list[str] | None = None
                   ) -> dict[str, Any]:
  rows = []
  for w in workloads:
    ours, vllm = w["ours"], w["vllm"]
    warn = []
    if ours.get("status") == "ok" and vllm.get("status") == "ok":
      po, pv = w.get("ours_prompt_len"), w.get("vllm_prompt_len")
      if po is not None and pv is not None and po != pv:
        warn.append(f"prompt length differs (ours P={po}, vLLM P={pv}): attention/context-dependent kernels are not "
                    "like-for-like")
      if ours.get("wall_source") or vllm.get("wall_source"):
        warn.append(f"wall source differs: ours = {ours.get('wall_source', 'GPU timeline')}; vLLM = "
                    f"{vllm.get('wall_source', 'GPU timeline')}")
    rows.append({"id": w["id"], "kind": w["kind"], "batch": w.get("batch"), "prompt_len": {"ours": w.get("ours_prompt_len"),
                 "vllm": w.get("vllm_prompt_len")}, "ours": ours, "vllm": vllm, "comparison": compare_sides(ours, vllm),
                 "warnings": warn})
  return {"schema": SCHEMA, "model": model, "layer_pattern": pattern,
          "definitions": {
            "wall_ms_per_step": "GPU-timeline step period (decode: mean interval between step anchors; prefill: first "
                                "kernel start .. last kernel end)",
            "kernel_sum_ms_per_step": "sum of kernel durations per step (overlapping kernels count twice)",
            "lifecycle_ms_per_step": "wall - kernel_sum (launch gaps, host sync, scheduling; negative => overlap)",
            "kernel_launches_per_step": "kernels executed per step (graph-replayed nodes included)",
            "graph_launches_per_step": "graph launches per step (vLLM cudaGraphLaunch; ours HCQ graph executions)",
            "roles": "GEMM ops by role, positional over the layer pattern, helpers (split-K reduce, hi/lo split/sum) "
                     "included",
            "categories": "non-GEMM kernels by the segment they sit in",
            "identity": "wall_delta = sum(role deltas) + sum(category deltas) + lifecycle delta"},
          "notes": notes or [], "workloads": rows}


def render_markdown(doc:dict[str, Any]) -> str:
  L = [f"# Lifecycle comparison: ours vs vLLM ({doc['model']})", "", f"schema `{doc['schema']}`", ""]
  L += ["| workload | side | wall ms/step | kernel-sum ms | lifecycle ms | launches/step | graphs/step |",
        "|---|---|---|---|---|---|---|"]
  for w in doc["workloads"]:
    for side in ("ours", "vllm"):
      s = w[side]
      if s.get("status") != "ok":
        L.append(f"| {w['id']} | {side} | missing | | | | |"); continue
      L.append(f"| {w['id']} | {side} | {s['wall_ms_per_step']:.3f} | {s['kernel_sum_ms_per_step']:.3f} | "
               f"{s['lifecycle_ms_per_step']:.3f} | {s['kernel_launches_per_step']:.0f} | "
               f"{s['graph_launches_per_step']:.2f} |")
  for w in doc["workloads"]:
    L += ["", f"## {w['id']} (B={w.get('batch')}, P ours={w['prompt_len']['ours']} / vLLM={w['prompt_len']['vllm']})", ""]
    for side in ("ours", "vllm"):
      s = w[side]
      if s.get("status") != "ok":
        L.append(f"- **{side}: missing** - {s.get('reason')}" + (f"; produce with `{s['command']}`" if s.get("command") else ""))
    for m in w.get("warnings", []): L.append(f"- warning: {m}")
    c = w["comparison"]
    if c.get("status") != "ok": continue
    lc = c["lifecycle"]
    L += [f"- wall gap {c['wall_delta_ms']:+.3f} ms = kernels {c['kernel_sum_delta_ms']:+.3f} (GEMM "
          f"{c['gemm_delta_ms']:+.3f}, non-GEMM {c['non_gemm_delta_ms']:+.3f}) + lifecycle {lc['delta_ms']:+.3f} ms; "
          f"launches {lc['launch_delta']:+.0f}/step, graph launches {lc['graph_launch_delta']:+.2f}/step", "",
          "| term | ours ms | vLLM ms | delta ms |", "|---|---|---|---|"]
    for k, v in c["role_deltas"].items():
      L.append(f"| {k} | {v['ours_ms']:.3f} | {v['vllm_ms']:.3f} | {v['delta_ms']:+.3f} |")
    for k, v in c["category_deltas"].items():
      L.append(f"| ({k}) | {v['ours_ms']:.3f} | {v['vllm_ms']:.3f} | {v['delta_ms']:+.3f} |")
    L.append(f"| lifecycle (wall - kernel sum) | {lc['ours_ms']:.3f} | {lc['vllm_ms']:.3f} | {lc['delta_ms']:+.3f} |")
  if doc.get("notes"):
    L += ["", "## Notes", ""] + [f"- {n}" for n in doc["notes"]]
  return "\n".join(L) + "\n"


# ---------------------------------------------------------------- manifest driver (the only I/O-heavy entry point)
# Where the traces, the reference bench and the GPU wrapper live is execution configuration, never experiment
# identity: the rule boltbeam/target/tinygrad_root.py states for the tinygrad checkout, applied to the other
# roots the same way. A root nobody configured is written as its $VARIABLE, so a generated command says plainly
# that the path was never known instead of carrying one developer's home directory; the data rows under it come
# out 'missing' with that command attached. The tinygrad checkout itself goes through resolve_tinygrad_root.
ROOT_ENV = {"vllm_bench": "VLLM_BENCH_ROOT",        # the vLLM bench checkout (prof.py, prof_prefill.py, runs/)
            "gpu_run": "GPU_RUN",                   # the wrapper that serialises GPU use on the bench box
            "traces": "LIFECYCLE_TRACES_DIR",       # where ours_b{B}_p{P}.jsonl/_meta.json and vLLM sqlites land
            "audit": "KERNEL_AUDIT_DIR"}            # where the kernel audit's ours.json lives
_SPEC = "docs/nemotron-vllm-parity/bench/spec"
_NSYS = ("nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node "
         "--trace=cuda,nvtx --sample=none --cpuctxsw=none -f true")


def _root(key:str) -> str:
  name = ROOT_ENV[key]
  return os.environ.get(name) or f"${name}"


def _tinygrad_root() -> str:
  return str(resolve_tinygrad_root(allow_unresolved=True))


def ours_decode_command(batch:int, prompt_len:int, out_dir:str) -> str:
  return (f"cd {_tinygrad_root()} && {_root('gpu_run')} time env PROFILE=1 "
          f"HCQ_GRAPH_PROFILE_JSON={out_dir}/ours_b{batch}_p{prompt_len}.jsonl DEV=NV PYTHONPATH=. python3 "
          f"{_SPEC}/ours_prof.py {batch} {prompt_len} 1024 {out_dir}/ours_b{batch}_p{prompt_len}_meta.json")


def ours_prefill_command(prompt_len:int, out_dir:str, piece:int = 1024) -> str:
  return (f"cd {_tinygrad_root()} && {_root('gpu_run')} time env PROFILE=1 "
          f"HCQ_GRAPH_PROFILE_JSON={out_dir}/ourspf_{prompt_len}.jsonl DEV=NV PYTHONPATH=. python3 "
          f"{_SPEC}/ours_pf_prof.py {prompt_len} {piece} ssd float {out_dir}/ourspf_{prompt_len}_meta.json")


def vllm_decode_command(batch:int, prompt_len:int, out_dir:str) -> str:
  """vprof64.sh recipe (bench/spec): prof.py under nsys, then export the capture to sqlite."""
  o = f"{out_dir}/prof_b{batch}_p{prompt_len}"
  vb = _root("vllm_bench")
  return (f"cd {vb} && {_root('gpu_run')} time env PATH={vb}/.venv/bin:/usr/local/cuda/bin:$PATH "
          f"CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0 GPU_UTIL=0.85 EXTRA='{{\"disable_log_stats\": false}}' "
          f"{_NSYS} -o {o} .venv/bin/python prof.py {batch}:{prompt_len} 300 && "
          f"nsys export --type sqlite -o {o}.1.sqlite {o}.1.nsys-rep")


def nemotron_default_manifest(ours_dir:str | None = None, audit_dir:str | None = None) -> dict[str, Any]:
  """Where the Nemotron-H 4B / RTX 5090 traces live (2026-09-26). Missing files become explicit 'missing' rows."""
  ours_dir = ours_dir or _root("traces")
  audit_dir = audit_dir or _root("audit")
  vb = _root("vllm_bench")
  runs = f"{vb}/runs"
  def od(b, p): return {"jsonl": f"{ours_dir}/ours_b{b}_p{p}.jsonl", "meta": f"{ours_dir}/ours_b{b}_p{p}_meta.json",
                        "prompt_len": p, "command": ours_decode_command(b, p, ours_dir)}
  return {
    "model": "NVIDIA-Nemotron-3-Nano-4B-BF16 (RTX 5090)", "pattern": NEMOTRON_H_4B_PATTERN,
    "ours_gemm_json": f"{audit_dir}/ours.json",
    "workloads": [
      {"id": "decode_b8", "kind": "decode", "batch": 8, "ours": od(8, 200),
       "vllm": {"sqlite": f"{runs}/prof_default.1.sqlite", "prompt_len": 200, "command": vllm_decode_command(8, 200, ours_dir)}},
      {"id": "decode_b32", "kind": "decode", "batch": 32, "ours": od(32, 200),
       "vllm": {"sqlite": f"{ours_dir}/prof_b32_p200.1.sqlite", "prompt_len": 200,
                "command": vllm_decode_command(32, 200, ours_dir)}},
      {"id": "decode_b64", "kind": "decode", "batch": 64, "ours": od(64, 2000),
       "vllm": {"sqlite": f"{runs}/prof_spec64.1.sqlite", "prompt_len": 2000, "command": vllm_decode_command(64, 2000, ours_dir)}},
      {"id": "decode_b128", "kind": "decode", "batch": 128, "ours": od(128, 2000),
       "vllm": {"sqlite": f"{runs}/prof_default.3.sqlite", "prompt_len": 2000,
                "command": vllm_decode_command(128, 2000, ours_dir)}},
      {"id": "prefill_10k", "kind": "prefill", "batch": 1,
       "ours": {"jsonl": f"{ours_dir}/ourspf_10000.jsonl", "meta": f"{ours_dir}/ourspf_10000_meta.json",
                "prompt_len": 10000, "command": ours_prefill_command(10000, ours_dir)},
       "vllm": {"sqlite": f"{runs}/prof_prefill10k.sqlite", "prompt_len": 10000,
                "command": f"cd {vb} && {_NSYS} -o runs/prof_prefill10k .venv/bin/python prof_prefill.py && "
                           "nsys export --type sqlite -o runs/prof_prefill10k.sqlite runs/prof_prefill10k.nsys-rep"}},
    ],
    "notes": [
      "ours = tinygrad-self-training exp (DEV=NV, HCQ graph profile, PROFILE=1), vLLM 0.30 (nsys, cuda-graph-trace=node);"
      " ours traces 2026-09-26 via bench/spec/ours_prof.py / ours_pf_prof.py (host wall/step printed in each meta).",
      "vLLM B=32 trace was produced 2026-09-26 with VLLM_CACHE_ROOT pointed at a fresh dir (the default cache holds a "
      "root-owned flashinfer autotune file that makes the engine fail at warmup).",
      "ours relu2 is fused into ffn_down's input prep (counted in ffn_down); vLLM's relu2 kernel is mlp_act.",
      "norm_residual includes the kernels before the first GEMM of a step (embedding, first norm); "
      "sampling_bookkeeping = everything after the output GEMM (sampler, vLLM input prep and mamba align copies).",
      "ours decode steps carry the amortized state flush (state_flush) that runs every ~16 steps.",
    ],
  }


def _load_ours(spec:dict[str, Any], kind:str, ours_rows:list[dict[str, Any]] | None, pattern:str) -> dict[str, Any]:
  jsonl, meta_p = spec.get("jsonl"), spec.get("meta")
  if not jsonl or not pathlib.Path(jsonl).exists() or not meta_p or not pathlib.Path(meta_p).exists():
    return missing(f"no HCQ graph profile at {jsonl}", spec.get("command"))
  if ours_rows is None:
    return missing("ours_gemm_json (audit ours.json) not found: needed to recognise GEMM ops and hi/lo helpers")
  meta = json.loads(pathlib.Path(meta_p).read_text())
  if kind == "decode":
    graphs = load_hcq_profile_lines(jsonl, tuple(meta["lines"]))
  else:  # ours_pf_prof.py: marks[i]..marks[i+1] = rep i; the last rep is the warmest
    marks = meta["marks"]
    graphs = load_hcq_profile_lines(jsonl, (marks[-2], marks[-1]))
  try:
    return ours_side(graphs, ours_rows, kind=kind, meta=meta, pattern=pattern, source={"jsonl": jsonl, "meta": meta_p})
  except ValueError as exc:
    return missing(f"unparseable ours trace: {exc}", spec.get("command"))


def _load_vllm(spec:dict[str, Any], kind:str, pattern:str) -> dict[str, Any]:
  db = spec.get("sqlite")
  if not db or not pathlib.Path(db).exists():
    return missing(f"no nsys sqlite at {db}", spec.get("command"))
  K, g = load_nsys_sqlite(db)
  try:
    return vllm_side(K, g, kind=kind, pattern=pattern, window=tuple(spec.get("steady_window", (0.2, 0.8))),
                     source={"sqlite": db})
  except ValueError as exc:
    return missing(f"unparseable vLLM trace: {exc}", spec.get("command"))


def run_manifest(manifest:dict[str, Any], only:Iterable[str] | None = None) -> dict[str, Any]:
  pattern = manifest.get("pattern", NEMOTRON_H_4B_PATTERN)
  gj = manifest.get("ours_gemm_json")
  ours_rows = json.loads(pathlib.Path(gj).read_text()) if gj and pathlib.Path(gj).exists() else None
  if isinstance(ours_rows, dict): ours_rows = ours_rows["rows"]
  only = set(only) if only else None
  rows = []
  for w in manifest["workloads"]:
    if only and w["id"] not in only: continue
    rows.append({"id": w["id"], "kind": w["kind"], "batch": w.get("batch"),
                 "ours_prompt_len": w["ours"].get("prompt_len"), "vllm_prompt_len": w["vllm"].get("prompt_len"),
                 "ours": _load_ours(w["ours"], w["kind"], ours_rows, pattern),
                 "vllm": _load_vllm(w["vllm"], w["kind"], pattern)})
  return build_document(rows, model=manifest.get("model", "?"), pattern=pattern, notes=manifest.get("notes"))
