"""`boltbeam tune MODEL --engine E`: this machine's values for an engine's knobs, each with where it came from, from one table
(data/engine_knobs.json): a row per knob, keyed by engine, says what it controls and how it is set.

  facts     a formula over named facts (host RAM and L3, the model's bytes, the GPU, the engine's own facts): instant
  measured  a short real-text prefill sweep under --measure, over the row's axis knob, with the row's points, tie and
            stop rule; without --measure the facts value or the default stands
  default   left at the default, with the evidence why

This module is engine-blind (policy/guards.py checks it): the table names engines, and each engine's adapter (the
module its table row names) does the four engine-specific things through one interface:

  detect(build) -> features       what the build is and the features it has, read without running a model
  facts(model, host, features, options) -> facts   the model's facts by their generic names, the engine's own as ns.name
  launch_measure(model, features, facts, port, out, extra) -> run(values, prompt, reps, tag) -> wall ms
  render(values, facts, features, model, extra) -> {env, flags, config, argv, command}

and declares NS (its fact namespace) and FACTS (its facts and what they are). Adding an engine is an adapter module and
rows; nothing here changes."""
from __future__ import annotations

import ast
import datetime
import importlib
import json
import math
import os
import pathlib
import statistics
from typing import Any, Callable

from boltbeam.core.refused import Refused
from boltbeam.workflow import mods

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "engine_knobs.json"
ADAPTERS = TABLE.parent / "engine_adapters.json"  # engines.<E>.adapter: the module table() puts in each engine's row
KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
PROMPT = 10000  # the prompt length the setup is stated at (the evidence's 10k)
DETAIL_CHARS = 300  # the end of an engine's error text kept as provenance: its last lines say why, its start is the command
SEED_BITS = 31  # a choice sweep's order seed: a positive whole number a reader can retype and JSON keeps exact
SERVED = 1  # tune states a setup for one request at a time: its context fit, its KV estimate and its decode rows are one request's
REPLY_TOKENS = 2048  # room for the reply beside the prompt when no --context is given (a context of the prompt alone holds no reply)


def prompt_at(context:int | None) -> int:
  """The prompt tune states a setup at for this context: PROMPT, or below it the context less room for the reply (at most half)."""
  return context - min(REPLY_TOKENS, context // 2) if context and context < PROMPT else PROMPT
FACTS = {  # the engine-neutral names a formula may read: the fact and what it is
  "host_expert_bytes": "the experts this placement keeps in host RAM (the model's tensor table, per layer)",
  "model_experts": "experts per layer (0: a dense model)",
  "model_weight_bytes": "the weights the engine holds on the GPU",
  "model_draft_layers": "the checkpoint's draft (multi-token prediction) layers",
  "kv_bytes_per_token": "KV cache bytes per token at the model's own type",
  "usable_ram_bytes": "available RAM less the margin, floored to a GiB (target/host.usable_ram)",
  "ram_available_bytes": "RAM available now (/proc/meminfo MemAvailable, macOS vm_stat)",
  "ram_total_bytes": "RAM installed", "ram_margin_bytes": "RAM kept free for the system and the engine (target/host.RAM_MARGIN_BYTES)",
  "l3_bytes": "one L3 instance (target/host.l3_cache)", "page_bytes": "the memory page size",
  "gpu_total_bytes": "the GPU's memory", "gpu_free_bytes": "the GPU's memory free now",  # over every card the engine splits over
  "gpu_count": "the cards the engine splits its layers over", "gpu_free_min_bytes": "the tightest card's memory free now",
  "prompt_tokens": "the prompt length the setup is stated at", "context_tokens": "the context the setup must hold", "reply_tokens": "the reply length the context leaves room for, from measured replies (--replies; absent when not measured)",
  "shared_memory": "the GPU reads host memory directly (no copies)",
  "h2d_pinned_gbs": "the measured copy rate to the GPU from pinned host memory, GB/s (target/host.facts)",
  "gpu_name": "the GPU's name (the hardware scan)", "model_file": "the model file's name",
  "cpu_physical_cores": "the CPU's physical cores (target/host.cpu_cores)", "cpu_logical_cores": "the CPU's logical CPUs (target/host.cpu_cores)",
  "cpu_l3_cores": "the physical cores that share one L3 (one core complex; target/host.cpu_cores)",
  "served_requests": "the requests the setup is stated for at once (SERVED)",
  "workload": "what the server will serve: one-shot (each prompt once) or chat (follow-ups that reuse earlier turns), --workload",
  "gpu_after_load_bytes": "the GPU's memory free once the engine has loaded the model, with the rows that take GPU memory at their defaults: the engine's estimate (its after_load)",
  "gpu_free_after_load_bytes": "--measure: the GPU's memory free right after the engine loaded the model, read in a run of the setup",
  "gpu_free_lowest_bytes": "--measure: the least GPU memory free read after each of that run's replies, short and at the context (not during one)",
  "value": "the row's own value (in sets)"}
FUNCS: dict[str, Callable[..., Any]] = {"min": min, "max": max, "pow2_floor": lambda x: 1 << (int(x).bit_length() - 1) if x >= 1 else 0,
                                         "gib_floor": lambda x: int(x // GiB), "gib_ceil": lambda x: int(-(-x // GiB)),
                                         "floor_to": lambda x, step: round(int(x / step) * step, 6),
                                         "ceil": lambda x: int(math.ceil(x)), "floor": lambda x: int(math.floor(x)),
                                         "has": lambda text, part: str(part).lower() in str(text).lower(),  # a name holds a part (a GPU, a model)
                                         "len": len, "count": lambda x: _count(x), "at": lambda d, key: _at(d, key)}
WORKLOADS = ("chat", "one-shot")
BUDGET_PROMPT_TOLERANCE = 0.05  # a time budget stands for a prompt within this share of its own length (10k runs read 9,8xx tokens)
SESSION_MARGIN = GiB  # GPU memory the memory fit keeps free beside a setup for what a session adds after load (the engine's pools grow
# with the context, CUDA graphs are built later). An estimate, not measured: on an RTX 5090 a server ran out at turn 7 of a chat
# with 0.77 GiB free after load, and a load with about 0.69 GiB projected free ran out at load; 1.08 GiB free after load held
# through four sessions (~/storage/boltbeam-runs/replay-h2h-20261113 runs 8-10 and the -bs A/B they cite). Those readings were
# taken after a load with no warm-up; under --measure the setup's own fall after its warmed load, over replies out to the
# context, replaces it (budget), and each combination's own run is read too (fit_measure): ~/storage/boltbeam-runs/joint-fit-20261113 run 4 read 0.28 GiB free after a warmed load and 0.27 GiB of
# fall for a setup that ran clean, which this estimate would have called over budget.
CONFIRM = 3  # --measure confirms this many of the fastest predicted combinations that fit, on real text
CONSTS = {"KiB": KiB, "MiB": MiB, "GiB": GiB, "True": True, "False": False, "None": None}


def table(path:pathlib.Path = TABLE) -> dict[str, Any]:
  """The knob table, each engine's row with its adapter and MTP modules (engine_adapters.json) first, as one table."""
  tab, mods = json.loads(path.read_text()), json.loads(ADAPTERS.read_text())["engines"]
  tab["engines"] = {e: {**mods.get(e, {}), **row} for e, row in tab["engines"].items()}
  return tab


def engines(tab:dict[str, Any] | None = None) -> list[str]:
  return list((tab or table())["engines"])


def adapter(engine:str, tab:dict[str, Any] | None = None):
  """The engine's adapter module, as its table row names it."""
  tab = tab or table()
  if engine not in tab["engines"]:
    raise Refused(f"BoltBeam has no knob table for {engine}; it has one for {', '.join(tab['engines'])}")
  return importlib.import_module(tab["engines"][engine]["adapter"])


def knobs(engine:str, tab:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  return [k for k in (tab or table())["knobs"] if k["engine"] == engine]


def unchecked(engine:str, a:list[dict[str, Any]], b:list[dict[str, Any]], tab:dict[str, Any] | None = None) -> list[str]:
  """The rows two setups differ in that the engine's output check cannot see (output_check false): a check of them is refused."""
  blind, x, y = {k["name"] for k in knobs(engine, tab) if k.get("output_check") is False}, *({v["name"]: v["value"] for v in vals} for vals in (a, b))
  return sorted(n for n in x.keys() | y.keys() if n in blind and x.get(n) != y.get(n))


def fact_names(engine:str, tab:dict[str, Any] | None = None) -> set[str]:
  """The facts a row of this engine may read: the generic ones and its adapter's namespaced ones."""
  ad, tab = adapter(engine, tab), tab or table()
  return set(FACTS) | {f"{ad.NS}.{n}" for n in ad.FACTS} | {f"load.{n}" for n in (tab["engines"][engine].get("after_load") or {}).get("terms") or {}}


def after_reads(engine:str, tab:dict[str, Any]) -> set[str]:
  """The facts the engine's after-load estimate reads (its formula and terms): a row's sets that moves one moves the estimate."""
  al = tab["engines"][engine].get("after_load") or {}
  return {x for e in [al["formula"], *al["terms"].values()] for x in names_in(e)} if al else set()


def after_load(f:dict[str, Any], engine:str, tab:dict[str, Any]) -> dict[str, Any]:
  """The engine's estimate of the GPU memory free once it has loaded the model (its table row's after_load: terms and a
  formula over the facts), as gpu_after_load_bytes and its terms as load.<name>, for any row and the memory fit to read.
  Unknown (None) when a fact it reads is."""
  al = tab["engines"][engine].get("after_load")
  if not al:
    return {}
  allowed = fact_names(engine, tab)
  t = terms(al, f, allowed)
  out = {f"load.{n[2:]}": v for n, v in t.items()}
  out["gpu_after_load_bytes"] = _try(al["formula"], {**f, **t}, allowed | term_names(al))
  return out


# --- formulas: a restricted expression over named facts -------------------------------------------------------------------

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: _div(a, b, False),
        ast.FloorDiv: lambda a, b: _div(a, b, True)}
_CMP = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
        ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


class Missing(Exception):
  """A formula read a fact this machine or model did not give (or divided by a zero one)."""


def _count(x) -> int:
  """A whole number given as a number or its digits; anything else (a word, a fraction) is unknown: Missing."""
  if isinstance(x, bool) or not str(x).strip().isdigit():
    raise Missing("a count")
  return int(x)


def _at(d, key):
  """A mapping's entry; a key it lacks is unknown (Missing)."""
  if not isinstance(d, dict) or key not in d:
    raise Missing("an entry")
  return d[key]


def _div(a, b, floor:bool):
  if not b:
    raise Missing("a zero divisor")
  return a // b if floor else a / b


def _name(n:ast.AST) -> str | None:
  """A fact's name: x, or an engine's own ns.x."""
  if isinstance(n, ast.Name):
    return n.id
  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
    return f"{n.value.id}.{n.attr}"
  return None


def names_in(expr:str) -> list[str]:
  """The facts a formula reads, in order, without functions and constants."""
  out = []
  for n in ast.walk(ast.parse(expr, mode="eval")):
    if isinstance(n, ast.Attribute) and (got := _name(n)):
      out.append(got)
    elif isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTS:
      out.append(n.id)
  dotted = {o.split(".")[0] for o in out if "." in o}
  return list(dict.fromkeys(o for o in out if o not in dotted))


def evaluate(expr:str, facts:dict[str, Any], allowed:set[str] | None = None) -> Any:
  """The formula's value over facts. Only fact names (FACTS, or `allowed` when given), numbers, strings, arithmetic,
  comparisons, and/or/not, x if c else y and FUNCS: anything else is a ValueError (the table is data, never code). A
  fact that is None raises Missing."""
  allowed = allowed if allowed is not None else set(FACTS) | {k for k in facts if "." in k}

  def ev(n:ast.AST) -> Any:
    if isinstance(n, ast.Expression):
      return ev(n.body)
    if isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, (int, float, bool, str))):
      return n.value
    name = _name(n)
    if name is not None:
      if name in CONSTS:
        return CONSTS[name]
      if name not in allowed and name != "value":
        raise ValueError(f"{name!r} is not a fact a knob formula may read")
      if name == "value":  # a row's own value is None when it keeps the default
        return facts.get("value")
      if facts.get(name) is None:
        raise Missing(name)
      return facts[name]
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
      return _BIN[type(n.op)](ev(n.left), ev(n.right))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.Not)):
      v = ev(n.operand)
      return -v if isinstance(n.op, ast.USub) else not v
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
      return _CMP[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
    if isinstance(n, ast.BoolOp):
      vals = (ev(v) for v in n.values)
      return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.IfExp):
      return ev(n.body) if ev(n.test) else ev(n.orelse)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
      return FUNCS[n.func.id](*(ev(a) for a in n.args))
    raise ValueError(f"{ast.dump(n)[:60]} is not allowed in a knob formula")
  return ev(ast.parse(expr, mode="eval"))


class _Unknown:
  def __format__(self, spec:str) -> str:
    return "unknown"

  def __str__(self) -> str:
    return "unknown"


class _View(dict):
  def __missing__(self, key):
    return _Unknown()

  def __getattr__(self, key):
    return self[key]


def say(template:str, facts:dict[str, Any]) -> str:
  """A row's reason with its facts filled in: *_bytes also as *_gib and *_mib, ns.x as an engine's own, a list as its
  items, a missing or None fact as "unknown"."""
  view = _View()
  for k, v in facts.items():
    ns, _, key = k.rpartition(".")
    into = view.setdefault(ns, _View()) if ns else view
    if not isinstance(into, _View):
      continue
    into[key] = _Unknown() if v is None else ", ".join(map(str, v)) if isinstance(v, list) else v
    if key.endswith("_bytes"):
      base = key[:-6]
      into[f"{base}_gib"], into[f"{base}_mib"] = (_Unknown(), _Unknown()) if not isinstance(v, (int, float)) else (v / GiB, v / MiB)
  return template.format_map(view)


# --- the host: engine-neutral facts --------------------------------------------------------------------------------------

def copy_rate(link:dict[str, Any] | None = None) -> tuple[float | None, str, str | None]:
  """The measured pinned copy rate to the run's GPU (GB/s), where it came from, and the GPU's name. The GPU's bus id and
  name come from the hardware scan; the rate is the figure host.json saved for that bus id (target/host.saved_h2d): no
  copy or RAM probe runs here. None when it was never measured. link stands in for the saved figure (tests; with its
  "gpu_name" when a test gives one)."""
  name = None
  if link is None:
    from boltbeam.target import host
    from boltbeam.target.hardware import hardware_profile
    try:
      dev = (hardware_profile()["gpu"].get("devices") or [None])[0] or {}
      name = dev.get("name")
      link = host.saved_h2d(dev.get("pci_bus_id")) or {}
    except Exception:  # noqa: BLE001  no reading is a stated unknown, never a crash
      return None, "the GPU could not be read here", None
  else:
    name = link.get("gpu_name")
  rate = (link.get("h2d_gbs") or {}).get("pinned")
  return ((float(rate), str(link.get("h2d_source") or "measured"), name) if rate else
          (None, str(link.get("h2d_source") or "never measured (boltbeam host measures it)"), name))


def host_facts(*, memory:dict[str, Any] | None = None, l3:dict[str, Any] | None = None, prompt:int = PROMPT,
               context:int | None = None, workload:str = "chat", link:dict[str, Any] | None = None, reply:dict[str, Any] | None = None,
               cpu:dict[str, Any] | None = None) -> dict[str, Any]:
  """RAM, L3, the CPU's cores, the copy rate to the GPU, the page size, the workload and the prompt and context the setup
  is stated at, with "sources". memory, l3, cpu and link (host.facts' record) stand in for the readings (tests)."""
  from boltbeam.target import host
  if workload not in WORKLOADS:
    raise Refused(f"--workload is one of {', '.join(WORKLOADS)}, not {workload!r}")
  rate, rate_source, gpu_name = copy_rate(link)
  if context is not None and reply:
    raise Refused("--context and --replies both say how much room the reply gets: give one or the other")
  if context is not None and context < prompt:
    raise Refused(f"--context {context} is shorter than the {prompt}-token prompt: the context holds the prompt and the reply")
  mem = memory or host.memory()
  ram = host.usable_ram(mem["total_bytes"], mem.get("available_bytes"))
  l3 = l3 or host.l3_cache()
  cpu = cpu or host.cpu_cores()
  page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
  out = {"usable_ram_bytes": ram["usable_bytes"], "ram_available_bytes": ram.get("available_bytes"), "ram_total_bytes": ram["total_bytes"],
         "ram_margin_bytes": ram["margin_bytes"], "l3_bytes": l3.get("bytes"), "l3_source": l3["source"], "page_bytes": page,
         "prompt_tokens": prompt, "context_tokens": context or prompt + (reply["value"] if reply else REPLY_TOKENS), "h2d_pinned_gbs": rate, "workload": workload,
         "gpu_name": gpu_name, "cpu_physical_cores": cpu["physical"], "cpu_logical_cores": cpu["logical"], "cpu_l3_cores": cpu.get("l3_cores"), "served_requests": SERVED, **({"reply_tokens": reply["value"]} if reply else {})}
  out["sources"] = {"usable_ram_bytes": ram["says"], "ram_available_bytes": mem["source"], "ram_total_bytes": mem["source"],
                    "ram_margin_bytes": "target/host.RAM_MARGIN_BYTES", "l3_bytes": l3["source"], "page_bytes": "os.sysconf SC_PAGE_SIZE",
                    "prompt_tokens": "--prompt", "context_tokens": "--context" if context else f"--prompt plus {reply['value']} tokens for the reply: {reply['says']}" if reply else
                    f"--prompt plus {REPLY_TOKENS} tokens for the reply (the unmeasured default: no --context and no --replies given)",
                    "h2d_pinned_gbs": rate_source, "workload": "--workload", "gpu_name": "the hardware scan",
                    "cpu_physical_cores": cpu["source"], "cpu_logical_cores": cpu["source"], "cpu_l3_cores": cpu["source"], "served_requests": "SERVED"}
  return out


def time_budget(path:str | pathlib.Path, prompt:int | None = None) -> dict[str, Any]:
  """A measured time budget (collectors/time_budget.py) read from a file: the record itself, or a run's results.json
  that holds one (loss.prefill.lengths[L].time_budget: the length nearest `prompt`, else the first). A Refused when the
  file holds none."""
  p = pathlib.Path(path).expanduser()
  try:
    d = json.loads(p.read_text())
  except (OSError, ValueError) as exc:
    raise Refused(f"{p.name}: no time budget could be read (missing or not JSON)") from exc
  if not isinstance(d, dict):
    raise Refused(f"{p.name}: holds no time budget (not a time budget or a run's results)")
  if d.get("schema") != "boltbeam.time_budget.v1":
    lengths = (((d.get("loss") or {}).get("prefill") or {}).get("lengths") or {})
    found = [{"prompt_tokens": int(L), **v["time_budget"]} for L, v in lengths.items() if isinstance(v, dict) and isinstance(v.get("time_budget"), dict)
             and str(L).isdigit()]
    d = min(found, key=lambda b: abs(b["prompt_tokens"] - prompt)) if found and prompt else (found or [{}])[0]
  if not isinstance(d.get("kernels_ms"), (int, float)):
    raise Refused(f"{p.name}: holds no time budget with a kernel time")
  return {**d, "what": d.get("what") or p.name}


def reply_budget(path:str | pathlib.Path, target:float = 0.05) -> dict[str, Any]:
  """The reply budget measured replies call for (a JSONL file, one row per reply: tokens, finish; "length" = cut at the
  budget it ran with): the (1 - target) quantile of the lengths, the smallest length that holds at least that share of
  replies. A cut reply is longer than it shows, so when more than `target` were cut the quantile is not placed by the
  data: a Refused. {value, n, cut, quantile, what, says}."""
  p = pathlib.Path(path).expanduser()
  got: list[tuple[int, bool]] = []
  try:
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
  except (OSError, ValueError) as exc:
    raise Refused(f"{p.name}: no replies could be read (a JSON line per reply with its tokens and finish is needed)") from exc
  for i, r in enumerate(rows, 1):
    ok = isinstance(r, dict) and isinstance(r.get("tokens"), int) and not isinstance(r.get("tokens"), bool) and r["tokens"] > 0
    if not ok or r.get("finish") not in ("stop", "length"):
      raise Refused(f"{p.name}: reply {i} does not say its length in tokens and how it ended (finish is stop, or length when it was cut off)")
    got.append((r["tokens"], r["finish"] == "length"))
  if not got:
    raise Refused(f"{p.name}: holds no replies")
  n, cut, q = len(got), sum(c for _, c in got), 1 - target
  if cut > target * n:
    at = sorted({t for t, c in got if c})
    raise Refused(f"{p.name}: {cut} of {n} replies were cut off at {', '.join(f'{t:,}' for t in at)} tokens ({cut / n:.1%} of them, more than the {target:.0%} the budget may leave cut), "
                  f"so how long they would have run is not known: measure the replies again at a larger budget")
  k = math.ceil(round(q * n, 9))
  value = int(sorted(math.inf if c else t for t, c in got)[k - 1])  # a cut reply is at least its length: it sorts last
  upto = sum(t <= value and not c for t, c in got)
  few = f"n={n}: too few replies for a {q:.0%} length, so this is the longest measured reply: " if n < 20 else ""
  return {"value": value, "n": n, "cut": cut, "quantile": q, "what": p.name,
          "says": f"{few}{upto} of {n} measured replies ({cut} cut) are {value:,} tokens or shorter, the smallest length that holds {q:.0%} of them ({p.name})"}


# --- the rows ----------------------------------------------------------------------------------------------------------

def _shown(v:Any) -> Any:
  return round(v / GiB, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= GiB else v


def groups_in(engine:str, features:dict[str, Any], tab:dict[str, Any]) -> dict[str, bool]:
  """The engine's named extras (mods) and whether the build has each (its marker is a feature)."""
  return {g: bool(group_markers(engine, v) & set(features["features"])) for g, v in tab["engines"][engine]["groups"].items()}


def group_markers(engine:str, group:dict[str, Any]) -> set[str]:
  """A group's markers: the arkey_ie_mods index's for its mod, in every version that has it."""
  return {m["marker"] for m in mods.named(engine, group["mod"])}


def _holds(expr:str | None, f:dict[str, Any], allowed:set[str]) -> bool:
  """A row's measured_when: true only when it reads true over facts that are all known (an unknown fact is not a match)."""
  try:
    return bool(evaluate(expr, f, allowed)) if expr else False
  except Missing:
    return False


def term_names(k:dict[str, Any]) -> set[str]:
  """The names a row's terms give (t.<name>), for its formula, sweep and reason to read."""
  return {f"t.{n}" for n in k.get("terms") or {}}


def terms(k:dict[str, Any], f:dict[str, Any], allowed:set[str]) -> dict[str, Any]:
  """A row's terms, in order: each a named formula over the facts and the terms before it (t.<name>), so a rule's steps
  are data in the row and its reason can show the arithmetic. A term that reads an unknown fact is None (unknown)."""
  out: dict[str, Any] = {}
  ok = allowed | term_names(k)
  for name, expr in (k.get("terms") or {}).items():
    try:
      out[f"t.{name}"] = evaluate(expr, {**f, **out}, ok)
    except Missing:
      out[f"t.{name}"] = None
  return out


def case(field:Any, f:dict[str, Any], allowed:set[str]) -> str:
  """A row's text: a string, or cases [{"when": expr, "text": ...}, ...] read in order, the first whose `when` holds over
  known facts (a case without one is the fallback), so a row says why for the setup it is in."""
  if isinstance(field, str):
    return field
  return next((c["text"] for c in field if "when" not in c or _holds(c["when"], f, allowed)), "")


def budget_unusable(budget:dict[str, Any], prompt:int) -> str | None:
  """Why a measured time budget cannot stand for this run's prompt, or None when it can: it must say its prompt length,
  and that length must be within BUDGET_PROMPT_TOLERANCE of the prompt being tuned."""
  n = budget.get("prompt_tokens")
  if not isinstance(n, (int, float)) or n <= 0:
    return "the time budget does not say what prompt length it was measured at"
  if abs(n - prompt) > BUDGET_PROMPT_TOLERANCE * prompt:
    return f"the time budget was measured on a {int(n)}-token prompt, not within {BUDGET_PROMPT_TOLERANCE:.0%} of the {prompt}-token prompt tuned here"
  return None


def placed(engine:str, model:str, *, host_expert_bytes:float, ram:dict[str, Any], features=None, profile:dict[str, Any] | None = None,
           draft:str | None = None, tab:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """The engine's rows that hang only on where the experts sit and on RAM (knob rows marked placed), at one placement:
  what the offload page sets beside that placement's flags. ram: target/host.usable_ram's figure. features: the build's
  (None: not read, so the row's `needs` is said beside it; a row whose requires a read build lacks is left out). Each
  row that derives a value: name, value, flags, says (its reason) and output (its label). A row whose facts are missing keeps its default and is left out."""
  tab = tab or table()
  rows = [k for k in knobs(engine, tab) if k.get("placed") and (features is None or all(r in features for r in k["requires"]))]
  ad = adapter(engine, tab)
  if not rows or not hasattr(ad, "placed_facts"):
    return []
  got, _ = ad.placed_facts(model, host_expert_bytes, features, profile=profile, draft=draft)
  f = {**got, "host_expert_bytes": host_expert_bytes, "usable_ram_bytes": ram.get("usable_bytes"), "ram_available_bytes": ram.get("available_bytes")}
  ok, out = fact_names(engine, tab), []
  for k in rows:
    try:
      value = evaluate(k["formula"], f, ok) if evaluate(k.get("applies", "True"), f, ok) else None
    except Missing:
      value = None
    if value is not None:
      out.append({"name": k["name"], "value": str(value), "flags": [k["name"], str(value)] if k["kind"] == "flag" else [k["name"]],
                  "says": say(k["reason"], f) + ("; " + k["needs"] if features is None and k.get("needs") else ""), "output": output_of(k, value)["output"]})
  return out


def derive(f:dict[str, Any], features:dict[str, Any], engine:str, tab:dict[str, Any] | None = None,
           measured:dict[str, Any] | None = None, into:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """Every row of the engine on this build and these facts: {name, kind, group, status (set | default | not_applicable |
  not_in_build), value, default, source (derived | measured | default), reason, provenance}. measured: --measure's
  result per knob name ({value, ...} or {status: failed}), which overrides the facts value. into: given the facts as the
  rows leave them (each row's sets written), what a later row's sweep reads."""
  tab, measured, f = tab or table(), measured or {}, dict(f)
  f = {**after_load(f, engine, tab), **f}
  allowed = fact_names(engine, tab)
  rows = []
  for k in knobs(engine, tab):
    default = f.get(k["default_from"]) if k.get("default_from") else k["default"]  # a default the build states is read from it
    row = {"name": k["name"], "kind": k["kind"], "group": k.get("group"), "default": default, "controls": k["controls"],
           "bitwise": k.get("bitwise", True), "with": k.get("with"), "with_draft": k.get("with_draft")}
    lacks = [r for r in k.get("requires", []) if r not in features["features"]]
    if lacks:
      rows.append({**row, "status": "not_in_build", "value": None, "source": None, "reason": None,
                   "provenance": {"build": features["source"], "lacks": lacks,
                                  **({"group": tab["engines"][engine]["groups"][k["group"]]["mod"]} if k.get("group") else {})}})
      continue
    unread = None
    try:
      applies = evaluate(k.get("applies", "True"), f, allowed)
      if applies and k.get("terms"):
        f = {**f, **terms(k, f, allowed)}  # dropped again below: a row's terms are its own
    except Missing as m:
      applies, unread = False, str(m)
    if not applies:
      reason = _missing(unread, adapter(engine, tab).FACTS) if unread else say(k.get("not_applicable", ""), f)
      rows.append({**row, "status": "not_applicable", "value": None, "source": None, "reason": reason,
                   "provenance": {"applies": k.get("applies"), "inputs": {n: _shown(f.get(n)) for n in names_in(k.get("applies", "True"))}}})
      continue
    ok = allowed | term_names(k)
    text = lambda field, _f=f, _ok=ok: say(case(field, _f, _ok), _f)  # noqa: E731
    value, source, reason = None, "default", text(k.get("reason_default") or k.get("reason", ""))
    prov: dict[str, Any] = {"evidence": k.get("evidence"), "features": {r: features["features"][r] for r in k.get("requires", [])}}
    if k.get("terms"):
      prov["terms"] = {n: {"formula": e, "value": _shown(f.get(f"t.{n}"))} for n, e in k["terms"].items()}
    if k["derive"] in ("facts", "measured") and k.get("formula"):
      inputs = names_in(k["formula"])
      prov.update(formula=k["formula"], inputs={n: {"value": _shown(f.get(n)), "source": f["sources"].get(n)} for n in inputs})
      try:
        value = evaluate(k["formula"], f, ok)
        source = "derived" if value is not None else "default"  # a formula that keeps the default derived nothing
        reason = text(k["reason"] if value is not None else k.get("reason_default") or k["reason"])
      except Missing as m:
        prov["missing"] = str(m)
        if k.get("reason_missing"):
          reason = text(k["reason_missing"])
    if k.get("measured") and _holds(k.get("measured_when"), f, ok):  # measured numbers only where their setup is this one
      reason += ("" if reason.endswith(".") else ".") + " Measured: " + say(k["measured"], f)
    got = measured.get(k["name"])
    if got and got.get("fit"):  # the memory fit chose it with the other rows that take GPU memory (default: None)
      value, source, reason = got["value"], got["source"], got["says"]
      prov["fit"] = got.get("record")
    elif got and got.get("value") is not None:
      value, source = got["value"], "measured"
      reason = got["says"]
      prov["sweep"] = got
    elif got:
      prov["sweep"] = got
      reason += f" (--measure: {got.get('says') or got.get('why')})"
    if k["derive"] == "measured" and not got and not held(k, f):
      s = k["sweep"]
      over = say("{x}", {"x": _try(s["candidates"], f, ok)}) if s["type"] == "choice" else s.get("points")  # a choice's points are this build's and machine's
      prov["measure"] = (f"confirm with --measure: {s['what']}" if s["type"] == "fit" else f"confirm with --measure: {s['type']} over {s['axis']} {over}")
    f = {n: v for n, v in f.items() if not n.startswith("t.")}
    was = dict(f)
    for name, expr in (k.get("sets") or {}).items():
      try:
        f[name] = evaluate(expr, {**f, "value": value}, allowed)
      except Missing:
        f[name] = None
    if any(f[n] != was.get(n) for n in (k.get("sets") or {}) if n in after_reads(engine, tab)):  # the estimate follows (a chunk row)
      f |= after_load(f, engine, tab)
    shown = str(value) if value is not None else default
    if isinstance(row["bitwise"], str):  # whether outputs stay bitwise identical can hang on the value (only at the default, say)
      row["bitwise"] = bool(evaluate(row["bitwise"], {**f, "value": value}, allowed))
    row["output"] = output_of(k, value, f, allowed)["output"]
    rows.append({**row, "status": "set" if value is not None and shown != default else "default", "value": shown, "source": source,
                 "reason": reason, "provenance": prov})
  for r, k in zip(rows, knobs(engine, tab)):  # a value measured with other rows at their defaults says when this setup moves them
    moved = [f"{x['name']} {x['value']}" for x in rows if x["name"] in (k.get("measured_with") or ()) and x["status"] == "set"]
    if moved and r["status"] == "set" and r["source"] != "measured":  # measured here, it ran with them
      r["reason"] += _words()["tune"]["measured_with"].format(knobs=", ".join(k["measured_with"]), moved=", ".join(moved))
  if into is not None:
    into.update(f)
  return rows


def _try(expr:str, f:dict[str, Any], allowed:set[str]) -> Any:
  """A formula's value, None when it reads an unknown fact."""
  try:
    return evaluate(expr, f, allowed)
  except Missing:
    return None


def _words() -> dict[str, Any]:
  from boltbeam.report.plain import words
  return words()


def fact_words(fact:str, engine_facts:dict[str, str] | None = None) -> str:
  """A fact as a reader knows it: what it is (FACTS, or the adapter's), with no code names or module paths."""
  import re
  what = FACTS.get(fact) or (engine_facts or {}).get(fact.rpartition(".")[2]) or fact.rpartition(".")[2].replace("_", " ")
  return re.sub(r"\s*\([^()]*[/.][^()]*\)", "", what).strip()


def _missing(fact:str, engine_facts:dict[str, str] | None = None) -> str:
  return _words()["tune"]["missing"].format(fact=fact_words(fact, engine_facts))


def values_of(rows:list[dict[str, Any]], **over:Any) -> list[dict[str, Any]]:
  """The rows set away from their defaults as {name, kind, value, with}, with `over` (name -> value) replacing or adding
  (None: back to its default, left off the command)."""
  kinds = {r["name"]: r for r in rows}
  out = {r["name"]: r["value"] for r in rows if r["status"] == "set"}
  out.update({k: str(v) for k, v in over.items()})
  out = {k: v for k, v in out.items() if over.get(k, "") is not None}
  return [{"name": n, "kind": kinds[n]["kind"], "value": v, "with": kinds[n].get("with"), "with_draft": kinds[n].get("with_draft")} for n, v in out.items()]


# --- the memory fit: the rows that take GPU memory, chosen together from one budget ------------------------------------------

def memory_knobs(engine:str, tab:dict[str, Any], rows:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """The rows that declare the GPU memory they take (vram) and are on this build and setup: the memory fit's rows."""
  st = {r["name"]: r["status"] for r in rows}
  return [k for k in knobs(engine, tab) if k.get("vram") and st.get(k["name"]) in ("set", "default")]


def held(k:dict[str, Any], f:dict[str, Any]) -> bool:
  """A row the setup holds (its held_by fact is known: the Recommended setup's own value): neither the memory fit nor
  --measure moves it, so this page's command keeps the setup's value."""
  return bool(k.get("held_by")) and f.get(k["held_by"]) is not None


def candidates(k:dict[str, Any], row:dict[str, Any], f:dict[str, Any], allowed:set[str]) -> list[str | None]:
  """A memory row's values for the fit, the setup's own first (None: the default): its fit candidates (a list, or a
  formula over the facts), less those its sweep's keep rule rejects; a candidate equal to the default is the default."""
  got, keep, ok = (k.get("fit") or {}).get("candidates"), (k.get("sweep") or {}).get("keep"), allowed | term_names(k) | {"point"}
  out = [row["value"] if row["status"] == "set" else None]
  for c in (got if isinstance(got, list) else _try(got, f, ok) or []) if got else []:
    c = None if c is None or str(c) == str(row["default"]) else str(c)
    if c not in out and (c is None or not keep or _holds(keep, {**f, "point": c}, ok)):
      out.append(c)
  return out


def fit_facts(ks:list[dict[str, Any]], combo:dict[str, Any], f:dict[str, Any], allowed:set[str], at:tuple | None = None) -> dict[str, Any]:
  """The facts as the memory rows' sets leave them at these values; at (engine, table): where a set moved a fact, the
  engine's after-load estimate and its terms follow (a row's sets may move a fact the estimate reads, a chunk row the chunk)."""
  g = dict(f)
  ks = [k for k in ks if k["name"] in combo]  # the rows the fit chose among (memory_fit leaves out a row with one value)
  for k, (name, expr) in ((k, x) for k in ks for x in (k.get("sets") or {}).items()):
    g[name] = _try(expr, {**g, "value": combo[k["name"]]}, allowed)
  moved = {n for k in ks for n in k.get("sets") or {} if g[n] != f.get(n)}
  return {**g, **after_load(g, *at)} if at and moved & after_reads(*at) else g


def costs(ks:list[dict[str, Any]], combo:dict[str, Any], f:dict[str, Any], allowed:set[str], at:tuple | None = None) -> dict[str, float]:
  """What each memory row takes of the GPU at these values (its vram over its terms); Missing when one is unknown."""
  g = fit_facts(ks, combo, f, allowed, at)
  return {k["name"]: evaluate(k["vram"], {**g, **terms(k, {**g, "value": combo[k["name"]]}, allowed), "value": combo[k["name"]]}, allowed | term_names(k))
          for k in ks if k["name"] in combo}


def budget(ks:list[dict[str, Any]], ref:dict[str, Any], f:dict[str, Any], allowed:set[str], mem:dict[str, Any] | None = None,
           at:tuple | None = None) -> dict[str, Any]:
  """The GPU memory the memory rows may take together: what is free after load (the engine's estimate with the rows at
  their defaults, or `mem`: read in a run of the setup at `ref`), plus what the rows take there, less a margin for what a
  session adds after load: the fall below the after-load reading measured over the run's replies, else SESSION_MARGIN
  (an estimate). Missing when a figure is unknown."""
  after = mem["loaded"] if mem else evaluate("gpu_after_load_bytes", fit_facts(ks, ref, f, allowed, at))  # the estimate at ref's values
  m = session_margin(mem, f.get("session_fall"))
  held, margin, growth = sum(costs(ks, ref, f, allowed, at).values()), m["value"], m["growth"]
  return {"bytes": after + held - margin, "after_load": after, "held": held, "margin": margin, "growth": growth, "measured": bool(mem), "ref": m.get("ref")}


def session_margin(mem:dict[str, Any] | None, fall:dict[str, Any] | None = None) -> dict[str, Any]:
  """The one margin rule for what a session adds after load: the fall from the free memory at load to the lowest free
  over a run of the setup (mem {loaded, lowest}, bytes: measured), else the largest such fall measured over load probes
  of this model at this context (fall {bytes, where}: spend.probe_fall), else SESSION_MARGIN (an estimate)."""
  if mem:
    g = max(0, mem["loaded"] - mem["lowest"])
    return {"value": g, "growth": g, "label": "measured", "why": "the fall from free at load to the lowest free over a run of the setup"}
  if fall:
    return {"value": fall["bytes"], "growth": fall["bytes"], "label": "measured", "why": fall["where"], "ref": fall["where"]}
  return {"value": SESSION_MARGIN, "growth": None, "label": "estimate", "why": "SESSION_MARGIN: no run of the setup measured; test: one session's device samples"}


def _bitwise(k:dict[str, Any], value:Any, f:dict[str, Any], allowed:set[str]) -> bool:
  return bool(evaluate(b, {**f, "value": value}, allowed)) if isinstance(b := k.get("bitwise", True), str) else bool(b)


def gated(k:dict[str, Any], f:dict[str, Any] | None) -> dict[str, str] | None:
  """A row's gate evidence for this setup: its `gated` rows ({when: facts it ran on, result, evidence}) matched on the
  facts given (model and target ids). quality-gated where one passed, gate failed where one failed (the prefill chunk's
  labels, roofline/chunk_size.output_label); None where no gate ran here."""
  hit = next((g for g in k.get("gated") or [] if f and all(f.get(x) == v for x, v in g["when"].items())), None)
  return {"output": "quality-gated" if hit["result"] == "passed" else "gate failed", "why": hit["evidence"]} if hit else None


def output_of(k:dict[str, Any] | None, value:Any = None, f:dict[str, Any] | None = None, allowed:set[str] | None = None) -> dict[str, str]:
  """A row's output label at a value (data/engine_knobs.json knobs and audit_flags, an arkey_ie_mods switch): identical
  where the row states `bitwise` and it holds there; else its gate evidence for these facts (gated); else the row's own `output`. The one statement of the default: a
  row that says neither, or no row, is untested (the bitwise default above is tune's own gating, not a label)."""
  k = k or {}
  try:
    same = "bitwise" in k and _bitwise(k, value, f or {}, allowed or {"value"})
  except Missing:  # a fact the rule needs is unknown: not shown identical
    same = False
  if same:
    return {"output": "identical", "why": "its row: bitwise at this value"}
  if gate := gated(k, f):
    return gate
  if (src := k.get("output_from")) and (f or {}).get(src) and value is not None and str(value) == str((f or {}).get(src.rsplit("_output", 1)[0])):
    return {"output": f[src], "why": f.get(src + "_why") or k.get("output_why", "")}  # a derived value's own label (its evidence), at that value only
  return {"output": k.get("output", "untested"), "why": k.get("output_why", "its row states no output" if k else "no data row names this setting")}


def named(r:dict[str, Any], value:Any, w:dict[str, Any]) -> str:
  """A row at a value (None: its default) as a reader reads it: name=value, name value, or a switch's bare flag (or not passed)."""
  val = next((v for v in (value, r.get("default"), w["unset"]) if v is not None))
  return (f"{r['name']}={val}" if r["kind"] in ("env", "config") else (r["name"] if val == "on" else w["switch_off"].format(name=r["name"]))
          if r["kind"] == "switch" else f"{r['name']} {val}")


def memory_fit(rows:list[dict[str, Any]], f:dict[str, Any], engine:str, tab:dict[str, Any], *, mem:dict[str, Any] | None = None,
               checked:bool = False, only:list[str] | None = None) -> dict[str, Any] | None:
  """The memory rows' values chosen together: every combination of their candidates, each costed (their vram summed),
  kept when the sum is within the budget, ranked by the product of the rows' predicted gains (fit.gain; a row without
  one counts 1), ties to the setup's own values. Unchecked (no --measure), a combination that changes the answers (a row
  not bitwise at its value) is not taken: only the output check can let it count. One row alone is checked the same way.
  only (--knob): the rows it may move; the others keep their values (kept). A combination whose cost is unknown is
  named with why (uncosted), never chosen. None with no memory row; {why} when the budget is unknown (each row keeps its
  own value). Engine-blind: it sums what the rows declare against one budget, and ranks."""
  import itertools
  allowed, by = fact_names(engine, tab), {r["name"]: r for r in rows}
  ks = memory_knobs(engine, tab, rows)
  cands = {k["name"]: candidates(k, by[k["name"]], f, allowed) for k in ks}
  # a row with no other value whose cost is all the engine's own estimate (its terms) is in the budget already: not chosen
  inside = after_reads(engine, tab) | {f"load.{t}" for t in ((tab["engines"][engine].get("after_load") or {}).get("terms") or {})}
  ks = [k for k in ks if len(cands[k["name"]]) > 1 or not set(names_in(k["vram"])) <= inside]
  cands = {k["name"]: cands[k["name"]] for k in ks}
  kept = [n for n in cands if only and n not in only and len(cands[n]) > 1]
  holds = {k["name"] for k in ks if held(k, f)}  # its value is the setup's: kept, and judged by its own label, not the fit's bitwise rule
  cands = {n: c[:1] if n in kept or n in holds else c for n, c in cands.items()}
  if not ks:
    return None
  here = {n: c[0] for n, c in cands.items()}
  try:
    # the estimate is after load with the rows at their defaults, a held row at the setup's value (the engine's after_load sizes it so)
    b = budget(ks, here if mem else {n: here[n] if n in holds else None for n in cands}, f, allowed, mem, (engine, tab))
  except Missing as m:
    return {"rows": list(cands), "why": fact_words(str(m), adapter(engine, tab).FACTS)}
  out, uncosted = [], []

  def label(combo:dict[str, Any]) -> str:
    return ", ".join(named(by[n], v, _words()["tune"]) for n, v in combo.items())
  for picks in itertools.product(*(list(enumerate(c)) for c in cands.values())):
    combo = {n: v for n, (_, v) in zip(cands, picks)}
    try:
      cost = costs(ks, combo, f, allowed, (engine, tab))
    except Missing as m:  # a value whose memory cannot be known is never chosen, and said
      uncosted.append((label(combo), fact_words(str(m), adapter(engine, tab).FACTS)))
      continue
    g = fit_facts(ks, combo, f, allowed, (engine, tab))
    said = [(_try((k.get("fit") or {}).get("gain") or "None", {**g, "value": combo[k["name"]]}, allowed),
             _try((k.get("fit") or {}).get("gain_words") or "None", {**g, "value": combo[k["name"]]}, allowed)) for k in ks]
    gain, how = math.prod(x for x, _ in said if x is not None), [h for x, h in said if x is not None and h]
    out.append({"values": combo, "order": tuple(i for i, _ in picks), "bytes": sum(cost.values()), "costs": cost, "gain": gain, "how": how,
                "bitwise": all(_bitwise(k, combo[k["name"]], g, allowed) for k in ks if k["name"] not in holds), "fits": sum(cost.values()) <= b["bytes"],
                "label": label(combo)})
  out.sort(key=lambda c: (-c["gain"], c["order"]))
  pick = next((c for c in out if c["fits"] and (c["bitwise"] or checked)), None)
  # none fits: the one that takes the least (and keeps the answers) rather than the setup's own, so a value over the budget is not shown when a smaller one is there
  least = min((c for c in out if c["bitwise"] or checked), key=lambda c: (c["bytes"], c["order"]), default=None) if pick is None else None
  return {"rows": list(cands), "budget": b, "here": here, "pick": pick, "combos": out, "uncosted": uncosted, "kept": kept,
          "choices": len(out) + len(uncosted) > 1, "least": least if least and least["values"] != here else None}


def fit_words(plan:dict[str, Any] | None) -> str | None:
  """The memory fit in words: the rows, the budget and where it came from, the choice and its gain, and the faster
  predicted combinations that did not fit or that only the output check could let count."""
  w, rows = _words()["tune"]["memory"], ", ".join((plan or {}).get("rows") or [])
  one = "_one" if len((plan or {}).get("rows") or []) == 1 else ""
  if not plan or "why" in plan:
    return plan and w["unknown" + one].format(rows=rows, what=plan["why"])
  b, pick = plan["budget"], plan["pick"]
  ref = next((c["label"] for c in plan["combos"] if c["values"] == plan["here"]), "the setup")
  margin = (w["margin_estimate"] if b["growth"] is None else w["margin_probes"] if b.get("ref") else w["margin_measured"]).format(growth=(b["growth"] or 0) / GiB, ref=b.get("ref") or ref)
  said = w["lead" + one].format(rows=rows) + w["budget"].format( after=b["after_load"] / GiB, how=w["measured"].format(ref=ref) if b["measured"] else w["estimated"], held=b["held"] / GiB,
                            margin=b["margin"] / GiB, margin_how=margin, room=b["bytes"] / GiB)
  why: dict[str, list[str]] = {}
  for lab, what in plan.get("uncosted") or []:
    why.setdefault(what, []).append(lab)
  tail = "".join(w["uncosted"].format(n=len(labs), items="; ".join(labs[:3]), what=what) for what, labs in why.items()) + (w["kept"].format(rows=", ".join(plan["kept"])) if plan.get("kept") else "")
  if pick is None:
    least = plan.get("least")
    return said + (w["none_least"].format(label=least["label"], gib=least["bytes"] / GiB) if least else w["none" + one]) + tail
  swept = (plan.get("as") or {}).get("says", "")
  said += w["pick"].format(label=pick["label"], gib=pick["bytes"] / GiB) + (w["gain"].format(gain=pick["gain"], how="; ".join(pick["how"])) if pick["how"] else w["no_gain"]) + \
    (w["failed"].format(why=plan["failed"]) if plan.get("failed") else "") + (w["swept"].format(says=swept[:1].upper() + swept[1:].rstrip(".")) if swept else "")
  better = [c for c in plan["combos"] if c["gain"] > pick["gain"]]
  over = [w["item"].format(label=c["label"], gib=c["bytes"] / GiB) for c in better if not c["fits"]][:4]
  unchecked = [c["label"] for c in better if c["fits"] and not c["bitwise"]][:3]
  return said + (w["over"].format(items="; ".join(over)) if over else "") + (w["unchecked"].format(items="; ".join(unchecked)) if unchecked else "") + tail


def fit_overrides(plan:dict[str, Any] | None, rows:list[dict[str, Any]], source:str = "derived", says:str | None = None) -> dict[str, Any]:
  """The rows the memory fit moves, as derive's overrides: {name: {fit, value, source, says, record}}."""
  pick, w, own = (plan or {}).get("pick"), _words()["tune"]["memory"], {r["name"]: r["value"] if r["status"] == "set" else None for r in rows}
  one = "_one" if plan and len(plan["rows"]) == 1 else ""
  why = pick and (w["row_measured" + one].format(label=pick["label"]) if says else w["row" + one].format(label=pick["label"], room=plan["budget"]["bytes"] / GiB, gain=pick["gain"]))
  if pick is None and (pick := (plan or {}).get("least")):
    why = w["row_least"].format(label=pick["label"], room=plan["budget"]["bytes"] / GiB)
  return {n: {"fit": True, "value": v, "source": source, "says": why, "record": {"label": pick["label"], "bytes": pick["bytes"], "gain": pick["gain"]}}
          for n, v in (pick or {}).get("values", {}).items() if v != own[n]}


def settle(f:dict[str, Any], feats:dict[str, Any], engine:str, tab:dict[str, Any], measured:dict[str, Any] | None = None,
           into:dict[str, Any] | None = None, plan:dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
  """Every row (derive), then the memory fit over the rows that take GPU memory (plan: one already made, measured);
  the rows it moves derived again with its values, so the facts their sets write follow. -> (rows, plan)."""
  into = {} if into is None else into
  rows = derive(f, feats, engine, tab, measured, into=into)
  plan = plan or memory_fit(rows, into, engine, tab)
  if over := fit_overrides(plan, rows, **(plan or {}).get("as", {})):
    rows = derive(f, feats, engine, tab, {**(measured or {}), **over}, into=into)
  return rows, plan


# --- --measure: sweeps over a knob's axis --------------------------------------------------------------------------------

def fit(knob:dict[str, Any], run:Callable[..., Any], rows:list[dict[str, Any]], *, prompt:int, facts:dict[str, Any],
        allowed:set[str]) -> dict[str, Any]:
  """A fit sweep: what a resource holds, read from the running engine rather than estimated. One probe point at the
  row's probe value, which reports the engine's measured facts (run(..., facts=True) -> {ms, facts}); the row's sweep
  formula sizes the value from them (value is the probe's); then the value runs, and while the run reports the sweep's
  `failed` fact above 0, the row's backoff formula lowers it, at most `tries` runs. The answer is the first value that ran
  clean (None when none did: the facts value or the default stands, said so)."""
  s, name = knob["sweep"], knob["name"]
  reps = s["reps"]
  prompt_n = prompt if s.get("prompt") == "prompt_tokens" else int(s.get("prompt") or prompt)
  ok = allowed | term_names(knob)
  base = {**facts, **terms(knob, facts, allowed)}
  got = run(values=values_of(rows, **{name: s["probe"]}), prompt=prompt_n, reps=reps, tag=f"{name}-probe-{s['probe']}", facts=True)
  base = {**base, **got["facts"]}
  base = {**base, **terms(knob, base, allowed), "value": int(s["probe"])}
  out = [{"point": int(s["probe"]), "probe": True, "ms": round(got["ms"], 1), "tok_s": round(prompt_n * 1000 / got["ms"], 1), "facts": got["facts"]}]
  try:
    value = int(evaluate(s["formula"], base, ok))
  except Missing as m:
    return {"value": None, "points": out, "prompt": prompt_n, "stop": s["stop"], "says": f"the probe did not report {fact_words(str(m))}"}
  first, answer = value, None
  for _ in range(s["tries"]):
    try:
      got = run(values=values_of(rows, **{name: value}), prompt=prompt_n, reps=reps, tag=f"{name}-{value}", facts=True)
      failed = got["facts"].get(s["failed"]) or 0
      out.append({"point": value, "ms": round(got["ms"], 1), "tok_s": round(prompt_n * 1000 / got["ms"], 1), "failed": failed, "facts": got["facts"]})
    except Exception as exc:  # noqa: BLE001  an engine that did not start or answer at this value counts as a failed run, and backs off
      failed = "a"
      out.append({"point": value, "ms": None, "tok_s": None, "failed": failed, "detail": f"{type(exc).__name__}: {str(exc)[-DETAIL_CHARS:]}"})
    if not failed:
      answer = value
      break
    lower = int(evaluate(s["backoff"], {**base, "value": value}, ok))
    if lower >= value:
      break
    value = lower
  runs = "; ".join(f"{r['point']}: {r['tok_s']:,.0f} tok/s, {r['failed']} {s['failed_words']}" if r["tok_s"] is not None else
                   f"{r['point']}: {s['run_failed_words']}" for r in out[1:])
  said = {**base, "measure.probe": out[0]["point"], "measure.first": first, "measure.value": answer, "measure.runs": runs}
  says = say(s["says"] if answer is not None else s["says_none"], said)
  return {"value": answer, "points": out, "prompt": prompt_n, "stop": s["stop"], "says": says}


def choice_points(knob:dict[str, Any], rows:list[dict[str, Any]], facts:dict[str, Any], allowed:set[str]) -> dict[str, Any]:
  """A choice sweep's points: the setup's own value (base: None when the row is at its default, which `here` names) and
  each candidate the row's `candidates` fact lists for this machine and build; a candidate the row's `keep` rule (over
  the facts and `point`) rejects is dropped, said, not run."""
  s, ok = knob["sweep"], allowed | term_names(knob) | {"point"}
  f = {**facts, **terms(knob, facts, allowed)}
  row = next(r for r in rows if r["name"] == knob["name"])
  base = row["value"] if row["status"] == "set" else None
  here = base if base is not None else row["default"]
  points, dropped = [base], []
  for c in (str(c) for c in _try(s["candidates"], f, ok) or []):
    if c != here and c not in points:
      (points if not s.get("keep") or _holds(s["keep"], {**f, "point": c}, ok) else dropped).append(c)
  return {"base": base, "here": here, "points": points, "dropped": dropped}


def choice(knob:dict[str, Any], run:Callable[..., Any], rows:list[dict[str, Any]], *, prompt:int, facts:dict[str, Any],
           allowed:set[str], seed:int | None = None, pts:dict[str, Any] | None = None, vals:Callable[[Any], list] | None = None,
           check:Callable[[Any], tuple | None] | None = None, first:dict[str, Any] | None = None,
           admit:Callable[[Any], str | None] | None = None) -> dict[str, Any]:
  """A choice sweep: decode speed at SERVED requests at a time (run(..., decode=True) -> {tokens, depths: {depth: {context, per_prompt}}},
  the engine's real-text prompts with their tasks, short and long, a reply each, held to the loop guard) of the setup's
  own value, run first and last, and of choice_points' candidates between, in a random order (seed, said). A candidate
  that does not run is said and left out. One rule, core/draft_count's: at every depth the paired bootstrap lower bound
  of its per-prompt speed over the setup's (the two runs averaged) is above 1 plus the setup's own drift between its two
  runs, at the level corrected for the candidates compared together (draft_count.corrected); of those, the one with the
  highest lower bound. A row with a gate checks the output too: the candidate against the setup's value, teacher-forced
  on the real-text long corpus at the setup's context (run(..., agree=setup values, texts=, context=) -> per-document
  figures), judged by Speed up's own gate: its agreement, the engine's own noise floor at the values it is checked against
  through the same function and cache (run(..., floor=True) -> quantization/speed_up.noise_floor, only for an engine that
  declares a perturbation; none: no floor) and the same bar() and words, at Speed up's own confidence; one that does not
  pass stage 1 would wait on a task eval in Speed up, and tune runs none, so it is reported, not picked. The memory fit
  runs its combinations through it: pts (its points), vals (a point's values), check (a point's output check: (values,
  against), or None when it needs none), first (the setup's first run) and admit (why a faster point's run is not kept
  before its check, or None)."""
  import random
  import re
  from boltbeam.collectors import mtp
  from boltbeam.core import draft_count as dc
  from boltbeam.quantization import speed_up
  s, name, w = knob["sweep"], knob["name"], _words()["tune"]["choice"]
  pts = pts or choice_points(knob, rows, facts, allowed)
  vals = vals or (lambda p: values_of(rows, **({name: p} if p is not None else {})))  # None: the setup as it is, its own value included
  pair = check or (lambda p: (vals(p), vals(pts["base"])) if s.get("gate") else None)
  seed = random.SystemRandom().getrandbits(SEED_BITS) if seed is None else seed
  order = random.Random(seed).sample(pts["points"][1:], len(pts["points"]) - 1)
  timed = lambda p, tag: run(values=vals(p), prompt=prompt, reps=1, tag=re.sub(r"[^\w.-]+", "_", f"{name}-{tag}"), decode=True)  # noqa: E731
  first, got, failed = first or timed(None, "setup"), {}, []
  for p in order:
    try:
      got[p] = timed(p, p)
    except Exception as exc:  # noqa: BLE001  one candidate that does not run is said, never the row's end
      failed.append({"point": p, "why": f"{type(exc).__name__}: {str(exc)[-DETAIL_CHARS:]}"})
  last = timed(None, "setup-again")
  depths, level = sorted(first["depths"]), dc.corrected(len(got))
  base = {d: {q: (first["depths"][d]["per_prompt"][q] + last["depths"][d]["per_prompt"][q]) / 2 for q in first["depths"][d]["per_prompt"]} for d in depths}
  drift = {d: abs(statistics.median(last["depths"][d]["per_prompt"][q] / first["depths"][d]["per_prompt"][q] for q in base[d]) - 1) for d in depths}
  out = [{"point": pts["here"], "runs": [first, last], "median": {d: statistics.median(base[d].values()) for d in depths}}]
  for p, r in got.items():
    if any(set(r["depths"][d]["per_prompt"]) != set(base[d]) for d in depths):
      raise Refused(f"{name} {p} did not run the setup's prompts: a speed is compared only on the same prompts")
    lo = {d: dc.bounds((r["depths"][d]["per_prompt"], None), (base[d], None), level)[0] for d in depths}
    out.append({"point": p, "runs": [r], "median": {d: statistics.median(r["depths"][d]["per_prompt"].values()) for d in depths}, "lower": lo,
                "beats": all(lo[d] > 1 + drift[d] for d in depths)})
  gates, answer, th, floors = [], None, speed_up.table()["threshold"], {}
  for r in sorted((r for r in out[1:] if r["beats"]), key=lambda r: -min(r["lower"].values())):
    if admit and (why := admit(r["runs"][0])):
      gates.append({"point": r["point"], "passed": False, "held": why})
      continue
    if two := pair(r["point"]):
      try:
        mtp.long_corpus()  # hash-checked
        texts = [d["text"] for d in json.loads(mtp.LONG.read_text())["documents"]]
        if blind := unchecked(knob["engine"], *two):
          raise Refused(f"{getattr(adapter(knob['engine']), 'CHECK_TOOL', 'the output check')} cannot see {', '.join(blind)}, so its effect on the answers cannot be checked")
        fig = speed_up.agreement(run(values=two[0], agree=two[1], texts=texts, context=facts.get("context_tokens"), prompt=prompt,
                                     reps=1, tag=re.sub(r"[^\w.-]+", "_", f"{name}-{r['point']}-agree")), texts)
        if (at := json.dumps(two[1], sort_keys=True)) not in floors and speed_up.perturbation(knob["engine"]):  # the engine's own noise at the values checked against: once each, cached across runs
          floors[at] = run(values=two[1], floor=True, texts=texts, context=facts.get("context_tokens"), prompt=prompt, reps=1, tag=f"{name}-floor")
        floor = floors.get(at)
        gate = speed_up.bar(fig, floor)  # Speed up's own bar at its own confidence: a candidate's speed rule does not widen it
      except (Refused, mtp.MtpRefused) as exc:
        gates.append({"point": r["point"], "passed": False, "why": str(exc)})
        continue
      short = (fig.get("ctx") or 0) < (facts.get("context_tokens") or 0)  # a check at a shorter context says nothing past it
      said = {"gate": gate, "noise_floor": floor, "accuracy": fig, "engine": knob["engine"]}  # read as Speed up's record is
      gates.append({"point": r["point"], "agrees": gate["verdict"] == "pass", "passed": gate["verdict"] == "pass" and not short, "agreement": fig[th["metric"]], "interval": fig["interval"],
                    "figures": fig, "gate": gate, "said": " ".join(speed_up.floor_words(said, r["point"])),
                    **({"short": True} if short else {})})
      if not gates[-1]["passed"]:
        continue
    answer = r["point"]
    break
  where = ", ".join(w["depth"].format(depth=d, context=first["depths"][d]["context"], n=len(base[d]), drift=drift[d]) for d in depths)
  runs = "; ".join(w["run"].format(point=r["point"], speeds=", ".join(f"{r['median'][d]:,.1f}" for d in depths)) +
                   ("" if "lower" not in r else w["lower"].format(lower=", ".join(f"{r['lower'][d] - 1:+.1%}" for d in depths))) for r in out)
  setup = w["setup"].format(tokens=first["tokens"], served=facts["served_requests"], where=where, here=pts["here"], seed=seed, level=level, runs=runs)
  says = w["picked" if answer is not None else "none"].format(setup=setup, value=answer, here=pts["here"])
  for g in gates:
    says += (w["held"].format(point=g["point"], why=g["held"]) if "held" in g else w["gate_failed"].format(point=g["point"], here=pts["here"], why=g["why"]) if "why" in g else
             w["gate_passed" if g["agrees"] else "gate_below"].format(point=g["point"], here=pts["here"], agreement=g["agreement"], lo=g["interval"][0],
                                                                       hi=g["interval"][1], said=g["said"], context=g["figures"]["ctx"],
                                                                       scored=g["figures"]["scored"], docs=len(g["figures"]["per_doc"])))
    if g.get("short"):
      says += w["gate_short"].format(point=g["point"], checked=g["figures"]["ctx"], context=facts.get("context_tokens"))
  for x in failed:
    says += w["failed"].format(point=x["point"])
  if pts["dropped"]:
    says += w["dropped"].format(points=", ".join(pts["dropped"]), why=say(s["keep_words"], facts))
  return {"value": answer, "points": out, "gates": gates, "failed": failed, "dropped": pts["dropped"], "seed": seed, "level": level, "drift": drift,
          "stop": s["stop"], "says": says}


def fit_measure(plan:dict[str, Any], rows:list[dict[str, Any]], run:Callable[..., Any], f:dict[str, Any], engine:str, tab:dict[str, Any], *,
                prompt:int, seed:int | None = None, only:list[str] | None = None) -> dict[str, Any]:
  """--measure's memory fit: the setup (the unchecked fit's choice) runs first and last on choice's real-text replies, its
  GPU memory read right after load and at its lowest while it writes (run(..., decode=True, facts=True)), and the budget
  is set again from those readings; the CONFIRM fastest predicted combinations within it run between, kept by choice's
  paired rule. One that changes the answers counts only past the output check, of its rows that change them against
  the setup's values, with the rows the check cannot see (output_check false) left out of both sides. -> the plan, its
  pick the measured winner (else the setup), with the sweep's words and record."""
  allowed, ks = fact_names(engine, tab), memory_knobs(engine, tab, rows)

  def read(**kw):  # every run reads the GPU's memory: a combination is kept only on its own reading too
    return run(**kw, facts=True)
  first = read(values=values_of(rows), prompt=prompt, reps=1, tag="memory-fit-setup", decode=True)
  m = first.get("facts") or {}
  mem = {"loaded": m["gpu_free_after_load_bytes"], "lowest": m["gpu_free_lowest_bytes"]} if None not in (m.get("gpu_free_after_load_bytes"), m.get("gpu_free_lowest_bytes")) else None
  plan = memory_fit(rows, f, engine, tab, mem=mem, checked=True, only=only) or plan
  if "why" in plan:
    return plan
  here, by = plan["here"], {k["name"]: k for k in ks}
  top = [c for c in plan["combos"] if c["fits"] and c["values"] != here][:CONFIRM]
  combos, blind = {c["label"]: c["values"] for c in top}, {k["name"] for k in knobs(engine, tab) if k.get("output_check") is False}
  setup = next(c for c in plan["combos"] if c["values"] == here)
  if not top:  # nothing else fits the measured budget: nothing to confirm
    return {**plan, "pick": setup, "as": {"source": "measured", "says": _words()["tune"]["memory"]["nothing"]}}

  def check(p):
    c = combos[p]
    moved = [n for n in c if c[n] != here[n] and not _bitwise(by[n], c[n], fit_facts(ks, c, f, allowed), allowed) and n not in blind]
    return ([v for v in values_of(rows, **c) if v["name"] not in blind],
            [v for v in values_of(rows, **{**c, **{n: here[n] for n in moved}}) if v["name"] not in blind]) if moved else None
  def admit(got):  # its own run read: a faster one is kept only when its own memory was read and stayed above zero
    lowest = (got.get("facts") or {}).get("gpu_free_lowest_bytes")
    return _words()["tune"]["memory"]["unread"] if lowest is None else None if lowest > 0 else _words()["tune"]["memory"]["own"]
  got = choice({"name": "memory fit", "engine": engine, "sweep": {"stop": _words()["tune"]["memory"]["stop"]}}, read, rows, prompt=prompt, facts=f,
               allowed=allowed, seed=seed, pts={"base": None, "here": setup["label"], "points": [None, *combos], "dropped": []},
               vals=lambda p: values_of(rows, **(combos[p] if p is not None else {})), check=check, first=first, admit=admit)
  win = next((c for c in top if c["label"] == got["value"]), setup)  # measured, the setup fits: its own fall is the margin
  return {**plan, "pick": win, "sweep": {k: v for k, v in got.items() if k != "points"}, "as": {"source": "measured", "says": got["says"]}}


def sweep(knob:dict[str, Any], run:Callable[..., float], rows:list[dict[str, Any]], *, prompt:int, facts:dict[str, Any] | None = None,
          allowed:set[str] | None = None) -> dict[str, Any]:
  """A row's sweep. crossover: per point of the axis knob (ascending) the knob's on and off arms, alternating order; the
  answer is the first point from which on beats off by more than the tie at two points in a row (or at the last point).
  best: one arm per point, the smallest point within the tie of the fastest; it stops after two points in a row slower
  than the best by more than the tie. run(values=, prompt=, reps=, tag=) -> wall ms; values are the setup's with the
  axis (and the arm) replaced."""
  s = knob["sweep"]
  if s["type"] == "choice":
    return choice(knob, run, rows, prompt=prompt, facts=facts or {}, allowed=allowed if allowed is not None else set(FACTS) | {k for k in facts or {} if "." in k})
  if s["type"] == "fit":
    return fit(knob, run, rows, prompt=prompt, facts=facts or {}, allowed=allowed if allowed is not None else set(FACTS) | {k for k in facts or {} if "." in k})
  tie, reps = s["tie_pct"] / 100, s["reps"]
  prompt_n = prompt if s.get("prompt") == "prompt_tokens" else int(s.get("prompt") or prompt)
  pts = [p for p in s["points"] if p < prompt_n] + [p for p in s["points"] if p >= prompt_n][:1]  # a chunk past the prompt is the prompt
  out, answer = [], None
  if s["type"] == "crossover":
    wins = 0
    for i, p in enumerate(pts):
      arms = {"on": s["on"].format(point=p), "off": s["off"]}
      ms = {}
      for arm in (("off", "on") if i % 2 == 0 else ("on", "off")):
        ms[arm] = run(values=values_of(rows, **{s["axis"]: p, knob["name"]: arms[arm]}), prompt=prompt_n, reps=reps, tag=f"{knob['name']}-{p}-{arm}")
      win = ms["on"] < ms["off"] * (1 - tie)
      out.append({"point": p, "on_ms": round(ms["on"], 1), "off_ms": round(ms["off"], 1), "on_wins": win})
      wins = wins + 1 if win else 0
      if wins == 2:
        answer = pts[i - 1]
        break
    if answer is None and out and out[-1]["on_wins"]:
      answer = out[-1]["point"]
    low = " (or lower: it already wins at the smallest point swept)" if answer is not None and answer == pts[0] else ""
    says = (f"measured crossover: on beats off by more than {s['tie_pct']}% from {s['axis']} {answer}{low} ({prompt_n}-token real-text prompt, wall: "
            + ", ".join(f"{r['point']}: {r['on_ms']:.0f} vs {r['off_ms']:.0f} ms" for r in out) + ")"
            if answer is not None else f"measured: on never beat off by more than {s['tie_pct']}% up to {s['axis']} {pts[-1] if pts else '-'}; "
            "the default is not confirmed (the crossover may be higher)")
  else:
    best = None
    worse = 0
    for p in pts:
      ms = run(values=values_of(rows, **{s["axis"]: p}), prompt=prompt_n, reps=reps, tag=f"{knob['name']}-{p}")
      out.append({"point": p, "ms": round(ms, 1), "tok_s": round(prompt_n * 1000 / ms, 1)})
      best = ms if best is None else min(best, ms)
      worse = worse + 1 if ms > best * (1 + tie) else 0
      if worse == 2:
        break
    answer = next(r["point"] for r in out if r["ms"] <= best * (1 + tie)) if out else None
    if answer is None:
      return {"value": None, "points": [], "prompt": prompt_n, "tie_pct": s["tie_pct"], "stop": s["stop"], "says": f"no {s['axis']} point to sweep"}
    says = f"measured: {s['axis']} {answer} is the smallest within {s['tie_pct']}% of the fastest ({prompt_n}-token real-text prompt, wall: " + \
           ", ".join(f"{r['point']}: {r['tok_s']:.0f} tok/s" for r in out) + ")"
  return {"value": answer, "points": out, "prompt": prompt_n, "tie_pct": s["tie_pct"], "stop": s["stop"], "says": says}


def fall_of(engine:str, tab:dict[str, Any], model:str, context:int | None, paths:list[str], ad, f:dict[str, Any]) -> dict[str, Any] | None:
  """What a session takes below the free memory at load, from the load probes of this model at this context and how much
  lower a full session went than a probe at the same setup (spend.probe_fall, spend.probe_gap)."""
  from boltbeam.workflow import spend as sp
  rows, recs = sp.rows_of(engine, tab), sp.read_records(paths)
  try:
    gap = sp.plan_gap(rows, recs, model, f, ad, fact_names(engine, tab))
  except Exception:  # noqa: BLE001  a setup's facts unread: the probes' own fall, the gap unknown
    gap = 0.0
  return sp.probe_fall(recs, model, context, rows, gap)


def kv_of(engine:str, tab:dict[str, Any], model:str, paths:list[str]) -> dict[str, Any]:
  """The KV type the measured runs of this model held their context at (the spend rows' argv fact spend.kv_type), as the
  adapter's options (kv, kv_from, words); several types: none taken, a note asks for one (--kv); none set: {}."""
  from boltbeam.workflow import spend as sp
  got = sp.recorded(sp.rows_of(engine, tab), sp.read_records(paths), model, "spend.kv_type")
  w = _words()["tune"]
  if len(got) == 1:
    (kv, where), = got.items()
    return {"kv": kv, "kv_from": w["kv_measured"].format(where=", ".join(where))}
  return {"kv_ask": w["kv_ask"].format(types="; ".join(f"{k} ({', '.join(v)})" for k, v in got.items()))} if got else {}


def tune(model:str, engine:str, build:str | None = None, *, prompt:int = PROMPT, context:int | None = None, measure:bool = False,
         knobs_only:list[str] | None = None, port:int = 0, out:pathlib.Path | None = None, extra:list[str] | None = None,
         experts_in_ram:str | None = None, run:Callable[..., float] | None = None, tab:dict[str, Any] | None = None,
         memory:dict[str, Any] | None = None, l3:dict[str, Any] | None = None, say:Callable[[str], None] | None = None,
         workload:str = "chat", link:dict[str, Any] | None = None, reply:dict[str, Any] | None = None, cpu:dict[str, Any] | None = None,
         spend:list[str] | None = None, spend_max:bool = False, **options) -> dict[str, Any]:
  """The setup: the build, the facts, every row, the rendered command, and (measure) the sweeps' records. A Refused when
  the engine, its build or the model cannot be read. options go to the adapter's facts (test stand-ins). say: --measure's
  progress as a screen's pipeline lines (a stage per swept knob, a progress line per point, at most the row's points)."""
  from boltbeam.quantization import speed_up
  tab = tab or table()
  ad = adapter(engine, tab)
  if knobs_only and not measure:  # --knob picks what --measure sweeps; alone it would be silently ignored
    raise Refused("--knob names the knobs --measure sweeps; without --measure nothing is swept, so it is refused. Add --measure, or drop --knob")
  if knobs_only:  # a flag's name may be given without its leading dashes
    names = {k["name"].lstrip("-"): k["name"] for k in knobs(engine, tab)}
    if unknown := sorted(q for q in knobs_only if q.lstrip("-") not in names):
      raise Refused(f"{engine} has no knob named {', '.join(unknown)} in BoltBeam's knob table")
    knobs_only = [names[q.lstrip("-")] for q in knobs_only]
  feats = ad.detect(build)
  from boltbeam.workflow import spend as sp
  if not spend and context is not None and sp.rows_of(engine, tab):  # stored measured runs of this model, this GPU's size and this build's switches
    spend = sp.defaults(model, getattr(ad, "gpu_total", lambda: None), set(feats["features"])) or None
  f = host_facts(memory=memory, l3=l3, prompt=prompt, context=context, workload=workload, link=link, reply=reply, cpu=cpu)
  if spend and options.get("kv") is None and (kv := kv_of(engine, tab, model, spend)):  # the measured runs' KV type, not the engine's default
    options = {**options, **kv}
  got = ad.facts(model, f, feats, {"experts_in_ram": experts_in_ram, "extra": extra, "spend": bool(spend), **options})
  f = {**f, **got, "model_file": pathlib.Path(model).name, "sources": {**f["sources"], **got.get("sources", {}), "model_file": "the model given"}}
  if spend and sp.rows_of(engine, tab) and (fall := fall_of(engine, tab, model, context, spend, ad, f)):  # what a session adds after load, measured on this model
    f["session_fall"] = fall
  if warn := speed_up.usable(model):  # a Speed up copy waiting on its task eval: tuned, and said; a failed one refused
    f["notes"] = [*f.get("notes", []), warn]
  first: dict[str, Any] = {}
  rows, plan = settle(f, feats, engine, tab, into=first)
  measured: dict[str, Any] = {}
  if measure:
    out = out or pathlib.Path.home() / ".boltbeam" / "tune" / f"{engine}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    go = run or ad.launch_measure(model, feats, f, port=port, out=out, extra=extra)
    scoped = memory_fit(rows, first, engine, tab, only=knobs_only) if plan and knobs_only else plan  # what --knob lets it move
    fitted = set(scoped["rows"]) if scoped and scoped.get("choices") else set()  # measured together, at the end
    todo = [k for k in knobs(engine, tab) if k["derive"] == "measured" and (not knobs_only or k["name"] in knobs_only)
            and next(x for x in rows if x["name"] == k["name"])["status"] not in ("not_in_build", "not_applicable") and k["name"] not in fitted
            and not held(k, f)]
    together = bool(fitted)
    say = say or (lambda _line: None)
    say(f"pipeline steps: {len(todo) + together}")
    allowed = fact_names(engine, tab)
    for k in todo:
      now: dict[str, Any] = {}
      rows, _ = settle(f, feats, engine, tab, measured, into=now)  # a later sweep runs with the earlier ones' answers and reads the facts they set
      s = k["sweep"]
      most = ((len(choice_points(k, rows, now, allowed)["points"]) + 1) * (2 if s.get("gate") else 1) if s["type"] == "choice" else
              (len(s["points"]) if s["type"] != "fit" else 1 + s["tries"]) * (2 if s["type"] == "crossover" else 1))
      say(f"stage {k['name']}: start")
      try:
        measured[k["name"]] = sweep(k, _counted(go, say, k["name"], most), rows, prompt=prompt, facts=now, allowed=allowed)
      except Exception as exc:  # noqa: BLE001  a failed sweep keeps the facts value and says why (in its row, not the stage)
        measured[k["name"]] = {"value": None, "status": "failed", "why": _words()["tune"]["sweep_failed"].format(out=short_path(str(out))),
                               "detail": f"{type(exc).__name__}: {str(exc)[-DETAIL_CHARS:]}"}  # the raw text: provenance, not the reason
      say(f"stage {k['name']}: done")
    if together:
      now = {}
      rows, plan = settle(f, feats, engine, tab, measured, into=now)
      say("stage memory fit: start")
      try:
        plan = fit_measure(plan, rows, _counted(go, say, "memory fit", 2 * (CONFIRM + 2)), now, engine, tab, prompt=prompt, only=knobs_only)
      except Exception as exc:  # noqa: BLE001  a failed fit sweep keeps the unchecked choice and says why
        plan = {**plan, "failed": _words()["tune"]["sweep_failed"].format(out=short_path(str(out))), "detail": f"{type(exc).__name__}: {str(exc)[-DETAIL_CHARS:]}"}
      say("stage memory fit: done")
    say("pipeline done")
    rows, plan = settle(f, feats, engine, tab, measured, plan=plan)
    (out / "tune.json").write_text(json.dumps({"engine": engine, "rows": rows, "measured": measured, "memory_fit": plan, "facts": f}, indent=1, default=str))
  md = mods.found(engine, feats.get("build"), feats["features"])  # its build record, else its mods' markers in its files
  shown = mods.with_switches(ad.render(values_of(rows), f, feats, model, extra), md)  # as a run on it sets them
  spent = None
  if spend:  # the spend fit: the ways to spend the GPU's memory at the context, ranked on the measured runs given (workflow/spend.py)
    spent = {"why": "no_facts", "what": f["spend_why"]} if f.get("spend_why") else sp.plan(f, engine, tab, model=model, records=sp.read_records(spend), context=None if spend_max else f["context_tokens"],
                    features=set(feats["features"]), ad=ad, allowed=fact_names(engine, tab), local=[options["draft"]] if options.get("draft") else None)
  if (sw := (spent or {}).get("margin", {}).get("switch")) and "argv" in shown:  # a bound headroom holds only with its switch set
    env = {**(shown.get("env") or {}), **sw}
    shown = {**shown, "env": env, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), *map(str, shown["argv"])])}
  return {"schema": "boltbeam.tune.v1", "spend": spent, "engine": engine, "model": model, "build": {**{k: v for k, v in feats.items() if k != "features"},
          "groups": groups_in(engine, feats, tab), "feature_count": len(feats["features"]), "modded": md, "mods_known": len(mods.mods(engine)),
          "series": sorted(m["id"] for m in mods.mods(engine) if m.get("state") == "series" and m.get("marker") in feats["features"])}, "facts": f, "rows": rows, **shown,
          "measured": measured, "memory_fit": plan, "out": str(out) if out else None}


def _counted(go:Callable[..., Any], say:Callable[[str], None], stage:str, most:int) -> Callable[..., Any]:
  """go, saying each call as the stage's progress line (at most `most`)."""
  n = [0]
  def run(*a, **kw):
    n[0] += 1
    say(f"stage {stage}: progress {min(n[0], most) - 1}/{most}")
    return go(*a, **kw)
  return run


# --- the words ---------------------------------------------------------------------------------------------------------

def short_path(path:str) -> str:
  """A build's path as a reader sees it: the home folder as ~, and a deep path as its first two folders, ... and the
  program (the full path stays in the command and in the record)."""
  p = pathlib.Path(path).expanduser()
  try:
    parts, head = p.relative_to(pathlib.Path.home()).parts, "~"
  except ValueError:
    parts, head = (p.parts[1:], "") if p.is_absolute() else (p.parts, ".")
  if len(parts) > 4:
    parts = (*parts[:2], "...", parts[-1])
  return "/".join((head, *parts)) if head else "/" + "/".join(parts)


def text(res:dict[str, Any], tab:dict[str, Any] | None = None) -> str:
  """The setup as plain lines (data/plain_words.json tune)."""
  from boltbeam.report.plain import words
  tab = tab or table()
  w, b, f, rows, eng = words()["tune"], res["build"], res["facts"], res["rows"], tab["engines"][res["engine"]]
  have = [k for k, v in b["groups"].items() if v]
  ver = f", {b['version']}" if b.get("version") else ""
  md = b.get("modded")  # a modded build (its record, or its mods' markers) is named as one; its mods are the line under it, not "none"
  fork = (w["modded_found"] if md.get("found") else w["modded_build"].format(key=md.get("key"))) if md else b.get("fork") or w["unlisted"]
  lines = [w["build"].format(engine=res["engine"], path=short_path(b["build"]), backend=(b.get("backend") or "?").upper(), version=ver, fork=fork)]
  if eng["groups"] and have:
    lines[-1] += " " + w["groups"].format(what=eng["groups_words"], groups=", ".join(have))
  if md:
    lines.append(w["modded_unverified"].format(status=md.get("status"), mods=", ".join(m["id"] for m in md.get("mods", []))))
    if md.get("other"):
      lines.append(w["modded_other"].format(mods=", ".join(md["other"])))
  elif b.get("mods_known"):  # no record and no marker: said, never read silently as a stock build
    lines.append(w["no_mods_found"].format(n=b["mods_known"]))
  lines += list(b.get("notes") or []) + list(f.get("notes") or [])
  lines.append(w["host"].format(total=f["ram_total_bytes"] / GiB, usable=f["usable_ram_bytes"] / GiB,
                                avail="unknown" if f["ram_available_bytes"] is None else f"{f['ram_available_bytes'] / GiB:.1f}",
                                l3="unknown" if f["l3_bytes"] is None else f"{f['l3_bytes'] / MiB:.0f} MiB", l3_source=f["l3_source"]))
  if fitted := fit_words(res.get("memory_fit")):
    lines.append(fitted)
  if res.get("spend"):
    from boltbeam.workflow import spend as sp
    lines += ["", *sp.words(res["spend"], tab["engines"][res["engine"]].get("spend") or {})]
  lines += ["", w["setup"]]
  for r in rows:
    if r["status"] in ("set", "default"):
      name = named(r, r["value"], w)
      tail = ("" if r["bitwise"] else " " + w["not_bitwise"]) + ("" if r["output"] == "identical" else f" [output {r['output']}]")
      lines.append(f"  {name:<34} {w['source_' + r['source']]}: {r['reason']}{tail}")
  na = [r for r in rows if r["status"] == "not_applicable"]
  if na:
    lines += ["", w["not_applicable"]] + [f"  {r['name']}: {r['reason']}" for r in na]
  absent = [r for r in rows if r["status"] == "not_in_build"]
  gone = sorted({r["group"] for r in absent if r["group"] and not b["groups"].get(r["group"])})
  older = [r["name"] for r in absent if r["group"] and b["groups"].get(r["group"])]
  lacks = [r["name"] for r in absent if not r["group"]]
  if older:
    lines += ["", w["older_group"].format(noun=eng["group_noun"], knobs=", ".join(older))]
  if gone and have:  # a build with none of the groups: its notes say so once (the adapter's), naming the build's own mods apart
    lines += ["", w["not_in_build"].format(groups=", ".join(gone))]
  if lacks:
    lines += ["", w["lacks"].format(engine=res["engine"], knobs=", ".join(lacks))]
  lines += ["", w["command"], "  " + res["command"]]
  if not res["measured"] and any(r["provenance"].get("measure") for r in rows):
    lines.append(w["measure_hint"])
  return "\n".join(lines)


def page(res:dict[str, Any], setting_of:Callable[[dict[str, Any]], str | None] | None = None,
         tab:dict[str, Any] | None = None) -> dict[str, Any]:
  """The setup as a screen lays it out (data/plain_words.json tune.page): the build line, the notes, one row per knob set
  or left at its default (its value, source, reason and tags), the knobs not used, the command, what --measure would
  confirm, and what "use these" copies. setting_of(row) names the run setting that takes a flag or config row's value
  (None: a run does not pass it); an env row is passed as an environment variable. Engine-blind, as the rest."""
  from boltbeam.report.plain import words
  w, wp = words()["tune"], words()["tune"]["page"]
  lines = text(res, tab).split("\n")
  head = lines[0]
  notes = [line for line in lines[1:lines.index(w["setup"])] if line.strip()]
  rows, settings, env, not_passed = [], {}, {}, []
  uses: dict[str, dict[str, Any]] = {}
  for r in res["rows"]:
    if r["status"] == "set" and r["value"] is not None:
      if r["kind"] == "env":
        uses[r["name"]] = {"env": r["name"], "value": r["value"]}
      elif (name := (setting_of(r) if setting_of else None)) and None not in (goes := {setting_of({**r, "name": x}): t for x, t in (r.get("with") or {}).items()} if r["kind"] == "flag" else {}):
        uses[r["name"]] = {"setting": name, "value": r["value"],  # the flags it goes with are copied beside it, or the row is not
                           **({"with": {x: say(t, {**res["facts"], "value": r["value"]}) for x, t in goes.items()}} if goes else {})}
      else:
        not_passed.append(r["name"])
  placed = [{**p, "use": setting_of({**p, "kind": "flag"}) if setting_of else None} for p in res.get("placement") or []]
  held = held_back(res, uses, placed, tab)  # a value whose formula assumes a row (or a placement) a run cannot pass
  for p in placed:
    if p["use"] and not held.get(None):
      settings[p["use"]] = p["value"]
  for r in res["rows"]:
    if r["status"] not in ("set", "default"):
      continue
    tags = ([] if r["bitwise"] else [w["not_bitwise"].strip("[]")]) + ([] if r["output"] == "identical" else [f"output {r['output']}"])
    use = None if r["name"] in held else uses.get(r["name"])
    if use and "env" in use:
      env[use["env"]] = use["value"]
    elif use:
      settings |= {use["setting"]: use["value"], **use.get("with", {})}
    confirm = (r["provenance"] or {}).get("measure")
    rows.append({"name": r["name"], "kind": r["kind"], "status": r["status"], "value": w["unset"] if r["value"] is None else r["value"],
                 "source": r["source"], "source_words": w["source_" + r["source"]], "reason": r["reason"], "tags": tags,
                 "group": r["group"], "use": use, "confirm": wp["confirm"].format(how=confirm.removeprefix("confirm with --measure: ")) if confirm else None,
                 "held": wp["held"].format(knob=r["name"], needs=", ".join(held[r["name"]])) if r["name"] in held else None})
  measured = [k for k, v in (res.get("measured") or {}).items() if v.get("value") is not None]
  to_measure = [r["name"] for r in res["rows"] if (r["provenance"] or {}).get("measure")]
  tail = [line for line in lines[lines.index(w["setup"]) + 1:] if line.strip()]
  absent = [line for line in tail if not line.startswith("  ") and line not in (w["not_applicable"], w["command"], w["measure_hint"])]
  return {"title": wp["title"], "head": head, "notes": notes, "says": wp["measured"].format(knobs=", ".join(measured)) if measured else wp["facts"],
          "rows": rows, "not_used_title": wp["not_used"],
          "not_used": [{"name": r["name"], "reason": r["reason"]} for r in res["rows"] if r["status"] == "not_applicable"],
          "absent": absent, "command": res["command"],
          "measure": {"label": wp["measure"], "one": wp["measure_one"], "helper": wp["measure_helper"], "knobs": to_measure, "none": None if to_measure else wp["measure_none"],
                      "running": wp["running"], "stop": wp["stop"], "stop_helper": wp["stop_helper"], "stopped": wp["stopped"],
                      "failed": wp["failed"], "starting": wp["starting"], "progress": wp["progress"]},
          "use": {"label": wp["use"], "helper": wp["use_helper"], "settings": settings, "env": env,
                  "none": None if settings or env else wp["use_none"], "used": wp["used"],
                  "not_passed": " ".join(filter(None, [wp["not_passed"].format(knobs=", ".join(not_passed)) if not_passed else None,
                                                       *(wp["held"].format(knob=k, needs=", ".join(v)) for k, v in held.items() if k)])) or None,
                  "env_about": wp["env"]}}


def held_back(res:dict[str, Any], uses:dict[str, dict[str, Any]], placed:list[dict[str, Any]], tab:dict[str, Any] | None = None) -> dict[str | None, list[str]]:
  """The rows "use these" must not copy, each with what its value assumes that a run cannot pass: a row whose formula (or
  `applies`) reads a fact another row's `sets` writes, when that row is set and not copied (through the chain), or a fact
  of the experts' placement (the adapter's PLACEMENT_FACTS) when the placement cannot be passed. Key None: the placement
  itself is held (a part of it cannot be passed, so none of it is copied)."""
  tab = tab or table()
  ks = {k["name"]: k for k in knobs(res["engine"], tab)}
  writes = {fact: k["name"] for k in ks.values() for fact in (k.get("sets") or {})}
  status = {r["name"]: r for r in res["rows"]}
  out: dict[str | None, list[str]] = {}
  stuck = [p for p in placed if not p["use"]]
  if stuck:
    out[None] = [f"{p['name']} {p['value']}" for p in stuck]
  where = set(getattr(adapter(res["engine"], tab), "PLACEMENT_FACTS", ())) if placed else set()

  def needs(name:str, seen:set[str]) -> list[str]:
    k = ks.get(name) or {}
    reads = names_in(k["formula"]) if k.get("formula") else []
    reads += names_in(k.get("applies", "True"))
    got = [" ".join(f"{p['name']} {p['value']}" for p in placed)] if stuck and where & set(reads) else []
    for fact in reads:
      dep = writes.get(fact)
      if not dep or dep == name or dep in seen:
        continue
      d = status.get(dep) or {}
      if d.get("status") == "set" and dep not in uses:
        got.append(f"{dep} {d['value']}")
      got += needs(dep, seen | {name})
    return list(dict.fromkeys(got))
  for name in uses:
    if got := needs(name, set()):
      out[name] = got
  return out
