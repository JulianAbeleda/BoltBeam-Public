"""The llama.cpp adapter of `boltbeam tune` (workflow/tune.py): the build read for the getenv names compiled into it, the
GGUF's expert bytes and host tables, the offload plan's placement and chunk, a measured point through the real-text
llama-server prefill driver, and the env + flags command.

The build is read, not run: a knob applies when its env-var name is a string in llama-server or the libraries beside it
(getenv literals), so a stock build gets only stock flags, and a v1 pinned-stream build no pin budget."""
from __future__ import annotations

import json
import math
import mmap
import pathlib
import platform
import re
import statistics
import sys
from typing import Any, Callable

from boltbeam.core.refused import Refused
from boltbeam.roofline import routing_profile as rp

ENGINE = "llama.cpp"
NS = "llama_cpp"
PLACEMENT_FACTS = ("host_expert_bytes",)  # facts that hold only with the experts placed as the command places them
FACTS = {"pin_in_build": "the build can pin host experts at load (LLAMA_WS_PIN_GIB is in it)",
         "prompt_cache_default_bytes": "llama-server's prompt cache in RAM by default (recommend.CACHE_RAM_DEFAULT)",
         "prompt_cache_bytes": "the prompt cache this setup keeps (set by the --cache-ram row)",
         "plan_ubatch": "the chunk the offload plan's prompt limit picks (workflow/offload.prefill_plan)",
         "plan_ubatch_why": "where the chunk came from, or why there is none",
         "routing_note": "a clause from the saved routing profile of this model (how concentrated its expert use is), or empty when there is none",
         "host_tables_resident_bytes": "the host-mapped input tables' rows one prompt reads (one row per token, at least a page each): an estimate"}
NAMES = re.compile(rb"(?:LLAMA|GGML)_[A-Z0-9_]{2,}")  # getenv literals in the build
GiB = 1 << 30


def linked(binary:pathlib.Path, run_command=None) -> list[pathlib.Path]:
  """The ggml and llama libraries a program links, as the loader lists them (ldd on Linux, otool -L on macOS); [] when
  neither tool answers. Read only: the program is not run."""
  from boltbeam.target.hardware import run_tool
  rc, out, _ = (run_command or run_tool)(("otool", "-L", str(binary)) if platform.system() == "Darwin" else ("ldd", str(binary)))
  if rc != 0:
    return []
  found = re.findall(r"(/\S*lib(?:ggml|llama)\S*\.(?:so|dylib)\S*)", out)
  return [pathlib.Path(f) for f in dict.fromkeys(found) if pathlib.Path(f).is_file()]


def detect(server:str | pathlib.Path | None) -> dict[str, Any]:
  """What a llama-server build is, read without running it: the env-var names compiled into it and the libraries beside it
  (each patch's knobs are getenv literals), its backend from those libraries' names and the fork row it matches
  (data/engine_families.json). A Refused when there is no such file."""
  from boltbeam.collectors import llama_server_decode as lsd
  server = server or lsd.find(None)
  if not server:
    raise Refused("no llama-server found (beside llama-bench, $BOLTBEAM_LLAMA_SERVER or on PATH); pass --build PATH")
  p = pathlib.Path(server).expanduser()
  if p.is_file() and p.name != lsd.TOOL and (p.parent / lsd.TOOL).is_file():  # another program of the build (the screen's llama-bench)
    p = p.parent / lsd.TOOL
  if not p.is_file():
    raise Refused(f"no llama-server at {p}; pass --build PATH or put llama-server on PATH")
  real = p.resolve()
  dirs = {real.parent, real.parent.parent / "lib"}
  for dep in linked(real):  # a packaged build (Homebrew) keeps ggml and its loadable backends (libexec) apart
    dirs |= {dep.parent, dep.parent.parent / "libexec"}
  files = [real] + sorted(f for d in dirs if d.is_dir() for f in d.iterdir() if f.is_file() and re.search(r"^lib(ggml|llama)\S*\.(so|dylib)", f.name))
  seen, names, libs, where = set(), set(), [], {}
  for f in files:
    r = f.resolve()
    if r in seen:
      continue
    seen.add(r)
    libs.append(f.name)
    try:
      with open(r, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for m in NAMES.finditer(mm):
          where.setdefault(m.group().decode(), f.name)
    except (OSError, ValueError):
      continue
  names = set(where)
  joined = " ".join(libs)
  kinds = (("hip", "ggml-hip", "GGML_HIP_"), ("metal", "ggml-metal", "GGML_METAL_"), ("cuda", "ggml-cuda", "GGML_CUDA_"))
  backend = next((k for k, lib, _ in kinds if lib in joined), None) or \
    next((k for k, _, pre in kinds if any(n.startswith(pre) for n in names)), "cpu")  # a HIP build carries GGML_CUDA_ names too
  fork = None
  try:
    from boltbeam.collectors import engine_families
    row = engine_families.build_of(str(p), ENGINE)
    fork = row.get("label") if row and not row.get("provider") else None  # the root row is build_of's fallback, not a match
  except Exception:  # noqa: BLE001  a fork name is a label, never a reason to refuse
    pass
  return {"build": str(p), "files": libs, "features": where, "backend": backend, "fork": fork, "version": None,
          "source": f"env-var names in {len(libs)} file(s) of the build (no run)"}


# --- the facts ---------------------------------------------------------------------------------------------------------

def expert_layers(source:str | pathlib.Path) -> dict[str, Any]:
  """Expert bytes per layer in layer order and the experts per layer, from the GGUF tensor table (0 for a dense model)."""
  from boltbeam.profile.decode_roles import decode_roofline_inventory_from_gguf
  p = pathlib.Path(source).expanduser()
  try:
    inv = decode_roofline_inventory_from_gguf(p)
  except (OSError, ValueError) as exc:
    raise Refused(f"{p.name}: its GGUF tensor table could not be read (the file is missing, cut short or not a GGUF)") from exc
  per: dict[int, int] = {}
  n = 0
  for t in inv["tensors"]:
    if t.get("n_expert"):
      layer = int(t["tensor_name"].split(".")[1])
      per[layer] = per.get(layer, 0) + t["packed_weight_bytes"] * t["n_expert"] // t["experts_read"]
      n = max(n, t["n_expert"])
  return {"per_layer": [per[k] for k in sorted(per)], "ids": sorted(per), "experts": n, "source": f"{p.name} GGUF tensor table"}


def cpu_moe_layers(ids:list[int], expert_layers:int) -> int:
  """llama.cpp's --n-cpu-moe N keeps the experts of the model's layers 0..N-1 in RAM: N counts the model's layers, dense
  ones included (a model whose first layers are dense has no experts there). The N that puts the first `expert_layers`
  expert layers (in layer order) in RAM."""
  return ids[min(expert_layers, len(ids)) - 1] + 1 if expert_layers > 0 and ids else 0


def host_tables(source:str | pathlib.Path, prompt:int, page:int) -> dict[str, Any]:
  """The input tables llama.cpp keeps mapped in host RAM (recommend.HOST_ONLY: token_embd, per_layer_token_embd): their bytes
  and the rows one prompt reads, one row per token and table, at least a page each (rows sit apart): an estimate."""
  from boltbeam.profile import gguf
  from boltbeam.workflow.recommend import HOST_ONLY, _gguf_files
  total = resident = 0
  for f in _gguf_files(pathlib.Path(source).expanduser()):
    _, ts, start = gguf.read_gguf_layout(f)
    offs = sorted((o, name, dims) for name, dims, _, o in ts)
    for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [f.stat().st_size - start]):
      if name.split(".")[0] in HOST_ONLY:
        rows = max(1, math.prod(dims[1:]))
        total += nxt - o
        resident += min(nxt - o, prompt * max((nxt - o) // rows, page))
  return {"bytes": total, "resident_bytes": resident}


def offload_plan(model:str, prompt:int) -> dict[str, Any]:
  """The offload page's setup for prompts on this machine's GPU (workflow/offload.prefill_plan through recommend): the
  layers whose experts it keeps in RAM and the chunk its prompt limit picks; {"why"} when there is none."""
  try:
    from boltbeam.profile.loaders import profile_from_model
    from boltbeam.target.targets import get_target
    from boltbeam.workflow import chips, screen
    tid = chips.detect().get("target_id")
    if not tid:
      return {"why": "no GPU target detected"}
    prof, tgt = profile_from_model(model, None).to_json(), get_target(tid)
    got = screen.recommended(prof, tgt, engine=ENGINE, context=prompt, offload=True)
    saved = rp.find(model, engine=ENGINE)
    flat = screen.recommended(prof, tgt, engine=ENGINE, context=prompt, offload=True, routing=False) if saved else None
  except Refused as exc:  # a refusal is already in words
    return {"why": str(exc)}
  except (Exception, SystemExit) as exc:  # noqa: BLE001  no plan is a stated default, never a crash; the raw text is provenance
    from boltbeam.report.plain import words
    return {"why": words()["tune_llama_cpp"]["no_plan"], "detail": f"{type(exc).__name__}: {exc}"}
  pre = (got.get("offload") or {}).get("prefill") or {}
  if pre.get("status") != "recommended":
    return {"why": pre.get("reason") or (got.get("offload") or {}).get("reason") or "no prompt setup"}
  out = {"layers_in_ram": int(pre["layers_in_ram"]), "ubatch": int(pre["ubatch"]), "source": f"the offload plan on {tid} (boltbeam offload)"}
  flat_pre = ((flat or {}).get("offload") or {}).get("prefill") or {}
  if saved and flat_pre.get("status") == "recommended":
    out |= {"ubatch_uniform": int(flat_pre["ubatch"]), "routing_profile": saved}
  return out


def routing_note(p:dict[str, Any] | None) -> str:
  """A clause for a knob's reason from the model's saved routing profile; empty without one."""
  return f"; on this model the hottest 10% of experts take {p['derived']['top10_share']:.0%} of selections" if p and p.get("derived") else ""


def facts(model:str, host:dict[str, Any], b:dict[str, Any], options:dict[str, Any]) -> dict[str, Any]:
  """The model's facts (experts per layer and those this placement keeps in host RAM, the host tables) and llama.cpp's own
  (llama_cpp.*), with "sources" and "notes". options: experts_in_ram ("all" or a layer count; default the offload
  plan's), plan (a stand-in; {} skips it), system. No GPU is touched: the plan reads saved figures only."""
  from boltbeam.report.plain import words
  from boltbeam.workflow.recommend import CACHE_RAM_DEFAULT
  w = words()["tune_llama_cpp"]
  system = options.get("system") or platform.system()
  unified = b["backend"] == "metal" or (system == "Darwin" and platform.machine() == "arm64")
  ex = expert_layers(model)
  layers = len(ex["per_layer"])
  ids = ex.get("ids") or list(range(layers))
  top = ids[-1] + 1 if ids else 0  # --n-cpu-moe at least this keeps every expert in RAM (--cpu-moe)
  plan = options.get("plan")
  if plan is None:
    plan = offload_plan(model, host["prompt_tokens"]) if not unified else {"why": "unified memory: no offload plan"}
  given = options.get("experts_in_ram")
  # in_ram: --n-cpu-moe's N, the model's layers 0..N-1 (dense layers count); the plan counts expert layers
  if given == "all":
    in_ram, where = top, "--experts-in-ram all (given)"
  elif given is not None:
    in_ram, where = min(max(int(given), 0), top), f"--experts-in-ram {given} (given)"
  elif "layers_in_ram" in plan:
    in_ram, where = cpu_moe_layers(ids, plan["layers_in_ram"]), plan["source"]
  else:
    in_ram, where = 0, f"none given and no offload plan ({plan.get('why')}): every expert on the GPU"
  host_experts = 0 if unified else sum(b_ for l, b_ in zip(ids, ex["per_layer"]) if l < in_ram)
  tabs = host_tables(model, host["prompt_tokens"], host["page_bytes"])
  placement = "unified memory: the GPU reads the experts where they are, nothing is copied" if unified else where
  from boltbeam.workflow import tune
  have = [g for g in tune.table()["engines"][ENGINE]["groups"].values() if g["marker"] in b["features"]]
  notes = ([] if have else [w["no_patch"]]) + ([w["unified"]] if unified else [])
  name = pathlib.Path(model).name
  if plan.get("ubatch_uniform") is not None and plan.get("ubatch") != plan["ubatch_uniform"] and plan.get("routing_profile"):
    p = plan["routing_profile"]
    notes.append(w["routing_changed"].format(what=rp.words(p, ENGINE).removeprefix(rp.MEASURED + " "), before=plan["ubatch_uniform"], after=plan["ubatch"]))
  notes.append(w["model_dense"].format(name=name) if not ex["experts"] else
               w["model"].format(name=name, layers=layers, experts=ex["experts"], gib=sum(ex["per_layer"]) / GiB, placement=placement, host=host_experts / GiB))
  out = {"host_expert_bytes": host_experts, "model_experts": ex["experts"], f"{NS}.host_tables_resident_bytes": tabs["resident_bytes"],
         "unified_memory": unified, f"{NS}.prompt_cache_default_bytes": CACHE_RAM_DEFAULT, f"{NS}.prompt_cache_bytes": CACHE_RAM_DEFAULT,
         f"{NS}.routing_note": routing_note(options.get("routing_profile", plan.get("routing_profile") or rp.find(model, engine=ENGINE))),
         f"{NS}.plan_ubatch": plan.get("ubatch"), f"{NS}.plan_ubatch_why": plan.get("why") or plan.get("source"),
         f"{NS}.pin_in_build": "LLAMA_WS_PIN_GIB" in b["features"], "layers": top, "layers_in_ram": 0 if unified else in_ram,
         "expert_bytes_total": sum(ex["per_layer"]), "host_tables_bytes": tabs["bytes"], "placement": placement, "notes": notes}
  out["sources"] = {"host_expert_bytes": f"{ex['source']}: layers 0-{in_ram - 1} ({where})" if in_ram and not unified else where,
                    "model_experts": ex["source"], f"{NS}.host_tables_resident_bytes": f"estimate: {host['prompt_tokens']} tokens x one row (at least a {host['page_bytes']} B page) per host table",
                    f"{NS}.prompt_cache_default_bytes": "llama-server --cache-ram default (8192 MiB)", f"{NS}.prompt_cache_bytes": "the --cache-ram row",
                    f"{NS}.plan_ubatch": plan.get("source") or plan.get("detail") or plan.get("why"), f"{NS}.pin_in_build": b["source"],
                    "unified_memory": f"build backend {b['backend']}, {system}"}
  return out


# --- the command and a measured point -----------------------------------------------------------------------------------

def _placement(f:dict[str, Any]) -> list[str]:
  if f.get("unified_memory") or not f.get("layers_in_ram"):
    return []
  return ["--cpu-moe"] if f["layers_in_ram"] >= f["layers"] else ["--n-cpu-moe", str(f["layers_in_ram"])]


def _env_flags(values:list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
  """The env block and the flags; -b is raised to -ub when -ub is larger than llama-server's default batch."""
  env = {v["name"]: v["value"] for v in values if v["kind"] == "env"}
  flags: list[str] = []
  given = {v["name"] for v in values}
  for v in values:
    if v["kind"] == "flag":
      flags += [v["name"], v["value"]]
      if v["name"] == "-ub" and "-b" not in given and int(v["value"]) > 2048:
        flags += ["-b", v["value"]]
  return env, flags


def render(values:list[dict[str, Any]], f:dict[str, Any], b:dict[str, Any], model:str, extra:list[str] | None = None) -> dict[str, Any]:
  env, flags = _env_flags(values)
  argv = [b["build"], "-m", model, "-ngl", "99", *_placement(f), *flags, *(extra or [])]
  place = _placement(f)  # where the experts sit: a run takes it as a setting, so "use these" copies it with the values that assume it
  placement = [{"name": "--cpu-moe", "value": "on"}] if place == ["--cpu-moe"] else [{"name": place[0], "value": place[1]}] if place else []
  return {"env": env, "flags": flags, "config": {}, "argv": argv, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), *argv]),
          "placement": placement}


def point_ms(server:str, model:str, *, prompt:int, flags:list[str], env:dict[str, str], reps:int, port:int, log:pathlib.Path) -> float:
  """One cold real-text prefill of `prompt` tokens in a fresh llama-server (the --prefill driver), the median client wall ms."""
  from boltbeam.collectors import engine_common as ec, llama_server_decode as lsd
  from boltbeam.collectors.llama_server_prefill import CORPUS
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(port),
          "--ctx-size", str(prompt + lsd.HEADROOM), "--contexts", str(prompt), "--reps", str(reps), "--flags", json.dumps(flags),
          "--log", str(log.with_suffix(".server.log")), "--prefill", "--corpus", str(CORPUS)]
  lines = ec.run_driver(argv, env=env, log=log)
  got = next(line for line in lines if line.get("prefill"))
  return statistics.median(got.get("wall_samples_ms") or got["samples_ms"])


def launch_measure(model:str, b:dict[str, Any], f:dict[str, Any], *, port:int, out:pathlib.Path, extra:list[str] | None = None) -> Callable[..., float]:
  """run(values=, prompt=, reps=, tag=) -> wall ms: a fresh llama-server per point with that setup."""
  def run(*, values, prompt, reps, tag):
    env, flags = _env_flags(values)
    return point_ms(b["build"], model, prompt=prompt, flags=["-ngl", "99", *_placement(f), *flags, *(extra or []), "--no-warmup"], env=env,
                    reps=reps, port=port, log=out / f"{tag}.driver.log")
  return run
