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
         "host_tables_resident_bytes": "the host-mapped input tables' rows one prompt reads (one row per token, at least a page each): an estimate",
         "on_card_bytes": "the model's weights on the GPU: the GGUF's tensor bytes less the host experts and the host-mapped tables",
         "kv_cache_bytes": "the KV cache at the context (f16, the attended layers) plus the recurrent state, from the model's profile",
         "host_expert_tensors": "the expert tensors this placement keeps in host RAM (a slot holds one)",
         "host_expert_tensor_max_bytes": "the largest of those tensors: a slot grows to the largest it has held",
         "host_expert_layers": "the layers whose experts this placement keeps in host RAM",
         "model_layers": "the model's layers (--n-cpu-moe counts them)",
         "chunk_tokens": "the chunk (-ub) the command runs: the offload plan's, else llama-server's 512",
         "hidden_size": "the model's hidden width", "experts_used": "experts each token is routed to", "expert_ffn": "one expert's FFN width",
         "prompt_compute_ms": "the GPU's compute time for the prompt: a measured time budget's kernel time when one is given, else the prompt limit's (a lower bound)",
         "prompt_compute_source": "where the compute time came from",
         "free_after_load_bytes": "--measure: the GPU's free memory once llama-server has loaded the model",
         "free_after_warm_bytes": "--measure: the GPU's free memory after llama-server read one prompt (its slots and pools grown)",
         "slot_alloc_failures": "--measure: the streamer's failed slot allocations in llama-server's log",
         "logged_compute_buffer_bytes": "--measure: the GPU compute buffer llama-server logged at load",
         "vram_why": "which of the facts the slot count reads could not be read, or empty"}
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


def server_of(build:str | pathlib.Path) -> pathlib.Path:
  """The llama-server of a build given as any of its programs: the screen's llama-bench maps to the server beside it."""
  from boltbeam.collectors import llama_server_decode as lsd
  p = pathlib.Path(build).expanduser()
  return p.parent / lsd.TOOL if p.is_file() and p.name != lsd.TOOL and (p.parent / lsd.TOOL).is_file() else p


def detect(server:str | pathlib.Path | None) -> dict[str, Any]:
  """What a llama-server build is, read without running it: the env-var names compiled into it and the libraries beside it
  (each patch's knobs are getenv literals), its backend from those libraries' names and the fork row it matches
  (data/engine_families.json). A Refused when there is no such file."""
  from boltbeam.collectors import llama_server_decode as lsd
  server = server or lsd.find(None)
  if not server:
    raise Refused("no llama-server found (beside llama-bench, $BOLTBEAM_LLAMA_SERVER or on PATH); pass --build PATH")
  p = server_of(server)
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
  tensors: list[tuple[int, int]] = []  # (layer, bytes) of each expert tensor: a slot holds one
  n = used = ff = 0
  for t in inv["tensors"]:
    if t.get("n_expert"):
      layer = int(t["tensor_name"].split(".")[1])
      size = t["packed_weight_bytes"] * t["n_expert"] // t["experts_read"]
      per[layer] = per.get(layer, 0) + size
      tensors.append((layer, size))
      n, used, ff = max(n, t["n_expert"]), max(used, t["experts_read"]), max(ff, min(t["shape"]))
  return {"per_layer": [per[k] for k in sorted(per)], "ids": sorted(per), "experts": n, "tensors": tensors, "experts_used": used, "expert_ffn": ff,
          "source": f"{p.name} GGUF tensor table"}


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
  total = resident = data = 0
  for f in _gguf_files(pathlib.Path(source).expanduser()):
    _, ts, start = gguf.read_gguf_layout(f)
    data += f.stat().st_size - start
    offs = sorted((o, name, dims) for name, dims, _, o in ts)
    for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [f.stat().st_size - start]):
      if name.split(".")[0] in HOST_ONLY:
        rows = max(1, math.prod(dims[1:]))
        total += nxt - o
        resident += min(nxt - o, prompt * max((nxt - o) // rows, page))
  return {"bytes": total, "resident_bytes": resident, "file_bytes": data}


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
  chunk = next((c for c in pre.get("chunks") or [] if c.get("ubatch") == out["ubatch"]), None)
  if chunk and (chunk.get("ms") or {}).get("gpu"):
    out["gpu_ms"] = float(chunk["ms"]["gpu"])  # the prompt limit's GPU time at that chunk: compute, the copy apart
  flat_pre = ((flat or {}).get("offload") or {}).get("prefill") or {}
  if saved and flat_pre.get("status") == "recommended":
    out |= {"ubatch_uniform": int(flat_pre["ubatch"]), "routing_profile": saved}
  return out


def routing_note(p:dict[str, Any] | None) -> str:
  """A clause for a knob's reason from the model's saved routing profile; empty without one."""
  return f"; on this model the hottest 10% of experts take {p['derived']['top10_share']:.0%} of selections" if p and p.get("derived") else ""


def compute_time(plan:dict[str, Any], budget:dict[str, Any] | None, prompt:int) -> tuple[float | None, str]:
  """The GPU's compute time for the prompt (ms) and where it came from: a measured time budget's kernel time
  (collectors/time_budget.py, boltbeam.time_budget.v1) when one is given and was measured at this prompt length
  (tune.budget_unusable), else the prompt limit's GPU time at the plan's chunk (a lower bound: the real kernels are
  slower), saying why a given budget was not used."""
  from boltbeam.workflow.tune import budget_unusable
  unused = ""
  if budget and isinstance(budget.get("kernels_ms"), (int, float)) and budget["kernels_ms"] > 0:
    if (why := budget_unusable(budget, prompt)) is None:
      return float(budget["kernels_ms"]), f"kernel time of a measured time budget ({budget.get('what') or budget.get('source') or 'given'})"
    unused = f"; the time budget given is not used: {why}"
  if plan.get("gpu_ms"):
    return float(plan["gpu_ms"]), "the prompt limit's GPU time, a lower bound" + unused
  return None, "no measured time budget and no prompt limit" + unused


def slot_facts(model:str, ex:dict[str, Any], tabs:dict[str, Any], host:dict[str, Any], *, in_ram:int, host_experts:int, plan:dict[str, Any],
               options:dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
  """The facts the streamer's slot count reads, read and not combined (the rule is the row's, data/engine_knobs.json): the
  host expert tensors (count, the largest, their layers), the GPU's free memory now, what the model takes of it on the
  card (weights, KV at the context and the recurrent state), the shapes the compute buffer scales with, the chunk, and
  the GPU's compute time for the prompt. vram_why names any that could not be read. options: gpu ({free, total}),
  profile, time_budget (a measured boltbeam.time_budget.v1 record of this setup's prompt)."""
  held = [(layer, b) for layer, b in ex.get("tensors") or [] if layer < in_ram]
  ids = ex.get("ids") or list(range(len(ex.get("per_layer") or [])))
  ms, ms_src = compute_time(plan, options.get("time_budget"), int(host["prompt_tokens"]))
  out: dict[str, Any] = {f"{NS}.host_expert_tensors": len(held), f"{NS}.host_expert_tensor_max_bytes": max((b for _, b in held), default=None),
                         f"{NS}.host_expert_layers": len({layer for layer, _ in held}) or None, f"{NS}.model_layers": (ids[-1] + 1) if ids else None,
                         f"{NS}.chunk_tokens": int(plan.get("ubatch") or 512), f"{NS}.experts_used": ex.get("experts_used"),
                         f"{NS}.expert_ffn": ex.get("expert_ffn"), f"{NS}.hidden_size": None, f"{NS}.on_card_bytes": None,
                         f"{NS}.kv_cache_bytes": None, f"{NS}.prompt_compute_ms": ms, f"{NS}.prompt_compute_source": ms_src,
                         f"{NS}.vram_why": "", "gpu_total_bytes": None, "gpu_free_bytes": None}
  src = {f"{NS}.host_expert_tensors": f"{ex.get('source')}: expert tensors of layers 0-{in_ram - 1}", f"{NS}.host_expert_tensor_max_bytes": ex.get("source"),
         f"{NS}.host_expert_layers": ex.get("source"), f"{NS}.model_layers": ex.get("source"), f"{NS}.experts_used": ex.get("source"),
         f"{NS}.expert_ffn": ex.get("source"), f"{NS}.chunk_tokens": plan.get("source") if plan.get("ubatch") else "llama-server's default -ub",
         f"{NS}.prompt_compute_ms": ms_src}
  gpu = options.get("gpu")
  if gpu is None:
    from boltbeam.workflow.tune_vllm import gpu_memory
    gpu = gpu_memory()
  out["gpu_total_bytes"], out["gpu_free_bytes"] = gpu.get("total"), gpu.get("free")
  src |= {"gpu_total_bytes": "the GPU's telemetry source (NVML / amdgpu sysfs)", "gpu_free_bytes": "the GPU's telemetry source (NVML / amdgpu sysfs)"}
  profile = options.get("profile")
  try:
    if profile is None:
      from boltbeam.profile.loaders import profile_from_model
      profile = profile_from_model(model, None).to_json()
    from boltbeam.roofline import prefill_work as pw
    out[f"{NS}.hidden_size"] = int(profile["hidden_size"]) if profile.get("hidden_size") else None
    src[f"{NS}.hidden_size"] = "the model's profile"
    per, state = pw.kv_bytes_per_token(profile), pw.recurrent_state_bytes(profile)
    if per is None and state is None:
      raise ValueError("the profile gives no attention or recurrent shape for a KV size")
    ctx = int(host.get("context_tokens") or host["prompt_tokens"])
    out[f"{NS}.kv_cache_bytes"] = int((per or 0) * ctx + (state or 0))
    src[f"{NS}.kv_cache_bytes"] = f"roofline/prefill_work at f16: {ctx} tokens x KV per token + the recurrent state"
  except Exception as exc:  # noqa: BLE001  an unsized piece is a stated unknown, never a crash; the raw text is provenance
    out[f"{NS}.vram_why"] = "the model's shapes could not be read for a KV size"
    src[f"{NS}.vram_why"] = f"{type(exc).__name__}: {exc}"
  if tabs.get("file_bytes") is not None:
    out[f"{NS}.on_card_bytes"] = max(tabs["file_bytes"] - host_experts - tabs["bytes"], 0)
    src[f"{NS}.on_card_bytes"] = "the GGUF files' tensor data less the host experts and the host-mapped tables"
  total, on_card, kv_b = out.get("gpu_total_bytes"), out.get(f"{NS}.on_card_bytes"), out.get(f"{NS}.kv_cache_bytes")
  sized = [x for x in options.get("extra") or [] if x in ("-c", "--ctx-size", "-ctk", "-ctv", "--cache-type-k", "--cache-type-v")]
  if sized:  # the KV here is sized at f16 and the context given: flags that change either are not seen, so no fit is claimed
    src[f"{NS}.kv_cache_bytes"] = src.get(f"{NS}.kv_cache_bytes", "") + f"; the fit of one request on the GPU is not checked: --flags sets {', '.join(sized)}"
  elif total and on_card is not None and kv_b is not None and on_card + kv_b > total:  # one request cannot fit this GPU: say so, never clamp the slots to 4 and go on
    from boltbeam.report.plain import words
    raise Refused(words()["tune_llama_cpp"]["no_fit_context"].format(context=int(host.get("context_tokens") or host["prompt_tokens"]),
                                                                       need=(on_card + kv_b) / GiB, gpu=total / GiB))
  need = {"gpu_free_bytes": "the GPU's free memory was not read", f"{NS}.on_card_bytes": "the GGUF's size was not read", f"{NS}.kv_cache_bytes": "no KV size",
          f"{NS}.hidden_size": "no hidden width", f"{NS}.host_expert_tensor_max_bytes": "no host expert tensor",
          f"{NS}.prompt_compute_ms": "no compute time for the prompt (no measured time budget and no prompt limit)"}
  missing = [why for k, why in need.items() if out.get(k) is None]
  if not host.get("h2d_pinned_gbs"):
    missing.append("no measured pinned copy rate to the GPU (boltbeam host measures it)")
  if missing:
    out[f"{NS}.vram_why"] = "; ".join(filter(None, [out[f"{NS}.vram_why"], *missing]))
  return out, src


def facts(model:str, host:dict[str, Any], b:dict[str, Any], options:dict[str, Any]) -> dict[str, Any]:
  """The model's facts (experts per layer and those this placement keeps in host RAM, the host tables) and llama.cpp's own
  (llama_cpp.*), with "sources" and "notes". options: experts_in_ram ("all" or a layer count; default the offload
  plan's), plan (a stand-in; {} skips it), system. No GPU is touched: the plan reads saved figures only."""
  from boltbeam.report.plain import words
  from boltbeam.workflow.recommend import CACHE_RAM_DEFAULT
  w = words()["tune_llama_cpp"]
  system = options.get("system") or platform.system()
  shared = b["backend"] == "metal" or (system == "Darwin" and platform.machine() == "arm64")
  ex = expert_layers(model)
  layers = len(ex["per_layer"])
  ids = ex.get("ids") or list(range(layers))
  top = ids[-1] + 1 if ids else 0  # --n-cpu-moe at least this keeps every expert in RAM (--cpu-moe)
  plan = options.get("plan")
  if plan is None:
    plan = offload_plan(model, host["prompt_tokens"]) if not shared else {"why": "shared memory: no offload plan"}
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
  host_experts = 0 if shared else sum(b_ for l, b_ in zip(ids, ex["per_layer"]) if l < in_ram)
  tabs = host_tables(model, host["prompt_tokens"], host["page_bytes"])
  placement = "shared memory: the GPU reads the experts where they are, nothing is copied" if shared else where
  from boltbeam.workflow import tune
  have = [g for g in tune.table()["engines"][ENGINE]["groups"].values() if g["marker"] in b["features"]]
  notes = ([] if have else [w["no_patch"]]) + ([w["shared_memory"]] if shared else [])
  name = pathlib.Path(model).name
  if plan.get("ubatch_uniform") is not None and plan.get("ubatch") != plan["ubatch_uniform"] and plan.get("routing_profile"):
    p = plan["routing_profile"]
    notes.append(w["routing_changed"].format(what=rp.words(p, ENGINE).removeprefix(rp.MEASURED + " "), before=plan["ubatch_uniform"], after=plan["ubatch"]))
  notes.append(w["model_dense"].format(name=name) if not ex["experts"] else
               w["model"].format(name=name, layers=layers, experts=ex["experts"], gib=sum(ex["per_layer"]) / GiB, placement=placement, host=host_experts / GiB))
  out = {"host_expert_bytes": host_experts, "model_experts": ex["experts"], f"{NS}.host_tables_resident_bytes": tabs["resident_bytes"],
         "shared_memory": shared, f"{NS}.prompt_cache_default_bytes": CACHE_RAM_DEFAULT, f"{NS}.prompt_cache_bytes": CACHE_RAM_DEFAULT,
         f"{NS}.routing_note": routing_note(options.get("routing_profile", plan.get("routing_profile") or rp.find(model, engine=ENGINE))),
         f"{NS}.plan_ubatch": plan.get("ubatch"), f"{NS}.plan_ubatch_why": plan.get("why") or plan.get("source"),
         f"{NS}.pin_in_build": "LLAMA_WS_PIN_GIB" in b["features"], "layers": top, "layers_in_ram": 0 if shared else in_ram,
         "expert_bytes_total": sum(ex["per_layer"]), "host_tables_bytes": tabs["bytes"], "placement": placement, "notes": notes}
  if not shared:
    vr, vsrc = slot_facts(model, ex, tabs, host, in_ram=in_ram, host_experts=host_experts, plan=plan, options=options)
    out |= vr
  else:
    vsrc = {}
  out["sources"] = {**vsrc, "host_expert_bytes": f"{ex['source']}: layers 0-{in_ram - 1} ({where})" if in_ram and not shared else where,
                    "model_experts": ex["source"], f"{NS}.host_tables_resident_bytes": f"estimate: {host['prompt_tokens']} tokens x one row (at least a {host['page_bytes']} B page) per host table",
                    f"{NS}.prompt_cache_default_bytes": "llama-server --cache-ram default (8192 MiB)", f"{NS}.prompt_cache_bytes": "the --cache-ram row",
                    f"{NS}.plan_ubatch": plan.get("source") or plan.get("detail") or plan.get("why"), f"{NS}.pin_in_build": b["source"],
                    "shared_memory": f"build backend {b['backend']}, {system}"}
  return out


# --- the command and a measured point -----------------------------------------------------------------------------------

def _placement(f:dict[str, Any]) -> list[str]:
  if f.get("shared_memory") or not f.get("layers_in_ram"):
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
  asked = {x for x in extra or [] if x in ("-c", "--ctx-size")}
  ctx = [] if asked or not f.get("context_tokens") else ["-c", str(f["context_tokens"])]  # the context the KV estimate used
  argv = [b["build"], "-m", model, "-ngl", "99", *_placement(f), *ctx, *flags, *(extra or [])]
  place = _placement(f)  # where the experts sit: a run takes it as a setting, so "use these" copies it with the values that assume it
  placement = [{"name": "--cpu-moe", "value": "on"}] if place == ["--cpu-moe"] else [{"name": place[0], "value": place[1]}] if place else []
  return {"env": env, "flags": flags, "config": {}, "argv": argv, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), *argv]),
          "placement": placement}


SLOT_FAILED = re.compile(r"cudaMalloc of a [\d.]+ MiB slot failed")  # the streamer's log line (llama-pinned-copy-stream.patch)
COMPUTE_BUFFER = re.compile(r"CUDA\d+ compute buffer size\s*=\s*([\d.]+) MiB")
MEMORY_CMD = [sys.executable, "-c", "import json; from boltbeam.workflow.tune_vllm import gpu_memory; print(json.dumps(gpu_memory()))"]


def server_log_facts(text:str) -> dict[str, Any]:
  """What a llama-server log says about GPU memory: the streamer's failed slot allocations and the compute buffer it
  logged at load (None when it logged none)."""
  bufs = [float(m) for m in COMPUTE_BUFFER.findall(text)]
  return {f"{NS}.slot_alloc_failures": len(SLOT_FAILED.findall(text)),
          f"{NS}.logged_compute_buffer_bytes": int(max(bufs) * 2**20) if bufs else None}


def point(server:str, model:str, *, prompt:int, flags:list[str], env:dict[str, str], reps:int, port:int, log:pathlib.Path,
          memory:bool = False, context:int | None = None) -> dict[str, Any]:
  """One cold real-text prefill of `prompt` tokens in a fresh llama-server (the --prefill driver): {"ms" (the median client
  wall), "facts"}. memory: the driver also reads the GPU's free memory right after load and after the warm-up prompt, and
  the facts add those and what the server's log says (server_log_facts). context: the server's context, as the command
  runs it (at least the prompt plus the driver's headroom), so a fit is measured beside the KV cache the command holds."""
  from boltbeam.collectors import engine_common as ec, llama_server_decode as lsd
  from boltbeam.collectors.llama_server_prefill import CORPUS
  srv_log = log.with_suffix(".server.log")
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(port),
          "--ctx-size", str(max(prompt + lsd.HEADROOM, context or 0)), "--contexts", str(prompt), "--reps", str(reps), "--flags", json.dumps(flags),
          "--log", str(srv_log), "--prefill", "--corpus", str(CORPUS)] + (["--memory-cmd", json.dumps(MEMORY_CMD)] if memory else [])
  lines = ec.run_driver(argv, env=env, log=log)
  got = next(line for line in lines if line.get("prefill"))
  out: dict[str, Any] = {"ms": statistics.median(got.get("wall_samples_ms") or got["samples_ms"]), "facts": {}}
  if memory:
    mem = got.get("memory") or {}
    out["facts"] = {f"{NS}.free_after_load_bytes": (mem.get("loaded") or {}).get("free"), f"{NS}.free_after_warm_bytes": (mem.get("warm") or {}).get("free"),
                    **server_log_facts(srv_log.read_text(errors="replace") if srv_log.is_file() else "")}
  return out


def point_ms(server:str, model:str, **kw) -> float:
  """One cold real-text prefill of `prompt` tokens in a fresh llama-server (the --prefill driver), the median client wall ms."""
  return point(server, model, **kw)["ms"]


def launch_measure(model:str, b:dict[str, Any], f:dict[str, Any], *, port:int, out:pathlib.Path, extra:list[str] | None = None) -> Callable[..., Any]:
  """run(values=, prompt=, reps=, tag=) -> wall ms: a fresh llama-server per point with that setup. facts=True: {"ms",
  "facts"} with the GPU's free memory after load and after one prompt, and the server log's slot failures (a fit sweep)."""
  def run(*, values, prompt, reps, tag, facts=False):
    env, flags = _env_flags(values)
    got = point(b["build"], model, prompt=prompt, flags=["-ngl", "99", *_placement(f), *flags, *(extra or []), "--no-warmup"], env=env,
                reps=reps, port=port, log=out / f"{tag}.driver.log", memory=facts, context=f.get("context_tokens"))
    return got if facts else got["ms"]
  return run
