"""The llama.cpp adapter of `boltbeam tune` (workflow/tune.py): the build read for the getenv names compiled into it, the
GGUF's expert bytes and host tables, the offload plan's placement and chunk, a measured point through the real-text
llama-server prefill driver, and the env + flags command.

The build is read, not run: a knob applies when its env-var name is a string in llama-server or the libraries beside it
(getenv literals), so a stock build gets only stock flags, and a v1 pinned-stream build no pin budget."""
from __future__ import annotations

import itertools
import json
import math
import mmap
import pathlib
import platform
import re
import statistics
import sys
from typing import Any, Callable

from boltbeam import formats
from boltbeam.collectors import capture_settings as cs
from boltbeam.collectors.replay import flag_value
from boltbeam.core.refused import Refused
from boltbeam.roofline import routing_profile as rp

ENGINE = "llama.cpp"
NS = "llama_cpp"
PLACEMENT_FACTS = ("host_expert_bytes",)  # facts that hold only with the experts placed as the command places them
FACTS = {"pin_in_build": "the build can pin host experts at load (LLAMA_WS_PIN_GIB is in it)",
         "prompt_cache_default_bytes": "llama-server's prompt cache in RAM by default (recommend.CACHE_RAM_DEFAULT)",
         "prompt_cache_bytes": "the prompt cache this setup keeps (set by the --cache-ram row)",
         "plan_ubatch": "the chunk the offload plan's prompt limit picks (workflow/offload.prefill_plan)",
         "plan_ubatch_why": "why the plan's chunk: the bytes = compute crossover and the fit (roofline/chunk_size), or why there is none",
         "plan_ubatch_output": "the plan chunk's output label: quality-gated only with gate evidence for that chunk on this model, else untested",
         "plan_ubatch_output_why": "the evidence behind that label, or why there is none",
         "plan_ubatch_candidates": "the memory fit's chunks: the plan's chunk and the smaller ones down to llama-server's default -ub, largest first",
         "plan_ubatch_gains": "each of those chunks' prompt speed over the default -ub's, from the plan's prompt limit (derived, not measured)",
         "routing_note": "a clause from the saved routing profile of this model (how concentrated its expert use is), or empty when there is none",
         "host_tables_resident_bytes": "the host-mapped input tables' rows one prompt reads (one row per token, at least a page each): an estimate",
         "on_card_bytes": "the model's weights on the GPU: the GGUF's tensor bytes less the host experts and the host-mapped tables",
         "kv_cache_bytes": "the KV cache llama.cpp allocates at the context and the command's type (load_facts: the attended layers, a sparse indexer's keys, a quantized type's dequant copy) plus the recurrent state",
         "kv_dequant_bytes": "the f16 copy of one attended layer's K and V a quantized KV cache adds per cell on a backend whose flash attention dequantizes it (kv_shape)",
         "kpool_indexer": "the model has a k-pool sparse attention indexer: its scores and selection masks add to the compute buffer per cell and chunk token",
         "kpool_ratio": "the k-pool indexer's pool ratio: cells per pooled key (the GGUF header's compress ratios)",
         "kpool_cell_bytes": "the compute buffer's bytes per cell and chunk token of a k-pool indexer's scores and masks (the stock path's, or an on mod's row)",
         "indexer_row": "the indexer cache's keys per cell, in key heads of its key length (the stock path's, or an on mod's row)",
         "kv_type": "the KV cache type the command runs, as llama-server spells it",
         "kv_dequant_types": "the block-quantized KV types this command keeps the f16 dequant copy for (an on mod may read some in place)",
         "draft_head_shared": "the draft reads the target's output head, so its own is not loaded (an on mod's row)",
         "mods_on": "the mods the command runs with on: the build has them and their switches are set (mods_on)",
         "first_layers_expert_bytes": "the experts of the model's first N layers, by N (what --n-cpu-moe N keeps in host RAM)",
         "layer_expert_bytes": "each expert layer's experts in bytes, by layer (the GGUF tensor table: what -ot keeps in host RAM per layer)",
         "expert_layer_count": "the model's layers that hold routed experts",
         "layer_reach": "per prompt chunk size, the share of each layer's experts a chunk reaches (a measured routing profile; None: uniform)",
         "layer_reach_source": "where the per-layer reach came from (a measured routing profile, or uniform routing assumed)",
         "draft_expert_bytes": "one routed expert of the draft head, its rows of every expert tensor and its router row (the draft file's tensors over its expert count)",
         "draft_input_row_bytes": "the draft's input rows per batch token: the target's hidden state, every stream, in f32",
         "recurrent_state_bytes": "one sequence's recurrent state (the model's profile)",
         "model_context": "the context the model was trained to (the GGUF header's context_length)",
         "kv_layer_elems": "one attended layer's K and V elements per cell (the model's profile)", "kv_stream_layers": "the layers the kv-stream mod streams: the target's sparse-attention layers (the GGUF header's compress_ratios above 0) of an architecture it streams (its spend row's archs), else 0",
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
         "vram_why": "which of the facts the slot count reads could not be read, or empty",
         "fa_modes": "the values this build's -fa takes (its --help)",
         "flash_attn": "the flash attention setting the command runs: the build's default, else the -fa row's value",
         "kv_types": "the KV cache types this build takes (its --help) that it has flash attention kernels for, K and V alike (its GPU library), with the default first",
         "kv_quantized_types": "those of the KV types that are block-quantized",
         "v_quant_needs_fa": "the build refuses a block-quantized V cache with flash attention off (its own error message is in it)",
         "kv_sizes": "the KV cache's bytes per token at each of those types",
         "thread_candidates": "the thread counts a decode sweep tries: the cores sharing one L3, the physical cores and the logical CPUs",
         "fa_default": "the build's own -fa default (its --help)", "kv_default": "the build's own KV cache type default (its --help)",
         "kv_setup": "the KV cache type the Recommended setup's command uses (screen tune --kv), when this build has flash attention kernels for it",
         "kv_setup_from": "where the setup's KV type came from (the Recommended setup's --kv, or the measured runs of this model)",
         "kv_setup_output": "its output label: untested, no agreement gate ran for it", "kv_setup_output_why": "why it is untested",
         "threads_default": "the build's own -t default (its --help)",
         "per_layer_embd_bytes": "the per-layer token embedding llama.cpp keeps in host RAM and gathers a row of per token and layer",
         "mtp_head": "the multi-token prediction head the command loads (--draft, else the one beside the model)",
         "mtp_k_measured": "the MTP draft count measured on this machine for this model", "mtp_why": "why there is no measured draft count",
         "mtp_says": "where the measured draft count was measured, its speed and spread", "vocab_size": "the model's vocabulary, in tokens",
         "server_slots": "llama-server's slots: -np in --flags, else the requests the setup is stated for (the command passes them as -np)", "draft_tokens": "the draft tokens per step the command runs (--flags' draft count, else the MTP draft count row's; 0 without)",
         "draft_from_flags": "--flags give the draft count (the command's own wins over the row's)",
         "kv_type_bytes": "bytes per element of each KV cache type this build takes (ggml's block layouts)",
         "kv_elems_per_token": "the KV cache's elements one token adds: K and V of the attended layers (the model's profile) and a sparse indexer's keys (the GGUF header; kv_shape)",
         "draft_kv_elems_per_token": "those the draft head adds: its layers at the model's elements per attended layer",
         "draft_weight_bytes": "the draft head's file as it sits on the GPU: its tensors less the host-RAM input tables, and less its output head when the draft reads the target's",
         "mtp_ks": "the draft counts the memory fit may choose: none and each one measured, when a measured one is used",
         "mtp_gains": "each measured draft count's speed over no draft (at the pick's depth, else the nearest it was measured at)",
         "mtp_gain_words": "where each of those speeds was measured, in words",
         "model_file_bytes": "the model's GGUF files on disk, every shard: what the engine maps when it maps the model",
         "load_none_ram_bytes": "what --load-mode none reads into RAM at load: the host experts and the host tables it does not leave mapped (engines.llama.cpp.load_none)",
         "ckpt_pool_bytes": "the pinned context-checkpoint pool of the ckpt-pinned mod: checkpoints + 1 buffers of one slot's recurrent state (an estimate); 0 when the build lacks the mod"}
NAMES = re.compile(rb"(?:LLAMA|GGML)_[A-Z0-9_]{2,}")  # getenv literals in the build
READ_FROM = "llama.cpp b12257-19-g7592b61d3 (2026-11-10)"  # the build LITERALS and FA_KERNELS were read from: a later build that renames them reads as lacking them, said
LITERALS = {"draft-mtp": rb"adding speculative implementation 'draft-mtp'",  # common/speculative.cpp: llama-server registers the MTP draft type
            "quantized-v-needs-flash-attention": rb"quantized V cache requires flash_attn"}  # src/llama-context.cpp: refused with flash attention off
FA_KERNELS = (re.compile(rb"flash_attn_ext_vecILi\d+ELi\d+EL9ggml_type(\d+)ELS0_(\d+)E"),  # ggml-cuda/fattn-vec.cuh: one CUDA/HIP template instance per (K, V) type id
              re.compile(rb"kernel_flash_attn_ext_vec_([a-z0-9]+(?:_[0-9])?)_dk\d+"))  # ggml-metal: one Metal kernel per type
HELP_TIMEOUT_S = 60  # --help loads no model; a build that has not answered in a minute is hung, and its help is read as empty
GiB = 1 << 30
NP = ("-np", "--parallel")  # llama-server's slots: the command passes the requests the setup is stated for (tools/server/server.cpp:126-134)
CHECK_TOOL = "llama-perplexity (it scores the logits)"  # the output check's tool, in tune's words
def kv_stream_archs() -> tuple[str, ...]:
  """The graphs the kv-stream mod streams: its spend row's archs (arkey_ie_mods llama.cpp 1.0.0 patch 20, llama-kv-stream.cpp plan)."""
  from boltbeam.workflow import tune
  return tuple(next((r.get("archs") or [] for r in tune.table()["engines"]["llama.cpp"]["spend"]["rows"] if r["name"] == "kv_stream"), []))
GATE_CHUNKS = 1  # the output check reads one chunk of the tuned context per document: its second half is what is scored, the depth tune is for


def _flag_of(fact:str) -> str:
  """The flag whose default this fact holds: the row of the knob table that reads its default from it (default_from)."""
  from boltbeam.workflow import tune
  return next(k["name"] for k in tune.knobs(ENGINE) if k.get("default_from") == f"{NS}.{fact}")


def _row() -> dict[str, Any]:
  """This engine's row of the knob table: the names of its draft flags and its per-layer host table."""
  from boltbeam.workflow import tune
  return tune.table()["engines"][ENGINE]


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
    raise Refused(f"no llama-server found ({lsd.WHERE}); pass --build PATH")
  p = server_of(server)
  if not p.is_file():
    raise Refused(f"no llama-server at {p}; pass --build PATH or put llama-server on PATH")
  real = p.resolve()
  dirs = {real.parent, real.parent.parent / "lib"}
  for dep in linked(real):  # a packaged build (Homebrew) keeps ggml and its loadable backends (libexec) apart
    dirs |= {dep.parent, dep.parent.parent / "libexec"}
  files = [real] + sorted(f for d in dirs if d.is_dir() for f in d.iterdir() if f.is_file() and re.search(r"^lib(ggml|llama)\S*\.(so|dylib)", f.name))
  seen, names, libs, where, fa = set(), set(), [], {}, set()
  from boltbeam.workflow import mods
  marks = {m["marker"]: m["marker"].encode() for m in mods.mods(ENGINE) if m.get("marker") and not NAMES.fullmatch(m["marker"].encode())}  # a mod known by a message, not a getenv name
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
        for name, lit in {**LITERALS, **marks}.items():
          if mm.find(lit) >= 0:
            where.setdefault(name, f.name)
        fa |= fa_kernel_types(mm)
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
          "fa_kv_types": sorted(fa) or None, "source": f"env-var names in {len(libs)} file(s) of the build (no run); messages and kernel names as {READ_FROM} has them"}


def fa_kernel_types(blob) -> set[str]:
  """The KV cache types a GPU library has flash attention kernels for with K and V of that same type (FA_KERNELS), by
  ggml's type names (profile/gguf.GGML_TYPE_NAMES)."""
  from boltbeam.profile.gguf import GGML_TYPE_NAMES
  out = {GGML_TYPE_NAMES.get(int(k), "").lower() for k, v in FA_KERNELS[0].findall(blob) if k == v}
  return (out | {m.decode() for m in FA_KERNELS[1].findall(blob)}) - {""}


def build_help(server:str, run_command=None) -> str:
  """llama-server --help, the build's own list of its flags and the values each takes, with every GPU hidden from it (it
  loads no model and touches no GPU); empty when it does not answer."""
  import os, subprocess
  from boltbeam.target import vendors
  hide = {name: "" for v in vendors.VENDORS.values() for name in v.visible}  # each vendor runtime's own variables that hide its GPUs
  try:
    got = (run_command or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=HELP_TIMEOUT_S, env={**os.environ, **hide})))([server, "--help"])
  except (OSError, subprocess.SubprocessError):
    return ""
  return (got.stdout or "") + (got.stderr or "")


def _help_block(text:str, flag:str) -> list[str]:
  """A flag's lines in --help: its own and the indented ones after it."""
  lines = text.splitlines()
  at = next((i for i, ln in enumerate(lines) if re.match(rf"\s*(?:\S+,\s*)*{re.escape(flag)}[,\s]", ln + " ")), None)
  return [] if at is None else [lines[at]] + list(itertools.takewhile(lambda ln: not re.match(r"\s*-", ln) and ln.strip(), lines[at + 1:]))


def help_default(text:str, flag:str) -> str | None:
  """The default a flag's --help states, (default: x) or default: 'x'."""
  m = re.search(r"default:\s*'?(-?[a-z0-9_]+)", " ".join(_help_block(text, flag)))
  return m.group(1) if m else None


def help_choices(text:str, flag:str) -> list[str]:
  """The values a flag takes, read from --help: its [a|b|c] or its "allowed values: a, b, ..." (wrapped lines joined)."""
  block = _help_block(text, flag)
  if not block:
    return []
  if m := re.search(r"\[([a-z0-9_]+(?:\|[a-z0-9_]+)+)\]", block[0]):
    return m.group(1).split("|")
  joined = " ".join(x.strip() for x in block)
  m = re.search(r"allowed values:\s*(.*?)(?:\(default|$)", joined)
  return [v.strip() for v in m.group(1).split(",") if v.strip()] if m else []


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


def kv_shape(profile:dict[str, Any], header:dict[str, Any]) -> dict[str, Any]:
  """llama.cpp's KV cache per token, in elements, as the stock path keeps it: K and V of the attended layers
  (prefill_work's rule), plus a sparse attention indexer's own cache, one key head of its key length per indexer layer,
  as wide as the engine row's stock indexer_row (src/llama-memory-hybrid-idx.cpp:55; measured: 204 MiB of q8_0 indexer
  cache at 65,536 cells beside 816 MiB of KV, Flash-Next IQ1_M on b11571). dequant: one attended layer's K and V, the f16
  copy CUDA flash attention makes of a quantized cache over the context (measured: 2,048 bytes per q8_0 cell over f16,
  same model). idx: the indexer's key elements per cell before that width; ratio: its pool ratio. A mod that changes
  either is a row of engines.llama.cpp.mod_memory (memory_facts). header: offload.header_table."""
  from boltbeam.roofline import prefill_work as pw
  meta, arch = header["meta"], header["meta"].get("general.architecture")
  attn = pw.kv_bytes_per_token(profile, element_bytes=1) or 0.0
  layers = pw.attention_layer_count(profile)
  idx_layers = sum(1 for n, _, _ in header["tensors"] if ".indexer.k_proj." in n)
  ratio = max((r or 0 for r in (meta.get(f"{arch}.attention.compress_ratios") or [])), default=0)
  idx = idx_layers * int(meta.get(f"{arch}.attention.indexer.key_length") or 0)
  kpool = ratio > 1 and idx > 0
  row = memory_facts({"attn": attn, "idx": idx, "kpool": kpool, "ratio": ratio, "dequant": 0.0}, kv_type="f16", backend=None)
  return {"elems": row[f"{NS}.kv_elems_per_token"], "attn": attn, "idx": idx, "kpool": kpool, "ratio": ratio,
          "dequant": attn / layers if layers and attn else 0.0}


def dequant_bytes(shape:dict[str, Any], backend:str | None) -> float:
  """Bytes per cell a quantized KV cache adds as the f16 copy of one attended layer, on a backend whose flash attention makes one."""
  from boltbeam.target import vendors
  return shape["dequant"] * 2 if any(v.ggml_kv_dequant for v in vendors.VENDORS.values() if v.ggml_backend == backend) else 0.0


def mods_on(b:dict[str, Any]) -> list[str]:
  """The mods a run of the command has on, by name: the build has the mod (its marker is compiled in) and its switch is
  set, as a run on a modded build sets it (mods.run_env of its record, or of its mods found by their markers: mods.found)
  or in this environment, at its on value. A mod on
  by default (switch unset) reads as off: the conservative side, it is charged the stock path's memory."""
  import os
  from boltbeam.workflow import mods
  feats, env = set(b.get("features") or {}), {**mods.run_env(mods.found(ENGINE, b.get("build"), b.get("features")) or {}), **os.environ}
  return list(dict.fromkeys(m["name"] for m in mods.mods(ENGINE) if m.get("marker") in feats and (sw := mods.switch(m)) in env  # once a name: versions share it
          and (env[sw] == mods.run_on(m) if mods.run_on(m) else env[sw] not in ("", "0"))))  # a value-valued switch (a path): set and not 0


def memory_facts(shape:dict[str, Any], *, kv_type:str, backend:str | None, mods:list[str] | tuple = (),
                 extra:dict[str, Any] | None = None) -> dict[str, Any]:
  """The per-cell memory facts of the command: the stock path's (the engine row's mod_memory.stock over the shape) with
  each on mod's row over them (mod_memory.rows, in the order given), then the KV's elements per cell. extra: further facts
  a row may read (the draft head's bytes). The one place a mod's memory effect is applied; the rows are data."""
  from boltbeam.workflow import tune
  rule = _row()["mod_memory"]
  f = {f"{NS}.kpool_indexer": shape["kpool"], f"{NS}.kpool_ratio": shape["ratio"] or 1, f"{NS}.kv_type": kv_type,
       f"{NS}.kv_dequant_bytes": dequant_bytes(shape, backend), **(extra or {})}
  allowed = set(f) | {f"{NS}.{n}" for n in FACTS}
  for name, expr in rule["stock"].items():
    f[name] = tune.evaluate(expr, f, allowed)
  for m in mods:
    for name, expr in (rule["rows"].get(m) or {}).get("sets", {}).items():
      f[name] = tune.evaluate(expr, f, allowed)
  f[f"{NS}.kv_elems_per_token"] = shape["attn"] + shape["idx"] * f[f"{NS}.indexer_row"]
  return f


def header_of(model:str | pathlib.Path) -> dict[str, Any]:
  """The model file's header (offload.header_table); an empty one when it cannot be read (kv_shape then sees no indexer)."""
  from boltbeam.workflow.offload import header_table
  try:
    return header_table(str(pathlib.Path(model).expanduser()))
  except (OSError, ValueError, KeyError):
    return {"meta": {}, "tensors": []}


def load_facts(profile:dict[str, Any], header:dict[str, Any], ex:dict[str, Any], *, kv_type:str, context:int, chunk:int,
               backend:str | None, slots:int = 1, mods:list[str] | tuple = ()) -> dict[str, Any]:
  """The facts llama.cpp's after_load formula (data/engine_knobs.json engines.llama.cpp.after_load) reads from the model
  and the command, besides the GPU's memory and the weights on the card: the one source tune's memory fit and the
  Recommended setup's context both read. kv_cache_bytes: the cells at the KV type, a quantized type's dequant copy (on a
  backend that makes one) and each slot's recurrent state; the expert shapes (ex: expert_layers) the compute buffer's
  chunk term reads. mods: the mods the command runs with on (mods_on), whose memory rows replace the stock path's terms."""
  from boltbeam.roofline import prefill_work as pw
  block = (formats.row(formats.canonical(kv_type)) or {"block": (1, 2)})["block"]
  f = memory_facts(kv_shape(profile, header), kv_type=str(kv_type).lower(), backend=backend, mods=mods)
  dequant = f[f"{NS}.kv_dequant_bytes"]
  return {**{k: v for k, v in f.items() if k != f"{NS}.kv_type"},
          f"{NS}.kv_cache_bytes": int(context * (f[f"{NS}.kv_elems_per_token"] * block[1] / block[0] + (dequant if block[0] > 1 else 0))
                                      + (pw.recurrent_state_bytes(profile) or 0) * slots),
          f"{NS}.chunk_tokens": chunk, f"{NS}.experts_used": ex.get("experts_used"), f"{NS}.expert_ffn": ex.get("expert_ffn"),
          f"{NS}.hidden_size": int(profile["hidden_size"]) if profile.get("hidden_size") else None, "context_tokens": context}


def after_load(facts:dict[str, Any]) -> float | None:
  """GPU memory free once llama-server has loaded the model: the engine's after_load formula over these facts (tune.after_load)."""
  from boltbeam.workflow import tune
  return tune.after_load(facts, ENGINE, tune.table()).get("gpu_after_load_bytes")


def host_tables(source:str | pathlib.Path, prompt:int, page:int) -> dict[str, Any]:
  """The input tables llama.cpp keeps mapped in host RAM (recommend.HOST_ONLY: token_embd, per_layer_token_embd): their bytes
  and the rows one prompt reads, one row per token and table, at least a page each (rows sit apart): an estimate.
  lazy_bytes: those that stay mapped even under --load-mode none (engines.llama.cpp.load_none)."""
  from boltbeam.profile import gguf
  from boltbeam.workflow import tune
  from boltbeam.workflow.recommend import HOST_ONLY, _gguf_files
  rule = tune.table()["engines"][ENGINE]["load_none"]
  total = resident = data = lazy = 0
  by: dict[str, int] = {}
  for f in _gguf_files(pathlib.Path(source).expanduser()):
    _, ts, start = gguf.read_gguf_layout(f)
    data += f.stat().st_size - start
    offs = sorted((o, name, dims) for name, dims, _, o in ts)
    for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [f.stat().st_size - start]):
      if name.split(".")[0] in HOST_ONLY:
        rows = max(1, math.prod(dims[1:]))
        total += nxt - o
        lazy += nxt - o if name.split(".")[0] in rule["lazy_tables"] and nxt - o > rule["lazy_min_bytes"] else 0
        by[name.split(".")[0]] = by.get(name.split(".")[0], 0) + nxt - o
        resident += min(nxt - o, prompt * max((nxt - o) // rows, page))
  return {"bytes": total, "resident_bytes": resident, "file_bytes": data, "by_table": by, "lazy_bytes": lazy}


def offload_plan(model:str, prompt:int, draft:str | None = None, context:int | None = None, features=None) -> dict[str, Any]:
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
    got = screen.recommended(prof, tgt, engine=ENGINE, context=context or prompt, offload=True, draft=draft, turn=prompt, build_features=features)
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
  if ch := pre.get("chunk"):  # the derived chunk (roofline/chunk_size): its words and its output label
    out |= {"why": "; ".join(n for n in pre["notes"] if n.startswith(("chunk ", "fit: ", "placement "))), "output": ch["output"]}
    if ch["placement"]["label"] != "measured":  # an unmeasured placement is said, never put in the session command
      out.pop("layers_in_ram")
      out["placement_why"] = "the derived placement is an estimate with no timed arm"
  else:
    out["why"] = words_of_tie()
  out["chunk_tok_s"] = {int(c["ubatch"]): float(c["tok_s"]) for c in pre.get("chunks") or [] if c.get("tok_s")}
  chunk = next((c for c in pre.get("chunks") or [] if c.get("ubatch") == out["ubatch"]), None)
  if chunk and (chunk.get("ms") or {}).get("gpu"):
    out["gpu_ms"] = float(chunk["ms"]["gpu"])  # the prompt limit's GPU time at that chunk: compute, the copy apart
  flat_pre = ((flat or {}).get("offload") or {}).get("prefill") or {}
  if saved and flat_pre.get("status") == "recommended":
    out |= {"ubatch_uniform": int(flat_pre["ubatch"]), "routing_profile": saved}
  return out


def chunk_fit(plan:dict[str, Any]) -> dict[str, Any]:
  """The chunks the memory fit may pick (the -ub row's vram): the plan's chunk and each smaller one of its prompt limit
  down to llama-server's default, largest first, with each one's prompt speed over the default's; none without a plan."""
  from boltbeam.workflow.offload import knob_default
  rate, top, low = plan.get("chunk_tok_s") or {}, plan.get("ubatch"), int(knob_default(ENGINE, "-ub"))
  if not top or not rate.get(low):
    return {f"{NS}.plan_ubatch_candidates": None, f"{NS}.plan_ubatch_gains": None}
  keep = sorted((u for u in rate if low <= u <= top), reverse=True)
  return {f"{NS}.plan_ubatch_candidates": [str(u) for u in keep], f"{NS}.plan_ubatch_gains": {str(u): rate[u] / rate[low] for u in keep}}


def words_of_tie() -> str:
  from boltbeam.report.plain import words
  return words()["tune_llama_cpp"]["ubatch_tie"]


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


def gpu_backend(b:dict[str, Any]) -> str | None:
  """The ggml backend whose GPUs the build runs on (its detect()), None for a CPU build or an unknown one: every GPU."""
  return None if b.get("backend") in (None, "cpu") else b["backend"]


def slot_facts(model:str, ex:dict[str, Any], tabs:dict[str, Any], host:dict[str, Any], *, in_ram:int, host_experts:int, plan:dict[str, Any],
               options:dict[str, Any], kv:str | None = None, header:dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
  """The facts the streamer's slot count reads, read and not combined (the rule is the row's, data/engine_knobs.json): the
  host expert tensors (count, the largest, their layers), the GPU's free memory now, what the model takes of it on the
  card (weights, KV at the context and the recurrent state), the shapes the compute buffer scales with, the chunk, and
  the GPU's compute time for the prompt. vram_why names any that could not be read. options: gpu ({free, total}),
  profile, time_budget (a measured boltbeam.time_budget.v1 record of this setup's prompt). kv: the KV type the command
  runs (None: f16, the build's type unread); the KV cache and the per-cell terms are load_facts', as the Recommended setup's."""
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
    gpu = gpu_memory(every=True, ggml_backend=options.get("ggml_backend"))  # llama-server splits its layers over every GPU of its backend
    if options.get("gpu_free") is not None:  # a recorded reading given in its place (--gpu-free-mib)
      gpu = {**gpu, "free": options["gpu_free"], "free_min": options["gpu_free"]}
  out["gpu_total_bytes"], out["gpu_free_bytes"] = gpu.get("total"), gpu.get("free")
  out["gpu_count"], out["gpu_free_min_bytes"] = gpu.get("cards", 1 if gpu else None), gpu.get("free_min", gpu.get("free"))
  src |= dict.fromkeys(("gpu_total_bytes", "gpu_free_bytes", "gpu_count", "gpu_free_min_bytes"), "the GPU's telemetry source (NVML / amdgpu sysfs)")
  profile = options.get("profile")
  try:
    if profile is None:
      from boltbeam.profile.loaders import profile_from_model
      profile = profile_from_model(model, None).to_json()
    from boltbeam.roofline import prefill_work as pw
    src[f"{NS}.hidden_size"] = "the model's profile"
    if pw.kv_bytes_per_token(profile) is None and pw.recurrent_state_bytes(profile) is None:
      raise ValueError("the profile gives no attention or recurrent shape for a KV size")
    ctx = int(host.get("context_tokens") or host["prompt_tokens"])
    got = load_facts(profile, header if header is not None else header_of(model), ex, kv_type=kv or "f16", context=ctx, chunk=out[f"{NS}.chunk_tokens"],
                     backend=options.get("ggml_backend"), mods=options.get("mods") or ())
    out |= {k: v for k, v in got.items() if k != "context_tokens"}
    at = f"the command's type, {kv}" if kv else "f16 (the build's cache type could not be read)"
    src[f"{NS}.kv_cache_bytes"] = f"tune_llama_cpp.load_facts at {at}: {ctx} cells x (KV and indexer keys per cell + a quantized type's dequant copy) + the recurrent state"
  except Exception as exc:  # noqa: BLE001  an unsized piece is a stated unknown, never a crash; the raw text is provenance
    out[f"{NS}.vram_why"] = "the model's shapes could not be read for a KV size"
    src[f"{NS}.vram_why"] = f"{type(exc).__name__}: {exc}"
  if tabs.get("file_bytes") is not None:
    out[f"{NS}.on_card_bytes"] = max(tabs["file_bytes"] - host_experts - tabs["bytes"], 0)
    src[f"{NS}.on_card_bytes"] = "the GGUF files' tensor data less the host experts and the host-mapped tables"
  total, on_card, kv_b = out.get("gpu_total_bytes"), out.get(f"{NS}.on_card_bytes"), out.get(f"{NS}.kv_cache_bytes")
  sized = [n[0] for n in (("-c", "--ctx-size"), ("-ctk", "--cache-type-k"), ("-ctv", "--cache-type-v"), NP) if flag_value([str(x) for x in options.get("extra") or []], n)]
  if sized:  # the KV here is one slot's, at f16 and the context given: flags that change these (slots split the context, each its own state) are not seen, so no fit is claimed
    src[f"{NS}.kv_cache_bytes"] = src.get(f"{NS}.kv_cache_bytes", "") + f"; the fit of one request on the GPU is not checked: --flags sets {', '.join(sized)}"
  elif not options.get("spend") and total and on_card is not None and kv_b is not None and on_card + kv_b > total:  # one request cannot fit this GPU: say so, never clamp the slots to 4 and go on (the spend fit's placement answers it instead)
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
    plan = offload_plan(model, int(host["prompt_tokens"]), options.get("draft"), int(host.get("context_tokens") or host["prompt_tokens"]),
                        b.get("features")) if not shared else {"why": "shared memory: no offload plan"}
  given = options.get("experts_in_ram")
  # in_ram: --n-cpu-moe's N, the model's layers 0..N-1 (dense layers count); the plan counts expert layers
  if given == "all":
    in_ram, where = top, "--experts-in-ram all (given)"
  elif given is not None:
    in_ram, where = min(max(int(given), 0), top), f"--experts-in-ram {given} (given)"
  elif "layers_in_ram" in plan:
    in_ram, where = cpu_moe_layers(ids, plan["layers_in_ram"]), plan["source"]
  else:
    in_ram, where = 0, f"none given and no offload plan ({plan.get('placement_why') or plan.get('why')}): every expert on the GPU"
  host_experts = 0 if shared else sum(b_ for l, b_ in zip(ids, ex["per_layer"]) if l < in_ram)
  tabs = host_tables(model, host["prompt_tokens"], host["page_bytes"])
  placement = "shared memory: the GPU reads the experts where they are, nothing is copied" if shared else where
  from boltbeam.workflow import mods, tune
  have = [g for g in tune.table()["engines"][ENGINE]["groups"].values() if tune.group_markers(ENGINE, g) & set(b["features"])]
  named = sorted({m["id"].split("/", 1)[1] for g in tune.table()["engines"][ENGINE]["groups"].values() for m in mods.named(ENGINE, g["mod"])})
  none = [] if have else [w["no_patch_modded" if mods.found(ENGINE, b.get("build"), b.get("features")) else "no_patch"].format(mods=", ".join(named))]
  notes = none + ([w["shared_memory"]] if shared else []) + ([options["kv_ask"]] if options.get("kv_ask") else [])
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
         f"{NS}.plan_ubatch_output": (plan.get("output") or {}).get("output"), f"{NS}.plan_ubatch_output_why": (plan.get("output") or {}).get("why"),
         **chunk_fit(plan),
         f"{NS}.pin_in_build": "LLAMA_WS_PIN_GIB" in b["features"], "layers": top, "layers_in_ram": 0 if shared else in_ram,
         "expert_bytes_total": sum(ex["per_layer"]), "host_tables_bytes": tabs["bytes"], "placement": placement, "notes": notes}
  lf, lsrc = placed_facts(model, host_experts, b["features"], tabs=tabs, draft=options.get("draft"))
  out |= lf
  header = options["header"] if "header" in options else header_of(model)  # a test's stand-in, else the file's
  on = mods_on(b)  # the mods the command runs with: their memory rows replace the stock path's terms
  out[f"{NS}.mods_on"] = on
  dr, dsrc = decode_facts(model, host, b, {**options, "mods": on}, tabs, header=header)
  out |= dr
  if not shared:  # the KV it sizes is at the type the command starts from: the Recommended setup's, else the build's own (decode_facts)
    kv_type = dr[f"{NS}.kv_setup"] or dr[f"{NS}.kv_default"]
    vr, vsrc = slot_facts(model, ex, tabs, host, in_ram=in_ram, host_experts=host_experts, plan=plan, options={"ggml_backend": gpu_backend(b), **options, "mods": on},
                          kv=kv_type, header=header)
    out |= vr
    try:
      out |= spend_facts(ex, options.get("profile") or _profile(model), header, dr[f"{NS}.mtp_head"],
                         routing=options["routing_profile"] if "routing_profile" in options else rp.find(model, engine=ENGINE))
    except (OSError, ValueError, KeyError) as exc:  # the spend fit then says which fact it lacks
      out["spend_why"] = f"{type(exc).__name__}: {exc}"
  else:
    vsrc = {}
  vsrc = {**vsrc, **dsrc}
  out["sources"] = {**vsrc, "host_expert_bytes": f"{ex['source']}: layers 0-{in_ram - 1} ({where})" if in_ram and not shared else where,
                    "model_experts": ex["source"], f"{NS}.host_tables_resident_bytes": f"estimate: {host['prompt_tokens']} tokens x one row (at least a {host['page_bytes']} B page) per host table",
                    f"{NS}.prompt_cache_default_bytes": "llama-server --cache-ram default (8192 MiB)", f"{NS}.prompt_cache_bytes": "the --cache-ram row",
                    f"{NS}.plan_ubatch": plan.get("source") or plan.get("detail") or plan.get("why"), f"{NS}.pin_in_build": b["source"],
                    "shared_memory": f"build backend {b['backend']}, {system}", **lsrc}
  return out


def placed_facts(model:str, host_experts:float, features=None, *, tabs:dict[str, Any] | None = None,
                 profile:dict[str, Any] | None = None, draft:str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
  """The facts of the rows that hang on where the experts sit and on RAM alone (knob rows marked placed: tune.placed, the
  offload page): the model's files on disk (and the draft's: -lm applies to -md too), what --load-mode none reads into
  RAM at that placement (the host experts, the host tables of both files it does not leave mapped), the pinned
  checkpoint pool, the prompt cache at its default (the --cache-ram row sets it in tune) and whether the build pins a
  second copy of the host experts. The engine's rules are engines.llama.cpp.load_none. features: the build's (None: not
  read: no pin, the pool counted). A fact that cannot be read is None (the row keeps its default), said in its source."""
  from boltbeam.workflow import tune
  from boltbeam.workflow.recommend import _gguf_files, CACHE_RAM_DEFAULT
  from boltbeam.roofline import prefill_work as pw
  from boltbeam.profile.loaders import profile_from_model
  rule = tune.table()["engines"][ENGINE]["load_none"]
  files = [model, *([draft] if draft else [])]
  out: dict[str, Any] = {f"{NS}.model_file_bytes": None, f"{NS}.load_none_ram_bytes": None, f"{NS}.prompt_cache_bytes": CACHE_RAM_DEFAULT,
                         f"{NS}.pin_in_build": features is not None and "LLAMA_WS_PIN_GIB" in features,
                         f"{NS}.ckpt_pool_bytes": 0 if features is not None and rule["ckpt_pool_marker"] not in features else None}
  src = {f"{NS}.model_file_bytes": "the GGUF files' sizes, the draft's included",
         f"{NS}.load_none_ram_bytes": "the host experts + the host tables (the draft's too) under the lazy floor (engines.llama.cpp.load_none)",
         f"{NS}.ckpt_pool_bytes": "(n_ctx_checkpoints + 1) x the profile's recurrent state (engines.llama.cpp.load_none)"}
  try:  # an unread fact is a stated unknown (None: the row keeps its default), never a crash
    out[f"{NS}.model_file_bytes"] = sum(f.stat().st_size for m in files for f in _gguf_files(pathlib.Path(m).expanduser()))
    ts = [tabs if tabs is not None and m == model else host_tables(m, 1, 4096) for m in files]
    if all(t.get("lazy_bytes") is not None for t in ts):
      out[f"{NS}.load_none_ram_bytes"] = int(host_experts + sum(t["bytes"] - t["lazy_bytes"] for t in ts))
  except Exception as exc:  # noqa: BLE001
    src[f"{NS}.model_file_bytes"] += f"; not read: {type(exc).__name__}: {exc}"
  if out[f"{NS}.ckpt_pool_bytes"] is None:
    try:  # a model with no recurrent state may still checkpoint (sliding-window attention): unknown, not 0
      state = pw.recurrent_state_bytes(profile or profile_from_model(model, None).to_json())
      out[f"{NS}.ckpt_pool_bytes"] = int((rule["ctx_checkpoints"] + 1) * state) if state else None
    except Exception as exc:  # noqa: BLE001
      src[f"{NS}.ckpt_pool_bytes"] += f"; not read: {type(exc).__name__}: {exc}"
  return out, src


def spend_facts(ex:dict[str, Any], profile:dict[str, Any] | None, header:dict[str, Any], head:str | None,
                routing:dict[str, Any] | None = None) -> dict[str, Any]:
  """The facts the spend rows (engines.llama.cpp.spend) read beyond tune's: the experts of the first N model layers by N,
  each expert layer's own bytes (the GGUF tensor table: a mixed-type model's layers differ), the share of a layer's
  experts a prompt chunk reaches per chunk size (a measured routing profile; None: uniform routing), the draft's bytes
  per routed expert, its input rows per batch token, one sequence's recurrent state and the KV the kv-stream mod streams."""
  from boltbeam.roofline import prefill_work as pw
  ids, per = ex.get("ids") or [], ex.get("per_layer") or []
  top = (ids[-1] + 1) if ids else 0
  first = {n: sum(b for i, b in zip(ids, per) if i < n) for n in range(top + 1)}
  meta, arch = header["meta"], header["meta"].get("general.architecture")
  streams = int(meta.get(f"{arch}.hyper_connection.count") or 1)
  reach = None
  if routing and routing.get("layers"):
    order = [r["layer"] for r in routing["layers"]]
    reach = {w: {l: x / routing["n_expert"] for l, x in zip(order, rp.reached(routing, w))} for w in rp.CHUNKS}
  out = {f"{NS}.first_layers_expert_bytes": first, f"{NS}.layer_expert_bytes": dict(zip(ids, per)), f"{NS}.expert_layer_count": len(ids),
         f"{NS}.layer_reach": reach, f"{NS}.layer_reach_source": rp.words(routing) if routing else rp.UNIFORM,
         f"{NS}.recurrent_state_bytes": (pw.recurrent_state_bytes(profile) if profile else None),
         f"{NS}.draft_input_row_bytes": 4 * int(profile["hidden_size"]) * streams if profile and profile.get("hidden_size") else None,
         f"{NS}.draft_expert_bytes": None, f"{NS}.model_context": meta.get(f"{arch}.context_length"),
         f"{NS}.kv_layer_elems": (pw.kv_bytes_per_token(profile, element_bytes=1) or 0) / max(pw.attention_layer_count(profile) or 0, 1) if profile else None,
         f"{NS}.kv_stream_layers": sum(1 for r in (meta.get(f"{arch}.attention.compress_ratios") or [])[:int(meta.get(f"{arch}.block_count") or 0)] if (r or 0) > 0) if arch in kv_stream_archs() else 0}
  if head:
    out[f"{NS}.draft_expert_bytes"] = draft_expert_bytes(pathlib.Path(head))
  return out


def draft_expert_bytes(head:pathlib.Path) -> float | None:
  """One routed expert of a draft file: the bytes of its tensors with an expert dimension (the expert stacks and the
  router's rows) over its expert count."""
  from boltbeam.profile import gguf
  meta, ts, start = gguf.read_gguf_layout(head)
  n = int(meta.get(f"{meta.get('general.architecture')}.expert_count") or 0)
  if not n:
    return None
  size = head.stat().st_size - start
  offs = sorted((o, name, dims) for name, dims, _, o in ts)
  return sum(nxt - o for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [size]) if dims and dims[-1] == n) / n


def gpu_total() -> int | None:
  """Every GPU of the build's backend, summed: what a stored spend record's GPU size is matched against."""
  from boltbeam.workflow.tune_vllm import gpu_memory
  return gpu_memory(every=True).get("total")


def spend_value(fact:str, path:str) -> Any:
  """What a spend row reads from a file a command names (its read.file_flags): "experts", the expert count of a draft file."""
  from boltbeam.profile import gguf
  if fact != "experts":
    raise ValueError(f"no file fact {fact!r}")
  meta = gguf.read_gguf_layout(pathlib.Path(path).expanduser())[0]
  return int(meta.get(f"{meta.get('general.architecture')}.expert_count") or 0)


def _profile(model:str) -> dict[str, Any] | None:
  from boltbeam.profile.loaders import profile_from_model
  try:
    return profile_from_model(model, None).to_json()
  except Exception:  # noqa: BLE001  no profile: the facts that need it are unknown
    return None


def draft_on_card(head:pathlib.Path, mods:list[str] | tuple = ()) -> int:
  """The draft head file's tensors on the GPU: every tensor less the input tables llama.cpp keeps in host RAM
  (recommend.HOST_ONLY), and less its output head when an on mod has the draft read the target's (a mod_memory row
  setting draft_head_shared)."""
  from boltbeam.profile import gguf
  from boltbeam.workflow.min_memory import kind
  from boltbeam.workflow.recommend import HOST_ONLY
  _, ts, start = gguf.read_gguf_layout(head)
  size = head.stat().st_size - start
  offs = sorted((o, name, dims) for name, dims, _, o in ts)
  by: dict[str, int] = {}
  for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [size]):
    k = "host" if name.split(".")[0] in HOST_ONLY else kind(name, list(dims))
    by[k] = by.get(k, 0) + nxt - o
  shared = memory_facts({"attn": 0.0, "idx": 0, "kpool": False, "ratio": 0, "dequant": 0.0}, kv_type="f16", backend=None, mods=mods)[f"{NS}.draft_head_shared"]
  return int(size - by.get("host", 0) - (by.get("head", 0) if shared else 0))


def decode_facts(model:str, host:dict[str, Any], b:dict[str, Any], options:dict[str, Any], tabs:dict[str, Any],
                 header:dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
  """The facts the decode rows read, each from where it is: the -fa, -ctk and -t values and defaults this build takes
  (its --help; options help stands in), the KV types it has flash attention kernels for (detect), the KV cache's elements
  per token (the model's profile) and its bytes at each type (ggml's block layouts), the thread counts to try (the CPU's
  topology), the per-layer embedding kept in host RAM, and the MTP head with the draft count measured for it (options:
  draft, the head file; machine_facts, where the measurement is kept)."""
  from boltbeam.collectors import llama_mtp
  from boltbeam.profile import gguf
  from boltbeam.report.plain import words
  from boltbeam.workflow import recommend as rec
  w = words()["tune_llama_cpp"]
  text = options["help"] if "help" in options else build_help(b["build"])
  said = "llama-server --help" if text else "llama-server --help did not answer"
  kv_flag, fa_flag, t_flag = (_flag_of(x) for x in ("kv_default", "fa_default", "threads_default"))
  np, dk = (flag_value([str(x) for x in options.get("extra") or []], n) for n in (NP, (_row()["draft"]["count"],)))  # the user's own, which win
  default = help_default(text, kv_flag)
  layout = {t: tuple(r["block"]) if (r := formats.row(formats.canonical(t))) else None for t in help_choices(text, kv_flag)}
  unknown = [t for t, lay in layout.items() if lay is None]  # a type ggml's block table here does not hold: its size is not known, so it is not a candidate
  kinds = [t for t, lay in layout.items() if lay and (b.get("fa_kv_types") is None or t in b["fa_kv_types"])]
  kinds = ([default] if default in kinds else []) + [t for t in kinds if t != default]
  out: dict[str, Any] = {f"{NS}.fa_modes": help_choices(text, fa_flag), f"{NS}.flash_attn": help_default(text, fa_flag), f"{NS}.fa_default": help_default(text, fa_flag),
                         f"{NS}.kv_default": default, f"{NS}.threads_default": help_default(text, t_flag),
                         f"{NS}.v_quant_needs_fa": "quantized-v-needs-flash-attention" in b["features"], f"{NS}.kv_types": kinds,
                         f"{NS}.kv_quantized_types": [t for t in kinds if layout[t][0] > 1], "kv_bytes_per_token": None, f"{NS}.kv_sizes": None,
                         f"{NS}.per_layer_embd_bytes": (tabs.get("by_table") or {}).get(_row()["per_layer_table"], 0), f"{NS}.vocab_size": None,
                         f"{NS}.server_slots": host["served_requests"] if np is None else int(np) if np.isdigit() and int(np) >= 1 else None,
                         f"{NS}.draft_tokens": 0 if dk is None else int(dk) if dk.isdigit() else None, f"{NS}.draft_from_flags": dk is not None}
  src = {f"{NS}.fa_modes": said, f"{NS}.flash_attn": f"{said}: its default, until the -fa row sets it", f"{NS}.v_quant_needs_fa": b["source"],
         f"{NS}.kv_types": f"{said}, " + ("the flash attention kernels in the build's GPU library" if b.get("fa_kv_types") else "no flash attention kernels read in the build: every type it takes")
         + (f"; not {', '.join(unknown)}: no block layout for them here" if unknown else ""), f"{NS}.per_layer_embd_bytes": "the GGUF tensor table",
         f"{NS}.server_slots": "the requests the setup is stated for (the command passes -np)" if np is None else f"--flags -np {np}" + ("" if out[f"{NS}.server_slots"] else
                                ": not a slot count of 1 or more (-1 is llama-server's auto: 4 slots, one KV), so the slots are unknown")}
  try:
    profile = options.get("profile")
    if profile is None:
      from boltbeam.profile.loaders import profile_from_model
      profile = profile_from_model(model, None).to_json()
    from boltbeam.roofline import prefill_work as pw
    shape = kv_shape(profile, header if header is not None else header_of(model))  # K and V of every attended layer, and a sparse indexer's keys
    on = options.get("mods") or ()
    mem = {t: memory_facts(shape, kv_type=t, backend=gpu_backend(b), mods=on) for t in kinds}  # the per-cell terms at each type, with the on mods' rows
    elems = memory_facts(shape, kv_type=default or "f16", backend=gpu_backend(b), mods=on)[f"{NS}.kv_elems_per_token"] or None
    per_layer = shape["dequant"] or None  # one attended layer's K and V: a draft head's layer
    out[f"{NS}.kv_dequant_bytes"] = dequant_bytes(shape, gpu_backend(b))  # the -ctk row's cost reads it on every backend (0 where none is made)
    out[f"{NS}.kv_dequant_types"] = [t for t in kinds if layout[t][0] > 1 and mem[t][f"{NS}.kv_dequant_bytes"] > 0]  # where the copy stays (an on mod reads some in place)
    out[f"{NS}.vocab_size"], src[f"{NS}.vocab_size"] = profile.get("vocab_size"), "the model's profile"
  except Exception as exc:  # noqa: BLE001  an unread shape is a stated unknown; the raw text is provenance
    elems, per_layer, src["kv_bytes_per_token"] = None, None, f"the model's shapes could not be read: {type(exc).__name__}: {exc}"
  size = lambda t: elems * layout[t][1] / layout[t][0]  # noqa: E731  bytes per token at a type: its block's bytes over its block's elements
  out[f"{NS}.kv_type_bytes"], out[f"{NS}.kv_elems_per_token"] = {t: layout[t][1] / layout[t][0] for t in kinds}, elems
  if elems and default in kinds:
    out["kv_bytes_per_token"] = size(default)
    out[f"{NS}.kv_sizes"] = ", ".join(w["kv_size"].format(type=t, bytes=size(t)) for t in kinds)
    src["kv_bytes_per_token"] = f"roofline/prefill_work.kv_bytes_per_token's elements (the attended layers) at the build's default type, {default}"
  setup = options.get("kv")  # the Recommended setup's type (--kv), else the measured runs' (tune.kv_of): the -ctk row starts from it
  out[f"{NS}.kv_setup"] = setup if setup in kinds else None
  out[f"{NS}.kv_setup_from"] = options.get("kv_from") or w["kv_setup_from"]
  out[f"{NS}.kv_setup_output"], out[f"{NS}.kv_setup_output_why"] = "untested", w["kv_setup_untested"]
  src[f"{NS}.kv_setup"] = (f"--kv {setup}: {out[f'{NS}.kv_setup_from']}" if setup in kinds else
                           f"--kv {setup}: not a type this build has flash attention kernels for, so not used" if setup else "no --kv given")
  topo = {"l3": host.get("cpu_l3_cores"), "physical": host.get("cpu_physical_cores"), "logical": host.get("cpu_logical_cores")}
  out[f"{NS}.thread_candidates"] = sorted({int(v) for v in topo.values() if v})
  src[f"{NS}.thread_candidates"] = "the CPU's topology (target/host.cpu_cores): " + ", ".join(f"{k} {v}" for k, v in topo.items() if v)
  given = options.get("draft")
  if given and not pathlib.Path(given).expanduser().is_file():
    raise Refused(w["no_head"].format(head=given))
  heads = [pathlib.Path(given).expanduser()] if given else llama_mtp.heads(model)
  head = str(heads[0].absolute()) if len(heads) == 1 else None
  layers = 0
  if head:
    kv = gguf.read_gguf_layout(heads[0])[0]
    layers = int(kv.get(f"{kv.get('general.architecture')}.nextn_predict_layers") or 0)
  pick = rec.mtp_tune(options.get("machine_facts"), ENGINE, str(pathlib.Path(model).expanduser()), host.get("context_tokens"), b.get("backend"), b.get("build"),
                      seqs=host["served_requests"]) if head else \
    {"k": None, "says": None, "why": w["heads"].format(heads=", ".join(h.name for h in heads)) if heads else w["no_heads"]}
  out |= {"model_draft_layers": layers, f"{NS}.mtp_head": head, f"{NS}.mtp_k_measured": pick["k"], f"{NS}.mtp_why": pick["why"], f"{NS}.mtp_says": pick["says"]}
  gains = {str(k): g for k, g in sorted((pick.get("gains") or {}).items())} if pick["k"] and dk is None else {}  # the fit's draft counts: only where one is used, never the flags' own
  out |= {f"{NS}.draft_kv_elems_per_token": per_layer * layers if per_layer else None, f"{NS}.draft_weight_bytes": draft_on_card(heads[0], options.get("mods") or ()) if head else None,
          f"{NS}.mtp_ks": [None, *gains] if gains else [], f"{NS}.mtp_gains": {k: g[0] for k, g in gains.items()},
          f"{NS}.mtp_gain_words": {k: w["mtp_gain"].format(k=k, speedup=g[0], context=g[1]) for k, g in gains.items()}}
  src |= {"model_draft_layers": f"{heads[0].name}: its nextn_predict_layers" if head else "no MTP head (--draft, or one beside the model)",
          f"{NS}.mtp_head": "--draft (given)" if given else "the one MTP head beside the model (collectors/llama_mtp.heads)"}
  return out, src


# --- the command and a measured point -----------------------------------------------------------------------------------

def _placement(f:dict[str, Any]) -> list[str]:
  if f.get("shared_memory") or not f.get("layers_in_ram"):
    return []
  return ["--cpu-moe"] if f["layers_in_ram"] >= f["layers"] else ["--n-cpu-moe", str(f["layers_in_ram"])]


def _env_flags(values:list[dict[str, Any]], f:dict[str, Any] | None = None) -> tuple[dict[str, str], list[str]]:
  """The env block and the flags; -b is raised to -ub when -ub is larger than llama-server's default batch. A flag row's
  `with` adds the flags it goes with (their values filled from the row's value and the facts), and its `with_draft` those
  for the draft model, only when a draft model (-md) is in the command."""
  from boltbeam.workflow.tune import say
  env = {v["name"]: v["value"] for v in values if v["kind"] == "env"}
  flags: list[str] = []
  given = {v["name"] for v in values}
  fill = lambda t, v: say(t, {**(f or {}), "value": v["value"]})  # noqa: E731
  for v in values:
    if v["kind"] == "flag":
      flags += [v["name"], v["value"], *(x for k, t in (v.get("with") or {}).items() for x in (k, fill(t, v)))]
      if v["name"] == "-ub" and "-b" not in given and int(v["value"]) > 2048:
        flags += ["-b", v["value"]]
  if _row()["draft"]["model"] in flags:
    flags += [x for v in values if v["kind"] == "flag" for k, t in (v.get("with_draft") or {}).items() for x in (k, fill(t, v))]
  return env, flags + [v["name"] for v in values if v["kind"] == "switch" and v["value"] == "on"]  # no value: after the pairs


def render(values:list[dict[str, Any]], f:dict[str, Any], b:dict[str, Any], model:str, extra:list[str] | None = None) -> dict[str, Any]:
  env, flags = _env_flags(values, f)
  ctx = [] if flag_value(extra or [], ("-c", "--ctx-size")) or not f.get("context_tokens") else ["-c", str(f["context_tokens"])]  # the context the KV estimate used
  ctx += [] if flag_value(extra or [], NP) or not f.get("served_requests") else ["-np", str(f["served_requests"])]  # not llama-server's auto: 4 slots, KV shared
  argv = [b["build"], "-m", model, *cs.all_layers(extra), *_placement(f), *ctx, *flags, *(extra or [])]
  place = _placement(f)  # where the experts sit: a run takes it as a setting, so "use these" copies it with the values that assume it
  placement = [{"name": "--cpu-moe", "value": "on"}] if place == ["--cpu-moe"] else [{"name": place[0], "value": place[1]}] if place else []
  return {"env": env, "flags": flags, "config": {}, "argv": argv, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), *argv]),
          "placement": placement}


SLOT_FAILED = re.compile(r"cudaMalloc of a [\d.]+ MiB slot failed")  # the streamer's log line (arkey_ie_mods llama.cpp/7592b61d3/pinned-copy-stream)
COMPUTE_BUFFER = re.compile(r"[A-Za-z]+\d+ compute buffer size\s*=\s*([\d.]+) MiB")  # any GPU backend: CUDA0, ROCm0, MTL0
def memory_cmd(backend:str | None) -> list[str]:
  """The driver's --memory-cmd: the tightest card of the build's backend, so a fit and its lowest reading are per card (a
  sum over cards hides one card at zero behind another's free memory)."""
  return [sys.executable, "-c", "import json; from boltbeam.workflow.tune_vllm import gpu_memory; "
          f"print(json.dumps(gpu_memory(every=True, ggml_backend={backend!r}, tightest=True)))"]


def server_log_facts(text:str) -> dict[str, Any]:
  """What a llama-server log says about GPU memory: the streamer's failed slot allocations and the compute buffer it
  logged at load (None when it logged none)."""
  bufs = [float(m) for m in COMPUTE_BUFFER.findall(text)]
  return {f"{NS}.slot_alloc_failures": len(SLOT_FAILED.findall(text)),
          f"{NS}.logged_compute_buffer_bytes": int(max(bufs) * 2**20) if bufs else None}


def point(server:str, model:str, *, prompt:int, flags:list[str], env:dict[str, str], reps:int, port:int, log:pathlib.Path,
          memory:bool = False, backend:str | None = None, context:int | None = None, slots:int = 1) -> dict[str, Any]:
  """One cold real-text prefill of `prompt` tokens in a fresh llama-server (the --prefill driver): {"ms" (the median client
  wall), "facts"}. memory: the driver also reads the GPU's free memory right after load and after the warm-up prompt, and
  the facts add those and what the server's log says (server_log_facts). context: the server's context, as the command
  runs it (at least the prompt plus the driver's headroom), so a fit is measured beside the KV cache the command holds."""
  from boltbeam.collectors import engine_common as ec, llama_server_decode as lsd
  from boltbeam.collectors.llama_server_prefill import CORPUS
  srv_log = log.with_suffix(".server.log")
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(port),
          "--ctx-size", str(max(prompt + lsd.HEADROOM, context or 0)), "--contexts", str(prompt), "--reps", str(reps), "--flags", json.dumps(flags),
          "--log", str(srv_log), "--prefill", "--corpus", str(CORPUS), "--slots", str(slots)] + (["--memory-cmd", json.dumps(memory_cmd(backend))] if memory else [])
  lines = ec.run_driver(argv, env=env, log=log)
  got = next(line for line in lines if line.get("prefill"))
  out: dict[str, Any] = {"ms": statistics.median(got.get("wall_samples_ms") or got["samples_ms"]), "facts": {}}
  if memory:
    mem = got.get("memory") or {}
    out["facts"] = {f"{NS}.free_after_load_bytes": (mem.get("loaded") or {}).get("free"), f"{NS}.free_after_warm_bytes": (mem.get("warm") or {}).get("free"),
                    **server_log_facts(srv_log.read_text(errors="replace") if srv_log.is_file() else "")}
  return out


def decode_point(server:str, model:str, *, flags:list[str], env:dict[str, str], port:int, log:pathlib.Path, context:int | None = None,
                 slots:int = 1, memory:bool = False, backend:str | None = None) -> dict[str, Any]:
  """Decode speed at one request in a fresh llama-server, as the MTP measurement times it (the driver's --mtp-prompts
  mode, collectors/llama_mtp): the shipped short prompts and the real-text long corpus cut to the setup's context, each
  with its task, a reply of collectors/mtp.TOKENS each, the server's own tok/s, held to the loop guard (collectors/mtp.guard:
  a reply that repeats itself is refused). The draft count is the setup's (the engine row's draft count flag, else none). {tokens, depths:
  {depth: {context, per_prompt}}}; memory: and facts, the GPU's free memory right after load and the least after any request (tune's
  gpu_free_after_load_bytes, gpu_free_lowest_bytes: the memory fit's budget)."""
  from boltbeam.collectors import engine_common as ec, llama_mtp, llama_server_decode as lsd, mtp
  prompts = mtp.prompt_set()
  count = _row()["draft"]["count"]
  k = int(flags[flags.index(count) + 1]) if count in flags else 0
  long_tokens = max(0, (context or 0) - prompts["room"] - mtp.TOKENS)  # the long prompts fill the setup's context, as screen mtp --context sizes them
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(port),
          "--ctx-size", str(max(prompts["room"] + long_tokens + mtp.TOKENS + lsd.HEADROOM, context or 0)), "--flags", json.dumps([*flags, "--metrics"]),
          "--mtp-prompts", prompts["path"], "--k", str(k), "--tokens", str(mtp.TOKENS), "--long-tokens", str(long_tokens), "--long-corpus", str(mtp.LONG),
          "--log", str(log.with_suffix(".server.log")), "--end-ids", json.dumps(llama_mtp.end_ids(model)), "--deadline-s", str(llama_mtp.deadline_s(prompts)), "--slots", str(slots)] + (["--memory-cmd", json.dumps(memory_cmd(backend))] if memory else [])
  lines = ec.run_driver(argv, env=env, log=log, timeout_s=llama_mtp.deadline_s(prompts) + llama_mtp.DRIVER_GRACE_S)
  rows, eng = [x for x in lines if "decode_tok_s" in x], next(x["engine"] for x in lines if "engine" in x)
  if why := mtp.guard(rows, eng["real_text"]) + list(mtp.ended(rows, eng["real_text"]).values()):  # paired across settings: an early end refuses
    raise Refused("; ".join(why))
  mem = next((x["memory"] for x in lines if "memory" in x), None) if memory else None
  loaded = (mem or {}).get("loaded") or {}
  return {"tokens": mtp.TOKENS, "depths": {d: {"context": statistics.median(r["context"] for r in rows if r["depth"] == d),
                                               "per_prompt": {r["prompt"]: r["decode_tok_s"] for r in rows if r["depth"] == d}}
                                           for d in sorted({r["depth"] for r in rows})},
          **({"facts": {"gpu_free_after_load_bytes": loaded.get("free"), "gpu_free_lowest_bytes": (mem.get("lowest") or loaded).get("free")}} if mem else {})}


def help_flags(text:str) -> set[str]:
  """Every flag a tool's --help lists on its option lines."""
  return {m for ln in text.splitlines() if re.match(r"\s*-", ln) for m in re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", ln.split("  ")[0])}


def gate_flags(values:list[dict[str, Any]], f:dict[str, Any], accepted:set[str]) -> tuple[dict[str, str], list[str]]:
  """The setup as the output check's tool runs it: the placement and each of the command's flags the tool accepts (its
  --help: the server's own and the draft model's are not its; a draft cache's flags only beside a draft model it loads)."""
  took = lambda d: {k: t for k, t in (d or {}).items() if k in accepted}  # noqa: E731
  env, flags = _env_flags([{**v, "with": took(v.get("with")), "with_draft": took(v.get("with_draft"))} for v in values
                           if v["kind"] == "env" or v["name"] in accepted], f)
  return env, [*_placement(f), *flags]


def launch_measure(model:str, b:dict[str, Any], f:dict[str, Any], *, port:int, out:pathlib.Path, extra:list[str] | None = None) -> Callable[..., Any]:
  """run(values=, prompt=, reps=, tag=) -> wall ms: a fresh llama-server per point with that setup. facts=True: {"ms",
  "facts"} with the GPU's free memory after load and after one prompt, and the server log's slot failures (a fit sweep).
  decode=True: decode speed per real-text prompt at one request (decode_point). agree=setup values, texts=, context=:
  Speed up's accuracy check (llama-perplexity, one pair per document, one chunk of the context each) of these values
  against those, on one model; refused when the two do not differ or the tool does not take a flag they differ in.
  floor=True, texts=, context=: the engine's own noise at these values (quantization/speed_up.noise_floor, Speed up's
  function and cache: the same check of the values against themselves, with the engine's declared change)."""
  def run(*, values, prompt, reps, tag, facts=False, decode=False, agree=None, texts=None, context=None, floor=False):
    if floor:
      from boltbeam.quantization import speed_up
      from boltbeam.quantization.head_writers import gguf_quantize as gq
      env, flags = gate_flags(values, f, help_flags(build_help(gq.tool(b["build"], "llama-perplexity"))))
      return speed_up.noise_floor(pathlib.Path(model), ENGINE, gq, b["build"], texts, src_flags=flags, dst_flags=flags, env=env, ctx=context, chunks=GATE_CHUNKS)
    env, flags = _env_flags(values, f)
    if agree is not None:
      from boltbeam.quantization.head_writers import gguf_quantize as gq
      accepted = help_flags(build_help(gq.tool(b["build"], "llama-perplexity")))
      a, c = ({v["name"]: v for v in vals} for vals in (agree, values))
      differ = [a.get(n) or c[n] for n in sorted(a.keys() | c.keys()) if (a.get(n) or {}).get("value") != (c.get(n) or {}).get("value")]
      if not differ:
        raise Refused("the output check would compare a setting with itself")
      if any(r["kind"] == "env" for r in differ):
        raise Refused("the output check runs both settings in one environment, and these two differ in their environment variables")
      draft = _row()["draft"]["model"] in accepted  # no draft model there: no draft cache
      if missing := [x for r in differ for x in (r["name"], *(r.get("with") or {}), *((r.get("with_draft") or {}) if draft else ())) if x not in accepted]:
        raise Refused(f"llama-perplexity does not take {', '.join(missing)}, so the answers with it cannot be checked")
      base_env, base = gate_flags(agree, f, accepted)
      cand = gate_flags(values, f, accepted)[1]
      return gq.accuracy(model, model, b["build"], texts, src_flags=tuple(base), dst_flags=tuple(cand), env=base_env, ctx=context, chunks=GATE_CHUNKS)
    if decode:
      return decode_point(b["build"], model, flags=[*cs.all_layers(extra), *_placement(f), *flags, *(extra or [])], env=env, port=port,
                          log=out / f"{tag}.driver.log", context=f.get("context_tokens"), slots=f["served_requests"], memory=facts, backend=gpu_backend(b))
    got = point(b["build"], model, prompt=prompt, flags=[*cs.all_layers(extra), *_placement(f), *flags, *(extra or []), "--no-warmup"], env=env,
                reps=reps, port=port, log=out / f"{tag}.driver.log", memory=facts, context=f.get("context_tokens"), slots=f["served_requests"], backend=gpu_backend(b))
    return got if facts else got["ms"]
  return run
