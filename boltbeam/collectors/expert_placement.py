"""Where a llama.cpp (or Strata) run kept its experts, as the run recorded it: the offload flags in its engine command
line (collectors/engine_args) and its engine settings (capture_settings.LLAMA_FLAGS). The offload limit
(workflow/offload.run_block) and the attribution of a capture (attribution.provider_trace) read it from here."""
from __future__ import annotations

import pathlib
import re
from typing import Any

from boltbeam.core.refused import Refused

_FLAGS = {"--cpu-moe": "cpu_moe", "-cmoe": "cpu_moe", "--n-cpu-moe": "n_cpu_moe", "-ncmoe": "n_cpu_moe", "--override-tensor": "ot", "-ot": "ot",
          "--moe-cache-mib": "cache_mib", "--moe-cache-cpu-share": "cpu_share", "-ngl": "ngl", "--n-gpu-layers": "ngl", "--gpu-layers": "ngl"}
_LONELY = ("cpu_moe",)  # flags with no value
PLACEMENT = ("cpu_moe", "n_cpu_moe", "override_tensor", "moe_cache_mib", "moe_cache_cpu_share")  # the engine settings that place experts in RAM


_FLAG = re.compile(r"-{1,2}[A-Za-z]")  # a flag; "-1" is a value


def flag_pairs(tokens:list[str], switches:set[str] | tuple[str, ...] = ()) -> list[tuple[str, str | None]]:
  """An engine command line as (flag, value) pairs, the one flag parser: --name=value is split first; a switch, and a
  flag followed by another flag or by nothing, has the value None; a token that is no flag's value is left out."""
  toks = [t for tok in tokens for t in (tok.split("=", 1) if tok.startswith("--") and "=" in tok else [tok])]
  out, i = [], 0
  while i < len(toks):
    t, nxt = toks[i], toks[i + 1] if i + 1 < len(toks) else None
    value = None if t in switches or nxt is None or _FLAG.match(nxt) else nxt
    if _FLAG.match(t):
      out.append((t, value))
    i += 2 if _FLAG.match(t) and value is not None else 1
  return out


def parse_flags(tokens:list[str]) -> dict[str, Any]:
  """Offload flags in an engine command line: cpu_moe, n_cpu_moe, ot (the -ot patterns), cache_mib, cpu_share."""
  out: dict[str, Any] = {}
  for t, value in flag_pairs(tokens, {f for f, k in _FLAGS.items() if k in _LONELY}):
    key = _FLAGS.get(t)
    if key is None:
      continue
    if key in _LONELY:
      out[key] = True
      continue
    if value is None: raise Refused(f"{t} has no value in the run's recorded engine arguments")
    if key == "ngl" and not value.lstrip("-").isdigit():
      continue  # "auto" and "all" put every layer on the GPU the model fits; only a count can leave layers behind
    try:
      out[key] = (out.get(key, []) + [value]) if key == "ot" else float(value) if key == "cpu_share" else int(value)
    except ValueError as exc:
      raise Refused(f"{t} {value!r} in the run's recorded engine arguments is not a number") from exc
  return out


def recorded(run) -> dict[str, Any] | None:
  """The offload a run recorded: its engine arguments (trace_request.json engine_settings: engine_args, or the keys cpu_moe,
  n_cpu_moe, override_tensor, moe_cache_mib, moe_cache_cpu_share, moe_cache_hit_rate), or a Strata fork row. None: no offload."""
  from boltbeam.core.shell import split_command
  from boltbeam.artifacts.run_io import load_manifest, optional
  s = optional(run, "trace_request.json").get("engine_settings") or optional(run, "workload_profile.json").get("engine_settings") or {}
  args = s.get("engine_args") or []
  toks = split_command(args) if isinstance(args, str) else [str(a) for a in args]
  from boltbeam.collectors.capture_settings import render_llama
  toks += render_llama({k: True if k == "cpu_moe" else s[k] for k in PLACEMENT if s.get(k) is not None and (k != "cpu_moe" or s[k])})
  flags = parse_flags(toks)
  if (left := cpu_layers(run, flags.pop("ngl", None))):  # a partial -ngl is an offload: whole layers (attention too) run on the CPU
    flags["ngl_cpu_layers"] = left
  fork = str(((optional(run, "timing_trace.json").get("rows") or [{}])[0].get("engine") or {}).get("fork") or (load_manifest(run).get("engine") or {}).get("fork") or "")
  strata = "strata" in (fork + str(s.get("engine") or "")).lower()
  if not flags and not strata:
    return None
  return {"flags": flags, "strata": strata, "hit_rate": s.get("moe_cache_hit_rate")}


def cpu_layers(run, ngl:int | None) -> int:
  """The layers a -ngl count leaves on the CPU: llama.cpp puts the last `ngl` of the n_layer + 1 (the output layer counts one)
  on the GPU (n_layer: the profile's layer_count, the header's block_count). 0 for no -ngl, for a negative count (llama.cpp's
  -1 is auto and -2 all: common/common.h n_gpu_layers, arg.cpp's "auto"/"all"), for a count that covers every layer, and for a
  model with no expert layers (the offload limit is for MoE models; a dense model's partial offload is not modeled here).
  -ngl 0 is every layer on the CPU, as llama.cpp reads it."""
  if ngl is None or ngl < 0:
    return 0
  from boltbeam.artifacts.run_io import optional
  profile = optional(run, "model_profile.json")
  experts = max([int(r["count"]) for r in profile.get("roles") or [] if r.get("n_expert") and r.get("count")] or [0])
  if not experts:
    return 0
  layers = int(profile.get("layer_count") or experts)
  return min(layers, max(0, layers + 1 - ngl))


def cpu_experts(run, profile:dict[str, Any]) -> dict[str, Any] | None:
  """The experts the CPU computes in this run (roofline/offload.cpu_share), with the profile's expert roles, or None when
  it kept no expert in RAM. An -ot pattern that moves other tensors (not every expert) has no share: it says so."""
  from boltbeam.roofline.offload import cpu_share
  experts = [r for r in profile.get("roles") or [] if r.get("n_expert") and r.get("count")]
  if not experts or not (pathlib.Path(run) / "run_manifest.json").is_file():
    return None
  if (rec := recorded(run)) is None:
    return None
  f, roles = rec["flags"], sorted({(r["role"], r["quant"]) for r in experts})
  patterns = f.get("ot", [])
  if f.get("ngl_cpu_layers"):
    return {"share": None, "low": None, "high": None, "roles": roles, "per_layer": False,
            "basis": f"-ngl leaves {f['ngl_cpu_layers']} whole layers on the CPU: their attention and experts both run there, which the expert share does not model"}
  if any("=CPU" in p and ("exps" not in p or "blk" in p) for p in patterns):
    return {"share": None, "low": None, "high": None, "roles": roles, "basis": "an -ot pattern moves tensors that are not all the experts"}
  layers = max(int(r["count"]) for r in experts)
  got = cpu_share(layers=layers, cpu_moe=bool(f.get("cpu_moe")), n_cpu_moe=f.get("n_cpu_moe"), all_in_ram=any("=CPU" in p for p in patterns),
                  cache=bool(f.get("cache_mib")), cpu_share=f.get("cpu_share", 0.0), hit_rate=rec["hit_rate"], strata=rec["strata"])
  return {**got, "roles": roles, "per_layer": bool(f.get("n_cpu_moe")) and not (f.get("cpu_moe") or patterns or f.get("cache_mib"))}
