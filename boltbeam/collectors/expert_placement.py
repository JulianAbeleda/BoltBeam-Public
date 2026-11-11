"""Where a llama.cpp (or Strata) run kept its experts, as the run recorded it: the offload flags in its engine command
line (collectors/engine_args) and its engine settings (capture_settings.LLAMA_FLAGS). The offload limit
(workflow/offload.run_block) and the attribution of a capture (attribution.provider_trace) read it from here."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.core.refused import Refused

_FLAGS = {"--cpu-moe": "cpu_moe", "-cmoe": "cpu_moe", "--n-cpu-moe": "n_cpu_moe", "-ncmoe": "n_cpu_moe", "--override-tensor": "ot", "-ot": "ot",
          "--moe-cache-mib": "cache_mib", "--moe-cache-cpu-share": "cpu_share"}
_LONELY = ("cpu_moe",)  # flags with no value
PLACEMENT = ("cpu_moe", "n_cpu_moe", "override_tensor", "moe_cache_mib", "moe_cache_cpu_share")  # the engine settings that place experts in RAM


def parse_flags(tokens:list[str]) -> dict[str, Any]:
  """Offload flags in an engine command line: cpu_moe, n_cpu_moe, ot (the -ot patterns), cache_mib, cpu_share. A flag
  given as --name=value is split first."""
  toks = [t for tok in tokens for t in (tok.split("=", 1) if tok.startswith("--") and "=" in tok else [tok])]
  out: dict[str, Any] = {}
  for i, t in enumerate(toks):
    key = _FLAGS.get(t)
    if key is None:
      continue
    if key in _LONELY:
      out[key] = True
      continue
    if i + 1 >= len(toks): raise Refused(f"{t} has no value in the run's recorded engine arguments")
    try:
      out[key] = (out.get(key, []) + [toks[i + 1]]) if key == "ot" else float(toks[i + 1]) if key == "cpu_share" else int(toks[i + 1])
    except ValueError as exc:
      raise Refused(f"{t} {toks[i + 1]!r} in the run's recorded engine arguments is not a number") from exc
  return out


def recorded(run) -> dict[str, Any] | None:
  """The offload a run recorded: its engine arguments (trace_request.json engine_settings: engine_args, or the keys cpu_moe,
  n_cpu_moe, override_tensor, moe_cache_mib, moe_cache_cpu_share, moe_cache_hit_rate), or a Strata fork row. None: no offload."""
  import shlex
  from boltbeam.artifacts.run_io import load_manifest, optional
  s = optional(run, "trace_request.json").get("engine_settings") or optional(run, "workload_profile.json").get("engine_settings") or {}
  args = s.get("engine_args") or []
  toks = shlex.split(args) if isinstance(args, str) else [str(a) for a in args]
  from boltbeam.collectors.capture_settings import render_llama
  toks += render_llama({k: True if k == "cpu_moe" else s[k] for k in PLACEMENT if s.get(k) is not None and (k != "cpu_moe" or s[k])})
  flags = parse_flags(toks)
  fork = str(((optional(run, "timing_trace.json").get("rows") or [{}])[0].get("engine") or {}).get("fork") or (load_manifest(run).get("engine") or {}).get("fork") or "")
  strata = "strata" in (fork + str(s.get("engine") or "")).lower()
  if not flags and not strata:
    return None
  return {"flags": flags, "strata": strata, "hit_rate": s.get("moe_cache_hit_rate")}


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
  if any("=CPU" in p and ("exps" not in p or "blk" in p) for p in patterns):
    return {"share": None, "low": None, "high": None, "roles": roles, "basis": "an -ot pattern moves tensors that are not all the experts"}
  layers = max(int(r["count"]) for r in experts)
  got = cpu_share(layers=layers, cpu_moe=bool(f.get("cpu_moe")), n_cpu_moe=f.get("n_cpu_moe"), all_in_ram=any("=CPU" in p for p in patterns),
                  cache=bool(f.get("cache_mib")), cpu_share=f.get("cpu_share", 0.0), hit_rate=rec["hit_rate"], strata=rec["strata"])
  return {**got, "roles": roles, "per_layer": bool(f.get("n_cpu_moe")) and not (f.get("cpu_moe") or patterns or f.get("cache_mib"))}
