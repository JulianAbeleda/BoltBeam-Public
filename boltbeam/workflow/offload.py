"""Models past GPU memory: the recommended setup when the experts live in host RAM with a cache on the card, and how a
run that did so reads against the offload limit (roofline/offload.py). Shapes come from a header or are stated as an
estimate; the host's RAM and PCIe come from target/host.py; the hit rate is a sweep, never fitted to a speed."""
from __future__ import annotations

from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.report.plain import words as plain_words
from boltbeam.roofline import offload as ol, offload_shapes as shapes

RAM_SHARE = 0.85  # of the machine's RAM the experts may fill: an assumed share, the rest is the OS, the engine and the page cache
GIB = 2**30


def _gb(n:float) -> str:
  return f"{n / 1e9:.1f} GB"


def feature_gaps(engine:str, flags:list[str]) -> list[str]:
  """What the engine's pinned build lacks of these flags (data/engine_pins.json features: the first build number with
  each), as sentences; empty when the pin has them all or the engine records no feature table."""
  from boltbeam.collectors.providers import engine_pin
  pin = engine_pin(engine) or {}
  tag = str(pin.get("tag") or "")
  built = int(tag[1:]) if tag[1:].isdigit() else None
  return [f"{f} needs a llama.cpp build of b{want['since']} or newer ({'master' if not want['release'] else 'release ' + want['release'] + ' and master'}); "
          f"BoltBeam's pin {tag} does not have it" + ("; release v0.9.0 does not either" if not want["release"] else "")
          for f, want in (pin.get("features") or {}).items() if f in flags and built is not None and built < want["since"]]


def plan(sh:dict[str, Any], *, usable_vram:float, kv_bytes:float, ram_total:float, vram_gbs:float, host:dict[str, Any],
         engine:str = "llama.cpp", context:int | None = None) -> dict[str, Any]:
  """The offload setup for a model whose weights do not fit usable VRAM: the cache size from the free VRAM after the dense
  parts and the KV, the CPU share that balances RAM against PCIe, and the limit as a band over hit rates. A Refused says
  why it cannot be done (dense parts and KV past the card; weights past VRAM + RAM)."""
  dense, total_exp = sh["dense_bytes"], sh["expert_total"]
  ram = RAM_SHARE * ram_total
  if dense + kv_bytes >= usable_vram:
    raise Refused(f"the dense parts ({_gb(dense)}) and the KV cache for {context} tokens ({_gb(kv_bytes)}) do not fit the usable "
                  f"GPU memory ({_gb(usable_vram)}), so no expert cache can be kept on the card")
  if dense + total_exp > usable_vram + ram:
    raise Refused(f"the weights need {_gb(dense + total_exp)} ({_gb(dense)} dense, {_gb(total_exp)} experts); this machine has "
                  f"{_gb(usable_vram)} usable GPU memory and {_gb(ram)} usable RAM ({RAM_SHARE:.0%} of {_gb(ram_total)}): {_gb(usable_vram + ram)} in all")
  cache = min(usable_vram - dense - kv_bytes, total_exp)
  mib = int(cache // 2**20)
  s = round(host["ram_gbs"] / (host["ram_gbs"] + host["pcie_gbs"]), 2)  # the share that loads RAM and PCIe equally
  common = dict(dense_bytes=dense, expert_bytes=sh["expert_bytes"], vram_gbs=vram_gbs, ram_gbs=host["ram_gbs"], pcie_gbs=host["pcie_gbs"],
                estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"))
  rates = shapes.hit_rates(cache, total_exp)
  band, plain = ol.band(rates, cpu_share=s, **common), ol.band(rates, cpu_share=0.0, **common)  # 0.0: llama.cpp's default share
  flags = ["--cpu-moe", "--moe-cache-mib", str(mib), "--moe-cache-cpu-share", f"{s:g}"] if engine == "llama.cpp" else None
  spill = max(total_exp - ram, 0.0)
  notes = list(band[0]["notes"]) + ([f"the experts ({_gb(total_exp)}) are more than the usable RAM ({_gb(ram)}): {_gb(spill)} would be read from disk, "
                                     "which this limit does not cover; the speed is below it"] if spill else [])
  tok, lo_hi = [r["overlap"]["tok_s"] for r in band], shapes.span(band)
  say = (f"at most {lo_hi['low']:.0f}–{lo_hi['high']:.0f} tokens a second, depending on how often the needed parts are already on the card"
         + (" (estimate: the model's shapes are not read from a header)" if sh["estimated"] else "") + ". " + plain_words()["offload"]["decode_tuned"])
  return {"schema": "boltbeam.offload_setup.v1", "status": "recommended", "estimated": sh["estimated"], "notes": notes, "engine": engine,
          "shapes": {k: sh[k] for k in ("source", "dense_bytes", "expert_bytes", "expert_total", "experts", "top_k", "layers")},
          "cache_mib": mib, "cache_share_of_experts": cache / total_exp, "cpu_share": s, "ram_usable_bytes": ram, "host": host, "hit_rates": rates,
          "band": [{"hit_rate": r["hit_rate"], "ms": r["ms"], "bound_by": r["bound_by"], "tok_s": r["overlap"]["tok_s"],
                    "tok_s_no_overlap": r["no_overlap"]["tok_s"]} for r in band],
          "tok_s": lo_hi, "tok_s_default_share": shapes.span(plain), "flags": flags, "build_gaps": feature_gaps(engine, flags or []),
          "context": context, "says": say,
          "arithmetic": [f"GPU: ({_gb(dense)} dense + hit x {_gb(sh['expert_bytes'])} experts + the uploaded misses) / {vram_gbs:.0f} GB/s",
                         f"CPU: {s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {min(host['ram_gbs'], host.get('cpu_gbs') or host['ram_gbs']):.1f} GB/s (RAM; CPU dequant rate unmeasured)",
                         f"PCIe: {1 - s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {host['pcie_gbs']:.1f} GB/s ({host['pcie_source']})",
                         "the limit is the slowest of the three; hit rates: the cache's share of the experts, then 0.8 to 0.99 (a sweep, not an estimate)"]}


def refusal(reason:str) -> dict[str, Any]:
  return {"schema": "boltbeam.offload_setup.v1", "status": "refused", "reason": reason, "says": f"This model cannot be offloaded here: {reason}."}


def for_setup(profile:dict[str, Any], provider:str, *, usable:float, kv_bytes:float, vram_gbs:float | None, pci_bus_id:str | None,
              context:int, host:dict[str, Any] | None = None, estimate:dict[str, Any] | None = None) -> dict[str, Any]:
  """recommend.setup's offload block: the plan, or the plain reason there is none. Nothing here raises."""
  from boltbeam.target import host as hostmod
  try:
    if provider != "llama.cpp":
      raise Refused(f"{provider} has no expert-cache offload BoltBeam can set up; llama.cpp has (--engine llama.cpp)")
    if not vram_gbs:
      raise Refused("this GPU has no measured memory bandwidth, so there is no limit to state")
    sh = shapes.moe_split(profile["source"], estimate)
    host = host or hostmod.facts(pci_bus_id)
    return plan(sh, usable_vram=usable, kv_bytes=kv_bytes, ram_total=host["ram_bytes"], vram_gbs=vram_gbs, host=host, engine=provider, context=context)
  except Refused as exc:
    return refusal(str(exc))


def settings(moe:dict[str, Any]) -> list[dict[str, Any]]:
  """The offload flags as setup rows (flag, why, when), beside the engine's own."""
  return [{"flag": "--cpu-moe", "recommended": True, "changes": "every expert stays in host RAM (the dense parts stay on the card). " + plain_words()["offload"]["decode_tuned"], "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-mib {moe['cache_mib']}", "recommended": True, "changes": "the card keeps this many MiB of the experts already used: the free memory after the dense parts and the KV",
           "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-cpu-share {moe['cpu_share']:g}", "recommended": True, "changes": "the share of cache misses the CPU computes instead of uploading; this balances RAM against PCIe",
           "when": "the weights do not fit GPU memory" + ("; " + moe["build_gaps"][0] if moe["build_gaps"] else "")}]


# --- a run that offloaded its experts, read against the offload limit -------------------------------------------------

_FLAGS = {"--cpu-moe": "cpu_moe", "-cmoe": "cpu_moe", "--n-cpu-moe": "n_cpu_moe", "-ncmoe": "n_cpu_moe", "--override-tensor": "ot", "-ot": "ot",
          "--moe-cache-mib": "cache_mib", "--moe-cache-cpu-share": "cpu_share"}
_LONELY = ("cpu_moe",)  # flags with no value


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
  toks += (["--cpu-moe"] if s.get("cpu_moe") else []) + [x for k, f in (("n_cpu_moe", "--n-cpu-moe"), ("override_tensor", "-ot"), ("moe_cache_mib", "--moe-cache-mib"),
                                                                        ("moe_cache_cpu_share", "--moe-cache-cpu-share")) if s.get(k) is not None for x in (f, str(s[k]))]
  flags = parse_flags(toks)
  fork = str(((optional(run, "timing_trace.json").get("rows") or [{}])[0].get("engine") or {}).get("fork") or (load_manifest(run).get("engine") or {}).get("fork") or "")
  strata = "strata" in (fork + str(s.get("engine") or "")).lower()
  if not flags and not strata:
    return None
  return {"flags": flags, "strata": strata, "hit_rate": s.get("moe_cache_hit_rate")}


def placement(run, profile:dict[str, Any]) -> dict[str, Any] | None:
  """A run whose engine arguments nobody recorded: llama.cpp (or Strata) can keep the experts in RAM, and without the arguments the
  report cannot say where the weights sat, so it says so and does not take it as all on the GPU. None: the arguments are recorded
  (collectors/engine_args), the model has no experts to keep in RAM, or the engine has no expert offload to ask about."""
  from boltbeam.artifacts.run_io import load_manifest, optional
  from boltbeam.collectors import providers
  s = optional(run, "trace_request.json").get("engine_settings") or optional(run, "workload_profile.json").get("engine_settings") or {}
  if "engine_args" in s or s.get("cpu_moe") or s.get("moe_cache_mib") is not None:
    return None
  try:
    shapes.moe_split(profile.get("source") or "")
  except Refused:
    return None
  fork = str(((optional(run, "timing_trace.json").get("rows") or [{}])[0].get("engine") or {}).get("fork") or (load_manifest(run).get("engine") or {}).get("fork") or "")
  if providers.run_provider(run) != "llama.cpp" and "strata" not in fork.lower():
    return None
  return {"status": "not_recorded", "says": "Where this run kept the model's weights (all on the GPU or experts in RAM) was not recorded, so it is not held against either limit."}


def run_block(run, profile:dict[str, Any], token:dict[str, Any] | None, vram_gbs:float | None, *, host:dict[str, Any] | None = None) -> dict[str, Any] | None:
  """A run that offloaded its experts against the offload limit: None when it did not. Else the limit as one tok/s
  (hit rate known: no cache, so every offloaded expert is a miss; or a hit rate the run recorded) or as a band over hit
  rates, the measured speed as a % of it, and `says` for layer 1. A run it cannot limit says why (status absent)."""
  rec = recorded(run)
  if rec is None:
    return None
  from boltbeam.target import host as hostmod
  f = rec["flags"]
  base = {"via": "Strata" if rec["strata"] else "engine arguments " + " ".join(sorted(f)), "schema": "boltbeam.offload_run.v1"}
  try:
    if not token or (token.get("batch") or 1) != 1:
      raise Refused("the offload limit is stated for one stream, and this run has no one-stream speed")
    if not vram_gbs:
      raise Refused("this GPU has no measured memory bandwidth")
    if any("=CPU" in p and "exps" not in p for p in f.get("ot", [])) or any("blk" in p for p in f.get("ot", [])):
      raise Refused("it moved tensors to the CPU that are not all the experts (an -ot pattern), which is not the model the limit is built on")
    sh = shapes.moe_split(profile.get("source") or "")
    host = host or hostmod.facts(None)
    cache = f.get("cache_mib")
    s = 0.0 if rec["strata"] else f.get("cpu_share", 0.0) if cache else 1.0  # Strata reads host weights over PCIe; no cache: the CPU computes every expert
    miss = 1 - f["n_cpu_moe"] / sh["layers"] if f.get("n_cpu_moe") and not cache else 1.0
    common = dict(dense_bytes=sh["dense_bytes"], expert_bytes=sh["expert_bytes"], vram_gbs=vram_gbs, ram_gbs=host["ram_gbs"], pcie_gbs=host["pcie_gbs"],
                  estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"), cpu_share=s)
    if cache or rec["strata"]:
      rates = [rec["hit_rate"]] if rec["hit_rate"] is not None else shapes.hit_rates((cache or 0) * 2**20, sh["expert_total"])
      note = None if cache else "Strata's cache size is not recorded"
    else:
      rates, note = [1 - miss], None  # --cpu-moe: hit rate 0 by construction; --n-cpu-moe N: the other layers' experts are on the card
    rows = ol.band(rates, **common)
  except Refused as exc:
    return {**base, "status": "absent", "reason": str(exc)}
  lim, tok = shapes.span(rows), token["tok_s"]
  return {**base, "status": "modeled", "estimated": sh["estimated"], "hit_rate_known": len(rows) == 1, "cpu_share": s, "tok_s": lim, "band": [
          {"hit_rate": r["hit_rate"], "tok_s": r["overlap"]["tok_s"], "bound_by": r["bound_by"]} for r in rows], "notes": rows[0]["notes"] + ([note] if note else []),
          "measured_tok_s": tok, "pct": {"low": 100 * tok / lim["high"], "high": 100 * tok / lim["low"]}, "host": {k: host[k] for k in ("ram_gbs", "pcie_gbs")}}
