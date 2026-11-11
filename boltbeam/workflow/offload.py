"""Models past GPU memory: the recommended setup when the experts live in host RAM with a cache on the card, and how a
run that did so reads against the offload limit (roofline/offload.py). Shapes come from a header or are stated as an
estimate; the host's RAM and PCIe come from target/host.py; the hit rate is a sweep, never fitted to a speed."""
from __future__ import annotations

from typing import Any

from boltbeam.collectors.expert_placement import parse_flags, recorded  # noqa: F401  (where a run kept its experts; tests read parse_flags here)
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
  from boltbeam.collectors.capture_settings import render_llama
  placed = {"cpu_moe": True, "moe_cache_mib": mib, "moe_cache_cpu_share": s} if engine == "llama.cpp" else None
  flags = render_llama(placed) if placed else None
  spill = max(total_exp - ram, 0.0)
  notes = list(band[0]["notes"]) + ([f"the experts ({_gb(total_exp)}) are more than the usable RAM ({_gb(ram)}): {_gb(spill)} would be read from disk, "
                                     "which this limit does not cover; the speed is below it"] if spill else [])
  tok, lo_hi = [r["overlap"]["tok_s"] for r in band], shapes.span(band)
  say = (f"at most {lo_hi['low']:.0f}–{lo_hi['high']:.0f} tokens a second, depending on how often the needed parts are already on the card"
         + (" (estimate: the model's shapes are not read from a header)" if sh["estimated"] else "") + ". " + plain_words()["offload"]["decode_tuned"])
  return {"schema": "boltbeam.offload_setup.v1", "status": "recommended", "workload": "decode", "for": plain_words()["offload"]["for_decode"],
          "estimated": sh["estimated"], "notes": notes, "engine": engine,
          "shapes": {k: sh[k] for k in ("source", "dense_bytes", "expert_bytes", "expert_total", "experts", "top_k", "layers")},
          "cache_mib": mib, "cache_share_of_experts": cache / total_exp, "cpu_share": s, "ram_usable_bytes": ram, "host": host, "hit_rates": rates,
          "band": [{"hit_rate": r["hit_rate"], "ms": r["ms"], "bound_by": r["bound_by"], "tok_s": r["overlap"]["tok_s"],
                    "tok_s_no_overlap": r["no_overlap"]["tok_s"]} for r in band],
          "tok_s": lo_hi, "tok_s_default_share": shapes.span(plain), "flags": flags, "settings": placed, "build_gaps": feature_gaps(engine, flags or []),
          "context": context, "says": say,
          "arithmetic": [f"GPU: ({_gb(dense)} dense + hit x {_gb(sh['expert_bytes'])} experts + the uploaded misses) / {vram_gbs:.0f} GB/s",
                         f"CPU: {s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {min(host['ram_gbs'], host.get('cpu_gbs') or host['ram_gbs']):.1f} GB/s (RAM; CPU dequant rate unmeasured)",
                         f"PCIe: {1 - s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {host['pcie_gbs']:.1f} GB/s ({host['pcie_source']})",
                         "the limit is the slowest of the three; hit rates: the cache's share of the experts, then 0.8 to 0.99 (a sweep, not an estimate)"]}


def _ubatches() -> list[int]:
  from boltbeam.core.policy import value
  lo, hi = int(value("prefill_ubatch_min")), int(value("prefill_ubatch_max"))
  return [1 << i for i in range(lo.bit_length() - 1, hi.bit_length()) if lo <= 1 << i <= hi]


LLAMA_BATCH = 2048  # llama.cpp's logical batch (-b) by default (common/common.h:472); it bounds the chunk (-ub)
# llama.cpp's prompt attention is the MMA flash-attention kernel, which takes K and V as fp16 and converts a quantized
# cache first (ggml-cuda/fattn-mma-f16.cuh:2113-2114 need_f16_K/V, fattn-common.cuh:1042; master 7592b61d3): its products
# run on the fp16 path. Its instruction sites are read from the engine's binary when a capture has it; a setup has none.
LLAMA_ATTENTION = {"fp16": 1}


def prompt_work(profile:dict[str, Any], target, *, vram_gbs:float, kv:str) -> dict[str, Any]:
  """What prefill_work.turn_ceiling needs for llama.cpp on this GPU, from the profile alone: the weight roles (the
  decode limit's bytes, collectors/prefill.prefill_roles), llama.cpp's formats with this KV type, its weight paths, the
  chip's measured matrix peaks and the read bandwidth. A Refused when the profile has no roles to limit."""
  from boltbeam.collectors import prefill as pf
  from boltbeam.collectors.replay import llama_weight_path
  from boltbeam.roofline import limit
  roles = pf.prefill_roles(profile, limit.ceiling(profile, target, peak_gbs=vram_gbs)["decode"]["roles"])
  if not roles:
    raise Refused("the model's profile has no weight roles to limit a prompt with")
  return {"profile": profile, "roles": roles, "formats": pf.llama_formats(kv, {str(r["quant"]).upper() for r in roles}),
          "weight_path": llama_weight_path, "attention_paths": LLAMA_ATTENTION, "peaks": dict(target.matrix_tflops or {}),
          "bandwidth_gbs": vram_gbs}


def prompt_limit(work:dict[str, Any], *, layers:int, in_ram:int, ubatch:int, prompt:int, h2d_gbs:float, ram_gbs:float,
                 estimated:bool) -> dict[str, Any]:
  """The limit of reading one prompt of `prompt` tokens in chunks of `ubatch` with the experts of `in_ram` of the
  model's `layers` expert layers in host RAM: prefill_work.turn_ceiling, the one prompt limit (each weight once per
  chunk, the experts the chunk reaches, attention and the recurrent block, each at its path's measured peak or the
  bandwidth), with its offload term for the layers in RAM. Per chunk llama.cpp uploads the experts of those layers the
  chunk reaches (a large batch: the CPU computes none, and the MoE cache, filled only by small batches, holds nothing
  for it); under uniform routing a chunk reaches as many experts in every layer, so the share of them in VRAM is the
  share of layers kept there."""
  from boltbeam.roofline.prefill_work import turn_ceiling
  off = {"hit_rate": 1 - in_ram / layers, "cpu_share": 0.0, "pcie_gbs": h2d_gbs, "ram_gbs": ram_gbs, "estimated": estimated} if in_ram else None
  try:
    c = turn_ceiling(work["profile"], work["roles"], prompt, 0, ubatch=ubatch, batch=max(ubatch, LLAMA_BATCH), formats=work["formats"],
                     weight_path=work["weight_path"], attention_paths=work["attention_paths"], peaks=work["peaks"],
                     bandwidth_gbs=work["bandwidth_gbs"], offload=off)
  except ValueError as exc:  # FormulaRefused (a Mamba2 hybrid), a routing profile that does not fit
    raise Refused(str(exc)) from exc
  o = c.get("offload") or {"bytes": {"pcie": 0.0}, "ms": {"pcie": 0.0}}
  ms = {"gpu": sum(r["ms"] or 0.0 for r in c["rows"]), "pcie": o["ms"]["pcie"]}
  pathless = [f"{r['role']} {r['quant']}: no matrix path known for it in llama.cpp, so only its bytes bound it" for r in work["roles"]
              if r["count"] != 1 and not work["weight_path"](r["quant"])]  # turn_ceiling notes a path with no peak, not a missing path
  return {"ubatch": ubatch, "flops": sum(r["flops"] for r in c["rows"]), "bytes": {"pcie": o["bytes"]["pcie"], "gpu_vram": sum(r["bytes"] for r in c["rows"])},
          "ms": ms, "bound_by": max(ms, key=ms.get), "tok_s": prompt * 1000.0 / c["ms"], "unmodelled": c["unmodelled"] + pathless}


def prefill_plan(sh:dict[str, Any], work:dict[str, Any], *, usable_vram:float, kv_bytes:float, host:dict[str, Any], prompt:int,
                 engine:str = "llama.cpp") -> dict[str, Any]:
  """The setup for reading prompts when the weights do not fit: derived from the prompt limit (prompt_limit, which is
  prefill_work.turn_ceiling), apart from the decode setup (plan), because what is fast for one is slow for the other.
  The memory the decode setup gives a cache, which a prompt never hits, holds whole layers' experts instead
  (--n-cpu-moe: the rest stay in RAM; every layer moved to RAM adds uploads and saves the GPU no work, so the limit is
  highest with the most layers that fit); the chunk (-ub) is the smallest whose prompt limit is within the policy's tie
  of the best one's. Both setups' prompt and reply limits are given, so the reader sees what each costs the other workload."""
  from boltbeam.core.policy import value
  if not any(r.get("routed") for r in work["roles"]):
    raise Refused("the model's profile names no routed experts, so the experts a prompt uploads cannot be limited")
  layers = sh["layers"]
  per_layer = sh["expert_total"] / layers
  room = usable_vram - sh["dense_bytes"] - kv_bytes
  if room <= 0:
    raise Refused(f"the dense parts ({_gb(sh['dense_bytes'])}) and the KV cache ({_gb(kv_bytes)}) do not fit the usable GPU memory ({_gb(usable_vram)})")
  in_ram = layers - min(layers, int(room // per_layer))
  h2d = (host.get("h2d_gbs") or {}).get("pageable")
  rate = h2d or host["pcie_gbs"]
  at = dict(layers=layers, prompt=prompt, h2d_gbs=rate, ram_gbs=host["ram_gbs"], estimated=sh["estimated"])
  rows = [prompt_limit(work, in_ram=in_ram, ubatch=u, **at) for u in _ubatches()]
  best = max(r["tok_s"] for r in rows)
  chosen = next(r for r in rows if r["tok_s"] >= best * (1 - value("prefill_ubatch_tie_pct") / 100))
  decode_setup = prompt_limit(work, in_ram=layers, ubatch=int(value("prefill_ubatch_min")), **at)
  replies = ol.offload_limit(dense_bytes=sh["dense_bytes"], expert_bytes=sh["expert_bytes"], hit_rate=1 - in_ram / layers, cpu_share=1.0,
                             vram_gbs=work["bandwidth_gbs"], ram_gbs=host["ram_gbs"], pcie_gbs=host["pcie_gbs"], estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"))
  from boltbeam.collectors.capture_settings import render_llama
  placed = ({"n_cpu_moe": in_ram} if in_ram else {}) | {"chunk_size": chosen["ubatch"]} \
           | ({"batch_size": chosen["ubatch"]} if chosen["ubatch"] > LLAMA_BATCH else {})  # llama.cpp's batch bounds the chunk
  w = plain_words()["offload"]
  say = (f"at most {chosen['tok_s']:.0f} tokens a second reading a {prompt}-token prompt (the setup for replies: at most "
         f"{decode_setup['tok_s']:.0f}); writing replies, at most {replies['overlap']['tok_s']:.0f} tokens a second. " + w["prefill_tuned"])
  return {"schema": "boltbeam.offload_setup.v1", "status": "recommended", "workload": "prefill", "for": w["for_prefill"], "engine": engine,
          "estimated": True, "notes": ["routing assumed uniform: no measured routing profile",
                                       "attention on llama.cpp's fp16 flash-attention path: its instruction sites were not read for a setup",
                                       "the compute buffer a larger chunk needs is not in the fit"] + (["shapes estimated, not read from a header"] if sh["estimated"] else [])
                                      + ([] if h2d else ["PCIe at the link's wire rate, an upper bound: no measured host-to-device rate"]) + list(chosen["unmodelled"]),
          "prompt": prompt, "layers_in_ram": in_ram, "ubatch": chosen["ubatch"], "chunks": rows, "h2d_gbs": rate,
          "tok_s": chosen["tok_s"], "tok_s_decode_setup": decode_setup["tok_s"], "replies_tok_s": replies["overlap"]["tok_s"],
          "settings": placed if engine == "llama.cpp" else None, "flags": render_llama(placed) if engine == "llama.cpp" else None, "says": say,
          "arithmetic": [f"GPU, per chunk of {chosen['ubatch']}: every weight once (the dense parts, the experts the chunk reaches, the uploaded ones), "
                         f"attention and the recurrent block, each at its path's measured peak or {work['bandwidth_gbs']:.0f} GB/s, the slower "
                         f"(prefill_work.turn_ceiling): {chosen['ms']['gpu']:.0f} ms for the prompt",
                         f"PCIe, per chunk: the experts the chunk reaches in the {in_ram} layers kept in RAM, at {rate:.1f} GB/s: {chosen['ms']['pcie']:.0f} ms",
                         "the prompt limit is the slower of the two; chunks: " + ", ".join(f"{r['ubatch']}: {r['tok_s']:.0f} tok/s" for r in rows)]}


def refusal(reason:str) -> dict[str, Any]:
  return {"schema": "boltbeam.offload_setup.v1", "status": "refused", "reason": reason, "says": f"This model cannot be offloaded here: {reason}."}


def for_setup(profile:dict[str, Any], provider:str, *, usable:float, kv_bytes:float, vram_gbs:float | None, pci_bus_id:str | None,
              context:int, target, kv:str = "f16", host:dict[str, Any] | None = None, estimate:dict[str, Any] | None = None) -> dict[str, Any]:
  """recommend.setup's offload block: the plan, or the plain reason there is none. kv: the setup's KV cache type, which
  the prompt limit's attention reads. Nothing here raises."""
  from boltbeam.target import host as hostmod
  try:
    if provider != "llama.cpp":
      raise Refused(f"{provider} has no expert-cache offload BoltBeam can set up; llama.cpp has (--engine llama.cpp)")
    if not vram_gbs:
      raise Refused("this GPU has no measured memory bandwidth, so there is no limit to state")
    sh = shapes.moe_split(profile["source"], estimate)
    host = host or hostmod.facts(pci_bus_id)
    out = plan(sh, usable_vram=usable, kv_bytes=kv_bytes, ram_total=host["ram_bytes"], vram_gbs=vram_gbs, host=host, engine=provider, context=context)
  except Refused as exc:
    return refusal(str(exc))
  try:  # the setup for prompts, apart: what is tuned for replies reads a long prompt 5 to 18 times slower
    work = prompt_work(profile, target, vram_gbs=vram_gbs, kv=kv)
    out["prefill"] = prefill_plan(sh, work, usable_vram=usable, kv_bytes=kv_bytes, host=host, prompt=context, engine=provider)
  except Refused as exc:
    out["prefill"] = refusal(str(exc))
  return out


def settings(moe:dict[str, Any]) -> list[dict[str, Any]]:
  """The offload flags as setup rows (flag, why, when), beside the engine's own."""
  return [{"flag": "--cpu-moe", "recommended": True, "changes": "every expert stays in host RAM (the dense parts stay on the card). " + plain_words()["offload"]["decode_tuned"], "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-mib {moe['cache_mib']}", "recommended": True, "changes": "the card keeps this many MiB of the experts already used: the free memory after the dense parts and the KV",
           "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-cpu-share {moe['cpu_share']:g}", "recommended": True, "changes": "the share of cache misses the CPU computes instead of uploading; this balances RAM against PCIe",
           "when": "the weights do not fit GPU memory" + ("; " + moe["build_gaps"][0] if moe["build_gaps"] else "")}]


# --- a run that offloaded its experts, read against the offload limit -------------------------------------------------

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


def run_block(run, profile:dict[str, Any], token:dict[str, Any] | None, vram_gbs:float | None, *, host:dict[str, Any] | None = None,
              withheld:str | None = None) -> dict[str, Any] | None:
  """A run that offloaded its experts against the offload limit: None when it did not. Else the limit as one tok/s
  (hit rate known: no cache, so every offloaded expert is a miss; or a hit rate the run recorded) or as a band over hit
  rates, the measured speed as a % of it, and `says` for layer 1. A run it cannot limit says why (status absent); one whose
  % is withheld (limit.pct_withheld: a speculative run, against a per-token limit) carries the plain-words key."""
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
    if withheld:  # a run the limit could hold, but not as a % of it: its absence keeps the key, so Emit's setups still derive
      return {**base, "status": "absent", "reason": plain_words()["no_limit"][withheld], "reason_key": withheld}
    host = host or hostmod.facts(None)
    cache = f.get("cache_mib")
    s = 0.0 if rec["strata"] else f.get("cpu_share", 0.0) if cache else 1.0  # Strata reads host weights over PCIe; no cache: the CPU computes every expert
    miss = min(f["n_cpu_moe"], sh["layers"]) / sh["layers"] if f.get("n_cpu_moe") and not cache else 1.0  # the first N layers' experts are in RAM: misses
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
  cpu = [(r["bytes"]["cpu_ram"], r["ms"]["cpu"]) for r in rows]  # the experts the CPU computes: their own part of the limit
  return {**base, "status": "modeled", "estimated": sh["estimated"], "hit_rate_known": len(rows) == 1, "cpu_share": s, "tok_s": lim, "band": [
          {"hit_rate": r["hit_rate"], "tok_s": r["overlap"]["tok_s"], "bound_by": r["bound_by"]} for r in rows], "notes": rows[0]["notes"] + ([note] if note else []),
          "cpu_experts": {"bytes_per_token": {"low": min(b for b, _ in cpu), "high": max(b for b, _ in cpu)},
                          "ms": {"low": min(m for _, m in cpu), "high": max(m for _, m in cpu)}, "rate_gbs": rows[0]["cpu_rate_gbs"],
                          "says": plain_words()["offload"]["cpu_experts"]},
          "measured_tok_s": tok, "pct": {"low": 100 * tok / lim["high"], "high": 100 * tok / lim["low"]}, "host": {k: host[k] for k in ("ram_gbs", "pcie_gbs")}}


def run_setups(run, res:dict[str, Any]) -> dict[str, Any] | None:
  """Emit's two setups for a run that kept its experts in RAM (results' offload): the setup for replies and the one for
  prompts that recommend.setup derives for this model on the run's GPU and context, each with the workload it is for,
  its flags, command and limits. None for a run that did not; status "not derived" with the reason when they cannot be
  (the model file is not here to read its shapes, the run's GPU is not recorded)."""
  o = res.get("offload") or {}
  if o.get("status") != "modeled" and not o.get("reason_key") or o.get("via") == "Strata":  # a llama.cpp run held against the offload limit, or whose % is withheld
    return None
  from boltbeam.artifacts.run_io import load_manifest, optional
  from boltbeam.target import gpu as tgpu
  from boltbeam.target.targets import get_target
  from boltbeam.workflow import recommend
  try:
    devs = tgpu.devices(optional(run, "hardware_profile.json").get("gpu") or {})[:1]
    if not devs:
      raise Refused("the run recorded no GPU")
    context = max(optional(run, "trace_request.json").get("contexts") or [recommend.CONTEXT])
    s = recommend.setup(optional(run, "model_profile.json"), None, "llama.cpp", devs, target=get_target(load_manifest(run)["target_id"]),
                        context=context, offload=True)
  except (Refused, OSError, ValueError, KeyError, SystemExit) as exc:
    return {"status": "not derived", "reason": str(exc)}
  moe = s.get("offload") or {}
  if moe.get("status") != "recommended":
    return {"status": "not derived", "reason": moe.get("reason")}
  keep = ("workload", "for", "flags", "settings", "says", "tok_s", "notes", "arithmetic", "estimated")
  pre = moe.get("prefill") or {}
  return {"status": "derived", "context": context, "decode": {**{k: moe[k] for k in keep if k in moe}, "command": s.get("command")},
          "prefill": {**{k: pre[k] for k in (*keep, "command", "ubatch", "layers_in_ram", "prompt", "replies_tok_s", "tok_s_decode_setup") if k in pre}}
                     if pre.get("status") == "recommended" else {"status": "not derived", "reason": pre.get("reason")}}
