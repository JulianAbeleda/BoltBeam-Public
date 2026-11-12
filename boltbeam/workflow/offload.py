"""Models past GPU memory: the recommended setup when the experts live in host RAM with a cache on the card, and how a
run that did so reads against the offload limit (roofline/offload.py). Shapes come from a header or are stated as an
estimate; the host's RAM and its link to the GPU come from target/host.py (the link: target/host_links.py); the hit rate is a sweep, never fitted to a speed."""
from __future__ import annotations

from typing import Any

from boltbeam.collectors.expert_placement import parse_flags, recorded  # noqa: F401  (where a run kept its experts; tests read parse_flags here)
from boltbeam.core.refused import Refused
from boltbeam.report.plain import words as plain_words
from boltbeam.roofline import offload as ol, offload_shapes as shapes, routing_profile as rp
from boltbeam.target import host_links

GIB = 2**30


def _gb(n:float) -> str:
  return f"{n / 1e9:.1f} GB"


def tok_range(span:dict[str, float]) -> str:
  """A tok/s band as words: one number when its rounded ends agree ("335"), else low–high ("120–335")."""
  lo, hi = f"{span['low']:.0f}", f"{span['high']:.0f}"
  return lo if lo == hi else f"{lo}–{hi}"


def replies_words(span:dict[str, float], *, all_cached:bool, unit:str = "tok/s") -> str:
  """The replies limit in words, the one place it is said: a band over hit rates, or one number with its own reason
  (every expert in the card's cache; or the same at every hit rate swept)."""
  one = tok_range(span)
  if "–" in one:
    return _w("replies_limit", span=one, unit=unit)
  return _w("replies_limit_cached" if all_cached else "replies_limit_flat", span=one, unit=unit)


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


def link(host:dict[str, Any]) -> tuple[float | None, str | None]:
  """The rate llama.cpp copies experts to the card at: the measured pageable rate (its experts sit in ordinary, mmap'd
  memory, which the driver stages through its own pinned buffer), else the link's own rate (the link record,
  target/host_links: a wire rate is an upper bound); None when neither is known (offload_limit refuses only a limit
  that moves bytes over the link)."""
  h2d = (host.get("h2d_gbs") or {}).get("pageable")
  if h2d:
    return h2d, "pageable"
  return (host_links.of(host) or {}).get("gbs"), None


def _rate(host:dict[str, Any]) -> tuple[float, str | None]:
  """link(), for a setup, which always balances RAM against the copies: a Refused when there is no rate."""
  rate, memory = link(host)
  if rate is None:
    raise Refused(_w("no_rate", the=host_links.words(host_links.of(host), "the")))
  return rate, memory


def _basis(hl:dict[str, Any] | None) -> str:
  return "reported" if (hl or {}).get("basis") == "reported" else "wire"


def _unmeasured(hl:dict[str, Any] | None) -> str:
  """The note a limit carries when no copy was measured: the link's own rate stands in, and what kind of figure it is."""
  return _w("unmeasured_" + _basis(hl), short=host_links.words(hl, "short"))


def ram_figure(host:dict[str, Any], host_tables:float = 0.0) -> dict[str, Any]:
  """The RAM the experts may fill (target/host.usable_ram): available now less a margin when it was read, else an
  assumed share of total, with the sentence saying which and why."""
  from boltbeam.target.host import usable_ram
  return usable_ram(host["ram_bytes"], host.get("ram_available_bytes"), host_tables=host_tables)


def plan(sh:dict[str, Any], *, usable_vram:float, kv_bytes:float, vram_gbs:float, host:dict[str, Any], ram:dict[str, Any] | None = None,
         engine:str = "llama.cpp", context:int | None = None, fits:bool = False, routing:dict[str, Any] | None = None) -> dict[str, Any]:
  """The offload setup for a model whose weights do not fit usable VRAM: the cache size from the free VRAM after the dense
  parts and the KV, the CPU share that balances RAM against the copy rate, and the limit as a band over hit rates. ram:
  ram_figure (default: from host). A Refused says why it cannot be done (dense parts and KV past the card; weights past
  VRAM + RAM). fits: the model fits the card whole, so any read from disk is only if the reader chooses to offload.
  routing: a measured routing profile (roofline/routing_profile.find): the band's first hit rate is then what a cache of the
  hottest experts serves (in-sample: an upper estimate), not the cache's share of them (uniform routing)."""
  dense, total_exp = sh["dense_bytes"], sh["expert_total"]
  ram = ram or ram_figure(host)
  usable_ram = ram["usable_bytes"]
  if dense + kv_bytes >= usable_vram:
    raise Refused(f"the dense parts ({_gb(dense)}) and the KV cache for {context} tokens ({_gb(kv_bytes)}) do not fit the usable "
                  f"GPU memory ({_gb(usable_vram)}), so no expert cache can be kept on the card")
  if dense + total_exp > usable_vram + usable_ram:
    raise Refused(f"the weights need {_gb(dense + total_exp)} ({_gb(dense)} dense, {_gb(total_exp)} experts); this machine has "
                  f"{_gb(usable_vram)} usable GPU memory and {_gb(usable_ram)} usable RAM: {_gb(usable_vram + usable_ram)} in all. {ram['says']}")
  cache = min(usable_vram - dense - kv_bytes, total_exp)
  mib = int(cache // 2**20)
  rate, memory = _rate(host)
  hl = host_links.of(host)
  s = round(host["ram_gbs"] / (host["ram_gbs"] + rate), 2)  # the share that loads RAM and the copies equally
  common = dict(dense_bytes=dense, expert_bytes=sh["expert_bytes"], vram_gbs=vram_gbs, ram_gbs=host["ram_gbs"], link_gbs=(hl or {}).get("gbs"), link=hl,
                estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"), h2d_gbs=host.get("h2d_gbs"), memory=memory)
  rates = shapes.hit_rates(cache, total_exp, hottest_hit(routing, cache, sh))
  band, plain = ol.band(rates, cpu_share=s, **common), ol.band(rates, cpu_share=0.0, **common)  # 0.0: llama.cpp's default share
  pinned = ol.band(rates, cpu_share=0.0, **{**common, "memory": "pinned"}) if (host.get("h2d_gbs") or {}).get("pinned") else None
  from boltbeam.collectors.capture_settings import render_llama
  placed = {"cpu_moe": True, "moe_cache_mib": mib, "moe_cache_cpu_share": s} if engine == "llama.cpp" else None
  flags = render_llama(placed) if placed else None
  spill = max(total_exp - cache - usable_ram, 0.0)  # the experts the card's cache holds are never read from RAM after loading
  notes = list(band[0]["notes"]) + ([("only if you choose to offload: " if fits else "") + f"the experts the cache does not hold "
                                     f"({_gb(total_exp - cache)}) are more than the usable RAM ({_gb(usable_ram)}): {_gb(spill)} would be read from disk, "
                                     "which this limit does not cover; the speed is below it"] if spill else []) \
          + ([] if memory else [_unmeasured(hl)]) + [hit_rate_note(routing, rates[0], engine)]
  tok, lo_hi = [r["overlap"]["tok_s"] for r in band], shapes.span(band)
  say = (replies_words(lo_hi, all_cached=cache >= total_exp, unit="tokens a second")
         + (" (estimate: the model's shapes are not read from a header)" if sh["estimated"] else "") + ". " + plain_words()["offload"]["decode_tuned"])
  return {"schema": "boltbeam.offload_setup.v1", "status": "recommended", "workload": "decode", "for": plain_words()["offload"]["for_decode"],
          "estimated": sh["estimated"], "notes": notes, "engine": engine,
          "shapes": {k: sh[k] for k in ("source", "dense_bytes", "expert_bytes", "expert_total", "experts", "top_k", "layers")},
          "cache_mib": mib, "cache_share_of_experts": cache / total_exp, "cpu_share": s, "ram_usable_bytes": usable_ram, "ram": ram, "spill_bytes": spill,
          "host": host, "hit_rates": rates, "routing": rp.words(routing, engine), "h2d_gbs": rate, "h2d_memory": memory, "vram_gbs": vram_gbs,
          "band": [{"hit_rate": r["hit_rate"], "ms": r["ms"], "bound_by": r["bound_by"], "tok_s": r["overlap"]["tok_s"],
                    "tok_s_no_overlap": r["no_overlap"]["tok_s"]} for r in band],
          "tok_s": lo_hi, "tok_s_default_share": shapes.span(plain), "tok_s_pinned": shapes.span(pinned) if pinned else None,
          "flags": flags, "settings": placed, "build_gaps": feature_gaps(engine, flags or []),
          "context": context, "says": say,
          "arithmetic": [f"GPU: ({_gb(dense)} dense + hit x {_gb(sh['expert_bytes'])} experts + the uploaded misses) / {vram_gbs:.0f} GB/s",
                         f"CPU: {s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {min(host['ram_gbs'], host.get('cpu_gbs') or host['ram_gbs']):.1f} GB/s (RAM; CPU dequant rate unmeasured)",
                         f"{host_links.words(hl, 'short')}: {1 - s:g} x (1 - hit) x {_gb(sh['expert_bytes'])} / {rate:.1f} GB/s ("
                         + (f"measured copies from {memory} memory: {host.get('h2d_source')}" if memory else hl['source']) + ")",
                         "the limit is the slowest of the three; hit rates: the first is " + rp.hit_words(routing) + ", then 0.8 to 0.99 (a sweep, not an estimate)"]}


def hottest_hit(routing:dict[str, Any] | None, cache:float, sh:dict[str, Any]) -> float | None:
  """The hit rate of a cache holding the hottest experts it fits (experts of one size: the experts' total over layers x
  experts each), over the layers the profile measured; None without a profile."""
  if not routing:
    return None
  slots = int(cache // (sh["expert_total"] / (sh["layers"] * sh["experts"])) * len(routing["layers"]) / sh["layers"])
  return round(rp.hit_rate(routing, slots), 4)


def hit_rate_note(routing:dict[str, Any] | None, rate:float, engine:str | None = None) -> str:
  """The band's first hit rate and what it is: with a profile, in-sample and hottest-first (an upper estimate for a cache
  that keeps the recently used experts), from the routing words; else the cache's share under uniform routing."""
  return (f"first hit rate {rate:.0%}: {rp.hit_words(routing)}; routing: {rp.words(routing, engine)}" if routing else
          f"first hit rate {rate:.0%}: {rp.hit_words(None)}, under {rp.UNIFORM}")


def _ubatches() -> list[int]:
  from boltbeam.core.policy import value
  lo, hi = int(value("prefill_ubatch_min")), int(value("prefill_ubatch_max"))
  return [1 << i for i in range(lo.bit_length() - 1, hi.bit_length()) if lo <= 1 << i <= hi]


LLAMA_BATCH = 2048  # llama.cpp's logical batch (-b) by default (common/common.h:472); it bounds the chunk (-ub)
# llama.cpp's prompt attention is the MMA flash-attention kernel, which takes K and V as fp16 and converts a quantized
# cache first (ggml-cuda/fattn-mma-f16.cuh:2113-2114 need_f16_K/V, fattn-common.cuh:1042; master 7592b61d3): its products
# run on the fp16 path. Its instruction sites are read from the engine's binary when a capture has it; a setup has none.
LLAMA_ATTENTION = {"fp16": 1}


def prompt_work(profile:dict[str, Any], target, *, vram_gbs:float, kv:str, routing:bool = True, engine:str = "llama.cpp") -> dict[str, Any]:
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
          "bandwidth_gbs": vram_gbs, "routing": rp.find(profile.get("source"), engine=engine) if routing else None, "engine": engine}


def prompt_limit(work:dict[str, Any], *, layers:int, in_ram:int, ubatch:int, prompt:int, h2d_gbs:float, ram_gbs:float,
                 estimated:bool, link:dict[str, Any] | None = None) -> dict[str, Any]:
  """The limit of reading one prompt of `prompt` tokens in chunks of `ubatch` with the experts of `in_ram` of the
  model's `layers` expert layers in host RAM: prefill_work.turn_ceiling, the one prompt limit (each weight once per
  chunk, the experts the chunk reaches, attention and the recurrent block, each at its path's measured peak or the
  bandwidth), with its offload term for the layers in RAM. Per chunk llama.cpp uploads the experts of those layers the
  chunk reaches (a large batch: the CPU computes none, and the MoE cache, filled only by small batches, holds nothing
  for it); under uniform routing a chunk reaches as many experts in every layer, so the share of them in VRAM is the
  share of layers kept there. link: the link record (its semantics set the upload term; copy when None)."""
  from boltbeam.roofline.prefill_work import turn_ceiling
  prof = work.get("routing")
  try:
    off = {"hit_rate": card_share(work, in_ram, layers, ubatch), "cpu_share": 0.0, "link_gbs": h2d_gbs, "ram_gbs": ram_gbs, "estimated": estimated, "link": link} if in_ram else None
    c = turn_ceiling(work["profile"], work["roles"], prompt, 0, ubatch=ubatch, batch=max(ubatch, LLAMA_BATCH), formats=work["formats"],
                     weight_path=work["weight_path"], attention_paths=work["attention_paths"], peaks=work["peaks"],
                     bandwidth_gbs=work["bandwidth_gbs"], offload=off, routing=prof["layers"] if prof else None)
  except ValueError as exc:  # FormulaRefused (a Mamba2 hybrid), a routing profile that does not fit
    raise Refused(str(exc)) from exc
  o = c.get("offload") or {"bytes": {"link": 0.0}, "ms": {"link": 0.0}}
  ms = {"gpu": sum(r["ms"] or 0.0 for r in c["rows"]), "link": o["ms"]["link"]}
  from boltbeam.collectors.replay import weight_path_note
  pathless = [weight_path_note(r["role"], r["quant"]) for r in work["roles"] if r["count"] != 1 and not work["weight_path"](r["quant"])]  # turn_ceiling notes a path with no peak, not a missing path
  return {"ubatch": ubatch, "flops": sum(r["flops"] for r in c["rows"]), "bytes": {"link": o["bytes"]["link"], "gpu_vram": sum(r["bytes"] for r in c["rows"])},
          "ms": ms, "bound_by": max(ms, key=ms.get), "tok_s": prompt * 1000.0 / c["ms"], "unmodelled": c["unmodelled"] + pathless,
          "routing": rp.words(prof, work.get("engine"))}


def card_share(work:dict[str, Any], in_ram:int, layers:int, ubatch:int) -> float:
  """The share of a chunk's reached expert bytes in the layers on the card (--n-cpu-moe keeps the first in_ram layers'
  experts in RAM): 1 - in_ram / layers under uniform routing, else from the profile's per-layer reach at this chunk."""
  prof = work.get("routing")
  if not prof:
    return 1 - in_ram / layers
  from boltbeam.roofline.prefill_work import experts_reached
  r = next((x for x in work["roles"] if x.get("routed") and x["count"] != 1), None)
  if r is None:
    return 1 - in_ram / layers
  reach = experts_reached(ubatch, r["n_expert"], r["routed"], layers, prof["layers"])
  return 1 - sum(reach[:in_ram]) / sum(reach)


def prefill_plan(sh:dict[str, Any], work:dict[str, Any], *, usable_vram:float, kv_bytes:float, host:dict[str, Any], prompt:int,
                 engine:str = "llama.cpp", ram:dict[str, Any] | None = None, fits:bool = False) -> dict[str, Any]:
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
  ram = ram or ram_figure(host)
  spill = max(in_ram * per_layer - ram["usable_bytes"], 0.0)
  rate, memory = _rate(host)
  hl = host_links.of(host)
  h2d = rate if memory else None
  at = dict(layers=layers, prompt=prompt, h2d_gbs=rate, ram_gbs=host["ram_gbs"], estimated=sh["estimated"], link=hl)
  rows = [prompt_limit(work, in_ram=in_ram, ubatch=u, **at) for u in _ubatches()]
  best = max(r["tok_s"] for r in rows)
  chosen = next(r for r in rows if r["tok_s"] >= best * (1 - value("prefill_ubatch_tie_pct") / 100))
  pinned_gbs = (host.get("h2d_gbs") or {}).get("pinned")
  pinned = prompt_limit(work, in_ram=in_ram, ubatch=chosen["ubatch"], **{**at, "h2d_gbs": pinned_gbs}) if pinned_gbs and in_ram else None
  decode_setup = prompt_limit(work, in_ram=layers, ubatch=int(value("prefill_ubatch_min")), **at)
  replies = ol.offload_limit(dense_bytes=sh["dense_bytes"], expert_bytes=sh["expert_bytes"], hit_rate=1 - in_ram / layers, cpu_share=1.0,
                             vram_gbs=work["bandwidth_gbs"], ram_gbs=host["ram_gbs"], link_gbs=(hl or {}).get("gbs"), link=hl, estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"))
  from boltbeam.collectors.capture_settings import render_llama
  placed = ({"n_cpu_moe": shapes.n_cpu_moe(sh, in_ram)} if in_ram else {}) | {"chunk_size": chosen["ubatch"]} \
           | ({"batch_size": chosen["ubatch"]} if chosen["ubatch"] > LLAMA_BATCH else {})  # llama.cpp's batch bounds the chunk
  w = plain_words()["offload"]
  say = (f"at most {chosen['tok_s']:.0f} tokens a second reading a {prompt}-token prompt (the setup for replies: at most "
         f"{decode_setup['tok_s']:.0f}); writing replies, at most {replies['overlap']['tok_s']:.0f} tokens a second. " + w["prefill_tuned"])
  return {"schema": "boltbeam.offload_setup.v1", "status": "recommended", "workload": "prefill", "for": w["for_prefill"], "engine": engine,
          "estimated": True, "notes": [("routing: " + rp.words(work.get("routing"), work.get("engine") or engine)) if work.get("routing") else "routing assumed uniform: no measured routing profile",
                                       "attention on llama.cpp's fp16 flash-attention path: its instruction sites were not read for a setup",
                                       "the compute buffer a larger chunk needs is not in the fit"] + (["shapes estimated, not read from a header"] if sh["estimated"] else [])
                                      + ([] if h2d else [_unmeasured(hl)]) + host_links.unverified(hl)
                                      + ([_w("direct_read_prompts")] if (hl or {}).get("semantics") == "direct_read" else []) + list(chosen["unmodelled"])
                                      + ([("only if you choose to offload: " if fits else "") + f"the experts kept in RAM ({_gb(in_ram * per_layer)}) are more than the usable RAM ({_gb(ram['usable_bytes'])}): "
                                          f"{_gb(spill)} would be read from disk, which this limit does not cover"] if spill else []),
          "prompt": prompt, "layers_in_ram": in_ram, "ubatch": chosen["ubatch"], "chunks": rows, "h2d_gbs": rate, "h2d_memory": memory,
          "experts_in_ram_bytes": in_ram * per_layer, "experts_on_card_bytes": (layers - in_ram) * per_layer, "spill_bytes": spill, "ram": ram,
          "tok_s_pinned": pinned["tok_s"] if pinned else None, "bound_by": chosen["bound_by"],
          "tok_s": chosen["tok_s"], "tok_s_decode_setup": decode_setup["tok_s"], "replies_tok_s": replies["overlap"]["tok_s"],
          "settings": placed if engine == "llama.cpp" else None, "flags": render_llama(placed) if engine == "llama.cpp" else None, "says": say,
          "arithmetic": [f"GPU, per chunk of {chosen['ubatch']}: every weight once (the dense parts, the experts the chunk reaches, the uploaded ones), "
                         f"attention and the recurrent block, each at its path's measured peak or {work['bandwidth_gbs']:.0f} GB/s, the slower "
                         f"(prefill_work.turn_ceiling): {chosen['ms']['gpu']:.0f} ms for the prompt",
                         f"{host_links.words(hl, 'short')}, per chunk: the experts the chunk reaches in the {in_ram} layers kept in RAM, at {rate:.1f} GB/s"
                         + (f" (measured copies from {memory} memory)" if memory else " (" + _w("rate_" + _basis(hl)) + ")")
                         + f": {chosen['ms']['link']:.0f} ms",
                         "the prompt limit is the slower of the two; chunks: " + ", ".join(f"{r['ubatch']}: {r['tok_s']:.0f} tok/s" for r in rows)]}


def _w(key:str, **kw) -> str:
  return plain_words()["offload_page"][key].format(**kw)


def _bound(which:str, *, vram_gbs:float, host:dict[str, Any], rate:float, memory:str | None) -> str:
  if which == "gpu":
    return _w("bound_gpu", gbs=f"{vram_gbs:.0f}")
  if which == "cpu":
    return _w("bound_cpu", gbs=f"{min(host['ram_gbs'], host.get('cpu_gbs') or host['ram_gbs']):.0f}")
  hl = host_links.of(host)
  if (hl or {}).get("semantics", "copy") != "copy":
    return host_links.words(hl, "reads", gbs=f"{rate:.0f}")
  return _w("bound_link_pageable" if memory == "pageable" else "bound_link_" + _basis(hl), gbs=f"{rate:.0f}") + "".join(f"; {n}" for n in host_links.unverified(hl))


def machine_rows(host:dict[str, Any]) -> list[dict[str, str]]:
  """This machine's host figures as label/value rows, the offload page's and doctor's: RAM now, RAM reads, the link
  and the measured copies to the card."""
  total, avail = host.get("ram_bytes"), host.get("ram_available_bytes")
  rows = []
  if total:
    rows.append({"label": "RAM", "value": _w("ram_now", total=_gb(total), available=_gb(avail)) if avail is not None else _w("ram_now_unknown", total=_gb(total))})
  if host.get("ram_gbs"):
    rows.append({"label": "RAM reads", "value": _w("ram_reads", gbs=f"{host['ram_gbs']:.1f}")})
  hl = host_links.of(host)
  if hl:
    d = hl.get("detail") or {}
    rows.append({"label": host_links.words(hl, "name"), "value": host_links.words(hl, "detail", gen=d.get("gen") or "?", width=d.get("width") or "?",
                                                                                      gbs=f"{hl['gbs']:.1f}" if hl["gbs"] else "?")
                                                                 + (f" ({host_links.unverified_words(hl)})" if host_links.unverified_words(hl) else "")})
  h2d = host.get("h2d_gbs") or {}
  if h2d.get("pageable") and h2d.get("pinned"):
    rows.append({"label": "Copies to the card", "value": _w("copies", pageable=f"{h2d['pageable']:.1f}", pinned=f"{h2d['pinned']:.1f}", when=host.get("h2d_measured_at") or "date not recorded")})
  elif hl and hl["semantics"] == "none":
    rows.append({"label": "Copies to the card", "value": _w("copies_none")})
  elif hl and hl["gbs"]:
    rows.append({"label": "Copies to the card", "value": _w("copies_unmeasured" + ("_reported" if _basis(hl) == "reported" else ""), gbs=f"{hl['gbs']:.1f}")})
  return rows


def page(out:dict[str, Any], *, usable_vram:float, kv_bytes:float, fits:bool = False, host:dict[str, Any] | None = None) -> dict[str, Any]:
  """The offload page the screen shows (tui: Offload), every sentence here and in plain_words.json's offload_page: what
  sits on the card and in RAM under each setup, each setup's settings and limits, the resource that limits each in plain
  words, the gap pinned copies would close (a limit, not a promise) and this machine's host figures. Sections of
  label/value rows and lines; no raw keys, no paths."""
  host = host or out.get("host") or {}
  head = _w("fits_card") if fits else _w("past_card")
  if out.get("status") != "recommended":
    return {"schema": "boltbeam.offload_page.v1", "status": "refused", "headline": _w("refused", reason=out.get("reason")),
            "sections": [{"title": _w("machine"), "rows": machine_rows(host), "lines": []}] if host else []}
  sh, ram, pre = out["shapes"], out["ram"], out.get("prefill") or {}
  rate, memory = out["h2d_gbs"], out.get("h2d_memory")
  vram_gbs = out["vram_gbs"]
  cache = out["cache_mib"] * 2**20
  spill = (lambda b: "; " + _w("spill_if_offloaded" if fits else "spill", spill=_gb(b)) if b else "")
  all_cached = out["cache_share_of_experts"] >= 1
  where = [{"label": "On the card, replies", "value": _w("card_replies", dense=_gb(sh["dense_bytes"]), kv=_gb(kv_bytes), cache=_gb(cache),
                                                          used=_gb(sh["dense_bytes"] + kv_bytes + cache), usable=_gb(usable_vram))},
           {"label": "In RAM, replies", "value": _w("ram_replies_cached" if all_cached else "ram_replies", experts=_gb(sh["expert_total"])) + spill(out.get("spill_bytes"))}]
  if pre.get("status") == "recommended":
    on = sh["layers"] - pre["layers_in_ram"]
    where += [{"label": "On the card, prompts", "value": _w("card_prompts", dense=_gb(sh["dense_bytes"]), kv=_gb(kv_bytes), on=on, layers=sh["layers"],
                                                             experts=_gb(pre["experts_on_card_bytes"]), used=_gb(sh["dense_bytes"] + kv_bytes + pre["experts_on_card_bytes"]),
                                                             usable=_gb(usable_vram))},
              {"label": "In RAM, prompts", "value": (_w("ram_prompts", n=pre["layers_in_ram"], layers=sh["layers"], experts=_gb(pre["experts_in_ram_bytes"]))
                                                     if pre["layers_in_ram"] else _w("ram_prompts_none"))
                                                    + spill(pre.get("spill_bytes"))}]
  if ram.get("host_tables_bytes"):
    where.append({"label": "Mapped from disk", "value": _w("tables", tables=_gb(ram["host_tables_bytes"]))})
  rows_r = [{"label": "Settings", "value": _w("set_replies_cached", mib=f"{out['cache_mib']:,}") if all_cached else
                                           _w("set_replies", mib=f"{out['cache_mib']:,}", pct=f"{100 * out['cpu_share']:.0f}")},
            {"label": "Flags", "value": " ".join(out.get("flags") or [])},
            {"label": "Replies", "value": replies_words(out["tok_s"], all_cached=all_cached)},
            {"label": "Limited by", "value": _bound(out["band"][0]["bound_by"], vram_gbs=vram_gbs, host=host, rate=rate, memory=memory)}]
  sections = [{"title": _w("where"), "rows": where, "lines": [ram["says"]]}, {"title": _w("replies"), "rows": rows_r, "lines": []}]
  if pre.get("status") == "recommended":
    set_p = _w("set_prompts", n=pre["layers_in_ram"], layers=sh["layers"], ubatch=pre["ubatch"]) if pre["layers_in_ram"] else _w("set_prompts_none", ubatch=pre["ubatch"])
    sections.append({"title": _w("prompts"), "rows": [
      {"label": "Settings", "value": set_p}, {"label": "Flags", "value": " ".join(pre.get("flags") or [])},
      {"label": "Prompts", "value": _w("prompts_limit", tok_s=f"{pre['tok_s']:.0f}", prompt=f"{pre['prompt']:,}", replies=f"{pre['replies_tok_s']:.0f}")},
      {"label": "Limited by", "value": _bound(pre["bound_by"], vram_gbs=vram_gbs, host=host, rate=pre["h2d_gbs"], memory=pre.get("h2d_memory"))}], "lines": []})
  else:
    sections.append({"title": _w("prompts"), "rows": [], "lines": [pre.get("says") or ""]})
  h2d = host.get("h2d_gbs") or {}
  if h2d.get("pinned") and h2d.get("pageable"):
    lines = [_w("pinned_gap", pinned=f"{h2d['pinned']:.1f}", pageable=f"{h2d['pageable']:.1f}")]
    if pre.get("status") == "recommended":
      lines.append(_w("pinned_prompts", now=f"{pre['tok_s']:.0f}", then=f"{pre['tok_s_pinned']:.0f}") if pre.get("tok_s_pinned") else _w("pinned_prompts_none"))
    if all_cached:
      lines.append(_w("pinned_replies_none"))
    elif out.get("tok_s_pinned"):
      lines.append(_w("pinned_replies", now=tok_range(out["tok_s_default_share"]), then=tok_range(out["tok_s_pinned"])))
    lines.append(_w("pinned_caveat"))
  else:
    lines = [_w("pinned_unmeasured", why=host.get("h2d_reason") or "no GPU runtime here")]
  sections += [{"title": _w("pinned"), "rows": [], "lines": lines},
               {"title": _w("machine"), "rows": machine_rows(host), "lines": []},
               {"title": _w("notes"), "rows": [], "lines": list(dict.fromkeys([*out.get("notes", []), *pre.get("notes", [])]))}]
  return {"schema": "boltbeam.offload_page.v1", "status": "recommended", "headline": head, "sections": sections}


def refusal(reason:str) -> dict[str, Any]:
  return {"schema": "boltbeam.offload_setup.v1", "status": "refused", "reason": reason, "says": f"This model cannot be offloaded here: {reason}."}


def for_setup(profile:dict[str, Any], provider:str, *, usable:float, kv_bytes:float, vram_gbs:float | None, pci_bus_id:str | None,
              context:int, target, kv:str = "f16", host:dict[str, Any] | None = None, estimate:dict[str, Any] | None = None,
              host_tables:float = 0.0, fits:bool = False, probe:bool = False, routing:bool = True) -> dict[str, Any]:
  """recommend.setup's offload block: the plan, or the plain reason there is none, and the offload page (page()).
  kv: the setup's KV cache type, which the prompt limit's attention reads. host_tables: the model's bytes the engine
  maps from disk whatever the plan (a GGUF's input embeddings, recommend.tensor_bytes' host): not taken from the RAM
  for the experts (only the rows read are resident), said on the page. probe: measure the copy rate to the GPU when
  none is saved (the offload page only; an idle GPU, target/host._h2d_probe); else the saved figure or the link's.
  Nothing here raises."""
  from boltbeam.target import host as hostmod
  try:
    if provider != "llama.cpp":
      raise Refused(f"{provider} has no expert-cache offload BoltBeam can set up; llama.cpp has (--engine llama.cpp)")
    if not vram_gbs:
      raise Refused("this GPU has no measured memory bandwidth, so there is no limit to state")
    sh = shapes.moe_split(profile["source"], estimate)
    host = host or hostmod.facts(pci_bus_id, h2d_probe=hostmod._h2d_probe if probe else None,
                                 integrated_probe=hostmod._integrated_probe if probe else None)
    if (host_links.of(host) or {}).get("semantics") == "none":
      raise Refused(plain_words()["offload_page"]["shared_memory"])
    ram = ram_figure(host, host_tables)
    out = plan(sh, usable_vram=usable, kv_bytes=kv_bytes, vram_gbs=vram_gbs, host=host, ram=ram, engine=provider, context=context, fits=fits,
               routing=rp.find(profile.get("source"), engine=provider) if routing else None)
  except Refused as exc:
    return {**refusal(str(exc)), "page": page(refusal(str(exc)), usable_vram=usable, kv_bytes=kv_bytes, host=host)}
  try:  # the setup for prompts, apart: what is tuned for replies reads a long prompt 5 to 18 times slower
    work = prompt_work(profile, target, vram_gbs=vram_gbs, kv=kv, routing=routing, engine=provider)
    out["prefill"] = prefill_plan(sh, work, usable_vram=usable, kv_bytes=kv_bytes, host=host, prompt=context, engine=provider, ram=ram, fits=fits)
  except Refused as exc:
    out["prefill"] = refusal(str(exc))
  out["page"] = page(out, usable_vram=usable, kv_bytes=kv_bytes, fits=fits)
  return out


def settings(moe:dict[str, Any]) -> list[dict[str, Any]]:
  """The offload flags as setup rows (flag, why, when), beside the engine's own."""
  return [{"flag": "--cpu-moe", "recommended": True, "changes": "every expert stays in host RAM (the dense parts stay on the card). " + plain_words()["offload"]["decode_tuned"], "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-mib {moe['cache_mib']}", "recommended": True, "changes": "the card keeps this many MiB of the experts already used: the free memory after the dense parts and the KV",
           "when": "the weights do not fit GPU memory"},
          {"flag": f"--moe-cache-cpu-share {moe['cpu_share']:g}", "recommended": True, "changes": "the share of cache misses the CPU computes instead of uploading; this balances RAM against " + host_links.words(host_links.of(moe.get("host") or {}), "short"),
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
    if f.get("ngl_cpu_layers"):
      raise Refused(f"-ngl leaves {f['ngl_cpu_layers']} whole layers on the CPU, attention included, which is not the model the offload limit is built on")
    if any("=CPU" in p and "exps" not in p for p in f.get("ot", [])) or any("blk" in p for p in f.get("ot", [])):
      raise Refused("it moved tensors to the CPU that are not all the experts (an -ot pattern), which is not the model the limit is built on")
    sh = shapes.moe_split(profile.get("source") or "")
    if withheld:  # a run the limit could hold, but not as a % of it: its absence keeps the key, so Emit's setups still derive
      return {**base, "status": "absent", "reason": plain_words()["no_limit"][withheld], "reason_key": withheld}
    host = host or hostmod.facts(_run_bus(run))  # the saved copy rate for the run's GPU, never a probe
    cache = f.get("cache_mib")
    s = 0.0 if rec["strata"] else f.get("cpu_share", 0.0) if cache else 1.0  # Strata reads host weights over the link; no cache: the CPU computes every expert
    miss = shapes.expert_layers_under(sh, f["n_cpu_moe"]) / sh["layers"] if f.get("n_cpu_moe") and not cache else 1.0  # the first N layers' experts are in RAM: misses
    hl = host_links.of(host)
    _, memory = link(host)  # llama.cpp copies from pageable memory, as a setup's limit says; Strata's copies are not known
    common = dict(dense_bytes=sh["dense_bytes"], expert_bytes=sh["expert_bytes"], vram_gbs=vram_gbs, ram_gbs=host["ram_gbs"], link_gbs=(hl or {}).get("gbs"), link=hl,
                  estimated=sh["estimated"], cpu_gbs=host.get("cpu_gbs"), cpu_share=s,
                  **({"h2d_gbs": host.get("h2d_gbs"), "memory": memory} if memory and not rec["strata"] else {}))
    if cache or rec["strata"]:  # the misses cross the link: with no rate for it, say so in words (not the limit's own check)
      if (hl or {}).get("semantics") != "none" and common["link_gbs"] is None and "memory" not in common:
        raise Refused(_w("no_rate", the=host_links.words(hl, "the")))
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
          "measured_tok_s": tok, "pct": {"low": 100 * tok / lim["high"], "high": 100 * tok / lim["low"]}, "host": {"ram_gbs": host["ram_gbs"], "link_gbs": (hl or {}).get("gbs"), "link": (hl or {}).get("kind")}}


def _run_bus(run) -> str | None:
  """The PCI bus id of the GPU a run recorded (hardware_profile.json), or None."""
  from boltbeam.artifacts.run_io import optional
  from boltbeam.target import gpu as tgpu
  try:
    devs = tgpu.devices(optional(run, "hardware_profile.json").get("gpu") or {})
  except (KeyError, TypeError, ValueError):
    return None
  return (devs[0].get("pci_bus_id") if devs else None) or None


def other_machine(run, here_gpu:dict[str, Any] | None = None) -> dict[str, str]:
  """{"host_says": ...} when the machine running Emit is not the one that ran the run: the setups are sized with the RAM and
  link to the GPU read here (target/host.py), and the run recorded its own RAM (hardware_profile.json system.memory_bytes) and its
  GPU's name and PCI bus id, but no link rate or RAM bandwidth, so a run from a machine with other RAM, another GPU or the
  GPU in another slot gets setups sized for this one. {} when the RAM matches (within 2%) and the GPU's name and bus id
  match where both sides know them; a run with no recorded RAM is also said, since nothing shows the machines agree.
  here_gpu: this machine's first GPU (target/gpu.devices), read from the hardware probe when not given; the link is named
  as read for it (target/host_links), "host link" when it cannot be read."""
  import os
  from boltbeam.artifacts.run_io import optional
  from boltbeam.target import gpu as tgpu
  here = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  hw = optional(run, "hardware_profile.json")
  there = (hw.get("system") or {}).get("memory_bytes")
  ran = (tgpu.devices(hw.get("gpu") or {}) or [{}])[0]
  if here_gpu is None and (ran.get("name") or ran.get("pci_bus_id")):
    from boltbeam.target.hardware import hardware_profile
    here_gpu = (tgpu.devices(hardware_profile().get("gpu") or {}) or [{}])[0]
  gpu = next(((k, ran[k], (here_gpu or {})[k]) for k in ("name", "pci_bus_id") if ran.get(k) and (here_gpu or {}).get(k)
              and str(ran[k]).lower() != str(here_gpu[k]).lower()), None)
  if there and abs(there - here) <= 0.02 * here and gpu is None:
    return {}
  was = (f"the run recorded {there / 2**30:.0f} GiB of RAM, this machine has {here / 2**30:.0f} GiB" if there else "the run recorded no RAM") \
        + (f"; the run's GPU was {'a' if gpu[0] == 'name' else 'at PCI'} {gpu[1]}, this machine's is {gpu[2]}" if gpu else "")
  try:
    from boltbeam.target.host import link as read_link
    hl = read_link((here_gpu or {}).get("pci_bus_id"), vendor=(here_gpu or {}).get("vendor"))
  except Refused:
    hl = None
  return {"host_says": f"These setups are sized with the RAM and {host_links.words(hl, 'name')} of the machine that ran BoltBeam just now, not of the machine that "
                       f"ran the measurement ({was}). Check them on the machine that will serve."}


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
  return {"status": "derived", "context": context, **other_machine(run), "decode": {**{k: moe[k] for k in keep if k in moe}, "command": s.get("command")},
          "prefill": {**{k: pre[k] for k in (*keep, "command", "ubatch", "layers_in_ram", "prompt", "replies_tok_s", "tok_s_decode_setup") if k in pre}}
                     if pre.get("status") == "recommended" else {"status": "not derived", "reason": pre.get("reason")}}
