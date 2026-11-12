"""The speed limit for a mixture-of-experts model whose experts are split between VRAM (a cache) and host RAM.

Per token, expert bytes E split by the hit rate h (the share of the token's expert bytes found in VRAM):
  GPU   reads  dense + h*E + (1-s)*(1-h)*E from VRAM     (the uploaded misses are read once in VRAM too)
  CPU   reads  s*(1-h)*E from host RAM, at min(RAM GB/s, CPU dequant GB/s): the CPU computes those experts
  link  moves  (1-s)*(1-h)*E to the GPU
s is the CPU share of the misses (llama.cpp --moe-cache-cpu-share). h=1 is exactly today's limit (bytes / VRAM
bandwidth, math.roofline.floor_ms_per_token). Each part is a floor on its own resource, so the token cannot be faster
than the slowest one when the three run in parallel (overlap, the limit) and cannot be slower than their sum when
nothing overlaps (no_overlap, the other bound). The link term follows the link record's semantics
(target/host_links.py): copy, the misses are copied over the link and read again from VRAM (above); direct_read, they are
read in place over the link during the GEMM, with no staging copy and no second read; none, the GPU shares the CPU's
memory, so the misses are plain memory reads at vram_gbs (which is then the memory's bandwidth) and the link term is 0. The hit rate is an input, never fitted to a measured speed: when the
engine does not report one, `band` gives the limit over a stated list of hit rates. Nothing here names a model."""
from __future__ import annotations

import math

from boltbeam.core.refused import Refused
from boltbeam.math.roofline import floor_ms_per_token
from boltbeam.target.host_links import SEMANTICS, unverified, words

_GB = 1e9
SEMANTICS_NOTE = {'direct_read': 'the misses are read in place over {the} during the GEMM (no staging copy)',
                  'none': 'no copy over {the}: the misses are plain memory reads at the memory bandwidth'}


def _check(name, value, *, positive):
  if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
    raise Refused(f'{name} must be a finite {"positive" if positive else "non-negative"} number, got {value!r}')


def offload_limit(*, dense_bytes: float, expert_bytes: float, hit_rate: float, vram_gbs: float, ram_gbs: float,
                  link_gbs: float | None, estimated: bool, cpu_share: float = 1.0, cpu_gbs: float | None = None,
                  h2d_gbs: dict | None = None, memory: str | None = None, link: dict | None = None) -> dict:
  """The per-token limit with a given hit rate. Bytes are per token: dense_bytes the always-read weights,
  expert_bytes the experts the token selects (top_k x bytes per expert). cpu_gbs is the CPU's dequant-and-multiply
  rate over those bytes when it is below RAM bandwidth (None: the CPU part is assumed RAM-bound, which makes the limit an
  upper bound that is only as tight as the RAM figure, and the result says so in `notes`). estimated is required: True when
  the byte counts come from an estimate, not a model header; the result carries it and says so in `notes`. h2d_gbs: a
  measured host-to-device rate per host memory kind ({"pinned": ..., "pageable": ...}, target/host.facts); memory: the
  kind the engine copies from, whose measured rate replaces the link's own rate link_gbs (an upper bound). link: the link
  record (target/host_links.detect); its semantics set the link term (module docstring), copy when there is none."""
  _check('dense_bytes', dense_bytes, positive=False); _check('expert_bytes', expert_bytes, positive=False)
  if dense_bytes + expert_bytes <= 0: raise Refused('a token reads no bytes')
  for n, v in (('hit_rate', hit_rate), ('cpu_share', cpu_share)):
    if type(v) not in (int, float) or not 0 <= v <= 1: raise Refused(f'{n} must be within [0, 1], got {v!r}')
  semantics = (link or {}).get('semantics') or 'copy'
  if semantics not in SEMANTICS: raise Refused(f'link semantics must be one of {SEMANTICS}, got {semantics!r}')
  for n, v in (('vram_gbs', vram_gbs), ('ram_gbs', ram_gbs)): _check(n, v, positive=True)
  if cpu_gbs is not None: _check('cpu_gbs', cpu_gbs, positive=True)
  if memory is not None:
    if memory not in (h2d_gbs or {}): raise Refused(f'no measured host-to-device rate for {memory!r} memory (have {sorted(h2d_gbs or {})})')
    _check(f'h2d_gbs[{memory!r}]', h2d_gbs[memory], positive=True); link_gbs = h2d_gbs[memory]
  miss = (1 - hit_rate) * expert_bytes
  cpu_bytes, link_bytes = cpu_share * miss, (1 - cpu_share) * miss
  if link_bytes and semantics != 'none': _check('link_gbs', link_gbs, positive=True)  # a rate only where the link moves bytes
  gpu_bytes = dense_bytes + hit_rate * expert_bytes + (0.0 if semantics == 'direct_read' else link_bytes)
  cpu_rate = min(ram_gbs, cpu_gbs) if cpu_gbs else ram_gbs
  gpu_ms = floor_ms_per_token(gpu_bytes, vram_gbs * _GB)
  cpu_ms = floor_ms_per_token(cpu_bytes, cpu_rate * _GB) if cpu_bytes else 0.0
  link_ms = floor_ms_per_token(link_bytes, link_gbs * _GB) if link_bytes and semantics != 'none' else 0.0
  parts = {'gpu': gpu_ms, 'cpu': cpu_ms, 'link': link_ms}
  overlap_ms, sum_ms = max(parts.values()), gpu_ms + cpu_ms + link_ms
  notes = (['CPU part assumed RAM-bound (dequant rate unmeasured)'] if cpu_bytes and not cpu_gbs else []) + (['shapes estimated, not read from a header'] if estimated else []) + ([f'{words(link, "short")} at the measured {memory} host-to-device rate'] if memory else []) \
          + ([SEMANTICS_NOTE[semantics].format(the=words(link, "the"))] if semantics != 'copy' and link_bytes else []) + (unverified(link) if link_bytes else [])
  return {'estimated': estimated, 'notes': notes, 'hit_rate': hit_rate, 'cpu_share': cpu_share, 'bytes': {'gpu_vram': gpu_bytes, 'cpu_ram': cpu_bytes, 'link': link_bytes},
          'ms': parts, 'bound_by': max(parts, key=parts.get), 'cpu_rate_gbs': cpu_rate, **({'link_gbs': link_gbs, 'memory': memory} if memory else {}),
          'overlap': {'ms': overlap_ms, 'tok_s': 1000.0 / overlap_ms},
          'no_overlap': {'ms': sum_ms, 'tok_s': 1000.0 / sum_ms}}


def band(hit_rates, **kwargs) -> list[dict]:
  """The limit at each stated hit rate: the answer when the engine reports none (llama.cpp's LLAMA_MOE_CACHE_TRACE
  reports one; others do not). The rates are a sweep to read across, not an estimate of the run."""
  rates = list(hit_rates)
  if not rates: raise Refused('a band needs at least one hit rate')
  return [offload_limit(hit_rate=h, **kwargs) for h in rates]


def cpu_share(*, layers:int, cpu_moe:bool = False, n_cpu_moe:int | None = None, all_in_ram:bool = False, cache:bool = False,
              cpu_share:float = 0.0, hit_rate:float | None = None, strata:bool = False) -> dict:
  """The share of a token's expert bytes the CPU computes: experts that are neither GPU time nor GPU bytes, read from
  host RAM by the CPU (offload_limit's CPU term). Without a cache the CPU computes every expert kept in RAM (--cpu-moe
  or an -ot exps=CPU pattern: all; --n-cpu-moe N: the first N of `layers` expert layers); with one, the CPU share s of
  the misses, s x (1 - hit_rate), which is a range from 0 to s when the hit rate is not reported. Strata uploads its
  misses: none. {"share": the share or None when only the range is known, "low", "high", "basis"}."""
  def out(share, basis, low=None, high=None):
    return {"share": share, "low": share if low is None else low, "high": share if high is None else high, "basis": basis}
  if strata:
    return out(0.0, "Strata uploads the experts it misses; its CPU computes none")
  in_ram = 1.0 if cpu_moe or all_in_ram else min(n_cpu_moe or 0, layers) / layers if layers else 0.0
  if not in_ram:
    return out(0.0, "no expert is kept in RAM")
  if not cache:
    return out(in_ram, "no expert cache: the CPU computes every expert kept in RAM" + ("" if in_ram == 1 else f" ({n_cpu_moe} of {layers} layers)"))
  if not cpu_share:
    return out(0.0, "the expert cache uploads every miss (CPU share 0)")
  if hit_rate is None:
    return out(None, f"the CPU computes {cpu_share:g} of the cache misses, and the run did not report its hit rate", 0.0, cpu_share)
  return out(cpu_share * (1 - hit_rate), f"the CPU computes {cpu_share:g} of the cache misses at the reported hit rate {hit_rate:g}")
