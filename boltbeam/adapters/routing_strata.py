"""The Strata routing adapter: Strata's own routing files read as a routing profile (no GPU, no run).

- `--dump-routing PATH` trace (UnbiasedStrata ff19037 src/program/generate.cpp:864, written at :1103-1122 and :1138-1145):
  records of i32 layer, i32 k, k i32 expert ids, k f32 weights, one per token per layer, in token order. Counts: a full
  profile.
- `--expert-profile` file (profile.bin; include/strata/core/expert_cache.hpp:40-55, read by src/core/expert_cache.cpp:74-140,
  written by tools/make_profile.py:61-70): "STRP", u32 version (1), n_layers, n_expert, slots, n_ranked, then n_ranked
  (u16 layer, u16 expert) pairs, hottest first, then an n_layers x n_expert i32 table of each pair's rank. A ranking with
  no counts: a cross-check of the hotness order only."""
from __future__ import annotations

import pathlib
import struct
from typing import Any

SOURCE = "UnbiasedStrata ff19037"


def kind(path:str | pathlib.Path) -> str:
  return "ranking" if pathlib.Path(path).read_bytes()[:4] == b"STRP" else "trace"


def read_ranking(path:str | pathlib.Path) -> dict[str, Any]:
  b = pathlib.Path(path).read_bytes()
  if b[:4] != b"STRP":
    raise ValueError(f"{path}: not a Strata expert profile (no STRP header)")
  version, nl, ne, _slots, n = struct.unpack_from("<5I", b, 4)
  if version != 1 or len(b) < 24 + 4 * n:
    raise ValueError(f"{path}: version {version} or truncated; this reader reads version 1")
  pairs = [list(struct.unpack_from("<HH", b, 24 + 4 * i)) for i in range(n)]
  if any(l >= nl or e >= ne for l, e in pairs):
    raise ValueError(f"{path}: a ranked pair is out of range")
  return {"n_layer": nl, "n_expert": ne, "ranking": pairs}


def read_trace(path:str | pathlib.Path, n_expert:int) -> dict[int, list[list[int]]]:
  b = pathlib.Path(path).read_bytes()
  off, tokens = 0, {}
  while off + 8 <= len(b):
    il, k = struct.unpack_from("<ii", b, off)
    if not 0 < k <= 64 or off + 8 + 8 * k > len(b):
      raise ValueError(f"{path}: a truncated or malformed record at byte {off} (k {k})")
    ids = struct.unpack_from(f"<{k}i", b, off + 8)
    off += 8 + 8 * k
    if not 0 <= il or any(not 0 <= e < n_expert for e in ids):
      raise ValueError(f"{path}: record out of range (layer {il}, ids {ids})")
    tokens.setdefault(il, []).append(list(ids))
  if off != len(b) or not tokens:
    raise ValueError(f"{path}: a truncated or empty routing trace")
  return tokens


def profile(path:str | pathlib.Path, model:dict[str, Any], *, n_expert:int, corpus:str, tokens:int | None = None) -> dict[str, Any]:
  from boltbeam.roofline import routing_profile as rp
  src = {"engine": "strata", "file": str(path), "corpus": corpus}  # no phase: the trace does not say if a record is a prompt token or a reply token
  if kind(path) == "ranking":
    r = read_ranking(path)
    return rp.record(None, model=model, n_expert=r["n_expert"], ranking=r["ranking"], source={**src, "how": f"{SOURCE} --expert-profile file (a ranking, no counts)"})
  sel = {il: rows[:tokens] if tokens else rows for il, rows in read_trace(path, n_expert).items()}
  return rp.record(rp.layers_from(sel), model=model, n_expert=n_expert, source={**src, "how": f"{SOURCE} --dump-routing trace"})
