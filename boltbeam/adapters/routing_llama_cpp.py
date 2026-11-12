"""The llama.cpp routing adapter: llama.cpp's own MoE cache trace (LLAMA_MOE_CACHE_TRACE, src/llama-moe-cache.cpp:390-391,
:494-499, master 7592b61d3) read as a routing profile. No patch: the trace holds the selected experts of every slot lookup
(:798-804, the [n_expert_used, n_tokens] ids the graph selected), in the format :642-645 states ("MCT1", u32 n_layer,
n_expert_used, n_group; per group u32 device, n_expert, n_layers, the layers, f64 split, split_sum, u64 group host bytes,
device host bytes, alignment, u32 n_tensors, u64 per tensor; then per lookup i32 layer, experts per token, n_ids, the ids).

What it takes: a lookup happens only with the MoE cache on (--moe-cache-mib, experts in host RAM: --cpu-moe) and only for
a batch of at most 32 tokens (get_slot_map, :693-695, max_batch :368): so the capture runs the real-text prefill driver
(runtime/llama_server_decode_drive.py --prefill, the corpus data/gate_corpus.json) with -b 32 -ub 32 and one pass
(--reps 0). Router choices do not depend on the chunk; only the speed does."""
from __future__ import annotations

import os
import pathlib
import struct
import sys
from typing import Any

MAGIC = b"MCT1"
UBATCH = 32  # llama-moe-cache.cpp:368 max_batch: a larger batch is not looked up, so not traced


def read_trace(path:str | pathlib.Path) -> dict[str, Any]:
  """{n_expert, top_k, tokens: {layer: [ids of each token]}} from an MCT1 trace. ValueError on another format."""
  b = pathlib.Path(path).read_bytes()
  if b[:4] != MAGIC:
    raise ValueError(f"{path}: not an MoE cache trace (no {MAGIC.decode()} header)")
  n_layer, k, n_group = struct.unpack_from("<3I", b, 4)
  off, n_expert = 16, 0
  for _ in range(n_group):
    _dev, ne, nl = struct.unpack_from("<3I", b, off)
    n_expert = max(n_expert, ne)
    off += 12 + 4 * nl + 16 + 24
    (nt,) = struct.unpack_from("<I", b, off)
    off += 4 + 8 * nt
  tokens: dict[int, list[list[int]]] = {}
  while off + 12 <= len(b):
    il, used, n = struct.unpack_from("<3i", b, off)
    off += 12
    if used <= 0 or n % used or off + 4 * n > len(b):
      raise ValueError(f"{path}: a truncated or malformed trace record at byte {off - 12}")
    ids = struct.unpack_from(f"<{n}i", b, off)
    off += 4 * n
    tokens.setdefault(il, []).extend(list(ids[i:i + used]) for i in range(0, n, used))
  if off != len(b) or not tokens:
    raise ValueError(f"{path}: a truncated or empty trace ({len(tokens)} layers)")
  return {"n_expert": n_expert, "top_k": k, "n_layer": n_layer, "tokens": tokens}


def profile(path:str | pathlib.Path, model:dict[str, Any], *, corpus:str, tokens:int | None = None, n_expert:int | None = None) -> dict[str, Any]:
  """The trace as a profile: each layer's first `tokens` tokens (the whole trace without it). n_expert: the trace states its own."""
  from boltbeam.roofline import routing_profile as rp
  t = read_trace(path)
  sel = {il: rows[:tokens] if tokens else rows for il, rows in t["tokens"].items()}
  return rp.record(rp.layers_from(sel), model=model, n_expert=t["n_expert"],
                   source={"engine": "llama.cpp", "how": "LLAMA_MOE_CACHE_TRACE (src/llama-moe-cache.cpp), the real-text prefill driver at -ub 32",
                           "phase": "prefill", "corpus": corpus, "file": str(path)})


def capture(model:str, out:pathlib.Path, *, tokens:int, ident:dict[str, Any], corpus:pathlib.Path, build:str | None = None,
            port:int = 0, cache_mib:int = 18000) -> dict[str, Any]:
  """Run the real-text prefill driver once with the trace on and read it. build: any program of the build, taken as its llama-server (default: as screen finds it)."""
  from boltbeam.collectors import engine_common as ec, llama_server_decode as lsd
  from boltbeam.workflow.tune_llama_cpp import server_of  # the build tune uses: any program of it maps to its llama-server
  server = str(server_of(build)) if build else lsd.find(os.environ.get("BOLTBEAM_LLAMA_BENCH"))
  if not server:
    raise FileNotFoundError("no llama-server found (beside llama-bench, $BOLTBEAM_LLAMA_SERVER or on PATH); pass --build")
  out.mkdir(parents=True, exist_ok=True)
  trace = out / "moe-cache-trace.bin"
  trace.unlink(missing_ok=True)
  flags = ["-ngl", "99", "--cpu-moe", "--moe-cache-mib", str(cache_mib), "-b", str(UBATCH), "-ub", str(UBATCH), "--no-warmup"]
  import json
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(port),
          "--ctx-size", str(tokens + lsd.HEADROOM), "--contexts", str(tokens), "--reps", "0", "--flags", json.dumps(flags),
          "--log", str(out / "server.log"), "--prefill", "--corpus", str(corpus)]
  ec.run_driver(argv, env={"LLAMA_MOE_CACHE_TRACE": str(trace)}, log=out / "driver.log")
  return profile(trace, ident, corpus=corpus.name, tokens=tokens)
