"""The llama.cpp hot-draft adapter (workflow/hot_draft.py): what is llama.cpp's in making a hot-set draft.

  experts  a GGUF's expert-axis tensors by llama.cpp's own naming (LLM_TENSOR_FFN_*_EXPS "*_exps*", the router
           FFN_GATE_INP "ffn_gate_inp.weight", its selection bias "exp_probs_b"), each with its expert axis last in GGUF
           order = outermost in the file, so one expert's bytes are one contiguous block. Names, not a bare shape test:
           other tensors can share the expert count's length.
  counts   the draft's route count, written by the engine's own diagnostic (LLAMA_DRAFT_ROUTE_COUNT=path, arkey_ie_mods
           llama.cpp 1.0.0 mod 18 draft-route-count): "# n_expert N n_used K tokens_decode D tokens_catchup C", then per
           expert "id hits mass hits mass" for 1-token passes (decode) and larger passes (catch-up).
  capture  Measure MTP's real-text run (collectors/llama_mtp.measure) with the diagnostic on: one run gives the counts and
           the full draft's acceptance; the same run on the trimmed draft gives its acceptance.
  trim     a copy keeping the chosen experts (expert_count = N; boltbeam.hot_draft.* keys say which of the parent's)."""
from __future__ import annotations

import pathlib
import struct
from typing import Any

SWITCH = "LLAMA_DRAFT_ROUTE_COUNT"
PASSES = ("decode", "catchup")
U32, ARRAY, STRING = 4, 9, 8


def is_expert(name:str, dims:tuple, n:int) -> bool:
  return bool(dims) and int(dims[-1]) == n and ("_exps" in name or name.endswith("ffn_gate_inp.weight") or "exp_probs_b" in name)


def experts(path:str | pathlib.Path) -> dict[str, Any]:
  """{n_expert, per_expert_bytes, arch}: one expert's bytes summed over the expert-axis tensors (file bytes = device bytes for weights)."""
  from boltbeam.profile import gguf
  from boltbeam.workflow.min_memory import nbytes
  kv, ts, _ = gguf.read_gguf_layout(path)
  arch = kv.get("general.architecture")
  n = int(kv.get(f"{arch}.expert_count") or 0)
  if n <= 0:
    raise ValueError(f"{pathlib.Path(path).name} has no experts ({arch}.expert_count)")
  per = sum(nbytes(nm, list(d), t) // n for nm, d, t, _ in ts if is_expert(nm, d, n))
  if per <= 0 or int(kv.get("split.count") or 1) > 1:
    raise ValueError(f"{pathlib.Path(path).name}: no expert tensors by llama.cpp's names, or a split file (one file only)")
  return {"n_expert": n, "per_expert_bytes": per, "arch": arch}


def counts(path:str | pathlib.Path) -> dict[str, Any]:
  """The route count as {n_expert, n_used, tokens: {pass: n}, hits: {pass: [..]}, mass: {pass: [..]}}."""
  lines = pathlib.Path(path).read_text().splitlines()
  head = lines[0].lstrip("# ").split()
  h = dict(zip(head[::2], map(int, head[1::2])))
  rows = [x.split() for x in lines[1:] if x.strip()]
  if len(rows) != h.get("n_expert") or any(len(r) != 5 for r in rows):
    raise ValueError(f"{path}: not a draft route count ({len(rows)} rows for n_expert {h.get('n_expert')})")
  rows.sort(key=lambda r: int(r[0]))
  return {"n_expert": h["n_expert"], "n_used": h.get("n_used"), "tokens": {p: h.get(f"tokens_{p}", 0) for p in PASSES},
          "hits": {p: [int(r[1 + 2 * i]) for r in rows] for i, p in enumerate(PASSES)},
          "mass": {p: [float(r[2 + 2 * i]) for r in rows] for i, p in enumerate(PASSES)}}


def capture(main:str, draft:str, k:int, *, out:pathlib.Path, server:str | None, routes:bool, backend:str) -> dict[str, Any]:
  """Measure MTP's real-text run of the draft at draft count k: its acceptance (accept_length per depth), and with routes
  the count file the diagnostic wrote ({acceptance, routes})."""
  from boltbeam.collectors import llama_mtp
  from boltbeam.workflow import mods
  path = out / "route-count.txt"
  env = {SWITCH: str(path)} if routes else {}
  path.unlink(missing_ok=True)
  with mods.switched({**mods.run_env(mods.modded("llama.cpp", server) or {}), **env}):
    rec = llama_mtp.measure(main, [k], backend=backend, out=out, head=draft, server=server)
  if routes and not path.is_file():
    raise ValueError(f"the server wrote no route count ({SWITCH}); is arkey_ie_mods draft-route-count in this build?")
  acc = {d: v[str(k)]["accept_length"] for d, v in rec["depths"].items() if str(k) in v}
  return {"acceptance": acc, "routes": str(path) if routes else None, "measurement": str(out)}


def _s(x:str) -> bytes:
  e = x.encode()
  return struct.pack("<Q", len(e)) + e


def trim(src:str | pathlib.Path, dst:str | pathlib.Path, ids:list[int], keep:dict[str, Any]) -> dict[str, Any]:
  """A copy of src keeping the experts ids (ascending) of every expert-axis tensor, expert_count = len(ids), and the
  provenance keys boltbeam.hot_draft.{parent_expert_count, expert_ids, routes_sha256}. {path, bytes, n}."""
  from boltbeam.profile import gguf
  from boltbeam.workflow.min_memory import nbytes
  src, dst, ids = pathlib.Path(src), pathlib.Path(dst), sorted(int(i) for i in ids)
  kv, ts, start = gguf.read_gguf_layout(src)
  arch, align = kv.get("general.architecture"), int(kv.get("general.alignment", 32))
  n = int(kv[f"{arch}.expert_count"])
  if not ids or ids[-1] >= n or len(set(ids)) != len(ids):
    raise ValueError(f"expert ids must be distinct and below {n}")
  meta = b""
  with gguf.GGUFReader(src) as r:  # copy every key as written, expert_count rewritten in its own width
    r.read(4)
    ver, nt, nkv = r.u32(), r.u64(), r.u64()
    for _ in range(nkv):
      k0 = r.f.tell()
      key, typ = r.string(), r.i32()
      v0 = r.f.tell()
      r.value(typ)
      end = r.f.tell()
      r.f.seek(k0)
      raw = r.read(end - k0)
      meta += raw[:v0 - k0] + len(ids).to_bytes(end - v0, "little") if key == f"{arch}.expert_count" else raw
  add = [("boltbeam.hot_draft.parent_expert_count", struct.pack("<i", U32) + struct.pack("<I", n)),
         ("boltbeam.hot_draft.expert_ids", struct.pack("<iiQ", ARRAY, U32, len(ids)) + struct.pack(f"<{len(ids)}I", *ids)),
         ("boltbeam.hot_draft.routes_sha256", struct.pack("<i", STRING) + _s(str(keep.get("routes_sha256", ""))))]
  meta += b"".join(_s(k) + v for k, v in add)
  plan, info, pos = [], b"", 0
  for name, dims, typ, o in ts:
    size = nbytes(name, list(dims), typ)
    cut = is_expert(name, dims, n)
    assert not cut or size % n == 0, name
    new = tuple(dims[:-1]) + (len(ids),) if cut else tuple(dims)
    plan.append((start + o, size, size // n if cut else None))
    info += _s(name) + struct.pack("<I", len(new)) + struct.pack(f"<{len(new)}Q", *new) + struct.pack("<iQ", typ, pos)
    pos += -(-(size // n * len(ids) if cut else size) // align) * align
  head = b"GGUF" + struct.pack("<IQQ", ver, nt, nkv + len(add)) + meta + info
  head += b"\0" * (-len(head) % align)
  tmp = dst.with_name(dst.name + ".part")
  with open(src, "rb") as f, open(tmp, "wb") as w:
    w.write(head)
    for at, size, per in plan:
      blocks = [(at + i * per, per) for i in ids] if per else [(at, size)]
      for a, ln in blocks:
        f.seek(a)
        w.write(f.read(ln))
      w.write(b"\0" * (-sum(ln for _, ln in blocks) % align))
  tmp.replace(dst)
  return {"path": str(dst), "bytes": dst.stat().st_size, "n": len(ids)}
