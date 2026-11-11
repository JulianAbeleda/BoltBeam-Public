"""Speed up: a copy of a model whose output layer is quantized lighter, kept only if it still answers like the original.
The original is never modified. This core knows no engine: data/speed_up.json has a row per engine and
head_writers/<writer>.py the parts of that engine (find, stock, plan, write, accuracy). Steps: plan (refuse unless the head is
wider than the target), write <source>-boltbeam, gate (pooled next-token agreement on data/gate_corpus.json against the
table's threshold; a failing copy is deleted unless keep_failed), speed (original then copy at a short and a long context;
kept only if faster at every one by more than the spread), record (speed_up.json in the copy).
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import pathlib
import shutil
import statistics
from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.quantization import head_writers

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "speed_up.json"
CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "gate_corpus.json"
BOOTSTRAPS = 10000
SCHEMA = "boltbeam.speed_up.v1"
RECORD = "speed_up.json"
TEXT = "speed_up_text.json"
SUFFIX = "-boltbeam"
PARTIAL = ".partial"  # beside a copy while it is written: still there after a crash
SHORT_CONTEXT = 128
BATCHES = (1, 8, 32)  # streams decoded together; GATE_BATCH alone decides whether the copy stays
GATE_BATCH = 1
WANT_LONG = 32768  # the long context asked for; the largest the model and the GPU hold if less
SPEED_TOKENS, SPEED_REPS = 64, 3  # the decode driver's own (collectors/vllm_decode.py TOKENS, REPS)
KV_ELEMENT_BYTES = 2  # the engine's default KV cache dtype is the model's 16 bit
RESERVE_BYTES = 6 << 30  # graphs, activations, sampler and recurrent state beside weights and KV


@functools.lru_cache(maxsize=1)
def table() -> dict[str, Any]:
  return json.loads(TABLE.read_text())


def writer(row:dict[str, Any]):
  return importlib.import_module("boltbeam.quantization.head_writers." + row["writer"])


def row_for(fmt:str, engine:str | None = None) -> dict[str, Any]:
  rows = [r for r in table()["engines"] if fmt in r["formats"] and engine in (None, r["engine"])]
  if not rows:
    raise Refused(f"no engine writes a lighter output layer for the {fmt} format" + (f" on {engine}" if engine else ""))
  row = rows[0]
  if not row["runs"]:
    raise Refused(f"{row['label']}: {row['why']}")
  return row


STAGES = ("speed_write", "speed_check", "speed_measure")  # the stage keys data/plain_words.json names, in order


def offer(checkpoint:str) -> dict[str, Any]:
  """What Speed up would do to this model, without an engine or a GPU: enabled, or why not in plain words
  (plain_words.json speed_up), the copy's name and its size at most, and the seconds each stage is expected to take."""
  from boltbeam.report import plain
  say = plain.words()["speed_up"]
  src = pathlib.Path(checkpoint).expanduser().resolve()
  base = {"model": str(src), "name": display_name(src.stem if src.is_file() else src.name),
          "is_copy": False, "enabled": False, "reason": None, "copy_bytes": None, "expect": None, "gpu": True}
  beside = src / RECORD if src.is_dir() else src.with_name(src.name + "." + RECORD)
  if not src.exists():
    return {**base, "reason": say["missing"]}
  if beside.is_file():
    return {**base, "is_copy": True, "reason": say["already"]}
  try:
    row = row_for(head_writers.detect(src))
  except Refused:
    return {**base, "reason": say["no_writer"]}
  gib, e = checkpoint_bytes(src) / 2**30, table()["expect"]
  stages = STAGES if row.get("speed") else STAGES[:2]
  expect = {k: round(e["floor"][k] + e["per_gib"][k] * gib, 1) for k in stages}
  c = say["confirm"]
  return {**base, "enabled": True, "engine": row["label"], "copy_bytes": checkpoint_bytes(src), "expect": expect, "expect_source": "estimate",
          "confirm": {"title": c["title"].format(model=src.name), "yes": c["yes"], "no": c["no"],
                      "lines": [say["helper"], c["space"].format(size=f"{gib:.1f} GB"), c["time"].format(minutes=math.ceil(sum(expect.values()) / 60)), c["gpu"]]}}


def progress_lines(checkpoint:str, speed:bool = True) -> list[str]:
  """The header a screen reads before the stages (the pipeline's own lines): how many, and the seconds each is expected to take."""
  o = offer(checkpoint)
  keys = [k for k in (o["expect"] or {}) if speed or k != "speed_measure"]
  return [f"pipeline steps: {len(keys)}", "pipeline expect: " + ",".join(f"{o['expect'][k]:.1f}" for k in keys), "pipeline expect source: estimate"] if keys else []


def display_name(model:str) -> str:
  return f"{model} · BoltBeam"


def default_out(src:pathlib.Path) -> pathlib.Path:
  """<source>-boltbeam beside the source (a GGUF keeps its extension)."""
  return src.with_name(src.stem + SUFFIX + src.suffix) if src.is_file() else src.with_name(src.name + SUFFIX)


def plan(checkpoint:str, engine:str | None = None, engine_path:str | None = None) -> dict[str, Any]:
  """What a copy would do, or Refused. Steps: format, row, engine found, stock, output layer wider than the target."""
  src = pathlib.Path(checkpoint).expanduser().resolve()
  if not src.exists():
    raise Refused(f"{checkpoint} does not exist")
  beside = src / RECORD if src.is_dir() else src.with_name(src.name + "." + RECORD)
  hand = sorted(x.name for x in src.glob("model-lmhead*.safetensors")) if src.is_dir() else []
  if beside.is_file() or hand:  # a copy of a copy, or the hand-made -lmhead4: the head is already lighter
    raise Refused(f"{src.name} is already a lighter-head copy ({beside.name if beside.is_file() else hand[0]} is in it)")
  if src.is_dir() and (src / "config.json").is_file() and text_config(json.loads((src / "config.json").read_text())).get("tie_word_embeddings"):
    raise Refused("tie_word_embeddings is true in config.json: the output layer is the 16 bit embedding, so there is no separate head to lighten")
  fmt = head_writers.detect(src)
  row = row_for(fmt, engine)
  w = writer(row)
  path = getattr(w, row.get("find", "find"))(engine_path)
  if not path:
    raise Refused(f"{row['label']} was not found: pass --engine or set ${row['env']}")
  facts = getattr(w, row["stock"])(path)
  pl = getattr(w, row["plan"])(str(src), path, target=row.get("target"))
  if not pl["bits_target"] < pl["bits_now"]:
    raise Refused(f"the output layer is {pl['held']} ({pl['bits_now']:.2f} bits a weight), not wider than {pl['target']} ({pl['bits_target']:.2f})")
  return {"source": str(src), "format": fmt, "row": row, "engine_path": path, "stock": facts, "layer": pl,
          "copy": str(default_out(src)), "name": display_name(src.stem if src.is_file() else src.name)}


def corpus() -> list[str]:
  """The gate's text: the shipped natural-text excerpts, refused unless they hash to the table's sha256 and each to its own."""
  data = json.loads(CORPUS.read_text())
  texts = [d["text"] for d in data["documents"]]
  th = table()["threshold"]
  if hashlib.sha256(json.dumps(texts).encode()).hexdigest() != th["corpus_sha256"]:
    raise Refused(f"{CORPUS.name} does not hash to the corpus_sha256 in the table: the gate's text was changed")
  bad = [d["gutenberg_id"] for d in data["documents"] if hashlib.sha256(d["text"].encode()).hexdigest() != d["sha256"]]
  if bad or len(texts) < th["min_documents"]:
    raise Refused(f"{CORPUS.name}: " + (f"documents {bad} do not match their sha256" if bad else f"{len(texts)} documents, the gate needs {th['min_documents']}"))
  return texts


def pooled(docs:list[dict[str, int]]) -> float:
  return sum(d["hits"] for d in docs) / sum(d["n"] for d in docs)


def clustered_interval(docs:list[dict[str, int]], resamples:int = BOOTSTRAPS, seed:int = 0) -> list[float]:
  """The 95% percentile interval of the pooled share when whole documents are resampled with replacement (positions inside
  one document are not independent, so a Wilson interval over positions would be too narrow)."""
  import random
  rng = random.Random(seed)
  k = len(docs)
  draws = sorted(pooled([docs[rng.randrange(k)] for _ in range(k)]) for _ in range(resamples))
  return [draws[int(resamples * 0.025)], draws[int(resamples * 0.975) - 1]]


def _remove(p:pathlib.Path) -> None:
  shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)


def derive(checkpoint:str, engine:str | None = None, out:str | None = None, engine_path:str | None = None,
           keep_failed:bool = False, speed:bool = True, progress=None) -> dict[str, Any]:
  """Plan, write, gate, record. Returns the record; "passed" says whether the copy stays. Refused if it cannot start.
  progress(stage, event) hears STAGES start and finish ("start", "done"); a screen draws its bar from them."""
  say = progress or (lambda stage, event: None)
  pl = plan(checkpoint, engine, engine_path)
  row, src = pl["row"], pathlib.Path(pl["source"])
  dst = pathlib.Path(out).expanduser().resolve() if out else pathlib.Path(pl["copy"])
  mark = dst.with_name(dst.name + PARTIAL)
  if dst.exists() or mark.exists():
    raise Refused(f"{dst} already exists; the copy is never written over anything" if dst.exists() else f"{mark} is left by a run that crashed mid-copy: delete it and {dst}")
  w, path, th = writer(row), pl["engine_path"], table()["threshold"]
  texts = corpus()  # the gate's text: natural prose the original did not write, shipped and hashed
  digest = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
  mark.touch()
  try:
    say("speed_write", "start")
    changed = getattr(w, row["write"])(str(src), str(dst), path, pl["layer"])
    say("speed_write", "done")
    say("speed_check", "start")
    figures = getattr(w, row["accuracy"])(str(src), str(dst), path, texts)
    say("speed_check", "done")
    docs = figures.get("per_doc")
    if not docs or len(docs) != len(texts) or sum(d["n"] for d in docs) != figures["n"]:
      raise Refused("the accuracy check did not score every document of the gate corpus")
    if figures["n"] < th["min_positions"]:
      raise Refused(f"the accuracy check scored {figures['n']} positions, fewer than the {th['min_positions']} it needs to decide")
    if figures.get("engine_version", pl["stock"]["engine_version"]) != pl["stock"]["engine_version"]:
      raise Refused(f"the accuracy check ran on {figures['engine_version']}, not the {pl['stock']['engine_version']} found stock")
  except BaseException:
    _remove(dst)
    mark.unlink(missing_ok=True)
    raise
  share = [d["hits"] / d["n"] for d in docs]
  figures[th["metric"]] = pooled(docs)
  figures["interval"] = clustered_interval(docs)
  figures["interval_method"] = f"95% percentile, {BOOTSTRAPS} resamples of the {len(docs)} documents with replacement (seed 0)"
  figures["doc_min"], figures["doc_median"], figures["docs_at_or_above"] = min(share), statistics.median(share), sum(x >= th["min"] for x in share)
  passed = figures[th["metric"]] >= th["min"]
  timed = None
  if passed and speed and row.get("speed"):
    try:
      say("speed_measure", "start")
      timed = measure_speed(src, dst, row, w, path, pl["layer"])
      say("speed_measure", "done")
      if not (timed["engine"] and pl["stock"]["engine_version"].endswith(str(timed["engine"]))):
        raise Refused(f"the speed ran on engine build {timed['engine']}, not the {pl['stock']['engine_version']} found stock")
    except BaseException:
      _remove(dst)
      mark.unlink(missing_ok=True)
      raise
  rec = {"schema": SCHEMA, "source": str(src), "copy": str(dst), "name": pl["name"], "format": pl["format"],
         "engine": row["engine"], "engine_version": pl["stock"]["engine_version"], "stock": pl["stock"],
         "changed": changed, "bytes_now": pl["layer"]["bytes_now"], "bytes_after": pl["layer"]["bytes_after"],
         "accuracy": figures, "threshold": th, "text": {"file": TEXT, "sha256": digest, "positions": figures["n"], "corpus": CORPUS.name, "corpus_sha256": th["corpus_sha256"]}, "passed": passed,
         "speed": [timed] if timed else [], "speed_note": None if timed else "not measured: " + ("switched off" if not speed else "the engine row has no speed measurement" if passed else "the accuracy check failed")}
  slower = timed is not None and not timed["passed"]  # timed exists only if passed, so at most one reason below
  rec["kept"] = (passed and not slower) or keep_failed
  if slower:
    rec["deleted_because"] = timed["why"]
  if not passed:
    lo, hi = figures["interval"]
    rec["deleted_because"] = (f"next-token agreement {figures[th['metric']]:.4f} on the natural-text gate corpus is below {th['min']} (documents {figures['doc_min']:.4f} to "
                              f"{max(d['hits'] / d['n'] for d in docs):.4f}, interval {lo:.4f} to {hi:.4f})")
  if "deleted_because" in rec and not keep_failed:
    dst.with_name(dst.name + ".speed_up.json").write_text(json.dumps(rec, indent=2))  # the copy goes; its reason (and the per-document figures) stay beside it
  if rec["kept"]:
    beside = (lambda name: dst / name) if dst.is_dir() else (lambda name: dst.with_name(dst.name + "." + name))
    rec["sha256"] = {f.name: _sha(f) for f in ([dst / x for x in changed["files"]] if dst.is_dir() else [dst])}  # verify() checks the copy against these
    beside(RECORD).write_text(json.dumps(rec, indent=2))
    beside(TEXT).write_text(json.dumps({"sha256": digest, "texts": texts}))
  if not rec["kept"]:
    _remove(dst)
  mark.unlink(missing_ok=True)
  return rec


def _sha(p:pathlib.Path) -> str:
  h = hashlib.sha256()
  with p.open("rb") as f:
    while chunk := f.read(1 << 24):
      h.update(chunk)
  return h.hexdigest()


def verify(copy:str, engine:str | None = None, engine_path:str | None = None) -> dict[str, Any]:
  """A copy against its record: it passed and was kept, its files and its gate text still hash to the record, this machine's
  stock engine is the record's version, and the record holds a speed for this GPU. Refused on the first mismatch."""
  p = pathlib.Path(copy).expanduser().resolve()
  beside = (lambda name: p / name) if p.is_dir() else (lambda name: p.with_name(p.name + "." + name))
  if not beside(RECORD).is_file():
    raise Refused(f"{p} has no {RECORD}: it is not a BoltBeam copy, or the record was removed")
  rec = json.loads(beside(RECORD).read_text())
  if rec.get("schema") != SCHEMA or not rec.get("sha256"):
    raise Refused(f"{beside(RECORD).name} is not a {SCHEMA} record with file hashes")
  if not (rec["kept"] and rec["passed"]) or any(not e["passed"] for e in rec["speed"]):
    raise Refused("the record says this copy failed its gate or its speed check and was kept only as asked")
  for name, want in rec["sha256"].items():
    f = p / name if p.is_dir() else p
    if not f.is_file() or _sha(f) != want:
      raise Refused(f"{name} " + ("is missing" if not f.is_file() else "no longer matches the record's sha256"))
  texts = json.loads(beside(TEXT).read_text()) if beside(TEXT).is_file() else {}
  if hashlib.sha256(json.dumps(texts.get("texts")).encode()).hexdigest() != rec["text"]["sha256"]:
    raise Refused(f"{beside(TEXT).name} is missing or no longer matches the record's gate text hash")
  row = row_for(rec["format"], rec["engine"] if engine is None else engine)
  w = writer(row)
  path = getattr(w, row.get("find", "find"))(engine_path)
  now = getattr(w, row["stock"])(path)["engine_version"] if path else None
  if now != rec["engine_version"]:
    raise Refused(f"the record was made on {rec['engine_version']}; this machine has {now or 'no ' + row['label']}")
  if rec["speed"]:
    gpu = getattr(w, row["device"])(path)
    if gpu["target"] not in {e["gpu"]["target"] for e in rec["speed"]}:
      raise Refused(f"the speed was measured on {', '.join(e['gpu']['name'] for e in rec['speed'])}, not on this {gpu['name']}")
  return rec


def text_config(cfg:dict[str, Any]) -> dict[str, Any]:
  return cfg.get("text_config") or cfg


def kv_bytes_per_token(cfg:dict[str, Any]) -> int:
  """Bytes of KV cache one token adds: only full-attention layers hold one (layer_types, else every layer)."""
  t = text_config(cfg)
  types = t.get("layer_types")
  layers = sum(1 for x in types if x == "full_attention") if types else t["num_hidden_layers"]
  heads = t.get("num_key_value_heads") or t["num_attention_heads"]
  head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
  return layers * 2 * heads * head_dim * KV_ELEMENT_BYTES


def checkpoint_bytes(src:pathlib.Path) -> int:
  return src.stat().st_size if src.is_file() else sum(f.stat().st_size for f in src.glob("*.safetensors"))


SAFETENSORS_DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1}
NOT_READ_PER_TOKEN = ("visual.", "vision_tower", "vision_model", "multi_modal_projector", "mtp.", "nextn", "audio_tower")
ENGINE_KEYS = ("build", "max_model_len", "compilation_config", "enforce_eager", "warm", "seed", "call_order", "speculative_config",
               "max_num_batched_tokens", "max_num_seqs", "chunked_prefill", "attention_backend", "gemm_backends", "tensor_parallel_size")


def decode_read_bytes(src:pathlib.Path, cfg:dict[str, Any]) -> int:
  """Estimated bytes one decode token reads from the weights, from the safetensors headers: not the vision or audio tower, not
  the MTP (nextn) layers, and the input embedding table as one row (a lookup) unless it is tied to the output layer."""
  if src.is_file():
    return src.stat().st_size
  tied, vocab = bool(cfg.get("tie_word_embeddings", text_config(cfg).get("tie_word_embeddings"))), text_config(cfg)["vocab_size"]
  total = 0
  for f in sorted(src.glob("*.safetensors")):
    with f.open("rb") as h:
      try:
        header = json.loads(h.read(int.from_bytes(h.read(8), "little")))
      except (ValueError, OverflowError, MemoryError) as exc:
        raise Refused(f"{f.name} is not a safetensors file, so the bytes a token reads cannot be counted: {exc}")
    for name, t in header.items():
      if name == "__metadata__" or any(x in name for x in NOT_READ_PER_TOKEN):
        continue
      size = t["data_offsets"][1] - t["data_offsets"][0]
      total += size // vocab if name.endswith("embed_tokens.weight") and not tied else size
  return total


def long_context(cfg:dict[str, Any], weight_bytes:int, budget_bytes:int, tokens:int = SPEED_TOKENS, want:int = WANT_LONG,
                 reserve:int = RESERVE_BYTES) -> dict[str, Any]:
  """The long decode context: `want` if the model's position limit and the KV cache that fits (budget_bytes, the GPU's
  usable memory, less the weights and a reserve) allow it, else the largest multiple of 1024 that does, with the reason."""
  limit, kv = text_config(cfg)["max_position_embeddings"], kv_bytes_per_token(cfg)
  room = budget_bytes - weight_bytes - reserve
  fit = room // kv if kv else limit
  cap = limit - tokens - 8
  bounds = {"asked": want, "model_limit": cap, "kv_fit": int(fit)}
  ctx = min(bounds.values()) // 1024 * 1024
  if ctx < 4 * SHORT_CONTEXT:
    raise Refused(f"no long context fits: model limit {cap}, KV room {room / 2**30:.1f} GiB at {kv} bytes a token")
  bound = min(bounds, key=bounds.get)
  why = {"asked": f"{want} fits the model's {limit} positions and the {room / 2**30:.1f} GiB left for KV ({kv * ctx / 2**30:.2f} GiB used)",
         "model_limit": f"the model's position limit {limit} (less {tokens + 8} for the measured tokens)",
         "kv_fit": f"the KV cache that fits: {room / 2**30:.1f} GiB at {kv} bytes a token"}[bound]
  return {"context": ctx, "why": why, "bounds": bounds, "kv_bytes_per_token": kv, "kv_room_bytes": int(room)}


def predicted_pct(cfg:dict[str, Any], weight_bytes:int, saved_bytes:int, context:int, batch:int = 1) -> float:
  """An estimate of the gain if a decode step were only bytes read: `weight_bytes` (decode_read_bytes: what a token reads, not the
  checkpoint's size) plus every stream's KV at this context, with `saved_bytes` fewer. It is an estimate, not a bound: it
  measured above and below the real gain, and fixed per-step costs are not byte-proportional."""
  read = weight_bytes + kv_bytes_per_token(cfg) * context * batch
  return (read / (read - saved_bytes) - 1) * 100


def fits(cfg:dict[str, Any], kv_room:int, context:int, batch:int, tokens:int = SPEED_TOKENS) -> bool:
  """Whether `batch` streams of `context` + `tokens` positions hold their KV in the room left (a lower bound on the need:
  recurrent state is not counted, so the engine may still refuse; that is recorded as a skip)."""
  return batch * (context + tokens + 8) * kv_bytes_per_token(cfg) <= kv_room


def arm(row:dict[str, Any]) -> dict[str, Any]:
  """One arm at one context and batch from a decode driver row: the median tok/s a stream gets (and the batch's total)
  over its repeats, and their spread."""
  speeds = [1000.0 / ms for ms in row["step_samples_ms"]]
  med, n = statistics.median(speeds), row["batch"]
  return {"batch": n, "engine": row.get("engine"), "tok_s": med, "tok_s_total": med * n, "min": min(speeds), "max": max(speeds), "spread": max(speeds) - min(speeds),
          "spread_pct": (max(speeds) - min(speeds)) / med * 100, "reps": len(speeds)}


def compare(original:list[dict[str, Any]], copy:list[dict[str, Any]]) -> dict[str, Any]:
  """{context: {batch: ...}}: original and copy tok/s per stream, the change in %, and whether the copy is faster by more
  than the larger spread."""
  out:dict[str, Any] = {}
  for o in original:
    c = next((r for r in copy if r["context"] == o["context"] and r["batch"] == o["batch"]), None)
    if c is None:
      raise Refused(f"the copy has no row at context {o['context']} batch {o['batch']}")
    a, b = arm(o), arm(c)
    differ = engine_differences(a["engine"], b["engine"])
    if differ:
      raise Refused(f"the two arms ran on different engines at context {o['context']} batch {o['batch']}: " + "; ".join(differ))
    gain, band = b["tok_s"] - a["tok_s"], max(a["spread"], b["spread"])
    out.setdefault(str(o["context"]), {})[str(o["batch"])] = {"original": a, "copy": b, "change_pct": gain / a["tok_s"] * 100, "spread": band, "faster": gain > band}
  return out


def engine_differences(a:dict[str, Any] | None, b:dict[str, Any] | None) -> list[str]:
  """What differs between the two arms' engine facts (ENGINE_KEYS: version, graphs, attention backend, batching, speculation);
  the model path is not among them. A missing record is itself a difference: the arms cannot be shown to be alike."""
  if not a or not b:
    return ["an arm recorded no engine facts"]
  return [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in ENGINE_KEYS if a.get(k) != b.get(k)]


def keep_rule(by_context:dict[str, Any]) -> tuple[bool, str]:
  """The copy stays only if it is faster at every measured context on one stream (GATE_BATCH). Larger batches are recorded, not gated."""
  g = str(GATE_BATCH)
  slow = [f"{k}: {v[g]['change_pct']:+.1f}% vs spread {v[g]['spread']:.2f} tok/s" for k, v in by_context.items() if not v[g]["faster"]]
  return (not slow, "faster than the original at every measured context on one stream by more than the spread" if not slow
          else "not faster by more than the spread at " + "; ".join(slow))


def measure_speed(src:pathlib.Path, dst:pathlib.Path, row:dict[str, Any], w, path:str, pl:dict[str, Any]) -> dict[str, Any]:
  """One GPU's entry: original then copy through the engine's decode measurement at the short and the long context and
  at each batch of BATCHES that fits. A pair that does not fit, or that the engine refuses beyond one stream, is skipped
  and recorded with the reason."""
  cfg = json.loads((src / "config.json").read_text())
  weights, read_bytes = checkpoint_bytes(src), decode_read_bytes(src, cfg)
  info = getattr(w, row["device"])(path)
  lc = long_context(cfg, weights, info["budget_bytes"])
  skipped:list[dict[str, Any]] = []
  rows:dict[str, list[dict[str, Any]]] = {"original": [], "copy": []}
  for ctx in (SHORT_CONTEXT, lc["context"]):
    run = [n for n in BATCHES if n == GATE_BATCH or fits(cfg, lc["kv_room_bytes"], ctx, n)]
    skipped += [{"context": ctx, "batch": n, "why": f"{n} streams of {ctx + SPEED_TOKENS + 8} positions need {n * (ctx + SPEED_TOKENS + 8) * lc['kv_bytes_per_token'] / 2**30:.1f} GiB of KV, {lc['kv_room_bytes'] / 2**30:.1f} GiB is left"}
                for n in BATCHES if n not in run]
    for name, model in (("original", src), ("copy", dst)):
      try:
        got = getattr(w, row["speed"])(str(model), path, [ctx], run, SPEED_TOKENS, SPEED_REPS)
      except Refused:
        if len(run) == 1:
          raise
        got = []  # the engine refused the group: one batch at a time, the larger ones skipped if refused
        for n in run:
          try:
            got += getattr(w, row["speed"])(str(model), path, [ctx], [n], SPEED_TOKENS, SPEED_REPS)
          except Refused as exc:
            if n == GATE_BATCH:
              raise
            skipped.append({"context": ctx, "batch": n, "why": f"{name}: the engine refused it: {exc}"})
      rows[name] += got
  have = {(r["context"], r["batch"]) for r in rows["original"]} & {(r["context"], r["batch"]) for r in rows["copy"]}
  every = {(r["context"], r["batch"]) for v in rows.values() for r in v}
  skipped += [{"context": c, "batch": n, "why": "only one of the two arms returned it"} for c, n in sorted(every - have)]
  rows = {k: [r for r in v if (r["context"], r["batch"]) in have] for k, v in rows.items()}
  by = compare(rows["original"], rows["copy"])
  saved = pl["bytes_now"] - pl["bytes_after"]
  for k, per in by.items():
    for n, v in per.items():
      v["predicted_pct"] = predicted_pct(cfg, read_bytes, saved, int(k), int(n))
  ok, why = keep_rule(by)
  return {"gpu": {k: info[k] for k in ("target", "name", "backend", "memory_bytes")}, "batches": list(BATCHES), "gate_batch": GATE_BATCH, "tokens": SPEED_TOKENS, "reps": SPEED_REPS,
          "order": "original then copy", "decode_read_bytes": read_bytes, "contexts": by, "skipped": skipped, "long_context": lc,
          "spread": "max - min of the repeats' tok/s; the copy must win by more than the larger of the two arms'",
          "passed": ok, "why": why, "engine": rows["copy"][0].get("engine", {}).get("build")}


def words(rec:dict[str, Any]) -> list[str]:
  a, th = rec["accuracy"], rec["threshold"]
  entries = rec["speed"]
  slower = any(e["passed"] is False for e in entries)
  head = (f"{rec['name']}: kept at {rec['copy']}" + ("" if entries else " (SPEED NOT MEASURED)") if rec["kept"] and rec["passed"] and not slower else
          f"{rec['name']}: " + ("failed the accuracy check" if not rec["passed"] else "is not faster than the original") + (" (kept, as asked)" if rec["kept"] else " and was deleted"))
  speed = [rec["speed_note"]] if not entries else [line for e in entries for line in (
    f"speed on {e['gpu']['name']} ({e['gpu']['target']}), tok/s per stream, original -> copy: " + "; ".join(
      f"context {k} batch {n}: {v['original']['tok_s']:.1f} -> {v['copy']['tok_s']:.1f} ({v['change_pct']:+.1f}%)" for k, per in e["contexts"].items() for n, v in per.items()),
    *(f"SKIPPED context {x['context']} batch {x['batch']}: {x['why']}" for x in e["skipped"]), *([] if e["skipped"] else ["nothing skipped"]),
    f"long context {e['long_context']['context']}: {e['long_context']['why']}", e["why"])]
  return [head, *speed, f"changed: {rec['changed']['layer']} {rec['changed']['from']} -> {rec['changed']['to']}, everything else {rec['changed']['all_else']}",
          f"gives the original's top answer {a['top1_agreement'] * 100:.1f} times in 100 (pooled; documents clustered interval {a['interval'][0] * 100:.1f} to {a['interval'][1] * 100:.1f} over {a['n']} words; worst document {a.get('doc_min', 0) * 100:.1f}, median {a.get('doc_median', 0) * 100:.1f}; needs {th['min'] * 100:.0f} pooled); perplexity "
          f"{a.get('ppl_original') or 0:.3f} -> {a.get('ppl_copy') or 0:.3f}", f"engine: {rec['engine_version']}"]
