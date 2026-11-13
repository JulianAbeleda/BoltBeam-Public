"""Speed up: a copy of a model whose parts (data/speed_up.json parts: the output layer, the MTP draft layer) are quantized
lighter, kept only if it passes every gate its parts name. The original is never modified. This core knows no engine: the
table has a row per engine and head_writers/<writer>.py the functions of that engine (find, stock, a plan per part, write,
accuracy, speed, mtp). One part per run, on the original or on a copy (a copy's copy, gated against it, its links into it
hashed). Steps: plan (refuse unless the part is wider than the target and not already light in the source), write the copy
(its name says the parts it carries: <source>-boltbeam, <source>-boltbeam-mtp), gates (agreement: pooled next-token
agreement on data/gate_corpus.json against the table's threshold; decode: source then copy at a short and a long context,
faster at every one by more than the spread; mtp: speed with MTP on, source, copy, source again, the copy's best k over the
source's per prompt clearing the source's own change), record (speed_up.json in the copy). A copy that fails a gate is deleted.
"""
from __future__ import annotations

import datetime
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
MTP_LOGS = "speed_up_mtp"  # in the copy: the MTP measurement's logs, kept with a kept copy and deleted with any other
MTP_RUNS = ("source", "copy", "source again")  # measure_mtp's runs, in order (mtp_seconds counts them)
SPEED_GATES = {"decode", "mtp"}  # the gates that measure speed: a part with no other gate is untested with speed off
OLD_PARTS = ["output"]  # a record from before parts lightened the output layer only
OWN = (RECORD, TEXT, MTP_LOGS)  # the names the core writes in a copy
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


def stages(row:dict[str, Any], p:dict[str, Any], speed:bool = True) -> list[str]:
  """The stages a run of part p on this engine row goes through: the one list a screen is told (offer, progress_lines) and
  derive tells it (write; the agreement check; a speed measurement the part's gates and the row have)."""
  measures = speed and ("decode" in p["gates"] and row.get("speed") or "mtp" in p["gates"] and row.get("mtp"))
  return [k for k, on in zip(STAGES, (True, "agreement" in p["gates"], measures)) if on]


def offer(checkpoint:str) -> dict[str, Any]:
  """What Speed up would do to this model, without an engine or a GPU: each part the engine row writes, offered (its confirm
  and the seconds each stage is expected to take) or why not in plain words (plain_words.json speed_up; a copy is offered
  the parts it does not have yet); enabled if any part is, else the first part's reason."""
  from boltbeam.report import plain, speed_up_plain
  say = plain.words()["speed_up"]
  src = pathlib.Path(checkpoint).expanduser().resolve()
  beside = src / RECORD if src.is_dir() else src.with_name(src.name + "." + RECORD)
  base = {"model": str(src), "name": display_name(src.stem if src.is_file() else src.name),
          "is_copy": beside.is_file(), "enabled": False, "reason": None, "copy_bytes": None, "gpu": True, "parts": []}
  if not src.exists():
    return {**base, "reason": say["missing"]}
  try:
    row = row_for(head_writers.detect(src))
  except Refused as exc:
    return {**base, "reason": speed_up_plain.refusal(str(exc))["reason"] if "could not be read" in str(exc) else say["no_writer"]}
  light, parts = light_parts(src, row), []
  for name in [r["part"] for r in table()["parts"] if r["part"] in row["parts"]]:
    p, words = part(name, row), {"part": name, "label": say["pick"].format(action=say["action"], part=say["parts"][name]["name"])}
    try:
      check(src, row, p, light)
    except Refused as exc:
      parts.append({**words, "enabled": False, "reason": speed_up_plain.refusal(str(exc))["reason"]})
      continue
    mins = minutes(src, row, p)
    parts.append({**words, "enabled": True, "reason": None, "helper": say["row"].format(short=say["parts"][name]["short"], minutes=mins),
                  "expect": expected(src, row, p), "confirm": confirm(src, row, p, say, mins)})
  on = any(x["enabled"] for x in parts)
  return {**base, "enabled": on, "reason": None if on else parts[0]["reason"], "engine": row["label"], "copy_bytes": checkpoint_bytes(src) if on else None,
          "expect_source": "estimate", "parts": parts}


def minutes(src:pathlib.Path, row:dict[str, Any], p:dict[str, Any]) -> str:
  """The minutes part p is expected to take, as words read them: one figure, or a range when its measurement's length is
  not known ahead (the MTP sweep)."""
  lo, hi = (math.ceil(sum(expected(src, row, p, most=m).values()) / 60) for m in (False, True))
  return str(hi) if lo == hi else f"{lo} to {hi}"


def confirm(src:pathlib.Path, row:dict[str, Any], p:dict[str, Any], say:dict[str, Any], mins:str) -> dict[str, Any]:
  """The confirm step of part p: what changes, the time and the GPU up front (with why when it is a range), the disk space."""
  c, words = say["confirm"], say["parts"][p["part"]]
  return {"title": c["title"].format(model=src.name), "yes": c["yes"], "no": c["no"],
          "lines": [say["helper"], words["what"], c["time"].format(minutes=mins), *([words["why_range"]] if " to " in mins else []),
                    c["space"].format(size=f"{checkpoint_bytes(src) / 2**30:.1f} GB")]}


def expected(src:pathlib.Path, row:dict[str, Any], p:dict[str, Any], speed:bool = True, most:bool = True) -> dict[str, float]:
  """Seconds each stage of part p is expected to take on this checkpoint (stages()): the table's floor + per GiB, but the MTP
  measurement's from what it runs (mtp_seconds; most: its longest sweep, else its shortest). An estimate."""
  gib, e = checkpoint_bytes(src) / 2**30, table()["expect"]
  out = {k: e["floor"][k] + e["per_gib"][k] * gib for k in stages(row, p, speed)}
  if "speed_measure" in out and "mtp" in p["gates"]:
    out["speed_measure"] = mtp_seconds(gib, row, p)[most]
  return {k: round(v, 1) for k, v in out.items()}


def mtp_seconds(gib:float, row:dict[str, Any], p:dict[str, Any]) -> tuple[float, float]:
  """measure_mtp's seconds, shortest and longest sweep: each of MTP_RUNS starts the engine at k=0 and once per k: the
  writer's first ks (the row's mtp_ks), and at most one more per k past the largest up to core/draft_count.K_MAX, at the
  table's seconds per start for this size."""
  from boltbeam.core import draft_count
  e, first = table()["expect"]["mtp"], set(getattr(writer(row), row["mtp_ks"])())
  start = e["per_start"]["floor"] + e["per_start"]["per_gib"] * gib
  return tuple(len(MTP_RUNS) * (1 + len(ks)) * start for ks in (first, first | set(range(max(first) + 1, draft_count.K_MAX + 1))))


def progress_lines(checkpoint:str, speed:bool = True, want:str | None = None) -> list[str]:
  """The header a screen reads before the stages (the pipeline's own lines): how many (stages: the ones derive tells), and
  the seconds each is expected to take."""
  src = pathlib.Path(checkpoint).expanduser().resolve()
  try:
    row = row_for(head_writers.detect(src))
    ex = expected(src, row, part(want, row), speed)
  except Refused:
    return []  # derive refuses with the reason
  return [f"pipeline steps: {len(ex)}", "pipeline expect: " + ",".join(f"{v:.1f}" for v in ex.values()), "pipeline expect source: estimate"]


def display_name(model:str) -> str:
  return f"{model} · BoltBeam"


def copy_tail(carried:list[str]) -> str:
  """The one naming rule: a copy's name says the parts it carries (table order); the default part alone is plain -boltbeam."""
  rows = table()["parts"]
  carried = [r["part"] for r in rows if r["part"] in carried]
  return SUFFIX if carried == [r["part"] for r in rows if r["default"]] else SUFFIX + "".join("-" + p for p in carried)


def default_out(src:pathlib.Path, carried:list[str] | None = None, had:list[str] | None = None) -> pathlib.Path:
  """The copy beside the source (a GGUF keeps its extension), named by copy_tail; had: the parts the source carries, whose
  tail comes off its name first."""
  base, ext = (src.stem, src.suffix) if src.is_file() else (src.name, "")
  tail = copy_tail(had) if had else ""
  base = base[:-len(tail)] if tail and base.endswith(tail) else base
  return src.with_name(base + copy_tail(carried or [r["part"] for r in table()["parts"] if r["default"]]) + ext)


def light_parts(src:pathlib.Path, row:dict[str, Any]) -> dict[str, str]:
  """{part: the file that shows it}: the parts a source already has light, from its record, or a part's file of the engine
  row in it (a hand-made copy, such as a -lmhead4 folder)."""
  beside = src / RECORD if src.is_dir() else src.with_name(src.name + "." + RECORD)
  files = {p: h[0].name for p, x in row["parts"].items() if src.is_dir() and "file" in x and (h := sorted(src.glob(x["file"].format(bits="[0-9]*"))))}
  return {**files, **(dict.fromkeys(json.loads(beside.read_text()).get("parts") or OLD_PARTS, beside.name) if beside.is_file() else {})}


def part(name:str | None, row:dict[str, Any]) -> dict[str, Any]:
  """The table's part row asked for (default: the one marked default), merged with what the engine row says it needs;
  Refused for an unknown part or one this engine does not write."""
  rows = {r["part"]: r for r in table()["parts"]}
  name = name or next(k for k, r in rows.items() if r["default"])
  if name not in rows:
    raise Refused(f"no part named {name}; the parts are {', '.join(rows)}")
  if name not in row["parts"]:
    raise Refused(f"{row['label']} has no writer for the {rows[name]['label']} of the {row['formats'][0]} format")
  return {**rows[name], **row["parts"][name]}


def check(src:pathlib.Path, row:dict[str, Any], p:dict[str, Any], light:dict[str, str]) -> dict[str, Any]:
  """Part p planned on this checkpoint from its files alone (no engine), or Refused: already light in it, a tied output layer,
  what the writer's plan refuses, not wider than the target."""
  if p["part"] in light:  # a copy may get another part, never the same one twice
    raise Refused(f"{src.name} is already a lighter-head copy ({light[p['part']]} is in it): its {p['label']} is already light")
  try:  # the files are read here: a truncated file or a weight type no table knows refuses in words, never raises
    tied = src.is_dir() and (src / "config.json").is_file() and text_config(json.loads((src / "config.json").read_text())).get("tie_word_embeddings")
    if tied and p.get("untied"):
      raise Refused("tie_word_embeddings is true in config.json: the output layer is the 16 bit embedding, so there is no separate head to lighten")
    pl = {**p, **getattr(writer(row), p["plan"])(str(src), None, target=row.get("target"), part=p)}  # the part's rows and what the writer planned
  except (ValueError, KeyError, IndexError, EOFError, OSError) as exc:
    raise Refused(f"{src.name} could not be read: {type(exc).__name__} {exc}") from exc
  if not pl["bits_target"] < pl["bits_now"]:
    raise Refused(f"the {p['label']} is {pl['held']} ({pl['bits_now']:.2f} bits a weight), not wider than {pl['target']} ({pl['bits_target']:.2f})")
  return pl


def plan(checkpoint:str, engine:str | None = None, engine_path:str | None = None, want:str | None = None) -> dict[str, Any]:
  """What a copy would do, or Refused. Steps: format, row, part, check (from the files), engine found, stock."""
  src = pathlib.Path(checkpoint).expanduser().resolve()
  if not src.exists():
    raise Refused(f"{checkpoint} does not exist")
  row = row_for(fmt := head_writers.detect(src), engine)
  p, light = part(want, row), light_parts(src, row)
  pl = check(src, row, p, light)
  w = writer(row)
  if not (path := getattr(w, row.get("find", "find"))(engine_path)):
    raise Refused(f"{row['label']} was not found: pass --engine or set ${row['env']}")
  facts = getattr(w, row["stock"])(path)
  out = default_out(src, [*light, p["part"]], list(light))
  return {"source": str(src), "format": fmt, "row": row, "engine_path": path, "stock": facts, "layer": pl, "light": list(light),
          "copy": str(out), "name": display_name((out.stem if src.is_file() else out.name).replace(SUFFIX, "", 1))}


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


def agreement(figures:dict[str, Any], texts:list[str]) -> dict[str, Any]:
  """The gate's judgment of one accuracy check (per_doc {n, hits} over the corpus's documents): the pooled agreement, its
  document-clustered interval, the worst and median document; refused when a document went unscored or too few positions
  were scored to decide. The one place the gate's method lives: Speed up's copies and tune's output-changing knobs read it."""
  th, docs = table()["threshold"], figures.get("per_doc")
  if not docs or len(docs) != len(texts) or sum(d["n"] for d in docs) != figures["n"]:
    raise Refused("the accuracy check did not score every document of the gate corpus")
  if figures["n"] < th["min_positions"]:
    raise Refused(f"the accuracy check scored {figures['n']} positions, fewer than the {th['min_positions']} it needs to decide")
  share = [d["hits"] / d["n"] for d in docs]
  return {**figures, th["metric"]: pooled(docs), "interval": bootstrap(docs, pooled), "interval_method": f"95% percentile, {BOOTSTRAPS} resamples of the {len(docs)} documents with replacement (seed 0)",
          "doc_min": min(share), "doc_median": statistics.median(share), "docs_at_or_above": sum(x >= th["min"] for x in share)}


def passes(figures:dict[str, Any] | None) -> bool:
  """The gate: the pooled agreement (agreement()) at or above the table's threshold; no figures (a part with no agreement gate) passes."""
  th = table()["threshold"]
  return figures is None or figures[th["metric"]] >= th["min"]


def bootstrap(items:list, stat, resamples:int = BOOTSTRAPS, seed:int = 0) -> list[float]:
  """The 95% percentile interval of stat(items) when the items are resampled with replacement (seeded, so it repeats). The
  gate's documents: positions inside one are not independent, so a Wilson interval over positions would be too narrow."""
  import random
  rng, k = random.Random(seed), len(items)
  draws = sorted(stat([items[rng.randrange(k)] for _ in range(k)]) for _ in range(resamples))
  return [draws[int(resamples * 0.025)], draws[int(resamples * 0.975) - 1]]


def keep_logs(dst:pathlib.Path, why:str) -> pathlib.Path | None:
  """A copy that goes keeps its MTP measurement's logs: moved beside it as <copy>.<why>-<time>.speed_up_mtp; None when it has none."""
  logs = dst / MTP_LOGS
  if not logs.is_dir() or logs.is_symlink():
    return None
  kept = dst.with_name(f"{dst.name}.{why}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.{MTP_LOGS}")
  shutil.move(str(logs), kept)
  return kept


def _remove(p:pathlib.Path) -> None:
  shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)


def derive(checkpoint:str, engine:str | None = None, out:str | None = None, engine_path:str | None = None,
           speed:bool = True, progress=None, want:str | None = None) -> dict[str, Any]:
  """Plan, write, gate, record. Returns the record; "kept" says whether the copy stays. Refused if it cannot start.
  progress(stage, event) hears STAGES start and finish ("start", "done"); a screen draws its bar from them."""
  say = progress or (lambda stage, event: None)
  pl = plan(checkpoint, engine, engine_path, want)
  row, src, layer = pl["row"], pathlib.Path(pl["source"]), pl["layer"]
  gates = layer["gates"]
  if not speed and set(gates) <= SPEED_GATES:
    raise Refused(f"the {layer['label']} is gated on speed alone, so with the speed step switched off nothing would test it")
  dst = pathlib.Path(out).expanduser().resolve() if out else pathlib.Path(pl["copy"])
  mark = dst.with_name(dst.name + PARTIAL)
  if dst.exists() or mark.exists():
    raise Refused(f"{dst} already exists; the copy is never written over anything" if dst.exists() else f"{mark} is left by a run that crashed mid-copy: delete it and {dst}")
  w, path, th = writer(row), pl["engine_path"], table()["threshold"]
  texts = corpus() if "agreement" in gates else None  # the gate's text: natural prose the original did not write, shipped and hashed
  digest = hashlib.sha256(json.dumps(texts).encode()).hexdigest() if texts else None
  mark.touch()
  try:
    told = stages(row, layer, speed)
    say = (lambda stage, event, say=say: say(stage, event) if stage in told else None)  # only the stages the header announced
    say("speed_write", "start")
    changed = getattr(w, row["write"])(str(src), str(dst), path, layer)
    _own(dst)
    say("speed_write", "done")
    figures = None  # a part without the agreement gate: its row's note says why
    if "agreement" in gates:
      say("speed_check", "start")
      figures = getattr(w, row["accuracy"])(str(src), str(dst), path, texts)
      say("speed_check", "done")
      if figures.get("engine_version", pl["stock"]["engine_version"]) != pl["stock"]["engine_version"]:
        raise Refused(f"the accuracy check ran on {figures['engine_version']}, not the {pl['stock']['engine_version']} found stock")
      figures = agreement(figures, texts)
      docs = figures["per_doc"]
    passed = passes(figures)
    timed = drafted = None
    if passed and speed:
      say("speed_measure", "start")
      timed = measure_speed(src, dst, row, w, path, layer) if "decode" in gates and row.get("speed") else None
      drafted = measure_mtp(src, dst, row, w, path, layer) if "mtp" in gates else None
      say("speed_measure", "done")
      for e in filter(None, (timed, drafted)):
        if not (e["engine"] and pl["stock"]["engine_version"].endswith(str(e["engine"]))):
          raise Refused(f"the speed ran on engine build {e['engine']}, not the {pl['stock']['engine_version']} found stock")
  except BaseException as exc:
    kept = keep_logs(dst, "refused")
    _remove(dst)
    mark.unlink(missing_ok=True)
    if kept and isinstance(exc, Refused):  # the refusal names the logs where they now are
      raise Refused(str(exc).replace(str(dst / MTP_LOGS), str(kept)) + ("" if str(kept) in str(exc) else f" (its speed logs are kept in {kept})")) from exc
    raise
  rec = {"schema": SCHEMA, "source": str(src), "copy": str(dst), "name": pl["name"], "format": pl["format"],
         "engine": row["engine"], "engine_version": pl["stock"]["engine_version"], "stock": pl["stock"], "parts": [*pl.get("light", []), layer["part"]],
         "changed": changed, "bytes_now": layer["bytes_now"], "bytes_after": layer["bytes_after"], "accuracy": figures, "accuracy_note": None if figures else layer.get("note"),
         "threshold": th if figures else None, "passed": passed,  # a part with no agreement gate used no threshold and no gate text
         "text": {"file": TEXT, "sha256": digest, "positions": figures["n"], "corpus": CORPUS.name, "corpus_sha256": th["corpus_sha256"]} if figures else None,
         "speed": [timed] if timed else [], "mtp_speed": [drafted] if drafted else [],
         "speed_note": None if timed or drafted else "not measured: " + ("switched off" if not speed else "the engine row has no speed measurement" if passed else "the accuracy check failed")}
  rec["kept"] = passed and not slower(rec)
  if passed and not rec["kept"]:
    rec["deleted_because"] = "; ".join(e["why"] for e in measured(rec) if not e["passed"])
  if not passed:
    lo, hi = figures["interval"]
    rec["deleted_because"] = (f"next-token agreement {figures[th['metric']]:.4f} on the natural-text gate corpus is below {th['min']} (documents {figures['doc_min']:.4f} to "
                              f"{max(d['hits'] / d['n'] for d in docs):.4f}, interval {lo:.4f} to {hi:.4f})")
  if "deleted_because" in rec:
    rec["logs_kept"] = str(kept) if (kept := keep_logs(dst, "not-kept")) else None
    dst.with_name(dst.name + ".speed_up.json").write_text(json.dumps(rec, indent=2))  # the copy goes; its reason (and the per-document figures) stay beside it
  if rec["kept"]:
    beside = (lambda name: dst / name) if dst.is_dir() else (lambda name: dst.with_name(dst.name + "." + name))
    rec["sha256"] = {f.name: _sha(f) for f in ([dst / x for x in changed["files"]] if dst.is_dir() else [dst])}  # verify() checks the copy against these
    if pl.get("light") and dst.is_dir():  # a copy's copy reads the files it links in the copy it was built on: those are hashed too
      rec["base"] = {"copy": str(src), "sha256": {f.name: _sha(f) for f in sorted(dst.iterdir()) if f.is_symlink() and f.resolve().parent == src}}
    beside(RECORD).write_text(json.dumps(rec, indent=2))
    if figures:
      beside(TEXT).write_text(json.dumps({"sha256": digest, "texts": texts}))
  if not rec["kept"]:
    _remove(dst)
  mark.unlink(missing_ok=True)
  return rec


def _own(dst:pathlib.Path) -> None:
  """Take back the names the core writes in a copy (OWN): a copy's copy links its source's record, text and MTP logs, and a
  write through one of those links would change the source."""
  for name in OWN:
    if dst.is_dir() and (dst / name).is_symlink():
      (dst / name).unlink()


def measured(rec:dict[str, Any]) -> list[dict[str, Any]]:
  """Every speed entry of a record: plain decode and with MTP on (a record from before parts has no mtp_speed)."""
  return rec["speed"] + rec.get("mtp_speed", [])


def mtp_depths_valid(e:dict[str, Any]) -> tuple[dict[str, Any], bool]:
  """An MTP gate entry's depths that still stand, and whether a depth was dropped: an entry measured before
  collectors/mtp.SCHEMA (its long depth the model's own looping continuation, or real text under a weaker guard) keeps
  only its short depth."""
  from boltbeam.collectors import mtp as core
  if e.get("schema") == core.SCHEMA or len(e["depths"]) < 2:
    return e["depths"], False
  short = min(e["depths"], key=lambda d: e["depths"][d]["context"])
  return {short: e["depths"][short]}, True


def slower(rec:dict[str, Any]) -> bool:
  return any(e["passed"] is False for e in measured(rec))


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
  if not (rec["kept"] and rec["passed"]) or slower(rec):
    raise Refused("the record says this copy failed its gate or its speed check and was kept only as asked")
  for name, want in rec["sha256"].items():
    f = p / name if p.is_dir() else p
    if not f.is_file() or _sha(f) != want:
      raise Refused(f"{name} " + ("is missing" if not f.is_file() else "no longer matches the record's sha256"))
  texts = json.loads(beside(TEXT).read_text()) if beside(TEXT).is_file() else {}
  if rec["text"] and hashlib.sha256(json.dumps(texts.get("texts")).encode()).hexdigest() != rec["text"]["sha256"]:
    raise Refused(f"{beside(TEXT).name} is missing or no longer matches the record's gate text hash")
  row = row_for(rec["format"], rec["engine"] if engine is None else engine)
  w = writer(row)
  path = getattr(w, row.get("find", "find"))(engine_path)
  now = getattr(w, row["stock"])(path)["engine_version"] if path else None
  if now != rec["engine_version"]:
    raise Refused(f"the record was made on {rec['engine_version']}; this machine has {now or 'no ' + row['label']}")
  base = rec.get("base") or {}
  for name, want in base.get("sha256", {}).items():
    f = p / name
    if not f.is_file() or _sha(f) != want:
      raise Refused(f"the copy it was built on, {base['copy']}, " + ("is missing or lost " + name if not f.is_file() else f"changed: {name} no longer matches the record"))
  if measured(rec):
    gpu = getattr(w, row["device"])(path)
    if gpu["target"] not in {e["gpu"]["target"] for e in measured(rec)}:
      raise Refused(f"the speed was measured on {', '.join(e['gpu']['name'] for e in measured(rec))}, not on this {gpu['name']}")
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
GPU_KEYS = ("target", "name", "backend", "memory_bytes")  # the GPU a speed entry was measured on (verify matches target)
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
                 reserve:int = RESERVE_BYTES, engine_fit:tuple[int | None, str] | None = None) -> dict[str, Any]:
  """The long decode context: `want` if the model's position limit and the KV cache that fits (budget_bytes, the GPU's
  usable memory, less the weights and a reserve) allow it, else the largest multiple of 1024 that does, with the reason.
  cfg: config.json, or another format's {max_position_embeddings, kv_bytes_per_token}. engine_fit: the context the engine's
  own fit holds and how it says so (None: the model's whole context), in place of the byte count where the engine has one."""
  limit, kv = text_config(cfg)["max_position_embeddings"], cfg.get("kv_bytes_per_token") or kv_bytes_per_token(cfg)
  room = budget_bytes - weight_bytes - reserve
  fit = (engine_fit[0] or limit) if engine_fit else room // kv if kv else limit
  cap = limit - tokens - 8
  bounds = {"asked": want, "model_limit": cap, "kv_fit": int(fit)}
  ctx = min(bounds.values()) // 1024 * 1024
  if ctx < 4 * SHORT_CONTEXT:
    raise Refused(f"no long context fits: model limit {cap}, " + (engine_fit[1] if engine_fit else f"KV room {room / 2**30:.1f} GiB at {kv} bytes a token"))
  bound = min(bounds, key=bounds.get)
  why = {"asked": f"{want} fits the model's {limit} positions and the {room / 2**30:.1f} GiB left for KV ({kv * ctx / 2**30:.2f} GiB used)",
         "model_limit": f"the model's position limit {limit} (less {tokens + 8} for the measured tokens)",
         "kv_fit": engine_fit[1] if engine_fit else f"the KV cache that fits: {room / 2**30:.1f} GiB at {kv} bytes a token"}[bound]
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
  return {"gpu": {k: info[k] for k in GPU_KEYS}, "batches": list(BATCHES), "gate_batch": GATE_BATCH, "tokens": SPEED_TOKENS, "reps": SPEED_REPS,
          "order": "original then copy", "decode_read_bytes": read_bytes, "contexts": by, "skipped": skipped, "long_context": lc,
          "spread": "max - min of the repeats' tok/s; the copy must win by more than the larger of the two arms'",
          "passed": ok, "why": why, "engine": rows["copy"][0].get("engine", {}).get("build")}


def compare_mtp(first:dict[str, Any], copy:dict[str, Any], again:dict[str, Any], min_prompts:int) -> dict[str, Any]:
  """{depth: ...} from three runs of the engine's MTP measurement ({depth: {context, base, ks, tokens}}; base: each prompt's
  tok/s without drafts): the source, the copy, the source again. Per prompt, the copy at its best k over the source at its
  best k (the source's two runs averaged); each arm's best k is the highest median tok/s, chosen again inside each bootstrap
  resample of the prompts. Faster if the lower end of the 95% interval of the median is above 1 plus the source's own change
  between its two runs (at its best k). Recorded: the factor with MTP over without (each arm over its own base: each arm
  compiles its own graph) and the factor without drafts (copy base over source base), each a median over the prompts; per
  prompt their product is that prompt's ratio, whose median is the headline (change_pct)."""
  if not set(first) == set(copy) == set(again):
    raise Refused(f"the runs of the MTP measurement ran different depths: {sorted(first)}, {sorted(copy)}, {sorted(again)}")
  out:dict[str, Any] = {}
  for depth, f in first.items():
    runs, prompts = [{"base": x["base"], **{k: v["per_prompt"] for k, v in x["ks"].items()}} for x in (f, copy[depth], again[depth])], sorted(f["base"])
    if any(not set(r) <= set(runs[2]) or any(sorted(v) != prompts for v in r.values()) for r in runs):  # the source again ran every k
      raise Refused(f"the runs of the MTP measurement ran different prompts or draft counts at the {depth} depth")
    if len(prompts) < min_prompts:
      raise Refused(f"the MTP measurement ran {len(prompts)} prompts at the {depth} depth; the gate needs {min_prompts}")
    reads = [x.get("read") for x in (f, copy[depth], again[depth])]
    if not all(reads) or any(None in x.values() for x in reads) or not reads[0] == reads[1] == reads[2]:
      raise Refused(f"the runs of the MTP measurement read different text at the {depth} depth, or did not say what they read (another tokenizer or chat template)")
    s1, c, s2 = runs
    src = {k: {p: (s1[k][p] + s2[k][p]) / 2 if k in s1 else s2[k][p] for p in prompts} for k in s2}
    med = lambda t, k, ps: statistics.median(t[k][p] for p in ps)  # noqa: E731
    best = lambda t, ps: max((k for k in t if k != "base"), key=lambda k: med(t, k, ps))  # noqa: E731
    lo, hi = bootstrap(prompts, lambda ps: statistics.median(c[best(c, ps)][p] / src[best(src, ps)][p] for p in ps))
    kc, ks = best(c, prompts), best(src, prompts)
    shift = lambda k: statistics.median(s2[k][p] / s1[k][p] for p in prompts) - 1  # noqa: E731
    drift = shift(ks) if ks in s1 else max((shift(k) for k in s1 if k != "base"), key=abs)  # k past the first sweep: the largest shared change
    ratio = {p: c[kc][p] / src[ks][p] for p in prompts}
    has = lambda x: f if x == "base" or x in s1 else again[depth]  # noqa: E731  the source run that ran that k
    same = {x: sum(has(x).get("tokens", {}).get(x, {}).get(p) == copy[depth].get("tokens", {}).get(x, {}).get(p) is not None for p in prompts) for x in (kc, "base")}
    out[depth] = {"context": f["context"], "copy": {"k": int(kc), "tok_s": med(c, kc, prompts), "base_tok_s": med(c, "base", prompts), "accept_length": copy[depth]["ks"][kc].get("accept_length")},
                  "source": {"k": int(ks), "tok_s": med(src, ks, prompts), "base_tok_s": med(src, "base", prompts), "accept_length": (f["ks"].get(ks) or again[depth]["ks"][ks]).get("accept_length")},
                  "ratio": ratio, "change_pct": (statistics.median(ratio.values()) - 1) * 100, "interval_pct": [(lo - 1) * 100, (hi - 1) * 100], "source_change_pct": drift * 100, "source_change_at": int(ks) if ks in s1 else "largest shared",
                  "single_run": None if ks in s1 else f"the source's k={ks} ran once, in its second run: the median may be off by about half the source's own change ({abs(drift) * 50:.2f}%)",
                  "mtp_factor_pct": (statistics.median(c[kc][p] / c["base"][p] / (src[ks][p] / src["base"][p]) for p in prompts) - 1) * 100,
                  "base_factor_pct": (statistics.median(c["base"][p] / src["base"][p] for p in prompts) - 1) * 100,
                  "same_tokens": {"at_k": same[kc], "without_drafts": same["base"], "of": len(prompts), "gated": False}, "faster": lo > 1 + abs(drift)}
  return out


def measure_mtp(src:pathlib.Path, dst:pathlib.Path, row:dict[str, Any], w, path:str, layer:dict[str, Any]) -> dict[str, Any]:
  """One GPU's entry with MTP on: the source and the copy each on the derived draft-count sweep, then the source again at every
  k either reached (all on the first run's long-context text), logs in the copy's MTP_LOGS, compared per depth; kept only if
  faster at every depth. sweeps: each run's ks and why its sweep stopped. long_text: the long depth's corpus (collectors/mtp.long_corpus):
  an entry without it is from before the long depth was real text (mtp_depths_valid)."""
  folder, how, run = dst / MTP_LOGS, layer["measure"], getattr(w, row["mtp"])
  runs = {"source": run(str(src), path, folder / "source", {**how, "grow": True})}
  runs["copy"] = run(str(dst), path, folder / "copy", {**how, "grow": True})
  ks = {n: sorted({int(k) for d in r["depths"].values() for k in d["ks"]}) for n, r in runs.items()}
  runs["source_again"] = run(str(src), path, folder / "source_again", {**how, "ks": sorted(set(ks["source"]) | set(ks["copy"]))})
  if len({r["engine"] for r in runs.values()}) != 1:
    raise Refused(f"the runs with MTP on ran on different engine builds: {', '.join(str(r['engine']) for r in runs.values())}")
  by = compare_mtp(*(r["depths"] for r in runs.values()), layer["min_prompts"])
  slow = [f"{d}: median {v['change_pct']:+.1f}%, 95% interval {v['interval_pct'][0]:+.1f} to {v['interval_pct'][1]:+.1f}%, the source's own change {v['source_change_pct']:+.2f}%"
          for d, v in by.items() if not v["faster"]]
  info = getattr(w, row["device"])(path)
  sweeps = {n: {"ks": sorted({int(k) for d in r["depths"].values() for k in d["ks"]}), "stopped": r.get("sweep")} for n, r in runs.items()}
  from boltbeam.collectors import mtp as core
  return {"gpu": {k: info[k] for k in GPU_KEYS}, **how, "schema": core.SCHEMA, "long_text": {k: core.long_corpus()[k] for k in ("version", "sha256")}, "sweeps": sweeps, "order": ", ".join(MTP_RUNS), "logs": MTP_LOGS, "depths": by, "runs": {k: r["depths"] for k, r in runs.items()},
          "engine": runs["copy"]["engine"], "passed": not slow,
          "why": "with MTP on, faster than the source at every depth: the 95% interval of the median per-prompt ratio clears the source's own change between its two runs" if not slow
          else "with MTP on, not shown faster than the source (the 95% interval of the median per-prompt ratio does not clear the source's own change) at " + "; ".join(slow)}


def words(rec:dict[str, Any]) -> list[str]:
  a, th, slow = rec["accuracy"], rec["threshold"], slower(rec)
  head = (f"{rec['name']}: kept at {rec['copy']}" + ("" if measured(rec) else " (SPEED NOT MEASURED)") if rec["kept"] and rec["passed"] and not slow else
          f"{rec['name']}: " + ("failed the accuracy check" if not rec["passed"] else "is not faster than the original") + (" (kept, as asked)" if rec["kept"] else " and was deleted"))
  speed = [rec["speed_note"]] if not measured(rec) else [line for e in rec["speed"] for line in (
    f"speed on {e['gpu']['name']} ({e['gpu']['target']}), tok/s per stream, original -> copy: " + "; ".join(
      f"context {k} batch {n}: {v['original']['tok_s']:.1f} -> {v['copy']['tok_s']:.1f} ({v['change_pct']:+.1f}%)" for k, per in e["contexts"].items() for n, v in per.items()),
    *(f"SKIPPED context {x['context']} batch {x['batch']}: {x['why']}" for x in e["skipped"]), *([] if e["skipped"] else ["nothing skipped"]),
    f"long context {e['long_context']['context']}: {e['long_context']['why']}", e["why"])]
  speed += [line for e in rec.get("mtp_speed", []) for depths, gone in [mtp_depths_valid(e)] for line in (
    f"speed with MTP on, on {e['gpu']['name']} ({e['gpu']['target']}), {rec['source']} -> copy, each at its best k, {e['order']}: " + "; ".join(
      f"{d} ({v['context']:.0f} tokens): {v['source']['tok_s']:.1f} (k={v['source']['k']}) -> {v['copy']['tok_s']:.1f} (k={v['copy']['k']}) tok/s, median per prompt {v['change_pct']:+.1f}% "
      f"(95% interval {v['interval_pct'][0]:+.1f} to {v['interval_pct'][1]:+.1f}% over {len(v['ratio'])} prompts, needs to clear the source's own change {v['source_change_pct']:+.2f}%; "
      f"with MTP over without {v['mtp_factor_pct']:+.1f}%, without drafts {v['base_factor_pct']:+.2f}%; same output tokens on {v['same_tokens']['at_k']} of {v['same_tokens']['of']} "
      f"prompts at the copy's k and {v['same_tokens']['without_drafts']} without drafts, not gated: near-ties" + (f"; {v['single_run']}" if v.get("single_run") else "") + ")"
      for d, v in depths.items()), *(["the long depth is not shown: it was measured before the replies were checked for repetition (before collectors/mtp's v3), and some measured repetition"] if gone else []),
    *(["draft counts measured: " + "; ".join(
      f"{n.replace('_', ' ')} k={','.join(map(str, x['ks']))}" + (f" (stopped: {x['stopped']})" if x.get("stopped") else "") for n, x in e["sweeps"].items())] if e.get("sweeps") else []),
    e["why"])]
  ch = rec["changed"]
  return [head, *speed, f"changed: {ch['layer']} {ch['from']} -> {ch['to']}, everything else {ch['all_else']}", rec.get("accuracy_note") if not a else
          f"gives the original's top answer {a['top1_agreement'] * 100:.1f} times in 100 (pooled; documents clustered interval {a['interval'][0] * 100:.1f} to {a['interval'][1] * 100:.1f} over {a['n']} words; worst document {a.get('doc_min', 0) * 100:.1f}, median {a.get('doc_median', 0) * 100:.1f}; needs {th['min'] * 100:.0f} pooled); perplexity "
          f"{a.get('ppl_original') or 0:.3f} -> {a.get('ppl_copy') or 0:.3f}", f"engine: {rec['engine_version']}"]
