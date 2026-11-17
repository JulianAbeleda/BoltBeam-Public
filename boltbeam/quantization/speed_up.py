"""Speed up: a copy of a model whose parts (data/speed_up.json parts: the output layer, the MTP draft layer) are quantized
lighter, kept only if it passes every gate its parts name. The original is never modified. This core knows no engine: the
table has a row per engine and head_writers/<writer>.py the functions of that engine (find, stock, a plan per part, write,
accuracy, speed, mtp). One part per run, on the original or on a copy (a copy's copy, gated against it, its links into it
hashed). Steps: plan (refuse unless the part is wider than the target and not already light in the source), write the copy
(its name says the parts it carries: <source>-boltbeam, <source>-boltbeam-mtp), gates (agreement: pooled next-token
agreement on data/gate_corpus.json judged by bar() against the engine's own noise, else a paired task eval; decode: source then copy at a short and a long context,
faster at every one by more than the spread; mtp: speed with MTP on, source, copy, source again, the copy's best k over the
source's per prompt clearing the source's own change), record (speed_up.json in the copy). A copy that fails a gate is deleted;
one that needs the task eval stays for it.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import importlib
import inspect
import json
import math
import pathlib
import re
import shutil
import statistics
import sys
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
MTP_LOGS = "speed_up_mtp"  # in the copy (beside a file copy): the MTP measurement's logs, kept with a kept copy and deleted with any other
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
FLOORS = pathlib.Path.home() / ".boltbeam" / "noise_floor"  # noise_floor()'s cache


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
  beside = own(src, RECORD)
  base = {"model": str(src), "name": display_name(src.stem if src.is_file() else src.name),
          "is_copy": beside.is_file(), "enabled": False, "reason": None, "copy_bytes": None, "gpu": True, "parts": []}
  if not src.exists():
    return {**base, "reason": say["missing"]}
  try:
    usable(src)  # a copy that failed is not offered another part
    row = row_for(head_writers.detect(src))
  except Refused as exc:
    return {**base, "reason": speed_up_plain.refusal(str(exc))["reason"] if "could not be read" in str(exc) or "delete it" in str(exc) else say["no_writer"]}
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
  from boltbeam.collectors import mtp
  from boltbeam.core import draft_count
  e, first = table()["expect"]["mtp"], set(getattr(writer(row), row["mtp_ks"])())
  s = e["per_start"]  # each short prompt and each long document adds its decode to a start
  start = s["floor"] + s["per_gib"] * gib + s["per_prompt"] * len(mtp.prompt_set()["ids"]) + s["per_document"] * len(mtp.long_corpus()["ids"])
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
  beside = own(src, RECORD)
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


def check(src:pathlib.Path, row:dict[str, Any], p:dict[str, Any], light:dict[str, str], path:str | None = None) -> dict[str, Any]:
  """Part p planned on this checkpoint from its files (path None: no engine; path: the engine found, for what only it can
  say), or Refused: already light in it, a tied output layer, what the writer's plan refuses, not wider than the target."""
  if p["part"] in light:  # a copy may get another part, never the same one twice
    raise Refused(f"{src.name} is already a lighter-head copy ({light[p['part']]} is in it): its {p['label']} is already light")
  try:  # the files are read here: a truncated file or a weight type no table knows refuses in words, never raises
    tied = src.is_dir() and (src / "config.json").is_file() and text_config(json.loads((src / "config.json").read_text())).get("tie_word_embeddings")
    if tied and p.get("untied"):
      raise Refused("tie_word_embeddings is true in config.json: the output layer is the 16 bit embedding, so there is no separate head to lighten")
    pl = {**p, **getattr(writer(row), p["plan"])(str(src), path, target=row.get("target"), part=p)}  # the part's rows and what the writer planned
  except (ValueError, KeyError, IndexError, EOFError, OSError) as exc:
    raise Refused(f"{src.name} could not be read: {type(exc).__name__} {exc}") from exc
  if not pl["bits_target"] < pl["bits_now"]:
    raise Refused(f"the {p['label']} is {pl['held']} ({pl['bits_now']:.2f} bits a weight), not wider than {pl['target']} ({pl['bits_target']:.2f})")
  return pl


def plan(checkpoint:str, engine:str | None = None, engine_path:str | None = None, want:str | None = None, main:str | None = None) -> dict[str, Any]:
  """What a copy would do, or Refused. Steps: format, row, part, check (from the files), engine found, stock, check with the
  engine. main: the main model a draft head that is its own file runs with (llama.cpp's -md sidecar)."""
  src = pathlib.Path(checkpoint).expanduser().resolve()
  if not src.exists():
    raise Refused(f"{checkpoint} does not exist")
  usable(src)  # a copy built on a failed copy is refused; one built on a copy waiting on its eval inherits the wait (base_verdict)
  row = row_for(fmt := head_writers.detect(src), engine)
  p, light = {**part(want, row), **({"main": str(pathlib.Path(main).expanduser().resolve())} if main else {})}, light_parts(src, row)
  check(src, row, p, light)
  w = writer(row)
  if not (path := getattr(w, row.get("find", "find"))(engine_path)):
    raise Refused(f"{row['label']} was not found: pass --engine or set ${row['env']}")
  facts = getattr(w, row["stock"])(path)
  pl = check(src, row, p, light, path)
  out = default_out(src, [*light, p["part"]], list(light))
  return {"source": str(src), "format": fmt, "row": row, "engine_path": path, "stock": facts, "layer": pl, "light": list(light), "base_verdict": base_verdict(src),
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
          "doc_min": min(share), "doc_median": statistics.median(share)}


def z(confidence:float | None = None, df:int | None = None) -> float:
  """The one-sided quantile at confidence (threshold.confidence): normal, or Student t with df degrees of freedom by the
  Cornish-Fisher series in 1/df (Abramowitz and Stegun 26.7.5; 1.8946 at df 7, 0.95)."""
  x = statistics.NormalDist().inv_cdf(table()["threshold"]["confidence"] if confidence is None else confidence)
  g = [(x**3 + x) / 4, (5 * x**5 + 16 * x**3 + 3 * x) / 96, (3 * x**7 + 19 * x**5 + 17 * x**3 - 15 * x) / 384,
       (79 * x**9 + 776 * x**7 + 1482 * x**5 - 1920 * x**3 - 945 * x) / 92160]
  return x + sum(c / df ** (i + 1) for i, c in enumerate(g)) if df else x


def bar(figures:dict[str, Any], floor:dict[str, Any] | None = None) -> dict[str, Any]:
  """Stage 1 of the agreement gate, the one place its rule is computed (Speed up's copies, tune's output-changing knobs).
  d = 1 - pooled agreement; pass iff d_copy - d_floor <= t SE and every document holds, else needs_eval (a paired
  downstream eval decides: noninferior). d_floor: the engine's own noise (noise_floor), paired by document and position.
  SE: the document bootstrap SD of d_copy - d_floor (one SE, so a wider one never comes from a second estimate); t: Student
  t at threshold.confidence, k - 1 df for k documents. Per document (it can only send a copy to the eval): copy_i -
  floor_i <= z sqrt(var_c,i + var_f,i), var = d(1 - d) / n, z Sidak's for k documents together, the quantile at
  confidence^(1/k). No floor (an engine that declares no change): nothing measured to credit, so needs_eval, never a
  test against an invented SE; only a copy that agrees at every position passes, as no noise is smaller than none."""
  th, docs = table()["threshold"], figures["per_doc"]
  conf, m = th["confidence"], th["metric"]
  if not floor:
    return {"verdict": "pass" if figures[m] == 1 else "needs_eval", "d_copy": 1 - figures[m], "d_floor": None, "confidence": conf, "se_method": "no floor"}
  if [d["n"] for d in floor["per_doc"]] != [d["n"] for d in docs]:
    raise Refused("the noise floor and the check scored different positions per document, so they cannot be paired")
  k, base, miss = len(docs), floor["per_doc"], lambda d: 1 - d["hits"] / d["n"]  # noqa: E731
  d_copy, d_floor = 1 - pooled(docs), 1 - pooled(base)
  se, t = statistics.stdev(draws(list(zip(base, docs)), lambda pairs: pooled([f for f, _ in pairs]) - pooled([c for _, c in pairs]))), z(conf, k - 1)
  zs = z(conf ** (1 / k))
  per = [(miss(c) - miss(f), zs * math.sqrt((miss(c) * (1 - miss(c)) + miss(f) * (1 - miss(f))) / c["n"])) for c, f in zip(docs, base)]
  e, allowed = max(per, key=lambda x: x[0] - x[1])
  doc = {"excess": e, "allowed": allowed, "z": zs, "holds": all(x <= a for x, a in per)}
  return {"verdict": "pass" if d_copy - d_floor <= t * se and doc["holds"] else "needs_eval", "d_copy": d_copy, "d_floor": d_floor, "excess": d_copy - d_floor,
          "se": se, "z": t, "df": k - 1, "confidence": conf, "allowed": t * se, "needs": 1 - d_floor - t * se, "worst": doc, "se_method": "the document bootstrap, paired"}


def paired(a:list[dict], b:list[dict]) -> dict[str, Any]:
  """Two arms of a paired eval (rows id, correct, finish): b items only a got right, c only b; refused unless both arms
  answered the same items, each once, every row whole."""
  for rows in (a, b):
    if not rows or any(not isinstance(r, dict) or not isinstance(r.get("correct"), bool) or "id" not in r or "finish" not in r for r in rows):
      raise Refused("an eval arm is empty or has a row without id, a true/false correct and finish")
    if len({r["id"] for r in rows}) != len(rows):
      raise Refused("an eval arm answers an item twice")
  x, y = ({r["id"]: r for r in rows} for rows in (a, b))
  if x.keys() != y.keys():
    raise Refused(f"the two eval arms did not answer the same items: {len(x.keys() ^ y.keys())} are in one arm only")
  return {"n": len(x), "b": sum(x[i]["correct"] > y[i]["correct"] for i in x), "c": sum(y[i]["correct"] > x[i]["correct"] for i in x),
          "truncated": [sum(r["finish"] == "length" for r in rows) for rows in (a, b)]}


def noninferior(original:list[dict], copy:list[dict], noise:list[dict], tolerance:str | float | None = None, how:str = "on its noise run") -> dict[str, Any]:
  """Stage 2: the copy's paired eval loss against the engine's own eval noise (noise: the original run again under the
  engine's declared change, perturbation(), or as it was when none is declared), on the same items. The two pairs share
  the original, so L_copy - L_noise is exactly D = mean(noise_i - copy_i), each item 1 right, 0 wrong. The standard
  non-inferiority rule against the tolerance, computed exactly per item: u_i = (noise_i - copy_i) - t_i, t_i = |noise_i -
  orig_i| for "psi0" (so mean(u) = D - psi0, the allowance measured on the same items and its correlation with D carried
  in u's own SE), t_i = the share for a fixed tolerance (0: shown no worse than the noise run). fail: mean(u) - z SE_u > 0
  (proven worse than allowed); pass: mean(u) + z SE_u <= 0; else inconclusive, with the n at which an equal copy passes at
  threshold.eval.power, (z_a + z_b)^2 var(u) / allowance^2, as a hint. SE_u = sqrt(var(u) / n). tolerance
  (threshold.eval.tolerance): "psi0", the engine's own gross churn, the share of answers its noise run changes against
  the original (refused when 0: nothing to measure the copy against); 0; or a fixed share. Truncation (finish=length)
  over threshold.eval.max_truncation in any arm: inconclusive, BoltBeam's eval budget rule. A "run N more" hint re-rolls
  all three arms if they are rerun: repeating the whole eval until it passes inflates the error rate, so run the noise
  arm once and add the new items to all three arms together."""
  th = table()["threshold"]
  ev, conf, tol = th["eval"], th["confidence"], th["eval"]["tolerance"] if tolerance is None else tolerance
  p, p0, d = paired(original, copy), paired(original, noise), paired(noise, copy)
  n, churn, za = d["n"], (p0["b"] + p0["c"]) / d["n"], z(conf)
  if tol == "psi0" and not churn:
    raise Refused("this engine shows no eval noise: its run of the original again changed no answer, so there is nothing to measure the copy "
                  "against; declare a perturbation for it (threshold.noise.engines)")
  o, c, y = ({r["id"]: int(r["correct"]) for r in rows} for rows in (original, copy, noise))
  u = [y[i] - c[i] - (abs(y[i] - o[i]) if tol == "psi0" else float(tol)) for i in o]
  allow, loss, mean = churn if tol == "psi0" else float(tol), (d["b"] - d["c"]) / n, sum(u) / n
  var = sum((x - mean) ** 2 for x in u) / n
  se, cut = math.sqrt(var / n), max(x / n for x in p["truncated"] + p0["truncated"])
  need = math.ceil((za + z(ev["power"])) ** 2 * var / allow ** 2) if allow else None
  verdict = ("inconclusive" if cut > ev["max_truncation"] else "fail" if mean - za * se > 0 else "pass" if mean + za * se <= 0 else "inconclusive")
  said = (f"the engine's own churn, {allow * 100:.1f} points on this model (answers it changes against itself {how})" if tol == "psi0" else
          "none: a pass is shown no worse than the noise run" if not allow else f"a fixed {allow * 100:.1f} points")
  return {"verdict": verdict, "n": n, "copy": p, "noise": p0, "against_noise": d, "loss": loss, "margin": mean, "se": se, "z": za, "churn": churn, "tolerance": tol,
          "allowed": allow, "allowed_said": said, "n_needed": need, "power": ev["power"], "confidence": conf,
          "why": (f"truncation {cut:.1%} is over {ev['max_truncation']:.0%}" if cut > ev["max_truncation"] else None if verdict != "inconclusive" else
                  f"underpowered: run {need - n} more items, added to all three arms (rerunning the whole eval until it passes inflates the error rate)" if need and need > n else "its loss is neither shown within the allowance nor shown beyond it")}


def model_files(src:pathlib.Path) -> list[pathlib.Path]:
  """The files an engine reads for a model: a folder's files, a file and the other shards of a split (<stem>-0000k-of-0000N)."""
  if src.is_dir():
    return sorted(f for f in src.rglob("*") if f.is_file())
  m = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})(\.\w+)", src.name)
  shard = m and re.compile(re.escape(m[1]) + r"-\d{5}-of-" + m[2] + re.escape(m[3]))
  return sorted(f for f in src.parent.iterdir() if shard.fullmatch(f.name)) if shard else [src]


def setting(flags:list[str], env:dict | None, spec:dict[str, Any]) -> str | None:
  """The value a setup gives one engine setting (spec: its flags, a '--' flag's '_' read as '-', and its env names): the
  last flag given, else its env; spec accumulates: env then every flag, joined by ','. None: unset."""
  flags = [f.replace("_", "-") if f.startswith("--") else f for f in flags]
  got = [flags[i + 1] for i, f in enumerate(flags[:-1]) if f in spec["flags"]]
  envs = [v for e in ([spec["env"]] if isinstance(spec["env"], str) else spec["env"]) if (v := (env or {}).get(e)) is not None]
  return ",".join(envs + got) or None if spec.get("accumulates") else got[-1] if got else next(iter(envs), None)


def drafted(row:dict[str, Any], text:str) -> list[str]:
  """The draft flags and types of an engine's tail rule that text (an eval's config) names."""
  t = row.get("tail") or {}
  return [w for w in [*t.get("draft", {}).get("flags", ()), *t.get("types", {}).get("rollback", ())] if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text)]


def perturbation(engine:str, flags=None, env:dict | None = None) -> dict[str, Any] | None:
  """The change an engine declares must not matter (threshold.noise.engines): the one place Speed up and tune read it.
  None: none declared, so no floor, and its eval noise is the original run again unchanged. Derived for the setup it
  changes (flags, env; flags None: not known): the batch, raised to the tail bound where the setup forces one; refused if
  the setup runs at it."""
  if not (row := table()["threshold"]["noise"]["engines"].get(engine)) or "batch" not in row:
    return row
  t, base, known, flags = row["tail"], row["batch"], flags is not None, [str(f) for f in flags or ()]
  types = setting(flags, env, t["types"])
  on = bool(set(types.split(",")) & set(t["types"]["rollback"])) if types else setting(flags, env, t["draft"]) is not None
  n = int(setting(flags, env, t["n_max"]) or t["n_max"]["default"])
  b, ub = (int(setting(flags, env, s) or s["default"]) for s in row["own"])  # -ub 0: -b; -b caps -ub
  batch, own = max(base, n + t["plus"]) if on else base, min(b, ub or b) if known else row["unknown"]
  fill = lambda x: x.format(batch=batch, own=own, base=base, n=n, plus=t["plus"])  # noqa: E731
  if own == batch:
    raise Refused(f"the setup already runs {engine} at a batch of {batch} tokens, so the declared change to {batch} would change nothing: "
                  "no noise to measure; run the setup at its own batch, or declare another change (threshold.noise.engines)")
  return {**row, "args": {k: [fill(x) for x in v] for k, v in row["args"].items()}, "batch": batch, "rule": fill(t["rule"]), "raised": batch != base,
          "says": fill(row["says"]) + (f" ({fill(t['says'])})" if batch != base else "")}


def stamp(f:pathlib.Path) -> list:
  return [str(f), (st := f.stat()).st_size, st.st_mtime_ns]


def noise_key(src:pathlib.Path, w, path:str, **more) -> dict[str, Any]:
  """What a measurement of the engine's own noise on this model depends on, the key of its cache entry: the model's
  files, the engine build and device (the writer's build_id and device: one identity, Speed up's and tune's), the
  accuracy module, this gate's agreement code, and what the caller adds (the texts, the arguments, the eval items)."""
  return {"model": [stamp(f) for f in model_files(src)], "build": w.build_id(path), "device": w.device(path)["target"],
          "check": _sha(pathlib.Path(inspect.getfile(w.accuracy))), "gate": hashlib.sha256(inspect.getsource(agreement).encode()).hexdigest(), **more}


def cached(key:dict[str, Any], measure, whole=lambda entry: True) -> dict[str, Any]:
  """FLOORS' entry for key, read back only when its stored key is this key and whole(entry); else measure(), stored."""
  key = json.loads(json.dumps(key))
  f = FLOORS / (hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:32] + ".json")
  try:
    if (got := json.loads(f.read_text())).get("key") == key and whole(got):
      return {**got, "cached": str(f)}
  except (OSError, ValueError):
    pass  # no entry, or one cut short: measure
  got = {**measure(), "key": key}
  f.parent.mkdir(parents=True, exist_ok=True)
  f.with_suffix(".part").write_text(json.dumps(got, indent=2))
  f.with_suffix(".part").replace(f)
  return {**got, "cached": None}


def noise_floor(src:pathlib.Path, engine:str, w, path:str, texts:list[str], **setup) -> dict[str, Any] | None:
  """The engine's own noise on this model (threshold.noise): agreement() of the original against itself, w.accuracy(src,
  src, path, texts, **setup) with the engine's perturbation() for dst_flags added (a list argument is extended, so the declared flags
  come last and win); None for an engine without one. Speed up and tune both call it, and share its cache."""
  if not (noise := perturbation(engine, setup.get("dst_flags", ()), setup.get("env"))):
    return None
  args = {**setup, **{k: [*setup.get(k, ()), *v] if isinstance(v, list) else v for k, v in noise["args"].items()}}
  key = noise_key(src, w, path, texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(), args=args)
  return cached(key, lambda: {**agreement(w.accuracy(str(src), str(src), path, texts, **args), texts), "says": noise["says"], "args": args},
                lambda e: len(e.get("per_doc") or ()) == len(texts))


def draws(items:list, stat, resamples:int = BOOTSTRAPS, seed:int = 0) -> list[float]:
  """stat(items) over resamples of the items with replacement (seeded, so it repeats), sorted. The gate's documents:
  positions inside one are not independent, so a Wilson interval over positions would be too narrow."""
  import random
  rng, k = random.Random(seed), len(items)
  return sorted(stat([items[rng.randrange(k)] for _ in range(k)]) for _ in range(resamples))


def bootstrap(items:list, stat, resamples:int = BOOTSTRAPS, seed:int = 0) -> list[float]:
  """The 95% percentile interval of draws()."""
  d = draws(items, stat, resamples, seed)
  return [d[int(resamples * 0.025)], d[int(resamples * 0.975) - 1]]


def own(p:pathlib.Path, name:str) -> pathlib.Path:
  """Where the core's file `name` (OWN) of copy p is: in a folder copy, beside a file copy as <file>.<name>."""
  return p / name if p.is_dir() else p.with_name(p.name + "." + name)


def keep_logs(dst:pathlib.Path, why:str) -> pathlib.Path | None:
  """A copy that goes keeps its MTP measurement's logs: moved beside it as <copy>.<why>-<time>.speed_up_mtp; None when it has none."""
  logs = own(dst, MTP_LOGS)
  if not logs.is_dir() or logs.is_symlink():
    return None
  kept = dst.with_name(f"{dst.name}.{why}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.{MTP_LOGS}")
  shutil.move(str(logs), kept)
  return kept


def _remove(p:pathlib.Path) -> None:
  shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)


def derive(checkpoint:str, engine:str | None = None, out:str | None = None, engine_path:str | None = None,
           speed:bool = True, progress=None, want:str | None = None, main:str | None = None) -> dict[str, Any]:
  """Plan, write, gate, record. Returns the record; "kept" says whether the copy stays. Refused if it cannot start.
  progress(stage, event) hears STAGES start and finish ("start", "done"); a screen draws its bar from them."""
  say = progress or (lambda stage, event: None)
  pl = plan(checkpoint, engine, engine_path, want, main)
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
    figures = floor = None  # a part without the agreement gate: its row's note says why
    if "agreement" in gates:
      say("speed_check", "start")
      figures = getattr(w, row["accuracy"])(str(src), str(dst), path, texts)
      if figures.get("engine_version", pl["stock"]["engine_version"]) != pl["stock"]["engine_version"]:
        raise Refused(f"the accuracy check ran on {figures['engine_version']}, not the {pl['stock']['engine_version']} found stock")
      figures = agreement(figures, texts)
      floor = noise_floor(src, row["engine"], w, path, texts)
      say("speed_check", "done")
    gate = bar(figures, floor) if figures else None
    timed = drafted = None
    if speed:  # stage 1 passes or needs the task eval: the copy is worth its speed either way
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
         "threshold": th if figures else None, "noise_floor": floor, "gate": gate,
         "verdict": "needs_eval" if "needs_eval" in ((gate or {}).get("verdict"), pl.get("base_verdict")) else "pass",  # never better than its base
         "base_verdict": pl.get("base_verdict"),  # a part with no agreement gate used no threshold and no gate text
         "text": {"file": TEXT, "sha256": digest, "positions": figures["n"], "corpus": CORPUS.name, "corpus_sha256": th["corpus_sha256"]} if figures else None,
         "speed": [timed] if timed else [], "mtp_speed": [drafted] if drafted else [],
         "speed_note": None if timed or drafted else "not measured: " + ("switched off" if not speed else "the engine row has no speed measurement")}
  rec["kept"] = not slower(rec)  # a copy that needs the task eval stays for it
  if not rec["kept"]:
    rec["deleted_because"] = "; ".join(e["why"] for e in measured(rec) if not e["passed"])
  if (gate or {}).get("verdict") == "needs_eval" and rec["kept"]:  # its own eval's rows go beside the copy; the command says how
    rec["eval"] = {k: str(own(dst, v)) for k, v in th["eval"]["files"].items()}
    noise = perturbation(row["engine"])
    rec["eval"]["command"] = th["eval"]["command"].format(source=src, copy=dst, **{f"{k}_rows": v for k, v in rec["eval"].items()},
                                                          again=f"with {noise['says']}" + (f" ({noise['rule']})" if noise.get("raised") else "") if noise else th["eval"]["unchanged"])
    rec["eval"]["noise_rule"] = (noise or {}).get("rule")  # the batch for a setup with a draft head: verify derives it from --eval-flags
  if "deleted_because" in rec:
    rec["logs_kept"] = str(kept) if (kept := keep_logs(dst, "not-kept")) else None
    dst.with_name(dst.name + ".speed_up.json").write_text(json.dumps(rec, indent=2))  # the copy goes; its reason (and the per-document figures) stay beside it
  if rec["kept"]:
    beside = functools.partial(own, dst)
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


def verify(copy:str, engine:str | None = None, engine_path:str | None = None, eval_flags=None) -> dict[str, Any]:
  """A copy against its record: it passed and was kept, its files and its gate text still hash to the record, this machine's
  stock engine is the record's version, and the record holds a speed for this GPU. Refused on the first mismatch."""
  p = pathlib.Path(copy).expanduser().resolve()
  beside = functools.partial(own, p)
  if not beside(RECORD).is_file():
    raise Refused(f"{p} has no {RECORD}: it is not a BoltBeam copy, or the record was removed")
  rec = json.loads(beside(RECORD).read_text())
  if rec.get("schema") != SCHEMA or not rec.get("sha256"):
    raise Refused(f"{beside(RECORD).name} is not a {SCHEMA} record with file hashes")
  if verdict(rec) == "fail" or not rec["kept"] or slower(rec):
    raise Refused("the record says this copy failed its gate or its speed check" + ("; delete it" if verdict(rec) == "fail" else " and was kept only as asked"))
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
  if (bv := base_verdict(pathlib.Path(rec["source"]))) in ("fail", "needs_eval"):
    raise Refused(f"the copy it was built on, {rec['source']}, " + ("failed: delete both" if bv == "fail" else "waits on its task eval: verify it first"))
  if bv and rec.get("base_verdict") == "needs_eval" and verdict(rec) == "needs_eval" and (rec.get("gate") or {}).get("verdict", "pass") == "pass":
    rec = {**rec, "verdict": "pass", "base_verdict": bv}  # it waited only on its base, which has passed since
    own(p, RECORD).write_text(json.dumps(rec, indent=2))
  base = rec.get("base") or {}
  for name, want in base.get("sha256", {}).items():
    f = p / name
    if not f.is_file() or _sha(f) != want:
      raise Refused(f"the copy it was built on, {base['copy']}, " + ("is missing or lost " + name if not f.is_file() else f"changed: {name} no longer matches the record"))
  if measured(rec):
    gpu = getattr(w, row["device"])(path)
    if gpu["target"] not in {e["gpu"]["target"] for e in measured(rec)}:
      raise Refused(f"the speed was measured on {', '.join(e['gpu']['name'] for e in measured(rec))}, not on this {gpu['name']}")
  for k, sha in (rec.get("stage2") or {}).get("rows_sha256", {}).items():  # the rows a verdict was judged on still say it
    if not (f := pathlib.Path(rec["eval"][k])).is_file() or _sha(f) != sha:
      raise Refused(f"{f.name}, an eval row file this copy was judged on, is missing or changed since")
  return judge(rec, p, w, path, eval_flags) if verdict(rec) == "needs_eval" else rec


def usable(path:str | pathlib.Path) -> str | None:
  """Whether a model path may be used (tune, recommend, the pipeline consult it): a Speed up copy whose record says it
  failed is refused, with the instruction to delete it; one waiting on its task eval is used with this warning; any other
  path (an original, a copy that passed) None."""
  p = pathlib.Path(path).expanduser().resolve()
  try:
    rec = json.loads(own(p, RECORD).read_text())
  except (OSError, ValueError):
    return None
  if rec.get("schema") != SCHEMA or verdict(rec) == "pass":
    return None
  if verdict(rec) == "fail":
    raise Refused(f"{p} is a Speed up copy that failed its " + ("task eval" if rec.get("stage2") else "accuracy check") + ": delete it")
  return f"{p} is a Speed up copy waiting on its task eval: its answers differ beyond the engine's own noise, and no eval has judged them yet"


def base_verdict(src:pathlib.Path) -> str | None:  # the verdict of the copy a copy is built on, None for an original
  try:
    return verdict(json.loads(own(src, RECORD).read_text()))
  except (OSError, ValueError):
    return None


def verdict(rec:dict[str, Any]) -> str:  # pass, needs_eval or fail; a record from before the derived gate said passed
  return rec.get("verdict") or ("pass" if rec.get("passed") else "fail")


def judge(rec:dict[str, Any], p:pathlib.Path, w, path:str, flags=None) -> dict[str, Any]:
  """Stage 2 for a copy waiting on its task eval: the eval's rows beside it (original, copy, and noise: the original run
  again as perturbation() says), each row naming the checkpoint it ran (model), judged by noninferior(). The engine's own
  eval noise is cached as the floor is (noise_key, plus the original's rows and the eval's settings, every row's config,
  one value for all), and refused when it is the original's rows again. The record takes the verdict, the figures, the
  rows' sha256 and the models; a fail says to delete the copy, an inconclusive eval writes nothing."""
  ev = rec["eval"]
  want = {"original": rec["source"], "copy": rec["copy"], "noise": rec["source"]}
  def rows(k:str) -> list[dict]:
    f = pathlib.Path(ev[k])
    if not f.is_file():
      raise Refused(f"this copy waits on its task eval: {ev['command']}")
    got = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if bad := {str(r.get("model")) for r in got if not isinstance(r, dict) or pathlib.Path(str(r.get("model"))).expanduser().resolve() != pathlib.Path(want[k]).resolve()}:
      raise Refused(f"{f.name} holds rows of {', '.join(sorted(bad))[:200]}, not of {want[k]}")
    return got
  def fresh() -> dict[str, Any]:
    if (got := rows("noise")) == orig:
      raise Refused(f"{pathlib.Path(ev['noise']).name} holds the original's rows again, not a second run of it")
    if {json.dumps(r.get("config"), sort_keys=True) for r in got} != {cfg}:
      raise Refused("the noise run's rows carry another config than the original's: it ran under other eval settings")
    return {"rows": got}
  paired(orig := rows("original"), copy := rows("copy"))  # whole rows, one per item, the same items, before they key anything
  if len(config := {json.dumps(r.get("config"), sort_keys=True) for r in orig + copy}) > 1:
    raise Refused("the eval rows carry more than one config: the arms ran under different eval settings")
  noise = perturbation(rec["engine"], flags)  # flags: the engine arguments the eval ran the original with
  if flags is None and noise and (named := drafted(noise, cfg := next(iter(config)))):
    raise Refused(f"the eval's config names a draft head ({', '.join(named)}) but --verify was not given the eval's engine flags (--eval-flags), "
                  f"so the noise run's batch cannot be derived: {ev.get('noise_rule') or noise['rule']}")
  key = noise_key(pathlib.Path(rec["source"]), w, path, args=noise and noise["args"], config=(cfg := config.pop()),
                  original=hashlib.sha256(json.dumps(orig, sort_keys=True).encode()).hexdigest())
  again = cached(key, fresh, lambda e: isinstance(e.get("rows"), list))
  got = noninferior(orig, copy, again["rows"], how=f"under {noise['says']}" if noise else "run again unchanged")
  if got["verdict"] == "inconclusive":
    raise Refused("the task eval is inconclusive: " + eval_words(got))
  rec = {**rec, "verdict": got["verdict"], "stage2": {**got, "noise_cached": again["cached"], "models": want,
                                                      "rows_sha256": {k: _sha(pathlib.Path(ev[k])) for k in want if pathlib.Path(ev[k]).is_file()}}}
  own(p, RECORD).write_text(json.dumps(rec, indent=2))
  if got["verdict"] == "fail":
    raise Refused(f"the task eval failed: {eval_words(got)}; delete the copy ({rec['copy']})")
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


def compare_mtp(first:dict[str, Any], copy:dict[str, Any], again:dict[str, Any], needed:int, size:dict[str, Any] | None = None) -> dict[str, Any]:
  """{depth: ...} from three runs of the engine's MTP measurement ({depth: {context, base, ks, tokens}}; base: each prompt's
  tok/s without drafts): the source, the copy, the source again. Per prompt, the copy at its best k over the source at its
  best k (the source's two runs averaged); each arm's best k is the highest median tok/s, chosen again inside each bootstrap
  resample of the prompts. Faster if the lower end of the 95% interval of the median is above 1 plus the source's own change
  between its two runs (at its best k). Recorded: the factor with MTP over without (each arm over its own base: each arm
  compiles its own graph) and the factor without drafts (copy base over source base), each a median over the prompts; per
  prompt their product is that prompt's ratio, whose median is the headline (change_pct). needed: the short depth's
  prompts (the set's derived count, collectors/mtp.prompt_set); a long depth with fewer documents is judged and said
  underpowered (its corpus is not sized yet). The count was derived for steps between adjacent ks, not for copy against
  source: the record says so (data/mtp_prompts.json size.population)."""
  from boltbeam.collectors import mtp
  if not set(first) == set(copy) == set(again):
    raise Refused(f"the runs of the MTP measurement ran different depths: {sorted(first)}, {sorted(copy)}, {sorted(again)}")
  out:dict[str, Any] = {}
  for depth, f in first.items():
    runs, prompts = [{"base": x["base"], **{k: v["per_prompt"] for k, v in x["ks"].items()}} for x in (f, copy[depth], again[depth])], sorted(f["base"])
    bare = set(runs[0]) == set(runs[2]) == {"base"}  # the source's head did not fit: its baseline is the model without drafts
    if any(not (set(r) <= set(runs[2]) or bare) or any(sorted(v) != prompts for v in r.values()) for r in runs):  # the source again ran every k
      raise Refused(f"the runs of the MTP measurement ran different prompts or draft counts at the {depth} depth")
    if depth == "short" and len(prompts) < needed:
      raise Refused(f"the MTP measurement ran {len(prompts)} prompts at the {depth} depth; the gate needs {needed}")
    reads = [x.get("read") for x in (f, copy[depth], again[depth])]
    if not all(reads) or any(None in x.values() for x in reads) or not reads[0] == reads[1] == reads[2]:
      raise Refused(f"the runs of the MTP measurement read different text at the {depth} depth, or did not say what they read (another tokenizer or chat template)")
    s1, c, s2 = runs
    src = {k: {p: (s1[k][p] + s2[k][p]) / 2 if k in s1 else s2[k][p] for p in prompts} for k in s2}
    med = lambda t, k, ps: statistics.median(t[k][p] for p in ps)  # noqa: E731
    best = lambda t, ps: max((k for k in t if k != "base"), key=lambda k: med(t, k, ps))  # noqa: E731
    best_src = lambda ps: "base" if bare else best(src, ps)  # noqa: E731
    lo, hi = bootstrap(prompts, lambda ps: statistics.median(c[best(c, ps)][p] / src[best_src(ps)][p] for p in ps))
    kc, ks = best(c, prompts), best_src(prompts)
    shift = lambda k: statistics.median(s2[k][p] / s1[k][p] for p in prompts) - 1  # noqa: E731
    drift = shift(ks) if ks in s1 else max((shift(k) for k in s1 if k != "base"), key=abs)  # k past the first sweep: the largest shared change
    ratio = {p: c[kc][p] / src[ks][p] for p in prompts}
    has = lambda x: f if x == "base" or x in s1 else again[depth]  # noqa: E731  the source run that ran that k
    same = {x: sum(has(x).get("tokens", {}).get(x, {}).get(p) == copy[depth].get("tokens", {}).get(x, {}).get(p) is not None for p in prompts) for x in (kc, "base")}
    out[depth] = {"context": f["context"], "copy": {"k": int(kc), "tok_s": med(c, kc, prompts), "base_tok_s": med(c, "base", prompts), "accept_length": copy[depth]["ks"][kc].get("accept_length")},
                  "source": {"k": 0 if bare else int(ks), "tok_s": med(src, ks, prompts), "base_tok_s": med(src, "base", prompts),
                             "accept_length": None if bare else (f["ks"].get(ks) or again[depth]["ks"][ks]).get("accept_length")},
                  "ratio": ratio, "change_pct": (statistics.median(ratio.values()) - 1) * 100, "interval_pct": [(lo - 1) * 100, (hi - 1) * 100], "source_change_pct": drift * 100,
                  "source_change_at": 0 if bare else int(ks) if ks in s1 else "largest shared",
                  "single_run": None if ks in s1 else f"the source's k={ks} ran once, in its second run: the median may be off by about half the source's own change ({abs(drift) * 50:.2f}%)",
                  "mtp_factor_pct": (statistics.median(c[kc][p] / c["base"][p] / (src[ks][p] / src["base"][p]) for p in prompts) - 1) * 100,
                  "base_factor_pct": (statistics.median(c["base"][p] / src["base"][p] for p in prompts) - 1) * 100,
                  "same_tokens": {"at_k": same[kc], "without_drafts": same["base"], "of": len(prompts), "gated": False}, "faster": lo > 1 + abs(drift),
                  "underpowered": mtp.underpowered(len(prompts), size) if len(prompts) < needed else None}
  return out


def measure_mtp(src:pathlib.Path, dst:pathlib.Path, row:dict[str, Any], w, path:str, layer:dict[str, Any]) -> dict[str, Any]:
  """One GPU's entry with MTP on: the source and the copy each on the derived draft-count sweep, then the source again at every
  k either reached (all on the first run's long-context text), logs in the copy's MTP_LOGS, compared per depth; kept only if
  faster at every depth. sweeps: each run's ks and why its sweep stopped. long_text: the long depth's corpus (collectors/mtp.long_corpus):
  an entry without it is from before the long depth was real text (mtp_depths_valid)."""
  from boltbeam.collectors import mtp as core
  folder, how, run = own(dst, MTP_LOGS), {**layer["measure"], **({"main": layer["main"]} if layer.get("main") else {})}, getattr(w, row["mtp"])
  runs = {"source": run(str(src), path, folder / "source", {**how, "grow": True})}
  runs["copy"] = run(str(dst), path, folder / "copy", {**how, "grow": True})
  ks = {n: sorted({int(k) for d in r["depths"].values() for k in d["ks"]}) for n, r in runs.items()}
  runs["source_again"] = run(str(src), path, folder / "source_again", {**how, "ks": sorted(set(ks["source"]) | set(ks["copy"]))})
  if runs["copy"].get("did_not_fit"):
    raise Refused(f"the copy's MTP draft head did not fit on this GPU beside the model either ({runs['copy']['did_not_fit']})")
  if len({bool(runs[n].get("did_not_fit")) for n in ("source", "source_again")}) != 1:
    raise Refused("the source's MTP draft head fit in one of its two runs and not in the other, so the runs cannot be compared")
  if len({r["engine"] for r in runs.values()}) != 1:
    raise Refused(f"the runs with MTP on ran on different engine builds: {', '.join(str(r['engine']) for r in runs.values())}")
  ps = core.prompt_set()
  by = compare_mtp(*(r["depths"] for r in runs.values()), ps["needed"], ps["size"])
  slow = [f"{d}: median {v['change_pct']:+.1f}%, 95% interval {v['interval_pct'][0]:+.1f} to {v['interval_pct'][1]:+.1f}%, the source's own change {v['source_change_pct']:+.2f}%"
          for d, v in by.items() if not v["faster"]]
  info = getattr(w, row["device"])(path)
  sweeps = {n: {"ks": sorted({int(k) for d in r["depths"].values() for k in d["ks"]}), "stopped": r.get("sweep")} for n, r in runs.items()}
  from boltbeam.report import plain
  bare = {"baseline": plain.words()["speed_up"]["kept"]["no_fit"], "did_not_fit": runs["source"]["did_not_fit"]} if runs["source"].get("did_not_fit") else {}
  return {"gpu": {k: info[k] for k in GPU_KEYS}, **how, **bare, "schema": core.SCHEMA,
          "prompts": {"version": ps["version"], "sha256": ps["sha256"], "needed": ps["needed"], "population": (ps["size"] or {}).get("population")}, "long_text": {k: core.long_corpus()[k] for k in ("version", "sha256")}, "sweeps": sweeps, "order": ", ".join(MTP_RUNS), "logs": folder.name, "depths": by, "runs": {k: r["depths"] for k, r in runs.items()},
          "engine": runs["copy"]["engine"], "passed": not slow,
          "why": (lambda was: f"with MTP on, faster than {was} at every depth: the 95% interval of the median per-prompt ratio clears {was}'s own change between its two runs" if not slow
                  else f"with MTP on, not shown faster than {was} (the 95% interval of the median per-prompt ratio does not clear {was}'s own change) at " + "; ".join(slow))(
                  "running the source without MTP (its head did not fit)" if bare else "the source"),
          "size_said": [f"{n}: {said}" for n, r in runs.items() if (said := (r.get("size_check") or {}).get("said"))]}  # the count's assumption, checked per run


def words(rec:dict[str, Any]) -> list[str]:
  a, v, slow, kept = rec["accuracy"], verdict(rec), slower(rec), " (kept, as asked)" if rec["kept"] else " and was deleted"
  head = (f"{rec['name']}: is not faster than the original{kept}" if slow else
          f"{rec['name']}: failed " + ("the task eval: delete it" if rec.get("stage2") else f"the accuracy check{kept}") if v == "fail" else
          f"{rec['name']}: kept at {rec['copy']}" + ("" if measured(rec) else " (SPEED NOT MEASURED)") + (", waiting on a task eval" if v == "needs_eval" else "") +
          (f" (the copy it is built on, {rec['source']}, waits on its own)" if rec.get("base_verdict") == "needs_eval" else "") +
          (", its task eval passed" if v == "pass" and rec.get("stage2") else ""))
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
    *([e["baseline"]] if e.get("baseline") else []), *(f"{d}: {v['underpowered']}" for d, v in depths.items() if v.get("underpowered")),
    *e.get("size_said", []), e["why"])]
  ch = rec["changed"]
  return [head, *speed, f"changed: {ch['layer']} {ch['from']} -> {ch['to']}, everything else {ch['all_else']}", rec.get("accuracy_note") if not a else
          f"gives the original's top answer {a['top1_agreement'] * 100:.1f} times in 100 (pooled; documents clustered interval {a['interval'][0] * 100:.1f} to {a['interval'][1] * 100:.1f} over {a['n']} words; worst document {a.get('doc_min', 0) * 100:.1f}, median {a.get('doc_median', 0) * 100:.1f}; " + (f"needs {n} pooled" if (n := need(rec)) else "no floor to judge it by") + "); perplexity "
          f"{a.get('ppl_original') or 0:.3f} -> {a.get('ppl_copy') or 0:.3f}", *floor_words(rec), f"engine: {rec['engine_version']}"]


def need(rec:dict[str, Any]) -> str | None:
  """The pooled agreement stage 1 needs, per 100: its bar, or an older record's own (bar, else the threshold's min); None: no floor."""
  x = rec["gate"].get("needs") if rec.get("gate") else rec.get("bar") or rec["threshold"]["min"]
  return None if x is None else f"{x * 100:.1f}".removesuffix(".0")


def floor_words(rec:dict[str, Any], who:str = "the copy") -> list[str]:
  """bar()'s judgment in words (Speed up's record, and tune's knob as who); a record from before it says nothing more."""
  g, f, a = rec.get("gate"), rec.get("noise_floor"), rec["accuracy"]
  if not (g and a):
    return []
  if not f:
    return [f"{rec['engine']} declares no change that must not matter, so it has no noise floor: every copy that changes any answer waits on a task eval"]
  wd = g["worst"]
  said = (f"the original against itself with {f['says']} gives its own top answer {(1 - g['d_floor']) * 100:.1f} times in 100 (the engine's noise on this "
          f"model{', cached' if f.get('cached') else ''}); {who} diverges {g['excess'] * 100:+.2f} in 100 beyond that noise, {g['allowed'] * 100:.2f} allowed "
          f"({g['confidence']:.0%} one-sided, t {g['z']:.3f} at {g['df']} df x SE {g['se'] * 100:.2f}, {g['se_method']}); the document furthest over "
          f"its own floor {wd['excess'] * 100:+.1f}, {wd['allowed'] * 100:.1f} allowed (z {wd['z']:.2f}, Sidak over the documents)")
  s2 = rec.get("stage2")
  return [said + ("" if g["verdict"] == "pass" else f": beyond the engine's own noise; its task eval: {s2['verdict']}, {eval_words(s2)}" if s2 else
                  f": beyond the engine's own noise, so {who} waits on a task eval" + (f": {rec['eval']['command']}" if rec.get("eval") else ""))]


def eval_words(got:dict[str, Any]) -> str:
  return (f"the copy's net loss against the engine's own noise run {got['loss'] * 100:+.2f} points over {got['n']} paired items, {got['margin'] * 100:+.2f} past the allowance, SE {got['se'] * 100:.2f}, "
          f"{got['confidence']:.0%} one-sided (against the original: {got['copy']['b']} only it right, {got['copy']['c']} only the copy); allowed: "
          f"{got['allowed_said']}" + (f"; {got['why']}" if got["why"] else ""))
