"""llama.cpp, GGUF: stock `llama-quantize --tensor-type-file`, the part's tensors at their planned types and nothing else
requantized. A plain ftype re-picks every tensor's type by its own rules, so every other tensor is pinned to the type it
already has: llama-quantize copies a tensor whose target is its own type byte for byte. The file type passed is the
source's own, or QUANTIZED_BASE for an unquantized source (llama-quantize applies --tensor-type-file only over a quantized
file type; every 2-D weight is pinned, so it lands only in general.file_type). `identical_but` then proves it on the files. The parts: the output layer (output.weight,
the row's target type), and the MTP draft head, a sidecar GGUF of its own (llama-server -md, --spec-type draft-mtp) at the
part row's per-tensor recipe. The output layer's gate is llama-perplexity's KL divergence run: its "Same top p" is the
share of positions with the same most likely token. The draft head's is speed with MTP on, measured by the module
data/engine_adapters.json names for llama.cpp, through collectors/mtp.measure, refused while it names none."""
from __future__ import annotations

import functools
import math
import os
import pathlib
import re
import subprocess
import tempfile

from boltbeam.collectors import capture_settings as cs
from boltbeam.collectors import engine_common as ec
from boltbeam.core.refused import Refused
from boltbeam.profile import gguf
from boltbeam.profile.decode_roles import GGML_BITS_PER_WEIGHT, GGML_BLOCK_LAYOUTS
from boltbeam.quantization import head_writers
from boltbeam.quantization.head_writers.vllm_ct import device, mtp_depths, mtp_ks  # noqa: F401  the row's device and mtp_ks; a measurement per depth

HEAD = "output.weight"
STOCK_OWNERS = ("ggml-org", "ggerganov")  # ggerganov/llama.cpp is the same repo under its former owner name
CTX = 512
TOOL_TIMEOUT_S = 7200  # one llama-quantize or llama-perplexity run: a guard on a hung tool, not a measured length (a big model's whole pass takes minutes)
ERROR_TAIL = 300  # the end of a tool's error text said in a refusal: its last lines say why
QUANTIZED_BASE = "Q4_K_M"  # the file type an unquantized (F32/F16/BF16) source is written under; see the module docstring


def find(explicit:str | None = None) -> str | None:
  from boltbeam.collectors import llama_bench_decode as lb
  return lb.find(explicit) if explicit else lb.find()


def _git(checkout:pathlib.Path, *args:str) -> str:
  p = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, timeout=60)
  return p.stdout.strip() if p.returncode == 0 else ""


def checkout_of(path:str) -> pathlib.Path | None:
  """The source checkout a build came from: its CMakeCache.txt's home folder (a build folder may be a link elsewhere),
  else the nearest parent that is a git checkout of ggml."""
  here = pathlib.Path(path).expanduser().resolve()
  for d in here.parents:
    cache = d / "CMakeCache.txt"
    if cache.is_file():
      m = re.search(r"^CMAKE_HOME_DIRECTORY:\w+=(.+)$", cache.read_text(errors="replace"), re.M)
      home = pathlib.Path(m.group(1)).resolve() if m else None
      return home if home and (home / ".git").exists() and (home / "ggml").is_dir() else None
  return next((d for d in here.parents if (d / ".git").exists() and (d / "ggml").is_dir()), None)


def tool(bench:str, name:str, env:dict | None = None) -> str:
  """llama-quantize / llama-perplexity: beside the llama-bench, else in a build folder of its checkout."""
  import shutil
  env = os.environ if env is None else env
  if env.get("BOLTBEAM_" + name.upper().replace("-", "_")):
    return env["BOLTBEAM_" + name.upper().replace("-", "_")]
  co = checkout_of(bench)
  for d in [pathlib.Path(bench).resolve().parent, *(sorted(co.glob("build*/bin")) if co else [])]:
    if (d / name).is_file():
      return str(d / name)
  if shutil.which(name):
    return shutil.which(name)
  raise Refused(f"{name} was not found beside {bench} or in its checkout's build folders")


def build_id(bench:str) -> list:
  """The engine build a noise measurement is keyed on (quantization/speed_up.noise_key): the llama-perplexity it runs."""
  f = pathlib.Path(tool(bench, "llama-perplexity"))
  return [str(f), (st := f.stat()).st_size, st.st_mtime_ns]


def stock(bench:str, env:dict | None = None) -> dict:
  """Stock = upstream llama.cpp: not another project's fork (a fork row of upstream's own repo, such as master, is upstream
  code), its source checkout found, ggml-org's, tracked files unchanged."""
  from boltbeam.collectors import engine_families, llama_bench_decode as lb
  if (fork := lb.fork_of(bench, env)) and not engine_families.upstream_code(fork["engine_family"], fork["fork"]):
    raise Refused(f"this llama.cpp is the {fork['label']} fork, not upstream ggml-org/llama.cpp")
  co = checkout_of(bench)
  if co is None:
    raise Refused(f"the source checkout {bench} was built from cannot be found, so it cannot be shown to be stock llama.cpp")
  url = _git(co, "remote", "get-url", "origin")
  m = re.search(r"github\.com[:/]([^/]+)/llama\.cpp(?:\.git)?/?$", url)
  if not m or m.group(1) not in STOCK_OWNERS:
    raise Refused(f"its source checkout {co.name} comes from {url or 'no remote'}, not ggml-org/llama.cpp")
  if (dirty := _git(co, "status", "--porcelain", "--untracked-files=no")):
    raise Refused(f"its source checkout {co.name} has local changes ({dirty.splitlines()[0].strip()})")
  for t in ("llama-quantize", "llama-perplexity", "llama-server"):
    if checkout_of(tool(bench, t, env)) != co:
      raise Refused(f"{t} does not come from the same checkout as {pathlib.Path(bench).name}")
  return {"engine_version": "llama.cpp " + _git(co, "rev-parse", "--short", "HEAD"), "source": url}


def plan(src:str, bench:str, target:str, part:dict | None = None) -> dict:
  kv, tensors, _ = gguf.read_gguf_layout(src)
  if int(kv.get("split.count") or 1) > 1:
    raise Refused("a split GGUF (several files) is not handled")
  head = next((t for t in tensors if t[0] == HEAD), None)
  if head is None:
    raise Refused("the model has no output.weight (the output layer is tied to the token embedding), so there is nothing to lighten")
  name, dims, kind, _ = head
  want = target.upper()
  if want not in gguf.GGML_TYPE_IDS or gguf.GGML_TYPE_IDS[want] not in GGML_BLOCK_LAYOUTS:
    raise Refused(f"{want} is not a GGML type BoltBeam knows")
  tid = gguf.GGML_TYPE_IDS[want]
  if dims[0] % GGML_BLOCK_LAYOUTS[tid][0]:
    raise Refused(f"the output layer's rows ({dims[0]}) do not divide into {want} blocks")
  now, after = GGML_BITS_PER_WEIGHT[kind], GGML_BITS_PER_WEIGHT[tid]
  n = dims[0] * dims[1]
  return {"output_layer": HEAD, "shape": list(dims), "held": gguf.GGML_TYPE_NAMES[kind], "bits_now": now, "target": want,
          "bits_target": after, "bytes_now": int(n * now / 8), "bytes_after": int(n * after / 8), "types": {HEAD: want}, "ftype": base_type(kv, bench) if bench else None}


@functools.lru_cache(maxsize=4)
def file_types(quantize:str) -> dict[int, str]:
  """{file type id: name} llama-quantize lists as allowed (its --help)."""
  p = subprocess.run([quantize, "--help"], capture_output=True, text=True, timeout=60)
  got = {int(i): n for i, n in re.findall(r"^\s*(\d+)\s+or\s+(\S+)\s*:", p.stdout + p.stderr, re.M)}
  if not got:
    raise Refused(f"{quantize} --help listed no file types, so the source's file type cannot be checked")
  return got


def base_type(kv:dict, bench:str) -> str:
  """The file type to pass llama-quantize: the source's own (general.file_type), QUANTIZED_BASE if that is unquantized, or
  Refused when llama-quantize does not list it."""
  types, ft = file_types(tool(bench, "llama-quantize")), kv.get("general.file_type")
  if ft is None or int(ft) not in types:
    raise Refused(f"llama-quantize has no file type {ft} (the model's general.file_type), so it cannot be told to keep the model as it is")
  name = types[int(ft)]
  if name in gguf.GGML_TYPE_IDS and GGML_BLOCK_LAYOUTS.get(gguf.GGML_TYPE_IDS[name], (0,))[0] == 1:
    if QUANTIZED_BASE not in types.values():
      raise Refused(f"llama-quantize has no file type {QUANTIZED_BASE} to write the unquantized {name} model under")
    return QUANTIZED_BASE
  return name


def _row() -> dict:
  from boltbeam.quantization import speed_up  # the table names this writer's functions; the core imports writers lazily
  return next(r for r in speed_up.table()["engines"] if r.get("writer") == pathlib.Path(__file__).stem)


def _measurer():
  """collectors/mtp.measure on this engine, or Refused while data/engine_adapters.json names no MTP measurement for it."""
  from boltbeam.collectors import mtp
  if mtp.engine(engine := _row()["engine"]) is None:
    raise Refused(f"no MTP measurement for {_row()['label']} yet, so a lighter MTP draft head cannot be gated")
  return functools.partial(mtp.measure, engine)


def _need_main(main:str | None) -> str:
  if not main:
    raise Refused("an MTP draft head is measured with the main model it drafts for: pass --main MAIN.gguf")
  return main


def _main_kv(main:str) -> tuple[dict, list[str]]:
  """The main model's metadata and tensor names (every file of a split one), or Refused in plain words."""
  p = pathlib.Path(main)
  if not p.is_file():
    raise Refused(f"the main model {main} is not a file")
  try:
    if int(gguf.read_gguf_layout(p)[0].get("split.no") or 0):
      raise Refused(f"the main model {p.name} is a later file of a split GGUF: pass its first file")
    kv, tensors, _ = gguf.read_gguf_split(p)
  except ValueError as exc:
    raise Refused(f"the main model {p.name} cannot be read as a GGUF: {exc}")
  return kv, [t[0] for t in tensors]


def _quantized(src:str) -> tuple[dict, dict]:
  """A GGUF's metadata and {2-D weight: (dims, type name)}; a split GGUF is refused."""
  kv, tensors, _ = gguf.read_gguf_layout(src)
  if int(kv.get("split.count") or 1) > 1:
    raise Refused("a split GGUF (several files) is not handled")
  return kv, {n: (d, gguf.GGML_TYPE_NAMES[k]) for n, d, k, _ in tensors if len(d) >= 2 and n.endswith(".weight")}


def _main_blocks(kv:dict, names) -> int:
  """How many of a GGUF's blk.N blocks are the main model's: N below block_count less nextn_predict_layers."""
  arch = kv.get("general.architecture")
  last = int(kv.get(f"{arch}.block_count") or 0) - int(kv.get(f"{arch}.nextn_predict_layers") or 0)
  return len({int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.", n)) and int(m.group(1)) < last})


def plan_mtp(src:str, bench:str, target:str | None = None, part:dict | None = None) -> dict:
  """The MTP draft head, a sidecar GGUF (its blk tensors only the last nextn_predict_layers blocks), at the part row's recipe:
  each 2-D weight the recipe's first matching row names goes to that row's type, every other tensor keeps its own. Refused: no
  draft layer, the main model's layers in the file, a recipe row that names no tensor, a type that does not divide a tensor's
  rows into its blocks (llama-quantize would pick another type) or is wider than the tensor now, no MTP measurement for this
  engine yet, no main model to measure it with, or one that does not pair with the head."""
  recipe = part["recipe"]
  kv, have = _quantized(src)
  arch = kv.get("general.architecture")
  if not kv.get(f"{arch}.nextn_predict_layers"):
    raise Refused(f"the GGUF has no MTP draft layer (no {arch}.nextn_predict_layers)")
  if (main := _main_blocks(kv, have)):
    raise Refused(f"the GGUF holds the main model's layers too ({main} blocks): the MTP draft head is made lighter as its own sidecar GGUF (llama-server -md)")
  rows = [(re.compile(r["match"]), r["type"].upper()) for r in recipe["tensors"]]
  types, now, after, n_all = {}, 0.0, 0.0, 0
  for name, (dims, held) in have.items():
    want = next((t for rx, t in rows if rx.search(name)), None)
    if want is None:
      continue
    if want not in gguf.GGML_TYPE_IDS or gguf.GGML_TYPE_IDS[want] not in GGML_BLOCK_LAYOUTS:
      raise Refused(f"the recipe's {want} is not a GGML type BoltBeam knows")
    block, bits, was = GGML_BLOCK_LAYOUTS[gguf.GGML_TYPE_IDS[want]][0], GGML_BITS_PER_WEIGHT[gguf.GGML_TYPE_IDS[want]], GGML_BITS_PER_WEIGHT[gguf.GGML_TYPE_IDS[held]]
    if dims[0] % block:
      raise Refused(f"the recipe puts {name} at {want}, but its rows ({dims[0]}) do not divide into {want} blocks of {block}: llama-quantize would pick another type")
    if bits > was:
      raise Refused(f"the recipe puts {name} at {want}, wider than its {held}")
    n = math.prod(dims)
    types[name], now, after, n_all = want, now + n * was, after + n * bits, n_all + n
  idle = [r["match"] for r in recipe["tensors"] if not any(re.search(r["match"], n) for n in have)]
  if idle:
    raise Refused(f"the recipe's {idle[0]} names no tensor of this draft head")
  _measurer()
  ftype = main = None
  if bench is not None:  # the run's own inputs: the engine's file types and the main model (the engine-free check has neither)
    ftype, (mkv, names) = base_type(kv, bench), _main_kv(main := _need_main(part.get("main")))
    same = lambda k: mkv.get(k) == kv.get(k)  # noqa: E731
    if not (same("general.architecture") and same(f"{arch}.embedding_length") and same("tokenizer.ggml.tokens")) or not _main_blocks(mkv, names):
      raise Refused(f"the main model {pathlib.Path(main).name} does not pair with this draft head (architecture, width or vocabulary differ, or it is a draft head itself)")
  held = sorted({have[n][1] for n in types})
  return {"layer": ", ".join(types), "held": "/".join(held), "bits_now": now / n_all, "target": f"recipe {recipe['recipe']}", "bits_target": after / n_all,
          "bytes_now": int(now / 8), "bytes_after": int(after / 8), "types": types, "main": main, "ftype": ftype}


def pins(src:str, types:dict[str, str] | None = None) -> str:
  """--tensor-type-file text: every 2-D weight at the type `types` gives it, else pinned to the type it has."""
  _, tensors, _ = gguf.read_gguf_layout(src)
  return "".join(f"^{re.escape(n)}$={(types or {}).get(n, gguf.GGML_TYPE_NAMES[k]).lower()}\n" for n, d, k, _ in tensors
                 if len(d) >= 2 and n.endswith(".weight"))


def write(src:str, dst:str, bench:str, pl:dict) -> dict:
  """The part's tensors (pl["types"]) at their types, every other one pinned, the source's own file type; then checked on
  the files: each part tensor at its planned type, every other tensor byte-identical."""
  q, types = tool(bench, "llama-quantize"), pl["types"]
  with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(pins(src, types))
  try:
    p = subprocess.run([q, "--allow-requantize", "--tensor-type-file", f.name, src, dst, pl["ftype"]],
                       capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
  finally:
    pathlib.Path(f.name).unlink(missing_ok=True)
  if p.returncode:
    pathlib.Path(dst).unlink(missing_ok=True)
    raise Refused(f"llama-quantize failed: {(p.stderr or p.stdout).strip()[-ERROR_TAIL:]}")
  got = {n: t for n, (_, t) in _quantized(dst)[1].items() if n in types}
  bad = identical_but(src, dst, set(types))
  wrong = [f"{n} is {got.get(n)}, not {t}" for n, t in types.items() if got.get(n) != t]
  if bad or wrong:
    pathlib.Path(dst).unlink(missing_ok=True)
    raise Refused(f"llama-quantize changed {len(bad)} other tensor(s), first {bad[0]}" if bad else f"llama-quantize did not write the planned types: {wrong[0]}")
  return {"layer": pl.get("output_layer") or pl["layer"], "from": pl["held"], "to": pl["target"], "types": types,
          "all_else": "every other tensor byte-identical (checked)", **({"main": pl["main"]} if pl.get("main") else {})}


def _spans(path:str) -> dict[str, tuple[tuple[int, ...], int, bytes]]:
  """name -> (dims, type, data span as written: the tensor and its alignment padding)."""
  kv, tensors, start = gguf.read_gguf_layout(path)
  size = pathlib.Path(path).stat().st_size
  by_off = sorted(tensors, key=lambda t: t[3])
  ends = {t[0]: (by_off[i + 1][3] if i + 1 < len(by_off) else size - start) for i, t in enumerate(by_off)}
  return {n: (d, k, (start + off, start + ends[n])) for n, d, k, off in tensors}


def identical_but(a:str, b:str, changed:set[str] = frozenset({HEAD}), chunk:int = 1 << 26) -> list[str]:
  """Names of tensors that differ between two GGUFs other than `changed`: missing, other shape or type, or other bytes.
  Empty means the copy changed those tensors and nothing else."""
  sa, sb = _spans(a), _spans(b)
  bad = [n for n in sa.keys() ^ sb.keys()]
  with open(a, "rb") as fa, open(b, "rb") as fb:
    for n in sa.keys() & sb.keys():
      if n in changed:
        continue
      (da, ka, (a0, a1)), (db, kb, (b0, b1)) = sa[n], sb[n]
      if (da, ka, a1 - a0) != (db, kb, b1 - b0):
        bad.append(n)
        continue
      fa.seek(a0), fb.seek(b0)
      left = a1 - a0
      while left:
        n_ = min(chunk, left)
        if fa.read(n_) != fb.read(n_):
          bad.append(n)
          break
        left -= n_
  return sorted(bad)


def scored(ctx:int) -> int:
  """The positions one chunk of ctx tokens scores: its second half, less the first of it."""
  return ctx - ctx // 2 - 1


def _kld(output:str, ctx:int = CTX) -> dict:
  """The last row of llama-perplexity's --kl-divergence table (running figures over every chunk): chunk, PPL(Q),
  ln(PPL(Q)/PPL(base)), KLD, delta-p RMS %, same top p %, each as `value ± uncertainty`. n, the scored positions, is
  exact: the chunks times scored(n_ctx), n_ctx as the run says it read from the base file ("computing over N chunks,
  n_ctx=C"; perplexity.cpp kl_divergence), whatever -b and -ub were, so two checks of one document pair position by
  position. Read back from the printed uncertainty it came out 1018 to 1022 for 1020."""
  import math
  rows = [re.findall(r"[-+]?\d+\.?\d*", x.replace("±", " ")) for x in output.splitlines() if re.match(r"\s*\d+\s+\d", x)]
  if not rows or len(rows[-1]) < 11:
    raise Refused("llama-perplexity printed no KL divergence table, so the copy cannot be checked")
  v, said = [float(x) for x in rows[-1]], re.search(r"computing over \d+ chunks, n_ctx=(\d+)", output)
  p, n = v[9] / 100, int(v[0]) * scored(int(said[1]) if said else ctx)
  return {"top1_agreement": p, "mean_kld": v[5], "ppl_copy": v[1], "ppl_original": v[1] / math.exp(v[3]), "chunks": int(v[0]), "n": n}


def accuracy(src:str, dst:str, bench:str, texts:list[str], *, src_flags:tuple[str, ...] = (), dst_flags:tuple[str, ...] = (),
             env:dict[str, str] | None = None, ctx:int | None = None, chunks:int | None = None) -> dict:
  """One llama-perplexity pair per document, so each document has its own agreement. src_flags, dst_flags: each arm's
  own flags (tune's output-changing knobs: one model, two settings); env: the engine's environment for both; ctx: the
  context each chunk is read at (default CTX), chunks: at most this many per document (each scores its second half)."""
  ctx = ctx or CTX
  ppl = tool(bench, "llama-perplexity")
  per_doc, last = [], {}
  with tempfile.TemporaryDirectory(prefix="speed-up-") as d:
    for i, doc in enumerate(texts):
      text = pathlib.Path(d, f"doc{i}.txt")
      text.write_text(doc)
      base = str(pathlib.Path(d, f"base{i}.kld"))
      common = ["-f", str(text), "-c", str(ctx), *cs.all_layers(None), "--kl-divergence-base", base, *(["--chunks", str(chunks)] if chunks else [])]
      out = ""
      for cmd in ([ppl, "-m", src, *common, *src_flags], [ppl, "-m", dst, *common, *dst_flags, "--kl-divergence"]):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S, env=None if env is None else {**os.environ, **env})
        if p.returncode:
          raise Refused(f"llama-perplexity failed: {(p.stderr or p.stdout).strip()[-ERROR_TAIL:]}")
        out = p.stdout + p.stderr
      last = _kld(out, ctx)
      per_doc.append({"n": last["n"], "hits": round(last["top1_agreement"] * last["n"]), "ppl_copy": last["ppl_copy"], "ppl_original": last["ppl_original"]})
  n = sum(x["n"] for x in per_doc)
  return {"method": "llama-perplexity --kl-divergence (Same top p), one run per document of the shipped natural-text corpus", "n": n, "per_doc": per_doc, "ctx": ctx, "scored": scored(ctx),
          "top1_agreement": sum(x["hits"] for x in per_doc) / n, "ppl_copy": last["ppl_copy"], "ppl_original": last["ppl_original"]}


def mtp(model:str, bench:str, out:pathlib.Path, how:dict) -> dict:
  """Speed with MTP on through collectors/mtp.measure: the main model (how["main"]) on this llama.cpp's llama-server with this
  draft head (head=), at how["ks"] or mtp_ks(); each engine line's llama.cpp build_info (b<N>-<commit>) read as its commit, so
  the core can hold it against the stock check's."""
  fn, main = _measurer(), _need_main(how.get("main"))
  out.mkdir(parents=True, exist_ok=True)
  try:
    rec = fn(main, how.get("ks") or mtp_ks(), backend=device(bench)["backend"], out=out, tokens=how["tokens"],
             long_tokens=how["long_tokens"], grow=bool(how.get("grow")), head=model, server=tool(bench, "llama-server"))
  except RuntimeError as exc:
    if (rec := did_not_fit(out)) is None:
      raise Refused(f"the MTP measurement failed on {model}: {exc}")
  builds = {str(e.get("llama.cpp")).rsplit("-", 1)[-1] for e in rec["engines"]}
  if len(builds) != 1:
    raise Refused(f"the MTP measurement of {model} ran on several llama.cpp builds: {sorted(map(str, builds))}")
  return {"depths": mtp_depths(rec), "engine": builds.pop(), "sweep": (rec.get("sweep") or [None])[-1], **({"did_not_fit": rec["did_not_fit"]} if "did_not_fit" in rec else {})}


def did_not_fit(out:pathlib.Path) -> dict | None:
  """A run whose draft head never ran, as a measurement without drafts (the k=0 run loads no head), else None. All of: k=0
  measured, no row at any k of 1 or more, and the lowest such k's driver stopped before measuring, in its own words (the
  driver alone reads the server's log, runtime/llama_server_decode_drive): out of GPU memory while loading ("before it was
  ready") or at its first request (no row at that k). A head that ran at one k and failed later fit."""
  from boltbeam.collectors import mtp
  from boltbeam.report import plain
  w = plain.words()["llama_server"]
  logs = {int(m.group(1)): f.read_text(errors="replace") for f in out.glob("mtp-k*.log") if (m := re.fullmatch(r"mtp-k(\d+)\.log", f.name))}
  drafted, k0 = sorted(k for k in logs if k), ec.json_lines(logs.get(0, ""))
  rows = [r for r in k0 if "prompt" in r and r.get("k") == 0]
  if not drafted or not rows or any("prompt" in r for k in drafted for r in ec.json_lines(logs[k])):
    return None
  oom = w["out_of_memory"].split("{")[0]
  said = next((ln.strip() for ln in logs[drafted[0]].splitlines() if oom in ln), None)
  if said is None:
    return None
  return {"depths": mtp.summarize(rows), "rows": rows, "engines": [r["engine"] for r in k0 if "engine" in r], "sweep": [],
          "did_not_fit": f"k={drafted[0]}: {said[-400:]}"}
