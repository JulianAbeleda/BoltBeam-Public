"""llama.cpp, GGUF: stock `llama-quantize --output-tensor-type`, and nothing else requantized. `COPY` would copy every
tensor but also ignore --output-tensor-type, and a plain ftype re-picks every tensor's type by its own rules, so every
other tensor is pinned to the type it already has with --tensor-type-file: llama-quantize copies a tensor whose target
is its own type byte for byte. `identical_but_head` then proves it on the files. The gate is llama-perplexity's KL
divergence run: its "Same top p" is the share of positions with the same most likely token."""
from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile

from boltbeam.core.refused import Refused
from boltbeam.profile import gguf
from boltbeam.profile.decode_roles import GGML_BITS_PER_WEIGHT, GGML_BLOCK_LAYOUTS

HEAD = "output.weight"
STOCK_OWNERS = ("ggml-org", "ggerganov")  # ggerganov/llama.cpp is the same repo under its former owner name
CTX = 512


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
  import os, shutil
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


def stock(bench:str, env:dict | None = None) -> dict:
  """Stock = upstream llama.cpp: not a fork's build, its source checkout found, ggml-org's, tracked files unchanged."""
  from boltbeam.collectors import llama_bench_decode as lb
  if (fork := lb.fork_of(bench, env)):
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


def plan(src:str, bench:str, target:str) -> dict:
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
          "bits_target": after, "bytes_now": int(n * now / 8), "bytes_after": int(n * after / 8)}


def pins(src:str) -> str:
  """--tensor-type-file text: every 2-D weight but the output layer pinned to the type it has."""
  _, tensors, _ = gguf.read_gguf_layout(src)
  return "".join(f"^{re.escape(n)}$={gguf.GGML_TYPE_NAMES[k].lower()}\n" for n, d, k, _ in tensors
                 if len(d) >= 2 and n.endswith(".weight") and n != HEAD)


def write(src:str, dst:str, bench:str, pl:dict) -> dict:
  q = tool(bench, "llama-quantize")
  with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(pins(src))
  try:
    p = subprocess.run([q, "--allow-requantize", "--output-tensor-type", pl["target"].lower(), "--tensor-type-file", f.name, src, dst, "Q4_K_M"],
                       capture_output=True, text=True, timeout=7200)
  finally:
    pathlib.Path(f.name).unlink(missing_ok=True)
  if p.returncode:
    pathlib.Path(dst).unlink(missing_ok=True)
    raise Refused(f"llama-quantize failed: {(p.stderr or p.stdout).strip()[-300:]}")
  bad = identical_but_head(src, dst)
  if bad:
    pathlib.Path(dst).unlink(missing_ok=True)
    raise Refused(f"llama-quantize changed {len(bad)} other tensor(s), first {bad[0]}")
  return {"layer": HEAD, "from": pl["held"], "to": pl["target"], "all_else": "every other tensor byte-identical (checked)"}


def _spans(path:str) -> dict[str, tuple[tuple[int, ...], int, bytes]]:
  """name -> (dims, type, data span as written: the tensor and its alignment padding)."""
  kv, tensors, start = gguf.read_gguf_layout(path)
  size = pathlib.Path(path).stat().st_size
  by_off = sorted(tensors, key=lambda t: t[3])
  ends = {t[0]: (by_off[i + 1][3] if i + 1 < len(by_off) else size - start) for i, t in enumerate(by_off)}
  return {n: (d, k, (start + off, start + ends[n])) for n, d, k, off in tensors}


def identical_but_head(a:str, b:str, chunk:int = 1 << 26) -> list[str]:
  """Names of tensors that differ between two GGUFs other than the output layer: missing, other shape or type, or other
  bytes. Empty means the copy changed the output layer and nothing else."""
  sa, sb = _spans(a), _spans(b)
  bad = [n for n in sa.keys() ^ sb.keys()]
  with open(a, "rb") as fa, open(b, "rb") as fb:
    for n in sa.keys() & sb.keys():
      if n == HEAD:
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


def _kld(output:str, ctx:int = CTX) -> dict:
  """The last row of llama-perplexity's --kl-divergence table (running figures over every chunk): chunk, PPL(Q),
  ln(PPL(Q)/PPL(base)), KLD, delta-p RMS %, same top p %, each as `value ± uncertainty`. n, the scored positions, is
  read back from the share's own uncertainty, p(1-p)/(n-1); a chunk scores its second half."""
  import math
  rows = [re.findall(r"[-+]?\d+\.?\d*", x.replace("±", " ")) for x in output.splitlines() if re.match(r"\s*\d+\s+\d", x)]
  if not rows or len(rows[-1]) < 11:
    raise Refused("llama-perplexity printed no KL divergence table, so the copy cannot be checked")
  v = [float(x) for x in rows[-1]]
  p, unc = v[9] / 100, v[10] / 100
  n = round(p * (1 - p) / unc ** 2) + 1 if unc > 0 else int(v[0]) * (ctx - ctx // 2 - 1)
  return {"top1_agreement": p, "mean_kld": v[5], "ppl_copy": v[1], "ppl_original": v[1] / math.exp(v[3]), "chunks": int(v[0]), "n": n}


def accuracy(src:str, dst:str, bench:str, texts:list[str]) -> dict:
  """One llama-perplexity pair per document, so each document has its own agreement."""
  ppl = tool(bench, "llama-perplexity")
  per_doc, last = [], {}
  with tempfile.TemporaryDirectory(prefix="speed-up-") as d:
    for i, doc in enumerate(texts):
      text = pathlib.Path(d, f"doc{i}.txt")
      text.write_text(doc)
      base = str(pathlib.Path(d, f"base{i}.kld"))
      common = ["-f", str(text), "-c", str(CTX), "-ngl", "99", "--kl-divergence-base", base]
      out = ""
      for cmd in ([ppl, "-m", src, *common], [ppl, "-m", dst, *common, "--kl-divergence"]):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if p.returncode:
          raise Refused(f"llama-perplexity failed: {(p.stderr or p.stdout).strip()[-300:]}")
        out = p.stdout + p.stderr
      last = _kld(out)
      per_doc.append({"n": last["n"], "hits": round(last["top1_agreement"] * last["n"]), "ppl_copy": last["ppl_copy"], "ppl_original": last["ppl_original"]})
  n = sum(x["n"] for x in per_doc)
  return {"method": "llama-perplexity --kl-divergence (Same top p), one run per document of the shipped natural-text corpus", "n": n, "per_doc": per_doc,
          "top1_agreement": sum(x["hits"] for x in per_doc) / n, "ppl_copy": last["ppl_copy"], "ppl_original": last["ppl_original"]}
