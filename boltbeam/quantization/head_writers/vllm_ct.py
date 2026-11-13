"""vLLM, compressed-tensors pack-quantized (W4, group): the parts data/speed_up.json names (the output layer lm_head, the
MTP draft layer's Linear modules) as RTN int4, asymmetric, nibble zero point, the group size and bits read from the
checkpoint's own quantization_config. No vLLM patch: the modules out of `ignore`, a target for lm_head (not a Linear),
the 16 bit weights out of the shards. Stock vLLM passes quant_config to the MTP layers. The torch work is vllm_engine.py,
run under the engine's own python."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from boltbeam.core.refused import Refused

ENGINE_SCRIPT = pathlib.Path(__file__).with_name("vllm_engine.py")
INDEX = "model.safetensors.index.json"


def _run(python:str, cmd:str, args:dict, timeout:int = 7200) -> dict:
  p = subprocess.run([python, "-I", str(ENGINE_SCRIPT), cmd, json.dumps(args)], capture_output=True, text=True, timeout=timeout)
  line = next((x for x in reversed(p.stdout.splitlines()) if x.startswith("@@")), None)
  if p.returncode or line is None:
    raise Refused(f"the vLLM python failed in `{cmd}`: {(p.stderr or p.stdout).strip()[-400:]}")
  return json.loads(line[2:])


def find(explicit:str | None = None) -> str | None:
  from boltbeam.collectors import vllm_decode
  return explicit or vllm_decode.python()


def stock(python:str) -> dict:
  """Stock = a PyPI install whose files still hash to its RECORD (checked under the engine's python)."""
  r = _run(python, "stock", {}, 600)
  if r["direct_url"]:
    raise Refused(f"this vLLM {r['version']} was installed from a VCS, local or editable source (direct_url.json), not from PyPI")
  if r["changed"]:
    raise Refused(f"this vLLM {r['version']} is patched: {r['changed']} no longer matches its RECORD hash")
  if not r["checked"]:
    raise Refused(f"this vLLM {r['version']} has no hashed files in its RECORD, so it cannot be shown to be stock")
  return {"engine_version": "vllm " + r["version"], "files_checked": r["checked"]}


def _group(src:pathlib.Path) -> dict:
  q = json.loads((src / "config.json").read_text())["quantization_config"]
  groups = q.get("config_groups") or {}  # read for any model a screen offers Speed up on: a missing key refuses, never raises
  if len(groups) != 1:
    raise Refused(f"the checkpoint has {len(groups)} quantization groups; only one group is handled")
  w = next(iter(groups.values()))["weights"]
  if w.get("strategy") != "group" or w.get("type") != "int" or w.get("symmetric") is not False or w.get("num_bits") != 4:
    raise Refused("only asymmetric int4 group quantization is handled (the checkpoint's weights say otherwise)")
  return w


def _gb(p:pathlib.Path) -> tuple[int, int]:
  w = _group(p)
  if w.get("group_size") is None:
    raise Refused("no group_size in the quantization config")
  return int(w["group_size"]), int(w["num_bits"])


def plan(src:str, python:str, target:str | None = None, part:dict | None = None) -> dict:
  p = pathlib.Path(src)
  g, b = _gb(p)
  if "lm_head" not in json.loads((p / "config.json").read_text())["quantization_config"].get("ignore", []):
    raise Refused("the output layer is already quantized (lm_head is not in the ignore list)")
  return plan_row([head(p, "lm_head.weight_packed", g)], f"int{b} group {g}", b + (16 / g + b / g), g, b)  # a bf16 scale and a nibble zero point per group


def plan_mtp(src:str, python:str, target:str | None = None, part:dict | None = None) -> dict:
  """The MTP draft layer: every module under part's prefix in `ignore` (kept 16 bit by the checkpoint), or Refused: no MTP
  layer, already quantized, a module not divisible into groups."""
  p, prefix = pathlib.Path(src), part["prefix"]
  (g, b), names = _gb(p), _index(p)
  if not any(k.startswith(prefix) for k in names):
    raise Refused(f"the checkpoint has no MTP draft layer (no {prefix}* weights in {'its index' if (p / INDEX).is_file() else 'model.safetensors'})")
  packed = next((k for k in names if k.startswith(prefix) and k.endswith(".weight_packed")), None)
  ignored = [m for m in json.loads((p / "config.json").read_text())["quantization_config"].get("ignore", []) if m.startswith(prefix)]
  if packed or not ignored:
    raise Refused(f"the MTP draft layer is already quantized ({packed + ' is in the checkpoint' if packed else 'no ' + prefix + '* module is in the ignore list'})")
  return plan_row([head(p, m + ".weight_packed", g, m, names) for m in ignored], f"int{b} group {g}", b + (16 / g + b / g), g, b)


def _header(f:pathlib.Path) -> dict:
  with f.open("rb") as h:  # a safetensors header is a length and JSON: BoltBeam's python needs no torch for it
    return json.loads(h.read(int.from_bytes(h.read(8), "little")))


def _index(p:pathlib.Path) -> dict[str, str]:
  """{tensor: file} of checkpoint p: its index, else the header of its one model.safetensors."""
  if (p / INDEX).is_file():
    return json.loads((p / INDEX).read_text())["weight_map"]
  return dict.fromkeys(_header(p / "model.safetensors"), "model.safetensors") if (p / "model.safetensors").is_file() else {}


def head(p:pathlib.Path, packed:str, g:int, module:str = "lm_head", wmap:dict[str, str] | None = None) -> dict:
  """module.weight's name, shape, dtype and bits now in checkpoint p (wmap: its _index), or Refused: already quantized
  (`packed` is in it), absent (lm_head: tied), not a float, or not divisible into groups of g."""
  wmap = _index(p) if wmap is None else wmap
  what = "the output layer" if module == "lm_head" else "the draft layer module " + module
  if packed in wmap:
    raise Refused(f"{what} is already quantized ({packed} is in the checkpoint)")
  h = _header(p / wmap[module + ".weight"]).get(module + ".weight") if module + ".weight" in wmap else None
  if h is None:
    raise Refused("the checkpoint has no lm_head.weight (tied to the embedding), so there is no separate output layer to lighten" if module == "lm_head"
                  else f"{what} has no weight in the checkpoint")
  shape, dtype = h["shape"], h["dtype"]
  if (bits_now := {"BF16": 16, "F16": 16, "F32": 32}.get(dtype)) is None:
    raise Refused(f"{what} is {dtype}, not a 16 or 32 bit float")
  if len(shape) != 2 or shape[1] % g or shape[1] % 8 or shape[0] % 8:
    raise Refused(f"{what} {shape} does not divide into groups of {g}")
  return {"module": module, "shape": shape, "dtype": dtype, "bits_now": bits_now}


def plan_row(layers:list[dict], target:str, bpw:float, g:int, b:int) -> dict:
  n = [x["shape"][0] * x["shape"][1] for x in layers]
  return {"layer": ", ".join(x["module"] + ".weight" for x in layers), "modules": [x["module"] for x in layers],
          "shape": layers[0]["shape"] if len(layers) == 1 else [x["shape"] for x in layers], "held": "/".join(sorted({x["dtype"] for x in layers})),
          "bits_now": max(x["bits_now"] for x in layers), "target": target, "bits_target": bpw, "group_size": g, "num_bits": b,
          "bytes_now": sum(k * x["bits_now"] // 8 for k, x in zip(n, layers)), "bytes_after": sum(int(k * bpw / 8) for k in n)}


def write(src:str, dst:str, python:str, pl:dict, **extra) -> dict:
  """The part's modules into its file in one pass (vllm_engine.write); the relative error per module, for the record."""
  out = _run(python, "write", {"src": src, "dst": dst, "bits": pl["num_bits"], "group_size": pl["group_size"], "modules": dict.fromkeys(pl["modules"], pl["file"].format(bits=pl["num_bits"])),
                               "targets": pl.get("targets", []), **extra})
  return {"part": pl["part"], "layer": pl["layer"], "from": pl["held"], "to": pl["target"], "relative_error": out["rel_error"],
          "files": out["resharded"] + out["files"] + ["config.json", INDEX], "all_else": "linked, unchanged"}


def accuracy(src:str, dst:str, python:str, texts:list[str]) -> dict:
  runs = [_run(python, "logprobs", {"model": m, "texts": texts}) for m in (src, dst)]
  a, b = runs[0]["tokens"], runs[1]["tokens"]
  if len(a) != len(b) or [x[0] for x in a] != [x[0] for x in b]:
    raise Refused("the original and the copy tokenize the prompts differently")
  import math
  ppl = lambda rows: math.exp(-sum(r[2] for r in rows) / len(rows))
  if runs[0]["version"] != runs[1]["version"]:
    raise Refused(f"the original ran on vLLM {runs[0]['version']} and the copy on {runs[1]['version']}")
  cuts = [0]
  for k in runs[0]["lengths"]:
    cuts.append(cuts[-1] + k)
  if cuts[-1] != len(a) or runs[0]["lengths"] != runs[1]["lengths"]:
    raise Refused("the original and the copy scored different numbers of positions per document")
  per_doc = [{"n": hi - lo, "hits": sum(x[1] == y[1] for x, y in zip(a[lo:hi], b[lo:hi]))} for lo, hi in zip(cuts, cuts[1:])]
  return {"method": "vLLM prompt_logprobs, teacher-forced on the shipped natural-text corpus", "n": len(a), "engine_version": "vllm " + runs[0]["version"], "per_doc": per_doc,
          "top1_agreement": sum(x[1] == y[1] for x, y in zip(a, b)) / len(a),
          "ppl_original": ppl(a), "ppl_copy": ppl(b)}


def mtp(model:str, python:str, out:pathlib.Path, how:dict) -> dict:
  """Speed with MTP on (collectors/vllm_mtp, the `screen mtp` path) at k=0 and each k of how["ks"] (default collectors/mtp.KS), more
  while how["grow"] and a larger k could still win (core/draft_count.sweep_next; its last reason returned as sweep), at the short and the long
  depth, as the core reads it (mtp_depths), on the engine found (python: the one the stock check passed). The long depth is the shipped real
  text, so every arm reads the same ids where the tokenizer is the same (compare_mtp checks the ids' hashes)."""
  from boltbeam.collectors import mtp as core, vllm_mtp
  out.mkdir(parents=True, exist_ok=True)
  try:
    rec = vllm_mtp.measure(model, how.get("ks") or mtp_ks(), backend=device(python)["backend"], out=out, tokens=how["tokens"],
                           long_tokens=how["long_tokens"], python=python, grow=bool(how.get("grow")))
  except (core.MtpRefused, RuntimeError) as exc:
    raise Refused(f"the MTP measurement failed on {model}: {exc}")
  builds = {e.get("vllm") for e in rec["engines"]}
  if len(builds) != 1:
    raise Refused(f"the MTP measurement of {model} ran on several vLLM builds: {sorted(map(str, builds))}")
  return {"depths": mtp_depths(rec), "engine": builds.pop(), "sweep": (rec.get("sweep") or [None])[-1]}


def mtp_ks() -> list[int]:
  """The draft counts mtp() starts from when how names none (Speed up's time estimate counts them)."""
  from boltbeam.collectors import mtp
  return list(mtp.KS)


def mtp_depths(rec:dict) -> dict:
  """An MTP measurement (collectors/mtp) as {depth: {context, base, ks, tokens}}: k=0 (vLLM without speculation) is the base, each
  prompt's tok/s; every other k a draft count with its prompts' tok/s and the acceptance length vLLM counted; tokens: each
  prompt's output hash per k and at the base, so the arms' outputs can be compared; read: each prompt's input ids' hash."""
  sha = lambda d, k, f="token_ids_sha": {r["prompt"]: r.get(f) for r in rec.get("rows", []) if r["depth"] == d and str(r["k"]) == k}  # noqa: E731
  return {d: {"context": v["0"]["context"], "base": v["0"]["per_prompt"], "tokens": {"base" if k == "0" else k: sha(d, k) for k in v}, "read": sha(d, "0", "prompt_sha"),
              "ks": {k: {"per_prompt": x["per_prompt"], "accept_length": x.get("accept_length")} for k, x in v.items() if k != "0"}} for d, v in rec["depths"].items()}


def device(python:str) -> dict:
  """The GPU this machine's engine runs on: target id (target/hardware.py's, e.g. nvidia_sm120), name, memory, and the
  bytes the engine may use (total times BOLTBEAM_VLLM_GPU_MEMORY, default 0.85, the driver's gpu_memory_utilization)."""
  import os
  out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True)
  if out.returncode or not out.stdout.strip():
    raise Refused("nvidia-smi did not say what the GPU is")
  name, cap, mib = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
  total = int(mib) * 2**20
  return {"target": "nvidia_sm" + cap.replace(".", ""), "name": name, "backend": "CUDA", "memory_bytes": total,
          "budget_bytes": int(total * float(os.environ.get("BOLTBEAM_VLLM_GPU_MEMORY", 0.85)))}


def speed(model:str, python:str, contexts:list[int], batches:list[int], tokens:int, reps:int) -> list[dict]:
  """The repo's own decode measurement (runtime/vllm_decode_drive.py, batches given; N tokens minus 1 token over N - 1) at each
  context, in the engine's python. Refuses to start while another program holds the GPU."""
  from boltbeam.collectors import engine_common as ec, providers, vllm_decode as vd
  free = providers.gpu_free("CUDA")
  if not free["free"]:
    raise Refused(f"speed not measured: {free['reason']}; a GPU another program holds gives slower numbers that look real")
  argv = [python, str(vd.DRIVER), "--model", model, "--contexts", ",".join(map(str, contexts)), "--batches", ",".join(map(str, batches)),
          "--tokens", str(tokens), "--reps", str(reps)]
  try:
    import hashlib, tempfile
    log = pathlib.Path(tempfile.gettempdir()) / f"boltbeam-speed-{hashlib.sha256(' '.join(argv).encode()).hexdigest()[:12]}.log"
    rows = ec.run_driver(argv, env={**ec.venv_env(python), **vd.RUN_ENV}, log=log, timeout_s=7200)
  except ec.DriverFailed as exc:
    raise Refused(f"the decode measurement of {model} failed: {exc}")
  after = providers.gpu_free("CUDA")
  if not after["free"]:
    raise Refused(f"the decode measurement left the GPU held: {after['reason']}")
  return [r for r in rows if "step_samples_ms" in r]
