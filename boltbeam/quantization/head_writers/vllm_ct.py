"""vLLM, compressed-tensors pack-quantized (W4, group): the output layer (lm_head) as RTN int4, asymmetric, nibble zero
point, the group size and bits read from the checkpoint's own quantization_config. Three checkpoint changes, no vLLM
patch: lm_head out of `ignore`, target `re:.*lm_head$` in the group, the bf16 head out of the shards. The torch work
is vllm_engine.py, run under the engine's own python."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from boltbeam.core.refused import Refused

ENGINE_SCRIPT = pathlib.Path(__file__).with_name("vllm_engine.py")


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
  groups = q["config_groups"]
  if len(groups) != 1:
    raise Refused("the checkpoint has more than one quantization group; only one group is handled")
  w = next(iter(groups.values()))["weights"]
  if w.get("strategy") != "group" or w.get("type") != "int" or w.get("symmetric") is not False or w.get("num_bits") != 4:
    raise Refused("only asymmetric int4 group quantization is handled (the checkpoint's weights say otherwise)")
  return w


def plan(src:str, python:str, target:str | None = None) -> dict:
  p = pathlib.Path(src)
  w = _group(p)
  cfg = json.loads((p / "config.json").read_text())
  if "lm_head" not in cfg["quantization_config"].get("ignore", []):
    raise Refused("the output layer is already quantized (lm_head is not in the ignore list)")
  if w.get("group_size") is None:
    raise Refused("no group_size in the quantization config")
  g, b = int(w["group_size"]), int(w["num_bits"])
  shape, dtype, bits_now = head(p, "lm_head.weight_packed", g)
  return plan_row(shape, dtype, bits_now, f"int{b} group {g}", b + (16 / g + b / g), g, b)  # a bf16 scale and a nibble zero point per group


def head(p:pathlib.Path, packed:str, g:int) -> tuple[list[int], str, int]:
  """lm_head.weight's (shape, dtype, bits now) in checkpoint p, or Refused: already quantized (`packed` is in it), tied, not a
  float, or not divisible into groups of g."""
  idx_path = p / "model.safetensors.index.json"
  wmap = json.loads(idx_path.read_text())["weight_map"] if idx_path.is_file() else None
  if packed in (wmap or {}):
    raise Refused(f"the output layer is already quantized ({packed} is in the checkpoint)")
  holder = wmap.get("lm_head.weight") if wmap is not None else "model.safetensors"
  tied = Refused("the checkpoint has no lm_head.weight (tied to the embedding), so there is no separate output layer to lighten")
  if holder is None:
    raise tied
  with (p / holder).open("rb") as f:  # a safetensors header is a length and JSON: BoltBeam's python needs no torch for it
    h = json.loads(f.read(int.from_bytes(f.read(8), "little"))).get("lm_head.weight")
  if h is None:
    raise tied
  shape, dtype = h["shape"], h["dtype"]
  bits_now = {"BF16": 16, "F16": 16, "F32": 32}.get(dtype)
  if bits_now is None:
    raise Refused(f"the output layer is {dtype}, not a 16 or 32 bit float")
  if shape[1] % g or shape[1] % 8 or shape[0] % 8:
    raise Refused(f"the output layer {shape} does not divide into groups of {g}")
  return shape, dtype, bits_now


def plan_row(shape:list[int], dtype:str, bits_now:int, target:str, bpw:float, g:int, b:int) -> dict:
  return {"output_layer": "lm_head.weight", "shape": shape, "held": dtype, "bits_now": bits_now, "target": target,
          "bits_target": bpw, "group_size": g, "num_bits": b,
          "bytes_now": shape[0] * shape[1] * bits_now // 8, "bytes_after": int(shape[0] * shape[1] * bpw / 8)}


def write(src:str, dst:str, python:str, pl:dict, **extra) -> dict:
  out = _run(python, "write", {"src": src, "dst": dst, "bits": pl["num_bits"], "group_size": pl["group_size"], **extra})
  return {"layer": "lm_head.weight", "from": pl["held"], "to": pl["target"], "relative_error": out["rel_error"],
          "files": out["resharded"] + [out["head_file"], "config.json", "model.safetensors.index.json"], "all_else": "linked, unchanged"}


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
