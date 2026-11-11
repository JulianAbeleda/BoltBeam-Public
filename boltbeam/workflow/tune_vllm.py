"""The vLLM adapter of `boltbeam tune` (workflow/tune.py): the installed vLLM read without starting a model (its version from
importlib.metadata, its features from its own source: the flags its arg parser adds and the Literal values of its config
classes, by an AST read), the checkpoint's facts (weights from the safetensors headers, the KV per token, the MTP head,
experts), the GPU's memory from its telemetry source, a measured point through runtime/vllm_decode_drive.py --prefill on
real text, and the `vllm serve` command. vLLM is found as the rest of BoltBeam finds it (collectors/vllm_decode.python:
$BOLTBEAM_VLLM_PYTHON, the python beside `vllm` on PATH, the engine scan); --build names that python or its venv."""
from __future__ import annotations

import ast
import json
import pathlib
import statistics
import subprocess
from typing import Any, Callable

from boltbeam.core.refused import Refused

ENGINE = "vllm"
NS = "vllm"
FACTS = {"gpu_share": "vLLM's share of the GPU (set by the --gpu-memory-utilization row; 0.92 is vLLM's default)",
         "kv_bytes_per_token": "KV bytes per token at the cache type the --kv-cache-dtype row picks",
         "offload_gib": "GiB of weights the --cpu-offload-gb row offloads",
         "process_headroom_gib": "GiB of RAM kept free for the vLLM process itself, beyond the margin usable RAM already leaves",
         "min_share": "the least share of the GPU the weights need: the weights plus a small allowance for activations and KV, over the GPU's total",
         "gpu_warning": "a sentence when another program holds a meaningful share of the GPU, else empty",
         "mtp_k_measured": "the MTP draft count measured on this chip", "mtp_why": "why there is no measured draft count"}
FEATURE_FILES = ("engine/arg_utils.py", "config/cache.py", "config/speculative.py", "config/offload.py")
PROBE = ("import importlib.metadata as m, importlib.util as u, json, pathlib\n"
         "s = u.find_spec('vllm')\n"
         "print(json.dumps({'version': m.version('vllm'), 'root': str(pathlib.Path(s.origin).parent) if s and s.origin else None}))")
GiB = 1 << 30
# Margins, not measurements: vLLM's own python process, CUDA context and pinned-buffer bookkeeping take host RAM beyond the
# weights it offloads, and activations take GPU memory beyond the weights. Round numbers chosen to be safely above what
# a start was seen to need, not derived from a model.
PROCESS_HEADROOM_GIB = 4
ACTIVATION_ALLOWANCE_GIB = 2
BUSY_GPU_SHARE = 0.05  # another program holding more than this share of the GPU's memory is called out


def _python(build:str | None) -> str | None:
  from boltbeam.collectors import vllm_decode as vd
  if not build:
    return vd.python()
  p = pathlib.Path(build).expanduser()
  p = p / "bin" / "python" if p.is_dir() else p
  if not p.name.startswith("python"):  # never run another program with -c
    raise Refused(f"{p} is not a python: pass --build the python of the venv vLLM is installed in, or the venv")
  return str(p)


def features(root:pathlib.Path) -> dict[str, str]:
  """The installed vLLM's features with where each is (file:line): every literal option string an add_argument call in its
  arg parser adds (--max-num-batched-tokens), and every value of a module-level Literal in its config classes, as
  Name:value (CacheDType:fp8, MTPModelTypes:mtp). Read from the source, nothing imported."""
  out: dict[str, str] = {}
  for rel in FEATURE_FILES:
    f = root / rel
    if not f.is_file():
      continue
    tree = ast.parse(f.read_text())
    for n in ast.walk(tree):
      if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument":
        for a in n.args:
          if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-"):
            out.setdefault(a.value, f"vllm/{rel}:{a.lineno}")
    for n in tree.body:
      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and \
         isinstance(n.value, ast.Subscript) and getattr(n.value.value, "id", None) == "Literal":
        for c in ast.walk(n.value.slice):
          if isinstance(c, ast.Constant) and isinstance(c.value, str):
            out.setdefault(f"{n.targets[0].id}:{c.value}", f"vllm/{rel}:{c.lineno}")
  return out


def _probe(py:str, code:str, timeout:float) -> dict[str, Any]:
  try:
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=timeout)
  except (OSError, subprocess.SubprocessError) as exc:
    raise Refused(f"the Python at {py} could not be run to read the vLLM build (missing, not executable or too slow to answer)") from exc
  if proc.returncode:
    why = Refused(f"the Python at {py} could not import vLLM, so its build cannot be read")
    why.detail = (proc.stderr.strip().splitlines() or ["failed"])[-1]  # the raw error: provenance, not the reason
    raise why
  return json.loads(proc.stdout.strip().splitlines()[-1])


def detect(build:str | None, *, target=None) -> dict[str, Any]:
  """The installed vLLM: version, source folder and features, without starting a model. A Refused when vLLM does not
  run on this machine's GPU (collectors/vllm_decode.available says why) or is not installed."""
  from boltbeam.collectors import vllm_decode as vd
  if target is None:
    from boltbeam.target.targets import get_target
    from boltbeam.workflow import chips
    tid = chips.detect().get("target_id")
    target = get_target(tid) if tid else None
  if target is not None and target.backend not in vd.BACKENDS:
    raise Refused(vd.available(target) or f"vLLM does not run on {target.backend}")
  py = _python(build)
  if not py or not pathlib.Path(py).is_file():
    raise Refused(f"vLLM was not found: set ${vd.ENV} to the python of the venv vLLM is installed in, or pass --build")
  got = _probe(py, PROBE, 60)
  if not got.get("root"):
    raise Refused(f"{py} has vLLM's metadata but not its package")
  root = pathlib.Path(got["root"])
  return {"build": py, "version": f"vLLM {got['version']}", "root": str(root), "features": features(root),
          "backend": getattr(target, "backend", None) or "?", "fork": None,
          "source": f"vLLM {got['version']} source at {root} (arg parser and config classes, read; no model started)"}


def gpu_memory() -> dict[str, Any]:
  """The first GPU's total and free memory, read through its vendor's telemetry source (NVML, amdgpu sysfs): no CUDA
  context, no model; {} when it cannot say."""
  from boltbeam.collectors import gpu_telemetry
  from boltbeam.target.hardware import hardware_profile
  try:
    dev = (hardware_profile()["gpu"].get("devices") or [None])[0]
    used, total = gpu_telemetry.source_for(dev).memory_mib()
  except Exception:  # noqa: BLE001  no reading is a stated unknown, never a crash
    return {}
  return {"free": (total - used) * 2**20, "total": total * 2**20} if used is not None and total else {}


def _config(model:str) -> dict[str, Any]:
  p = pathlib.Path(model).expanduser() / "config.json"
  try:
    c = json.loads(p.read_text())
  except (OSError, ValueError) as exc:
    raise Refused(f"{pathlib.Path(model).name}: no Hugging Face config.json that vLLM could read (missing or not valid JSON)") from exc
  return {**c, **(c.get("text_config") or {})}


def facts(model:str, host:dict[str, Any], b:dict[str, Any], options:dict[str, Any]) -> dict[str, Any]:
  """The checkpoint's weights on the GPU (text only, no MTP head: workflow/recommend), its KV per token at the model's
  bf16, MTP layers and experts (config.json), and the GPU's memory. options: gpu ({free, total}), profile, tensors
  (stand-ins)."""
  from boltbeam.report.plain import words
  from boltbeam.roofline import prefill_work as pw
  from boltbeam.workflow import recommend as rec
  w = words()["tune_vllm"]
  c = _config(model)
  profile = options.get("profile")
  if profile is None:
    from boltbeam.profile.loaders import profile_from_model
    profile = profile_from_model(model, None).to_json()
  tb = options.get("tensors") or rec.tensor_bytes(model)
  weights = rec.weights_resident(tb, text_only=True)
  kv = pw.kv_bytes_per_token(profile, element_bytes=2) or 0
  draft = int(c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers") or 0)
  experts = int(c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts") or 0)
  gpu = options.get("gpu")
  if gpu is None:
    gpu = gpu_memory()
  notes = [w["model"].format(name=pathlib.Path(model).name, gib=weights / GiB, kv=kv, draft=draft, experts=experts or "no")]
  if options.get("experts_in_ram") is not None:
    notes.append(w["placement"])
  total, free = gpu.get("total"), gpu.get("free")
  share0 = 0.92 if not (total and free) else min(0.92, max(0.01, (free - GiB) / total))
  if total and free:
    usable = int(host.get("usable_ram_bytes") or 0)
    kv_ctx = kv * (1 if not host.get("context_tokens") else int(host["context_tokens"])) / (2 if "CacheDType:fp8" in (b.get("features") or {}) else 1)
    need, have = weights + kv_ctx, share0 * total + max(0, usable - PROCESS_HEADROOM_GIB * GiB)
    if need > have:
      raise Refused(w["no_fit"].format(need=need / GiB, gpu=share0 * total / GiB, ram=usable / GiB, headroom=PROCESS_HEADROOM_GIB,
                                       context=int(host.get("context_tokens") or 0)))
  held = (total - free) if total and free else 0
  warning = w["busy"].format(held=held / GiB, total=total / GiB, pct=BUSY_GPU_SHARE * 100) if total and free and held > BUSY_GPU_SHARE * total else ""
  out = {"model_weight_bytes": weights, "kv_bytes_per_token": kv, "model_draft_layers": draft, "model_experts": experts,
         "host_expert_bytes": 0, "gpu_total_bytes": gpu.get("total"), "gpu_free_bytes": gpu.get("free"), "unified_memory": False,
         f"{NS}.gpu_share": 0.92, f"{NS}.kv_bytes_per_token": kv, f"{NS}.offload_gib": 0, f"{NS}.process_headroom_gib": PROCESS_HEADROOM_GIB,
         f"{NS}.min_share": (weights + ACTIVATION_ALLOWANCE_GIB * GiB) / total if total else None, f"{NS}.gpu_warning": warning, f"{NS}.mtp_k_measured": None,
         f"{NS}.mtp_why": "no MTP measurement in this machine's facts", "notes": notes}
  out["sources"] = {"model_weight_bytes": f"{tb.get('source', 'safetensors headers')}: all tensors less the vision tower and the MTP head",
                    "kv_bytes_per_token": "roofline/prefill_work.kv_bytes_per_token at 2 bytes", "model_draft_layers": "config.json",
                    "model_experts": "config.json", "gpu_total_bytes": "the GPU's telemetry source (NVML / amdgpu sysfs)" if gpu else "not read",
                    "gpu_free_bytes": "the GPU's telemetry source (NVML / amdgpu sysfs)" if gpu else "not read"}
  return out


def _typed(v:str) -> Any:
  try:
    return json.loads(v)
  except ValueError:
    return v


def engine_args(values:list[dict[str, Any]]) -> dict[str, Any]:
  """The setup as LLM(...) keyword arguments: a flag --a-b is a_b, a config key a.b is a: {b: value, **with}."""
  out: dict[str, Any] = {}
  for v in values:
    if v["kind"] == "config":
      top, _, key = v["name"].partition(".")
      out[top] = {**out.get(top, {}), **(v.get("with") or {}), key: _typed(v["value"])}
    elif v["name"] == "--cpu-offload-params":
      out["cpu_offload_params"] = v["value"].split()
    else:
      out[v["name"].lstrip("-").replace("-", "_")] = _typed(v["value"])
  return out


def render(values:list[dict[str, Any]], f:dict[str, Any], b:dict[str, Any], model:str, extra:list[str] | None = None) -> dict[str, Any]:
  flags: list[str] = []
  config: dict[str, Any] = {}
  for v in values:
    if v["kind"] == "config":
      top, _, key = v["name"].partition(".")
      config[top] = {**config.get(top, {}), **(v.get("with") or {}), key: _typed(v["value"])}
    else:
      flags += [v["name"], *(v["value"].split() if v["name"] == "--cpu-offload-params" else [v["value"]])]
  for top, d in config.items():
    flags += [f"--{top.replace('_', '-')}", json.dumps(d, separators=(",", ":"))]
  argv = ["vllm", "serve", model, *flags, *(extra or [])]
  return {"env": {}, "flags": flags, "config": config, "argv": argv,
          "command": " ".join(f"'{a}'" if a.startswith("{") else a for a in argv)}


def launch_measure(model:str, b:dict[str, Any], f:dict[str, Any], *, port:int, out:pathlib.Path, extra:list[str] | None = None) -> Callable[..., float]:
  """run(values=, prompt=, reps=, tag=) -> wall ms: one vLLM LLM per point (runtime/vllm_decode_drive.py --prefill on the
  in-tree real-text corpus, as collectors/prefill.vllm_truth runs it), the median wall of its timed prompts, at the
  setup's GPU share. The driver sizes the context to the prompt itself. vLLM's offline engine opens no port, so --port is
  not used, and the driver takes no serve flags, so --flags reaches the command only."""
  from boltbeam.collectors import engine_common as ec, prefill, vllm_decode as vd
  from boltbeam.collectors.llama_server_prefill import CORPUS

  def run(*, values, prompt, reps, tag):
    args = engine_args(values)
    args.pop("max_model_len", None)
    args.setdefault("gpu_memory_utilization", f.get(f"{NS}.gpu_share") or 0.92)
    pick = {k: args.pop(k) for k in ("max_num_batched_tokens", "gpu_memory_utilization", "speculative_config") if k in args}
    argv = [b["build"], str(vd.DRIVER), "--model", str(pathlib.Path(model).absolute()), "--prefill", str(prompt), "--warm", str(prefill.VLLM_WARM),
            "--reps", str(reps), "--corpus", str(CORPUS), *[x for k, v in pick.items() for x in (f"--{k.replace('_', '-')}", json.dumps(v) if isinstance(v, dict) else str(v))],
            "--engine-args", json.dumps(args)]
    line = ec.run_driver(argv, env={**ec.venv_env(b["build"]), **vd.RUN_ENV}, log=out / f"{tag}.driver.log")[-1]
    return statistics.median(s * 1e3 for s in line["wall_samples_s"])
  return run
