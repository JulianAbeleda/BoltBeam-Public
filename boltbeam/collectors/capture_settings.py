"""Capture and untraced measurement answer the same persisted request."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.artifacts.run_io import read_json

# The engine settings the vLLM adapter can honor: the one list every vLLM run and capture checks a request against.
MTP_CONTEXT = "mtp_context.json"  # a speculative run's natural context ids, the same for every process of the run
VLLM_SETTINGS = ("chunk_size", "attention_backend", "tensor_parallel_size", "gemm_backends", "mtp", "mtp_prompt")
# llama.cpp's flag per setting, in the order they are passed. Where the experts sit (n_gpu_layers, n_cpu_moe,
# override_tensor, cpu_moe), the MoE cache (moe_cache_mib, moe_cache_cpu_share) and the draft (spec_type, draft_model,
# cache_type_k_draft): which program of which build has which is the fork row's flags (data/engine_families.json).
LLAMA_FLAGS = {"chunk_size": "-ub", "batch_size": "-b", "cache_type_k": "-ctk",
               "cache_type_v": "-ctv", "flash_attention": "-fa",
               "n_gpu_layers": "-ngl", "n_cpu_moe": "-ncmoe", "override_tensor": "-ot", "cpu_moe": "--cpu-moe",
               "moe_cache_mib": "--moe-cache-mib", "moe_cache_cpu_share": "--moe-cache-cpu-share",
               "spec_type": "--spec-type", "draft_model": "-md", "cache_type_k_draft": "-ctkd"}
LLAMA_SWITCHES = ("cpu_moe",)  # a flag with no value: the setting is true
LLAMA_WHOLE = {"chunk_size": 1, "batch_size": 1, "n_gpu_layers": 0, "n_cpu_moe": 0, "moe_cache_mib": 1}  # whole numbers, at least this
LLAMA_TEXT = ("override_tensor", "spec_type", "draft_model", "cache_type_k_draft")  # passed as written
LLAMA_FIELDS = {"n_ubatch": "chunk_size", "n_batch": "batch_size", "type_k": "cache_type_k",
                "type_v": "cache_type_v", "flash_attn": "flash_attention"}
GEMM_BACKENDS = ("ATEN", "TRITON", "CUTLASS", "CPP")  # Inductor's matrix backends; a request is a comma list of these


def switch(value:str) -> bool:
  """A switch setting on the command line (llama.cpp's --cpu-moe): `on` is its one value, as the screen passes it."""
  if str(value).lower() not in ("on", "true"):
    raise ValueError(f"a switch is on or not asked, got {value!r}")
  return True


# The settings a run asks for on the pipeline's command line, each its value type, its name on screen and its help:
# the one table the parser (screen pipeline), the request (pipeline._engine_settings) and the screen's Settings page
# read. Which engine honors which is ACCEPTED; tensor_parallel_size is not asked, the layout sets it.
ASKED = {
  "chunk_size": (int, "chunk size", "requested engine chunk size; unsupported adapters refuse it"),
  "attention_backend": (str, "attention backend", "requested attention backend by the engine's name for it (vLLM: "
                        "TRITON_ATTN, ROCM_ATTN, ...); recorded with the run, and adapters that cannot honor it refuse it"),
  "gemm_backends": (str, "matrix backends", "vLLM: the compiler's matrix backends (TRITON: its own Triton GEMMs, not "
                    "cuBLAS, whose closed kernels need GPU counters to read); recorded with the run, others refuse it"),
  "mtp": (int, "MTP draft tokens", "vLLM: multi-token prediction with this many draft tokens per step (speculative "
          "decoding); decode runs on natural text (data/mtp_prompts.json) at one context; recorded with the run"),
  "mtp_prompt": (str, "MTP prompt", "vLLM with --mtp: the prompt id (data/mtp_prompts.json) the context is built from "
                 "(default: the set's first)"),
  "n_gpu_layers": (int, "GPU layers", "llama.cpp -ngl: layers kept on the GPU (default: all)"),
  "n_cpu_moe": (int, "layers with experts in RAM", "llama.cpp -ncmoe: the experts of the first N layers stay in host RAM "
                "and the CPU computes them (the model's layer count: all of them, as --cpu-moe)"),
  "override_tensor": (str, "tensor placement", "llama.cpp -ot: tensor name pattern=buffer type, as llama.cpp takes it "
                      "(exps=CPU keeps every expert in RAM)"),
  "cpu_moe": (switch, "every expert in RAM", "llama.cpp --cpu-moe (on): every expert stays in host RAM and the CPU computes "
              "it; the dense parts stay on the GPU"),
  "moe_cache_mib": (int, "expert cache MiB", "llama.cpp --moe-cache-mib (llama-server): MiB of GPU memory caching the "
                    "experts kept in RAM; needs experts in RAM"),
  "moe_cache_cpu_share": (float, "CPU share of cache misses", "llama.cpp --moe-cache-cpu-share (llama-server, master "
                          "builds): the share of cache misses the CPU computes instead of uploading; needs the cache"),
  "spec_type": (str, "speculation", "llama.cpp --spec-type (llama-server): draft-mtp with --draft-model for an MTP head"),
  "draft_model": (str, "draft model", "llama.cpp -md (llama-server): the draft or MTP head file"),
  "cache_type_k_draft": (str, "draft K cache type", "llama.cpp -ctkd (llama-server): the draft head's K cache type (q8_0, f16)")}
TINYGRAD_WARM = 3  # warm tokens before the measured window (runtime/tinygrad_decode_profile.py --warm)
ACCEPTED = {"vllm": VLLM_SETTINGS, "llama.cpp": tuple(LLAMA_FLAGS)}  # per engine (collectors/providers.ENGINES)


def window_context(row:dict[str, Any]) -> float | None:
  """Only explicit, internally consistent differential windows replace the requested base context."""
  window = (row.get("engine") or {}).get("timing_window") or {}
  if window.get("method") != "long_minus_short":
    return None
  from boltbeam.collectors.llama_bench_decode import decode_window
  from boltbeam.collectors.vllm_decode import decode_window_vllm
  vllm = "vllm_version" in window  # vLLM's window is only checked against its own row's recorded fields (the rest of the rule is the same)
  expected = decode_window_vllm(row.get("context"), row.get("decode_tokens"), row.get("batch"), row["engine"]) if vllm else \
    decode_window(row.get("context"), window.get("long_tokens"), window.get("short_tokens"), window.get("independent_pairs"))
  if window != expected or any(type(window.get(k)) is not int for k in ("long_tokens", "short_tokens", "context_start", "context_end", *(() if vllm else ("repetitions",)))) \
      or not vllm and (window.get("warmup") is not False or type(row.get("decode_tokens")) is not int or row["decode_tokens"] != window["long_tokens"] - window["short_tokens"]):
    raise ValueError("decode timing window is inconsistent with the recorded work")
  if vllm:return None  # a vLLM capture is keyed by the requested context, which stays the headline
  mean = (window["context_start"] + window["context_end"]) / 2
  return int(mean) if mean.is_integer() else mean


def attended_context(trace:dict[str, Any], provider:str) -> float | None:
  """The mean context the measured tokens attended to. Captures subtract a 1-token run from an N-token run at the
  same depth, so the window is tokens 2..N; tinygrad's own timing measures tokens 1..N after its warm tokens."""
  from boltbeam.collectors.engine_kernels import METHOD as ISOLATED  # the engine's kernels timed alone
  from boltbeam.collectors.tinygrad_role_time import OWN_TIMING
  whole = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step"), None)
  if not whole or whole.get("context") is None:
    return None
  if (explicit := window_context(whole)) is not None:
    return explicit
  method = (trace.get("capture") or {}).get("method")
  if method == ISOLATED:  # the kernels ran alone: the context is step 4's, as the row carries it
    return float(whole["context"])
  n = whole.get("decode_tokens") or 1
  engine = whole.get("engine") or {}
  if provider == "tinygrad" and method == OWN_TIMING and all(engine.get(k) is not None for k in ("context_start", "context_end")):
    return (engine["context_start"] + engine["context_end"]) / 2
  base = float(whole["context"]) + (TINYGRAD_WARM if provider == "tinygrad" else 0)
  captured = method not in (None, OWN_TIMING)
  return base + ((n + 3) / 2 if captured else (n + 1) / 2)


def measured_step(trace:dict[str, Any] | None, provider:str, near:float | None = None) -> dict[str, Any] | None:
  """THE measured token of step 4: one whole-step row, picked by one rule for every reader (the headline, the Run
  line, the tie-out, summary.txt, results.json). Rows of another provider never count: one engine's kernels never
  tie out against another's token. With a per-role capture, the row nearest the context that capture attended
  (near); without one, the row at the smallest context, the token closest to the context-1 limit. The other rows
  are "also measured", labelled, never the headline."""
  if not trace:
    return None
  measured_by = trace.get("provider") or ("tinygrad" if str(trace.get("provider_id", "")).startswith("tinygrad")
                                          else "llama.cpp")
  if measured_by != provider:
    return None
  rows = [r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("tok_s")]
  if not rows:
    return None
  def context(r):
    explicit = window_context(r)
    return explicit if explicit is not None else r.get("context")
  key = (lambda r: abs(float(context(r) or 0) - near)) if near is not None else (lambda r: float(context(r) or 0))
  row = min(rows, key=key)
  index = trace["rows"].index(row)
  return {"context": context(row), "requested_context": row.get("context"), "tok_s": row["tok_s"], "ms": 1000.0 / row["tok_s"], "wall_us": row.get("wall_us"),
          "source": row.get("source"),
          "timing_window": (row.get("engine") or {}).get("timing_window"),
          "graph_error": row.get("graph_error"), "index": index, "decode_tokens": row.get("decode_tokens"),
          "rule": "nearest the context the per-role capture attended" if near is not None else "the smallest context",
          "also": [{"context": context(r), "tok_s": r["tok_s"], "ms": 1000.0 / r["tok_s"]} for r in rows if r is not row]}


def step4_token(run:pathlib.Path, context:float | None, provider:str) -> tuple[float | None, int | None]:
  """The untraced whole-step ms per token from step 4, at the context nearest the measured one, only when step 4
  measured the same provider: one provider's kernels never tie out against another's token."""
  p = run / "timing_trace.json"
  if not p.exists():
    return None, None
  token = measured_step(json.loads(p.read_text()), provider, near=context)
  return (token["ms"], token["context"]) if token else (None, None)


def offered(provider:str) -> list[dict[str, Any]]:
  """The settings a run on this engine may ask for, as the screen offers them: the flag, a whole number or text, the
  values when they are a fixed set, the help, and whether the checkpoint needs a draft head (the mtp rows)."""
  from boltbeam.collectors.vllm_mtp import prompt_set
  choices = {"gemm_backends": lambda: list(GEMM_BACKENDS), "mtp_prompt": lambda: prompt_set()["ids"], "cpu_moe": lambda: ["on"]}
  return [{"name": k, "flag": "--" + k.replace("_", "-"), "label": label, "whole": kind is int,
           "choices": choices[k]() if k in choices else None, "about": about, "draft_head": k.startswith("mtp")}
          for k, (kind, label, about) in ASKED.items() if k in ACCEPTED.get(provider, ())]


def contexts(run:pathlib.Path) -> list[int]:
  values = read_json(run / "trace_request.json").get("contexts")
  if not isinstance(values, list) or not values or any(type(c) is not int or c < 0 for c in values):
    raise ValueError("trace_request.json needs explicit nonnegative contexts; capture settings cannot be guessed")
  return values


def selected_context(run:pathlib.Path, context:int | None = None) -> int:
  requested = contexts(run)
  chosen = max(requested) if context is None else context
  if chosen not in requested:
    raise ValueError(f"capture context {chosen} is absent from the requested contexts {requested}")
  return chosen


RECORDED = ("engine", "engine_args", "moe_cache_hit_rate", "moe_cache_trace")  # what the engine ran with (collectors/engine_args), not what was asked


def requested(run:pathlib.Path) -> dict[str, Any]:
  settings = read_json(run / "trace_request.json").get("engine_settings") or {}
  if not isinstance(settings, dict):
    raise ValueError("engine_settings must be an object")
  return {k: v for k, v in settings.items() if k not in RECORDED}


def setting_words(settings:dict[str, Any]) -> list[str]:
  """The requested engine settings as the reader sees them, one phrase each, in every output (summary, report, TUI).
  A matrix-backend request changes which kernels run, so it says the run is not stock vLLM."""
  words = {"chunk_size": lambda v: f"chunk size {v}",
           "attention_backend": lambda v: f"attention backend {v}",
           "tensor_parallel_size": lambda v: f"split across {v} GPUs by rows (tensor parallel)",
           "gemm_backends": lambda v: (f"matrix backends {str(v).upper()}: the compiler's own GEMM kernels in place of "
                                       f"cuBLAS; not stock vLLM"),
           "mtp": lambda v: f"MTP k={v}: speculative decoding, {v} draft token{'s' if v != 1 else ''} per step",
           "mtp_prompt": lambda v: f"MTP context: natural text from prompt {v}",
           "n_gpu_layers": lambda v: f"{v} layers on the GPU",
           "n_cpu_moe": lambda v: f"the experts of the first {v} layers in host RAM",
           "override_tensor": lambda v: f"tensor placement {v}",
           "cpu_moe": lambda v: "every expert in host RAM",
           "moe_cache_mib": lambda v: f"expert cache {v} MiB on the GPU",
           "moe_cache_cpu_share": lambda v: f"the CPU computes {v:g} of the cache misses",
           "spec_type": lambda v: f"speculative decoding {v}",
           "draft_model": lambda v: f"draft {pathlib.Path(str(v)).name}"}
  return [words[k](v) if k in words else f"{k} {v}" for k, v in settings.items() if v is not None]


def mtp_request(settings:dict[str, Any]) -> dict[str, Any] | None:
  """The requested speculation: vLLM's speculative_config for MTP with k draft tokens, and the natural prompt the
  context is built from (collectors/vllm_mtp.PROMPTS, the prompt set's first prompt unless one is named). None when
  the run requests none. A prompt without k, or a k that is not a positive whole number, is refused."""
  k, prompt = settings.get("mtp"), settings.get("mtp_prompt")
  if k is None:
    if prompt is not None:
      raise ValueError("mtp_prompt names the natural text a speculative decode runs on; it needs mtp (draft tokens per step)")
    return None
  if type(k) is not int or k < 1:
    raise ValueError("mtp is the draft tokens per step, a whole number of 1 or more")
  from boltbeam.collectors.vllm_mtp import PROMPTS, prompt_set
  ids = prompt_set()["ids"]
  if prompt is not None and prompt not in ids:
    raise ValueError(f"mtp_prompt {prompt!r} is not in {PROMPTS.name} ({', '.join(ids)})")
  return {"config": {"method": "mtp", "num_speculative_tokens": k}, "prompt": prompt or ids[0], "prompt_set": str(PROMPTS)}


def observed(run:pathlib.Path, context:int) -> dict[str, Any]:
  path = run / "timing_trace.json"
  trace = read_json(path) if path.exists() else {}
  return next((r.get("engine") or {} for r in trace.get("rows", []) if r.get("context") == context), {})


def vllm_streams(run:pathlib.Path, context:int) -> list[str]:
  """The stream slots the untraced engine was built with. A sweep times every batch on one engine sized for its
  largest, so a capture of a smaller batch asks for the same slots: sized for itself it is another allocation (and
  vLLM 0.31.0 faults warming up a 1-stream hybrid 27B with the sweep's 8192 batched tokens)."""
  slots = observed(run, context).get("max_num_seqs")
  return ["--max-num-seqs", str(slots)] if slots is not None else []


def llama_flags(run:pathlib.Path, context:int, *, capture:bool = False, binary:str | None = None,
                tool:str = "llama-bench") -> list[str]:
  """The requested settings as llama.cpp flags. capture: take what the untraced run recorded (a recorded value that
  differs from the request is an error). binary: the program that will run them, whose build must have each flag."""
  settings = requested(run)
  if capture:
    engine = observed(run, context)
    for field, name in LLAMA_FIELDS.items():
      if engine.get(field) is not None:
        actual = engine[field]
        if name in settings and str(settings[name]).lower() != str(actual).lower():
          raise ValueError(f"untraced {name} {actual!r} differs from request {settings[name]!r}")
        settings[name] = actual
  check_llama(settings)
  if tool == "llama-bench" and "," in str(settings.get("override_tensor", "")):
    raise ValueError("override_tensor for llama-bench separates patterns with ';' (a ',' starts another configuration of the benchmark)")
  if binary is not None:
    missing_flags(settings, binary, tool)
  return render_llama(settings)


def all_layers(extra:list[str] | None) -> list[str]:
  """-ngl 99 (every layer on the GPU) unless the flags already place the layers: llama-bench reads a second -ngl as
  a second configuration to run, not a replacement."""
  return [] if "-ngl" in (extra or []) else ["-ngl", "99"]


def check_llama(settings:dict[str, Any]) -> None:
  """The requested llama.cpp settings, each of its kind, and the ones that only mean something with another: a ValueError
  in plain words for the first that is not."""
  unknown = set(settings) - LLAMA_FLAGS.keys()
  if unknown:
    raise ValueError(f"llama.cpp does not support requested settings {sorted(unknown)}")
  for key, least in LLAMA_WHOLE.items():
    if key in settings and (type(settings[key]) is not int or settings[key] < least):
      raise ValueError(f"{key} must be a positive integer" if least else f"{key} must be a whole number, 0 or more")
  for key in LLAMA_SWITCHES:
    if key in settings and settings[key] is not True:
      raise ValueError(f"{key} is on or not asked: true is its one value")
  for key in LLAMA_TEXT:
    if key in settings and (type(settings[key]) is not str or not settings[key].strip()):
      raise ValueError(f"{key} must be text, as llama.cpp takes it")
  if "draft_model" in settings and not pathlib.Path(settings["draft_model"]).is_absolute():
    raise ValueError("draft_model must be an absolute path: a capture runs the engine in its own folder")
  share = settings.get("moe_cache_cpu_share")
  if share is not None and (type(share) not in (int, float) or not 0 <= share <= 1):
    raise ValueError("moe_cache_cpu_share is a share of the cache misses, from 0 to 1")
  in_ram = settings.get("cpu_moe") or settings.get("n_cpu_moe") or "override_tensor" in settings
  if "moe_cache_mib" in settings and not in_ram:
    raise ValueError("moe_cache_mib caches experts kept in RAM, and this run keeps none there (ask cpu_moe, n_cpu_moe or override_tensor)")
  if share is not None and "moe_cache_mib" not in settings:
    raise ValueError("moe_cache_cpu_share is a share of the expert cache's misses, and this run asks for no cache (moe_cache_mib)")


def render_llama(settings:dict[str, Any]) -> list[str]:
  """The settings as llama.cpp's flags, in LLAMA_FLAGS order: the one place a setting becomes a flag."""
  return [value for key, flag in LLAMA_FLAGS.items() if key in settings
          for value in ((flag,) if key in LLAMA_SWITCHES else (flag, f"{settings[key]:g}" if type(settings[key]) is float else str(settings[key])))]


def missing_flags(settings:dict[str, Any], binary:str, tool:str = "llama-bench") -> None:
  """Refuse a setting whose flag this build's program lacks (the fork row's flags, collectors/engine_families.tool_flags),
  saying which build, which flag, and where it is instead."""
  from boltbeam.collectors import engine_families
  row, have = engine_families.tool_flags(binary, tool)
  if have is None:
    return
  lacking = [(k, LLAMA_FLAGS[k]) for k in LLAMA_FLAGS if k in settings and LLAMA_FLAGS[k] not in have]
  if not lacking:
    return
  key, flag = lacking[0]
  build = f"llama.cpp {row['label']}" + (f" ({row['commit']})" if row.get("commit") else "")
  other = [t for t, flags in (row.get("flags") or {}).items() if t != tool and flag in flags]
  raise ValueError(f"{tool} of {build} has no {flag}, so it cannot run the requested {key}"
                   + (f"; {other[0]} of this build has it" if other else "; no program of this build has it"))


def require_supported(run:pathlib.Path, supported:tuple[str, ...] = ()) -> dict[str, Any]:
  settings = requested(run)
  if unknown := set(settings) - set(supported):
    raise ValueError(f"this adapter cannot honor requested settings {sorted(unknown)}")
  return settings


def context_limit(run:pathlib.Path, context:int, field:str) -> int:
  value = observed(run, context).get(field)
  if type(value) is not int or value <= context:
    raise ValueError(f"the untraced run did not record {field}; a capture cannot choose another allocation")
  return value


def vllm_flags(run:pathlib.Path, context:int, *, capture:bool = False) -> list[str]:
  """The driver flags that hold the requested engine settings: the prefill chunk, the attention backend and the
  tensor-parallel split (collectors/multi_gpu.py). A
  capture takes what the untraced run recorded, so the two cannot differ; a recorded value that differs from the
  request is an error, not a choice."""
  settings = require_supported(run, VLLM_SETTINGS)
  size, backend, split = settings.get("chunk_size"), settings.get("attention_backend"), settings.get("tensor_parallel_size")
  gemm = settings.get("gemm_backends")
  if capture:
    engine = observed(run, context)
    if engine.get("chunked_prefill"):
      actual = engine.get("max_num_batched_tokens")
      if size is not None and size != actual:
        raise ValueError("untraced chunk size differs from the request")
      size = actual
    if engine.get("attention_backend"):
      actual = engine["attention_backend"]
      if backend is not None and str(backend).upper() != str(actual).upper():
        raise ValueError(f"untraced attention backend {actual!r} differs from the request {backend!r}")
      backend = actual if backend is not None else None
    if "gemm_backends" in engine and engine["gemm_backends"] != gemm:
      raise ValueError(f"untraced matrix backends {engine['gemm_backends']!r} differ from the request {gemm!r}")
    if (actual := engine.get("tensor_parallel_size")) is not None and actual != (split or 1):
      raise ValueError(f"untraced tensor parallel size {actual} differs from the request {split or 1}")
    if engine and engine.get("speculative_config") != ((mtp_request(settings) or {}).get("config")):
      raise ValueError(f"untraced speculation {engine.get('speculative_config')!r} differs from the request "
                       f"{(mtp_request(settings) or {}).get('config')!r}")
  if split is not None and (type(split) is not int or split < 1):
    raise ValueError("tensor_parallel_size must be a positive integer")
  if size is not None and (type(size) is not int or size <= 0):
    raise ValueError("chunk_size must be a positive integer")
  if gemm is not None and (type(gemm) is not str or not set(gemm.upper().split(",")) <= set(GEMM_BACKENDS)):
    raise ValueError("gemm_backends is a comma list of Inductor's matrix backends: ATEN, TRITON, CUTLASS, CPP")
  if backend is not None and (type(backend) is not str or not backend.strip()):
    raise ValueError("attention_backend must name one of vLLM's attention backends")
  flags = [*(["--max-num-batched-tokens", str(size)] if size is not None else []),
           *(["--attention-backend", backend] if backend is not None else []),
           *(["--tensor-parallel-size", str(split)] if split and split > 1 else []),
           *(["--gemm-backends", gemm.upper()] if gemm is not None else [])]
  if (spec := mtp_request(settings)) is not None:
    import json
    # one natural context per run, read by every process that decodes it (vllm_decode_drive.natural_context)
    flags += ["--speculative-config", json.dumps(spec["config"]), "--mtp-prompt", spec["prompt"], "--prompt-set",
              spec["prompt_set"], "--mtp-context", str(run / MTP_CONTEXT)]
  if capture and observed(run, context).get("max_model_len"):
    flags += ["--model-context-limit", str(context_limit(run, context, "max_model_len"))]
  return flags


def decode_tokens(run:pathlib.Path, context:int) -> int:
  trace = read_json(run / "timing_trace.json")
  row = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("context") == context), {})
  tokens = row.get("decode_tokens")
  window = (row.get("engine") or {}).get("timing_window") or {}
  if window.get("method") == "long_minus_short":
    window_context(row)
    return window["long_tokens"]
  if type(tokens) is not int or tokens < 2:
    raise ValueError("untraced run must record at least two decode tokens before taking a matching capture")
  return tokens
