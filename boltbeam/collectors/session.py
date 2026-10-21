"""A chat session's costs as its context fills: the turn at each depth, decode at that depth, the KV cache it holds,
whether the engine still reads the cache right, and what compaction costs.

    pipeline MODEL --run RUN --target T --provider P --workload session --ctxs 1024,8192,32768 --analyze --measure auto --no-search

The depths are the run's contexts. Step 4 (`measure`):

  tokens       real text: a fixed document set (DOCUMENTS) tokenized by the reference engine's own tokenizer, saved in
               the run (TOKENS) with its sha256, so every depth and every engine reads the same varied ids and a
               rerun repeats them. llama-bench times its own prompts: random ids from its fixed default seed
  reference    a llama.cpp server started for the run (the reference engine), with its prompt cache off per request:
               for each depth the C + TURN tokens and the next CHECK + 1 greedy tokens with the top probabilities of
               each (the reference); the same through the append path (C cached, TURN appended) for llama.cpp's own
               check; at the focus depth the same C + TURN again with the cache on (the warm-cache row)
  turn         per depth, the untraced time to append TURN fresh tokens to a cache already holding C (the per-turn
               time to first token), what the engine batched, decode tokens/s at that depth, the clock, power and
               device memory through the run (gpu_telemetry), and the next tokens checked against the reference
               (AGREEMENT). llama.cpp: llama-bench -p TURN -n DECODE -d C. The fork: its own driver
               (runtime/tinygrad_session_profile.py), where the engine's own prefix match keeps the C tokens
  compaction   a SUMMARY-token prompt on an empty cache (collectors/prefill.py's truth), against the turn at depth
  kv           KV cache bytes per token (prefill_work.kv_bytes_per_token, the engine's KV element) and the context
               that fits on the device beside the weights: (device memory - what the engine held less its KV at a
               depth) / bytes per token, from the memory the run logged

Step 5 (`role_time`) captures one turn at the focus depth and attributes it as a prefill at that depth
(prefill.role_time with depth: prefill_attribution, the same pairing, ceilings and MFU): session_trace.json.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sysconfig
import time
from typing import Any, Callable

from boltbeam.collectors import gpu_telemetry
from boltbeam.collectors import prefill as pf

TURN = 512        # fresh tokens a turn appends
SUMMARY = 2048    # tokens a compaction's summary is re-prefilled as
CHECK = 16        # greedy tokens compared with the reference after the turn (each with its probabilities)
DECODE = 64       # tokens decoded at each depth for its rate
FOCUS = 8192      # the depth nearest this is attributed and gets the warm-cache row
AGREEMENT = {"top": 5, "max_dp": 0.05}  # the next tokens agree: argmax equal, top sets equal, |dp| within max_dp
AGREEMENT_WORDS = (f"after the turn, the next {CHECK + 1} greedy tokens equal the reference's, the {AGREEMENT['top']} most "
                   f"probable tokens at each of the {CHECK} with probabilities are the same set, and no probability among "
                   f"them differs by more than {AGREEMENT['max_dp']}")
TOKENS = "session_tokens.json"
REFERENCE = "session_reference.json"
TRACE = "session_trace.json"
SERVER_LOG = "session_server.log"
DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_session_profile.py"


def documents() -> tuple[list[pathlib.Path], str]:
  """The document set: the standard library's top-level modules of the Python running BoltBeam, in name order."""
  root = pathlib.Path(sysconfig.get_paths()["stdlib"])
  return sorted(root.glob("*.py")), f"the Python {platform.python_version()} standard library's top-level modules ({root}), in name order"


# --- the reference: a llama.cpp server with its prompt cache off per request ------------------------------------

def _server(run:pathlib.Path) -> pathlib.Path:
  return pathlib.Path(pf._bench(run)).resolve().parent / "llama-server"


def tokenize(url:str, count:int) -> tuple[list[int], str]:
  from boltbeam.adapters.llama_server_control import _json_post
  files, words = documents()
  ids: list[int] = []
  for f in files:
    ids += _json_post(url + "/tokenize", {"content": f.read_text(errors="replace"), "add_special": False}, 120.0)["tokens"]
    if len(ids) >= count:
      return ids[:count], words
  raise RuntimeError(f"the document set holds {len(ids)} tokens; the run needs {count}")


def completion(url:str, ids:list[int], n:int, *, cache:bool) -> dict[str, Any]:
  """n greedy tokens after `ids`, each with its top AGREEMENT['top'] probabilities (the full softmax, before
  sampling), and the server's own prompt timing."""
  from boltbeam.adapters.llama_server_control import _json_post
  r = _json_post(url + "/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0, "cache_prompt": cache,
                                       "n_probs": AGREEMENT["top"], "post_sampling_probs": False, "return_tokens": True}, 600.0)
  top = []
  for step in r.get("completion_probabilities") or []:
    cands = step.get("top_logprobs") or step.get("probs") or []
    top.append([[c["id"], math.exp(c["logprob"]) if "logprob" in c else c["prob"]] for c in cands])
  t = r.get("timings") or {}
  return {"tokens": [int(x) for x in r.get("tokens") or []], "top": top, "prompt_n": t.get("prompt_n"),
          "prompt_ms": t.get("prompt_ms"), "cached": r.get("tokens_cached")}


def make_tokens(run:pathlib.Path, model:str, count:int) -> dict[str, Any]:
  """The run's ids: the document set tokenized by the reference engine (a small server, started for it), saved."""
  from boltbeam.adapters.llama_server_control import serve, token_evidence
  with serve(executable=_server(run), model=model, context_size=4096, log_path=run / SERVER_LOG,
             extra=["--parallel", "1"]) as (url, _):
    ids, words = tokenize(url, count)
  ev = token_evidence(ids)
  (run / TOKENS).write_text(json.dumps({"ids": ids, "source": words, "sha256": ev["sha256"], "count": ev["count"]}))
  return {"ids": ids, "source": words, "sha256": ev["sha256"], "count": ev["count"]}


def reference(run:pathlib.Path, model:str, ids:list[int], depth:int, warm:bool) -> dict[str, Any]:
  """The reference at one depth, from a server sized for it with its prompt cache off per request: the C + TURN
  tokens cold and their next tokens; the same through the append path (C cached, then the turn appended: the
  self-check, a second run on the same tokens); with warm, the same again with the cache on."""
  from boltbeam.adapters.llama_server_control import serve
  with serve(executable=_server(run), model=model, context_size=depth + TURN + CHECK + 64,
             log_path=run / SERVER_LOG, extra=["--parallel", "1"]) as (url, command):
    began = time.monotonic()
    cold = completion(url, ids[:depth + TURN], CHECK + 1, cache=False)
    fill_s = time.monotonic() - began
    completion(url, ids[:depth], 1, cache=True)
    appended = completion(url, ids[:depth + TURN], CHECK + 1, cache=True)
    row = {"cold": cold, "appended": appended, "self": agree(cold, appended), "command": command, "cold_s": round(fill_s, 1)}
    if warm:
      row["warm"] = completion(url, ids[:depth + TURN], 1, cache=True)
  path = run / REFERENCE
  saved = json.loads(path.read_text()) if path.exists() else {}
  saved[str(depth)] = row
  path.write_text(json.dumps(saved, indent=1) + "\n")
  return row


def agree(ref:dict[str, Any], got:dict[str, Any]) -> dict[str, Any]:
  """The next tokens against the reference's (AGREEMENT): greedy ids compared up to the first that differs, and the
  top sets and probabilities at every step both report probabilities for, up to and including that one (after it
  the two continue from different text)."""
  steps = min(len(ref["tokens"]), len(got["tokens"]))
  first = next((i for i in range(steps) if ref["tokens"][i] != got["tokens"][i]), None)
  upto = steps if first is None else first + 1
  sets_differ, dp, compared = [], 0.0, 0
  for i in range(upto):
    a = ref["top"][i] if i < len(ref["top"]) else None
    b = got["top"][i] if i < len(got["top"]) else None
    if not a or not b:
      continue
    compared += 1
    pa, pb = {int(k): float(v) for k, v in a}, {int(k): float(v) for k, v in b}
    if set(pa) != set(pb):
      sets_differ.append(i)
    dp = max([dp] + [abs(pa[k] - pb[k]) for k in set(pa) & set(pb)])
  ok = first is None and steps >= CHECK + 1 and not sets_differ and compared >= CHECK and dp <= AGREEMENT["max_dp"]
  return {"pass": ok, "tokens": steps, "first_mismatch": first, "with_probabilities": compared,
          "top_sets_differ_at": sets_differ, "max_dp": round(dp, 5), "rule": AGREEMENT_WORDS}


# --- the turn per engine ---------------------------------------------------------------------------------------

def llama_turn(run:pathlib.Path, model:str, depth:int, target, root, warm:bool) -> dict[str, Any]:
  """llama-bench at the depth: the TURN-token prompt after `depth` tokens it filled untimed, REPS times (its depth
  is filled once and restored for each repetition), and DECODE generated tokens at that depth, its own row."""
  bench = pf._bench(run)
  from boltbeam.workflow.layout import one_gpu
  pin, env = one_gpu(run, "llama.cpp")
  cmd = pf.llama_argv(bench, model, TURN, pf.REPS, depth=depth, gen=DECODE, extra=pin)
  with gpu_telemetry.run_telemetry(run, 0) as tele:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=run, env={**os.environ, **env})
  if proc.returncode != 0:
    raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  row, gen = pf.llama_row(proc.stdout), pf.llama_row(proc.stdout, "n_gen")
  samples = [float(x) / 1e6 for x in row.get("samples_ns") or [row["avg_ns"]]]
  rates = [DECODE / (float(x) / 1e9) for x in gen.get("samples_ns") or [gen["avg_ns"]]]
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  qualified = pf.qualification_sample(row, TURN, depth)
  return {"ms": statistics.median(samples), "qualification_ms": qualified, "samples_ms": samples, "computed": [TURN] * len(samples),
          "chunk": pf.chunk_record(pf.split_chunks(TURN, ub) if ub else [], f"llama-bench n_ubatch {ub}, n_batch {b}",
                                   outputs=-(-TURN // b) if b else None),
          "decode": {"tok_s": statistics.median(rates), "tokens": DECODE, "from": depth, "samples_tok_s": rates},
          "telemetry": tele.summary(), "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": (f"llama-bench -p {TURN} -n {DECODE} -d {depth}: the prompt after {depth} tokens it filled untimed "
                     f"(its own random ids, fixed seed), {len(samples)} repetitions after its warm-up, untraced"),
          "engine": {"timing_window": pf.timing_window(TURN, depth) if qualified is not None else None, "build": row.get("build_commit"), "build_number": row.get("build_number"), "flash_attn": row.get("flash_attn"), "n_ubatch": ub, "n_batch": b,
                     "type_k": row.get("type_k"), "type_v": row.get("type_v"), "command": cmd}}


def _fork_session(run, model, depth, target, root, mode, extra:list[str]) -> dict[str, Any]:
  root, python = pf._fork(root)
  argv = ["--model", model, "--tokens", str((run / TOKENS).absolute()), "--depth", str(depth), "--turn", str(TURN),
          "--samples", str(pf.REPS), "--decode", str(DECODE), "--check", str(CHECK), "--mode", mode, *extra]
  return pf.fork_exec(root, python, target, DRIVER, argv, mode, "session")


def tinygrad_turn(run:pathlib.Path, model:str, depth:int, target, root, warm:bool) -> dict[str, Any]:
  with gpu_telemetry.run_telemetry(run, 0) as tele:
    got = _fork_session(run, model, depth, target, root, "truth", ["--warm"] if warm else [])
  computed = got["computed"]
  return {"ms": got["wall_s"] * 1e3, "samples_ms": [s * 1e3 for s in got["samples_s"]], "computed": computed,
          "chunk": pf.chunk_record(got["chunks"], "the fork's forward calls during the turn"),
          "decode": {**got["decode"], "words": f"the fork's generate() after its last turn, from depth "
                                               f"{got['decode']['from']}, the first {got['decode']['warm_left_out']} tokens left out"},
          "check": got["check"], "warm": got.get("warm"), "telemetry": tele.summary(),
          "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": (f"the fork's generate() with {TURN} new tokens appended to what its cache held (a recurrent state "
                     f"cannot go back, so the turns follow each other, at depths {', '.join(map(str, got['turn_depths']))} after "
                     f"the {depth}-token fill and the check): from its first forward call to the first token, "
                     f"{len(computed)} turns after one compiled and one captured, untraced"),
          "turn_depths": got["turn_depths"],
          "engine": {"fill_ms": got["fill_s"] * 1e3, "fill_chunks": len(got["fill_chunks"]),
                     "compile_ms": got["compile_s"] * 1e3, "capture_ms": (got["capture_s"] or 0) * 1e3,
                     "workload_reuse_default": got.get("workload_reuse_default")}}


def tinygrad_capture(run:pathlib.Path, model:str, length:int, target, root, depth:int) -> dict[str, Any]:
  raw = run / pf.RAW / f"{length}-at-{depth}"
  raw.mkdir(parents=True, exist_ok=True)
  got = _fork_session(run, model, depth, target, root, "capture", ["--out", str(raw / "launches.json")])
  return pf.fork_capture(raw, got)


ENGINES: dict[str, dict[str, Callable[..., dict[str, Any]]]] = {
  "llama.cpp": {"turn": llama_turn,
                "capture": lambda run, model, length, target, root, depth: pf.llama_capture(run, model, length, target, depth)},
  "tinygrad": {"turn": tinygrad_turn, "capture": tinygrad_capture},
}


def _engine(provider:str) -> dict[str, Callable[..., dict[str, Any]]]:
  if provider not in ENGINES:
    raise RuntimeError(f"a session is measured for {', '.join(ENGINES)}; {provider} has no session adapter")
  return ENGINES[provider]


# --- step 4 and step 5 -----------------------------------------------------------------------------------------

def focus_of(depths:list[int]) -> int:
  return min(depths, key=lambda d: (abs(d - FOCUS), d))


FIRST, STEP = 1024, 4  # derived depths: FIRST, then STEP times deeper, then the binding cap
RESERVE = 0.03         # the share of device memory a memory cap leaves free (buffers that grow with the context)


def model_cap(profile:dict[str, Any]) -> dict[str, Any] | None:
  """The context the model declares (profile.gguf.declared_context): its context_length, or a rope-scaling
  extension of it, with the key it was read from."""
  ctx = (profile.get("metadata") or {}).get("context") or {}
  return {"tokens": int(ctx["length"]), "key": ctx["source"]} if ctx.get("length") else None


def caps(profile:dict[str, Any], total_bytes:float, fixed_bytes:float, fixed_from:str, by_type:dict[str, float]) -> dict[str, Any]:
  """Per KV cache type: the memory cap (the device less RESERVE and the engine's fixed part, over that type's bytes
  per token), the model's cap, and which binds."""
  mc = model_cap(profile)
  out: dict[str, Any] = {"model": mc, "fixed_bytes": fixed_bytes, "fixed_from": fixed_from, "reserve": RESERVE, "types": {}}
  for name, b in by_type.items():
    mem = int((total_bytes * (1 - RESERVE) - fixed_bytes) // b)
    binds = "model" if mc and mc["tokens"] <= mem else "memory"
    out["types"][name] = {"memory": mem, "binding": min(mem, mc["tokens"]) if mc else mem, "binds": binds}
  return out


def deepest(cap:int) -> int:
  """The deepest turn whose tokens and the ones after it end at the cap."""
  return cap - TURN - max(DECODE, CHECK + 1)


def next_depth(done:list[int], cap:int) -> int | None:
  """The next derived depth: FIRST, then STEP times the last, then the cap's own (deepest), then none."""
  last = deepest(cap)
  d = FIRST if not done else done[-1] * STEP
  if done and done[-1] >= last:
    return None
  return d if d < last else last


def ladder(cap:int) -> list[int]:
  """Every derived depth next_depth gives under this cap, in order."""
  out: list[int] = []
  while (d := next_depth(out, cap)) is not None:
    out.append(d)
  return out


def fixed_part(rows:list[dict[str, Any]], per_token:float) -> float | None:
  """What the engine held beside its KV cache, the most over the depths measured (device memory in use less the
  depth's KV)."""
  got = [(r["telemetry"]["memory_mib"]["used_max"] * 2**20 - per_token * (r["context"] + TURN))
         for r in rows if ((r.get("telemetry") or {}).get("memory_mib") or {}).get("used_max")]
  return max(got) if got else None


CACHE_TYPES = ("f16", "q8_0")  # the KV cache types the context that fits is given for (formats in data/quants.json)


def cache_bytes(profile:dict[str, Any]) -> dict[str, float]:
  """KV bytes per token for each CACHE_TYPES entry (prefill_work.kv_bytes_per_token)."""
  from boltbeam.roofline.prefill_work import kv_bytes_per_token
  return {t: b for t in CACHE_TYPES if (b := kv_bytes_per_token(profile, t.upper()))}


def measure(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> pathlib.Path:
  """trace_request.json answered: one row per depth (the turn, decode, the check), the compaction and warm rows,
  the KV cache and the caps. A requested depth of 0 (--ctxs cap) is the derived depths: FIRST, then STEP times
  deeper, then the binding cap of this engine's cache, each next one decided after the last was measured."""
  from boltbeam.cli._common import CAP
  from boltbeam.collectors import providers
  from boltbeam.core.canonical import pretty_json
  from boltbeam.roofline.prefill_work import kv_bytes_per_token, recurrent_state_bytes
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest, read_json
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  model = str(manifest["model_path"])
  profile = read_json(run / "model_profile.json")
  asked = sorted(int(c) for c in read_json(run / "trace_request.json").get("contexts") or [])
  derive = CAP in asked
  element = providers.kv_element(provider)
  per_token = kv_bytes_per_token(profile, element_bytes=element[0])
  state = recurrent_state_bytes(profile) or 0.0
  weight_bytes = float((profile.get("metadata", {}).get("weights") or {}).get("total_bytes") or 0.0)
  by_type = cache_bytes(profile)
  with gpu_telemetry.run_telemetry(run, 0) as probe:
    pass
  total = ((probe.summary().get("memory_mib") or {}).get("total") or 0.0) * 2**20
  own = {"engine": per_token} if per_token else {}

  def cap_now(rows):
    fixed = fixed_part(rows, per_token) if per_token else None
    how = "measured: what the engine held beside its KV cache" if fixed else "before any depth: the weights and the recurrent state"
    return caps(profile, total, fixed if fixed else weight_bytes + state, how, {**by_type, **own})

  if not (per_token and total):
    raise RuntimeError("a session needs the KV bytes per token (the profile's attention shape) and the device memory")
  mc = model_cap(profile)
  upper = min(mc["tokens"] if mc else 1 << 30, cap_now([])["types"]["engine"]["memory"])
  ids = make_tokens(run, model, max(upper, max(asked)) + TURN * (pf.REPS + 4) + CHECK + 64)["ids"]
  # the focus is a depth that runs: the derived one nearest FOCUS when none was asked (8192 is not one)
  focus = focus_of([c for c in asked if c != CAP] or ladder(cap_now([])["types"]["engine"]["binding"]))
  rows: list[dict[str, Any]] = []
  plan = [c for c in asked if c != CAP]
  done: list[int] = []
  steps: list[int] = []  # the derived depths so far
  while True:
    if derive and not plan:
      nxt = next_depth(steps, cap_now(rows)["types"]["engine"]["binding"])
      if nxt is None:
        break
      steps.append(nxt)
      if nxt in done:
        continue
      plan.append(nxt)
    if not plan:
      break
    c = plan.pop(0)
    done.append(c)
    began = time.monotonic()
    base = {"scope": "whole_step", "workload": "session", "context": c, "turn_tokens": TURN, "kv_bytes": per_token * c}
    binding = cap_now(rows)["types"]["engine"]["binding"]
    if c > deepest(binding):
      rows.append({**base, "failed": f"beyond the binding cap: {binding:,} tokens for this engine's cache"})
      continue
    try:
      say(f"reference at depth {c}: llama.cpp server")
      r = reference(run, model, ids, c, warm=(c == focus and provider == "llama.cpp"))
      say(f"turn of {TURN} tokens at depth {c}: {provider}")
      got = _engine(provider)["turn"](run, model, c, target, root, c == focus)
    except Exception as exc:  # noqa: BLE001  a depth the engine cannot run is a row with its reason
      rows.append({**base, "failed": str(exc)[-600:]})
      continue
    check = (agree(r["cold"], got["check"]) if got.get("check") else
             {**r["self"], "kind": "llama.cpp against itself: the append path (depth cached, turn appended) against the "
                                   "same tokens cold, two runs"} if provider == "llama.cpp" else None)
    if check and "kind" not in check:
      check["kind"] = "against llama.cpp, an independent engine, on the same tokens"
    tele = got["telemetry"]
    rows.append({**base, "turn_ms": got["ms"], "wall_us": got["ms"] * 1e3, "tok_s": TURN / got["ms"] * 1e3,
                 "samples_ms": got["samples_ms"], "qualification_ms": got.get("qualification_ms"), "computed": got["computed"], "chunk": got["chunk"],
                 "batched": got["chunk"]["batched"], "decode": got["decode"], "decode_tok_s": got["decode"]["tok_s"],
                 "check": check, "reference_fill_s": r.get("cold_s"), "telemetry": tele, "throttled": tele.get("throttled"),
                 "telemetry_samples": got.get("telemetry_samples"), "source": got["source"], "engine": got["engine"],
                 **({"turn_depths": got["turn_depths"]} if got.get("turn_depths") else {}),
                 "seconds": round(time.monotonic() - began, 1)})
    if c == focus:
      w = got.get("warm") if provider != "llama.cpp" else (r.get("warm") and {"s": (r["warm"]["prompt_ms"] or 0) / 1e3,
                                                                                    "widths": [r["warm"]["prompt_n"]]})
      if w:
        rows.append({"scope": "warm", "workload": "session", "context": c, "turn_tokens": TURN, "ms": w["s"] * 1e3,
                     "computed": sum(x for x in w["widths"] if isinstance(x, int)),
                     "source": ("the same tokens sent again with the engine's prefix cache on: "
                                + ("llama.cpp server, its own prompt timing" if provider == "llama.cpp" else
                                   "the fork's generate() over what its cache held, first forward call to first token"))})
  say(f"compaction: a {SUMMARY}-token summary on an empty cache: {provider}")
  try:
    got = pf._engine(provider)["truth"](run, model, SUMMARY, target, root)
    rows.append({"scope": "compaction", "workload": "session", "context": 0, "prompt_tokens": SUMMARY, "ms": got["ms"],
                 "samples_ms": got["samples_ms"], "chunk": got["chunk"], "telemetry": got["telemetry"],
                 "throttled": got["telemetry"].get("throttled"), "source": got["source"]})
  except Exception as exc:  # noqa: BLE001
    rows.append({"scope": "compaction", "workload": "session", "context": 0, "prompt_tokens": SUMMARY, "failed": str(exc)[-600:]})
  measured = [r for r in rows if r["scope"] == "whole_step" and r.get("turn_ms")]
  final = cap_now(measured)
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": "session", "provider": provider, "provider_id": provider, "collector_id": "session",
           "timing_source": "untraced turn wall time at depth", "contexts": sorted(done), "derived": derive, "rows": rows,
           "focus": focus, "turn_tokens": TURN, "summary_tokens": SUMMARY,
           "tokens": json.loads((run / TOKENS).read_text()) | {"ids": None}, "agreement": AGREEMENT_WORDS,
           "kv": {"bytes_per_token": per_token, "element_bytes": element[0], "element": element[1],
                  "by_cache_type": by_type, "fixed_state_bytes": state or None,
                  "attention_layers": ((profile.get("metadata") or {}).get("attention") or {}).get("layer_count")
                                      or profile.get("layer_count"),
                  "caps": final, "device_bytes": total},
           "clock_policy": gpu_telemetry.run_clock_policy(run),
           "measured": ["whole_step.turn_ms (TURN tokens appended at depth C, untraced)", "whole_step.decode_tok_s",
                        "compaction.ms", "warm.ms"]}
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """One turn at the focus depth captured in the model and attributed: session_trace.json."""
  from boltbeam.workflow.common import read_json
  trace = read_json(run / "timing_trace.json")
  focus = int(trace["focus"])
  row = next((r for r in trace["rows"] if r["scope"] == "whole_step" and r["context"] == focus and r.get("turn_ms")), None)
  if row is None:
    raise RuntimeError(f"no turn measured at depth {focus}, so there is nothing to tie a capture out to")
  cap = _engine(provider)["capture"]
  return pf.role_time(run, provider, root=root, say=say, depth=focus, lengths=[TURN], truth_ms={TURN: row["turn_ms"]},
                      capture=lambda run, model, length, target, root: cap(run, model, length, target, root, focus),
                      trace=TRACE)
