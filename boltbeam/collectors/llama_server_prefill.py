"""llama.cpp's prefill through llama-server, on real text: the path an MoE model's prefill is measured by.

llama-bench -p N feeds random token ids. An MoE routes random ids to far more experts than text does, so its prefill reads
~2x slow (Qwen3.8-Flash-Next Coder, 10k prompt, -ub 2048: 3,179-3,219 tok/s by llama-bench, 5,941 on real text in the same
server, 2,778-2,855 on random ids in that server: evidence in the 2026-11-11 prefill runs). So for an MoE this prefill
is taken from llama-server (runtime/llama_server_decode_drive.py --prefill): the shipped Gutenberg excerpts
(data/gate_corpus.json) cut to the prompt length, cold cache (cache_prompt off, prompt_n equal to the length, cache_n 0),
the server's own prompt_ms as the sample. Five timed requests after a warm-up, like llama-bench's five after its own.

Which path: BOLTBEAM_PREFILL_PROMPT = real | random | auto (the default). auto: real text for an MoE model when a
llama-server is found (beside llama-bench, $BOLTBEAM_LLAMA_SERVER, PATH); llama-bench for every dense model (its kernels do
not depend on the token values, so the tool, its recorded history and its tests are unchanged) and for an MoE model with no
server, which is then labelled in the report as random-token. real: refuses when there is no server. random: llama-bench.
The capture (nsys) runs the same driver, so the kernels attributed are the ones the real-text time was spent in.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
from typing import Any

from boltbeam.collectors import gpu_telemetry, llama_server_decode as lsd
from boltbeam.collectors import prefill as pf

ENV = "BOLTBEAM_PREFILL_PROMPT"
CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "gate_corpus.json"
NATIVE = "llama_server_prefill"
DEFAULT_BATCH, DEFAULT_UBATCH = 2048, 512  # llama-server's (and llama-bench's) defaults, pinned so the chunks are known
HEAD = 1  # the server reads one output token per prompt, at its end
STARTUP_HEADS = 1  # and (measured, 2026-11-11, llama.cpp master) runs one single-token decode as the first request starts
WORDS = ("real text: the shipped Gutenberg excerpts (data/gate_corpus.json) cut to the prompt length and sent to llama-server as "
         "token ids, the prompt cache off (prompt_n equal to the length, cache_n 0 on every request), the server's own prompt_ms")
CHAT_WORDS = ("real text sent as a chat request: the shipped Gutenberg excerpts (data/gate_corpus.json) as one user message to "
              "llama-server's /v1/chat/completions, cut so the chat template makes a prompt of the length, the prompt cache off "
              "(the same prompt_n on every request, cache_n 0), the server's own prompt_ms. The server breaks the first batch "
              "where the user message starts, as it does for any chat client")


CHECKPOINT_TAIL = 4  # the server ends a batch where 4 + n_ubatch and 4 tokens of the prompt remain (chunks)


def checkpoints(run:pathlib.Path) -> bool:
  """Whether the server stores context checkpoints for this model: memory it cannot roll back (a recurrent state; the
  profile's recurrent layers). llama.cpp's server-context.cpp (do_checkpoint) also checkpoints SWA models; no profile
  names a sliding window, so such a model reads as False and its chunk record is checked by the walk's layer count."""
  from boltbeam.collectors.pairing import layer_kinds
  p = run / "model_profile.json"
  return p.exists() and bool(layer_kinds(json.loads(p.read_text())).get("recurrent"))


def server_chunks(length:int, batch:int, ubatch:int, checkpoint:bool, head:int = 0) -> list[int]:
  """The server's chunks of a prompt. Like llama-bench it fills batches of n_batch tokens, each split into n_ubatch
  chunks; for a model it checkpoints (checkpoints) it also ends a batch where min(n_batch, 4 + n_ubatch) and min(n_batch,
  4) tokens remain, so the last tokens run apart and a checkpoint can be stored before them (llama.cpp server-context.cpp,
  "process the last few tokens of the prompt separately", PR 20288): 10,000 tokens at 2,048 / 2,048 are 2,048 x 3,
  1,808, 2,044 and 4, not 2,048 x 4 and 1,808. The capture's block kernels check it: the walk refuses a window whose
  layers do not group into these chunks. head: a chat prompt's template tokens before the user message (the driver's
  chat_head_tokens): the server also ends the first batch there, where the user message starts."""
  if not checkpoint or batch <= 0 or ubatch <= 0:
    return pf.llama_chunks(length, batch, ubatch)
  cuts = {length - min(batch, n) for n in (CHECKPOINT_TAIL + ubatch, CHECKPOINT_TAIL)} | ({head} if 0 < head < length else set())
  out, pos = [], 0
  while pos < length:
    start = pos
    while pos < length and pos - start < batch:
      pos += 1
      if pos in cuts:
        break
    out += pf.split_chunks(pos - start, ubatch)
  return out


def mode() -> str:
  m = os.environ.get(ENV, "auto").lower()
  if m not in ("auto", "real", "random"):
    raise ValueError(f"{ENV} is real, random or auto, not {m!r}")
  return m


def server_for(bench:str | None) -> str | None:
  return lsd.find(bench)


def use_server(run:pathlib.Path, bench:str | None) -> str | None:
  """The llama-server to measure this run's prefill with, or None to use llama-bench."""
  m, chat = mode(), pf.prompt_form() == "chat"
  if chat and m == "random":
    raise ValueError(f"{pf.FORM_ENV}=chat sends text, and {ENV}=random asks for llama-bench's random token ids: set one of them otherwise")
  if m == "random" or (m == "auto" and not chat and not pf.moe_model(run)):
    return None
  server = server_for(bench)
  if server is None and (m == "real" or chat):
    from boltbeam.collectors.llama_bench_decode import CannotMeasure
    raise CannotMeasure(f"{ENV}=real (or {pf.FORM_ENV}=chat) needs llama-server, which was not found beside llama-bench, in ${lsd.ENV} or on PATH",
                        f"export {lsd.ENV}=/path/to/llama-server")
  return server


def flags(run:pathlib.Path, length:int, server:str, *, capture:bool = False) -> list[str]:
  """The run's settings as llama-server flags, every layer on the GPU, batch and ubatch always given (so the chunks are known
  and the capture runs the untraced run's), no start-up warm-up (its decode would sit in a capture's launches)."""
  got = lsd._flags(run, length, server, capture=capture)
  return [*got, *(["-b", str(DEFAULT_BATCH)] if "-b" not in got else []), *(["-ub", str(DEFAULT_UBATCH)] if "-ub" not in got else []),
          "--no-warmup"]


def argv(server:str, model:str, length:int, flag_list:list[str], reps:int, log:str) -> list[str]:
  return [lsd.sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", "0",
          "--ctx-size", str(length + lsd.HEADROOM), "--contexts", str(length), "--reps", str(reps), "--flags", json.dumps(flag_list),
          "--log", log, "--prefill", "--corpus", str(CORPUS), "--prompt-form", pf.prompt_form()]


def row_of(driver:dict[str, Any], length:int) -> dict[str, Any]:
  """The driver's line as llama-bench's row, which the qualification and the records read."""
  return {"n_prompt": length, "n_gen": 0, "n_depth": 0, "n_batch": driver["n_batch"], "n_ubatch": driver["n_ubatch"],
          "samples_ns": [ms * 1e6 for ms in driver["samples_ms"]], "avg_ns": statistics.fmean(driver["samples_ms"]) * 1e6,
          "build_commit": (m.group(1) if (m := re.search(r"-([0-9a-f]{7,})$", str((driver.get("engine") or {}).get("build") or ""))) else None)}


def truth(run:pathlib.Path, model:str, length:int, server:str, *, device:int = 0) -> dict[str, Any]:
  from boltbeam.collectors import engine_common as ec
  from boltbeam.target.gpu import one_gpu
  pin, env = one_gpu(run, "llama.cpp")
  folder = run.absolute() / pf.NATIVE_RAW
  folder.mkdir(parents=True, exist_ok=True)
  cmd = argv(server, model, length, [*pin, *flags(run, length, server)], pf.LLAMA_REPS, str((folder / f"{length}.server.log").absolute()))
  with gpu_telemetry.run_telemetry(run, device) as tele:
    lines = ec.run_driver(cmd, env=env, log=folder / f"{length}.driver.log", cwd=run)
  driver = next(l for l in lines if l.get("prefill"))
  (folder / f"{length}.command.json").write_text(json.dumps(cmd))
  (folder / f"{length}.stdout").write_text(json.dumps(driver) + "\n")
  form = driver.get("prompt_form") or "ids"
  words = CHAT_WORDS if form == "chat" else WORDS
  return pf.llama_truth_record(row_of(driver, length), length, cmd, tele,
                               f"llama-server prompt_ms, a cold {length}-token prompt of real text{' sent as a chat request' if form == 'chat' else ''}, "
                               f"{len(driver['samples_ms'])} requests after a warm-up, untraced",
                               words + f" (corpus repeated {driver['corpus']['repeats']}x)" if driver["corpus"]["repeats"] > 1 else words,
                               "real_text", pf.moe_model(run), tool="llama-server", binary=server,
                               engine={"n_ctx": length + lsd.HEADROOM, "n_ctx_source": "llama-server -c: the prompt and a small headroom",
                                       "server": True, "corpus": driver["corpus"], "prompt_form": form,
                                       "prompt_tokens": driver.get("prompt_tokens") or length,
                                       **pf.outside_timer(driver["samples_ms"], driver.get("wall_samples_ms") or [])})


def capture(run:pathlib.Path, model:str, length:int, target, server:str) -> dict[str, Any]:
  """llama_capture's attribution for the server's prefill: the driver under the vendor tool, five timed requests after a
  warm-up, the steady median repetition's kernels. The server reads one output token per prompt (HEAD)."""
  from boltbeam.collectors import vendor_capture
  from boltbeam.collectors.engine_common import json_lines
  from boltbeam.collectors.prefill_attribution import REPEAT, UnnamedCapture, measured_window
  from boltbeam.target.gpu import one_gpu, run_gpu
  raw = run.absolute() / pf.RAW / str(length)
  pin, env = one_gpu(run, "llama.cpp")
  cmd = argv(server, model, length, [*pin, *flags(run, length, server, capture=True)], pf.LLAMA_REPS, str((raw / "llama-server.log").absolute()))
  launches = vendor_capture.capture(target.backend, cmd, raw, env=env, device=run_gpu(run), flush_ms=lsd.FLUSH_MS)
  driver = next(l for l in json_lines(vendor_capture.program_output(raw)) if l.get("prefill"))
  row = row_of(driver, length)
  selected = pf.qualification_index(row, length)
  if selected is None:
    raise ValueError("matched steady capture needs five complete positive host elapsed samples")
  qualified, ub, b = pf.qualification_sample(row, length), row["n_ubatch"], row["n_batch"]
  heads = [(pf.LLAMA_REPS + 1) * HEAD, (pf.LLAMA_REPS + 1) * HEAD + STARTUP_HEADS]
  try:
    measured, _ = measured_window(launches, REPEAT, repetition_index=selected, outputs=HEAD, head_count_bounds=heads)
  except UnnamedCapture as exc:
    raise UnnamedCapture(str(exc), {"method": vendor_capture.plan(target.backend)["method"], "token_ms": qualified, "prefills": pf.COPIES,
                                    "busy_ms": sum(l["wall_us"] for l in launches) / 1e3 / pf.COPIES,
                                    "timing_window": pf.timing_window(length)} if qualified else None) from exc
  chunks = server_chunks(int(driver.get("prompt_tokens") or length), b, ub, checkpoints(run), int(driver.get("chat_head_tokens") or 0))
  ends = [0] * (len(chunks) - 1) + [1]
  rec = pf.llama_capture_record(run, target, server, row, launches, measured, raw, cmd, env, length, chunks, HEAD, selected, qualified,
                                head_bounds=heads, prune=pf.llama_prunes(measured, chunks, HEAD, copies=1, ends=ends))
  if budget := request_budget(raw, driver, selected, server, target.backend):
    rec["time_budget"] = budget
  return rec


def request_budget(raw:pathlib.Path, driver:dict[str, Any], index:int, server:str, backend:str) -> dict[str, Any] | None:
  """The time budget of the capture's measured request (collectors/time_budget.py), written beside the capture; None
  when the capture has no sqlite export or the driver no request spans. Kernel families: the fork's rows, then llama.cpp's."""
  import sqlite3
  from boltbeam.collectors import time_budget
  from boltbeam.collectors.llama_bench_decode import fork_of
  from boltbeam.target.vendors import NoVendorRow, for_backend
  db = raw / "capture.sqlite"
  if not db.is_file():
    return None
  try:
    vendor = for_backend(backend).vendor
  except NoVendorRow:
    vendor = None
  fork = (fork_of(server) or {}).get("fork")
  try:
    got = time_budget.request_budget(db, driver, index, engine=[*([fork] if fork else []), lsd.PROVIDER], vendor=vendor,
                                     what=f"one cold {driver.get('prompt_tokens')}-token prefill, {driver.get('prompt_form') or 'ids'} form, llama-server")
  except (sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as exc:  # the budget is a disclosure: never fails the capture
    (raw / "time_budget.json").write_text(json.dumps({"refused": "The time budget could not be read from this capture's trace.",
                                                      "detail": f"{type(exc).__name__}: {exc}"}, indent=1))  # the raw text: provenance
    return None
  if got:
    (raw / "time_budget.json").write_text(json.dumps(got, indent=1))
  return got
