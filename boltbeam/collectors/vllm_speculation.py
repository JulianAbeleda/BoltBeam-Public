"""Speculative decoding (vLLM MTP) as captured steps: the one place that says what a speculative decode step did.

One engine step of MTP with k draft tokens, one stream (vLLM 0.31, model runner v2, async scheduling, read from a run):
  verify   the target model over k + 1 query tokens (the last emitted token and the k drafts), its lm_head over all k + 1
           rows, and rejection sampling (greedy: a draft is accepted while it equals the target's own choice);
  draft    the draft head (one MTP layer, its own weights) over the same k + 1 positions, then the lm_head over the last
           one (draft 1); k - 1 more passes of one token each, each with the lm_head (drafts 2..k).
The step emits a = accepted + 1 tokens (accepted drafts and the target's own next token), so a decode of N tokens takes
a variable number of steps. vLLM's scheduler says, per step, the tokens it scheduled and the drafts among them; the
engine says the tokens each step emitted; vLLM's spec-decode counters (SpecDecodingStats) say drafts and accepted
tokens. runtime/vllm_semantics.observe_speculation records all three per scheduler step, in both capture passes.

The ledger (ledger()) is the capture's own account, checked against itself: one scheduler step per engine step; step 0
the prefill; every decode step k + 1 query tokens; emitted = accepted + 1 where the engine reported the step; the
counters over the measured call equal the steps' sum; the emitted tokens sum to the call's tokens. A step the async
scheduler ran ahead of the request's end (scheduled, never reported back) runs on the GPU and emits nothing, and
whether it runs is a host race (a capture ran one, the same call under ncu did not, 2026-11-08): it is recorded and
said, never joined or attributed (accounted()), so the account is of the verify steps every pass ran.

Per token, a step's time and its operations' limits are divided by the tokens the captured steps emitted (the decode
tokens of the measured call), never by the steps: the time per token is (time per step) / (tokens per step), and so is
the limit, at the acceptance this capture measured. Two passes are comparable only when their ledgers are identical
(same steps, same acceptance at every step, same tokens): greedy speculation is lossless, but the graphless labelling
pass rounds differently and a near tie can change a draft's acceptance without changing the text. A token script
cannot force speculation (it holds one logits row per request; verification scores k + 1), so a divergence is refused,
never forced.
"""
from __future__ import annotations

from typing import Any


class SpeculationDiverge(ValueError):
  """The two capture passes took different speculative steps: labels cannot move from one to the other."""


def _request(record:dict[str, Any], where:str) -> dict[str, Any]:
  requests = record.get("requests") or {}
  if len(requests) != 1:
    raise ValueError(f"{where}: speculative capture is one stream; the scheduler step holds {len(requests)} requests")
  return next(iter(requests.values()))


def ledger(engine:dict[str, Any]) -> list[dict[str, Any]]:
  """The measured call's engine steps, from one pass's record (engine['speculation']): [{step, query_tokens,
  draft_tokens, emitted, accepted, reported, context}], step 0 the prefill. context is the tokens in the cache when the
  step starts: the prompt and every emitted token but the last, which is the step's first query (so its first query
  attends context + 1 keys, as a non-speculative decode step at context + step - 1 does)."""
  spec = engine.get("speculation") or {}
  steps, config = spec.get("steps"), spec.get("config") or {}
  k = config.get("num_speculative_tokens")
  if config.get("method") != "mtp" or type(k) is not int or k < 1:
    raise ValueError(f"speculative capture names no MTP draft count: {config!r}")
  if not isinstance(steps, list) or not steps:
    raise ValueError("speculative capture recorded no scheduler steps")
  context, tokens = engine.get("context"), engine.get("tokens")
  if type(context) is not int or type(tokens) is not int or tokens < 2:
    raise ValueError("speculative capture needs its context and measured tokens")
  if engine.get("batch", 1) != 1:
    raise ValueError("speculative capture is one stream: the acceptance of B streams differs stream by stream")
  out, done, ahead, empty = [], 0, False, False
  for i, record in enumerate(steps):
    if record.get("schedule") != i or record.get("engine_step") != i:
      raise ValueError(f"scheduler step {i} was not scheduled in engine step {i} (one scheduler step per engine step)")
    if not record.get("requests") and not record.get("scheduled_tokens"):
      empty = True  # the scheduler found nothing to run (the request is done): no work, no tokens
      continue
    if empty:
      raise ValueError(f"scheduler step {i} has work after an empty step")
    row = _request(record, f"scheduler step {i}")
    query, drafts = row.get("tokens"), row.get("draft_tokens")
    if i == 0:
      if query != context or drafts != 0 or not record.get("reported") or row.get("emitted") != 1:
        raise ValueError(f"step 0 is not the {context}-token prefill emitting one token: {record!r}")
      out.append({"step": 0, "query_tokens": query, "draft_tokens": 0, "emitted": 1, "accepted": 0, "reported": True,
                  "context": 0, "phase": "prefill"})
      done = 1
      continue
    if drafts != k or query != k + 1:
      raise ValueError(f"decode step {i} scheduled {query} tokens with {drafts} drafts, not k + 1 = {k + 1} with {k}")
    if not record.get("reported"):
      ahead = True
      out.append({"step": i, "query_tokens": query, "draft_tokens": drafts, "emitted": 0, "accepted": 0,
                  "reported": False, "context": context + done - 1, "phase": "run_ahead"})
      continue
    if ahead:
      raise ValueError(f"decode step {i} was reported after a step the engine never reported: steps out of order")
    stats, emitted = record.get("spec_decoding_stats") or {}, row.get("emitted")
    accepted = stats.get("accepted_tokens")
    if stats.get("drafts") != 1 or stats.get("draft_tokens") != k or type(accepted) is not int:
      raise ValueError(f"decode step {i}: vLLM's counters are not one draft of {k} tokens: {stats!r}")
    # the step that reaches the call's token count keeps only what the request asked for: vLLM trims its output, its
    # counters still count every accepted draft (acceptance is theirs; the per-token account is of kept tokens)
    last = type(emitted) is int and 0 < emitted < accepted + 1 and done + emitted == tokens
    if emitted != accepted + 1 and not last:
      raise ValueError(f"decode step {i} emitted {emitted} tokens with {accepted} drafts accepted (expected accepted + 1, "
                       "or fewer only at the call's last token)")
    positions = stats.get("accepted_per_position") or []
    if len(positions) != k or positions != [1] * accepted + [0] * (k - accepted):
      raise ValueError(f"decode step {i}: accepted per draft position {positions} is not its {accepted} accepted drafts")
    out.append({"step": i, "query_tokens": query, "draft_tokens": drafts, "emitted": emitted, "accepted": accepted,
                "reported": True, "context": context + done - 1, "phase": "decode",
                **({"trimmed": accepted + 1 - emitted} if emitted != accepted + 1 else {})})
    done += emitted
  if done != tokens:
    raise ValueError(f"the captured steps emitted {done} tokens; the measured call generated {tokens}")
  counters = spec.get("counters") or {}
  decode = [s for s in out if s["phase"] == "decode"]
  sums = {"drafts": len(decode), "draft_tokens": k * len(decode), "accepted_tokens": sum(s["accepted"] for s in decode)}
  if any(counters.get(key) != value for key, value in sums.items()):
    raise ValueError(f"vLLM's counters over the measured call ({counters!r}) do not equal its steps' sum ({sums})")
  return out


def acceptance(steps:list[dict[str, Any]], k:int) -> dict[str, Any]:
  """What speculation did over the captured decode: drafts (verify steps the engine reported), the tokens they
  emitted, accept length (tokens per step, 1 + accepted / drafts: vLLM's and vllm_mtp.summarize's definition), the
  accepted share of draft tokens, the share accepted at each draft position, and the steps run ahead."""
  decode = [s for s in steps if s["phase"] == "decode"]
  if not decode:
    raise ValueError("speculative capture has no reported decode step")
  accepted = sum(s["accepted"] for s in decode)
  return {"k": k, "drafts": len(decode), "emitted": sum(s["emitted"] for s in decode), "accepted": accepted,
          "accept_length": 1 + accepted / len(decode), "accept_rate": accepted / (k * len(decode)),
          "accept_rate_per_position": [sum(s["accepted"] > i for s in decode) / len(decode) for i in range(k)],
          "run_ahead_steps": sum(s["phase"] == "run_ahead" for s in steps)}


def compare(timing:dict[str, Any], label:dict[str, Any]) -> list[dict[str, Any]]:
  """The timing pass's verify steps (prefill and decode), when the labelling pass's are the same step for step; else
  SpeculationDiverge, naming the first step that differs. Run-ahead steps are a host race and are not compared."""
  a, b = ([s for s in ledger(p) if s["phase"] != "run_ahead"] for p in (timing, label))
  ta, tb = ((timing.get("speculation") or {}).get("prompt") or {}), ((label.get("speculation") or {}).get("prompt") or {})
  if ta.get("context_sha256") != tb.get("context_sha256"):
    raise SpeculationDiverge("capture passes decoded from different contexts (context_sha256)")
  for x, y in zip(a, b):
    if x != y:
      raise SpeculationDiverge(f"capture passes took different speculative steps at step {x['step']}: timing {x}, "
                               f"labelling {y}; speculation cannot be forced to one pass's tokens, so the capture is refused")
  if len(a) != len(b):
    raise SpeculationDiverge(f"capture passes took {len(a)} and {len(b)} steps")
  return a


def same_as_reference(capture:dict[str, Any], reference:dict[str, Any]) -> list[str]:
  """Why the untraced reference did not run the captured speculation ([] when it did): the same MTP configuration,
  the same natural context (sha256), and in every repeat's long call the captured call's drafts and accepted tokens
  (with greedy decoding, the same steps); its short call (one token: the prefill) drafts nothing."""
  spec, ref = capture.get("speculation") or {}, reference.get("speculation") or {}
  why = []
  if spec.get("config") != ref.get("config"):
    why.append(f"speculative configuration {ref.get('config')!r}, captured {spec.get('config')!r}")
  if (spec.get("prompt") or {}).get("context_sha256") != (ref.get("prompt") or {}).get("context_sha256"):
    why.append("a different context (context_sha256)")
  counters = spec.get("counters") or {}
  for i, r in enumerate(ref.get("repeats") or []):
    long, short = r.get("long") or {}, r.get("short") or {}
    got = {key: long.get(key) for key in ("tokens", "drafts", "accepted_tokens")}
    want = {key: counters.get(key) for key in ("tokens", "drafts", "accepted_tokens")}
    if got != want:
      why.append(f"repeat {i + 1} ran {got}, the capture {want}")
    if short.get("drafts"):
      why.append(f"repeat {i + 1}'s one-token call drafted {short['drafts']} times")
  if not ref.get("repeats"):
    why.append("no spec-decode counters per repeat")
  return why


def words(summary:dict[str, Any]) -> str:
  """The measured speculation in one phrase, for every place a run is read."""
  per = ", ".join(f"{100 * p:.0f}%" for p in summary["accept_rate_per_position"])
  ahead = summary.get("run_ahead_steps") or 0
  return (f"MTP k={summary['k']}: {summary['drafts']} verify steps emitted {summary['emitted']} tokens, accept length "
          f"{summary['accept_length']:.2f} tokens per step ({100 * summary['accept_rate']:.0f}% of drafts accepted; "
          f"by position {per})" + (f"; {ahead} step{'s' if ahead != 1 else ''} the scheduler ran past the last token, "
                                   "no tokens" if ahead else ""))


def measured_words(run) -> list[str]:
  """The speculation a run measured, as the reader sees it in every output (summary, report, results, TUI): the
  capture's acceptance per context, else the untraced timing's, from vLLM's counters over each repeat's long call.
  [] for a run without speculation."""
  import json
  from pathlib import Path
  run, out = Path(run), []
  capture = run / "decode_capture.json"
  contexts = (json.loads(capture.read_text()).get("contexts") or {}) if capture.exists() else {}
  for context, c in sorted(contexts.items(), key=lambda kv: int(kv[0])):
    if (spec := c.get("speculation")) and spec.get("words"):
      prompt = (spec.get("prompt") or {}).get("prompt_id")
      out.append(f"{spec['words']} (captured at context {context}" + (f", natural text from prompt {prompt})" if prompt else ")"))
  if out:
    return out
  trace = run / "timing_trace.json"
  rows = (json.loads(trace.read_text()).get("rows") or []) if trace.exists() else []
  for r in rows:
    spec = (r.get("engine") or {}).get("speculation") or {}
    k = (spec.get("config") or {}).get("num_speculative_tokens")
    longs = [x.get("long") or {} for x in spec.get("repeats") or []]
    drafts, accepted = sum(x.get("drafts") or 0 for x in longs), sum(x.get("accepted_tokens") or 0 for x in longs)
    if k and drafts:
      out.append(f"MTP k={k}: accept length {1 + accepted / drafts:.2f} tokens per step, {100 * accepted / (k * drafts):.0f}% "
                 f"of drafts accepted (untraced at context {r.get('context')}, {len(longs)} repeats; vLLM's counters)")
  return out


def accounted(steps:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
  """The decode steps a per-token account holds, and what it leaves out. A step the async scheduler ran past the
  request's last token emits nothing, and whether it runs at all is a host race between the scheduler and the
  request's end (the capture's timing pass ran one, the same call under ncu did not), so no pass can be held to it:
  it is left out of the table and said. The untraced long-minus-short difference may hold such a step; the capture
  comparison says how far the two are apart."""
  ahead = sum(s['phase'] == 'run_ahead' for s in steps)
  decode = [s for s in steps if s['phase'] == 'decode']
  if not ahead:
    return decode, 'no step ran past the last token'
  return decode, (f"the scheduler ran {ahead} step{'s' if ahead != 1 else ''} past the last token (no tokens; whether it "
                  'runs is a host race, so it is not in the table)')


def horizon(identities:list[dict[str, Any]]) -> dict[str, Any]:
  """A speculative decode horizon (cuda_memory.decode_horizon's counterpart): steps 1..S in order, one stream, each
  the first's capture but for its own depth and query tokens; with the per-step ledger and the acceptance."""
  from boltbeam.collectors.cuda_memory import same_capture
  if not identities:
    raise ValueError("empty speculative decode horizon")
  first = identities[0]
  for offset, identity in enumerate(identities):
    row = identity.get("speculation_step") or {}
    if identity.get("decode_step") != offset + 1 or row.get("step") != offset + 1 or identity.get("batch") != 1:
      raise ValueError("speculative decode horizon requires contiguous steps of one stream")
    if identity.get("context") != row.get("context") or identity.get("length") != row.get("query_tokens"):
      raise ValueError(f"decode step {offset + 1}'s depth or query tokens differ from its ledger row")
    same_capture({**first, "decode_step": offset + 1, "context": identity["context"], "length": identity["length"]},
                 identity)
  rows = [i["speculation_step"] for i in identities]
  k = ((first.get("speculation") or {}).get("config") or {}).get("num_speculative_tokens")
  return {**first, "length": len(identities), "decode_steps": list(range(1, len(identities) + 1)),
          "speculative_steps": rows, "acceptance": acceptance(rows, k)}
