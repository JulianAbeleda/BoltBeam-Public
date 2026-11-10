"""The matched whole-model A/B and the decision: the driver loads the model once and alternates default and plan arms
(runtime/tinygrad_decode_ab.py), the medians and noise read from the interleaved arms, and the existing evaluator's
CandidateDecision from them, or a firm refutation when the search's fastest kernel is the reference or the model's own.
"""
from __future__ import annotations

import json, os, pathlib, statistics, subprocess
from collections.abc import Mapping
from typing import Any

from boltbeam.artifacts.base import EvidenceFlags, EvidenceRow, EvidenceSource, NormalizedEvidence
from boltbeam.eval.evaluator import evaluate
from boltbeam.ledger.model import CandidateDecision, EvidenceRef
from boltbeam.manifest import Candidate
from boltbeam.search import compare_record, role_numbers, tinygrad_fork
from boltbeam.vocab import Verdict

AB_DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_decode_ab.py"


def run_ab(root:pathlib.Path, model:str, *, transforms:list[Mapping[str, Any]], shape:tuple[int, int],
           pairs:int = 3, timeout_s:float = 1800.0, env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """The driver loads the model once and alternates default and plan arms in that one process."""
  argv = [str(tinygrad_fork.fork_python(root)), str(AB_DRIVER), "--model", str(model), "--opts", json.dumps(list(transforms)),
          "--shape", f"{shape[0]},{shape[1]}", "--pairs", str(pairs)]
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", **(env or {})})
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError(f"decode A/B failed: {(proc.stderr or proc.stdout).strip()[-400:]}")
  return json.loads(lines[-1])


def matched_ab(raw:Mapping[str, Any], env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """Medians and noise from the interleaved arms. The plan counts as bound only when it reached decode calls:
  a plan applied to other kernels leaves both arms running the same decode."""
  b = [x for arm in raw["baseline"] for x in arm["tok_s_samples"]]
  c = [x for arm in raw["candidate"] for x in arm["tok_s_samples"]]
  if not c: c = list(b)
  bm, cm = statistics.median(b), statistics.median(c)
  first = raw["baseline"][0]["tokens"]
  binding = dict(raw["binding"])
  return {"timing_source": compare_record.TIMING_SOURCE, "context": raw["context"], "runtime_env": dict(env or {}),
          "pairs": raw["pairs"], "baseline_tok_s": bm, "candidate_tok_s": cm, "delta_pct": (cm - bm) / bm * 100.0,
          # noise is the wider of the two sides' ranges; overlapping ranges mean no measurable change
          "spread_pct": max(max(b) - min(b), max(c) - min(c)) / bm * 100.0,
          "ranges_overlap": min(c) <= max(b) and min(b) <= max(c),
          "baseline_samples": b, "candidate_samples": c,
          "token_match": all(arm["tokens"] == first for arm in raw["candidate"]),
          "deterministic": all(arm["tokens"] == first for arm in raw["baseline"]),
          "binding": binding, "route_bound": binding.get("calls_reached", 0) > 0 and binding.get("errors", 0) == 0,
          "load_s": raw.get("load_s"), "max_rss_bytes": raw.get("max_rss_bytes")}


def decide(route:Mapping[str, Any], winner:Mapping[str, Any], ab:Mapping[str, Any], *, model_id:str, target_id:str,
           ab_path:str, ab_sha:str) -> CandidateDecision:
  """The existing evaluator judges the A/B: correctness, binding, speed tier, rollback, target completeness."""
  role, quant = str(route["role"]), str(route["quant"])
  cid = role_numbers.plan_id(role, quant, winner)
  candidate = Candidate(candidate_id=cid, workload="decode", quant=(quant,), roles=(role,),
                        required_evidence_kinds=("wd_speed",), rollback={"kernel_plan": "default"},
                        route_family=(route.get("candidates") or [""])[0], origin="search",
                        description=f"{winner['plan']} for {role} {quant}")
  row = EvidenceRow(kind="wd_speed", metric="tok_s", value=float(ab["candidate_tok_s"]), unit="tok/s", role=role,
                    quant=quant, context=int(ab["context"]),
                    extra={"baseline_tok_s": float(ab["baseline_tok_s"]), "spread_pct": float(ab["spread_pct"]),
                           "timing_source": compare_record.TIMING_SOURCE})
  evidence = NormalizedEvidence(model_id=model_id, target_id=target_id, workload="decode",
                                source=EvidenceSource("tinygrad", "boltbeam/runtime/tinygrad_decode_ab.py", ab_path, f"sha256:{ab_sha}"),
                                rows=(row,), contexts=(int(ab["context"]),),
                                flags=EvidenceFlags(route_bound=bool(ab["route_bound"]), token_match=bool(ab["token_match"]),
                                                    deterministic=bool(ab["deterministic"])))
  return evaluate(candidate, [evidence], model_id=model_id, target_id=target_id, workload="decode")


def default_wins(route:Mapping[str, Any], winner:Mapping[str, Any], *, model_id:str, target_id:str,
                 search_path:str, search_sha:str, model_kernel:bool = False) -> CandidateDecision:
  """The search's fastest kernel is the reference or the model's own kernel: no plan to try, a firm refutation."""
  role, quant = str(route["role"]), str(route["quant"])
  which = "the model's own kernel" if model_kernel else "the search reference kernel"
  ref = EvidenceRef(evidence_id=search_sha, path=search_path, kind="boltbeam_kernel_check", fingerprint=f"sha256:{search_sha}",
                    claim=f"no reproduced plan beats {which} outside the chip band for {role} {quant}")
  return CandidateDecision(candidate_id=role_numbers.plan_id(role, quant, winner), model_id=model_id, target_id=target_id,
                           workload="decode", verdict=Verdict.REFUTE.value, evidence=(ref,),
                           reason=f"no searched plan beat {which} alone",
                           next_action="keep the default kernel")


def route_status(verdict:str) -> str:
  """route_policy.json status for a verdict: promote and refute are firm; anything else is not decided yet."""
  return {Verdict.PROMOTE.value: "promoted", Verdict.REFUTE.value: "refuted"}.get(verdict, "blocked")
