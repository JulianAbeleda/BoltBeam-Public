"""Flash decode: one fusion over a layer's attention nodes. BubbleBeam proposes flash tiles for the model's attention
shape from the chip's facts (search/flash_space.py), FutureSight applies the schema's legality and orders them, the
provider (--live-flash) compiles, checks and times each, BoltBeam rebuilds, checks and times both launches of each
(role_check.Checker.flash), and the verdict takes two bars: the geometry the model's decode installs, timed the same
way, and the in-model attention time. One fusions.json row.
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.plan.resolved_target import candidate_target, resolved_target_document
from boltbeam.search import compare_record, provider_check, role_check, tinygrad_fork

FLASH_STEM = "flash"


def in_model_attention(run:pathlib.Path) -> dict[str, Any] | None:
  """The attention kernels' time per token in the model (the tie-out's attention row: tinygrad's role-time rows of
  kind attention), the context it was captured at, and the KV read at that context (the limit's part for it)."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.workflow.common import read_json
  from boltbeam.workflow.tie_out import kv_ms
  path = run / tinygrad_role_time.TRACE
  if not path.is_file(): return None
  trace = read_json(path)
  whole = next((r for r in trace.get("rows") or [] if r.get("scope") == "whole_step"), None)
  if not whole or not whole.get("decode_tokens"): return None
  tokens = float(whole["decode_tokens"])
  ms = sum(float(r.get("wall_us") or 0.0) for r in trace.get("rows") or [] if r.get("scope") == "kernel" and r.get("kind") == flash_space_kind()) / tokens / 1000.0
  context = int((trace.get("contexts") or [whole.get("context") or 0])[0] or 0)
  profile = read_json(run / "model_profile.json") if (run / "model_profile.json").is_file() else {}
  limit = kv_ms(profile, context, tinygrad_role_time.KV_ELEMENT[0], float(trace.get("peak_gbs") or 0.0))
  return {"ms": ms, "context": context, "limit_ms": limit, "lost_ms": ms - limit}


def flash_space_kind() -> str:
  from boltbeam.search import flash_space
  return flash_space.ATTENTION


def compare_flash(run:pathlib.Path, *, root:pathlib.Path, device:str, checker:Any, band:float, say:Callable[[str], None],
                  ask:Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
                  session:Callable[[], Any] | None = None, profile:Mapping[str, Any] | None = None,
                  in_model:Mapping[str, Any] | None = None) -> dict[str, Any]:
  """BubbleBeam proposes flash tiles for the model's attention shape from the chip's facts, FutureSight applies the
  schema's legality and orders them, the provider (--live-flash) compiles, checks and times each, BoltBeam rebuilds,
  checks and times both launches of each, and the verdict takes two bars: the geometry the model's decode installs,
  timed the same way, and the in-model attention time. Returns one fusions.json row."""
  from boltbeam.search import flash_space
  from boltbeam.workflow.common import read_json
  folder = run / compare_record.FOLDER
  profile = profile if profile is not None else (read_json(run / "model_profile.json") if (run / "model_profile.json").is_file() else {})
  shape = flash_space.attention_shape(profile)
  row: dict[str, Any] = {"name": "+".join(flash_space.COVERS), "family": "flash_decode_live_split", "covers": list(flash_space.COVERS),
                         "quant": "fp16 KV", "stem": FLASH_STEM, "note": flash_space.LIMIT_NOTE, "rejected_rows": []}
  if shape is None:
    return row | {"verdict": "rejected", "reason": "rejected by BubbleBeam: the model profile names no attention shape"}
  attn = in_model if in_model is not None else in_model_attention(run)
  if not attn or not attn.get("context"):
    return row | {"verdict": "not_searched", "reason": "the run has no in-model attention time to judge against"}
  from boltbeam.roofline.prefill_work import attention_layer_count
  layers = attention_layer_count(profile)
  if not layers:
    return row | {"verdict": "not_searched", "reason": "the profile has no attended layer count"}
  row.update(shape=shape, instance={"calls_per_token": layers, "shape": shape, "context": attn["context"]},
             in_model_ms=attn["ms"], lost_ms=attn["lost_ms"],
             in_model_rows=[{"row": flash_space.ATTENTION, "ms": attn["ms"], "lost_ms": attn["lost_ms"], "calls_per_token": layers}])
  ask = ask or (lambda action, payload: tinygrad_fork.ask(root, action, {"device": device, **payload}))
  schema = {"schema_version": flash_space.FLASH_DECODE_CANDIDATE_SCHEMA_VERSION}
  described = ask("describe", {"candidate": schema, "attention": shape})
  installed = described.get("installed") or {}
  if installed.get("refused"):
    return row | {"verdict": "rejected", "reason": f"rejected by the kernel's own validate: {installed['refused']}"}
  if installed.get("model_bound") is not True:
    return row | {"verdict": "not_searched", "reason": "not decided: provider describes a candidate route, not the model-bound flash kernel"}

  facts = dict(described.get("target") or {})
  target = candidate_target(resolved_target_document(str(load_target_id(run)), facts)) if facts.get("backend") else None
  if target is None:
    return row | {"verdict": "not_searched", "reason": "the provider's flash describe reported no target facts"}
  maxc = flash_space.max_context(attn["context"])
  proposal = flash_space.propose(shape, attn["context"], target, installed)
  evidence = flash_space.assess(proposal["candidates"], facts)
  rejected = {r["candidate_hash"]: r["reason"] for r in evidence["rejections"]}
  order = [a["candidate_hash"] for a in evidence["assessments"]]
  by_hash = {c.candidate_hash: c for c in proposal["candidates"]}
  row.update(proposed_rows=len(proposal["candidates"]), rejected_rows=[{"row": r["tile"], "reason": r["reason"]} for r in proposal["refused"]],
             futuresight_rejected=[{"candidate_hash": h, "reason": why} for h, why in rejected.items()],
             installed=dict(installed), installed_hash=proposal["installed_hash"])
  compare_record.write(folder / f"{FLASH_STEM}-search-request.json", {"shape": shape, "context": attn["context"], "max_context": maxc,
                                                         "installed": installed, "facts": facts,
                                                         "candidates": [c.to_dict() | {"candidate_hash": c.candidate_hash} for c in proposal["candidates"]],
                                                         "refused": proposal["refused"], "futuresight_evidence": evidence})
  say(f"flash {shape['Hq']}x{shape['Hkv']}x{shape['Hd']}: {len(order)} of {len(proposal['candidates'])} pass FutureSight")
  measured, refused = [], []
  open_session = session or (lambda: _flash_session(root))
  with open_session() as worker:
    for h in order:
      doc = by_hash[h].to_dict() | {"candidate_hash": h}
      payload = {"candidate": doc, "candidate_hash": h, "execution": {"context_tokens": attn["context"], "samples": 7, "warmups": 0}}
      stages = {}
      for action in ("compile", "check", "measure"):
        got = worker.invoke(action, payload, 300.0)
        if "result" not in got:
          refused.append({"candidate_hash": h, "reason": got.get("blocked_reason"), "stage": action}); break
        stages[action] = got["result"]
      else:
        if not stages["check"].get("correct"):
          refused.append({"candidate_hash": h, "reason": "the provider's own check failed", "stage": "check"}); continue
        summary = stages["measure"].get("summary_ns") or {}
        measured.append({"candidate_hash": h, "tile": doc["descriptor"]["tile"], "provider_median_ns": summary.get("median"),
                         "kernels": stages["compile"].get("kernels") or []})
  compare_record.write(folder / f"{FLASH_STEM}-search-result.json", {"measured": measured, "refused": refused})
  row["search"] = {"counts": {"total": len(proposal["candidates"]), "measured_correct": len(measured)}, "refused": refused}
  row["search_result"] = f"{compare_record.FOLDER}/{FLASH_STEM}-search-result.json"
  if not measured:
    return row | {"verdict": "not_reproduced", "reason": "the provider measured no flash candidate"}
  say(f"flash: check {len(measured)}")
  check = {"floor_us": checker.floor_us, "band": band, "candidates": []}
  for m in measured:
    plan = flash_plan(m["tile"])
    mine = checker.flash(m["kernels"], shape=shape, maxc=maxc, context=attn["context"], label=plan)
    check["candidates"].append({"candidate_hash": m["candidate_hash"], "plan": plan, "boltbeam": mine,
                                **provider_check.judge(m["provider_median_ns"], mine, band)})
  compare_record.write(folder / f"{FLASH_STEM}-check.json", check)
  row["check_result"] = f"{compare_record.FOLDER}/{FLASH_STEM}-check.json"
  good = [c for c in check["candidates"] if c.get("verdict") == provider_check.REPRODUCED]
  mine_installed = next((c for c in good if c["candidate_hash"] == proposal["installed_hash"]), None)
  best = role_check.pick_winner(check["candidates"], mine_installed, band)
  if best is None or best.get("verdict") != provider_check.REPRODUCED:
    why = "; ".join(sorted({str(c.get("reason")) for c in check["candidates"]}))
    return row | {"verdict": "not_reproduced", "reason": f"{provider_check.NOT_REPRODUCED}: {why}"}
  mine_installed = next((c for c in good if c["candidate_hash"] == proposal["installed_hash"]), None)
  fused_ms = best["boltbeam_us_less_floor"] * layers / 1000.0
  row["fused"] = {"plan": best["plan"], "candidate_hash": best["candidate_hash"], "us_less_floor": best["boltbeam_us_less_floor"],
                  "ms_per_token": fused_ms, "correctness": (best.get("boltbeam") or {}).get("correctness"),
                  "launches": (best.get("boltbeam") or {}).get("launches"), "provider_note": best.get("provider_note")}
  if mine_installed is None:
    return row | {"verdict": "not_reproduced", "reason": "BoltBeam did not reproduce the geometry the model installs, so there is no like-for-like bar"}
  inst_ms = mine_installed["boltbeam_us_less_floor"] * layers / 1000.0
  row["unfused"] = {"nodes": [], "ms_per_token": None, "installed_ms": inst_ms, "installed_plan": mine_installed["plan"],
                    "why": "the bar timed alone is the model's installed flash geometry, not unfused attention"}
  if best["candidate_hash"] == proposal["installed_hash"] or fused_ms >= inst_ms * (1.0 - band):
    return row | {"verdict": "none_faster", "decided_by": "kernel alone",
                  "reason": (f"decided by the kernel alone: BoltBeam's fastest flash geometry ({best['plan']}) takes {fused_ms:.3f} ms per token "
                             f"against the installed geometry's {inst_ms:.3f} ms, timed the same way (the chip's band is ±{100 * band:.1f}%)")}
  if fused_ms >= attn["ms"] * (1.0 - band):
    return row | {"verdict": "none_faster", "decided_by": "kernel alone",
                  "reason": f"decided by the kernel alone: {fused_ms:.3f} ms per token is not under the model's {attn['ms']:.3f} ms of attention"}
  return row | {"verdict": "found_not_applied", "decided_by": "binding",
                "reason": (f"decided by binding: {best['plan']} takes {fused_ms:.3f} ms per token against the installed geometry's {inst_ms:.3f} ms "
                           f"and the model's {attn['ms']:.3f} ms; the whole-model A/B does not install a flash geometry yet")}


def flash_plan(tile:Mapping[str, Any]) -> str:
  return (f"flash: split {tile['split_count']}, lane {tile['lane_width']}, query group {tile['query_group_size'] or 1}, "
          f"stage {tile['stage_width']}, {tile['reduce_structure']} reduce")


def load_target_id(run:pathlib.Path) -> str:
  from boltbeam.workflow.common import load_manifest
  return str(load_manifest(run).get("target_id"))


def _flash_session(root:pathlib.Path):
  from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession
  # the provider reads flash candidates with BoltBeam's own schema module, from this checkout
  return PersistentJSONLSession((str(tinygrad_fork.fork_python(root)), "-m", "extra.llm_research.search_provider", "--live-flash"), cwd=str(root),
                                env={"BOLTBEAM_ROOT": str(pathlib.Path(__file__).resolve().parents[2])})
