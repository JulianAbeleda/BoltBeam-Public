"""`boltbeam derive hot-draft DRAFT`: a copy of a mixture-of-experts draft (an MTP head) keeping only its hottest experts,
sized to a measured GPU memory budget, so the freed memory buys something else (the target's experts on the GPU).

  routing   the draft's own routing while it drafts: the engine's route count (--routes FILE, or captured: Measure MTP's
            real-text run with the engine's count switch on, which also gives the full draft's acceptance).
  ranking   each expert by its share of the routed weight, per pass (1-token decode passes and catch-up passes over a
            prompt), each normalised to its pass's total and summed; ties by id.
  size      N = the most experts the budget holds (budget / one expert's bytes), never more than were routed at all (an
            expert never routed frees memory at no loss) nor than the draft has. The budget is measured: --budget-mib, or
            a session-ab arm (--measured REC --arm A): its lowest free memory over the session less the run-to-run spread
            of that low (a margin; one run: unknown, refused), plus what the arm's own draft experts take.
  copy      the engine adapter writes the trimmed file and names the kept experts in it.
  record    boltbeam.hot_draft.v1 beside the copy: ids, coverage of the routed mass and hits, budget and its source, and
            the acceptance (accept length per depth) of the full and the trimmed draft where measured.
Engine-blind: an engine is a row of data/hot_draft.json and its adapter module."""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
from typing import Any

from boltbeam.core.refused import Refused

SCHEMA = "boltbeam.hot_draft.v1"
TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "hot_draft.json"
MiB = 1 << 20


def row(engine:str) -> dict[str, Any]:
  rows = json.loads(TABLE.read_text())["engines"]
  if engine not in rows:
    raise Refused(f"no hot-draft adapter for {engine}; the engines with one: {', '.join(sorted(rows))}")
  return rows[engine]


def adapter(engine:str):
  return importlib.import_module(row(engine)["adapter"])


def rank(c:dict[str, Any]) -> list[int]:
  """Experts hottest first: the summed per-pass shares of routed weight (a pass with no tokens adds nothing); ties by id."""
  score = [0.0] * c["n_expert"]
  for p, mass in c["mass"].items():
    tot = sum(mass)
    for e, m in enumerate(mass):
      score[e] += m / tot if tot else 0.0
  return sorted(range(c["n_expert"]), key=lambda e: (-score[e], e))


def routed(c:dict[str, Any]) -> int:
  return sum(1 for e in range(c["n_expert"]) if any(h[e] for h in c["hits"].values()))


def size(n_expert:int, n_routed:int, per_bytes:int, budget_bytes:float) -> int:
  return max(0, min(n_expert, n_routed, int(budget_bytes // per_bytes)))


def coverage(c:dict[str, Any], ids:list[int]) -> dict[str, float | None]:
  return {f"{p}_mass": sum(m[e] for e in ids) / sum(m) if sum(m) else None for p, m in c["mass"].items()}


def budget_from_arm(record:str, arm:str, ad, per_bytes:int, margin_mib:float | None = None) -> dict[str, Any]:
  """The draft's budget from a session-ab arm: lowest free over its valid runs (a sampled peak), less the margin (the
  spread of that low over the runs, or margin_mib), plus the experts its own draft holds (a trim of the same parent:
  per_bytes each; their count from the record's file_facts, else the file)."""
  from boltbeam.collectors.replay import flag_value
  d = json.loads(pathlib.Path(record).expanduser().read_text())
  if d.get("schema") != "boltbeam.session_ab.v1":
    raise Refused(f"{record} is not a session-ab record")
  a = next((x for x in d["inputs"]["arms"] if x["name"] == arm), None)
  if a is None:
    raise Refused(f"{record} has no arm {arm}; its arms: {', '.join(x['name'] for x in d['inputs']['arms'])}")
  mem = [(r.get("telemetry") or {}).get("memory_mib") or {} for r in d["arms"] if r["name"] == arm and r.get("valid")]
  free = [m["total"] - m["used_max"] for m in mem if m.get("total") and m.get("used_max")]
  if not free:
    raise Refused(f"arm {arm} has no valid run with its GPU memory read")
  spread = max(free) - min(free) if len(free) > 1 else None
  margin = margin_mib if margin_mib is not None else spread
  if margin is None:
    raise Refused(f"arm {arm} ran once: the run-to-run spread of its lowest free memory is unknown; repeat it (session-ab --repeats) or pass --margin-mib")
  draft = flag_value([str(x) for x in d["inputs"]["server_argv"] + (a.get("flags") or [])], ("-md", "--model-draft"))
  from boltbeam.workflow.spend import file_facts_of
  known = (file_facts_of(d).get(pathlib.Path(draft).name) or {}).get("experts") if draft else None
  held = (known if known is not None else ad.experts(draft)["n_expert"] if draft else 0) * per_bytes
  return {"bytes": (min(free) - margin) * MiB + held, "lowest_free_mib": min(free), "margin_mib": margin, "runs": len(free),
          "draft_held_mib": held / MiB,
          "source": f"{pathlib.Path(record).parent.name} {arm}: lowest free {min(free):,.0f} MiB less a {margin:,.0f} MiB margin "
                    f"({'given' if margin_mib is not None else f'the spread of the sampled peak over {len(free)} runs'}), plus {held / MiB:,.0f} MiB its own draft's experts hold"}


def derive(draft:str, *, engine:str, out:str | None = None, routes:str | None = None, main:str | None = None, k:int | None = None,
           budget_mib:float | None = None, measured:str | None = None, arm:str | None = None, margin_mib:float | None = None,
           build:str | None = None, backend:str = "CUDA", acceptance:bool = True) -> dict[str, Any]:
  """The trimmed draft and its record (written as <out>.json). A Refused says what is missing."""
  ad = adapter(engine)
  src = pathlib.Path(draft).expanduser()
  dst = pathlib.Path(out).expanduser() if out else src.with_name(f"{src.stem}-hot.gguf")
  if (budget_mib is None) == (measured is None):
    raise Refused("give the budget once: --budget-mib N (measured MiB the draft's experts may take), or --measured REC --arm A")
  if not routes and not (main and k):
    raise Refused("routing: pass --routes FILE (the engine's draft route count), or --main MODEL --k K to capture it")
  e = ad.experts(src)
  work = dst.with_name(dst.stem + "-work")
  full = None
  if not routes:
    full = ad.capture(main, str(src), k, out=work / "full", server=build, routes=True, backend=backend)
    routes = full["routes"]
  c = ad.counts(routes)
  if c["n_expert"] != e["n_expert"]:
    raise Refused(f"the route count has {c['n_expert']} experts, the draft {e['n_expert']}: a count of another draft file")
  b = {"bytes": budget_mib * MiB, "source": "--budget-mib"} if budget_mib is not None else budget_from_arm(measured, arm, ad, e["per_expert_bytes"], margin_mib)
  order, n_routed = rank(c), routed(c)
  n = size(e["n_expert"], n_routed, e["per_expert_bytes"], b["bytes"])
  if n < max(1, c.get("n_used") or 1):
    raise Refused(f"the budget ({b['bytes'] / MiB:.0f} MiB) holds {n} experts of {e['per_expert_bytes'] / MiB:.2f} MiB; the router picks {c.get('n_used')} a token")
  ids = sorted(order[:n])
  sha = hashlib.sha256(pathlib.Path(routes).read_bytes()).hexdigest()
  made = ad.trim(src, dst, ids, {"routes_sha256": sha})
  rec = {"schema": SCHEMA, "engine": engine, "draft": str(src), "copy": made["path"], "n": n, "n_expert": e["n_expert"], "n_routed": n_routed,
         "per_expert_mib": e["per_expert_bytes"] / MiB, "freed_mib": (e["n_expert"] - n) * e["per_expert_bytes"] / MiB,
         "budget": {**b, "mib": b["bytes"] / MiB}, "rule": "the most measured-hottest experts the budget holds, at most those routed at all",
         "ids": ids, "coverage": coverage(c, ids), "routes": {"path": str(routes), "sha256": sha, "tokens": c["tokens"]},
         "acceptance": {"full": full["acceptance"] if full else None, "hot": None, "k": k, "main": main}}
  if acceptance and main and k:
    rec["acceptance"]["hot"] = ad.capture(main, made["path"], k, out=work / "hot", server=build, routes=False, backend=backend)["acceptance"]
  pathlib.Path(made["path"] + ".json").write_text(json.dumps(rec, indent=1))
  return rec


def words(rec:dict[str, Any]) -> list[str]:
  from boltbeam.report.plain import words as all_words
  w = all_words()["hot_draft"]
  cov = rec["coverage"]
  out = [w["made"].format(n=rec["n"], of=rec["n_expert"], routed=rec["n_routed"], copy=rec["copy"], freed=rec["freed_mib"]),
         w["budget"].format(mib=rec["budget"]["mib"], source=rec["budget"]["source"], per=rec["per_expert_mib"]),
         w["coverage"].format(**{k: "?" if v is None else f"{v:.1%}" for k, v in cov.items()})]
  a = rec["acceptance"]
  if a["full"] and a["hot"]:
    out += [w["acceptance"].format(depth=d, full=a["full"][d], hot=a["hot"][d], pct=100 * (a["hot"][d] / a["full"][d] - 1)) for d in a["full"] if d in a["hot"]]
  else:
    out.append(w["acceptance_unmeasured"])
  out.append(w["next"].format(copy=rec["copy"]))
  return out
