"""search/identity: the identity contract a kernel (a kernel list entry), a session-ab arm or a searched kernel is held to.

    exact     the default: the device's results are the engine's own kernels' bit for bit (the region timer's identity
              read, stock against kernel, on the same buffers; role_check's, the model's own kernel against each
              candidate), and a session's replies are the base's sha256 for sha256. No tolerance: anything that differs
              is refused for timing, install, realization and a speed claim.
    lossy     declared, never inferred: identity "lossy" with a lossy block (kernel_manifest_v2 $defs/lossy, the one
              schema; a session-ab arm's --lossy file holds the same two fields). Judged here, not only shape-checked:
      evidence   a quality run's record (a file and its sha256, read and hashed here; $defs/lossy_evidence). It names
                 the kernel it ran (kernel_sha256: the kernel's sha256, a launches entry's sequence(), an
                 arm's build or kernel file); the gate checks that is the kernel judged.
      quality    the stated deltas (copy less original, at level over n) are the metrics of QUALITY, each once, and
                 the evidence's own: delta, its paired interval [lo, hi] and n are computed again here from the
                 evidence's rows (each with an id and a finish) and must agree. The evidence names its engine, which
                 must declare a change that must not matter (data/speed_up.json threshold.noise.engines).
                 Over at least threshold.lossy.min_items paired items (the owner's 100, 2026-11-21), each must pass
                 non-inferiority against a margin from the engine's numerics-only noise pair (math/noninferior.
                 noise_pair: the original again under its declared change, the same items): margin = |d0| + z(two-
                 sided) SE0, the noise pair's own upper bound; the copy's loss + z(one-sided) SE must be at or below
                 it. It replaced psi0 here (Speed up's gross-churn tolerance, kept there): simulated at a 3% loop
                 rate over 100 items, a true null fails psi0 ~60% and this margin ~14-22% (attn-quality-20261121);
                 the run-local A1 margin 1.96 sqrt(2 p (1 - p) q / n) fails one 37.6% of the time at any n.
      divergence the token-probability check (llama.cpp tools/perplexity's KL-divergence standard): the copy's and
                 the noise run's mean KL from the original and mean change of the reference token's probability,
                 teacher-forced on the same positions at the kernel's own shape, in blocks of positions; each held
                 to the same form, its line the noise run's own upper bound (math/noninferior.divergence), over at
                 least threshold.lossy.min_items blocks. The blocks are made here from the evidence's raw rows
                 (the original's, the copy's and the noise run's: pos, ref, top, prompt_n; math/noninferior.blocks),
                 never read as written. The rows are tied to the kernel's shape: the positions are exactly start +
                 stride x (1..positions) (depth and stride), the same reference tokens in every run, only positions
                 every run appended exactly stride tokens at (cache misses dropped); the window (candidate_hash) is
                 the one its envelope was measured on and, at the region timer, the one timed, and stride a batch
                 width that window runs at (its passes' columns; manifest_writer takes the candidates region_stage
                 gives it, else binds the entry's evidence.candidate_hash alone; session-ab has no window). The
                 scored depths and stride lie inside the declaration's range (depth [lo, hi], widths: where the
                 kernel is dispatched), on every path. Blocks of at least threshold.lossy.min_block positions,
                 rows of exactly n_probs (at least min_n_probs) top tokens, one token list (tokens: each run's sha256).
      range      the depths [lo, hi] and batch widths the kernel is declared lossy over; its divergence is scored inside.
                 The derivation is said in words on the result (judged).
      envelope   the largest difference the device read may show (abs or rel_max) and in words how it was derived. Its
                 ceiling is the quality run's own measured deviation, a reproducibility tie: the evidence's
                 envelope_measured, the region timer's read of this kernel against stock on that window
                 (candidate_hash; worst by mode). A wider declared tolerance admits device results the quality run
                 never produced, so it is refused; evidence with no such measurement is refused, saying what to
                 measure. No stock-against-stock elementwise noise source exists in BoltBeam (one could be measured:
                 the region timer on stock under the engine's declared change), so no number is picked.
              A declaration that cannot stand is refused, as if undeclared. KV types are the precedent: q8_0 unless proven.
Engine-blind: nothing here names an engine; the region timer, region_match.timed, manifest_writer, role_check and
session-ab read it."""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics
import struct
from collections.abc import Iterable, Mapping
from typing import Any

from boltbeam.core.canonical import sha256_file, sha256_hex


def sequence(shas:Iterable[str]) -> str:
  """One identity for a launches entry's kernels: the sha256 of their sha256s in launch order, one a line."""
  return sha256_hex("".join(f"{s}\n" for s in shas).encode())


EXACT, LOSSY = "exact", "lossy"
MODES = ("abs", "rel_max")  # the hook's tolerance_mode words (kernel_manifest_v2): abs |ext - stock|, rel_max that over max |stock|
EVIDENCE = "boltbeam.lossy_evidence.v1"   # kernel_manifest_v2 $defs/lossy_evidence
ARMS = ("original", "copy", "noise")       # the evidence's three paired runs (math/noninferior's)
QUALITY = {"pass_rate": ("correct", True), "loop_rate": ("loop", False)}   # each required metric: the evidence row's field, and whether True is the good value
_SHA = re.compile(r"[0-9a-f]{64}")
FINISH = ("stop", "length")   # a row's finish: a loop is detected (loop: true) and stops; length is truncation, judged as such
RESTATED = 1e-6   # a declared figure restates the evidence's: equal up to printing it with six figures, not a tolerance on the data
FORCED = ("stride", "start", "positions", "block", "n_probs")   # the divergence's shape: tokens an append, the prefix, positions, positions a block, top tokens a row
THRESHOLD = pathlib.Path(__file__).resolve().parents[1] / "data" / "speed_up.json"   # threshold: the non-inferiority rule's confidence, power, truncation


class NotIdentical(ValueError):
  """The device's results, or an arm's replies, are not the engine's own and nothing declares the difference: refused."""


def _int(x:Any) -> bool:
  return isinstance(x, int) and not isinstance(x, bool)


def _num(x:Any) -> bool:
  return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def malformed(lossy:Any) -> str | None:
  """What is wrong with a lossy block's shape (the schema's rules, said in words), or None."""
  if not isinstance(lossy, Mapping):
    return "no lossy block: a lossy declaration states its evidence, its quality deltas and its check envelope"
  ev, q, env = lossy.get("evidence"), lossy.get("quality"), lossy.get("envelope")
  if not (isinstance(ev, Mapping) and isinstance(ev.get("path"), str) and ev["path"] and isinstance(ev.get("sha256"), str) and len(ev["sha256"]) == 64):
    return "no evidence: a quality run's record (path and sha256) is required"
  if not (isinstance(q, list) and q):
    return "no quality deltas: at least one metric's delta with its interval is required"
  for r in q:
    if not (isinstance(r, Mapping) and isinstance(r.get("metric"), str) and r["metric"] and all(_num(r.get(k)) for k in ("delta", "lo", "hi", "level"))
            and isinstance(r.get("n"), int) and r["n"] >= 2 and r["lo"] <= r["delta"] <= r["hi"] and 0 < r["level"] < 1):
      return f"a quality delta is not a metric with delta inside [lo, hi] at a level in (0, 1) over n >= 2: {r!r}"
  if missing := [m for m in QUALITY if m not in {r["metric"] for r in q}]:
    return f"the quality deltas do not state {' and '.join(missing)}: every lossy declaration states {', '.join(QUALITY)}"
  if len({r["metric"] for r in q}) != len(q) or (other := {r["metric"] for r in q} - set(QUALITY)):
    return (f"the quality deltas state {'a metric twice' if len({r['metric'] for r in q}) != len(q) else 'a metric not judged: ' + ', '.join(sorted(other))}: "
            f"each of {', '.join(QUALITY)} once, and only those (a delta shown beside the judged ones must be judged too)")
  rg = lossy.get("range")
  if not (isinstance(rg, Mapping) and isinstance(rg.get("depth"), list) and len(rg["depth"]) == 2 and all(_int(x) and x >= 1 for x in rg["depth"])
          and rg["depth"][0] <= rg["depth"][1] and isinstance(rg.get("widths"), list) and rg["widths"] and all(_int(x) and x >= 1 for x in rg["widths"])):
    return ("no range: the depths [lo, hi] and batch widths the kernel is declared lossy over (where it is dispatched) are required; its "
            "divergence must be scored inside them")
  if not (isinstance(env, Mapping) and env.get("mode") in MODES and _num(env.get("tolerance")) and env["tolerance"] > 0
          and isinstance(env.get("derived"), str) and env["derived"].strip()):
    return f"no check envelope: mode ({' or '.join(MODES)}), a tolerance above 0 and how it was derived from measured noise are required"
  return None


def _rows(doc:Mapping[str, Any], metric:str) -> dict[str, list[dict[str, Any]]]:
  """The evidence's three runs as math/noninferior.paired rows for one metric (correct: the metric's good value)."""
  field, good = QUALITY[metric]
  arms = doc.get("arms")
  if not isinstance(arms, Mapping) or any(not isinstance(arms.get(a), list) for a in ARMS):
    raise NotIdentical(f"its evidence has no arms {', '.join(ARMS)}: each run's rows (id, correct, loop, finish) on the same items")
  out = {}
  for a in ARMS:
    if any(not isinstance(r, Mapping) or not isinstance(r.get(field), bool) or r.get("id") is None or r.get("finish") not in FINISH for r in arms[a]):
      raise NotIdentical(f"its evidence's {a} run has a row without an id, a true/false {field} ({metric}) or a finish of {' or '.join(FINISH)} "
                         "(truncation is judged by finish: a row without it would skip that rule)")
    out[a] = [{"id": r.get("id"), "correct": r[field] is good, "finish": r.get("finish")} for r in arms[a]]
  return out


def _forced(kl:Mapping[str, Any], doc:Mapping[str, Any], lossy:Mapping[str, Any], candidate:str | Mapping[str, Any] | None, th:Mapping[str, Any]) -> dict[str, Any]:
  """The divergence's blocks made here from the evidence's raw teacher-forced rows (math/noninferior.blocks), never
  read as written, and the rows tied to the kernel's shape and declared range (module docstring, divergence); raises
  NotIdentical."""
  from boltbeam.core.refused import Refused
  from boltbeam.math import noninferior as ni
  lz = th["lossy"]
  if not all(_int(kl.get(f)) and kl[f] >= 1 for f in FORCED):
    raise NotIdentical(f"its divergence does not state its shape ({', '.join(FORCED)}: whole numbers >= 1) and the window it ran on (candidate_hash)")
  stride, start, n, size, k, win = kl["stride"], kl["start"], kl["positions"], kl["block"], kl["n_probs"], kl.get("candidate_hash")
  if size < lz["min_block"] or k < lz["min_n_probs"]:
    raise NotIdentical(f"its divergence is scored in blocks of {size} positions with the top {k} tokens, under the lossy gate's {lz['min_block']} and "
                       f"{lz['min_n_probs']} (threshold.lossy min_block, min_n_probs): adjacent positions are not independent units, and a short top "
                       "list leaves the KL's lower bound loose")
  toks = kl.get("tokens")
  if not (isinstance(toks, Mapping) and all(isinstance(toks.get(a), str) and _SHA.fullmatch(toks[a]) for a in ARMS) and len({toks[a] for a in ARMS}) == 1):
    raise NotIdentical("its divergence's runs do not name one token list (tokens: the sha256 teacher_forced records, the same for original, copy and noise)")
  measured = (doc.get("envelope_measured") or {}).get("candidate_hash") if isinstance(doc.get("envelope_measured"), Mapping) else None
  if not (isinstance(win, str) and _SHA.fullmatch(win)) or win != measured:
    raise NotIdentical(f"its divergence was teacher-forced on window {str(win)[:12]}, not the window its envelope was measured on ({str(measured)[:12]}): "
                       "both are reads of the one kernel at its own shape")
  rg = lossy["range"]
  lo, hi, deep = start + stride, start + stride * n, rg["depth"]
  if stride not in rg["widths"] or lo < deep[0] or hi > deep[1]:
    raise NotIdentical(f"its divergence was scored at depth {lo}-{hi} with appends of {stride} tokens, not inside the range the kernel is declared lossy "
                       f"over (depth {deep[0]}-{deep[1]}, widths {', '.join(map(str, rg['widths']))}): the check must run where the kernel is dispatched")
  if candidate is not None:
    want = candidate if isinstance(candidate, str) else candidate.get("candidate_hash")
    if win != want:
      raise NotIdentical(f"its divergence was teacher-forced on window {win[:12]}, not this one ({str(want)[:12]})")
    if isinstance(candidate, Mapping):
      widths = {r.get("columns") for ps in (candidate.get("passes") or {}).values() if isinstance(ps, Mapping) for r in ps.get("per_kind") or []} - {None}
      if stride not in widths or not set(rg["widths"]) <= widths:
        raise NotIdentical(f"its divergence appends {stride} tokens a request and its range names widths {', '.join(map(str, rg['widths']))}, not all batch "
                           f"widths this window runs at ({', '.join(map(str, sorted(widths))) or 'it states none'}): the check must run the kernel at its own shape")
  want = {start + stride * (j + 1) for j in range(n)}   # teacher_forced's positions: the prefix, then an append of stride tokens each
  arms = {}
  for a in ARMS:
    rows = kl.get("base" if a == "original" else a)
    if not isinstance(rows, list) or any(not isinstance(r, Mapping) or not _int(r.get("pos")) or not _int(r.get("ref")) or "prompt_n" not in r
                                         or not (r["prompt_n"] is None or _int(r["prompt_n"])) or not isinstance(r.get("top"), list) or len(r["top"]) != k
                                         or any(not (isinstance(t, list) and len(t) == 2 and _int(t[0]) and _num(t[1]) and t[1] <= 0) for t in r["top"])
                                         for r in rows):
      raise NotIdentical(f"its divergence's {a} rows are not teacher-forced rows (pos, ref, top: {k} [[token, logprob <= 0]] (n_probs), prompt_n)")
    if len(rows) != len(want) or {r["pos"] for r in rows} != want:
      raise NotIdentical(f"its divergence's {a} rows are not the {n} positions {start} + {stride} x (1..{n}) its shape states: the depth and stride the rows ran at must be the stated ones")
    arms[a] = rows
  ref = {r["pos"]: r["ref"] for r in arms["original"]}
  if any(r["ref"] != ref[r["pos"]] for a in ARMS[1:] for r in arms[a]):
    raise NotIdentical("its divergence's runs were not teacher-forced on the same tokens (a reference token differs)")
  keep = set.intersection(*({r["pos"] for r in arms[a] if r["prompt_n"] == stride} for a in ARMS))   # every arm appended exactly stride tokens there
  try:
    bc, bn = (ni.blocks(arms["original"], arms[a], size, stride, keep) for a in ARMS[1:])
    got = ni.divergence(bc["blocks"], bn["blocks"], th)
  except Refused as e:
    raise NotIdentical(f"divergence: {e}") from e
  return {**got, "depth": [lo, hi], "stride": stride, "dropped": n - len(keep), "ref_outside_top": [bc["ref_outside_top"], bn["ref_outside_top"]]}


def judged(doc:Mapping[str, Any], lossy:Mapping[str, Any], candidate:str | Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Each required metric and the token-probability divergence judged from the evidence's own rows (module docstring,
  quality and divergence); candidate: the window the kernel runs on (its hash, or the candidate whose batch widths
  bind the divergence's stride), None when the caller has none. Raises NotIdentical."""
  from boltbeam.core.refused import Refused
  from boltbeam.math import noninferior as ni
  th = json.loads(THRESHOLD.read_text())["threshold"]   # Speed up's table: one confidence and truncation rule; lossy: the owner's count
  conf, said, least = th["confidence"], {r["metric"]: r for r in lossy["quality"]}, th["lossy"]["min_items"]
  change = (th["noise"]["engines"].get(str(doc.get("engine"))) or {}).get("says")
  if not change:
    raise NotIdentical(f"its evidence names engine {doc.get('engine')!r}, which declares no change that must not matter (data/speed_up.json "
                       "threshold.noise.engines): with no declared change there is no measured noise run to take a margin from")
  change = doc["noise_change"] if isinstance(doc.get("noise_change"), str) and doc["noise_change"].strip() else change
  out = {}
  for metric, (_, good) in QUALITY.items():
    rows = _rows(doc, metric)
    try:
      got = ni.noise_pair(rows["original"], rows["copy"], rows["noise"], th)
    except Refused as e:
      raise NotIdentical(f"{metric}: {e}") from e
    n = got["n"]
    if n < least:
      raise NotIdentical(f"its {metric} is judged over {n} paired items, under the {least} the lossy gate requires (threshold.lossy.min_items, owner 2026-11-21)")
    delta = got["loss"] * (-1 if good else 1)   # copy less original, of the metric itself
    r = said[metric]
    o, c = ({x["id"]: int(x["correct"]) * (1 if good else -1) for x in rows[a]} for a in ("original", "copy"))   # the metric's own value per item
    d = [c[i] - o[i] for i in o]
    half = statistics.NormalDist().inv_cdf(0.5 + conf / 2) * math.sqrt(sum((x - delta) ** 2 for x in d) / n / n)   # the paired interval at level
    lo, hi = delta - half, delta + half
    if r["n"] != n or abs(r["level"] - conf) > 1e-12 or any(abs(r[k] - v) > RESTATED for k, v in (("delta", delta), ("lo", lo), ("hi", hi))):
      raise NotIdentical(f"its {metric} delta is not its evidence's: declared {r['delta']:+.6g} [{r['lo']:+.6g}, {r['hi']:+.6g}] over {r['n']} at "
                         f"{r['level']:g}, the evidence's rows give {delta:+.6g} [{lo:+.6g}, {hi:+.6g}] over {n} at {conf:g} (paired, normal)")
    words = (f"{metric}: non-inferiority, paired over the evidence's {n} items at {conf:.0%} one-sided (math/noninferior.noise_pair): the margin "
             f"is the engine's own numerics-only noise pair (the original against its run {change}) on the same items, |loss {got['noise_loss']:+.4g}| "
             f"+ z {got['z_margin']:.3f} SE {got['noise_se']:.4g} = {got['margin']:.4g}; the copy's loss {got['loss']:+.4g} has its worse bound "
             f"{got['worse_bound']:+.4g} (loss + z {got['z']:.3f} SE {got['se']:.4g}), which must be at or below the margin")
    if got["verdict"] != "pass":
      raise NotIdentical(f"its {metric} is not shown non-inferior ({got['verdict']}{': ' + got['why'] if got.get('why') else ''}): {words}")
    out[metric] = {"delta": delta, "n": n, "margin": got["margin"], "worse_bound": got["worse_bound"], "verdict": got["verdict"], "derived": words}
  kl = doc.get("divergence")
  if not (isinstance(kl, Mapping) and isinstance(kl.get("how"), str) and kl["how"].strip()):
    raise NotIdentical("its evidence has no token-probability divergence (divergence: the original's, the copy's and the noise run's teacher-forced "
                       "rows, their shape and how they were taken): a lossy kernel is also judged on how far it moves the engine's token "
                       "probabilities, teacher-forced at the kernel's own shape, against the noise run's own")
  got = _forced(kl, doc, lossy, candidate, th)
  k, dv = got["blocks"], got["kld"]
  sk = got["dp"]
  words = (f"divergence over {got['positions']} teacher-forced positions (depth {got['depth'][0]}-{got['depth'][1]}, appends of {got['stride']} tokens; "
           f"{got['dropped']} cache misses dropped; reference outside a top list, counted 0, at {got['ref_outside_top'][0]} and {got['ref_outside_top'][1]}) in {k} blocks at {conf:.0%} one-sided, t {got['t']:.3f} at {k - 1} df "
           f"(math/noninferior.divergence): mean KL from the original {dv['copy']:.4g} (SE {dv['copy_se']:.3g}), worse bound {dv['worse_bound']:.4g}, "
           f"against the noise run's ({change}) {dv['noise']:.4g} + t {got['t_line']:.3f} SE {dv['noise_se']:.3g} = {dv['line']:.4g}; the reference "
           f"token's probability change {sk['copy']:+.4g} (SE {sk['copy_se']:.3g}), its loss's worse bound {sk['worse_bound']:+.4g}, against the noise "
           f"run's |{sk['noise']:+.4g}| + t SE {sk['noise_se']:.3g} = {sk['line']:.4g}")
  if got["verdict"] != "pass":
    raise NotIdentical(f"its token probabilities move beyond the engine's own noise ({'mean KL' if dv['verdict'] != 'pass' else 'skewed toward worse'}): {words}")
  out["divergence"] = {"blocks": k, "positions": got["positions"], "depth": got["depth"], "stride": got["stride"], "kld": dv, "dp": sk, "verdict": "pass", "derived": words}
  return out


def ceiling(doc:Mapping[str, Any], env:Mapping[str, Any], candidate:str | None = None) -> float:
  """The envelope's measured ceiling (module docstring, envelope) in its mode; raises NotIdentical when the declared
  tolerance is above it, nothing measured it, or (candidate given: the region timer's window) it was read on another
  window."""
  m = doc.get("envelope_measured")
  if candidate is not None and isinstance(m, Mapping) and m.get("candidate_hash") != candidate:
    raise NotIdentical(f"its envelope was measured on window {str(m.get('candidate_hash'))[:12]}, not this one ({candidate[:12]}): "
                       "envelope_measured names the candidate_hash whose window the region timer read")
  if not (isinstance(m, Mapping) and _num(m.get(env["mode"])) and m[env["mode"]] >= 0 and isinstance(m.get("how"), str) and m["how"].strip()):
    raise NotIdentical(f"no measured ceiling for its {env['mode']} envelope: its evidence must carry envelope_measured ({', '.join(MODES)}, how), "
                       "the region timer's identity read of this kernel against stock (worst by mode over the window's results) on the kernel "
                       "the quality run ran; nothing else in BoltBeam measures it, and a tolerance is never picked")
  if env["tolerance"] > m[env["mode"]]:
    raise NotIdentical(f"its envelope {env['mode']} {env['tolerance']:g} is above the {m[env['mode']]:g} its evidence measured ({m['how']}): a wider "
                       "envelope admits device results the quality run never produced")
  return float(m[env["mode"]])


def declared(holder:Mapping[str, Any], folder:str | pathlib.Path, kernel:str | Iterable[str] | None = None, candidate:str | Mapping[str, Any] | None = None) -> dict[str, Any]:
  """A holder's (a list entry, an arm's declaration file) identity: {identity: exact}, or {identity: lossy, lossy: the
  block, evidence_sha256, kernel_sha256: the kernel its evidence ran, ceiling, judged: each metric's judgment}. The
  evidence path is a name beside the holder's file (or an absolute path). kernel: the sha256 (or any of several) the
  evidence must name; None when the caller ties it itself (session-ab, against each arm run's files). Raises
  NotIdentical when the declaration cannot stand: an unknown identity, a lossy block on an exact holder, a malformed
  block, evidence missing, of another sha256 or of another kernel, a delta not the evidence's or not non-inferior, an
  envelope over its measured ceiling."""
  kind = holder.get("identity", EXACT)
  if kind == EXACT:
    if "lossy" in holder:
      raise NotIdentical("a lossy block on an entry that is not declared identity: lossy")
    return {"identity": EXACT}
  if kind != LOSSY:
    raise NotIdentical(f"identity {kind!r}: it is {EXACT} (the default) or {LOSSY}")
  if why := malformed(holder.get("lossy")):
    raise NotIdentical(f"declared lossy without evidence: {why}")
  lz = holder["lossy"]
  ev = lz["evidence"]
  path = pathlib.Path(ev["path"]) if pathlib.Path(ev["path"]).is_absolute() else pathlib.Path(folder) / ev["path"]
  if not path.is_file():
    raise NotIdentical(f"declared lossy without evidence: no quality run's record at {path}")
  if (got := sha256_file(path)) != ev["sha256"]:
    raise NotIdentical(f"declared lossy without evidence: {path} hashes {got[:12]}, not the declared {ev['sha256'][:12]}")
  try:
    doc = json.loads(path.read_text())
  except (OSError, ValueError) as e:
    raise NotIdentical(f"declared lossy without evidence: {path} is not a quality run's record ({EVIDENCE}): {e}") from e
  if not isinstance(doc, Mapping) or doc.get("schema") != EVIDENCE:
    raise NotIdentical(f"declared lossy without evidence: {path} is not a quality run's record (schema {EVIDENCE})")
  ks = doc.get("kernel_sha256")
  if not (isinstance(ks, str) and _SHA.fullmatch(ks)):
    raise NotIdentical(f"declared lossy without evidence: {path} does not name the kernel its quality run ran (kernel_sha256)")
  if kernel is not None and ks not in ({kernel} if isinstance(kernel, str) else set(kernel)):
    raise NotIdentical(f"declared lossy with another kernel's evidence: {path} ran {ks[:12]}, not this kernel's "
                       f"{kernel[:12] if isinstance(kernel, str) else ', '.join(k[:12] for k in kernel)}")
  try:
    top = ceiling(doc, lz["envelope"], candidate if candidate is None or isinstance(candidate, str) else candidate.get("candidate_hash"))
    judgment = judged(doc, lz, candidate)   # after the envelope: the divergence's window is held to the one it was measured on
  except NotIdentical as e:
    raise NotIdentical(f"declared lossy, refused on its evidence: {e}") from e
  return {"identity": LOSSY, "lossy": dict(lz), "evidence_sha256": got, "kernel_sha256": ks, "ceiling": top, "judged": judgment}


def worst(letter:str | None, want:bytes, got:bytes) -> dict[str, float]:
  """How far a result is from the engine's own (want), by each envelope mode: a float type (its struct letter in the
  format registry) as abs, max |want - got|, and rel_max, that over max |want| (a NaN or inf that differs is inf);
  anything else, or bytes past the last whole value that differ, 0 or inf by bits. Only a declared lossy entry is
  judged by it: exact is bits."""
  if want == got:
    return {"abs": 0.0, "rel_max": 0.0}
  if not letter or len(want) != len(got):
    return {"abs": math.inf, "rel_max": math.inf}
  size = struct.calcsize(letter)
  n = len(want) // size
  if want[n * size:] != got[n * size:]:
    return {"abs": math.inf, "rel_max": math.inf}
  a, b = struct.unpack(f"<{n}{letter}", want[:n * size]), struct.unpack(f"<{n}{letter}", got[:n * size])
  scale = max((abs(x) for x in a if math.isfinite(x)), default=0.0) or 1.0
  d = max((abs(x - y) if math.isfinite(x) and math.isfinite(y) else 0.0 if x == y or (x != x and y != y) else math.inf) for x, y in zip(a, b))
  return {"abs": d, "rel_max": d / scale}


def read(results:Mapping[str, tuple[str | None, bytes, bytes]]) -> dict[str, Any]:
  """An identity read of named results ({name: (struct letter or None, the engine's bytes, the kernel's bytes)}):
  exact per result, 4-byte words that differ, worst by mode: what device() judges."""
  ws = {k: worst(*v) for k, v in results.items()}
  return {"exact": {k: w == g for k, (_, w, g) in results.items()},
          "differ": {k: sum(g[i:i + 4] != w[i:i + 4] for i in range(0, max(len(w), len(g)), 4)) for k, (_, w, g) in results.items()},
          "worst": {m: {k: ws[k][m] for k in results} for m in MODES}}


def device(read:Mapping[str, Any], decl:Mapping[str, Any] | None) -> str | None:
  """The refusal of a device identity read (the region timer's record identity: exact, worst by mode) under a
  declaration, in words, or None. Exact: every result bit for bit. Lossy: worst within the envelope, by its mode."""
  decl = decl or {"identity": EXACT}
  if decl["identity"] == EXACT:
    same = read.get("exact") if "exact" in read else {k: n == 0 for k, n in (read.get("differ") or {}).items()}   # a read before "exact": its 4-byte words
    if not isinstance(same, Mapping) or not same or not all(v is True for v in same.values()):
      bad = ({k: n for k, n in (read.get("differ") or {}).items() if n} or {k: "bytes past the last whole word" for k, v in same.items() if v is not True}
             if isinstance(same, Mapping) and same else "no identity read")
      return (f"the kernel's results on the device are not the engine's own bit for bit ({bad}: 4-byte words that differ), and its entry is not "
              "declared identity: lossy with evidence: nothing is timed, installed or realized")
    return None
  env = decl["lossy"]["envelope"]
  worst_ = (read.get("worst") or {}).get(env["mode"]) or {}
  if not worst_ or not all(_num(w) and w <= env["tolerance"] for w in worst_.values()):
    return f"the kernel's results stray beyond its declared lossy envelope ({worst_ or 'no read'} against {env['mode']} {env['tolerance']:g}): nothing is timed"
  return None


def label(decl:Mapping[str, Any] | None) -> str:
  """A declaration in one line: exact, or lossy with its evidence and deltas, for a report beside the deltas."""
  if not decl or decl.get("identity", EXACT) == EXACT:
    return "identity exact (bit for bit)"
  lz, j = decl["lossy"], decl.get("judged") or {}
  q = "; ".join(f"{r['metric']} {r['delta']:+.4g} [{r['lo']:+.4g}, {r['hi']:+.4g}] at {r['level']:.0%} over {r['n']}"
                + (f" (non-inferior: worse bound {j[r['metric']]['worse_bound']:+.4g} within a margin of {j[r['metric']]['margin']:.4g}, the engine's noise pair)"
                   if r["metric"] in j else "") for r in lz["quality"])
  e, dv = lz["envelope"], j.get("divergence")
  if dv:
    q += f"; mean KL {dv['kld']['copy']:.3g} against the noise run's line {dv['kld']['line']:.3g} over {dv['positions']} positions"
  return (f"identity LOSSY (declared): quality {q}; envelope {e['mode']} {e['tolerance']:g} ({e['derived']}; measured ceiling {decl.get('ceiling', float('nan')):g}); "
          f"evidence {lz['evidence']['path']} sha256 {lz['evidence']['sha256'][:12]} of kernel {str(decl.get('kernel_sha256'))[:12]}")


def read_file(path:str | pathlib.Path) -> dict[str, Any]:
  """A declaration file (session-ab --lossy): {identity: lossy, lossy: {...}} as a list entry holds them, judged here;
  its kernel tie is the caller's (session-ab: kernel_sha256 against each arm run's build and files)."""
  p = pathlib.Path(path).expanduser()
  doc = json.loads(p.read_text())
  if not isinstance(doc, Mapping) or doc.get("identity") != LOSSY:
    raise NotIdentical(f"{p} is not a lossy declaration (identity: lossy and its lossy block)")
  return {**declared(doc, p.parent), "file": str(p.absolute()), "file_sha256": sha256_file(p)}


__all__ = ["ARMS", "EVIDENCE", "EXACT", "LOSSY", "MODES", "QUALITY", "NotIdentical", "ceiling", "declared", "device", "judged", "label", "malformed",
           "read", "read_file", "worst"]
