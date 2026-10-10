"""Self-contained HTML run report: the TUI results screen on paper.

The point of the page is to show where the token loses time against the roofline, and what to try about it. It
reads `workflow.screen.results(run)`, the same seam the TUI reads (loss, tie_out, roles), and computes nothing
new: every number on the page is a measured or derived number from that seam, or a share of two of them.

Sections, in order (a measured run):
  1. The answer: model, chip, engine, batch, weight format; measured speed against the limit; ms per token lost.
  2. Where the time goes: the tie-out from the limit to the measured token; then Where the token goes, the same
     token as one ranked table of what each part costs it (tie_out.where_token_goes).
  3. Per role, sorted by lost ms, in the TUI's words.
  4. What to try next, from the reasons by a stated rule table (NEXT_RULES, LINE_RULES): data, not code.
  5. Facts: chip profile, capture method, stages that ran, files in the run.
An unmeasured run has section 1 with one sentence on what is missing and the command, then section 5.

Load-bearing properties: deterministic (no timestamps, stable order, so two renders are byte identical); no
external requests (CSS and JS inline, system fonts); both themes; a phone at 390 px scrolls only vertically.
"""
from __future__ import annotations

import html
import re
from typing import Any

# canonical pipeline order. The keys are the `stage=` values workflow/*.py pass to update_manifest; the label
# is the stage's short id. A stage absent from run_manifest["stages"] renders as not-run, never hidden.
STAGES: tuple[tuple[str, str, str], ...] = (
  ("load",          "load",        "model / weight / workload facts"),
  ("autoscan",      "autoscan",    "machine + provider capabilities"),
  ("analyze",       "analyze",     "search space + measurement plan"),
  ("runner_plan",   "runner-plan", "audit-tracer handoff bundle"),
  ("ingest_probe",  "probe",       "primitive evidence classified"),
  ("ingest_timing", "timing",      "timing trace classified"),
  ("output",        "output",      "policy / report / provider plan"),
)

# Plain words, the same ones boltbeam-tui prints (tui/internal/ui/render.go: plainStage, plainNeed, plainStatus,
# plainBucket, plainRegime, plainRoute). The page shows the plain word first and the record's id small beside it.
# tests/test_report_html.py checks these tables against render.go, so the two screens cannot drift apart.
PLAIN_STAGE = {"load": "Read the model", "autoscan": "Check the machine", "analyze": "Plan what to try",
               "runner_plan": "Prepare the handoff", "ingest_probe": "Test the building blocks",
               "ingest_timing": "Time the real run", "output": "Package the result"}
PLAIN_STATUS = {"not_analyzed": "not planned yet", "needs_measurement": "needs measuring", "policy_seeded": "plan ready"}
PLAIN_NEED = {"probe_evidence": "the building-block tests", "timing_trace": "a timing trace"}
PLAIN_ROUTE = {"promoted": "kept", "refuted": "ruled out", "blocked": "undecided", "unmeasured": "default kernel, none compared yet",
               "candidate": "to try"}
PLAIN_BUCKET = {"at_peak": "at the speed limit", "gemv_codegen_capped": "reads memory slower than it could",
                "latency_bound": "waiting on memory", "elementwise_dilution": "small kernel, dead time",
                "activation_bound": "real activation work", "timing_inconclusive": "not clear yet"}
PLAIN_REGIME = {"streaming_bound": "reads memory at full speed", "occupancy_starved": "not enough work in flight",
                "dequant_bound": "slowed by unpacking the numbers", "metadata_bound": "slowed by the scale tables",
                "latency_bound": "waiting on memory", "compute_bound": "limited by the sums",
                "inconclusive": "not clear yet"}

# Severity of every word the classifiers emit, so the colour follows the meaning. Sources:
# roofline/roofline_trace.py (kernel buckets), trace/timing.py (role, candidate and dominant buckets),
# quantization/quant_gemv.py (regimes), trace/schedule_trace.py (fused_ok). A word not listed gets no colour.
SEVERITY = {
  "at_peak": "ok", "streaming_bound": "ok", "timing_win": "ok", "fused_ok": "ok", "promoted": "ok",
  "compute_bound": "ok", "activation_bound": "ok",
  "latency_bound": "warn", "elementwise_dilution": "warn", "timing_flat": "warn", "timing_inconclusive": "warn",
  "inconclusive": "warn", "mixed": "warn", "route_not_bound": "warn", "unmeasured": "warn", "blocked": "warn",
  "gemv_codegen_capped": "bad", "occupancy_starved": "bad", "dequant_bound": "bad", "metadata_bound": "bad",
  "timing_loss": "bad", "correctness_failed": "bad", "refuted": "bad",
  # timing_hot / timing_observed say how big a role is, not whether it is healthy: no colour.
}
PLAIN_ROLE = {"attn_kv": "attention keys and values", "attn_qo": "attention query and output", "ffn_gate_up": "feed-forward in",
              "ffn_down": "feed-forward out", "lm_head": "vocabulary output", "embed": "token embedding",
              "attn_qkv": "attention query, keys and values, fused", "attn_gate": "attention output gate",
              "ssm_out": "state-space output", "ssm_alpha_beta": "state-space decay and update gates"}

_CSS = """
/* Glass over a gradient. Cards keep an opaque enough fill that text reads without backdrop-filter (print, old
   browsers). System fonts only: this file makes no external requests. */
:root{--ui:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--bg:#0d1022;--bg-a:#141a3d;--bg-b:#2a1747;
--glass:rgba(30,34,64,.78);--glass-2:rgba(255,255,255,.05);--edge:rgba(190,200,255,.16);--shadow:0 10px 30px rgba(0,0,0,.35);
--tx:#eef0ff;--tx-2:#c3c8ea;--tx-3:#9198c2;
--g1:#ff7ac6;--g2:#a98bff;--g3:#5fd6ff;
--ideal:#7f88b8;--excess:#f2b84b;--other:#b48cff;--gaps:#ff6b81;--ok:#7fd88f;--track:rgba(255,255,255,.08)}
@media (prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#eef0fa;--bg-a:#e3e8ff;--bg-b:#f3e6ff;
--glass:rgba(255,255,255,.82);--glass-2:rgba(40,50,120,.04);--edge:rgba(60,70,140,.16);--shadow:0 10px 30px rgba(60,70,140,.12);
--tx:#1d2250;--tx-2:#3f4677;--tx-3:#666c96;--g1:#d63c96;--g2:#7c4ddb;--g3:#0a8fbf;
--ideal:#8790b8;--excess:#b06d00;--other:#7c4ddb;--gaps:#d0334f;--ok:#2f8a46;--track:rgba(40,50,120,.08)}}
:root[data-theme=light]{--bg:#eef0fa;--bg-a:#e3e8ff;--bg-b:#f3e6ff;--glass:rgba(255,255,255,.82);
--glass-2:rgba(40,50,120,.04);--edge:rgba(60,70,140,.16);--shadow:0 10px 30px rgba(60,70,140,.12);--tx:#1d2250;
--tx-2:#3f4677;--tx-3:#666c96;--g1:#d63c96;--g2:#7c4ddb;--g3:#0a8fbf;--ideal:#8790b8;--excess:#b06d00;
--other:#7c4ddb;--gaps:#d0334f;--ok:#2f8a46;--track:rgba(40,50,120,.08)}
*,*::before,*::after{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;min-height:100vh;color:var(--tx);font-family:var(--ui);font-size:14px;line-height:1.55;
-webkit-font-smoothing:antialiased;padding:32px 24px 64px;
background:radial-gradient(1200px 700px at 10% -10%,var(--bg-a),transparent 60%),
radial-gradient(1000px 800px at 100% 10%,var(--bg-b),transparent 60%),var(--bg);background-attachment:fixed}
.wrap{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:20px;min-width:0}
.mast{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.mark{font-weight:800;font-size:17px;letter-spacing:-.01em;background:linear-gradient(90deg,var(--g1),var(--g2),var(--g3));
-webkit-background-clip:text;background-clip:text;color:transparent}
.run-id{color:var(--tx-3);font-family:var(--mono);font-size:11.5px;overflow-wrap:anywhere;flex:1;min-width:0}
.toggle{cursor:pointer;font:inherit;font-size:12px;color:var(--tx-2);background:var(--glass);border:1px solid var(--edge);
border-radius:20px;padding:4px 12px}
.card{background:var(--glass);border:1px solid var(--edge);border-radius:16px;box-shadow:var(--shadow);
-webkit-backdrop-filter:blur(18px) saturate(140%);backdrop-filter:blur(18px) saturate(140%);padding:22px 24px;min-width:0}
.card h2{margin:0 0 14px;font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--tx-3)}
.card h2 .n{display:inline-block;width:20px;height:20px;line-height:20px;text-align:center;border-radius:50%;
margin-right:8px;color:#fff;background:linear-gradient(135deg,var(--g1),var(--g2));letter-spacing:0;font-size:11px}
.who{font-size:13.5px;color:var(--tx-2);margin:0 0 14px;overflow-wrap:anywhere}
.who b{color:var(--tx);font-size:20px;font-weight:700;margin-right:6px}
.hero{display:flex;align-items:baseline;flex-wrap:wrap;gap:6px 18px}
.big{font-size:clamp(40px,8vw,96px);font-weight:800;line-height:1;letter-spacing:-.03em;font-variant-numeric:tabular-nums;
background:linear-gradient(90deg,var(--g1),var(--g2) 55%,var(--g3));-webkit-background-clip:text;background-clip:text;color:transparent}
.big-u{font-size:16px;color:var(--tx-2)}
.lim{font-size:clamp(18px,3vw,28px);font-weight:700;color:var(--tx-2);font-variant-numeric:tabular-nums}
.lim small{font-size:13px;font-weight:500;color:var(--tx-3)}
.big.dim{font-size:clamp(30px,6vw,56px)}
.gauge{position:relative;height:16px;border-radius:9px;background:var(--track);margin:22px 0 8px;border:1px solid var(--edge)}
.gauge i{position:absolute;left:0;top:0;bottom:0;border-radius:9px;background:linear-gradient(90deg,var(--g1),var(--g2),var(--g3))}
.gauge s{position:absolute;right:-1px;top:-6px;bottom:-6px;width:3px;border-radius:2px;background:var(--g3);
box-shadow:0 0 10px 2px var(--g3)}
.gauge-l{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--tx-2)}
.gauge-l b{color:var(--tx);font-size:15px}
.lost{color:var(--gaps);font-weight:700}
.stack{display:flex;height:34px;border-radius:10px;overflow:hidden;border:1px solid var(--edge);background:var(--track);position:relative}
.stack span{display:flex;align-items:center;justify-content:center;font-size:11.5px;font-weight:700;color:#10132a;
white-space:nowrap;overflow:hidden;min-width:0;font-variant-numeric:tabular-nums}
.stack .band{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,rgba(255,255,255,.45) 0 3px,transparent 3px 6px);
border-left:1px solid var(--tx);border-right:1px solid var(--tx);opacity:.7}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:baseline}
.c-ideal{background:var(--ideal)}.c-excess{background:var(--excess)}.c-other{background:var(--other)}.c-gaps{background:var(--gaps)}
.c-ok{background:var(--ok)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:14px}
td{padding:8px 6px;border-bottom:1px solid var(--edge);text-align:right;color:var(--tx-2);vertical-align:top}
td:first-child{text-align:left;color:var(--tx)}
td.how{color:var(--tx-3);font-size:12px;text-align:left;padding-left:16px}
.where th{padding:6px;font-size:12px;font-weight:600;color:var(--tx-3);text-align:right;border-bottom:1px solid var(--edge)}
.where th:first-child{text-align:left}.where td.lost{color:var(--tx);font-weight:700}
tr.sum td{font-weight:800;color:var(--tx);border-bottom:0;border-top:2px solid var(--edge)}
.sub{display:block;color:var(--tx-3);font-size:12px;font-weight:400}
.note{color:var(--tx-2);font-size:13px;margin:10px 0 0}
.muted{color:var(--tx-3);font-size:12.5px;margin:8px 0 0}
.roles{display:flex;flex-direction:column;gap:6px}
.role{border-radius:12px;background:var(--glass-2);border:1px solid var(--edge)}
.role summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:minmax(150px,1fr) minmax(0,2fr) 30ch;
gap:6px 14px;align-items:center;padding:10px 14px}
.role summary::-webkit-details-marker{display:none}
.rname{font-weight:600;overflow-wrap:anywhere}
.rname .q{color:var(--tx-3);font-weight:500;font-size:12px;margin-left:6px}
.rbar{position:relative;height:12px;border-radius:6px;background:var(--track);overflow:hidden}
.rbar::after{content:"";position:absolute;right:0;top:0;bottom:0;width:2px;background:var(--tx-2)}
.rbar i{display:block;height:100%;border-radius:6px}
.q{margin-left:6px}
.rnum{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;display:flex;flex-wrap:wrap;justify-content:flex-end;column-gap:8px;row-gap:2px;align-items:baseline}
.rtags{white-space:normal;display:inline}
.rnum b{color:var(--tx)}.rnum span{color:var(--tx-3);font-size:12px;margin-left:8px}
.why{font-size:12px;font-weight:600}
.r-ok{color:var(--ok)}.r-small{color:var(--excess)}.r-slow{color:var(--gaps)}
.rdet{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px 12px;padding:0 14px 12px;font-size:12.5px;color:var(--tx-2)}
.rdet div span{display:block;color:var(--tx-3);font-size:10.5px;letter-spacing:.06em}
.next{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px;counter-reset:n}
.next li{counter-increment:n;position:relative;padding:14px 16px 14px 56px;border-radius:12px;background:var(--glass-2);
border:1px solid var(--edge)}
.next li::before{content:counter(n);position:absolute;left:16px;top:14px;width:26px;height:26px;line-height:26px;
text-align:center;border-radius:50%;font-weight:800;font-size:13px;color:var(--tx);border:1px solid var(--edge)}
.next li.top{border:1px solid transparent;background:linear-gradient(var(--glass),var(--glass)) padding-box,
linear-gradient(90deg,var(--g1),var(--g2),var(--g3)) border-box}
.next li.top::before{color:#fff;border:0;background:linear-gradient(135deg,var(--g1),var(--g2))}
.next .what{font-weight:700;color:var(--tx)}
.next .ms{color:var(--tx-2);font-variant-numeric:tabular-nums}
.ev{display:block;margin-top:4px;font-size:12px;color:var(--tx-2);opacity:.8;overflow-wrap:anywhere}
.ev a{color:inherit}
.next .do{display:block;margin-top:4px;color:var(--tx-2)}
.rules{margin-top:12px}
.rules summary{cursor:pointer;color:var(--tx-3);font-size:12.5px}
.rules table td{font-size:12.5px}
dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:8px 18px;margin:0;font-size:13.5px}
dt{color:var(--tx-3)}dd{margin:0;color:var(--tx-2);overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:11px;padding:3px 10px;border-radius:12px;background:var(--glass-2);color:var(--tx-2);
border:1px solid var(--edge);font-family:var(--mono);text-decoration:none;overflow-wrap:anywhere}
a.chip:hover{color:var(--tx)}
a{color:var(--g3)}
code{background:var(--glass-2);border:1px solid var(--edge);padding:1px 6px;border-radius:5px;font-family:var(--mono);
font-size:12px;overflow-wrap:anywhere}
.foot{color:var(--tx-3);font-size:11.5px;text-align:center}
.ties{margin-top:18px;padding-top:14px;border-top:1px solid var(--edge)}
.ties h3{margin:0 0 6px;font-size:13px;color:var(--tx)}
.eq{overflow-x:auto;overflow-y:hidden;padding:4px 0}
.eq math{width:max-content;margin:0 auto;font-family:"STIX Two Math","Cambria Math",math;font-size:17px;color:var(--tx)}
.eq mtext.lab{font-family:var(--ui);font-size:11.5px;color:var(--tx-3)}
.eq mtext.u{font-family:var(--ui);font-size:13px;color:var(--tx-2)}
@media (max-width:640px){
body{padding:16px 16px 40px}.card{padding:18px 16px}dl{grid-template-columns:1fr;gap:2px}dd{margin-bottom:8px}
.role summary{grid-template-columns:minmax(0,1fr)}.rnum{text-align:left;white-space:normal}
.rdet{grid-template-columns:repeat(2,minmax(0,1fr))}
td.how{display:none}.eq math{font-size:13px}.eq mtext.lab{font-size:10px}.stack span{font-size:10px}
.next li{padding-left:48px}.next li::before{left:12px}
}
.card .card{box-shadow:none;border-radius:12px;margin-top:16px;padding:0;overflow:hidden}
.card-hd{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;padding:10px 14px;border-bottom:1px solid var(--edge)}
.card-ttl{margin:0;font-size:13px;color:var(--tx)}
.card-sub,.id{color:var(--tx-3);font-family:var(--mono);font-size:11px}
.tscroll{overflow-x:auto}.tscroll table{margin:0}.tscroll td{white-space:nowrap}td.wrap{white-space:normal;min-width:220px}
.tag{font-size:11px;padding:1px 8px;border-radius:10px;border:1px solid var(--edge)}
.empty{color:var(--tx-3);font-size:12.5px;padding:10px 14px;margin:0}
@media print{:root,:root[data-theme]{--bg:#fff;--bg-a:#fff;--bg-b:#fff;--glass:#fff;--tx:#1d2250;--tx-2:#3f4677;--tx-3:#666c96;
--edge:rgba(60,70,140,.25)}body{background:#fff}.card{box-shadow:none;-webkit-backdrop-filter:none;backdrop-filter:none}.toggle{display:none}}
"""

# theme toggle only; everything else is static so the page is readable with JS disabled.
_JS = """
(function(){
  var r=document.documentElement,b=document.getElementById('theme');
  if(!b)return;
  b.addEventListener('click',function(){
    var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&
      window.matchMedia('(prefers-color-scheme:dark)').matches);
    r.setAttribute('data-theme',dark?'light':'dark');
  });
})();
"""


_NUM = "\x00n\x00"  # the section number placeholder, filled in order at the end


def _number(sections:str) -> str:
  parts = sections.split(_NUM)
  return "".join(p + (str(i + 1) if i < len(parts) - 1 else "") for i, p in enumerate(parts))


def _e(value:Any) -> str:
  """Escape any value for HTML text/attribute context. Kernel names come from provider traces — untrusted."""
  return html.escape("" if value is None else str(value), quote=True)


def _bucket_class(bucket:Any) -> str:
  sev = SEVERITY.get(str(bucket)) if bucket else None
  return f"t-{sev}" if sev else ""


def _named(table:dict[str, str], key:Any, *, tag:bool = False) -> str:
  """The plain word with the record's id small and muted beside it (the TUI's `named`)."""
  key_s = "" if key is None else str(key)
  plain = table.get(key_s)
  if tag:
    text = _e(plain or key_s or "unknown")
    inner = f'<span class="tag {_bucket_class(key_s)}">{text}</span>'
    return inner + (f' <span class="id">{_e(key_s)}</span>' if plain else "")
  return f'{_e(plain)} <span class="id">{_e(key_s)}</span>' if plain else _e(key_s or "unknown")


def _card(title:str, source:str, body:str, *, flag:bool = False, cls:str = "") -> str:
  """A titled panel. `source` names the artifact the body was read from: no section without a citation."""
  classes = " ".join(c for c in ("card", "flag" if flag else "", cls) if c)
  return (f'<section class="{classes}"><div class="card-hd"><h2 class="card-ttl">{title}</h2>'
          f'<span class="card-sub">{_e(source)}</span></div>{body}</section>')


# Stages a run does not need once step 4 measured on this machine (measure_status.json "measured"): the
# handoff bundle is for an outside runner, and there is none.
NOT_NEEDED = {"runner_plan": "not needed: measured on this machine"}


def stage_state(key:str, entry:Any, measure:dict[str, Any] | None) -> tuple[str, str | None]:
  """done, not_needed (with its sentence) or open. The one place this is decided; the screen draws it."""
  if entry:
    return "done", None
  if key in NOT_NEEDED and (measure or {}).get("status") == "measured":
    return "not_needed", NOT_NEEDED[key]
  return "open", None

def next_step(report:dict[str, Any], plan:dict[str, Any], probe:dict[str, Any] | None = None,
              measure:dict[str, Any] | None = None) -> str:
  """The one next-step ladder, plain text with `code` in backticks; summary.md (workflow/output.py), the screen
  (workflow/screen.py show) and this page all print it, so they never disagree. A probe the run already took, or
  one this GPU cannot give (measure_status probe absent), is never asked for again: its own record is read."""
  probe, measure = probe or {}, measure or {}
  if probe.get("status") == "measured":
    absent = probe.get("absent") or {}
    missing = "; ".join(sorted({v for v in absent.values()})) if absent else ""
    head = f"The building-block probe ran: {len(probe.get('rows') or [])} roles timed with BoltBeam's own kernel (reference)."
    return head + (f" Not reported by this GPU: {missing}." if missing else "")
  if measure.get("probe") == "absent":
    return f"No building-block probe here: {measure.get('probe_reason') or 'this chip has none'}."
  if plan.get("timing_profile", {}).get("status") == "requested":
    return ("Run an external timing trace from `trace_request.json`, ingest `boltbeam.timing_trace.v1`, then re-run "
            "`boltbeam analyze`.")
  if plan.get("primitive_profile", {}).get("status") == "requested":
    return ("Run an external probe from `probe_request.json`, ingest `boltbeam.probe_evidence.v1`, then re-run "
            "`boltbeam analyze`.")
  if report.get("status") == "policy_seeded":
    return "Review `route_policy.json`; selected routes still require normalized evidence before promotion unless already ledgered."
  if plan:
    return "Run or translate `measurement_plan.json` through a provider adapter, then ingest normalized evidence and re-analyze."
  return "Run `boltbeam analyze --run <run>` to build the search and measurement plan."


def _code(text:str) -> str:
  """Plain text with `backticks` as an HTML fragment with <code>."""
  parts = text.split("`")
  return "".join(_e(p) if i % 2 == 0 else f"<code>{_e(p)}</code>" for i, p in enumerate(parts))


def roofline_kernels(timing:dict[str, Any], context:Any = None) -> tuple[list[dict[str, Any]], Any]:
  """Hottest per-kernel rows at one context, plus that context: the measured token's context when given
  (tie_out.measured_step picks it), else the widest summary. Deterministic tie-break."""
  summaries = [s for s in timing.get("context_summaries", []) or [] if isinstance(s, dict)]
  if not summaries: return [], context
  at = [s for s in summaries if context is not None and s.get("context") == context]
  chosen = at[0] if at else max(summaries, key=lambda s: (s.get("context") or 0, str(s.get("dominant_timing_bucket") or "")))
  kernels = [k for k in (chosen.get("roofline", {}) or {}).get("kernels", []) or [] if isinstance(k, dict)]
  kernels.sort(key=lambda k: (-(k.get("us") or 0.0), str(k.get("name") or "")))
  return kernels, chosen.get("context")


# Shown instead of the per-role table while no role has had kernels compared; boltbeam-tui prints the same line.
NO_KERNEL_CHOICE = "No kernels compared yet. Every role runs the default kernel."
# Plain words for a role's kernel search verdict (search/role_compare.py role_verdict); boltbeam-tui uses the same.
PLAIN_VERDICT = {"applied": "applied", "found_not_applied": "found, not applied", "none_faster": "none faster",
                 "not_reproduced": "provider claim not reproduced",
                 "not_searched": "not searched"}
VERDICT_CLASS = {"applied": "r-ok", "found_not_applied": "r-slow", "none_faster": "", "not_reproduced": "r-slow", "not_searched": ""}

# Where every compare time comes from; boltbeam-tui prints the same sentence (render.go compareNote).
COMPARE_NOTE = "Times are from tinygrad's Metal runtime, not llama.cpp."


def _compare_card(policy:dict[str, Any]) -> str:
  """The per-role kernel comparison (boltbeam/search/role_compare.py), the same table step 5 shows."""
  routes = [r for r in policy.get("routes", []) or [] if isinstance(r, dict)]
  body = []
  for r in routes:
    c = r.get("compare") if isinstance(r.get("compare"), dict) else {}
    ab = c.get("ab") if isinstance(c.get("ab"), dict) else {}
    whole = "—"
    if all(isinstance(ab.get(k), (int, float)) for k in ("baseline_tok_s", "candidate_tok_s", "delta_pct")):
      whole = f'{ab["baseline_tok_s"]:.2f} to {ab["candidate_tok_s"]:.2f} ({ab["delta_pct"]:+.1f}%)'
    kn = c.get("kernel") if isinstance(c.get("kernel"), dict) else None
    kernel = "—"
    if kn and kn.get("model_us_per_call") is not None:
      kernel = (f'model {kn["model_us_per_call"]:.0f} µs to plan {kn["plan_us"]:.0f} µs '
                f'({"faster" if kn["faster_than_model"] else "slower"})')
    elif kn:
      kernel = f'plan {kn["plan_us"]:.0f} µs alone'
    body.append(f'<tr><td>{_e(r.get("role") or "unknown")}</td><td>{_e(r.get("quant") or "unknown")}</td>'
                f'<td class="l">{_named(PLAIN_ROUTE, r.get("status") or "unknown", tag=True)}</td>'
                f'<td class="l">{_e(c.get("plan") or "—")}</td>'
                f'<td>{_e(kernel)}</td><td>{_e(whole)}</td><td class="l wrap">{_e(c.get("reason") or "")}</td></tr>')
  table = ('<div class="tscroll"><table class="wide"><thead><tr><th>role</th><th>quant</th><th class="l">choice</th>'
           '<th class="l">best plan</th><th>kernel per call</th><th>whole model tok/s</th>'
           f'<th class="l">why</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
           f'<p class="empty">{_e(COMPARE_NOTE)} Kernel per call is the kernel of the role inside the running model, to '
           'the best plan alone at the same shape. Whole model is the default kernel to the plan, in a matched decode A/B. '
           'One number never stands in for the other.</p>')
  return _card("Kernel choice per role", "route_policy.json · kernel_compare/", table)


def _routes_card(policy:dict[str, Any]) -> str:
  routes = [r for r in policy.get("routes", []) or [] if isinstance(r, dict)]
  if any(r.get("status") not in (None, "unmeasured") and isinstance(r.get("compare"), dict) for r in routes):
    return _compare_card(policy) + _selected_routes(policy)
  rows = [r for r in routes if r.get("selected_route")]
  if not rows:
    text = NO_KERNEL_CHOICE
    return _card("Kernel choice per role", "route_policy.json", f'<p class="empty">{_e(text)}</p>', cls="compact")
  return _selected_routes(policy)


def _selected_routes(policy:dict[str, Any]) -> str:
  rows = [r for r in policy.get("routes", []) or [] if isinstance(r, dict) and r.get("selected_route")]
  if not rows: return ""
  rows = sorted(rows, key=lambda r: (str(r.get("selected_route")), str(r.get("role") or "")))
  blocks = []
  for row in rows:
    rollback = row.get("rollback") or {}
    cmd = " ".join(f"{k}={rollback[k]}" for k in sorted(rollback)) or "# no rollback recorded"
    refs = row.get("evidence_refs") or []
    cite = ", ".join(str(r) for r in refs) if refs else "no evidence refs"
    blocks.append(
      f'<details><summary><span class="rb-name">{_e(row.get("selected_route"))}</span>'
      f'<span class="tag">{_e(row.get("role") or "unknown")}</span>'
      f'<span class="tag">{_e(row.get("quant") or "unknown")}</span>'
      f'{_named(PLAIN_ROUTE, row.get("status") or "unknown", tag=True)}'
      f'<span class="card-sub">{_e(cite)}</span></summary>'
      f'<pre>{_e(cmd)}</pre></details>')
  return _card("Selected routes", "route_policy.json · rollback commands", "".join(blocks))


# --- What to try next: the rule table (Prefer data over code). Each rule maps a measured reason or tie-out line
# to one lever. The page states the rule beside its advice; it never adds a number the seam did not give.
REASON_CLASS = {"at the limit": "r-ok", "too small to fill memory": "r-small", "slow kernel": "r-slow",
                "compute bound": "r-small", "unexplained": "r-slow"}
REASON_COLOR = {"at the limit": "var(--ok)", "too small to fill memory": "var(--excess)", "slow kernel": "var(--gaps)"}
NEXT_RULES = {  # per-role reason word (tie_out.REASONS) to the lever; "at the limit" has nothing to gain
  "too small to fill memory": "Fuse it with the roles next to it (Q, K and V) or batch more tokens, so each call moves more bytes.",
  "slow kernel": "Try other kernels for it.",
}
# a slow kernel whose search found nothing faster: the lever is the kernel itself, never "try other kernels" again
SEARCHED_LEVER = "{n} searched, none faster than the model's kernel; the kernel itself is the lever: write a better one for this shape."
GAPS_SHARE = 0.10  # gaps between kernels at or above this share of the token: launch fewer kernels
GAPS_LEVER = "Launch fewer kernels: run the token as one graph (CUDA graphs or Metal command buffer reuse) or fuse kernels."
NOT_TIMED_LEVER = ("The kernels were timed alone, so this time is not split by kernel. An in-model capture (Metal System "
                   "Trace, with Xcode; nsys on NVIDIA) splits it into weight kernels, attention, norms and idle time.")
OTHER_SHARE = 0.05  # other kernels above their ideal at or above this share of the token get an item
FUSIBLE = ("quantize", "norm", "elementwise", "rope", "copy")  # kernels small enough to fold into a neighbour
OTHER_LEVER_FUSE = "Fuse {kinds} into the weight kernels next to them."
OTHER_LEVER_KERNEL = "Try a faster {kind} kernel."
SPLIT_LEVER = "Split the loss by role first: {missing}."
COMMAND = "python -m boltbeam.workflow.screen pipeline {model} --run {run} --target {target} --measure auto"


def _f(v:Any, fmt:str = "{:.2f}") -> str:
  return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else ""


def _pct(part:float, whole:float) -> float:
  return max(0.0, min(100.0, 100.0 * part / whole)) if whole else 0.0


def _role_name(role:Any, quant:Any) -> str:
  return f'{_e(PLAIN_ROLE.get(str(role), role))}<span class="q">{_e(quant)}</span>'


def _other_label(line:dict[str, Any] | None) -> str:
  """The other-kernels line in plain words, with its kinds inline when the capture named them."""
  kinds = [p["kind"] for p in (line or {}).get("parts") or []]
  return "Other kernels" + (f" ({', '.join(kinds)})" if kinds else "")


def _dedupe(ptrs:list[dict[str, Any]]) -> list[dict[str, Any]]:
  seen, out = set(), []
  for p in ptrs:
    key = (p.get("file"), p.get("path"))
    if key not in seen:
      seen.add(key)
      out.append(p)
  return out


def _evidence(ptrs:list[dict[str, Any]] | None) -> str:
  """A muted line of pointers into the run's raw JSON: file › path, each a relative link to the file."""
  if not ptrs:
    return ""
  # One link per file. Its paths are folded into one bracket so a card never prints thirty links.
  by_file:dict[str, list[str]] = {}
  for ptr in _dedupe(ptrs):
    by_file.setdefault(str(ptr["file"]), [])
    if ptr.get("path"):
      by_file[str(ptr["file"])].append(str(ptr["path"]))
  links = []
  for file, paths in by_file.items():
    label = _e(file)
    if paths:
      heads = [m.group(1) for m in (re.match(r"^([a-z_]+)\[", x) for x in paths) if m]
      inner = [x[x.index("[") + 1:-1] for x in paths if "[" in x and x.endswith("]")]
      if heads and len(set(heads)) == 1 and len(inner) == len(paths):
        more = f", +{len(inner) - 6} more" if len(inner) > 6 else ""
        label += f" › {_e(heads[0])}[{_e(', '.join(inner[:6]))}{more}]"
      else:
        label += " › " + _e(", ".join(paths[:4])) + (f", +{len(paths) - 4} more" if len(paths) > 4 else "")
    links.append(f'<a href="{_e(file)}">{label}</a>')
  return f'<span class="ev">evidence: {", ".join(links)}</span>'


def _section(n:Any, title:str, body:str) -> str:
  """A numbered card. n is a placeholder here; render_run_html numbers the sections it keeps, in order, so a
  section that has nothing to say never leaves a gap in the numbers."""
  return f'<section class="card"><h2><span class="n">{_NUM}</span>{_e(title)}</h2>{body}</section>'


def _measured(loss:dict[str, Any]) -> dict[str, Any] | None:
  """The engine's untraced whole step (the TUI's Result line); the per-role GPU time only when it is all there is."""
  runs = loss.get("runtimes") or []
  return next((r for r in runs if not r.get("per_role")), runs[0] if runs else None)


# --- How this ties out: the headline, the limit and the bar as equations. MathML, not KaTeX: KaTeX inline with its
# fonts is about 645 KB, MathML needs no assets and no JS. Each equation carries its LaTeX source as an annotation.
# Every number is one the page already prints, at the same rounding; nothing is recomputed here.
SHORT_LINE = (("weight kernels", "weight excess"), ("other kernels", "other kernels"), ("gaps", "gaps"),
              ("kernels and gaps, not split", "kernels and gaps"), ("all kernels", "all kernels"))


def _mn(v:float, fmt:str = "{:.1f}") -> str:
  return f"<mn>{fmt.format(v)}</mn>"


def _t(sub:str, arg:str = "") -> str:
  """t with a text subscript, and an optional argument in parentheses: t_ideal(136)."""
  base = f'<mrow><msub><mi>t</mi><mtext>{_e(sub)}</mtext></msub>'
  return base + (f'<mo stretchy="false">(</mo><mn>{_e(arg)}</mn><mo stretchy="false">)</mo>' if arg else "") + "</mrow>"


def _u(unit:str) -> str:
  return f'<mspace width="0.25em"></mspace><mtext class="u">{_e(unit)}</mtext>'


def _brace(num:str, label:str) -> str:
  return (f'<munder><munder accentunder="true">{num}<mo stretchy="true">&#x23DF;</mo></munder>'
          f'<mtext class="lab">{_e(label)}</mtext></munder>')


def _eq(tex:str, mml:str) -> str:
  return (f'<div class="eq"><math display="block"><semantics><mrow>{mml}</mrow>'
          f'<annotation encoding="application/x-tex">{_e(tex)}</annotation></semantics></math></div>')


def _short(label:str) -> str:
  return next((s for p, s in SHORT_LINE if label.startswith(p)), label)


def _ties(loss:dict[str, Any], results:dict[str, Any], m:dict[str, Any]) -> str:
  """The equations behind section 1 and the bar in section 2."""
  t = loss.get("tie_out") or {}
  limit1 = loss.get("limit_ms")
  if not limit1 or not m.get("ms") or not m.get("tok_s"):
    return ""
  inputs = {i.get("what", ""): i.get("value") for i in (loss.get("layout") or {}).get("inputs") or []}
  ceil = results.get("ceiling") or {}
  nbytes = inputs.get("weight bytes per token") or ceil.get("bytes_moved")
  bw = next((v for k, v in inputs.items() if k.endswith("read bandwidth")), None) or ceil.get("peak_bandwidth_gbs")
  eqs = [_eq(f"t_{{token}} = \\frac{{1000}}{{{m['tok_s']:.2f}\\ \\text{{tok/s}}}} = {m['ms']:.1f}\\ \\text{{ms}}",
             _t("token") + f"<mo>=</mo><mfrac><mn>1000</mn><mrow>{_mn(m['tok_s'], '{:.2f}')}{_u('tok/s')}</mrow></mfrac>"
             f"<mo>=</mo>{_mn(m['ms'])}{_u('ms')}")]
  if nbytes and bw:
    eqs.append(_eq(f"t_{{ideal}}(1) = \\frac{{{nbytes / 1e9:.2f}\\ \\text{{GB}}}}{{{bw:.1f}\\ \\text{{GB/s}}}} = {limit1:.1f}\\ \\text{{ms}}",
                   _t("ideal", "1") + f"<mo>=</mo><mfrac><mrow>{_mn(nbytes / 1e9, '{:.2f}')}{_u('GB')}</mrow>"
                   f"<mrow>{_mn(bw)}{_u('GB/s')}</mrow></mfrac><mo>=</mo>{_mn(limit1)}{_u('ms')}"))
  ctx = f'{t["context"]:.0f}' if t.get("context") is not None else None
  if ctx and t.get("limit_ms") is not None and t.get("kv_ms") is not None:
    eqs.append(_eq(f"t_{{ideal}}({ctx}) = t_{{ideal}}(1) + t_{{KV}} = {limit1:.1f} + {t['kv_ms']:.1f} = {t['limit_ms']:.1f}\\ \\text{{ms}}",
                   _t("ideal", ctx) + "<mo>=</mo>" + _t("ideal", "1") + "<mo>+</mo>" + _t("KV")
                   + f"<mo>=</mo>{_mn(limit1)}<mo>+</mo>{_mn(t['kv_ms'])}<mo>=</mo>{_mn(t['limit_ms'])}{_u('ms')}"))
  lines = t.get("lines") or []
  same = t.get("token_ms") is not None and abs(t["token_ms"] - m["ms"]) < 1e-9
  other = "" if same or t.get("token_ms") is None else (
    f'The bar below splits a different token: {t["token_ms"]:.1f} ms, {t.get("token_source") or "the tie-out token"}.')
  if same and lines and not t.get("refused"):
    first, rest = lines[0], lines[1:]
    head = [("ideal", first["ms"])] + [(_short(l["label"]), l["ms"]) for l in rest]
    eqs.append(_eq(f"t_{{token}} = t_{{ideal}}" + (f"({ctx})" if ctx else "") + " + " + " + ".join(f"\\text{{{n}}}" for n, _ in head[1:]),
                   _t("token") + "<mo>=</mo>" + _t("ideal", ctx or "") + "".join(f'<mo>+</mo><mtext>{_e(n)}</mtext>' for n, _ in head[1:])))
    tex = f"{t['token_ms']:.1f} = " + " + ".join(f"\\underbrace{{{v:.1f}}}_{{\\text{{{n}}}}}" for n, v in head)
    mml = f"{_mn(t['token_ms'])}<mo>=</mo>" + "<mo>+</mo>".join(
      _brace(_mn(v), f"{n} ({ctx})" if i == 0 and ctx else n) for i, (n, v) in enumerate(head)) + _u("ms")
    eqs.append(_eq(tex, mml))
  pct = 100.0 * limit1 / m["ms"]
  eqs.append(_eq(f"\\text{{lost}} = t_{{token}} - t_{{ideal}}(1) = {m['ms']:.1f} - {limit1:.1f} = {m['lost_ms']:.1f}\\ \\text{{ms}}",
                 f'<mtext>lost</mtext><mo>=</mo>{_t("token")}<mo>&#x2212;</mo>{_t("ideal", "1")}<mo>=</mo>'
                 f"{_mn(m['ms'])}<mo>&#x2212;</mo>{_mn(limit1)}<mo>=</mo>{_mn(m['lost_ms'])}{_u('ms')}"))
  eqs.append(_eq(f"\\text{{roofline share}} = \\frac{{t_{{ideal}}(1)}}{{t_{{token}}}} = \\frac{{{limit1:.1f}}}{{{m['ms']:.1f}}} = {pct:.0f}\\%",
                 f'<mtext>roofline share</mtext><mo>=</mo><mfrac>{_t("ideal", "1")}{_t("token")}</mfrac><mo>=</mo>'
                 f"<mfrac>{_mn(limit1)}{_mn(m['ms'])}</mfrac><mo>=</mo><mn>{pct:.0f}</mn><mo>%</mo>"))
  lead = (f"The headline is the time for one token. {m['tok_s']:.2f} tok/s rounds to {m['tok_s']:.1f}. "
          "The limit is the weight bytes over the read bandwidth. The bar below splits the same token.")
  if other:
    lead = lead.replace(" The bar below splits the same token.", "")
  tail = f'<p class="muted">{_e(other)}</p>' if other else ""
  return f'<div class="ties"><h3>How this ties out</h3><p class="muted" style="margin:0 0 6px">{_e(lead)}</p>{"".join(eqs)}{tail}</div>'


def _answer(manifest:dict[str, Any], results:dict[str, Any], measure:dict[str, Any] | None, source_run:str) -> str:
  loss = results.get("loss") or {}
  engine = results.get("engine") or {}
  tie = loss.get("tie_out") or {}
  who = (f'<p class="who"><b>{_e(manifest.get("model_id") or "unknown model")}</b> on {_e(manifest.get("target_id") or "unknown chip")}'
         f' · {_e(engine.get("provider") or loss.get("provider") or "unknown engine")} · batch {_e(tie.get("batch") or 1)}'
         f' · {_e(engine.get("weight_format") or "unknown weights")}</p>')
  limit_tok, limit_ms = loss.get("limit_tok_s"), loss.get("limit_ms")
  if limit_tok is None and (results.get("ceiling") or {}).get("status") == "modeled":
    limit_tok = results["ceiling"].get("tok_s")
  lim = f'<span class="lim">limit {_f(limit_tok, "{:.1f}")} <small>tok/s</small></span>' if limit_tok else ""
  m = _measured(loss)
  if not m:
    reason = (measure or {}).get("reason") or (f"the measure step ended {measure['status']}" if (measure or {}).get("status")
                                                else "the run never reached the measure step")
    cmd = COMMAND.format(model=manifest.get("model_path") or "MODEL", run=source_run or "RUN",
                         target=manifest.get("target_id") or "TARGET")
    return _section(1, "The answer", who + f'<div class="hero"><span class="big dim">not measured yet</span>{lim}</div>'
                    f'<p class="note">Missing: a measured token ({_e(reason)}). Measure it with <code>{_e(cmd)}</code>.</p>')
  pct = 100.0 * limit_ms / m["ms"] if limit_ms and m.get("ms") else None
  gauge = (f'<div class="gauge" role="img" aria-label="{pct:.0f}% of roofline"><i style="width:{_pct(pct, 100):.1f}%"></i><s></s></div>'
           f'<div class="gauge-l"><span><b>{pct:.0f}%</b> of roofline</span><span>roofline {limit_ms:.1f} ms per token</span></div>'
           if pct is not None else "")
  step = loss.get("step") or {}
  also = "".join(f'<p class="muted">Also measured at context {a["context"]}: {a["tok_s"]:.1f} tok/s. Not the headline: the headline '
                 f'is the row the tie-out uses ({_e(step.get("rule") or "")}).</p>' for a in step.get("also") or [])
  return _section(1, "The answer", who +
                  f'<div class="hero"><span class="big">{m["tok_s"]:.1f}</span><span class="big-u">tok/s measured</span>{lim}</div>'
                  + gauge +
                  f'<p class="note">{m["tok_s"]:.1f} tok/s = {m["ms"]:.1f} ms per token.</p>'
                  f'<p class="note"><span class="lost">{m["lost_ms"]:.1f} ms per token lost</span> against the limit: '
                  f'{m["ms"]:.1f} ms measured, {_ideal_words(loss, limit_ms)}. <span class="muted">{_e(m.get("note") or "")}</span></p>'
                  + also + _ties(loss, results, m))


def _ideal_words(loss:dict[str, Any], limit_ms:float) -> str:
  """The ideal once, at context 1 and, when the bar uses another context, there too with the KV read."""
  t = loss.get("tie_out") or {}
  if t.get("limit_ms") is None or t.get("context") is None or f'{t["limit_ms"]:.1f}' == f"{limit_ms:.1f}":
    return f"{limit_ms:.1f} ms ideal"
  return f'{limit_ms:.1f} ms ideal at context 1, {t["limit_ms"]:.1f} ms at context {t["context"]:.0f} with the KV read'


def _seg_class(line:dict[str, Any]) -> str:
  if line["how"] == "derived": return "c-ideal"
  if line["how"] == "difference": return "c-gaps"
  return "c-excess" if line["label"].startswith(("weight", "all kernels")) else "c-other"


def _line_label(line:dict[str, Any]) -> str:
  return _other_label(line) + " above their ideal" if line.get("parts") is not None and line["label"].startswith("other") else line["label"]


def _where(loss:dict[str, Any]) -> str:
  t = loss.get("tie_out") or {}
  if t.get("refused"):
    return _section(2, "Where the time goes", f'<p class="note">Not tied out. {_e(t["refused"])}</p>')
  if t.get("token_ms") is None or not t.get("lines"):
    return ""
  token = t["token_ms"]
  words = t.get("band")
  frac = loss.get("band") if isinstance(loss.get("band"), (int, float)) else None
  if frac is None:  # the band fraction is not in the seam's tie-out; the ±X% is in its words
    got = re.search(r"±([0-9.]+)%", str(words or ""))
    frac = float(got.group(1)) / 100 if got else None
  segs, rows = [], []
  for i, l in enumerate(t["lines"]):
    cls = _seg_class(l)
    w = _pct(max(l["ms"], 0.0), token)
    text = f'roofline {l["ms"]:.1f}' if i == 0 and l["how"] == "derived" else f'{l["ms"]:.1f}'
    segs.append(f'<span class="{cls}" style="width:{w:.2f}%" title="{_e(_line_label(l))}">{text}</span>')
    tag = f' <span class="sub" style="display:inline">±{frac * 100:.1f}%</span>' if i == 0 and frac else ""
    rows.append(f'<tr><td><span class="sw {cls}"></span>{"" if i == 0 else "+ "}{_e(_line_label(l))}{tag}</td>'
                f'<td>{l["ms"]:.3f}</td><td>{_pct(l["ms"], token):.0f}%</td><td class="how">{_e(l["how"])}</td></tr>')
  band = ""
  if frac:
    ideal = t["lines"][0]["ms"]
    band = (f'<span class="band" style="left:{_pct(ideal * (1 - frac), token):.2f}%;'
            f'width:{_pct(2 * ideal * frac, token):.2f}%" title="band ±{frac * 100:.1f}%"></span>')
  m = _measured(loss)
  same = m is not None and m.get("tok_s") and abs(m["ms"] - token) < 1e-9
  total = f'= measured token (the {m["tok_s"]:.1f} tok/s above)' if same else "= measured token"
  rows.append(f'<tr class="sum"><td>{_e(total)}</td><td>{token:.3f}</td><td>100%</td>'
              f'<td class="how">{_e(t.get("token_source") or "")}</td></tr>')
  notes = []
  if t.get("untraced_ms") is not None and str(t.get("token_source", "")).startswith("the captured run"):
    notes.append(f'Tracing slowed the token: {token:.3f} ms here, {t["untraced_ms"]:.3f} ms untraced.')
  for l in t["lines"]:
    if l.get("kernels"):  # GEMVs no role took: named, with their launches; the capture carries no bytes for them
      notes.append("Weight kernels no role took (the limit has no bytes for them): " + "; ".join(
        f'{k["kernel"]}, {k["calls_per_token"]:.0f} per token, {k["us_per_call"]:.1f} µs each' for k in l["kernels"]) + ".")
  if est := t.get("estimate"):
    up, least = ("up to ", "at least ") if est["scaled"] else ("", "")
    floor = f', {est["floor_words"]}' if est.get("floor_words") else ''
    notes.append(f'Estimated split (isolated): weight kernels {up}{est["weight_ms"]:.1f} ms, other and gaps {least}'
                 f'{est["other_ms"]:.1f} ms ({_e(est["other_how"])}). Not measured in the token: each weight kernel was timed alone{_e(floor)}'
                 + (f', and the times are scaled by {est["scale"]:.3f} to fit the token less the KV-read floor.' if est["scaled"] else '.'))
  if t.get("show_both"):
    notes.append(f'The limit is {t["limit_ms_ctx1"]:.3f} ms at context 1 and {t["limit_ms"]:.3f} ms at context {t["context"]:.0f}.')
  body = (f'<p class="muted" style="margin:0 0 10px">ms per token at context {t["context"]:.0f}</p>'
          f'<div class="stack">{"".join(segs)}{band}</div>'
          f'<table class="tie"><tbody>{"".join(rows)}</tbody></table>'
          + "".join(f'<p class="muted">{_e(n)}</p>' for n in notes))
  return _section(2, "Where the time goes", body)


def _where_token(loss:dict[str, Any]) -> str:
  """tie_out.where_token_goes as a table: what each part costs the token, worst first. The numbers are the table's."""
  w = loss.get("where_token_goes")
  if not w:
    return ""
  head = "".join(f"<th>{_e(c)}</th>" for c in w["columns"])
  rows = []
  for r in w["rows"]:
    at = f'{r["limit_ms"]:.2f}' if r["limit_ms"] is not None else _e(w["no_limit"])
    rows.append(f'<tr><td>{_e(r["name"])}</td><td>{r["now_ms"]:.2f}</td><td>{at}</td><td class="lost">{r["lost_ms"]:.2f}</td>'
                f'<td>{_f(None if r["share"] is None else 100 * r["share"], "{:.0f}%")}</td><td>{_f(r["tok_s_if_fixed"], "{:.1f}")}</td>'
                f'<td>{_f(r["pct_peak"], "{:.1f}%")}</td></tr>')
  rows.append(f'<tr class="sum"><td>= token</td><td>{w["token_ms"]:.2f}</td><td>{w["limit_ms"]:.2f}</td><td>{w["lost_ms"]:.2f}</td>'
              f'<td>100%</td><td></td><td></td></tr>')
  body = (f'<p class="muted" style="margin:0 0 10px">{_e(w["words"])}</p>'
          f'<div class="tscroll"><table class="tie where"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
  return _section(2, w["title"] + (" (estimate)" if w["estimate"] else ""), body)


def _other_line(t:dict[str, Any]) -> dict[str, Any] | None:
  return next((l for l in t.get("lines") or [] if l.get("parts") is not None), None)


def _per_role(loss:dict[str, Any]) -> str:
  if loss.get("refused"):
    return _section(3, "Per role", f'<p class="note">Per role: not shown. {_e(loss["refused"])}</p>')
  est = loss.get("estimate")
  lost = (lambda r: r["est_lost_ms"]) if est else (lambda r: r["lost_ms"])
  roles = sorted(loss.get("roles") or [], key=lambda r: (-lost(r), str(r["role"]), str(r["quant"])))
  if not roles:
    return ""
  items = []
  for r in roles:
    why = r.get("reason") or ""  # shown as it is; the class, the colour and the rules key on the firm word
    word = r.get("reason_word") or why
    floor = bool(est and est.get("columns_words"))  # the rows carry a dispatch floor: the rate columns are less it
    rate = " (less floor)" if floor else " (isolated)"
    nums = (("bytes/call", _f(r.get("mb_per_call"), "{:.1f} MB")),
            ("µs/call (measured, floor included)" if floor else "µs/call (isolated)", _f(r.get("us_per_call"), "{:.1f}")))
    nums += (("µs/call less floor", _f(r.get("us_per_call_less_floor"), "{:.1f}")),) if floor else ()
    nums = nums + (("GB/s" + rate, _f(r.get("gbs"), "{:.1f}")), ("% of peak" + rate, _f(r.get("pct_peak"), "{:.1f}%")),
            ("ideal ms", f'{r["ideal_ms"]:.2f}'), ("est. ms in token", f'{r["est_ms"]:.2f}'),
            ("est. lost ms", f'{r["est_lost_ms"]:.2f}')) if est else (
      ("ideal ms", f'{r["ideal_ms"]:.2f}'), ("actual ms", f'{r["actual_ms"]:.2f}'),
      ("µs/call", _f(r.get("us_per_call"), "{:.1f}")), ("share of loss", f'{r["share"] * 100:.0f}%'))
    det = "".join(f'<div><span>{k}</span>{v}</div>' for k, v in nums + (
      ("best found", _e((r.get("best_found") or {}).get("text") or "none")),
      ("verdict", _e(PLAIN_VERDICT.get(str(r.get("verdict")), r.get("verdict") or "not searched"))
       + (f' ({r["candidates"]} plans searched)' if r.get("candidates") else ""))))
    if r.get("verdict_reason"):
      det += f'<p class="muted wrap">{_e(r["verdict_reason"])}</p>'
    det += _evidence(r.get("evidence"))
    found = (r.get("best_found") or {}).get("text")
    tag = (f'<span class="why {VERDICT_CLASS.get(str(r.get("verdict")), "")}">'
           f'{_e(PLAIN_VERDICT.get(str(r.get("verdict")), ""))}{" · " + _e(found) if found else ""}</span>') if r.get("verdict") else ""
    items.append(
      f'<details class="role"><summary><span class="rname">{_role_name(r["role"], r["quant"])}</span>'
      f'<span class="rbar" title="{_f(r.get("pct_peak"), "{:.1f}%")} of the roofline"><i style="width:{_pct(r.get("pct_peak") or 0.0, 100.0):.1f}%;background:{REASON_COLOR.get(word, "var(--ideal)")}"></i></span>'
      f'<span class="rnum"><b>{lost(r):.2f} ms{" est." if est else ""}</b><span>{_f(r.get("pct_peak"), "{:.1f}%")} of peak{rate if est else ""}</span>'
      f'<span class="rtags"><span class="why {REASON_CLASS.get(word, "")}">{_e(why)}</span>{tag}</span></span></summary><div class="rdet">{det}</div></details>')
  t = loss.get("tie_out") or {}
  if est:
    from boltbeam.workflow.screen import estimate_how
    columns = f' {_e(est["columns_words"])}.' if est.get("columns_words") else ""
    body = (f'<p class="note">{_e(est["method"])}</p>'
            f'<p class="muted" style="margin:0 0 10px">est. lost ms per token {_e(est["label"])}, {_e(estimate_how(est))}; '
            f'the bar is filled to the {"less-floor" if est.get("columns_words") else "isolated"} share of peak; open a row for its numbers.{columns}</p>')
  else:
    body = f'<p class="muted" style="margin:0 0 10px">{_e(loss.get("source") or "")} · lost ms per token, longest first; the bar is filled to the share of the roofline this role reaches; the empty part is its loss; open a row for its numbers</p>'
  body += f'<div class="roles">{"".join(items)}</div>'
  search = loss.get("search") or {}
  if search.get("status") == "skipped":
    body += f'<p class="note">Kernel search: skipped. {_e(search.get("reason") or "")}</p>'
  elif search.get("status") == "searched" and search.get("seconds") is not None:
    body += f'<p class="muted">Kernel search: {search["seconds"]:.0f} s in this Run. Speedup is the best plan alone against the model\'s own kernel per call.</p>'
  if loss.get("not_attributed_ms") is not None:
    body += (f'<p class="note">{_e(_other_label(_other_line(t)))}: {loss["not_attributed_ms"]:.2f} ms of kernel time per token.</p>')
  unsplit = loss.get("unpaired_roles") or []
  if unsplit:
    names = ", ".join(f'{PLAIN_ROLE.get(str(u.get("role")), u.get("role"))} {u.get("quant") or ""}'.strip() for u in unsplit)
    body += f'<p class="note">Roles that could not be split: {_e(names)}. Their time is inside other kernels.</p>'
  for n in loss.get("not_timed") or []:  # a limit role the timer had no adapter for: its own row, out of the floor rule and the sums
    body += (f'<p class="note">{_role_name(n.get("role"), n.get("quant"))}: not timed: {_e(n.get("reason") or n.get("status") or "")}. '
             f'Left out of the floor rule and the sums.</p>')
  if cc := loss.get("cross_check"):
    body += _rate_table("Cross-check", cc.get("words") or "", cc.get("rows") or [], cc.get("reason"), cc.get("columns_words"))
  return _section(3, "Per role, estimated from isolated kernel times" if est else "Per role", body)


def _rate_table(title:str, words:str, rows:list[dict[str, Any]], reason:str | None = None, columns_words:str | None = None) -> str:
  """A labelled row group of GB/s and share of peak per role: the probe's reference kernel, or an isolated
  cross-check. Rows that carry a time less the dispatch floor get that column, and columns_words says once which
  column the rates are on."""
  floor = any(r.get("us_per_call_less_floor") is not None for r in rows)
  less = (lambda r: f'<td>{_f(r.get("us_per_call_less_floor"), "{:.1f}")}</td>') if floor else (lambda r: "")  # noqa: E731
  cells = "".join(f'<tr><td>{_role_name(r["role"], r["quant"])}</td><td>{_f(r.get("us_per_call"), "{:.1f}")}</td>{less(r)}'
                  f'<td>{_f(r.get("gbs"), "{:.1f}")}</td><td>{_f(r.get("pct_peak"), "{:.0f}%")}</td></tr>'
                  for r in rows)
  head = f'<tr><td>role</td><td>µs/call{" (measured)" if floor else ""}</td>{"<td>less floor</td>" if floor else ""}<td>GB/s</td><td>of peak</td></tr>'
  table = f'<table class="tie"><tbody>{head}{cells}</tbody></table>' if rows else f'<p class="empty">{_e(reason or "no rows")}</p>'
  note = f'<p class="muted">{_e(columns_words)}.</p>' if columns_words and rows else ""
  return f'<div class="card"><div class="card-hd"><h2 class="card-ttl">{_e(title)}</h2><span class="card-sub">{_e(words)}</span></div>{table}{note}</div>'


def _probe(results:dict[str, Any]) -> str:
  """BoltBeam's own kernel (reference) per role: GB/s and share of peak. Clearly not the engine's kernel."""
  probe = results.get("probe") or {}
  if probe.get("status") != "measured":
    return ""
  rows = probe.get("rows") or []
  body = (f'<p class="muted" style="margin:0 0 10px">{_e(probe.get("label") or "")}. A plain correct GEMV on the model\'s real '
          f'bytes, timed alone after a cache flush: what this GPU reads for each role\'s shape, not what the engine reads.</p>'
          + _rate_table("BoltBeam's own kernel (reference)", "collectors/boltbeam_gemv.py through the kernel timer", rows))
  absent = probe.get("absent") or {}
  if absent:
    body += f'<p class="muted">Not reported by this GPU: {_e("; ".join(sorted(set(absent.values()))))}.</p>'
  if probe.get("dispatch_floor_us") is not None:
    body += f'<p class="muted">Dispatch floor measured here: {probe["dispatch_floor_us"]:.1f} µs per launch.</p>'
  return _section(0, "Building blocks", body)


def next_items(loss:dict[str, Any], compare_runs:bool = True) -> list[dict[str, Any]]:
  """What to try next, from NEXT_RULES and the line rules; each item restates measured ms and its share of the
  token. Sorted by ms, largest first."""
  t = loss.get("tie_out") or {}
  token = t.get("token_ms")
  if not token or t.get("refused"):
    return []
  out = []
  groups: dict[str, list[dict[str, Any]]] = {}
  roles = [{**r, "lost_ms": r["est_lost_ms"]} if r.get("estimate") else r for r in loss.get("roles") or []]
  for r in roles:
    word = r.get("reason_word") or r.get("reason")  # the firm word; a noisy suffix never changes the lever
    if word in NEXT_RULES and r["lost_ms"] > 0:
      groups.setdefault(word, []).append(r)
  for reason, rs in groups.items():
    rs = sorted(rs, key=lambda r: -r["lost_ms"])
    # a slow kernel already searched with nothing faster found gets its own item: the search result is the fact
    searched = [r for r in rs if reason == "slow kernel" and r.get("verdict") == "none_faster"]
    for part, lever in ((searched, None), ([r for r in rs if r not in searched], NEXT_RULES[reason])):
      if not part:
        continue
      ms = sum(r["lost_ms"] for r in part)
      if lever is None:
        counts = [int(r.get("candidates") or 0) for r in part if r.get("candidates")]
        n = (f"{counts[0]} plans per role" if len(set(counts)) == 1 and len(counts) > 1 else f"{sum(counts)} plans") if counts else "the kernel search"
        lever = SEARCHED_LEVER.format(n=n)
      out.append({"what": f"{reason.capitalize()}: " + ", ".join(
        f'{PLAIN_ROLE.get(r["role"], r["role"])} {r["quant"]} {r["lost_ms"]:.2f} ms' for r in part),
        "ms": ms, "share": ms / token, "do": lever, "rule": reason if part is not searched else f"{reason}, searched: none faster",
        "evidence": _dedupe([p for r in part for p in r.get("evidence") or []])})
  ev = loss.get("evidence") or {}
  found_verdicts = {f.get("verdict") for f in loss.get("findings") or []}
  out += [dict(f) for f in loss.get("findings") or []]
  if (found := (loss.get("search") or {}).get("next")) and found.get("ms", 0) > 0:
    out.append({"what": found["what"], "ms": found["ms"], "share": found["ms"] / token, "do": found["do"],
                "rule": found.get("rule") or "a faster kernel found", "evidence": found.get("evidence") or []})
  for l in t.get("lines") or []:
    if l["how"] == "difference" and l["label"].startswith("gaps between kernels (GPU idle)") and l["ms"] >= GAPS_SHARE * token \
        and "launch_heavy" not in found_verdicts:
      out.append({"what": "Gaps between kernels (GPU idle)", "ms": l["ms"], "share": l["ms"] / token, "do": GAPS_LEVER,
                  "rule": f"gaps ≥ {GAPS_SHARE:.0%} of the token", "evidence": ev.get("whole_step") or []})
    if l.get("parts") is not None and l["ms"] >= OTHER_SHARE * token:
      parts = l.get("parts") or []
      fuse = [p["kind"] for p in parts if p["kind"] in FUSIBLE]
      lead = parts[0]["kind"] if parts else None
      do = (OTHER_LEVER_FUSE.format(kinds=", ".join(fuse)) if fuse and lead in FUSIBLE else
            OTHER_LEVER_KERNEL.format(kind=lead) if lead else OTHER_LEVER_FUSE.format(kinds="them"))
      if fuse and lead not in FUSIBLE and lead:
        do = OTHER_LEVER_KERNEL.format(kind=lead) + " " + OTHER_LEVER_FUSE.format(kinds=", ".join(fuse))
      out.append({"what": _other_label(l) + " above their ideal", "ms": l["ms"], "share": l["ms"] / token, "do": do,
                  "rule": f"other kernels ≥ {OTHER_SHARE:.0%} of the token", "evidence": ev.get("other_kernels") or []})
    if l["label"] == "kernels and gaps, not split" and loss.get("missing"):
      out.append({"what": "Kernels and gaps, not split", "ms": l["ms"], "share": l["ms"] / token,
                  "do": SPLIT_LEVER.format(missing=loss["missing"]), "rule": "no per-role time",
                  "evidence": (ev.get("whole_step") or []) + (ev.get("common") or [])})
    if l["how"] == "difference" and l["label"] == "kernels and gaps, not split" and loss.get("estimate") and l["ms"] > 0:
      out.append({"what": "Kernels and gaps above the ideal, not split (isolated timing)", "ms": l["ms"], "share": l["ms"] / token,
                  "do": NOT_TIMED_LEVER, "rule": "isolated kernels: the token minus the ideal is one difference",
                  "evidence": ev.get("whole_step") or []})
  return sorted(out, key=lambda x: (-x["ms"], x["what"]))


def _next(loss:dict[str, Any], policy:dict[str, Any]) -> str:
  items = next_items(loss)
  routes = [r for r in policy.get("routes", []) or [] if isinstance(r, dict)]
  compared = any(r.get("status") not in (None, "unmeasured") and isinstance(r.get("compare"), dict) for r in routes)
  if not items and not compared:
    return ""
  lis = "".join(
    f'<li class="{"top" if i == 0 else ""}"><span class="what">{_e(x["what"])}</span> '
    f'<span class="ms">· {x["ms"]:.2f} ms, {x["share"] * 100:.0f}% of the token</span>'
    f'<span class="do">{_e(x["do"])}</span>{_evidence(x.get("evidence"))}</li>' for i, x in enumerate(items))
  body = f'<ol class="next">{lis}</ol>' if lis else ""
  if compared:
    body += _routes_card(policy)
  return _section(4, "What to try next", body)


def _stage_list(manifest:dict[str, Any], measure:dict[str, Any] | None) -> str:
  stages = manifest.get("stages", {}) or {}
  known = [k for k, _, _ in STAGES]
  keys = known + sorted(k for k in stages if k not in known)
  out = []
  for key in keys:
    state, _ = stage_state(key, stages.get(key), measure)
    if state != "done":
      continue
    n = len((stages.get(key) or {}).get("artifacts", []) or [])
    out.append(f'<span class="chip">{_e(PLAIN_STAGE.get(key, key))} · {n} file{"" if n == 1 else "s"}</span>')
  return "".join(out) or '<span class="chip">no stage ran</span>'


def _facts(manifest:dict[str, Any], results:dict[str, Any], measure:dict[str, Any] | None, gameplan:str | None = None) -> str:
  loss = results.get("loss") or {}
  ceil = results.get("ceiling") or {}
  machine = loss.get("machine") or {}
  rows = []
  bw = ceil.get("peak_bandwidth_gbs")
  if bw is not None:
    src = ceil.get("bandwidth_source") or "the chip registry"
    rows.append(("Chip profile", f'{manifest.get("target_id")}: {bw:.1f} GB/s read bandwidth, {src}'
                 + (f', machine facts {machine["measured_at"]}' if machine.get("measured_at") else "")))
  if (loss.get("layout") or {}).get("formula"):
    rows.append(("Limit", f'{loss["layout"]["formula"]} ({loss["layout"].get("label") or ""})'))
  elif ceil.get("bytes_moved"):
    rows.append(("Limit", f'{ceil["bytes_moved"] / 1e9:.2f} GB of weights per token over {bw:.1f} GB/s'))
  if w := ceil.get("weights"):  # every tensor counted or excluded with a reason (profile/weight_ledger.py)
    rows.append(("Weights", w["line"]))
    for u in w["unclassified"]:
      rows.append(("Not classified, counted", f'{u["pattern"]} {u["quant"]} x{u["tensors"]}: {u["bytes"] / 1e9:.3f} GB per token'))
    if w["excluded"]:
      rows.append(("Excluded from the limit", "; ".join(f'{x["pattern"]} x{x["tensors"]} {x["bytes"] / 1e6:.2f} MB ({x["reason"]})'
                                                      for x in w["excluded"])))
  step = loss.get("step") or {}
  if step:
    graph = ""
    if step.get("graph_failed"):
      graph = " Graph replay failed on this run" + (f' ({step["graph_error"]})' if step.get("graph_error") else "") + ": the token ran without graphs."
    rows.append(("Measured token", f'context {step["context"]}, {step["tok_s"]:.1f} tok/s ({step.get("rule") or ""}); '
                 f'{step.get("source") or "step 4"}.{graph}'))
  cap = loss.get("capture") or {}
  if cap.get("method"):
    from boltbeam.workflow.screen import CAPTURE_WORDS
    rows.append(("Capture", CAPTURE_WORDS.get(cap["method"], cap["method"]) + (f'; {cap["reason"]}' if cap.get("reason") else "")))
  elif loss.get("runtimes"):
    rows.append(("Capture", "whole step only, untraced" + (f'; per role: {loss["missing"]}' if loss.get("missing") else "")))
  if loss.get("role_source_words"):
    rows.append(("Per-role source", loss["role_source_words"]))
  from boltbeam.workflow.screen import measurement_words
  if words := measurement_words(results.get("measurement")):
    rows.append(("Measurement", words.removeprefix("Measurement: ")))
  if lat := loss.get("latency"):
    rows.append(("Latency in the reason rule", f'{lat["us"]:.1f} µs, {lat["source"]}'))
  if (loss.get("tie_out") or {}).get("kv_source"):
    rows.append(("KV cache", loss["tie_out"]["kv_source"]))
  if measure and measure.get("status"):
    rows.append(("Measure step", str(measure["status"]) + (f': {measure["reason"]}' if measure.get("reason") else "")))
  for o in loss.get("others") or []:
    label = o.get("provider") + (f' (run {o["run"]})' if o.get("run") else "")
    rows.append(("Also measured", f'{label}: ' + (f'{o["tok_s"]:.1f} tok/s' if o.get("tok_s") else str(o.get("missing") or "no speed"))))
  dl = "".join(f"<dt>{_e(a)}</dt><dd>{_e(b)}</dd>" for a, b in rows)
  files = "".join(f'<a class="chip" href="{_e(a)}">{_e(a)}</a>' for a in manifest.get("artifacts", []) or [])
  dl += f'<dt>Stages that ran</dt><dd><div class="chips">{_stage_list(manifest, measure)}</div></dd>'
  dl += f'<dt>Files in the run</dt><dd><div class="chips">{files or "none listed"}</div></dd>'
  if gameplan:  # Emit wrote it (workflow/gameplan.py): per role, worst first, what kernel to emit, from this run's files
    dl += f'<dt>Gameplan</dt><dd><a href="{_e(gameplan)}">{_e(gameplan)}</a>: what to emit per role, worst first (Emit)</dd>'
  return _section(5, "Facts", f"<dl>{dl}</dl>")


def render_run_html(*, manifest:dict[str, Any], profile:dict[str, Any], report:dict[str, Any],
                    plan:dict[str, Any], policy:dict[str, Any], providers:dict[str, Any],
                    primitive:dict[str, Any], timing:dict[str, Any], runner:dict[str, Any],
                    source_run:str = "", results:dict[str, Any] | None = None,
                    measure:dict[str, Any] | None = None, gameplan:str | None = None) -> str:
  """Render one run as a standalone HTML document from `results` (workflow.screen.results). The other arguments
  are the run's artifacts, `{}` when absent; the page reads only the manifest and the policy from them. gameplan is
  the gameplan file's name when Emit wrote one into the run; the Facts link it."""
  results = results or {}
  loss = results.get("loss") or {}
  model_id = manifest.get("model_id") or profile.get("model_id") or "unknown"
  measured = _measured(loss) is not None
  sections = _answer(manifest, results, measure, source_run)
  if measured:
    sections += _where(loss) + _where_token(loss) + _per_role(loss) + _next(loss, policy)
  sections += _probe(results) + _facts(manifest, results, measure, gameplan)
  sections = _number(sections)
  return (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    f"<title>BoltBeam {_e(model_id)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">"
    f'<header class="mast"><div class="mark">BoltBeam</div><div class="run-id">{_e(source_run or model_id)}</div>'
    '<button id="theme" class="toggle" type="button">theme</button></header>'
    f"{sections}"
    '<p class="foot">Written by <code>boltbeam output</code> from the same results the TUI shows. Times are ms per token.</p>'
    f"</div><script>{_JS}</script></body></html>\n")
