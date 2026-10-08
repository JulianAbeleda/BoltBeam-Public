"""Self-contained HTML run report (companion to `summary.md`).

`_summary_md` in workflow/output.py renders the same run-directory facts as markdown; this renders them as one
standalone HTML file you can open, scp, or attach to a handoff without a server, a build step, or a network
fetch. Nothing here computes new facts: every number on the page already exists in a staged artifact, and every
section names the artifact it came from, so a reader can always go from a claim back to its file.

Two properties are load-bearing, both inherited from report/markdown.py:
  * deterministic — stable ordering, no timestamps, no randomness, so two runs over the same run directory are
    byte identical and the report can be committed / diffed;
  * absence is rendered, not omitted — a stage with no artifact, a plan blocked on missing evidence, and a
    route with no measurement all get visible rows. A report that silently drops what it lacks reads as
    complete when it is not.

The CSS/JS are inlined deliberately. A report that needs a CDN is a report that renders blank on the machine
you actually want to read it on.
"""
from __future__ import annotations

import html
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

_CSS = """
/* Tokyo Night, matching the BoltBeam run graph: dark canvas, panel cards, typed accent colours.
   Light mode maps onto Tokyo Night Day rather than inverting the dark palette.
   Type is a system UI/mono stack — no @font-face, so this file issues no external requests. */
:root{--ui:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--canvas:#1a1b26;--grid:rgba(192,202,245,.055);--grid-2:rgba(192,202,245,.10);
--node:#24283b;--node-hd:#1f2335;--node-br:#414868;--widget:#1a1b26;--widget-br:#3b4261;
--tx:#c0caf5;--tx-2:#a9b1d6;--tx-3:#565f89;
--profile:#bb9af7;--plan:#7aa2f7;--evidence:#9ece6a;--trace:#ff9e64;--policy:#f7768e;
--run:#9ece6a;--warn:#e0af68;--bad:#db4b4b}
@media (prefers-color-scheme:light){:root{--canvas:#e1e2e7;--grid:rgba(55,96,191,.09);--grid-2:rgba(55,96,191,.14);
--node:#e9e9ec;--node-hd:#d5d6db;--node-br:#a8aecb;--widget:#d5d6db;--widget-br:#c4c8da;
--tx:#3760bf;--tx-2:#6172b0;--tx-3:#848cb5;--profile:#9854f1;--plan:#2e7de9;--evidence:#587539;
--trace:#b15c00;--policy:#f52a65;--run:#587539;--warn:#8c6c3e;--bad:#c64343}}
:root[data-theme=light]{--canvas:#e1e2e7;--grid:rgba(55,96,191,.09);--grid-2:rgba(55,96,191,.14);--node:#e9e9ec;
--node-hd:#d5d6db;--node-br:#a8aecb;--widget:#d5d6db;--widget-br:#c4c8da;--tx:#3760bf;--tx-2:#6172b0;
--tx-3:#848cb5;--profile:#9854f1;--plan:#2e7de9;--evidence:#587539;--trace:#b15c00;--policy:#f52a65;
--run:#587539;--warn:#8c6c3e;--bad:#c64343}
:root[data-theme=dark]{--canvas:#1a1b26;--grid:rgba(192,202,245,.055);--grid-2:rgba(192,202,245,.10);
--node:#24283b;--node-hd:#1f2335;--node-br:#414868;--widget:#1a1b26;--widget-br:#3b4261;--tx:#e9e9e9;
--tx-2:#a8a8a8;--tx-3:#767676;--profile:#b39ddb;--plan:#64b5f6;--evidence:#81c784;--trace:#ff8a65;
--policy:#f06292;--run:#7ec86a;--warn:#e0a33e;--bad:#e5534b}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;min-height:100vh;background:var(--canvas);color:var(--tx);font-family:var(--ui);font-size:13px;
line-height:1.6;-webkit-font-smoothing:antialiased;padding:30px 20px 64px;
background-image:radial-gradient(var(--grid-2) 1px,transparent 1px),radial-gradient(var(--grid) 1px,transparent 1px);
background-size:100px 100px,20px 20px;background-attachment:fixed}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
.mast{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:2px 2px 6px}
.mark{font-weight:700;font-size:17px;letter-spacing:-.01em}
.mark span{color:var(--profile)}
.run-id{color:var(--tx-3);font-family:var(--mono);font-size:11.5px;word-break:break-all}
.meta{color:var(--tx-3);font-size:11.5px;display:flex;gap:16px;flex-wrap:wrap;margin-left:auto}
.meta b{color:var(--tx-2);font-weight:500;font-family:var(--mono)}
.card,.rail{background:var(--node);border:1px solid var(--node-br);border-radius:9px;
box-shadow:0 4px 16px rgba(6,8,18,.32);overflow:hidden}
.rail{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.stage{padding:10px 14px;border-right:1px solid var(--node-br);border-bottom:1px solid var(--node-br);
display:flex;flex-direction:column;gap:2px;margin:0 -1px -1px 0;min-width:0}
.stage-top{display:flex;align-items:center;gap:8px}
.stage-name{font-size:12.5px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;flex:none;background:var(--run)}
.stage-note{font-size:10.5px;color:var(--tx-3);padding-left:16px;font-family:var(--mono)}
.s-none .dot{background:transparent;border:1.5px solid var(--tx-3)}
.s-none .stage-name,.s-none .stage-note{color:var(--tx-3)}
.card-hd{padding:10px 15px;border-bottom:1px solid var(--node-br);background:var(--node-hd);
display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.card-ttl{margin:0;font-size:12.5px;font-weight:600}
.card-sub{font-size:10.5px;color:var(--tx-3);margin-left:auto;font-family:var(--mono)}
.card-bd{padding:14px 15px}
.card.flag{border-left:3px solid var(--warn)}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
.cols.one{grid-template-columns:minmax(0,1fr)}
@media (max-width:920px){.cols{grid-template-columns:1fr}.meta{margin-left:0}}
.tscroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tscroll table{min-width:640px}
.tscroll table.wide{min-width:900px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
thead th{font-size:9.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--tx-3);
text-align:right;padding:10px 12px 8px;border-bottom:1px solid var(--node-br);white-space:nowrap}
thead th:first-child,tbody td:first-child{text-align:left}
tbody td{padding:8px 12px;border-bottom:1px solid color-mix(in srgb,var(--node-br) 55%,transparent);
text-align:right;font-size:12px;color:var(--tx-2);font-family:var(--mono);white-space:nowrap;vertical-align:middle}
th.l,td.l{text-align:left!important}
td.wrap{white-space:normal;min-width:280px;font-family:var(--ui)}
tbody tr:last-child td{border-bottom:0}
tbody td:first-child{color:var(--tx)}
tbody tr:hover td{background:var(--node-hd)}
.name{max-width:32ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pct{position:relative}
.pct i{position:absolute;left:6px;right:6px;bottom:2px;height:2px;background:var(--widget-br);border-radius:2px}
.pct i b{position:absolute;left:0;top:0;bottom:0;background:var(--plan);border-radius:2px}
.tag{font-size:10.5px;padding:2px 9px;border-radius:14px;white-space:nowrap;color:var(--tx-2);
background:var(--widget);border:1px solid var(--widget-br);font-family:var(--mono)}
.t-bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 45%,transparent)}
.t-warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.t-ok{color:var(--run);border-color:color-mix(in srgb,var(--run) 45%,transparent)}
ul{margin:0;padding-left:17px;color:var(--tx-2)}
li{padding:3px 0}
.need{display:grid;grid-template-columns:26px 1fr;gap:12px;align-items:start;padding:5px 0}
.need-k{color:var(--warn);font-family:var(--mono);font-size:11.5px}
.need-b{color:var(--tx-2);font-size:12.5px;line-height:1.7}
code{background:var(--widget);border:1px solid var(--widget-br);padding:1px 6px;border-radius:4px;
color:var(--tx);font-family:var(--mono);font-size:11px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{font-size:10.5px;padding:3px 10px;border-radius:14px;background:var(--widget);color:var(--tx-2);
border:1px solid var(--widget-br);font-family:var(--mono);text-decoration:none}
a.chip:hover{border-color:var(--plan);color:var(--tx)}
.chip.on{color:var(--run);border-color:color-mix(in srgb,var(--run) 45%,transparent)}
a{color:var(--plan)}
.id{font-family:var(--mono);font-size:10px;color:var(--tx-3)}
.lbl{font-size:9.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--tx-3);margin:0 0 6px}
.lbl .id{letter-spacing:0;text-transform:none;font-weight:400}
.compact .card-bd{padding:9px 15px}
.compact .empty{padding:9px 15px}
.head-bd{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr);gap:18px;padding:18px 20px;align-items:start}
.head-model{font-size:22px;font-weight:700;letter-spacing:-.01em;word-break:break-word}
.head-chip{color:var(--tx-2);font-family:var(--mono);font-size:12px}
.big{font-size:30px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.big.dim{font-size:18px;color:var(--warn)}
.big-u{color:var(--tx-2);font-size:12px}
.big-n{color:var(--tx-3);font-size:11px;margin-top:4px}
.head{border-left:3px solid var(--profile)}
@media (max-width:640px){.head-bd{grid-template-columns:1fr 1fr}.head-who{grid-column:1/-1}body{padding:16px 16px 48px}}
details{border-bottom:1px solid var(--node-br)}
details:last-child{border-bottom:0}
summary{padding:11px 15px;cursor:pointer;display:flex;gap:11px;align-items:center;flex-wrap:wrap}
summary:hover{background:var(--widget)}
summary:focus-visible{outline:2px solid var(--plan);outline-offset:-2px}
summary .rb-name{font-size:13px;font-weight:600;color:var(--tx)}
pre{margin:0 15px 14px 32px;padding:11px 13px;background:var(--widget);border:1px solid var(--widget-br);
border-radius:7px;overflow-x:auto;font-size:11px;color:var(--tx-2);font-family:var(--mono)}
.empty{color:var(--tx-3);font-size:12px;padding:15px}
.foot{color:var(--tx-3);font-size:11px;text-align:center;padding-top:10px;line-height:1.85}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
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


def _e(value:Any) -> str:
  """Escape any value for HTML text/attribute context. Kernel names come from provider traces — untrusted."""
  return html.escape("" if value is None else str(value), quote=True)


def _num(value:Any, fmt:str = "{:.1f}", dash:str = "—") -> str:
  return fmt.format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else dash


def _shape_str(shape:Any) -> str:
  if isinstance(shape, dict):
    return "x".join(str(shape[k]) for k in ("rows", "cols") if k in shape) or "unknown"
  if isinstance(shape, (list, tuple)) and shape: return "x".join(str(x) for x in shape)
  return "unknown"


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


def _ms(us:Any) -> str:
  """Every time on the page is milliseconds. Artifacts store microseconds; this is the one conversion."""
  return _num(us / 1000.0, "{:.2f}") if isinstance(us, (int, float)) and not isinstance(us, bool) else "—"


def _card(title:str, source:str, body:str, *, flag:bool = False, cls:str = "") -> str:
  """A titled panel. `source` names the artifact the body was read from: no section without a citation."""
  classes = " ".join(c for c in ("card", "flag" if flag else "", cls) if c)
  return (f'<section class="{classes}"><div class="card-hd"><h2 class="card-ttl">{title}</h2>'
          f'<span class="card-sub">{_e(source)}</span></div>{body}</section>')


def _bar_cell(pct:Any) -> str:
  width = max(0.0, min(100.0, float(pct))) if isinstance(pct, (int, float)) and not isinstance(pct, bool) else 0.0
  return f'<td class="pct">{_num(pct)}<i><b style="width:{width:.1f}%"></b></i></td>'


def _rail(manifest:dict[str, Any]) -> str:
  stages = manifest.get("stages", {}) or {}
  cells = []
  for i, (key, label, note) in enumerate(STAGES):
    entry = stages.get(key)
    if entry:
      count = len(entry.get("artifacts", []) or [])
      detail, cls = f"{count} file{'' if count == 1 else 's'}", "stage"
    else:
      detail, cls = "not run", "stage s-none"
    cells.append(f'<div class="{cls}" title="{_e(note)}"><div class="stage-top"><i class="dot"></i>'
                 f'<span class="stage-name">{i + 1}. {_e(PLAIN_STAGE.get(key, label))}</span></div>'
                 f'<div class="stage-note">{_e(label)} · {_e(detail)}</div></div>')
  return '<nav class="rail" aria-label="Pipeline stages">' + "".join(cells) + "</nav>"


def _headline(manifest:dict[str, Any], results:dict[str, Any] | None) -> str:
  """The answer first: model, chip, speed limit, measured speed. Numbers come from `screen.results`, the same
  facts the TUI's Speed limit and Result steps print. Nothing is computed here."""
  results = results or {}
  model = manifest.get("model_id") or "unknown model"
  chip = manifest.get("target_id") or "unknown chip"
  ceil = results.get("ceiling") or {}
  limit = ceil.get("tok_s") if ceil.get("status") == "modeled" else None
  timing = results.get("timing") or {}
  measured = timing.get("tok_s")
  if isinstance(limit, (int, float)):
    limit_html = (f'<div class="big">{limit:.1f}</div><div class="big-u">tokens per second</div>'
                  f'<div class="big-n">memory {_num(ceil.get("peak_bandwidth_gbs"))} GB/s · context '
                  f'{_e(ceil.get("context"))}</div>')
  else:
    limit_html = (f'<div class="big dim">none</div><div class="big-n">{_e(ceil.get("reason") or "not computed for this page")}'
                  '</div>')
  if isinstance(measured, (int, float)):
    pct = f" · {measured / limit * 100:.0f}% of the limit" if isinstance(limit, (int, float)) and limit else ""
    meas_html = (f'<div class="big">{measured:.1f}</div><div class="big-u">tokens per second{pct}</div>'
                 f'<div class="big-n">timing trace · context {_e(timing.get("context"))}</div>')
  else:
    blocked = results.get("blocked") or []
    if blocked:
      missing = "Needs " + " and ".join(
        f'{_e(PLAIN_NEED.get(b.get("need"), b.get("need")))} (<a href="{_e(b.get("request"))}">{_e(b.get("request"))}</a>)'
        for b in blocked) + "."
    elif not results:
      missing = "Results were not read for this page."
    elif results.get("measured"):
      missing = "Needs a timing trace."
    else:
      missing = "Needs a plan: run <code>boltbeam analyze</code>."
    meas_html = f'<div class="big dim">not measured yet</div><div class="big-n">{missing}</div>'
  body = (f'<div class="head-bd"><div class="head-who"><div class="head-model">{_e(model)}</div>'
          f'<div class="head-chip">on {_e(chip)} · {_e(manifest.get("workload") or "unknown workload")}</div></div>'
          f'<div class="head-num"><div class="lbl">Speed limit</div>{limit_html}</div>'
          f'<div class="head-num"><div class="lbl">Measured</div>{meas_html}</div></div>')
  return f'<section class="card head" aria-label="Answer">{body}</section>'


def next_step(report:dict[str, Any], plan:dict[str, Any]) -> str:
  """Same decision ladder as workflow/output.py::_summary_md, so the two reports never disagree."""
  if plan.get("timing_profile", {}).get("status") == "requested":
    return ("Run an external timing trace from <code>trace_request.json</code>, ingest "
            "<code>boltbeam.timing_trace.v1</code>, then re-run <code>boltbeam analyze</code>.")
  if plan.get("primitive_profile", {}).get("status") == "requested":
    return ("Run an external probe from <code>probe_request.json</code>, ingest "
            "<code>boltbeam.probe_evidence.v1</code>, then re-run <code>boltbeam analyze</code>.")
  if report.get("status") == "policy_seeded":
    return ("Review <code>route_policy.json</code>; selected routes still require normalized evidence before "
            "promotion unless already ledgered.")
  if plan:
    return ("Run or translate <code>measurement_plan.json</code> through a provider adapter, then ingest "
            "normalized evidence and re-analyze.")
  return "Run <code>boltbeam analyze --run &lt;run&gt;</code> to build the search and measurement plan."


def roofline_kernels(timing:dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
  """Hottest per-kernel rows from the widest context summary, plus that context. Deterministic tie-break."""
  summaries = [s for s in timing.get("context_summaries", []) or [] if isinstance(s, dict)]
  if not summaries: return [], None
  chosen = max(summaries, key=lambda s: (s.get("context") or 0, str(s.get("dominant_timing_bucket") or "")))
  kernels = [k for k in (chosen.get("roofline", {}) or {}).get("kernels", []) or [] if isinstance(k, dict)]
  kernels.sort(key=lambda k: (-(k.get("us") or 0.0), str(k.get("name") or "")))
  return kernels, chosen.get("context")


def _kernel_table(timing:dict[str, Any]) -> str:
  kernels, context = roofline_kernels(timing)
  if not kernels:
    rows = [r for r in timing.get("role_timing", []) or [] if isinstance(r, dict)]
    if not rows: return ""
    rows.sort(key=lambda r: (-(r.get("pct_step") or 0.0), str(r.get("role") or "")))
    body = "".join(
      f'<tr><td class="name" title="{_e(r.get("role"))}">{_e(r.get("role") or "unknown")}</td>'
      f'<td>{_e(_shape_str(r.get("shape")))}</td><td>{_ms(r.get("wall_us"))}</td>'
      f'{_bar_cell(r.get("pct_step"))}'
      f'<td class="l">{_named(PLAIN_BUCKET, r.get("classification") or "unclassified", tag=True)}</td></tr>'
      for r in rows[:12])
    table = ('<div class="tscroll"><table><thead><tr><th>role</th><th>shape</th><th>wall ms</th>'
             '<th>% step</th><th class="l">classification</th></tr></thead>'
             f'<tbody>{body}</tbody></table></div>')
    return _card("Hot roles", "timing_profile.json · role_timing", table)

  body = "".join(
    f'<tr><td class="name" title="{_e(k.get("name"))}">{_e(k.get("name") or "unnamed")}</td>'
    f'<td>{_e(k.get("kind") or "—")}</td><td>{_ms(k.get("us"))}</td>'
    f'{_bar_cell(k.get("pct_step"))}<td>{_num(k.get("phys_util_pct"))}</td>'
    f'<td>{_ms(k.get("loss_us"))}</td>'
    f'<td class="l">{_named(PLAIN_BUCKET, k.get("bucket") or "unclassified", tag=True)}</td></tr>'
    for k in kernels[:12])
  more = ""
  if len(kernels) > 12:
    more = f'<p class="empty">{len(kernels) - 12} more kernels in <code>timing_profile.json</code>.</p>'
  table = ('<div class="tscroll"><table><thead><tr><th>kernel</th><th>kind</th><th>ms</th><th>% step</th>'
           '<th>% peak</th><th>loss ms</th><th class="l">bucket</th></tr></thead>'
           f'<tbody>{body}</tbody></table></div>{more}')
  ctx = f"timing_profile.json · context {context}" if context is not None else "timing_profile.json"
  return _card("Hot kernels", ctx, table)


def _timing_card(timing:dict[str, Any]) -> str:
  bucket = timing.get("dominant_timing_bucket") or "timing_inconclusive"
  actions = [a for a in timing.get("next_actions", []) or []][:4]
  items = "".join(f"<li>{_e(a)}</li>" for a in actions) or '<li class="empty">no recorded next action</li>'
  counts = (f'<div class="chips"><span class="chip">role rows {len(timing.get("role_timing", []) or [])}</span>'
            f'<span class="chip">candidate rows {len(timing.get("candidate_timing", []) or [])}</span></div>')
  body = (f'<div class="card-bd"><p style="margin:0 0 12px">Most time lost to '
          f'{_named(PLAIN_BUCKET, bucket, tag=True)}</p><ul>{items}</ul><div style="margin-top:12px">{counts}</div></div>')
  return _card("Timing verdict", "timing_profile.json", body)


def _regime_card(primitive:dict[str, Any]) -> str:
  regimes = [r for r in primitive.get("quant_gemv_regimes", []) or [] if isinstance(r, dict)]
  if not regimes: return ""
  regimes = sorted(regimes, key=lambda r: (str(r.get("role") or ""), str(r.get("quant") or "")))
  body = "".join(
    f'<tr><td>{_e(r.get("role") or "unknown")}</td><td>{_e(r.get("quant") or "unknown")}</td>'
    f'<td>{_e(_shape_str(r.get("shape")))}</td>'
    f'<td class="l">{_named(PLAIN_REGIME, r.get("classification") or "unknown", tag=True)}</td>'
    f'<td class="l">{_e(r.get("visible_bottleneck") or "unknown")}</td>'
    f'<td class="l wrap">{_e(r.get("next_action") or "collect more evidence")}</td></tr>'
    for r in regimes[:12])
  table = ('<div class="tscroll"><table class="wide"><thead><tr><th>role</th><th>quant</th><th>shape</th>'
           '<th class="l">regime</th><th class="l">bottleneck</th>'
           f'<th class="l">next action</th></tr></thead><tbody>{body}</tbody></table></div>')
  return _card("Building blocks (quant GEMV regimes)", "primitive_profile.json", table)


def _not_measured_card(primitive:dict[str, Any], timing:dict[str, Any]) -> str:
  """One compact card for every measured section that has no data yet, instead of one empty card each."""
  missing = []
  if not timing: missing.append(("Timing verdict and hot kernels", "timing_profile.json", "a timing trace"))
  if not primitive: missing.append(("Building blocks", "primitive_profile.json", "the building-block tests"))
  if not missing: return ""
  items = "".join(f'<li>{_e(what)}: needs {_e(need)} <span class="id">{_e(src)}</span></li>'
                  for what, src, need in missing)
  return _card("Not measured yet", " · ".join(src for _, src, _ in missing),
               f'<div class="card-bd"><ul>{items}</ul></div>', cls="compact")


def _blocked_card(report:dict[str, Any], plan:dict[str, Any], policy:dict[str, Any], primitive:dict[str, Any],
                  timing:dict[str, Any], runner:dict[str, Any]) -> str:
  """Absence as a first-class panel. A run that is missing evidence should say so above the fold."""
  needs = []
  if plan.get("primitive_profile", {}).get("status") == "requested" or not primitive:
    needs.append("The building-block tests (<code>probe_evidence.json</code>). Until they land no route may be "
                 "promoted on primitive grounds.")
  if plan.get("timing_profile", {}).get("status") == "requested" or not timing:
    needs.append("A timing trace (<code>hw_trace.json</code> to <code>timing_trace.json</code>). Without it every "
                 "speed on this page is a prediction, not a measurement.")
  if runner:
    missing = []
    if not primitive: missing.append("probe_evidence")
    if not timing: missing.append("timing_trace")
    if missing:
      needs.append("The handoff returned no " + _e(", ".join(missing)) +
                   ". The bundle was prepared but the provider has not written back.")
  # needs_measurement means no route is selected yet (workflow/analyze.py), even when the probe and the trace are
  # in: the route candidates themselves (plan phase M3, `wd_speed`) are still unmeasured.
  if report.get("status") == "needs_measurement":
    routes = [r for r in policy.get("routes", []) or [] if isinstance(r, dict)]
    open_routes = [r for r in routes if not r.get("selected_route")]
    count = f"{len(open_routes)} of {len(routes)} roles have" if routes else "No role has"
    needs.append(f"Route measurements. {count} no measured route yet (<code>route_policy.json</code>). Measure the "
                 "route candidates from <code>measurement_plan.json</code> (phase M3), then re-run "
                 "<code>boltbeam analyze</code>.")
  if not needs:
    return _card("Blocked on", "measurement_plan.json",
                 '<div class="card-bd"><p style="margin:0;color:var(--tx-2)">Nothing outstanding. Every '
                 'requested measurement has been ingested and classified.</p></div>')
  items = "".join(f'<div class="need"><span class="need-k">{i + 1:02d}</span>'
                  f'<span class="need-b">{n}</span></div>' for i, n in enumerate(needs))
  return _card("Blocked on", f"measurement_plan.json · {len(needs)} item{'' if len(needs) == 1 else 's'}",
               f'<div class="card-bd">{items}</div>', flag=True)


def _routes_card(policy:dict[str, Any]) -> str:
  rows = [r for r in policy.get("routes", []) or [] if isinstance(r, dict) and r.get("selected_route")]
  if not rows:
    return _card("Selected routes", "route_policy.json",
                 '<p class="empty">No route selected. Nothing to roll back.</p>', cls="compact")
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


def render_run_html(*, manifest:dict[str, Any], profile:dict[str, Any], report:dict[str, Any],
                    plan:dict[str, Any], policy:dict[str, Any], providers:dict[str, Any],
                    primitive:dict[str, Any], timing:dict[str, Any], runner:dict[str, Any],
                    source_run:str = "", results:dict[str, Any] | None = None) -> str:
  """Render one staged run directory as a standalone HTML document.

  Every argument is the parsed contents of a run artifact, or `{}` when that artifact does not exist. `results`
  is `workflow.screen.results` for the run (speed limit and measured tokens/s); without it the headline says the
  numbers were not read. Missing inputs are rendered as missing: this function never invents a value.
  """
  model_id = manifest.get("model_id") or profile.get("model_id") or "unknown"
  provider_names = sorted(str((p.get("provider_id") or p.get("provider")) if isinstance(p, dict) else p)
                          for p in providers.get("providers", []) or [])
  ready = {str(p.get("provider_id")) for p in providers.get("providers", []) or []
           if isinstance(p, dict) and p.get("status") not in (None, "missing")}

  meta = "".join(f"<div>{_e(label)} <b>{_e(value)}</b></div>" for label, value in (
    ("format", manifest.get("model_format") or "unknown"),
    ("arch", profile.get("architecture_class") or "unknown"),
    ("last stage", manifest.get("latest_stage") or "unknown"),
  ))

  artifacts = "".join(f'<a class="chip" href="{_e(a)}">{_e(a)}</a>' for a in manifest.get("artifacts", []) or [])
  providers_html = ("".join(f'<span class="chip{" on" if p in ready else ""}">{_e(p)}</span>' for p in provider_names)
                    or '<span class="chip">none recorded</span>')

  status = report.get("status") or "not_analyzed"
  status_card = _card("Where this run stands", "run_manifest.json · analysis_report.json",
    f'<div class="card-bd"><p style="margin:0 0 10px;color:var(--tx-2)">Status '
    f'{_named(PLAIN_STATUS, status)}</p>'
    f'<p style="margin:0 0 12px;color:var(--tx-2)"><b style="color:var(--tx)">Next step.</b> '
    f'{next_step(report, plan)}</p>'
    f'<div class="lbl">Providers <span class="id">provider_capabilities.json</span></div>'
    f'<div class="chips">{providers_html}</div></div>')

  measured = ""
  if timing: measured += f'<div class="cols one">{_timing_card(timing)}</div>'
  measured += _regime_card(primitive) + (_kernel_table(timing) if timing else "")

  return (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    f"<title>BoltBeam {_e(model_id)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">"
    f'<header class="mast"><div class="mark">Bolt<span>Beam</span></div>'
    f'<div class="run-id">{_e(source_run or model_id)}</div>'
    f'<div class="meta">{meta}</div>'
    '<button id="theme" class="chip" type="button" style="cursor:pointer;font:inherit;font-size:11px">'
    'theme</button></header>'
    f"{_headline(manifest, results)}"
    f"{_rail(manifest)}"
    f'<div class="cols">{status_card}{_blocked_card(report, plan, policy, primitive, timing, runner)}</div>'
    f"{_not_measured_card(primitive, timing)}"
    f"{measured}"
    f"{_routes_card(policy)}"
    f'{_card("Files in this run", "run_manifest.json", f"<div class=\'card-bd\'><div class=\'chips\'>{artifacts}</div></div>")}'
    '<p class="foot">Generated by <code>boltbeam output</code>. Deterministic: no timestamps, stable ordering.'
    '<br>Every panel names the file it was read from. Times are milliseconds.</p>'
    f"</div><script>{_JS}</script></body></html>\n")
