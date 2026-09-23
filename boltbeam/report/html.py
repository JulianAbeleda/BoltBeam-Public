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
# is what a human calls the step. A stage absent from run_manifest["stages"] renders as not-run, never hidden.
_STAGES: tuple[tuple[str, str, str], ...] = (
  ("load",          "load",        "model / weight / workload facts"),
  ("autoscan",      "autoscan",    "machine + provider capabilities"),
  ("analyze",       "analyze",     "search space + measurement plan"),
  ("runner_plan",   "runner-plan", "audit-tracer handoff bundle"),
  ("ingest_probe",  "probe",       "primitive evidence classified"),
  ("ingest_timing", "timing",      "timing trace classified"),
  ("output",        "output",      "policy / report / provider plan"),
)

# timing buckets that mean "this kernel is fine" — used only to pick a severity colour, never to drop a row.
_HEALTHY_BUCKETS = frozenset({"at_speed_of_light", "memory_bound_ok", "fused_ok", "timing_ok"})

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
.rail{display:flex;overflow-x:auto}
.stage{flex:1 1 0;min-width:150px;padding:12px 15px;border-right:1px solid var(--node-br);
display:flex;flex-direction:column;gap:3px}
.stage:last-child{border-right:0}
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
@media (max-width:920px){.cols{grid-template-columns:1fr}.meta{margin-left:0}}
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
thead th{font-size:9.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--tx-3);
text-align:right;padding:10px 12px 8px;border-bottom:1px solid var(--node-br);white-space:nowrap}
thead th:first-child,tbody td:first-child{text-align:left}
tbody td{padding:8px 12px;border-bottom:1px solid color-mix(in srgb,var(--node-br) 55%,transparent);
text-align:right;font-size:12px;color:var(--tx-2);font-family:var(--mono)}
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
border:1px solid var(--widget-br);font-family:var(--mono)}
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
  if not bucket: return ""
  if str(bucket) in _HEALTHY_BUCKETS: return "t-ok"
  return "t-warn" if "latency" in str(bucket) or "dilution" in str(bucket) else "t-bad"


def _card(title:str, source:str, body:str, *, flag:bool = False) -> str:
  """A titled panel. `source` names the artifact the body was read from — no section without a citation."""
  cls = "card flag" if flag else "card"
  return (f'<section class="{cls}"><div class="card-hd"><h2 class="card-ttl">{_e(title)}</h2>'
          f'<span class="card-sub">{_e(source)}</span></div>{body}</section>')


def _bar_cell(pct:Any) -> str:
  width = max(0.0, min(100.0, float(pct))) if isinstance(pct, (int, float)) and not isinstance(pct, bool) else 0.0
  return f'<td class="pct">{_num(pct)}<i><b style="width:{width:.1f}%"></b></i></td>'


def _rail(manifest:dict[str, Any]) -> str:
  stages = manifest.get("stages", {}) or {}
  cells = []
  for key, label, note in _STAGES:
    entry = stages.get(key)
    if entry:
      count = len(entry.get("artifacts", []) or [])
      detail = f"{count} artifact{'' if count == 1 else 's'}"
      cls = "stage"
    else:
      detail, cls = "not run", "stage s-none"
    cells.append(f'<div class="{cls}"><div class="stage-top"><i class="dot"></i>'
                 f'<span class="stage-name">{_e(label)}</span></div>'
                 f'<div class="stage-note">{_e(detail)}</div>'
                 f'<div class="stage-note">{_e(note)}</div></div>')
  return '<nav class="rail" aria-label="Pipeline stages">' + "".join(cells) + "</nav>"


def _next_step(report:dict[str, Any], plan:dict[str, Any]) -> str:
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


def _roofline_kernels(timing:dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
  """Hottest per-kernel rows from the widest context summary, plus that context. Deterministic tie-break."""
  summaries = [s for s in timing.get("context_summaries", []) or [] if isinstance(s, dict)]
  if not summaries: return [], None
  chosen = max(summaries, key=lambda s: (s.get("context") or 0, str(s.get("dominant_timing_bucket") or "")))
  kernels = [k for k in (chosen.get("roofline", {}) or {}).get("kernels", []) or [] if isinstance(k, dict)]
  kernels.sort(key=lambda k: (-(k.get("us") or 0.0), str(k.get("name") or "")))
  return kernels, chosen.get("context")


def _kernel_table(timing:dict[str, Any]) -> str:
  kernels, context = _roofline_kernels(timing)
  if not kernels:
    rows = [r for r in timing.get("role_timing", []) or [] if isinstance(r, dict)]
    if not rows: return _card("Hot kernels", "timing_profile.json",
                              '<p class="empty">No timing profile ingested. Absent, not zero.</p>')
    rows.sort(key=lambda r: (-(r.get("pct_step") or 0.0), str(r.get("role") or "")))
    body = "".join(
      f'<tr><td class="name" title="{_e(r.get("role"))}">{_e(r.get("role") or "unknown")}</td>'
      f'<td>{_e(_shape_str(r.get("shape")))}</td><td>{_num(r.get("wall_us"))}</td>'
      f'{_bar_cell(r.get("pct_step"))}'
      f'<td style="text-align:left"><span class="tag {_bucket_class(r.get("classification"))}">'
      f'{_e(r.get("classification") or "unclassified")}</span></td></tr>' for r in rows[:12])
    table = ('<div class="tscroll"><table><thead><tr><th>role</th><th>shape</th><th>wall µs</th>'
             '<th>% step</th><th style="text-align:left">classification</th></tr></thead>'
             f'<tbody>{body}</tbody></table></div>')
    return _card("Hot roles", "timing_profile.json · role_timing", table)

  body = "".join(
    f'<tr><td class="name" title="{_e(k.get("name"))}">{_e(k.get("name") or "unnamed")}</td>'
    f'<td>{_e(k.get("kind") or "—")}</td><td>{_num(k.get("us"))}</td>'
    f'{_bar_cell(k.get("pct_step"))}<td>{_num(k.get("phys_util_pct"))}</td>'
    f'<td>{_num(k.get("loss_us"))}</td>'
    f'<td style="text-align:left"><span class="tag {_bucket_class(k.get("bucket"))}">'
    f'{_e(k.get("bucket") or "unclassified")}</span></td></tr>' for k in kernels[:12])
  more = ""
  if len(kernels) > 12:
    more = f'<p class="empty">… {len(kernels) - 12} more kernels in <code>timing_profile.json</code></p>'
  table = ('<div class="tscroll"><table><thead><tr><th>kernel</th><th>kind</th><th>µs</th><th>% step</th>'
           '<th>% peak</th><th>loss µs</th><th style="text-align:left">bucket</th></tr></thead>'
           f'<tbody>{body}</tbody></table></div>{more}')
  ctx = f"timing_profile.json · context {context}" if context is not None else "timing_profile.json"
  return _card("Hot kernels", ctx, table)


def _timing_card(timing:dict[str, Any]) -> str:
  if not timing:
    return _card("Timing verdict", "timing_profile.json",
                 '<p class="empty">No timing trace ingested — no timing claim can be made.</p>', flag=True)
  bucket = timing.get("dominant_timing_bucket") or "timing_inconclusive"
  actions = [a for a in timing.get("next_actions", []) or []][:4]
  items = "".join(f"<li>{_e(a)}</li>" for a in actions) or '<li class="empty">no recorded next action</li>'
  counts = (f'<div class="chips"><span class="chip">role rows {len(timing.get("role_timing", []) or [])}</span>'
            f'<span class="chip">candidate rows {len(timing.get("candidate_timing", []) or [])}</span></div>')
  body = (f'<div class="card-bd"><p style="margin:0 0 12px"><span class="tag {_bucket_class(bucket)}">'
          f'{_e(bucket)}</span></p><ul>{items}</ul><div style="margin-top:12px">{counts}</div></div>')
  return _card("Timing verdict", "timing_profile.json", body)


def _regime_card(primitive:dict[str, Any]) -> str:
  regimes = [r for r in primitive.get("quant_gemv_regimes", []) or [] if isinstance(r, dict)]
  if not regimes:
    return _card("Quant GEMV regimes", "primitive_profile.json",
                 '<p class="empty">No primitive probe evidence ingested.</p>', flag=True)
  regimes = sorted(regimes, key=lambda r: (str(r.get("role") or ""), str(r.get("quant") or "")))
  body = "".join(
    f'<tr><td>{_e(r.get("role") or "unknown")}</td><td>{_e(r.get("quant") or "unknown")}</td>'
    f'<td>{_e(_shape_str(r.get("shape")))}</td>'
    f'<td style="text-align:left"><span class="tag {_bucket_class(r.get("classification"))}">'
    f'{_e(r.get("classification") or "unknown")}</span></td>'
    f'<td style="text-align:left">{_e(r.get("visible_bottleneck") or "unknown")}</td>'
    f'<td style="text-align:left">{_e(r.get("next_action") or "collect more evidence")}</td></tr>'
    for r in regimes[:12])
  table = ('<div class="tscroll"><table><thead><tr><th>role</th><th>quant</th><th>shape</th>'
           '<th style="text-align:left">regime</th><th style="text-align:left">bottleneck</th>'
           f'<th style="text-align:left">next action</th></tr></thead><tbody>{body}</tbody></table></div>')
  return _card("Quant GEMV regimes", "primitive_profile.json", table)


def _blocked_card(plan:dict[str, Any], primitive:dict[str, Any], timing:dict[str, Any],
                  runner:dict[str, Any]) -> str:
  """Absence as a first-class panel. A run that is missing evidence should say so above the fold."""
  needs = []
  if plan.get("primitive_profile", {}).get("status") == "requested" or not primitive:
    needs.append("Primitive probe evidence (<code>probe_evidence.json</code>). Until it lands the search space "
                 "is only partially authorised and no route may be promoted on primitive grounds.")
  if plan.get("timing_profile", {}).get("status") == "requested" or not timing:
    needs.append("A timing trace (<code>hw_trace.json</code> → <code>timing_trace.json</code>). Without it every "
                 "speed claim below is a prediction, not a measurement.")
  if runner:
    missing = []
    if not primitive: missing.append("probe_evidence")
    if not timing: missing.append("timing_trace")
    if missing:
      needs.append("Runner bundle returned no " + _e(", ".join(missing)) +
                   ". The handoff was prepared but the provider-side executor has not written back.")
  if not needs:
    return _card("Blocked on", "measurement_plan.json",
                 '<div class="card-bd"><p style="margin:0;color:var(--ink-2)">Nothing outstanding — every '
                 'requested measurement has been ingested and classified.</p></div>')
  items = "".join(f'<div class="need"><span class="need-k">{i + 1:02d}</span>'
                  f'<span class="need-b">{n}</span></div>' for i, n in enumerate(needs))
  return _card("Blocked on", f"measurement_plan.json · {len(needs)} item{'' if len(needs) == 1 else 's'}",
               f'<div class="card-bd">{items}</div>', flag=True)


def _routes_card(policy:dict[str, Any]) -> str:
  rows = [r for r in policy.get("routes", []) or [] if isinstance(r, dict) and r.get("selected_route")]
  if not rows:
    return _card("Selected routes", "route_policy.json",
                 '<p class="empty">No route selected. Nothing to roll back.</p>')
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
      f'<span class="tag {"t-ok" if row.get("status") == "promoted" else ""}">'
      f'{_e(row.get("status") or "unknown")}</span>'
      f'<span class="card-sub">{_e(cite)}</span></summary>'
      f'<pre>{_e(cmd)}</pre></details>')
  return _card("Selected routes", "route_policy.json · rollback commands", "".join(blocks))


def render_run_html(*, manifest:dict[str, Any], profile:dict[str, Any], report:dict[str, Any],
                    plan:dict[str, Any], policy:dict[str, Any], providers:dict[str, Any],
                    primitive:dict[str, Any], timing:dict[str, Any], runner:dict[str, Any],
                    source_run:str = "") -> str:
  """Render one staged run directory as a standalone HTML document.

  Every argument is the parsed contents of a run artifact, or `{}` when that artifact does not exist. Missing
  inputs are rendered as missing — this function never invents a value to fill a panel.
  """
  model_id = manifest.get("model_id") or profile.get("model_id") or "unknown"
  provider_names = [str(p.get("provider") if isinstance(p, dict) else p)
                    for p in providers.get("providers", []) or []]

  meta = "".join(f"<div>{_e(label)} <b>{_e(value)}</b></div>" for label, value in (
    ("target", manifest.get("target_id") or "unknown"),
    ("format", manifest.get("model_format") or "unknown"),
    ("workload", manifest.get("workload") or "unknown"),
    ("arch", profile.get("architecture_class") or "unknown"),
    ("stage", manifest.get("latest_stage") or "unknown"),
  ))

  artifacts = "".join(f'<span class="chip">{_e(a)}</span>' for a in manifest.get("artifacts", []) or [])
  providers_html = ("".join(f'<span class="chip">{_e(p)}</span>' for p in sorted(provider_names))
                    or '<span class="chip">none recorded</span>')

  status_card = _card("Where this run stands", "run_manifest.json · analysis_report.json",
    f'<div class="card-bd"><p style="margin:0 0 10px;color:var(--ink-2)">Analysis status '
    f'<span class="tag">{_e(report.get("status") or "not_analyzed")}</span></p>'
    f'<p style="margin:0 0 12px;color:var(--ink-2)"><b style="color:var(--ink)">Next step.</b> '
    f'{_next_step(report, plan)}</p>'
    f'<div class="chips">{providers_html}</div></div>')

  return (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    f"<title>BoltBeam — {_e(model_id)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">"
    f'<header class="mast"><div class="mark">Bolt<span>Beam</span></div>'
    f'<div class="run-id">{_e(source_run or model_id)}</div>'
    f'<div class="meta">{meta}</div>'
    '<button id="theme" class="chip" type="button" style="cursor:pointer;border:1px solid var(--rule);'
    'font:inherit;font-size:11px">theme</button></header>'
    f"{_rail(manifest)}"
    f'<div class="cols">{status_card}{_blocked_card(plan, primitive, timing, runner)}</div>'
    f'<div class="cols">{_timing_card(timing)}{_regime_card(primitive)}</div>'
    f"{_kernel_table(timing)}"
    f"{_routes_card(policy)}"
    f'{_card("Artifacts in this run", "run_manifest.json", f"<div class=\'card-bd\'><div class=\'chips\'>{artifacts}</div></div>")}'
    '<p class="foot">Generated by <code>boltbeam output</code>. Deterministic — no timestamps, stable ordering;'
    '<br>every panel cites the artifact it was read from.</p>'
    f"</div><script>{_JS}</script></body></html>\n")
