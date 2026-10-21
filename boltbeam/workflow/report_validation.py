"""Fail closed on inconsistent report arithmetic, before rendering or emitting advice.

Raw ms use 1e-6 absolute / 1e-8 relative tolerance. Percentages that the producer
rounds to one decimal use 0.051 percentage points. Capture coverage has a separate
10% window tolerance against the engine's own profiled repetition, not an untraced
performance target. No expected model speeds are encoded here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from boltbeam.math.numerics import ACCOUNTING_RTOL, ACCOUNTING_ATOL


class ReportInvalid(ValueError):
  pass


def unmatched_window_failures(loss:dict, captures:list) -> list[str]:
  """A decode capture whose window is not the native one (both from verified work evidence) is not comparable: no
  overhead comparison, ceiling words or throughput gains."""
  cap=next((c for c in captures if c.get('context')==loss.get('limit_context')),None) if loss.get('semantic_capture') else None
  window=((cap or {}).get('reference_band') or {}).get('timing_window');w=loss.get('where_token_goes') or {}
  if cap is None or (window and cap.get('timing_window')==window):return []
  return [f'loss: native and captured timing windows differ or are unrecorded, yet the report shows {what}' for what,bad in
          (('a capture comparison',loss.get('capture_comparison')),('capture-overhead ceiling words','capture-overhead ceiling' in json.dumps([w,loss.get('capture_comparison')])),
           ('throughput gains',any(r.get('tok_s_gain') is not None for r in w.get('rows') or []))) if bad]


def validate(run:Path, res:dict) -> dict:
  failures = []
  checks = 0
  if (run/'forced_replay.json').exists() or (res.get('loss',{}).get('replay') or {}).get('forced'):
    from boltbeam.workflow.forced_replay import table
    expected=table(run)
    if res['loss']['replay']!=expected:failures.append('forced replay display differs from checked capture evidence')
    checks+=1
  trace_path = run/('decode_capture.json' if res.get('workload') == 'decode' else 'prefill_trace.json')
  trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
  captures = list((trace.get('contexts' if res.get('workload') == 'decode' else 'lengths') or {}).values())
  verified_work = bool(captures)
  from boltbeam.workflow.work_evidence import verify
  for cap in captures:
    if not cap.get('work_evidence'):
      verified_work = False
      if trace_path.name == 'decode_capture.json':
        failures.append('decode capture is missing semantic work evidence')
      continue
    checks += 1
    try:
      verify(run, cap, require_limits=True)
    except (KeyError, TypeError, ValueError, OSError) as exc:
      verified_work = False
      failures.append(f'captured work evidence: {exc}')
  profile_path = run/'model_profile.json'
  if profile_path.is_file():
    profile = json.loads(profile_path.read_text())
    metadata = profile.get('metadata') or {}
    recurrent = metadata.get('recurrent') or {}
    arch = (metadata.get('architecture_signals') or {}).get('arch') or profile.get('architecture')
    unsupported_scan = recurrent.get('kind') == 'mamba2' or (recurrent and arch in ('nemotron_h', 'mamba', 'mamba2'))
    if unsupported_scan and (res.get('ceiling') or {}).get('floor_ms') and not (
        res.get('workload') in ('prefill','decode') and verified_work):
      failures.append('roofline: Mamba2 state traffic and SSD work are not validated against engine evidence; '
                      'the Gated DeltaNet formula cannot supply this model’s limit. Raw timings are retained.')

  def equal(path, actual, expected, tolerance=ACCOUNTING_ATOL):
    nonlocal checks
    checks += 1
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (actual, expected)) or not math.isclose(
        actual, expected, rel_tol=ACCOUNTING_RTOL, abs_tol=tolerance):
      failures.append(f"{path}: reported {actual!r}, derived {expected!r}")

  def cell(path, actual, expected):
    nonlocal checks
    checks += 1
    if actual != expected:
      failures.append(f"{path}: displayed {actual!r}, derived {expected!r}")

  def where(w, path):
    if not w or not w.get('rows'):
      return
    rows = w['rows']
    for field, total in [('now_ms', 'token_ms'), ('limit_ms', 'limit_ms'), ('lost_ms', 'lost_ms'), ('best_ms', 'best_ms')]:
      if w.get(total) is not None:
        equal(f'{path}.{total} = sum(rows.{field})', w[total], sum(r.get(field) or 0 for r in rows))
    for i, r in enumerate(rows):
      p = f'{path}.rows[{i}] ({r.get("name")})'
      if r.get('lost_ms') is not None:
        equal(p+'.lost_ms', r['lost_ms'], r['now_ms']-(r.get('limit_ms') or 0.0))
      if r.get('ms_if_fixed') is not None and r['ms_if_fixed'] < 0:
        failures.append(p+'.ms_if_fixed: negative projected duration')
      if w.get('base_ms') is not None:
        after = w['base_ms'] - r.get('lost_ms', 0)
        if r.get('tok_s_if_fixed') is not None and after > 0:
          equal(p+'.tok_s_if_fixed', r['tok_s_if_fixed'], 1000/after)
        if r.get('ms_if_fixed') is not None:
          equal(p+'.ms_if_fixed', r['ms_if_fixed'], after)
      if r.get('cells'):
        for index, field in [(0,'now_ms'),(1,'limit_ms'),(2,'best_ms'),(3,'lost_ms')]:
          if r.get(field) is not None:
            cell(p+f'.cells[{index}]',r['cells'][index],f'{r[field]:.2f}')
      if r.get('pct_peak') is not None and r.get('limit_ms') is not None and r.get('now_ms'):
        actual = r['now_ms']
        if w.get('estimate'):
          measured = [s for s in loss.get('roles', [])
                      if f"{s['role']} {s['quant']}" == r['name']]
          if len(measured) != 1:
            failures.append(p+'.pct_peak: no unique measured role behind the estimate')
            continue
          actual = measured[0].get('less_floor_ms')
          if actual is None:
            actual = measured[0]['actual_ms']
        equal(p+'.pct_peak', r['pct_peak'], 100*r['limit_ms']/actual, .051)

  loss = res.get('loss') or {}
  timing = res.get('timing') or {}
  ceiling = res.get('ceiling') or {}
  if res.get('workload') == 'decode' and ceiling.get('floor_ms') is not None and ceiling.get('tok_s') is not None:
    equal('ceiling.tok_s = 1000 / floor_ms', ceiling['tok_s'],1000/ceiling['floor_ms'])
  for runtime in loss.get('runtimes') or []:
    if not runtime.get('per_role') and runtime.get('lost_ms') is not None and loss.get('limit_ms') is not None:
      equal('loss.runtimes.lost_ms',runtime['lost_ms'],runtime['ms']-loss['limit_ms'])
  if res.get('workload') == 'decode' and timing.get('tok_s') is not None and timing.get('total_us') is not None:
    equal('timing.tok_s = 1e6 / timing.total_us', timing['tok_s'], 1e6/timing['total_us'], .0051)
  for i, r in enumerate(loss.get('roles') or []):
    if r.get('actual_ms') is not None and r.get('ideal_ms') is not None and not r.get('est_lost_ms'):
      if r.get('lost_ms') is not None:
        equal(f'loss.roles[{i}].lost_ms', r['lost_ms'], r['actual_ms']-r['ideal_ms'])
      if r.get('pct_peak') is not None and r['actual_ms']:
        equal(f'loss.roles[{i}].pct_peak', r['pct_peak'], 100*r['ideal_ms']/r['actual_ms'], .051)
      if r.get('mb_per_call') and r.get('calls_per_token') and r.get('gbs') and r.get('us_per_call'):
        equal(f'loss.roles[{i}].gbs = bytes/call / us/call', r['gbs'],r['mb_per_call']*1000/(r.get('us_per_call_less_floor') or r['us_per_call']), .051)
  tie = loss.get('tie_out') or {}
  if tie.get('lines') and tie.get('token_ms') is not None and not tie.get('refused'):
    equal('loss.tie_out.token_ms = sum(lines.ms)', tie['token_ms'], sum(r['ms'] for r in tie['lines']))
  where(loss.get('where_token_goes'), 'loss.where_token_goes')
  from boltbeam.workflow.tie_out import check_end_to_end, overhead_words
  def verdicts(node, path):
    nonlocal checks
    if isinstance(node, dict) and 'confirms' in node and (node['confirms'] is not None or node.get('status') == 'checked'):
      checks += 1
      failures.extend(f'{path}: {why}' for why in check_end_to_end(node))
    for key, child in (node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()):
      verdicts(child, f'{path}.{key}')
  verdicts(loss, 'loss')
  failures.extend(unmatched_window_failures(loss, captures))
  if comparison:=loss.get('capture_comparison'):
    rows=comparison.get('rows') or []
    refs=[cap for cap in captures if cap.get('reference_band') and cap.get('host_step_ms') and cap.get('context')==loss.get('limit_context')]
    if len(rows)!=2 or len(refs)!=1:failures.append('capture comparison lacks one reference and two accounting rows')
    else:
      cap=refs[0];wall=cap['host_step_ms'];median=cap['reference_band']['median_ms']
      for what,got,want in (('reference',rows[0]['now_ms'],median),('difference',rows[1]['now_ms'],wall-median),
                            ('sum',sum(r['now_ms'] for r in rows),wall),('captured',comparison['token_ms'],wall)):
        equal('capture comparison '+what,got,want)
      if overhead_words(100*(wall-median)/median) not in comparison.get('words',''):failures.append('capture comparison words do not disclose the capture overhead derived from its reference')
      for row in rows:cell('capture comparison cell',row['cells'][0],f"{row['now_ms']:.6f}")
      cell('capture comparison total cell',comparison['sum_cells'][1],f'{wall:.6f}')
  if loss.get('semantic_capture'):
    w=loss.get('where_token_goes') or {}
    if not verified_work:failures.append('decode operation ledger has no verified work evidence')
    if w.get('limit_ms') is not None:
      equal('decode headline limit = captured operation limits',ceiling.get('floor_ms'),w['limit_ms'])
  for point in (loss.get('prefill_curve') or {}).get('points') or []:
    if point.get('ms') is not None and point.get('tok_s') is not None:
      equal(f'prefill[{point["length"]}].tok_s', point['tok_s'], point['length']*1000/point['ms'])
  for filename, collection in [('prefill_trace.json','lengths'),('decode_capture.json','contexts')]:
    path=run/filename
    if not path.exists():
      continue
    trace = json.loads(path.read_text())
    for length, cap in (trace.get(collection) or {}).items():
      if not cap.get('rows'):
        continue
      if cap.get('ambiguous'):
        failures.append(f'{filename} {collection}.{length}: unresolved matrix attribution: {cap["ambiguous"]}')
      for row in cap['rows']:
        if row.get('refused') and (row.get('what') in ('role', 'weight', 'matrix', 'attention', 'recurrent') or 'matrix kernel no role' in row['refused']):
          failures.append(f'{filename} {collection}.{length} {row.get("name")}: {row["refused"]}')
      p = f'{filename} {collection}.{length}'
      equal(p+'.busy_ms = sum(rows.ms)', cap['busy_ms'], sum(r['ms'] for r in cap['rows']))
      equal(p+'.wall_ms = busy_ms + gaps_ms', cap['wall_ms'], cap['busy_ms']+cap['gaps_ms'])
      if cap.get('measured_ms'):
        equal(p+'.window covers engine repetition', cap['wall_ms'], cap['measured_ms'], .1*cap['measured_ms'])
      for i, r in enumerate(cap['rows']):
        if r.get('components'):
          for field in ('ceiling_ms','compute_ms','memory_ms'):
            equal(p+f'.rows[{i}].'+field,r[field],sum(c[field] for c in r['components']))
          from boltbeam.roofline.memory import hierarchy_ms
          rates=r['memory_rates']
          for j,c in enumerate(r['components']):
            memory=(hierarchy_ms(c['traffic'],rates['dram_gbs'],rates['l2_gbs'])['memory_ms']
                    if c.get('traffic') else c['bytes']/(rates['dram_gbs']*1e6))
            equal(p+f'.rows[{i}].components[{j}].memory_ms',c['memory_ms'],memory)
            equal(p+f'.rows[{i}].components[{j}].ceiling_ms',c['ceiling_ms'],max(c['compute_ms'],c['memory_ms']))
        elif r.get('ceiling_ms') is not None and r.get('compute_ms') is not None and r.get('memory_ms') is not None:
          equal(p+f'.rows[{i}].ceiling_ms', r['ceiling_ms'], max(r['compute_ms'],r['memory_ms']))
  replay = loss.get('replay') or {}
  turns = replay.get('turns') or []
  if turns:
    warm = [t for t in turns if not t['cold']] or turns
    head = replay['headline']
    equal('replay.headline.time_ms = sum(warm turn.wall_ms)', head['time_ms'], sum(t['wall_ms'] for t in warm))
    equal('replay.headline.time_ms = sum(split)', head['time_ms'], sum(head['split'].values()))
    for label, field in [('prefill','prompt_ms'),('decode','predicted_ms'),('overhead','overhead_ms')]:
      equal('replay.headline.split.'+label,head['split'][label],sum(t[field] for t in warm))
    if head.get('roofline_ms') is not None:
      equal('replay.headline.roofline_ms',head['roofline_ms'],sum(t['roofline_ms'] for t in warm))
      equal('replay.headline.e_pct',head['e_pct'],100*head['roofline_ms']/head['time_ms'])
      equal('replay.headline.lost',sum(head['lost'].values()),head['time_ms']-head['roofline_ms'])
    columns = replay.get('columns') or []
    display = {r['i']: r['cells'] for r in replay.get('rows') or []}
    for t in turns:
      equal(f'replay.turn[{t["i"]}].wall_ms',t['wall_ms'],t['prompt_ms']+t['predicted_ms']+t['overhead_ms'])
      if t.get('predicted_n', 0) > 1 and t.get('predicted_ms') and t.get('decode_tok_s') is not None:
        equal(f'replay.turn[{t["i"]}].decode_tok_s',t['decode_tok_s'],1000*(t['predicted_n']-1)/t['predicted_ms'])
      for title, field in [('prompt ms','prompt_ms'),('overhead ms','overhead_ms'),('ceiling ms','roofline_ms')]:
        if t['i'] in display and title in columns and t.get(field) is not None:
          cell(f'replay.turn[{t["i"]}].display.{title}',display[t['i']][columns.index(title)],f'{t[field]:.0f}')
      if t['i'] in display and '% of roofline' in columns and t.get('roofline_ms') is not None and t['wall_ms']:
        cell(f'replay.turn[{t["i"]}].display.percent',display[t['i']][columns.index('% of roofline')],
             f'{100*t["roofline_ms"]/t["wall_ms"]:.0f}%')
    if replay.get('total') and 'overhead ms' in columns:
      cell('replay.total.display.overhead',replay['total'][columns.index('overhead ms')],
           f'{sum(t["overhead_ms"] for t in warm):,.0f}')
  return {'status': 'failed' if failures else 'passed', 'checks': checks, 'failures': failures}


def require(run:Path, res:dict) -> dict:
  try:
    result = validate(run, res)
  except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
    raise ReportInvalid(f'FAILED: malformed report evidence: {exc}') from exc
  if result['failures']:
    raise ReportInvalid('FAILED: report does not reconcile with evidence\n'+'\n'.join(result['failures']))
  return result


def require_emit(run:Path, res:dict, plan:dict) -> None:
  """Check the emitted plan itself against its input, including every claimed evidence pointer."""
  import re
  require(run, res)
  failures = []
  h = plan['header']; loss = res.get('loss') or {}
  if (loss.get('replay') or {}).get('forced'):
    if plan.get('kind')!='replay_report' or plan.get('replay')!=loss['replay'] or plan.get('roles')!=[]:
      raise ReportInvalid('FAILED: emitted replay differs from the checked Run report')
    return
  if plan.get('kind')=='gap_report':
    from boltbeam.workflow.gameplan import gameplan
    expected=gameplan(run,res,now=plan.get('footer',{}).get('date',''))
    if plan!=expected:
      raise ReportInvalid('FAILED: gap report differs from the validated Run evidence')
    return
  token = next((r for r in loss.get('runtimes', []) if not r.get('per_role')), None)
  for key, expected in [('limit_ms',loss.get('limit_ms')),('limit_tok_s',loss.get('limit_tok_s')),
                         ('token_ms',(token or {}).get('ms')),('tok_s',(token or {}).get('tok_s')),
                         ('lost_ms',(token or {}).get('lost_ms')),
                         ('pct_of_limit',100*loss['limit_ms']/token['ms'] if token and token.get('ms') and loss.get('limit_ms') else None),
                         ('where_token_goes',loss.get('where_token_goes'))]:
    if h.get(key) != expected:
      failures.append(f'gameplan.header.{key}: {h.get(key)!r} != run {expected!r}')
  source = {(r['role'], r['quant'], tuple(r.get('shape') or [])): r for r in loss.get('roles', [])}
  if len(plan.get('roles', [])) != len(source):
    failures.append('gameplan.roles: emitted role count differs from measured role count')
  for i, r in enumerate(plan.get('roles', [])):
    original = source.get((r['role'], r['quant'], (r['rows'],r['cols'])))
    if original is None:
      failures.append(f'gameplan.roles[{i}]: no matching measured role')
      continue
    expected_loss = original.get('est_lost_ms') if r.get('lost_is_estimate') else original.get('lost_ms')
    if r.get('lost_ms') != expected_loss:
      failures.append(f'gameplan.roles[{i}].lost_ms: differs from measured role')
    for key in ('ideal_ms','mb_per_call','calls_per_token','us_per_call','pct_peak','gbs'):
      if r.get(key) != original.get(key):
        failures.append(f'gameplan.roles[{i}].{key}: differs from measured role')
  def evidence(value):
    if isinstance(value, dict):
      for ref in value.get('evidence', []):
        if not isinstance(ref,dict) or not ref.get('file'):continue
        f=run/ref['file']
        if not f.is_file():
          failures.append(f'gameplan evidence missing: {ref["file"]}')
          continue
        pointer=ref.get('path')
        if pointer and f.suffix=='.json':
          try:
            node=json.loads(f.read_text())
            for field,index in re.findall(r'([^\.\[\]]+)|\[(\d+)\]',pointer):
              node=node[int(index)] if index else node[field]
          except (KeyError,IndexError,TypeError,ValueError):
            failures.append(f'gameplan evidence pointer missing: {ref["file"]} {pointer}')
      for v in value.values():evidence(v)
    elif isinstance(value,list):
      for v in value:evidence(v)
  evidence(plan)
  if failures:
    raise ReportInvalid('FAILED: Emit does not reconcile with its evidence\n'+'\n'.join(failures))
