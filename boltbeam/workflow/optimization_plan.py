"""Measured performance gaps for engines without kernel-specific Emit evidence."""
from __future__ import annotations

from pathlib import Path


def tables(res:dict) -> list[dict]:
  loss=res.get('loss') or {}
  direct=loss.get('where_token_goes')
  if direct and direct.get('rows'):
    return [direct]
  return [w for w in (loss.get('replay') or {}).get('wheres', [])
          if w.get('status') in ('measured','benchmark_only') and w.get('rows')]


def build(run:Path, res:dict, date:str) -> dict:
  from boltbeam.workflow.gameplan import SCHEMA
  from boltbeam.workflow.screen import measurement_words
  loss=res.get('loss') or {}
  breakdowns=tables(res)
  if not breakdowns:
    raise ValueError('no measured work breakdown for a gap report')
  files=[name for name in ('run_manifest.json','model_profile.json','timing_trace.json','decode_capture.json',
         'prefill_trace.json','replay_trace.json','report_validation.json','machine_facts.json') if (run/name).is_file()]
  evidence=[{'file':name} for name in files if name!='report_validation.json']
  gaps=[]
  summaries=[]
  for table_index,w in enumerate(breakdowns):
    summary=dict.fromkeys(('modeled_difference_ms','unknown_ms','timing_gaps_ms','overlap_ms','limit_adjustment_ms'),0.0)
    summary.update(table_index=table_index,title=w['title'],unknown_rows=0,
      captured_ms=w['token_ms'],modeled_limit_ms=w['limit_ms'],
      capture_reference_difference_ms=w['token_ms']-w['base_ms'] if w.get('base_ms') is not None else None)
    for index,row in enumerate(w['rows']):
      limit=row.get('limit_ms')
      what=row.get('what')
      if what=='overlap':
        status='overlap correction';summary['overlap_ms']+=row['now_ms']
      elif what=='gaps':
        status='measured interval; recoverability unknown';summary['timing_gaps_ms']+=row['now_ms']
      elif what=='limit':
        status='modeled budget correction';summary['limit_adjustment_ms']+=row['lost_ms']
      elif limit is None:
        status='measured; ceiling unknown';summary['unknown_ms']+=row['now_ms'];summary['unknown_rows']+=1
      else:
        status='above model' if row['now_ms']>limit else 'at or below model'
        summary['modeled_difference_ms']+=row['now_ms']-limit
      gaps.append({'table_index':table_index,'row_index':index,'name':row['name'],'section':row.get('section'),
        'what':what,'measured_ms':row['now_ms'],'modeled_limit_ms':limit,
        'modeled_difference_ms':row['now_ms']-limit if limit is not None else None,
        'allocated_tok_s_share':row.get('tok_s_gain'),
        'allocation_basis':'proportional accounting share; not independently recoverable throughput',
        'bound':row.get('bound'),'status':status,'evidence':evidence})
    summaries.append(summary)
  plan={'schema':SCHEMA,'kind':'gap_report','id':run.name,
    'header':{'model_id':res.get('model_id'),'target_id':res.get('target_id'),
      'engine':(res.get('engine') or {}).get('provider') or loss.get('provider'),
      'weight_format':(res.get('engine') or {}).get('weight_format'),'workload':res.get('workload'),
      'engine_details':res.get('engine'),'toolchain':res.get('toolchain'),
      'measurement':measurement_words(res.get('measurement')),'measured_runtimes':loss.get('runtimes') or [],
      'modeled_limit_ms':loss.get('limit_ms'),'modeled_limit_tok_s':loss.get('limit_tok_s'),'evidence':evidence},
    'tables':breakdowns,'gaps':gaps,'gap_summaries':summaries,'roles':[],
    'limitations':['Roofline is an upper bound; modeled gaps are not proven savings.',
      'No model = measured time, unknown savings.',
      'Kernel intervals may include tracing overhead; their recoverability is unknown.',
      'Tok/s shares reconcile the bound; they do not predict individual optimizations.'],
    'footer':{'files_read':files,'date':date,'sentence':'Evidence only; no engine changes applied.'}}
  plan['presentation']=presentation(plan)
  return plan


def presentation(plan:dict) -> dict:
  h=plan['header']
  out=[f"# Performance gaps: {h.get('model_id')} on {h.get('target_id')}",'',
       f"Run {plan['id']} · {h.get('engine')} · {h.get('workload')} · {h.get('weight_format') or 'format not recorded'}",'',
       plan['limitations'][0]]
  for runtime in h['measured_runtimes']:
    out+=['',f"Measured {runtime['tok_s']:.1f} tok/s · {runtime['ms']:.3f} ms."]
  summary='\n'.join(out)
  out=['','## Gap summary','']
  for s in plan['gap_summaries']:
    if len(plan['gap_summaries'])>1:
      out += [f"### {s['title']}"]
    out += [f"- Known operations: {s['modeled_difference_ms']:+.3f} ms vs model.",
      f"- No model: {s['unknown_rows']} rows · {s['unknown_ms']:.3f} ms measured · unknown savings.",
      f"- Kernel intervals: {s['timing_gaps_ms']:.3f} ms · unknown savings.",
      f"- Overlap correction: {s['overlap_ms']:+.3f} ms."]
    if s['limit_adjustment_ms']:
      out += [f"- Unsplit model correction: {s['limit_adjustment_ms']:+.3f} ms."]
    if s['capture_reference_difference_ms'] is not None:
      out += [f"- Captured minus untraced: {s['capture_reference_difference_ms']:+.3f} ms."]
  out+=['',plan['limitations'][2],plan['limitations'][3],'',
        'Evidence files: '+', '.join(plan['footer']['files_read']),f"Date: {plan['footer']['date']}",'',plan['footer']['sentence'],'']
  return {'summary':summary,'details':'\n'.join(out)}


def markdown(plan:dict) -> str:
  from boltbeam.workflow.screen import where_lines
  parts=plan['presentation']
  out=[parts['summary']]
  for w in plan['tables']:
    lines=where_lines(w)
    out+=['',f"## {lines[0].rstrip(':')}",lines[1].strip(),'','```text',*(line[2:] for line in lines[2:]),'```']
  return '\n'.join(out)+'\n'+parts['details']
