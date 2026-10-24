"""Join semantic labels to production times only for identical recorded programs."""
from pathlib import Path
from boltbeam.collectors import semantic_capture as sc


class TokensDiverge(ValueError):
  """Both passes recorded every stream's tokens, and they differ: the passes took different steps."""


def reference_band(engine:dict, trace:dict) -> dict:
  """Use only repeats of the same production configuration and token horizon."""
  import math,statistics
  from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL
  batch=engine.get('batch',1)
  if batch==1:
    rows=[r for r in trace.get('rows',[]) if r.get('scope')=='whole_step' and r.get('context')==engine['context']]
  else:  # B streams: the untraced point of that batch at that context (providers.engine_trace keeps every batch there)
    points=[p for p in trace.get('batches',[]) if p.get('context')==engine['context'] and p.get('batch')==batch]
    if len(points)==1 and (points[0].get('step_samples_ms') is None or points[0].get('repeat_band_ms') is None):
      raise ValueError(f'the untraced point of {batch} streams has no paired repeats (step_samples_ms, repeat_band_ms): '
                       'time that batch again before capturing it')
    # the median and token horizon are the point's own step time and tokens; a point cannot state them twice
    if any(p.get(k) is not None and p[k]!=v for p in points for k,v in (('wall_us',p.get('step_ms',0)*1000),('decode_tokens',p.get('tokens')))):
      raise ValueError('untraced point states a median or token horizon other than its own step time and tokens')
    rows=[{**p,'wall_us':p['step_ms']*1000,'decode_tokens':p.get('tokens')} for p in points]
  if len(rows)!=1:raise ValueError('capture needs one matching untraced reference')
  row=rows[0];actual=row.get('engine') or {}
  for field in ('model_config_sha256','activation_dtype','kv_cache_dtype','compilation_config',
                'enforce_eager','max_model_len','max_num_batched_tokens'):
    if field not in engine or actual.get(field)!=engine[field]:
      raise ValueError(f'untraced reference has different production setting: {field}')
  if actual.get('build')!=engine['vllm'] or row.get('decode_tokens')!=engine['tokens'] or row.get('batch')!=batch:
    raise ValueError('untraced reference build, batch or token horizon differs')
  samples=row.get('step_samples_ms') or []
  if len(samples)<2 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
    raise ValueError('untraced reference needs positive finite paired repeats')
  interval=[min(samples),max(samples)];median=statistics.median(samples)
  if row.get('repeat_band_ms')!=interval or not math.isclose(row['wall_us']/1000,median,
      rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
    raise ValueError('untraced repeat band or median does not derive from recorded samples')
  # the native arm is never forced: a forced capture says so beside it (vllm_semantic_decode's token_forcing)
  if actual.get('token_forcing'):raise ValueError('untraced reference was forced: it is not the native engine')
  return {'low_ms':interval[0],'high_ms':interval[1],'median_ms':median,'repeats':len(samples),'timing_window':actual.get('timing_window'),
          'basis':'paired N-token minus one-token calls in the same unprofiled engine',
          **({'token_forcing':{'capture':engine['token_forcing'],'reference':None}} if engine.get('token_forcing') else {})}


def host_step(engine:dict) -> dict:
  """The profiled arm's host-elapsed decode step and its window, from the recorded long and short call seconds
  (the median of its one sample, as the native arm takes the median of its samples). {} for older records."""
  from boltbeam.collectors.vllm_decode import decode_window_vllm
  if engine.get('short_seconds') is None:return {}
  return {'host_samples':engine['samples'],'host_step_ms':(engine['seconds']-engine['short_seconds'])*1000/(engine['tokens']-1),'timing_window':decode_window_vllm(engine['context'],engine['tokens'],engine['batch'],engine)}


def join(path: Path, records: list[dict]) -> dict:
  passes={}
  for record in records:
    name=record.get('capture_pass')
    if name not in ('timing','label'):continue
    if name in passes:raise ValueError('duplicate recorded capture pass')
    passes[name]=record
  if set(passes)!={'timing','label'}:raise ValueError('capture needs both timing and labeling passes')
  timing,labels=passes['timing'],passes['label']
  for field in ('model_config_sha256','vllm','engine','compilation_config','enforce_eager','context','batch',
                'tokens','output_token_ids','steps','seed','max_model_len','max_num_batched_tokens'):
    if field not in timing or timing[field] is None or timing[field]!=labels.get(field):
      if field=='output_token_ids' and labels.get(field) is not None:  # both recorded, and they differ
        raise TokensDiverge(f'capture passes have absent or different {field}')
      raise ValueError(f'capture passes have absent or different {field}')
  if (timing['enforce_eager'] or not timing['compilation_config'].get('mode') or
      timing['runtime_graph_mode']!='production' or labels['runtime_graph_mode']!='NONE'):
    raise ValueError('capture requires production compilation and graphless labeling')
  for field in ('token_forcing','token_script_sha256','engine_nondeterministic'):
    if timing.get(field)!=labels.get(field):raise ValueError(f'capture passes were forced differently: {field}')
  batch=timing['batch']
  if (type(batch) is not int or batch<1 or len(timing['output_token_ids'])!=batch or
      any(len(ids)!=timing['tokens'] for ids in timing['output_token_ids'])):
    raise ValueError('capture requires complete output token evidence for every stream')
  if timing['steps']!=list(range(len(timing['steps']))):raise ValueError('capture has missing or reordered step identities')
  spans=sc.step_index(path);steps=[];empty=[]
  with sc.Capture(path) as capture:
    for step in timing['steps']:
      pair={m:sc.read_step(capture,spans.get((m,step),[]),allow_empty=True) for m in ('label','timing')}
      if not pair['label'] and not pair['timing']:
        empty.append(step);continue
      joined=sc.join_step(*(sc.submission_order(pair[m]) for m in ('label','timing')),step)
      steps.append({'step':step,'launches':joined})
  if len(steps)!=timing['tokens']:
    raise ValueError('captured steps do not cover one prefill and all generated decode tokens')
  return {'engine':timing,'steps':steps,'empty_steps':empty,
          'basis':'production timing; labels joined by exact recorded kernel submission order and geometry'}


def replay_steps(path:Path, passes:dict, *, segments:list[dict] | None = None, activities=None):
  """Yield exact production steps; callers must finish validation before publishing.

  Equal kernel sequences alone do not prove equal data-dependent MoE work. Bind
  each step to its consumed tokens and cache depth before transferring labels.
  The last emitted token is not consumed and does not change captured work.
  """
  from contextlib import ExitStack
  from itertools import groupby
  from boltbeam.collectors.replay_semantics import session_identities,SETTINGS
  if set(passes)!={'timing','label'}:raise ValueError('replay needs timing and labeling passes')
  timing,labels=(passes[m]['model'] for m in ('timing','label'))
  for field in (*SETTINGS,'model_config_sha256','activation_dtype','kv_cache_dtype','build',
                'compilation_config','logits_processors','triton_autotune_cache'):
    if field not in timing or timing[field]!=labels.get(field):
      raise ValueError(f'replay passes have absent or different {field}')
  if (timing['enforce_eager'] or not timing['compilation_config'].get('mode') or
      timing.get('runtime_graph_mode')!='production' or labels.get('runtime_graph_mode')!='NONE'):
    raise ValueError('replay requires production compilation and graphless labeling')
  sessions={m:session_identities(p['model'],p['records'],p['responses']) for m,p in passes.items()}
  if segments is not None and (len(segments)!=len(sessions['timing']) or any(set(s)!={'timing','label'} for s in segments)):
    raise ValueError('replay trace segments do not cover every turn in both passes')
  spans=sc.step_index(path)
  native=groupby(activities,key=lambda row:row['step']) if activities is not None else None
  with sc.Capture(path) as capture, ExitStack() as readers:
    previous={};sources={m:(capture,spans) for m in passes}
    scopes={m:readers.enter_context(ExitStack()) for m in passes}
    for turn,(timed,labeled) in enumerate(zip(sessions['timing'],sessions['label']),1):
      if timed['cached_tokens']!=labeled['cached_tokens'] or len(timed['steps'])!=len(labeled['steps']):
        raise ValueError(f'replay capture divergence at turn {turn}: cached prefix or step count')
      for m,p in (segments[turn-1].items() if segments is not None else ()):
        if native is not None and m=='timing':continue
        if previous.get(m)!=p:
          scopes[m].close()
          sources[m]=(scopes[m].enter_context(sc.Capture(p,lineage=capture)),sc.step_index(p))
          previous[m]=p
      for position,(t,l) in enumerate(zip(timed['steps'],labeled['steps'])):
        if any(t[k]!=l[k] for k in ('context','length','replay')):
          raise ValueError(f'replay input divergence at turn {turn}, step {position}: consumed tokens or scheduler work')
        pair={m:sc.submission_order(sc.read_step(sources[m][0],sources[m][1].get((m,i['decode_step']),[]),
                       phases=('execute','sample'))) for m,i in [('timing',t),('label',l)] if native is None or m=='label'}
        if native is not None:
          step,rows=next(native,(None,()))
          if step!=t['decode_step']:raise ValueError('native activity scheduler order differs')
          pair['timing']=native_launches(pair['label'],list(rows))
        yield {'turn':turn,'step':t['decode_step'],'identity':t,
               'label_step':l['decode_step'],
               'launches':sc.join_step(pair['label'],pair['timing'],t['decode_step'])}
  if len(sessions['timing'])!=len(sessions['label']):raise ValueError('replay passes have different turn counts')
  if native is not None and next(native,None) is not None:raise ValueError('unused native activity steps')


def native_launches(labels,activities):
  """Transfer display keys only after exact observed entry and geometry checks."""
  from boltbeam.collectors.attention_auxiliary import geometry
  if len(labels)!=len(activities):raise ValueError('native activity launch coverage differs')
  result=[]
  for label,activity in zip(labels,activities,strict=True):
    grid,block=geometry(label)
    if label.get('phase') not in ('execute','sample') or label['phase']!=activity['phase']:
      raise ValueError('native activity scheduler phase differs')
    if (not label.get('raw_entry') or label['raw_entry']!=activity['raw_entry'] or
        list(grid)!=activity['grid'] or list(block)!=activity['block']):
      raise ValueError('native activity raw entry or geometry differs')
    result.append({**activity,'key':label['key']})
  return result
