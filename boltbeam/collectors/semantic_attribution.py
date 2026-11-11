"""Attribute a recorded engine step using shared operation math.

This stage is CPU-only. Captures and counters are supplied by the provider; all
rates are supplied by measured hardware facts. It never substitutes timings or
fills missing instruction paths.
"""
import hashlib
import json
from pathlib import Path

from boltbeam.collectors.prefill_attribution import attribute_prefill
from boltbeam.collectors.work_evidence import verify


def bind_cache_rate(run:Path, raw:Path, capture:dict, cache:dict | None) -> None:
  """Persist the independently measured calibration used by semantic attribution."""
  calibration=(cache or {}).get('calibration')
  if calibration is None or not capture.get('work_evidence'):return
  from boltbeam.roofline.memory import peak_bandwidth
  if cache['gbs']!=peak_bandwidth(calibration):
    raise ValueError('cache rate differs from its independent calibration')
  path=raw/'cache_bandwidth.json'
  if not path.resolve().is_relative_to(run.resolve()):
    raise ValueError('cache calibration evidence must be stored inside its run')
  path.write_text(json.dumps(calibration,indent=2)+'\n')
  proof=capture['work_evidence'];relative=str(path.relative_to(run))
  proof['sources']=[s for s in proof['sources'] if s['file']!=relative]+[
    {'file':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
  proof['cache_rate_source']=path.name


def step_replicas(launches:list[dict], rows_of) -> dict:
  """A semantic decode capture's repeats for tie_out.over_ceiling: it is one capture of many decode steps, so its
  steps are dealt into engine_common.REPLICAS groups, every third step to a group (the context grows a token a step, so each
  group spans the same contexts; contiguous thirds do not), and each group is attributed as the whole capture is:
  ms per step by kernel row, the overlap row left out. One process, not fresh captures, and said so. Nothing with
  fewer than two steps a group."""
  from boltbeam.collectors.engine_common import REPLICAS
  steps=sorted({l['capture_step'] for l in launches})
  if len(steps)<2*REPLICAS:return {}
  groups=[set(steps[g::REPLICAS]) for g in range(REPLICAS)]
  return {'replicas':[{r['name']:r['ms']/len(own) for r in rows_of([l for l in launches if l['capture_step'] in own],[1]*len(own),len(own))['rows']
                       if r['what']!='overlap'} for own in groups],
          'replicas_of':f"groups of one capture's {len(steps)} decode steps (every {REPLICAS}rd step; one process, not fresh captures)"}


def attribute(run:Path, raw:Path, *, peaks:dict, bandwidth_gbs:float,
              cache:dict | None, uncertainty:dict, formats:dict) -> dict:
  profile=json.loads((run/'model_profile.json').read_text())
  facts=json.loads((raw/'semantic_work.json').read_text())
  paths=json.loads((raw/'instruction_paths.json').read_text())
  engine=facts['engine']
  replay=engine.get('replay')
  if replay is not None:
    counts=[replay['prefill_tokens'],replay['decode_tokens']]
    if engine.get('batch')!=1 or sum(counts)!=engine['length'] or sum(v>0 for v in counts)!=1:
      raise ValueError('replay attribution requires one unmixed single-request step')
    workload='prefill' if counts[0] else 'decode'
  else:
    if (engine.get('execution_mode')!='production CUDA graph timing' or
        engine.get('capture_pass')!='timing' or engine.get('enforce_eager') is not False or
        type(engine.get('batch')) is not int or engine['batch']<1 or engine.get('decode_step')!=1):
      raise ValueError('decode attribution requires recorded production decode of a whole number of streams')
    workload='decode'
  length=engine['length']
  context=engine['context']
  if type(context) is not int or context<0:
    raise ValueError('decode capture has no explicit prompt context')
  speculative=engine.get('speculative_steps')
  # a speculative step verifies k + 1 query tokens (collectors/vllm_speculation); a plain decode step, one
  chunks=([s['query_tokens'] for s in speculative] if speculative else [1]*length if engine.get('decode_steps') else [length])
  rows_of=lambda launches,chunks,n=length:attribute_prefill(launches,[],chunk_widths=chunks,outputs=len(chunks),  # noqa: E731
    layers=profile['layer_count'],kernel_paths=paths['kernels'],observed=paths.get('observed'),observed_by=paths.get('observed_by'),
    peaks=peaks,bandwidth_gbs=bandwidth_gbs,profile=profile,length=n,warmup=False,
    formats=formats,cache=cache,
    depth=context,unc=uncertainty,explicit_overlap=True,streams=1 if replay is not None else engine['batch'])
  attr=rows_of(facts['launches'],chunks)
  if engine.get('decode_steps') and not speculative:
    attr.update(step_replicas(facts['launches'],rows_of))
  digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  sources=[raw/name for name in ('semantic_work.json','instruction_paths.json')]
  if engine.get('decode_steps'):
    from boltbeam.collectors.vllm_two_pass import reference_band,host_step
    sources.append(run/'timing_trace.json')
    attr['reference_band']=reference_band(engine,json.loads(sources[-1].read_text()))
    attr.update(host_step(engine))
  if any((l.get('semantic_work') or {}).get('requires_memory_evidence') for l in facts['launches']):
    sources.append(raw/'memory.json')
  if replay is not None:sources.append(raw/'replay_evidence.json')
  if any(not p.resolve().is_relative_to(run.resolve()) for p in sources):
    raise ValueError('decode evidence must be stored inside its run')
  if paths.get('observed_scope'):attr['observed_scope']=paths['observed_scope']  # which decode steps ncu observed
  if paths.get('module_load'):attr['module_load']=sorted(paths['module_load']['kernels'])  # kernels no file holds
  if speculative:
    from boltbeam.collectors.vllm_speculation import words
    spec=engine.get('speculation') or {}
    attr['speculation']={'config':spec.get('config'),'prompt':spec.get('prompt'),'acceptance':engine['acceptance'],
      'tokens':sum(s['emitted'] for s in speculative),'steps':len(speculative),
      'accounting':engine.get('speculation_accounting'),'words':words(engine['acceptance']),
      'per_token':'each step\'s time and limits over the tokens the captured steps emitted (time per step / tokens '
                  'per step, at the acceptance measured over these steps)'}
  attr.update(workload=workload,context=context,length=length,engine=engine,
    wall_source=f'the captured {workload} step, first kernel to last',
    peaks_at=peaks,bandwidth_gbs=bandwidth_gbs,cache=cache,
    work_evidence={'schema':'boltbeam.semantic_work.v1','execution_mode':engine['execution_mode'],
      'model_profile_sha256':digest(run/'model_profile.json'),
      'sources':[{'file':str(p.relative_to(run)),'sha256':digest(p)} for p in sources]})
  bind_cache_rate(run,raw,attr,cache)
  verify(run,attr)
  return attr
