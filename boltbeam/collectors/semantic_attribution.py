"""Attribute a recorded engine step using shared operation math.

This stage is CPU-only. Captures and counters are supplied by the provider; all
rates are supplied by measured hardware facts. It never substitutes timings or
fills missing instruction paths.
"""
import hashlib
import json
from pathlib import Path

from boltbeam.collectors.prefill_attribution import attribute_prefill
from boltbeam.workflow.work_evidence import verify


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
        engine.get('batch')!=1 or engine.get('decode_step')!=1):
      raise ValueError('decode attribution requires recorded production single-stream decode')
    workload='decode'
  length=engine['length']
  context=engine['context']
  if type(context) is not int or context<0:
    raise ValueError('decode capture has no explicit prompt context')
  chunks=[1]*length if engine.get('decode_steps') else [length]
  attr=attribute_prefill(facts['launches'],[],chunk_widths=chunks,outputs=len(chunks),
    layers=profile['layer_count'],kernel_paths=paths['kernels'],observed=paths.get('observed'),observed_by=paths.get('observed_by'),
    peaks=peaks,bandwidth_gbs=bandwidth_gbs,profile=profile,length=length,warmup=False,
    formats=formats,cache=cache,
    depth=context,unc=uncertainty,explicit_overlap=True)
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
  attr.update(workload=workload,context=context,length=length,engine=engine,
    wall_source=f'the captured {workload} step, first kernel to last',
    peaks_at=peaks,bandwidth_gbs=bandwidth_gbs,cache=cache,
    work_evidence={'schema':'boltbeam.semantic_work.v1','execution_mode':engine['execution_mode'],
      'model_profile_sha256':digest(run/'model_profile.json'),
      'sources':[{'file':str(p.relative_to(run)),'sha256':digest(p)} for p in sources]})
  bind_cache_rate(run,raw,attr,cache)
  verify(run,attr)
  return attr
