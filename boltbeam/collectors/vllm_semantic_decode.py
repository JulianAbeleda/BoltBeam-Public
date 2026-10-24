"""vLLM graph decode capture, counter collection, and operation-ledger output."""
import json
from pathlib import Path


# B streams read the experts each stream's routing picks; graphed CUDA replay runs no router code to observe, and the
# graphless labelling pass's routing is not the timed pass's (a near tie in top-k can differ while greedy tokens agree)
BATCHED_MOE = ('batched routed-expert decode is refused: the experts B streams read are known only from routing in '
               'the graphless labelling pass, and nothing ties that to the graphed timed pass; dense models and one '
               'stream (whose top-k reads exactly k experts) are captured')


def collect(run:Path, *, target, model:str, context:int, tokens:int, batch:int) -> dict:
  from boltbeam.collectors import (vllm_decode as vd, engine_common as ec, vendor_capture,
    vllm_two_pass, vllm_semantic_work, cuda_memory, instruction_path as ip, semantic_attribution)
  from boltbeam.collectors.capture_settings import vllm_flags
  from boltbeam.collectors.prefill import library_paths, gpu_arch, last_level_cache, vllm_formats
  from boltbeam.target.targets import fact_uncertainty
  from boltbeam.workflow.screen import run_bandwidth
  from boltbeam.workflow.common import run_gpu
  if type(batch) is not int or batch<1:
    raise ValueError('semantic graph decode needs a positive whole number of streams')
  if target.backend not in ('CUDA','AMD'):
    raise RuntimeError(f"vLLM's per-role decode capture joins a graph trace from nsys or rocprofv3; {target.backend} has neither")
  profile=json.loads((run/'model_profile.json').read_text())
  if batch>1 and any(r.get('n_expert') for r in profile.get('roles') or []):
    raise ValueError(BATCHED_MOE)
  raw=run/'kernel_compare'/'vllm_graph_capture'/str(context)
  command=[*vd.argv(model,[context],[batch],tokens=tokens,reps=1,once=tokens),
           *vllm_flags(run,context,capture=True)]
  if '--tensor-parallel-size' in command:
    raise RuntimeError("vLLM's per-role decode of a tensor-parallel split is not measured yet: its kernels run in one "
                       "worker per GPU, while the capture marks each decode step in the engine's process")
  env={**ec.venv_env(vd.python()),**vd.CAPTURE_ENV}
  def paired(extra):
    vendor_capture.capture(target.backend,[*command,*extra,'--capture-pass','paired'],raw,env=env,profiler_range=True,device=run_gpu(run))
    return vllm_two_pass.join(raw/'capture.sqlite',ec.json_lines(vendor_capture.program_output(raw)))
  def passes(forced):
    pair=paired(forced)
    if batch>1 and any((l.get('semantic') or {}).get('class')=='MoERunner' for s in pair['steps'] for l in s['launches']):
      raise ValueError(BATCHED_MOE)
    for s in pair['steps'] if batch>1 else []:
      if s['step']>0:require_streams(s['launches'],batch)
    attach=vllm_semantic_work.prepare(Path(model),profile)
    steps=[{'identity':{**pair['engine'],'decode_step':s['step'],'context':context+s['step']-1},
            'launches':attach(s['launches'],ssd_chunk=pair['engine'].get('mamba_chunk_size'),
                              **({'streams':batch} if batch>1 else {}))}
           for s in pair['steps'] if s['step']>0]
    engine=cuda_memory.decode_horizon([s['identity'] for s in steps])
    counter_command=[*command,*forced,'--capture-pass','timing','--profile-step','1','--profile-end',str(len(steps))]
    counter_env={**env,'TRITON_CACHE_DIR':engine['triton_cache_dir']}
    if target.backend=='AMD' and any((l.get('semantic_work') or {}).get('requires_memory_evidence') for s in steps for l in s['launches']):
      raise ValueError('decode kernels need measured memory traffic, and rocprofv3 has no memory counter on AMD here')
    return steps,engine,counter_command,counter_env,cuda_memory.collect_horizon(counter_command,counter_env,raw,steps)
  forced=[]
  try:
    steps,engine,counter_command,counter_env,launches=passes([])
  except vllm_two_pass.TokensDiverge:
    # Two passes took different tokens (a near tie in greedy argmax, or an engine that does not repeat itself): every
    # pass is forced to one production-graph generation's (--token-script), recorded as token_forcing
    script=raw/'token_script.json';script.unlink(missing_ok=True)
    forced=['--token-script',str(script)]
    steps,engine,counter_command,counter_env,launches=passes(forced)
  command=[*command,*forced]
  (raw/'semantic_work.json').write_text(json.dumps({'engine':engine,'launches':launches,
                                                 'steps':[s['identity'] for s in steps]},indent=1))
  bins=sorted(Path(vd.python()).parent.parent.glob('lib/python*/site-packages/nvidia/*/bin'))
  tools=bins[0] if bins else None
  paths=library_paths([Path(p) for p in engine['libraries']],launches,tools=tools,
                      jit_caches=[engine['triton_cache_dir']],capture=raw,device=run_gpu(run))
  closed={ip.template(l['key']) for l in launches if l.get('semantic_work') or
          l.get('semantic_kind')=='attention'}-set(paths['kernels'])
  if closed and target.backend=='AMD':
    raise ValueError(f'no code object was read for decode kernels {sorted(closed)[:3]}')
  if closed:
    arch=gpu_arch()
    if not arch or not ip.counters_open():
      raise ValueError('closed decode kernels require accessible instruction counters')
    sigs=ip.sass_signatures(arch,nvdisasm=str(tools/'nvdisasm') if tools else None)
    if batch==1:
      observed=ip.ncu_observe(counter_command,counter_env,raw/'ncu',closed,sigs,launches=len(launches))
      scope='every decode step'
    else:
      # B streams launch closed kernels every step, and observing every launch replays the program tens of thousands of
      # times: the first and the last decode step are observed, each holding every closed template, and each template's
      # path must agree between them
      observed,scope=observe_ends(lambda step,name,count:ip.ncu_observe(
        [*command,'--capture-pass','timing','--profile-step',str(step),'--profile-end',str(step)],
        counter_env,raw/name,closed,sigs,launches=count),[s['launches'] for s in steps],closed)
    paths.update(observed=observed['kernels'],observed_how=observed['how'],observed_by=ip.FROM_NCU,observed_scope=scope)
  (raw/'instruction_paths.json').write_text(json.dumps(paths,indent=1))
  bandwidth,source=run_bandwidth(run,target)
  peaks=dict(target.matrix_tflops or {})
  uncertainty={'bandwidth':fact_uncertainty(target,'memory_bandwidth_gbs'),
               'peaks':{p:fact_uncertainty(target,'matrix_tflops',p) for p in peaks}}
  attr=semantic_attribution.attribute(run,raw,peaks=peaks,bandwidth_gbs=bandwidth,
    cache=last_level_cache(target),uncertainty=uncertainty,
    formats=vllm_formats({r['quant'] for r in profile['roles']},engine))
  attr.update(bandwidth_source=source,peaks_clock='rates at their measured probe clocks; no step-local clock sample')
  return store(run,context,batch,attr)


def observe_ends(observe, steps:list[list[dict]], closed:set[str]) -> tuple[dict, str]:
  """The closed kernels' executed paths from the first and the last decode step (observe(step, folder, launches)),
  each step launching every closed template, and every template's path the same in both; with the scope said."""
  from boltbeam.collectors.instruction_path import template
  ends=[]
  for step,name in ((1,'ncu-first'),(len(steps),'ncu-last')):
    own=[l for l in steps[step-1] if template(l['key']) in closed]
    if {template(l['key']) for l in own}!=closed:
      raise ValueError(f'decode step {step} does not launch every closed decode kernel')
    ends.append(observe(step,name,len(own)))
  if ends[0]['kernels']!=ends[1]['kernels']:
    raise ValueError('closed decode kernels executed different paths in the first and last decode step')
  return ends[0],f'decode steps 1 and {len(steps)} of {len(steps)}, agreeing'


def store(run:Path, context:int, batch:int, attr:dict) -> dict:
  """The capture at its context in the run's decode_capture.json. A context holds one batch: a capture of another
  batch there is refused, never written over it."""
  from boltbeam.collectors import vllm_decode as vd
  path=run/'decode_capture.json'
  trace=json.loads(path.read_text()) if path.exists() else {
    'schema':'boltbeam.decode_capture.v1','provider':vd.PROVIDER,'contexts':{}}
  held={(c.get('engine') or {}).get('batch') for c in trace['contexts'].values()}-{None}
  if held-{batch}:
    raise ValueError(f'decode_capture.json already holds a capture of {sorted(held)[0]} stream(s); a capture of '
                     f'{batch} goes in a run of its own')
  trace['contexts'][str(context)]=attr
  path.write_text(json.dumps(trace,indent=1))
  return trace


def require_streams(launches:list[dict], batch:int) -> None:
  """A decode step of `batch` streams holds one token per stream: every labelled module's first input is `batch`
  rows. A step that also carries a prefill chunk, or a stream that sat out, is not that step and is refused."""
  seen={}
  for l in launches:
    s=l.get('semantic') or {}
    shape=((s.get('inputs') or [{}])[0] or {}).get('shape')
    if shape:seen[(s.get('module'),s.get('operation_id'))]=shape[0]
  wrong=sorted({rows for rows in seen.values() if rows!=batch})
  if not seen or wrong:
    raise ValueError(f'decode step is not {batch} streams of one token each: labelled modules see {wrong or "no"} rows')
