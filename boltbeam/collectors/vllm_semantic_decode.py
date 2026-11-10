"""vLLM graph decode capture, counter collection, and operation-ledger output."""
import json
from pathlib import Path

from boltbeam.collectors.attribution import NotSplitByRole
from boltbeam.target.gpu import run_gpu


# B streams read the experts each stream's routing picks; graphed CUDA replay runs no router code to observe, and the
# graphless labelling pass's routing is not the timed pass's (a near tie in top-k can differ while greedy tokens agree)
BATCHED_MOE = ('batched routed-expert decode is refused: the experts B streams read are known only from routing in '
               'the graphless labelling pass, and nothing ties that to the graphed timed pass; dense models and one '
               'stream (whose top-k reads exactly k experts) are captured')


def collect(run:Path, *, target, model:str, context:int, tokens:int, batch:int) -> dict:
  from boltbeam.collectors import (vllm_decode as vd, engine_common as ec, vendor_capture,
    vllm_two_pass, vllm_semantic_work, cuda_memory, instruction_path as ip, semantic_attribution)
  from boltbeam.collectors.capture_settings import vllm_flags, vllm_streams
  from boltbeam.collectors.cuda_module_load import NoModuleCapture
  from boltbeam.collectors.prefill import library_paths, gpu_arch, last_level_cache, vllm_formats
  from boltbeam.target.targets import fact_uncertainty
  from boltbeam.workflow.limit import run_bandwidth
  if type(batch) is not int or batch<1:
    raise ValueError('semantic graph decode needs a positive whole number of streams')
  if target.backend not in ('CUDA','AMD'):
    raise RuntimeError(f"vLLM's per-role decode capture joins a graph trace from nsys or rocprofv3; {target.backend} has neither")
  profile=json.loads((run/'model_profile.json').read_text())
  if batch>1 and any(r.get('n_expert') for r in profile.get('roles') or []):
    raise ValueError(BATCHED_MOE)
  raw=run/'kernel_compare'/'vllm_graph_capture'/str(context)
  command=[*vd.argv(model,[context],[batch],tokens=tokens,reps=1,once=tokens),
           *vllm_flags(run,context,capture=True),*vllm_streams(run,context)]
  if '--tensor-parallel-size' in command:
    raise NotSplitByRole("vLLM's per-role decode of a tensor-parallel split is not measured yet: its kernels run in one "
                         "worker per GPU, while the capture marks each decode step in the engine's process")
  speculative='--speculative-config' in command
  if speculative and batch!=1:
    raise ValueError('speculative decode is captured for one stream: each stream accepts its own drafts, so B streams '
                     'take different numbers of steps for the same tokens')
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
    attach=vllm_semantic_work.prepare(Path(model),profile,**({'speculative':True} if speculative else {}))
    if speculative:
      from boltbeam.collectors import vllm_speculation
      kept,accounting=vllm_speculation.accounted(vllm_speculation.ledger(pair['engine'])[1:])
      held={r['step'] for r in kept}
      steps=[{'identity':{**pair['engine'],'decode_step':s['step'],'context':s['speculation']['context'],
                          'length':s['speculation']['query_tokens'],'speculation_step':s['speculation'],
                          'speculation_accounting':accounting},
              'launches':attach(s['launches'],ssd_chunk=pair['engine'].get('mamba_chunk_size'),speculation=s['speculation'])}
             for s in pair['steps'] if s['step'] in held]
    else:
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
  except vllm_two_pass.TokensDiverge as e:
    if speculative:
      from boltbeam.collectors.vllm_speculation import SpeculationDiverge
      raise SpeculationDiverge(f'{e}: speculative passes cannot be forced to one token script (verification scores '
                               'k + 1 rows a request), so the capture is refused') from e
    # Two passes took different tokens (a near tie in greedy argmax, or an engine that does not repeat itself): every
    # pass is forced to one production-graph generation's (--token-script), recorded as token_forcing
    script=raw/'token_script.json';script.unlink(missing_ok=True)
    forced=['--token-script',str(script)]
    steps,engine,counter_command,counter_env,launches=passes(forced)
  command=[*command,*forced]
  (raw/'semantic_work.json').write_text(json.dumps({'engine':engine,'launches':launches,
                                                 'steps':[s['identity'] for s in steps]},indent=1))
  tools=ip.engine_tools(vd.python())
  paths=library_paths([Path(p) for p in engine['libraries']],launches,tools=tools,
                      jit_caches=[engine['triton_cache_dir']],capture=raw,device=run_gpu(run))
  closed={ip.template(l['key']) for l in launches if l.get('semantic_work') or
          l.get('semantic_kind')=='attention'}-set(paths['kernels'])
  if closed and target.backend=='AMD':
    raise ValueError(f'no code object was read for decode kernels {sorted(closed)[:3]}')
  if closed:
    # kernels built in the process (cuBLASLt's nvjet_*) are in no library file: the modules the driver loads are
    # captured in one more pass of the same program, outside the timed captures, and read by entry symbol
    arch=gpu_arch()
    nvdisasm=next((str(d/'nvdisasm') for d in tools if (d/'nvdisasm').is_file()),None) or ip.cuda_tool('nvdisasm')  # tools is a list of folders
    sigs=ip.sass_signatures(arch,nvdisasm=nvdisasm) if arch and nvdisasm else None
    said='did not run: '+('nvdisasm was not found, so no machine code was read' if arch else 'nvidia-smi gave no GPU architecture')
    if sigs is not None:
      try:
        loaded=read_loaded([*command,'--capture-pass','timing'],counter_env,raw/'module_load',closed,sigs,engine,nvdisasm=nvdisasm)
      except NoModuleCapture as e:
        said=f'did not run: {e}'
      else:
        said=f"ran and did not see them in the {len(loaded['modules'])} modules the driver loaded"
        if loaded['kernels']:
          paths['kernels'].update(loaded['kernels'])
          paths['libraries'].append({'library':str(raw/'module_load'),'sha256':None,'kernels':len(loaded['kernels']),'code':loaded['how']})
          paths['module_load']={'how':loaded['how'],'kernels':loaded['evidence'],'modules':loaded['modules'],
                                'identity':json.loads((raw/'module_load'/'capture_identity.json').read_text())}
          closed-=set(loaded['kernels'])
  if closed:
    if sigs is None or not ip.counters_open():  # the run stands whole: the report keeps its measured step and limit
      raise NotSplitByRole(f'closed decode kernels require accessible instruction counters: no binary was read for {sorted(closed)}'
                       +''.join(f'; {u}' for u in paths.get('unread') or [])+f'; the module-load capture {said}'
                       +(f'; {jet} are generated by cuBLASLt at runtime, so no library file holds their machine code' if (jet:=sorted(k for k in closed if 'nvjet_' in k)) else ''))
    if speculative:
      # under ncu the engine does not repeat the capture's speculative steps (it serializes kernels, and a bf16 near tie
      # moves an acceptance: the 27B diverged at token 16, 2026-11-08), so no step past the first is the capture's.
      # Every verify step launches the same programs at the same shapes; the first decode step must hold every closed
      # template, and its executed paths are theirs
      first=[l for l in steps[0]['launches'] if ip.template(l['key']) in closed]
      if {ip.template(l['key']) for l in first}!=closed:
        raise ValueError('the first speculative decode step does not launch every closed decode kernel')
      observed=ip.ncu_observe([*command,*forced,'--capture-pass','timing','--profile-step','1','--profile-end','1'],
                              counter_env,raw/'ncu',closed,sigs,launches=len(first))
      scope='decode step 1 (every speculative verify step launches the same kernels at the same shapes)'
    elif batch==1:
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
  from boltbeam.workflow.limit import bandwidth_uncertainty, run_read
  # the uncertainty of the bandwidth the limit uses: this run's read probe when it has one, as the ceiling's is
  uncertainty={'bandwidth':bandwidth_uncertainty(target,run_read(run,target.target_id)),
               'peaks':{p:fact_uncertainty(target,'matrix_tflops',p) for p in peaks}}
  attr=semantic_attribution.attribute(run,raw,peaks=peaks,bandwidth_gbs=bandwidth,
    cache=last_level_cache(target),uncertainty=uncertainty,
    formats=vllm_formats({r['quant'] for r in profile['roles']},engine))
  attr.update(bandwidth_source=source,peaks_clock='rates at their measured probe clocks; no step-local clock sample')
  bare=unmeasured(attr['rows'])
  if bare:  # nothing to divide their work by, so no limit by captured operations: the run stands whole and says which
    raise NotSplitByRole(f"the chip profile holds no measured rate for the matrix instruction of {', '.join(r['name'] for r in bare)} "
                         f"({'; '.join(sorted({r['path'] for r in bare}))}): they have no ceiling, so this run has the whole step only")
  return store(run,context,batch,attr)


def unmeasured(rows:list[dict]) -> list[dict]:
  """The role and attention rows with no ceiling because the chip profile holds no rate for their matrix instruction:
  an opcode no probe path issues (an H200's HGMMA before its probe, 2026-11-09), or a path the probe did not measure."""
  return [r for r in rows if r['what'] in ('role','attention') and r.get('ceiling_ms') is None and
          (r.get('path_status') in ('one','mixed') and r.get('peak_tflops') is None or
           any(p.startswith('unknown:') for p in r.get('compute_paths') or {}))]


def read_loaded(argv:list[str], env:dict, out:Path, closed:set[str], signatures:dict, engine:dict, *, nvdisasm:str | None) -> dict:
  """The closed kernels' matrix paths from the modules the driver loads in one more production-graph pass with no
  profiler around it (cuda_module_load.table). The pass must be the timed program: same model, engine, settings and
  shape. Its tokens are not compared: a kernel is picked by shape and type, and one the pass did not load is not found."""
  from boltbeam.collectors import cuda_module_load, cuda_memory, vllm_decode as vd
  ran=next(r for r in reversed(cuda_module_load.capture(argv,env,out,python=vd.python())) if 'execution_mode' in r)
  for key in (*cuda_memory.IDENTITY_FIELDS,'context','batch','tokens','compilation_config','enforce_eager','capture_pass','triton_cache_dir'):
    if key!='length' and key in engine and engine[key]!=ran.get(key):
      raise ValueError(f'module-load pass ran a different program than the timed capture: {key}')
  return cuda_module_load.table(out,closed,signatures,nvdisasm=nvdisasm)


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
