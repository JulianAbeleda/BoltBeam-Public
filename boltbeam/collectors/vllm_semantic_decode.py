"""vLLM graph decode capture, counter collection, and operation-ledger output."""
import json
from pathlib import Path


def collect(run:Path, *, target, model:str, context:int, tokens:int, batch:int) -> dict:
  from boltbeam.collectors import (vllm_decode as vd, engine_common as ec, vendor_capture,
    vllm_two_pass, vllm_semantic_work, cuda_memory, instruction_path as ip, semantic_attribution)
  from boltbeam.collectors.capture_settings import vllm_flags
  from boltbeam.collectors.prefill import library_paths, gpu_arch, last_level_cache, vllm_formats
  from boltbeam.target.targets import fact_uncertainty
  from boltbeam.workflow.screen import run_bandwidth
  from boltbeam.workflow.common import run_gpu
  if batch!=1:
    raise ValueError('semantic graph decode currently requires a single stream')
  if target.backend not in ('CUDA','AMD'):
    raise RuntimeError(f"vLLM's per-role decode capture joins a graph trace from nsys or rocprofv3; {target.backend} has neither")
  profile=json.loads((run/'model_profile.json').read_text())
  raw=run/'kernel_compare'/'vllm_graph_capture'/str(context)
  command=[*vd.argv(model,[context],[batch],tokens=tokens,reps=1,once=tokens),
           *vllm_flags(run,context,capture=True)]
  if '--tensor-parallel-size' in command:
    raise RuntimeError("vLLM's per-role decode of a tensor-parallel split is not measured yet: its kernels run in one "
                       "worker per GPU, while the capture marks each decode step in the engine's process")
  env={**ec.venv_env(vd.python()),**vd.CAPTURE_ENV}
  vendor_capture.capture(target.backend,[*command,'--capture-pass','paired'],raw,env=env,profiler_range=True,device=run_gpu(run))
  pair=vllm_two_pass.join(raw/'capture.sqlite',ec.json_lines(vendor_capture.program_output(raw)))
  attach=vllm_semantic_work.prepare(Path(model),profile)
  steps=[{'identity':{**pair['engine'],'decode_step':s['step'],'context':context+s['step']-1},
          'launches':attach(s['launches'],ssd_chunk=pair['engine'].get('mamba_chunk_size'))}
         for s in pair['steps'] if s['step']>0]
  engine=cuda_memory.decode_horizon([s['identity'] for s in steps])
  counter_command=[*command,'--capture-pass','timing','--profile-step','1','--profile-end',str(len(steps))]
  counter_env={**env,'TRITON_CACHE_DIR':engine['triton_cache_dir']}
  if target.backend=='AMD' and any((l.get('semantic_work') or {}).get('requires_memory_evidence') for s in steps for l in s['launches']):
    raise ValueError('decode kernels need measured memory traffic, and rocprofv3 has no memory counter on AMD here')
  launches=cuda_memory.collect_horizon(counter_command,counter_env,raw,steps)
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
    observed=ip.ncu_observe(counter_command,counter_env,raw/'ncu',closed,sigs,launches=len(launches))
    paths.update(observed=observed['kernels'],observed_how=observed['how'],observed_by=ip.FROM_NCU)
  (raw/'instruction_paths.json').write_text(json.dumps(paths,indent=1))
  bandwidth,source=run_bandwidth(run,target)
  peaks=dict(target.matrix_tflops or {})
  uncertainty={'bandwidth':fact_uncertainty(target,'memory_bandwidth_gbs'),
               'peaks':{p:fact_uncertainty(target,'matrix_tflops',p) for p in peaks}}
  attr=semantic_attribution.attribute(run,raw,peaks=peaks,bandwidth_gbs=bandwidth,
    cache=last_level_cache(target),uncertainty=uncertainty,
    formats=vllm_formats({r['quant'] for r in profile['roles']},engine))
  attr.update(bandwidth_source=source,peaks_clock='rates at their measured probe clocks; no step-local clock sample')
  path=run/'decode_capture.json'
  trace=json.loads(path.read_text()) if path.exists() else {
    'schema':'boltbeam.decode_capture.v1','provider':vd.PROVIDER,'contexts':{}}
  trace['contexts'][str(context)]=attr
  path.write_text(json.dumps(trace,indent=1))
  return trace
