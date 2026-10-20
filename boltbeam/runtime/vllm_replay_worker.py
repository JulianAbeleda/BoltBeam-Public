"""Production server timing and temporary compiled labeling in one loaded worker.

The controller switches passes through collective_rpc between complete requests.
Autotuning stays unprofiled. Graph provenance may be captured after kernel warmup.
Only labeling disables replay and observes modules/operators; timing records scheduler identities and NVTX.
"""
import json
import os
import sys
from pathlib import Path
from contextlib import ExitStack,contextmanager,nullcontext

from vllm.v1.worker.gpu_worker import Worker
from boltbeam.runtime.vllm_replay_steps import ScheduledWork
from boltbeam.runtime import vllm_semantics as semantics


class ReplayWorker(semantics.GraphProvenance, Worker):
  def __init__(self,*args,**kwargs):
    folder=os.environ.get('BOLTBEAM_REPLAY_CAPTURE_DIR')
    if not folder:raise ValueError('replay capture worker requires BOLTBEAM_REPLAY_CAPTURE_DIR')
    super().__init__(*args,**kwargs)
    self._bb_root=Path(folder);self._bb_pass=None;self._bb_stack=ExitStack()
    self._bb_provenance=ExitStack();self._bb_triton=None
    if os.environ.get('BOLTBEAM_REPLAY_TRITON_EVIDENCE')=='1':
      from boltbeam.runtime.triton_evidence import Capture
      self._bb_triton=self._bb_provenance.enter_context(Capture(self._bb_root/f'triton-{os.getpid()}').install())
    self._bb_stream=None;self._bb_pending=None;self._bb_profile_active=False
    self._bb_routing_installed=False;self._bb_routing_active=False
    selected=os.environ.get('BOLTBEAM_REPLAY_PROFILE_STEP')
    last=os.environ.get('BOLTBEAM_REPLAY_PROFILE_END_STEP',selected)
    self._bb_profile_step=int(selected) if selected is not None else None
    self._bb_profile_end_step=int(last) if last is not None else None
    if last is not None and (selected is None or not 0<=int(selected)<=int(last)):
      raise ValueError('replay profiler range requires ordered nonnegative steps')

  def load_model(self,**kwargs):
    if os.environ.get('BOLTBEAM_REPLAY_CAPTURE_INITIALIZATION')=='1':
      raise ValueError('profiling initialization can change autotuned kernels; start replay capture after warmup')
    result=super().load_model(**kwargs)
    self._bb_model=semantics.RunnerModel(self.model_runner)
    return result

  @property
  def capture_graphs(self):
    return os.environ.get('BOLTBEAM_REPLAY_CAPTURE_GRAPHS')=='1'

  @contextmanager
  def _get_cudagraph_capture_context(self):
    with super()._get_cudagraph_capture_context():
      with self._bb_triton.phase(identity_only=True) if self._bb_triton and self.capture_graphs else nullcontext():yield

  def boltbeam_capture(self,name):
    """Switch only between requests; 'off' flushes evidence and restores production."""
    if name in ('pause','resume'):
      if self._bb_stream is None or self._bb_pending is not None or self._bb_profile_active:
        raise ValueError('profiler boundaries require an active pass between scheduler steps')
      if self._bb_profile_step is not None:
        raise ValueError('turn boundaries cannot override a selected profiler step range')
      import torch
      torch.cuda.synchronize()
      (torch.cuda.profiler.stop if name=='pause' else torch.cuda.profiler.start)()
      self._bb_stream.flush()
      return {'capture_pass':self._bb_pass,'profiler':name}
    if name=='flush':
      if self._bb_stream is None:raise ValueError('no replay capture is active')
      self._bb_stream.flush();return {'capture_pass':self._bb_pass}
    if name not in ('off','timing','label'):raise ValueError('unknown replay capture pass')
    if self._bb_pending is not None or self._bb_profile_active:
      raise ValueError('cannot switch capture pass inside a scheduler step')
    self._bb_stack.close();self._bb_stack=ExitStack()
    if self._bb_stream is not None:
      import torch
      torch.cuda.synchronize();torch.cuda.profiler.stop()
      self._bb_stream.close();self._bb_stream=None
      self._bb_log.with_suffix('.libraries.json').write_text(json.dumps(semantics.loaded_libraries()))
    self._bb_pass=None
    if name=='off':
      import torch
      torch.cuda.synchronize();torch.cuda.profiler.stop()
      return {'capture_pass':'off'}
    config=self.vllm_config
    if config.model_config.enforce_eager or not config.compilation_config.mode:
      raise ValueError('paired replay requires production compilation')
    root=self._bb_root/name;root.mkdir(parents=True,exist_ok=True)
    self._bb_log=root/f'worker-{os.getpid()}.jsonl'
    self._bb_stream=self._bb_log.open('x',buffering=1024*1024)
    self._bb_steps=ScheduledWork()
    if name=='label':
      self._bb_stack.enter_context(semantics.without_graph_replay(self._bb_model))
      self._bb_stack.enter_context(semantics.module_ranges(self._bb_model))
      self._bb_dispatch_mode=semantics.dispatch_mode(self._bb_model,runtime_operators=self._bb_triton is not None)
      if not self._bb_routing_installed:
        semantics.install_routing(self._bb_model,active=lambda:self._bb_routing_active)
        self._bb_routing_installed=True
    import torch
    if self._bb_profile_step is None:torch.cuda.profiler.start()
    self._bb_pass=name
    import vllm
    from triton import knobs
    facts={**semantics.config_facts(config),'build':vllm.__version__,
      'loaded_packages':{name:getattr(sys.modules[name],'__version__',None)
                         for name in ('torch','triton','flashinfer') if name in sys.modules},
      **{key:getattr(config.model_config,key) for key in ('seed','max_model_len','enforce_eager')},
      **{key:getattr(config.scheduler_config,key) for key in ('max_num_batched_tokens','max_num_seqs','async_scheduling')},
      **{key:getattr(config.cache_config,key) for key in ('gpu_memory_utilization','mamba_cache_mode','mamba_ssm_cache_dtype')},
      'chunked_prefill':config.scheduler_config.enable_chunked_prefill,
      'prefix_caching':config.cache_config.enable_prefix_caching,
      'compilation_config':semantics.compilation_facts(config.compilation_config),
      'cudagraph_mode':config.compilation_config.cudagraph_mode.name,
      'runtime_graph_mode':'production' if name=='timing' else 'NONE',
      'profile_step':self._bb_profile_step,'profile_end_step':self._bb_profile_end_step,
      'capture_pass':name,'triton_cache_dir':knobs.cache.dir,'triton_autotune_cache':bool(knobs.autotuning.cache),
      'logits_processors':[f'{p.__module__}:{p.__qualname__}' if isinstance(p,type) else str(p)
                           for p in (config.model_config.logits_processors or [])],
      'execution_mode':'server production CUDA graph timing' if name=='timing' else 'server compiled graphless labeling'}
    self._bb_log.with_suffix('.model.json').write_text(json.dumps(facts))
    self._bb_stream.write(json.dumps({'schema':'boltbeam.replay_steps.v1','pid':os.getpid(),'rank':self.rank})+'\n')
    return {'capture_pass':name}

  def execute_model(self,scheduler_output):
    if self._bb_pass is None:return super().execute_model(scheduler_output)
    if self._bb_pending is not None:raise ValueError('replay execution would overwrite an unsampled step')
    record=self._bb_steps.record(scheduler_output)
    with self._capture_phase({**record,'phase':'execute'}):
      result=super().execute_model(scheduler_output)
    if record.get('scheduled_tokens',0):self._bb_pending=record
    return result

  def sample_tokens(self,grammar_output):
    if self._bb_pass is None:return super().sample_tokens(grammar_output)
    record=self._bb_pending
    if record is None:raise ValueError('replay sampling has no pending scheduler step')
    self._bb_pending=None
    with self._capture_phase({**record,'phase':'sample'}):
      return super().sample_tokens(grammar_output)

  @contextmanager
  def _capture_phase(self,record):
    import torch
    record['capture_pass']=self._bb_pass
    if record['step']==self._bb_profile_step and record['phase']=='execute':
      if not record['scheduled_tokens']:raise ValueError('selected replay profiler step has no work')
      torch.cuda.synchronize();torch.cuda.profiler.start();self._bb_profile_active=True
    labeling=self._bb_pass=='label';self._bb_routing_active=labeling
    torch.cuda.nvtx.range_push('boltbeam-step:'+json.dumps(record))
    try:
      with self._bb_dispatch_mode if labeling else nullcontext():
        with self._bb_triton.phase(identity_only=not labeling) if self._bb_triton else nullcontext():yield
    except Exception:
      record['status']='failed';raise
    else:record['status']='submitted'
    finally:
      self._bb_routing_active=False;torch.cuda.nvtx.range_pop()
      if self._bb_profile_active and (record['status']=='failed' or
          (record['phase']=='sample' and record['step']==self._bb_profile_end_step)):
        torch.cuda.synchronize();torch.cuda.profiler.stop();self._bb_profile_active=False
      try:
        if labeling and record['status']=='submitted':record['routing']=semantics.read_routing(self._bb_model)
      except Exception as exc:
        record.update(status='capture_failed',capture_error=str(exc))
        if self._bb_profile_active:
          torch.cuda.synchronize();torch.cuda.profiler.stop();self._bb_profile_active=False
        raise
      finally:self._bb_stream.write(json.dumps(record)+'\n')

  def shutdown(self):
    self._bb_stack.close()
    self._bb_provenance.close()
    if self._bb_stream is not None:self._bb_stream.close()
    return super().shutdown()
