"""NVTX evidence from loaded vLLM modules and dispatched operators.

Facts are shared by both observers. Execution mode belongs to the driver and is
recorded with the capture; these observers never substitute benchmark timings.
"""
import json
from contextvars import ContextVar
from contextlib import contextmanager

_operation = ContextVar('boltbeam_operation', default=None)
_drafting = ContextVar('boltbeam_drafting', default=False)  # inside the speculator's propose (draft_scope)
DRAFT_ROOT = 'mtp'  # vLLM's own prefix for the draft head's layers (its attention layer is mtp.layers.0.self_attn.attn)


def hip_profiler_window():
  """On ROCm, torch.cuda.profiler.start/stop open and close rocprofv3's --selected-regions window (roctx), which
  hipProfilerStart does not; torch's nvtx ranges already reach rocprofv3 as roctx markers."""
  import ctypes,os
  try:import torch
  except ImportError:return
  if not getattr(torch.version,'hip',None) or getattr(torch.cuda.profiler,'_boltbeam_roctx',False):return
  for name in ('librocprofiler-sdk-roctx.so.1',os.path.join(os.environ.get('ROCM_PATH','/opt/rocm'),'lib','librocprofiler-sdk-roctx.so.1')):
    try:lib=ctypes.CDLL(name);break
    except OSError:lib=None
  if lib is None:raise RuntimeError("ROCm's roctx library (librocprofiler-sdk-roctx) is not here: a capture window cannot open")
  torch.cuda.profiler.start=lambda:lib.roctxProfilerResume(0);torch.cuda.profiler.stop=lambda:lib.roctxProfilerPause(0)
  torch.cuda.profiler._boltbeam_roctx=True


hip_profiler_window()


class GraphProvenance:
  """Start tracing only after vLLM kernel warmup has chosen production tactics."""
  stop_after_graphs = True

  @contextmanager
  def _get_cudagraph_capture_context(self):
    with super()._get_cudagraph_capture_context():
      if not self.capture_graphs:
        yield;return
      import torch
      torch.cuda.synchronize();torch.cuda.profiler.start()
      try:yield
      except BaseException:
        torch.cuda.synchronize();torch.cuda.profiler.stop()
        raise
      else:
        if self.stop_after_graphs:
          torch.cuda.synchronize();torch.cuda.profiler.stop()


class RunnerModel:
  """The same model-observer interface for offline LLMs and server workers."""
  def __init__(self, runner):
    self.runner = runner

  def apply_model(self, fn):
    return [fn(self.runner.model)]


def model_runner(llm):
  """The engine's GPU model runner, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) or a server worker's."""
  return (llm.runner if isinstance(llm,RunnerModel) else
          llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner)


def speculator(llm):
  """The runner's speculative-decoding proposer (vLLM's MTPSpeculator), or None without speculation. An engine
  configured to speculate must have one: a missing proposer there is an error, not 'no speculation'."""
  if not isinstance(llm,RunnerModel):
    config=getattr(getattr(getattr(llm,'llm_engine',None),'vllm_config',None),'speculative_config',None)
    if config is None:return None
  spec=getattr(model_runner(llm),'speculator',None)
  if spec is None and not isinstance(llm,RunnerModel):
    raise ValueError('the engine is configured to speculate, and its model runner holds no speculator')
  return spec


def draft_name(name):
  """A draft-head module's name as its checkpoint stores it: vLLM loads mtp.* into the drafter's model.* (Qwen3_5MTP),
  so model.layers.0.mlp.down_proj is mtp.layers.0.mlp.down_proj, and the drafter's own lm_head is mtp.lm_head."""
  return DRAFT_ROOT+'.'+name.removeprefix('model.') if name else DRAFT_ROOT


@contextmanager
def draft_scope(llm):
  """Mark every operation the speculator dispatches (its whole propose) as draft work, so a weight the draft head
  shares with the target (lm_head, the embedding) is labelled by the phase that read it, not only by its owner."""
  spec=speculator(llm)
  if spec is None:
    yield;return
  original=spec.propose
  import torch
  # one range over the whole propose: a kernel no module or operator range names (the drafter's fused norms, its
  # input updates) is still the draft head's
  scope='boltbeam:'+json.dumps({'module':DRAFT_ROOT,'class':type(spec).__name__,'draft':True},separators=(',',':'))
  def propose(*args,**kwargs):
    token=_drafting.set(True)
    torch.cuda.nvtx.range_push(scope)
    try:return original(*args,**kwargs)
    finally:
      torch.cuda.nvtx.range_pop()
      _drafting.reset(token)
  spec.propose=propose
  try:yield
  finally:del spec.propose  # the instance override; the class's method again


@contextmanager
def without_graph_replay(llm):
  """Run the already compiled production model without replaying its graphs; with speculation, the draft head's
  graphs too (its prefill over the verified tokens and its fused draft decode), each built with the attention plan the
  production graph was captured with."""
  from dataclasses import replace
  from vllm.config import CUDAGraphMode
  runner=model_runner(llm)
  spec=getattr(runner,'speculator',None)
  restore=[]
  if spec is not None:
    for manager_name in ('prefill_cudagraph_manager','decode_cudagraph_manager'):
      draft_manager=getattr(spec,manager_name,None)
      if draft_manager is None:continue
      chosen=draft_manager.dispatch
      def draft_dispatch(*args,_chosen=chosen,_name=manager_name,**kwargs):
        selected_desc=_chosen(*args,**kwargs)
        spec._boltbeam_selected=getattr(spec,'_boltbeam_selected',{});spec._boltbeam_selected[_name]=selected_desc
        return replace(selected_desc,cg_mode=CUDAGraphMode.NONE)
      draft_manager.dispatch=draft_dispatch
      restore.append((draft_manager,'dispatch',None))
    # Draft decode builds its own attention metadata; a FULL production graph was captured planning attention for
    # max_model_len keys, so the graphless pass plans it the same way (as prepare_labels does for the target)
    plan=spec._build_attn_metadata
    def draft_metadata(*args,**kwargs):
      selected_desc=(getattr(spec,'_boltbeam_selected',{}) or {}).get('decode_cudagraph_manager')
      if selected_desc is None or selected_desc.cg_mode!=CUDAGraphMode.FULL:
        return plan(*args,**kwargs)
      held=spec.draft_max_seq_len;spec.draft_max_seq_len=spec.max_model_len
      try:return plan(*args,**kwargs)
      finally:spec.draft_max_seq_len=held
    spec._build_attn_metadata=draft_metadata
    restore.append((spec,'_build_attn_metadata',None))
  manager=runner.cudagraph_manager
  original=manager.dispatch
  state=runner.model_state;prepare=state.prepare_attn
  metadata_globals=prepare.__func__.__globals__
  build=metadata_globals['build_attn_metadata']
  selected=None
  def dispatch(*args,**kwargs):
    nonlocal selected
    selected=original(*args,**kwargs)
    return replace(selected,cg_mode=CUDAGraphMode.NONE)
  def metadata(*args,**kwargs):
    # Graph construction fixes the maximum sequence bound at max_model_len.
    # Keep actual sequence lengths/state; only preserve that launch-planning bound.
    if selected.cg_mode==CUDAGraphMode.FULL:kwargs['max_seq_len']=state.max_model_len
    return build(*args,**kwargs)
  def prepare_labels(batch,mode,*args,**kwargs):
    if selected is None:raise ValueError('label metadata has no production batch descriptor')
    metadata_globals['build_attn_metadata']=metadata
    try:return prepare(batch,selected.cg_mode,*args,**kwargs)
    finally:metadata_globals['build_attn_metadata']=build
  manager.dispatch=dispatch;state.prepare_attn=prepare_labels
  try:yield
  finally:
    manager.dispatch=original;state.prepare_attn=prepare
    for owner,attribute,value in restore:
      if value is None:delattr(owner,attribute)  # an instance override over the class's method
      else:setattr(owner,attribute,value)



ENGINE_STEP=[None]  # the engine step observe_steps is inside, read by observe_speculation


@contextmanager
def observe_speculation(llm):
  """Every scheduler step of speculative decoding, as vLLM's scheduler records it: the engine step it was scheduled
  in (observe_steps), the tokens it scheduled and the draft tokens among them per request, and, once the engine has
  run it, the tokens it emitted per request and vLLM's own spec-decode counters for it (SpecDecodingStats: drafts,
  draft tokens, accepted tokens, accepted per draft position). Host-side bookkeeping only: nothing is read from the
  GPU, and no step is changed. A step the scheduler ran ahead (async scheduling) that the engine never reported back
  keeps no emitted tokens, and says so (reported False)."""
  scheduler=llm.llm_engine.engine_core.engine_core.scheduler
  schedule,update=scheduler.schedule,scheduler.update_from_output
  records=[];mark=object()
  def scheduled(*args,**kwargs):
    out=schedule(*args,**kwargs)
    out._boltbeam_schedule=(mark,len(records))
    records.append({'schedule':len(records),'engine_step':ENGINE_STEP[0],
                    'scheduled_tokens':int(out.total_num_scheduled_tokens),'reported':False,
                    'requests':{str(r):{'tokens':int(n),'draft_tokens':len(out.scheduled_spec_decode_tokens.get(r) or ())}
                                for r,n in out.num_scheduled_tokens.items()}})
    return out
  def updated(scheduler_output,runner_output):
    result=update(scheduler_output,runner_output)
    owner,index=getattr(scheduler_output,'_boltbeam_schedule',(None,None))
    if owner is mark:
      record=records[index];record['reported']=True
      sampled=runner_output.sampled_token_ids or []
      for request,row in record['requests'].items():
        at=runner_output.req_id_to_index.get(request)
        row['emitted']=len(sampled[at]) if at is not None and at<len(sampled) else 0
      stats=[o.scheduler_stats for o in result.values() if getattr(o,'scheduler_stats',None) is not None]
      spec=stats[0].spec_decoding_stats if stats else None
      record['spec_decoding_stats']=None if spec is None else {
        'drafts':int(spec.num_drafts),'draft_tokens':int(spec.num_draft_tokens),'accepted_tokens':int(spec.num_accepted_tokens),
        'accepted_per_position':[int(v) for v in spec.num_accepted_tokens_per_pos]}
    return result
  scheduler.schedule,scheduler.update_from_output=scheduled,updated
  try:yield records
  finally:
    del scheduler.schedule,scheduler.update_from_output


@contextmanager
def observe_steps(llm, profile_step=None, *, profile_end=None, capture_pass=None):
  """Label actual engine steps; optionally expose just one to GPU counters."""
  import torch
  engine = llm.llm_engine
  original = engine.step
  steps = []
  end=profile_step if profile_end is None else profile_end
  if end is not None and (profile_step is None or end<profile_step):
    raise ValueError('profiler range requires ordered start and end steps')
  active=False
  def observed(*args, **kwargs):
    nonlocal active
    index = len(steps)
    steps.append(index)
    selected = profile_step == index
    if selected:
      torch.cuda.synchronize()
      torch.cuda.profiler.start()
      active=True
    torch.cuda.nvtx.range_push('boltbeam-step:'+json.dumps({'step':index,**({'capture_pass':capture_pass} if capture_pass else {})}))
    ENGINE_STEP[0]=index
    try:
      return original(*args, **kwargs)
    finally:
      ENGINE_STEP[0]=None
      torch.cuda.nvtx.range_pop()
      if active and index==end:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        active=False
  engine.step = observed
  try:
    yield steps
    if end is not None and end not in steps:
      raise ValueError('requested profiler step was not executed')
  finally:
    if active:
      torch.cuda.synchronize()
      torch.cuda.profiler.stop()
    engine.step = original


def module_facts(name, mod):
  """One schema for loaded dimensions/dtypes, independent of observation method."""
  import torch
  facts={'module':name,'class':type(mod).__name__}
  if hasattr(mod,'get_state_shape') and hasattr(mod,'get_state_dtype'):
    facts.update(state_shapes=mod.get_state_shape(),
                 state_dtypes=[str(d).removeprefix('torch.') for d in mod.get_state_dtype()])
    # Small recurrent constants can use a different dtype from activations and
    # state. Record loaded tensors instead of assuming FP32 or checkpoint dtype.
    parameters = {}
    for key in ('A', 'D', 'dt_bias', 'conv_weights'):
      value = getattr(mod, key, None)
      if isinstance(value, torch.Tensor):
        parameters[key] = {'shape': list(value.shape),
                           'dtype': str(value.dtype).removeprefix('torch.')}
    conv = getattr(mod, 'conv1d', None)
    bias = getattr(conv, 'bias', None)
    parameters['conv_bias'] = ({'shape': list(bias.shape),
                                'dtype': str(bias.dtype).removeprefix('torch.')}
                               if isinstance(bias, torch.Tensor) else None)
    facts['recurrent_parameters'] = parameters
  weight=getattr(mod,'weight',None)
  if isinstance(weight,torch.Tensor):
    facts.update(weight_shape=list(weight.shape),weight_dtype=str(weight.dtype).removeprefix('torch.'))
  for key in ('input_size_per_partition','output_size_per_partition','num_experts',
              'top_k','hidden_size','intermediate_size_per_partition','chunk_size',
              'num_heads','head_dim','ssm_state_size','n_groups'):
    value=getattr(mod,key,None)
    if isinstance(value,(int,str,bool)):
      facts[key]=value
  return facts


def model_modules(model, *, draft=False):
  """(name, module, facts) for every module. draft: the speculator's draft head, its modules named as its checkpoint
  names them (draft_name) and marked draft, so its work is never read as the target model's."""
  packed=getattr(model,'packed_modules_mapping',{})
  for name,mod in model.named_modules():
    shown=draft_name(name) if draft else name
    parent,_,leaf=shown.rpartition('.')
    parts=[f'{parent}.{part}' if parent else part for part in packed.get(leaf,[leaf])]
    yield shown,mod,{**module_facts(shown,mod),'checkpoint_modules':parts,**({'draft':True} if draft else {})}


def _owners(model, by_ptr, by_name, *, draft=False):
  for name,mod,f in model_modules(model,**({'draft':True} if draft else {})):
    # Quantized engines keep weights/scales under names such as weight_packed
    # and qweight. Attribute by actual storage, not one parameter spelling.
    for _,w in mod.named_parameters(recurse=False):
      address=w.untyped_storage().data_ptr()
      # a tensor the draft head shares with the target (lm_head, the embedding) keeps its one owner, the target's
      # module; which phase read it is the dispatch's draft mark (draft_scope), not a second owner
      if draft and address in by_ptr and not any(x.get('draft') for x in by_ptr[address]):continue
      by_ptr.setdefault(address,[]).append(f)
    by_name[name]=f


def parameter_owners(llm):
  """{parameter storage address: [module facts]} and {module name: facts}, from the loaded model and, with
  speculation, its draft head."""
  def facts(model):
    by_ptr={};by_name={}
    _owners(model,by_ptr,by_name)
    return by_ptr,by_name
  by_ptr,by_name=llm.apply_model(facts)[0]
  if (spec:=speculator(llm)) is not None:
    _owners(spec.model,by_ptr,by_name,draft=True)
  return by_ptr,by_name


@contextmanager
def inductor_ranges(llm):
  """Label the matrix products Inductor writes as its own Triton templates (triton_tem_*) with the module whose
  weights they read. Such a product is no dispatched operator, so dispatch_mode cannot see it: under
  max_autotune_gemm_backends=TRITON, Qwen3.8-27B's in_proj_ba ran as triton_tem_fused_mm_t_1 with no label (RTX 5090,
  2026-11-08). The owner is found the way dispatch_mode finds one, by the storage its tensor arguments point into;
  a kernel reading no parameter, or several modules' parameters, stays unlabeled."""
  import itertools,torch
  from torch._inductor.runtime.triton_heuristics import CachingAutotuner
  pointers,_=parameter_owners(llm)
  original=CachingAutotuner.run;ids=itertools.count(1_000_000)  # apart from dispatch_mode's operation ids
  def run(self,*args,stream,**kwargs):
    name=(getattr(self,'inductor_meta',None) or {}).get('kernel_name') or getattr(getattr(self,'fn',None),'__name__','')
    if not name.startswith('triton_tem_'):
      return original(self,*args,stream=stream,**kwargs)
    tensors=[x for x in args if isinstance(x,torch.Tensor)]
    owners={f['module']:f for x in tensors for f in pointers.get(x.untyped_storage().data_ptr(),[])}
    if len(owners)!=1:
      return original(self,*args,stream=stream,**kwargs)
    f=next(iter(owners.values()))
    inputs=[{'shape':list(x.shape),'dtype':str(x.dtype).removeprefix('torch.')}
            for x in tensors if x.untyped_storage().data_ptr() not in pointers]
    torch.cuda.nvtx.range_push('boltbeam:'+json.dumps({**f,'operation_id':next(ids),'operator':'inductor:'+name,
                                                       'inputs':inputs},separators=(',',':')))
    try:return original(self,*args,stream=stream,**kwargs)
    finally:torch.cuda.nvtx.range_pop()
  CachingAutotuner.run=run
  try:yield
  finally:CachingAutotuner.run=original


def dispatch_mode(llm,*,runtime_operators=False):
  """Annotate actual dispatched operators using loaded parameter storage identity.

  Unlike module hooks this does not require eager execution. CUDA graph replay
  may bypass operator dispatch; those launches remain unassigned and must fail
  coverage, never inherit an eager calibration by time/order guesses.
  """
  import torch
  from torch.utils._python_dispatch import TorchDispatchMode
  from torch.utils._pytree import tree_leaves,tree_map
  from vllm.utils.torch_utils import LayerName
  pointers,names=parameter_owners(llm)
  class Trace(TorchDispatchMode):
    operation_id = 0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
      kwargs=kwargs or {};leaves=tree_leaves((args,kwargs));owners=[]
      for x in leaves:
        if isinstance(x,torch.Tensor):owners.extend(pointers.get(x.untyped_storage().data_ptr(),[]))
        elif isinstance(x,str) and x in names:owners.append(names[x])
        elif isinstance(x,LayerName) and x.value in names:owners.append(names[x.value])
      unique={f['module']:f for f in owners}
      if len(unique)==1:f=next(iter(unique.values()))
      elif runtime_operators and (any(isinstance(x,torch.Tensor) and x.is_cuda for x in leaves) or
                                 str(kwargs.get('device','')).startswith('cuda')):
        from boltbeam.runtime.triton_evidence import argument
        def leaf(x):return str(x) if isinstance(x,(torch.dtype,torch.device)) else argument(x)
        f={'module':'<runtime>','class':'TorchOperator','parameter_owners':sorted(unique),
           'arguments':tree_map(leaf,args),'keyword_arguments':tree_map(leaf,kwargs)}
      else:return func(*args,**kwargs)
      self.operation_id += 1
      f={**f,'operation_id':self.operation_id,**({'draft':True} if _drafting.get() else {})}
      inputs=[{'shape':list(x.shape),'dtype':str(x.dtype).removeprefix('torch.')}
              for x in leaves if isinstance(x,torch.Tensor) and x.untyped_storage().data_ptr() not in pointers]
      torch.cuda.nvtx.range_push('boltbeam:'+json.dumps({**f,'operator':str(func),'inputs':inputs},separators=(',',':')))
      token=_operation.set(f)
      try:return func(*args,**kwargs)
      finally:
        _operation.reset(token)
        torch.cuda.nvtx.range_pop()
  return Trace()


def install_routing(llm, *, active=None):
  """Retain actual selected IDs; read them on the CPU only after the capture.

  No GPU copies, reductions or synchronizations are inserted in the measured
  call. The operation ID ties routing to its dispatch range, not its timing.
  This observer is for dispatched prefill, not CUDA-graph replay.
  """
  def attach(model):
    model._boltbeam_routing_modules = []
    for name,mod in model.named_modules():
      router=getattr(mod,'router',None)
      if type(mod).__name__!='MoERunner' or not hasattr(router,'select_experts'):
        continue
      model._boltbeam_routing_modules.append((name,mod))
      records=[]
      mod._boltbeam_routing=records
      original=router.select_experts
      def select(*args,_original=original,_records=records,_name=name,**kwargs):
        result=_original(*args,**kwargs)
        if active is not None and not active():
          return result
        operation=_operation.get()
        if operation is None or operation['module']!=_name:
          raise ValueError('routing has no matching dispatched operation')
        _records.append((operation['operation_id'],result[1].detach()))
        return result
      router.select_experts=select
  llm.apply_model(attach)


def read_routing(llm):
  def read(model):
    import numpy as np
    out=[]
    for name,mod in model._boltbeam_routing_modules:
      for operation,ids in getattr(mod,'_boltbeam_routing',[]):
        array=ids.cpu().numpy()
        if array.ndim!=2 or (array<0).any():
          raise ValueError('unsupported expert routing indices')
        ordered=np.sort(array,axis=1)
        if (ordered[:,1:]==ordered[:,:-1]).any():
          bad = np.flatnonzero((ordered[:,1:]==ordered[:,:-1]).any(axis=1))
          raise ValueError(f'expert selection repeats an expert within one token: module={name}, '
                           f'operation={operation}, shape={list(array.shape)}, '
                           f'bad_rows={bad[:8].tolist()}, first_ids={array[bad[0]].tolist()}')
        values,counts=np.unique(array,return_counts=True)
        out.append({'module':name,'operation_id':operation,'shape':list(array.shape),
                    'expert_counts':{str(int(k)):int(v) for k,v in zip(values,counts)}})
      if hasattr(mod,'_boltbeam_routing'):
        mod._boltbeam_routing.clear()
    return out
  return llm.apply_model(read)[0]


def _module_ranges(model, *, draft=False):
  import torch
  records = []; handles=[]
  for name, mod, facts in model_modules(model, **({'draft':True} if draft else {})):
    if not name or (draft and name==DRAFT_ROOT):
      continue
    records.append(facts)
    def enter(module, args, kwargs, facts=facts):
      tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
      torch.cuda.nvtx.range_push('boltbeam:' + json.dumps({**facts,
        'inputs': [{'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.')} for t in tensors]}, separators=(',', ':')))
    def leave(module, args, kwargs, output):
      torch.cuda.nvtx.range_pop()
    handles.append(mod.register_forward_pre_hook(enter, with_kwargs=True))
    handles.append(mod.register_forward_hook(leave, with_kwargs=True, always_call=True))
  return records,handles


def install(llm):
  return llm.apply_model(_module_ranges)[0][0]


@contextmanager
def module_ranges(llm):
  """Temporary module labels installed after production compilation and warmup."""
  records,handles=llm.apply_model(_module_ranges)[0]
  if (spec:=speculator(llm)) is not None:
    more,draft_handles=_module_ranges(spec.model,draft=True)
    records,handles=records+more,handles+draft_handles
  try:yield records
  finally:
    for handle in handles:handle.remove()


def config_facts(config):
  """Resolved model formats and configuration identity, shared by capture drivers."""
  from pathlib import Path
  dtype = str(config.model_config.dtype).removeprefix('torch.')
  cache = str(config.cache_config.cache_dtype)
  import hashlib
  config_path=Path(config.model_config.model)/'config.json'
  return {"model_path": str(Path(config.model_config.model).resolve()),
          "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
          "activation_dtype": dtype, "kv_cache_dtype": dtype if cache == 'auto' else cache,
          "model_type": config.model_config.hf_config.model_type,
          **({'mamba_chunk_size': config.model_config.get_mamba_chunk_size()}
             if hasattr(config.model_config.hf_config, 'ssm_state_size') else {})}


def compilation_facts(config):
  """Serialize settings without copying loaded model objects in runtime state."""
  from dataclasses import fields
  from pydantic import TypeAdapter
  exclude={field.name:True for field in fields(config) if not field.init}
  exclude.update(static_forward_context=True,inductor_compile_config={'post_grad_custom_post_pass':True})
  return json.loads(json.dumps(TypeAdapter(type(config)).dump_python(config,
    exclude=exclude),default=lambda value:sorted(value) if isinstance(value,set) else str(value)))


def loaded_libraries():
  with open("/proc/self/maps") as stream:
    return sorted({line.split()[-1] for line in stream
                   if ".so" in line.split()[-1] and line.split()[-1].startswith("/")})
