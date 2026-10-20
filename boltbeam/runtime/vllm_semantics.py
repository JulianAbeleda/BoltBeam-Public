"""NVTX evidence from loaded vLLM modules and dispatched operators.

Facts are shared by both observers. Execution mode belongs to the driver and is
recorded with the capture; these observers never substitute benchmark timings.
"""
import json
from contextvars import ContextVar
from contextlib import contextmanager

_operation = ContextVar('boltbeam_operation', default=None)


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


@contextmanager
def without_graph_replay(llm):
  """Run the already compiled production model without replaying its graphs."""
  from dataclasses import replace
  from vllm.config import CUDAGraphMode
  runner=(llm.runner if isinstance(llm,RunnerModel) else
          llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner)
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
  finally:manager.dispatch=original;state.prepare_attn=prepare



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
    try:
      return original(*args, **kwargs)
    finally:
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


def model_modules(model):
  packed=getattr(model,'packed_modules_mapping',{})
  for name,mod in model.named_modules():
    parent,_,leaf=name.rpartition('.')
    parts=[f'{parent}.{part}' if parent else part for part in packed.get(leaf,[leaf])]
    yield name,mod,{**module_facts(name,mod),'checkpoint_modules':parts}


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
  def facts(model):
    by_ptr={};by_name={}
    for name,mod,f in model_modules(model):
      # Quantized engines keep weights/scales under names such as weight_packed
      # and qweight. Attribute by actual storage, not one parameter spelling.
      for _,w in mod.named_parameters(recurse=False):
        by_ptr.setdefault(w.untyped_storage().data_ptr(),[]).append(f)
      by_name[name]=f
    return by_ptr,by_name
  pointers,names=llm.apply_model(facts)[0]
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
      f={**f,'operation_id':self.operation_id}
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


def _module_ranges(model):
  import torch
  records = []; handles=[]
  for name, mod, facts in model_modules(model):
    if not name:
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
