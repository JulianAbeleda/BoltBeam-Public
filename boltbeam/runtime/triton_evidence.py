"""Opt-in compiled-kernel provenance for paired captures, never native timing inference.

Observe loaded binaries rather than searching a shared compiler cache by name.
Runtime arguments contain shapes and scalar launch parameters, not tensor data.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace


def argument(value):
  if value is None or type(value) in (bool,int,str):return value
  if type(value) is float and math.isfinite(value):return value
  if hasattr(value,'shape') and hasattr(value,'dtype') and hasattr(value,'stride'):
    return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype)}
  return {'unsupported_type':type(value).__name__}


class Capture:
  def __init__(self,root):
    self.root=Path(root);self.active=False;self.identity_only=False;self.registered={}

  def register(self,kernel):
    binary=kernel.asm['cubin'];digest=hashlib.sha256(binary).hexdigest()
    key=kernel.hash
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',key):raise ValueError('invalid compiled-kernel identity key')
    if key not in self.registered:
      folder=self.root/key;folder.mkdir(parents=True,exist_ok=True)
      artifacts={}
      for ext in ('cubin','ptx','ttir','ttgir','llir','source'):
        value=kernel.asm.get(ext)
        if value is None:continue
        content=value if isinstance(value,bytes) else value.encode()
        (folder/('kernel.'+ext)).write_bytes(content)
        artifacts[ext]=hashlib.sha256(content).hexdigest()
      fn=getattr(kernel.src,'fn',None)
      python_source=getattr(fn,'src',None)
      location=re.search(r'loc\("([^"\n]+\.py)"',kernel.asm.get('source',''))
      if python_source is None and location and Path(location[1]).is_file():
        python_source=Path(location[1]).read_text()
      if isinstance(python_source,str):
        content=python_source.encode();(folder/'kernel.py').write_bytes(content)
        artifacts['py']=hashlib.sha256(content).hexdigest()
      record={'name':kernel.name,'cache_key':key,'binary_sha256':digest,'artifacts':artifacts,
              'signature':{str(k):v for k,v in kernel.src.signature.items()},
              'constants':{str(k):argument(v) for k,v in getattr(kernel.src,'constants',{}).items()},
              'num_warps':kernel.metadata.num_warps,'shared_bytes':kernel.metadata.shared}
      (folder/'identity.json').write_text(json.dumps(record,indent=2))
      self.registered[key]=record
    elif self.registered[key]['binary_sha256']!=digest:
      raise ValueError('Triton cache key refers to different loaded binaries')
    return key

  @contextmanager
  def install(self):
    from triton.compiler.compiler import CompiledKernel
    from triton.backends.nvidia.driver import CudaLauncher
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult,CachingAutotuner
    from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
    import torch
    initialize=CompiledKernel._init_handles;launch=CudaLauncher.__call__
    make_static=StaticTritonCompileResult.make_launcher;static_run=StaticallyLaunchedCudaKernel.run
    build_fast=CachingAutotuner._build_fast_launcher
    fast_runners=[]
    capture=self
    def init(kernel):
      initialize(kernel)
      if getattr(kernel._run,'_boltbeam_kernel',None) not in capture.registered:
        kernel._run._boltbeam_kernel=capture.register(kernel)
    def make(result):
      kernel=result.kernel
      if not kernel.cubin_path:result.reload_cubin_path()
      path=Path(kernel.cubin_path)
      asm={ext:p.read_bytes() if ext=='cubin' else p.read_text() for ext in ('cubin','ptx','ttir','ttgir','llir','source')
           if (p:=path.with_suffix('.'+ext)).exists()}
      proxy=SimpleNamespace(hash=kernel.hash,name=kernel.name,asm=asm,
        metadata=SimpleNamespace(num_warps=kernel.num_warps,shared=kernel.shared),
        src=SimpleNamespace(signature=result.compile_meta['signature'],constants=result.compile_meta['constants']))
      kernel._boltbeam_kernel=capture.register(proxy)
      _,_,none_args=result._get_arg_lists(kernel.arg_names,kernel.declared_constexprs)
      kernel._boltbeam_arg_names=[n for i,n in enumerate(kernel.arg_names) if i not in kernel.full_constexprs and n not in none_args]
      return make_static(result)
    @contextmanager
    def scope(key,grid,args,names=None):
      if key not in capture.registered:raise ValueError('Triton launch lacks loaded binary provenance')
      record={'cache_key':key,'binary_sha256':capture.registered[key]['binary_sha256']}
      if not capture.identity_only:
        record.update(grid=grid,args=[argument(v) for v in args])
        if names is not None:record['argument_names']=names
      torch.cuda.nvtx.range_push('boltbeam-triton:'+json.dumps(record,separators=(',',':')))
      try:yield
      finally:torch.cuda.nvtx.range_pop()
    def static(kernel,gx,gy,gz,stream,*args):
      if not capture.active:return static_run(kernel,gx,gy,gz,stream,*args)
      with scope(kernel._boltbeam_kernel,[gx,gy,gz],args,kernel._boltbeam_arg_names):
        return static_run(kernel,gx,gy,gz,stream,*args)
    def fast(autotuner,launcher):
      result=build_fast(autotuner,launcher)
      if result is None:return None
      kernel=launcher.__globals__['runner'].__self__;original=result.__globals__['runner']
      fast_runners.append((result.__globals__,original))
      def observed(gx,gy,gz,stream,*args):
        if not capture.active:return original(gx,gy,gz,stream,*args)
        with scope(kernel._boltbeam_kernel,[gx,gy,gz],args,kernel._boltbeam_arg_names):
          return original(gx,gy,gz,stream,*args)
      result.__globals__['runner']=observed
      return result
    def call(launcher,gx,gy,gz,stream,function,metadata,launch_metadata,enter,leave,*args):
      if not capture.active:
        return launch(launcher,gx,gy,gz,stream,function,metadata,launch_metadata,enter,leave,*args)
      key=getattr(launcher,'_boltbeam_kernel',None)
      with scope(key,[gx,gy,gz],args):
        return launch(launcher,gx,gy,gz,stream,function,metadata,launch_metadata,enter,leave,*args)
    CompiledKernel._init_handles=init;CudaLauncher.__call__=call
    StaticTritonCompileResult.make_launcher=make;StaticallyLaunchedCudaKernel.run=static
    CachingAutotuner._build_fast_launcher=fast
    try:yield self
    finally:
      for namespace,runner in reversed(fast_runners):namespace['runner']=runner
      CompiledKernel._init_handles=initialize;CudaLauncher.__call__=launch
      StaticTritonCompileResult.make_launcher=make_static;StaticallyLaunchedCudaKernel.run=static_run
      CachingAutotuner._build_fast_launcher=build_fast

  @contextmanager
  def phase(self,*,identity_only=False):
    previous=self.active,self.identity_only;self.active=True;self.identity_only=identity_only
    try:yield
    finally:self.active,self.identity_only=previous
