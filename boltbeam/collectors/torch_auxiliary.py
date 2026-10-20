"""Minimum operand work for observed single-kernel Torch operators.

The replay caller binds loaded libraries and exact counter launch identity.
Only the listed operator ABIs and kernel families are admitted. Memory counters
remain mandatory; integer arithmetic, comparisons and conversions are excluded.
"""
import math
from boltbeam.collectors.instruction_path import template
from boltbeam.collectors.attention_auxiliary import geometry

TYPES={'bool':(1,'bool'),'int32':(4,'int'),'int64':(8,'long'),
       'float32':(4,'float'),'bfloat16':(2,'__nv_bfloat16')}


def shape(value):
  if not isinstance(value,list) or any(type(n) is not int or n<=0 for n in value):raise ValueError('invalid Torch operand dimensions')
  return value


def tensor(value):
  if not isinstance(value,dict):raise ValueError('Torch operand metadata missing')
  dims=shape(value.get('shape'));dtype=value.get('dtype','').removeprefix('torch.')
  if dtype not in TYPES or value.get('stride')!=[math.prod(dims[i+1:]) for i in range(len(dims))]:
    raise ValueError('unsupported Torch operand dtype or layout')
  return dims,dtype


def work(group):
  if len(group)!=1:return
  launch=group[0];s=launch.get('semantic') or {}
  if s.get('class')!='TorchOperator':return
  op=s.get('operator');args=s.get('arguments');kw=s.get('keyword_arguments')
  if not isinstance(args,list) or not isinstance(kw,dict):raise ValueError('Torch operator arguments missing')
  key=template(launch['key']);grid,block=geometry(launch)
  if grid[1:]!=(1,1) or block[1:]!=(1,1):return
  if op in ('aten.fill_.Scalar','aten.zeros.default','aten.new_ones.default'):
    if op=='aten.fill_.Scalar':
      if len(args)!=2 or kw:return
      dims,dtype=tensor(args[0])
    else:
      if set(kw)-{'dtype','device','pin_memory','layout'}:return
      if kw.get('layout') not in (None,'torch.strided') or kw.get('pin_memory') not in (None,False):return
      if kw.get('device') is not None and not str(kw['device']).startswith('cuda'):return
      if op=='aten.zeros.default':
        if len(args)!=1 or not str(kw.get('device','')).startswith('cuda'):return
        dims=shape(args[0]);dtype=str(kw.get('dtype','')).removeprefix('torch.')
      else:
        if len(args)!=2:return
        _,dtype=tensor(args[0]);dims=shape(args[1])
        if kw.get('dtype') is not None:dtype=str(kw['dtype']).removeprefix('torch.')
    if dtype not in TYPES:return
    if not key.startswith(('at::native::vectorized_elementwise_kernel<','at::native::unrolled_elementwise_kernel<')) or f'FillFunctor<{TYPES[dtype][1]}>' not in key:return
    nbytes=math.prod(dims)*TYPES[dtype][0];name='fill '+dtype
  elif op=='aten.arange.default':
    if len(args)!=1 or type(args[0]) is not int or args[0]<=0:return
    if set(kw)-{'dtype','device','pin_memory','layout'} or kw.get('dtype')!='torch.int32' or not str(kw.get('device','')).startswith('cuda'):return
    if kw.get('layout') not in (None,'torch.strided') or kw.get('pin_memory') not in (None,False):return
    if not key.startswith('<unnamed>::elementwise_kernel_with_index<int,') or 'arange_cuda_out' not in key:return
    dtype='int32';nbytes=4*args[0];name='arange int32'
  elif op=='aten.index.Tensor':
    if len(args)!=2 or kw or not isinstance(args[1],list) or len(args[1])!=1:return
    dims,dtype=tensor(args[0]);indices,index_type=tensor(args[1][0])
    if not dims or len(indices)!=1 or index_type!='int64':return
    width=TYPES[dtype][0]
    if not key.startswith('at::native::index_elementwise_kernel<') or f'index_kernel_impl<at::native::OpaqueType<{width}>>' not in key:return
    nbytes=indices[0]*(2*math.prod(dims[1:])*width+8);name='index '+dtype
  elif op=='aten.copy_.default':
    if len(args) not in (2,3) or kw or (len(args)==3 and type(args[2]) is not bool):return
    dims,dtype=tensor(args[0]);src,src_type=tensor(args[1])
    if src!=dims:return
    if not key.startswith('at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda('):return
    nbytes=math.prod(dims)*(TYPES[dtype][0]+TYPES[src_type][0]);name='copy '+src_type+' to '+dtype
  elif op=='aten.argmax.default':
    if len(args)!=3 or kw or type(args[1]) is not int or type(args[2]) is not bool:return
    dims,dtype=tensor(args[0])
    if not dims or args[1] not in (-1,len(dims)-1) or dtype!='float32':return
    if not key.startswith('at::native::reduce_kernel<') or 'ArgMaxOps<float>' not in key:return
    nbytes=4*math.prod(dims)+8*math.prod(dims[:-1]);name='argmax float32'
  elif op=='aten.gather.default':
    if len(args)!=3 or kw or type(args[1]) is not int:return
    dims,dtype=tensor(args[0]);indices,index_type=tensor(args[2]);width=TYPES[dtype][0]
    if not dims or args[1] not in (-1,len(dims)-1) or index_type!='int64' or len(indices)!=len(dims) or indices[:-1]!=dims[:-1]:return
    if not key.startswith('at::native::_scatter_gather_elementwise_kernel<') or f'_cuda_scatter_gather_internal_kernel<0,at::native::OpaqueType<{width}>,long>' not in key or 'TensorAssign' not in key:return
    nbytes=math.prod(indices)*(2*width+8);name='gather '+dtype
  else:return
  yield launch,{'name':'Torch '+name,'what':'auxiliary','path':'memory_only','quant':dtype,
    'flops':0.,'bytes':float(nbytes),'requires_memory_evidence':True,'source_module':op,
    'basis':'observed Torch operator ABI, operand dimensions and single matching GPU kernel; loaded libraries bound by replay',
    'limitations':['Memory-only lower bound; comparisons, integer arithmetic, casts and address operations are excluded. No floating-point throughput or MFU is claimed.']}
