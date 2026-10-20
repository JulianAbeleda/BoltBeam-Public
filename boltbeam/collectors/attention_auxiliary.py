"""Auxiliary attention work from observed operator operands and launch geometry.

FA2 combine's scalar dot products are a lower bound: softmax transcendental and
indexing work is not included. Intermediate bytes require matched DRAM/L2
counters, never an assumption that a small buffer resides in cache.

FA2 contract: vllm-project/flash-attention, 2c839c33742309ec41e620bf837495ec9926c56e,
csrc/flash_attn/src/flash_fwd_kernel.h, combine_attn_seqk_parallel.
Combine admits only the matching single-token ABI and observed geometry.
Unquantized cache writes use vLLM db9527a46, csrc/libtorch_stable/cache_kernels.cu:
every token reads its int64 slot; a negative slot skips the K/V transfer.
"""
import math
import re
from boltbeam.collectors.instruction_path import template
from boltbeam.roofline.prefill_work import elem_bytes


def geometry(launch):
  match=re.search(r'grid=(\d+)x(\d+)x(\d+) block=(\d+)x(\d+)x(\d+)',launch['key'])
  if not match:raise ValueError('attention auxiliary needs recorded launch geometry')
  values=tuple(map(int,match.groups()))
  if min(values)<=0:raise ValueError('invalid attention auxiliary launch geometry')
  return values[:3],values[3:]


def work(group):
  """Yield (launch, work) for reviewed attention auxiliary operators."""
  if not group:return
  semantic=group[0].get('semantic') or {}
  if semantic.get('class')!='Attention':return
  inputs=semantic.get('inputs') or []
  operator=semantic.get('operator')
  formats={'bfloat16':'BF16','float16':'F16'}
  if operator=='vllm.unified_kv_cache_update.default':
    if len(group)!=1 or len(inputs)!=2:return
    key=template(group[0]['key'])
    dtype=inputs[0].get('dtype');scalar={'bfloat16':'__nv_bfloat16','float16':'__half'}.get(dtype)
    if key!=f'vllm::reshape_and_cache_flash_kernel<{scalar},{scalar},(vllm::Fp8KVCacheDataType)0>':return
    shape=inputs[0]['shape']
    if inputs[1]!=inputs[0] or len(shape)!=3 or any(type(n) is not int or n<=0 for n in shape):return
    grid,_=geometry(group[0])
    if grid[1:]!=(1,1) or grid[0]>shape[0]:raise ValueError('KV update has inconsistent launch geometry')
    # Padding can make the operand larger than the slot vector. Each launched
    # block reads one slot, but negative slots skip the data transfer. Actual
    # K/V traffic must come from matched counters, not assumed valid slots.
    yield group[0],{'name':'KV cache write','flops':0.,'bytes':8.*grid[0],'path':'memory_only',
                   'quant':formats[dtype],'basis':'unquantized copy and one int64 slot read per launched token block',
                   'limitations':['Logical bytes count mandatory slot reads only; conditional K/V transfers require measured memory traffic. Integer indexing is excluded; no floating-point throughput or MFU is claimed.']}
  elif operator=='vllm.unified_attention_with_output.default':
    combine=[l for l in group if template(l['key']).startswith('flash::flash_fwd_splitkv_combine_kernel<')]
    main=[l for l in group if template(l['key']).startswith('flash::flash_fwd_splitkv_kernel<')]
    if not combine:return
    if len(combine)!=1 or len(main)!=1 or len(group)!=2:return
    if len(inputs)<4:return
    q,k,v,o=inputs[:4];shape=q.get('shape',[])
    if (len(shape)!=3 or shape[0]!=1 or o!=q or k!=v or len(k['shape'])!=3 or k['shape'][0]!=1 or
        q['dtype'] not in formats or k['dtype']!=q['dtype'] or k['shape'][2]!=shape[2]):return
    _,heads,dim=shape;kvheads=k['shape'][1]
    if min(heads,dim,kvheads)<=0 or heads%kvheads:raise ValueError('invalid grouped attention operands')
    key=template(combine[0]['key'])
    traits=re.search(r'Flash_fwd_kernel_traits<(\d+),',key)
    tail=re.search(r'>,(\d+),(\d+),([01])>$',key)
    if not traits or not tail or int(traits[1])!=dim:return
    block_m,log_splits,even=map(int,tail.groups())
    if not even or block_m not in (4,8,16,32) or not 0<=log_splits<=7:return
    grid,block=geometry(main[0]);splits=grid[1]
    cgrid,cblock=geometry(combine[0])
    if (grid[0]!=1 or grid[2]!=kvheads or splits<=1 or splits>2**log_splits or
        cgrid!=(math.ceil(heads/block_m),1,1) or cblock!=(128,1,1)):
      raise ValueError('split-KV combine geometry disagrees with observed attention operands')
    # Each FP32 partial vector is scaled and accumulated. FP32 LSE partials
    # are read; the final LSE and output vector are written.
    yield combine[0],{'name':'attention split-KV combine','flops':2.*heads*dim*splits,
      'bytes':4*heads*splits*(dim+1)+4*heads+elem_bytes(formats[q['dtype']])*heads*dim,
      'quant':formats[q['dtype']],'splits':splits,
      'basis':'FA2 split-KV scalar accumulation from observed Q/K/output shapes and producer split grid',
      'limitations':['Compute is a lower bound from scalar multiply-adds; softmax transcendental, reduction and address work are excluded.']}
