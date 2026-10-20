"""Greedy sampling work from reviewed complete PTX and observed device controls.

Both programs branch directly to BB0_34 when the loaded temperature is zero.
That path performs comparisons, selection, shuffles and integer indexing, but
no floating-point arithmetic. The PTX digest covers all inlined helpers too.
The stochastic path is deliberately unsupported, even with APPLY_TEMPERATURE=False.
"""
import math
from boltbeam.collectors.attention_auxiliary import geometry
from boltbeam.collectors.instruction_path import template

PROGRAMS = {
  'b9a2640a699d5fe4d53a58d6c649ef46e6d611d5825c4827e45a460d3bb64be8':('torch.float32','*fp32',4),
  'b82864c3ed39aae4238293aa177bf7812c63d9724fbb8b6f24a0081d50be7b6a':('torch.bfloat16','*bf16',2),
}
SIGNATURE = dict(zip(
  ('local_argmax_ptr','local_argmax_stride','local_max_ptr','local_max_stride',
   'logits_cache_ptr','logits_cache_stride_0','logits_cache_stride_1','logits_cache_col_ptr',
   'logits_ptr','logits_stride','expanded_idx_mapping_ptr','seeds_ptr','pos_ptr','temp_ptr',
   'vocab_size','BLOCK_SIZE','IS_DRAFTING','APPLY_TEMPERATURE','USE_FP64','PER_TOKEN_COL'),
  ('*i64','i32','*fp32','i32','constexpr','i32','i32','constexpr','*fp32','i32',
   '*i32','*i64','*i64','*fp32','i32','constexpr','constexpr','constexpr','constexpr','constexpr')))


def supports(record):
  return record['artifacts'].get('ptx') in PROGRAMS


def work(record,launch,evidence):
  control=launch.get('sampling_control')
  if control is None:return None
  if not isinstance(control,dict):raise ValueError('invalid sampling control witness')
  if template(launch['key'])!='_gumbel_sample_kernel' or record['name']!='_gumbel_sample_kernel':
    raise ValueError('sampling entry point mismatch')
  dtype,ty,size=PROGRAMS[record['artifacts']['ptx']]
  signature={**SIGNATURE,'logits_ptr':ty}
  if record['signature']!=signature:raise ValueError('sampling ABI mismatch')
  names=evidence.get('argument_names',list(SIGNATURE));values=evidence['args']
  if names!=list(SIGNATURE) or len(values)!=len(names):raise ValueError('sampling argument order mismatch')
  args=dict(zip(names,values));grid,block=geometry(launch);n=grid[0];vocab=args['vocab_size']
  if (type(vocab) is not int or vocab!=151936 or grid!=(n,149,1) or block!=(128,1,1)
      or record['num_warps']!=4 or grid!=tuple(evidence['grid'])):
    raise ValueError('sampling geometry or vocabulary mismatch')
  for key in ('cache_key','binary_sha256','grid'):
    if control.get(key)!=evidence[key]:raise ValueError('sampling control identity mismatch')
  mapping=control.get('mapping');temperatures=control.get('selected_temperatures')
  if (not isinstance(mapping,list) or len(mapping)!=n or any(type(i) is not int for i in mapping)
      or not isinstance(temperatures,list) or len(temperatures)!=n
      or any(type(t) not in (int,float) or not math.isfinite(t) for t in temperatures)):
    raise ValueError('invalid sampling device controls')
  if not all(t==0 for t in temperatures):return None
  if control.get('all_zero') is not True:raise ValueError('inconsistent sampling temperature witness')
  if any(args[k] is not False for k in ('IS_DRAFTING','APPLY_TEMPERATURE','USE_FP64','PER_TOKEN_COL')):
    raise ValueError('sampling specialization mismatch')
  if (args['BLOCK_SIZE']!=1024 or args['logits_cache_ptr'] is not None or args['logits_cache_col_ptr'] is not None
      or args['logits_cache_stride_0']!=0 or args['logits_cache_stride_1']!=0
      or args['logits_stride']!=vocab or args['local_argmax_stride']!=149 or args['local_max_stride']!=149):
    raise ValueError('sampling strides or cache specialization mismatch')
  if not isinstance(args['temp_ptr'],dict):raise ValueError('sampling temperature operand missing')
  reqs=args['temp_ptr'].get('shape',[])
  if len(reqs)!=1 or type(reqs[0]) is not int or reqs[0]<=0 or any(i>=reqs[0] for i in mapping):
    raise ValueError('sampling request index outside temperature buffer')
  shapes={'local_argmax_ptr':([n,149],'torch.int64'),'local_max_ptr':([n,149],'torch.float32'),
    'logits_ptr':([n,vocab],dtype),'expanded_idx_mapping_ptr':([n],'torch.int32'),
    'seeds_ptr':(reqs,'torch.int64'),'pos_ptr':([n],'torch.int64'),'temp_ptr':(reqs,'torch.float32')}
  for key,(shape,expected_dtype) in shapes.items():
    value=args[key]
    if (not isinstance(value,dict) or value.get('shape')!=shape or value.get('dtype')!=expected_dtype
        or value.get('stride')!=[math.prod(shape[i+1:]) for i in range(len(shape))]):
      raise ValueError('sampling operand mismatch: '+key)
  # Logical lower bound: logits; mapping and position; valid temperature/seed;
  # one int64 index and FP32 value per block. Repeated control loads are excluded.
  byte_count=n*vocab*size+12*n+12*sum(i>=0 for i in mapping)+12*n*149
  return {'name':'greedy sampling block argmax','what':'auxiliary','path':'memory_only',
    'quant':'F32' if size==4 else 'BF16','flops':0.,'bytes':float(byte_count),
    'requires_memory_evidence':True,'source_module':'Triton binary '+record['binary_sha256'],
    'basis':'complete reviewed PTX, exact loaded binary, runtime operands and same-launch zero-temperature device witness',
    'limitations':['Memory-only lower bound; comparisons, casts, integer indexing and shared-memory reductions are excluded. No floating-point throughput or MFU is claimed.']}
