"""Integer scheduler programs with source-proved minimum operand work.

The complete programs have no floating-point arithmetic. Programs calling the
pointer-loading helper additionally require the reviewed, fully inlined PTX.
Conditional and loop-dependent operands are not guessed. Matched counters still
supply the complete measured DRAM/L2 traffic used by the memory-only bound.
"""
import math
from boltbeam.collectors.attention_auxiliary import geometry

PROGRAMS = {
 '24b22f2e4c0ece2e6d1267cec0951493d01ca82877acf3212e4bf9cbcb1f3673':('position preparation',
  'pos_ptr:*i64 seq_lens_ptr:*i32 idx_mapping_ptr:*i32 query_start_loc_ptr:*i32 num_computed_tokens_ptr:*i32 max_num_reqs:constexpr BLOCK_SIZE:constexpr'),
 '910ea3f1b075feb5891a9e8a55c66c8a75c2106f1e75aa97eec3d3e50f0a6155':('sampled token preparation',
  'input_ids_ptr:*i32 idx_mapping_ptr:*i32 last_sampled_tokens_ptr:*i64 query_start_loc_ptr:*i32 seq_lens_ptr:*i32 prefill_len_ptr:*i32 draft_tokens_ptr:*i64 draft_tokens_stride:constexpr cu_num_logits_ptr:*i32 logits_indices_ptr:*i64 BLOCK_SIZE:constexpr NUM_NEW_SAMPLED_TOKENS:constexpr'),
 '4433c1fec46333b17d7e5b96baea870071b119f2c85208a162f53088bb11fe34':('request state update',
  'idx_mapping_ptr:*i32 num_computed_tokens_ptr:*i32 last_sampled_tokens_ptr:*i64 output_bin_counts_ptr:*i32 output_bin_counts_stride:i32 sampled_tokens_ptr:*i64 sampled_tokens_stride:constexpr num_sampled_ptr:*i32 num_rejected_ptr:*i32 query_start_loc_ptr:*i32 all_token_ids_ptr:*i32 all_token_ids_stride:i32 total_len_ptr:*i32'),
 '6f690c35acb689130f3b6554fb43920dd8d222c36795ecadc5370f5e8b96d565':('block table gathering',
  'batch_idx_to_req_idx:*i32 src_block_table_ptrs:*u64 dst_block_table_ptrs:*u64 block_table_strides:*i64 num_blocks_ptr:*i32 num_blocks_stride:constexpr num_reqs:i32 BLOCK_SIZE:constexpr'),
 '4261dc9cc571136ea6708c0d52b6533e60e12a6a9bfeca3866522f6a3da35227':('slot mapping',
  'max_num_tokens:i32 idx_mapping:*i32 query_start_loc:*i32 pos:*i64 block_table_ptrs:*u64 block_table_strides:*i64 block_sizes:*i32 kernel_block_sizes:*i32 slot_mapping_enabled:*u1 dcp_sharded:*u1 slot_mappings_ptr:*i64 slot_mappings_stride:i32 cp_rank:i32 CP_SIZE:constexpr CP_INTERLEAVE:constexpr PAD_ID:constexpr TRITON_BLOCK_SIZE:constexpr'),
 '2111026a6507a6574e667b46ee991ce0ff8c75a7e1dbadc026ae1ca3e11134d8':('prefill token preparation',
  'input_ids_ptr:*i32 next_prefill_tokens_ptr:*i32 next_prefill_tokens_stride:constexpr num_lookahead:constexpr idx_mapping_ptr:*i32 query_start_loc_ptr:*i32 all_token_ids_ptr:*i32 all_token_ids_stride:i32 prefill_lens_ptr:*i32 num_computed_tokens_ptr:*i32 BLOCK_SIZE:constexpr LOOKAHEAD_BLOCK:constexpr'),
 '614c09dbf9fb09f779864fa398d8ddc0051a6c9a7b86bfc11978973428b101d5':('staged buffer writes',
  'output_ptr:copy output_stride:stride write_indices_ptr:*i32 write_starts_ptr:*i32 write_contents_ptr:copy write_cu_lens_ptr:*i32 write_group_ids_ptr:constexpr BLOCK_SIZE:constexpr MULTI_GROUP:constexpr'),
}
INLINED_PTX = {
 'block table gathering':'0055ab3f320d3747ce5778f370031f4390f2f43d7fe11613e1510c6c2a7a01bf',
 'slot mapping':'6c4fd708573350d5a98798946b0aa559b70d0d6b338a601f282552a0ddbe94af',
}


def signature(contract):return dict(s.split(':') for s in PROGRAMS[contract][1].split())


def work(record,args,launch,evidence):
  kind,_=PROGRAMS[record['contract']];abi=signature(record['contract'])
  if kind=='staged buffer writes':
    ty=record['signature'].get('output_ptr');row_stride=record['signature'].get('output_stride')
    if ty not in ('*i32','*fp32') or row_stride not in ('constexpr','i32'):return None
    abi.update(output_ptr=ty,write_contents_ptr=ty,output_stride=row_stride)
  if kind in INLINED_PTX and record['artifacts'].get('ptx')!=INLINED_PTX[kind]:return None
  if list(record['signature'].items())!=list(abi.items()) or list(args)!=list(abi):raise ValueError('scheduler program ABI mismatch')
  for i,(name,ty) in enumerate(abi.items()):
    value=args[name]
    if ty.startswith('*'):
      shape=value.get('shape') if isinstance(value,dict) else None
      if (not shape or any(type(n) is not int or n<0 for n in shape) or
          value.get('dtype')!={'*i32':'torch.int32','*i64':'torch.int64','*u64':'torch.uint64','*u1':'torch.bool','*fp32':'torch.float32'}[ty] or
          value.get('stride')!=[math.prod(max(n,1) for n in shape[j+1:]) for j in range(len(shape))]):
        raise ValueError('scheduler operand layout or dtype mismatch: '+name)
      if math.prod(shape)==0 and name!='draft_tokens_ptr':raise ValueError('empty scheduler operand: '+name)
    elif kind=='staged buffer writes' and name in ('write_group_ids_ptr','MULTI_GROUP'):
      expected=None if name=='write_group_ids_ptr' else False
      if value is not expected or f'({i},)' not in record['constants'] or record['constants'][f'({i},)'] is not expected:
        raise ValueError('staged writes require the single-output specialization')
    elif type(value) is not int or (ty=='constexpr' and record['constants'].get(f'({i},)')!=value):
      raise ValueError('scheduler scalar or specialization mismatch: '+name)
  grid,block=geometry(launch)
  grouped=kind in INLINED_PTX
  if grid!=tuple(evidence['grid']) or grid[2]!=1 or (not grouped and grid[1]!=1) or block!=(32*record['num_warps'],1,1):
    raise ValueError('scheduler launch geometry mismatch')
  n=grid[0]-(kind=='position preparation')
  if n<=0:raise ValueError('scheduler launch has no requests')
  def vector(name,size,exact=False):
    shape=args[name]['shape']
    if len(shape)!=1 or shape[0]<size or (exact and shape[0]!=size):raise ValueError('scheduler buffer dimensions mismatch: '+name)
  def stride(pointer,scalar):
    if len(args[pointer]['shape'])!=2 or args[pointer]['stride'][0]!=args[scalar]:raise ValueError('scheduler row stride mismatch: '+pointer)
  if not grouped and kind!='staged buffer writes':vector('idx_mapping_ptr',n,True);vector('query_start_loc_ptr',n+1,True)
  if kind=='position preparation':
    maximum=args['max_num_reqs']
    if maximum<n or args['BLOCK_SIZE']<=0:raise ValueError('invalid position preparation specialization')
    vector('pos_ptr',1);vector('seq_lens_ptr',maximum,True);vector('num_computed_tokens_ptr',n)
    # Four int32 reads and one int32 write per real request, plus padding
    # writes. The runtime query-length loop's int64 stores are excluded here.
    minimum=20*n+4*(maximum-n)
  elif kind=='sampled token preparation':
    if args['BLOCK_SIZE']<=0 or args['NUM_NEW_SAMPLED_TOKENS']<0:raise ValueError('invalid sampled token specialization')
    vector('seq_lens_ptr',n,True);vector('prefill_len_ptr',n);vector('cu_num_logits_ptr',n+1,True)
    vector('input_ids_ptr',1);vector('logits_indices_ptr',1);stride('draft_tokens_ptr','draft_tokens_stride')
    # Six unconditional int32 reads. Masked logits-index writes and branch-
    # dependent sampled/draft token operations are excluded from logical bytes.
    minimum=24*n
  elif kind=='request state update':
    for name in ('num_sampled_ptr','num_rejected_ptr'):vector(name,n,True)
    for name in ('num_computed_tokens_ptr','total_len_ptr'):vector(name,n)
    for pointer,scalar in (('sampled_tokens_ptr','sampled_tokens_stride'),('all_token_ids_ptr','all_token_ids_stride'),('output_bin_counts_ptr','output_bin_counts_stride')):stride(pointer,scalar)
    # A negative request index returns immediately. Only that first int32
    # read is guaranteed without reading runtime GPU control values.
    minimum=4*n
  elif kind=='block table gathering':
    groups,batches,_=grid
    if not 0<args['num_reqs']<=batches or args['BLOCK_SIZE']<=0:raise ValueError('invalid block-table request count or tile')
    vector('batch_idx_to_req_idx',args['num_reqs'])
    for name in ('src_block_table_ptrs','dst_block_table_ptrs','block_table_strides'):vector(name,groups,True)
    stride('num_blocks_ptr','num_blocks_stride')
    if args['num_blocks_ptr']['shape'][0]!=groups:raise ValueError('block-table group count mismatch')
    # Each CTA reads the int64 stride and uint64 destination pointer before
    # either the padding path or the runtime block-count copy loop.
    minimum=16*groups*batches
  elif kind=='slot mapping':
    groups,batches,_=grid;requests=batches-1
    if requests<=0 or args['max_num_tokens']<=0 or args['CP_SIZE']!=1 or args['cp_rank']!=0:raise ValueError('unsupported slot-mapping dimensions')
    vector('idx_mapping',requests,True);vector('query_start_loc',requests+1,True);vector('pos',1)
    for name in ('block_table_ptrs','block_table_strides','block_sizes','kernel_block_sizes','slot_mapping_enabled','dcp_sharded'):vector(name,groups,True)
    stride('slot_mappings_ptr','slot_mappings_stride')
    if args['slot_mappings_ptr']['shape']!=[groups,args['max_num_tokens']]:raise ValueError('slot-mapping output dimensions mismatch')
    # Guaranteed int32 query-bound reads: two per real request and one for
    # the padding CTA. Other operands and masked loop stores are excluded.
    minimum=4*groups*(2*requests+1)
  elif kind=='prefill token preparation':
    if not 0<args['num_lookahead']<=args['LOOKAHEAD_BLOCK'] or args['BLOCK_SIZE']<=0:raise ValueError('invalid prefill lookahead specialization')
    for name in ('prefill_lens_ptr','num_computed_tokens_ptr'):vector(name,n)
    vector('input_ids_ptr',1);stride('all_token_ids_ptr','all_token_ids_stride');stride('next_prefill_tokens_ptr','next_prefill_tokens_stride')
    if args['next_prefill_tokens_ptr']['shape'][0]!=args['num_lookahead'] or args['next_prefill_tokens_ptr']['shape'][1]<n:raise ValueError('prefill lookahead buffer mismatch')
    # Three unconditional int32 reads precede the no-prefill early return.
    minimum=12*n
  else:
    for name in ('write_indices_ptr','write_starts_ptr','write_cu_lens_ptr'):vector(name,n,True)
    vector('write_contents_ptr',1)
    output=args['output_ptr']['shape']
    if args['BLOCK_SIZE']<=0 or args['output_stride']<=0 or len(output)>2:raise ValueError('invalid staged write layout')
    if args['output_stride']!=(args['output_ptr']['stride'][0] if len(output)==2 else 1):raise ValueError('staged write row stride mismatch')
    # Row, start and cumulative end for every write, and cumulative start
    # for all but the first. Runtime content lengths are not inferred.
    minimum=12*n+4*(n-1)
  quant=args['output_ptr']['dtype'].removeprefix('torch.') if kind=='staged buffer writes' else 'integer'
  return {'name':kind+(' '+quant if kind=='staged buffer writes' else ''),'what':'auxiliary','path':'memory_only','quant':quant,
    'flops':0.,'bytes':float(minimum),'requires_memory_evidence':True,
    'source_module':'Triton binary '+record['binary_sha256'],
    'basis':'reviewed program without floating-point arithmetic, loaded binary and checked runtime ABI; logical operand bytes are a source-proved minimum',
    'limitations':['Memory-only lower bound; integer arithmetic, comparisons and control flow are excluded. Logical bytes exclude conditional and loop-dependent operands; matched counters supply measured memory-tier traffic. No floating-point throughput or MFU is claimed.']}
