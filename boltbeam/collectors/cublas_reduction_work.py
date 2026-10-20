"""Reviewed BF16 workspace reduction for two exact loaded CUTLASS producers.

For a nonbatched aten.mm.out, producer grid Z partitions K. Its XY grid covers
the output tiles once per partition. These binaries store BF16 partials; the
reducer loads U16 values, widens them, and sums with FP32 FADD before its BF16
store. Count only S-1 adds per output; scaling, casts and indexing are excluded.
Neither an entry name nor an adjacent launch alone admits this contract.
"""
import math
from boltbeam.collectors.attention_auxiliary import geometry
from boltbeam.collectors.instruction_path import template

PRODUCERS = {
  'cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>':
    ('5ecaec2f7a0c934d5cf0316da6f8e3def48bdf2a071556c783f54c5b361695e2',128,64,64,128),
  'cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>':
    ('00457ad5c576d4006a46abde5976760118881522539d5d4a636739532ef08f87',16,16,128,32),
}
REDUCER = 'cublasLt::splitKreduce_kernel<32,16,int,__nv_bfloat16,__nv_bfloat16,float,__nv_bfloat16,0,__nv_bfloat16,__nv_bfloat16,__nv_bfloat16,1,0,0,0>'
REDUCER_BINARY = 'bd88210a10fe809f4f666195b073b9546aee125138c95418bc9882ba43a675f4'


def work(group):
  if len(group)!=2:return
  producer,reducer=group
  contract=PRODUCERS.get(template(producer['key']))
  if contract is None or template(reducer['key'])!=REDUCER:return
  digest,tile_n,tile_m,tile_k,threads=contract
  if producer.get('native_binary_sha256')!=digest or reducer.get('native_binary_sha256')!=REDUCER_BINARY:return
  s=producer.get('semantic') or {}
  if s!=reducer.get('semantic') or s.get('operator')!='aten.mm.out':return
  inputs=s.get('inputs',[])
  if len(inputs)!=2 or any(x.get('dtype')!='bfloat16' or len(x.get('shape',[]))!=2 for x in inputs):return
  (m,k),(out_m,n)=(x['shape'] for x in inputs)
  if any(type(x) is not int or x<=0 for x in (m,k,out_m,n)) or m!=out_m:
    raise ValueError('split-K reduction operand dimensions mismatch')
  grid,block=geometry(producer);rgrid,rblock=geometry(reducer);splits=grid[2]
  if (splits<2 or block!=(threads,1,1) or grid[0]*grid[1]!=math.ceil(n/tile_n)*math.ceil(m/tile_m)
      or math.ceil(k/(splits*tile_k))*tile_k*(splits-1)>=k
      or rblock!=(32,16,1) or rgrid[0]!=math.ceil(n/32) or rgrid[2]!=1 or rgrid[1]>math.ceil(m/16)):
    raise ValueError('split-K producer/reducer geometry disagrees with operands')
  yield reducer,{'name':'GEMM split-K reduction','what':'auxiliary','path':'fp32_fma','quant':'BF16',
    'flops':float(m*n*(splits-1)),'bytes':float(2*m*n*(splits+1)),'splits':splits,
    'requires_memory_evidence':True,'source_module':s['module'],
    'basis':'exact loaded producer/reducer binaries, one nonbatched matrix operator and observed K-partition grid',
    'limitations':['FP32 sum and BF16 workspace bytes are lower bounds; scaling, conversions and address arithmetic are excluded. Matched DRAM/L2 counters are required.']}
