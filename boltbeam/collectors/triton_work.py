"""Reviewed scalar work contracts, admitted by exact loaded-code provenance.

The AST fingerprints identify complete generated programs, not kernel names.
Changed programs stay unsupported. RMS arithmetic counts the algorithm's minimum
adds/multiplies (including reduction), excluding rsqrt and address operations.
Operand bytes never establish cache residency: a limit still requires counters.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from boltbeam.collectors.attention_auxiliary import geometry
from boltbeam.collectors.instruction_path import template
from boltbeam.collectors import triton_scheduler_work as scheduler

# (kind, residual input count, output count, weight pointer). Width is read from the admitted
# program's r0_numel assignment and checked against every runtime operand.
CONTRACTS = {
  '2722797c462c860fcf04bfb88a6c2f911a1adb179cacefcf01d1f500e3e9541c':('residual RMSNorm',2,1,'in_ptr2'),
  'fd8e1f32170d9b69cc6967183f8374e7218ca96fcad276615bb64e1470693b98':('residual RMSNorm',3,2,'in_ptr3'),
  'd6354694caf60db05b4153b94aac4b20988045b429671cbfb260d381bb5646d7':('residual RMSNorm',3,1,'in_ptr2'),
  '2ef2c9a3dc66a86cce131ff6b931e4ae4ca3f9eb3a1045a6d42e3a2d0dce695b':('embedding RMSNorm',1,2,'in_ptr2'),
}
FUSED = {
  'c4bddb56e0f785c1d8fb003cf769d4f67e54d53a062402e56c43d91653815150':'SiLU gate',
  'a05324b81d75566065841db53f1b5759978182aea9d628b8e251b65b72e5c20e':'Q/K RMSNorm and rotary',
}
COUNTS = '615d4d8386724aec9555c7ffcb407aead17b6e67df03321e6a55bac19de36000'
COUNT_ARGS = ['num_sampled_ptr','num_rejected_ptr','seq_lens_ptr','cu_num_logits_ptr','idx_mapping_ptr','prefill_len_ptr']


def sampling_counts(record,args,launch,evidence):
  grid,block=geometry(launch);n=grid[0]
  if grid!=(n,1,1) or grid!=tuple(evidence['grid']) or block!=(32*record['num_warps'],1,1):
    raise ValueError('sampling-count launch geometry mismatch')
  if list(record['signature'])!=COUNT_ARGS or set(args)!=set(COUNT_ARGS):raise ValueError('sampling-count ABI mismatch')
  for name in COUNT_ARGS:
    value=args[name];shape=value.get('shape') if isinstance(value,dict) else None
    expected=n+1 if name=='cu_num_logits_ptr' else n
    if (not shape or len(shape)!=1 or type(shape[0]) is not int or shape[0]<expected or
        (name!='prefill_len_ptr' and shape[0]!=expected) or value.get('stride')!=[1] or
        value.get('dtype')!='torch.int32' or record['signature'][name]!='*i32'):
      raise ValueError('sampling-count operand mismatch: '+name)
  # Six scalar int32 loads and two scalar int32 stores per request, on every
  # branch. Conditional selection changes values, not the accessed sizes.
  return {'name':'sampling count update','what':'auxiliary','path':'memory_only','quant':'I32',
    'flops':0.,'bytes':32.*n,'requires_memory_evidence':True,
    'source_module':'Triton binary '+record['binary_sha256'],
    'basis':'reviewed integer-only program, loaded binary identity and observed request buffers',
    'limitations':['Memory-only lower bound; integer arithmetic, comparisons and control flow are excluded. No floating-point throughput or MFU is claimed.']}


def fused(record,args,launch,evidence):
  """Constants below are address formulas in the admitted complete programs."""
  kind=FUSED[record['contract']];tile=record['constants'].get('XBLOCK')
  if type(tile) is not int or tile<=0:raise ValueError('invalid fused launch tile')
  if kind=='SiLU gate':
    n=args.get('xnumel')
    if type(n) is not int or n<=0 or n%12288 or n%tile:raise ValueError('unmasked SiLU launch dimensions mismatch')
    rows=n//12288;blocks=n//tile
    shapes={'in_ptr0':[rows,24576],'out_ptr0':[rows,12288]};scalars={'xnumel'}
    flops=2*n;byte_count=6*n;excluded='exp, division, negation, casts and address arithmetic'
  else:
    q=args.get('xnumel_0');k=args.get('xnumel_1')
    if type(q) is not int or q<=0 or q%32 or type(k) is not int or k!=q//4:
      raise ValueError('Q/K head counts disagree with executed program')
    rows=q//32;blocks=math.ceil(q/tile)+math.ceil(k/tile)
    shapes={'in_ptr0':[rows,6144],'in_ptr1':[128],'in_ptr2':[rows],'in_ptr3':[40960,128],
            'in_ptr4':[128],'out_ptr1':[rows,32,128],'out_ptr3':[rows,8,128]}
    scalars={'xnumel_0','xnumel_1'}
    # Per head: RMS minimum 4D+1, plus two multiplies and an add per rotary output.
    flops=(q+k)*(7*128+1)
    # Logical operands: Q/K input and output, two weight vectors, one rotary row
    # and int64 position per token. V is present in the packed tensor but not read.
    byte_count=rows*(4*40*128+2*128+8)+2*2*128
    excluded='rsqrt, redundant normalization, casts and address arithmetic'
  if set(args)!=set(shapes)|scalars:raise ValueError('fused argument signature mismatch')
  for name,shape in shapes.items():
    dtype,ty=('torch.int64','*i64') if kind!='SiLU gate' and name=='in_ptr2' else ('torch.bfloat16','*bf16')
    value=args[name]
    if (not isinstance(value,dict) or value.get('shape')!=shape or any(type(n) is not int for n in value['shape']) or value.get('dtype')!=dtype or
        value.get('stride')!=[math.prod(shape[i+1:]) for i in range(len(shape))] or record['signature'].get(name)!=ty):
      raise ValueError('fused operand shape, stride or dtype differs from executed program: '+name)
  grid,block=geometry(launch)
  if grid!=(blocks,1,1) or grid!=tuple(evidence['grid']) or block!=(32*record['num_warps'],1,1):
    raise ValueError('fused launch geometry disagrees with executed program')
  return result(record,kind,flops,byte_count,excluded)


def result(record,kind,flops,byte_count,excluded):
  return {'name':kind,'what':'auxiliary','path':'fp32_fma','quant':'BF16',
    'flops':float(flops),'bytes':float(byte_count),'requires_memory_evidence':True,
    'source_module':'Triton binary '+record['binary_sha256'],
    'basis':'reviewed generated program AST, loaded binary digest and observed runtime operand dimensions',
    'limitations':[f'Scalar arithmetic is a lower bound; {excluded} are excluded. Operand bytes do not establish memory-tier traffic.']}


def program(source,name):
  matches=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
  if len(matches)!=1:raise ValueError('Triton source has no unique entry point')
  fn=matches[0];fn.name='kernel';fn.decorator_list=[]
  digest=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
  width=next((n.value.value for n in fn.body if isinstance(n,ast.Assign) and len(n.targets)==1
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='r0_numel' and isinstance(n.value,ast.Constant)),None)
  return digest,width


class Contracts:
  def __init__(self,root):self.root=Path(root);self.records={};self.used=set()

  def sources(self):
    return [{k:self.records[key][k] for k in ('cache_key','binary_sha256','artifacts','contract')}
            for key in sorted(self.used)]

  def record(self,key):
    if not key or any(c not in '0123456789abcdef' for c in key):raise ValueError('invalid compiled-kernel identity key')
    if key not in self.records:
      folder=self.root/key;record=json.loads((folder/'identity.json').read_text())
      if record['cache_key']!=key:raise ValueError('Triton identity directory disagrees with record')
      for ext,digest in record['artifacts'].items():
        if ext not in ('cubin','ptx','ttir','ttgir','llir','source','py'):raise ValueError('unknown Triton source artifact')
        if hashlib.sha256((folder/('kernel.'+ext)).read_bytes()).hexdigest()!=digest:
          raise ValueError('Triton source artifact changed since capture')
      if record['artifacts'].get('cubin')!=record['binary_sha256']:raise ValueError('Triton binary digest disagrees with artifact')
      record['contract'],record['width']=program((folder/'kernel.py').read_text(),record['name']) if 'py' in record['artifacts'] else (None,None)
      self.records[key]=record
    return self.records[key]

  def work(self,launch):
    evidence=launch.get('triton')
    if not evidence:return None
    record=self.record(evidence['cache_key'])
    if record['binary_sha256']!=evidence.get('binary_sha256'):raise ValueError('launch used a different Triton binary')
    from boltbeam.collectors import triton_sampling_work as sampling
    if sampling.supports(record):
      derived=sampling.work(record,launch,evidence)
      if derived is not None:self.used.add(evidence['cache_key'])
      return derived
    contract=CONTRACTS.get(record['contract'])
    if contract is None and record['contract'] not in FUSED and record['contract']!=COUNTS and record['contract'] not in scheduler.PROGRAMS:return None
    if template(launch['key'])!=record['name']:raise ValueError('Triton launch symbol disagrees with loaded entry point')
    defaults=list(scheduler.signature(record['contract'])) if record['contract'] in scheduler.PROGRAMS else COUNT_ARGS if record['contract']==COUNTS else []
    names=evidence.get('argument_names',defaults);values=evidence['args']
    if len(names)!=len(values) or len(set(names))!=len(names):raise ValueError('Triton launch lacks unambiguous runtime argument names')
    args=dict(zip(names,values))
    if record['contract'] in scheduler.PROGRAMS:
      derived=scheduler.work(record,args,launch,evidence)
      if derived is not None:self.used.add(evidence['cache_key'])
      return derived
    if record['contract']==COUNTS:
      derived=sampling_counts(record,args,launch,evidence);self.used.add(evidence['cache_key']);return derived
    if record['contract'] in FUSED:
      derived=fused(record,args,launch,evidence);self.used.add(evidence['cache_key']);return derived
    width=record['width'];rows=args.get('xnumel')
    if type(rows) is not int or rows<=0 or type(width) is not int or width<=0 or args.get('r0_numel')!=width:
      raise ValueError('RMSNorm dimensions disagree with executed program')
    grid,block=geometry(launch);constants=record['constants']
    if type(constants.get('XBLOCK')) is not int or constants['XBLOCK']<=0:raise ValueError('invalid RMSNorm launch tile')
    if (grid!=tuple(evidence['grid']) or grid!=(math.ceil(rows/constants['XBLOCK']),1,1)
        or block!=(32*record['num_warps'],1,1)):
      raise ValueError('RMSNorm geometry disagrees with executed program')
    kind,inputs,outputs,weight=contract
    pointers=[(name,ty) for name,ty in record['signature'].items() if ty.startswith('*')]
    if set(args)!={name for name,_ in pointers}|{'xnumel','r0_numel'}:raise ValueError('RMSNorm argument signature mismatch')
    for name,ty in pointers:
      value=args.get(name)
      if not isinstance(value,dict) or not value.get('shape'):raise ValueError('RMSNorm operand shape was not observed')
      shape=value['shape'];dtype=value['dtype']
      if any(type(n) is not int or n<=0 for n in shape):raise ValueError('invalid RMSNorm operand dimensions')
      if value.get('stride')!=[math.prod(shape[i+1:]) for i in range(len(shape))]:
        raise ValueError('RMSNorm operand is not contiguous as required by executed program')
      if ty=='*i32':
        if dtype!='torch.int32' or math.prod(shape)!=rows:raise ValueError('embedding index shape mismatch')
      elif ty!='*bf16' or dtype!='torch.bfloat16' or shape[-1]!=width:
        raise ValueError('RMSNorm operand width or dtype mismatch')
      elif name==weight and shape!=[width]:raise ValueError('RMSNorm weight shape mismatch')
      elif kind=='embedding RMSNorm' and name=='in_ptr1':
        if shape!=[151936,width]:raise ValueError('embedding table differs from executed program')
      elif name!=weight and shape!=[rows,width]:
        raise ValueError('RMSNorm operand token count mismatch')
    self.used.add(evidence['cache_key'])
    return result(record,kind,rows*((inputs+3)*width+1),
      rows*width*2*(inputs+1+outputs)+(4*rows if kind=='embedding RMSNorm' else 0),'rsqrt, casts and address arithmetic')
