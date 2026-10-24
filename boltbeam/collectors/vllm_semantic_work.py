"""Work attached to NVTX-correlated vLLM launches, checked against tensors.

No time-ranking or nearest-shape pairing. A missing/ambiguous module or a changed
fused-kernel contract is an error. MoE weight coverage comes from observed routing;
for one token, distinct top-k selection itself determines the exact footprint.
"""
from __future__ import annotations
import collections
from functools import partial
import json
import math
from pathlib import Path

from boltbeam.collectors.instruction_path import template
from boltbeam.profile.loaders import _quant_config, _safetensors_specs
from boltbeam.profile.roles import role_from_tensor_name, classify_tensor_role
from boltbeam.quantization.quant import load_quant_registry
from boltbeam.kernel_analysis.theoretical_roofline import bits_per_weight
from boltbeam.roofline import mamba2, moe, prefill_work

SSD = {'_causal_conv1d_fwd_kernel':'convolution', '_chunk_cumsum_fwd_kernel':'cumsum',
       '_chunk_state_fwd_kernel':'chunk_state','_state_passing_fwd_kernel':'state_passing',
       '_bmm_chunk_fwd_kernel':'bmm','_chunk_scan_fwd_kernel':'chunk_scan'}
DECODE = {'_causal_conv1d_update_kernel':'convolution',
          '_selective_scan_update_kernel':'state_update'}


def is_projection(key):
  name=key.lower()
  gemm=any(x in name for x in ('gemmuniversal','cutlass::kernel2<','gemvx::kernel'))
  # cuBLAS uses a strided batched dot plus a separate reduction for a router
  # with one input token. The dot owns the model's matrix product; reduction
  # overhead stays separate, like split-K reduction for a GEMM.
  dot='dot_kernel<' in name and 'cublasgemvtensorstridedbatched<' in name
  xmma='_xmma_gemm_' in name and '_execute_kernel' in name and '_cublas' in name
  marlin='marlin::marlin<' in name
  # ROCm: vLLM's skinny GEMMs (wvSplitK, its int4 and FP8 forms), its Triton W4A16 GEMM, hipBLASLt's Tensile (Cijk_)
  rocm=name.startswith(('wvsplitk','void wvsplitk','cijk_','_triton_w4a16'))
  return (gemm or dot or xmma or marlin or rocm) and 'splitkreduce' not in name


def attach(launches: list[dict], model: Path, profile: dict, *, ssd_chunk: int,
           routing: list[dict] | None = None) -> list[dict]:
  return prepare(model,profile)(launches,ssd_chunk=ssd_chunk,routing=routing)


def canonical(name):
  return name[name.index('.layers.'):] if '.layers.' in name else name


def prepare(model: Path, profile: dict, *, auxiliary=False):
  """Snapshot checkpoint metadata once for attribution across captured steps."""
  cfg=json.loads((model/'config.json').read_text())
  specs=_safetensors_specs(model,qcfg=_quant_config(model,cfg))
  registry=load_quant_registry(activations=True)
  tensors={canonical(s.name):s for s in specs}
  projection_roles={r['role'] for r in profile['roles']} - {'ssm_conv','ssm_state'}
  expected_tensors={canonical(t.name) for t in specs if len(t.shape)==2 and
                    role_from_tensor_name(t.name,t.shape) in projection_roles}
  return partial(_attach,cfg=cfg,profile=profile,registry=registry,
                 tensors=tensors,expected_tensors=expected_tensors,auxiliary=auxiliary)


def _attach(launches, *, cfg, profile, registry, tensors, expected_tensors,
            ssd_chunk, routing=None, auxiliary=False, streams=1):
  routes={(r['module'],r['operation_id']):r for r in routing or []}
  if routing is not None and len(routes)!=len(routing):
    raise ValueError('duplicate routing operation evidence')
  covered_tensors=set()
  def weight(s):
    return s.weight_bytes if s.weight_bytes is not None else math.prod(s.shape)*bits_per_weight({'quant':s.quant})/8
  def tensor(module):
    key=canonical(module+'.weight')
    matches=([tensors[key]] if key in tensors else []) if '.layers.' in key else [
      s for name,s in tensors.items() if name.endswith(key)]
    # Multimodal vLLM wraps the text head in language_model, while its
    # checkpoint may store lm_head at the root. Only relax this known wrapper.
    if not matches and key == 'language_model.lm_head.weight':
      matches=[s for name,s in tensors.items() if name == 'lm_head.weight' or name.endswith('.lm_head.weight')]
    if len(matches)!=1: raise ValueError(f'engine module {module} has {len(matches)} checkpoint matches')
    return matches[0]
  groups=collections.defaultdict(list)
  for l in launches:
    if l.get('semantic'):
      groups[tuple(l['semantic_range'])].append(l)
  result=[dict(l) for l in launches]
  by_launch={id(original):copy for original,copy in zip(launches,result)}
  recurrent_modules={group[0]['semantic']['module'] for group in groups.values()
                     if group[0]['semantic']['class']=='MambaMixer2'}
  recurrent=(profile.get('metadata') or {}).get('recurrent') or {}
  if len(recurrent_modules)!=(mamba2.shape(profile)['layer_count'] if recurrent.get('kind')=='mamba2' else 0):
    raise ValueError('captured recurrent modules do not cover the model layers')
  for group in groups.values():
    s=group[0]['semantic'];name=s['module'];cls=s['class']
    if auxiliary:
      from boltbeam.collectors.attention_auxiliary import work
      for launch,derived in work(group):
        by_launch[id(launch)]['semantic_work']={**derived,'what':'auxiliary','path':derived.get('path','fp32_fma'),
          'requires_memory_evidence':True,'source_module':name}
      from boltbeam.collectors.torch_auxiliary import work as torch_work
      for launch,derived in torch_work(group):by_launch[id(launch)]['semantic_work']=derived
      from boltbeam.collectors.cublas_reduction_work import work as reduction_work
      for launch,derived in reduction_work(group):by_launch[id(launch)]['semantic_work']=derived
    if cls=='Attention':
      # The TRT-LLM decode backend's symbol contains neither "attention" nor
      # "flash". Its module identity and exact entry point identify the main
      # attention work; cache writes in the same module remain separate.
      for l in group:
        if template(l['key']).split('(',1)[0] == 'kernel_mha':
          by_launch[id(l)]['semantic_kind']='attention'
    matrix=[l for l in group if is_projection(l['key'])]
    if cls=='MambaMixer2':
      length=s['inputs'][0]['shape'][0]
      decode = any(template(l['key']) in DECODE for l in group)
      contract = DECODE if decode else SSD
      if decode:
        # a decode step holds one token per stream, each with its own state rows (mamba2.decode_stages)
        if length != streams:
          raise ValueError('Mamba2 decode requires one observed token per stream' if streams > 1 else
                           'Mamba2 decode requires one observed token until batched state selection is modeled')
        # mamba_mixer2 explicitly casts A to FP32 before selective_state_update.
        params={**s.get('recurrent_parameters',{})}
        if 'A' in params:params['A']={**params['A'],'dtype':'float32'}
        work=mamba2.decode_stages(profile,{**s,'recurrent_parameters':params},s['inputs'][0]['dtype'],streams)
      else:
        work=mamba2.ssd(profile,length,[length],ssd_chunk,s,s['inputs'][0]['dtype'])
      stages={x['stage']:x for x in work}
      seen=collections.Counter()
      for l in group:
        stage=contract.get(template(l['key']))
        if stage:
          seen[stage]+=1
          by_launch[id(l)]['semantic_work']={'name':'mamba2 '+stage, 'what':'recurrent', **stages[stage],
            'source_module':name, 'quant':'BF16' if s['inputs'][0]['dtype']=='bfloat16' else 'F16'}
      if set(seen)!=set(stages) or any(v!=1 for v in seen.values()):
        raise ValueError(f'{name}: recurrent stage launches do not match the engine contract {list(stages)}: {seen}')
    if not matrix:
      continue
    if cls=='MoERunner':
      if len(matrix)!=2 or not all('GroupProblemShape' in l['key'] for l in matrix):
        raise ValueError(f'{name}: expected two grouped expert GEMMs from vLLM MoERunner')
      topk=cfg['num_experts_per_tok'];length=s['inputs'][0]['shape'][0]
      selected=topk
      if routing is not None:
        route=routes.get((name,s.get('operation_id')))
        if route is None:
          raise ValueError(f'{name}: expert routing does not match the dispatched operation')
        selected=moe.selected_experts(route,tokens=length,top_k=topk,experts=cfg['n_routed_experts'])
      exact = routing is not None or length == 1
      for l,projection in zip(matrix,('up_proj','down_proj')):
        ts=[tensor(name+f'.{i}.{projection}') for i in range(cfg['n_routed_experts'])]
        if len({(t.shape,t.quant,weight(t)) for t in ts})!=1:
          raise ValueError('heterogeneous experts need actual routing evidence')
        t=ts[0];n,k=t.shape
        covered_tensors.update(canonical(t.name) for t in ts)
        q=t.quant;acts={'in':q,'out':'BF16'}
        w=prefill_work.gemm(length*topk,n,k,weight(t)*selected,acts)
        by_launch[id(l)]['semantic_work']={'name':role_from_tensor_name(t.name,t.shape)+' '+q,
          'what':'role','quant':q,**w,'source_module':name,
          'basis':('grouped MoE: exact token products and distinct selected expert weights from observed routing'
                   if routing is not None else 'grouped MoE: one token selects exactly top-k distinct experts'
                   if length == 1 else 'grouped MoE: exact top-k token products; minimum top-k distinct weights, not measured routing traffic'),
          'limitations':([] if exact else ['Expert weight bytes are a theoretical minimum over top-k distinct experts; actual routing coverage was not measured.']),
          'routing_basis':('observed' if routing is not None else 'single_token_topk' if length == 1 else 'minimum'),
          'selected_experts':selected,'routing_observed':routing is not None,
          'routing_operation_id':s.get('operation_id'),'role':role_from_tensor_name(t.name,t.shape),
          'activation_formats':acts,
          'top_k':topk,'tokens':length,'shape':[n,k]}
      continue
    if len(matrix)!=1:
      raise ValueError(f'{name}: expected one matrix launch, found {len(matrix)}')
    parts=[tensor(part) for part in s.get('checkpoint_modules',['lm_head' if cls=='LogitsProcessor' else name])]
    covered_tensors.update(canonical(t.name) for t in parts)
    n=sum(t.shape[0] for t in parts);k=parts[0].shape[1]
    if any(len(t.shape)!=2 or t.shape[1]!=k for t in parts) or len({t.quant for t in parts})!=1:
      raise ValueError(f'{name}: incompatible packed matrix tensors')
    runtime_shape=(s.get('output_size_per_partition'),s.get('input_size_per_partition'))
    if runtime_shape==(None,None):runtime_shape=tuple(s.get('weight_shape', []))
    if cls!='LogitsProcessor' and (n,k)!=runtime_shape:
      raise ValueError(f'{name}: runtime logical shape disagrees with checkpoint')
    inputs=s['inputs'];length=math.prod(inputs[0]['shape'][:-1])
    q=parts[0].quant
    runtime_quant={'float32':'F32','bfloat16':'BF16','float16':'F16','float8_e4m3fn':'FP8_E4M3'}.get(s.get('weight_dtype'))
    if runtime_quant and runtime_quant!=q:
      # GateLinear can promote BF16 checkpoint router weights to FP32 in memory.
      q=runtime_quant
    roles=sorted({role_from_tensor_name(t.name,t.shape) for t in parts})
    wb=n*k*registry[q].bytes_per_elem if runtime_quant and q!=parts[0].quant else sum(weight(t) for t in parts)
    w=prefill_work.gemm(length,n,k,wb,{'in':q if q in ('NVFP4_G16','FP8_E4M3','F32') else 'BF16','out':'F32' if q=='F32' else 'BF16'})
    by_launch[id(matrix[0])]['semantic_work']={'name':' + '.join(roles)+' '+q,'what':'role','quant':q,
      **w,'source_module':name,'shape':[n,k],'tokens':length,'basis':'NVTX module and runtime dimensions checked against checkpoint tensors'}
    by_launch[id(matrix[0])]['semantic_work']['role_classes'] = sorted({classify_tensor_role(t.name,t.shape).value for t in parts})
  missing=[l['key'] for l in result if is_projection(l['key']) and not l.get('semantic_work')]
  if missing:raise ValueError('matrix launches lack semantic work: '+str(missing[:3]))
  if expected_tensors - covered_tensors:
    raise ValueError('model projection tensors lack captured operations: '+str(sorted(expected_tensors-covered_tensors)[:3]))
  return result
