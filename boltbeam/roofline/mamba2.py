"""Mamba2 work from rectangular state shapes, not the Gated DeltaNet rule.

Counts are useful arithmetic / compulsory tensor traffic lower bounds, like the
attention model. They exclude padding, cache rereads and transcendental latency;
those are not measured DRAM traffic or executed-instruction counts. SSD is split
by its actual algorithm stages rather than assigning all work to a blended peak.

Sources: vLLM mamba_mixer2.py and ops/ssd_{bmm,chunk_state,chunk_scan,state_passing}.py;
Tri Dao, State Space Duality, Part III (2024), four-stage SSD decomposition.
"""
from __future__ import annotations
import math

DTYPE_BYTES = {'float32':4, 'bfloat16':2, 'float16':2}


def shape(profile: dict) -> dict:
  r = (profile.get('metadata') or {}).get('recurrent') or {}
  required = ('layer_count','head_count','inner_size','state_size','group_count','conv_kernel')
  if r.get('kind') != 'mamba2' or any(type(r.get(k)) is not int or r[k] <= 0 for k in required):
    raise ValueError('Mamba2 requires explicit positive state/head/group dimensions')
  if r['inner_size'] % r['head_count'] or r['head_count'] % r['group_count']:
    raise ValueError('Mamba2 head/group dimensions do not divide')
  return {**r, 'head_dim':r['inner_size']//r['head_count'],
          'channels':r['inner_size']+2*r['group_count']*r['state_size']}


def state_bytes(profile: dict, evidence: dict) -> float:
  """One layer's resolved engine state, checked against the checkpoint config.

Evidence is a loaded MambaMixer2's get_state_shape/get_state_dtype; TP and
speculative expansions are refused until their partition semantics are modeled.
"""
  r = shape(profile)
  shapes, dtypes = evidence.get('state_shapes', []), evidence.get('state_dtypes', [])
  expected = (r['head_count'],r['head_dim'],r['state_size'])
  conv = (r['channels'], r['conv_kernel']-1)
  if len(shapes) != 2 or tuple(shapes[1]) != expected or tuple(shapes[0]) not in (conv,conv[::-1]):
    raise ValueError('engine Mamba2 state shapes disagree with model or use unsupported sharding/speculation')
  if len(dtypes) != 2 or any(d not in DTYPE_BYTES for d in dtypes):
    raise ValueError('engine Mamba2 state dtypes are absent or unsupported')
  return float(sum(math.prod(s)*DTYPE_BYTES[d] for s,d in zip(shapes,dtypes)))


def decode(profile: dict, evidence: dict, activation_dtype: str) -> dict:
  rows = decode_stages(profile, evidence, activation_dtype)
  return {'flops':sum(r['flops'] for r in rows), 'bytes':sum(r['bytes'] for r in rows),
          'state_bytes':state_bytes(profile,evidence), 'path':'fp32_fma',
          'layers':shape(profile)['layer_count'], 'requires_memory_evidence':True,
          'basis':'sum of per-layer convolution and rectangular state-update work'}


def decode_stages(profile: dict, evidence: dict, activation_dtype: str) -> list[dict]:
  """One token in one layer, split at the actual convolution/update boundary.

  Constants use loaded tensor formats. The adapter supplies A in the format
  consumed by the kernel (vLLM explicitly converts it to FP32). Transcendentals
  and address/index work are excluded from useful FLOPs, as in the SSD model.
  Compulsory bytes are not DRAM bytes: cache traffic requires counter evidence.
  """
  r = shape(profile)
  state_bytes(profile, evidence)
  a = DTYPE_BYTES[activation_dtype]
  h,p,n,g = r['head_count'],r['head_dim'],r['state_size'],r['group_count']
  constants = evidence.get('recurrent_parameters') or {}
  expected = {'A':[h], 'D':[h], 'dt_bias':[h],
              'conv_weights':[r['channels'],r['conv_kernel']]}
  def parameter_bytes(name, dims):
    value = constants.get(name)
    if not value or value.get('shape') != dims or value.get('dtype') not in DTYPE_BYTES:
      raise ValueError(f'Mamba2 {name} requires resolved parameter shape and dtype')
    return math.prod(dims)*DTYPE_BYTES[value['dtype']]
  sizes = {name:parameter_bytes(name,dims) for name,dims in expected.items()}
  if 'conv_bias' not in constants:
    raise ValueError('Mamba2 convolution bias presence is unresolved')
  bias = parameter_bytes('conv_bias',[r['channels']]) if constants['conv_bias'] is not None else 0
  conv_state, scan_state = [math.prod(s)*DTYPE_BYTES[d] for s,d in
                           zip(evidence['state_shapes'],evidence['state_dtypes'])]
  def row(stage, flops, traffic):
    return {'stage':stage, 'flops':float(flops), 'bytes':float(traffic),
            'path':'fp32_fma', 'layers':r['layer_count'], 'requires_memory_evidence':True,
            'basis':'per layer, one token; useful arithmetic / compulsory tensor bytes'}
  return [
    row('convolution', 2*r['conv_kernel']*r['channels'],
        2*conv_state+2*a*r['channels']+sizes['conv_weights']+bias),
    # dt+bias, A*dt; dB=B*dt; state*dA+dB*x; sum(state*C)+D*x.
    row('state_update', 2*h+h*n+5*h*p*n+2*h*p,
        2*scan_state+a*(2*h*p+2*g*n+h)+sizes['A']+sizes['D']+sizes['dt_bias']),
  ]


def ssd(profile: dict, length: int, engine_chunks: list[int], ssd_chunk: int,
        evidence: dict, activation_dtype: str) -> list[dict]:
  r=shape(profile); state_bytes(profile,evidence)
  if length <= 0 or sum(engine_chunks)!=length or min(engine_chunks)<=0 or ssd_chunk<=0:
    raise ValueError('SSD chunks must cover the prompt exactly')
  a=DTYPE_BYTES[activation_dtype]; s=DTYPE_BYTES[evidence['state_dtypes'][1]]
  h,p,n,g=r['head_count'],r['head_dim'],r['state_size'],r['group_count']
  widths=[min(ssd_chunk,w-i) for w in engine_chunks for i in range(0,w,ssd_chunk)]
  c=len(widths); pairs=sum(w*(w+1)//2 for w in widths)
  # vLLM calls _bmm_chunk_fwd with causal=False: it materializes full CB
  # blocks even though chunk_scan masks their upper triangles.
  squares=sum(w*w for w in widths)
  state_elements=h*p*n
  def row(stage, flops, traffic, path):
    return {'stage':stage,'flops':float(flops),'bytes':float(traffic),'path':path,
            'layers':r['layer_count'],'requires_memory_evidence':True,
            'basis':'per layer; useful arithmetic / compulsory bytes; not padded instruction count'}
  return [
    row('convolution',2*r['conv_kernel']*r['channels']*length,
        a*(2*r['channels']*length+r['channels']*r['conv_kernel'])+
        2*len(engine_chunks)*math.prod(evidence['state_shapes'][0])*DTYPE_BYTES[evidence['state_dtypes'][0]],'fp32_fma'),
    row('cumsum',3*length*h, a*length*h+8*length*h+8*h,'fp32_fma'),
    row('chunk_state',2*length*state_elements,
        a*length*(h*p+g*n)+8*length*h+4*c*state_elements,activation_dtype_to_path(activation_dtype)),
    row('state_passing',2*c*state_elements,
        4*c*state_elements+s*(c+len(engine_chunks)-1)*state_elements+4*c*h,'fp32_fma'),
    row('bmm',2*g*n*squares,2*a*length*g*n+4*g*squares,activation_dtype_to_path(activation_dtype)),
    row('chunk_scan',2*length*state_elements+2*h*p*pairs,
        a*length*(g*n+2*h*p)+4*g*pairs+s*(c-1)*state_elements+8*length*h,activation_dtype_to_path(activation_dtype)),
  ]


def activation_dtype_to_path(dtype):
  return {'bfloat16':'bf16','float16':'fp16'}[dtype]
