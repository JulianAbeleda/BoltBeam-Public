"""Bind server scheduler work to returned token IDs and resolved engine settings."""
from boltbeam.runtime.vllm_replay_steps import token_hash

SETTINGS = ('seed','max_model_len','max_num_batched_tokens','max_num_seqs','chunked_prefill',
            'prefix_caching','async_scheduling','gpu_memory_utilization','mamba_cache_mode',
            'mamba_ssm_cache_dtype','enforce_eager','cudagraph_mode','triton_cache_dir')


def capture_identity(facts, record, responses):
  """No request IDs or raw tokens escape this identity; changed input/state must not match."""
  from boltbeam.collectors.cuda_memory import IDENTITY_FIELDS
  if record.get('phase')!='execute' or record.get('status')!='submitted' or len(record.get('requests',[]))!=1:
    raise ValueError('replay memory identity requires one completed single-request execution phase')
  for key in SETTINGS:
    if key not in facts or facts[key] is None:
      raise ValueError(f'replay capture has no resolved setting: {key}')
  request=record['requests'][0]
  matches=[r for r in responses if request['request_id']==r['id'] or request['request_id'].startswith(r['id']+'-')]
  if len(matches)!=1 or len(matches[0]['choices'])!=1:
    raise ValueError('scheduler request has no unique response token evidence')
  response=matches[0];choice=response['choices'][0]
  prompt,generated=choice.get('prompt_token_ids'),choice.get('token_ids')
  if not isinstance(prompt,list) or not isinstance(generated,list) or any(type(t) is not int or t<0 for t in prompt+generated):
    raise ValueError('response has no valid prompt and generated token IDs')
  if (len(prompt)!=request['prompt_tokens'] or token_hash(prompt)!=request.get('prompt_sha256') or
      len(prompt)!=response['usage']['prompt_tokens'] or len(generated)!=response['usage']['completion_tokens']):
    raise ValueError('response token evidence differs from scheduler prompt or usage')
  depth,count=request['depth'],request['scheduled_tokens']
  prefill=min(count,max(len(prompt)-depth,0))
  if (depth<0 or count<=0 or depth+count>len(prompt)+len(generated) or
      count!=record['scheduled_tokens'] or request['prefill_tokens']!=prefill or request['decode_tokens']!=count-prefill):
    raise ValueError('scheduler counts do not match the observed token sequence')
  identity={**facts,'vllm':facts['build'],'length':count,'batch':1,'context':depth,'decode_step':record['step']}
  fields=[k for k in IDENTITY_FIELDS if k!='mamba_chunk_size' or k in identity]
  missing=[k for k in fields if identity.get(k) is None]
  if missing:raise ValueError(f'replay memory identity lacks {missing}')
  return {**{k:identity[k] for k in fields},'triton_cache_dir':facts['triton_cache_dir'],
          'context':depth,'batch':1,'decode_step':record['step'],
          'replay':{'settings':{k:facts[k] for k in (*SETTINGS,'logits_processors','triton_autotune_cache') if k in facts},
                    'prefill_tokens':prefill,'decode_tokens':count-prefill,
                    'prompt_sha256':request['prompt_sha256'],
                    'processed_tokens_sha256':token_hash((prompt+generated)[:depth+count])}}


def session_identities(facts, records, responses):
  """Prove complete serial request coverage before summing any step ceilings.

  An independently reported cached prefix may precede the first token. Subsequent depths must
  be contiguous; recomputation and concurrent requests need separate accounting.
  The final emitted token is not fed back into the model.
  """
  ids=[r['id'] for r in responses]
  if not ids or len(set(ids))!=len(ids):raise ValueError('replay session requires unique response identities')
  groups={i:[] for i in ids}
  phases={}
  for record in records:
    if not record.get('scheduled_tokens'):continue
    requests=record.get('requests',[])
    matches=[i for i in ids if any(r['request_id']==i or r['request_id'].startswith(i+'-') for r in requests)]
    if not matches:continue  # warmup requests are outside this supplied session
    if len(matches)!=1 or len(requests)!=1:raise ValueError('replay session requires serial request evidence')
    key=record['step'];phase=record.get('phase')
    if phase not in ('execute','sample') or record.get('status')!='submitted':
      raise ValueError('replay session has an incomplete phase')
    pair=phases.setdefault(key,{})
    if phase in pair:raise ValueError('replay session has a duplicate phase')
    pair[phase]=record
  for step,pair in sorted(phases.items()):
    if set(pair)!={'execute','sample'}:raise ValueError('replay session is missing a phase')
    execute,sample=pair['execute'],pair['sample']
    if any(execute[k]!=sample[k] for k in ('requests','scheduled_tokens')):
      raise ValueError('replay session phases disagree')
    identity=capture_identity(facts,execute,responses)
    request=execute['requests'][0]['request_id']
    match=next(i for i in ids if request==i or request.startswith(i+'-'))
    groups[match].append(identity)
  out=[]
  for response in responses:
    steps=groups[response['id']]
    if not steps:raise ValueError('replay session has an uncaptured response')
    prompt=response['usage']['prompt_tokens'];generated=response['usage']['completion_tokens']
    depth=steps[0]['context']
    if generated<1 or depth>=prompt:raise ValueError('replay session has no captured prompt completion')
    cached=(response['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
    if type(cached) is not int or cached<0 or depth!=cached:
      raise ValueError('replay session initial depth lacks matching cache evidence')
    for identity in steps:
      if identity['context']!=depth:raise ValueError('replay session has missing or repeated token work')
      depth+=identity['length']
    if depth!=prompt+generated-1:raise ValueError('replay session does not cover the complete response')
    out.append({'cached_tokens':cached,'steps':steps})
  return out


class StreamTokens:
  """Retain a streamed chat response's token identity only when explicitly requested."""
  def __init__(self):
    self.request_id=None
    self.prompt=None
    self.generated=[]

  def add(self, chunk):
    request=chunk.get('id')
    if not isinstance(request,str) or not request or (self.request_id is not None and request!=self.request_id):
      raise ValueError('streamed replay changed or omitted its request identity')
    self.request_id=request
    prompt=chunk.get('prompt_token_ids')
    if prompt is not None:
      if not isinstance(prompt,list) or (self.prompt is not None and prompt!=self.prompt):
        raise ValueError('streamed replay changed its prompt token evidence')
      self.prompt=prompt
    choices=chunk.get('choices') or []
    if len(choices)>1:raise ValueError('streamed replay requires one response choice')
    for choice in choices:
      if choice.get('index')!=0:
        raise ValueError('streamed replay requires one response choice')
      ids=choice.get('token_ids')
      if ids is not None:
        if not isinstance(ids,list):raise ValueError('streamed replay has invalid generated token evidence')
        self.generated.extend(ids)

  def finish(self, usage):
    if (self.prompt is None or self.request_id is None or
        any(type(t) is not int or t<0 for t in self.prompt+self.generated) or
        len(self.prompt)!=usage.get('prompt_tokens') or len(self.generated)!=usage.get('completion_tokens')):
      raise ValueError('streamed replay token evidence does not reconcile with usage')
    return {'id':self.request_id,'usage':usage,'choices':[
      {'prompt_token_ids':self.prompt,'token_ids':self.generated}]}
