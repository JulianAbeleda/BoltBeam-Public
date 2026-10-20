"""Match a full eager counter replay to the original paired production replay."""
import json
from itertools import zip_longest
from pathlib import Path
from boltbeam.collectors.cuda_counter_stream import Stream,bind_launches
from boltbeam.collectors.cuda_memory import same_capture
from boltbeam.collectors.replay_semantics import SETTINGS,session_identities


def matched_steps(passes):
  """Scheduler IDs may differ; turn position, consumed tokens and settings may not."""
  original=passes['timing']['model'];responses=passes['timing']['responses']
  for name,p in passes.items():
    for key in (*SETTINGS,'model_config_sha256','activation_dtype','kv_cache_dtype','build',
                'compilation_config','logits_processors','triton_autotune_cache','loaded_packages'):
      if key not in p['model'] or p['model'][key]!=original.get(key):raise ValueError(f'counter replay {name}: different {key}')
    if len(p['responses'])!=len(responses):raise ValueError('counter replay has different turn count')
    for a,b in zip(p['responses'],responses,strict=True):
      if a['choices']!=b['choices'] or a['usage']!=b['usage']:raise ValueError('counter replay output/cache differs')
  for name in ('label','counter'):
    if passes[name]['model'].get('runtime_graph_mode')!='NONE':raise ValueError('counter binding requires eager labeling')
  sessions={name:session_identities(p['model'],p['records'],p['responses']) for name,p in passes.items()}
  out=[]
  for turn,(timed,label,counter) in enumerate(zip(sessions['timing'],sessions['label'],sessions['counter'],strict=True),1):
    if not timed['cached_tokens']==label['cached_tokens']==counter['cached_tokens']:
      raise ValueError('counter replay cached prefix differs')
    for t,l,c in zip_longest(timed['steps'],label['steps'],counter['steps']):
      if t is None or l is None or c is None:raise ValueError('counter replay scheduler coverage differs')
      # This is the explicit step-number mapping, not a rewritten engine identity.
      same_capture({k:v for k,v in l.items() if k!='decode_step'},{k:v for k,v in c.items() if k!='decode_step'})
      if any(t[k]!=l[k] for k in ('context','length','replay')):raise ValueError('original paired scheduler work differs')
      out.append((turn,t,c))
  return out


class ReplayCounters:
  def __init__(self,source,paired, *, passes=None):
    from boltbeam.collectors.forced_replay import evidence
    folder=Path(source['folder']);paired=Path(paired)
    if passes is None:
      responses=json.loads((paired/'responses.json').read_text())
      passes={name:evidence(paired/name,responses[name]) for name in ('timing','label')}
    else:passes=dict(passes)
    counter=json.loads((folder/'responses.json').read_text())
    passes['counter']=evidence(folder/'label',counter['label'])
    self.mapping=matched_steps(passes);self.position=0;self.launches=0
    roots=list(folder.glob('counters-*'))
    if len(roots)!=1:raise ValueError('counter replay needs exactly one worker stream')
    self.stream=Stream(roots[0],[c['decode_step'] for _,_,c in self.mapping],source['collector_sha256'],
                       cuobjdump=source.get('cuobjdump','cuobjdump'))
    self.chunks=iter(self.stream)
  def attach(self,item):
    if self.position>=len(self.mapping):raise ValueError('counter replay has unexpected production step')
    turn,timed,counter=self.mapping[self.position]
    if item['turn']!=turn or item['step']!=timed['decode_step'] or item['identity']!=timed:
      raise ValueError('counter replay production order or identity differs')
    chunk=next(self.chunks)
    if chunk['step']!=counter['decode_step']:raise ValueError('counter replay step mapping differs')
    launches=bind_launches(item['launches'],chunk['rows'])
    self.position+=1;self.launches+=len(launches)
    return launches
  def finish(self):
    if self.position!=len(self.mapping):raise ValueError('counter replay production coverage incomplete')
    if next(self.chunks,None) is not None:raise ValueError('counter replay has unused chunks')
    return {'steps':self.position,'launches':self.launches,'manifest_sha256':self.stream.manifest_sha256,
      'collector_sha256':self.stream.collector_sha256,
      'scope':'equal replay inputs/settings; exact raw entry/geometry and compiled Triton identities; native binary digests observed in the counter pass; raw memory counters remain unqualified'}
