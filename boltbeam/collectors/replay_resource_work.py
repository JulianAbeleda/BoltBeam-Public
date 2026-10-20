"""Dynamic work totals bound to each observed native launch before grouping.

Primary dense attention core work and auxiliary compiled work are disjoint.
Execution-resource rates and temporal memory demand are admitted separately.
"""
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
from boltbeam.collectors.vllm_semantic_work import prepare
from boltbeam.collectors.triton_work import Contracts
from boltbeam.collectors.prefill import vllm_formats
from boltbeam.collectors.replay_counter_stream import ReplayCounters
from boltbeam.roofline.prefill_work import attention
from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL


class ResourceWork:
 def __init__(self,source,model,folder,passes):
  self.profile=json.loads((Path(source['run'])/'model_profile.json').read_text())
  config=Path(model['model_path'])/'config.json'
  if hashlib.sha256(config.read_bytes()).hexdigest()!=model['model_config_sha256']:
   raise ValueError('resource work checkpoint changed')
  self.attach=prepare(Path(model['model_path']),self.profile,auxiliary=True)
  self.contracts=Contracts(source['triton_evidence'])
  self.counter=ReplayCounters(source['counter_stream'],folder,passes=passes)
  self.formats=vllm_formats({x['quant'] for x in self.profile['roles']},model,require_resolved=True)
  self.layers=self.profile['layer_count'];self.shape=self.profile['metadata']['attention']
  if self.shape['layer_count']!=self.layers:raise ValueError('resource work requires uniform dense attention layers')
  self.modules={f'model.layers.{i}.self_attn.attn' for i in range(self.layers)}
  self.routing={r['step']:r['routing'] for r in map(json.loads,next((folder/'label').glob('*.jsonl')).open()) if r.get('phase')=='execute' and 'routing' in r}
  self.groups={};self.turns=defaultdict(lambda:{'steps':0,'kernels':0,'work_kernels':0,'attention_kernels':0,'kernel_ms':0.})
 def consume(self,item):
  engine=item['identity'];routing=self.routing.get(item.get('label_step',item['step']),[])
  launches=self.attach(self.counter.attach(item),ssd_chunk=engine.get('mamba_chunk_size'),routing=routing)
  for launch in launches:
   if launch.get('triton'):
    work=self.contracts.work(launch)
    if work:
     if launch.get('semantic_work'):raise ValueError('overlapping resource work')
     launch['semantic_work']=work
  selected=[l for l in launches if not l.get('semantic_work')]
  if len(selected)!=self.layers:raise ValueError('unassigned resource launches differ from attention layer count')
  heads=self.shape['head_count'];kv_heads=self.shape['head_count_kv'];dim=self.shape['head_dim']
  expected=[[engine['length'],heads,dim],[engine['length'],kv_heads,dim],[engine['length'],kv_heads,dim],[engine['length'],heads,dim]]
  found=[]
  for launch in selected:
   semantic=launch.get('semantic') or {};witness=semantic.get('inputs',[])[:4];module=semantic.get('module');found.append(module)
   if semantic.get('class')!='Attention' or semantic.get('operator')!='vllm.unified_attention_with_output.default' or semantic.get('num_heads')!=heads:
    raise ValueError('resource attention operator differs')
   if [x.get('shape') for x in witness]!=expected or any(x.get('dtype')!='bfloat16' for x in witness):
    raise ValueError('resource attention operands differ')
   if semantic.get('checkpoint_modules')!=[module]:raise ValueError('ambiguous resource attention module')
  if set(found)!=self.modules or len(set(found))!=self.layers:raise ValueError('resource attention layer coverage differs')
  core=attention(self.profile,engine['length'],[engine['length']],self.formats['kv'],self.formats['attention_activations'],engine['context'])
  if not core or core['layers']!=self.layers:raise ValueError('resource attention core work unavailable')
  for launch in selected:
   launch['semantic_work']={'flops':core['flops']/self.layers,'path':'unresolved_attention_execution',
    'limitations':['causal attention core FLOPs; auxiliary work accounted separately']}
  counts=self.turns[item['turn']];counts['steps']+=1;counts['kernels']+=len(launches);counts['attention_kernels']+=len(selected)
  for launch in launches:
   work=launch.get('semantic_work')
   if work is None:raise ValueError('unassigned resource work')
   if not math.isfinite(work['flops']) or work['flops']<0 or not math.isfinite(launch['wall_us']) or launch['wall_us']<=0:
    raise ValueError('invalid resource work or duration')
   counts['work_kernels']+=1;counts['kernel_ms']+=launch['wall_us']/1000;key=launch['native_binary_sha256'],launch['raw_entry']
   group=self.groups.setdefault(key,{'calls':0,'flops':0.,'kernel_ms':0.,'declared_paths':set(),'limitations':set(),'example_key':launch['key'],'example_step':item['step']})
   group['calls']+=1;group['kernel_ms']+=launch['wall_us']/1000;group['flops']+=work['flops'];group['declared_paths'].add(work.get('path','unresolved_projection_execution'));group['limitations'].update(work.get('limitations',[]))
 def result(self,provenance):
  if not self.turns:raise ValueError('empty resource work')
  totals={k:sum(t[k] for t in self.turns.values()) for k in next(iter(self.turns.values()))}
  if totals['kernels']!=provenance['kernels'] or totals['work_kernels']!=totals['kernels']:
   raise ValueError('resource work does not cover native launches')
  if sum(g['calls'] for g in self.groups.values())!=totals['kernels'] or any(not math.isfinite(g['flops']) for g in self.groups.values()):
   raise ValueError('resource groups do not reconcile')
  for duration in (sum(g['kernel_ms'] for g in self.groups.values()),provenance['kernel_ms']):
   if not math.isclose(duration,totals['kernel_ms'],rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
    raise ValueError('resource durations do not reconcile')
  return {'status':'work_and_identity_passed','native_manifest_sha256':provenance['manifest_sha256'],
   'counter_stream':self.counter.finish(),'totals':totals,'turns':dict(self.turns),
   'groups':[{'binary_sha256':key[0],'entry':key[1],**{k:sorted(v) if isinstance(v,set) else v for k,v in g.items()}} for key,g in self.groups.items()],
   'scope':'each launch bound by native binary/entry/geometry/phase and consumed operand witnesses before grouping; required semantic FLOPs are a lower bound, not all executed machine instructions; attention core excludes noncore operations unless separately derived; execution rates and memory qualification remain separate'}
