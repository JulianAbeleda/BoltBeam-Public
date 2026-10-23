"""Streaming role arithmetic from joined kernels and checkpoint metadata."""
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
from boltbeam.collectors.vllm_semantic_work import prepare
from boltbeam.collectors.prefill import vllm_formats
from boltbeam.collectors.prefill_attribution import attribute_prefill
from boltbeam.workflow.work_evidence import verify_launches
from boltbeam.math.numerics import ACCOUNTING_ATOL,ACCOUNTING_RTOL

class Roles:
 def __init__(self,source,model,folder, *, timing_folder=None,passes=None):
  if timing_folder is not None and source.get('auxiliary_memory'):
   raise ValueError('native activity replay does not support legacy auxiliary-memory step mapping')
  self.run=Path(source['run']);self.model=model
  self.routing={r['step']:r['routing'] for r in map(json.loads,next((folder/'label').glob('*.jsonl')).open()) if r.get('phase')=='execute' and 'routing' in r}
  self.profile=json.loads((self.run/'model_profile.json').read_text())
  self.calibration=json.loads((self.run/'decode_capture.json').read_text())['contexts'][str(source['context'])]
  self.paths=json.loads(Path(source['instruction_paths']).read_text())
  config=Path(model['model_path'])/'config.json'
  if hashlib.sha256(config.read_bytes()).hexdigest()!=model['model_config_sha256']:raise ValueError('checkpoint config changed since capture')
  loaded={str(Path(f).resolve()) for f in json.loads(next((timing_folder or folder/'timing').glob('*.libraries.json')).read_text())}
  for record in self.paths['libraries']:
   file=Path(record['library']).resolve()
   if file.is_dir():continue
   with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   if str(file) not in loaded or digest!=record['sha256']:raise ValueError('instruction library differs from loaded replay binary: '+file.name)
  from boltbeam.workflow.work_evidence import verify
  verify(self.run,self.calibration,require_limits=True)
  self.attach=prepare(Path(model['model_path']),self.profile)
  self.auxiliary=prepare(Path(model['model_path']),self.profile,auxiliary=True) if source.get('auxiliary_memory') or source.get('counter_stream') else None
  from boltbeam.collectors.replay_counter_stream import ReplayCounters
  self.counter_stream=ReplayCounters(source['counter_stream'],folder,passes=passes) if source.get('counter_stream') else None
  from boltbeam.collectors.triton_work import Contracts
  self.triton=Contracts(source['triton_evidence']) if source.get('triton_evidence') else None
  self.memory={};self.memory_sources=[];self.used_memory=set()
  for evidence in source.get('auxiliary_memory',[]):
   from boltbeam.collectors.cuda_memory import read_raw
   from boltbeam.collectors.replay_semantics import capture_identity
   paths={k:Path(evidence[k]) for k in ('model','records','responses','counters')}
   facts=json.loads(paths['model'].read_text())
   if facts.get('profile_step')!=evidence['step'] or facts.get('profile_end_step')!=evidence['step']:
    raise ValueError('auxiliary counter profiler range differs from the selected scheduler step')
   if facts.get('loaded_packages')!=model.get('loaded_packages') or facts.get('compilation_config')!=model.get('compilation_config'):
    raise ValueError('auxiliary counters used different loaded packages or compilation settings')
   responses=json.loads(paths['responses'].read_text())[evidence['repeat']]
   records=[json.loads(line) for line in paths['records'].read_text().splitlines()]
   selected=[r for r in records if r.get('phase')=='execute' and r['step']==evidence['step']]
   if len(selected)!=1 or evidence['step'] in self.memory:raise ValueError('auxiliary counters need one unique scheduler step')
   identity=capture_identity(facts,selected[0],responses)
   self.memory[evidence['step']]={'identity':identity,**read_raw(paths['counters'].read_text())}
   self.memory_sources.append({'step':evidence['step'],'repeat':evidence['repeat'],
                              **{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}})
  self.formats=vllm_formats({r['quant'] for r in self.profile['roles']},model,require_resolved=True)
  self.rows=defaultdict(lambda:defaultdict(float));self.unknown=set();self.reasons=defaultdict(set);self.limitations=defaultdict(set);self.steps=0
 def consume(self,item):
  engine={**item['identity'],'routing':self.routing.get(item.get('label_step',item['step']),[])}
  counters=self.memory.get(item['step'])
  attach=self.auxiliary if counters or self.counter_stream else self.attach
  bound=self.counter_stream.attach(item) if self.counter_stream else item['launches']
  launches=attach(bound,ssd_chunk=engine.get('mamba_chunk_size'),routing=engine['routing'])
  if counters:
   from boltbeam.collectors.cuda_memory import same_capture,bind
   same_capture(engine,counters['identity'])
  if counters or self.counter_stream:
   if self.triton:
    for launch in launches:
     derived=self.triton.work(launch)
     if derived is not None:
      if launch.get('semantic_work'):raise ValueError('compiled work overlaps existing semantic attribution')
      launch['semantic_work']=derived
  if counters:
   bind(launches,counters,source='matched replay auxiliary counter pass')
   self.used_memory.add(item['step'])
  attr=attribute_prefill(launches,[],chunk_widths=[engine['length']],outputs=1,layers=self.profile['layer_count'],
   kernel_paths=self.paths['kernels'],observed=self.paths.get('observed'),observed_by=self.paths.get('observed_by'),peaks=self.calibration['peaks_at'],
   bandwidth_gbs=self.calibration['bandwidth_gbs'],profile=self.profile,length=engine['length'],warmup=False,
   formats=self.formats,cache=self.calibration['cache'],depth=engine['context'],unc=self.calibration['uncertainty'],explicit_overlap=True)
  attr.update(engine=engine,bandwidth_gbs=self.calibration['bandwidth_gbs'],cache=self.calibration['cache'],peaks_at=self.calibration['peaks_at'])
  self.last_attr=attr
  verify_launches(self.run,self.profile,attr,{'engine':engine,'launches':launches},require_limits=True)
  if not math.isclose(sum(r['ms'] for r in attr['rows'])+attr['gaps_ms'],attr['wall_ms'],rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):raise ValueError('role rows do not reconcile')
  phase='prefill' if engine['replay']['prefill_tokens'] else 'decode'
  for r in [*attr['rows'],{'name':'gaps within scheduler steps','ms':attr['gaps_ms'],'ceiling_ms':0}]:
   out=self.rows[item['turn'],phase,r['name']]
   self.limitations[r['name']].update(r.get('limitations',[]))
   for k in ('ms','ceiling_ms','bytes','flops','calls'):out[k]+=r.get(k) or 0
   if r.get('ceiling_ms') is None:
    self.unknown.add(r['name'])
    self.reasons[r['name']].add(r.get('refused') or 'missing verified work dimensions and instruction-path evidence')
  self.steps+=1
 def result(self):
  if set(self.memory)!=self.used_memory:raise ValueError('auxiliary counter evidence contains uncaptured steps')
  return {'basis':'checkpoint-derived work on exact joined production kernels; rates from retained independent calibration; summed scheduler-step windows exclude gaps between steps',
   **({'counter_stream':self.counter_stream.finish()} if self.counter_stream else {}),
   'steps':self.steps,'unmodeled_rows':sorted(self.unknown),'auxiliary_counter_sources':self.memory_sources,
   'compiled_work_sources':self.triton.sources() if self.triton else [],
   'unmodeled_reasons':{k:sorted(v) for k,v in self.reasons.items()},
   'row_limitations':{k:sorted(v) for k,v in self.limitations.items() if v},
   'rows':[{'turn':t,'phase':phase,'name':name,**r} for (t,phase,name),r in self.rows.items()]}
