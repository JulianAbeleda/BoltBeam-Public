"""Exact scripted replay reconciliation, independent of report presentation.

The input manifest is local: paths and token content never enter the public result.
A failed join produces no result. Timing qualification is a separate, disclosed verdict.
"""
import hashlib,json,math,statistics
from pathlib import Path
from itertools import zip_longest
from boltbeam.collectors.replay_semantics import SETTINGS,session_identities
from boltbeam.collectors.vllm_two_pass import replay_steps
from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL,INTERVAL_ATOL
from boltbeam.artifacts.run_io import OUTPUT_ARTIFACTS, load_manifest

ARTIFACT='forced_replay.json'
SCHEMA='boltbeam.forced_replay.v1'


def read(path):return json.loads(Path(path).read_text())


def evidence(folder, responses):
  folder=Path(folder)
  def one(pattern):
    files=list(folder.glob(pattern))
    if len(files)!=1:raise ValueError(f'replay needs exactly one {pattern} worker artifact')
    return files[0]
  return {'model':read(one('*.model.json')),'records':[json.loads(l) for l in one('*.jsonl').read_text().splitlines()],
          'responses':responses}


def schedule(records, rows):
  ids={r['id']:i for i,r in enumerate(rows)};out=[]
  for record in records:
    if not record.get('scheduled_tokens'):continue
    requests=[]
    for request in record['requests']:
      matches=[i for key,i in ids.items() if request['request_id']==key or request['request_id'].startswith(key+'-')]
      if len(matches)!=1:raise ValueError('scheduler request has no unique response')
      requests.append({**request,'request_id':matches[0]})
    out.append({**record,'requests':requests})
  return out


def interval(xs):
  if not xs or any(not math.isfinite(a) or not math.isfinite(b) or b<=a for a,b in xs):
    raise ValueError('invalid or missing captured kernel interval')
  origin=min(a for a,b in xs);xs=sorted((a-origin,b-origin) for a,b in xs)
  first=last=xs[0][0];busy=total=0.
  for start,end in xs:
    busy+=max(0,end-max(start,last));last=max(last,end);total+=end-start
  wall=last-first;gap=wall-busy;overlap=busy-total
  if not math.isclose(total+overlap+gap,wall,rel_tol=ACCOUNTING_RTOL,abs_tol=INTERVAL_ATOL*1000):
    raise ValueError('launch/overlap/gap arithmetic mismatch')
  return {'captured_ms':wall/1000,'launch_ms':total/1000,'overlap_ms':overlap/1000,'gaps_ms':gap/1000,'kernels':len(xs)}


def band(samples):
  if len(samples)<2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples):
    raise ValueError('reference band requires at least two positive finite repeats')
  return [min(samples),max(samples)]


def check_clocks(row):
  sent,first,last=(row[k] for k in ('t_send','t_first','t_last'))
  if not all(math.isfinite(t) for t in (sent,first,last)) or not sent<=first<last:
    raise ValueError('invalid replay client clock order')
  for key,value in [('client_ms',(last-sent)*1000),('decode_client_ms',(last-first)*1000)]:
    if not math.isclose(row[key],value,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
      raise ValueError('replay client latency differs from recorded clocks: '+key)


def forcing_cost(rows):
  if rows and 'on' in rows[0]:
    for pair in rows:
      for mode in ('on','off'):check_clocks(pair[mode])
      if (pair['on']['usage']['completion_tokens']!=pair['off']['usage']['completion_tokens'] or
          pair['on']['choices'][0]['prompt_token_ids']!=pair['off']['choices'][0]['prompt_token_ids']):
        raise ValueError('forcing cost pair has different token counts or context')
    rows=[{key:r[mode]['decode_client_ms']/(r[mode]['usage']['completion_tokens']-1)
           for key,mode in [('forced','on'),('free','off')]} for r in rows]
  forced=[r['forced'] for r in rows];free=[r['free'] for r in rows]
  a,b=band(forced),band(free)
  return {'name':'token forcing','pairs':len(rows),'forced_band_ms':a,'free_band_ms':b,
          'delta_ms':statistics.mean(x-y for x,y in zip(forced,free)),
          'bands_overlap':max(a[0],b[0])<=min(a[1],b[1]),'samples':rows,
          'basis':'paired native decode ms/token; diagnostic, not added to captured session time'}


def diagnostics(manifest, *, responses=None):
  paired=Path(manifest.get('native_activity',{}).get('folder',manifest['paired']))
  responses=responses if responses is not None else read(paired/'responses.json');telemetry={}
  for rows in [*responses.values(),*read(Path(manifest['reference'])/'responses.json').values()]:
    for row in rows:check_clocks(row)
  if (paired/'telemetry.json').exists():
    from boltbeam.collectors.gpu_telemetry import summarize
    recorded=read(paired/'telemetry.json');windows=responses['timing']
    samples=[s for s in recorded['samples'] if any(t['t_first']<=s['t']<=t['t_last'] for t in windows)]
    telemetry=summarize(samples,power_limit_w=recorded.get('summary',{}).get('power_limit_w'))
  return {'telemetry':telemetry,'forcing_cost':forcing_cost(read(manifest['cost'])) if manifest.get('cost') else None}


def qualification(turns, *, native=False,overhead=None):
  missed={k:[t['turn'] for t in turns if not t[k+'_inside_band']] for k in ('client','decode')}
  if overhead is not None:
    if not native:raise ValueError('prospective overhead requires native activity capture')
    from boltbeam.collectors.replay_overhead import verdict
    return {**verdict(overhead),'outside_band_diagnostics':missed}
  if native:return {'status':'unavailable','outside_band':missed,
                    'reason':'native activity timing has no prospective overhead qualification'}
  return {'status':'failed' if any(missed.values()) else 'passed','outside_band':missed}


def review(manifest, *, say=lambda _:None):
  paired=Path(manifest['paired']);responses=read(paired/'responses.json')
  passes={m:evidence(paired/m,responses[m]) for m in ('timing','label')}
  activities=None
  if source:=manifest.get('native_activity'):
    from boltbeam.collectors.cuda_activity_stream import Stream
    activities=Stream(source);folder=Path(source['folder']);repeat=source.get('repeat','0')
    responses['timing']=read(folder/'responses.json')[repeat]
    passes['timing']=evidence(folder/repeat,responses['timing'])
  if passes['timing']['model'].get('loaded_packages')!=passes['label']['model'].get('loaded_packages'):
    raise ValueError('label and timing loaded package versions differ')
  script=read(manifest['script']);count=len(script)
  if not count or len(responses['timing'])!=count:raise ValueError('incomplete scripted replay')
  for mode,rows in responses.items():
    if len(rows)!=count:raise ValueError(f'{mode}: incomplete turns')
    for i,(row,t) in enumerate(zip(rows,script),1):
      if row['choices']!=[{'prompt_token_ids':t['prompt'],'token_ids':t['tokens']}]:
        raise ValueError(f'{mode}, turn {i}: response differs from token script')
  identities=session_identities(passes['timing']['model'],passes['timing']['records'],responses['timing'])
  expected=schedule(passes['timing']['records'],responses['timing'])
  reference=read(Path(manifest['reference'])/'responses.json')
  if len(reference)<2:raise ValueError('replay requires at least two forced reference repeats')
  for name,rows in reference.items():
    ref=evidence(Path(manifest['reference'])/name,rows)
    for key in (*SETTINGS,'model_config_sha256','activation_dtype','kv_cache_dtype','build','compilation_config',
                'logits_processors','triton_autotune_cache','loaded_packages'):
      if key not in ref['model'] or ref['model'][key]!=passes['timing']['model'].get(key):
        raise ValueError(f'reference {name}: different {key}')
    if len(rows)!=count:raise ValueError(f'reference {name}: incomplete turns')
    for i,(row,timed) in enumerate(zip(rows,responses['timing']),1):
      if row['choices']!=timed['choices'] or row['usage']!=timed['usage']:
        raise ValueError(f'reference {name}, turn {i}: response/cache mismatch')
    actual=schedule(ref['records'],rows)
    if actual!=expected:
      first=next(i for i,(a,b) in enumerate(zip_longest(actual,expected)) if a!=b)
      raise ValueError(f'reference {name}: scheduler record {first} differs')
  resources=None
  if resource_source:=manifest.get('resource_work'):
    if activities is None:raise ValueError('resource work requires verified native activities')
    from boltbeam.collectors.replay_resource_work import ResourceWork
    resources=ResourceWork(resource_source,passes['timing']['model'],paired,passes)
  roles=None;role_error='per-operation DRAM unavailable for full-replay resource accounting' if resources else 'no per-launch work and instruction-path evidence supplied'
  if manifest.get('calibration') and resources is None:
    from boltbeam.collectors.replay_roles import Roles
    try:
      options={'timing_folder':folder/repeat,'passes':passes} if activities is not None else {}
      roles=Roles(manifest['calibration'],passes['timing']['model'],paired,**options);role_error=None
    except (ValueError,KeyError,OSError) as exc:role_error=str(exc)
  segments=[{m:Path(p) for m,p in s.items()} for s in manifest['segments']]
  turns=[];current=None;groups={};steps=0
  def finish():
    row=responses['timing'][current-1];n=row['usage']['completion_tokens']-1
    if n<=0:raise ValueError('replay decode needs at least two generated tokens per turn')
    out={'turn':current,'steps':steps,'cached_tokens':identities[current-1]['cached_tokens'],
         'prefix':row['usage']['prompt_tokens'],'decode_tokens':n,**{k:interval(v) for k,v in groups.items()}}
    out['total']=interval(groups['prefill']+groups['decode'])
    out['phase_transition_ms']=out['total']['captured_ms']-sum(out[k]['captured_ms'] for k in ('prefill','decode'))
    out['client_ms']=row['client_ms'];out['client_boundary_ms']=row['client_ms']-out['total']['captured_ms']
    out['decode_client_ms_per_token']=row['decode_client_ms']/n
    out['client_samples_ms']=[r[current-1]['client_ms'] for r in reference.values()]
    out['decode_samples_ms']=[r[current-1]['decode_client_ms']/n for r in reference.values()]
    for key,value,samples in [('client',row['client_ms'],out['client_samples_ms']),
                              ('decode',out['decode_client_ms_per_token'],out['decode_samples_ms'])]:
      bounds=band(samples);out[key+'_band_ms']=bounds;out[key+'_inside_band']=bounds[0]<=value<=bounds[1]
    turns.append(out)
  for item in replay_steps(Path(manifest['graph']),passes,segments=segments,activities=activities):
    if item['turn']!=current:
      if current is not None:finish()
      current=item['turn'];groups={'prefill':[],'decode':[]};steps=0;say(f'joining turn {current} of {count}')
    if resources is not None:resources.consume(item)
    if roles is not None and role_error is None:
      try:roles.consume(item)
      except (ValueError,KeyError) as exc:role_error=f'turn {item["turn"]}, step {item["step"]}: {exc}'
    phase='prefill' if item['identity']['replay']['prefill_tokens'] else 'decode'
    groups[phase].extend((l['start_us'],l['start_us']+l['wall_us']) for l in item['launches']);steps+=1
  if current is not None:finish()
  if len(turns)!=count:raise ValueError('missing turn coverage')
  resource_window=None
  if resources is not None:
    from boltbeam.collectors.replay_resources import read_window
    try:resource_window=read_window(folder,activities.provenance())
    except ValueError as exc:resource_window={'status':'unavailable','reason':str(exc)}
  overhead=None
  if source:=manifest.get('overhead_experiment'):
    raw=Path(source['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('overhead evidence digest differs')
    overhead=json.loads(raw)
    if activities is None or overhead['reported_capture_manifest_sha256']!=activities.provenance()['manifest_sha256']:
      raise ValueError('overhead report source differs from selected native capture')
  acceptance_data={}
  if source:=manifest.get('observed_overhead_acceptance'):
    from boltbeam.collectors.replay_overhead import observed_acceptance
    raw=Path(source['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('observed acceptance policy digest differs')
    policy=json.loads(raw)
    if overhead is None:raise ValueError('observed acceptance requires the retained overhead experiment')
    acceptance_data={'overhead_acceptance_policy':policy,'overhead_acceptance':observed_acceptance(overhead,policy),
                     'overhead_acceptance_policy_sha256':source['sha256']}
  resource_data={};timing=qualification(turns,native=activities is not None,overhead=overhead)
  if resources is not None:
    from boltbeam.collectors.replay_resource_roofline import estimate
    work=resources.result(activities.provenance());calibration=None
    if source:=manifest.get("resource_clock_scope"):
      from boltbeam.collectors.replay_resource_roofline import clock_scope_from_capture
      work["clock_scope"]=clock_scope_from_capture(source,activities)
      work["clock_scope_source_sha256"]=source["sha256"]
    if source:=manifest.get('resource_calibration'):
      raw=Path(source['file']).read_bytes()
      if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('resource calibration digest differs')
      calibration={**json.loads(raw),'source_sha256':source['sha256']}
      work['gpu_identity']=json.loads((folder/'configuration.json').read_text())['gpu_identity']
    resource_data={'resource_work':work,'resource_window':resource_window,'resource_calibration':calibration,
                   'resource_roofline':estimate(work,resource_window,calibration,timing)}
  return {'schema':SCHEMA,'pairing':'passed','divergences':0,'turns':turns,
          **({'stopped':manifest['stopped']} if manifest.get('stopped') else {}),  # the trace's turns that did not fit
          **({'native_activity':activities.provenance()} if activities is not None else {}),
          **resource_data,**acceptance_data,
          'script_sha256':hashlib.sha256(Path(manifest['script']).read_bytes()).hexdigest(),
          'turn_output_lengths':[len(t['tokens']) for t in script],
          'loaded_packages':passes['timing']['model']['loaded_packages'],**diagnostics(manifest,responses=responses),
          'model_config_sha256':passes['timing']['model']['model_config_sha256'],
          'timing_qualification':timing,
          **({'overhead_experiment':overhead} if overhead is not None else {}),
          'limits':{'status':'unavailable','reason':role_error} if role_error else {'status':'partial' if roles.unknown else 'complete',**roles.result()},
          'totals':{phase:{key:sum(t[phase][key] for t in turns) for key in turns[0][phase]}
                    for phase in ('prefill','decode','total')}}


def checked(run):
  data=json.loads((Path(run)/ARTIFACT).read_text())
  return validate(data)


def validate(data):
  if data['schema']!=SCHEMA or data['pairing']!='passed' or data['divergences']!=0:
    raise ValueError('forced replay has no successful exact pairing')
  turns=data['turns']
  if not turns or [t['turn'] for t in turns]!=list(range(1,len(turns)+1)):
    raise ValueError('forced replay has incomplete turn coverage')
  def eq(name,a,b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
      raise ValueError('forced replay does not reconcile: '+name)
  for t,n in zip(turns,data['turn_output_lengths'],strict=True):
    if t['decode_tokens']!=n-1 or n<2:raise ValueError('forced replay output counts differ')
    for phase in ('prefill','decode','total'):
      x=t[phase]
      if x['captured_ms']<=0 or x['launch_ms']<=0 or x['gaps_ms']<0 or x['overlap_ms']>0:
        raise ValueError('forced replay interval signs are invalid')
      eq(phase,x['captured_ms'],x['launch_ms']+x['overlap_ms']+x['gaps_ms'])
    eq('phase split',t['total']['captured_ms'],t['prefill']['captured_ms']+t['decode']['captured_ms']+t['phase_transition_ms'])
    eq('client boundary',t['client_ms'],t['total']['captured_ms']+t['client_boundary_ms'])
    for key,value in [('client',t['client_ms']),('decode',t['decode_client_ms_per_token'])]:
      bounds=band(t[key+'_samples_ms'])
      if bounds!=t[key+'_band_ms'] or t[key+'_inside_band']!=(bounds[0]<=value<=bounds[1]):
        raise ValueError('forced replay repeat band does not derive from samples')
  if set(data['totals'])!={'prefill','decode','total'}:raise ValueError('missing replay phase totals')
  if len({len(t['decode_samples_ms']) for t in turns})!=1:raise ValueError('inconsistent reference repeat counts')
  for phase,total in data['totals'].items():
    for key,value in total.items():eq('session '+phase+'.'+key,value,sum(t[phase][key] for t in turns))
  if 'native_activity' in data:
    source=data['native_activity']
    if source['kernels']!=data['totals']['total']['kernels'] or len(source['manifest_sha256'])!=64:
      raise ValueError('native activity provenance does not reconcile')
  if work:=data.get('resource_work'):
    source=data.get('native_activity',{})
    if work['native_manifest_sha256']!=source.get('manifest_sha256') or work['status'] not in ('work_and_identity_passed','work_and_identity_passed_execution_rates_pending'):
      raise ValueError('resource work has no matching native provenance')
    totals=work['totals'];groups=work['groups']
    identities=[(g['binary_sha256'],g['entry']) for g in groups]
    if len(set(identities))!=len(identities) or any(len(binary)!=64 or any(c not in '0123456789abcdef' for c in binary) or not entry for binary,entry in identities):
      raise ValueError('invalid or duplicate resource binary/entry groups')
    if any(type(g['calls']) is not int or g['calls']<=0 for g in groups) or any(type(t[k]) is not int or t[k]<0 for t in [totals,*work['turns'].values()] for k in ('kernels','work_kernels','steps','attention_kernels')):
      raise ValueError('invalid resource integer counts')
    if totals['kernels']!=source['kernels'] or totals['work_kernels']!=source['kernels'] or sum(g['calls'] for g in groups)!=source['kernels']:
      raise ValueError('resource work launch coverage differs')
    if not groups or any(g['calls']<=0 or not math.isfinite(g['flops']) or g['flops']<0 or not math.isfinite(g['kernel_ms']) or g['kernel_ms']<=0 for g in groups):
      raise ValueError('invalid grouped resource work')
    eq('resource durations',sum(g['kernel_ms'] for g in groups),totals['kernel_ms'])
    eq('resource native durations',totals['kernel_ms'],data['totals']['total']['launch_ms'])
    if set(map(int,work['turns']))!=set(range(1,len(turns)+1)):raise ValueError('resource turns differ')
    for key,value in totals.items():eq('resource total '+key,value,sum(t[key] for t in work['turns'].values()))
    window=data['resource_window']
    if window['status']=='unavailable':
      if not window.get('reason'):raise ValueError('unavailable resource window has no reason')
    elif window['status']=='recorded_decode_complete':
      if window['native_manifest_sha256']!=source['manifest_sha256']:raise ValueError('resource window capture differs')
      start,end=window['window_ns'];first,last=source['kernel_window_ns']
      if not start<=first<last<=end:raise ValueError('resource window excludes native work')
      eq('resource window duration',window['window_ms'],(end-start)/1e6)
      if window['samples']<=0 or not 0<=window['boundary_samples']<=window['samples']:raise ValueError('invalid resource sample counts')
      for key in ('dram_read_bytes','dram_write_bytes','l1tex_l2_sectors'):
        values=[window[k][key] for k in ('contained','intersecting','outside_window')]
        if any(not math.isfinite(v) or v<0 for v in values) or values[0]>values[1]:raise ValueError('invalid temporal resource bounds')
    else:raise ValueError('resource window is not admitted')
  if data['timing_qualification']!=qualification(turns,native='native_activity' in data,overhead=data.get('overhead_experiment')):
    raise ValueError('forced replay timing verdict differs from repeat bands')
  if data.get('overhead_experiment') and data['overhead_experiment']['reported_capture_manifest_sha256']!=data['native_activity']['manifest_sha256']:
    raise ValueError('overhead capture provenance differs')
  if 'overhead_acceptance' in data:
    from boltbeam.collectors.replay_overhead import observed_acceptance
    if data['overhead_acceptance']!=observed_acceptance(data['overhead_experiment'],data['overhead_acceptance_policy']):
      raise ValueError('observed overhead acceptance differs from retained evidence and policy')
  if 'resource_roofline' in data:
    from boltbeam.collectors.replay_resource_roofline import estimate
    if data['resource_roofline']!=estimate(data['resource_work'],data['resource_window'],data['resource_calibration'],data['timing_qualification']):
      raise ValueError('resource roofline differs from admitted work, window and rates')
  if data.get('forcing_cost') and data['forcing_cost']!=forcing_cost(data['forcing_cost']['samples']):
    raise ValueError('token forcing cost does not derive from paired samples')
  limits=data['limits']
  if limits['status'] not in ('unavailable','partial','complete'):raise ValueError('invalid replay limit status')
  if limits['status']!='unavailable':
    for row in limits['rows']:
      if row['turn'] not in range(1,len(turns)+1) or row['phase'] not in ('prefill','decode'):
        raise ValueError('role row has no matching captured turn and phase')
      for field in ('ms','ceiling_ms','bytes','flops','calls'):
        if not math.isfinite(row[field]):raise ValueError('nonfinite replay role evidence')
    for t in turns:
      for phase in ('prefill','decode'):
        accounted=sum(r['ms'] for r in limits['rows'] if r['turn']==t['turn'] and r['phase']==phase)
        if accounted>t[phase]['captured_ms']+ACCOUNTING_ATOL:raise ValueError('role rows exceed the captured phase')
    names={r['name'] for r in limits['rows']};unknown=limits['unmodeled_rows']
    if len(set(unknown))!=len(unknown) or set(unknown)-names:
      raise ValueError('unmodeled roles do not match captured rows')
    if (limits['status']=='partial')!=bool(unknown):
      raise ValueError('replay limit status differs from role coverage')
  return data


def collect(run, *, provider='vllm', say=lambda _:None):
  """Own every child process; never attach to or stop an existing server."""
  import os,shutil,sys
  from boltbeam.runtime.bounded_job import run as bounded
  from boltbeam.collectors.providers import gpu_free
  from boltbeam.target.targets import get_target
  from boltbeam.collectors import vendor_capture,rocpd_trace
  run=Path(run).resolve();facts=load_manifest(run);backend=get_target(facts['target_id']).backend
  if provider!='vllm' or not vendor_capture.windowed(backend):
    raise ValueError(f'forced replay requires vLLM and a capture of a program-marked window; {backend} has none')
  if vendor_capture.plan(backend)['method']=='cupti-hes':
    raise ValueError('forced replay needs semantic/API tracing; cupti-hes captures kernel timing only')
  nsys=vendor_capture.plan(backend)['tool']  # rocprofv3 on AMD: one rocpd database, cut into nsys's files below
  if not nsys:raise ValueError('forced replay needs '+vendor_capture.plan(backend)['reason'])
  root=run/'forced_replay';root.mkdir(exist_ok=False)
  repo=Path(__file__).resolve().parents[2]
  env={**os.environ,'PYTHONPATH':os.pathsep.join([str(repo),str(repo/'boltbeam/runtime'),os.environ.get('PYTHONPATH','')])}
  env.setdefault('TRITON_CACHE_AUTOTUNING','1')
  env.setdefault('TRITON_CACHE_DIR',str(root/'triton-cache'))
  env.setdefault('VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR',str(root/'flashinfer-autotune'))
  def job(argv,name):
    result=bounded(argv,log=root/(name+'.log'),env=env,timeout=14400)
    if result['exit_code']!=0:raise RuntimeError(f'forced replay {name} failed; inspect its local log')
  for mode,repeats in [('record',5),('cost',30),('reference',5),('paired',5)]:
    free=gpu_free(backend)
    if not free['free']:raise RuntimeError(free['reason'])
    say('forced replay: '+mode)
    argv=[sys.executable,'-m','boltbeam.runtime.vllm_replay_drive',str(run),mode,str(repeats)]
    if mode=='paired':
      count=len(read(root/'token-script.json'))
      argv=([nsys,'profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi',f'--capture-range-end=repeat:{count+2}:async',
             '--output',str(root/'capture'),*argv] if backend=='CUDA' else
            vendor_capture.VENDORS['AMD'][2](nsys,argv,root,env,None,True))
    job(argv,mode)
  dbs=sorted(root.rglob('*_results.db'))
  if backend=='AMD' and len(cuts:=rocpd_trace.windows(dbs))!=count+2:
    raise ValueError(f'rocprofv3 recorded {len(cuts)} capture windows; the replay opens {count+2}')
  for i in range(1,count+3):  # the first window built the graphs: it alone carries their nodes' origins
    if backend=='AMD':rocpd_trace.export(dbs,root/f'capture.{i}.sqlite',window=cuts[i-1],nodes=i==1)
    else:job([nsys,'export','--type=sqlite','--output',str(root/f'capture.{i}.sqlite'),
              str(root/f'capture.{i}.nsys-rep')],f'export-{i}')
  manifest={**({'stopped':json.loads((root/'record-stopped.json').read_text())} if (root/'record-stopped.json').exists() else {}),
            'paired':str(root/'paired'),'reference':str(root/'reference'),'script':str(root/'token-script.json'),
            'cost':str(root/'cost/cost.json'),'graph':str(root/'capture.1.sqlite'),
            'segments':[{'timing':str(root/'capture.2.sqlite'),'label':str(root/f'capture.{i+3}.sqlite')} for i in range(count)]}
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  return publish(run,manifest,say=say)


def publish(run, manifest, *, say=lambda _:None):
  """Reconcile retained or newly captured passes, then atomically publish the result."""
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  run=Path(run);facts=load_manifest(run)
  (run/'forced_replay_request.json').write_text('{}')
  for name in ('report_validation.json','emit_validation.json'):
    (run/name).write_text(json.dumps({'status':'failed','error':'forced replay review has not completed'}))
  for name in (ARTIFACT,'gameplan.json','gameplan.md',*OUTPUT_ARTIFACTS): (run/name).unlink(missing_ok=True)
  result=review(manifest,say=say)
  validate(result)
  config=Path(facts['model_path'])/'config.json'
  if hashlib.sha256(config.read_bytes()).hexdigest()!=result['model_config_sha256']:
    raise ValueError('forced replay model config differs from this run')
  trace={'schema':SCHEMA_TIMING_TRACE,'model_id':facts['model_id'],'target_id':facts['target_id'],
         'workload':'replay','provider':'vllm','provider_id':'vllm','collector_id':'forced-replay',
         'timing_source':'exact scripted production capture; qualification is reported separately',
         'contexts':[max(t['prefix'] for t in result['turns'])],
         'rows':[{'scope':'turn','workload':'replay','i':t['turn'],'context':t['prefix'],
                  'prefix':t['prefix'],'wall_ms':t['total']['captured_ms'],
                  'prompt_ms':t['prefill']['captured_ms'],'predicted_ms':t['decode']['captured_ms'],
                  'predicted_n':t['decode_tokens']+1,'cache_n':t['cached_tokens'],'cold':False}
                 for t in result['turns']]}
  (run/'timing_trace.json').write_text(json.dumps(trace))
  temporary=run/(ARTIFACT+'.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(run/ARTIFACT)
  return run/'timing_trace.json'
