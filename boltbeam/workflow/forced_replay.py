"""One checked scripted-replay table for Run, HTML and Emit."""
import json,math,statistics
from pathlib import Path
from boltbeam.collectors.forced_replay import ARTIFACT,SCHEMA,band,forcing_cost,qualification
from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL

ACCOUNTING_ROWS = {'gaps within scheduler steps','concurrent kernel overlap (subtract)'}


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


def coverage(data):
  """Kernel-duration coverage; gaps and overlap never count as modeled work."""
  limits=data['limits']
  if limits['status']=='unavailable':return {'status':'unavailable','reason':limits['reason']}
  unknown=set(limits['unmodeled_rows']);groups={};modeled=0.
  for row in limits['rows']:
    if row['name'] in ACCOUNTING_ROWS:continue
    if row['ms']<0:raise ValueError('negative kernel duration in roofline coverage')
    if row['name'] in unknown:groups[row['name']]=groups.get(row['name'],0.)+row['ms']
    else:modeled+=row['ms']
  total=data['totals']['total'];unmodeled=sum(groups.values());accounted=modeled+unmodeled
  if not math.isclose(accounted,total['launch_ms'],rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
    return {'status':'unavailable','reason':'role durations do not cover the raw kernel duration'}
  return {'status':limits['status'],'basis':'sum of raw kernel durations; excludes gaps and overlap correction',
          'modeled_kernel_ms':modeled,'unmodeled_kernel_ms':unmodeled,'kernel_ms':accounted,
          'modeled_kernel_pct':100*modeled/accounted,'gaps_ms':total['gaps_ms'],'overlap_ms':total['overlap_ms'],
          'missing':[{'name':name,'ms':ms,'kernel_pct':100*ms/accounted,
                      'reasons':limits.get('unmodeled_reasons',{}).get(name,['missing verified limit evidence'])}
                     for name,ms in sorted(groups.items(),key=lambda item:-item[1])]}


def table(run):
  data=checked(run);rows=data['turns'];total=data['totals']['total'];tokens=sum(r['decode_tokens'] for r in rows)
  native=[sum(r['decode_samples_ms'][i]*r['decode_tokens'] for r in rows)/tokens for i in range(len(rows[0]['decode_samples_ms']))]
  measured=sum(r['decode_client_ms_per_token']*r['decode_tokens'] for r in rows)/tokens
  bounds=band(native);cost=data['forcing_cost'];limits=data['limits'];qualified=data['timing_qualification']['status']
  words=['Exact scripted production capture: all turns included, including the first. Launch time plus overlap correction plus GPU gaps equals captured time.',
         'Prefill and decode are kernel-window times; transition is the interval between their windows. Client timing is reported separately.',
         'GPU gaps include launch, scheduling and profiler overhead; they are not attributed model loss.',
         'Full session limits, % of floor and largest loss are unavailable unless every launch has verified work and instruction-path evidence.']
  lines=[f"Exact pairing: {len(rows)} turns, {data['divergences']} divergences. Timing qualification: {qualified.upper()}.",
         f'Captured client decode {measured:.6f} ms/token; forced untraced band [{bounds[0]:.6f}, {bounds[1]:.6f}] ms/token.',
         f"GPU gaps: {total['gaps_ms']:.3f} ms ({100*total['gaps_ms']/total['captured_ms']:.2f}% of captured time).",
         'Session limit: unavailable; % of floor: unavailable; largest loss: unavailable.']
  if data.get('overhead_experiment'):
    q=data['timing_qualification']
    lines.append(f"Prospective session overhead: {100*q['observed_mean_overhead']:.3f}%; four fixed pairs; one-sided 95% upper excess over the 5% limit {q['one_sided_95_upper_excess_seconds']:.6f} s.")
    lines.append('Qualification protocol: '+q['protocol_sha256']);h=q['thermal_disclosure'];o=h['overhead_vs_temperature']
    lines.append(f"Thermal disclosure (not a gate): instrumented minus native temperature {h['temperature_C']['min']:+.2f} to {h['temperature_C']['max']:+.2f} C (mean {h['temperature_C']['mean']:+.2f}, SD {h['temperature_C']['sd']:.2f}), power {h['power_W']['mean']:+.1f} W; SM-clock difference at most {100*h['max_sm_clock_speed_bound']:.3f}%; 2 C reference exceeded: {'yes' if h['reference_temperature_exceeded'] else 'no'}.")
    words.extend([f"Thermal influence: timing is affected through clock speed, bounded per pair by the relative SM-clock difference; overhead vs temperature difference correlation {o['correlation'] or 0:+.2f}, slope {100*(o['slope_overhead_per_C'] or 0):+.3f} %/C ({o['status']}).",q['scope'],q['assumptions'],'Reference turn bands are diagnostics; session-level qualification does not certify individual turns.'])
  if limits['status']=='unavailable':words.append('Limit evidence: '+limits['reason'])
  else:
    modeled=sum(r['ceiling_ms'] for r in limits['rows'] if r['name'] not in limits['unmodeled_rows'])
    grouped={}
    for r in limits['rows']:
      if r['name'] not in limits['unmodeled_rows'] and r['name'] not in ACCOUNTING_ROWS:
        grouped[r['name']]=grouped.get(r['name'],0)+r['ms']-r['ceiling_ms']
    if grouped:
      largest=max(grouped,key=grouped.get);lines.append(f'Largest modeled loss: {largest}, {grouped[largest]:.3f} ms; excludes unmodeled rows.')
    lines.append(f'Partial modeled floor: {modeled:.3f} ms ({100*modeled/total["captured_ms"]:.2f}% of captured time); not the full session floor.')
  for name,notes in limits.get('row_limitations',{}).items():words.append(name+': '+' '.join(notes))
  covered=coverage(data)
  if covered['status']!='unavailable':
    lines.append(f"Roofline coverage: {covered['modeled_kernel_pct']:.2f}% of summed kernel time; {covered['unmodeled_kernel_ms']:.3f} ms unmodeled. Gaps and overlap excluded.")
    if covered['missing']:
      missing=covered['missing'][0]
      lines.append(f"Largest coverage gap: {missing['name']}, {missing['ms']:.3f} ms. "+'; '.join(missing['reasons']))
  else:words.append('Roofline coverage unavailable: '+covered['reason'])
  clock_words=None
  if work:=data.get('resource_work'):
    if scope:=work.get('clock_scope'):
      from boltbeam.collectors.replay_resource_roofline import _clock_scope
      _clock_scope(work)
      observations=scope['observations'];samples=sum(row['samples'] for row in observations)
      mean=sum(row['sm_mhz']['mean']*row['samples'] for row in observations)/samples
      peak=max(row['sm_mhz']['max'] for row in observations)
      clock_words=(f"Timing scope: SM clock requested externally at {scope['requested_sm_mhz']:g} MHz; "
                   f"observed capture mean {mean:g} MHz, peak {peak:g} MHz. Timing qualification is reported separately.")
    lines[3]='Per-operation session limit: unavailable; per-operation % of floor and largest loss: unavailable.'
    words=[s.replace('Roofline coverage unavailable:', 'Per-operation roofline coverage unavailable:') for s in words]
    counts=work['totals']
    lines.append(f"Resource work accounting: {counts['work_kernels']:,}/{counts['kernels']:,} native launches; {counts['kernel_ms']:.3f} ms summed kernel duration. Per-operation DRAM unavailable.")
    words.append(work['scope'])
    window=data['resource_window'];lines.append('Replay PM window: '+window['status']+('. '+window['reason'] if window['status']=='unavailable' else '; temporal device traffic, including background/control demand.'))
    resource=data.get('resource_roofline')
    if resource and resource['status']!='unavailable':
      lines[3]=f"Full-replay empirical resource estimate: {resource['floor_low_ms']:.3f}–{resource['floor_high_ms']:.3f} ms over its {resource['window_ms']:.3f} ms worker window ({resource['estimate_pct'][0]:.2f}–{resource['estimate_pct'][1]:.2f}%). {resource['status']}."
      lines.append(f"Resource provenance: capture {resource['native_manifest_sha256']}; rates and exact binary/entry bindings {resource['calibration_source_sha256']}.")
      if resource['estimate_exceeds_observed_window']:lines.append('Empirical rate sensitivity exceeds the observed window; reported without clamping.')
      words.append(resource['scope'])
    else:lines.append('Resource estimate unavailable: '+'; '.join(resource['reasons'] if resource else ['execution-resource rates pending']))
  if accepted:=data.get('overhead_acceptance'):
    low,high=accepted['observed_overhead_range'];ceiling=accepted['maximum_observed_overhead']
    lines.insert(0,f"Retrospective observed-overhead acceptance: {accepted['status'].upper()}; observed {100*low:.3f}–{100*high:.3f}% against a {100*ceiling:g}% ceiling. Original prospective {100*data['timing_qualification']['maximum_overhead']:g}% qualification: {qualified.upper()}.")
    words.append(accepted['scope'])
  telemetry=data.get('telemetry') or {}
  packages=data.get('loaded_packages') or {}
  if packages:lines.append('Loaded packages: '+', '.join(f'{k} {v or "version unavailable"}' for k,v in sorted(packages.items())))
  if telemetry.get('words'):lines.append('Capture clock basis: '+telemetry['words'])
  if telemetry.get('throttled'):lines.append('THROTTLED capture: modeled limits use the recorded calibration; clock variation can affect reference-band qualification.')
  excess=[r['decode_client_ms_per_token']-statistics.median(r['decode_samples_ms']) for r in rows]
  lines.append(f'Capture/reference difference: {min(excess):+.6f} to {max(excess):+.6f} ms/token across turns; not added to the session.')
  words.append('Gaps between kernels can include tracer overhead that grows with session length, as well as CUDA memory activity. They are not proven GPU idle.')
  if cost:
    lines.append(f"Token forcing: {cost['delta_ms']:+.6f} ms/token; forced band {cost['forced_band_ms']}, free band {cost['free_band_ms']}; "
                 f"bands {'overlap' if cost['bands_overlap'] else 'DO NOT OVERLAP'}. Separate diagnostic, not added to the session.")
  else:lines.append('Token forcing: cost unavailable; no paired cost measurement supplied.')
  turns=[{'i':r['turn'],'cold':False,'prefix':r['prefix'],'ttft_ms':r['prefill']['captured_ms'],
          'wall_ms':r['total']['captured_ms'],'prompt_ms':r['prefill']['captured_ms'],'predicted_ms':r['decode']['captured_ms'],
          'overhead_ms':r['phase_transition_ms'],'predicted_n':r['decode_tokens']+1,
          'decode_tok_s':1000*r['decode_tokens']/r['decode']['captured_ms'],'roofline_ms':None} for r in rows]
  head={'turns':len(turns),'warm_turns':len(turns),'time_ms':total['captured_ms'],'roofline_ms':None,'e_pct':None,'lost':None,
        'split':{k:sum(t[f] for t in turns) for k,f in [('prefill','prompt_ms'),('decode','predicted_ms'),('overhead','overhead_ms')]},
        'decode_tok_s_median':statistics.median(t['decode_tok_s'] for t in turns),'deepest_prefix':max(t['prefix'] for t in turns)}
  floors={r['turn']:sum(x['ceiling_ms'] for x in limits.get('rows',[]) if x['turn']==r['turn'] and x['name'] not in limits.get('unmodeled_rows',[])) for r in rows}
  columns=['turn','prefix','cache','prefill ms','decode ms','transition ms','launch ms','overlap ms','gaps ms','captured ms','modeled limit ms','% modeled floor','timing']
  def cells(r):return [str(r['turn']),str(r['prefix']),str(r['cached_tokens']),*[f'{v:.3f}' for v in
    [r['prefill']['captured_ms'],r['decode']['captured_ms'],r['phase_transition_ms'],r['total']['launch_ms'],
     r['total']['overlap_ms'],r['total']['gaps_ms'],r['total']['captured_ms']]],
    *([f"{floors[r['turn']]:.3f}",f"{100*floors[r['turn']]/r['total']['captured_ms']:.2f}"] if limits['status']!='unavailable' else ['unavailable','unavailable']),
    'session only' if data.get('overhead_experiment') else 'UNAVAILABLE' if qualified=='unavailable' else 'pass' if r['client_inside_band'] and r['decode_inside_band'] else 'FAILED']
  totals=['all','','',*[f'{v:.3f}' for v in [head['split']['prefill'],head['split']['decode'],head['split']['overhead'],
          total['launch_ms'],total['overlap_ms'],total['gaps_ms'],total['captured_ms']]],*([f'{sum(floors.values()):.3f}',f"{100*sum(floors.values())/total['captured_ms']:.2f}"] if limits['status']!='unavailable' else ['unavailable','unavailable']),qualified.upper()]
  return {**({'timing_scope':clock_words} if clock_words else {}),'title':'Forced replay: exact production work','columns':columns,'rows':[{'i':r['turn'],'cells':cells(r)} for r in rows],
          'total':totals,'turns':turns,'headline':head,'headline_lines':lines,'words':' '.join(words),
          'evidence_words':'Exact production kernel pairing; native reference bands; limits never inferred from model names.',
          'next_step':'Review the reconciled replay report and its disclosed timing and resource ranges.',
          'forced':True,'overhead_acceptance':data.get('overhead_acceptance'),'limit_coverage':covered,'resource_roofline':data.get('resource_roofline'),'qualification':data['timing_qualification'],'wheres':role_table(data),'decode_where':None}


def role_table(data):
  limits=data['limits']
  if limits['status']=='unavailable':return []
  groups={}
  for row in limits['rows']:
    out=groups.setdefault(row['name'],[0.,0.]);out[0]+=row['ms'];out[1]+=row['ceiling_ms']
  wall=data['totals']['total']['captured_ms']
  groups['between-step interval correction']=[wall-sum(v[0] for v in groups.values()),0.]
  unknown=set(limits['unmodeled_rows']);floor=sum(v[1] for k,v in groups.items() if k not in unknown)
  rows=[{'name':name,'section':('Timing adjustments' if name=='between-step interval correction' else 'Supporting kernels — no model' if name in unknown else 'Model operations'),'cells':[f'{ms:.3f}',*([f'{limit:.3f}',f'{ms-limit:.3f}',f'{100*limit/ms:.2f}' if ms else '-'] if name not in unknown else ['unavailable']*3)]}
        for name,(ms,limit) in sorted(groups.items(),key=lambda item:item[1][0],reverse=True)]
  order={'Model operations':0,'Supporting kernels — no model':1,'Timing adjustments':2}
  rows.sort(key=lambda row:order[row['section']])
  return [{'status':'measured','estimate':False,'title':'Captured roles: partial limits','columns':['role','captured ms','limit ms','lost ms','% of floor'],
           'rows':rows,'sum_cells':['all',f'{wall:.3f}',f'{floor:.3f} modeled','unavailable','unavailable'],
           'words':limits['basis']+'. Unknown role limits remain unavailable; correction closes step windows to the full interval.',
           'server_line':'Captured timing qualification remains separate from arithmetic validation.'}]


def loss_block(run,base):
  t=table(run);h=t['headline']
  return {**base,**({'timing_scope':t['timing_scope']} if t.get('timing_scope') else {}),'workload':'replay','per':'per session','status':'measured','limit_ms':None,'limit_tok_s':None,
          'roles':[],'replay':t,'resource_roofline':t.get('resource_roofline'),
          'reason':'per-operation replay limits unavailable; full-replay resources reported separately' if t.get('resource_roofline') else 'full captured replay limits unavailable',
          'runtimes':[{'provider':'vllm','ms':h['time_ms'],'tok_s':None,'lost_ms':None,'per_role':False,
                       'note':'all scripted turns, captured kernel windows; timing qualification '+t['qualification']['status']}]}
