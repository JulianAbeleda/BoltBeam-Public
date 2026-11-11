"""One checked scripted-replay table for Run, HTML and Emit."""
import math,statistics
from boltbeam.collectors.forced_replay import band,checked
from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL

ACCOUNTING_ROWS = {'gaps within scheduler steps','concurrent kernel overlap (subtract)'}


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
