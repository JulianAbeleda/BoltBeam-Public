"""Prospective four-pair session-overhead verdict, separate from turn bands.

The one-sided t bound is conditional on independent approximately normal paired
excesses. Four observations cannot establish those assumptions or bound a
population maximum. Capture-validity evidence is supplied by the experiment
collector, and is kept separate from the arithmetic gate. Paired thermal/clock balance is a
mandatory disclosure (per pair, with the clock bound on its timing influence),
not a validity check.
"""
import math,statistics

CHECKS=('artifact_integrity','exclusive_gpu','requests_tokens_cache','config_libraries',
        'native_observers_absent','complete_activity','complete_pm')
THERMAL_KEYS=('sm_mhz','mem_mhz','temp_c','power_w','limit_duty')
REFERENCE_TEMPERATURE_C=2.0  # the former engineering gate; reported, never gating
PAIRS=4
ORDER=['NP','NP','PN','PN']
MAX_OVERHEAD=.05
T95_DF3=2.3533634348018264


def thermal_disclosure(pairs,overheads):
  """Per-pair clock/temperature/power evidence plus the descriptive influence assessment; never gates."""
  rows=[]
  for pair in pairs:
    arms=pair.get('thermal')
    if not isinstance(arms,dict) or set(arms)!={'native','instrumented'}:raise ValueError('thermal disclosure missing')
    for arm in arms.values():
      if not isinstance(arm,dict) or any(type(arm.get(k)) not in (int,float) or not math.isfinite(arm[k]) or arm[k]<0 for k in THERMAL_KEYS) or not arm['sm_mhz']*arm['mem_mhz']:raise ValueError('thermal disclosure invalid')
    n,i=arms['native'],arms['instrumented'];d={'sm_relative':i['sm_mhz']/n['sm_mhz']-1,'memory_relative':i['mem_mhz']/n['mem_mhz']-1,'temperature_C':i['temp_c']-n['temp_c'],'power_W':i['power_w']-n['power_w'],'limit_duty':i['limit_duty']-n['limit_duty']}
    rows.append({'pair':pair['pair'],'native':n,'instrumented':i,'differences':d,'sm_clock_speed_bound':abs(d['sm_relative']),'reference_temperature_exceeded':abs(d['temperature_C'])>REFERENCE_TEMPERATURE_C})
  t=[r['differences']['temperature_C'] for r in rows];w=[r['differences']['power_W'] for r in rows];fit=[None,None] if len(set(t))<2 else statistics.linear_regression(t,overheads);r=statistics.correlation(t,overheads) if len(set(t))>1 and len(set(overheads))>1 else None  # None: undefined when an input is constant
  return {'pairs':rows,'gating':False,'temperature_C':{'min':min(t),'max':max(t),'mean':statistics.mean(t),'sd':statistics.stdev(t)},'power_W':{'min':min(w),'max':max(w),'mean':statistics.mean(w),'sd':statistics.stdev(w)},
          'reference_temperature_limit_C':REFERENCE_TEMPERATURE_C,'reference_temperature_exceeded':any(r['reference_temperature_exceeded'] for r in rows),'max_sm_clock_speed_bound':max(r['sm_clock_speed_bound'] for r in rows),
          'overhead_vs_temperature':{'n':len(rows),'correlation':r,'slope_overhead_per_C':fit[0],'status':'descriptive only; n=4 supports no inference'},
          'basis':'thermal effect on timing acts through clock speed; the relative SM-clock difference bounds that influence'}

def verdict(evidence):
  pairs=evidence['pairs']
  if len(pairs)!=PAIRS or [p['pair'] for p in pairs]!=list(range(1,PAIRS+1)):
    raise ValueError('overhead experiment requires all four fixed pairs')
  if evidence['order']!=ORDER or evidence['maximum_overhead']!=MAX_OVERHEAD:
    raise ValueError('overhead experiment differs from frozen design')
  for key in ('protocol_sha256','reported_capture_manifest_sha256'):
    digest=evidence[key]
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('overhead provenance digest invalid')
  invalid=[];native=[];profiled=[]
  for pair in pairs:
    for key,out in [('native_seconds',native),('profiled_seconds',profiled)]:
      value=pair[key]
      if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('invalid overhead elapsed interval')
      out.append(value)
    if set(pair['checks'])-{'thermal_balance'}!=set(CHECKS):raise ValueError('incomplete overhead validity evidence')
    invalid.extend({'pair':pair['pair'],'check':key} for key,value in pair['checks'].items() if key in CHECKS and value is not True)
  differences=[p-(1+MAX_OVERHEAD)*n for n,p in zip(native,profiled)]
  ratios=[p/n-1 for n,p in zip(native,profiled)]
  if not all(math.isfinite(x) for x in [*differences,*ratios]):raise ValueError('overhead arithmetic overflow')
  observed=statistics.mean(profiled)/statistics.mean(native)-1
  mean=statistics.mean(differences);sd=statistics.stdev(differences);upper=mean+T95_DF3*sd/math.sqrt(PAIRS)
  if not all(math.isfinite(x) for x in [observed,mean,sd,upper]):raise ValueError('overhead arithmetic overflow')
  within=all(p<=(1+MAX_OVERHEAD)*n for n,p in zip(native,profiled));passed=upper<=0 and within
  thermal=thermal_disclosure(pairs,ratios)
  return {'status':'unavailable' if invalid else 'passed' if passed else 'failed','scope':'full-session elapsed overhead relative to untraced forced replay; per-turn qualification unavailable',
          'maximum_overhead':MAX_OVERHEAD,'observed_mean_overhead':observed,'pair_overheads':ratios,
          'paired_excess_seconds':differences,'mean_excess_seconds':mean,'sd_excess_seconds':sd,'one_sided_95_upper_excess_seconds':upper,
          'invalid':invalid,'thermal_disclosure':thermal,'statistical_gate_passed':upper<=0,'all_observed_pairs_within_limit':within,
          'assumptions':'conditional on independent approximately normal paired excesses; four pairs do not establish these assumptions or guarantee a population maximum',
          'protocol_sha256':evidence['protocol_sha256'],'reported_capture_manifest_sha256':evidence['reported_capture_manifest_sha256']}

def observed_acceptance(evidence,policy):
  """Apply an explicit retrospective user policy without changing qualification."""
  import hashlib,json
  digest=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if policy['kind']!='retrospective_observed_overhead_acceptance' or policy['experiment_digest']!=digest:
    raise ValueError('observed acceptance is not bound to this experiment')
  ceiling=policy['maximum_observed_overhead']
  if type(ceiling) not in (int,float) or not math.isfinite(ceiling) or not 0<=ceiling<=1:
    raise ValueError('invalid observed acceptance ceiling')
  original=verdict(evidence)
  within=all(p['profiled_seconds']<=(1+ceiling)*p['native_seconds'] for p in evidence['pairs'])
  return {'status':'unavailable' if original['invalid'] else 'accepted' if within else 'rejected',
          'maximum_observed_overhead':ceiling,'observed_overhead_range':[min(original['pair_overheads']),max(original['pair_overheads'])],
          'observed_mean_overhead':original['observed_mean_overhead'],'pair_overheads':original['pair_overheads'],
          'invalid':original['invalid'],'thermal_disclosure':original['thermal_disclosure'],
          'original_qualification_status':original['status'],'experiment_digest':digest,
          'scope':'retrospective acceptance of these four observed pairs only; clock variation confounds isolated profiler cost; no population or future guarantee; original prospective qualification unchanged'}
