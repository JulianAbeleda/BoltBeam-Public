"""Prospective four-pair session-overhead verdict, separate from turn bands.

The one-sided t bound is conditional on independent approximately normal paired
excesses. Four observations cannot establish those assumptions or bound a
population maximum. Capture-validity evidence is supplied by the experiment
collector, and is kept separate from the arithmetic gate.
"""
import math,statistics

CHECKS=('artifact_integrity','exclusive_gpu','requests_tokens_cache','config_libraries',
        'native_observers_absent','complete_activity','complete_pm','thermal_balance')
PAIRS=4
ORDER=['NP','NP','PN','PN']
MAX_OVERHEAD=.05
T95_DF3=2.3533634348018264


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
    if set(pair['checks'])!=set(CHECKS):raise ValueError('incomplete overhead validity evidence')
    invalid.extend({'pair':pair['pair'],'check':key} for key,value in pair['checks'].items() if value is not True)
  differences=[p-(1+MAX_OVERHEAD)*n for n,p in zip(native,profiled)]
  ratios=[p/n-1 for n,p in zip(native,profiled)]
  if not all(math.isfinite(x) for x in [*differences,*ratios]):raise ValueError('overhead arithmetic overflow')
  observed=statistics.mean(profiled)/statistics.mean(native)-1
  mean=statistics.mean(differences);sd=statistics.stdev(differences);upper=mean+T95_DF3*sd/math.sqrt(PAIRS)
  if not all(math.isfinite(x) for x in [observed,mean,sd,upper]):raise ValueError('overhead arithmetic overflow')
  within=all(p<=(1+MAX_OVERHEAD)*n for n,p in zip(native,profiled));passed=upper<=0 and within
  return {'status':'unavailable' if invalid else 'passed' if passed else 'failed','scope':'full-session elapsed overhead relative to untraced forced replay; per-turn qualification unavailable',
          'maximum_overhead':MAX_OVERHEAD,'observed_mean_overhead':observed,'pair_overheads':ratios,
          'paired_excess_seconds':differences,'mean_excess_seconds':mean,'sd_excess_seconds':sd,'one_sided_95_upper_excess_seconds':upper,
          'invalid':invalid,'statistical_gate_passed':upper<=0,'all_observed_pairs_within_limit':within,
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
  invalid=[v for v in original['invalid'] if v['check']!='thermal_balance']
  within=all(p['profiled_seconds']<=(1+ceiling)*p['native_seconds'] for p in evidence['pairs'])
  return {'status':'unavailable' if invalid else 'accepted' if within else 'rejected',
          'maximum_observed_overhead':ceiling,'observed_overhead_range':[min(original['pair_overheads']),max(original['pair_overheads'])],
          'observed_mean_overhead':original['observed_mean_overhead'],'pair_overheads':original['pair_overheads'],
          'invalid':invalid,'disclosed_clock_balance_failures':[v for v in original['invalid'] if v['check']=='thermal_balance'],
          'original_qualification_status':original['status'],'experiment_digest':digest,
          'scope':'retrospective acceptance of these four observed pairs only; clock variation confounds isolated profiler cost; no population or future guarantee; original prospective qualification unchanged'}
