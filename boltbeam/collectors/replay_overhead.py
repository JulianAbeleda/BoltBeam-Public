"""The capture-overhead ceiling (MAX_OVERHEAD) and every verdict against it: within_ceiling and its words, a
capture's end-to-end check against its untraced run (end_to_end), whether a capture takes repeats
(capture_over_ceiling); and the prospective four-pair session-overhead verdict, separate from turn bands.

The one-sided t bound is conditional on independent approximately normal paired
excesses. Four observations cannot establish those assumptions or bound a
population maximum. Capture-validity evidence is supplied by the experiment
collector, and is kept separate from the arithmetic gate. Paired thermal/clock balance is a
mandatory disclosure (per pair, with the clock bound on its timing influence),
not a validity check.
"""
import math,pathlib,statistics
from typing import Any

from boltbeam.collectors.capture_settings import step4_token
from boltbeam.core.policy import value

CHECKS=('artifact_integrity','exclusive_gpu','requests_tokens_cache','config_libraries',
        'native_observers_absent','complete_activity','complete_pm')
THERMAL_KEYS=('sm_mhz','mem_mhz','temp_c','power_w','limit_duty')
REFERENCE_TEMPERATURE_C=2.0  # the former engineering gate; reported, never gating
PAIRS=4
ORDER=['NP','NP','PN','PN']
MAX_OVERHEAD=value('replay_max_overhead')  # the capture-overhead ceiling, a stated choice (data/policy.json)
T95_DF3=2.3533634348018264
CEILING_PCT = 100 * MAX_OVERHEAD  # the user's 5% ceiling, from the one authority (data/policy.json, read above)


within_ceiling = lambda pct: abs(pct) <= CEILING_PCT + 1e-9  # noqa: E731  # a capture far faster than native is not comparable


def overhead_words(pct:float, reference:str = "native") -> str:
  """THE disclosure of a measured capture overhead: signed, against its reference, against the ceiling. A capture
  faster than its reference qualifies within the same ceiling; beyond it the arms differ in more than the capture
  (first-use costs, environment) and are not comparable. Nothing is subtracted from any number."""
  shown = f"{pct:+.4f}" if not within_ceiling(pct) and f"{abs(pct):.2f}" == f"{CEILING_PCT:.2f}" else f"{pct:+.2f}"  # never round an excess into the ceiling
  return (f"capture {shown}% {'over' if pct >= 0 else 'under'} {reference}, {'within' if within_ceiling(pct) else 'above' if pct >= 0 else 'beyond'} "
          f"the {CEILING_PCT:g}% capture-overhead ceiling" + ("" if pct >= 0 or within_ceiling(pct) else " below it: the arms are not comparable"))


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


def capture_over_ceiling(run:pathlib.Path, context:float | None, provider:str, captured_ms:float | None) -> bool:
  """Whether a capture's own token is over the ceiling against the run's untraced one: when its collector takes the
  repeats over_ceiling needs (engine_common.captured_role_time)."""
  untraced, _ = step4_token(run, context, provider)
  return bool(untraced and captured_ms) and not within_ceiling(100 * (captured_ms / untraced - 1))


def end_to_end(attr:dict[str, Any], untraced_ms:float | None, what:str = "prefill", *, band:dict | None = None,
               reference:str = "untraced median", reference_window:dict | None = None) -> dict[str, Any]:
  """Qualify a capture by its measured overhead against the untraced run, from full engine elapsed timers with
  identical recorded measurement windows. diff_pct = 100 * (captured - untraced) / untraced; `confirms` is True when
  it is at most the 5% ceiling (negative qualifies), False above it, None ("not checked") when the windows differ or
  are unrecorded. The repeat band no longer gates: it is only said, as "within repeat noise". The GPU kernel span and
  its gaps reconcile separately; neither substitutes for the complete host interval. gap_growth_ms is its own figure:
  the capture's idle beyond what the untraced prefill leaves room for (the gaps less the untraced time over the
  kernels' busy time; zero when the gaps fit in it). An engine's own timing sums to the untraced prefill already, so
  there is nothing to check."""
  captured, gaps, busy = attr.get("measured_ms"), float(attr.get("gaps_ms") or 0.0), float(attr.get("busy_ms") or 0.0)
  if untraced_ms is None or not captured or not (math.isfinite(captured) and math.isfinite(untraced_ms) and captured > 0 and untraced_ms > 0):
    return {"confirms": None, "words": "no matching captured engine elapsed timer and untraced time: the GPU kernel span alone cannot confirm end-to-end timing"}
  if abs(float(attr["wall_ms"]) - float(untraced_ms)) < 1e-9 and abs(float(attr["wall_ms"]) - captured) > 1e-9:
    return {"confirms": None, "words": "the rows sum to the untraced prefill (the engine's own timing): no capture to check"}
  diff = captured - float(untraced_ms)
  share = diff / float(untraced_ms)
  growth = max(gaps - max(float(untraced_ms) - busy, 0.0), 0.0)
  from boltbeam.collectors.capture_scope import end_to_end_width
  matched = bool(reference_window) and attr.get("timing_window") == reference_window
  width = end_to_end_width(band, captured_key="engine_elapsed_ms") if matched else None
  ok = within_ceiling(100 * share) if matched else None
  noise = "no engine-elapsed repeat band" if width is None else f"{'yes' if abs(diff) <= width else 'no'}, band ±{width:.3f} ms"
  verdict = (overhead_words(100 * share, "native" if reference.startswith("untraced") else reference) if matched else
             "not checked: native and captured timing windows differ or are unrecorded")
  return {"confirms": ok, "captured_ms": captured, "kernel_span_ms": attr.get("profiled_wall_ms"), "untraced_ms": float(untraced_ms), "diff_ms": diff, "diff_pct": 100 * share,
          "reference": reference,
          "band_pct": 100 * width / float(untraced_ms) if width is not None else None, "band_ms": width,
          "status": "checked" if ok is not None else "not checked", "gaps_ms": gaps, "gap_growth_ms": growth,
          "words": (f"{verdict} (the captured engine timer for {what} took {captured:.1f} ms against the {float(untraced_ms):.1f} ms {reference}, {100 * share:+.1f}%; "
                    f"within repeat noise: {noise}); its gaps are {gaps:.1f} ms, "
                    f"{growth:.1f} ms more than the untraced {what} leaves room for"
                    + ("" if ok is not False else f". The rows stand, but this capture does not qualify the {what} end to end"))}
