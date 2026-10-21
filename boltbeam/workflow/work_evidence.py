"""Verify captured work provenance independently of model and engine families.

Collectors produce facts; this boundary checks their identity and conservation.
It never fills missing facts from architecture names or expected performance.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from boltbeam.math.numerics import ACCOUNTING_RTOL, ACCOUNTING_ATOL, INTERVAL_ATOL


def verify_components(row:dict, works:list[dict]) -> None:
  """Recompute each launch bound before checking the grouped row's totals."""
  from boltbeam.roofline.memory import hierarchy_ms
  components=row.get('components') or []
  if len(components)!=len(works):
    raise ValueError(f'work evidence {row["name"]} components do not cover launches')
  def equal(field,actual,expected):
    if not math.isclose(actual,expected,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
      raise ValueError(f'work evidence {row["name"]} {field} differs from derived work/rates')
  rates=row['memory_rates'];dram=rates['dram_gbs'];peak=row['peak_tflops']
  memory_only=all(w.get('path')=='memory_only' and w['flops']==0 for w in works)
  if memory_only and (peak is not None or row.get('compute_paths')):raise ValueError('memory-only evidence cannot claim floating-point throughput')
  if any(not math.isfinite(v) or v<=0 for v in ((dram,) if memory_only else (dram,peak))):
    raise ValueError('work evidence requires positive finite rates')
  totals=defaultdict(float)
  for component,work in zip(components,works):
    compute=0. if memory_only else work['flops']/(peak*1e9)
    if work.get('requires_memory_evidence'):
      strip=lambda traffic:{k:v for k,v in traffic.items() if k!='source'}
      if strip(component['traffic'])!=strip(work['memory_traffic']):
        raise ValueError(f'work evidence {row["name"]} component traffic differs from launches')
      memory=hierarchy_ms(work['memory_traffic'],dram,rates['l2_gbs'])
      for field in ('dram_ms','l2_ms'):equal(field,component[field],memory[field])
      if component['memory_tier']!=memory['memory_tier']:
        raise ValueError(f'work evidence {row["name"]} memory tier differs from derived work/rates')
      memory_ms=memory['memory_ms']
    else:
      if component['bytes']!=work['bytes'] or component['flops']!=work['flops']:
        raise ValueError(f'work evidence {row["name"]} component work differs from launches')
      memory_ms=work['bytes']/(dram*1e6)
    for field,value in (('compute_ms',compute),('memory_ms',memory_ms),('ceiling_ms',max(compute,memory_ms))):
      equal('component '+field,component[field],value)
      totals[field]+=value
  for field,value in totals.items():equal(field,row[field],value)


def verify_routed_work(profile: dict, engine: dict, work: dict, semantic: dict | None = None) -> None:
  from boltbeam.roofline import moe, prefill_work
  matches=[r for r in profile['roles'] if r['role']==work['role'] and
           [r['rows'],r['cols']]==work['shape'] and r['quant']==work['quant']]
  shapes={(r['n_expert'],r['experts_read'],r.get('weight_bytes')) for r in matches}
  if len(shapes)!=1:
    raise ValueError('routed work has no unique model tensor specification')
  experts,top_k,stored=next(iter(shapes))
  if work['top_k']!=top_k:
    raise ValueError('routed top-k differs from model profile')
  if work.get('routing_basis') == 'single_token_topk':
    s=semantic or {}
    inputs=s.get('inputs') or []
    if (work['tokens'] != 1 or s.get('class') != 'MoERunner' or
        s.get('module') != work['source_module'] or
        s.get('operation_id') != work['routing_operation_id'] or
        not inputs or not inputs[0].get('shape') or inputs[0]['shape'][0] != 1 or
        not 0 < top_k <= experts):
      raise ValueError('single-token expert footprint has no matching one-token operation')
    selected=top_k
  else:
    routes=[r for r in engine.get('routing',[]) if r['module']==work['source_module'] and
            r['operation_id']==work['routing_operation_id']]
    if len(routes)!=1:
      raise ValueError('routed work has no unique observed operation')
    selected=moe.selected_experts(routes[0],tokens=work['tokens'],top_k=top_k,experts=experts)
  if selected!=work['selected_experts']:
    raise ValueError('selected expert footprint differs from observed routing')
  n,k=work['shape']
  weight=stored if stored is not None else n*k*prefill_work.elem_bytes(work['quant'])
  derived=prefill_work.gemm(work['tokens']*top_k,n,k,weight*selected,work['activation_formats'])
  for field in ('flops','bytes'):
    if not math.isclose(work[field],derived[field],rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
      raise ValueError(f'routed {field} differs from tensor shapes, formats and observed routing')


def verify(run: Path, capture: dict, *, require_limits: bool = False) -> None:
  proof = capture.get('work_evidence') or {}
  if proof.get('schema') != 'boltbeam.semantic_work.v1':
    raise ValueError('missing semantic work evidence')
  digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
  if proof.get('model_profile_sha256') != digest(run/'model_profile.json'):
    raise ValueError('work evidence model profile differs from this run')
  profile=json.loads((run/'model_profile.json').read_text())
  sources = {}
  for ref in proof.get('sources', []):
    path = (run/ref['file']).resolve()
    if not path.is_relative_to(run.resolve()) or not path.is_file() or digest(path) != ref['sha256']:
      raise ValueError(f'work evidence source missing or changed: {ref["file"]}')
    if path.name in sources:
      raise ValueError('duplicate work evidence source')
    sources[path.name] = json.loads(path.read_text())
  if rate_source:=proof.get('cache_rate_source'):
    from boltbeam.roofline.memory import peak_bandwidth
    if rate_source not in sources:raise ValueError('cache rate source is missing from hashed evidence')
    rate=peak_bandwidth(sources[rate_source])
    for row in capture['rows']:
      recorded=(row.get('memory_rates') or {}).get('l2_gbs')
      if recorded is not None and recorded!=rate:
        raise ValueError('row cache rate differs from independent probe evidence')
    if capture.get('cache') and capture['cache']['gbs']!=rate:
      raise ValueError('capture cache rate differs from independent probe evidence')
  facts = sources.get('semantic_work.json')
  if not facts or not facts.get('launches'):
    raise ValueError('work evidence has no launches')
  if facts['engine'].get('execution_mode') != proof.get('execution_mode'):
    raise ValueError('work evidence execution mode differs')
  launches = facts['launches']
  if facts['engine'].get('decode_steps'):
    from boltbeam.collectors.cuda_memory import decode_horizon,same_capture
    same_capture(facts['engine'],decode_horizon(facts.get('steps',[])))
    same_capture(facts['engine'],capture.get('engine') or {})
    if set(l.get('capture_step') for l in launches)!=set(facts['engine']['decode_steps']):
      raise ValueError('decode horizon launches do not cover every recorded step')
    from boltbeam.collectors.vllm_two_pass import reference_band,host_step
    if capture.get('reference_band')!=reference_band(facts['engine'],sources.get('timing_trace.json') or {}):
      raise ValueError('decode reference band differs from hashed untraced evidence')
    if {k:capture[k] for k in ('host_samples','host_step_ms','timing_window') if k in capture}!=host_step(facts['engine']):raise ValueError('decode host step or window differs from engine seconds')
  if facts['engine'].get('replay') is not None:
    from boltbeam.collectors.replay_semantics import capture_identity
    from boltbeam.collectors.cuda_memory import same_capture
    replay=sources.get('replay_evidence.json')
    if not replay:raise ValueError('replay work has no scheduler and response evidence')
    same_capture(facts['engine'],capture_identity(replay['model'],replay['step'],replay['responses']))
    same_capture(facts['engine'],capture.get('engine') or {})
  if any((l.get('semantic_work') or {}).get('requires_memory_evidence') for l in launches):
    from boltbeam.collectors.cuda_memory import same_capture, bind
    memory = sources.get('memory.json')
    if not memory:
      raise ValueError('work evidence has no memory counters')
    same_capture(facts['engine'], memory['identity'])
    if facts['engine'].get('decode_steps'):
      same_capture(facts['engine'],decode_horizon(memory.get('steps',[])))
    # Rebind a copy and compare against the traffic used to calculate the rows.
    import copy
    rebound = bind(copy.deepcopy(launches), memory, source='verified')
    for original, checked in zip(launches, rebound):
      a = (original.get('semantic_work') or {}).get('memory_traffic')
      b = (checked.get('semantic_work') or {}).get('memory_traffic')
      if a is not None and {k:v for k,v in a.items() if k!='source'} != {k:v for k,v in b.items() if k!='source'}:
        raise ValueError('work evidence memory traffic differs from counters')
  verify_launches(run,profile,capture,facts,require_limits=require_limits)


def verify_launches(run:Path, profile:dict, capture:dict, facts:dict, *, require_limits:bool = False) -> None:
  """Check work and arithmetic; the caller must separately bind source provenance."""
  launches=facts['launches']
  sums = defaultdict(lambda: defaultdict(float))
  work_by_row = defaultdict(list)
  raw_ms = defaultdict(float)
  for launch in launches:
    if work := launch.get('semantic_work'):
      raw_ms[work['name']] += launch['wall_us']/1000
  from boltbeam.collectors.attribution import share_overlap
  shared, overlap = share_overlap(launches)
  if capture.get('overlap_policy')=='explicit subtraction':
    shared=launches
    from boltbeam.roofline.concurrency import require_shared_limit
    require_shared_limit(launches,capture['rows'],capture['bandwidth_gbs'],capture.get('cache'))
    correction=(overlap['union_us']-overlap['sum_us'])/1000
    adjustments=[r for r in capture['rows'] if r.get('what')=='overlap']
    expected=1 if correction < -INTERVAL_ATOL else 0
    if len(adjustments)!=expected or (adjustments and (
        not math.isclose(adjustments[0]['ms'],correction,rel_tol=ACCOUNTING_RTOL,abs_tol=INTERVAL_ATOL) or
        adjustments[0].get('ceiling_ms')!=0 or adjustments[0].get('best_ms')!=0)):
      raise ValueError('overlap subtraction differs from captured kernel intervals')
  for launch in shared:
    w = launch.get('semantic_work')
    if w:
      if w.get('top_k') and not w.get('routing_observed') and w.get('routing_basis') != 'single_token_topk':
        raise ValueError('expert weight footprint is only a lower bound; observed routing is required')
      if w.get('top_k'):
        verify_routed_work(profile,facts['engine'],w,launch.get('semantic'))
      work_by_row[w['name']].append(w)
      totals = sums[w['name']]
      for field in ('flops','bytes'):
        totals[field] += w[field]
      totals['calls'] += 1
      totals['ms'] += launch['wall_us']/1000
  rows = {r['name']:r for r in capture['rows']}
  required=set(sums) | {r['name'] for r in capture['rows'] if r.get('flops') is not None or r.get('bytes') is not None}
  from boltbeam.workflow.tie_out import kind_of
  attention_launches=[l for l in launches if not l.get('semantic_work') and (l.get('semantic_kind')=='attention' or
                     kind_of({'kernel':l.get('key','').split(' grid=')[0]})=='attention')]
  if attention_launches:required.add('attention')
  if require_limits:
    for name in sorted(required):
      limit=rows.get(name,{}).get('ceiling_ms')
      if type(limit) not in (int,float) or not math.isfinite(limit) or limit<0:
        raise ValueError(f'work evidence {name} has no finite nonnegative ceiling; report limit refused')
  if 'attention' in required and 'attention' in rows:
    from boltbeam.collectors.prefill import vllm_formats, vllm_chunk
    from boltbeam.roofline.prefill_work import attention
    engine=facts['engine'];length=engine.get('length')
    depth=engine.get('context') if engine.get('decode_step') is not None else engine.get('context',0)
    if type(length) is not int or length<=0 or type(depth) is not int or depth<0:
      raise ValueError('work evidence attention has no valid recorded length/context')
    formats=vllm_formats({r['quant'] for r in profile.get('roles',[])},engine,require_resolved=True)
    chunks=([1]*length if engine.get('decode_steps') else [length] if engine.get('decode_step') is not None
            else vllm_chunk(length,engine)['widths'])
    derived=attention(profile,length,chunks,formats['kv'],formats['attention_activations'],depth)
    if derived is None:
      raise ValueError('work evidence attention has no model tensor shape')
    sums['attention'].update({field:derived[field] for field in ('flops','bytes')})
    raw_ms['attention']=sum(l['wall_us']/1000 for l in attention_launches)
  for name, totals in sums.items():
    if name not in rows:
      raise ValueError(f'work evidence row absent: {name}')
    limit=rows[name].get('ceiling_ms')
    peaks=(capture.get('uncertainty') or {}).get('peaks') or {}
    paths=rows[name].get('compute_paths') or {}
    uncertainty=max(peaks[p] for p in paths) if rows[name].get('bound')=='compute' and paths and all(p in peaks for p in paths) else 0.
    if type(uncertainty) not in (int,float) or not math.isfinite(uncertainty) or uncertainty<0:
      raise ValueError('work evidence has invalid measured peak uncertainty')
    observed=raw_ms[name]*(1+uncertainty)
    # Compare against full launch durations, before allocating overlap. A rate
    # calibration contradicted by the capture cannot certify its own minimum.
    if limit is not None and limit>observed and not math.isclose(
        limit,observed,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
      raise ValueError(f'work evidence {name} limit {limit:.6f} ms exceeds raw captured duration '
                       f'{raw_ms[name]:.6f} ms; work and rate calibration require requalification')
    for field, expected in totals.items():
      value = rows[name].get(field)
      if not isinstance(value,(int,float)) or not math.isclose(value,expected,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL):
        raise ValueError(f'work evidence {name}.{field} does not sum to launches')
    expected_components=work_by_row[name] if rows[name].get('components') else [
      w for w in work_by_row[name] if w.get('requires_memory_evidence')]
    if rows[name].get('path_status')=='unconfirmed':
      from boltbeam.collectors.prefill_attribution import UNCONFIRMED_SHARE
      row=rows[name];candidates=row.get('candidate_peaks') or {}
      recorded=capture.get('peaks_at') or {}
      unknown=sum(r['ms'] for r in capture['rows'] if r.get('path_status')=='unconfirmed')
      if (set(candidates)!={'fp32_fma','tf32'} or any(recorded.get(p)!=v for p,v in candidates.items()) or
          any(w.get('quant')!='F32' or w.get('role_classes')!=['moe_router'] for w in work_by_row[name]) or
          unknown>UNCONFIRMED_SHARE*capture['wall_ms']):
        raise ValueError('unconfirmed instruction range lacks eligible work, measured rates or small-row coverage')
      dram=row['memory_rates']['dram_gbs']
      bounds=sorted(sum(max(w['flops']/(peak*1e9),w['bytes']/(dram*1e6)) for w in work_by_row[name])
                    for peak in candidates.values())
      if (row.get('ceiling_range_ms')!=bounds or row['ceiling_ms']!=bounds[0] or
          row['peak_tflops']!=max(candidates.values())):
        raise ValueError('unconfirmed instruction range does not derive from work and measured rates')
    if expected_components:
      verify_components(rows[name],expected_components)
    elif require_limits or rows[name].get('memory_ms') is not None:
      row=rows[name]
      bandwidth=(row.get('memory_rates') or {}).get('dram_gbs') or capture.get('bandwidth_gbs')
      if bandwidth is None and (run/'prefill_trace.json').is_file():
        bandwidth=json.loads((run/'prefill_trace.json').read_text()).get('bandwidth_gbs')
      if type(bandwidth) not in (int,float) or not math.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError(f'work evidence {name} has no positive recorded bandwidth')
      memory=totals['bytes']/(bandwidth*1e6)
      compute=totals['flops']/(row['peak_tflops']*1e9) if row.get('peak_tflops') else None
      for field,expected in [('memory_ms',memory),('compute_ms',compute),('ceiling_ms',max(memory,compute or 0.))]:
        actual=row.get(field)
        if expected is not None and (type(actual) not in (int,float) or not math.isfinite(actual) or
            not math.isclose(actual,expected,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL)):
          raise ValueError(f'work evidence {name}.{field} differs from derived work/rates')

  if 'wall_ms' in capture and 'gaps_ms' in capture:
    if any(type(l.get(k)) not in (int,float) or not math.isfinite(l[k]) or l[k]<0
           for l in launches for k in ('start_us','wall_us')):
      raise ValueError('work evidence has invalid captured kernel intervals')
    wall=(max(l['start_us']+l['wall_us'] for l in launches)-min(l['start_us'] for l in launches))/1000
    for key,value in [('wall_ms',wall),('gaps_ms',wall-overlap['union_us']/1000)]:
      if not math.isclose(capture[key],value,rel_tol=ACCOUNTING_RTOL,abs_tol=INTERVAL_ATOL):
        raise ValueError(f'work evidence {key} differs from captured kernel intervals')
