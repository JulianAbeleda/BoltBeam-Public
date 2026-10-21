"""Whole replay empirical resource estimate, with independent admission gates."""
import hashlib,json,math,statistics
from pathlib import Path
from boltbeam.roofline.resources import aggregate
from boltbeam.math.numerics import ACCOUNTING_RTOL,ACCOUNTING_ATOL


def _clock_scope(record):
  """Bind a declared regime to retained complete observation summaries, not a claim of continuous lock."""
  if 'clock_scope' not in record:return None
  scope=record['clock_scope']
  def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
  try:
    requested=scope['requested_sm_mhz'];observations=scope['observations'];evidence=scope['evidence']
    if type(requested) is not int or requested<=0 or not observations:raise ValueError('invalid requested regime')
    if not all(digest(evidence[key]) for key in ('protocol_sha256','raw_manifest_sha256','apply_sha256')):
      raise ValueError('clock evidence hashes unavailable')
    names=set()
    for row in observations:
      values=[row['sm_mhz'][key] for key in ('min','mean','max')]
      if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in values):
        raise ValueError('invalid observed clock')
      if values!=sorted(values) or any(abs(v/requested-1)>.02 for v in values[1:]):
        raise ValueError('observed clock regime differs')
      if row['complete'] is not True or type(row['samples']) is not int or row['samples']<=0 or not digest(row['source_sha256']):
        raise ValueError('incomplete clock observations')
      name=row['name']
      if not isinstance(name,str) or not name or name in names:raise ValueError('invalid clock observation name')
      names.add(name)
  except (KeyError,TypeError,ValueError) as exc:
    raise ValueError('invalid resource clock scope or observation evidence') from exc
  return {'requested_sm_mhz':requested}


def clock_scope_from_capture(source,activities):
  """Authenticate the declaration, then recompute observations from the completed capture's raw telemetry."""
  def checked(ref):
    raw=Path(ref['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('resource clock evidence digest differs')
    return json.loads(raw)
  proof=checked(source);scope=proof['clock_scope'];_clock_scope(proof)
  if scope['evidence']['raw_manifest_sha256']!=activities.provenance()['manifest_sha256']:
    raise ValueError('resource clock capture provenance differs')
  protocol,apply=checked(proof['protocol']),checked(proof['apply'])
  if any(scope['evidence'][key+'_sha256']!=proof[key]['sha256'] for key in ('protocol','apply')):
    raise ValueError('resource clock control provenance differs')
  requested=scope['requested_sm_mhz'];argv=apply['argv']
  if protocol['clock_policy']['requested_sm_mhz']!=requested or apply['exit_code']!=0 or not any(
      Path(arg).name=='nvidia-smi' for arg in argv) or argv[-2:]!=['-lgc',f'{requested},{requested}']:
    raise ValueError('resource clock application differs from requested setup')
  def retained(name):
    path=activities.folder/name;raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=activities.manifest.get(name):
      raise ValueError('resource clock observation differs from completed capture')
    return json.loads(raw)
  window=retained('resource-client-window.json');start,end=window['first_send'],window['last_token']
  if not all(isinstance(t,(int,float)) and math.isfinite(t) for t in (start,end)) or not start<end:
    raise ValueError('invalid resource clock observation window')
  rows=retained('telemetry.json')['samples']
  if any(any(type(row[key]) not in (int,float) or not math.isfinite(row[key]) for key in ('t','sm_mhz')) or row['sm_mhz']<=0
         or type(row['reasons']) is not int or row['reasons']<0 for row in rows):
    raise ValueError('invalid retained resource clock samples')
  rows=[row for row in rows if start<=row['t']<=end];stamps=[start,*[row['t'] for row in rows],end]
  if not rows or any(not 0<=b-a<=1 for a,b in zip(stamps,stamps[1:])):
    raise ValueError('incomplete resource clock observation coverage')
  values=[row['sm_mhz'] for row in rows if not row['reasons']&1]
  if not values:raise ValueError('no active resource clock observations')
  observed={'min':min(values),'mean':statistics.mean(values),'max':max(values)}
  claims=scope['observations']
  if len(claims)!=1 or claims[0]['samples']!=len(values) or claims[0]['source_sha256']!=activities.manifest['telemetry.json'] or any(
      not math.isclose(claims[0]['sm_mhz'][key],value,rel_tol=ACCOUNTING_RTOL,abs_tol=ACCOUNTING_ATOL) for key,value in observed.items()):
    raise ValueError('resource clock claim differs from retained observations')
  return scope


def estimate(work,window,calibration,timing):
  missing=[]
  if window['status']!='recorded_decode_complete':missing.append('complete temporal memory evidence unavailable')
  if calibration is None:missing.append('independent resource calibration unavailable')
  if missing:return {'status':'unavailable','reasons':missing}
  if calibration['kind']!='empirical_resource_calibration':raise ValueError('unsupported resource calibration')
  if window['native_manifest_sha256']!=work['native_manifest_sha256']:raise ValueError('resource work/window provenance differs')
  if not calibration.get('gpu_identity') or calibration['gpu_identity']!=work.get('gpu_identity'):
    raise ValueError('resource calibration device/driver identity differs')
  clock=_clock_scope(work)
  if clock!=_clock_scope(calibration):raise ValueError('resource calibration clock regime differs')
  bindings={}
  for row in calibration['bindings']:
    key=row['binary_sha256'],row['entry']
    if key in bindings:raise ValueError('duplicate resource execution binding')
    bindings[key]=row['path']
  terms=[];covered=0
  for group in work['groups']:
    path=bindings.get((group['binary_sha256'],group['entry']))
    if path is None:missing.append('execution path unavailable: '+group['entry']);continue
    if path=='memory_only':
      if group['flops']!=0:raise ValueError('nonzero work assigned to memory-only path')
    else:
      rate=calibration['compute'].get(path)
      if rate is None:missing.append('compute calibration unavailable: '+path);continue
      if rate['unit']!='FLOPs/ms':raise ValueError('compute calibration unit differs')
      terms.append({'resource':rate['resource'],'demand':[group['flops']]*2,'rate':rate['rate_per_ms']})
    covered+=group['calls']
  low,high=window['contained'],window['intersecting']
  for name,keys,unit in [('dram',('dram_read_bytes','dram_write_bytes'),'bytes/ms'),('l2',('l1tex_l2_sectors',),'sectors/ms')]:
    rate=calibration['memory'].get(name)
    if rate is None:missing.append('memory calibration unavailable: '+name);continue
    if rate['unit']!=unit:raise ValueError('memory calibration unit differs: '+name)
    terms.append({'resource':name,'demand':[sum(low[k] for k in keys),sum(high[k] for k in keys)],'rate':rate['rate_per_ms']})
  reads,writes=high['dram_read_bytes'],high['dram_write_bytes']
  if reads+writes<=0 or low['dram_read_bytes']+low['dram_write_bytes']<=0:raise ValueError('empty contained DRAM demand')
  # Edge bins also bound the read/write fraction; do not select a convenient midpoint.
  fraction=[low['dram_write_bytes']/(high['dram_read_bytes']+low['dram_write_bytes']),
            high['dram_write_bytes']/(low['dram_read_bytes']+high['dram_write_bytes'])]
  span=calibration['dram_write_fraction_span']
  if not 0<=span[0]<=span[1]<=1:raise ValueError('invalid calibrated traffic mix span')
  if fraction[0]<span[0] or fraction[1]>span[1]:missing.append('DRAM write fraction outside calibrated traffic mix span')
  if covered!=work['totals']['kernels']:missing.append('execution-resource coverage incomplete')
  if missing:return {'status':'unavailable','reasons':missing,'covered_launches':covered,'total_launches':work['totals']['kernels']}
  result=aggregate(terms);wall=window['window_ms']
  if not math.isfinite(wall) or wall<=0:raise ValueError('invalid resource elapsed window')
  percentages=[result[k]/wall*100 for k in ('floor_low_ms','floor_high_ms')]
  if not all(math.isfinite(v) for v in percentages):raise ValueError('resource percentage overflow')
  return {'status':'qualified_empirical_estimate' if timing['status']=='passed' else 'timing_unqualified_empirical_estimate',
          **result,'window_ms':wall,'estimate_pct':percentages,
          'estimate_exceeds_observed_window':result['floor_high_ms']>wall,
          'covered_launches':covered,'total_launches':work['totals']['kernels'],'dram_write_fraction':fraction,
          'native_manifest_sha256':work['native_manifest_sha256'],'calibration_source_sha256':calibration['source_sha256'],
          **({'clock_scope':clock,'clock_provenance':{'work':work['clock_scope']['evidence'],
              'calibration':calibration['clock_scope']['evidence']}} if clock else {}),
          'scope':'required semantic arithmetic and temporal device memory demand; empirical attainable-rate sensitivity, not a guaranteed physical floor or confidence interval; per-operation DRAM unavailable'}
