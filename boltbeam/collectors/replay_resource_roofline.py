"""Whole replay empirical resource estimate, with independent admission gates."""
import math
from boltbeam.roofline.resources import aggregate


def estimate(work,window,calibration,timing):
  missing=[]
  if window['status']!='recorded_decode_complete':missing.append('complete temporal memory evidence unavailable')
  if calibration is None:missing.append('independent resource calibration unavailable')
  if missing:return {'status':'unavailable','reasons':missing}
  if calibration['kind']!='empirical_resource_calibration':raise ValueError('unsupported resource calibration')
  if window['native_manifest_sha256']!=work['native_manifest_sha256']:raise ValueError('resource work/window provenance differs')
  if not calibration.get('gpu_identity') or calibration['gpu_identity']!=work.get('gpu_identity'):
    raise ValueError('resource calibration device/driver identity differs')
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
          'scope':'required semantic arithmetic and temporal device memory demand; empirical attainable-rate sensitivity, not a guaranteed physical floor or confidence interval; per-operation DRAM unavailable'}
