"""Temporal PM demand for the same completed native replay capture.

Device traffic includes inter-request/background/control traffic. Sample bins
are never assigned causally to individual kernels. Every sample must be valid,
including startup samples; malformed bins are not silently discarded.
"""
import csv,hashlib,json,math
from pathlib import Path

METRICS=('dram_read_bytes','dram_write_bytes','l1tex_l2_sectors')


def read_window(folder,provenance, *, diagnostic=False):
  roots=list(Path(folder).glob('identity-*'))
  if len(roots)!=1:raise ValueError('resource capture needs one identity worker')
  root=roots[0]
  paths=[root/'resource-window.window.json',root/'resource-window.csv']
  prefix_paths=[root/('resource-window'+suffix) for suffix in ('.early.csv','.early.json','.early.counters','.clocks.jsonl')]
  has_prefix=any(p.exists() for p in prefix_paths)
  if has_prefix:paths+=prefix_paths
  def verify():
    for path in paths:
      name=str(path.relative_to(folder));expected=provenance.get('resource_artifacts',{}).get(name)
      if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('resource artifact differs from completed native manifest')
  verify();window=json.loads(paths[0].read_text())
  start,end=window['work_start_ns'],window['work_end_ns']
  first,last=provenance['kernel_window_ns']
  if not start<=first<last<=end:raise ValueError('resource window does not contain all application kernels')
  if window['overflow']:raise ValueError('resource sampler overflow')
  if ('populated_samples' in window and window['populated_samples']!=window['completed_samples']) or ('decode_stop_reason' in window and window['decode_stop_reason']!='END_OF_RECORDS'):
    raise ValueError('incomplete resource decode')
  complete=window.get('populated_samples')==window['completed_samples'] and window.get('decode_stop_reason')=='END_OF_RECORDS'
  if not complete and not diagnostic:raise ValueError('resource decode completeness is not recorded')
  prefix=None
  if has_prefix:
    from boltbeam.collectors.replay_resource_prefix import completed_prefix
    rows,prefix=completed_prefix(root,window,METRICS)
    window={**window,'first_sample_start_ns':rows[0]['start_ns']}
  else:
    with (root/'resource-window.csv').open() as stream:rows=list(csv.DictReader(stream))
  low={k:0 for k in METRICS};high=low.copy();outside=low.copy();previous=None;count=edges=0
  for index,row in enumerate(rows):
      a,b=int(row['start_ns']),int(row['end_ns'])
      if int(row['index'])!=index or a>=b or (previous is not None and previous!=a):
        raise ValueError('invalid resource sample interval sequence')
      if index==0 and (a!=window['first_sample_start_ns'] or a>start):
        raise ValueError('resource samples do not bracket replay start')
      previous=b;count+=1;inside=start<=a and b<=end;overlap=a<end and b>start
      edges+=int(overlap and not inside)
      for key in METRICS:
        value=float(row[key])
        if not math.isfinite(value) or value<0 or not value.is_integer():raise ValueError('invalid resource counter')
        if inside:low[key]+=int(value)
        if overlap:high[key]+=int(value)
        else:outside[key]+=int(value)
  if not count or count!=window['completed_samples'] or previous!=window['last_sample_end_ns'] or previous<end:
    raise ValueError('incomplete resource sample coverage')
  verify()
  return {'status':'recorded_decode_complete' if complete else 'diagnostic_decode_metadata_unavailable','window_ns':[start,end],'window_ms':(end-start)/1e6,'samples':count,'boundary_samples':edges,
          **({'completed_prefix':prefix} if prefix is not None else {}),
          'contained':low,'intersecting':high,'outside_window':outside,
          'native_manifest_sha256':provenance['manifest_sha256'],
          'scope':'device traffic within the worker replay interval, including inter-request/background/control demand; no per-kernel DRAM attribution'}
