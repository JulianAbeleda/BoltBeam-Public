"""Join engine NVTX module ranges to GPU launches through CUPTI correlation IDs.

GPU timestamp containment is not attribution: asynchronous launches can execute
after a CPU range closes. Only the launching CUDA API's CPU timestamp/thread
selects a module. Ambiguous or missing correlations are left unassigned.
"""
from __future__ import annotations

import collections
import bisect
import json
import sqlite3
from functools import lru_cache
from pathlib import Path


def read(path: Path) -> list[dict]:
  return list(iter_read(path))


def join_step(labels: list[dict], timing: list[dict], step: int) -> list[dict]:
  """Transfer labels only across identical ordered launch sequences.

  Keep every timestamp, graph node and correlation from the production pass.
  Geometry is part of the identity: matching names alone cannot establish that
  compilation selected the same specialization and amount of work.
  """
  from itertools import zip_longest
  joined=[]
  for position,(label,timed) in enumerate(zip_longest(labels,timing)):
    left=label.get('key') if label is not None else '<end>'
    right=timed.get('key') if timed is not None else '<end>'
    if label is None or timed is None or not left or left!=right:
      raise ValueError(f'capture sequence divergence at step {step}, kernel {position}: '
                       f'labeling={left!r}, timing={right!r}')
    if (label.get('raw_entry') or timed.get('raw_entry')) and label.get('raw_entry')!=timed.get('raw_entry'):
      raise ValueError(f'raw kernel entry divergence at step {step}, kernel {position}')
    if label.get('triton') or timed.get('triton'):
      if (not label.get('triton') or not timed.get('triton') or
          any(not label['triton'].get(k) or label['triton'][k]!=timed['triton'].get(k) for k in ('cache_key','binary_sha256'))):
        raise ValueError(f'compiled Triton binary divergence at step {step}, kernel {position}')
    joined.append({**timed,'semantic':label.get('semantic'),'semantic_range':label.get('semantic_range'),
                   **({'triton':label['triton']} if label.get('triton') else {}),
                   'label_position':position,'capture_step':step})
  if not joined:raise ValueError(f'capture step {step} has no kernels in either pass')
  return joined


def submission_order(launches: list[dict]) -> list[dict]:
  """Order by recorded CPU submission and original graph-node construction.

  GPU start order is not program order when independent streams overlap.
  Missing or ambiguous submission evidence cannot be repaired by sorting names.
  """
  keys=[tuple(l['launch_order']) if l.get('launch_order') is not None else None for l in launches]
  if None in keys or len(set(keys))!=len(keys):
    raise ValueError('capture has missing or ambiguous kernel submission order')
  return sorted(launches,key=lambda l:l['launch_order'])


def iter_read(path: Path, *, step_spans=None):
  with Capture(path) as capture:
    yield from capture.iter_read(step_spans)


class Capture:
  """One open trace, with immutable names and graph lineage shared across steps."""
  def __init__(self,path, *, lineage=None):
    self.db=sqlite3.connect(path);self.db.row_factory=sqlite3.Row;self.lineage=lineage

  def __enter__(self):
    db=self.db
    try:
      self.tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
      if 'NVTX_EVENTS' not in self.tables:raise ValueError('capture has no NVTX module evidence')
      self.strings=dict(db.execute('select id,value from StringIds'))
      if self.lineage is not None:
        files=lambda c:set(map(tuple,c.execute('select filename,globalPid from ANALYSIS_FILE')))
        if not files(db)&files(self.lineage.db):
          raise ValueError('graph lineage belongs to a different profiler session')
        clock=lambda c:c.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
        self.lineage_offset=clock(self.lineage.db)-clock(db)
        if self.lineage_offset>0:raise ValueError('graph lineage recording follows its replay')
      self.nodes=collections.defaultdict(list)
      if 'CUDA_GRAPH_NODE_EVENTS' in self.tables:
        for r in db.execute('select * from CUDA_GRAPH_NODE_EVENTS'):
          if r['globalTid'] is not None:
            self.nodes[r['globalTid'] >> 24,r['graphNodeId']].append(dict(r))
      self.origins=lru_cache(None)(self._origins)
      return self
    except BaseException:
      db.close();raise

  def __exit__(self,*exc):
    self.origins.cache_clear();self.db.close()

  def _origins(self,pid,node,visited=frozenset()):
    key=(pid,node)
    if key in visited:return ()
    records=self.nodes.get(key,());clones=[r for r in records if r['originalGraphNodeId'] is not None]
    if not records and self.lineage is not None:
      return tuple((tid,start+self.lineage_offset,end+self.lineage_offset)
                   for tid,start,end in self.lineage.origins(pid,node))
    return tuple(origin for r in clones or records for origin in
      (self.origins(pid,r['originalGraphNodeId'],visited|{key}) if r['originalGraphNodeId'] is not None
       else [(r['globalTid'],r['start'],r['end'])]))

  def iter_read(self,step_spans=None):
    db,tables,strings=self.db,self.tables,self.strings
    ranges=collections.defaultdict(list)
    bounds=(min(r['start'] for r,_ in step_spans),max(r['end'] for r,_ in step_spans)) if step_spans else None
    sources=[(self,0,bounds)]
    if self.lineage is not None:sources.append((self.lineage,self.lineage_offset,None))
    for reader,offset,window in sources:
      nvtx=(reader.db.execute('select * from NVTX_EVENTS where start>=? and start<=?',window)
            if window else reader.db.execute('select * from NVTX_EVENTS'))
      for row in nvtx:
        r=dict(row);text=r.get('text') or reader.strings.get(r.get('textId'),'')
        for prefix,kind in (('boltbeam:','semantic'),('boltbeam-triton:','triton')):
          if text.startswith(prefix) and r.get('end') is not None:
            ranges[r['globalTid'],kind].append((r['start']+offset,r['end']+offset,json.loads(text[len(prefix):])))
    index={}
    for tid,spans in ranges.items():
      spans.sort(key=lambda x:x[0])
      ends=[];latest=float('-inf')
      for span in spans:
        latest=max(latest,span[1]);ends.append(latest)
      index[tid]=([x[0] for x in spans],ends,spans)

    def enclosing(tid, start, end, kind):
      if (tid,kind) not in index:
        return None
      starts,ends,spans=index[tid,kind]
      i=bisect.bisect_right(starts,start)-1
      scope=[]
      while i>=0 and ends[i]>=end:
        if spans[i][1]>=end:
          scope.append(spans[i])
        i-=1
      return min(scope,key=lambda x:x[1]-x[0]) if scope else None

    def scopes(tid,start,end):
      return {kind:enclosing(tid,start,end,kind) for kind in ('semantic','triton')}

    api = collections.defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
      if table not in tables:
        continue
      calls = db.execute(f'select * from {table} where start>=? and start<=?',bounds) if bounds else db.execute(f'select * from {table}')
      for r in calls:
        inner = scopes(r['globalTid'],r['start'],r['end'])
        api[r['correlationId']].append((r['globalTid'] >> 24, inner, r['start']))
    phases=collections.defaultdict(set)
    if step_spans:
      selected={(span['globalTid'] >> 24,r[0],label.get('phase')) for span,label in step_spans for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') if table in tables
           for r in db.execute(f'select correlationId from {table} where start>=? and start<=? and end<=? and globalTid=?',
                               (span['start'],span['end'],span['end'],span['globalTid']))}
      for pid,correlation,phase in selected:phases[pid,correlation].add(phase)
      ids=set(phases)
      db.execute('create temp table if not exists selected_calls(pid integer,correlation integer,primary key(pid,correlation))')
      db.execute('delete from selected_calls')
      db.executemany('insert into selected_calls values(?,?)',ids)
      kernels=db.execute('select k.* from selected_calls s cross join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=s.correlation and (k.globalPid >> 24)=s.pid order by k.start')
    else:kernels=db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')
    for r in kernels:
      pid=r['globalPid'] >> 24
      node=r['graphNodeId'] if 'graphNodeId' in r.keys() else None
      calls=[(scope,start) for p,scope,start in api[r['correlationId']] if p==pid]
      origins=[(scopes(tid,start,end),start) for tid,start,end in self.origins(pid,node)] if node is not None else calls
      api_times={start for _,start in calls};node_times={start for _,start in origins}
      order=([next(iter(api_times)),next(iter(node_times)) if node is not None else 0]
             if len(api_times)==1 and (node is None or len(node_times)==1) else None)
      # Runtime and driver can both describe the same launch; identical evidence is one match.
      found={}
      for kind in ('semantic','triton'):
        unique={json.dumps(scope[kind],sort_keys=True):scope[kind] for scope,_ in origins if scope[kind] is not None}
        found[kind]=next(iter(unique.values())) if len(unique)==1 else None
      info=found['semantic'];triton=found['triton']
      grid = 'x'.join(str(r['grid'+a]) for a in 'XYZ')
      block = 'x'.join(str(r['block'+a]) for a in 'XYZ')
      yield {'key': f"{strings[r['demangledName']]} grid={grid} block={block}",
                  **({'raw_entry':strings[r['mangledName']]} if 'mangledName' in r.keys() and r['mangledName'] is not None else {}),
                  'start_us': r['start']/1000, 'wall_us': (r['end']-r['start'])/1000,
                  'stream': str(r['streamId']), 'correlation_id': r['correlationId'],
                  'process_id':pid,
                  'phase':next(iter(phases[pid,r['correlationId']])) if len(phases[pid,r['correlationId']])==1 else None,
                  'graph_node_id':node,
                  'launch_order':order,
                  **({'triton':triton[2]} if triton else {}),
                  'semantic': info[2] if info else None,
                  'semantic_range': [info[0], info[1]] if info else None}


def validate_step(spans, phases=None):
  labels=[label for _,label in spans]
  found=[label.get('phase') for label in labels]
  identities={json.dumps({k:v for k,v in label.items() if k!='phase'},sort_keys=True) for label in labels}
  if (not spans or len(found)!=len(set(found)) or len(identities)!=1 or
      (len(spans)>1 and not set(found)<= {'execute','sample'}) or
      (phases is not None and set(found)!=set(phases))):
    raise ValueError('capture has no unique recorded engine step with the required phases')
  processes={span['globalTid'] >> 24 for span,_ in spans}
  if len(processes)!=1:
    raise ValueError('engine step phases belong to different processes')
  return processes.pop()


def step_index(path: Path) -> dict:
  """Index recorded step spans once; kernel reads can then stay bounded per step."""
  groups=collections.defaultdict(list)
  with sqlite3.connect(path) as db:
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table,columns in [('NVTX_EVENTS','start'),('CUPTI_ACTIVITY_KIND_KERNEL','correlationId,globalPid'),
                          ('CUPTI_ACTIVITY_KIND_RUNTIME','start'),('CUPTI_ACTIVITY_KIND_DRIVER','start')]:
      if table in tables:db.execute(f'create index if not exists boltbeam_{table} on {table}({columns})')
    strings=dict(db.execute('select id,value from StringIds'))
    for row in db.execute('select * from NVTX_EVENTS'):
      r=dict(row);text=r.get('text') or strings.get(r.get('textId'),'')
      if not text.startswith('boltbeam-step:') or r.get('end') is None:continue
      label=json.loads(text[len('boltbeam-step:'):])
      groups[(label.get('capture_pass'),label['step'])].append((r,label))
  return dict(groups)


def read_step(path: Path, spans, *, phases=None, allow_empty=False) -> list[dict]:
  validate_step(spans,phases)
  rows=list(path.iter_read(spans) if isinstance(path,Capture) else iter_read(path,step_spans=spans))
  if not rows and not allow_empty:raise ValueError('recorded engine step has no correlated GPU launches')
  return rows
