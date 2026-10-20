"""Disk-backed submission validation; exhaust the iterator to prove coverage."""
import collections,csv,gzip,hashlib,json,sqlite3,tempfile
from pathlib import Path
from boltbeam.collectors.cuda_counter_stream import catalog,csvrows,inventory_bindings

def _text(path):
 compressed=path.with_suffix(path.suffix+'.gz')
 if path.exists() and compressed.exists():raise ValueError('ambiguous raw and compressed activity evidence')
 if path.exists():return path.open()
 root=path.parent.parent;archive=json.loads((root/'lossless-archive.json').read_text())
 name=str(path.relative_to(root));rows=[r for r in archive['files'] if r['raw']==name]
 if len(rows)!=1 or rows[0]['compressed']!=str(compressed.relative_to(root)):
  raise ValueError('compressed activity has no unique raw provenance')
 row=rows[0]
 if archive['raw_manifest'].get(name)!=row['raw_sha256']:raise ValueError('raw activity manifest differs')
 with gzip.open(compressed,'rb') as source:
  actual=hashlib.file_digest(source,'sha256').hexdigest();size=source.tell()
 if actual!=row['raw_sha256'] or size!=row['raw_bytes']:raise ValueError('decompressed activity digest differs')
 return gzip.open(compressed,'rt')

def _loads(root):
 result={}
 for line in (root/'compiled/loads.jsonl').read_text().splitlines():
  row=json.loads(line);key=row['context_handle'],row['function']
  if key in result:raise ValueError('duplicate loaded function')
  binary=root/'compiled'/(row['binary_sha256']+'.cubin')
  if hashlib.sha256(binary.read_bytes()).hexdigest()!=row['binary_sha256']:raise ValueError('loaded cubin changed')
  result[key]=row
 return result

def _bind(rows,handles,inventory,modules,loaded):
 byhandle={(h['context_handle'],h['launch_handle']):h for h in handles};annotated=[]
 for row in rows:
  handle=byhandle[row['context_handle'],row['function']]
  record=loaded.get((handle['context_handle'],handle['function_handle']))
  notes=json.loads(row['annotations'])
  if record:
   if record['module_handle']!=handle['module_handle'] or record['entry']!=row['name']:raise ValueError('loader function/module/entry mismatch')
   notes.append('boltbeam-triton:'+json.dumps({k:record[k] for k in ('cache_key','binary_sha256')}))
  annotated.append({**row,'annotations':json.dumps(notes)})
 return inventory_bindings(annotated,handles,inventory,modules)


def _read(folder,cuobjdump,repeat):
 folder=Path(folder);roots=list(folder.glob('identity-*'))
 if len(roots)!=1:raise ValueError('expected one identity worker')
 root=roots[0];loaded=_loads(root);modules=catalog(root/'modules',cuobjdump)
 def rows(p):return csvrows(p.read_bytes())
 creations=rows(root/'timing.created.csv');creation={}
 for ordinal,row in enumerate(creations):
  if int(row['ordinal'])!=ordinal or row['node_id'] in creation:raise ValueError('ambiguous graph construction order')
  creation[row['node_id']]=row
 nodes={};bindings={};graph_masks={};node_bits={}
 for p in (root/'graphs').glob('*.graph.csv'):
  launches=rows(p);prefix=str(p).removesuffix('.graph.csv')
  all_nodes=rows(Path(prefix+'.nodes.csv'));inventory={x['node_id']:x for x in all_nodes}
  if len(inventory)!=len(all_nodes):raise ValueError('duplicate inventoried graph node')
  for node,row in inventory.items():
   observed=creation[node]
   if any(observed[k]!=row[k] for k in ('graph_id','node_handle')):raise ValueError('graph construction identity differs')
  edges=rows(Path(prefix+'.edges.csv'));seen_edges=set()
  for edge in edges:
   source,target=edge['source_node_id'],edge['target_node_id']
   if source not in inventory or target not in inventory or any(inventory[n]['graph_id']!=edge['graph_id'] for n in (source,target)):raise ValueError('graph dependency endpoint differs')
   key=source,target,edge['from_port'],edge['to_port'],edge['type']
   if key in seen_edges:raise ValueError('duplicate graph dependency')
   seen_edges.add(key)
   if int(creation[source]['ordinal'])>=int(creation[target]['ordinal']):raise ValueError('graph dependency contradicts observed construction order')
  if not launches:continue
  bound=_bind(launches,rows(Path(prefix+'.handles.csv')),rows(Path(prefix+'.inventory.csv')),modules,loaded)
  for row,b in zip(launches,bound,strict=True):
   if row['node_id'] in nodes:raise ValueError('duplicate graph node')
   node=row['node_id'];graph=row['graph_id'];mask=graph_masks.get(graph,0)
   nodes[node]=row;bindings[node]=b;node_bits[node]=mask+1;graph_masks[graph]=mask|(mask+1)
 handles=rows(root/'timing.handles.csv');inventory=rows(root/'timing.inventory.csv');eager_binary={};eager_unique={}
 lineage={}
 for row in rows(root/'timing.clones.csv'):
  child,source=row['cloned_node_id'],row['source_node_id']
  if child in lineage and lineage[child]!=source:raise ValueError('ambiguous graph clone')
  lineage[child]=source
 phases={}
 with next((folder/repeat).glob('*.jsonl')).open() as source:
  for line in source:
   row=json.loads(line)
   if 'phase' in row:
    key=row['step'],row['phase']
    if key in phases:raise ValueError('duplicate scheduler phase')
    phases[key]=row
 phase_counts=collections.Counter();exec_graph={};activity_count=0
 windows=rows(root/'timing.activity-window.csv')
 if len(windows)!=1:raise ValueError('expected one activity window')
 window=windows[0]
 if int(window['dropped']):raise ValueError('activity loss')
 with tempfile.TemporaryDirectory(prefix='identity-validation-') as temporary:
  db=sqlite3.connect(str(Path(temporary)/'submissions.sqlite'))
  try:
   db.execute('pragma cache_size=-65536');db.execute('pragma journal_mode=off');db.execute('pragma synchronous=off')
   db.execute('pragma page_size=4096');db.execute('pragma max_page_count=1048576') # 4 GiB explicit disk ceiling at default 4 KiB pages.
   db.execute('create table submissions (correlation text primary key, payload text, seen text, complete integer)')
   with _text(root/'timing.submissions.csv') as source:
    for index,row in enumerate(csv.DictReader(source)):
     if int(row['index'])!=index:raise ValueError('reordered submissions')
     if row['kind']=='kernel':
      # A handle's identity is invariant for this capture; validate every entry name too.
      key=row['context_handle'],row['function'],row['name']
      eager_unique.setdefault(key,row)
     elif row['kind']!='graph':raise ValueError('unsupported submission')
     db.execute('insert into submissions values (?, ?, ?, 0)',(row['correlation'],json.dumps(row),'0'))
   eager_binary={key:b for key,b in zip(eager_unique,_bind(list(eager_unique.values()),handles,inventory,modules,loaded),strict=True)}
   db.commit()
   with _text(root/'timing.activity.csv') as source:
    for activity in csv.DictReader(source):
     activity_count+=1
     if not int(window['start_ns'])<=int(activity['start_ns'])<int(activity['end_ns'])<=int(window['end_ns']):raise ValueError('activity outside window')
     found=db.execute('select payload,seen,complete from submissions where correlation=?',(activity['correlation'],)).fetchone()
     if found is None:raise ValueError('activity has no submission')
     payload,mask,complete=found;submit=json.loads(payload);mask=int(mask,16)
     if complete:raise ValueError('extra activity for completed submission')
     if submit['context']!=activity['context']:raise ValueError('activity context differs')
     notes=json.loads(submit['annotations']);step_notes=[json.loads(n.partition(':')[2]) for n in notes if n.startswith('boltbeam-step:')]
     if len(step_notes)!=1:raise ValueError('missing unique scheduler annotation')
     note=step_notes[0];phase=note['step'],note['phase'];expected=phases[phase]
     if expected['status']!='submitted' or note!={k:v for k,v in expected.items() if k not in ('status','capture_pass')}:raise ValueError('scheduler annotation differs')
     if submit['kind']=='kernel':
      if activity['graph_node']!='0':raise ValueError('eager activity has graph node')
      launch=submit;binary=eager_binary[submit['context_handle'],submit['function'],submit['name']];complete=1
     else:
      current=activity['graph_node'];seen=set()
      while current not in nodes:
       if current in seen:raise ValueError('cyclic clone lineage')
       seen.add(current);current=lineage[current]
      launch=nodes[current];binary=bindings[current];bit=node_bits[current]
      if launch['context_handle']!=submit['context_handle']:raise ValueError('graph node belongs to different context')
      if mask&bit:raise ValueError('duplicate graph execution node')
      mask|=bit;graph=launch['graph_id'];key=submit['graph_exec']
      if key in exec_graph and exec_graph[key]!=graph:raise ValueError('graph executable changed source')
      exec_graph[key]=graph;complete=int(mask==graph_masks[graph])
     if launch['name']!=activity['name'] or any(launch[k]!=activity[k] for stem in ('grid','block') for axis in 'xyz' for k in [stem+'_'+axis]):raise ValueError('actual entry/geometry differs')
     db.execute('update submissions set seen=?,complete=? where correlation=?',(format(mask,'x'),complete,activity['correlation']))
     phase_counts[phase]+=1
     yield {'step':phase[0],'phase':phase[1],'order':[int(submit['index']),int(creation[current]['ordinal']) if submit['kind']=='graph' else 0],
            'entry':activity['name'],'binary_sha256':binary[0],'triton':binary[1].get('boltbeam-triton'),'grid':[int(activity['grid_'+a]) for a in 'xyz'],
            'block':[int(activity['block_'+a]) for a in 'xyz'],'origin_ns':int(window['start_ns']),'start_ns':int(activity['start_ns']),'end_ns':int(activity['end_ns'])}
   if db.execute('select count(*) from submissions where complete=0').fetchone()[0]:raise ValueError('submission activity coverage incomplete')
   if set(phase_counts)!={k for k,v in phases.items() if v.get('scheduled_tokens')}:raise ValueError('scheduler phase coverage differs')
   if activity_count!=int(window['kernel_count']):raise ValueError('activity count mismatch')
  finally:db.close()


class Stream:
 """Validate a frozen capture, then order by observed submission and construction.

 The complete iterator must be consumed before its provenance can be published.
 GPU timestamps and file arrival order never determine semantic launch order.
 """
 def __init__(self,source):
  self.folder=Path(source['folder']);self.repeat=source.get('repeat','0')
  self.cuobjdump=source.get('cuobjdump','cuobjdump');self.count=0;self.complete=False;self.started=False
  raw=Path(source['manifest']).read_bytes();self.manifest_sha256=hashlib.sha256(raw).hexdigest()
  if self.manifest_sha256!=source['manifest_sha256']:raise ValueError('activity manifest digest differs')
  self.manifest=json.loads(raw)
  if not self.manifest:raise ValueError('empty activity manifest')
  for name in self.manifest:
   if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('activity manifest path escapes capture')
  self._check_files()
 def _check_files(self):
  files={str(p.relative_to(self.folder)):p for p in self.folder.rglob('*') if p.is_file()}
  if set(files)!=set(self.manifest):raise ValueError('activity manifest file coverage differs')
  for name,path in files.items():
   with path.open('rb') as source:actual=hashlib.file_digest(source,'sha256').hexdigest()
   if actual!=self.manifest[name]:raise ValueError('activity artifact digest differs: '+name)
 def __iter__(self):
  if self.started:raise ValueError('activity stream cannot be reused')
  self.started=True
  identities={};by_id=[];origin=None;first=last=None;duration_ns=0
  with tempfile.TemporaryDirectory(prefix='ordered-activities-') as temporary:
   db=sqlite3.connect(str(Path(temporary)/'activities.sqlite'))
   try:
    db.execute('pragma page_size=4096');db.execute('pragma cache_size=-32768');db.execute('pragma journal_mode=off');db.execute('pragma synchronous=off');db.execute('pragma max_page_count=1048576')
    db.execute('create table activities (step integer, submission integer, ordinal integer, phase integer, identity integer, start integer, end integer, primary key(step,submission,ordinal)) without rowid')
    for row in _read(self.folder,self.cuobjdump,self.repeat):
     duration_ns+=row['end_ns']-row['start_ns']
     first=row['start_ns'] if first is None else min(first,row['start_ns']);last=row['end_ns'] if last is None else max(last,row['end_ns'])
     value=[row[k] for k in ('entry','binary_sha256','grid','block','triton')];key=json.dumps(value,separators=(',',':'))
     if key not in identities:
      if len(identities)>=100000:raise ValueError('activity identity budget exceeded')
      identities[key]=len(by_id);by_id.append(value)
     if row['phase'] not in ('execute','sample'):raise ValueError('unsupported scheduler phase')
     if origin is None:origin=row['origin_ns']
     if origin!=row['origin_ns']:raise ValueError('activity clock origin changed')
     db.execute('insert into activities values (?,?,?,?,?,?,?)',(row['step'],*row['order'],int(row['phase']=='sample'),identities[key],row['start_ns'],row['end_ns']));self.count+=1
    db.commit();returned=0
    for step,submission,ordinal,phase,key,start,end in db.execute('select * from activities order by step,submission,ordinal'):
     entry,binary,grid,block,triton=by_id[key];returned+=1
     yield {'step':step,'phase':('execute','sample')[phase],'launch_order':[submission,ordinal],
      'raw_entry':entry,'native_binary_sha256':binary,'grid':grid,'block':block,
      'start_us':(start-origin)/1000,'wall_us':(end-start)/1000,**({'triton':triton} if triton else {})}
    if not returned or returned!=self.count:raise ValueError('ordered activity coverage differs')
    self._check_files();self.kernel_window_ns=[first,last];self.kernel_ms=duration_ns/1e6;self.complete=True
   finally:db.close()
 def provenance(self):
  if not self.complete:raise ValueError('activity stream has not completed validation')
  return {'kernels':self.count,'manifest_sha256':self.manifest_sha256,'kernel_window_ns':self.kernel_window_ns,
   'kernel_ms':self.kernel_ms,'resource_artifacts':{k:v for k,v in self.manifest.items() if Path(k).name.startswith('resource-window.')},
   'scope':'executed binary, entry, geometry, scheduler phases and observed submission/construction order; activity timestamps only, no timing or memory qualification'}
