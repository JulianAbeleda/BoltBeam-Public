"""Read lossless CUPTI range chunks and bind actual loaded module inventories.

This establishes observed identities and raw counters only. It does not qualify
traffic scope or transfer counter-pass timing to the production trace. A caller
must establish equal replay inputs/settings and exhaust the stream before
publishing any whole-session result.
"""
import csv,gzip,hashlib,io,json,math,re,subprocess
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from boltbeam.collectors.attention_auxiliary import geometry

SUFFIXES=('.launches.csv','.csv','.counters','.config','.inventory.csv','.handles.csv','.validation.json')
METRICS=('dram_read_bytes','dram_write_bytes','l1tex_l2_sectors')
NOTES=('boltbeam-step','boltbeam-triton','boltbeam-sampling-control')


def digest(data):return hashlib.sha256(data).hexdigest()
def csvrows(data):return list(csv.DictReader(io.StringIO(data.decode())))


def catalog(folder,cuobjdump):
  """Whole cubin digests keyed by context and complete CUDA entry inventories."""
  result=defaultdict(set)
  for row in csvrows((folder/'modules.csv').read_bytes()):
    name=row['file']
    if Path(name).name!=name:raise ValueError('module file is not a basename')
    path=folder/name;raw=path.read_bytes()
    if len(raw)!=int(row['cubin_size']):raise ValueError('captured cubin size differs from module record')
    symbols=subprocess.check_output([cuobjdump,'--dump-elf-symbols',str(path)],text=True)
    names=frozenset(m[1] for line in symbols.splitlines() if (m:=re.fullmatch(
      r'STT_FUNC\s+STB_(?:GLOBAL|LOCAL)\s+STO_ENTRY\s+(\S+)\s*',line.strip())))
    if names:result[row['context_handle'],names].add(digest(raw))
  if not result:raise ValueError('no captured CUDA entry inventories')
  return result


def annotations(row):
  result={}
  for text in json.loads(row['annotations']):
    key,sep,value=text.partition(':')
    if key not in NOTES:continue
    if not sep or key in result:raise ValueError('ambiguous counter launch annotation')
    result[key]=json.loads(value)
  return result


def inventory_bindings(launches,handles,inventory,modules):
  """Resolve actual launch handles, never an entry-name/proximity guess."""
  byhandle={(h['context_handle'],h['launch_handle']):h for h in handles}
  if len(byhandle)!=len(handles):raise ValueError('duplicate CUDA launch handle')
  entries=defaultdict(dict)
  for row in inventory:
    key=row['context_handle'],row['module_handle'];functions=entries[key]
    if row['function_handle'] in functions:raise ValueError('duplicate CUDA module function handle')
    functions[row['function_handle']]=row['name']
  notes=list(map(annotations,launches));constraints={}
  for launch,note in zip(launches,notes):
    h=byhandle[launch['context_handle'],launch['function']];key=h['context_handle'],h['module_handle']
    if entries[key].get(h['function_handle'])!=launch['name']:raise ValueError('CUDA launch handle entry mismatch')
    if t:=note.get('boltbeam-triton'):
      value=t['binary_sha256']
      if key in constraints and constraints[key]!=value:raise ValueError('inconsistent compiled module digests')
      constraints[key]=value
  bound={}
  for key,functions in entries.items():
    candidates=set(modules.get((key[0],frozenset(functions.values())),()))
    if key in constraints:candidates&={constraints[key]}
    if len(candidates)!=1:raise ValueError('CUDA module has no unique captured binary inventory')
    bound[key]=next(iter(candidates))
  result=[]
  for launch,note in zip(launches,notes):
    h=byhandle[launch['context_handle'],launch['function']]
    result.append((bound[h['context_handle'],h['module_handle']],note))
  return result


def chunk_rows(data,chunk,modules,collector_sha256):
  validation=json.loads(data['.validation.json']);step=chunk['step']
  if (validation.get('status')!='captured' or validation.get('step')!=step or
      validation.get('collector_sha256')!=collector_sha256):raise ValueError('counter collector or validation mismatch')
  launches=csvrows(data['.launches.csv']);counters=csvrows(data['.csv'])
  if not launches or len(launches)!=len(counters) or len(launches)!=chunk['launches'] or len(launches)!=validation['launches']:
    raise ValueError('counter chunk launch coverage differs')
  bindings=inventory_bindings(launches,csvrows(data['.handles.csv']),csvrows(data['.inventory.csv']),modules)
  result=[]
  for i,(launch,counter,(binary,note)) in enumerate(zip(launches,counters,bindings,strict=True)):
    if not launch['range_index']==counter['range_index']==counter['range_name']==str(i):
      raise ValueError('counter range order differs')
    phase=note.get('boltbeam-step',{})
    if phase.get('step')!=step or phase.get('capture_pass')!='label' or phase.get('phase') not in ('execute','sample'):
      raise ValueError('counter launch lacks matching scheduler phase')
    axes={k:[int(launch[k+'_'+a]) for a in 'xyz'] for k in ('grid','block')}
    if any(v<=0 for values in axes.values() for v in values):raise ValueError('invalid counter launch geometry')
    memory={k:float(counter[k]) for k in METRICS}
    if any(not math.isfinite(v) or v<0 for v in memory.values()):raise ValueError('invalid raw counter value')
    triton=note.get('boltbeam-triton');control=note.get('boltbeam-sampling-control')
    if triton and (triton['binary_sha256']!=binary or triton['grid']!=axes['grid']):
      raise ValueError('compiled counter launch identity mismatch')
    if control and (not triton or any(control.get(k)!=triton.get(k) for k in ('cache_key','binary_sha256','grid'))):
      raise ValueError('sampling controls belong to a different compiled launch')
    result.append({'entry':launch['name'],'binary_sha256':binary,**axes,'phase':phase,
      'triton':triton,'sampling_control':control,'observed_memory':memory})
  return result


class Stream:
  def __init__(self,root,expected_steps,collector_sha256,*,cuobjdump='cuobjdump'):
    self.root=Path(root);self.expected_steps=tuple(expected_steps);self.collector_sha256=collector_sha256
    if (not self.expected_steps or any(type(s) is not int or s<0 for s in self.expected_steps) or
        list(self.expected_steps)!=sorted(set(self.expected_steps))):raise ValueError('invalid expected counter steps')
    if not re.fullmatch('[0-9a-f]{64}',collector_sha256):raise ValueError('missing reviewed collector digest')
    self.modules=catalog(self.root/'modules',cuobjdump)
    self.manifest_sha256=digest((self.root/'chunks.jsonl').read_bytes())
  def __iter__(self):
    path=self.root/'chunks.jsonl'
    if digest(path.read_bytes())!=self.manifest_sha256:raise ValueError('counter manifest changed')
    with path.open() as stream:
      for line,step in zip_longest(stream,self.expected_steps):
        if line is None or step is None:raise ValueError('counter stream has missing or extra scheduler steps')
        chunk=json.loads(line)
        if type(chunk['step']) is not int or chunk['step']!=step:raise ValueError('counter scheduler order differs')
        if set(chunk['artifacts'])!=set(SUFFIXES):raise ValueError('incomplete raw counter artifacts')
        data={}
        for suffix in SUFFIXES:
          facts=chunk['artifacts'][suffix];packed=(self.root/f'step-{step}{suffix}.gz').read_bytes()
          if len(packed)!=facts['compressed_bytes'] or digest(packed)!=facts['compressed_sha256']:
            raise ValueError('compressed counter artifact integrity mismatch')
          raw=gzip.decompress(packed)
          if len(raw)!=facts['raw_bytes'] or digest(raw)!=facts['raw_sha256']:
            raise ValueError('raw counter artifact integrity mismatch')
          data[suffix]=raw
        yield {'step':step,'rows':chunk_rows(data,chunk,self.modules,self.collector_sha256)}
    if digest(path.read_bytes())!=self.manifest_sha256:raise ValueError('counter manifest changed during review')


def bind_launches(launches,rows):
  """Transfer identities/controls after caller checks equal replay work/settings.

  Raw metric scope remains explicit and unqualified; no memory roofline is
  admitted here. Missing raw symbols cannot be repaired by demangling.
  """
  if len(launches)!=len(rows):raise ValueError('counter/timing launch counts differ')
  out=[]
  for launch,row in zip(launches,rows):
    grid,block=geometry(launch)
    if not launch.get('raw_entry') or launch['raw_entry']!=row['entry'] or list(grid)!=row['grid'] or list(block)!=row['block']:
      raise ValueError('counter/timing raw entry or geometry differs')
    a,b=launch.get('triton'),row['triton']
    # Work contracts use the label pass's operand shapes, strides and scalars.
    # A matching binary/grid alone cannot transfer that work to counter rows.
    if bool(a)!=bool(b) or (a and any(a.get(k)!=b.get(k) for k in ('cache_key','binary_sha256','grid','args'))):
      raise ValueError('counter/timing compiled identity differs')
    if 'native_binary_sha256' in launch and launch['native_binary_sha256']!=row['binary_sha256']:
      raise ValueError('counter/timing observed native binary differs')
    out.append({**launch,'native_binary_sha256':row['binary_sha256'],
      **({'sampling_control':row['sampling_control']} if row['sampling_control'] is not None else {}),
      'observed_memory':row['observed_memory']})
  return out
