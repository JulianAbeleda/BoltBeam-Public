"""Normalize Nsight Compute memory counters without assuming metric spelling/units.

This module knows CUDA counters, not model families. Missing counters are an
error; Nsight can otherwise silently omit a requested metric it does not know.
"""
import csv
import io
import math
import re
import subprocess
import json
from pathlib import Path

METRICS = {'dram_read_bytes':('dram__bytes_op_read','dram__bytes_read'),
           'dram_write_bytes':('dram__bytes_op_write','dram__bytes_write'),
           'l2_bytes':('lts__t_bytes',), 'duration_ns':('gpu__time_duration',)}
UNITS = {'byte':1.,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'ns':1.,'us':1e3,'ms':1e6,'s':1e9}


def select_metrics(available):
  selected={}
  for field, aliases in METRICS.items():
    name=next((x for x in aliases if x in available),None)
    if not name:raise ValueError(f'no counter available for {field}')
    selected[field]=name+'.sum'
  return selected


def read_raw(text):
  """NCU --page raw --csv has one units row then one row per launch."""
  offset=text.find('"ID",')
  if offset<0:raise ValueError('NCU did not export raw counter rows')
  reader=csv.DictReader(io.StringIO(text[offset:]));units=next(reader)
  selected=select_metrics({k.removesuffix('.sum') for k in units if k.endswith('.sum')})
  result=[]
  for row in reader:
    item={'kernel':row['Kernel Name'],'grid':list(map(int,re.findall(r'\d+',row['Grid Size']))),
          'block':list(map(int,re.findall(r'\d+',row['Block Size']))),'id':int(row['ID'])}
    for field,name in selected.items():
      if units[name] not in UNITS:raise ValueError(f'unknown counter unit {units[name]}')
      value=float(row[name].replace(',',''))*UNITS[units[name]]
      if not math.isfinite(value) or value<0:raise ValueError(f'invalid counter {name}')
      item[field]=value
    result.append(item)
  if not result:raise ValueError('no measured GPU launches')
  return {'metrics':selected,'rows':result}


def bind(launches, counters, *, source):
  """Bind a complete, ordered counter capture to exactly matching launch geometry.

The caller must establish matching engine/settings/input evidence. This function
refuses a missing, extra, reordered or retuned kernel rather than extrapolating.
"""
  from boltbeam.collectors.instruction_path import template
  wanted=[l for l in launches if (l.get('semantic_work') or {}).get('requires_memory_evidence')]
  rows=counters['rows']
  # NCU may expand a selected graph node to its entire CUDA graph. Verify all
  # nodes before retaining the subset; never discard unexplained extra rows.
  selected=launches if len(rows)==len(launches) else wanted
  if len(selected)!=len(rows):raise ValueError('memory counter capture does not cover all required launches')
  for l,r in zip(selected,rows):
    grid=list(map(int,re.search(r'grid=(\d+)x(\d+)x(\d+)',l['key']).groups()))
    block=list(map(int,re.search(r'block=(\d+)x(\d+)x(\d+)',l['key']).groups()))
    if template(l['key'])!=template(r['kernel']) or grid!=r['grid'] or block!=r['block']:
      raise ValueError('memory counter launch identity/geometry differs from timed capture')
  for l,r in zip(selected,rows):
    if (l.get('semantic_work') or {}).get('requires_memory_evidence'):
      l['semantic_work']['memory_traffic']={**r,'source':source}
  return launches


IDENTITY_FIELDS = ('model_path','model_config_sha256','activation_dtype','kv_cache_dtype','model_type',
                   'mamba_chunk_size','vllm','execution_mode','length','seed','max_model_len','max_num_batched_tokens')


def same_capture(timed, counter):
  for key in IDENTITY_FIELDS:
    if key=='mamba_chunk_size' and key not in timed and key not in counter:continue
    if key not in timed or timed[key] is None or timed[key]!=counter.get(key):
      raise ValueError(f'memory evidence has absent or mismatched capture setting: {key}')
  for key in ('context','batch','decode_step','decode_steps','tokens','replay'):
    if (key in timed or (key=='replay' and key in counter)) and timed.get(key)!=counter.get(key):
      raise ValueError(f'memory evidence has mismatched decode setting: {key}')
  for key in ('compilation_config','enforce_eager','capture_pass','output_token_ids'):
    if key in timed and timed[key]!=counter.get(key):
      raise ValueError(f'memory evidence has mismatched production program: {key}')
  if timed.get('triton_cache_dir') and timed['triton_cache_dir']!=counter.get('triton_cache_dir'):
    raise ValueError('memory evidence used a different observed Triton cache')


def _measure(argv, env, out: Path, launches):
  from boltbeam.collectors.engine_common import run_driver
  from boltbeam.collectors.instruction_path import template
  # NCU filters its own demangled spelling, not our whitespace-normalized template.
  # Select function families; bind() still requires exact full symbols and geometry.
  names=sorted({template(l['key']).split('<',1)[0] for l in launches if (l.get('semantic_work') or {}).get('requires_memory_evidence')})
  if not names:raise ValueError('counter measurement has no required memory launches')
  query=subprocess.run(['ncu','--query-metrics','--csv'],capture_output=True,text=True,check=True)
  available={next(iter(r.values())) for r in csv.DictReader(io.StringIO(query.stdout))}
  metrics=select_metrics(available)
  base=out/'memory';base.parent.mkdir(parents=True,exist_ok=True)
  cmd=['ncu','--target-processes','all','--profile-from-start','off','--cache-control','none',
       '--clock-control','none','--replay-mode','application','--kernel-name-base','demangled','--rename-kernels','off','--kernel-name','regex:('+'|'.join(map(re.escape,names))+')',
       '--metrics',','.join(metrics.values()),'--csv','--log-file',str(base.with_suffix('.log')),
       '--export',str(base),'--force-overwrite',*argv]
  base.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2))
  rows=run_driver(cmd,env=env,log=base.with_suffix('.engine.log'),timeout_s=7200)
  raw=subprocess.run(['ncu','--import',str(base.with_suffix('.ncu-rep')),'--page','raw','--csv','--rename-kernels','off','--print-units','base'],capture_output=True,text=True,check=True)
  base.with_suffix('.csv').write_text(raw.stdout)
  parsed=read_raw(raw.stdout)
  return rows,parsed,base


def collect(argv, env, out: Path, launches, timed):
  """Measure required memory traffic in the same application, without cache flushing."""
  if not any((l.get('semantic_work') or {}).get('requires_memory_evidence') for l in launches):return launches,None
  rows,parsed,base=_measure(argv,env,out,launches)
  counter=next(r for r in reversed(rows) if 'execution_mode' in r)
  same_capture(timed,counter)
  identity={k:counter[k] for k in IDENTITY_FIELDS if k in counter}
  identity.update({k:counter[k] for k in ('context','batch','decode_step','tokens','replay') if k in counter})
  identity.update({k:counter[k] for k in ('compilation_config','enforce_eager','capture_pass','output_token_ids') if k in counter})
  if counter.get('triton_cache_dir'):
    identity['triton_cache_dir']=counter['triton_cache_dir']
  if counter.get('routing') is not None:
    identity['routing']=counter['routing']
  base.with_suffix('.json').write_text(json.dumps({'identity':identity,**parsed},indent=1))
  return bind(launches,parsed,source=str(base.with_suffix('.json'))),base.with_suffix('.json')


def partition_steps(steps, identities, counters):
  """Partition one serial counter pass by exact step identity and ordered launch coverage."""
  ids=[s['identity'].get('decode_step') for s in steps]
  if len(steps)!=len(identities) or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):
    raise ValueError('counter pass has missing, duplicate or invalid step identities')
  if ids!=sorted(ids):raise ValueError('counter pass steps are not in execution order')
  full=len(counters['rows'])==sum(len(s['launches']) for s in steps)
  offset=0;parts=[]
  for step,identity in zip(steps,identities):
    same_capture(step['identity'],identity)
    count=len(step['launches']) if full else sum(bool((l.get('semantic_work') or {}).get('requires_memory_evidence')) for l in step['launches'])
    parts.append({'identity':identity,'metrics':counters['metrics'],'rows':counters['rows'][offset:offset+count],
                  'counter_offset':offset})
    offset+=count
  if offset!=len(counters['rows']):raise ValueError('counter pass launch count differs from complete step coverage')
  return parts


def decode_horizon(identities):
  """A complete N-minus-one decode horizon, with its observed starting context."""
  if not identities:raise ValueError('empty decode horizon')
  first=identities[0];count=len(identities)
  if first.get('tokens')!=count+1:raise ValueError('decode horizon omits generated steps')
  for offset,identity in enumerate(identities):
    if identity.get('decode_step')!=offset+1 or identity.get('length')!=1 or identity.get('batch')!=1:
      raise ValueError('decode horizon requires contiguous single-token steps')
    expected={**first,'decode_step':offset+1,'context':first['context']+offset}
    same_capture(expected,identity)
  return {**first,'length':count,'decode_steps':list(range(1,count+1))}


def collect_horizon(argv,env,out:Path,steps):
  launches=[l for step in steps for l in step['launches']]
  if not any((l.get('semantic_work') or {}).get('requires_memory_evidence') for l in launches):return launches
  records,counters,base=_measure(argv,env,out,launches)
  identities=next(r['counter_steps'] for r in reversed(records) if 'counter_steps' in r)
  partition_steps(steps,identities,counters)
  identity=decode_horizon(identities)
  same_capture(decode_horizon([s['identity'] for s in steps]),identity)
  base.with_suffix('.json').write_text(json.dumps({'identity':identity,'steps':identities,**counters},indent=1))
  return bind(launches,counters,source=str(base.with_suffix('.json')))
