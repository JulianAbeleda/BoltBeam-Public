"""Reconstruct PM records from a prospectively preserved completed prefix.

Callers bind every input to the completed capture manifest before and after
reading. Late SDK timestamp changes remain explicit; source files are untouched.
"""
import csv,json,math


def completed_prefix(root,window,metrics):
  def read(suffix):
    rows=list(csv.DictReader((root/('resource-window'+suffix)).open()))
    for row in rows:
      for key in ('index','start_ns','end_ns'):row[key]=int(row[key])
      for key in metrics:
        value=float(row[key])
        if not math.isfinite(value) or value<0 or not value.is_integer():raise ValueError('invalid prefix counter')
        row[key]=int(value)
    return rows
  early=read('.early.csv');late=read('.csv')
  meta=json.loads((root/'resource-window.early.json').read_text())
  clocks=[json.loads(line) for line in (root/'resource-window.clocks.jsonl').read_text().splitlines()]
  barriers=[row['cupti_ns'] for row in clocks if row['stage']=='after_early_decode']
  if len(barriers)!=1 or not barriers[0]<window['work_start_ns']:raise ValueError('prefix snapshot did not precede work')
  if not early or len(early)!=meta['completed'] or len(early)!=meta['populated'] or len(early)>len(late):
    raise ValueError('incomplete preserved prefix')
  if len(late)!=window['completed_samples'] or len(late)!=window['populated_samples']:
    raise ValueError('prefix sample count changed')
  if late[0]['start_ns']!=window['first_sample_start_ns'] or late[-1]['end_ns']!=window['last_sample_end_ns']:
    raise ValueError('late sample metadata differs from raw records')
  changes=[]
  for index,row in enumerate(early):
    final=late[index]
    if row['index']!=index or final['index']!=index or not row['start_ns']<row['end_ns']<=barriers[0]:
      raise ValueError('invalid preserved prefix interval')
    if any(row[key]!=final[key] for key in (*metrics,'end_ns')):raise ValueError('completed prefix values changed')
    if row['start_ns']!=final['start_ns']:
      changes.append({'index':index,'early_start_ns':row['start_ns'],'late_start_ns':final['start_ns'],'end_ns':row['end_ns']})
  if early[0]['start_ns']!=meta['start_ns'] or early[-1]['end_ns']!=meta['end_ns']:
    raise ValueError('prefix metadata differs from preserved records')
  return early+late[len(early):],{'preserved_samples':len(early),'late_prefix_start_changes':changes,
      'late_first_sample_start_ns':window['first_sample_start_ns'],
      'scope':'completed records observed before work; later queries retained separately; no timestamp repair'}
