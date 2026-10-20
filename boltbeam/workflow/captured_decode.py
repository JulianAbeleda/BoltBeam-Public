"""Decode presentation from a captured operation ledger, independent of engine.

No measurements or model formulas belong here. The collector supplies operation
work and its provenance; the common report gate checks it before Run or Emit.
"""
import json
from pathlib import Path
from boltbeam.workflow import tie_out

TRACE = 'decode_capture.json'


def loss_block(run:Path, provider:str, token:dict | None) -> dict:
  trace=json.loads((run/TRACE).read_text())
  if trace.get('schema')!='boltbeam.decode_capture.v1' or trace.get('provider')!=provider:
    raise ValueError('decode capture schema or provider differs from this run')
  if token is None:
    raise ValueError('decode capture has no untraced token measurement')
  context=int(token['context'])
  capture=(trace.get('contexts') or {}).get(str(context))
  if not capture:
    raise ValueError('decode capture does not cover the headline context')
  count=len((capture.get('engine') or {}).get('decode_steps') or [1])
  fields=('ms','calls','flops','bytes','ceiling_ms','best_ms','compute_ms','memory_ms')
  capture={**capture,**{key:capture[key]/count for key in ('wall_ms','profiled_wall_ms','busy_ms','gaps_ms') if key in capture},
           'rows':[{**row,**({'ceiling_range_ms':[v/count for v in row['ceiling_range_ms']]} if row.get('ceiling_range_ms') else {}),**{key:row[key]/count for key in fields if row.get(key) is not None}}
                   for row in capture['rows']]}
  from boltbeam.workflow.repeat_bands import find
  capture={**capture,'end_to_end':tie_out.end_to_end(
    capture,token['ms'],what='decode step',band=find(run,f'decode {context}'))}
  if band:=capture.get('reference_band'):
    wall=capture['wall_ms'];low,high=band['low_ms'],band['high_ms']
    capture['end_to_end'].update(confirms=low<=wall<=high,
      words=f'captured {wall:.6f} ms/token; untraced paired-repeat band [{low:.6f}, {high:.6f}] ms/token')
  comparison=None
  if band:
    delta=capture['wall_ms']-band['median_ms']
    label='remaining capture overhead (unassigned)' if delta>0 and not capture['end_to_end']['confirms'] else 'capture/reference difference'
    comparison={'title':'Capture versus untraced reference','columns':['comparison','ms/token'],'estimate':False,
      'token_ms':capture['wall_ms'],
      'rows':[{'name':name,'now_ms':value,'cells':[f'{value:.6f}']}
              for name,value in [('untraced median',band['median_ms']),(label,delta)]],
      'sum_cells':['= captured',f"{capture['wall_ms']:.6f}"],
      'words':capture['end_to_end']['words']+'. Nsight graph-node tracing is enabled in the capture. '
              'The difference is not assigned to a model operation; this comparison does not alter the measured kernel rows.'}
  table=tie_out.where_decode_capture_goes(capture,untraced_ms=token['ms'],context=context)
  if count>1:
    table['words']+=f' Mean of {count} decode steps, including gaps.'
  limit=table['limit_ms']
  if limit<=0:
    raise ValueError('decode capture has no positive operation limit')
  return {'status':'modeled','provider':provider,'workload':'decode','per':'per token',
          'limit_ms':limit,'limit_tok_s':1000/limit,'limit_context':context,
          'source':TRACE,'semantic_capture':True,'where_token_goes':table,
          'capture_comparison':comparison,
          'runtimes':[{'provider':provider,'ms':token['ms'],'tok_s':token['tok_s'],
                       'lost_ms':token['ms']-limit,'per_role':False,
                       'note':f'Untraced at context {context}; table uses captured time.'}],
          'roles':[],'others':[],'unpaired_roles':[],'not_timed':[],
          'tie_out':None,'estimate':None,'missing':None,'refused':None,'findings':[]}
