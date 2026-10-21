"""Decode presentation from a captured operation ledger, independent of engine.

No measurements or model formulas belong here. The collector supplies operation
work and its provenance; the common report gate checks it before Run or Emit.
"""
import json
from pathlib import Path
from boltbeam.workflow import tie_out

TRACE = 'decode_capture.json'


def _graph_trace(run:Path) -> str | None:
  """How the run's vendor capture traced the engine's graphs (target/vendors.py terms), from the run's target."""
  from boltbeam.target import vendors
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  return vendors.for_backend(get_target(load_manifest(run).get('target_id')).backend).term('graph_trace')


def loss_block(run:Path, provider:str, token:dict | None) -> dict:
  trace=json.loads((run/TRACE).read_text())
  if trace.get('schema')!='boltbeam.decode_capture.v1' or trace.get('provider')!=provider:raise ValueError('decode capture schema or provider differs from this run')
  if token is None:raise ValueError('decode capture has no untraced token measurement')
  context=int(token['context'])
  capture=(trace.get('contexts') or {}).get(str(context))
  if not capture:raise ValueError('decode capture does not cover the headline context')
  count=len((capture.get('engine') or {}).get('decode_steps') or [1]);fields=('ms','calls','flops','bytes','ceiling_ms','best_ms','compute_ms','memory_ms')
  capture={**capture,**{key:capture[key]/count for key in ('wall_ms','profiled_wall_ms','busy_ms','gaps_ms') if key in capture},
           'rows':[{**row,**({'ceiling_range_ms':[v/count for v in row['ceiling_range_ms']]} if row.get('ceiling_range_ms') else {}),**{key:row[key]/count for key in fields if row.get(key) is not None}}
                   for row in capture['rows']]}
  from boltbeam.workflow.repeat_bands import find
  band=capture.get('reference_band');host=capture.get('host_step_ms');comparison=None
  capture={**capture,'end_to_end':tie_out.end_to_end({**capture,'measured_ms':host},token['ms'],what='decode step',
    band=find(run,f'decode {context}'),reference_window=(band or {}).get('timing_window'))}
  if capture['end_to_end'].get('status')=='checked':  # only matched windows are compared; otherwise nothing is labelled overhead
    comparison={'title':'Capture versus untraced reference','columns':['comparison','ms/token'],'estimate':False,'token_ms':host,
      'rows':[{'name':name,'now_ms':v,'cells':[f'{v:.6f}']} for name,v in
              [('untraced median',token['ms']),('remaining capture overhead (unassigned)' if host>token['ms'] else 'capture/reference difference',host-token['ms'])]],
      'sum_cells':['= captured',f'{host:.6f}'],
      'words':capture['end_to_end']['words']+'. '+(f'{w} is enabled in the capture. ' if (w:=_graph_trace(run)) else '')
              +f"Both are host-elapsed decode steps (long then short call): untraced is the median of {band['repeats']} samples, captured is the median of {capture['host_samples']}; the difference is not assigned to a model operation."}
  table=tie_out.where_decode_capture_goes(capture,untraced_ms=token['ms'],context=context)
  if count>1:table['words']+=f' Mean of {count} decode steps, including gaps.'
  limit=table['limit_ms']
  if limit<=0:raise ValueError('decode capture has no positive operation limit')
  return {'status':'modeled','provider':provider,'workload':'decode','per':'per token',
          'limit_ms':limit,'limit_tok_s':1000/limit,'limit_context':context,
          'source':TRACE,'semantic_capture':True,'where_token_goes':table,
          'capture_comparison':comparison,
          'runtimes':[{'provider':provider,'ms':token['ms'],'tok_s':token['tok_s'],'lost_ms':token['ms']-limit,'per_role':False,
                       'note':f'Untraced at context {context}; table uses captured time.'}],
          'roles':[],'others':[],'unpaired_roles':[],'not_timed':[],'tie_out':None,'estimate':None,'missing':None,'refused':None,'findings':[]}
