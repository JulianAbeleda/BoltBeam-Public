"""Decode presentation from a captured operation ledger, independent of engine.

No measurements or model formulas belong here. The collector supplies operation
work and its provenance; the common report gate checks it before Run or Emit.
"""
import json
from pathlib import Path
from boltbeam.collectors import replay_overhead
from boltbeam.artifacts.run_io import DECODE_CAPTURE
from boltbeam.workflow import tie_out

FORCING_WORDS = (" Every pass was forced to one production-graph generation's tokens (graphed and graphless passes had"
                 " taken different tokens), so labels and times are of the same steps; the untraced reference was not"
                 " forced, so the capture comparison includes the forcing.")
NONDETERMINISM_WORDS = (" The engine's own graphed generation did not repeat itself (stream, first differing token: {where}),"
                        " so the tokens forced are the first of two production generations that differ; one stream's"
                        " work does not depend on which tokens it takes.")
MODULE_LOAD_WORDS = (" No library file holds the code of {kernels}: the roles on them take their matrix path from a"
                     " module-load capture, the modules the driver loaded in one more pass of the same program outside"
                     " the timed captures, each kernel found by its entry symbol.")


def _graph_trace(run:Path) -> str | None:
  """How the run's vendor capture traced the engine's graphs (target/vendors.py terms), from the run's target."""
  from boltbeam.target import vendors
  from boltbeam.target.targets import get_target
  from boltbeam.artifacts.run_io import load_manifest
  return vendors.for_backend(get_target(load_manifest(run).get('target_id')).backend).term('graph_trace')


def loss_block(run:Path, provider:str, token:dict | None) -> dict:
  trace=json.loads((run/DECODE_CAPTURE).read_text())
  if trace.get('schema')!='boltbeam.decode_capture.v1' or trace.get('provider')!=provider:raise ValueError('decode capture schema or provider differs from this run')
  if token is None:raise ValueError('decode capture has no untraced token measurement')
  context=int(token['context'])
  capture=(trace.get('contexts') or {}).get(str(context))
  if not capture:raise ValueError('decode capture does not cover the headline context')
  spec=capture.get('speculation')
  # per token: a plain decode step emits one; a speculative one emits its accepted drafts and one more, so the
  # captured steps are divided by the tokens they emitted (collectors/vllm_speculation), never by the steps
  steps=len((capture.get('engine') or {}).get('decode_steps') or [1])
  count=spec['tokens'] if spec else steps;fields=('ms','calls','flops','bytes','ceiling_ms','best_ms','compute_ms','memory_ms')
  capture={**capture,**{key:capture[key]/count for key in ('wall_ms','profiled_wall_ms','busy_ms','gaps_ms') if key in capture},
           'rows':[{**row,**({'ceiling_range_ms':[v/count for v in row['ceiling_range_ms']]} if row.get('ceiling_range_ms') else {}),**{key:row[key]/count for key in fields if row.get(key) is not None}}
                   for row in capture['rows']]}
  from boltbeam.collectors.capture_scope import find
  batch=(capture.get('engine') or {}).get('batch') or 1
  if batch!=(token.get('batch') or 1):
    from boltbeam.core.refused import Refused
    raise Refused(f'the decode capture is of {batch} stream(s) and the untraced step of {token.get("batch") or 1}: '
                  'no report ties them out')
  band=capture.get('reference_band');host=capture.get('host_step_ms');comparison=None
  # the repeat-band record is the run's setup's (repeat_bands.scope holds the timed batches), so a B-stream run finds
  # its own; the band only says whether the difference is within repeat noise, it never qualifies the capture
  capture={**capture,'end_to_end':replay_overhead.end_to_end({**capture,'measured_ms':host},token['ms'],what='decode step',
    band=find(run,f'decode {context}'),reference_window=(band or {}).get('timing_window'))}
  if capture['end_to_end'].get('status')=='checked':  # only matched windows are compared; otherwise nothing is labelled overhead
    comparison={'title':'Capture versus untraced reference','columns':['comparison','ms/token'],'estimate':False,'token_ms':host,
      'rows':[{'name':name,'now_ms':v,'cells':[f'{v:.6f}']} for name,v in
              [('untraced median',token['ms']),('remaining capture overhead (unassigned)' if host>token['ms'] else 'capture/reference difference',host-token['ms'])]],
      'sum_cells':['= captured',f'{host:.6f}'],
      'words':capture['end_to_end']['words']+'. '+(f'{w} is enabled in the capture. ' if (w:=_graph_trace(run)) else '')
              +f"Both are host-elapsed decode steps (long then short call): untraced is the median of {band['repeats']} samples, captured is the median of {capture['host_samples']}; the difference is not assigned to a model operation."}
  table=tie_out.where_decode_capture_goes(capture,untraced_ms=token['ms'],context=context)
  if spec:
    table['words']+=(f" {spec['words']}. Per token: the {steps} captured steps' time and limits over the {count} tokens "
                     f"they emitted, so the limit is this capture's at its measured acceptance; {spec['accounting']}.")
  elif count>1:table['words']+=f' Mean of {count} decode steps, including gaps.'
  limit=table['limit_ms']
  if limit<=0:raise ValueError('decode capture has no positive operation limit')
  if batch>1:table['words']+=f' One decode step of {batch} streams, one token each.'
  forcing=(capture.get('engine') or {}).get('token_forcing')
  if forcing:table['words']+=FORCING_WORDS
  if nondeterministic:=(capture.get('engine') or {}).get('engine_nondeterministic'):
    table['words']+=NONDETERMINISM_WORDS.format(where=nondeterministic)
  if capture.get('observed_scope') and capture['observed_scope']!='every decode step':
    table['words']+=f" Closed kernels' executed paths were observed in {capture['observed_scope']}."
  if capture.get('module_load'):table['words']+=MODULE_LOAD_WORDS.format(kernels=', '.join(capture['module_load']))
  return {'status':'modeled','provider':provider,'workload':'decode','per':'per token' if batch==1 else f'per step of {batch} streams',
          **({'batch':batch} if batch>1 else {}),**({'token_forcing':forcing} if forcing else {}),
          **({'engine_nondeterministic':nondeterministic} if nondeterministic else {}),
          **({'observed_scope':capture['observed_scope']} if capture.get('observed_scope') else {}),
          **({'module_load':capture['module_load']} if capture.get('module_load') else {}),
          **({'speculation':{k:spec[k] for k in ('config','acceptance','tokens','steps','accounting','words','prompt')}} if spec else {}),
          'limit_ms':limit,'limit_tok_s':1000/limit,'limit_context':context,
          'source':DECODE_CAPTURE,'semantic_capture':True,'where_token_goes':table,
          'capture_comparison':comparison,
          'runtimes':[{'provider':provider,'ms':token['ms'],'tok_s':token['tok_s'],'lost_ms':token['ms']-limit,'per_role':False,
                       'note':f'Untraced at context {context}'+(f', {batch} streams' if batch>1 else '')+('; table sums to it, gaps as a bound.' if table.get('gaps_bound') else '; table uses captured time.')}],
          'roles':[],'others':[],'unpaired_roles':[],'not_timed':[],'tie_out':None,'estimate':None,'missing':None,'refused':None,'findings':[]}
