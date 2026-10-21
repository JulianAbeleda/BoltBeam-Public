"""Run identical scripted requests through production references and paired captures.

Raw token content stays in the run directory. Public review output contains only
script hashes, lengths and derived timing. The server context owns cleanup.
"""
import hashlib,json,os,sys,time,urllib.request
from boltbeam.collectors.providers import gpu_free
from pathlib import Path
from boltbeam.adapters.llama_server_control import serve_argv,_json_post
from boltbeam.collectors.replay import vllm_env,Words,REPLAY_ENGINES,flag_value
from boltbeam.collectors.gpu_telemetry import Telemetry, run_telemetry


def stream(url,body):
  request=urllib.request.Request(url+'/v1/completions',data=json.dumps({**body,'stream':True,
    'stream_options':{'include_usage':True}}).encode(),headers={'Content-Type':'application/json'})
  sent=time.perf_counter();first=last=None;ids=[];prompt=None;usage=None;request_id=None;text=''
  with urllib.request.urlopen(request,timeout=1800) as response:
    for raw in response:
      line=raw.decode().strip()
      if not line.startswith('data:') or line=='data: [DONE]':continue
      chunk=json.loads(line[5:]);request_id=request_id or chunk['id']
      if request_id!=chunk['id']:raise ValueError('stream changed request identity')
      if chunk.get('usage'):usage=chunk['usage']
      for choice in chunk.get('choices',[]):
        text+=choice.get('text') or ''
        if choice.get('prompt_token_ids') is not None:prompt=choice['prompt_token_ids']
        tokens=choice.get('token_ids') or []
        if tokens:
          now=time.perf_counter();first=now if first is None else first;last=now;ids.extend(tokens)
  if first is None or last is None or not request_id or prompt!=body['prompt'] or len(ids)!=body['max_tokens'] or not usage or usage['completion_tokens']!=len(ids):
    raise ValueError('stream token evidence differs from the request')
  if 'vllm_xargs' in body and ids!=body['vllm_xargs']['boltbeam_token_script']:
    raise ValueError('generated tokens differ from the script')
  return {'id':request_id,'usage':usage,'choices':[{'prompt_token_ids':prompt,'token_ids':ids}],'text':text,
          't_send':sent,'t_first':first,'t_last':last,'prefill_client_ms':(first-sent)*1000,
          'decode_client_ms':(last-first)*1000,'client_ms':(last-sent)*1000}


def request(model,prompt,tokens,*,force=True):
  body={'model':model,'prompt':prompt,'max_tokens':len(tokens) if force else tokens,
        'temperature':0,'ignore_eos':True,'return_token_ids':True}
  if force:body['vllm_xargs']={'boltbeam_token_script':tokens}
  return body


def drive(run:Path,mode:str,repeats:int=5):
  from boltbeam.trace.replay_trace import read
  if mode not in ('record','reference','paired','cost') or type(repeats) is not int or repeats<2:
    raise ValueError('replay needs a supported pass and at least two repeats')
  from boltbeam.target.targets import get_target  # the run's GPU; a run with no manifest predates AMD (CUDA)
  facts=run/'run_manifest.json'
  free=gpu_free(get_target(json.loads(facts.read_text())['target_id']).backend if facts.exists() else 'CUDA')
  if not free['free']:raise RuntimeError(free['reason'])
  if os.environ.get('VLLM_BATCH_INVARIANT') not in (None,'','0'):
    raise ValueError('forced replay requires production compilation, not VLLM_BATCH_INVARIANT')
  root=run/'forced_replay';out=root/mode
  out.mkdir(parents=True,exist_ok=False)
  argv=json.loads((run/'replay_server.json').read_text())['argv']
  if any(int(flag_value(argv,(flag,)) or 1)!=1 for flag in ('--tensor-parallel-size','--pipeline-parallel-size')):
    raise ValueError('forced replay currently requires one worker and one GPU')
  if any(flag_value(argv,(flag,)) is not None for flag in ('--worker-cls','--logits-processors')):
    raise ValueError('forced replay cannot replace a configured worker or logits processor')
  argv=[*argv,'--enable-prompt-tokens-details','--logits-processors','boltbeam.runtime.vllm_token_script:TokenScriptProcessor']
  if mode in ('reference','paired'):argv+=['--worker-cls','boltbeam.runtime.vllm_replay_worker.ReplayWorker']
  repo=Path(__file__).resolve().parents[2]
  env={**os.environ,**vllm_env(argv),'VLLM_SERVER_DEV_MODE':'1','BOLTBEAM_REPLAY_CAPTURE_DIR':str(out),
       'PYTHONPATH':os.pathsep.join([str(repo),str(repo/'boltbeam/runtime'),os.environ.get('PYTHONPATH','')])}
  for key in ('BOLTBEAM_REPLAY_PROFILE_STEP','BOLTBEAM_REPLAY_PROFILE_END_STEP','BOLTBEAM_REPLAY_CAPTURE_INITIALIZATION'):
    env.pop(key,None)
  if mode=='paired':env['BOLTBEAM_REPLAY_CAPTURE_GRAPHS']='1'
  else:env.pop('BOLTBEAM_REPLAY_CAPTURE_GRAPHS',None)
  script_path=root/'token-script.json'
  with serve_argv(argv,log_path=out/'server.log',environ=env,startup_timeout_s=900,
                  stop_timeout_s=1800 if mode=='paired' else 5) as (url,command,load), run_telemetry(run, make=Telemetry) as tele:
    post=lambda path,data:_json_post(url+path,data,1800)
    def reset():
      with urllib.request.urlopen(urllib.request.Request(url+'/reset_prefix_cache',data=b'{}',
          headers={'Content-Type':'application/json'}),timeout=60) as response:
        if response.status!=200:raise ValueError('prefix cache reset failed')
    with urllib.request.urlopen(url+'/v1/models',timeout=60) as response:
      model=json.loads(response.read())['data'][0]['id']
    if mode=='record':
      trace=read(run/'replay_trace.json');words=Words(REPLAY_ENGINES['vllm'],url)
      system={'role':'system','content':words.take(trace['system_tokens'])};messages=[system];script=[]
      for turn in trace['turns']:
        if turn['compaction']:messages=[system]
        messages.append({'role':'user','content':words.take(turn['new_tokens'])+words.take(turn['tool_tokens'])})
        prompt=post('/tokenize',{'model':model,'messages':messages,'add_generation_prompt':True,
                               'chat_template_kwargs':{'enable_thinking':False}})['tokens']
        row=stream(url,request(model,prompt,turn['reply_tokens'],force=False))
        script.append({'turn':turn['i'],'prompt':prompt,'tokens':row['choices'][0]['token_ids']})
        messages.append({'role':'assistant','content':row['text']})
      script_path.write_text(json.dumps(script))
    else:
      script=json.loads(script_path.read_text())
      requests=[request(model,t['prompt'],t['tokens']) for t in script]
      if not requests or repeats<2:raise ValueError('replay needs turns and at least two reference repeats')
      warm=request(model,script[0]['prompt'][:512],script[0]['tokens'][:8])
      for _ in range(3):stream(url,warm)
      if mode=='cost':
        rows=[]
        for i in range(repeats):
          pair={}
          for force in ((False,True) if i%2==0 else (True,False)):
            reset()
            body=request(model,script[0]['prompt'],script[0]['tokens'] if force else len(script[0]['tokens']),force=force)
            row=stream(url,body);n=row['usage']['completion_tokens']-1
            if n<1:raise ValueError('forcing cost requires at least two output tokens')
            pair['forced' if force else 'free']=row['decode_client_ms']/n
          rows.append(pair)
          (out/'cost.json').write_text(json.dumps(rows))
        return {'pairs':len(rows)}
      rpc=lambda name:post('/collective_rpc',{'method':'boltbeam_capture','args':[name]})
      rpc('off');results={}
      for name in (['timing','label'] if mode=='paired' else [str(i) for i in range(repeats)]):
        reset()
        rpc(name if mode=='paired' else 'timing');rows=[];results[name]=rows
        try:
          for i,body in enumerate(requests,1):
            if name=='label' and i>1:rpc('resume')
            row=stream(url,body);row['turn']=i;rows.append(row)
            if name=='label':rpc('pause')
            (out/'responses.json').write_text(json.dumps(results))
            print(name,'turn',i,flush=True)
        finally:rpc('off')
        if mode=='reference':(out/'timing').rename(out/name)
    (out/'telemetry.json').write_text(json.dumps({'summary':tele.summary(),'samples':tele.samples}))
  return {'script_sha256':hashlib.sha256(script_path.read_bytes()).hexdigest(),
          'turn_output_lengths':[len(t['tokens']) for t in script]}


if __name__=='__main__':
  drive(Path(sys.argv[1]).resolve(),sys.argv[2],int(sys.argv[3]))
