"""Translate vLLM scheduling facts into request/phase evidence.

This adapter records counts and token hashes, never prompt text or token IDs. Timing and
roofline arithmetic belong to consumers of the resulting step ledger.
"""

import hashlib
import json


def token_hash(tokens):
  return hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()


class ScheduledWork:
  def __init__(self):
    self.prompts = {}
    self.prompt_hashes = {}
    self.step = 0

  def record(self, output):
    if output.scheduled_spec_decode_tokens:
      raise ValueError('replay step accounting requires separate speculative verification work')
    for request in output.finished_req_ids:
      self.prompts.pop(request, None)
      self.prompt_hashes.pop(request, None)
    computed = {}
    for request in output.scheduled_new_reqs:
      if request.prompt_token_ids is None or request.mm_features:
        raise ValueError('replay step accounting currently requires text token prompts')
      self.prompts[request.req_id] = len(request.prompt_token_ids)
      self.prompt_hashes[request.req_id] = token_hash(request.prompt_token_ids)
      computed[request.req_id] = request.num_computed_tokens
    cached = output.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.num_computed_tokens):
      raise ValueError('cached request identities do not cover computed token counts')
    for request, depth in zip(cached.req_ids, cached.num_computed_tokens):
      if request in computed:
        raise ValueError('request appears as both new and cached in one step')
      computed[request] = depth
    rows = []
    for request, count in output.num_scheduled_tokens.items():
      if request not in self.prompts or request not in computed:
        raise ValueError('scheduled request has no observed prompt or computed depth')
      depth = computed[request]
      if type(count) is not int or count <= 0 or type(depth) is not int or depth < 0:
        raise ValueError('scheduled token count and depth must be valid integers')
      prompt = self.prompts[request]
      prefill = min(count, max(prompt-depth, 0))
      rows.append({'request_id': request, 'prompt_tokens': prompt, 'prompt_sha256':self.prompt_hashes[request], 'depth': depth,
                   'scheduled_tokens': count, 'prefill_tokens': prefill,
                   'decode_tokens': count-prefill})
    if sum(r['scheduled_tokens'] for r in rows) != output.total_num_scheduled_tokens:
      raise ValueError('scheduled request counts do not sum to the step total')
    result = {'step': self.step, 'requests': rows,
              'scheduled_tokens': output.total_num_scheduled_tokens}
    self.step += 1
    return result
