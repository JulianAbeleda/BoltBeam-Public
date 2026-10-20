"""Check when summing per-operation bounds is valid for concurrent work."""
from boltbeam.roofline.memory import hierarchy_ms


def require_shared_limit(launches,rows,bandwidth_gbs,cache):
  """Refuse overlapping modeled operations with independent limiting resources.

  Two DRAM-bound operations share the same bandwidth budget, so their byte
  bounds add. A compute-bound and a memory-bound operation may overlap their
  ideal times: adding those bounds would overstate the ideal elapsed time.
  """
  by_name={r['name']:r for r in rows}
  active=[]
  for launch in sorted(launches,key=lambda l:l['start_us']):
    work=launch.get('semantic_work')
    if not work and launch.get('semantic_kind')!='attention':
      continue
    resource='unresolved attention'
    if work:
      row=by_name[work['name']]
      peak=row.get('peak_tflops')
      compute=work['flops']/(peak*1e9) if peak else 0.
      if work.get('requires_memory_evidence'):
        if not work.get('memory_traffic') or not cache or not cache.get('gbs'):
          raise ValueError(f"{work['name']}: captured memory traffic and cache bandwidth are required for its limit")
        memory=hierarchy_ms(work['memory_traffic'],bandwidth_gbs,cache['gbs'])
        time,tier=memory['memory_ms'],memory['memory_tier']
      else:
        time,tier=work['bytes']/(bandwidth_gbs*1e6),'DRAM'
      resource=tier if time>=compute else 'compute:'+row['path']
    active=[(end,r) for end,r in active if end>launch['start_us']]
    if any(r!=resource or resource=='unresolved attention' for _,r in active):
      raise ValueError('concurrent modeled operations have independent or unresolved limiting resources; '
                       'summing their serial limits is not a validated elapsed-time roofline')
    active.append((launch['start_us']+launch['wall_us'],resource))
