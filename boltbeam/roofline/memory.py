"""Memory hierarchy bounds from measured bytes and independently measured rates."""
import math


def peak_bandwidth(evidence):
  """An attainable peak comes from independent probe samples, never model timings."""
  if evidence.get('schema')!='boltbeam.bandwidth_peak.v1' or evidence.get('quantity')!='l2_request_gbs':
    raise ValueError('unsupported cache bandwidth evidence')
  values=[sample.get('gbs') for sample in evidence.get('samples',[])]
  if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
    raise ValueError('cache bandwidth evidence requires positive finite probe samples')
  return max(values)


def reread_ms(nbytes, weight_bytes, bandwidth_gbs, cache):
  """The least time to read `nbytes` of which `weight_bytes` are a weight the same step already read: at most the
  last-level cache's size of that weight can still be there, so DRAM serves the rest, and every byte passes the cache
  at its measured rate (as hierarchy_ms holds the tiers). An upper bound on the hits, so a lower bound on the time: a
  limit. A speculative step's draft head reads its weights once per draft, back to back (vllm_semantic_work)."""
  if not cache or not cache.get('bytes') or not cache.get('gbs'):
    raise ValueError("a weight read again in the same step needs the chip's last-level cache size and rate for its limit")
  values=[nbytes,weight_bytes,bandwidth_gbs,cache['bytes'],cache['gbs']]
  if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values) or weight_bytes>nbytes:
    raise ValueError('a cached re-read needs finite bytes, a weight within them and positive rates')
  cached=min(float(weight_bytes),float(cache['bytes']))
  dram=(nbytes-cached)/(bandwidth_gbs*1e6)
  l2=nbytes/(cache['gbs']*1e6)
  return {'dram_ms':dram,'l2_ms':l2,'memory_ms':max(dram,l2),'cached_bytes':cached,
          'memory_tier':'DRAM' if dram>=l2 else 'L2'}


def hierarchy_ms(traffic, bandwidth_gbs, cache_bandwidth_gbs):
  required=('dram_read_bytes','dram_write_bytes','l2_bytes')
  values=[traffic.get(k) for k in required]+[bandwidth_gbs,cache_bandwidth_gbs]
  if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
    raise ValueError('memory hierarchy requires finite measured traffic and rates')
  if bandwidth_gbs<=0 or cache_bandwidth_gbs<=0:raise ValueError('memory rates must be positive')
  dram=(traffic['dram_read_bytes']+traffic['dram_write_bytes'])/(bandwidth_gbs*1e6)
  l2=traffic['l2_bytes']/(cache_bandwidth_gbs*1e6)
  return {'dram_ms':dram,'l2_ms':l2,'memory_ms':max(dram,l2),
          'memory_tier':'DRAM' if dram>=l2 else 'L2'}
