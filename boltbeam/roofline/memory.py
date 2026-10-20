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
