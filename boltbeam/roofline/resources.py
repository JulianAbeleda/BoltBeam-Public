"""Whole-window resource demand: shared engines add, independent engines overlap.

Callers own evidence admission. Rates are demand units per millisecond and must
bound the same measured resource and clock regime as their demand. The result is
an empirical resource roofline, not a sum of serial per-operation ceilings.
"""
import math
from collections import defaultdict


def aggregate(terms):
  resources=defaultdict(lambda:{'floor_low_ms':0.,'floor_high_ms':0.,'terms':0})
  if not terms:raise ValueError('resource roofline has no admitted demand')
  for term in terms:
    name=term['resource'];demand=term['demand'];rate=term['rate']
    if not isinstance(name,str) or not name:raise ValueError('resource identity missing')
    if len(demand)!=2 or len(rate)!=2:raise ValueError('resource bounds need two endpoints')
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in (*demand,*rate)):
      raise ValueError('resource demand or rate is not finite')
    if not 0<=demand[0]<=demand[1] or not 0<rate[0]<=rate[1]:
      raise ValueError('invalid resource demand or rate bounds')
    row=resources[name];row['floor_low_ms']+=demand[0]/rate[1];row['floor_high_ms']+=demand[1]/rate[0];row['terms']+=1
    if not all(math.isfinite(row[k]) for k in ('floor_low_ms','floor_high_ms')):raise ValueError('resource arithmetic overflow')
  return {'resources':dict(resources),'floor_low_ms':max(r['floor_low_ms'] for r in resources.values()),
          'floor_high_ms':max(r['floor_high_ms'] for r in resources.values()),
          'basis':'maximum of accumulated demand time per shared resource; independent resources may overlap'}
