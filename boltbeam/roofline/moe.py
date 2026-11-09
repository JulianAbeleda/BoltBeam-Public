"""Model work constraints for observed mixture-of-experts routing."""


def selected_experts(route: dict, *, tokens: int, top_k: int, experts: int) -> int:
  if not 0<top_k<=experts or tokens<=0 or route.get('shape')!=[tokens,top_k]:
    raise ValueError('expert routing shape disagrees with model work')
  counts=route.get('expert_counts') or {}
  if any(type(count) is not int or not 0<count<=tokens or not 0<=int(expert)<experts
         for expert,count in counts.items()):
    raise ValueError('expert routing contains invalid expert counts')
  if len({int(k) for k in counts})!=len(counts) or sum(counts.values())!=tokens*top_k:
    raise ValueError('expert routing counts do not cover token products')
  return len(counts)


def routing_basis(tokens: int, observed: bool) -> str:
  """How a launch's distinct experts are known: observed routing; one token, which selects exactly top-k distinct
  experts; else top-k distinct experts is only the minimum (actual coverage not measured)."""
  return 'observed' if observed else 'single_token_topk' if tokens == 1 else 'minimum'
