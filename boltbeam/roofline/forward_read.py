"""The weight bytes one forward reads: a decode step's verify of w tokens, and one pass of an MTP draft head.

  verify   the profile's limit rows (kernel_analysis/theoretical_roofline.limit_roles, the decode limit's own list) at
           their stored widths; each expert stack reads the distinct experts w tokens reach (prefill_work.experts_reached
           over the role's own stacks: the saved routing profile's per-layer counts when one is given, uniform routing
           otherwise, said in `how`), so w = 1 is the decode limit's bytes per token
  draft    a GGUF's draft head (llama.cpp's MTP sidecar, or the head inside a combined file): only the draft blocks
           (weight_ledger.draft_blocks, the last nextn_predict_layers) and the output head, counted as the ledger counts
           a block (k of n experts per stack)

Bytes only: the read rate that turns them into time is a measurement (measured_read)."""
from __future__ import annotations

import pathlib
from typing import Any


def verify_bytes(profile:dict[str, Any], tokens:float = 1.0, routing:list[dict[str, Any]] | None = None) -> dict[str, Any]:
  """{"bytes", "tokens", "how"}: the weight bytes a forward of `tokens` tokens reads. routing: a routing profile's layers
  (roofline/routing_profile.find(...)["layers"])."""
  from boltbeam.kernel_analysis.theoretical_roofline import bits_per_weight, limit_roles
  from boltbeam.roofline.prefill_work import experts_reached, role_routing
  total = 0.0
  for r in limit_roles(profile):
    per = int(r.get("rows") or 0) * int(r.get("cols") or 0) * bits_per_weight(r) / 8
    if r.get("n_expert"):
      n, k, stacks = int(r["n_expert"]), int(r.get("experts_read") or r["n_expert"]), int(r.get("calls_per_token") or 1)
      total += per * sum(experts_reached(tokens, n, k, stacks, role_routing(r, routing)))
    else:
      total += per * int(r.get("count") or 1)
  return {"bytes": total, "tokens": tokens,
          "how": f"the decode limit's weight rows at {tokens:.2f} tokens a forward, each expert stack reading the experts that "
                 f"many tokens reach ({'the saved routing profile' if routing else 'uniform routing: no routing profile saved'})"}


def draft_bytes(path:str | pathlib.Path) -> dict[str, Any]:
  """{"bytes", "how"}: the weight bytes one pass of a GGUF's draft head reads (the module docstring). Refuses a file
  with no draft blocks."""
  from boltbeam.core.refused import Refused
  from boltbeam.profile import weight_ledger
  from boltbeam.profile.gguf import read_gguf_split
  from boltbeam.profile.roles import role_from_tensor_name
  kv, tensors, _ = read_gguf_split(path)
  arch = kv.get("general.architecture")
  head = weight_ledger.draft_blocks(kv, arch)
  if not head:
    raise Refused(f"{pathlib.Path(path).name} has no draft blocks ({arch}.nextn_predict_layers): it is not an MTP head")
  mine = [t for t in tensors if t[0].startswith(head) or role_from_tensor_name(t[0], tuple(t[1])) == "lm_head"]
  live = {k: v for k, v in kv.items() if k != f"{arch}.nextn_predict_layers"}  # its blocks counted as blocks
  got = weight_ledger.build(live, mine, arch=arch)
  return {"bytes": float(got["counted_bytes"]),
          "how": f"the draft head of {pathlib.Path(path).name}: {', '.join(head)} and the output head, counted as the weight ledger "
                 "counts a block (k of n experts per stack)"}


def measured_read(target_id:str) -> dict[str, Any]:
  """{"gbs", "source"}: the GPU's measured read rate, the one the decode limit uses: this machine's chip profile
  (limit.profile_read), else the registry's figure only when it is a measurement (its scope and date said). A
  datasheet figure is refused: what bounds a step is read at a measured rate or not at all."""
  from boltbeam.core.refused import Refused
  from boltbeam.roofline import limit
  from boltbeam.target.targets import get_target
  if got := limit.profile_read({"target_id": target_id}):
    return {"gbs": got[0], "source": got[1]}
  src = ((get_target(target_id).capabilities or {}).get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  if src.get("kind") != "measurement" or not src.get("value_gbs"):
    raise Refused(f"{target_id} has no measured read rate (only a datasheet figure): measure it on this GPU "
                  "(`python -m boltbeam.workflow.screen autoscan`) before reading what bounds a step")
  return {"gbs": float(src["value_gbs"]), "source": f"measured: {src.get('scope', target_id)}, {src.get('observed_at', 'date unknown')}",
          "method": src.get("method")}
