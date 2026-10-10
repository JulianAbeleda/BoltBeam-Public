"""The weight ledger: every tensor in the file, counted in the decode limit or excluded with a stated reason.

The limit (kernel_analysis/theoretical_roofline.model_roofline) is the weight bytes a decode reads per token. The rule:
every weight the decode reads once per token counts in it. A tensor whose name the classifier (profile/roles.py) maps
to a limit role (vocab.WEIGHT_GEMV_ROLES) is counted through the profile's roles. Every other tensor is still counted,
as the row "unclassified weight: <tensor name pattern>" with its bytes and count, unless one of the stated exclusions
below applies. An unrecognised tensor never drops out of the limit: the 27B hybrid's recurrent-block matrices
(5.9 GB per token) once did, and the limit read 137 tok/s against a truth of 92.8.

The exclusions, each with its reason (EXCLUSIONS):

    row_gather       token_embd: the decode reads one row of the table per token, not the table. When the file has no
                     output head the table is the output head, read whole, and it is counted.
    small_vector     a vector, or a matrix with a side under VECTOR_SIDE (norm, bias, conv taps, per-head scale),
                     under SMALL_BYTES per tensor.
    inactive_experts an expert stack (rank 3, n experts): the decode reads expert_used_count of them per token, so
                     the other n - k are not read. The k are counted; without expert_used_count all n are counted.

The invariant (check): bytes counted in the limit + bytes excluded with a reason = total tensor bytes in the file, and
that total fills the file's data region to within its alignment padding. The counted bytes are read back from the
profile's own limit roles, not from this ledger's view of them, so a reader that drops a tensor breaks the sum. The
profile writer refuses with the difference named.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

from boltbeam.profile.decode_roles import GGML_TYPE_NAMES, _physical_bytes
from boltbeam.profile.roles import role_from_tensor_name
from boltbeam.vocab import WEIGHT_GEMV_ROLES, is_ssm_role

SMALL_BYTES = 1 << 20  # 1 MiB per tensor: below this a vector-like tensor is a stated exclusion
VECTOR_SIDE = 16  # a matrix with a side under this is vector-like (conv taps 4, norm groups 8, a (1, n) scale)
UNCLASSIFIED = "unclassified weight"

EXCLUSIONS = {
  "row_gather": "token_embd is read as one row per token (a gather), not as the table; the output head is its own tensor",
  "small_vector": f"vector-like (rank 1, or a side under {VECTOR_SIDE}) and under {SMALL_BYTES >> 20} MiB per tensor: "
                  "a norm, bias, conv tap or scale",
  "inactive_experts": "expert stack: only expert_used_count of the experts are read per token; the rest are not read",
}


class WeightLedgerError(ValueError):
  """The limit's bytes and the file's bytes do not add up: the profile is refused."""


def name_pattern(name:str) -> str:
  """The tensor name with its block index replaced: blk.12.ssm_in.weight -> blk.*.ssm_in.weight."""
  return re.sub(r"(?<=\.)\d+(?=\.)", "*", name)


def is_limit_role_tensor(name:str, dims:tuple[int, ...]) -> bool:
  """True for a tensor the profile's roles carry into the limit: a rank-two weight of a limit role. The same test
  gguf.profile_from_gguf applies (a `.weight`, or any ssm tensor, of rank two)."""
  role = role_from_tensor_name(name)
  return role in WEIGHT_GEMV_ROLES and len(dims) == 2 and (name.endswith(".weight") or is_ssm_role(role))


def _vector_like(dims:tuple[int, ...]) -> bool:
  return len(dims) <= 1 or min(int(d) for d in dims) < VECTOR_SIDE


def _expert_used(kv:Mapping[str, Any], arch:str | None) -> int | None:
  k = kv.get(f"{arch}.expert_used_count") if arch else None
  return int(k) if isinstance(k, int) and k > 0 else None


def build(kv:Mapping[str, Any], tensors:Iterable[tuple[str, tuple[int, ...], int, int]], *, arch:str | None,
          data_bytes:int | None = None, alignment:int = 32) -> dict[str, Any]:
  """Sort every tensor of the file into the limit's roles, an unclassified counted row, or a stated exclusion."""
  tensors = list(tensors)
  has_head = any(role_from_tensor_name(n) == "lm_head" for n, *_ in tensors)
  k_used = _expert_used(kv, arch)
  total = limit_bytes = limit_tensors = 0
  counted: dict[tuple, dict[str, Any]] = {}
  excluded: dict[tuple, dict[str, Any]] = {}
  notes: list[str] = []

  def exclude(pattern:str, reason:str, nbytes:int, detail:str | None = None) -> None:
    row = excluded.setdefault((pattern, reason, detail), {"pattern": pattern, "reason": reason, "why": detail or EXCLUSIONS[reason],
                                                          "tensors": 0, "bytes": 0})
    row["tensors"] += 1
    row["bytes"] += nbytes

  unsized: list[dict[str, Any]] = []
  for name, dims, typ, _off in tensors:
    dims = tuple(int(d) for d in dims)
    try:
      nbytes = _physical_bytes(math.prod(dims) if dims else 1, typ, tensor_name=name)
    except ValueError as exc:  # a type with no block layout: its bytes are unknown, said, never guessed into a sum
      unsized.append({"tensor_name": name, "ggml_type": typ, "why": str(exc)})
      continue
    total += nbytes
    role, pattern = role_from_tensor_name(name), name_pattern(name)
    if is_limit_role_tensor(name, dims):
      limit_bytes += nbytes
      limit_tensors += 1
      continue
    if role == "embedding" and has_head:
      exclude(pattern, "row_gather", nbytes)
      continue
    if nbytes < SMALL_BYTES and _vector_like(dims) and role != "embedding":  # a tied head is never "small"
      exclude(pattern, "small_vector", nbytes)
      continue
    cols = dims[0] if dims else 1
    reads, note = 1, None
    if len(dims) == 3 and "expert" in role and "shared" not in role:  # GGUF ne = [in, out, n_expert]
      n_expert, rows = dims[2], dims[1]
      reads = min(k_used, n_expert) if k_used else n_expert
      note = f"{reads} of {n_expert} experts read per token" + ("" if k_used else " (no expert_used_count: all counted)")
      if reads < n_expert:
        exclude(pattern, "inactive_experts", nbytes // n_expert * (n_expert - reads),
                f"{n_expert - reads} of {n_expert} experts not read per token (expert_used_count {reads})")
      nbytes = nbytes // n_expert * reads
    else:
      rows = math.prod(dims[1:]) if len(dims) > 1 else 1
    if role == "embedding":
      note = "no output head in the file: the output head reads this table whole"
    key = (pattern, typ, rows, cols)
    row = counted.setdefault(key, {"role": f"{UNCLASSIFIED}: {pattern}", "pattern": pattern, "classifier_role": role,
                                   "quant": GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"), "ggml_type": typ, "rows": rows,
                                   "cols": cols, "tensors": 0, "count": 0, "bytes": 0, "note": note})
    row["tensors"] += 1
    row["count"] += reads
    row["bytes"] += nbytes
  if unsized:
    notes.append(f"{len(unsized)} tensors have a type with no block layout, so their bytes are unknown and neither in "
                 "the limit nor in the file total: the limit is incomplete")
  if not k_used and any("expert" in r["classifier_role"] and "shared" not in r["classifier_role"] for r in counted.values()):
    notes.append(f"{arch}.expert_used_count is absent, so every expert of every stack is counted")
  unclassified = sorted(counted.values(), key=lambda r: -r["bytes"])
  dropped = sorted(excluded.values(), key=lambda r: -r["bytes"])
  out = {
    "rule": "every tensor in the file is counted in the limit or excluded with a stated reason",
    "tensor_count": len(tensors), "total_bytes": total,
    "limit_role_tensors": limit_tensors, "limit_role_bytes": limit_bytes,
    "unclassified": unclassified,
    "unclassified_tensors": sum(r["tensors"] for r in unclassified),
    "unclassified_bytes": sum(r["bytes"] for r in unclassified),
    "excluded": dropped, "excluded_bytes": sum(r["bytes"] for r in dropped),
    "unsized": unsized, "exclusion_rules": dict(EXCLUSIONS), "notes": notes,
  }
  out["counted_bytes"] = limit_bytes + out["unclassified_bytes"]
  if data_bytes is not None and data_bytes > 0:
    out["file"] = {"data_bytes": int(data_bytes), "alignment": int(alignment), "padding_bytes": int(data_bytes) - total}
  elif data_bytes is not None:  # a shape-only file (synth/): the table is checked, there is no data region to check it against
    out["file"] = {"data_bytes": 0, "alignment": int(alignment), "header_only": True}
  return out


def check(ledger:Mapping[str, Any], roles:Iterable[Any]) -> None:
  """The invariant: the bytes the limit counts (the profile's own limit roles plus the unclassified rows) plus the bytes
  excluded with a reason equal every tensor byte in the file, and those fill the file's data region but for padding.
  Raises WeightLedgerError naming the difference."""
  from_roles = 0
  for r in roles:
    role = r if isinstance(r, Mapping) else r.to_json()
    if role.get("role") not in WEIGHT_GEMV_ROLES:
      continue
    try:
      from_roles += _physical_bytes(int(role["rows"]) * int(role["cols"]), int(role["ggml_type"]),
                                    tensor_name=str(role.get("tensor_name"))) * int(role.get("count") or 1)
    except ValueError:  # unsized: listed in ledger["unsized"] and left out of both sides
      continue
  counted = from_roles + int(ledger["unclassified_bytes"])
  diff = counted + int(ledger["excluded_bytes"]) - int(ledger["total_bytes"])
  if diff:
    raise WeightLedgerError(
      f"weight ledger does not add up: counted in the limit {counted} B (limit roles {from_roles} B, unclassified "
      f"{ledger['unclassified_bytes']} B) + excluded {ledger['excluded_bytes']} B = {counted + int(ledger['excluded_bytes'])} B, "
      f"file tensors {ledger['total_bytes']} B: {'+' if diff > 0 else ''}{diff} B "
      f"({'counted twice' if diff > 0 else 'dropped without a reason'}); the ledger's limit roles say {ledger['limit_role_bytes']} B")
  f = ledger.get("file")
  if f and not f.get("header_only") and not ledger.get("unsized"):
    slack = int(f["data_bytes"]) - int(ledger["total_bytes"])
    if slack < 0 or slack > int(f["alignment"]) * max(int(ledger["tensor_count"]), 1):
      raise WeightLedgerError(f"weight ledger does not add up: the tensor table holds {ledger['total_bytes']} B but the "
                              f"file's data region is {f['data_bytes']} B ({slack:+d} B beyond alignment padding of "
                              f"{f['alignment']} B per tensor): a tensor is missing from the table or mis-sized")


def summary_line(ledger:Mapping[str, Any]) -> str:
  """One line for inspect: what is unclassified (counted) and what is excluded, in GB per token."""
  gb = lambda b: f"{b / 1e9:.2f} GB" if b >= 1e7 else f"{b / 1e6:.2f} MB"
  head = (f"weights: {gb(ledger['counted_bytes'])} per token counted in the limit, {gb(ledger['excluded_bytes'])} "
          f"excluded with a reason, {gb(ledger['total_bytes'])} in the file")
  n = ledger["unclassified_tensors"]
  unc = (f"{n} tensors, {gb(ledger['unclassified_bytes'])} per token, not classified; counted in the limit"
         if n else "0 tensors not classified")
  return f"{head}; {unc}"


def detail_lines(ledger:Mapping[str, Any]) -> list[str]:
  """The unclassified rows and the exclusions, one line each."""
  out = [f"  unclassified weight: {r['pattern']} {r['quant']} x{r['tensors']}, {r['bytes'] / 1e6:.2f} MB per token"
         + (f" ({r['note']})" if r.get("note") else "") for r in ledger["unclassified"]]
  out += [f"  excluded {r['reason']}: {r['pattern']} x{r['tensors']}, {r['bytes'] / 1e6:.2f} MB: {r['why']}"
          for r in ledger["excluded"]]
  out += [f"  note: {n}" for n in ledger.get("notes") or []]
  return out


def facts(ledger:Mapping[str, Any] | None) -> dict[str, Any] | None:
  """The ledger as the run's results carry it beside the limit: the totals, the line, and the rows."""
  if not ledger:
    return None
  keep = ("total_bytes", "counted_bytes", "limit_role_bytes", "unclassified_bytes", "unclassified_tensors", "excluded_bytes")
  return {**{k: ledger[k] for k in keep}, "line": summary_line(ledger),
          "unclassified": [{k: r[k] for k in ("pattern", "quant", "tensors", "count", "bytes", "classifier_role")}
                           for r in ledger["unclassified"]],
          "excluded": [{k: r[k] for k in ("pattern", "reason", "tensors", "bytes")} for r in ledger["excluded"]],
          "notes": list(ledger.get("notes") or [])}
