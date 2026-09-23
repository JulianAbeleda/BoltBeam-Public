#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any


def load_json(path:pathlib.Path) -> Any:
  return json.loads(path.read_text())


def write_json(path:pathlib.Path, obj:Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _shape(row:dict[str, Any]) -> tuple[int, int]:
  shape = row.get("shape") or {}
  return int(shape["rows"]), int(shape["cols"])


def _result(row:dict[str, Any]) -> dict[str, Any]:
  lowering = row["current_lowering"]
  out = {
    "candidate": lowering.get("candidate_name"),
    "device_ms": lowering.get("device_ms"),
    "family": lowering.get("family"),
    "format": row.get("format"),
    "gemv_max_abs": lowering.get("gemv_max_abs"),
    "opts": lowering.get("opts", []),
    "parts": lowering.get("parts"),
    "quant_gbs": lowering.get("quant_gbs"),
    "requires": lowering.get("requires", []),
    "status": lowering.get("result_status"),
    "tensor": row.get("tensor"),
    "unpack_max_abs": lowering.get("unpack_max_abs"),
    "wall_ms": lowering.get("wall_ms"),
  }
  return {k: v for k, v in out.items() if v is not None}


def policy_entry_from_descriptor(row:dict[str, Any], descriptor_set:dict[str, Any]) -> dict[str, Any]:
  rows, cols = _shape(row)
  lowering = row["current_lowering"]
  layout = row.get("layout") or {}
  storage = row.get("storage") or {}
  family = lowering.get("family")
  candidate_name = lowering.get("candidate_name") or lowering.get("winner")
  entry = {
    "candidate": {
      "activation": row.get("activation_dtype"),
      "family": family,
      "name": candidate_name,
      "opts": lowering.get("opts", []),
      "parts": int(lowering.get("parts") or 0),
      "reduction": lowering.get("reduction"),
      "requires": lowering.get("requires", []),
    },
    "descriptor": {
      "block_bytes": layout.get("block_bytes"),
      "block_elems": layout.get("block_elems"),
      "byte_start": layout.get("byte_start"),
      "cols": cols,
      "data_start": layout.get("data_start"),
      "dtype_activation": row.get("activation_dtype"),
      "dtype_output": row.get("output_dtype"),
      "format": row.get("format"),
      "ggml_type": int(row["ggml_type"]),
      "model": descriptor_set.get("model"),
      "packed_bytes": layout.get("packed_bytes"),
      "role": row.get("role"),
      "rows": rows,
      "tensor": row.get("tensor"),
      "tensor_offset": layout.get("tensor_offset"),
    },
    "format": row.get("format"),
    "key": {
      "activation": row.get("activation_dtype"),
      "format": row.get("format"),
      "generator_version": descriptor_set.get("generator_version"),
      "ggml_type": int(row["ggml_type"]),
      "shape": [rows, cols],
    },
    "metric": lowering.get("metric"),
    "metric_value": lowering.get("metric_value"),
    "policy_reason": lowering.get("policy_reason"),
    "reason": "reproduced from qk_semantic_descriptor_set",
    "result": _result(row),
    "scope": storage.get("scope") or "shape",
    "shape": [rows, cols],
    "storage": {
      "benefit_ms": storage.get("benefit_ms"),
      "benefit_ms_per_mb": storage.get("benefit_ms_per_mb"),
      "decision": storage.get("scope") or "shape_policy",
      "persistent_bytes": storage.get("policy_persistent_bytes") or 0,
    },
    "tensor": row.get("tensor"),
    "winner": lowering.get("winner"),
  }
  return entry


def build_policy_from_descriptor(descriptor_set:dict[str, Any]) -> dict[str, Any]:
  if descriptor_set.get("kind") != "qk_semantic_descriptor_set":
    raise ValueError("expected kind=qk_semantic_descriptor_set")
  entries = descriptor_set.get("descriptors")
  if not isinstance(entries, list) or not entries:
    raise ValueError("descriptor set must contain a non-empty descriptors list")
  return {
    "kind": "qk_generated_policy",
    "generator_version": descriptor_set.get("generator_version"),
    "commit": descriptor_set.get("policy_commit"),
    "created_at": "reproduced-from-qk-semantic-descriptor",
    "model": descriptor_set.get("model"),
    "model_size": descriptor_set.get("model_size"),
    "source_descriptor": descriptor_set.get("source_policy"),
    "storage_policy": descriptor_set.get("storage_policy") or {},
    "entries": [policy_entry_from_descriptor(row, descriptor_set) for row in entries],
  }


def runtime_entries(policy:dict[str, Any]) -> dict[str, dict[str, Any]]:
  if policy.get("kind") != "qk_generated_policy":
    raise ValueError("expected kind=qk_generated_policy")
  out: dict[str, dict[str, Any]] = {}
  for entry in policy.get("entries", []):
    desc, cand = entry.get("descriptor") or {}, entry.get("candidate") or {}
    scope = entry.get("scope") or "shape"
    tensor = str(desc.get("tensor") or entry.get("tensor") or "")
    key_parts = [scope]
    if scope == "tensor": key_parts.append(tensor)
    key_parts += [str(int(desc["ggml_type"])), str(int(desc["rows"])), str(int(desc["cols"]))]
    key = "|".join(key_parts)
    value = {
      "scope": scope,
      "tensor": tensor if scope == "tensor" else None,
      "ggml_type": int(desc["ggml_type"]),
      "rows": int(desc["rows"]),
      "cols": int(desc["cols"]),
      "winner": entry.get("winner"),
      "family": cand.get("family"),
      "parts": int(cand.get("parts", 0)),
      "opts": list(cand.get("opts", [])),
    }
    if key in out and out[key] != value:
      raise ValueError(f"conflicting runtime entry for {key}: {out[key]} vs {value}")
    out[key] = value
  if not out: raise ValueError("policy has no runtime entries")
  return out


def diff_policies(accepted:dict[str, Any], reproduced:dict[str, Any]) -> dict[str, Any]:
  left, right = runtime_entries(accepted), runtime_entries(reproduced)
  changed = []
  for key in sorted(set(left) & set(right)):
    if left[key] != right[key]:
      changed.append({"key": key, "accepted": left[key], "reproduced": right[key]})
  diff = {
    "kind": "qk_descriptor_policy_diff",
    "semantic_equal": not changed and set(left) == set(right),
    "accepted_entries": len(left),
    "reproduced_entries": len(right),
    "missing_runtime_keys": sorted(set(left) - set(right)),
    "extra_runtime_keys": sorted(set(right) - set(left)),
    "changed_runtime_entries": changed,
    "metadata_note": "Only runtime dispatch semantics are compared; timestamps, benchmark metadata, and archival fields may differ.",
  }
  return diff


def diff_markdown(diff:dict[str, Any], *, label:str="policy") -> str:
  lines = [
    f"# QK Descriptor Policy Diff: {label}",
    "",
    f"- semantic equal: `{diff['semantic_equal']}`",
    f"- accepted runtime entries: `{diff['accepted_entries']}`",
    f"- reproduced runtime entries: `{diff['reproduced_entries']}`",
    f"- missing keys: `{len(diff['missing_runtime_keys'])}`",
    f"- extra keys: `{len(diff['extra_runtime_keys'])}`",
    f"- changed entries: `{len(diff['changed_runtime_entries'])}`",
    "",
  ]
  if diff["missing_runtime_keys"]:
    lines += ["## Missing Runtime Keys", ""]
    lines += [f"- `{key}`" for key in diff["missing_runtime_keys"]]
    lines.append("")
  if diff["extra_runtime_keys"]:
    lines += ["## Extra Runtime Keys", ""]
    lines += [f"- `{key}`" for key in diff["extra_runtime_keys"]]
    lines.append("")
  if diff["changed_runtime_entries"]:
    lines += ["## Changed Runtime Entries", "", "```json", json.dumps(diff["changed_runtime_entries"], indent=2, sort_keys=True), "```", ""]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Reproduce a QK generated policy from a semantic descriptor set")
  parser.add_argument("--descriptor", type=pathlib.Path, required=True)
  parser.add_argument("--accepted", type=pathlib.Path)
  parser.add_argument("--policy-json", type=pathlib.Path, required=True)
  parser.add_argument("--diff-json", type=pathlib.Path)
  parser.add_argument("--diff-md", type=pathlib.Path)
  args = parser.parse_args()

  descriptor = load_json(args.descriptor.expanduser())
  policy = build_policy_from_descriptor(descriptor)
  write_json(args.policy_json, policy)
  if args.accepted is not None:
    diff = diff_policies(load_json(args.accepted.expanduser()), policy)
    if args.diff_json: write_json(args.diff_json, diff)
    if args.diff_md:
      args.diff_md.parent.mkdir(parents=True, exist_ok=True)
      args.diff_md.write_text(diff_markdown(diff, label=descriptor.get("model_size", args.descriptor.stem.upper())))
    if not diff["semantic_equal"]:
      raise SystemExit(f"{args.descriptor}: reproduced policy differs semantically from {args.accepted}")
  elif args.diff_json or args.diff_md:
    raise ValueError("--diff-json/--diff-md require --accepted")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
