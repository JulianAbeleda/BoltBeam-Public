"""BoltBeam-owned, versioned route-selection policy authority."""
from __future__ import annotations
import json
import pathlib
from collections.abc import Mapping

from boltbeam.core.canonical import canonical_json as _canonical_json, sha256_hex as _sha256_hex, pretty_json

_ASSET = pathlib.Path(__file__).with_name("assets") / "route_manifest.v1.json"
_DIGEST = _ASSET.with_suffix(_ASSET.suffix + ".sha256")

def _canonical_bytes(value: object) -> bytes:
  return (_canonical_json(value) + "\n").encode("ascii")

def load_authoritative_asset(path: pathlib.Path = _ASSET) -> dict:
  """Read the policy asset and fail closed on stale, missing, or mismatched hashes."""
  digest_path = path.with_suffix(path.suffix + ".sha256")
  try:
    raw, expected = path.read_bytes(), digest_path.read_text().split()[0]
  except (OSError, IndexError) as exc:
    raise RuntimeError(f"route policy unavailable: {path}") from exc
  actual = _sha256_hex(raw)
  if actual != expected:
    raise RuntimeError(f"route policy hash mismatch: expected {expected}, got {actual}")
  try: asset = json.loads(raw)
  except json.JSONDecodeError as exc: raise RuntimeError("route policy JSON is invalid") from exc
  if asset.get("schema") != "boltbeam.route-manifest.v1" or asset.get("authority") != "BoltBeam" or asset.get("version") != 1:
    raise RuntimeError("route policy schema/authority/version mismatch")
  if _canonical_bytes(asset) != raw: raise RuntimeError("route policy is not canonical serialized JSON")
  if not isinstance(asset.get("routes"), dict) or not isinstance(asset.get("refuted_axes"), list):
    raise RuntimeError("route policy missing routes/refuted axes")
  return asset

_ASSET_DATA = load_authoritative_asset()
ROUTES = _ASSET_DATA["routes"]
REFUTED = _ASSET_DATA["refuted_axes"]
PROFILE_DECODE = "qwen3_8b_q4_k_m_gfx1100_decode"
PROFILE_PREFILL = "qwen3_8b_q4_k_m_gfx1100_prefill"
ROUTE_PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated", "hand_authored_uop_template", "compiler_primitive_spec_owned", "external_handwritten_kernel", "rollback_oracle")
FINAL_DEFAULT_PROVENANCE = {"machine_authored_generated", "tinygrad_scheduler_generated"}
TRANSITIONAL_DEFAULT_PROVENANCE = {"hand_authored_uop_template"}
FORBIDDEN_DEFAULT_PROVENANCE = {"external_handwritten_kernel", "rollback_oracle"}

def derive_purity_status(status: str, provenance: str) -> str:
  if status in ("promoted_default", "default_shipped") and provenance in FINAL_DEFAULT_PROVENANCE: return "search_generated_promoted"
  return "refuted" if status == "refuted" else "research"

def route(route_id: str) -> dict:
  if route_id not in ROUTES: raise KeyError(f"unknown route_id {route_id!r}; known: {sorted(ROUTES)}")
  return ROUTES[route_id]
def default_routes() -> list[str]: return [rid for rid, row in ROUTES.items() if row["status"] in ("promoted_default", "default_shipped")]
def routes_by_status(status: str) -> list[str]: return [rid for rid, row in ROUTES.items() if row["status"] == status]
def route_provenance(route_id: str) -> str:
  value = str(route(route_id).get("provenance", ""))
  if value not in ROUTE_PROVENANCE: raise ValueError(f"route {route_id!r} has invalid provenance {value!r}")
  return value
def default_purity_report() -> dict:
  rows=[{"route_id":rid,"status":route(rid)["status"],"provenance":route_provenance(rid),"replacement_scope":route(rid).get("replacement_scope",""),"final_default_allowed":route_provenance(rid) in FINAL_DEFAULT_PROVENANCE} for rid in default_routes()]
  forbidden=[r["route_id"] for r in rows if r["provenance"] in FORBIDDEN_DEFAULT_PROVENANCE]
  transitional=[r["route_id"] for r in rows if r["provenance"] in TRANSITIONAL_DEFAULT_PROVENANCE]
  return {"verdict":"TINYGRAD_DEFAULT_PURITY_PASS" if not forbidden and not transitional else "TINYGRAD_DEFAULT_PURITY_FAIL","default_routes":default_routes(),"rows":rows,"forbidden_default_routes":forbidden,"transitional_default_routes":transitional,"final_default_allowed_provenance":sorted(FINAL_DEFAULT_PROVENANCE)}
def validate_manifest() -> list[str]:
  errors=[]
  for rid,row in ROUTES.items():
    p=row.get("provenance")
    if p not in ROUTE_PROVENANCE: errors.append(f"{rid}: invalid or missing provenance {p!r}")
    if row["status"] in ("promoted_default", "default_shipped"):
      if p == "rollback_oracle": errors.append(f"{rid}: default route cannot be provenance=rollback_oracle")
      if p in ("hand_authored_uop_template", "external_handwritten_kernel") and not row.get("replacement_scope"):
        errors.append(f"{rid}: non-pure default provenance={p} requires replacement_scope")
    if row.get("research_only"):
      if row["status"] != "research": errors.append(f"{rid}: research_only route cannot claim final status {row["status"]!r}")
      if not {"correctness evidence", "resource evidence", "timing evidence"}.issubset({part.strip() for part in str(row.get("authority_gate", "")).split("+")}):
        errors.append(f"{rid}: research_only route must require correctness/resource/timing evidence")
    if "purity_status" in row and row["purity_status"] != derive_purity_status(row["status"], str(p)): errors.append(f"{rid}: purity_status drift")
  return errors
def to_manifest_dict() -> dict:
  return {"_schema":"default route manifest (PMS-R1)","generated_by":"boltbeam.policy.route_manifest","policy_asset_version":1,"policy_asset_sha256":_sha256_hex(_ASSET.read_bytes()),"profiles":{"decode":PROFILE_DECODE,"prefill":PROFILE_PREFILL},"provenance_vocabulary":list(ROUTE_PROVENANCE),"routes":ROUTES,"refuted_axes":REFUTED,"default_routes":default_routes(),"promoted_defaults":routes_by_status("promoted_default"),"owned_defaults":routes_by_status("default_shipped"),"default_purity":default_purity_report()}
def dump(out_path: str | None = None) -> str:
  p=pathlib.Path(out_path) if out_path else pathlib.Path(__file__).resolve().parents[2] / "bench/default-route-manifest.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(pretty_json(to_manifest_dict())); return str(p)
def dump_refuted(out_path: str | None = None) -> str:
  p=pathlib.Path(out_path) if out_path else pathlib.Path(__file__).resolve().parents[2] / "bench/qk-search-spaces/refuted_axes.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(pretty_json({"_schema":"canonical refuted axes (generated FROM BoltBeam route policy)","generated_by":"boltbeam.policy.route_manifest.dump_refuted","refuted_axes":REFUTED})); return str(p)
