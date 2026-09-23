"""Identity-bound PMC adapters with deliberately narrow calibrated semantics."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.core.facts import Fact

PMC_SCHEMA = "tinygrad.amd_pmc_result.v1"
PROBE_SCHEMA = "tinygrad.mmq_differential_probe.v1"
STORE_SCOPE = "64B output-line write transactions only"


@dataclass(frozen=True)
class StoreProxyEvidence:
  candidate_id: str
  binary_sha256: str
  system_snapshot_id: str
  facts: tuple[Fact, ...]

  def fact(self, name: str) -> Fact | None: return next((f for f in self.facts if f.name == name), None)


@dataclass(frozen=True)
class LoadProxyEvidence:
  candidate_id: str
  binary_sha256: str
  system_snapshot_id: str
  facts: tuple[Fact, ...]

  def fact(self, name: str) -> Fact | None: return next((f for f in self.facts if f.name == name), None)


def adapt_mmq_load_proxy(*, artifact: Mapping[str, Any], candidate_id: str, binary_sha256: str,
                         system_snapshot_id: str, system_facts: Mapping[str, Any]) -> LoadProxyEvidence:
  if artifact.get("schema") != "tinygrad.mmq_load_proxy.v1": raise ValueError("expected MMQ load proxy")
  fingerprint = artifact.get("system_fingerprint", {})
  for key, fact_name in (("gpu_uuid", "gpu.uuid"), ("architecture", "gpu.architecture"), ("compute_units", "gpu.compute_units")):
    if fingerprint.get(key) != system_facts.get(fact_name): raise ValueError(f"canonical system bridge mismatch: {key}")
  if artifact.get("counter_liveness", {}).get("status") != "live": raise ValueError("GL2C_MC_RDREQ is not live")
  row = next((x for x in artifact.get("candidates", []) if x.get("candidate_id") == candidate_id), None)
  if row is None or row.get("binary_sha256") != binary_sha256: raise ValueError("candidate load-proxy identity mismatch")
  samples = row.get("samples", [])
  interval = row.get("measured_request_interval", [])
  floor = row.get("semantic_unique_input_line_floor")
  if len(samples) < 12 or len(interval) != 2 or min(samples) != interval[0] or max(samples) != interval[1] or floor != 58:
    raise ValueError("candidate load-proxy interval is invalid")
  redacted = _redact_timing(artifact)
  source = EvidenceSource("tinygrad", "mmq_load_proxy", "", sha256_json(redacted))
  facts = (
    Fact("mmq.load.external_128b_request_transactions", statistics.median(samples), "modeled", unit="requests",
      sources=(source,), derivation="candidate-specific live GL2C_MC_RDREQ interval",
      uncertainty={"low": interval[0], "high": interval[1], "sample_count": len(samples),
                   "scope": "GL2 external-address 128B read requests; not physical HBM bytes"}),
    Fact("mmq.load.semantic_unique_128b_line_floor", 58, "derived", unit="lines", sources=(source,),
      derivation="identity-bound bounded MMQ allocation/address contract", uncertainty={"low": 58, "high": 58}),
    Fact("mmq.load.microbenchmark_plus4_transfer", False, "derived", sources=(source,),
      derivation="producer calibration_provenance.transfer_to_mmq is false"),
  )
  return LoadProxyEvidence(candidate_id, binary_sha256, system_snapshot_id, facts)


def adapt_global_load_calibration(artifact: Mapping[str, Any]) -> Fact:
  if artifact.get("schema") != PROBE_SCHEMA or artifact.get("probe") != "global_load_transaction_proxy":
    raise ValueError("expected global-load transaction calibration")
  result = artifact.get("calibration_result", {})
  if (result.get("status") != "live" or result.get("all_samples_exact") is not True or
      result.get("supporting_samples", 0) < 12 or result.get("fixed_case_request_overhead") != 4):
    raise ValueError("global-load calibration is not live and exact")
  if artifact.get("production_dispatch_changed") is not False: raise ValueError("calibration changed production dispatch")
  return Fact("calibration.global_load_wg96.mc_rdreq_rule",
    "GL2C_MC_RDREQ - 4 == unique 128B input lines", "modeled", sources=(_source(artifact, "global_load_proxy"),),
    derivation="12 exact controlled samples across stride 1..32",
    uncertainty={"low": 4, "high": 4, "supporting_samples": 12,
                 "scope": "global_load.wg96 generated microbenchmark family only",
                 "mmq_transfer": "blocked pending separate MMQ-specific join"})


def adapt_store_proxy(*, pmc: Mapping[str, Any], calibration: Mapping[str, Any], isa: Mapping[str, Any],
                      ownership: Mapping[str, Any]) -> StoreProxyEvidence:
  if pmc.get("schema") != PMC_SCHEMA or pmc.get("kind") != "candidate_pmc": raise ValueError("expected candidate PMC result")
  if calibration.get("schema") != PROBE_SCHEMA or calibration.get("probe") != "store_active_lane_transaction_calibration":
    raise ValueError("expected store transaction calibration")
  identity = {key: pmc.get(key) for key in ("candidate_id", "binary_sha256", "system_snapshot_id")}
  if any(not value for value in identity.values()): raise ValueError("PMC identity is incomplete")
  for name, artifact in (("isa", isa), ("ownership", ownership)):
    for key, value in identity.items():
      if artifact.get(key) != value: raise ValueError(f"{name} {key} mismatch")
  if calibration.get("system_snapshot_id") != identity["system_snapshot_id"]: raise ValueError("calibration system mismatch")
  result = calibration.get("calibration_result", {})
  if not _calibration_is_exact(calibration, result):
    raise ValueError("GL2C_MC_WRREQ calibration is not live and exact")
  samples = []
  for sample in pmc.get("samples", []):
    if sample.get("status") == "live" and isinstance(sample.get("counters", {}).get("GL2C_MC_WRREQ"), (int, float)):
      samples.append(float(sample["counters"]["GL2C_MC_WRREQ"]))
  if len(samples) < int(pmc.get("repetitions", 0)) or len(samples) < 3: raise ValueError("insufficient live GL2C_MC_WRREQ samples")
  output_elements = _store_count(ownership.get("expected_stores"))
  if output_elements != 256: raise ValueError("store proxy is calibrated only for the 256-element bounded output")
  store_sites = int(ownership.get("final_isa_store_instruction_sites", -1))
  expected_sites = 256 if ".gated_matrix." in identity["candidate_id"] else 1 if ".direct_owner." in identity["candidate_id"] else -1
  if store_sites != expected_sites: raise ValueError("candidate/store-site semantic mismatch")
  pmc_source = _source(pmc, "candidate_pmc")
  calibration_source = _source(calibration, "store_transaction_calibration")
  isa_source, ownership_source = _source(isa, "final_isa"), _source(ownership, "ownership")
  median, low, high = statistics.median(samples), min(samples), max(samples)
  facts = (
    Fact("mmq.store.output_line_transactions_64b", median, "modeled", unit="transactions",
         sources=(pmc_source, calibration_source), derivation="live GL2C_MC_WRREQ mapped by exact controlled probe",
         uncertainty={"low": low, "high": high, "sample_count": len(samples), "scope": STORE_SCOPE,
                      "outside_scope": "unknown"}),
    Fact("mmq.store.executed_lane_stores", 8192, "derived", unit="lane_stores",
         sources=(isa_source, ownership_source), derivation="256 waves * 32 semantically active output lanes",
         uncertainty={"low": 8192, "high": 8192, "scope": "bounded MMQ output writeback"}),
    Fact("mmq.store.output_elements", 256, "derived", unit="elements", sources=(ownership_source,),
         derivation="unique bounded output ownership", uncertainty={"low": 256, "high": 256}),
    Fact("mmq.store.false_sites_per_wave", 255 if expected_sites == 256 else 0, "derived", unit="sites_per_wave",
         sources=(isa_source, ownership_source), derivation="final ISA sites minus the one semantically selected site",
         uncertainty={"low": 255 if expected_sites == 256 else 0, "high": 255 if expected_sites == 256 else 0}),
    Fact("mmq.store.transaction_proxy_scope", STORE_SCOPE, "modeled", sources=(calibration_source,),
         derivation="scope frozen by store calibration address contract"),
  )
  return StoreProxyEvidence(str(identity["candidate_id"]), str(identity["binary_sha256"]),
                            str(identity["system_snapshot_id"]), facts)


def _store_count(value: Any) -> int | None:
  if isinstance(value, int): return value
  if isinstance(value, Mapping): return value.get("store_count")
  return None


def _source(artifact: Mapping[str, Any], tool: str) -> EvidenceSource:
  return EvidenceSource("tinygrad", tool, "", sha256_json(artifact))


def _redact_timing(value: Any) -> Any:
  if isinstance(value, Mapping): return {k: _redact_timing(v) for k, v in value.items() if k not in ("median_ms", "elapsed_ms", "samples_ms")}
  if isinstance(value, list): return [_redact_timing(x) for x in value]
  return value


def _calibration_is_exact(calibration: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
  if result.get("status") == "live" and result.get("all_samples_exact") is True and result.get("supporting_samples", 0) >= 2:
    return True
  rows = [sample for point in calibration.get("points", []) for sample in point.get("samples", [])
          if sample.get("status") == "live"]
  return len(rows) >= 2 and all(sample.get("counters", {}).get("GL2C_MC_WRREQ") == sample.get("unique_64b_lines")
                                for sample in rows)


__all__ = ["STORE_SCOPE", "LoadProxyEvidence", "StoreProxyEvidence", "adapt_global_load_calibration",
           "adapt_mmq_load_proxy", "adapt_store_proxy"]
