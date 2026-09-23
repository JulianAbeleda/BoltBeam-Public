"""Interval-valued analytical cycle accounting for bounded MMQ kernels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.core.facts import TruthStatus
from boltbeam.perf.calibration import CalibrationProfile, Interval
from boltbeam.perf.isa_graph import ISAGraph
from boltbeam.perf.occupancy import OccupancyBound, derive_occupancy
from boltbeam.perf.pmc_proxy import STORE_SCOPE, LoadProxyEvidence, StoreProxyEvidence

MODEL_SCHEMA = "boltbeam.mmq_cycle_model.v1"
PREDICTION_SCHEMA = "boltbeam.mmq_prediction.v1"


def cycle_model_manifest(calibration: CalibrationProfile) -> dict[str, Any]:
  """Durable description of the analytical rules, separate from a candidate prediction."""
  body = {"schema": MODEL_SCHEMA, "model_version": calibration.model_version,
          "calibration_id": calibration.calibration_id,
          "overlap_rule": "max(dependency_floor,issue_floor,memory_completion)+synchronization",
          "kernel_rule": "launch+wave_cycles*((full_batches)+tail_batch_cost)",
          "required_static_evidence": ["ordered_isa", "structured_dependencies", "physical_transactions",
                                       "vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup", "grid"],
          "timing_inputs_allowed": False}
  return {**body, "cycle_model_id": sha256_json(body)}


@dataclass(frozen=True)
class CyclePrediction:
  system_snapshot_id: str
  calibration_id: str
  candidate_id: str
  binary_sha256: str
  cycles: Interval | None
  milliseconds: Interval | None
  components: Mapping[str, Any]
  occupancy: OccupancyBound
  assumptions: tuple[str, ...] = ()
  blockers: tuple[str, ...] = ()
  model_version: str = "mmq-cycle-v1"

  @property
  def complete(self) -> bool: return self.cycles is not None and self.milliseconds is not None and not self.blockers

  @property
  def prediction_id(self) -> str:
    return sha256_json({"system_snapshot_id": self.system_snapshot_id, "calibration_id": self.calibration_id,
      "candidate_id": self.candidate_id, "binary_sha256": self.binary_sha256, "model_version": self.model_version,
      "cycles": self.cycles.to_json() if self.cycles else None,
      "milliseconds": self.milliseconds.to_json() if self.milliseconds else None,
      "components": self.components, "assumptions": list(self.assumptions), "blockers": list(self.blockers)})

  def to_json(self) -> dict[str, Any]:
    return {"schema": PREDICTION_SCHEMA, "prediction_id": self.prediction_id,
      "system_snapshot_id": self.system_snapshot_id, "calibration_id": self.calibration_id,
      "candidate_id": self.candidate_id, "binary_sha256": self.binary_sha256, "model_version": self.model_version,
      "cycles": self.cycles.to_json() if self.cycles else None,
      "milliseconds": self.milliseconds.to_json() if self.milliseconds else None,
      "components": dict(self.components), "occupancy": _occupancy_json(self.occupancy),
      "assumptions": list(self.assumptions), "blockers": list(self.blockers)}


def predict_cycles(*, graph: ISAGraph, resources: Mapping[str, Any], launch: Mapping[str, Any],
                   calibration: CalibrationProfile, candidate_id: str, binary_sha256: str,
                   system_snapshot_id: str, store_proxy: StoreProxyEvidence | None = None,
                   load_proxy: LoadProxyEvidence | None = None) -> CyclePrediction:
  blockers = []
  if system_snapshot_id != calibration.system_snapshot_id: blockers.append("system/calibration identity mismatch")
  if not graph.complete: blockers.extend(graph.blockers or ("ISA graph incomplete",))
  if any(node.epoch in ("", "unknown") for node in graph.nodes): blockers.append("instruction epoch assignment missing")
  if not binary_sha256: blockers.append("binary identity missing")
  if any(key in launch for key in ("samples_ms", "timings_ms", "median_ms", "candidate_ms")):
    raise ValueError("prediction inputs must not contain measured candidate timing")
  occupancy = derive_occupancy(resources, calibration)
  blockers.extend(occupancy.blockers)
  grid = launch.get("grid_workgroups")
  if not isinstance(grid, int) or grid <= 0: blockers.append("grid_workgroups missing")
  cu = _point(calibration, "gpu.compute_units")
  clock = calibration.interval("gpu.clock_hz")
  launch_cycles = calibration.interval("perf.launch_cycles")
  tail_efficiency = calibration.interval("perf.tail_efficiency")
  required_domains = sorted({n.issue_domain for n in graph.nodes})
  rates = {domain: calibration.interval(f"perf.issue_rate.{domain}") for domain in required_domains}
  latencies = {kind: calibration.interval(f"perf.latency.{kind}") for kind in sorted({n.instruction_class for n in graph.nodes})}
  for name, value in (("gpu.compute_units", cu), ("gpu.clock_hz", clock), ("perf.launch_cycles", launch_cycles),
                      ("perf.tail_efficiency", tail_efficiency)):
    if value is None: blockers.append(f"missing calibration {name}")
  blockers.extend(f"missing calibration perf.issue_rate.{name}" for name, value in rates.items() if value is None)
  blockers.extend(f"missing calibration perf.latency.{name}" for name, value in latencies.items() if value is None)
  used_names = ("gpu.compute_units", "gpu.clock_hz", "perf.launch_cycles", "perf.tail_efficiency",
                *(f"perf.issue_rate.{name}" for name in rates), *(f"perf.latency.{name}" for name in latencies))
  for name in used_names:
    fact = calibration.fact(name)
    if fact is not None and fact.status not in {TruthStatus.MEASURED.value, TruthStatus.DERIVED.value, TruthStatus.MODELED.value}:
      blockers.append(f"uncalibrated truth status for {name}: {fact.status}")
  load_nodes = [n for n in graph.nodes if n.instruction_class == "global_load"]
  store_nodes = [n for n in graph.nodes if n.instruction_class == "global_store"]
  load_proxy_blockers = _load_proxy_blockers(load_proxy, candidate_id, binary_sha256, system_snapshot_id)
  if any(n.transactions is None for n in load_nodes) and (load_proxy is None or load_proxy_blockers):
    blockers.append("global-load transaction estimate missing")
  blockers.extend(load_proxy_blockers)
  proxy_blockers = _store_proxy_blockers(store_proxy, candidate_id, binary_sha256, system_snapshot_id)
  store_transactions_known = not any(n.transactions is None for n in store_nodes)
  store_lanes_known = not any(n.active_lanes is None for n in store_nodes)
  if not store_transactions_known and (store_proxy is None or proxy_blockers): blockers.append("output-store transaction estimate missing")
  if not store_lanes_known and (store_proxy is None or proxy_blockers): blockers.append("executed store lane estimate missing")
  blockers.extend(proxy_blockers)
  if blockers:
    return CyclePrediction(system_snapshot_id, calibration.calibration_id, candidate_id, binary_sha256, None, None,
                           {}, occupancy,
                           assumptions=tuple((*graph.assumptions, *(("GL2C_MC_WRREQ calibrated only for " + STORE_SCOPE,) if store_proxy else ()))),
                           blockers=tuple(dict.fromkeys(blockers)))
  scenarios = [_scenario(graph, occupancy, grid, int(cu), launch_cycles, clock, tail_efficiency, rates, latencies, mode)
               for mode in ("optimistic", "median", "pessimistic")]
  cycles = Interval(scenarios[0]["kernel_cycles"], scenarios[1]["kernel_cycles"], scenarios[2]["kernel_cycles"])
  milliseconds = Interval(scenarios[0]["milliseconds"], scenarios[1]["milliseconds"], scenarios[2]["milliseconds"])
  components = {"optimistic": scenarios[0], "median": scenarios[1], "pessimistic": scenarios[2],
                "instruction_counts": graph.counts(), "model_schema": MODEL_SCHEMA,
                "store": _store_summary(graph, store_proxy), "load": _load_summary(graph, load_proxy)}
  return CyclePrediction(system_snapshot_id, calibration.calibration_id, candidate_id, binary_sha256,
                         cycles, milliseconds, components, occupancy,
                         assumptions=tuple((*graph.assumptions, *(("GL2C_MC_WRREQ calibrated only for " + STORE_SCOPE,) if store_proxy else ()))))


def _scenario(graph, occupancy, grid, cu, launch_cycles, clock, tail_efficiency, rates, latencies, mode):
  latency = {key: _pick(value, mode, cost=True) for key, value in latencies.items()}
  issue_rate = {key: _pick(value, mode, cost=False) for key, value in rates.items()}
  path = {}
  for node in graph.nodes:
    path[node.index] = latency[node.instruction_class] + max((path.get(dep, 0.0) for dep in node.dependencies), default=0.0)
  dependency_floor = max(path.values(), default=0.0)
  domain_counts = {domain: sum(node.issue_domain == domain for node in graph.nodes) for domain in rates}
  issue_by_domain = {domain: count / issue_rate[domain] for domain, count in domain_counts.items()}
  issue_floor = max(issue_by_domain.values(), default=0.0)
  memory_completion = max((latency[n.instruction_class] + 1 / issue_rate[n.issue_domain]
                           for n in graph.nodes if n.instruction_class in ("global_load", "global_store", "lds_load", "lds_store")), default=0.0)
  sync = sum(latency[n.instruction_class] for n in graph.nodes if n.instruction_class in ("barrier", "waitcnt"))
  wave_cycles = max(dependency_floor, issue_floor, memory_completion) + sync
  capacity = cu * occupancy.resident_workgroups_per_cu
  batches = math.ceil(grid / capacity)
  tail = (grid - (batches - 1) * capacity) / capacity
  tail_batch_cost = max(tail, _pick(tail_efficiency, mode, cost=True))
  execution = wave_cycles * ((batches - 1) + tail_batch_cost)
  total = _pick(launch_cycles, mode, cost=True) + execution
  hz = _pick(clock, mode, cost=False)
  return {"dependency_floor": dependency_floor, "issue_floor": issue_floor, "issue_by_domain": issue_by_domain,
          "memory_completion": memory_completion, "synchronization": sync, "wave_cycles": wave_cycles,
          "batches": batches, "tail_fraction": tail, "tail_batch_cost": tail_batch_cost,
          "execution_cycles": execution, "kernel_cycles": total,
          "milliseconds": total / hz * 1000.0}


def _store_summary(graph: ISAGraph, proxy: StoreProxyEvidence | None) -> dict[str, Any]:
  stores = [n for n in graph.nodes if n.instruction_class == "global_store"]
  out = {"static_sites": len(stores), "executed_lanes": sum(n.active_lanes or 0 for n in stores),
         "transactions": sum(n.transactions or 0 for n in stores), "transaction_scope": "instruction exact"}
  if proxy:
    out.update({"executed_lanes": proxy.fact("mmq.store.executed_lane_stores").value,
                "transactions": proxy.fact("mmq.store.output_line_transactions_64b").value,
                "transaction_interval": proxy.fact("mmq.store.output_line_transactions_64b").uncertainty,
                "transaction_scope": STORE_SCOPE,
                "false_sites_per_wave": proxy.fact("mmq.store.false_sites_per_wave").value})
  return out


def _store_proxy_blockers(proxy: StoreProxyEvidence | None, candidate_id: str, binary_sha256: str,
                          system_snapshot_id: str) -> list[str]:
  if proxy is None: return []
  blockers = []
  if proxy.candidate_id != candidate_id: blockers.append("store proxy candidate mismatch")
  if proxy.binary_sha256 != binary_sha256: blockers.append("store proxy binary mismatch")
  if proxy.system_snapshot_id != system_snapshot_id: blockers.append("store proxy system mismatch")
  scope = proxy.fact("mmq.store.transaction_proxy_scope")
  transactions = proxy.fact("mmq.store.output_line_transactions_64b")
  lanes = proxy.fact("mmq.store.executed_lane_stores")
  if scope is None or scope.value != STORE_SCOPE: blockers.append("store proxy scope mismatch")
  if transactions is None or lanes is None: blockers.append("store proxy facts incomplete")
  elif transactions.status != "modeled" or lanes.status != "derived": blockers.append("store proxy truth status mismatch")
  return blockers


def _load_summary(graph: ISAGraph, proxy: LoadProxyEvidence | None) -> dict[str, Any]:
  loads = [n for n in graph.nodes if n.instruction_class == "global_load"]
  out = {"static_sites": len(loads), "transactions": sum(n.transactions or 0 for n in loads),
         "transaction_scope": "instruction exact"}
  if proxy:
    fact = proxy.fact("mmq.load.external_128b_request_transactions")
    out.update({"transactions": fact.value, "transaction_interval": fact.uncertainty,
                "semantic_unique_line_floor": proxy.fact("mmq.load.semantic_unique_128b_line_floor").value,
                "transaction_scope": "GL2 external-address 128B requests; not physical HBM bytes"})
  return out


def _load_proxy_blockers(proxy: LoadProxyEvidence | None, candidate_id: str, binary_sha256: str,
                         system_snapshot_id: str) -> list[str]:
  if proxy is None: return []
  blockers = []
  if proxy.candidate_id != candidate_id: blockers.append("load proxy candidate mismatch")
  if proxy.binary_sha256 != binary_sha256: blockers.append("load proxy binary mismatch")
  if proxy.system_snapshot_id != system_snapshot_id: blockers.append("load proxy system mismatch")
  requests, floor, transfer = (proxy.fact("mmq.load.external_128b_request_transactions"),
    proxy.fact("mmq.load.semantic_unique_128b_line_floor"), proxy.fact("mmq.load.microbenchmark_plus4_transfer"))
  if requests is None or floor is None or transfer is None: blockers.append("load proxy facts incomplete")
  elif requests.status != "modeled" or transfer.value is not False: blockers.append("load proxy scope mismatch")
  return blockers


def _pick(interval: Interval, mode: str, *, cost: bool) -> float:
  if mode == "median": return interval.median
  if mode == "optimistic": return interval.low if cost else interval.high
  return interval.high if cost else interval.low


def _point(profile: CalibrationProfile, name: str) -> float | None:
  value = profile.interval(name); return value.median if value else None


def _occupancy_json(value: OccupancyBound) -> dict[str, Any]:
  return {"waves_per_workgroup": value.waves_per_workgroup, "resident_workgroups_per_cu": value.resident_workgroups_per_cu,
          "resident_waves_per_cu": value.resident_waves_per_cu, "limiting_resource": value.limiting_resource,
          "limits": dict(value.limits), "blockers": list(value.blockers)}


__all__ = ["CyclePrediction", "cycle_model_manifest", "predict_cycles"]
