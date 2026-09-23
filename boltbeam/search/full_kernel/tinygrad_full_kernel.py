"""One subprocess adapter for the target-neutral tinygrad search provider."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib, json, math, os, pathlib, statistics, subprocess, time
import select
from collections.abc import Mapping
from typing import Any

from boltbeam.plan.resolved_target import resolved_target_document, validate_candidate_target, validate_resolved_target_document
from boltbeam.search.spec import FullKernelCandidate

_COMPATIBILITY_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "tinygrad_search_compatibility.json"
_COMPATIBILITY = json.loads(_COMPATIBILITY_PATH.read_text())
if _COMPATIBILITY.get("schema") != "boltbeam.cross_repo_compatibility.v1":
  raise RuntimeError(f"{_COMPATIBILITY_PATH} has unsupported compatibility schema")
PROTOCOL = _COMPATIBILITY["required_interfaces"]["provider_protocol"]
ACTIONS = tuple(_COMPATIBILITY["required_interfaces"]["provider_actions"])
DEFAULT_PROVIDER_RELPATH = _COMPATIBILITY["entrypoint"]
DEFAULT_PROVIDER_ARGS = tuple(_COMPATIBILITY["command"][2:])
_INVALID_ADMISSION_CODES = frozenset(("identity_mismatch", "admission_rejected"))
_UNSUPPORTED_CANDIDATE_CODES = frozenset(("unsupported_plan", "resource_limit"))


@dataclass(frozen=True)
class TinygradWorkerConfig:
  root: pathlib.Path
  python: str = "python3"
  timeout_s: float = 30.0
  worker_relpath: str = DEFAULT_PROVIDER_RELPATH
  backend: str = "METAL"
  requested_provider_revision: str | None = None

  def __post_init__(self):
    root = pathlib.Path(self.root).expanduser().resolve()
    if self.timeout_s <= 0: raise ValueError("Tinygrad provider timeout_s must be positive")
    if not self.python: raise ValueError("Tinygrad provider python must be non-empty")
    if not self.worker_relpath: raise ValueError("Tinygrad provider path must be non-empty")
    if not self.backend: raise ValueError("Tinygrad provider backend must be non-empty")
    if self.requested_provider_revision is not None and not self.requested_provider_revision:
      raise ValueError("requested_provider_revision must be non-empty when supplied")
    object.__setattr__(self, "root", root)

  @property
  def worker_path(self) -> pathlib.Path:
    path = pathlib.Path(self.worker_relpath).expanduser()
    return path.resolve() if path.is_absolute() else (self.root / path).resolve()

  @property
  def command(self) -> tuple[str, ...]:
    args = tuple(self.backend if value == "METAL" else value for value in DEFAULT_PROVIDER_ARGS)
    return (self.python, str(self.worker_path), *args)


@dataclass(frozen=True)
class TinygradWorkerError:
  code: str
  detail: str
  exit_code: int | None = None
  def to_dict(self) -> dict[str, Any]: return {"code": self.code, "detail": self.detail, "exit_code": self.exit_code}


@dataclass(frozen=True)
class TinygradAdmissionResult:
  candidate_hash: str
  status: str
  classification: str
  capability_id: str | None = None
  response: Mapping[str, Any] | None = None
  error: TinygradWorkerError | None = None
  @property
  def admitted(self) -> bool: return self.status == "admitted" and self.error is None
  def to_dict(self) -> dict[str, Any]:
    return {"candidate_hash": self.candidate_hash, "status": self.status, "classification": self.classification,
            "capability_id": self.capability_id, "error": self.error.to_dict() if self.error else None,
            "response": dict(self.response) if self.response is not None else None}


def _provider_revision(result:Mapping[str, Any]) -> str | None:
  value = result.get("provider_revision")
  if isinstance(value, Mapping): value = value.get("revision")
  return value if isinstance(value, str) and value else None


def _provider_is_clean(result:Mapping[str, Any]) -> bool:
  value = result.get("provider_revision")
  return isinstance(value, Mapping) and value.get("dirty") is False


def _generated_identity(result:Mapping[str, Any]) -> dict[str, Any]:
  return {"source_hash": result.get("source_sha256"),
          "plan_hash": result.get("candidate_plan_hash", result.get("plan_hash")),
          "binary_hash": result.get("mtlb_sha256", result.get("binary_sha256"))}


def _evidence_hash(result:Mapping[str, Any]) -> str:
  """Hash the complete canonical stage response; providers do not self-attest evidence identity."""
  return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                   allow_nan=False).encode("ascii")).hexdigest()


def validate_execution(value:Mapping[str, Any]) -> dict[str, Any]:
  """Validate the target-neutral measurement regime shared by search and provider."""
  if not isinstance(value, Mapping): raise ValueError("execution must be an object")
  mode = value.get("shape_mode")
  required = {"shape_mode", "warmups", "samples"} | ({"fixture_shape"} if mode == "bounded_fixture" else set())
  if set(value) != required or mode not in {"exact_workload", "bounded_fixture"}:
    raise ValueError("execution must select exact_workload or bounded_fixture with exact fields")
  warmups, samples = value["warmups"], value["samples"]
  if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0: raise ValueError("execution.warmups must be a non-negative int")
  if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0: raise ValueError("execution.samples must be a positive int")
  if mode == "bounded_fixture":
    shape = value["fixture_shape"]
    if not isinstance(shape, Mapping) or set(shape) != {"m", "n", "k"} or any(
        not isinstance(shape[key], int) or isinstance(shape[key], bool) or shape[key] <= 0 for key in ("m", "n", "k")):
      raise ValueError("execution.fixture_shape must contain positive integer m/n/k")
  return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass
class PersistentJSONLSession:
  """One campaign-scoped provider process; every request still has a hard deadline."""
  command: tuple[str, ...]; cwd: str | None = None
  proc: Any = None
  _pending: str = ""
  def __enter__(self):
    self.proc = subprocess.Popen(self.command, cwd=self.cwd, env={**os.environ, "PYTHONPATH": "."}, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    return self
  def __exit__(self, *_exc): self.close()
  def close(self):
    if self.proc is not None and self.proc.poll() is None:
      self.proc.terminate()
      try: self.proc.wait(timeout=2)
      except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
    self.proc = None
  def invoke(self, action:str, payload:Mapping[str, Any], timeout_s:float) -> dict[str, Any]:
    if self.proc is None or self.proc.poll() is not None: return {"blocked_reason": f"provider_{action}:session_unavailable"}
    request_id = f"{payload.get('candidate_hash', 'provider')}:{action}"
    try:
      self.proc.stdin.write(json.dumps({"protocol": PROTOCOL, "request_id": request_id, "action": action, "payload": dict(payload)}, sort_keys=True, separators=(",", ":")) + "\n"); self.proc.stdin.flush()
    except OSError: self.close(); return {"blocked_reason": f"provider_{action}:session_unavailable"}
    deadline = time.monotonic() + timeout_s
    while "\n" not in self._pending:
      ready, _, _ = select.select([self.proc.stdout], [], [], max(0, deadline-time.monotonic()))
      if not ready: self.close(); return {"blocked_reason": f"provider_{action}:timeout"}
      chunk = os.read(self.proc.stdout.fileno(), 65536)
      if not chunk: self.close(); return {"blocked_reason": f"provider_{action}:invalid_json"}
      self._pending += chunk.decode()
    line, self._pending = self._pending.split("\n", 1)
    try: response = json.loads(line)
    except json.JSONDecodeError: return {"blocked_reason": f"provider_{action}:invalid_json"}
    if not isinstance(response, Mapping) or response.get("protocol") != PROTOCOL or response.get("request_id") != request_id or response.get("action") != action or response.get("status") not in ("ok", "blocked"): return {"blocked_reason": f"provider_{action}:invalid_envelope"}
    if response["status"] == "blocked":
      error=response.get("error"); return {"blocked_reason": f"provider_{action}:{error.get('code') if isinstance(error, Mapping) else 'invalid_error'}", "response":dict(response)}
    return {"result":dict(response["result"]), "response":dict(response)} if isinstance(response.get("result"), Mapping) else {"blocked_reason": f"provider_{action}:missing_result"}

def _invoke(command:tuple[str, ...] | PersistentJSONLSession, cwd:str|None, timeout_s:float, action:str, payload:Mapping[str, Any]) -> dict[str, Any]:
  if isinstance(command, PersistentJSONLSession): return command.invoke(action, payload, timeout_s)
  if action not in ACTIONS: return {"blocked_reason": f"unsupported_provider_action:{action}"}
  request_id = f"{payload.get('candidate_hash', 'provider')}:{action}"
  request = {"protocol": PROTOCOL, "request_id": request_id, "action": action, "payload": dict(payload)}
  try:
    proc = subprocess.run(command, cwd=cwd, env={**os.environ, "PYTHONPATH": "."},
                          input=json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
                          text=True, capture_output=True, timeout=timeout_s, start_new_session=True, check=False)
  except subprocess.TimeoutExpired: return {"blocked_reason": f"provider_{action}:timeout"}
  except OSError as exc: return {"blocked_reason": f"provider_{action}:launch_failed:{exc}"}
  if proc.returncode: return {"blocked_reason": f"provider_{action}:exit_{proc.returncode}"}
  lines = [line for line in proc.stdout.splitlines() if line.strip()]
  if len(lines) != 1: return {"blocked_reason": f"provider_{action}:invalid_json_lines"}
  try: response = json.loads(lines[0])
  except json.JSONDecodeError: return {"blocked_reason": f"provider_{action}:invalid_json"}
  if not isinstance(response, Mapping) or response.get("protocol") != PROTOCOL or response.get("request_id") != request_id \
      or response.get("action") != action or response.get("status") not in ("ok", "blocked"):
    return {"blocked_reason": f"provider_{action}:invalid_envelope"}
  if response["status"] == "blocked":
    error = response.get("error")
    code = error.get("code") if isinstance(error, Mapping) else "invalid_error"
    return {"blocked_reason": f"provider_{action}:{code}", "response": dict(response)}
  result = response.get("result")
  if not isinstance(result, Mapping): return {"blocked_reason": f"provider_{action}:missing_result"}
  return {"result": dict(result), "response": dict(response)}


provider_invoke = _invoke


def _candidate_rejection(action:str, stage:Mapping[str, Any]) -> dict[str, str] | None:
  """Classify only candidate-local admission/compiler failures; transport/runtime failures remain blockers."""
  if action not in {"admit", "compile"}: return None
  response = stage.get("response")
  error = response.get("error") if isinstance(response, Mapping) else None
  code = error.get("code") if isinstance(error, Mapping) else None
  details = error.get("details") if isinstance(error, Mapping) else None
  compiler_local = action == "compile" and code == "provider_failure" and isinstance(details, Mapping) \
    and details.get("exception_type") == "CompileError"
  if code in _INVALID_ADMISSION_CODES:
    state = "REJECTED_INVALID"
  elif code in _UNSUPPORTED_CANDIDATE_CODES or compiler_local:
    state = "REJECTED_UNSUPPORTED"
  else:
    return None
  return {"state": state, "stage": action, "code": str(code), "reason": str(stage["blocked_reason"])}


@dataclass(frozen=True)
class TinygradSearchProviderWorker:
  """Run all provider stages under independent process/deadline boundaries."""
  command: tuple[str, ...]
  cwd: str | None = None
  timeout_s: float = 60.0
  resolved_target: Mapping[str, Any] | None = None
  requested_provider_revision: str | None = None
  execution: Mapping[str, Any] | None = None
  exact_gguf: Mapping[str, Any] | None = None
  compiled_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)

  def __post_init__(self):
    if not self.command or (not isinstance(self.command, PersistentJSONLSession) and any(not isinstance(x, str) or not x for x in self.command)) or self.timeout_s <= 0:
      raise ValueError("invalid tinygrad search provider command")
    if self.resolved_target is not None: validate_resolved_target_document(self.resolved_target)
    if self.execution is not None: object.__setattr__(self, "execution", validate_execution(self.execution))

  def _base_payload(self, candidate:FullKernelCandidate) -> dict[str, Any]:
    identity = candidate.candidate_hash
    base = {"candidate_hash": identity, "candidate": candidate.to_dict(), "workload": candidate.workload,
            "target_identity": candidate.target}
    if self.exact_gguf is not None: base["exact_gguf"] = dict(self.exact_gguf)
    if self.execution is not None:
      if self.execution["shape_mode"] == "bounded_fixture" or self.exact_gguf is None:
        base["fixture_shape"] = (candidate.workload["shape"] if self.execution["shape_mode"] == "exact_workload" else self.execution["fixture_shape"])
      base["warmups"], base["samples"] = self.execution["warmups"], self.execution["samples"]
    if self.requested_provider_revision is not None:
      base["provider_identity"] = {"revision": self.requested_provider_revision, "require_clean": True}
    return base

  def measure_only(self, candidate:FullKernelCandidate) -> dict[str, Any]:
    """Repeat a compiled finalist's measurement without a second compile/check protocol stage."""
    base = self._base_payload(candidate)
    stage = _invoke(self.command, self.cwd, self.timeout_s, "measure", base)
    if "blocked_reason" in stage: return {"candidate_hash":candidate.candidate_hash, **stage}
    result = stage["result"]; samples = result.get("samples_ns")
    generated = _generated_identity(result)
    if generated != self.compiled_artifacts.get(candidate.candidate_hash):
      return {"candidate_hash":candidate.candidate_hash, "blocked_reason":"provider_measure:compiled_artifact_drift",
              "generated":generated, "provider":{"measure":stage.get("response")}}
    required = self.execution["samples"] if self.execution is not None else 5
    if not isinstance(samples, list) or len(samples) < max(5, required) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in samples):
      return {"candidate_hash":candidate.candidate_hash, "blocked_reason":"provider_measure:invalid_raw_samples",
              "provider":{"measure":stage.get("response")}}
    return {"candidate_hash":candidate.candidate_hash, "status":"MEASURED",
      "generated":generated, "performance":{"evidence_hash":_evidence_hash(result),
        "measurement":{"median_ns":statistics.median(samples), "samples_ns":list(samples)}, "evidence":dict(result)},
      "provider":{"measure":stage["response"]}}

  def __call__(self, candidate:FullKernelCandidate) -> dict[str, Any]:
    identity = candidate.candidate_hash
    base = self._base_payload(candidate)
    actions: dict[str, Any] = {}
    describe = _invoke(self.command, self.cwd, self.timeout_s, "describe", base)
    if "blocked_reason" in describe: return {"candidate_hash": identity, **describe}
    actions["describe"] = describe["response"]
    description = describe["result"]
    revision = _provider_revision(description)
    if self.requested_provider_revision is not None and revision != self.requested_provider_revision:
      return {"candidate_hash": identity, "blocked_reason": "provider_describe:revision_mismatch", "provider": actions}
    if self.requested_provider_revision is not None and not _provider_is_clean(description):
      return {"candidate_hash": identity, "blocked_reason": "provider_describe:dirty_or_missing_clean_evidence", "provider": actions}
    facts = description.get("target")
    if not isinstance(facts, Mapping):
      return {"candidate_hash": identity, "blocked_reason": "provider_describe:missing_target_facts", "provider": actions}
    if candidate.schema_version in ("boltbeam.full_kernel_candidate.v2", "boltbeam.full_kernel_candidate.v3"):
      if self.resolved_target is None:
        return {"candidate_hash": identity, "blocked_reason": "provider_describe:missing_resolved_target", "provider": actions}
      live = resolved_target_document(candidate.target["target_id"], facts)
      try:
        validate_candidate_target(candidate.target, live)
        if dict(live) != dict(self.resolved_target): raise ValueError("request/live target drift")
      except ValueError:
        return {"candidate_hash": identity, "blocked_reason": "provider_describe:target_identity_mismatch", "provider": actions}

    results: dict[str, Mapping[str, Any]] = {}
    for action in ACTIONS[1:]:
      stage = _invoke(self.command, self.cwd, self.timeout_s, action, base)
      if "blocked_reason" in stage:
        rejection = _candidate_rejection(action, stage)
        if rejection is not None:
          return {"candidate_hash": identity, "candidate_rejection": rejection,
                  "provider": actions | {action: stage.get("response")}}
        return {"candidate_hash": identity, **stage, "provider": actions | {action: stage.get("response")}}
      actions[action], results[action] = stage["response"], stage["result"]
      if action == "admit" and results[action].get("admitted") is not True:
        return {"candidate_hash": identity, "candidate_rejection": {"state": "REJECTED_INVALID", "stage": "admit",
                "code": "not_admitted", "reason": "provider_admit:not_admitted"}, "provider": actions}

    compile_result, check_result, measure_result = results["compile"], results["check"], results["measure"]
    samples = measure_result.get("samples_ns")
    measurement = dict(measure_result.get("summary_ns", {})) if isinstance(measure_result.get("summary_ns"), Mapping) else {}
    if isinstance(samples, list) and samples and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in samples):
      measurement["median_ns"] = statistics.median(samples)
      measurement["samples_ns"] = list(samples)
    generated = _generated_identity(compile_result)
    if _generated_identity(measure_result) != generated:
      return {"candidate_hash":identity, "blocked_reason":"provider_measure:compiled_artifact_drift",
              "generated":generated, "provider":actions}
    self.compiled_artifacts[identity] = generated
    return {"candidate_hash": identity, "target": candidate.target, "workload": candidate.workload,
            "hardware": {"target": candidate.target, "device_id": str(facts.get("name", candidate.target.get("target_id", "provider-device"))),
                         "facts": dict(facts), "provider_revision": revision},
            "correctness": {"passed": check_result.get("correct"), "evidence_hash": _evidence_hash(check_result),
                            "evidence": dict(check_result)},
            "performance": {"evidence_hash": _evidence_hash(measure_result), "measurement": measurement,
                            "evidence": dict(measure_result)},
            "generated": generated, "provider": actions}


def admit_candidate(candidate:FullKernelCandidate, config:TinygradWorkerConfig) -> TinygradAdmissionResult:
  """Compatibility facade over the canonical provider's ``admit`` action."""
  identity = candidate.candidate_hash
  worker = config.worker_path
  if not worker.is_file(): return _worker_failure(identity, "worker_missing", f"Tinygrad provider does not exist: {worker}")
  payload = {"candidate_hash": identity, "candidate": candidate.to_dict(), "workload": candidate.workload,
             "target_identity": candidate.target}
  if config.requested_provider_revision is not None:
    payload["provider_identity"] = {"revision": config.requested_provider_revision, "require_clean": True}
  stage = _invoke(config.command, str(config.root), config.timeout_s, "admit", payload)
  if "blocked_reason" in stage:
    response, reason = stage.get("response"), stage["blocked_reason"]
    error = response.get("error") if isinstance(response, Mapping) else None
    code = error.get("code") if isinstance(error, Mapping) else reason.rsplit(":", 1)[-1]
    if code in _INVALID_ADMISSION_CODES or code in _UNSUPPORTED_CANDIDATE_CODES:
      classification = "candidate_invalid" if code in _INVALID_ADMISSION_CODES else "compiler_unsupported"
      return TinygradAdmissionResult(identity, "rejected", classification, response=response)
    return _worker_failure(identity, code, reason)
  result = stage["result"]
  if result.get("admitted") is not True: return _worker_failure(identity, "worker_invalid_response", "provider did not admit candidate")
  return TinygradAdmissionResult(identity, "admitted", "admitted", result.get("capability_id"), response=stage["response"])


def _worker_failure(identity:str, code:str, detail:str, exit_code:int|None=None) -> TinygradAdmissionResult:
  return TinygradAdmissionResult(identity, "worker_error", "worker_error", error=TinygradWorkerError(code, detail, exit_code))


__all__ = ["ACTIONS", "DEFAULT_PROVIDER_ARGS", "DEFAULT_PROVIDER_RELPATH", "PROTOCOL", "TinygradAdmissionResult", "TinygradSearchProviderWorker",
           "TinygradWorkerConfig", "TinygradWorkerError", "admit_candidate", "provider_invoke", "validate_execution"]
