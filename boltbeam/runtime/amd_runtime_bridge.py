"""Small, deliberately closed bridge from validated binaries to tinygrad AMD runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from typing import Any, Mapping

from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


@dataclass(frozen=True)
class CompileArtifact:
  """The only compile-result fields this bridge consumes."""

  binary: bytes
  binary_sha256: str


def executable_for_amd(artifact: CompileArtifact | Mapping[str, Any],
                      program: UOp, info: ProgramInfo) -> Any:
  """Return an AMDProgram-backed executable without submitting work.

  Compilation/validation is intentionally outside this function.  This function
  only admits a complete artifact whose bytes match its declared identity and a
  matching PROGRAM UOp.  ``get_runtime`` creates the tinygrad runtime (and hence
  uses ``AMDDevice.runtime``/``AMDProgram``); calling the returned object is the
  caller's responsibility.
  """
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("missing or invalid program UOp")
  if not isinstance(info, ProgramInfo) or program.arg != info:
    raise ValueError("program UOp and ProgramInfo do not match")

  if isinstance(artifact, CompileArtifact):
    binary, declared = artifact.binary, artifact.binary_sha256
  elif isinstance(artifact, Mapping):
    binary, declared = artifact.get("binary"), artifact.get("binary_sha256")
  else:
    raise ValueError("missing compile artifact")
  if not isinstance(binary, bytes) or not isinstance(declared, str) or not declared:
    raise ValueError("compile artifact is missing binary identity")
  actual = sha256(binary).hexdigest()
  if not compare_digest(actual, declared):
    raise ValueError("compile artifact binary_sha256 mismatch")

  # The binary is the renderer/runtime payload (ast.src[4].arg in tinygrad).
  # Keep this assignment local and do not invoke the returned executable here.
  runtime_program = program.replace(src=program.src[:4] + (program.src[4].replace(arg=binary),) + program.src[5:])
  return get_runtime("AMD", runtime_program, cache=False)
