"""Run one callback in a short-lived, process-group-isolated boundary."""
from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import multiprocessing
import os
import signal
import time
from typing import Any, Callable


@dataclass(frozen=True)
class StageResult:
  status: str
  value: Any = None
  error: str | None = None
  stdout: str = ""
  stderr: str = ""

  @property
  def ok(self) -> bool:
    return self.status == "success"


class _BoundedText(io.TextIOBase):
  def __init__(self, limit: int):
    self._limit = limit
    self._buf = io.StringIO()
    self._size = 0

  def write(self, text: str) -> int:
    remaining = max(0, self._limit - self._size)
    kept = text[:remaining]
    self._buf.write(kept)
    self._size += len(kept)
    return len(text)

  def getvalue(self) -> str:
    return self._buf.getvalue()

  def flush(self) -> None:
    pass


def _invoke(callback: Callable[[], Any], conn, output_limit: int) -> None:
  out, err = _BoundedText(output_limit), _BoundedText(output_limit)
  try:
    if hasattr(os, "setsid"):
      os.setsid()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
      value = callback()
    conn.send(("success", value, None, out.getvalue(), err.getvalue()))
  except BaseException as exc:  # callback failures are data at this boundary
    conn.send(("callback_error", None, f"{type(exc).__name__}: {exc}", out.getvalue(), err.getvalue()))
  finally:
    conn.close()


def _stop(process: multiprocessing.Process, grace_s: float) -> None:
  if process.pid is not None and hasattr(os, "killpg"):
    try:
      os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
  else:
    process.terminate()
  process.join(grace_s)
  if process.is_alive():
    if process.pid is not None and hasattr(os, "killpg"):
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    else:
      process.kill()
    process.join()


def run_isolated(callback: Callable[[], Any], *, timeout_s: float,
                 output_limit: int = 64 * 1024, terminate_grace_s: float = 0.2) -> StageResult:
  """Run ``callback`` with a hard deadline; never treat absent output as success."""
  if timeout_s <= 0 or output_limit < 0 or terminate_grace_s < 0:
    raise ValueError("timeout_s must be positive and limits must be non-negative")
  # This API intentionally accepts closures. POSIX spawn cannot pickle them,
  # so use an explicit local context rather than inheriting a platform/global
  # start-method default. Platforms without fork retain their native context
  # and therefore require a picklable callback.
  methods = multiprocessing.get_all_start_methods()
  context = multiprocessing.get_context("fork" if "fork" in methods else None)
  parent, child = context.Pipe(duplex=False)
  process = context.Process(target=_invoke, args=(callback, child, output_limit))
  process.start()
  child.close()
  deadline = time.monotonic() + timeout_s
  process.join(timeout_s)
  if process.is_alive():
    _stop(process, terminate_grace_s)
    parent.close()
    return StageResult("timeout", error=f"stage exceeded {timeout_s:g}s")
  if parent.poll(max(0.0, deadline - time.monotonic())):
    try:
      status, value, error, stdout, stderr = parent.recv()
    except (EOFError, OSError):
      parent.close()
      return StageResult("no_result", error="stage exited without a structured result")
    parent.close()
    return StageResult(status, value, error, stdout, stderr)
  parent.close()
  return StageResult("no_result", error="stage exited without a structured result")


__all__ = ["StageResult", "run_isolated"]
