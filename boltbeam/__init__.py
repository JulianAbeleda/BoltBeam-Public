"""BoltBeam: translate weights into codegen search."""

import sys

# The floor, checked before anything else is imported. `pip install` enforces requires-python in
# pyproject.toml, but running from a clone does not go through packaging, and the modules below use
# syntax an older interpreter cannot parse. Without this the first thing a new reader sees is
# "TypeError: unsupported operand type(s) for |" from a type annotation, which does not answer the
# question they have. Written in syntax every supported interpreter can read, or it cannot report
# the one thing it exists to report. tests/test_python_floor.py keeps it equal to pyproject.
MINIMUM_PYTHON = (3, 10)
if sys.version_info < MINIMUM_PYTHON:
  raise SystemExit(
    "BoltBeam needs Python %d.%d or newer. This is Python %d.%d.%d, at %s.\n"
    "The system python3 on macOS is usually older than the floor: try python3.12, or any newer one."
    % (MINIMUM_PYTHON + sys.version_info[:3] + (sys.executable,)))

__all__ = ["__version__", "MINIMUM_PYTHON", "FullKernelCandidateEntry", "FullKernelCandidateSet",
           "build_qwen3_8b_buffer2_candidate_set"]
__version__ = "0.1.0"

from boltbeam.full_kernel_candidate_set import (FullKernelCandidateEntry, FullKernelCandidateSet,
                                                build_qwen3_8b_buffer2_candidate_set)
