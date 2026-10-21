"""Where the GPU vendor toolkits are installed: the one place BoltBeam answers "where is CUDA / ROCm".

Each toolkit names its root through its own environment variables; the documented default install location is the
last resort. Tools are found on PATH first, then under each candidate root's bin/. Callers ask by tool name and never
spell an install path themselves.
"""
from __future__ import annotations

import os
import pathlib
from typing import Callable, Mapping

from boltbeam.target.vendors import VENDORS

# each vendor's toolkit (target/vendors.py): its root's variables and default, and the tools under its bin/
TOOLKITS = {v.toolkit.name: (v.toolkit.env, v.toolkit.default) for v in VENDORS.values() if v.toolkit}
TOOLKIT_OF = {t: v.toolkit.name for v in VENDORS.values() if v.toolkit for t in v.toolkit.tools}


def roots(toolkit:str, env:Mapping[str, str] | None = None) -> list[pathlib.Path]:
  """The toolkit's candidate install roots, in the order they are tried."""
  env = os.environ if env is None else env
  names, default = TOOLKITS[toolkit]
  return [pathlib.Path(r) for r in dict.fromkeys([env[n] for n in names if env.get(n)] + [default])]


def tool(name:str, resolver:Callable[[str], str | None], env:Mapping[str, str] | None = None) -> str | None:
  """A toolkit tool: on PATH (`resolver`, normally shutil.which), else in a candidate root's bin/; None if absent."""
  found = resolver(name)
  if found:
    return str(found)
  for root in roots(TOOLKIT_OF[name], env) if name in TOOLKIT_OF else ():
    if (root / "bin" / name).is_file():
      return str(root / "bin" / name)
  return None


def file(toolkit:str, relative:str, env:Mapping[str, str] | None = None) -> pathlib.Path | None:
  """A file inside the toolkit (a header, a library), from the first root that has it."""
  return next((r / relative for r in roots(toolkit, env) if (r / relative).is_file()), None)
