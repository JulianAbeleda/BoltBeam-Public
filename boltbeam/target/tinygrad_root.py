"""One portable authority for locating a tinygrad checkout."""
from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping


TINYGRAD_ROOT_ENV = "TINYGRAD_ROOT"
_SIBLING_NAMES = ("tinygrad-arkey", "tinygrad-arkey-exp", "tinygrad")


def _is_checkout(path:pathlib.Path) -> bool:
  return path.is_dir() and (path / "tinygrad" / "__init__.py").is_file()


def resolve_tinygrad_root(value:str | pathlib.Path | None = None, *, environ:Mapping[str, str] | None = None,
                          boltbeam_root:str | pathlib.Path | None = None,
                          require_checkout:bool = False) -> pathlib.Path:
  """Resolve explicit config, then environment, then a recognized sibling checkout.

  Checkout location is execution configuration, never experiment identity. An
  unresolved location fails with an actionable message instead of silently
  embedding a developer's absolute path in generated commands.
  """
  env = os.environ if environ is None else environ
  configured = value or env.get(TINYGRAD_ROOT_ENV)
  if configured:
    # Preserve the caller's path spelling (notably macOS /tmp -> /private/tmp)
    # while still making relative configuration unambiguous.
    path = pathlib.Path(configured).expanduser().absolute()
  else:
    repo = pathlib.Path(boltbeam_root).expanduser().resolve() if boltbeam_root else pathlib.Path(__file__).parents[2]
    candidates = [repo.parent / name for name in _SIBLING_NAMES]
    path = next((candidate.resolve() for candidate in candidates if _is_checkout(candidate)), None)
    if path is None:
      raise ValueError(f"tinygrad checkout is unresolved; pass --tinygrad-root or set {TINYGRAD_ROOT_ENV}")
  if require_checkout and not _is_checkout(path):
    raise ValueError(f"tinygrad root is not a checkout: {path}")
  return path


__all__ = ["TINYGRAD_ROOT_ENV", "resolve_tinygrad_root"]
