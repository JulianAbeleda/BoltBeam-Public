"""The one accessor for BoltBeam's stated choices: data/policy.json. A missing entry is an error, never a default."""
import functools
import json
import pathlib


@functools.cache
def _table() -> dict:
  return json.loads((pathlib.Path(__file__).resolve().parent.parent / "data" / "policy.json").read_text())["policy"]


def value(name:str) -> float:
  """One policy value by name; KeyError when the table lacks it or it is not a number (fail closed)."""
  v = (_table().get(name) or {}).get("value")
  if not isinstance(v, (int, float)) or isinstance(v, bool):
    raise KeyError(f"policy {name!r} is not in data/policy.json; add it with a reason, never a default in code")
  return float(v)
