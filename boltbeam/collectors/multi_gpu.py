"""An engine split across GPUs (tensor parallel): how many GPUs, how they connect, and what one join costs. Generic over
engine and vendor; an engine adapter says only how it asks for the split (vLLM: --tensor-parallel-size).

    split_size  the most GPUs (up to those here) that divide every head count of the model: the rule vLLM enforces,
                so a split BoltBeam picks is one the engine accepts
    topology    the link between each GPU pair, from the vendor's tool (`nvidia-smi topo -m`, `amd-smi topology`)
    join        a tensor-parallel split joins its partial results with an all-reduce across its GPUs:
                runtime/gpu_join_probe.py times one under the engine's own torch and collective library (NCCL or
                RCCL), captured in a graph as the engine runs it, at a ladder of message sizes. A copy between two
                GPUs is no stand-in: back-to-back copies overlap, and one copy is not a reduction.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from typing import Any

PROBE = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "gpu_join_probe.py"
HEAD_FIELDS = ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads")


def heads(model:str | pathlib.Path) -> dict[str, int]:
  """The model's head counts from its config.json (a multimodal checkpoint's text_config when it has one)."""
  from boltbeam.profile.loaders import _config_for
  cfg = _config_for(pathlib.Path(model))
  text = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) and cfg["text_config"].get("num_attention_heads") else cfg
  return {k: int(text[k]) for k in HEAD_FIELDS if isinstance(text.get(k), int) and text[k] > 0}


def split_size(model:str | pathlib.Path, gpus:int) -> int:
  """The most GPUs, up to `gpus`, that divide every head count; 1 when none does or the config has none."""
  counts = heads(model)
  return next((n for n in range(gpus, 1, -1) if counts and all(c % n == 0 for c in counts.values())), 1)


def read_amd_topology(text:str) -> dict[tuple[int, int], str]:
  """`amd-smi topology`: the link type between each pair of GPUs (PCIE, XGMI), GPUs in the tool's order."""
  table = text.split("LINK TYPE TABLE:", 1)[1] if "LINK TYPE TABLE:" in text else ""
  rows = []
  for line in table.splitlines()[2:]:  # the rest of the title line, then the column header
    if not re.match(r"\s*[0-9a-fA-F]{4}:", line):
      break
    rows.append(line.split()[1:])
  return {(i, j): cell for i, row in enumerate(rows) for j, cell in enumerate(row) if i != j and cell != "SELF"}


def topology(backend:str) -> dict[tuple[int, int], str]:
  """The link between each GPU pair from the vendor's tool; empty when the tool is not here or fails."""
  from boltbeam.target import toolkits, vendors
  row = vendors.BY_BACKEND.get(backend)
  if row is None or row.topology is None:
    return {}
  (tool, *args), reader = row.topology
  argv = [toolkits.tool(tool, shutil.which), *args]
  if not argv[0] or not shutil.which(str(argv[0])):
    return {}
  read = vendors.resolve(reader)
  try:
    proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=60)
  except (OSError, subprocess.SubprocessError):
    return {}
  return read(proc.stdout) if proc.returncode == 0 else {}


def join(python:str, gpus:list[dict[str, Any]], *, timeout_s:float = 600.0) -> dict[str, Any]:
  """One all-reduce across these GPUs (hardware-profile devices, named by PCI address, else index), timed under the
  engine's torch: {"joins": [{"bytes", "join_us"}], "backend", "graph", "torch", ...}."""
  names = ",".join(str(d.get("pci_bus_id") or d["index"]) for d in gpus)
  proc = subprocess.run([python, str(PROBE), "--gpus", names], capture_output=True, text=True, timeout=timeout_s)
  lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError((proc.stderr or proc.stdout).strip()[-300:] or f"gpu_join_probe exited {proc.returncode}")
  return json.loads(lines[-1])


def join_us(joins:list[dict[str, Any]], nbytes:float) -> float | None:
  """One join of nbytes from a measured ladder: linear between the two measured sizes around it; None outside it."""
  pts = sorted((j["bytes"], j["join_us"]) for j in joins)
  for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
    if b0 <= nbytes <= b1:
      return t0 + (t1 - t0) * (nbytes - b0) / (b1 - b0)
  return None

