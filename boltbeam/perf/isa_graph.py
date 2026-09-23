"""Ordered, explicit ISA dependency graph for gfx1100 MMQ evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

INSTRUCTION_CLASSES = frozenset(("valu_int", "valu_float", "dot_mfma", "salu", "global_load", "global_store",
                                 "lds_load", "lds_store", "branch_predicate", "barrier", "waitcnt"))
MEMORY_CLASSES = frozenset(("global_load", "global_store", "lds_load", "lds_store"))


@dataclass(frozen=True)
class InstructionNode:
  index: int
  pc: int
  mnemonic: str
  instruction_class: str
  issue_domain: str
  reads: tuple[str, ...]
  writes: tuple[str, ...]
  dependencies: tuple[int, ...]
  epoch: str
  active_lanes: int | None = None
  transactions: int | None = None


@dataclass(frozen=True)
class ISAGraph:
  nodes: tuple[InstructionNode, ...]
  complete: bool
  blockers: tuple[str, ...] = ()
  assumptions: tuple[str, ...] = ()

  def counts(self) -> dict[str, int]:
    return {kind: sum(n.instruction_class == kind for n in self.nodes) for kind in sorted(INSTRUCTION_CLASSES)}


def build_isa_graph(manifest: Mapping[str, Any]) -> ISAGraph:
  """Build dependencies from structured reads/writes; raw operand strings are intentionally not parsed."""
  rows = manifest.get("instructions")
  if not isinstance(rows, list) and isinstance(manifest.get("final_isa"), Mapping):
    rows = manifest["final_isa"].get("instructions")
  if not isinstance(rows, list): rows = manifest.get("rows")
  if not isinstance(rows, list): return ISAGraph((), False, ("missing ordered instructions",))
  nodes, blockers, assumptions, writers = [], [], [], {}
  previous_pc = -1
  epoch_barrier: int | None = None
  for index, row in enumerate(rows):
    if not isinstance(row, Mapping): blockers.append(f"instructions[{index}] is not an object"); continue
    kind = row.get("instruction_class")
    reads, writes = row.get("reads"), row.get("writes")
    pc = row.get("pc")
    if kind not in INSTRUCTION_CLASSES: blockers.append(f"instructions[{index}] unknown class {kind!r}"); continue
    if not isinstance(pc, int) or pc <= previous_pc: blockers.append(f"instructions[{index}] PC is not ordered"); continue
    previous_pc = pc
    if not isinstance(reads, list) or not isinstance(writes, list):
      blockers.append(f"instructions[{index}] lacks structured reads/writes"); reads, writes = [], []
    deps = {writers[r] for r in reads if r in writers}
    explicit = row.get("dependencies", [])
    if isinstance(explicit, list):
      invalid_deps = [d for d in explicit if not isinstance(d, int) or not 0 <= d < index]
      if invalid_deps: blockers.append(f"instructions[{index}] has invalid dependencies")
      deps.update(d for d in explicit if isinstance(d, int) and 0 <= d < index)
    if epoch_barrier is not None: deps.add(epoch_barrier)
    epoch = str(row.get("epoch", "unknown"))
    if epoch in ("", "unknown") and _stage_visibility_boundary(rows, index):
      epoch = "stage"
      assumptions.append(f"instruction {index} epoch inferred as stage from stage/visibility_sync boundary")
    node = InstructionNode(index, pc, str(row.get("mnemonic", "")), kind, str(row.get("issue_domain", _domain(kind))),
      tuple(map(str, reads)), tuple(map(str, writes)), tuple(sorted(deps)), epoch,
      row.get("active_lanes"), row.get("transactions"))
    nodes.append(node)
    for register in writes: writers[str(register)] = index
    if kind in ("barrier", "waitcnt"): epoch_barrier = index
  return ISAGraph(tuple(nodes), not blockers and bool(nodes), tuple(blockers), tuple(assumptions))


def _stage_visibility_boundary(rows: list[Any], index: int) -> bool:
  if index == 0 or index + 1 >= len(rows): return False
  before, after = rows[index - 1], rows[index + 1]
  return isinstance(before, Mapping) and isinstance(after, Mapping) and before.get("epoch") == "stage" and after.get("epoch") == "visibility_sync"


def _domain(kind: str) -> str:
  if kind in ("valu_int", "valu_float", "dot_mfma"): return "valu"
  if kind == "salu" or kind in ("branch_predicate", "barrier", "waitcnt"): return "salu"
  if kind.startswith("global_"): return "vmem"
  if kind.startswith("lds_"): return "lds"
  return "unknown"


__all__ = ["INSTRUCTION_CLASSES", "ISAGraph", "InstructionNode", "build_isa_graph"]
