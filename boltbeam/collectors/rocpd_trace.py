"""rocprofv3's rocpd database written as the Nsight Systems tables semantic_capture reads: AMD's path to the same join.

    rocpd (rocprofv3 --output-format rocpd)          Nsight Systems export
    MARKER_CORE_RANGE_API regions (torch nvtx: roctx) NVTX_EVENTS
    HIP_RUNTIME_API regions, their stack_id          CUPTI_ACTIVITY_KIND_RUNTIME, correlationId
    kernels, stack_id of the launching call          CUPTI_ACTIVITY_KIND_KERNEL, correlationId
    a replayed graph's node -> its capture's launch  CUDA_GRAPH_NODE_EVENTS (a node's origin call)

Times stay rocprofv3's (host and GPU nanoseconds on one clock), so the session start is 0. HIP reports a grid in
work-items; it is written in blocks, as CUDA counts it. rocpd names no node's origin: graph_exec_id counts
instantiations in order, so the k-th hipGraphInstantiate is exec k, its capture the last one its thread ended,
and node n the capture's n-th kernel launch, copy or fill. Every node is checked against that launch's grid and block; a
graph whose launches do not line up is refused, never guessed. ROCm runs a copy or fill as its own blit kernel
(__amd_rocclr_*); Nsight records those as memory operations, not kernels, so they are left out of the kernels here.
"""
from __future__ import annotations

import collections
import re
import sqlite3
from pathlib import Path

LAUNCHES = ("hipLaunchKernel", "hipExtLaunchKernel", "hipModuleLaunchKernel", "hipExtModuleLaunchKernel",
            "hipLaunchCooperativeKernel", "hipExtLaunchKernelGGL", "hipLaunchKernelExC")
BLIT = "copy or fill"
SCHEMA = """create table StringIds(id integer primary key, value text);
create table NVTX_EVENTS(start int, end int, text text, textId int, globalTid int, eventType int);
create table CUPTI_ACTIVITY_KIND_RUNTIME(start int, end int, globalTid int, correlationId int, nameId int);
create table CUPTI_ACTIVITY_KIND_KERNEL(start int, end int, correlationId int, globalPid int, deviceId int, streamId int,
  graphNodeId int, demangledName int, mangledName int, gridX int, gridY int, gridZ int, blockX int, blockY int, blockZ int);
create table CUDA_GRAPH_NODE_EVENTS(start int, end int, globalTid int, graphNodeId int, originalGraphNodeId int);
create table ANALYSIS_FILE(filename text, globalPid int); create table TARGET_INFO_SESSION_START_TIME(systemClockNs int);"""


def tid(pid:int, thread:int) -> int:
  return pid << 24 | thread & 0xFFFFFF


def node(exec_id:int, index:int) -> int:
  return exec_id << 20 | index


def geometry(args:dict[str, str]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
  """A launch call's grid and block, in blocks and threads; None when its arguments do not say."""
  dims = lambda text: tuple(int(re.search(rf"{a}=(\d+)", text)[1]) for a in "xyz")  # noqa: E731
  if "numBlocks" in args and "dimBlocks" in args:
    return dims(args["numBlocks"]), dims(args["dimBlocks"])
  if "gridDimX" in args:
    return tuple(int(args["gridDim" + a]) for a in "XYZ"), tuple(int(args["blockDim" + a]) for a in "XYZ")
  if "globalWorkSizeX" in args:
    block = tuple(int(args["localWorkSize" + a]) for a in "XYZ")
    return tuple(-(-int(args["globalWorkSize" + a]) // b) for a, b in zip("XYZ", block)), block
  return None


def captures(db:sqlite3.Connection) -> list[list[dict]]:
  """Each instantiated graph's captured kernel launches, in instantiation order: [{tid, start, end, shape}]."""
  marks = ",".join("?" * len(LAUNCHES))
  calls = db.execute("select id, name, pid, tid, start, end from regions where category like 'HIP_RUNTIME_API%' and "
                     "(name like 'hipStream%Capture' or name like 'hipGraphInstantiate%' or name like 'hipMem%Async' "
                     "or name in (" + marks + ")) "
                     "order by start", LAUNCHES).fetchall()
  args = collections.defaultdict(dict)
  for r in db.execute("select id, name, value from region_args where id in (select id from regions where name in ("
                      + marks + "))", LAUNCHES):
    args[r[0]][r[1]] = str(r[2])
  open_, ended, graphs = {}, {}, []
  for i, name, pid, thread, start, end in calls:
    if name == "hipStreamBeginCapture":
      open_[pid, thread] = []
    elif name == "hipStreamEndCapture":
      ended[pid, thread] = open_.pop((pid, thread), None)
    elif name.startswith("hipGraphInstantiate"):
      graphs.append(ended.pop((pid, thread), None))
    elif (pid, thread) in open_:  # a captured copy or fill is a node too, replayed as ROCm's own blit kernel
      open_[pid, thread].append({"tid": tid(pid, thread), "start": start, "end": end,
                                 "shape": BLIT if name.startswith("hipMem") else geometry(args[i])})
  return [launches or [] for launches in graphs]


def origins(graphs:list[list[dict]], replayed:dict[int, dict[int, tuple]]) -> dict[tuple[int, int], dict]:
  """(graph_exec_id, node index) -> the launch that captured it. Exec ids count instantiations, including those
  made before the trace's window opened, so the k-th instantiation seen is exec k + offset for one offset: the one
  under which every replayed node has its captured launch's grid and block. None or several is a refusal."""
  def same(launch, shape, name):
    return name.startswith("__amd_rocclr_") if launch["shape"] == BLIT else launch["shape"] == shape
  def fits(offset):
    return all(0 < e - offset <= len(graphs) and all(n < len(graphs[e - offset - 1]) and
               same(graphs[e - offset - 1][n], *seen) for n, seen in nodes.items()) for e, nodes in replayed.items())
  found = [o for o in range(max(replayed, default=0)) if fits(o)] if replayed else [0]
  if len(found) != 1:
    raise ValueError(f"replayed graphs {sorted(replayed)[:5]} line up with {len(found)} placements among the {len(graphs)} "
                     "captured graphs (one is required): a node's origin cannot be named")
  return {(k + found[0], n): launch for k, launches in enumerate(graphs, 1) for n, launch in enumerate(launches)}


def export(sources:list[Path], dst:Path) -> Path:
  """Write dst (an Nsight-shaped capture.sqlite) from rocprofv3's rocpd databases, one per traced process."""
  dst = Path(dst)
  dst.unlink(missing_ok=True)
  out = sqlite3.connect(dst)
  out.executescript(SCHEMA)
  strings: dict[str, int] = {}
  sid = lambda s: strings.setdefault(s, len(strings) + 1) if s is not None else None  # noqa: E731
  for src in sources:
    with sqlite3.connect(src) as db:
      _copy(db, out, sid)
  out.executemany("insert into StringIds values(?,?)", [(i, s) for s, i in strings.items()])
  out.execute("insert into TARGET_INFO_SESSION_START_TIME values(0)")
  out.commit()
  out.close()
  return dst


def _copy(db:sqlite3.Connection, out:sqlite3.Connection, sid) -> None:
  # rocprofv3 writes its tables without indexes; its views join a million call arguments to their calls
  for (table,) in db.execute("select name from sqlite_master where type='table' and "
                             "(name like 'rocpd_arg%' or name like 'rocpd_region%')").fetchall():
    db.execute(f"create index if not exists boltbeam_{table} on {table}(event_id)")
  out.executemany("insert into NVTX_EVENTS values(?,?,?,null,?,59)",
                  [(s, e, name, tid(p, t)) for name, p, t, s, e in db.execute(
                    "select name, pid, tid, start, end from regions where category='MARKER_CORE_RANGE_API'")])
  out.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)",
                  [(s, e, tid(p, t), c, sid(name)) for name, p, t, s, e, c in db.execute(
                    "select name, pid, tid, start, end, stack_id from regions where category like 'HIP_RUNTIME_API%'")])
  rows = db.execute("select k.start, k.end, k.stack_id, k.pid, k.agent_abs_index, k.stream_id, k.graph_exec_id, "
                    "k.graph_node_id, k.name, s.kernel_name, k.grid_x, k.grid_y, k.grid_z, k.workgroup_x, "
                    "k.workgroup_y, k.workgroup_z from kernels k left join kernel_symbols s on s.id = k.kernel_id "
                    "order by k.start").fetchall()
  shape = lambda dims: (tuple(-(-g // b) for g, b in zip(dims[:3], dims[3:])), tuple(dims[3:]))  # noqa: E731
  replayed: dict[int, dict[int, tuple]] = collections.defaultdict(dict)
  for r in rows:
    if r[6]:
      replayed[r[6]][r[7]] = (shape(r[10:]), r[8])
  captured = origins(captures(db), replayed)
  kernels, nodes = [], {}
  for (start, end, corr, pid, dev, stream, exec_id, index, name, raw, *dims) in rows:
    grid, block = shape(dims)
    graph = None
    if exec_id:
      launch = captured[exec_id, index]
      graph = node(exec_id, index)
      nodes[graph] = (launch["start"], launch["end"], launch["tid"], graph)
    if not name.startswith("__amd_rocclr_"):  # ROCm runs copies and fills as kernels; CUDA records them as copies
      kernels.append((start, end, corr, pid << 24, dev, stream, graph, sid(name), sid(raw), *grid, *block))
  out.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", kernels)
  out.executemany("insert into CUDA_GRAPH_NODE_EVENTS values(?,?,?,?,null)", nodes.values())
  out.executemany("insert into ANALYSIS_FILE values(?,?)",
                  [(str(g), p << 24) for g, p in db.execute("select guid, pid from processes")])
