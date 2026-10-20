#!/usr/bin/env python3
"""One join of a tensor-parallel split, timed with the engine's own all-reduce: collectors/multi_gpu.join.

A row (tensor-parallel) split joins its partial results with an all-reduce across every GPU of the split. vLLM
picks how (its custom all-reduce where its platform has one, else NCCL / RCCL through pynccl), so this asks vLLM's
own tensor-parallel group, set up as the engine sets it, one process per GPU, and captures the calls with vLLM's graph_capture
as its decode graphs hold them, timed between two GPU events, per call, best of --reps; the slowest rank's time is
the join. torch.distributed's collective is no stand-in: launched eagerly it adds the launch (22 us a call on two
R9700s), and captured in a graph through it RCCL hangs. Left on, vLLM's custom all-reduce took 6.5 us where the
engine, which turns it off on an R9700, ran RCCL at 17.7.

--gpus names the split's GPUs by PCI address (0000:13:00.0) or torch index, comma-separated; --bytes is a
comma-separated ladder of message sizes in bytes (bfloat16 elements). Prints one JSON line
{"gpus", "torch_gpus", "backend", "graph", "dtype", "reps", "torch", "vllm", "joins": [{"bytes", "join_us", "calls"}]}.
"""
from __future__ import annotations

import argparse, json, socket

LADDER = ",".join(str(1 << k) for k in range(12, 27))  # 4 KiB .. 64 MiB, doubling: a join between is interpolated


def index(torch, name:str) -> int:
  if name.isdigit():
    return int(name)
  domain, bus = (int(x, 16) for x in name.split(":")[:2])
  found = [i for i in range(torch.cuda.device_count()) if (p := torch.cuda.get_device_properties(i)) and
           getattr(p, "pci_bus_id", None) == bus and getattr(p, "pci_domain_id", domain) == domain]
  if not found:
    raise SystemExit(f"no GPU at PCI address {name}")
  return found[0]


def rank_main(rank:int, a, port:int, out) -> None:
  import torch, vllm
  from vllm.config import VllmConfig, set_current_vllm_config
  from vllm.distributed import init_distributed_environment, initialize_model_parallel
  from vllm.distributed.parallel_state import get_tp_group, graph_capture, set_custom_all_reduce
  names = a.gpus.split(",")
  # rank i on the i-th of the GPUs in torch's order, as vLLM places its workers: its custom all-reduce assumes it
  dev = sorted(index(torch, n) for n in names)[rank]
  torch.cuda.set_device(dev)
  config = VllmConfig()
  # the engine's own choice: vLLM's config turns its custom all-reduce off where the platform has none (an R9700)
  set_custom_all_reduce(not config.parallel_config.disable_custom_all_reduce)
  with set_current_vllm_config(config):
    init_distributed_environment(world_size=len(names), rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                 local_rank=dev, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=len(names))
    tp, joins = get_tp_group(), []
    for size in (int(b) for b in a.bytes.split(",")):
      x = torch.ones(max(size // 2, 1), dtype=torch.bfloat16, device="cuda")
      calls = max(4, min(a.calls, (64 << 20) * 4 // max(size, 1)))
      for _ in range(10):
        tp.all_reduce(x)
      torch.cuda.synchronize()
      with graph_capture(device=torch.device("cuda", dev)) as ctx, torch.cuda.graph(g := torch.cuda.CUDAGraph(), stream=ctx.stream):
        for _ in range(calls):
          tp.all_reduce(x)
      best = float("inf")
      for _ in range(a.reps):
        torch.distributed.barrier(); torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); g.replay(); end.record(); end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / calls)
      slowest = torch.tensor([best], dtype=torch.float64, device="cuda")
      torch.distributed.all_reduce(slowest, op=torch.distributed.ReduceOp.MAX)  # the join is done when the slowest rank is
      joins.append({"bytes": x.numel() * 2, "join_us": float(slowest.item()), "calls": calls})
    custom = getattr(tp.device_communicator, "ca_comm", None) is not None and not tp.device_communicator.ca_comm.disabled
    if rank == 0:
      out.put({"gpus": names, "backend": f"vLLM tensor-parallel all-reduce ({'custom all-reduce, ' if custom else ''}"
               f"{'RCCL' if torch.version.hip else 'NCCL'} {'.'.join(map(str, torch.cuda.nccl.version()))})",
               "graph": True, "dtype": "bfloat16", "reps": a.reps, "torch": torch.__version__, "vllm": vllm.__version__,
               "joins": joins, "torch_gpus": sorted(index(torch, n) for n in names)})


def main() -> int:
  ap = argparse.ArgumentParser()
  for flag, kind, default in (("--gpus", str, None), ("--bytes", str, LADDER), ("--reps", int, 10), ("--calls", int, 100)):
    ap.add_argument(flag, type=kind, default=default, required=default is None)
  a = ap.parse_args()
  if len(a.gpus.split(",")) < 2:
    raise SystemExit("a join needs two or more GPUs")
  import torch.multiprocessing as mp
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
  out = mp.get_context("spawn").SimpleQueue()
  mp.start_processes(rank_main, args=(a, port, out), nprocs=len(a.gpus.split(",")), join=True, start_method="spawn")
  print(json.dumps(out.get()), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
