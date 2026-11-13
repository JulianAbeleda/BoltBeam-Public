"""The host side of an offload limit: host RAM read bandwidth (probed once, remembered per machine in
~/.boltbeam/host.json like the engine scan), the copy rate to the GPU from pinned and from pageable memory (probed once
per GPU, runtime/h2d_probe.py, remembered in host.json under the GPU's PCI bus id), the GPU's link to host memory
(target/host_links.py: read, not measured, the fallback when no copy can be measured), and the RAM there is now (/proc/meminfo, read at every
plan and never saved: what is free changes with what else runs). The CPU's dequant rate is not probed: nothing here
returns one, and offload_limit says so in its notes."""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import platform
import time
from typing import Any, Callable

from boltbeam.core.refused import Refused
from boltbeam.target.hardware import run_tool

FILE_ENV = "BOLTBEAM_HOST_FILE"
PROBE_BYTES, REPS = 256 << 20, 5  # far beyond any cache; the best repetition (a lower bound on peak)
RAM_MARGIN_BYTES = 2 * 2**30  # kept free of the plan: the system, the engine's own process and its page-cache churn
RAM_SHARE = 0.85  # of total RAM, when what is available cannot be read (an assumed share)


def host_file() -> pathlib.Path:
  return pathlib.Path(os.environ.get(FILE_ENV) or pathlib.Path.home() / ".boltbeam" / "host.json").expanduser()


def _sum_probe() -> tuple[float, str]:
  """Read bandwidth in GB/s with the standard library only: bytes.find of an absent byte over PROBE_BYTES of ones is one
  C memchr pass that reads every byte and writes none, best of REPS. One thread: find holds the GIL, so a second
  thread would not run beside it. A lower bound on the DDR peak (one core), about 80 percent of two cores' numpy sum."""
  try:
    buf = bytes(b"\x01") * PROBE_BYTES  # every page touched, far beyond any cache
  except MemoryError as exc:
    raise Refused(f"the RAM read probe needs {PROBE_BYTES >> 20} MiB of free memory: close other programs and try again") from exc
  best = 0.0
  for _ in range(REPS):
    t0 = time.perf_counter()
    buf.find(b"\x02")
    best = max(best, PROBE_BYTES / (time.perf_counter() - t0) / 1e9)
  return best, f"memory scan (Python standard library), {PROBE_BYTES >> 20} MiB, best of {REPS}: a lower bound on the DDR peak"


RAM_METHOD = "memscan-stdlib-1t"  # _sum_probe's method; a saved figure of another method (numpy's 2-thread sum, before 0.3.21) is not this one


def ram_read(*, probe:Callable[[], tuple[float, str]] = _sum_probe, path:pathlib.Path | None = None, remeasure:bool = False,
             method:str = RAM_METHOD) -> dict[str, Any]:
  """This machine's host RAM read bandwidth: the saved figure when it was measured by this method, else measured now and
  saved with its method. A figure saved by another method (or none recorded) is measured again; when that cannot run,
  the old figure stands, labelled with its method. A bad probe is a Refused."""
  path = path or host_file()
  saved = _load(path)
  have = isinstance(saved.get("ram_gbs"), (int, float)) and saved["ram_gbs"] > 0
  if have and not remeasure and saved.get("ram_method") == method:
    return saved
  try:
    gbs, how = probe()
  except Refused:
    if have and not remeasure:  # the old figure, said as another method's
      return {**saved, "ram_source": f"{saved.get('ram_source') or 'saved'} (measured by another method than BoltBeam's own now; not re-measured)"}
    raise
  if not isinstance(gbs, (int, float)) or not gbs > 0: raise Refused(f"the RAM probe returned {gbs!r} GB/s")
  saved = {**saved, "schema": "boltbeam.host.v1", "ram_gbs": gbs, "ram_source": how, "ram_method": method,
           "measured_at": datetime.date.today().isoformat()}
  _save(path, saved)
  return saved


def _load(path:pathlib.Path) -> dict[str, Any]:
  try:
    out = json.loads(path.read_text())
    return out if isinstance(out, dict) else {}
  except (OSError, ValueError):
    return {}


def _save(path:pathlib.Path, saved:dict[str, Any]) -> None:
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(saved, indent=1)); os.replace(tmp, path)  # a reader never sees half a file
  except OSError:
    pass  # an unwritable home only means the next call measures again


def valid_h2d(entry:Any) -> dict[str, Any] | None:
  """A saved copy-rate entry when it has both positive rates, a source and a date; else None (not measured)."""
  if not isinstance(entry, dict) or not isinstance(entry.get("h2d_gbs"), dict):
    return None
  rates = entry["h2d_gbs"]
  if any(not isinstance(rates.get(k), (int, float)) or not rates[k] > 0 for k in ("pinned", "pageable")):
    return None
  return {"h2d_gbs": {k: rates[k] for k in ("pinned", "pageable")}, "h2d_source": str(entry.get("h2d_source") or "measured"),
          "measured_at": str(entry.get("measured_at") or "date not recorded")}


def saved_h2d(pci_bus_id:str | None, path:pathlib.Path | None = None) -> dict[str, Any] | None:
  h = _load(path or host_file()).get("h2d")
  return valid_h2d(h.get((pci_bus_id or "").lower())) if isinstance(h, dict) else None


PROBE_TIMEOUT_S = 30  # the probe copies 256 MiB a dozen times: about a second; a hung driver must not hang the plan
_CHILD = """import json, sys
from boltbeam.runtime import h2d_probe
try:
  print(json.dumps(getattr(h2d_probe, sys.argv[2])(sys.argv[1])))
except h2d_probe.Unavailable as exc:
  print(json.dumps({"unavailable": str(exc)}))
except Exception as exc:  # any failure is a figure not measured
  print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
"""


def _probe_child(pci_bus_id:str, what:str = "measure") -> dict[str, Any]:
  """runtime/h2d_probe.measure (or .integrated) in its own process, killed after PROBE_TIMEOUT_S: what it printed, or {"error": why}."""
  import subprocess
  import sys
  root = str(pathlib.Path(__file__).resolve().parents[2])
  env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
  try:
    proc = subprocess.run([sys.executable, "-c", _CHILD, pci_bus_id, what], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, env=env)
  except subprocess.TimeoutExpired:
    return {"error": f"it did not finish in {PROBE_TIMEOUT_S} s and was stopped"}
  except OSError as exc:
    return {"error": f"it could not start: {exc}"}
  try:
    return json.loads(proc.stdout.strip().splitlines()[-1])
  except (ValueError, IndexError):
    tail = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
    return {"error": f"exit {proc.returncode}: {tail}"}


def gpu_busy(pci_bus_id:str) -> str | None:
  """Why a GPU here is not idle (another program holds one: providers.gpu_free, per vendor whose management tool is
  installed: target/vendors.py scan_tools), or None when every one is idle or no tool says. Any GPU in use counts:
  the probe's copies share the host's memory and link."""
  import shutil
  from boltbeam.collectors import providers
  from boltbeam.target import vendors
  for v in vendors.VENDORS.values():
    if v.profile is not None and v.scan_tools and shutil.which(v.scan_tools[0]):
      free = providers.gpu_free(v.backend)
      if not free["free"]:
        return free["reason"]
  return None


def _h2d_probe(pci_bus_id:str, *, child:Callable[[str], dict[str, Any]] = _probe_child,
               busy:Callable[[str], str | None] = gpu_busy) -> dict[str, Any]:
  """The copy rates measured now (runtime/h2d_probe.py, in a child process with a timeout), on an idle GPU only: a GPU
  another program uses before or during the probe gives a slow figure that looks real, so it is refused, not saved."""
  if why := busy(pci_bus_id):
    raise Refused(f"not measured while another program uses the GPU ({why})")
  got = child(pci_bus_id)
  if "unavailable" in got:
    raise Refused(f"no GPU runtime here (CUDA or HIP) can copy to this GPU: {got['unavailable']}")
  if "error" in got:
    raise Refused(f"the copy probe failed ({got['error']})")
  if why := busy(pci_bus_id):
    raise Refused(f"not kept: another program used the GPU while the copies were measured ({why})")
  return got


def h2d(pci_bus_id:str | None, *, probe:Callable[[str], dict[str, Any]] = _h2d_probe, path:pathlib.Path | None = None,
        remeasure:bool = False) -> dict[str, Any]:
  """This GPU's host-to-device copy rate, {"h2d_gbs": {"pinned", "pageable"}, "h2d_source", "measured_at"}: the saved
  figure for this PCI bus id, else measured now (runtime/h2d_probe.py) and saved under host.json's "h2d" by bus id. A
  Refused when there is no bus id, no runtime, or a probe that returned a rate that is not positive."""
  if not pci_bus_id:
    raise Refused("the GPU's PCI bus id is not known, so no copy to it can be measured")
  path = path or host_file()
  key = pci_bus_id.lower()
  have = saved_h2d(key, path)
  if have and not remeasure:
    return have
  got = probe(pci_bus_id)
  rates = {k: got.get(k) for k in ("pinned", "pageable")}
  if any(not isinstance(v, (int, float)) or not v > 0 for v in rates.values()):
    raise Refused(f"the copy probe returned {rates!r} GB/s")
  have = {"h2d_gbs": rates, "h2d_source": got.get("source") or "measured", "measured_at": datetime.date.today().isoformat()}
  saved = _load(path)  # read again after the probe: another process may have saved meanwhile
  prior = saved.get("h2d") if isinstance(saved.get("h2d"), dict) else {}
  saved = {**saved, "schema": "boltbeam.host.v1", "h2d": {**prior, key: have}}
  _save(path, saved)
  return have


def parse_meminfo(text:str) -> dict[str, int | None]:
  """/proc/meminfo's MemTotal and MemAvailable in bytes (the file states kB, which is KiB); None for a line it lacks.
  MemAvailable is the kernel's own estimate of what can be given to a new program without swapping: free pages plus
  the page cache and reclaimable slabs it can drop."""
  out: dict[str, int | None] = {"total": None, "available": None}
  for line in text.splitlines():
    key, _, rest = line.partition(":")
    name = {"MemTotal": "total", "MemAvailable": "available"}.get(key.strip())
    parts = rest.split()
    if name and parts and parts[0].isdigit():
      out[name] = int(parts[0]) * (1024 if parts[1:2] == ["kB"] else 1)
  return out


def parse_vm_stat(text:str, page_bytes:int | None = None) -> int | None:
  """macOS vm_stat's free, inactive, speculative and purgeable pages, in bytes: what can be given to a new program
  without swapping, the counterpart of MemAvailable. None when the page size or the free count is missing."""
  import re
  size = re.search(r"page size of (\d+) bytes", text)
  page = page_bytes or (int(size.group(1)) if size else None)
  pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages (free|inactive|speculative|purgeable):\s+(\d+)", text, re.M)}
  if not page or "free" not in pages:
    return None
  return page * sum(pages.values())


def memory(*, meminfo:str | None = None, system:str | None = None, run_command=run_tool) -> dict[str, Any]:
  """RAM now: {"total_bytes", "available_bytes" (None when it cannot be read), "source"}. Linux reads /proc/meminfo,
  macOS sysctl hw.memsize and vm_stat. Read at every call, never saved: what is available changes with what else runs."""
  system = system or platform.system()
  if system == "Linux" or meminfo is not None:
    try:
      m = parse_meminfo(meminfo if meminfo is not None else pathlib.Path("/proc/meminfo").read_text())
    except OSError:
      m = {"total": None, "available": None}
    total = m["total"] or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {"total_bytes": total, "available_bytes": m["available"], "source": "/proc/meminfo MemTotal and MemAvailable" if m["available"] else
            "/proc/meminfo has no MemAvailable line"}
  if system == "Darwin":
    rc, out, _ = run_command(("sysctl", "-n", "hw.memsize"))
    total = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
    rc, vm, _ = run_command(("vm_stat",))
    avail = parse_vm_stat(vm) if rc == 0 else None
    return {"total_bytes": total or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "available_bytes": avail,
            "source": "sysctl hw.memsize; vm_stat free, inactive, speculative and purgeable pages" if avail else "sysctl hw.memsize; vm_stat unreadable"}
  return {"total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "available_bytes": None,
          "source": f"no reading of available memory on {system}"}


def _g(n:float) -> str:
  return f"{n / 1e9:.1f} GB"


def usable_ram(total:float, available:float | None, *, host_tables:float = 0.0, margin:float = RAM_MARGIN_BYTES) -> dict[str, Any]:
  """The RAM a plan may fill with experts, and in plain words which figure it came from and why. With a reading of
  available memory: available less the margin, floored to a whole GiB so a rerun a few MB apart gives the same plan.
  Without one: RAM_SHARE of total, said to be an assumption. Never below zero. host_tables (the model's input
  embeddings, which llama.cpp mmaps: only the rows a prompt touches become resident, as page cache the system may
  drop) is not taken from it, only carried so the page can say it competes for RAM under pressure."""
  if available is not None:
    raw = available - margin
    usable = max(0.0, (raw // 2**30) * 2**30)
    why = (f"{_g(usable)} of RAM for the experts: {_g(available)} available now (of {_g(total)}), less {_g(margin)} kept free for the system and the engine"
           + ". Read when this plan was made: it changes with what else runs.")
    return {"usable_bytes": usable, "basis": "available", "total_bytes": total, "available_bytes": available, "margin_bytes": margin,
            "host_tables_bytes": host_tables, "says": why}
  usable = max(0.0, RAM_SHARE * total)
  return {"usable_bytes": usable, "basis": "share_of_total", "total_bytes": total, "available_bytes": None, "margin_bytes": total - RAM_SHARE * total,
          "host_tables_bytes": host_tables,
          "says": f"{_g(usable)} of RAM for the experts: {RAM_SHARE:.0%} of {_g(total)}, an assumed share, because how much is available now could not be read."}


def _integrated_probe(pci_bus_id:str, *, child:Callable[..., dict[str, Any]] = _probe_child) -> dict[str, Any]:
  """runtime/h2d_probe.integrated in a child process with a timeout: the GPU runtime's integrated attribute (a fact of the
  chip, so no idle GPU is needed). A Refused when no runtime can say."""
  got = child(pci_bus_id, "integrated")
  if "unavailable" in got:
    raise Refused(f"no GPU runtime here (CUDA or HIP) can say: {got['unavailable']}")
  if "error" in got or not isinstance(got.get("integrated"), bool):
    raise Refused(f"the runtime's answer was not read ({got.get('error') or got!r})")
  return got


def integrated(pci_bus_id:str | None, *, probe:Callable[[str], dict[str, Any]] | None = None, path:pathlib.Path | None = None,
               remeasure:bool = False) -> bool | None:
  """Whether this GPU shares the host's memory, as its runtime said it (host.json "integrated" by PCI bus id): the saved
  answer, else asked now when probe is given (`boltbeam host`, the offload page) and saved. None when not known."""
  if not pci_bus_id:
    return None
  path, key = path or host_file(), pci_bus_id.lower()
  saved = _load(path).get("integrated")
  have = saved.get(key) if isinstance(saved, dict) else None
  if isinstance(have, dict) and isinstance(have.get("integrated"), bool) and not remeasure:
    return have["integrated"]
  if probe is None:
    return None
  got = probe(pci_bus_id)
  doc = _load(path)
  prior = doc.get("integrated") if isinstance(doc.get("integrated"), dict) else {}
  _save(path, {**doc, "schema": "boltbeam.host.v1", "integrated": {**prior, key: {"integrated": got["integrated"], "source": got.get("source") or "runtime",
                                                                                   "measured_at": datetime.date.today().isoformat()}}})
  return got["integrated"]


def link(pci_bus_id:str | None, *, vendor:str | None = None, integrated:bool | None = None, **kw) -> dict[str, Any]:
  """The GPU's link to host memory as a record (target/host_links.py: the kind's row, its own rate, how engines use it
  and whether the path that decided it was checked on real hardware); the unknown row's when no detector answers.
  integrated: the runtime's saved answer (integrated()), which a verified PCIe record needs."""
  from boltbeam.target import host_links
  return host_links.detect(pci_bus_id, vendor=vendor, integrated=integrated, **kw)


def facts(pci_bus_id:str | None = None, *, h2d_probe:Callable[[str], dict[str, Any]] | None = None, meminfo:str | None = None,
          vendor:str | None = None, integrated_probe:Callable[[str], dict[str, Any]] | None = None, **kw) -> dict[str, Any]:
  """The host inputs of an offload limit: ram_gbs with its source, link (the GPU's link record, host_links.detect; vendor
  names the GPU's vendor when known) and cpu_gbs None (unmeasured).
  ram_bytes and ram_available_bytes: RAM now (memory(), read at this call, not saved). h2d_gbs: the measured
  host-to-device rate per host memory kind ({"pinned": GB/s, "pageable": GB/s}, with h2d_source) for this GPU, from
  host.json or measured now (h2d()); None when it cannot be measured (no bus id, no GPU runtime, a busy GPU). h2d_probe
  None (the default) uses only a saved figure: only `boltbeam host` and the offload page (screen offload) probe, passing
  _h2d_probe. h2d_source then says why and the limit uses the link rate. integrated_probe: likewise for whether the GPU
  shares the host's memory (integrated(); they pass _integrated_probe), else the saved answer only."""
  ram = ram_read(**{k: v for k, v in kw.items() if k in ("probe", "path", "remeasure")})
  try:
    integ = integrated(pci_bus_id, probe=integrated_probe, path=kw.get("path"))
  except Refused:
    integ = None
  rec = link(pci_bus_id, vendor=vendor, integrated=integ, **{k: v for k, v in kw.items() if k in ("sysfs", "run_command", "system")})
  mem = memory(meminfo=meminfo, **{k: v for k, v in kw.items() if k == "run_command"})
  try:
    if h2d_probe is None:
      have = saved_h2d(pci_bus_id, kw.get("path"))
      if not have:
        raise Refused("not measured yet")
    else:
      have = h2d(pci_bus_id, probe=h2d_probe, **{k: v for k, v in kw.items() if k in ("path", "remeasure")})
    rates, how, when, reason = have["h2d_gbs"], f"{have['h2d_source']} ({have['measured_at']})", have["measured_at"], None
  except Refused as exc:
    from boltbeam.target.host_links import stands_in
    rates, how, when, reason = None, f"no measured copy rate to the GPU: {exc}; {stands_in(rec)}", None, str(exc)
  return {"ram_bytes": mem["total_bytes"], "ram_available_bytes": mem["available_bytes"], "ram_bytes_source": mem["source"],
          "ram_gbs": ram["ram_gbs"], "ram_source": f"{ram['ram_source']} ({ram['measured_at']})", "link": rec, "cpu_gbs": None, "cpu_source": "CPU dequant rate not measured",
          "h2d_gbs": rates, "h2d_source": how, "h2d_measured_at": when, "h2d_reason": reason}


def _size_bytes(text:str) -> int | None:
  """A cache size as sysfs or lscpu print it ("32768K", "32 MiB", "1.5 MiB") in bytes; None when it does not parse."""
  import re
  m = re.fullmatch(r"\s*([\d.]+)\s*([KMG]?)(?:i?B)?\s*", text)
  return int(float(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]) if m else None


def l3_cache(*, sysfs:str = "/sys/devices/system/cpu/cpu0/cache", system:str | None = None, run_command=run_tool) -> dict[str, Any]:
  """One L3 instance's size, the cache the cores of cpu0 share (a chip with several L3s, one per core complex, has a copy
  thread see only its own): {"bytes" (None when unknown), "source"}. Linux: sysfs cache/index*/level 3, else lscpu's total
  over its instances. macOS: sysctl hw.l3cachesize (Apple silicon reports none: its last cache is the system cache)."""
  system = system or platform.system()
  if system == "Linux":
    for d in sorted(pathlib.Path(sysfs).glob("index*")):
      try:
        if (d / "level").read_text().strip() == "3" and (n := _size_bytes((d / "size").read_text())):
          return {"bytes": n, "source": f"sysfs {d.name}: {n >> 20} MiB shared by cpus {(d / 'shared_cpu_list').read_text().strip()}"}
      except OSError:
        continue
    rc, out, _ = run_command(("lscpu",))
    import re
    m = re.search(r"^L3 cache:\s*([\d.]+\s*[KMG]i?B)(?:\s*\((\d+) instances?\))?", out, re.M) if rc == 0 else None
    if m and (n := _size_bytes(m.group(1))):
      k = int(m.group(2) or 1)
      return {"bytes": n // k, "source": f"lscpu: {m.group(1)} in {k} instance(s)"}
    return {"bytes": None, "source": "no L3 in sysfs or lscpu"}
  if system == "Darwin":
    rc, out, _ = run_command(("sysctl", "-n", "hw.l3cachesize"))
    n = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
    return {"bytes": n, "source": "sysctl hw.l3cachesize"} if n > 0 else {"bytes": None, "source": "sysctl reports no L3 (Apple silicon: a system-level cache instead)"}
  return {"bytes": None, "source": f"no reading of the L3 cache on {system}"}


def _cpus(cpu_list:str) -> int:
  """How many CPUs a sysfs list names ("0-7,16-23")."""
  return sum(int(b) - int(a) + 1 for a, _, b in (x.partition("-") if "-" in x else (x, "", x) for x in cpu_list.strip().split(",") if x))


def cpu_cores(*, cpuinfo:str = "/proc/cpuinfo", sysfs:str = "/sys/devices/system/cpu/cpu0", system:str | None = None, run_command=run_tool) -> dict[str, Any]:
  """The CPU's topology: {"physical" cores, "logical" CPUs, "l3_cores" (the physical cores sharing cpu0's L3: one core
  complex on a chip with several), "source"}, None where unknown. Linux: the distinct (physical id, core id) pairs of
  /proc/cpuinfo, and cpu0's L3 sharing list over its SMT siblings (sysfs); macOS: sysctl hw.physicalcpu and hw.logicalcpu."""
  system = system or platform.system()
  if system == "Linux":
    try:
      blocks = [dict(ln.split(":", 1) for ln in b.splitlines() if ":" in ln) for b in pathlib.Path(cpuinfo).read_text().split("\n\n") if b.strip()]
    except OSError:
      blocks = []
    blocks = [{k.strip(): v.strip() for k, v in b.items()} for b in blocks if "processor" in {k.strip() for k in b}]
    cores = {(b.get("physical id"), b.get("core id")) for b in blocks if "core id" in b}
    l3 = None
    try:
      shared = next(_cpus((d / "shared_cpu_list").read_text()) for d in sorted(pathlib.Path(sysfs, "cache").glob("index*")) if (d / "level").read_text().strip() == "3")
      l3 = shared // _cpus(pathlib.Path(sysfs, "topology", "thread_siblings_list").read_text())
    except (OSError, StopIteration, ValueError, ZeroDivisionError):
      pass
    return {"physical": len(cores) or None, "logical": len(blocks) or None, "l3_cores": l3,
            "source": f"{cpuinfo} (physical id, core id) pairs and processors; {sysfs} L3 sharing over SMT siblings"}
  if system == "Darwin":
    got = [run_command(("sysctl", "-n", f"hw.{k}cpu")) for k in ("physical", "logical")]
    n = [int(o.strip()) if rc == 0 and o.strip().isdigit() else None for rc, o, _ in got]
    return {"physical": n[0], "logical": n[1], "l3_cores": None, "source": "sysctl hw.physicalcpu, hw.logicalcpu"}
  return {"physical": None, "logical": None, "l3_cores": None, "source": f"no reading of the CPU's cores on {system}"}
