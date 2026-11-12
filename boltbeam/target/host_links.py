"""The link between host memory and a GPU, whatever it is: the kinds are rows of data/host_links.json, tried in their
order, each with the detectors that recognise it, how its own rate is read, and how the engines BoltBeam sets up use it
(semantics: copy, direct_read or none). This module is the only place a link kind's own terms live (policy/guards.py
keeps them out of the offload and limit code); the rest of BoltBeam reads the record detect() returns:

  {"kind", "gbs" (the link's own rate, None when not known), "basis" ("wire": an upper bound from the line code;
   "reported": the tool's stated speed; "memory": no link, the GPU reads system memory at its bandwidth; None),
   "source", "semantics", "detector" (the detector that decided it), "verified" (True only when that detector is one of
   the row's verified_by, checked on real hardware, and the row's requires hold), "why_unverified", "words", "detail"}

The measured copy rate (target/host.h2d, keyed by bus id) is not part of the record: it is per host memory kind
(pinned, pageable) and always comes before the record's own rate."""
from __future__ import annotations

import json
import pathlib
import platform
import re
from functools import lru_cache
from typing import Any, Callable

from boltbeam.target.hardware import run_tool

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "host_links.json"
SEMANTICS = ("copy", "direct_read", "none")


@lru_cache(maxsize=1)
def rows() -> tuple[dict[str, Any], ...]:
  out = tuple(json.loads(TABLE.read_text())["links"])
  for r in out:
    if r["semantics"] not in SEMANTICS or not set(r["can"]) <= set(SEMANTICS) or r["semantics"] not in r["can"]:
      raise ValueError(f"host_links.json row {r['kind']}: semantics {r['semantics']!r} / can {r['can']!r} outside {SEMANTICS}")
    if any(d not in DETECTORS for d in r["detect"]) or not set(r["verified_by"]) <= set(r["detect"]):
      raise ValueError(f"host_links.json row {r['kind']}: unknown detector in {r['detect']!r} or verified_by {r['verified_by']!r} outside it")
  if [r.get("fallback") for r in out].count(True) != 1 or not out[-1].get("fallback"):
    raise ValueError("host_links.json: exactly one fallback row, the last")
  return out


def row(kind:str) -> dict[str, Any]:
  return next(r for r in rows() if r["kind"] == kind)


# --- detectors: (ctx, row) -> {"gbs", "basis", "source", "detail"} or None (not this kind) --------------------------------

def _line_rate(gts:float, width:int, code:list[list[Any]]) -> float:
  """One direction's wire rate in GB/s: GT/s x the line code's payload over code bits / 8 x lanes."""
  num, den = next((n, d) for top, n, d in code if top is None or gts <= top)
  return gts * (num / den) / 8 * width


def _sysfs_link(d:pathlib.Path, which:str) -> tuple[float, int]:
  return float((d / f"{which}_link_speed").read_text().split()[0]), int((d / f"{which}_link_width").read_text())


def _sysfs_pcie(ctx:dict[str, Any], r:dict[str, Any]) -> dict[str, Any] | None:
  """sysfs max_link_speed/width of the GPU and its upstream port (the smaller: a gen5 GPU in a gen4 slot is gen4), and
  what it trained to now (an idle GPU trains down). The limit uses the maximum: a wire rate, an upper bound."""
  parts = (ctx.get("pci_bus_id") or "").lower().split(":")
  if len(parts) != 3:
    return None
  d = pathlib.Path(ctx["sysfs"]) / f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
  try:
    (gts, width), (pgts, pwidth), now = _sysfs_link(d, "max"), _sysfs_link(d.resolve().parent, "max"), _sysfs_link(d, "current")
  except (OSError, ValueError, IndexError):
    return None
  gts, width = min(gts, pgts), min(width, pwidth)
  return {"gbs": _line_rate(gts, width, r["line_code"]), "basis": "wire",
          "source": f"sysfs {d.name}: {gts:g} GT/s x{width}, the smaller of the GPU's and its slot's maximum",
          "detail": {"gen": {float(k): v for k, v in r["gens_by_gts"].items()}.get(gts), "width": width, "now_gbs": _line_rate(*now, r["line_code"])}}


def _nvidia_smi_pcie(ctx:dict[str, Any], r:dict[str, Any]) -> dict[str, Any] | None:
  """nvidia-smi's maximum gen and width for this GPU and system."""
  _, out, _ = ctx["run_command"](("nvidia-smi", "--query-gpu=pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"))
  try:
    gen, width = (int(x) for x in out.splitlines()[0].split(","))
    gts = {v: float(k) for k, v in r["gens_by_gts"].items()}[gen]
  except (ValueError, KeyError, IndexError):
    return None
  return {"gbs": _line_rate(gts, width, r["line_code"]), "basis": "wire", "source": f"nvidia-smi: gen {gen} x{width}, the maximum for this GPU and system",
          "detail": {"gen": gen, "width": width, "now_gbs": None}}


def _nvidia_smi_addressing_mode(ctx:dict[str, Any], r:dict[str, Any]) -> dict[str, Any] | None:
  """A GPU that reads host memory through the host's own page tables (nvidia-smi -q 'Addressing Mode: ATS', the hardware
  path of an NVLink-C2C system; a PCIe GPU says HMM or None), with the C2C links' speed from 'nvidia-smi c2c -s' when it
  states one (their sum; whether one direction or both is not verified)."""
  bus = ctx.get("pci_bus_id")
  if not bus:
    return None
  rc, out, _ = ctx["run_command"](("nvidia-smi", "-q", "-i", bus))
  mode = re.search(r"^\s*Addressing Mode\s*:\s*(\S+)", out or "", re.M) if rc == 0 else None
  if not mode or mode.group(1).upper() != "ATS":
    return None
  c2c = re.search(r"^\s*GPU C2C Mode\s*:\s*(\S+)", out, re.M)
  rc, status, _ = ctx["run_command"](("nvidia-smi", "c2c", "-s", "-i", bus))
  speeds = [float(x) for x in re.findall(r"C2C Link \d+\s*:\s*([\d.]+)\s*GB/s", status or "")] if rc == 0 else []
  return {"gbs": sum(speeds) or None, "basis": "reported" if speeds else None,
          "source": "nvidia-smi -q: Addressing Mode ATS" + (f", GPU C2C Mode {c2c.group(1)}" if c2c else "")
                    + (f"; nvidia-smi c2c -s: {len(speeds)} link(s), {sum(speeds):g} GB/s in all" if speeds else "; nvidia-smi c2c -s stated no speed"),
          "detail": {"links": len(speeds)}}


def _runtime_integrated(ctx:dict[str, Any], r:dict[str, Any]) -> dict[str, Any] | None:
  """The GPU runtime said the GPU is integrated with host memory (CUDA's CU_DEVICE_ATTRIBUTE_INTEGRATED, HIP's
  hipDeviceAttributeIntegrated; target/host.integrated saves the answer per bus id)."""
  if ctx.get("integrated") is not True:
    return None
  return {"gbs": None, "basis": "memory", "source": "the GPU runtime says the GPU is integrated with host memory", "detail": {}}


def _vendor_shares_memory(ctx:dict[str, Any], r:dict[str, Any]) -> dict[str, Any] | None:
  """A GPU whose vendor row has no profile shape of its own shares the CPU's memory (target/vendors.py). Without a named
  vendor: no PCI bus id, and such a row answers on this OS."""
  from boltbeam.target import vendors
  name = ctx.get("vendor")
  if name:
    v = vendors.VENDORS.get(name)
    shares = v is not None and v.profile is None
  else:
    shares = not ctx.get("pci_bus_id") and any(v.profile is None and v.platform == ctx["system"] for v in vendors.VENDORS.values())
  if not shares:
    return None
  return {"gbs": None, "basis": "memory", "source": "the GPU's vendor row shares the CPU's memory (target/vendors.py)", "detail": {}}


DETECTORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]] = {
  "sysfs_pcie": _sysfs_pcie, "nvidia_smi_pcie": _nvidia_smi_pcie, "nvidia_smi_addressing_mode": _nvidia_smi_addressing_mode,
  "vendor_shares_memory": _vendor_shares_memory, "runtime_integrated": _runtime_integrated}


def _record(r:dict[str, Any], got:dict[str, Any], detector:str | None, ctx:dict[str, Any]) -> dict[str, Any]:
  missing = [k for k, want in (r.get("requires") or {}).items() if ctx.get(k) != want]
  checked = detector in r["verified_by"]
  why = None if checked and not missing else r.get("requires_why") if checked else "this way of reading it was never checked on real hardware"
  return {"kind": r["kind"], "gbs": got["gbs"], "basis": got["basis"], "source": got["source"], "semantics": r["semantics"], "detector": detector,
          "verified": why is None, "why_unverified": why, "words": r["words"], "detail": got["detail"]}


def detect(pci_bus_id:str | None, *, vendor:str | None = None, integrated:bool | None = None, sysfs:str = "/sys/bus/pci/devices",
           run_command=run_tool, system:str | None = None) -> dict[str, Any]:
  """The GPU's link to host memory as a record (module docstring): the first row whose detector answers, else the
  fallback row's (unverified, no rate). integrated: the GPU runtime's saved answer (target/host.integrated), None when
  not read."""
  ctx = {"pci_bus_id": pci_bus_id, "vendor": vendor, "integrated": integrated, "sysfs": sysfs, "run_command": run_command,
         "system": system or platform.system()}
  for r in rows():
    for name in r["detect"]:
      got = DETECTORS[name](ctx, r)
      if got is not None:
        return _record(r, got, name, ctx)
  tried = ", ".join(d for r in rows() for d in r["detect"])
  return _record(rows()[-1], {"gbs": None, "basis": None, "source": f"no detector answered ({tried})", "detail": {}}, None, ctx)


def of(host:dict[str, Any]) -> dict[str, Any] | None:
  """The link record of a host facts dict: its "link", or one read from the fields a host dict carried before the record
  (pcie_gbs, pcie_source, pcie_gen, pcie_width: a PCIe link's wire rate); None when it has neither."""
  if isinstance(host.get("link"), dict):  # the row's words as they are now; verified only if its detector still counts
    saved = host["link"]
    r = next((x for x in rows() if x["kind"] == saved.get("kind")), rows()[-1])
    ok = saved.get("verified") is True and saved.get("detector") in r["verified_by"]
    return {**saved, "words": r["words"], "verified": ok, "why_unverified": None if ok else saved.get("why_unverified") or "saved before its check was recorded"}
  if not isinstance(host.get("pcie_gbs"), (int, float)):
    return None
  r = row("pcie")
  return {**_record(r, {"gbs": host["pcie_gbs"], "basis": "wire", "source": host.get("pcie_source") or "saved before the link record",
                        "detail": {"gen": host.get("pcie_gen"), "width": host.get("pcie_width"), "now_gbs": None}}, None, {}),
          "why_unverified": "saved before the link record, without whether the GPU shares the CPU's memory"}


def words(link:dict[str, Any] | None, key:str, **kw) -> str:
  """A plain-words phrase of the link's row ("name", "short", "the", "detail", "reads"); a link not known reads as
  'the host link'."""
  if not link:
    return {"name": "host link", "short": "the host link", "the": "the host link"}.get(key, "")
  w = link["words"]
  text = w.get(f"{key}_unknown") if key == "detail" and link.get("gbs") is None and f"{key}_unknown" in w else w.get(key, "")
  return text.format(**{**(link.get("detail") or {}), **{k: v for k, v in kw.items()}})


def unverified(link:dict[str, Any] | None) -> list[str]:
  """The note an unverified row's number carries: [] for a verified one."""
  said = unverified_words(link)
  return [f"{words(link, 'name')}: {said}"] if said else []


def unverified_words(link:dict[str, Any] | None) -> str | None:
  """'not verified on this hardware' and why, for a record that is not verified (or no record); None when verified."""
  if link and link.get("verified"):
    return None
  why = (link or {}).get("why_unverified") or ("no link was read" if not link else None)
  return "not verified on this hardware" + (f": {why}" if why else "")


_BASIS = {"wire": "{the}'s wire rate stands in, an upper bound", "reported": "{the}'s reported speed stands in (not measured)",
          "memory": "no copy is needed: the GPU reads {the} at the memory's bandwidth", None: "nothing stands in: {the}'s speed was not read"}


def stands_in(link:dict[str, Any] | None) -> str:
  """What stands in for a copy rate not measured, in words: the link's own rate and what kind of figure it is."""
  return _BASIS[(link or {}).get("basis")].format(the=words(link, "the"))
