"""Best-effort capture of the exact AMD execution environment."""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from boltbeam.artifacts.base import EvidenceSource, sha256_bytes
from boltbeam.core.facts import Fact, TruthStatus
from boltbeam.core.system import SystemSnapshot

RunCommand = Callable[[tuple[str, ...]], tuple[int, str, str]]


def _run(argv:tuple[str, ...]) -> tuple[int, str, str]:
  try:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    return proc.returncode, proc.stdout, proc.stderr
  except (OSError, subprocess.TimeoutExpired) as exc:
    return 127, "", str(exc)


def _source(argv:tuple[str, ...], stdout:str, stderr:str = "") -> EvidenceSource:
  raw = (stdout + ("\nSTDERR:\n" + stderr if stderr else "")).encode()
  return EvidenceSource("boltbeam", "system_snapshot", "command:" + " ".join(argv), sha256_bytes(raw),
                        {"argv": list(argv)})


def _fact(name:str, value:object, source:EvidenceSource | None, unit:str | None = None) -> Fact:
  if value is None: return Fact(name, None, TruthStatus.UNKNOWN.value, unit=unit)
  return Fact(name, value, TruthStatus.MEASURED.value, unit=unit, sources=(source,) if source else ())


def _match(text:str, patterns:tuple[str, ...], cast:Callable[[str], object] = str) -> object | None:
  for pattern in patterns:
    match = re.search(pattern, text, re.I | re.M)
    if match:
      try: return cast(match.group(1).strip())
      except (TypeError, ValueError): return None
  return None


def _rocminfo_gpu_agent(text:str) -> str:
  """Return the rocminfo agent section whose Name is a gfx architecture."""
  starts = [match.start() for match in re.finditer(r"(?m)^\s*Agent\s+\d+\s*$", text)]
  if not starts:
    return text if re.search(r"(?m)^\s*Name:\s*gfx\d+\s*$", text) else ""
  starts.append(len(text))
  for begin, end in zip(starts, starts[1:]):
    section = text[begin:end]
    if re.search(r"(?m)^\s*Name:\s*gfx\d+\s*$", section): return section
  return ""


def _git_fact(repo:Path, name:str, run:RunCommand) -> Fact:
  argv = ("git", "-C", str(repo), "rev-parse", "HEAD")
  code, out, err = run(argv); source = _source(argv, out, err)
  value = out.strip() if code == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", out.strip()) else None
  return _fact(name, value, source)


def capture_system_snapshot(*, run_command:RunCommand = _run, boltbeam_repo:str | Path | None = None,
                            tinygrad_repo:str | Path | None = None,
                            environment:tuple[str, ...] = ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES",
                                                            "ROCR_VISIBLE_DEVICES"),
                            captured_at:str | None = None) -> SystemSnapshot:
  facts: list[Fact] = []
  outputs: dict[str, tuple[str, EvidenceSource]] = {}
  probes = {
    "rocminfo": ("rocminfo",), "smi": ("rocm-smi", "--showproductname", "--showuniqueid", "--showvbios",
                                          "--showclocks", "--showmeminfo", "vram"),
    "lspci": ("lspci", "-Dnn"), "uname": ("uname", "-srvmo"), "hipcc": ("hipcc", "--version"),
  }
  for key, argv in probes.items():
    code, out, err = run_command(argv)
    outputs[key] = (out if code == 0 else "", _source(argv, out, err))

  rocm, rocm_src = outputs["rocminfo"]
  rocm_gpu = _rocminfo_gpu_agent(rocm)
  smi, smi_src = outputs["smi"]
  pci, pci_src = outputs["lspci"]
  uname, uname_src = outputs["uname"]
  hipcc, hipcc_src = outputs["hipcc"]

  facts.extend([
    _fact("gpu.rocminfo.marketing_name", _match(rocm_gpu, (r"Marketing Name:\s*(.+)", r"Name:\s*(AMD Radeon.+)")), rocm_src),
    _fact("gpu.smi.marketing_name", _match(smi, (r"Card series:\s*(.+)", r"Card model:\s*(.+)", r"Device Name:\s*(.+)")), smi_src),
    _fact("gpu.uuid", _match(rocm_gpu, (r"Uuid:\s*(\S+)",)), rocm_src),
    _fact("gpu.unique_id", _match(smi, (r"Unique ID:\s*(\S+)",)), smi_src),
    _fact("gpu.architecture", _match(rocm_gpu, (r"Name:\s*(gfx\d+)",)), rocm_src),
    _fact("gpu.compute_units", _match(rocm_gpu, (r"Compute Unit:\s*(\d+)", r"Compute Unit Count:\s*(\d+)"), int), rocm_src,
          "count"),
    _fact("gpu.simd_per_cu", _match(rocm_gpu, (r"SIMDs per CU:\s*(\d+)",), int), rocm_src, "count"),
    _fact("gpu.vram_bytes", _match(smi, (r"Total Memory \(B\):\s*(\d+)", r"VRAM Total Memory \(B\):\s*(\d+)"), int),
          smi_src, "bytes"),
    _fact("gpu.cache_l1_bytes", _match(rocm_gpu, (r"Cache Info:.*?L1.*?Size:\s*(\d+)",), int), rocm_src, "bytes"),
    _fact("gpu.cache_l2_bytes", _match(rocm_gpu, (r"Cache Info:.*?L2.*?Size:\s*(\d+)",), int), rocm_src, "bytes"),
    _fact("gpu.max_waves_per_cu", _match(rocm_gpu, (r"Max Waves Per CU:\s*(\d+)",), int), rocm_src, "count"),
    _fact("gpu.vbios", _match(smi, (r"VBIOS version:\s*(\S+)",)), smi_src),
    _fact("gpu.core_clock_mhz", _match(smi, (r"sclk clock level:\s*\d+:\s*(\d+)Mhz", r"SCLK:\s*(\d+)Mhz"), int),
          smi_src, "MHz"),
    _fact("gpu.memory_clock_mhz", _match(smi, (r"mclk clock level:\s*\d+:\s*(\d+)Mhz", r"MCLK:\s*(\d+)Mhz"), int),
          smi_src, "MHz"),
    _fact("gpu.pci.address", _match(pci, (r"^(\S+)\s+.*(?:VGA|Display).*AMD",)), pci_src),
    _fact("gpu.pci.vendor_device", _match(pci, (r"(?:VGA|Display).+\[([0-9a-f]{4}:[0-9a-f]{4})\]",)), pci_src),
    _fact("gpu.pci.revision", _match(pci, (r"\(rev\s+([0-9a-f]+)\)",)), pci_src),
    _fact("gpu.power_policy", _match(smi, (r"Power Profile:\s*(.+)",)), smi_src),
    _fact("software.kernel", uname.strip() or None, uname_src),
    _fact("software.rocm_runtime", _match(rocm, (r"ROCm Runtime Version:\s*(.+)", r"Runtime Version:\s*(.+)")), rocm_src),
    _fact("software.hip_compiler", hipcc.strip() or None, hipcc_src),
    _fact("software.amdgpu_driver", None, None),
    _fact("software.gpu_firmware", None, None),
  ])
  names = [f.value for f in facts if f.name in {"gpu.rocminfo.marketing_name", "gpu.smi.marketing_name"} and f.value]
  conflict_inputs = tuple(f.fact_id for f in facts if f.name in {"gpu.rocminfo.marketing_name", "gpu.smi.marketing_name"})
  facts.append(Fact("gpu.identity.conflict", len(set(names)) > 1, TruthStatus.DERIVED.value,
                    derivation="distinct observed GPU marketing identities", input_fact_ids=conflict_inputs))

  bb_repo = Path(boltbeam_repo or Path(__file__).resolve().parents[2])
  tg_repo = Path(tinygrad_repo or bb_repo.parent / "tinygrad-arkey")
  facts.extend((_git_fact(bb_repo, "software.boltbeam.commit", run_command),
                _git_fact(tg_repo, "software.tinygrad.commit", run_command)))
  for key in environment:
    value = os.environ.get(key)
    facts.append(Fact(f"environment.{key.lower()}", value, TruthStatus.IMPORTED.value) if value is not None
                 else Fact(f"environment.{key.lower()}", None, TruthStatus.UNKNOWN.value))
  timestamp = captured_at or datetime.now(timezone.utc).isoformat()
  return SystemSnapshot(tuple(facts), captured_at=timestamp)


def write_system_snapshot(snapshot:SystemSnapshot, path:str | Path) -> None:
  import json
  target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(snapshot.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
