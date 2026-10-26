"""Conservative matrix-grid constraints from the installed engine's configuration tables.

Every configuration matching the launch template and block size remains possible. No GPU generation is selected
here. A grid rules out a logical shape only when none of those source configurations could have launched it.
"""
from __future__ import annotations

import hashlib, math, pathlib, re, subprocess
from typing import Any

from boltbeam.collectors.instruction_path import template
from boltbeam.collectors.gemv_reference import GGML_TYPES


def constraints(root:pathlib.Path, sm_count:int, build_commit:str | None = None) -> dict[str, Any] | None:
  path = root / 'mmq.cuh'
  if not path.is_file() or sm_count <= 0: return None
  text = path.read_text()
  # Refuse a changed launch contract. The branch threshold itself is read, never retuned by BoltBeam.
  needed = ('(args.nrows_x   + config.I - 1) / config.I', '(args.ncols_max + config.J - 1) / config.J',
            'const int ntiles_dst = ntx * nty * ntzw;', '100 * ntiles_dst / (nsm*tiles_nwaves)')
  threshold = re.search(r'GGML_CUDA_CC_IS_NVIDIA\(cc\) && tiles_efficiency_percent >= (\d+) \? ntiles_dst : nsm', text)
  if not all(x in text for x in needed) or threshold is None: return None
  # The fallback template argument is chosen by the matrix's rows; an older source without this rule constrains nothing.
  fallback = re.search(r'ggml_cuda_mmq_needs_fallback\(const int64_t nrows_x\) \{\s*return nrows_x % (\d+) != 0;', text)
  if build_commit is not None:
    # The captured engine reports its source revision. Only use those exact committed source bytes.
    try:
      commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
      if not build_commit or not commit.startswith(build_commit): return None
      prefix = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-prefix"], text=True).strip()
    except (OSError, subprocess.SubprocessError): return None
  configs = set()
  sources = {path.name: hashlib.sha256(text.encode()).hexdigest()}
  for name in re.findall(r'#include "(mmq-config-[^"]+)"', text):
    table = root / name
    if not table.is_file(): return None
    content = table.read_text(); sources[name] = hashlib.sha256(content.encode()).hexdigest()
    for args in re.findall(r'CASE\(([^)]*)\)', content):
      fields = [x.strip() for x in args.split(',')]
      if fields[0].removeprefix('GGML_TYPE_') not in GGML_TYPES: continue
      if len(fields) != 9 or not all(fields[i].isdigit() for i in (1, 3, 4)) or fields[7] not in ('true', 'false') or fields[8] not in ('true', 'false'):
        return None  # A known type with an unsupported configuration cannot be omitted from the union.
      configs.add((GGML_TYPES[fields[0].removeprefix('GGML_TYPE_')], int(fields[1]), int(fields[3]), int(fields[4]), fields[7] == 'true', int(fields[8] == 'true')))
  if build_commit is not None:
    for name, digest in sources.items():
      try:
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{prefix}{name}"], stderr=subprocess.DEVNULL)
      except (OSError, subprocess.SubprocessError): return None
      if hashlib.sha256(committed).hexdigest() != digest: return None
  return {'build_commit': build_commit, 'configs': sorted(configs), 'sm_count': sm_count, 'efficiency_percent': int(threshold[1]), 'sources': sources,
          **({'fallback_rows': int(fallback[1])} if fallback else {})} if configs else None


def fits(proof:dict, launch:dict, rows:list[dict], m:float, fused:int, split:int) -> bool:
  name = re.fullmatch(r'mul_mat_q<\(ggml_type\)(\d+),(\d+),(0|1),\(ggml_prec\)\d+>', template(launch['key']))
  geometry = re.search(r' grid=(\d+)x(\d+)x(\d+) block=(\d+)x(\d+)x(\d+)', launch['key'])
  if not name or not geometry or not rows or any(not r.get('n') or not r.get('k') for r in rows): return True
  quant, width, fallback = map(int, name.groups())
  gx, gy, gz, bx, by, bz = map(int, geometry.groups())
  options = [c for c in proof['configs'] if (c[0], c[1], c[3], c[5]) == (quant, bx * by * bz, width, fallback)]
  if not options: return True
  for row in rows:
    if proof.get('fallback_rows') and (row['n'] % proof['fallback_rows'] != 0) != bool(fallback): continue
    for _, _, height, width, stream, _ in options:
      ny = math.ceil(m / split / width)
      # Fused products can pack rows or occupy separate tensor planes. Preserve both layouts;
      # rounding the combined row extent alone would rule out valid non-tile-aligned products.
      # An expert stack (mul_mat_id) tiles every expert's plane at the call's token columns.
      e = row.get('n_expert') or 1
      grids = {(math.ceil(row['n'] * fused / height), ny, e), (math.ceil(row['n'] / height), ny, fused * e)}
      for expected in grids:
        if stream:
          tiles, sm = math.prod(expected), proof['sm_count']
          efficiency = 100 * tiles // (sm * math.ceil(tiles / sm))
          expected = (tiles if efficiency >= proof['efficiency_percent'] else sm, 1, 1)
        if expected == (gx, gy, gz): return True
  return False
