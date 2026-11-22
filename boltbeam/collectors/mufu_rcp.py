"""collectors/mufu-rcp: the device's fast reciprocal (MUFU.RCP, PTX rcp.approx.ftz.f32) as a table, captured once on the
GPU, so BoltBeam's reference can compute what the engine's fast-math code computes for 1 / d.

Why: a kernel built with -use_fast_math computes x / d as x * MUFU.RCP(d) (an engine's q8_1 activation quantize: the
order row's source in data/region_reference.json cites it). MUFU.RCP is not the correctly rounded 1 / d, so a q at a
half-integer tie can differ by one (gen-quant-matmul-20261121 F3: 3.1e-6 of the values for a 1-ulp change). The
CPU check calls the host's correctly rounded 1 / d on both sides (device_only "reciprocal"); exactness against the
device needs this table (region_reference.rcp(d, device=True)). Until it is captured, a check whose bits depend on it
says so: pending rcp table.

    capture   python -m boltbeam.collectors.mufu_rcp capture OUT.bin [--arch sm_120]: on a machine with the GPU and nvcc.
              One kernel writes rcp.approx.ftz.f32 of every fp32 in [1, 2) (2^23 values: the mantissa's table, uint32
              bits), then checks on the device that the exponent only scales it (d x 2^k for k in -125 .. 125 on a
              sample of mantissas: rcp(d x 2^k) == rcp(d) x 2^-k bit for bit, where neither is past fp32's normal range)
              and the special inputs (+-0, +-inf, NaN, a subnormal, the largest normal). OUT.bin (32 MiB) and OUT.json
              (sha256, the device, driver and arch, the scaling check's counts) are written; the table is used only if
              the scaling check found no exception.
    table     the captured table (BOLTBEAM_MUFU_RCP: the .bin path; its .json beside it), or None: not captured.
    apply     MUFU.RCP of an fp32 array through the table; NoTable when there is none.

No GPU is touched outside `capture`."""
from __future__ import annotations

import functools, hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile
from boltbeam.target import toolkits

ENV = "BOLTBEAM_MUFU_RCP"
COUNT = 1 << 23

SOURCE = r"""
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
__device__ __forceinline__ float rcp(float d) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d)); return r; }
__global__ void table(uint32_t* out) {
  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < (1u << 23)) out[m] = __float_as_uint(rcp(__uint_as_float(0x3f800000u | m)));
}
// exponent scaling: mantissa m (every 4099th), exponent shift k: rcp(d 2^k) against rcp(d) 2^-k, where both are normal
__global__ void scaling(const uint32_t* tab, unsigned long long* bad, unsigned long long* seen) {
  uint32_t m = (blockIdx.x * blockDim.x + threadIdx.x) * 4099u;
  if (m >= (1u << 23)) return;
  for (int k = -125; k <= 125; k++) {
    float d = __uint_as_float(((uint32_t)(127 + k) << 23) | m), r = rcp(d);
    uint32_t t = tab[m], te = (t >> 23) & 0xff;
    int e = (int)te - k;
    if (e <= 0 || e >= 255) continue;   // the scaled table value leaves the normal range: not a claim here
    atomicAdd(seen, 1ull);
    if (__float_as_uint(r) != ((t & 0x807fffffu) | ((uint32_t)e << 23))) atomicAdd(bad, 1ull);
  }
}
__global__ void specials(uint32_t* out) {
  const uint32_t in[8] = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00000001u, 0x7f7fffffu, 0x00800000u};
  if (threadIdx.x < 8) out[threadIdx.x] = __float_as_uint(rcp(__uint_as_float(in[threadIdx.x])));
}
int main(int argc, char** argv) {
  if (argc < 2) return 2;
  uint32_t *dtab, *dsp; unsigned long long *dbad, *dseen;
  if (cudaMalloc(&dtab, 4u << 23) || cudaMalloc(&dsp, 32) || cudaMalloc(&dbad, 8) || cudaMalloc(&dseen, 8)) { fprintf(stderr, "cudaMalloc failed\n"); return 3; }
  cudaMemset(dbad, 0, 8); cudaMemset(dseen, 0, 8);
  table<<<(1u << 23) / 256, 256>>>(dtab);
  scaling<<<((1u << 23) / 4099 + 256) / 256, 256>>>(dtab, dbad, dseen);
  specials<<<1, 32>>>(dsp);
  if (cudaDeviceSynchronize()) { fprintf(stderr, "kernel failed\n"); return 4; }
  std::vector<uint32_t> tab(1u << 23); uint32_t sp[8]; unsigned long long bad, seen;
  cudaMemcpy(tab.data(), dtab, 4u << 23, cudaMemcpyDeviceToHost); cudaMemcpy(sp, dsp, 32, cudaMemcpyDeviceToHost);
  cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost); cudaMemcpy(&seen, dseen, 8, cudaMemcpyDeviceToHost);
  FILE* f = fopen(argv[1], "wb"); if (!f || fwrite(tab.data(), 4, tab.size(), f) != tab.size()) return 5; fclose(f);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int drv = 0; cudaDriverGetVersion(&drv);
  printf("{\"device\": \"%s\", \"cc\": %d, \"driver\": %d, \"scaling_checked\": %llu, \"scaling_exceptions\": %llu, \"specials\": [", p.name, p.major * 10 + p.minor, drv, seen, bad);
  for (int i = 0; i < 8; i++) printf("%s\"0x%08x\"", i ? ", " : "", sp[i]);
  printf("]}\n");
  return 0;
}
"""
SPECIAL_INPUTS = ("+0", "-0", "+inf", "-inf", "nan", "smallest subnormal", "largest normal", "smallest normal")


class NoTable(RuntimeError):
  """The device's reciprocal is not captured here: a value that depends on it is pending (rcp table)."""


def capture(out:str | pathlib.Path, arch:str = "sm_120") -> dict:
  """Build and run the capture on this machine's GPU (module docstring); returns the record written beside the table."""
  if (nvcc := toolkits.tool("nvcc", shutil.which)) is None:  # PATH, else the CUDA toolkit's roots (target/vendors.py)
    raise RuntimeError("no nvcc on this machine: the capture needs the CUDA toolkit and the GPU")
  out = pathlib.Path(out)
  with tempfile.TemporaryDirectory(prefix="mufu-rcp-") as tmp:
    src, exe = pathlib.Path(tmp) / "capture.cu", pathlib.Path(tmp) / "capture"
    src.write_text(SOURCE)
    subprocess.run([nvcc, f"-arch={arch}", "-O3", "-o", str(exe), str(src)], check=True)
    done = subprocess.run([str(exe), str(out)], check=True, capture_output=True, text=True)
  rec = json.loads(done.stdout)
  rec.update({"schema": "boltbeam.mufu_rcp.v1", "instruction": "rcp.approx.ftz.f32 (MUFU.RCP)", "arch": arch, "count": COUNT,
              "table": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "special_inputs": list(SPECIAL_INPUTS),
              "usable": rec["scaling_exceptions"] == 0 and rec["scaling_checked"] > 0})
  out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
  return rec


@functools.lru_cache(maxsize=1)
def table():
  """(the mantissa table as uint32 [2^23], its record) from BOLTBEAM_MUFU_RCP, or None when not captured (or not usable)."""
  import numpy as np
  path = os.environ.get(ENV)
  if not path or not pathlib.Path(path).exists():
    return None
  rec = json.loads(pathlib.Path(path).with_suffix(".json").read_text())
  data = pathlib.Path(path).read_bytes()
  if not rec.get("usable") or len(data) != 4 * COUNT or hashlib.sha256(data).hexdigest() != rec["sha256"]:
    return None
  return np.frombuffer(data, "<u4"), rec


def apply(d, tab=None):
  """MUFU.RCP of fp32 values through the table: rcp(m x 2^e) = table[m] x 2^-e; subnormal inputs read as zero (ftz);
  rcp(+-0) = +-inf, rcp(+-inf) = +-0, NaN stays NaN; a result past the normal range is flushed to a signed zero.
  tab: (table, record) (default: the captured one; NoTable when there is none)."""
  import numpy as np
  if (tab := tab or table()) is None:
    raise NoTable(f"the device's MUFU.RCP is not captured ({ENV}: python -m boltbeam.collectors.mufu_rcp capture): pending rcp table")
  t = tab[0]
  u = np.ascontiguousarray(np.asarray(d, np.float32)).view(np.uint32)
  sign, e, m = u & 0x80000000, ((u >> 23) & 0xff).astype(np.int64), u & 0x7fffff
  r = t[m]
  ne = ((r >> 23) & 0xff).astype(np.int64) - (e - 127)
  out = np.where((ne > 0) & (ne < 255), (r & 0x807fffff) | (np.clip(ne, 0, 254).astype(np.uint32) << 23), np.where(ne >= 255, 0x7f800000, 0)).astype(np.uint32)
  out = np.where(e == 0, 0x7f800000, out)                                    # +-0 and subnormals (read as zero): inf
  out = np.where(e == 255, np.where(m == 0, 0, 0x7fc00000), out)             # inf: 0; NaN: NaN
  return (out.astype(np.uint32) | sign.astype(np.uint32)).view(np.float32).reshape(np.shape(d))


def main(argv:list[str]) -> int:
  if len(argv) >= 2 and argv[0] == "capture":
    arch = argv[argv.index("--arch") + 1] if "--arch" in argv else "sm_120"
    print(json.dumps(capture(argv[1], arch), indent=1))
    return 0
  print(__doc__)
  return 2


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
