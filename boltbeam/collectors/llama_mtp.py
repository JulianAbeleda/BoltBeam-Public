"""llama.cpp's multi-token prediction (MTP) on llama-server, measured as collectors/mtp.py measures any engine (one
record shape, one sweep). The draft head is a GGUF of its own beside the model (a sidecar, heads()), passed as
-md HEAD --spec-type draft-mtp --spec-draft-n-max K; k=0 is the same server without it. Each k is one server start
through runtime/llama_server_decode_drive.py --mtp-prompts on its own free port, in its own process group: the driver
stops the group on any exit it can catch (its own deadline too), and collectors/mtp.sweep stops the group the driver
recorded (SERVER_GROUP) when the driver itself was killed. Decode speed is the server's own predicted_per_second,
(predicted_n - 1) / predicted_ms: no prefill. The acceptance is the server's spec-decode counters (--metrics) per request.

Folded in from the hand runs of 2026-11-10/12 (boltbeam-runs/headtohead-20261110 arm4.sh/drive4.py,
decode-parity-20261112 llama_arm.sh/drive.py): the flags (-ngl 99, -np 1, -md, --spec-type draft-mtp,
--spec-draft-n-max), temperature 0 and the prompt cache off per request, the timings fields, and the quiet-CPU gate
(runtime/mtp_shared.quiet, now both engines'). Not folded: their fixed port and kill by
pid (the driver's free port and process group replace them), their warm passes and repetitions (one warm request and one
timed request per prompt and depth, as on vLLM), and chat requests (token ids here, the chat template applied by the
server's /apply-template, so every k reads the same ids)."""
from __future__ import annotations

import json, pathlib, sys
from typing import Any

from boltbeam.collectors import capture_settings as cs, engine_common as ec, llama_server_decode as lsd, mtp

PROVIDER = "llama.cpp"
GPU_SHARE = 1.0  # llama.cpp takes no share of the card: the long context is sized on all of it, less RESERVE
RESERVE = 1 << 30  # llama.cpp's own -fit target, 1024 MiB free (common_params_fit, -fitt), as workflow/recommend.ENGINES states it
DRIVER_GRACE_S = 300  # how long past its own deadline the driver may take to stop its server before it is killed
TIMEOUT_S = 2 * 3600  # one server start: a stated guard on a hung server, not a measured length (a start, 16 prompts, a corpus read)


def _blocks(names:list[str]) -> set[int]:
  return {int(n.split(".")[1]) for n in names if n.startswith("blk.")}


def heads(model:str | pathlib.Path) -> list[pathlib.Path]:
  """The MTP head sidecars beside a GGUF model: the GGUFs in its folder of its architecture that declare draft layers
  (<arch>.nextn_predict_layers) and hold no block but those (the last of <arch>.block_count). An mmproj, a shard or
  another model is no head."""
  from boltbeam.profile import gguf
  path = pathlib.Path(model).expanduser()
  if path.suffix != ".gguf" or not path.is_file():
    return []
  arch = gguf.read_gguf_layout(path)[0].get("general.architecture")
  out = []
  for f in sorted(path.parent.glob("*.gguf")):
    if f.resolve() == path.resolve():
      continue
    try:
      kv, ts, _ = gguf.read_gguf_layout(f)
    except (ValueError, EOFError, OSError):
      continue
    draft, blocks = int(kv.get(f"{arch}.nextn_predict_layers") or 0), _blocks([t[0] for t in ts])
    if kv.get("general.architecture") == arch and draft and blocks and min(blocks) >= int(kv.get(f"{arch}.block_count") or 0) - draft:
      out.append(f)
  return out


def offer(model:str, draft_bytes:int, *, facts:bool) -> dict[str, Any] | None:
  """What Measure MTP runs for this checkpoint: the one MTP head beside it, else None; several are named, not guessed."""
  from boltbeam.report.plain import words
  found, w = heads(model), words()["mtp_pick"]
  if not found:
    return None
  blocked = w["offer_heads"].format(heads=", ".join(f.name for f in found)) if len(found) > 1 else None if facts else w["offer_facts"]
  return {"ks": list(mtp.KS), "blocked": blocked, "head": str(found[0]) if len(found) == 1 else None}


def resolve(model:str, head:str | None, server:str | None) -> tuple[pathlib.Path, str]:
  """The draft head (the one named, else the one heads() finds) and the server binary, or MtpRefused saying which is missing."""
  if head and not pathlib.Path(head).expanduser().is_file():
    raise mtp.MtpRefused(f"the draft head {head} is not a file")
  found = [pathlib.Path(head).expanduser()] if head else heads(model)
  if len(found) != 1:
    said = ", ".join(f.name for f in found) + " are all MTP heads" if found else "no MTP head is"
    raise mtp.MtpRefused(f"{said} beside {pathlib.Path(model).name}: name the one to measure (--draft HEAD)")
  if (server := server or lsd.find()) is None:
    raise mtp.MtpRefused(f"llama-server was not found: set ${lsd.ENV}")
  return found[0], server


def spec_flags(head:pathlib.Path, k:int) -> list[str]:
  return ["-md", str(head.absolute()), "--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)] if k else []


def fit_context(model:str, head:str | None, *, k:int, need:int, out:pathlib.Path, server:str | None = None) -> tuple[int | None, str]:
  """The largest server context llama.cpp's own fit (common_fit_params, with the draft head and its context at draft
  count k, to its free-memory target) holds on this GPU's free memory now, and how it says so: a server started with the fit on
  (no -c, no -ngl) and stopped as soon as it has logged the fit, before the weights load. None: the model's own context
  fits. need and the context returned are the measurement's, the server's less the driver's HEADROOM; the fit never goes
  below need (-fitc): MtpRefused when even that fits only with layers off the GPU."""
  from boltbeam.report.plain import words
  hd, server = resolve(model, head, server)
  w = words()["llama_server"]
  argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", "0", "--ctx-size", "0",
          "--flags", json.dumps(spec_flags(hd, k)), "--fit-probe", str(need + lsd.HEADROOM), "--log", str(out / "llama-server-fit.log"),
          "--deadline-s", str(TIMEOUT_S), "--pgid-file", str(out / mtp.SERVER_GROUP.format(k="fit"))]
  out.mkdir(parents=True, exist_ok=True)
  try:
    got = next(x["fit"] for x in ec.run_driver(argv, log=out / "mtp-fit.log", timeout_s=TIMEOUT_S + DRIVER_GRACE_S) if "fit" in x)
  finally:
    mtp.stop_group(out / mtp.SERVER_GROUP.format(k="fit"))
  if not got["fits"]:
    raise mtp.MtpRefused(w["fit_refused"].format(k=k, why=got["why"]))
  return None if got["context"] is None else got["context"] - lsd.HEADROOM, w["fit_bound"].format(k=k, why=got["why"])


def measure(model:str, ks:list[int], *, backend:str, out:pathlib.Path, tokens:int = mtp.TOKENS, long_tokens:int = 2048,
            long_why:str = "asked (--long-tokens)", grow:bool = False, head:str | None = None, server:str | None = None) -> dict[str, Any]:
  """collectors/mtp.sweep on llama-server. head: the draft head (default: the one heads() finds); server: the binary
  (default: $BOLTBEAM_LLAMA_SERVER, else llama-server on PATH)."""
  hd, server = resolve(model, head, server)
  prompts = mtp.prompt_set()

  def run_k(k:int) -> list[dict[str, Any]]:
    flags = [*cs.all_layers(None), "--metrics", *spec_flags(hd, k)]
    argv = [sys.executable, str(lsd.DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", "0",
            "--ctx-size", str(prompts["room"] + long_tokens + tokens + lsd.HEADROOM), "--flags", json.dumps(flags),
            "--mtp-prompts", prompts["path"], "--k", str(k), "--tokens", str(tokens), "--long-tokens", str(long_tokens),
            "--long-corpus", str(mtp.LONG), "--log", str(out / f"llama-server-k{k}.log"),
            "--deadline-s", str(TIMEOUT_S), "--pgid-file", str(out / mtp.SERVER_GROUP.format(k=k))]
    return ec.run_driver(argv, log=out / f"mtp-k{k}.log", timeout_s=TIMEOUT_S + DRIVER_GRACE_S)  # the driver's own deadline first: it stops its server
  return mtp.sweep(run_k, ks, model=model, backend=backend, out=out, tokens=tokens, long_tokens=long_tokens, long_why=long_why, grow=grow,
                   decode_speed="(predicted_n - 1) / predicted_ms, llama-server's own timings: no prefill")
