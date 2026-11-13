"""Runs under the vLLM venv's python (torch, safetensors, vllm), never BoltBeam's: `python vllm_engine.py CMD JSON_ARGS`
prints one JSON object. It imports nothing from boltbeam. Commands: stock, write, logprobs."""
import base64, hashlib, json, os, sys


def stock(a):
  from importlib import metadata
  d = metadata.distribution("vllm")
  direct = d.read_text("direct_url.json")
  out = {"version": d.version, "direct_url": json.loads(direct) if direct else None, "changed": None, "checked": 0}
  for f in d.files or ():
    if not f.hash or f.hash.mode != "sha256":
      continue
    want = f.hash.value
    try:
      got = base64.urlsafe_b64encode(hashlib.sha256(f.locate().read_bytes()).digest()).rstrip(b"=").decode()
    except OSError:
      got = None
    out["checked"] += 1
    if got != want:
      out["changed"] = str(f)
      break
  return out


_DT = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
       "F8_E4M3": 1, "F8_E5M2": 1}


AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)  # AWQ gemm: nibble i of a packed int32 holds column AWQ_ORDER[i] of its 8


def _pack(t, dim, order=range(8)):  # 8 nibbles per int32 along dim, first element in the lowest bits (order: which element sits in nibble j)
  import torch
  t = t.transpose(0, 1) if dim == 0 else t
  t = t.reshape(t.shape[0], -1, 8)
  out = torch.zeros(t.shape[:2], dtype=torch.int64, device=t.device)
  for j, o in enumerate(order):
    out |= t[..., o] << (4 * j)
  out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
  return out.transpose(0, 1) if dim == 0 else out


def _quant(m, W, G, top, dev, awq, sdt):
  """One 2-D weight as RTN int`bits`, asymmetric, a nibble zero point per group of G: the tensors that replace m.weight
  (AWQ gemm or compressed-tensors names and layout), and the squared error and squared norm for the relative error."""
  import torch
  O, I = W.shape
  assert I % G == 0 and I % 8 == 0 and O % 8 == 0, f"{m} shape does not divide the group"
  parts, err, norm = [], 0.0, 0.0
  for r in range(0, O, 8192):  # row chunks: the int64 codes of a whole head would not fit next to the model
    w = W[r:r + 8192].to(dev).float()
    n = w.shape[0]
    wg = w.view(n, I // G, G)
    mn, mx = wg.amin(-1), wg.amax(-1)
    scale = ((mx - mn) / top).clamp(min=1e-8)
    zp = (-mn / scale).round().clamp(0, top)  # nibble-domain zero point
    q = (wg / scale[..., None] + zp[..., None]).round().clamp(0, top).to(torch.int64).view(n, I)
    sc = scale.to(torch.bfloat16)
    deq = ((q.view(n, I // G, G) - zp[..., None]) * sc.float()[..., None]).view(n, I)
    err += ((deq - w) ** 2).sum().item()
    norm += (w ** 2).sum().item()
    if awq:
      parts.append((_pack(q.T.contiguous(), 1, AWQ_ORDER).cpu(), sc.T.contiguous().to(sdt).cpu(), _pack(zp.to(torch.int64).T.contiguous(), 1, AWQ_ORDER).cpu()))
    else:
      parts.append((_pack(q, 1).contiguous().cpu(), sc.cpu(), _pack(zp.to(torch.int64), 0).contiguous().cpu()))
  cat = lambda i, d: torch.cat([p[i] for p in parts], dim=d)  # noqa: E731
  out = ({m + ".qweight": cat(0, 1), m + ".scales": cat(1, 1), m + ".qzeros": cat(2, 1)} if awq else
         {m + ".weight_packed": cat(0, 0), m + ".weight_scale": cat(1, 0), m + ".weight_zero_point": cat(2, 0), m + ".weight_shape": torch.tensor([O, I], dtype=torch.int64)})
  return out, err, norm


def _fresh(D, name):
  """A path in the copy that nothing is at yet: a file linked from the source is the source's own, and a save there would
  write into it (a copy's copy links the first copy's model-part files)."""
  p = os.path.join(D, name)
  if os.path.lexists(p):
    raise FileExistsError(f"{name} is already in the copy, linked from the source: writing it would write into the source")
  return p


def write(a):
  """modules {name: file}: each name.weight quantized into its file; every shard that held one is re-sharded without it
  into model-part files no link in the copy holds, everything else linked; nothing written over anything already there.
  targets: patterns added to the quantization group (lm_head is not a Linear)."""
  import torch
  from safetensors import safe_open
  from safetensors.torch import save_file
  S, D, bits, G, mods = a["src"], a["dst"], int(a["bits"]), int(a["group_size"]), a["modules"]
  awq = a.get("fmt") == "awq"  # AWQ gemm: qweight [in, out/8], qzeros [in/G, out/8] (AWQ nibble order), scales [in/G, out]
  top = 2**bits - 1
  dev = a.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
  assert bits == 4, "the packer writes 8 nibbles per int32"
  idx_path = os.path.join(S, "model.safetensors.index.json")
  idx = json.load(open(idx_path)) if os.path.isfile(idx_path) else None
  holders = {m: idx["weight_map"][m + ".weight"] if idx else "model.safetensors" for m in mods}
  os.makedirs(D, exist_ok=True)
  sdt = torch.bfloat16
  if awq:  # the head's scales take the dtype the checkpoint's own scales have
    sdt = next(safe_open(os.path.join(S, fn), "pt").get_tensor(k).dtype for k, fn in idx["weight_map"].items() if k.endswith(".scales")) if idx else sdt
  written, rel = {}, {}
  for m, fn in mods.items():
    out, err, norm = _quant(m, safe_open(os.path.join(S, holders[m]), "pt", device="cpu").get_tensor(m + ".weight"), G, top, dev, awq, sdt)
    written.setdefault(fn, {}).update(out)
    rel[m] = (err / norm) ** 0.5
  # everything but the weights is linked; each shard that held a quantized weight is rewritten without it
  skip = {"config.json", "model.safetensors.index.json", *holders.values()}
  for f in os.listdir(S):
    if f not in skip and not os.path.lexists(os.path.join(D, f)):
      os.symlink(os.path.realpath(os.path.join(S, f)), os.path.join(D, f))
  wm = dict(idx["weight_map"]) if idx else {}
  names, gone = [], {m + ".weight" for m in mods}
  for holder in sorted(set(holders.values())):
    with safe_open(os.path.join(S, holder), "pt") as f:  # re-sharded in ~1 GiB parts, so no part is whole in memory
      part, size = [], 0
      def flush():
        nonlocal part, size
        if part:
          n = next(x for i in range(len(names) + 1, 1 << 20) if not os.path.lexists(os.path.join(D, x := "model-part-%05d.safetensors" % i)))
          save_file({k: f.get_tensor(k) for k in part}, _fresh(D, n), metadata={"format": "pt"})
          for k in part:
            wm[k] = n
          names.append(n)
        part, size = [], 0
      for k in f.keys():
        if k in gone:
          continue
        sl = f.get_slice(k)
        nbytes = _DT[sl.get_dtype()]  # real bytes from the header: 4 GiB parts of bf16-assumed int32 came out at 6-8 GiB
        for d in sl.get_shape():
          nbytes *= d
        part.append(k)
        size += nbytes
        if size >= 1 << 30:
          flush()
      flush()
  for fn, tensors in written.items():
    save_file(tensors, _fresh(D, fn), metadata={"format": "pt"})
    wm.update({k: fn for k in tensors})
  for k in gone:
    wm.pop(k, None)
  idx = {**(idx or {"metadata": {}}), "weight_map": wm}
  json.dump(idx, open(_fresh(D, "model.safetensors.index.json"), "w"))
  c = json.load(open(os.path.join(S, "config.json")))
  q_cfg = c["quantization_config"]
  if awq:  # stock vLLM's AWQ config reads this key and quantizes ParallelLMHead (the AWQ row writes only the output layer)
    q_cfg["lm_head"] = True
  else:
    q_cfg["ignore"] = [x for x in q_cfg["ignore"] if x not in mods]
    g = q_cfg["config_groups"]["group_0"]
    g["targets"] = list(g["targets"]) + [t for t in a.get("targets", []) if t not in g["targets"]]
  json.dump(c, open(_fresh(D, "config.json"), "w"), indent=2)
  return {"rel_error": rel, "group_size": G, "bits": bits, "files": sorted(written), "resharded": names}


def _llm(a):
  os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
  from vllm import LLM
  return LLM(model=a["model"], max_model_len=int(a.get("max_len", 4096)), enable_prefix_caching=False, enforce_eager=True,
            gpu_memory_utilization=float(a.get("gpu_memory", 0.85)), limit_mm_per_prompt={"image": 0, "video": 0})


def generate(a):
  from vllm import SamplingParams
  outs = _llm(a).generate(a["texts"], SamplingParams(max_tokens=int(a["tokens"]), temperature=0))
  return {"texts": [o.outputs[0].text for o in outs]}


def logprobs(a):
  from vllm import SamplingParams
  llm = _llm(a)
  tok = llm.get_tokenizer()
  prompts = [{"prompt_token_ids": tok.encode(t)} for t in a["texts"]]
  outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
  rows, lengths = [], []
  for o in outs:
    lengths.append(len(o.prompt_token_ids) - 1)
    for tid, lp in zip(o.prompt_token_ids[1:], o.prompt_logprobs[1:]):
      best = max(lp.items(), key=lambda kv: kv[1].logprob)
      rows.append([tid, best[0], lp[tid].logprob])
  return {"version": __import__("vllm").__version__, "tokens": rows, "lengths": lengths}


if __name__ == "__main__":
  print("@@" + json.dumps(globals()[sys.argv[1]](json.loads(sys.argv[2]))))
