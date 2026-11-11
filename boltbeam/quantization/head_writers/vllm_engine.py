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


def write(a):
  import torch
  from safetensors import safe_open
  from safetensors.torch import save_file
  S, D, bits, G = a["src"], a["dst"], int(a["bits"]), int(a["group_size"])
  awq = a.get("fmt") == "awq"  # AWQ gemm: qweight [in, out/8], qzeros [in/G, out/8] (AWQ nibble order), scales [in/G, out]
  top = 2**bits - 1
  dev = a.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
  assert bits == 4, "the packer writes 8 nibbles per int32"
  idx_path = os.path.join(S, "model.safetensors.index.json")
  idx = json.load(open(idx_path)) if os.path.isfile(idx_path) else None
  files = sorted({v for v in idx["weight_map"].values()}) if idx else ["model.safetensors"]
  holder = idx["weight_map"]["lm_head.weight"] if idx else "model.safetensors"
  os.makedirs(D, exist_ok=True)
  W = safe_open(os.path.join(S, holder), "pt", device="cpu").get_tensor("lm_head.weight")
  O, I = W.shape
  assert I % G == 0 and I % 8 == 0 and O % 8 == 0, "lm_head shape does not divide the group"
  parts, err, norm = [], 0.0, 0.0
  sdt = torch.bfloat16
  if awq:  # the head's scales take the dtype the checkpoint's own scales have
    sdt = next(safe_open(os.path.join(S, fn), "pt").get_tensor(k).dtype for k, fn in idx["weight_map"].items() if k.endswith(".scales")) if idx else sdt
  for r in range(0, O, 8192):  # row chunks: the int64 codes of the whole head would not fit next to the model
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
  rel = (err / norm) ** 0.5
  head = {"lm_head.qweight": torch.cat([p[0] for p in parts], dim=1), "lm_head.scales": torch.cat([p[1] for p in parts], dim=1),
          "lm_head.qzeros": torch.cat([p[2] for p in parts], dim=1)} if awq else {"lm_head.weight_packed": torch.cat([p[0] for p in parts]), "lm_head.weight_scale": torch.cat([p[1] for p in parts]),
          "lm_head.weight_zero_point": torch.cat([p[2] for p in parts], dim=0),
          "lm_head.weight_shape": torch.tensor([O, I], dtype=torch.int64)}
  # everything but the weights is linked; the shard that held the bf16 head is rewritten without it
  skip = {"config.json", "model.safetensors.index.json", holder}
  for f in os.listdir(S):
    if f not in skip and not os.path.lexists(os.path.join(D, f)):
      os.symlink(os.path.realpath(os.path.join(S, f)), os.path.join(D, f))
  wm = dict(idx["weight_map"]) if idx else {}
  names = []
  with safe_open(os.path.join(S, holder), "pt") as f:  # the holder is re-sharded in ~1 GiB parts, so no part is whole in memory
    part, size = [], 0
    def flush():
      nonlocal part, size
      if part:
        n = "model-part-%05d.safetensors" % (len(names) + 1)
        save_file({k: f.get_tensor(k) for k in part}, os.path.join(D, n), metadata={"format": "pt"})
        for k in part:
          wm[k] = n
        names.append((n, part))
      part, size = [], 0
    for k in f.keys():
      if k == "lm_head.weight":
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
  head_file = "model-lmhead%d.safetensors" % bits
  save_file(head, os.path.join(D, head_file), metadata={"format": "pt"})
  wm.pop("lm_head.weight", None)
  for n in (("qweight", "qzeros", "scales") if awq else ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")):
    wm["lm_head." + n] = head_file
  if idx:
    idx["weight_map"] = wm
  else:
    idx = {"metadata": {}, "weight_map": wm}
  json.dump(idx, open(os.path.join(D, "model.safetensors.index.json"), "w"))
  c = json.load(open(os.path.join(S, "config.json")))
  q_cfg = c["quantization_config"]
  if awq:  # stock vLLM's AWQ config reads this key and quantizes ParallelLMHead
    q_cfg["lm_head"] = True
    json.dump(c, open(os.path.join(D, "config.json"), "w"), indent=2)
    return {"rel_error": rel, "shape": [O, I], "group_size": G, "bits": bits, "head_file": head_file, "resharded": [n for n, _ in names]}
  if "lm_head" in q_cfg["ignore"]:
    q_cfg["ignore"].remove("lm_head")
  g = q_cfg["config_groups"]["group_0"]
  if "re:.*lm_head$" not in g["targets"]:
    g["targets"] = list(g["targets"]) + ["re:.*lm_head$"]
  json.dump(c, open(os.path.join(D, "config.json"), "w"), indent=2)
  return {"rel_error": rel, "shape": [O, I], "group_size": G, "bits": bits, "head_file": head_file, "resharded": [n for n, _ in names]}


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
