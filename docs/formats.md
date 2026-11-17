# Formats: one registry

Every weight, KV-cache and activation format BoltBeam knows is one row in
`boltbeam/data/formats.json`, read through `boltbeam/formats.py`. Every component
reads it: the GGUF and safetensors readers, the decode and prefill rooflines,
attribution, the CPU reference, the timer, the search, the emitters and the writers.
A fact said once there applies everywhere; nothing is fixed "for decode" and
missed in search.

## Layout

What is quantized, then family, then format rows:

- `weights`: ggml types (`ggml_legacy`, `ggml_k`, `ggml_iq`, `ggml_ternary`,
  `microscaling_fp4`), floats and ints (safetensors / torch dtypes), and the
  vLLM families (`compressed_tensors`, `awq`, `gptq`, `quark`, `modelopt`,
  `bitsandbytes`, `fp8`, and `auto_round`, `fbgemm`, `torchao`, whose tensors are read at their dtypes).
  A row's `quant_method` lists the config.json values it covers (`awq_marlin` is AWQ's layout, `gptq_marlin` GPTQ's).
- `kv_cache`: llama.cpp `-ctk` types and vLLM `--kv-cache-dtype` words.
- `activations`: the formats engines run activations in.

A row holds:

- its names: `name`, `aliases`, `ggml` id, `safetensors` dtype, `torch` name, `onnx` id;
- `engines`, the engines that run it;
- the layout: `block` [values, bytes] (bytes per element when values is 1);
- `bits` per weight, scales and zero points included;
- `dequant`, BoltBeam's own reference function, if it has one;
- `oracle`, llama.cpp's gguf-py implementation at the pinned engine source, used from there and never copied;
- `capability`, the search facts;
- `status`: one entry per component. The components are `profile_bytes`,
  `roofline`, `attribution`, `reference`, `timer`, `search`, `emitters` and
  `writers`. Each entry is `supported`, `n/a`, `unaudited`, or
  `{"refuses": "<words>"}`.

A `kv_cache` or `activations` row names its layout with `same_as`, a weights row,
instead of repeating it.

## Unknown formats refuse

When a reader meets a format with no row, it refuses with:

`unknown format <X> in <where>: add a row to data/formats.json`

This covers an unknown ggml type id in a GGUF, an unknown safetensors, numpy or
ONNX dtype, and an unknown `quant_method` in config.json. Bytes are never guessed,
the tensor is never skipped, and the format is never counted as a different one.
A component that knows the format but does not support it refuses with the words
in that row's status.

## Adding a format

1. Add one row under its family, with its names, engines, block layout and bits.
2. Set each component's status. `tests/test_formats.py` fails until every row
   names every component. `unaudited` is allowed for now, while the audit of the
   vLLM, AWQ and GPTQ families lands. A refusal must carry words.
3. Add the component work where a status is `supported`: a dequant function, a
   search capability, and so on.

## No other lists

No other module may list formats. `sz.py`'s `formats` authority fails a commit or
merge, through `.githooks`, when any literal outside `boltbeam/formats.py` names two
or more registry formats. That covers a tuple in a condition, a call argument, or
a dict keyed by them. It also fails on a dict keyed by quant type ids, and on a
table pairing quants with their block bytes. Read the registry instead:
`formats.row`, `canonical`, `field`, `capable`, `kv_words` or `kv_offer`.

The one exception is the PTX and HIP matrix-instruction operand types (`f16`,
`bf16`, `f32` beside `s8`, `tf32` and `iu8`) in `collectors/cuda_path_scan.py`,
`cuda_compute.py` and `hip_path_scan.py`. These are the register types of mma and
wmma instructions, not storage formats, and each one is listed in `sz.ALLOWED`
with that reason.
