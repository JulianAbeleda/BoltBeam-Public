# The bench: reproduce a row on your own GPU

BoltBeam's cross-GPU table (decode, batch 1) is meant to be rerun by anyone. Each row is one model on one engine on
one GPU, measured against that GPU's own limit: the bytes a token must read over the memory bandwidth BoltBeam
measures on the machine. This page fixes everything else, so your row is comparable with ours.

## What is fixed

| | Pin |
|---|---|
| llama.cpp | b11571 (`097cf33d9`), BoltBeam's engine pin (`boltbeam/data/engine_pins.json`). On Linux, `tools/pod-setup.sh` fetches the prebuilt archive (CUDA 12.8, sm 80–120, driver 525+, release `llama-097cf33d9` on the public mirror, sha256 `2b5e75859f32f4f88e09139038e75d8e7196a160039ea5657653ec7ca7547f8e`) or compiles that commit |
| vLLM | 0.31.0 (the pin), in its own venv |
| Workload | decode, batch 1, contexts 128 and 512, in-model measurement (nsys on NVIDIA) |

| Model | Engine | File | Size | sha256 / revision |
|---|---|---|---|---|
| Qwen3-14B Q4_K_M | llama.cpp | `unsloth/Qwen3-14B-GGUF` · `Qwen3-14B-Q4_K_M.gguf` | 9.0 GB | `5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66` |
| Qwen3-14B AWQ | vLLM | `Qwen/Qwen3-14B-AWQ` | 10 GB | revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3` |
| Qwen3.8-27B UD-Q4_K_M | llama.cpp | `unsloth/Qwen3.8-27B-GGUF` · `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16.5 GB | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.8-27B W4A16 AWQ | vLLM | `philbert440/Qwen3.8-27B-W4A16-AWQ` | 19.5 GB | revision `7908d42a71077a5e4dc458f273682b12dfe384a0` |

`hf download REPO [FILE] --local-dir DIR` fetches them in several streams; `tools/pod-setup.sh` downloads its
`MODEL_URL` in 8 byte ranges at once (one curl stream was 13 MB/s on a rented host, eight were 68 MB/s).
A model that does not fit a GPU is recorded as not fitting, not forced (the 27B AWQ fits 24 GB: an RTX 3090 ran it).

## Run it

On a rented Linux GPU, `tools/pod-setup.sh` does everything up to the first measured run ([pods.md](pods.md)):
packages, BoltBeam, both engines, the chip profile (`boltbeam` measures the GPU's bandwidth and keeps it as a chip
profile), the model, a run. On your own machine, `boltbeam` opens the screen: pick the model, the engine BoltBeam found
for its format, Measurement in-model, and Run. The CLI form of one row:

```
python -m boltbeam.workflow.screen pipeline MODEL --run RUN --target TARGET --id ID \
  --provider llama.cpp|vllm --analyze --measure auto --role-time in-model --no-search
python -m boltbeam.workflow.screen results --run RUN > RUN/results.json
```

`TARGET` is the chip profile id `boltbeam` made (`~/.boltbeam/chips/`), or a registry row such as `nvidia_sm120`.

## Check it

```
tools/check_run_math.py RUN
```

recomputes the run's headline from its own files: the speed is the engine's timing, the share is speed over limit,
the limit is no more than the measured bandwidth over the weight bytes, the tie-out and the table add up, and the
capture overhead is said. A run is quotable when it prints `all figures check`. Every run keeps the chip profile it was
measured against (`RUN/chip_profile.json`), so a run folder you send opens on any machine with `boltbeam`.

## What to quote

- The speed, the limit and the share: always (measured untraced; the limit is derived from measured bandwidth).
- The per-kernel table ("where the token goes"): when the capture is within 5% of the untraced token, or when the
  run's repeats replicate its kernel rows, for any engine: fresh captures, or for vLLM groups of one capture's decode
  steps (the note above the table says which, and that the gaps are then a bound). A host with a slow CPU per thread (rented servers) adds per-launch recording cost; a desktop CPU did not
  (RTX 5090 on a Ryzen 7 5800X: +2.7%).
- When the table cannot be built (a closed kernel nothing could read, or a matrix instruction the chip profile holds
  no measured rate for, such as Hopper's warpgroup GEMMs against a profile made before 0.3.15), the report keeps the
  speed and limit and says why there are no roles.

The 2026-11-09 sweep's runs, chip profiles and TUI screens are in `docs/evidence/gpu-sweep-20261109/`.
