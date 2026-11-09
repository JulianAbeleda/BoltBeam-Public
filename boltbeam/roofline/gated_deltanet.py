"""Gated DeltaNet work in one speculative verify step, from the engine's loaded state shapes.

Without speculation a GDN layer keeps one state per stream and a decode token reads and writes it once
(prefill_work.recurrent). With speculation vLLM verifies T = k + 1 query tokens per step and must be able to roll back
to whichever of them the target accepts, so it keeps a state per draft position (the engine's state indices are
[requests, k + 1]): the step reads the state at the last accepted position and writes the state after every one of
its T tokens. The convolution's state likewise holds conv_kernel - 1 + k columns. That is the partition this module
models; its source is vLLM's own reference for the step, fused_sigmoid_gating_delta_rule_update(inplace_final_state=
True, num_accepted_tokens=...) (third_party/flash_linear_attention/ops/fused_sigmoid_gating.py: one initial state load
at num_accepted - 1, one store per token), which the CUDA kernel it runs (fused_gdn_decode_post_conv_mtp) replaces with
the same arguments, and causal_conv1d_update with num_accepted_tokens.

The bytes are compulsory tensor traffic, the limit's basis for every recurrent row (prefill_work.recurrent): a
limit needs the bytes the step must move, not what a kernel moves. The CUDA kernel's code is not in vLLM's wheel, so
the partition is held to the engine's own structure instead: its state indices hold k + 1 slots a request, and the
next step reads the one at its accepted count. Measured DRAM traffic cannot be required here: under ncu the engine
does not repeat a capture's speculative steps (it serializes kernels, and a bf16 near tie moved an acceptance at token
16 of the 27B, 2026-11-08), so no counter run is the captured program. Shapes come from the loaded module
(get_state_shape and its dtypes), checked against the step: a conv state that does not hold k more columns than the
kernel's window is not the speculative partition and is refused, never guessed.
"""
from __future__ import annotations

from typing import Any

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


def speculative_stages(profile:dict[str, Any], evidence:dict[str, Any], step:dict[str, Any]) -> list[dict[str, Any]]:
  """One GDN layer in one verify step of T = step['query_tokens'] tokens: [{stage, flops, bytes, ...}] for the
  convolution and the state update. evidence: the layer's module facts (state_shapes, state_dtypes, inputs)."""
  r = (profile.get("metadata") or {}).get("recurrent") or {}
  width = r.get("conv_kernel")
  shapes, dtypes = evidence.get("state_shapes") or [], evidence.get("state_dtypes") or []
  inputs = evidence.get("inputs") or []
  if type(width) is not int or width < 2 or len(shapes) != 2 or len(dtypes) != 2 or not inputs:
    raise ValueError("speculative GDN work needs the conv kernel width and the loaded conv and recurrent state shapes")
  if any(d not in DTYPE_BYTES for d in dtypes) or inputs[0].get("dtype") not in DTYPE_BYTES:
    raise ValueError(f"speculative GDN formats {dtypes}, {inputs[0].get('dtype')} are not modeled")
  t, k = step["query_tokens"], step["draft_tokens"]
  conv, state = [list(s) for s in shapes]
  if len(conv) != 2 or len(state) != 3:
    raise ValueError(f"GDN state shapes {shapes} are not [window, channels] and [heads, value, key]")
  window, channels = (conv if conv[0] < conv[1] else conv[::-1])
  if window != width - 1 + k:
    raise ValueError(f"the conv state holds {window} columns, not the kernel's {width - 1} plus {k} draft positions: "
                     "not vLLM's speculative partition, so its work is not modeled")
  heads, vdim, kdim = state
  if inputs[0].get("shape") != [t, channels + heads * vdim]:  # the layer's input is q, k, v and the output gate z
    raise ValueError(f"GDN input {inputs[0].get('shape')} is not the step's {t} tokens of {channels} channels and "
                     f"a {heads} x {vdim} gate")
  qk = channels - heads * vdim
  if qk <= 0 or qk % 2:
    raise ValueError(f"GDN channels {channels} do not split into q, k and {heads} x {vdim} v")
  a, cs, ss = DTYPE_BYTES[inputs[0]["dtype"]], DTYPE_BYTES[dtypes[0]], DTYPE_BYTES[dtypes[1]]
  matrix = heads * vdim * kdim * ss
  basis = (f"per layer, one verify step of {t} tokens: state read once at the last accepted position and written "
           f"after each of the {t} (a state per draft position); compulsory bytes")
  row = lambda stage, flops, nbytes: {"stage": stage, "flops": float(flops), "bytes": float(nbytes), "path": "fp32_fma",  # noqa: E731
                                      "layers": r.get("layer_count"), "basis": basis,
                                      "tokens": t}
  return [
    # the window before the step read, the T new columns written; T inputs in, T outputs out; the weights once
    row("convolution", 2 * width * channels * t,
        (width - 1) * channels * cs + t * channels * cs + 2 * t * channels * a + width * channels * a),
    # per token per head the delta rule's 7 x V x K (prefill_work.recurrent); q, k, v, a, b, the gate in and the
    # normed output out per token
    row("state_update", 7 * heads * vdim * kdim * t,
        matrix + t * matrix + t * (qk + 2 * heads * vdim + 2 * heads + heads * vdim) * a),
  ]
