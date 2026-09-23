# Qwen3.5 Hybrid Support Result

Date: 2026-07-01

## Summary

BoltBeam now profiles Qwen3.5-style hybrid SSM models from GGUF facts instead of model-name hardcodes.
The shipped code recognizes:

- `hybrid_decoder` from `<arch>.ssm.*` metadata or SSM/DeltaNet tensor names.
- `hybrid_moe_decoder` when SSM signals and MoE expert/router signals are both present.
- SSM roles as first-class route surfaces: `ssm_projection`, `ssm_conv`, `ssm_state`, `ssm_scan`.

Dense attention/FFN roles on a hybrid profile inherit dense decoder route candidates. Hybrid-MoE profiles
also inherit MoE expert/router candidates. SSM tensors do **not** flow through the quant-GEMV registry; they
emit SSM-specific route families.

## Sources

- Qwen model card: <https://huggingface.co/Qwen/Qwen3.5-27B>
- GGUF artifact used locally: <https://huggingface.co/unsloth/Qwen3.5-27B-GGUF>
- Downloaded file: `/home/ubuntu/models/Qwen3.5-27B-Q4_K_M.gguf`

## Real GGUF Profile

Command:

```bash
PYTHONPATH=. python3 -m boltbeam.cli inspect \
  /home/ubuntu/models/Qwen3.5-27B-Q4_K_M.gguf \
  --target amd_gfx1100 \
  --out outputs/qwen3.5-27b-hybrid/model_profile.json
```

Result:

- `general.architecture`: `qwen35`
- `architecture_class`: `hybrid_decoder`
- `complete`: `true`
- `hidden_size`: 5120
- `ffn_size`: 17408
- `layer_count`: 64
- `vocab_size`: 248320
- SSM metadata present:
  - `qwen35.ssm.conv_kernel = 4`
  - `qwen35.ssm.group_count = 16`
  - `qwen35.ssm.inner_size = 6144`
  - `qwen35.ssm.state_size = 128`
  - `qwen35.ssm.time_step_rank = 48`

No MoE expert/router signals were present in this GGUF:

- `expert_count`: absent
- `has_expert_stacks`: false
- `has_router`: false
- `has_3d_tensors`: false

So this local artifact is classified as `hybrid_decoder`, not `hybrid_moe_decoder`.

## Role Census

Key grouped roles from the real GGUF:

| role | representative | shape | quant | count |
|---|---|---:|---|---:|
| `attn_kv` | `blk.3.attn_k.weight` | 1024 x 5120 | Q4_K | 22 |
| `attn_kv` | `blk.3.attn_v.weight` | 1024 x 5120 | Q6_K | 10 |
| `attn_qo` | `blk.3.attn_output.weight` | 5120 x 6144 | Q4_K | 16 |
| `attn_qo` | `blk.3.attn_q.weight` | 12288 x 5120 | Q4_K | 16 |
| `ffn_down` | `blk.8.ffn_down.weight` | 5120 x 17408 | Q4_K | 32 |
| `ffn_down` | `blk.0.ffn_down.weight` | 5120 x 17408 | Q6_K | 32 |
| `ffn_gate_up` | `blk.0.ffn_gate.weight` | 17408 x 5120 | Q4_K | 128 |
| `lm_head` | `output.weight` | 248320 x 5120 | Q6_K | 1 |
| `ssm_conv` | `blk.0.ssm_conv1d.weight` | 10240 x 4 | F32 | 48 |
| `ssm_projection` | `blk.0.ssm_alpha.weight` | 48 x 5120 | Q8_0 | 96 |
| `ssm_projection` | `blk.0.ssm_out.weight` | 5120 x 6144 | Q5_K | 48 |
| `ssm_state` | `blk.0.ssm_a` | 48 x 1 | F32 | 96 |

`ssm_norm.weight` remains `norm`, not an SSM route candidate.

## Search Space

Command:

```bash
PYTHONPATH=. python3 -m boltbeam.cli emit-search \
  /home/ubuntu/models/Qwen3.5-27B-Q4_K_M.gguf \
  --target amd_gfx1100 \
  --out outputs/qwen3.5-27b-hybrid/search_space.json
```

Real search-space families emitted:

- `lanemap_gemv`: 5 role rows
- `q6k_route`: 3 role rows
- `ssm_projection`: 2 role rows
- `ssm_state_update`: 1 role row
- `ssm_conv`: 1 role row

Reachable SSM candidates:

- `decode_ssm_projection_route`
- `decode_ssm_conv_route`
- `decode_ssm_state_update_route`

No `decode_ssm_scan_route` is reachable for this GGUF because no tensor role currently classifies as
`ssm_scan`; the state path is represented by `ssm_state` tensors.

## Next Tinygrad Work

The next implementation phase should not add a Qwen3.5 hardcoded route. It should consume BoltBeam's profile
and build generic hybrid SSM support:

1. Add tinygrad role attribution for Qwen3.5 decode using the real GGUF profile.
2. Measure the wall-share of `ssm_projection`, `ssm_conv`, and `ssm_state` against the existing 2.3 tok/s
   baseline.
3. Select the hottest SSM candidate via BoltBeam reachability/evidence.
4. Implement only the selected generic route family, with fallback flags and rollback.
5. Promote only after token correctness, route binding, memory fit, deterministic repeats, and speed tier
   gates pass.

The dense/Q4/Q6 parts should reuse the existing generic candidates (`decode_q4k_g3_anyshape`,
`decode_q4k_g3_anyshape_attn_k`, `decode_q6k_coop_shipped`, `decode_q6k_ffn_down_longk`) rather than adding
Qwen3.5-specific copies.

## Verification

```bash
python3 -m unittest discover -s tests
```

Result: 197 tests pass.

