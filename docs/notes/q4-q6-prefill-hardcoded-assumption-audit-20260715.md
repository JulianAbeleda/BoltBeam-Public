# Q4/Q6 prefill hard-coded-assumption audit

Audit date: 2026-07-15. Scope: the Q4/Q6 prefill path spanning
`/home/ubuntu/tinygrad-arkey` and `/home/ubuntu/BoltBeam`. This is an inventory,
not a runtime change. Runtime dispatch and emitters are intentionally untouched.

## Executive result

The following assumptions are fully evidenced in source and tests:

- Q4_K and Q6_K are the only default direct-packed prefill quant labels
  (`tinygrad-arkey/tinygrad/llm/prefill_routes.py:81-88`).
- Q4_K uses 256-element/144-byte blocks and Q6_K uses 256-element/210-byte
  blocks; both specs require K to be a block multiple
  (`tinygrad-arkey/extra/qk/q4k_prefill_route_spec.py:13,64-65`,
  `tinygrad-arkey/extra/qk/q6k_prefill_route_spec.py:18,66-67`).
- The generated kernels expose logical row, token, K-block, and 8-lane reduce
  axes. Q4 uses `(row, bb, blk, lane4)` and Q6 uses `(row, bb, blk, lane2)`
  (`tinygrad-arkey/extra/qk/q4k_prefill_route_spec.py:98-108`,
  `tinygrad-arkey/extra/qk/q6k_prefill_route_spec.py:97-107`). Partials add a
  parts axis; direct/reduce output requires `parts==1`
  (`q4k_prefill_route_spec.py:113-131`, `q6k_prefill_route_spec.py:112-130`).
- The default concrete token microbatch is `PREFILL_UBATCH=512`, and the direct
  packed Q4 schedule hard-codes a 4x4 register/upcast tile for the 14B pp512
  case (`tinygrad-arkey/tinygrad/llm/model.py:45`,
  `tinygrad-arkey/tinygrad/llm/prefill_routes.py:91-94,147-151`). This is a
  benchmarked assumption, not a general tile contract.
- Role normalization is explicit for gate/up, down, Q/O, K/V, and lm_head
  (`tinygrad-arkey/tinygrad/llm/prefill_routes.py:96-108`). BoltBeam’s all-four
  authority requires exactly `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`
  (`/home/ubuntu/BoltBeam/boltbeam/prefill_authority.py:25-30`).
- The route spec advertises fp16 activations, Q4_K/Q6_K tensor formats, and
  packed-dequant-dot lowering (`tinygrad-arkey/tinygrad/llm/prefill_routes.py:187-194`).
  KV storage is a separate concern: the model uses int8 plus fp16 scales only
  when `config.kv_quant` is enabled (`tinygrad-arkey/tinygrad/llm/model.py:347-367`),
  otherwise the cache is fp16 for the promoted generated shape
  (`model.py:421-454`).
- The pinned authority protocol is K=8, 4 warmups, 3 rounds, pinned clock,
  whole lengths 512/1024/2048/4096, and generated-pure/no-rollback metadata
  (`/home/ubuntu/BoltBeam/docs/prefill-policy-benchmark-protocol-20260712.md:12-23`).
  The validator enforces those fields and the context sweep
  (`/home/ubuntu/BoltBeam/boltbeam/prefill_authority.py:17-30`).

## Assumption ledger

| Area | Observed hard-coded or default fact | Evidence | Status |
|---|---|---|---|
| Tile sizes | Q4 direct packed defaults to `LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4`; Q4 B upcast is capped at 16 and defaults to 4. Q6 does not have an equivalent prefill tile default in its route spec. | `tinygrad-arkey/tinygrad/llm/prefill_routes.py:91-94,147-151`; `tinygrad-arkey/extra/qk/q6k_prefill_route_spec.py:37-47` | Q4 covered; Q6 default schedule still unknown |
| Logical axes | Output is indexed `[token,row]`; reduction is over K blocks and an 8-lane axis. Partials use `[row,token,part]`. | `q4k_prefill_route_spec.py:98-131`; `q6k_prefill_route_spec.py:97-130` | Covered for generated specs |
| Context windows | Attention reads `start_pos+T`, except full ring reads `max_context`; authority uses max context 4608 with start positions through 3584 and chunk 512. | `tinygrad-arkey/tinygrad/llm/model.py:336-340`; `tinygrad-arkey/extra/qk/prefill_harness.py:20-36,47-49` | Covered for harness; Q4/Q6 GEMM itself has no context axis |
| Batch/microbatch | `PREFILL_UBATCH=512`; authority `chunk_n=512`; 14B harness forces direct-packed route. | `model.py:45`; `prefill_harness.py:35-36,73-75` | Covered for pinned authorities; arbitrary ubatch performance unknown |
| Role mapping | Name/attribute fallback maps gate/up, down, Q/O, K/V, output to five role families; authority “all-four” uses four names and excludes lm_head. | `prefill_routes.py:96-108`; `BoltBeam/boltbeam/prefill_authority.py:25-30` | Covered for current dense census; lm_head prefill authority unknown |
| Dtype/formats | Activations are declared fp16; weights are Q4_K or Q6_K packed; Q4/Q6 block validation is explicit. KV may be fp16 or int8+fp16 scale, independently of weight quant. | `prefill_routes.py:187-194`; `q4k_prefill_route_spec.py:57-74`; `q6k_prefill_route_spec.py:59-76`; `model.py:347-367,446-454` | Covered structurally; numerical/per-role dtype evidence incomplete |
| Benchmark authority | BoltBeam protocol is generated-pure 8B; tinygrad also has a 14B direct-packed authority profile. BoltBeam roofline reports only a trusted Q4 rate and marks Q6 untrusted. | `BoltBeam/docs/prefill-policy-benchmark-protocol-20260712.md:3-6,20-35`; `prefill_harness.py:70-75`; `BoltBeam/boltbeam/prefill_roofline.py:80-110,153-180` | Alignment boundary covered; cross-regime comparison is invalid |

## Fully covered versus still unknown

Fully covered: route labels and eligibility, block sizes, generated logical axis
shape, direct/partial output invariants, current role aliases, pinned authority
protocol fields, the 8B all-four role census, and the fact that Q6 is not a
trusted roofline ceiling in the current imported baseline.

Still unknown or not evidenced by this inventory: a Q6-specific prefill tile
schedule equivalent to Q4’s 4x4 assumption; measured Q6 prefill performance at
each authority context; Q4/Q6 mixed-role attribution for lm_head and any MoE
expert roles; whether arbitrary `PREFILL_UBATCH` values preserve correctness and
the same generated schedule; exact activation dtype at every non-linear boundary;
and a single apples-to-apples benchmark that compares Q4 and Q6 on the same
model, route family, context sweep, clock, and role census. The Q6 roofline
exclusion is deliberate: `prefill_roofline.py:179` records that its imported Q6
rows exceed raw HBM peak.

## Authority alignment rule

Use the pinned whole-prefill artifact for promotion decisions only when K,
warmups, rounds, clock status, route purity, rollback status, context lengths,
model profile, and role census match. Do not merge the 8B generated-pure series
with the 14B direct-packed series, and do not call the raw HBM byte floor a Q6
authority. Typed host-side fixtures also do not establish execution:
`/home/ubuntu/BoltBeam/boltbeam/prefill_authority.py:5-15`.

## Audit limits

This pass is source/test/documentation evidence only. No runtime dispatch,
emitter, route default, benchmark artifact, or generated kernel was changed.
