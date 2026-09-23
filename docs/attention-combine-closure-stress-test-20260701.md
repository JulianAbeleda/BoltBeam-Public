# Attention Combine Closure Stress Test

Date: 2026-07-01

## Verdict

The broad wording "attention combine is closed" was too strong. The defensible claim is narrower:

> The currently reachable attention-combine route family is refuted for Qwen 14B/32B decode on gfx1100 because every tested way to remove or shrink the combine gives up the split parallelism that makes flash decode fast.

This does not prove that no future attention route can win. It proves that a future route must preserve the `Hq*S` partial-workgroup parallelism or introduce a new global/cooperative coordination primitive that can merge split results without collapsing the launch to `Hq` workgroups.

## Evidence

| Path | Mechanism result | W==D result | Source |
| --- | --- | --- | --- |
| `FLASH_L` knob | `attention_combine` shrank from 13.57% to 12.01% | ctx512 regressed 50.2 -> 43.5 tok/s | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attention-combine-flash-l-search-result.md` |
| 14B wholecache / score-broadcast route | token-correct | ctx512 regressed 50.2 -> 45.3 tok/s | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attention-combine-inkernel-result.md` |
| flash at ctx128 | earlier route selection | slower than shipped threshold | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attention-combine-inkernel-result.md` |
| fused-flash in-kernel LSE combine | bucket removed: 13.57% -> 1.71%; total reduce 19.0% -> 2.5%; token-identical | ctx512 regressed 50.2 -> 6.2 tok/s | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attention-fused-combine-result.md` |

The fused-combine result is the key stress test. It separates two questions:

1. Can generated code remove the combine? Yes.
2. Does removing it improve wall speed? No, not with the tested topology.

The failure mode is not math correctness or lack of primitives. It is topology. The shipped flash path runs partials as `Hq*S` workgroups. The fused path places the `S` splits as waves inside one workgroup per head, reducing launch parallelism to `Hq=40` workgroups for 14B. On a 96-CU GPU, that occupancy loss dominates the saved combine.

## What Would Falsify This Closure

Reopen `decode_attention_combine_reduce_fusion` only if a new candidate satisfies at least one of these conditions before W==D:

1. **Split-preserving combine:** keeps roughly `Hq*S` partial workgroups and eliminates at least one external combine kernel through a correct global or cooperative LSE merge.
2. **New target geometry:** a model/target has enough heads or different workgroup economics that `Hq` workgroups still saturate the GPU; this must be shown by occupancy or W==D evidence, not assumed.
3. **New backend primitive:** hardware/runtime support appears for global barriers, cooperative groups, or safe fast atomics that make cross-workgroup LSE combine cheaper than separate kernels.
4. **Measured attention dominance changes:** attention-combine becomes the dominant wall bucket after other wins, and a new candidate preserves protected-context speed.

Do not reopen for another Hq-only in-workgroup fused-combine attempt. That is exactly the route that removed the bucket and still lost 88%.

## Actionable Classification

BoltBeam should treat this as:

- `refuted_axis_tag`: `attention_combine_fusion_occupancy`
- `do_not_retry`: Hq-only fused in-workgroup combine
- `reopen_condition`: split-preserving/global-coordination attention LSE combine or a measured target geometry where Hq-only occupancy is sufficient

That keeps the lesson useful without overclaiming that attention can never produce another win.
