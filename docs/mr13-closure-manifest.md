# MR13 closure manifest

`boltbeam.artifacts.mr13_closure` produces a fresh, deterministic closure manifest
of named JSON packets. It stores schema, status, byte length, and SHA-256 only;
never trace, model, GGUF, or cache payloads. Every packet is reread during
validation, so missing, stale, tampered, duplicate, or wrong-schema/status inputs
fail closed.

Every closure requires the canonical `boltbeam.mr12_static_audit.v1` packet with
status `pass`. MR13 validates its check disposition and clean pinned BoltBeam and
tinygrad EXP commit/tree identities; a generic substitute or failed/dirty audit
cannot close the campaign.

The `mr9_result` input must use schema `boltbeam.mr9_semantic_search.v1` and
status `COMPLETE`. MR13 derives its branch from the canonical role decisions:
at least one `MACHINE_WINNER` selects the winner branch; all
`REFUTED_NO_MATERIAL_WIN` selects refutation. Blocked or unknown decisions fail
closed.

If MR9 has a winner, include `mr10_result`, `selected_plan`, and `route_census`.
If it is refuted, omit all three and provide a nonempty reason for each. This is a
protocol builder only; it makes no measurement or promotion claim.
