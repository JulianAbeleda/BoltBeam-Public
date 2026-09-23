# BoltBeam Commit Discipline

Every commit must carry one owning-subsystem prefix.

## Allowed Prefixes

| prefix | owns |
|---|---|
| `[profile]` | artifact readers, `ProfileIR`, role classification |
| `[search]` | candidate-space emission and topology/search families |
| `[policy]` | route policy documents and promotion policy |
| `[eval]` | evaluation and promotion contracts |
| `[synth]` | synthetic fixture generation |
| `[cli]` | command-line wiring and handoff bundles |
| `[schema]` | JSON schema contracts |
| `[docs]` | documentation-only changes |
| `[test]` | tests and test-only fixtures |
| `[repo]` | repo plumbing, hooks, packaging, CI/config |
| `[trace]` | trace/timing artifact plumbing |
| `[target]` | target capability and hardware-root resolution |
| `[bench]` | benchmark harnesses and fixtures |
| `[path]` | path/asset resolution |
| `[collectors]` | external-tool collector adapters |

Use the subsystem that owns the behavior. Documentation-only changes use
`[docs]` unless the document is the contract for another subsystem.

## NFC

Behavior-preserving changes must be marked:

```text
[search] NFC - extract route-family helper
```

Do not use `NFC` if any output bytes change. Do not mix NFC refactors with
functional changes.

## Examples

Valid:

```text
[profile] add GGUF tensor role for attn_k
[cli] add analyze handoff workflow
[repo] NFC - install commit hook checker
[docs] document search-space-incomplete policy
```

Invalid:

```text
Add analyze workflow
[misc] update stuff
[repo] NFC - change analysis output
[profile] add reader and update CLI docs
```

## Hook

The machine-enforced version is `.githooks/commit-msg`.

Install it in a checkout with:

```bash
python3 tools/install_hooks.py
```

