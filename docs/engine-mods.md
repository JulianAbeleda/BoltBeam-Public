# Engine mods

## BoltBeam vs arkey_ie_mods

- **arkey_ie_mods** holds every engine artifact: patches and kernels (a kernel's source, built binaries if kept, its
  manifest entry for the install hook mod), each with the same rules: one switch (a kernel: through the hook's
  manifest), off = stock, a check mode (a kernel: the hook's check), an output label with evidence, a line budget,
  gains against the same engine's stock build. Its `tools/check.py` enforces them; its `index.json` is the export.
- **BoltBeam** holds no engine patch and no engine kernel artifact. It is the tool: it measures, searches and applies
  mods (a clone of the user's tree, built). Producing mods in the repo's format (a writer that runs the repo's check)
  and verifying a modded build come next. It keeps BoltBeam-side facts only: `boltbeam/data/engine_patches.json` says
  where the repo is, the commit it reads (the pin) and which BoltBeam features use which mod; it ships a copy of the
  repo's index at the pin.
- **arkey** is BoltBeam's own provider and search machinery: the owner's fork of tinygrad (`tinygrad-arkey`), run by
  the `arkey` provider. Upstream tinygrad is a different engine row and is not run. A run record that says
  `tinygrad` (the provider's name before 2026-11-15) reads as `arkey`. The arkey
  capture patch (`tools/patches/arkey-capture-support.patch`) and arkey's kernel routes
  (`boltbeam/data/nv_sm120_kernel_authority.json`) are BoltBeam's own, not engine mods, and stay in BoltBeam.

## Where and which commit

The repo: `$BOLTBEAM_IE_MODS`, else `~/arkey_ie_mods` (`engine_patches.json` `mods_repo`). BoltBeam ships the repo's
index.json at `mods_repo.pin` as `boltbeam/data/ie_mods_index.json`; every reader (results, the context audit, tune,
recommend, the kernel ledger) answers from it, so the answers are the same with or without the repo. The repo itself
is read only for a patch's text when mods are applied (`git show <pin>:<patch>`). Moving the pin is a BoltBeam commit
that copies the new index; tests/test_engine_patches.py fails until the copy equals the pinned index (a named setup
skip without the repo). A switch that is both a mod's and a knob row's takes the mod's output label; the knob row's
evidence is kept beside it. A run through the kernel hook mod (`install_paths`) reads untested: the mods repo has
no kernel rows yet to label what a manifest installs.

## Applying mods to the build you run

`boltbeam screen mods --engine llama.cpp [--src DIR]` lists the mods that fit the build: base = its commit, or for
another commit each mod tried in series order in a scratch clone, applies or conflicts by name, never forced. It names
a mod left out because an apply dependency is not chosen, and the uncommitted or untracked changes in the checkout
that a clone does not carry. `--apply` takes the identical ones (`--label quality-gated|untested` or `--mod ID` for
others), applies all of them or none to a clone of the checkout at its commit (yours is never changed; a rerun resets
the clone), builds it with the engine's install recipe (workflow/engine_update.install) and records it in
`~/.boltbeam/builds.json` under `modded`, once per set of mods, as "applied, not verified". tune names such a build
"Modded build, applied, not verified" beside its setup.

Every mods build is configured with its base commit (`engine_patches.json` `apply` `commit_cmake`; llama.cpp:
`-DLLAMA_BUILD_COMMIT=<base> -DGGML_BUILD_COMMIT=<base> -DGIT_EXE=OFF`), a checkout's too: its ggml commit is the
full 40-character base and ggml's `-dirty` marker no longer appears (the applied mods would have made it dirty). Once
built, the commit it says (`commit_header`: build/ggml/src/ggml-version.h) must be the base, or it is refused and not
recorded. A source tree without .git (git archive, tarball, a copy) is accepted: it cannot say its commit, so it is
taken as the base of the newest released mods version in the pinned index (or `--base VERSION|COMMIT`), copied
(top-level build directories left out) instead of cloned, and never checked: its header agrees by construction, so
its commit line and every kernel line say "base taken, unchecked: the build was told it is <base>", its status is
"applied, not verified; base unchecked (no .git)" and its key is marked `-copy`, so it never replaces a checkout's
build. Without the flags such a tree builds ggml commit "unknown" and the ext-kernels hook refuses every manifest,
running stock kernels with only a WARN line (boltbeam-runs/decode-topk-fused-20261115 Round 2). With the hook among
the applied mods, each kernel row's manifest at the pin is checked against the built commit by the hook's own rule
(`install_paths` `commit_match`: the build's commit cut at `-`, then a common prefix of 7 or more characters with a
manifest entry): "will not load", "unknown: <reason>" for a manifest that cannot be read (the build is still
recorded), or "not confirmed" with the log lines that confirm it (`ext_load: loaded <id>` per entry, WARN;
llama-server drops ggml INFO lines) and the manifest path, said when that working-tree file is not the pinned one
checked. No load check runs here: it needs the GPU and a model. vLLM lists nothing yet: its mods use patch_p1 (into a copy of
a venv), and no applier for that is wired (`engine_patches.json` `apply` has llama.cpp only).

## Next: verifying a modded build

Not built yet; it lands with the writer that produces mods. A check that a mod is safe to turn on must reach the
mod's own code path, so: each mod's check mode run on a workload that exercises it (server mods through the server,
long-context mods at their depth), a check-count pattern in the index so a run with zero comparisons reads "not
exercised", not "passed"; explicit off values for switches that are on by default or take a value (a manifest, a
token count); run dependencies set; and stock against switches-off on the same workload. Until then a modded build is
never presented as checked.
