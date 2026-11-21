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
- **Kernels** come with `boltbeam mods`: the kernels module's kernel list, one build per chip architecture. Chips with
  builds today: sm_120 (RTX 5090) for every searched kernel; gfx1201 (R9700) none, so there the engine's own kernels run
  (the kernel hook is not ported to HIP yet: it loads nothing there and its log says nothing of it).

## Where and which commit

The repo: `$BOLTBEAM_IE_MODS`, else `~/arkey_ie_mods` (`engine_patches.json` `mods_repo`). BoltBeam ships the repo's
index.json at `mods_repo.pin` as `boltbeam/data/ie_mods_index.json`; every reader (results, the context audit, tune,
recommend, the kernel ledger) answers from it, so the answers are the same with or without the repo. The repo itself
is read only for a patch's text when mods are applied (`git show <commit>:<patch>`). Moving the pin is a BoltBeam
commit that copies the new index; tests/test_engine_patches.py fails until the copy equals the pinned index (a named
setup skip without the repo).

The pin is the floor and the offline default, not the only commit `boltbeam mods` reads (`workflow/mods.resolve`).
It fetches BoltBeam's own clone, then walks origin's default branch from its newest commit back to the pin and reads
the first commit whose index schema this BoltBeam reads (`mods.SCHEMAS`) and whose `requires.boltbeam` (arkey_ie_mods
engines.json, checked by its tools/check.py) is this version or older. Offline, a branch without the pin, or nothing
newer it can read: the pin, and it says why. A configured clone (`$BOLTBEAM_IE_MODS` or the default path) is never
fetched into: its `origin/HEAD` as last fetched is walked. `--pin COMMIT` or `$BOLTBEAM_IE_MODS_PIN` reads exactly
that commit (refused if missing or unreadable). The commit is said first (`mods: arkey_ie_mods <commit> (newest
compatible)`), kept in the plan (`mods_commit`), in the build record (`mods_commit`, `mods_resolved`) and in a run on
the build (`workload_profile.json` `modded_build`), so `--pin` with it rebuilds the same mods. The build key hashes
the chosen mods' patch and kernel sha256, so a commit that changes a patch builds anew. Only mods of a version the
index marks verified (a tested tree) are applied; the rest are listed with why (`--mod` naming one is refused).
What this trusts: by default the patches come from the tip of the repo's default branch, not a commit a BoltBeam
release reviewed; the sha256 checks catch a file that differs from its index at the same commit, not a change made in
the repo. `--pin` (the pin's full id is in `engine_patches.json`) reads only a reviewed commit. A switch that is both a mod's and a knob row's takes the mod's output label; the knob row's
evidence is kept beside it. A run through the kernel hook mod (`install_paths`) reads untested: the mods repo has
no kernel rows yet to label what a manifest installs.

## Applying mods to the build you run

`boltbeam mods [--engine llama.cpp] [--src DIR]` lists the mods that fit the build: base = its commit, or for
another commit each mod tried in series order in a scratch clone, applies or conflicts by name, never forced. It names
a mod left out because an apply dependency is not chosen, and the uncommitted or untracked changes in the checkout
that a clone does not carry. By default it takes the series for the base: its identical and quality-gated mods, the
set the published benchmark measured (llama.cpp 1.0.0: all 13, 04 mtp-share-head quality-gated; never a parked one).
The list marks the quality-gated ones and names their evidence (`engine_patches.json` `benchmarks`, keyed
engine/version: arkey_ie_mods benchmarks/llama.cpp-1.0.0-0.3.26). `--label identical|quality-gated|untested` takes
only the series mods with those labels, `--mod ID` exactly those mods. `boltbeam mods apply` (the 0.3.26 `--apply`
still works) applies all of them or none to a clone of the checkout at its commit (yours is never changed; a rerun
resets the clone), builds it with the engine's install recipe (workflow/engine_update.install) and records it in
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
