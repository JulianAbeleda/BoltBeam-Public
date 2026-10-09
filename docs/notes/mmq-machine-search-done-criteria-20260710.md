# MMQ Machine Search Done Criteria

Purpose: define the full evidence bar for saying the 14B MMQ machine-search
path is ready, and define the contract BoltBeam must transfer to tinygrad.

This is not a claim that we know everything about the GPU. It is the line where
search becomes constrained, falsifiable, and useful.

## Reference Grounding

The done criteria follow standard GPU-kernel optimization practice:

- Verify correctness before optimization.
- Profile and compare against a real baseline.
- Optimize memory access, coalescing, tiling, and shared/LDS reuse.
- Manage register/shared-memory resources and occupancy.
- Use barriers/waits only where producer/consumer ordering requires them.
- Place results against a Roofline-style performance model.
- For machine search, define a legal search space, cost/evidence model, and
  promotion gate before exploring variants.

Reference anchors:

| Source | Relevant practice |
|---|---|
| NVIDIA CUDA C++ Best Practices Guide | correctness, profiling workflow, memory coalescing, shared memory, occupancy-aware optimization |
| AMD HIP performance guidelines | profile first, maximize coalescing, use LDS, balance registers/shared memory/occupancy, minimize divergence |
| AMD HIP hardware/performance docs | memory hierarchy, coalescing, Wave32/Wave64, matrix cores, resource limits |
| Roofline model / NERSC Roofline docs | operational intensity, compute ceiling, bandwidth ceiling, bottleneck attribution |
| Ansor / TVM auto-scheduler | legal tensor search space, cost-model-guided exploration, generated schedule candidates |

Links:

```text
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
https://rocmdocs.amd.com/projects/HIP/en/develop/understand/performance_optimization.html
https://docs.nersc.gov/tools/performance/roofline/
https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
https://tvm.apache.org/2021/03/03/intro-auto-scheduler
https://arxiv.org/pdf/2006.06762
```

## Definitions

| Term | Meaning |
|---|---|
| Probe | A research artifact that answers one question but is not eligible for promotion. |
| Candidate | A generated or hand-authored kernel variant with a declared workload and search knobs. |
| Structural proof | Evidence that emitted instructions preserve ownership, accumulator identity, epochs, and stores. |
| Numeric proof | Bounded output equality against the Q4_K x Q8_1/DS4 oracle. |
| Performance proof | Same-session timing plus resource/counter/roofline attribution. |
| Promotion candidate | A candidate that passes structural, numeric, resource, timing, and rollback gates. |

## Done Levels

### D0: Research Taxonomy Done

Done when BoltBeam can name every category the search must answer.

Already implemented:

```text
boltbeam/search/research_categories.py
docs/machine-search-research-categories-20260710.md
```

Required outputs:

```text
research_categories()
research_category_slugs()
validate_research_category_coverage(...)
```

Pass criteria:

```text
18 categories exist
each category has a question
each category has evidence fields
coverage reports missing and unknown answers
```

### D1: Epoch Taxonomy Done

Done when BoltBeam can name the lifecycle epochs that matter for MMQ.

Already implemented:

```text
boltbeam/search/epoch_model.py
docs/mmq-epoch-model-exhaustive-scope-20260710.md
```

Required epochs:

```text
launch_ownership
q4k_tile_x_load_decode
q8_tile_y_stage
visibility_sync
dot_accumulate
k_advance
stage_reuse_or_overwrite
writeback
epilogue
```

Pass criteria:

```text
9 epochs exist and are ordered
each epoch declares required events
each epoch maps to research categories
coverage validator reports PASS/FAIL/BLOCKED/UNKNOWN
blocked reason extraction works
```

### D2: Llama Epoch Oracle Done

Done when BoltBeam can describe what llama MMQ is expected to do for bounded
shapes, without running GPU code.

Already implemented:

```text
boltbeam/search/mmq_epoch_oracle.py
```

Required shapes:

```text
16x16x256
128x128x256
```

Pass criteria:

```text
oracle emits all 9 epochs
source anchors point to llama MMQ concepts
ownership claims exist
reuse claims exist
sync claims exist
resource expectations are explicit source-level claims
unsupported shapes fail loudly
```

### D3: tinygrad Evidence Manifest Contract Done

Done when tinygrad emits enough opt-in evidence for BoltBeam to build an epoch
report for a compiled candidate.

Already partially implemented in tinygrad:

```text
AMD_ISA_PROOF_MANIFEST=1
ACCUM_READ
ACCUM_WRITE
V_WMMA
GLOBAL_STORE
GLOBAL_LOAD
GLOBAL_LOAD_B128
DS_LOAD
DS_LOAD_B128
DS_STORE
DS_STORE_B128
BARRIER
WAITCNT
```

Required tinygrad artifact:

```json
{
  "schema": "tinygrad.amd_isa_proof_manifest.v1",
  "candidate_id": "string",
  "kernel_name": "string",
  "source_sha256": "sha256",
  "binary_sha256": "sha256",
  "rows": [
    {
      "schema": "amd-isa-renderer-proof-manifest-row.v1",
      "kind": "global_load|ds_store|wmma|global_store|waitcnt|...",
      "logical_op": "GLOBAL_LOAD|DS_STORE|V_WMMA|...",
      "emitted": "instruction text",
      "epoch_id": "optional initially, required for promotion",
      "logical_identity": {},
      "physical_identity": {},
      "resources": {}
    }
  ]
}
```

Pass criteria:

```text
manifest is default-off
manifest does not change binary/mnemonic fixture hashes when off
manifest rows are tied to one compiled kernel
manifest rows include enough physical register/address/store identity for joins
missing epoch_id is allowed for D3, but must be recoverable by BoltBeam heuristics
```

### D4: Epoch Join Done

Done when BoltBeam can ingest a tinygrad manifest and produce an `EpochReport`.

Required BoltBeam module:

```text
boltbeam/search/epoch_join.py
```

Required input:

```text
llama_mmq_epoch_sequence(...)
mmq_epoch_specs()
tinygrad AMD ISA proof manifest
candidate metadata
```

Required output:

```text
EpochReport(candidate_id, workload, epochs, missing_epochs, blockers)
```

Per-epoch minimum evidence:

| Epoch | Minimum PASS evidence |
|---|---|
| launch_ownership | owner map exists and matches tile geometry claim |
| q4k_tile_x_load_decode | Q4_K global load/decode/stage rows or exact BLOCKED reason |
| q8_tile_y_stage | Q8/DS4 load/stage rows or exact BLOCKED reason |
| visibility_sync | barrier/wait rows between stage and consume |
| dot_accumulate | dot/WMMA/custom dot rows and accumulator identity rows |
| k_advance | K loop/carry evidence or exact bounded single-panel reason |
| stage_reuse_or_overwrite | reuse/overwrite boundary evidence or exact BLOCKED reason |
| writeback | global stores tied to owner and accumulator identity |
| epilogue | final wait/exit evidence or exact BLOCKED reason |

Pass criteria:

```text
every epoch is PASS, FAIL, BLOCKED, or UNKNOWN
UNKNOWN is allowed for probes only
promotion candidates cannot contain UNKNOWN
blocked rows name missing primitive/API/evidence
```

### D5: Numeric Candidate Done

Done when a compiled candidate computes bounded Q4_K x Q8_1/DS4 correctly.

Required tinygrad output:

```text
candidate output tensor
input seed / fixture identity
candidate_id
kernel manifest id
```

Required BoltBeam/tinygrad oracle:

```text
Q4_K x Q8_1/DS4 reference output
tolerance
shape
role
quant
```

Pass criteria:

```text
16x16x256 numeric pass
128x128x256 numeric pass before performance search
failure reports max error and failing coordinates
numeric test is tied to the same candidate/kernel identity as the manifest
```

### D6: Resource Snapshot Done

Done when resource facts are known for the exact compiled candidate.

Required fields:

```text
vgpr
sgpr
lds_bytes
scratch_bytes
workgroup
grid
occupancy estimate
source_sha256
binary_sha256
```

Pass criteria:

```text
resource row joins to manifest kernel identity
missing values are absent, not fabricated as zero
occupancy limiting resource is named when calculable
```

### D7: Comparator Timing Done

Done when the candidate is timed against direct packed and llama under the same
role/shape/session constraints.

Required comparisons:

```text
tinygrad direct packed baseline
tinygrad candidate
llama MMQ reference if available
same role
same shape
same prompt/prefill length
same measurement protocol
```

Pass criteria:

```text
timing artifacts are reproducible
warmup/repetition policy is recorded
candidate does not silently fall back
candidate beats baseline by predefined threshold or is marked refuted
```

### D8: Roofline Attribution Done

Done when BoltBeam can explain whether the candidate is memory-bound,
compute-bound, occupancy-bound, synchronization-bound, or structurally bad.

Required fields:

```text
estimated bytes moved
estimated ops
operational intensity
measured throughput
bandwidth ceiling
compute ceiling
dominant bottleneck
epoch attribution
```

Pass criteria:

```text
candidate is placed against a roofline or marked insufficient evidence
dominant bottleneck is named
next useful search axis is named
```

### D9: Search Integration Done

Done when machine search can vary only legal knobs and can reject candidates by
evidence instead of by vibes.

Required:

```text
candidate family declares owning epochs
each search knob maps to an epoch
each candidate emits manifest + numeric + resource + timing artifacts
BoltBeam ranks candidates with correctness/resource/timing constraints
refutations are durable
```

Pass criteria:

```text
search cannot promote UNKNOWN epochs
search cannot time candidates that fail structural/numeric gates
search reports failed epoch for every rejection
search records exact rollback for accepted candidates
```

### D10: Promotion Done

Done when a candidate can be made live/default without violating the project’s
principles.

Required:

```text
numeric gates pass
epoch gates pass
resource gates pass
same-session comparator passes
roofline attribution is coherent
route binding is modular, not model-size if-tree
rollback exists
tests prevent silent fallback
```

Pass criteria:

```text
candidate route can be selected by role/quant/shape/arch/resource facts
default behavior is intentional
manifest/provenance/evidence refs are committed
rollback flag or route exists
```

## BoltBeam To tinygrad Transfer Contract

BoltBeam should transfer these concrete requirements to tinygrad:

| Requirement | tinygrad responsibility |
|---|---|
| Candidate identity | emit stable `candidate_id`, kernel name, source hash, binary hash |
| Manifest rows | emit opt-in AMD ISA proof rows for load/stage/sync/dot/store |
| Epoch tags | preserve or expose enough tags for BoltBeam to assign rows to epochs |
| Numeric fixture | run bounded Q4_K x Q8_1/DS4 candidate against reference |
| Resource snapshot | expose VGPR/SGPR/LDS/scratch/workgroup/grid for same kernel |
| No silent fallback | fail if claimed candidate was not actually compiled/executed |
| Route facts | bind by role/quant/shape/arch/resource facts, not model labels |

Expected tinygrad artifact bundle:

```text
candidate_manifest.json
amd_isa_proof_manifest.json
kernel_resource_trace.json
numeric_result.json
timing_result.json
source or source hash
binary hash
```

## Final Done Statement

Machine search is **ready to run** when:

```text
For a candidate family, BoltBeam can generate or receive candidates, ingest a
tinygrad manifest, produce an EpochReport, run/verify bounded numeric output,
join resources, and reject failures by exact epoch/blocker.
```

Machine search is **ready to optimize** when:

```text
The candidate family passes structural epoch proof and numeric proof, and timing
plus resource data can be collected for each candidate.
```

Machine search is **ready to promote** when:

```text
The candidate beats the baseline under the comparator protocol, has coherent
roofline/resource attribution, binds through modular route facts, and has a
tested rollback.
```

Anything below these bars is a probe.
