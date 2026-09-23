# Machine Search Research Categories

Purpose: ground GPU-kernel machine search in research facts before candidate
generation. Search should only vary knobs after these categories have an
oracle, measurement, or explicit blocked state.

| # | Category | What It Defines | External Reference Anchor |
|---:|---|---|---|
| 1 | Correctness contract | The exact operation, inputs, outputs, dtype, shape, and tolerance a candidate must satisfy. | TVM Ansor starts from a tensor computation definition before constructing schedules/search space. |
| 2 | Numeric oracle | The trusted implementation or reference output used to reject wrong candidates before timing. | CUDA Best Practices emphasizes verification and profiling as separate steps in optimization workflow. |
| 3 | Data layout | Packed tensor formats, scale/min placement, alignment, stride, and addressability. | CUDA guides cover memory hierarchy and access patterns; coalescing depends on layout and addressing. |
| 4 | Tile geometry / thread-block geometry | CTA/workgroup size, wave/warp count, tile M/N/K, and per-block resource shape. | CUDA Programming Guide covers thread blocks, grids, and shared memory per block. |
| 5 | Work ownership / thread-to-output mapping | Which thread, lane, warp/wave, or fragment owns each output element. | CUDA execution model defines warps/threads as the mapping substrate for parallel work. |
| 6 | Accumulator / `sum[]` mapping | Logical accumulator slot to lane/register/output mapping. | AMD occupancy guidance treats VGPR use as a primary resource limiter; accumulator placement determines VGPR pressure. |
| 7 | Shared memory / LDS lifecycle | What is staged, how it is padded, when it is visible, and how many consumers reuse it. | CUDA shared memory and AMD LDS guidance both frame local memory as a reuse mechanism with occupancy tradeoffs. |
| 8 | K-loop cadence / reuse cadence | How the reduction loop advances through K, when panels reload, and what work is amortized per load. | Roofline analysis depends on operational intensity, i.e. work per byte moved. |
| 9 | Dot primitive / tensor instruction choice | Which scalar/vector/tensor instruction family performs the inner product. | CUDA and AMD optimization material distinguish instruction throughput ceilings from memory ceilings. |
| 10 | Register, VGPR, SGPR, LDS, occupancy model | Resource usage and resident-wave limits for each candidate. | AMD GPUOpen occupancy explains VGPR/LDS as limiting resources for resident wavefronts. |
| 11 | Wait / barrier / synchronization cadence | Required ordering for memory visibility and dependency correctness, plus unnecessary wait overhead. | CUDA Programming Guide covers synchronization; AMD tuning material treats barriers/LDS visibility as performance-critical. |
| 12 | Store path / memory coalescing | Final output store mapping, vector width, address pattern, and coalescing behavior. | CUDA Best Practices highlights global memory coalescing as a high-priority performance factor. |
| 13 | Baseline and comparator | Same-session baseline, candidate, and external implementation timing under controlled conditions. | Roofline/NERSC docs position performance models as comparative tools against machine capability and measured bottlenecks. |
| 14 | Roofline / operational-intensity target | Whether the candidate is memory-bound, compute-bound, or blocked by a lower ceiling. | Roofline model: attainable performance is bounded by peak compute and bandwidth times operational intensity. |
| 15 | Search-space knobs | The legal dimensions search can vary: tile sizes, staging, vector width, unroll, waits, placement. | Ansor constructs and explores a hierarchical search space with cost-model-guided search. |
| 16 | Route binding rules | The modular facts that select a candidate: role, quant, shape, architecture, and resource support. | Search systems separate workload description from schedule/candidate choices; avoid model-name branches. |
| 17 | Promotion gates | Criteria for moving a research candidate into a live/default route. | Optimization workflow requires correctness, reproducible measurement, and resource evidence before promotion. |
| 18 | Stop / blocked criteria | The exact missing fact, primitive, API, or hardware constraint that halts the search. | Roofline and occupancy analysis both identify hard ceilings; blocked criteria prevent search from optimizing impossible paths. |

## Reference Links

- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- AMD GPUOpen, Occupancy Explained: https://gpuopen.com/learn/occupancy-explained/
- AMD ROCm blog, Occupancy Math on AMD MI355X: https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
- NERSC Roofline documentation: https://docs.nersc.gov/tools/performance/roofline/
- Roofline paper: https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
- TVM Ansor introduction: https://tvm.apache.org/2021/03/03/intro-auto-scheduler
- Ansor paper: https://arxiv.org/pdf/2006.06762

## Use Rule

Before launching a machine-search pass, every candidate family should cite which
rows it satisfies, which rows are measured, and which rows are blocked. A search
result without this table filled in is a probe, not a promotion candidate.
