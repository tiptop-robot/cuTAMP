# cuTAMP GPU Profiling Analysis

**Date:** 2025-04-09 (initial), 2026-04-14 (updated with concat + FK Pose optimizations)
**Hardware:** NVIDIA RTX 3090, tested via `cutamp-demo` with `--torch-profile`
**Profiler:** `torch.profiler` with CPU + CUDA activities, record_function annotations

## How to reproduce

```bash
pixi shell

# Lightweight workload (tetris, 6 spheres/object)
cutamp-demo --env tetris_3 --disable_visualizer --num_opt_steps 100 \
  --torch-profile --torch-profile-output trace_tetris.json

# Realistic workload (blocks_5, 50 spheres/object, 5 objects)
cutamp-demo --env blocks_5 --disable_visualizer --num_opt_steps 100 \
  --coll_n_spheres 50 --prop_satisfying_break 0 \
  --torch-profile --torch-profile-output trace_blocks5.json
```

Open the `.json` trace in `chrome://tracing`.

The `--torch-profile` flag captures full GPU kernel timing. The `record_function` annotations
in `optimize_plan.py`, `cost_function.py`, and `rollout.py` label the key sections so they
appear as named spans in the trace.

## Pre-optimization baseline

### Blocks (200 spheres/object, 4 objects) — 100 opt steps

Total CUDA time: 10.46s

| Component | CUDA time | % |
|---|---|---|
| **`coll::robot_to_movables`** | **3.46s** | **33.1%** |
| `aten::mul` | 2.74s | 26.2% |
| `aten::sum` | 1.75s | 16.7% |
| `aten::neg` (backward) | 984ms | 9.4% |
| `aten::sub` | 848ms | 8.1% |
| `SqrtBackward0` | 611ms | 5.8% |
| `aten::add` | 704ms | 6.7% |
| cuRobo FK kernels | ~400ms | ~4% |

The `coll::robot_to_movables` call dominated at **33% of GPU time**. The `aten::*` entries
are element-wise ops inside `sphere_to_sphere_overlap_pytorch` and its autograd backward
pass, materializing an O(B × T × n1 × n2 × 3) intermediate tensor (~1.4 GB per call).

## Optimization 1: Warp kernel for `sphere_to_sphere_overlap`

Replaced the PyTorch pairwise distance implementation with a fused NVIDIA Warp kernel
(`costs_warp.py`) that never materializes the intermediate tensor.

**Architecture:**

- **Kernel 1** (`_sphere_overlap_fwd_kernel_1`): One thread per `(batch_elem, sphere_1_idx)`.
  Loops over all spheres in set 2, accumulates partial cost and analytical gradient for
  spheres_1. Early-exits pairs where `dist² >= (r1 + r2 + act_dist)²`.

- **Kernel 2** (`_sphere_overlap_fwd_kernel_2`): One thread per `(batch_elem, sphere_2_idx)`.
  Computes gradients for spheres_2. Skipped when `spheres_2.requires_grad` is False.

- **`torch.autograd.Function`** wrapper (`_SphereOverlapWarp`): Forward launches both
  kernels, stores analytical gradients. Backward is just `stored_grad * grad_output`.

**Benchmark results** (RTX 3090, forward + backward, 100 iterations):

| Workload | PyTorch | Warp | Speedup |
|---|---|---|---|
| Small (32×2, 10 vs 50) | 0.13 ms | 0.08 ms | 1.6x |
| Realistic (512×8, 38 vs 200) | 3.1 ms | 0.42 ms | 7.4x |
| Large (512×8, 38 vs 800) | 12.8 ms | 0.42 ms | 30.8x |

## Optimization 2: Concatenated `robot_to_movables` call

The initial Warp integration used per-object kernel launches (one Warp call per movable
object). Benchmarking with `benchmark_per_object_vs_concat.py` showed that at our sphere
counts (50–100), a single concatenated call is significantly faster:

| Objects | Spheres/obj | Per-object | Concat | Winner |
|---|---|---|---|---|
| 3 | 50 | 1.05 ms | 0.57 ms | concat 1.8x |
| 5 | 50 | 1.56 ms | 0.66 ms | concat 2.4x |
| 10 | 50 | 2.95 ms | 1.12 ms | concat 2.6x |
| 3 | 200 | 1.18 ms | 1.29 ms | per-object 1.1x |
| 5 | 200 | 1.97 ms | 2.09 ms | per-object 1.1x |

Per-object launches only win at 200+ spheres where the kernel's early-exit check benefits
from smaller n2, but the margin is within noise (~5-9%). At 50 spheres, kernel launch
overhead dominates.

**End-to-end impact on blocks_5** (5 objects, 50 spheres, 50 opt steps):

`coll::robot_to_movables` CUDA time: **179.8ms → 70.1ms** (2.6x faster)

## Optimization 3: Avoid Pose round-trip in kinematic costs

cuRobo's FK returns a `Pose` object (position + quaternion). Previously the rollout
converted this to a 4×4 matrix via `Pose.get_matrix()`, then `kinematic_costs` converted
it back via `Pose.from_matrix()` — wasting a quaternion→matrix→quaternion round-trip
every optimization step.

Now the rollout stores `ee_position` and `ee_quaternion` directly. `kinematic_costs`
constructs a `Pose` from these without matrix conversion. The desired side still needs
`from_matrix` since it's built from matrix multiplication in the rollout.

**Benchmark** (`benchmark_kinematic_cost.py`, batch=5120):

| Path | ms/iter | Speedup |
|---|---|---|
| Current (matrix round-trip) | 1.01 ms | — |
| Keep FK Pose (1 from_matrix) | 0.67 ms | 1.5x |
| Both Poses (no conversion) | 0.30 ms | 3.4x |

**End-to-end impact**: `cost::kinematic` dropped from 158ms to 123ms (22%).
Optimization loop wall time improved ~8.5% (2.13s → 1.95s on blocks_5, 100 steps).

## Current state (post-optimization)

### Blocks_5 (50 spheres/object, 5 objects) — 100 opt steps

Total CUDA time: 1.225s (excluding init)

**By phase (per step):**

| Phase | CUDA ms/step | % |
|---|---|---|
| Rollout (FK + spheres) | 4.1 | 34% |
| Cost function | 1.8 | 15% |
| Cost reduction + satisfying mask | 1.3 | 10% |
| Backward (FK kernels) | ~2.5 | ~20% |
| Optimizer step (Adam) | 3.9 | 32% |

**Cost function breakdown:**

| Sub-component | CUDA ms (total) | % of total |
|---|---|---|
| `cost::kinematic` | 165ms | 13.5% |
| `cost::collision` | 160ms | 13.1% |
| — `coll::movable_to_movable` | 149ms | 12.2% |
| — `coll::robot_to_movables` | 136ms | 11.1% |
| `cost::transform_spheres` | 115ms | 9.4% |
| `cost::stable_placement` | 53ms | 4.3% |

The remaining time is dominated by cuRobo internals (FK forward/backward kernels,
Adam optimizer, L-BFGS line search) which cannot be optimized from our side.

## Other observations

- **Per-step `torch.cuda.synchronize()` (removed)**: The original code had a sync every
  step for wall-clock timing. This forced CPU-GPU serialization — removed early in this PR.

- **AABB gating on `robot_to_movables`**: Tested and rejected. Robot arm AABB spans the
  entire workspace (39-78% overlap rate), and boolean indexing overhead exceeds savings.

- **AABB on `movable_to_movable`**: Still valuable — legitimately skips entire object-pair
  batch elements where AABBs don't overlap.

- **`torch.compile`**: Not viable due to recompilation overhead when shape combinations
  vary across plan skeletons.

## Benchmark scripts

- `cutamp/scripts/benchmark_sphere_overlap.py` — Warp vs PyTorch at various sizes
- `cutamp/scripts/benchmark_per_object_vs_concat.py` — per-object launches vs concatenated call
- `cutamp/scripts/benchmark_kinematic_cost.py` — Pose round-trip overhead measurement
