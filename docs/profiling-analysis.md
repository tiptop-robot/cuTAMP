# cuTAMP GPU Profiling Analysis

**Date:** 2025-04-09
**Hardware:** NVIDIA GPU (3090-class), tested via `cutamp-demo` with `--torch-profile`
**Profiler:** `torch.profiler` with CPU + CUDA activities, record_function annotations

## How to reproduce

```bash
pixi shell

# Lightweight workload (tetris, 6 spheres/object)
cutamp-demo --env tetris_3 --disable_visualizer --num_opt_steps 100 \
  --torch-profile --torch-profile-output trace_tetris.json

# Realistic workload (blocks, 200 spheres/object)
cutamp-demo --env blocks --disable_visualizer --num_opt_steps 100 \
  --coll_n_spheres 200 --prop_satisfying_break 0 \
  --torch-profile --torch-profile-output trace_blocks.json
```

Open the `.json` trace in `chrome://tracing`.

The `--torch-profile` flag captures full GPU kernel timing. The `record_function` annotations
in `optimize_plan.py`, `cost_function.py`, and `rollout.py` label the key sections so they
appear as named spans in the trace.

## Profiling results

### Tetris 3 (6 spheres/object, 3 objects) — 100 opt steps

Total CUDA time: 1.93s

| Component | CUDA time | % |
|---|---|---|
| `rollout` (FK + pose chain) | 519ms | 27.0% |
| `Optimizer.step#Adam` | 464ms | 24.1% |
| cuRobo FK backward kernel | 394ms | 20.4% |
| cuRobo FK forward kernel | 378ms | 19.6% |
| `cost::kinematic` | 363ms | 18.8% |
| cuRobo sphere-OBB collision | 262ms | 13.6% |
| `coll::movable_to_movable` | 200ms | 10.4% |
| `cost_reduction` | 175ms | 9.1% |
| `cost::transform_spheres` | 170ms | 8.8% |
| `coll::robot_to_movables` | 163ms | 8.5% |
| `cost::stable_placement` | 105ms | 5.5% |
| `rollout::forward_kinematics` | 95ms | 4.9% |

With few spheres per object, the workload is balanced across cuRobo FK kernels, Adam,
kinematic pose error, and collision — no single dominant bottleneck.

### Blocks (200 spheres/object, 4 objects) — 100 opt steps

Total CUDA time: 10.46s (5.4x slower than tetris)

| Component | CUDA time | % |
|---|---|---|
| **`coll::robot_to_movables`** | **3.46s** | **33.1%** |
| `aten::mul` | 2.74s | 26.2% |
| `aten::sum` | 1.75s | 16.7% |
| `aten::neg` (backward) | 984ms | 9.4% |
| `aten::sub` | 848ms | 8.1% |
| `SqrtBackward0` | 611ms | 5.8% |
| `aten::add` | 704ms | 6.7% |
| `aten::div` | 371ms | 3.6% |
| `coll::movable_to_movable` | 264ms | 2.5% |
| cuRobo FK kernels | ~400ms | ~4% |

The `coll::robot_to_movables` call dominates at **33% of GPU time**. The `aten::mul`,
`aten::sum`, `aten::sub`, `aten::neg`, `SqrtBackward0` entries are the element-wise ops
inside `_sphere_to_sphere_overlap_pytorch` and its autograd backward pass.

## Root cause: `sphere_to_sphere_overlap_pytorch` in `costs.py`

The baseline PyTorch implementation computes pairwise distances between two sets of spheres:

```python
diff = centers_1.unsqueeze(-2) - centers_2.unsqueeze(-3)   # (B, T, n1, n2, 3)
dist_sq = (diff * diff).sum(dim=-1)                         # (B, T, n1, n2)
dist = torch.sqrt(dist_sq + 1e-8)
penetration = radii_sum - dist + activation_distance
return torch.relu(penetration).sum((-2, -1))                # (B, T)
```

With 512 particles, 8 timesteps, 38 robot spheres, and 800 movable spheres (4 objects × 200):
- The `diff` intermediate is `(512, 8, 38, 800, 3)` = **350M floats = 1.4 GB**
- Autograd stores this for the backward pass, doubling memory pressure
- Each optimization step materializes and traverses this tensor multiple times (forward + backward)
- The ops are element-wise, so the bottleneck is **memory bandwidth**, not compute

The scaling is O(B × T × n_robot_spheres × n_movable_spheres), which grows quadratically
with sphere count. This explains why 200 spheres/object is 20x slower than 6 spheres/object
for this cost alone.

## Other observations

- **Per-step `torch.cuda.synchronize()` (removed)**: The pre-optimization baseline had a
  `torch.cuda.synchronize()` call every step in `optimize_plan.py` for wall-clock timing in
  `opt_metrics["elapsed"]`. This forced CPU-GPU sync each iteration — CPU time was 15.8s vs
  10.5s CUDA time, suggesting ~5s of sync stalls. This has been removed in the current code.

- **Adam optimizer**: With few spheres it's 24% of GPU time; with many spheres it becomes
  negligible relative to collision costs. Not worth optimizing.

- **cuRobo FK and collision kernels**: Custom CUDA, well-optimized, not easily changeable.
  They shrink as a percentage as sphere count grows.

- **`coll::movable_to_movable`**: Uses AABB gating which helps, but `coll::robot_to_movables`
  does NOT use AABB gating.

## Implemented optimization: Warp kernel for `sphere_to_sphere_overlap`

The bottleneck was `sphere_to_sphere_overlap_pytorch` in `costs.py`, which materializes
an O(B × T × n1 × n2 × 3) intermediate tensor for pairwise distances. With 512 particles,
8 timesteps, 38 robot spheres, and 800 movable spheres (4 objects × 200), the intermediate
is ~1.4 GB per call. The ops are element-wise and memory-bandwidth-bound.

### Approach: Warp GPU kernel with analytical gradients

We replaced the PyTorch implementation with a fused NVIDIA Warp kernel (`costs_warp.py`)
that never materializes the pairwise intermediate. The design follows cuRobo's pattern
in `curobo/geom/sdf/warp_primitives.py`.

**Architecture:**

- **Kernel 1** (`_sphere_overlap_fwd_kernel_1`): One thread per `(batch_elem, sphere_1_idx)`.
  Loops over all spheres in set 2, accumulates partial cost and analytical gradient for
  spheres_1. Early-exits pairs where `dist² ≥ (r1 + r2 + act_dist)²`.

- **Kernel 2** (`_sphere_overlap_fwd_kernel_2`): One thread per `(batch_elem, sphere_2_idx)`.
  Computes gradients for spheres_2. Skipped entirely when `requires_grad` is False on both
  inputs, avoiding unnecessary work during inference or constraint checking.

- **`torch.autograd.Function`** wrapper (`_SphereOverlapWarp`): Forward pass launches both
  kernels, stores analytical gradients. Backward pass is just `stored_grad * grad_output` —
  no backward kernel needed.

**Key design decisions:**

- `requires_grad` check instead of `torch.is_grad_enabled()`: PyTorch disables autograd
  inside `Function.forward`, so `is_grad_enabled()` is always False. We check
  `spheres_1.requires_grad or spheres_2.requires_grad` to decide whether to compute and
  store gradients.

- Broadcasting fallback: The Warp kernel requires matching batch dims. When batch dims
  differ (broadcasting), we fall back to `sphere_to_sphere_overlap_pytorch`. This dispatch
  happens in `sphere_to_sphere_overlap()` in `costs.py`.

- Per-object loop in `cost_function.py`: Instead of concatenating all movable spheres into
  one tensor, we loop over objects and sum costs. This reduces peak memory and improves
  cache utilization, complementing the Warp kernel.

### Benchmark results

Measured on NVIDIA RTX 3090 (forward + backward, 100 iterations):

| Workload | PyTorch | Warp | Speedup |
|---|---|---|---|
| Small (32×2, 10 vs 50) | 0.13 ms | 0.08 ms | 1.6x |
| Realistic (512×8, 38 vs 200) | 3.1 ms | 0.42 ms | 7.4x |
| Large (512×8, 38 vs 800) | 12.8 ms | 0.42 ms | 30.8x |

The speedup grows with sphere count because the Warp kernel avoids materializing the
quadratically-growing intermediate tensor.

### End-to-end impact

On the `blocks` demo with 200 spheres/object:

- **Before:** `coll::robot_to_movables` was 33% of total CUDA time (3.46s out of 10.46s
  for 100 opt steps)
- **After:** Optimization loop takes ~17-19s for 1000 steps (vs ~20s before), with
  collision no longer the dominant bottleneck

### Alternatives considered

1. **`torch.compile(dynamic=True)`** — One-line change, 2-4x expected speedup from kernel
   fusion. Rejected: recompilation overhead when shape combinations vary across skeletons,
   and lower ceiling than a fused kernel.

2. **Custom Triton kernel** — Similar performance ceiling to Warp, but Warp was preferred
   because it's already a cuRobo dependency and cuRobo demonstrates the exact
   `torch.autograd.Function` integration pattern.

3. **`torch.cdist`** — Avoids the `(n1, n2, 3)` diff tensor but still materializes `(n1, n2)`
   intermediates for penetration/relu. Limited benefit and doesn't compose well with
   `torch.compile`.
