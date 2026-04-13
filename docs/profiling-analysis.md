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

## Root cause: `_sphere_to_sphere_overlap_pytorch` in `costs.py`

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

## Optimization approaches for `_sphere_to_sphere_overlap`

The fundamental problem is materializing the `(B, T, n1, n2, 3)` pairwise diff tensor and
then traversing it multiple times for the forward pass (sub, mul, sum, sqrt, sub, relu, sum)
and again in reverse for autograd. With dynamic shapes (B, T, n1, n2 all vary per skeleton),
we need approaches that don't require fixed-size compiled kernels.

### 1. `torch.compile` with `dynamic=True` (low effort, moderate impact)

```python
_sphere_to_sphere_overlap = torch.compile(_sphere_to_sphere_overlap, dynamic=True)
```

**What it does:** Traces the function and fuses consecutive element-wise ops into single
GPU kernels. Instead of launching separate kernels for sub, mul, sum, sqrt, sub, relu, sum
(each reading/writing the full tensor from global memory), the compiler fuses them so data
stays in registers/L2 cache. Also compiles a fused backward pass.

**Why dynamic=True:** cuTAMP has variable batch sizes, timesteps, and sphere counts per
skeleton. `dynamic=True` generates kernels with symbolic shapes, avoiding recompilation
when dimensions change. There's a small overhead for shape guards but it's negligible
compared to the kernel fusion gains.

**Expected speedup:** 2-4x on this function. The current code launches ~7 separate CUDA
kernels per forward call, each doing a full memory round-trip on the massive intermediate
tensor. Fusion reduces this to 1-2 kernels. The backward pass benefits similarly.

**Risks:** The code was already written to be compile-friendly (manual distance instead of
`torch.cdist`, per the comment on line 82 of `costs.py`). Main risk is first-call
compilation latency (~10-30s), but this is a one-time cost per unique set of shapes and is
amortized over the 100-1000 optimization steps. Worst case: if the number of distinct shape
combinations is large, compilation overhead could dominate. In practice, cuTAMP typically
evaluates a handful of skeletons with the same particle count, so recompilation is rare.

**Try first.** This is one line of code.

### 2. Per-object AABB gating for `robot_to_movables` (low effort, variable impact)

Currently `robot_to_movables` concatenates ALL movable object spheres and checks the robot
against all of them:

```python
all_movable_spheres = torch.cat(list(obj_to_spheres.values()), dim=2)  # (B, T, 800, 4)
sphere_to_sphere_overlap(robot_spheres, all_movable_spheres[:, all_pose_ts])
```

With 4 objects × 200 spheres = 800 movable spheres checked against 38 robot spheres. But at
any given timestep, the robot arm is physically near at most 1-2 objects. AABB pre-filtering
per object would skip the sphere-sphere computation for objects whose bounding boxes don't
overlap the robot's bounding box.

**Implementation:** Loop over objects, compute AABB overlap between robot and each object at
each timestep, only run `_sphere_to_sphere_overlap` for overlapping (object, timestep) pairs.

**Expected speedup:** Depends heavily on the scene. In a scattered blocks scene, could skip
50-75% of sphere pairs. In a tightly packed scene, minimal benefit. Also adds overhead for
the AABB computation and the loop, plus complicates batching.

**Risk:** The boolean indexing `spheres[intersect]` breaks contiguous memory layout, which
can actually slow down the subsequent dense computation. Also harder to batch efficiently
when different (particle, timestep) combinations pass the filter.

### 3. Chunked per-object computation (low effort, moderate impact)

Instead of concatenating all movable spheres and computing one giant pairwise distance
matrix, compute robot-vs-each-object separately and sum the costs:

```python
for obj in obj_to_spheres:
    coll_values["robot_to_movables"] += sphere_to_sphere_overlap(
        robot_spheres, obj_to_spheres[obj][:, all_pose_ts], ...
    )
```

**What it does:** Trades one `(B, T, 38, 800, 3)` intermediate for four `(B, T, 38, 200, 3)`
intermediates computed sequentially. Peak memory drops 4x, and each chunk fits better in L2
cache. The total FLOPS are identical.

**Expected speedup:** Potentially 1.5-2x due to better cache utilization, even though the
total work is the same. The L2 cache on a 3090 is 6MB; a `(512, 8, 38, 200, 3)` float32
tensor is ~350MB, still far larger than L2. So the cache benefit is modest per chunk, but
the reduced peak memory helps avoid HBM thrashing and reduces autograd tape size.

**Combines well with approach 1** (`torch.compile`). Compiling smaller chunks is also faster
and more likely to produce optimal fused kernels.

### 4. Custom Triton kernel (high effort, high impact)

Write a fused forward+backward kernel in Triton that:
1. Reads sphere centers and radii from both sets
2. Computes pairwise distance, penetration, relu in registers
3. Accumulates the sum reduction on-the-fly
4. Never materializes the `(n1, n2)` intermediate in global memory

```python
@triton.jit
def sphere_overlap_fwd_kernel(centers_1, radii_1, centers_2, radii_2, output, ...):
    # Each program instance handles one (batch, timestep) pair
    # Tiles over n1 and n2 in shared memory
    ...
```

**Why Triton over raw CUDA:** Triton handles dynamic shapes natively (grid dimensions are
computed at launch time from Python). No recompilation needed when B, T, n1, n2 change —
these are just kernel launch parameters. Triton also auto-tunes tile sizes.

**Expected speedup:** 5-10x on this function. Eliminates all intermediate memory traffic.
The current code reads/writes the `(B, T, n1, n2, 3)` tensor ~7 times (forward) plus
~7 times (backward). A fused kernel reads the inputs once and writes the `(B, T)` output
once. That's roughly 14x less memory traffic.

**Risks:** Significant development effort. Need to write both forward and backward kernels
and wrap in a `torch.autograd.Function`. Testing requires careful numerical verification
against the PyTorch reference. Also need to handle the autograd integration correctly.

### 5. `torch.cdist` for distance computation (low effort, uncertain impact)

Replace the manual pairwise distance with `torch.cdist`:

```python
dist = torch.cdist(centers_1, centers_2, p=2)  # (B, T, n1, n2)
```

**What it does:** `torch.cdist` has optimized CUDA implementations that avoid materializing
the `(n1, n2, 3)` diff tensor. Instead it computes distances directly using a tiled
matrix-multiplication approach: `||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b`.

**Expected speedup:** Eliminates the biggest intermediate (the 3-channel diff). But the
remaining ops (radii_sum, penetration, relu, sum) still create `(n1, n2)` intermediates.
Roughly 2-3x on the forward pass distance computation alone.

**Risks:** The existing code comment (line 82) says the manual approach is "more efficient
than cdist for fusing with torch.compile." This suggests cdist may not compose well with
torch.compile, possibly because cdist dispatches to a specialized CUDA kernel that the
compiler can't fuse with the subsequent element-wise ops. Worth benchmarking both
`torch.compile(manual)` vs `torch.compile(cdist)` vs `cdist` alone.

### 6. Warp kernel (moderate effort, high impact)

Write a fused kernel in NVIDIA Warp (`@wp.kernel`) following cuRobo's existing pattern in
`curobo/geom/sdf/warp_primitives.py`. Warp is already a dependency and cuRobo demonstrates
the exact `torch.autograd.Function` integration pattern we'd need.

**Design:** Following cuRobo's approach, the forward kernel computes both the cost AND the
analytical gradient in a single pass. The backward pass just multiplies the stored gradient
by `grad_output` — no backward kernel needed.

```
Forward kernel (launched with dim = B * T):
  for each (batch, timestep):
    accumulate = 0.0
    grad_centers_1[n1][3] = 0.0  // local accumulators
    grad_centers_2[n2][3] = 0.0
    for i in range(n1):
      for j in range(n2):
        diff = centers_1[i] - centers_2[j]
        dist = sqrt(dot(diff, diff) + 1e-8)
        penetration = radii_1[i] + radii_2[j] - dist + activation_distance
        if penetration > 0:
          accumulate += penetration
          // gradient: d(relu(r1+r2-dist+a))/d(c1) = -diff/dist (when penetration > 0)
          grad = -diff / dist
          grad_centers_1[i] += grad
          grad_centers_2[j] -= grad
    output[batch, timestep] = accumulate
    // store gradients for backward
```

**Why this works for dynamic shapes:** `wp.launch(dim=B*T)` — B and T are just Python ints
passed at launch time. n1 and n2 are loop bounds read from tensor metadata. No
recompilation when shapes change.

**Key advantage over `torch.compile`:** No online compilation overhead. Warp compiles the
kernel once on first import (cached to `~/.cache/warp/`), then all subsequent calls are
just kernel launches. `torch.compile` re-traces and recompiles when it encounters new
shape patterns, which can cost 10-30s each time.

**Key advantage over current PyTorch:** Never materializes the `(B, T, n1, n2, 3)` or
`(B, T, n1, n2)` intermediates. The current code creates ~1.4GB of intermediates per call
at 200 spheres/object. The Warp kernel uses O(n1 + n2) local memory per thread.

**Expected speedup:** 5-10x on this function (similar to Triton estimate). Combined with
the fact that `robot_to_movables` is 33% of total CUDA time, this translates to ~25-30%
reduction in overall optimization time for high sphere counts.

**Concern — inner loop size:** With n1=38 and n2=800, the inner loop is 30,400 iterations
per thread. This is fine for a GPU kernel — each iteration is a few FLOPs. But we could
also tile the computation: launch with `dim = B * T * n1` and have each thread iterate
over n2, accumulating one row of the pairwise matrix. This gives better parallelism
(more threads) at the cost of requiring a reduction over n1 for the scalar output.

### Recommended order of attack

1. **`torch.compile` with warmup** — one-line change to try first. Add a warmup call at
   init time to pay the compilation cost upfront. If the number of distinct shape
   combinations is small (it likely is — B is fixed per run, T varies per skeleton), the
   compiled kernel is reused across optimization steps. Quick to validate. Abandoned if
   recompilation is triggered too frequently.

2. **Chunked per-object computation** — restructure `robot_to_movables` to loop per object
   instead of concatenating all movable spheres. Reduces peak memory 4x, independent of
   kernel strategy.

3. **Warp kernel** — the highest-impact change. Follows cuRobo's established pattern,
   eliminates the massive intermediate tensor, compiles once and caches. This is the
   recommended approach if `torch.compile` doesn't deliver sufficient speedup or has
   compilation overhead issues.

4. **Benchmark `torch.cdist`** — quick A/B test. If cdist's internal kernel is faster than
   the manual distance for the shapes we care about, it's a one-line swap.

Approaches 1-2 can be tried in an afternoon. Approach 3 is a focused day of work.
Approach 4 is a 5-minute test.
