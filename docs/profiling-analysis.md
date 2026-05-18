# Profiling cuTAMP

How to profile the cuTAMP optimization loop and where time is currently spent.
Historical optimization-by-optimization analysis (Warp kernel, concat launch, FK Pose
round-trip) lives in the PR that introduced them (#11).

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

The `--torch-profile` flag captures full GPU kernel timing. `record_function` annotations
in `optimize_plan.py`, `cost_function.py`, and `rollout.py` label key sections so they
show up as named spans in the trace.

## Where time goes today

Blocks_5 (5 objects, 50 spheres/object, 512 particles, 100 opt steps, RTX 3090).
Total CUDA time: ~1.2s excluding init.

**By phase (per step):**

| Phase | CUDA ms/step | % |
|---|---|---|
| Rollout (FK + spheres) | 4.1 | 34% |
| Optimizer step (Adam) | 3.9 | 32% |
| Backward (FK kernels) | ~2.5 | 20% |
| Cost function | 1.8 | 15% |
| Cost reduction + satisfying mask | 1.3 | 10% |

**Cost function breakdown:**

| Sub-component | CUDA ms (total) | % of total |
|---|---|---|
| `cost::kinematic` | 165ms | 13.5% |
| `cost::collision` | 160ms | 13.1% |
| — `coll::movable_to_movable` | 149ms | 12.2% |
| — `coll::robot_to_movables` | 136ms | 11.1% |
| `cost::transform_spheres` | 115ms | 9.4% |
| `cost::stable_placement` | 53ms | 4.3% |

The remaining time is dominated by cuRobo internals (FK kernels forward/backward, Adam
optimizer) which cannot be optimized from our side.

## Things we tried that didn't help

- **Per-step `torch.cuda.synchronize()`**: Removed. The original code synced every step
  for wall-clock timing, which forced CPU-GPU serialization.
- **AABB gating on `robot_to_movables`**: Rejected. Robot arm AABB spans the workspace
  (~40–80% overlap rate); boolean indexing overhead exceeds savings.
- **AABB on `movable_to_movable`**: Kept — legitimately skips entire object-pair batches
  where AABBs don't overlap.
- **`torch.compile`**: Not viable; recompilation overhead when shape combinations vary
  across plan skeletons.
