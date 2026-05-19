# Changelog

## [0.0.5] - 2026-05-19

### Added
- `pick_block` environment with a `Holding` goal for pick-only planning (#18)
- `tests/test_pick_only_goal.py` covering planning with a `Holding`-only goal (#18)
- `tests/test_task_planning.py` covering BFS goal-literal validation (#14)

### Fixed
- BFS now rejects goal atoms whose non-fabricable literals (e.g. `movable`, `surface`) do not appear in the initial state, raising `ValueError` instead of expanding the frontier forever. Fixes a hang triggered by hallucinated perception objects in the goal (#14)
- Pick-only goals no longer crash in cost-function setup: object pairs and movable-to-world collision masks now handle objects that are picked but never placed (#18)

### Changed
- Consolidated fresh-symbol minting in `get_valid_ground_operators` behind a single `_FABRICABLE_TYPE_PREFIXES` table so the sampler and the BFS goal validator cannot drift out of sync (#14)

## [0.0.4] - 2026-04-14

All changes in this release are from #11 (Warp sphere overlap + FK Pose optimizations).

### Added
- NVIDIA Warp kernel for `sphere_to_sphere_overlap` with fused cost + analytical gradients, replacing the PyTorch pairwise implementation
- Concatenated `robot_to_movables` kernel launch — single call over all movable spheres instead of per-object launches
- `torch.profiler.record_function` annotations through the optimization loop, cost function, and rollout
- `--torch-profile`, `--torch-profile-output`, `--coll_n_spheres`, `--placement_shrink_dist`, and `--prop_satisfying_break` CLI flags on `cutamp-demo`
- `blocks_5` environment for benchmarking
- `tests/test_sphere_overlap.py` correctness + gradient tests for the Warp kernel
- `docs/profiling-analysis.md` covering how to profile cuTAMP and where time goes today

### Changed
- Rollout stores `ee_position` and `ee_quaternion` directly from cuRobo FK; `kinematic_costs` no longer round-trips through `Pose.from_matrix`
- Removed per-step `torch.cuda.synchronize()` in the optimization loop that was forcing CPU-GPU pipeline stalls

### Performance

End-to-end optimization loop wall time, 100 steps, 512 particles, RTX 3090, median of 3 runs. Speedup scales with the size of the movable-sphere pairwise tensor (more objects × more spheres/object → larger win):

| Env | Before (0.0.3) | After (0.0.4) | Speedup |
|---|---|---|---|
| `tetris_3` (3 blocks, ~6 sph/obj) | 1.46s | 1.36s | 1.07x |
| `blocks` (4 blocks, ~50 sph/obj) | 3.43s | 1.68s | 2.04x |
| `blocks_5` (5 blocks, 50 sph/obj) | 4.98s | 1.87s | 2.66x |

## [0.0.3] - 2026-04-13

### Added
- `max_motion_refine_attempts` config option to cap the number of satisfying particles tried during motion refinement per skeleton (#9)
- Retry next plan skeleton when motion refinement fails instead of breaking out of the loop (#9)
- Test environment and pytest for movable-to-world initial collision fix (#10)

### Fixed
- Movable-to-world collision now masks initial timesteps per object, matching the movable-to-movable approach, so objects with perception noise at their initial pose are not incorrectly penalized (#10)
- Vectorized movable-to-world collision into a single batched `collision_fn` call with cached mask for better performance (#10)
- `break_on_satisfying` no longer exits the skeleton loop when motion planning is enabled but all particles fail (#9)

## [0.0.2] - 2026-04-01

### Added
- Expose `__version__` in `__init__.py` via `importlib.metadata`

### Fixed
- Disable CPU pose update to avoid mutating object pose in `TAMPEnvironment`

## [0.0.1] - 2026-03-22

### Added
- Initial release of cuTAMP from NVLabs — GPU-parallelized TAMP solver with core algorithm, cost functions, rollout, samplers, task planning search, and environment definitions (book shelf, stick button, tetris)
- Robot support for Franka Panda, FR3, and UR5e with Robotiq 2F-85/140 grippers
- TiPToP integration: extended algorithm and motion solver, added Franka+Robotiq robot config, OBB collision utilities, new environment assets
- Return failure reason from planner for better diagnostics
- Log git status in experiment logger
