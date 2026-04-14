# Changelog

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
