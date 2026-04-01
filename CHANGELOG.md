# Changelog

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
