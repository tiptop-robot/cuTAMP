"""
Test script to verify the movable-to-world initial collision masking.

Creates an environment where blocks slightly penetrate the floor (simulating
perception noise). Without masking, movable_to_world collision penalizes all
particles uniformly at pose_ts=0, making it harder to find a feasible plan.
With masking, initial timesteps are ignored per-object.

Usage:
    pixi run python -m cutamp.scripts.test_movable_to_world_fix
"""

import logging
import os
import sys

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol, setup_logging

_log = logging.getLogger(__name__)


def run_test(mask: bool, max_duration: float = 20.0, num_particles: int = 512) -> bool:
    """Run planning with masking on or off, return whether satisfying particles were found."""
    env = load_env(os.path.join(get_env_dir(), "blocks_activation_dist_test.yml"))
    config = TAMPConfiguration(
        num_particles=num_particles,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=max_duration,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        world_activation_distance=0.0,
        mask_initial_movable_world_collision=mask,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    _, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)
    _log.info(f"mask={mask}: num_satisfying={num_satisfying}, failure_reason={failure_reason}")
    return failure_reason is None


def main():
    setup_logging()

    unmasked = run_test(mask=False)
    masked = run_test(mask=True)

    _log.info(f"Without masking: {'PASS' if unmasked else 'FAIL'}")
    _log.info(f"With masking:    {'PASS' if masked else 'FAIL'}")

    if not unmasked and masked:
        _log.info("Fix correctly resolves initial collision edge case")
        return 0
    elif unmasked and masked:
        _log.warning("Both passed — environment may not trigger the bug")
        return 1
    elif not unmasked and not masked:
        _log.warning("Both failed — problem may be too hard or time limit too short")
        return 1
    else:
        _log.error("Unmasked passed but masked failed — regression!")
        return 2


if __name__ == "__main__":
    sys.exit(main())
