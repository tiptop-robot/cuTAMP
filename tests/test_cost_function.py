"""Tests for cutamp.cost_function."""

import os

import pytest
import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _run_blocks_activation_test(mask: bool) -> int:
    """Run planning on the blocks_activation_dist_test env and return num satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "blocks_activation_dist_test.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        world_activation_distance=0.0,
        mask_initial_movable_world_collision=mask,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, _ = run_cutamp(env, config, cost_reducer, constraint_checker)
    return num_satisfying


@gpu
def test_movable_to_world_masking():
    """Masking initial movable-to-world collisions should find satisfying particles when blocks
    slightly penetrate the floor (simulating perception noise), while disabling masking should not."""
    assert _run_blocks_activation_test(mask=False) == 0
    assert _run_blocks_activation_test(mask=True) > 0
