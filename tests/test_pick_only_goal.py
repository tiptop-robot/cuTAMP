"""Tests for planning with pick-only (Holding) goals."""

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


@gpu
def test_pick_only_goal_finds_plan():
    """Planning with a Holding-only goal should produce satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "pick_block.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)
    assert failure_reason is None
    assert num_satisfying > 0
