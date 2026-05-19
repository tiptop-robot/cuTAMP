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


@gpu
def test_pick_only_goal_with_motion_plan_endpoints_consistent():
    """Every trajectory step in the returned motion plan must have its interpolated `plan`
    and `optimized_plan` end at the same joint configuration — they are the resampled and
    raw forms of the same trajopt output."""
    env = load_env(os.path.join(get_env_dir(), "pick_block.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        curobo_plan=True,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    plan, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)

    assert failure_reason is None
    assert num_satisfying > 0
    assert plan is not None

    trajectories = [step for step in plan if step["type"] == "trajectory"]
    assert len(trajectories) > 0

    for step in trajectories:
        interp_end = step["plan"].position[-1]
        opt_end = step["optimized_plan"].position[-1]
        assert torch.allclose(interp_end, opt_end, atol=1e-3), (
            f"{step['label']}: optimized_plan endpoint {opt_end.tolist()} diverges from "
            f"interpolated plan endpoint {interp_end.tolist()}"
        )
