"""Tests for the PlaceNear operator and the config.near_placement gating.

Run with: pytest tests/test_near_placement.py -v
"""

import os

import pytest
import torch

from cutamp.tamp_domain import HandEmpty, Near, On, PlaceNear, all_tamp_operators, get_initial_state
from cutamp.task_planning.search import breadth_first_search

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_near_goal_reachable_with_place_near():
    """With the full operator set, a Near goal is solved by a plan that uses PlaceNear."""
    initial = get_initial_state(movables=["a", "b"], surfaces=["table"])
    goal = frozenset({Near.ground("a", "b"), HandEmpty.ground()})

    plan = next(breadth_first_search(initial, goal, all_tamp_operators))
    assert any(op.operator is PlaceNear for op in plan), f"expected a PlaceNear in the plan, got {plan}"


def test_only_place_near_achieves_near():
    """PlaceNear is the only operator that adds a Near atom.
    If a Near effect were ever added elsewhere, the gating would silently break.
    """
    achievers = [op for op in all_tamp_operators if any(eff.name == Near.name for eff in op.add_effects)]
    assert achievers == [PlaceNear]


def test_gated_operators_still_solve_plain_placement():
    """Dropping PlaceNear (near_placement=False) must not regress ordinary On-goal planning."""
    initial = get_initial_state(movables=["a"], surfaces=["table"])
    goal = frozenset({On.ground("a", "table"), HandEmpty.ground()})
    gated = [op for op in all_tamp_operators if op is not PlaceNear]

    plan = next(breadth_first_search(initial, goal, gated))
    assert any(op.operator.name == "Place" for op in plan), f"expected a Place in the plan, got {plan}"


def _make_config(near_placement: bool) -> "TAMPConfiguration":
    from cutamp.config import TAMPConfiguration

    return TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        near_placement=near_placement,
    )


@gpu
def test_near_goal_without_flag_raises():
    """A Near goal with near_placement=False must fail fast rather than search forever."""
    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

    env = load_env(os.path.join(get_env_dir(), "place_near.yml"))
    config = _make_config(near_placement=False)
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    with pytest.raises(ValueError, match=r"Near atom"):
        run_cutamp(env, config, cost_reducer, constraint_checker)


@gpu
def test_place_near_finds_satisfying_plan():
    """With near_placement=True, planning the Near goal yields satisfying particles."""
    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

    env = load_env(os.path.join(get_env_dir(), "place_near.yml"))
    config = _make_config(near_placement=True)
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    _, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)
    assert failure_reason is None
    assert num_satisfying > 0
