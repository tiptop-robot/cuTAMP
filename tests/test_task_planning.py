"""Tests for cutamp.task_planning.search.

Run with: pytest tests/test_task_planning.py -v
"""

import os

import pytest
import torch

from cutamp.task_planning import Fluent, Parameter
from cutamp.task_planning.search import breadth_first_search

On = Fluent("On", [Parameter("?o", "movable"), Parameter("?s", "surface")])
HandEmpty = Fluent("HandEmpty", [])
IsSurface = Fluent("IsSurface", [Parameter("?s", "surface")])
IsMovable = Fluent("IsMovable", [Parameter("?o", "movable")])

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _initial_state():
    return frozenset({HandEmpty.ground(), IsSurface.ground("bowl"), IsMovable.ground("book")})


def test_unknown_movable_in_goal_raises():
    """Goal referencing a movable not in the initial state should fail fast, not hang BFS."""
    initial = _initial_state()
    goal = frozenset({On.ground("spoon", "bowl"), HandEmpty.ground()})

    with pytest.raises(ValueError, match=r"unknown movable literal 'spoon'"):
        list(breadth_first_search(initial, goal, []))


def test_unknown_surface_in_goal_raises():
    """Goal referencing a surface not in the initial state should fail fast, not hang BFS."""
    initial = _initial_state()
    goal = frozenset({On.ground("book", "shelf"), HandEmpty.ground()})

    with pytest.raises(ValueError, match=r"unknown surface literal 'shelf'"):
        list(breadth_first_search(initial, goal, []))


def test_known_goal_literals_pass_validation():
    """Goal whose literals all appear in the initial state should not raise during validation."""
    initial = _initial_state()
    goal = frozenset({On.ground("book", "bowl"), HandEmpty.ground()})

    # No operators provided, so BFS finishes without yielding plans — the point is that the
    # goal-literal validation does not raise on a well-formed goal.
    plans = list(breadth_first_search(initial, goal, []))
    assert plans == []


@gpu
def test_unknown_goal_literal_propagates_through_run_cutamp():
    """Regression for the perception-hallucination hang: a goal that names an object not in the
    scene must surface as a `ValueError` from `run_cutamp` (not hang BFS forever). Mirrors the
    failure mode reported from tiptop where a hallucinated `On(spoon, bowl)` goal caused cuTAMP
    to spin in BFS until the process was killed."""
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.tamp_domain import HandEmpty as DomainHandEmpty
    from cutamp.tamp_domain import On as DomainOn

    env = load_env(os.path.join(get_env_dir(), "blocks_5.yml"))

    # Override the goal with a hallucinated movable ("spoon") that does not exist in blocks_5.
    # The "goal" surface does exist in the env (it's the green target region); the bug is in the
    # movable arg.
    env.goal_state = frozenset({DomainOn.ground("spoon", "goal"), DomainHandEmpty.ground()})

    config = TAMPConfiguration(
        num_particles=64,
        robot="fr3_robotiq",
        num_opt_steps=10,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    with pytest.raises(ValueError, match=r"unknown movable literal 'spoon'"):
        run_cutamp(env, config, cost_reducer, constraint_checker)
