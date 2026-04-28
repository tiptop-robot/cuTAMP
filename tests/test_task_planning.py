"""Tests for cutamp.task_planning.search.

Run with: pytest tests/test_task_planning.py -v
"""

import pytest

from cutamp.task_planning import Fluent, Parameter
from cutamp.task_planning.search import breadth_first_search

On = Fluent("On", [Parameter("?o", "movable"), Parameter("?s", "surface")])
HandEmpty = Fluent("HandEmpty", [])
IsSurface = Fluent("IsSurface", [Parameter("?s", "surface")])
IsMovable = Fluent("IsMovable", [Parameter("?o", "movable")])


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
