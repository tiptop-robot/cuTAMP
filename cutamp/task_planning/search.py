# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Sequence

from cutamp.task_planning import Atom, GroundOperator, Operator, State

_log = logging.getLogger(__name__)


@dataclass
class _Node:
    state: frozenset[Atom]
    parent: Optional["_Node"]
    operator: Optional[GroundOperator]
    depth: int
    _cached_params: Optional[dict[str, set[str]]] = field(default=None, compare=False, hash=False, repr=False)
    _cached_fluent_map: Optional[dict[str, set[Atom]]] = field(default=None, compare=False, hash=False, repr=False)

    def fluent_to_atoms_map(self) -> dict[str, set[Atom]]:
        """Returns a mapping from fluent names to their corresponding atoms in this state."""
        if self._cached_fluent_map is None:
            # Build the map more efficiently by pre-sizing and using direct dict operations
            fluent_map = {}
            for atom in self.state:
                name = atom.name
                if name in fluent_map:
                    fluent_map[name].add(atom)
                else:
                    fluent_map[name] = {atom}
            self._cached_fluent_map = fluent_map
        return self._cached_fluent_map

    def parameters(self) -> dict[str, set[str]]:
        """Returns all the parameters ever encountered"""
        if self._cached_params is not None:
            return self._cached_params

        if self.parent:
            # Shallow copy the dict and shallow copy each set
            param_type_to_literals = {k: v.copy() for k, v in self.parent.parameters().items()}
        else:
            param_type_to_literals = defaultdict(set)

        # All the literals in the state
        for atom in self.state:
            param_types = [param.type for param in atom.fluent.parameters]
            literals = atom.values
            for param_type, literal in zip(param_types, literals):
                if param_type not in param_type_to_literals:
                    param_type_to_literals[param_type] = set()
                param_type_to_literals[param_type].add(literal)

        # All the literals in the operator
        if self.operator is not None:
            for param, values in zip(self.operator.operator.parameters, self.operator.values):
                if param.type not in param_type_to_literals:
                    param_type_to_literals[param.type] = set()
                param_type_to_literals[param.type].add(values)

        # Cache the result
        self._cached_params = param_type_to_literals
        return param_type_to_literals

    def extract_solution(self) -> list[GroundOperator]:
        """Extract the solution path from this node."""
        solution = []
        node = self
        while node.parent is not None:
            assert node.operator is not None
            solution.append(node.operator)
            node = node.parent
        return list(reversed(solution))


def get_valid_ground_operators(
    node: _Node,
    operators: Sequence[Operator],
    operator_precond_fluents: Sequence[frozenset[str]],
    verbose: bool = False,
) -> list[GroundOperator]:
    """
    Get all valid ground operators by testing the operators, binding samples for the unspecified variables, and
    checking preconditions are satisfied.
    """
    ground_ops = []
    state = node.state

    # Use cached fluent-to-atoms mapping from the node
    fluent_to_atoms = node.fluent_to_atoms_map()

    # Pre-filter operators based on precondition fluent availability
    # Skip operators whose required fluents don't exist in the current state
    # Use pre-computed precondition fluent sets for faster subset checking
    available_fluent_names = frozenset(fluent_to_atoms.keys())
    relevant_operators = [
        op
        for op, pre_fluents in zip(operators, operator_precond_fluents)
        if pre_fluents.issubset(available_fluent_names)
    ]

    if verbose and len(relevant_operators) < len(operators):
        _log.debug(
            f"Filtered operators from {len(operators)} to {len(relevant_operators)} "
            f"based on available fluents: {available_fluent_names}"
        )

    # For each operator, we try to ground it given the current state
    for operator in relevant_operators:
        # FIXME: this sampling is slow once we have lots of plan skeletons, so improve it in the future
        # Got to do it inside here so we reset for each operator
        param_type_to_literals = node.parameters()

        # This is hacky and could break things due to naming, but ok for now
        def _sample_param_type(param_type: str) -> str:
            if param_type in param_type_to_literals:
                if param_type == "conf":
                    # remove the 'q' prefix
                    conf_nums = {int(lit[1:]) for lit in param_type_to_literals[param_type]}
                    conf_max = max(conf_nums)
                    return f"q{conf_max + 1}"
                elif param_type == "pose":
                    pose_nums = {int(lit[4:]) for lit in param_type_to_literals[param_type]}
                    pose_max = max(pose_nums)
                    return f"pose{pose_max + 1}"
                elif param_type == "traj":
                    traj_nums = {int(lit[4:]) for lit in param_type_to_literals[param_type]}
                    traj_max = max(traj_nums)
                    return f"traj{traj_max + 1}"
                elif param_type == "grasp":
                    grasp_nums = {int(lit[5:]) for lit in param_type_to_literals[param_type]}
                    grasp_max = max(grasp_nums)
                    return f"grasp{grasp_max + 1}"
                else:
                    raise NotImplementedError
            else:
                return f"{param_type}1"

        def sample_param_type(param_type: str) -> str:
            new_sample = _sample_param_type(param_type)
            if param_type not in param_type_to_literals:
                param_type_to_literals[param_type] = set()
            param_type_to_literals[param_type].add(new_sample)
            return new_sample

        # 1. Check all preconditions are satisfied, and map each parameter to it's associated values in the ground atom
        param_to_literals: dict[str, set[str]] = defaultdict(set)
        pre_check = True
        for pre_fluent in operator.preconditions:
            if pre_fluent.name not in fluent_to_atoms:
                pre_check = False
                break

            # Bind the operator parameters to their values in the ground atom
            atoms = fluent_to_atoms[pre_fluent.name]
            for atom in atoms:
                if len(atom.values) != len(pre_fluent.parameters):
                    raise RuntimeError(
                        f"Number of values in atom {atoms} does not match number of parameters in {pre_fluent}"
                    )
                for param, value in zip(pre_fluent.parameters, atom.values):
                    param_to_literals[param.name].add(value)

        if not pre_check:
            if verbose:
                _log.debug(f"Skipping {operator} because not all preconditions are satisfied.")
            continue

        if verbose:
            _log.debug(
                f"Preconditions all satisfied for {operator}, parameter to literal bindings: {param_to_literals}"
            )

        # 2. Since we're dealing with funny planning, some of the parameters might not be bound. We need to check and
        # sample placeholder values for them.
        for param in operator.parameters:
            if param.name not in param_to_literals:
                sampled_value = sample_param_type(param.type)
                # sampled_value = param.sample_name()
                param_to_literals[param.name].add(sampled_value)
                if verbose:
                    _log.debug(f"Parameter '{param}' not bound, sampled placeholder value {sampled_value}")

        if len(param_to_literals) != len(operator.parameters):
            raise RuntimeError(
                f"Number of parameter bindings {len(param_to_literals)} "
                f"does not match number of parameters in {operator}"
            )

        # 3. Generate all combinations of the parameter bindings, so we ground the operator
        param_names = list(param_to_literals.keys())
        combinations = list(itertools.product(*param_to_literals.values()))
        for combo in combinations:
            substitutions = dict(zip(param_names, combo))
            ground_op = operator.ground(substitutions)

            # Check preconditions are satisfied for this specific parameter combination
            # Note: We checked fluent availability earlier, but not all parameter combinations
            # will have their preconditions satisfied in the state
            if not ground_op.preconditions.issubset(state):
                if verbose:
                    _log.debug(f"Skipping {ground_op} because not all preconditions are satisfied.")
            else:
                ground_ops.append(ground_op)
                if verbose:
                    _log.debug(f"Grounded operator {operator} with substitutions {substitutions} to get {ground_op}")

    return ground_ops


def breadth_first_search(
    initial_state: State,
    goal_state: State,
    operators: Sequence[Operator],
    continue_branch_after_goal: bool = False,
    explored_state_check: bool = True,
    verbose: bool = False,
) -> Generator[List[GroundOperator], None, None]:
    """
    Performs a breadth-first search to find a solution to the given planning problem.

    Parameters
    ----------
    initial_state: State
    goal_state: State
    operators: List[Operator]
        The possible operators for the given search problem.
    continue_branch_after_goal: bool
        Whether to continue searching on a given branch after finding a solution.
    explored_state_check: bool
        Whether to check if a state has been explored before adding it to the frontier.
    verbose: bool
        Whether to log debugging information. We have a flag so the log doesn't become polluted.

    Returns
    -------
    Generator[List[GroundOperator], None, None]
        A generator that yields plans which are lists of ground operators.
    """
    # Sanity check, make sure the initial and goal state contain atoms only
    for elem in initial_state:
        if not isinstance(elem, Atom):
            raise ValueError(f"Initial state must contain only atoms, got {elem.__class__.__name__} {elem}")
    for elem in goal_state:
        if not isinstance(elem, Atom):
            raise ValueError(f"Goal state must contain only atoms, got {elem.__class__.__name__} {elem}")

    # Pre-compute precondition fluent sets for each operator (optimization)
    operator_precond_fluents = [frozenset(pre.name for pre in op.preconditions) for op in operators]

    initial_node = _Node(state=initial_state, parent=None, operator=None, depth=0)
    frontier: deque[_Node] = deque([initial_node])
    explored: set[State] = {initial_node.state}

    while frontier:
        node = frontier.popleft()
        if verbose:
            _log.debug(f"Exploring node: {node}")

        # Check if goal is satisfied
        if goal_state.issubset(node.state):
            plan = node.extract_solution()
            yield plan
            if not continue_branch_after_goal:
                continue  # stop exploring this branch as we already satisfied the goal

        # Get successor and add to frontier
        ground_ops = get_valid_ground_operators(node, operators, operator_precond_fluents, verbose=verbose)
        for ground_op in ground_ops:
            successor_state = ground_op.apply(node.state)

            # If the successor state hasn't been explored, add it to the frontier.
            # Also if successor is a goal state (we want different ways of getting to the goal)
            if not explored_state_check or successor_state not in explored or goal_state.issubset(successor_state):
                child = _Node(state=successor_state, parent=node, operator=ground_op, depth=node.depth + 1)
                frontier.append(child)
                explored.add(successor_state)
