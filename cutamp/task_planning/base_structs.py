# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field
from typing import Sequence, Tuple

# Global cache for interning ground atoms
_ATOM_CACHE: dict[tuple, "Atom"] = {}


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Fluent:
    """Represents a predicate with typed parameters."""

    name: str
    parameters: Sequence[Parameter] = field(default_factory=list)

    def __call__(self, *parameters: Parameter) -> "Fluent":
        """Create a new Fluent with the given parameters. Makes sure the types match."""
        if len(parameters) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameters, got {len(parameters)}")

        for param, new_param in zip(self.parameters, parameters):
            if param.type != new_param.type:
                raise ValueError(f"Expected type {param.type}, got {new_param.type}")

        return Fluent(self.name, parameters)

    def ground(self, *values: str) -> "Atom":
        """Ground a fluent by providing values for its parameters. We're only assume boolean atoms."""
        if not values:
            if self.parameters:
                raise ValueError(f"Expected {len(self.parameters)} values, got 0")
            # Cache zero-parameter atoms too
            cache_key = (self.name, ())
            if cache_key not in _ATOM_CACHE:
                _ATOM_CACHE[cache_key] = Atom(self)
            return _ATOM_CACHE[cache_key]

        if len(self.parameters) != len(values):
            raise ValueError(f"Expected {len(self.parameters)} values, got {len(values)}")

        values_tuple = tuple(val.name if isinstance(val, Parameter) else val for val in values)

        # Check cache first
        cache_key = (self.name, values_tuple)
        if cache_key not in _ATOM_CACHE:
            _ATOM_CACHE[cache_key] = Atom(self, values_tuple)
        return _ATOM_CACHE[cache_key]

    def __str__(self) -> str:
        param_str = ", ".join(str(param) for param in self.parameters)
        return f"{self.name}({param_str})"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Atom:
    """Represents a fluent with concrete values (no variables)."""

    fluent: Fluent
    values: Tuple[str, ...] = field(default_factory=tuple)
    _cached_str: str = field(default=None, init=False, compare=False, repr=False, hash=False)
    _cached_hash: int = field(default=None, init=False, compare=False, repr=False, hash=False)

    def __post_init__(self):
        """Eagerly compute and cache string representation and hash."""
        val_str = ", ".join(str(val) for val in self.values)
        object.__setattr__(self, "_cached_str", f"{self.name}({val_str})")
        object.__setattr__(self, "_cached_hash", hash(self._cached_str))

    @property
    def name(self) -> str:
        return self.fluent.name

    def __str__(self) -> str:
        return self._cached_str

    def __hash__(self) -> int:
        return self._cached_hash

    def __eq__(self, other):
        # Equality also based on string representation (same fluent + same values)
        return isinstance(other, Atom) and str(self) == str(other)

    def __repr__(self) -> str:
        return str(self)


State = frozenset[Atom]


def _ground_fluents(fluents: Sequence[Fluent], substitutions: dict[str, str]) -> set[Atom]:
    """Helper method to ground a sequence of fluents with given substitutions."""
    ground_atoms = set()
    for fluent in fluents:
        param_names = tuple(param.name for param in fluent.parameters)
        values = tuple(substitutions[name] for name in param_names)
        # Let fluent.ground() handle caching to avoid double-counting
        ground_atoms.add(fluent.ground(*values))
    return ground_atoms


@dataclass(frozen=True)
class Operator:
    """Represents a lifted (parameterized) planning operator."""

    name: str
    parameters: Sequence[Parameter] = field(default_factory=list)
    preconditions: Sequence[Fluent] = field(default_factory=list)
    add_effects: Sequence[Fluent] = field(default_factory=list)
    del_effects: Sequence[Fluent] = field(default_factory=list)

    def ground(self, substitutions: dict[str, str]) -> "GroundOperator":
        """Ground this operator using the given values for the parameters."""
        # Order substitutions by parameter name
        substitutions = {param.name: substitutions[param.name] for param in self.parameters}
        ground_name = f"{self.name}({', '.join(substitutions.values())})"
        ground_op = GroundOperator(
            name=ground_name,
            values=list(substitutions.values()),
            operator=self,
            preconditions=_ground_fluents(self.preconditions, substitutions),
            add_effects=_ground_fluents(self.add_effects, substitutions),
            del_effects=_ground_fluents(self.del_effects, substitutions),
        )
        return ground_op

    def __str__(self) -> str:
        param_str = ", ".join(str(param) for param in self.parameters)
        return f"{self.name}({param_str})"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class GroundOperator:
    """Represents a ground (fully instantiated) planning operator."""

    name: str
    values: Sequence[str]
    operator: Operator
    preconditions: set[Atom]
    add_effects: set[Atom]
    del_effects: set[Atom]

    def apply(self, state: State) -> State:
        """Apply this ground operator to a state."""
        # Make sure preconditions are satisfied
        if not self.preconditions.issubset(state):
            raise ValueError(f"Not all preconditions are satisfied for {self.name}")

        new_state = set(state)
        for eff in self.del_effects:
            if eff not in new_state:
                raise RuntimeError(f"Effect {eff} not in state {state} for operator {self.name}")
        new_state.difference_update(self.del_effects)
        new_state.update(self.add_effects)
        return frozenset(new_state)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)
