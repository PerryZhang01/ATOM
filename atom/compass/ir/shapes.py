# SPDX-License-Identifier: MIT
"""Shapes: every dimension a concrete `int`.

A live symbolic size (a torch `SymInt`) is refused: hashing one raises, and
asking it most other questions installs a guard. Every check here is an
`isinstance`, which asks a symbolic value nothing.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def as_dim(dim: Any) -> int:
    """`dim` if it is an `int` >= 0."""
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"a dimension is a concrete int, got {type(dim).__name__}")
    if dim < 0:
        raise ValueError(f"a concrete dimension cannot be negative, got {dim}")
    return dim


def as_shape(dims: Sequence[Any]) -> tuple:
    """One tensor's shape as a tuple of dimensions."""
    return tuple(as_dim(dim) for dim in _sequence(dims))


def as_shapes(shapes: Sequence[Sequence[Any]]) -> tuple[tuple, ...]:
    """One shape per operand, in operand order."""
    return tuple(as_shape(shape) for shape in _sequence(shapes))


def _sequence(value: Any) -> Sequence:
    # A mapping iterates as its keys and a str as its characters, so either
    # would be read as a shape nobody wrote.
    if isinstance(value, (Mapping, str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"expected a sequence, got {type(value).__name__}")
    return value
