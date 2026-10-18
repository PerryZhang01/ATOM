# SPDX-License-Identifier: MIT
"""Shapes, and the form a size that is not known yet is held in.

A dimension is a concrete `int` or a `SymDim`. A live symbolic size (a torch
`SymInt`) is never stored: hashing one raises, and asking it most other
questions installs a guard. The capture renders it once with
`SymDim.of(expr, scope)`, and the node holds the text. Every check here is an
`isinstance`, which asks a symbolic value nothing.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SymDim:
    """An open size: its rendering, and the capture whose symbols it names.

    Symbols are numbered per capture, so `s26` from two captures is two sizes.
    """

    text: str
    scope: str

    def __post_init__(self) -> None:
        if not self.text.strip() or "\n" in self.text or not self.scope.strip():
            raise ValueError(
                "a symbolic size needs a one-line rendering and the capture it "
                f"was read against, got {self.text!r} in {self.scope!r}"
            )
        try:
            float(self.text)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"{self.text!r} is a known size and belongs in the shape as an int"
            )
        if self.text.startswith("<") and self.text.endswith(">"):
            raise ValueError(f"{self.text!r} is an object with no rendering of its own")

    @classmethod
    def of(cls, expr: object, scope: str) -> "SymDim":
        """Render a live symbolic size; `str()` installs no guard."""
        return cls(str(expr), scope)


def as_dim(dim: Any) -> int | SymDim:
    """`dim` if it is an `int` >= 0 or a `SymDim`; nothing is rendered for you."""
    if isinstance(dim, SymDim):
        return dim
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(
            "a dimension is a concrete int or SymDim.of(expr, scope), got "
            f"{type(dim).__name__}"
        )
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
