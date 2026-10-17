# SPDX-License-Identifier: MIT
"""The regions a cost graph is built from.

* `Op`      one operator, the only leaf.
* `Seq`     regions one after another.
* `Repeat`  one body run `count` times; instance *i* binds `index` to
            `start + step * i`. The body is any region, nested to any depth.
* `Par`     regions that overlap on the device, combined by a `JoinPolicy`.

An operator's cost-deciding per-step state (block tables, context lengths) is
ambient, not an argument, so an `Op` names where to find it again
(`context_ref`) instead of holding a copy. A context key written as
`model.layers.{layer}.self_attn` is bound per instance by the `Repeat` that
binds `layer`; every region reports the indices it binds and the ones it
leaves free.
"""

import enum
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .shapes import as_shapes


class NodeKind(enum.Enum):
    """Where an operator came from."""

    CAPTURED = "captured"  # a dispatcher event, internals visible
    OPAQUE_LEAF = "opaque_leaf"  # one dispatcher op hiding a whole subsystem
    DECLARED = "declared"  # stated, not observed by the tracer


class JoinPolicy(enum.Enum):
    """How overlapping branches combine."""

    MAX = "max"  # the shorter branch hides inside the longer
    RESOURCE_BOUND = "resource_bound"  # branches contend for the same device
    EXCLUSIVE = "exclusive"  # a branch holds the whole device, costs add


@dataclass(frozen=True, slots=True)
class ContextRef:
    """The key an operator's ambient state is looked up by, with `{index}` holes."""

    key: str

    def __post_init__(self) -> None:
        try:
            names = self.index_names
        except ValueError as exc:
            raise ValueError(
                f"{self.key!r} is not a usable context key: {exc}"
            ) from None
        if not self.key.strip() or not all(name.isidentifier() for name in names):
            raise ValueError(
                f"{self.key!r}: a context key is non-empty and its placeholders "
                "are plain names a repeat index can carry"
            )

    @property
    def index_names(self) -> tuple[str, ...]:
        """The repeat indices this key is written in terms of, in order."""
        fields = (f for _, f, _, _ in string.Formatter().parse(self.key))
        return tuple(dict.fromkeys(f for f in fields if f is not None))

    def bind(self, **values: int) -> "ContextRef":
        """This key with its indices substituted, for one repeat instance."""
        if set(values) != set(self.index_names):
            raise KeyError(
                f"cannot bind {self.key!r}: it is written in terms of "
                f"{list(self.index_names)}, got {sorted(values)}"
            )
        return ContextRef(self.key.format(**values))


class Region:
    """One node of a cost graph: an `Op`, `Seq`, `Repeat` or `Par`.

    Each has `bound_indices`, the repeat indices bound at or below it, and
    `free_indices`, the ones its context keys name and nothing below binds.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Op(Region):
    """One operator: what ran, on what shapes, on which stream.

    `attrs` holds what the operator is (a dtype, a flag), sorted by name so two
    recordings compare equal. A value is a number, str, bytes, enum member,
    `None` or a tuple of those: a tensor or a live symbolic value is refused by
    its type, without being asked anything.
    """

    name: str
    kind: NodeKind
    in_shapes: tuple[tuple, ...]
    out_shapes: tuple[tuple, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    stream_id: int = 0
    context_ref: ContextRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", NodeKind(self.kind))
        object.__setattr__(self, "in_shapes", as_shapes(self.in_shapes))
        object.__setattr__(self, "out_shapes", as_shapes(self.out_shapes))
        object.__setattr__(self, "attrs", _as_attrs(self.attrs))

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset()

    @property
    def free_indices(self) -> frozenset[str]:
        return frozenset(self.context_ref.index_names if self.context_ref else ())


def _as_attrs(attrs: Any) -> tuple[tuple[str, Any], ...]:
    items = tuple(attrs.items() if isinstance(attrs, Mapping) else attrs)
    if not all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in items
    ):
        raise TypeError(f"attributes are (name, value) pairs, got {items!r}")
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"an attribute name is given twice in {names}")
    return tuple(sorted((name, _as_attr_value(name, value)) for name, value in items))


def _as_attr_value(name: str, value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes, enum.Enum)):
        return value
    if isinstance(value, tuple):
        return tuple(_as_attr_value(name, item) for item in value)
    raise TypeError(
        f"attribute {name!r} is a {type(value).__name__}. An attribute is a "
        "number, str, bytes, enum member, None, or a tuple of those."
    )


@dataclass(frozen=True, slots=True)
class Seq(Region):
    """Regions one after another, in the order they run."""

    items: tuple[Region, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _as_regions(self.items, 1))

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset().union(*(item.bound_indices for item in self.items))

    @property
    def free_indices(self) -> frozenset[str]:
        return frozenset().union(*(item.free_indices for item in self.items))


@dataclass(frozen=True, slots=True)
class Repeat(Region):
    """One body, run `count` times, binding `index` to `start + step * i`."""

    body: Region
    count: int
    index: str
    start: int = 0
    step: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.body, Region):
            raise TypeError(
                f"a repeat body is a region, got {type(self.body).__name__}"
            )
        if self.count < 2:
            raise ValueError(f"a repeat runs its body at least twice, got {self.count}")
        if (
            not (isinstance(self.index, str) and self.index.isidentifier())
            or not self.step
        ):
            raise ValueError(
                f"a repeat binds a plain name over distinct values, got "
                f"{self.index!r} with step {self.step}"
            )
        if self.index in self.body.bound_indices:
            raise ValueError(
                f"{self.index!r} is already bound inside this body; the inner "
                "binding would shadow this one"
            )

    @property
    def bound_indices(self) -> frozenset[str]:
        return self.body.bound_indices | {self.index}

    @property
    def free_indices(self) -> frozenset[str]:
        return self.body.free_indices - {self.index}

    def index_values(self) -> tuple[int, ...]:
        """The index value each instance binds, in order."""
        return tuple(self.start + self.step * i for i in range(self.count))


@dataclass(frozen=True, slots=True)
class Par(Region):
    """Regions that overlap on the device, and how their costs combine."""

    branches: tuple[Region, ...]
    join: JoinPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "branches", _as_regions(self.branches, 2))
        object.__setattr__(self, "join", JoinPolicy(self.join))

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset().union(*(branch.bound_indices for branch in self.branches))

    @property
    def free_indices(self) -> frozenset[str]:
        return frozenset().union(*(branch.free_indices for branch in self.branches))


def _as_regions(regions: Any, least: int) -> tuple[Region, ...]:
    items = tuple(regions)
    if len(items) < least or not all(isinstance(item, Region) for item in items):
        raise TypeError(f"expected at least {least} regions, got {items!r}")
    return items
