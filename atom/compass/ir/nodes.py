# SPDX-License-Identifier: MIT
"""The regions a cost graph is built from.

* `Op`      one operator, the only leaf.
* `Seq`     regions one after another.
* `Repeat`  one body run `count` times; instance *i* binds `index` to
            `start + step * i`. The body is any region, nested to any depth.
* `Par`     regions that overlap on the device, combined by a `JoinPolicy`.

An `Op` holds its arguments by schema slot, which is what calling it again
needs: a tensor as a `Tensor` (shape, dtype, stride, role), a module as a
`ModuleRef`, an integer affine in a decode step's context as an `Affine`,
anything else as its value.

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

from .shapes import as_shape, as_shapes


class NodeKind(enum.Enum):
    """Where an operator came from."""

    CAPTURED = "captured"  # a dispatcher event, internals visible
    OPAQUE_LEAF = "opaque_leaf"  # one dispatcher op hiding a whole subsystem
    DECLARED = "declared"  # stated, not observed by the tracer


class JoinPolicy(enum.Enum):
    """How overlapping branches combine."""

    MAX = "max"  # the shorter branch hides inside the longer
    RESOURCE_BOUND = "resource_bound"  # branches contend for the same device


class Role(enum.Enum):
    """Where a tensor argument's storage came from."""

    ACTIVATION = "activation"  # allocated by the model's forward or after it
    WEIGHT = "weight"  # a parameter, buffer or tensor attribute of the model
    KV = "kv"  # the KV cache and per-request state
    METADATA = "metadata"  # the runner's buffers and its input preparation


@dataclass(frozen=True, slots=True)
class Tensor:
    """A tensor argument: enough to make one like it, and where it came from."""

    shape: tuple
    dtype: str
    stride: tuple
    role: Role

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", as_shape(self.shape))
        object.__setattr__(self, "stride", as_shape(self.stride))
        object.__setattr__(self, "role", Role(self.role))


@dataclass(frozen=True, slots=True)
class Affine:
    """An integer argument that is `slope * context + intercept` at a decode
    step whose rows hold `context` tokens."""

    slope: int
    intercept: int

    def at(self, context: int) -> int:
        return self.slope * context + self.intercept


@dataclass(frozen=True, slots=True)
class ContextRef:
    """The key an operator's ambient state is looked up by, with `{index}` holes.

    A hole is a sum of repeat indices and at most one trailing offset, so a
    layer two repeats deep is named `{outer+inner}` and its neighbour
    `{outer+inner+1}`.
    """

    key: str

    def __post_init__(self) -> None:
        try:
            holes = self._holes()
        except ValueError as exc:
            raise ValueError(
                f"{self.key!r} is not a usable context key: {exc}"
            ) from None
        if not self.key.strip() or not all(
            _is_sum(terms) for _, terms in holes if terms is not None
        ):
            raise ValueError(
                f"{self.key!r}: a context key is non-empty and each placeholder "
                "is a sum of plain names a repeat index can carry, then at most "
                "one offset"
            )

    def _holes(self) -> list[tuple[str, list[str] | None]]:
        holes = []
        for text, field, spec, conversion in string.Formatter().parse(self.key):
            if spec or conversion:
                raise ValueError(f"the placeholder {field!r} takes no format")
            holes.append((text, None if field is None else field.split("+")))
        return holes

    @property
    def index_names(self) -> tuple[str, ...]:
        """The repeat indices this key is written in terms of, in order."""
        terms = (t for _, ts in self._holes() if ts for t in ts if not t.isdigit())
        return tuple(dict.fromkeys(terms))

    def bind(self, **values: int) -> "ContextRef":
        """This key with its indices substituted, for one repeat instance."""
        if set(values) != set(self.index_names):
            raise KeyError(
                f"cannot bind {self.key!r}: it is written in terms of "
                f"{list(self.index_names)}, got {sorted(values)}"
            )
        return type(self)(
            "".join(
                text
                + (
                    ""
                    if terms is None
                    else str(sum(int(values.get(t, t)) for t in terms))
                )
                for text, terms in self._holes()
            )
        )


def _is_sum(terms: list[str]) -> bool:
    *names, last = terms
    return all(t.isidentifier() for t in names) and (
        last.isidentifier() or bool(names) and last.isdigit()
    )


class Region:
    """One node of a cost graph: an `Op`, `Seq`, `Repeat` or `Par`.

    Each has `bound_indices`, the repeat indices bound at or below it, and
    `free_indices`, the ones its context keys name and nothing below binds.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ModuleRef(ContextRef):
    """A module argument, by its path in the model, with `{index}` holes as a
    context key has."""


@dataclass(frozen=True, slots=True)
class Op(Region):
    """One operator: what ran, with which arguments, on which stream.

    `args` holds each argument by its slot, in the order the operator takes
    them: a `Tensor`, a `ModuleRef`, a `ContextRef`, an `Affine`, a number, str,
    bytes, enum member, `None` or a tuple of those. Anything else, a live
    symbolic value among it, is refused by its type, without being asked
    anything.
    """

    name: str
    kind: NodeKind
    args: tuple[tuple[str, Any], ...]
    out_shapes: tuple[tuple, ...]
    stream_id: int = 0
    context_ref: ContextRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", NodeKind(self.kind))
        object.__setattr__(self, "args", _as_args(self.args))
        object.__setattr__(self, "out_shapes", as_shapes(self.out_shapes))

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset()

    @property
    def free_indices(self) -> frozenset[str]:
        refs = [self.context_ref, *_refs(self.args)]
        return frozenset(name for ref in refs if ref for name in ref.index_names)


def _as_args(args: Any) -> tuple[tuple[str, Any], ...]:
    items = tuple(args.items() if isinstance(args, Mapping) else args)
    if not all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in items
    ):
        raise TypeError(f"arguments are (slot, value) pairs, got {items!r}")
    slots = [slot for slot, _ in items]
    if len(set(slots)) != len(slots):
        raise ValueError(f"a slot is given twice in {slots}")
    return tuple((slot, _as_value(slot, value)) for slot, value in items)


def _as_value(slot: str, value: Any) -> Any:
    if value is None or isinstance(
        value, (bool, int, float, str, bytes, enum.Enum, Tensor, ContextRef, Affine)
    ):
        return value
    if isinstance(value, tuple):
        return tuple(_as_value(slot, item) for item in value)
    raise TypeError(
        f"argument {slot!r} is a {type(value).__name__}. An argument is a Tensor, "
        "ModuleRef, ContextRef, Affine, number, str, bytes, enum member, None, or a "
        "tuple of those."
    )


def _refs(value: Any):
    if isinstance(value, ContextRef):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _refs(item)


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
