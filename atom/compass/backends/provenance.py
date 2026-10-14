# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""How a cost was obtained, and what a declined answer carries.

Every number a cost backend produces says how it was obtained. `Provenance`
takes a `Method` with no default, and `CostTerm` takes a `Provenance` with no
default, so there is no constructor path that produces a number whose origin
is unknown.

The four methods:

- `ANALYTICAL`: a form from first principles whose every parameter is model
  geometry, deployment configuration, or a device parameter from a datasheet
  or a model-independent probe on this platform.
- `DECLARED`: some input is assumed, or was observed on another subject
  (another platform, width, model or engine configuration). A declared
  provenance carries a `note` stating that origin, and what replaces it or why
  nothing does.
- `MEASURED`: the query equals an observation in every variable that affects
  the value, with no projection.
- `ESTIMATED`: anything else derived from observations of the subject. An
  estimated answer is `validated` when the query lies inside the region where
  its algorithm's held-out error was measured, and is not otherwise. The
  algorithm, extrapolation included, is named in `source`.

`MEASURED` and `ESTIMATED` are the two `empirical` methods. `combine` gives
the method of an answer built from several: any declared input makes it
declared, all analytical stays analytical, all measured stays measured, and
anything else is estimated.

A `Refusal` is a declined answer with a named reason, and it is a value rather
than an error. Every refusal collected while resolving a cost is carried on the
provenance of the cost that was produced instead -- all of them, earliest
first, not just the one that renders -- so a cost that came from a later source
cannot be mistaken for one that came from the first, and a count of
reasons is not quietly missing the middle of the chain.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class Method(enum.Enum):
    """How one answer was obtained."""

    ANALYTICAL = "analytical"
    DECLARED = "declared"
    MEASURED = "measured"
    ESTIMATED = "estimated"

    @property
    def empirical(self) -> bool:
        return self in (Method.MEASURED, Method.ESTIMATED)

    def __str__(self) -> str:
        return self.value


def combine(methods: Iterable[Method]) -> Method:
    """The method of an answer built from parts obtained by `methods`."""
    methods = set(methods)
    if not methods:
        raise ValueError("an answer built from no parts has no method")
    if Method.DECLARED in methods:
        return Method.DECLARED
    return methods.pop() if len(methods) == 1 else Method.ESTIMATED


@dataclass(frozen=True)
class Refusal:
    """A source declining to answer, with the reason it declined.

    `source` names the thing that declined so a reader can go and fix it, and
    `reason` says what it could not do. Both are required: a refusal nobody can
    act on is as useless as a guess, and the counting a run reports groups by
    the pair.
    """

    source: str
    reason: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("a refusal must name the source that declined")
        if not self.reason.strip():
            raise ValueError(f"{self.source} declined without naming a reason")

    def __str__(self) -> str:
        return f"refused({self.source}: {self.reason})"


@dataclass(frozen=True)
class Provenance:
    """The origin of one cost.

    `detail` is the unit or the key that was matched, free text, and is how
    `measured (op-level)` is distinguished from `measured (step-level)` without
    inventing a method for each. `source` names the thing that answered, so
    the record identifies the source and not only the method. `refusals` holds
    everything that declined on the way to this answer, earliest first, which
    is what makes a fall-through visible in the record instead of an inference
    from a changed number. `validated` is false unless an estimated answer's
    query lies where its algorithm's held-out error was measured; `note` is a
    declared answer's origin and successor.
    """

    method: Method
    detail: str = ""
    source: str = ""
    refusals: tuple[Refusal, ...] = ()
    validated: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.method, Method):
            raise TypeError(f"method must be a Method, got {self.method!r}")
        if not isinstance(self.refusals, tuple):
            raise TypeError(f"refusals must be a tuple, got {self.refusals!r}")
        for refusal in self.refusals:
            if not isinstance(refusal, Refusal):
                raise TypeError(f"not a refusal: {refusal!r}")
        if self.validated is not False and (
            self.validated is not True or self.method is not Method.ESTIMATED
        ):
            raise ValueError(
                f"a {self.method} answer cannot be validated={self.validated!r}: "
                "only an estimated answer has held-out error to be inside"
            )
        if self.method is Method.DECLARED and not self.note.strip():
            raise ValueError(
                "a declared answer needs a note stating its origin, and its "
                "successor or why there is none"
            )

    @property
    def is_refused(self) -> bool:
        return bool(self.refusals)

    @property
    def refusal(self) -> Refusal | None:
        """The earliest refusal, which is the one an operator has to close."""
        return self.refusals[0] if self.refusals else None

    def resolved(self, source: str, declined: Sequence[Refusal] = ()) -> Provenance:
        """This origin as a resolver produced it.

        `declined` is prepended rather than merged or dropped, because a source
        may itself be resolver-backed: its own refusals are already here, and
        they happened *after* the ones being added, so earliest-first ordering
        puts the new ones in front and the first refusal is genuinely the
        first. Nothing is discarded -- dropping the inner chain would let a
        composed resolver report the wrong source and undercount the reasons.

        The answering name composes the same way: an answer from a source that
        is itself a resolver reads `outer/inner`, so the record names the path
        that produced the number rather than only its outermost step.
        """
        if not source.strip():
            raise ValueError("an answer has to name the source that produced it")
        named = f"{source}/{self.source}" if self.source else source
        return dataclasses.replace(
            self, source=named, refusals=tuple(declined) + self.refusals
        )

    def __str__(self) -> str:
        named = str(self.method)
        if self.method is Method.ESTIMATED:
            named += ", validated" if self.validated else ", unvalidated"
        if self.detail:
            named = f"{named} ({self.detail})"
        if self.source:
            named = f"{named} via {self.source}"
        if self.note:
            named = f"{named}; {self.note}"
        if not self.refusals:
            return named
        listed = "; ".join(f"{r.source}: {r.reason}" for r in self.refusals)
        return f"refused({listed}) -> {named}"
