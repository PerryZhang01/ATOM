# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""How a cost was obtained, and what a declined answer carries.

Every number a cost backend produces says where it came from. The vocabulary
is closed and small, and an unlabelled cost cannot be built: `Provenance`
takes a `Species` with no default, and `CostTerm` takes a `Provenance` with no
default, so there is no constructor path that produces a number whose origin
is unknown.

The five species answer "how was this answer obtained", which is a different
question from "which cost model was asked". A step-level timing read back out
of a file is still `MEASURED`; what a lookup changes is whether the key matched
exactly (`MEASURED`), matched the nearest neighbour (`INTERPOLATED`) or sat
outside the range that was measured at all (`EXTRAPOLATED`). "Priced" is not a
species either -- a price is a measurement of a smaller unit, and the unit goes
in `detail`, as `measured (op-level)` against `measured (step-level)`.

`ANALYTICAL` is the one word that is reserved rather than aspirational: it
means the subject was computed without ever being measured, and a backend that
fits coefficients to timings is `FITTED`, not analytical, however tidy the
closed form looks.

A `Refusal` is a declined answer with a named reason, and it is a value rather
than an error. Every refusal collected while resolving a cost is carried on the
provenance of the cost that was produced instead -- all of them, earliest
first, not just the one that renders -- so a cost that came from a later source
cannot be mistaken for one that came from the first, and a count of
reasons is not quietly missing the middle of the chain.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass


class Species(enum.Enum):
    """How one answer was obtained."""

    ANALYTICAL = "analytical"
    MEASURED = "measured"
    FITTED = "fitted"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"

    def __str__(self) -> str:
        return self.value


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
    inventing a species for each. `source` names the thing that answered, so
    the record identifies the source and not only the species. `refusals` holds
    everything that declined on the way to this answer, earliest first, which
    is what makes a fall-through visible in the record instead of an inference
    from a changed number.
    """

    species: Species
    detail: str = ""
    source: str = ""
    refusals: tuple[Refusal, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.species, Species):
            raise TypeError(f"species must be a Species, got {self.species!r}")
        if not isinstance(self.refusals, tuple):
            raise TypeError(f"refusals must be a tuple, got {self.refusals!r}")
        for refusal in self.refusals:
            if not isinstance(refusal, Refusal):
                raise TypeError(f"not a refusal: {refusal!r}")

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
        return Provenance(
            self.species, self.detail, named, tuple(declined) + self.refusals
        )

    def __str__(self) -> str:
        named = f"{self.species} ({self.detail})" if self.detail else str(self.species)
        if self.source:
            named = f"{named} via {self.source}"
        if not self.refusals:
            return named
        listed = "; ".join(f"{r.source}: {r.reason}" for r in self.refusals)
        return f"refused({listed}) -> {named}"
