# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The order cost sources are consulted in, and what a fall-through leaves behind.

A backend rarely has one source of truth. It has several, ranked: an exact
price for the thing being asked about, a nearest-neighbour lookup, a fitted
form, a law computed from geometry alone. `Resolver` walks them in order and
takes the first answer.

Falling through is allowed and is not free. A source that cannot answer says
why, as a `Refusal`, and *every* refusal collected on the way down is carried
on the answer's provenance -- not the first one with the rest discarded, which
would undercount the reasons a coverage report exists to list. The answer is
also stamped with the name of the source that produced it, always and not only
after a fall-through, so the record says what answered as well as what did
not. There is no path through this class that produces an unannotated answer
from a lower source. The failure being designed out is a run that quietly
answers from the last source and reports a tidy number.

A source may itself be resolver-backed, which is the shape a layered backend
takes. The chain composes rather than collapsing: the outer refusals go in
front of the inner ones, so the earliest refusal is still the first, and the
answering name reads `outer/inner`.

A source whose own resolver is exhausted is a source that declined, not an escape
from this one: its `CostRefused` is caught here, its refusals join this
resolver's, and the next source is consulted. Letting it propagate would discard
the refusals collected above it *and* skip every source below it, which is the
silent-fall-through failure inverted -- a run that refuses where it had an
answer available, and cannot say what it passed over.

When no source answers, nothing is invented. `CostRefused` is raised carrying
every refusal in order, inner resolvers' included, so the caller gets the
complete list of what would have to be measured rather than the first item of
it. What the caller does next --
charge the step from a lower-fidelity model and mark it, or stop the run -- is
the caller's policy, and this module deliberately has no opinion beyond
refusing to make the number up.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from atom.compass.backends.cost import CostTerm
from atom.compass.backends.provenance import Refusal


class CostSource(abc.ABC):
    """One source: something that can price a request, or say why it cannot."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, stable, and specific enough to act on when it declines."""

    @abc.abstractmethod
    def price(self, request: Any) -> CostTerm | Refusal:
        """Answer with a term, or decline with `self.refuse(reason)`."""

    def refuse(self, reason: str) -> Refusal:
        """Build a refusal that names this source, so it cannot name another."""
        return Refusal(self.name, reason)


@dataclass(frozen=True)
class Resolution:
    """Which source answered, and what was passed over to get there."""

    term: CostTerm
    source: str
    declined: tuple[Refusal, ...]

    @property
    def fell_through(self) -> bool:
        return bool(self.declined)


class CostRefused(Exception):
    """No source could answer. Carries the whole list, not the first entry."""

    def __init__(self, request: Any, declined: Sequence[Refusal]) -> None:
        self.request = request
        self.declined = tuple(declined)
        listed = "; ".join(str(r) for r in self.declined) or "no sources were consulted"
        super().__init__(f"no cost source answered for {request!r}: {listed}")


class Resolver:
    """A fixed sequence of sources, consulted highest fidelity first."""

    def __init__(self, sources: Sequence[CostSource]) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError(
                "a resolver with no sources can only refuse; give it sources"
            )
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, CostSource):
                raise TypeError(f"not a cost source: {source!r}")
            if source.name in seen:
                raise ValueError(
                    f"duplicate source {source.name!r}: a source is read by name"
                )
            seen.add(source.name)
        self._sources = sources

    @property
    def sources(self) -> tuple[CostSource, ...]:
        return self._sources

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._sources)

    def resolve(self, request: Any) -> Resolution:
        """The first source that answers, with every refusal above it attached.

        The answer carries the whole chain, earliest refusal first, and the
        name of the source that produced it. `Resolution.declined` holds every
        refusal this resolver collected, including those an exhausted inner
        resolver handed up; the provenance holds those plus any the answering
        source had already collected inside itself.
        """
        declined: list[Refusal] = []
        for source in self._sources:
            try:
                answer = source.price(request)
            except CostRefused as exhausted:
                # A resolver-backed source that ran out of its own sources. Its
                # refusals are this resolver's refusals -- keyed by the inner
                # sources that actually declined, not flattened onto the wrapper,
                # so the reason counts still name something measurable.
                declined.extend(
                    exhausted.declined
                    or (source.refuse("its own sources refused, naming none"),)
                )
                continue
            if isinstance(answer, Refusal):
                if answer.source != source.name:
                    raise ValueError(
                        f"source {source.name!r} returned a refusal naming "
                        f"{answer.source!r}; a refusal has to name who declined"
                    )
                declined.append(answer)
                continue
            if not isinstance(answer, CostTerm):
                raise TypeError(
                    f"source {source.name!r} returned {answer!r}, not a cost term"
                )
            stamped = CostTerm(
                answer.name,
                answer.seconds,
                answer.provenance.resolved(source.name, declined),
            )
            return Resolution(stamped, source.name, tuple(declined))
        raise CostRefused(request, declined)
