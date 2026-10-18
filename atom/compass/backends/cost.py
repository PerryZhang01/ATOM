# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The cost of one step, and the parts it was built from.

A `StepCost` does not store a total. `seconds` is folded from `terms` on every
read, and the breakdown is re-checked on every read rather than only at
construction, so the whole cannot come to disagree with its parts by any route
that leaves the object itself in place. Subclassing is refused where it would
shadow the total, the terms or the rows, because a subclass needs no bypass at
all to return one number while its breakdown says another.

An empty breakdown is refused outright. The failure behind that is the mean of
an empty sample, which is not an error, is not approximately zero, and is
exactly 0.0 -- a precise and entirely fictional answer that reads as a fast
step rather than as a missing model. The same shape is refused one layer up:
`ProvenanceMix` will not divide by an empty run, because "nothing refused" and
"nothing recorded" would otherwise be the same number, and it is the
reassuring one.

Every addition of seconds in this module goes through `fold_seconds`: a left
fold from 0.0 over values in the order given, with `fold_step` as its one-step
form for running totals. The only `+` outside them counts whole steps and
reasons, which are integers. Float addition is not associative, so two readers
who sum the same terms in different orders get different bits, and a run that
has to be reproducible cannot leave that to whoever writes the next summing
loop. Routing every sum through one function is what makes the order a
property of the module rather than a coincidence that holds until someone
adds a `+`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from atom.compass.backends.provenance import Method, Provenance, Refusal


def fold_seconds(values: Iterable[float]) -> float:
    """Sum in the given order, left fold from 0.0.

    The only summation this package performs, so that a total and a check of
    that total agree bit for bit.
    """
    total = 0.0
    for value in values:
        total += value
    return total


def fold_step(total: float, value: float) -> float:
    """One step of the same fold, for a running total kept across calls."""
    return fold_seconds((total, value))


@dataclass(frozen=True)
class CostTerm:
    """One labelled part of a step's cost.

    Terms are flat. A backend that wants a hierarchy spells it in the name --
    `attention.decode` under `attention` -- rather than nesting, because the
    consumers of a breakdown want to group and sum, not to walk a tree.

    `draft` tags a speculative drafter's device work, which runs after the
    target's; every other term is the target's.
    """

    name: str
    seconds: float
    provenance: Provenance
    draft: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a cost term must be named")
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"{self.name}: provenance is required, got {self.provenance!r}"
            )
        if not math.isfinite(self.seconds):
            raise ValueError(f"{self.name}: {self.seconds} is not a finite duration")
        if self.seconds < 0.0:
            raise ValueError(f"{self.name}: {self.seconds} s is negative")
        if self.provenance.is_refused and self.seconds <= 0.0:
            raise ValueError(
                f"{self.name}: a refusal was priced at {self.seconds} s. Something "
                "declined and the stand-in charged nothing, which removes the step "
                "from the schedule while still reporting it as priced."
            )

    @property
    def refusals(self) -> tuple[Refusal, ...]:
        return self.provenance.refusals


def _checked(terms: Sequence[CostTerm]) -> tuple[CostTerm, ...]:
    """The breakdown, or the reason it is not one. Run at build and at read."""
    terms = tuple(terms)
    if not terms:
        raise ValueError(
            "a step cost with no terms is an aggregate with no decomposition; "
            "name at least one part, even if the model has only one"
        )
    seen: set[str] = set()
    for term in terms:
        if not isinstance(term, CostTerm):
            raise TypeError(f"not a cost term: {term!r}")
        if term.name in seen:
            raise ValueError(
                f"duplicate term {term.name!r}: a breakdown is read by name"
            )
        seen.add(term.name)
    return terms


@dataclass(frozen=True)
class StepCost:
    """What one step costs, and what that cost is made of.

    Construct it from its parts. There is no constructor that takes a total,
    because a total that can be set independently of the parts is a total that
    will eventually disagree with them.

    `treatments` names each assumed treatment the step was priced under: a
    factor that moves cost and that no feature of the query carries. It
    belongs to the query, so no term's method changes with it.
    """

    terms: tuple[CostTerm, ...]
    treatments: tuple[str, ...]

    def __init__(
        self, terms: Sequence[CostTerm], treatments: Sequence[str] = ()
    ) -> None:
        object.__setattr__(self, "terms", _checked(terms))
        object.__setattr__(self, "treatments", tuple(treatments))

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass that shadows anything this class reports.

        Overriding a reader needs no bypass and defeats every check here: the
        object is valid, the breakdown is real, and the answer returned has
        nothing to do with it. `seconds` is the obvious one; `is_refused` is
        the dangerous one, because hiding it makes a run report no refused
        steps over steps that refused, at the layer that reads that fraction
        to decide whether the run is evidence.

        The protected set is read off this class rather than listed, so a
        reader added later is covered the day it is added and not the day
        somebody remembers to extend a tuple.
        """
        super().__init_subclass__(**kwargs)
        owned = {
            name
            for name in (*vars(StepCost), *StepCost.__annotations__)
            if not name.startswith("_")
        }
        shadowed = sorted(owned & set(cls.__dict__))
        if shadowed:
            raise TypeError(
                f"{cls.__name__} redefines {', '.join(shadowed)}; an answer that does "
                "not come from the terms is the thing this class exists to prevent"
            )

    @property
    def seconds(self) -> float:
        """The total, folded from the terms in their stored order."""
        return fold_seconds(term.seconds for term in _checked(self.terms))

    @property
    def target_seconds(self) -> float:
        """The fold of the terms not tagged `draft`."""
        return fold_seconds(t.seconds for t in _checked(self.terms) if not t.draft)

    @property
    def draft_seconds(self) -> float:
        """The fold of the terms tagged `draft`."""
        return fold_seconds(t.seconds for t in _checked(self.terms) if t.draft)

    @property
    def refusals(self) -> tuple[Refusal, ...]:
        """Every refusal behind this step, term order then chain order."""
        return tuple(r for term in self.terms for r in term.refusals)

    @property
    def is_refused(self) -> bool:
        return any(t.provenance.is_refused for t in self.terms)

    @property
    def refused_seconds(self) -> float:
        """Seconds priced by a stand-in after something declined.

        Reported alongside the count because they answer different questions:
        a run can refuse 2% of its steps and 40% of its time.
        """
        return fold_seconds(t.seconds for t in self.terms if t.provenance.is_refused)

    def seconds_by_method(self) -> Mapping[tuple[Method, bool], float]:
        """Seconds grouped by (method, validated), terms in stored order.

        Each group is the fold of its own terms, in the order they appear in
        the breakdown. Grouping re-associates, so folding these group totals
        is a *different* sum from `seconds` and is not bit-equal to it: three
        terms of 0.1, 0.2 and 0.15 whose first and third share a group fold
        flat to 0.45000000000000007 and grouped to 0.45. The mixture is for
        reporting how the time was obtained. `rows()` is the view that
        re-folds to the total, and is what a reader checks a total against.
        """
        grouped: dict[tuple[Method, bool], list[float]] = {}
        for term in self.terms:
            key = (term.provenance.method, term.provenance.validated)
            grouped.setdefault(key, []).append(term.seconds)
        return {key: fold_seconds(vals) for key, vals in grouped.items()}

    def rows(self) -> tuple[tuple[str, float, str], ...]:
        """The breakdown as an artifact writes it: name, seconds, provenance."""
        return tuple(
            (t.name, t.seconds, str(t.provenance)) for t in _checked(self.terms)
        )


#: How a run summary groups seconds: a method each, estimated seconds split
#: into validated (`estimated`) and not (`unvalidated`).
GROUPS = ("analytical", "declared", "measured", "estimated", "unvalidated")


def group(method: Method, validated: bool) -> str:
    """The `GROUPS` entry of a (method, validated) key."""
    unvalidated = method is Method.ESTIMATED and not validated
    return "unvalidated" if unvalidated else method.value


class ProvenanceMix:
    """The mixture a run reports, accumulated one step at a time.

    Refusals are counted the three ways they are read: how many, what fraction
    of steps, and what fraction of predicted seconds. The seconds fraction is
    the one that decides whether a run is worth anything, and the reason counts
    are what say which measurement would close the gap.

    A mix with nothing in it has no fractions. Returning 0.0 would make a run
    that recorded no steps indistinguishable from one that refused none, at
    exactly the layer that reads the fraction to decide whether a run counts as
    evidence -- and of the two readings the wrong one is the reassuring one.
    Ask `is_empty` first, or let the refusal say so.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.seconds = 0.0
        self.refused_steps = 0
        self.refused_seconds = 0.0
        #: Steps with a term its source extrapolated past its support domain.
        self.extrapolated_steps = 0
        self._method_seconds: dict[tuple[Method, bool], float] = {}
        self._method_steps: dict[tuple[Method, bool], int] = {}
        self._reasons: dict[tuple[str, str], int] = {}
        #: Steps priced under each assumed treatment, by its name.
        self.treatments: Counter[str] = Counter()

    def record(self, step: StepCost) -> None:
        self.steps += 1
        self.treatments.update(step.treatments)
        self.seconds = fold_step(self.seconds, step.seconds)
        self.extrapolated_steps += any(t.provenance.extrapolated for t in step.terms)
        if step.is_refused:
            self.refused_steps += 1
            self.refused_seconds = fold_step(self.refused_seconds, step.refused_seconds)
        for key, seconds in step.seconds_by_method().items():
            self._method_seconds[key] = fold_step(
                self._method_seconds.get(key, 0.0), seconds
            )
            self._method_steps[key] = self._method_steps.get(key, 0) + 1
        for refusal in step.refusals:
            key = (refusal.source, refusal.reason)
            self._reasons[key] = self._reasons.get(key, 0) + 1

    @property
    def is_empty(self) -> bool:
        return self.steps == 0

    @property
    def refused_step_fraction(self) -> float:
        if self.steps == 0:
            raise ValueError(
                "no steps were recorded, so there is no refused fraction of them; "
                "0.0 here would read as a run that refused nothing"
            )
        return self.refused_steps / self.steps

    @property
    def refused_second_fraction(self) -> float:
        if self.seconds <= 0.0:
            raise ValueError(
                f"no predicted seconds were recorded over {self.steps} step(s), so "
                "there is no refused fraction of them; 0.0 here would read as a run "
                "that refused nothing"
            )
        return self.refused_seconds / self.seconds

    def seconds_by_method(self) -> Mapping[tuple[Method, bool], float]:
        """Seconds by (method, validated)."""
        return dict(self._method_seconds)

    def steps_by_method(self) -> Mapping[tuple[Method, bool], int]:
        """Steps with at least one term of each (method, validated)."""
        return dict(self._method_steps)

    def reasons(self) -> Mapping[tuple[str, str], int]:
        """Distinct (source, reason) pairs with the number of terms each hit."""
        return dict(self._reasons)
