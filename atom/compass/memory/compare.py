# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A predicted non-KV footprint held to its gate one term at a time.

The rule this module implements is **validate per term, never as a sum**, and
the incident behind it is why it has no call that returns a total. A summed
non-KV check once read **+13.8%** while holding three errors, two of which
cancelled: weights over by **+0.280 GB** (a tied `lm_head` that is never
resident), activations compared at the wrong shape (**-0.015 GB**), and
**-0.084 GB** of a resident term nobody had noticed existed. The largest of the
three was **25% of its own term**, and the sum said 13.8%.

So this module compares term by term, and the aggregate is available only as a
`SummedCheck`, which cannot be constructed without the comparison it summarises
and renders the per-term table underneath itself. That is the rule against
reporting an aggregate without its decomposition, made structural rather than
remembered, the same way `terms.Reading` makes it structural for a reading.

**It costs no GPU time.** Every hardware run already prints its own breakdown;
a `Recorded` is that printout, handed in as input data. Nothing here runs a
model, imports the engine or touches a device, and a comparator that needed one
would be a comparator nobody could run against the card they are sizing for.

## What a term can come back as

| outcome | when |
|---|---|
| `Verdict.PASS` | both sides present, within the gate, and the predicted side carries a basis that can discharge one -- see below |
| `Verdict.FAIL` | both sides present and outside the gate -- **including** when the predicted side is a declared coefficient, because a recording that contradicts a coefficient is evidence |
| `Verdict.NOT_DISCHARGED` | both sides present and within the gate, but the predicted side carries a basis that cannot discharge one. The error is reported and the gate is not called discharged |
| a `TermRefusal` | the two sides cannot be compared at all: the run records no such term, the run records it as zero bytes, the shapes disagree, or the recorded peak is the warmup prefill's |

**Which bases discharge a gate.** It is a stated list, not the single word
"obtained", because the bases divide three ways and only one of the three is
about a number having been read off the thing it describes.

- `Basis.SPEC` **discharges.** A machine specification's runtime constants are
  filled by probe runs on a card, so a spec number is a measurement taken
  elsewhere rather than a coefficient somebody wrote down, and holding it
  against this run is a real check. What a pass does *not* say is that a law
  was validated: two of the constants this package reads -- the driver and
  collective reserve, and the load residue -- are width tables exactly because
  no closed form fits them, so a pass means the table's entry for this width
  agrees with this run and means nothing at any other width.
- `Basis.DERIVED` **discharges**, being arithmetic over readings that are
  themselves one of these.
- `Basis.OBTAINED` **discharges**: the number was read off the thing itself.
- `Basis.DECLARED` **does not.** It is a coefficient with a named successor,
  and agreement with one run does not turn a coefficient into a measurement.
- `Basis.DEPLOYMENT` **does not.** It is a knob the serving config states, and
  a knob agreeing with a run is not evidence about bytes in either direction.
  No footprint term carries it today; the only one in this package labels the
  eager-mode branch of the graph-pool estimator, which is not a footprint term.

The asymmetry between the second and third rows is deliberate: a declared term
can **fail** its gate but cannot **pass** it. Two of the three model terms have
a successor that makes them exact -- weights from a meta build, buffers from a
recording off a card -- and the third has none yet: a traced graph without the
invisible-scratch table is not enough to discharge the 10% gate on the
activation term.

## The traps this module refuses rather than papers over

- **Shape.** Both sides state the shape they were taken at and a term taken at
  a shape refuses against a side taken at another. The -0.015 GB above is an
  activation term compared at the wrong shape, and a comparison that does not
  say what shape it is at cannot notice.
- **The allocator's high-water mark.** A tracing run must reset it around the
  step it writes a graph for, or the peak belongs to the warmup prefill, whose
  shape is nobody's choice -- **-14.7%** on one configuration. `Recorded` takes
  that as a field with no default, and a recording that did not reset it
  refuses every term it took at a shape.
- **A term the run does not record.** Not a term of zero bytes. The refusal
  names it and, when the predicted side is declared, carries that term's own
  note -- which for buffers is the reason for recording them rather than
  computing them.
- **A term the run records as zero bytes.** An error relative to zero has no
  value, and the gate's unit is relative. So it refuses in the comparison
  rather than dividing, which also keeps it out of the renderer: one term that
  cannot state a relative error must not take the table down for the terms that
  can, in the module whose product *is* the table.

## The graph pool is two numbers and stays two

`graph_pool.reserves()` and `graph_pool.predicts()` disagree by 4-19x and only
the first reserves anything. `compare_graph_pool` reports **both** against the
recorded pool and labels which is which; there is no call here that returns one
of them, and `compare` refuses a footprint term carrying either name so the
pool cannot be folded into the per-term table as a single row.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from atom.compass.memory import graph_pool
from atom.compass.memory.readings import DeviceReadings, MemoryRefusal
from atom.compass.memory.terms import Basis, Reading, Term

#: The acceptance gate on each non-KV term, as a fraction of the recorded term.
NON_KV_TERM_GATE = 0.10

#: The bases that can discharge a gate. Stated as a set rather than as "not
#: DECLARED" so that adding a member to `Basis` is a decision about this list
#: rather than a silent grant. See the module docstring for why each is here.
DISCHARGES = frozenset({Basis.SPEC, Basis.DERIVED, Basis.OBTAINED})


class Verdict(enum.Enum):
    """What a comparison of one term concluded. See the module docstring."""

    PASS = "pass"
    FAIL = "fail"
    NOT_DISCHARGED = "not discharged"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Shape:
    """The shape a footprint was taken at, which both sides have to state.

    Two sides that disagree here are measuring different things, and the
    -0.015 GB of the incident this module exists for is what that looks like
    when nobody says so. Equality is the whole point of the type.

    **What it deliberately does not carry, so that the absence is a decision
    rather than a hole.** There is no batch dimension: the footprint terms this
    package predicts are linear in tokens and none of them is a function of how
    those tokens are split across sequences, so a batch field would be a
    distinction nothing here can act on. The consequence is real and worth
    stating -- two decode breakdowns at 256 tokens compare equal whatever the
    batch, so a term that *does* move with batch would be compared across two
    of them without a refusal. The first such term is where the field gets
    added, and adding it is a new field rather than a change of shape.

    `phase` is a free string rather than an enumeration for the same reason in
    the other direction: this package has no list of phases to close over, and
    an enumeration invented here would refuse a phase a caller has and this
    module has not heard of. Equality is exact, so two spellings of one phase
    refuse each other, which is the safe direction.
    """

    tokens: int
    phase: str

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise ValueError(f"a shape is at least one token: {self.tokens}")
        if not self.phase.strip():
            raise ValueError("a shape names its phase; prefill and decode differ")

    def __str__(self) -> str:
        return f"{self.tokens} tokens, {self.phase}"


def _named(terms: Iterable[Term], where: str) -> dict[str, Term]:
    by_name: dict[str, Term] = {}
    for term in terms:
        if not isinstance(term, Term):
            raise TypeError(f"{where}: not a term: {term!r}")
        if term.name in by_name:
            raise ValueError(f"{where} carries two terms called {term.name!r}")
        by_name[term.name] = term
    return by_name


@dataclass(frozen=True, slots=True)
class Recorded:
    """The breakdown one hardware run printed. Input data; nothing here makes one.

    `run` identifies the run, because a recorded number whose run nobody can
    name is a number without a source. `high_water_reset` has no default on
    purpose: it is the difference between a peak that belongs to the step this
    breakdown is about and one that belongs to the warmup prefill, and a
    default would answer that question for a reader who never asked it.

    `at_shape` names the terms that were taken at `shape`. It is stated rather
    than inferred: which terms move with the shape is a property of how the
    breakdown was produced, and the one thing this module must not do is guess
    it from a term's name.
    """

    run: str
    shape: Shape
    high_water_reset: bool
    terms: tuple[Term, ...]
    at_shape: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.run.strip():
            raise ValueError("a recording names its run, or its numbers have no source")
        by_name = _named(self.terms, f"recording {self.run!r}")
        if not by_name:
            raise ValueError(f"recording {self.run!r} records nothing")
        for term in self.terms:
            if term.basis is not Basis.OBTAINED:
                raise ValueError(
                    f"{self.run}.{term.name} is {term.basis}; every term of a "
                    "recording was read off the card, which is Basis.OBTAINED"
                )
        unknown = self.at_shape - by_name.keys()
        if unknown:
            raise ValueError(
                f"recording {self.run!r} says {sorted(unknown)} were taken at "
                f"{self.shape} and records no such term"
            )

    def by_name(self) -> dict[str, Term]:
        return _named(self.terms, f"recording {self.run!r}")


@dataclass(frozen=True, slots=True)
class Predicted:
    """The non-KV terms Compass predicts, and the shape they were taken at."""

    label: str
    shape: Shape
    terms: tuple[Term, ...]
    at_shape: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("a prediction says what produced it")
        by_name = _named(self.terms, f"prediction {self.label!r}")
        if not by_name:
            raise ValueError(f"prediction {self.label!r} predicts nothing")
        unknown = self.at_shape - by_name.keys()
        if unknown:
            raise ValueError(
                f"prediction {self.label!r} says {sorted(unknown)} were taken "
                f"at {self.shape} and predicts no such term"
            )

    @classmethod
    def from_readings(
        cls, readings: DeviceReadings, *, shape: Shape, at_shape: frozenset[str]
    ) -> Predicted:
        """The footprint terms of a `DeviceReadings`: `peak_torch` and `non_torch`.

        Three of the five readings are deliberately not here. `total` is the
        card and `free` is the box that is left after the footprint, so neither
        is a footprint term; folding either in would compare the card against
        itself. `cudagraph_overhead` is one of **two** numbers that disagree by
        4-19x, and a single row for it would pick one -- `compare_graph_pool`
        takes both, and `compare` refuses a term carrying either name.

        `at_shape` is required rather than defaulted to this package's own
        shape-dependent term. A caller that adds a term to `ModelTerms` has to
        say whether it moves with the shape, and a default here would answer
        for them.
        """
        return cls(
            label=f"{readings.spec_digest} at TP{readings.tp_width}",
            shape=shape,
            terms=tuple(readings.peak_torch.terms) + tuple(readings.non_torch.terms),
            at_shape=at_shape,
        )

    def by_name(self) -> dict[str, Term]:
        return _named(self.terms, f"prediction {self.label!r}")


@dataclass(frozen=True, slots=True)
class TermRefusal:
    """One term that could not be compared, and why. A declined term is a result."""

    name: str
    what: str
    remedy: str

    def __str__(self) -> str:
        return f"{self.name}: {self.what}. {self.remedy}"


@dataclass(frozen=True, slots=True)
class TermComparison:
    """One term's predicted bytes against one run's, and the gate it carries.

    `predicted` is `None` for a term the run records and nothing predicted --
    the -0.084 GB of the incident this module exists for, a resident term
    nobody had noticed existed. It is reported as the whole of itself rather
    than passed over,
    because a term that is absent from a prediction is absent from its sum too
    and that is exactly what makes a sum unable to see it.
    """

    name: str
    predicted: Term | None
    recorded: Term
    gate: float
    verdict: Verdict
    why: str = ""

    @property
    def predicted_bytes(self) -> int:
        return 0 if self.predicted is None else self.predicted.nbytes

    @property
    def delta_bytes(self) -> int:
        return self.predicted_bytes - self.recorded.nbytes

    @property
    def relative(self) -> float:
        """The error as a fraction of the recorded term, which is the gate's unit."""
        if self.recorded.nbytes == 0:
            raise MemoryRefusal(
                f"{self.name} was recorded as zero bytes, so an error relative "
                "to it has no value",
                "compare this term in bytes, or record the run that allocated it",
            )
        return self.delta_bytes / self.recorded.nbytes

    def row(self) -> tuple[str, ...]:
        predicted = "-" if self.predicted is None else f"{self.predicted_bytes:,}"
        return (
            self.name,
            predicted,
            f"{self.recorded.nbytes:,}",
            f"{self.delta_bytes:+,}",
            f"{self.relative:+.2%}",
            str(self.verdict),
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Every non-KV term of one prediction against one run, and nothing summed.

    There is no `total` here and no `__int__`. The aggregate is `summed`, which
    returns an object that cannot be built without this one and prints this
    table underneath itself: the rule is that an overall figure may be
    emitted only beside the decomposition that produced it, and a call that
    could return the figure alone is the defect rather than the convenience.
    """

    label: str
    run: str
    predicted_shape: Shape
    recorded_shape: Shape
    gate: float
    compared: tuple[TermComparison, ...]
    refused: tuple[TermRefusal, ...]

    def failures(self) -> tuple[TermComparison, ...]:
        return tuple(t for t in self.compared if t.verdict is Verdict.FAIL)

    def not_discharged(self) -> tuple[TermComparison, ...]:
        return tuple(t for t in self.compared if t.verdict is Verdict.NOT_DISCHARGED)

    def _largest(self, key) -> TermComparison:
        if not self.compared:
            raise MemoryRefusal(
                f"{self.run} and {self.label} share no comparable term, so "
                "there is no largest error",
                "check the refusals: every term of this comparison is in them",
            )
        return max(self.compared, key=key)

    def worst(self) -> TermComparison:
        """The term furthest outside its gate, ranked in the gate's own unit.

        The gate is a fraction of the recorded term, so that is what "worst"
        is ranked by. The two orderings genuinely disagree -- a term the run
        records and nothing predicted is 100% of itself and is usually not the
        largest number of bytes -- and handing a reader the second-worst term
        by the unit they are being held to is the wrong end of the finding.
        `largest_by_bytes` is the other ordering, named for what it is.
        """
        return self._largest(lambda t: abs(t.relative))

    def largest_by_bytes(self) -> TermComparison:
        """The term whose error is the most bytes, which is not the same term.

        This is the ordering the historical incident is stated in: the largest
        single error was 25% of its term. That is a claim about bytes, and it
        is history rather than a contract, so it is a separate accessor with
        the unit in its name.
        """
        return self._largest(lambda t: abs(t.delta_bytes))

    def table(self) -> str:
        """The per-term table, then every refusal by name. Never a total."""
        header = ("term", "predicted", "recorded", "delta", "of term", "verdict")
        rows = [header] + [t.row() for t in self.compared]
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        shapes = (
            f"  predicted at {self.predicted_shape}; recorded at "
            f"{self.recorded_shape}; gate {self.gate:.0%} per term"
        )
        lines = [f"{self.label} against run {self.run}", shapes]
        for row in rows:
            lines.append(
                "  "
                + "  ".join(
                    cell.rjust(widths[i]) if i else cell.ljust(widths[i])
                    for i, cell in enumerate(row)
                ).rstrip()
            )
        for comparison in self.compared:
            if comparison.why:
                lines.append(f"  * {comparison.name}: {comparison.why}")
        for refusal in self.refused:
            lines.append(f"  ! {refusal}")
        return "\n".join(lines)

    def summed(self, *, band: float) -> SummedCheck:
        """The instrument the per-term rule rejects, kept so it can be shown wrong.

        `band` has no default because no band for a sum exists. Both bands in
        use -- the 10% a non-KV memory term carries, and the 25% allowed one on
        a device nobody has measured -- are **per term**, and
        choosing one of those two for a sum is the substitution that produced
        the +13.8%. Naming it at the call site is the moment a caller has to
        notice that.
        """
        return SummedCheck(comparison=self, band=band)

    def __str__(self) -> str:
        return self.table()


@dataclass(frozen=True, slots=True)
class SummedCheck:
    """One aggregate, and the table it came from, which it cannot be printed without.

    This is the instrument that read +13.8% over three errors, two of which
    cancelled. It is here so that the two can be run on one breakdown and the
    difference shown, which is a stronger argument for the rule than restating
    the rule. It holds its `Comparison` rather than a pair of totals, so there
    is no way to obtain the figure without also holding the decomposition.
    """

    comparison: Comparison
    band: float

    def __post_init__(self) -> None:
        if not 0 < self.band < 10:
            raise ValueError(f"a band is a fraction of the recorded sum: {self.band}")

    @property
    def folded(self) -> tuple[TermComparison, ...]:
        return self.comparison.compared

    @property
    def predicted_total(self) -> int:
        return sum(t.predicted_bytes for t in self.folded)

    @property
    def recorded_total(self) -> int:
        return sum(t.recorded.nbytes for t in self.folded)

    @property
    def delta_bytes(self) -> int:
        return self.predicted_total - self.recorded_total

    @property
    def relative(self) -> float:
        if self.recorded_total == 0:
            raise MemoryRefusal(
                "the recorded terms sum to zero bytes",
                "a relative error against nothing is not a figure; read the table",
            )
        return self.delta_bytes / self.recorded_total

    @property
    def passed(self) -> bool:
        return abs(self.relative) <= self.band

    def table(self) -> str:
        """The aggregate, and underneath it the terms it folded."""
        verdict = "passes" if self.passed else "fails"
        refused = len(self.comparison.refused)
        aggregate = (
            f"summed check: {self.relative:+.1%} against a {self.band:.0%} "
            f"band -- {verdict}"
        )
        folded = (
            f"  folded {len(self.folded)} terms, {refused} refused and so not "
            "folded; a sum names none of them"
        )
        return f"{aggregate}\n{folded}\n{self.comparison.table()}"

    def __str__(self) -> str:
        return self.table()


def _refuse_shape(name: str, predicted: Shape, recorded: Shape) -> TermRefusal:
    return TermRefusal(
        name,
        f"this term was taken at a shape on each side and the two disagree -- "
        f"predicted at {predicted}, recorded at {recorded}",
        "take the two sides at one shape. A -0.015 GB activation error in the "
        "incident this comparison exists for is exactly this, and a difference "
        "in shape looks like a difference in the model until somebody states "
        "both",
    )


def _refuse_high_water(name: str, run: str) -> TermRefusal:
    return TermRefusal(
        name,
        f"run {run} did not reset the allocator's high-water mark around the "
        "step this breakdown is about, so its peak belongs to whatever ran "
        "before it -- on a tracing run, the warmup prefill",
        "reset the peak around the step the breakdown is written for and "
        "record again; the warmup prefill's shape is nobody's choice and "
        "taking its peak for the step's was worth -14.7% on one configuration",
    )


def _refuse_unrecorded(name: str, run: str, term: Term) -> TermRefusal:
    note = f" The predicted term says: {term.note}" if term.note else ""
    return TermRefusal(
        name,
        f"run {run} records no {name!r}, so there is nothing to compare it "
        "against; a term absent from a breakdown is not a term of zero bytes",
        f"record this term separately in the run, or say that the run's own "
        f"split cannot isolate it.{note}",
    )


def _refuse_zero(name: str, run: str) -> TermRefusal:
    return TermRefusal(
        name,
        f"run {run} records {name!r} as zero bytes, and the gate this term "
        "carries is a fraction of the recorded term, so there is no error to "
        "state",
        "compare this term in bytes, or record the run that allocated it. It "
        "refuses here rather than in the renderer, so that one term with no "
        "relative error does not take the table down for the terms that have "
        "one",
    )


def _verdict(predicted: Term, relative: float, gate: float) -> tuple[Verdict, str]:
    if abs(relative) > gate:
        return Verdict.FAIL, ""
    if predicted.basis in DISCHARGES:
        return Verdict.PASS, ""
    if predicted.basis is Basis.DECLARED:
        return Verdict.NOT_DISCHARGED, (
            f"within the gate, and the gate is not discharged: the predicted "
            f"side is a declared coefficient over {predicted.source}. "
            f"{predicted.note}"
        )
    return Verdict.NOT_DISCHARGED, (
        f"within the gate, and the gate is not discharged: the predicted side "
        f"is {predicted.basis}, over {predicted.source}, and a knob the serving "
        "config states agreeing with a run is not evidence about bytes"
    )


def compare(
    predicted: Predicted,
    recorded: Recorded,
    *,
    gate: float = NON_KV_TERM_GATE,
) -> Comparison:
    """Every term of a predicted footprint against one run's, one term at a time.

    A term appears in exactly one of two places. It is **compared** when both
    sides carry it and the pair is comparable, and it is **refused** by name
    when it is not -- an unrecorded term, a shape the two sides disagree on, or
    a recorded peak that belongs to the warmup prefill. There is no third
    place, and in particular there is no silent omission: the recorded-only
    terms are compared against a predicted zero, because a resident term nobody
    predicted is the error that a sum cannot see.
    """
    if not 0 < gate < 1:
        raise ValueError(f"a gate is a fraction of a term: {gate}")
    for side in (predicted.by_name(), recorded.by_name()):
        for name in (graph_pool.RESERVES, graph_pool.PREDICTS):
            if name in side:
                raise MemoryRefusal(
                    f"{name!r} was handed in as a footprint term, and the "
                    "graph pool is two numbers that disagree by 4-19x",
                    "pass both to `compare_graph_pool`, which reports each "
                    "against the recorded pool and labels which one reserves; "
                    "one row here would pick one and lose the finding",
                )
    at_shape = predicted.at_shape | recorded.at_shape
    if predicted.shape != recorded.shape and not at_shape:
        raise MemoryRefusal(
            f"the two sides were taken at different shapes -- predicted at "
            f"{predicted.shape}, recorded at {recorded.shape} -- and neither "
            "names a term it took at one",
            "state `at_shape` on the side whose terms move with the shape. A "
            "comparison that cannot say which terms move with the shape cannot "
            "say whether the two sides measured the same thing",
        )
    if not recorded.high_water_reset and not at_shape:
        raise MemoryRefusal(
            f"run {recorded.run} says its peak was not reset around the step "
            "this breakdown is about, and neither side names a term it took "
            "at a shape",
            "state `at_shape` on the side whose terms carry a peak. A "
            "recording that says the peak is wrong and then declines to say "
            "which terms it is wrong for leaves nothing for the guard to "
            "refuse, which is how a term gets compared against the warmup "
            "prefill's peak without anybody noticing",
        )
    shapes_agree = predicted.shape == recorded.shape
    predicted_terms = predicted.by_name()
    recorded_terms = recorded.by_name()
    compared: list[TermComparison] = []
    refused: list[TermRefusal] = []
    for name, term in predicted_terms.items():
        if name not in recorded_terms:
            refused.append(_refuse_unrecorded(name, recorded.run, term))
    for name, recording in recorded_terms.items():
        # Both shape guards read the same union, so the two agree on what
        # "moves with the shape" means. Reading only the recording's own set
        # here made this guard silent whenever the recording named none, which
        # is the field's default.
        if name in at_shape and not recorded.high_water_reset:
            refused.append(_refuse_high_water(name, recorded.run))
            continue
        if name in at_shape and not shapes_agree:
            refused.append(_refuse_shape(name, predicted.shape, recorded.shape))
            continue
        if recording.nbytes == 0:
            refused.append(_refuse_zero(name, recorded.run))
            continue
        term = predicted_terms.get(name)
        if term is None:
            compared.append(
                TermComparison(
                    name=name,
                    predicted=None,
                    recorded=recording,
                    gate=gate,
                    verdict=Verdict.FAIL,
                    why=(
                        "the run records this term and nothing predicted it, so "
                        "the whole of it is missing from the prediction and from "
                        "any sum over it"
                    ),
                )
            )
            continue
        relative = (term.nbytes - recording.nbytes) / recording.nbytes
        verdict, why = _verdict(term, relative, gate)
        compared.append(
            TermComparison(
                name=name,
                predicted=term,
                recorded=recording,
                gate=gate,
                verdict=verdict,
                why=why,
            )
        )
    return Comparison(
        label=predicted.label,
        run=recorded.run,
        predicted_shape=predicted.shape,
        recorded_shape=recorded.shape,
        gate=gate,
        compared=tuple(compared),
        refused=tuple(refused),
    )


@dataclass(frozen=True, slots=True)
class GraphPoolComparison:
    """Both graph-pool numbers against the recorded pool, labelled by which reserves.

    ATOM's estimator and the measured predictor stay two numbers because they
    disagree by 4-19x, and the disagreement is the finding. So this type has
    no accessor for *the* error: it has one for each, and its table has a row
    for each, labelled with what that function does. A comparator that
    reported one of them would reconcile the two numbers `graph_pool` keeps
    apart.
    """

    recorded: Term
    reserves: Reading
    predicts: Reading

    def __post_init__(self) -> None:
        if self.recorded.basis is not Basis.OBTAINED:
            raise ValueError(
                f"the recorded pool is {self.recorded.basis}; a pool to compare "
                "against was read off a card, which is Basis.OBTAINED"
            )
        if self.reserves.name != graph_pool.RESERVES:
            raise MemoryRefusal(
                f"the reserving reading given is {self.reserves.name!r}",
                f"pass `graph_pool.reserves(...)`, whose reading is named "
                f"{graph_pool.RESERVES!r}; it is the number ATOM subtracts "
                "from the budget, so it is the one that reserves",
            )
        if self.predicts.name != graph_pool.PREDICTS:
            raise MemoryRefusal(
                f"the predicting reading given is {self.predicts.name!r}",
                f"pass `graph_pool.predicts(...)`, whose reading is named "
                f"{graph_pool.PREDICTS!r}; it reserves nothing and says what "
                "the pool really costs",
            )

    @property
    def reserves_relative(self) -> float:
        """What the number that reserves is off the recorded pool by."""
        return (self.reserves.total - self.recorded.nbytes) / self.recorded.nbytes

    @property
    def predicts_relative(self) -> float:
        """What the number that predicts is off the recorded pool by."""
        return (self.predicts.total - self.recorded.nbytes) / self.recorded.nbytes

    @property
    def disagreement(self) -> float:
        """How many times the reserving number is the predicting one."""
        if self.predicts.total == 0:
            raise MemoryRefusal(
                "the predicted pool is zero bytes, so the two functions have "
                "no ratio",
                "read the two rows of the table, which are both still there",
            )
        return self.reserves.total / self.predicts.total

    def table(self) -> str:
        rows = [
            ("function", "role", "bytes", "vs recorded"),
            (
                "reserves()",
                "reserves the memory",
                f"{self.reserves.total:,}",
                f"{self.reserves_relative:+.2%}",
            ),
            (
                "predicts()",
                "reserves nothing",
                f"{self.predicts.total:,}",
                f"{self.predicts_relative:+.2%}",
            ),
            (
                "recorded",
                self.recorded.source,
                f"{self.recorded.nbytes:,}",
                "",
            ),
        ]
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        lines = [f"graph pool against {self.recorded.name}"]
        for row in rows:
            lines.append(
                "  "
                + "  ".join(
                    cell.rjust(widths[i]) if i > 1 else cell.ljust(widths[i])
                    for i, cell in enumerate(row)
                ).rstrip()
            )
        lines.append(
            f"  reserves() is {self.disagreement:.4f}x predicts(); only the "
            "first is subtracted from the budget"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def compare_graph_pool(
    recorded: Term, *, reserves: Reading, predicts: Reading
) -> GraphPoolComparison:
    """Both graph-pool numbers against one recorded pool. Neither is picked."""
    return GraphPoolComparison(recorded=recorded, reserves=reserves, predicts=predicts)


def footprint_terms(mapping: Mapping[str, int], *, source: str) -> tuple[Term, ...]:
    """Named byte counts from a run's printout, as recorded terms.

    A convenience for the one shape a recording arrives in -- a printed
    breakdown is names and byte counts -- and it does the one thing that must
    not be got wrong, which is to label every one of them `Basis.OBTAINED` and
    to make each name its run.

    **It converts nothing.** `Term` rejects a non-integer byte count and says
    why -- round at the call site, so the rounding is visible where it happens
    -- and a helper that called `int()` here would put the rounding in the one
    place a reader is not looking, in the function whose input is a hardware
    run's printout. That input is exactly where it bites: the machine
    specification this package already reads writes its own byte counts as
    floats, so a breakdown arriving with `1.1e6` in it is the expected shape
    rather than a contrived one, and truncating it would be silent.
    """
    return tuple(
        Term(name, nbytes, Basis.OBTAINED, source) for name, nbytes in mapping.items()
    )
