# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A stand-in cost model: declared coefficients multiplied into a step's shapes.

**What this produces is not an estimate of anything.** The coefficients were
chosen, not measured and not regressed, and multiplying one by a real shape
does not make the product a prediction of a real duration. Nothing from this
backend may be quoted as accuracy. That sentence is not left in this file for
a reader who happens to open it: the provenance on every term carries it and
`describe()` repeats it, so whoever reads the record an artifact writes is
told without reading any source.

What it is for is the loop. A fixed pair of constants prices every step the
same, so nothing the scheduler does can depend on the price and no run can
show whether it did. That was measured rather than argued: an offline workload
presents one shape per kind, so an answer per kind looked exact -- and the same
answer cost +47.9% time-to-first-token the moment shapes varied under real
serving, three prefill steps of 16, 256 and 16128 tokens costing 38.6, 151.0
and 177.6 ms and all three answered as about 47 ms. A form that reads the
shapes makes a longer chunk cost longer, a longer step is when the next
request arrives, and the next batch is a different batch. That loop is the
thing this exists to make testable; accuracy is somebody else's milestone.

The form, declared rather than derived:

    prefill:  a  + b .tokens   + c .Sum N_Q^2 + d .Sum N_Q.(N_KV - N_Q)
    decode:   a' + b'.requests + c'.Sum N_KV  + e'.(rung.max N_KV - Sum N_KV)

plus one term per collective the deployment's widths make possible. A step
carrying both kinds of work is priced as both and pays both intercepts, because
a step that did both launched both.

A tensor-parallel all-reduce runs once per layer this worker holds, so its
count is the depth of the worker's span and not the number of layers in it that
hold a cache of every past token. The two are the same number only on a stack
whose layers are all paged, and they diverge with the architecture rather than
by a factor somebody could fold into the coefficient: a layer keeping a bounded
recurrent state costs a KV block nothing and still takes part in the
all-reduce, so a stack that is one paged layer in four charges four times what
the paged count would. That depth is handed in -- the KV geometry cannot supply
it, since what it counts is the paged subset -- and a deployment with a
collective to price and no depth stated is refused rather than charged on the
count that happens to be at hand.

**That argument is the all-reduce's and it does not reach the other collective
this charges.** An expert all-to-all runs on the layers that hold experts,
which is a third count: not the paged subset, not the depth of the span, and
not something anything here can supply, since no expert arrangement is read
anywhere in this package. It is charged on the stack depth regardless, because
that is the count in hand -- so on a stack whose experts sit on a period, or
one mixing dense layers among expert ones, it is wrong in exactly the way the
paged count was wrong for the all-reduce. Two consequences are stated rather
than left to be noticed. The term for any collective other than the all-reduce
says, where its charge is read, that the depth stands in for this collective's
own count, which nothing here argues it equals; and moving the all-reduce onto
the stack depth moved the all-to-all by the same four times on the hybrid
here, which is a side effect of sharing the count in hand and not a result
about experts. Settling it needs an expert geometry and a term that reads one,
which belongs to a backend that means its numbers rather than to this one.

Two things the form does not know, stated because a reader of a number will
otherwise assume it does. It has one quadratic coefficient for the whole batch
and no notion of a layer kind, so a stack whose layers do not all grow with
history -- a recurrent or windowed layer among full-attention ones -- is priced
as if they all did; whoever sets the coefficient owns that, and a form that
needs to tell them apart needs a term per kind rather than a different number.
And the collective list it charges from is necessary rather than sufficient: an
absence is an absence, a presence is a candidate the widths admit and two
conditions outside them can still rule out. Every collective term says so in
its own provenance, so a charge that may not correspond to a collective cannot
be read back as one that did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.backends.provenance import Method, Provenance
from atom.compass.backends.shape import (
    BatchView,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

SOURCE = "step-stub"

# The note on every term's provenance, and repeated by `describe()`, so the
# record an artifact writes says it and not only the source that produced it:
# where the coefficients came from, and what replaces them.
DECLARED_NOTE = (
    "coefficients stated in the run file, not measured -- a plumbing figure, "
    "not an accuracy claim; successor: the step linear law fitted on real "
    "step records (step_linear)"
)

# Added to a collective's own provenance. The widths rule a collective out
# conclusively and rule it in only conditionally, so a charge for one states
# which of the two it is rather than leaving a reader to assume the stronger.
CANDIDATE = "a collective these widths admit, not one observed to run"

# Added to the decode padding term whenever a rung was supplied. The rows can
# show a rung is too narrow and can never show it is too wide, so that count
# rests on the caller in a way the coefficient's own note does not cover.
UNCHECKED_RUNG = "padding from a supplied rung width; nothing here bounds it above"

# Added to a collective's own provenance, because the count it multiplies is
# the one a reader is most likely to assume wrong. Two layer counts describe
# one worker and a record that states seconds states neither, so the term says
# which of them it was charged on where the charge is read.
PER_STACK_LAYER = "one charge per layer of the worker's stack, not per paged layer"

# Added to the provenance of every collective except the one whose count is
# argued for below. The depth is the count in hand rather than the count that
# collective runs on, and a term charged on a stand-in says so where the charge
# is read; a reader who divides the charge back out otherwise recovers a number
# and no indication that nothing here chose it. It names no collective and no
# kind of layer, because it is attached by not being in the argued set: a
# sentence about the layers one of them runs on would be a claim about every
# later name that joins it unargued, made by the default rather than by
# anybody.
DEPTH_STANDS_IN = (
    "the stack depth standing in for this collective's own count, which "
    "nothing here argues it equals"
)

# The collectives whose charge on the worker's stack depth has an argument
# behind it: an all-reduce runs on every layer the worker runs, and that is the
# depth. Membership is stated rather than inferred so that a collective added
# to the named set later is labelled a stand-in until somebody argues its
# count, instead of inheriting an argument made for a different collective.
_COUNT_ARGUED = frozenset({"tp-all-reduce"})


@dataclass(frozen=True)
class Coefficients:
    """The declared multipliers of the form. None of them was measured.

    The defaults are round numbers in seconds-per-unit, picked so a test's
    arithmetic can be checked by hand and so a long chunk costs visibly more
    than a short one. They are not a fit and they carry no physics; a backend
    that means its numbers is a later milestone with measurements behind it.
    """

    prefill_step: float = 2.0e-5
    prefill_token: float = 5.0e-7
    prefill_query_square: float = 2.0e-11
    prefill_query_cached: float = 4.0e-12
    decode_step: float = 2.0e-5
    decode_request: float = 2.0e-6
    decode_context: float = 4.0e-9
    decode_padding: float = 1.0e-9
    collective_token_layer: float = 1.0e-9

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{spec.name}={value} is not a usable coefficient")

    @property
    def shape_blind(self) -> bool:
        """True when only the two intercepts are non-zero: constant pricing."""
        shaped = [f.name for f in fields(self) if f.name not in _INTERCEPTS]
        return not any(getattr(self, name) for name in shaped)

    @classmethod
    def constant(cls, prefill_seconds: float, decode_seconds: float) -> Coefficients:
        """Bring-up pricing: the same form with every shape coefficient zero.

        Constant pricing is a coefficient set rather than a mode on the
        backend. `price` gains no branch, the breakdown still names every
        term, and the zeros are in the record instead of implied by a flag
        somewhere else. It is kept for first bring-up and is not the default,
        because a price that ignores the shape cannot show whether anything
        downstream reacted to the shape.
        """
        blind = {f.name: 0.0 for f in fields(cls)}
        blind["prefill_step"] = prefill_seconds
        blind["decode_step"] = decode_seconds
        return cls(**blind)


_INTERCEPTS = ("prefill_step", "decode_step")


class StepStubBackend(CostBackend):
    """Prices a step by multiplying declared coefficients into its shapes.

    A stub, not an analytical model. `Granularity.STEP` says this answers for
    a whole step, which is true; what the granularity cannot say, and
    what the provenance on every term says instead, is that the numbers behind
    the answer were declared rather than obtained.

    `parallelism` decides which collectives are charged, by asking it rather
    than by reasoning from the expert width -- an all-to-all exists on the
    data-parallel width alone, and a backend that decided from the experts
    would charge a single-rank deployment for a collective it never builds.
    What it returns is a candidate set: nothing it omits can run, and what it
    names can still be ruled out by conditions the widths do not express, so
    each such term carries that qualification in its own provenance rather
    than being charged as a certainty.

    Pricing one needs a layer count, and there are two of them. `stack_layers`
    is the one the charge is made on: every layer this worker runs, whatever
    each layer keeps. `geometry` is the KV a block is sized from, whose own
    `layers` counts only the layers holding a cache of every past token -- the
    same number on a uniform stack and a quarter of it on a stack that is one
    paged layer in four. Both are held, neither is derived from the other, and
    a deployment with a candidate collective and no stack depth is refused: a
    count taken from the geometry there would be wrong in kind on every hybrid
    and right by coincidence on everything else.

    **A geometry is not required to price a collective, and that is a
    loosening of what this constructor used to refuse.** Nothing that computes
    seconds reads it -- a collective costs the tokens times the stack depth
    times a declared coefficient, and every other term reads the batch -- so
    declining for want of a paged count would refuse a price this would then
    make identically, which is a refusal with nothing behind it. It stays on
    the constructor because it is the pool declaration a price is paired with
    where there is one, and `describe()` writes it. What its absence costs is
    two things, stated here rather than left to be discovered. The check that
    catches the two counts crossed -- a span shallower than the paged subset
    inside it -- compares two stated counts, so with one stated there is
    nothing to cross and it does not run. And `describe()` has one count to
    write, so it says the paged one was not stated rather than printing a lone
    number that reads like the whole description of a worker.

    `stage_share` prices one pipeline stage: the stage's share of the stack's
    layers, then of its paged layers. The coefficients are the whole model's;
    the history terms (query square, cached query, context, graph padding)
    scale by the paged share, every other step term by the layer share, so
    the stages of one partition sum to the whole-model price. `None` is the
    whole model.
    """

    def __init__(
        self,
        coefficients: Coefficients | None = None,
        parallelism: Parallelism | None = None,
        geometry: KvGeometry | None = None,
        stack_layers: int | None = None,
        stage_share: tuple[float, float] | None = None,
    ) -> None:
        self.coefficients = Coefficients() if coefficients is None else coefficients
        self.parallelism = Parallelism() if parallelism is None else parallelism
        self.geometry = geometry
        if stage_share is not None and not all(0 <= s <= 1 for s in stage_share):
            raise ValueError(
                f"stage_share={stage_share} is not a stage's (layer, paged layer) "
                "share of the stack; each is between 0 and 1"
            )
        self.stage_share = stage_share
        # A whole number of layers, or nothing. `int()` on the argument would
        # take 64.7 for a precise 64 and "64" for a count that was never
        # stated as one, which is the guess the refusals below exist to
        # decline, made in the constructor that makes them. The wording and
        # the `TypeError` are the ones this package already uses for a dialled
        # count. A `bool` is an `int` to Python and is taken as the one-layer
        # span it equals; refusing that here alone would be a second
        # convention for one question, which is worse than the case it would
        # catch. It is stored as that span and not as `True`: the check has
        # already declined everything that is not an `int`, so converting
        # after it narrows nothing else, and `describe()` writes this value as
        # the count the charge was made on, which `True` is not.
        if stack_layers is not None and not isinstance(stack_layers, int):
            raise TypeError(
                f"stack_layers must be a whole number of at least 1, not "
                f"{stack_layers!r}"
            )
        self.stack_layers = None if stack_layers is None else int(stack_layers)
        named = self.parallelism.collectives()
        if named and self.stack_layers is None:
            raise ValueError(
                f"{', '.join(named)} to price and no stack depth to price it from; "
                "a collective runs once per layer this worker runs, and a KV "
                "geometry counts only the layers that hold a cache of every past "
                "token, so state how many layers deep the span is"
            )
        if self.stack_layers is not None and self.stack_layers < 1:
            raise ValueError(
                "a worker runs at least one layer, got "
                f"stack_layers={self.stack_layers}"
            )
        # Two counts are what this compares, and a geometry is optional above,
        # so a deployment that states one count is priced with this check not
        # run. That is the cost of the geometry being optional and it is named
        # in the class docstring beside the reason.
        if (
            self.stack_layers is not None
            and geometry is not None
            and self.stack_layers < geometry.layers
        ):
            raise ValueError(
                f"a {self.stack_layers}-layer span holding {geometry.layers} paged "
                "layers is not a span: the layers that hold a cache are a subset "
                "of the layers there are, so the two counts have been crossed"
            )

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def _term(
        self, name: str, count: int, coefficient: float, qualifier: str = ""
    ) -> CostTerm:
        """One term: a count read off the batch times a number nobody measured.

        `qualifier` is for a term whose count is less certain than the reading
        it came from, and it goes in the record beside the count rather than
        in a note somewhere else.
        """
        detail = f"{coefficient:g} s x {count}"
        return CostTerm(
            name,
            count * coefficient,
            Provenance(
                Method.DECLARED,
                f"{detail}; {qualifier}" if qualifier else detail,
                SOURCE,
                note=DECLARED_NOTE,
            ),
        )

    def price(self, batch_view: BatchView) -> StepCost:
        """Price one step as the parts its total is folded from.

        No seconds are added here. Each term is one product, and the total is
        whatever `StepCost` folds from them, so there is no second summation
        to disagree with the first.

        The two refusals below and in `BatchView.__post_init__` are `TypeError`
        and `ValueError` rather than the `CostRefused` the seam names, because
        a batch that is not a batch is a caller defect and not a step this
        backend declines to price; the asymmetry is worth stating because a
        resolver that wraps this backend catches `CostRefused` and falls
        through to the next source, and will not catch these.
        """
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}; "
                "the projection is built by whoever holds the scheduled batch"
            )
        c = self.coefficients
        layer, paged = self.stage_share or (1.0, 1.0)
        counted: list[tuple[str, int, float] | tuple[str, int, float, str]] = []
        prefill = batch_view.prefill
        if prefill:
            counted += [
                ("prefill.step", 1, c.prefill_step * layer),
                ("prefill.tokens", sum_tokens(prefill), c.prefill_token * layer),
                (
                    "prefill.query_square",
                    sum_query_square(prefill),
                    c.prefill_query_square * paged,
                ),
                (
                    "prefill.query_cached",
                    sum_query_cached(prefill),
                    c.prefill_query_cached * paged,
                ),
            ]
        decode = batch_view.decode
        if decode:
            counted += [
                ("decode.step", 1, c.decode_step * layer),
                ("decode.requests", len(decode), c.decode_request * layer),
                ("decode.context", sum_context(decode), c.decode_context * paged),
                (
                    "decode.graph_padding",
                    batch_view.graph_padding,
                    c.decode_padding * paged,
                    "" if batch_view.capture_rung is None else UNCHECKED_RUNG,
                ),
            ]
        terms = [self._term(*row) for row in counted]
        collectives = self.parallelism.collectives()
        if collectives:
            # The constructor refuses a candidate collective with no depth
            # stated, so there is a count here and it is the stack's.
            moved = sum_tokens(batch_view.requests) * self.stack_layers
            for collective in collectives:
                # One count for all of them, and an argument for it behind
                # only one. The rest say in their own provenance that the
                # depth is standing in, because the seconds do not and a
                # reader who recovers the count from them recovers no sign
                # that nothing here chose it.
                qualifier = f"{CANDIDATE}; {PER_STACK_LAYER}"
                if collective not in _COUNT_ARGUED:
                    qualifier = f"{qualifier}; {DEPTH_STANDS_IN}"
                terms.append(
                    self._term(
                        f"collective.{collective}",
                        moved,
                        c.collective_token_layer,
                        qualifier,
                    )
                )
        return StepCost(terms)

    def all_gather_s(self, rows: int) -> float:
        """One TP all-gather of a `rows`-row tensor, as a pipeline stage runs on
        each sharded tensor it receives: one collective on one layer."""
        return rows * self.coefficients.collective_token_layer

    def describe(self) -> str:
        """One line for the run record, including what the number is not."""
        pricing = "constant bring-up" if self.coefficients.shape_blind else "shape-read"
        widths = (
            f"tp{self.parallelism.tp_size}"
            f" pp{self.parallelism.pp_size}"
            f" dp{self.parallelism.dp_size}"
        )
        named = ", ".join(self.parallelism.collectives())
        if named:
            # The depth belongs in the line because the seconds do not carry
            # it: a collective charged on the paged layers of a hybrid and one
            # charged on its stack read alike once they are in a total. With
            # no geometry held there is one count to write, and a lone number
            # here would read as the worker's whole description, so the line
            # says which count is missing instead of quietly emitting one.
            paged = (
                ", paged count not stated"
                if self.geometry is None
                else f", {self.geometry.layers} paged"
            )
            named += f" on {self.stack_layers} layers{paged}"
        else:
            named = "no collectives"
        if self.stage_share is not None:
            named += ", stage share {:g} of layers, {:g} of paged".format(
                *self.stage_share
            )
        return (
            f"step-level stand-in, {pricing} coefficients, {widths}, {named}"
            f" -- {DECLARED_NOTE}"
        )
