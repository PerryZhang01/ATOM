# SPDX-License-Identifier: MIT
"""Consulting cost sources in order, and what falling through leaves behind.

The failure these tests exist for is a run that silently answers from the
last source: every number present, every total plausible, and nothing
in the record saying the measured price was never found. So the assertions are
about what survives a fall-through, not about which source won -- the answer
carries every refusal that preceded it and the name of the source that produced
it, the resolution carries this resolver's own refusals, and an exhausted resolver
raises with the complete list rather than the first entry.

The composed case is asserted separately because it is the shape a layered
backend takes, and it is where a chain that keeps only one refusal reports the
wrong source to whoever reads the reason counts.
"""

import pytest

from atom.compass.backends import (
    CostRefused,
    CostSource,
    CostTerm,
    Provenance,
    ProvenanceMix,
    Refusal,
    Resolver,
    Species,
    StepCost,
)


class Answers(CostSource):
    """A source that always answers, at a stated species."""

    def __init__(self, name, seconds, species):
        self._name = name
        self._seconds = seconds
        self._species = species

    @property
    def name(self):
        return self._name

    def price(self, request):
        return CostTerm(request, self._seconds, Provenance(self._species))


class Declines(CostSource):
    """A source that always declines, for a stated reason."""

    def __init__(self, name, reason):
        self._name = name
        self._reason = reason

    @property
    def name(self):
        return self._name

    def price(self, request):
        return self.refuse(self._reason)


class Layered(CostSource):
    """A source that is itself a resolver, which is how a real backend stacks."""

    def __init__(self, name, inner):
        self._name = name
        self._inner = inner

    @property
    def name(self):
        return self._name

    def price(self, request):
        return self._inner.resolve(request).term


PRICED = Answers("price_list", 0.004, Species.MEASURED)
LAW = Answers("analytic_law", 0.005, Species.ANALYTICAL)
ROOFLINE = Answers("roofline", 0.006, Species.ANALYTICAL)
UNPRICED = Declines("price_list", "no entry for this leaf")
OUTSIDE = Declines("nearest_key", "outside the measured range")


def test_the_top_source_answers_and_nothing_is_marked():
    got = Resolver([PRICED, LAW]).resolve("attention")
    assert got.source == "price_list"
    assert not got.fell_through
    assert not got.term.provenance.is_refused
    assert str(got.term.provenance) == "measured via price_list"


def test_a_fall_through_is_stamped_on_the_answer():
    got = Resolver([UNPRICED, LAW]).resolve("attention")
    assert got.source == "analytic_law"
    assert got.fell_through
    assert got.term.provenance.refusal == Refusal(
        "price_list", "no entry for this leaf"
    )
    assert str(got.term.provenance) == (
        "refused(price_list: no entry for this leaf) -> analytical via analytic_law"
    )


def test_the_artifact_names_the_source_that_answered():
    """A row that names only a species cannot say which source to go and fix."""
    step = StepCost([Resolver([UNPRICED, OUTSIDE, LAW]).resolve("attention").term])
    ((_, _, rendered),) = step.rows()
    assert "via analytic_law" in rendered


def test_every_source_passed_over_reaches_the_answer_and_the_count():
    """A middle source's reason is a gap too, and has to be countable."""
    got = Resolver([UNPRICED, OUTSIDE, LAW]).resolve("attention")
    assert [r.source for r in got.declined] == ["price_list", "nearest_key"]
    assert [r.source for r in got.term.provenance.refusals] == [
        "price_list",
        "nearest_key",
    ]

    mix = ProvenanceMix()
    mix.record(StepCost([got.term]))
    assert mix.reasons() == {
        ("price_list", "no entry for this leaf"): 1,
        ("nearest_key", "outside the measured range"): 1,
    }


def test_a_resolver_backed_source_keeps_both_halves_of_the_chain():
    """The outer refusal is the earlier one, and neither source is lost."""
    inner = Resolver([Declines("inner_price_list", "no entry"), LAW])
    got = Resolver([UNPRICED, Layered("layered", inner)]).resolve("attention")

    assert [r.source for r in got.term.provenance.refusals] == [
        "price_list",
        "inner_price_list",
    ]
    assert got.term.provenance.refusal.source == "price_list"
    assert got.term.provenance.source == "layered/analytic_law"


def test_an_exhausted_inner_resolver_does_not_escape_the_outer_one():
    """It declined; the sources below it have not been asked yet."""
    inner = Resolver([Declines("inner_price_list", "no entry"), OUTSIDE])
    got = Resolver([UNPRICED, Layered("layered", inner), ROOFLINE]).resolve("attention")

    assert got.source == "roofline"
    assert [r.source for r in got.declined] == [
        "price_list",
        "inner_price_list",
        "nearest_key",
    ]
    assert [r.source for r in got.term.provenance.refusals] == [
        "price_list",
        "inner_price_list",
        "nearest_key",
    ]


def test_an_exhausted_inner_resolver_keeps_its_reasons_countable():
    """Flattening them onto the wrapper would name something unmeasurable."""
    inner = Resolver([Declines("inner_price_list", "no entry"), OUTSIDE])
    got = Resolver([UNPRICED, Layered("layered", inner), ROOFLINE]).resolve("attention")

    mix = ProvenanceMix()
    mix.record(StepCost([got.term]))
    assert mix.reasons() == {
        ("price_list", "no entry for this leaf"): 1,
        ("inner_price_list", "no entry"): 1,
        ("nearest_key", "outside the measured range"): 1,
    }


def test_an_inner_exhaustion_that_names_nobody_still_names_the_source():
    """A hand-rolled source can raise with an empty list; silence is not allowed."""

    class Silent(CostSource):
        @property
        def name(self):
            return "silent_layer"

        def price(self, request):
            raise CostRefused(request, ())

    got = Resolver([Silent(), ROOFLINE]).resolve("attention")
    assert [r.source for r in got.declined] == ["silent_layer"]


def test_an_exhausted_resolver_refuses_with_the_whole_list():
    """One refusal per run turns one coverage gap into one iteration each."""
    with pytest.raises(CostRefused) as raised:
        Resolver([UNPRICED, OUTSIDE]).resolve("attention")
    assert [r.source for r in raised.value.declined] == ["price_list", "nearest_key"]
    assert raised.value.request == "attention"
    assert "outside the measured range" in str(raised.value)


def test_an_exhausted_resolver_lists_the_inner_sources_too():
    """The whole list means the whole list, however deeply it was nested."""
    inner = Resolver([Declines("inner_price_list", "no entry"), OUTSIDE])
    with pytest.raises(CostRefused) as raised:
        Resolver([UNPRICED, Layered("layered", inner)]).resolve("attention")
    assert [r.source for r in raised.value.declined] == [
        "price_list",
        "inner_price_list",
        "nearest_key",
    ]


def test_a_resolver_needs_sources():
    with pytest.raises(ValueError, match="no sources"):
        Resolver([])


def test_sources_are_read_by_name_so_names_are_unique():
    with pytest.raises(ValueError, match="duplicate source"):
        Resolver([PRICED, Answers("price_list", 0.009, Species.FITTED)])


def test_the_resolver_reports_the_order_it_will_consult():
    assert Resolver([UNPRICED, OUTSIDE, LAW]).names == (
        "price_list",
        "nearest_key",
        "analytic_law",
    )


def test_a_source_cannot_pin_its_refusal_on_another_source():
    """Misattributed refusals make the reason counts name the wrong thing."""

    class Misattributes(Declines):
        def price(self, request):
            return Refusal("somebody_else", self._reason)

    resolver = Resolver([Misattributes("price_list", "no entry"), LAW])
    with pytest.raises(ValueError, match="has to name who declined"):
        resolver.resolve("attention")


def test_a_source_that_returns_a_bare_number_is_a_bug_not_a_cost():
    class ReturnsSeconds(CostSource):
        @property
        def name(self):
            return "loose"

        def price(self, request):
            return 0.004

    with pytest.raises(TypeError, match="not a cost term"):
        Resolver([ReturnsSeconds()]).resolve("attention")
