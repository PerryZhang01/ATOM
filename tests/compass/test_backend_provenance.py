# SPDX-License-Identifier: MIT
"""The vocabulary every cost is labelled with.

The point of these tests is that the label cannot be omitted. A cost whose
origin is unknown is worse than a missing cost, because it is counted, summed
and reported like a real one; the constructors are written so that producing
one takes a deliberate lie rather than an oversight, and these assert that
property rather than the spellings.

The refusal rendering is asserted too. It is not cosmetic: the string is what
lands in a run record, and a stand-in answer that renders the same as a
first-choice one would hide exactly the fall-through the record exists to
show. So is the composition rule -- a provenance that is resolved twice keeps
both refusals and both names, because a source that is itself a resolver is
the ordinary shape and dropping half the chain there names the wrong source.
"""

import pytest

from atom.compass.backends import Provenance, Refusal, Species


def test_the_species_vocabulary_is_closed():
    """Five ways an answer is obtained, and no sixth added in passing."""
    assert {s.value for s in Species} == {
        "analytical",
        "measured",
        "fitted",
        "interpolated",
        "extrapolated",
    }


def test_provenance_has_no_default_species():
    """There is no constructor call that leaves the origin unstated."""
    with pytest.raises(TypeError):
        Provenance()


def test_provenance_refuses_a_species_shaped_string():
    """A bare string would be silently accepted by the dataclass otherwise."""
    with pytest.raises(TypeError):
        Provenance("measured")


def test_provenance_refuses_a_refusal_shaped_thing():
    with pytest.raises(TypeError):
        Provenance(Species.MEASURED, "", "", ("price_list declined",))


def test_a_refusal_names_who_declined_and_why():
    r = Refusal("price_list", "no entry for gemm(4096,8192)")
    assert r.source == "price_list"
    assert str(r) == "refused(price_list: no entry for gemm(4096,8192))"


@pytest.mark.parametrize(
    "source,reason",
    [("", "a reason"), ("  ", "a reason"), ("a_source", ""), ("a_source", " ")],
)
def test_a_refusal_with_a_blank_half_is_not_a_refusal(source, reason):
    """Half a refusal cannot be acted on and cannot be grouped when counted."""
    with pytest.raises(ValueError):
        Refusal(source, reason)


def test_plain_provenance_renders_species_and_unit():
    assert str(Provenance(Species.MEASURED)) == "measured"
    assert str(Provenance(Species.MEASURED, "op-level")) == "measured (op-level)"


def test_an_answer_names_the_source_that_produced_it():
    """The record says what answered, not only what species the answer was."""
    p = Provenance(Species.MEASURED, "op-level").resolved("price_list")
    assert str(p) == "measured (op-level) via price_list"
    assert not p.is_refused


def test_an_answer_must_name_a_source():
    with pytest.raises(ValueError, match="name the source"):
        Provenance(Species.MEASURED).resolved("  ")


def test_a_stand_in_answer_renders_the_refusal_it_replaced():
    """Reading the record, a fall-through is visible without comparing runs."""
    p = Provenance(Species.ANALYTICAL, "roofline").resolved(
        "analytic_law", [Refusal("price_list", "unpriced leaf")]
    )
    assert p.is_refused
    assert str(p) == (
        "refused(price_list: unpriced leaf) -> analytical (roofline) via analytic_law"
    )


def test_every_source_that_declined_is_kept_not_just_the_first():
    """A count of reasons that drops the middle of the chain undercounts."""
    p = Provenance(Species.ANALYTICAL).resolved(
        "analytic_law",
        [
            Refusal("price_list", "unpriced leaf"),
            Refusal("nearest_key", "outside the measured range"),
        ],
    )
    assert [r.source for r in p.refusals] == ["price_list", "nearest_key"]
    assert p.refusal == Refusal("price_list", "unpriced leaf")
    assert str(p) == (
        "refused(price_list: unpriced leaf; nearest_key: outside the measured range)"
        " -> analytical via analytic_law"
    )


def test_resolving_twice_composes_rather_than_collapses():
    """A source that is itself a resolver: outer refusals are the earlier ones."""
    inner = Provenance(Species.ANALYTICAL).resolved(
        "inner_law", [Refusal("inner_price_list", "unpriced leaf")]
    )
    outer = inner.resolved("layered", [Refusal("price_list", "unpriced leaf")])

    assert [r.source for r in outer.refusals] == ["price_list", "inner_price_list"]
    assert outer.refusal == Refusal("price_list", "unpriced leaf")
    assert outer.source == "layered/inner_law"
