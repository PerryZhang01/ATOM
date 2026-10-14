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

from atom.compass.backends import Method, Provenance, Refusal, combine


def test_the_method_vocabulary_is_closed():
    """Four ways an answer is obtained, and no fifth added in passing."""
    assert {m.value: m.empirical for m in Method} == {
        "analytical": False,
        "declared": False,
        "measured": True,
        "estimated": True,
    }


@pytest.mark.parametrize(
    "methods,combined",
    [
        ({Method.DECLARED, Method.MEASURED}, Method.DECLARED),
        ({Method.DECLARED, Method.ANALYTICAL}, Method.DECLARED),
        ({Method.ANALYTICAL}, Method.ANALYTICAL),
        ({Method.MEASURED}, Method.MEASURED),
        ({Method.ESTIMATED}, Method.ESTIMATED),
        ({Method.ANALYTICAL, Method.MEASURED}, Method.ESTIMATED),
        ({Method.MEASURED, Method.ESTIMATED}, Method.ESTIMATED),
    ],
)
def test_any_declared_part_makes_an_answer_declared_and_a_mixture_estimated(
    methods, combined
):
    assert combine(methods) is combined


def test_nothing_combines_to_no_method():
    with pytest.raises(ValueError, match="no parts"):
        combine([])


@pytest.mark.parametrize("method", [m for m in Method if m is not Method.ESTIMATED])
def test_only_an_estimated_answer_can_be_validated(method):
    with pytest.raises(ValueError, match="only an estimated answer"):
        Provenance(method, validated=True, note="stated in this test; none")
    assert Provenance(Method.ESTIMATED, validated=True).validated
    assert not Provenance(Method.ESTIMATED).validated


def test_a_declared_answer_states_its_origin_and_successor():
    with pytest.raises(ValueError, match="needs a note"):
        Provenance(Method.DECLARED, "a guess")
    p = Provenance(Method.DECLARED, "1e-3 s", "stub", note="typed in; successor: a fit")
    assert str(p) == "declared (1e-3 s) via stub; typed in; successor: a fit"


def test_an_estimate_renders_whether_it_was_validated():
    assert str(Provenance(Method.ESTIMATED, validated=True)) == "estimated, validated"
    assert str(Provenance(Method.ESTIMATED)) == "estimated, unvalidated"


def test_resolving_keeps_validated_and_the_note():
    p = Provenance(Method.ESTIMATED, validated=True).resolved("outer")
    d = Provenance(Method.DECLARED, note="typed in; none").resolved("outer")
    assert p.validated and d.note == "typed in; none"


def test_provenance_has_no_default_method():
    """There is no constructor call that leaves the origin unstated."""
    with pytest.raises(TypeError):
        Provenance()


def test_provenance_refuses_a_method_shaped_string():
    """A bare string would be silently accepted by the dataclass otherwise."""
    with pytest.raises(TypeError):
        Provenance("measured")


def test_provenance_refuses_a_refusal_shaped_thing():
    with pytest.raises(TypeError):
        Provenance(Method.MEASURED, "", "", ("price_list declined",))


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


def test_plain_provenance_renders_method_and_unit():
    assert str(Provenance(Method.MEASURED)) == "measured"
    assert str(Provenance(Method.MEASURED, "op-level")) == "measured (op-level)"


def test_an_answer_names_the_source_that_produced_it():
    """The record says what answered, not only what method the answer was."""
    p = Provenance(Method.MEASURED, "op-level").resolved("price_list")
    assert str(p) == "measured (op-level) via price_list"
    assert not p.is_refused


def test_an_answer_must_name_a_source():
    with pytest.raises(ValueError, match="name the source"):
        Provenance(Method.MEASURED).resolved("  ")


def test_a_stand_in_answer_renders_the_refusal_it_replaced():
    """Reading the record, a fall-through is visible without comparing runs."""
    p = Provenance(Method.ANALYTICAL, "roofline").resolved(
        "analytic_law", [Refusal("price_list", "unpriced leaf")]
    )
    assert p.is_refused
    assert str(p) == (
        "refused(price_list: unpriced leaf) -> analytical (roofline) via analytic_law"
    )


def test_every_source_that_declined_is_kept_not_just_the_first():
    """A count of reasons that drops the middle of the chain undercounts."""
    p = Provenance(Method.ANALYTICAL).resolved(
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
    inner = Provenance(Method.ANALYTICAL).resolved(
        "inner_law", [Refusal("inner_price_list", "unpriced leaf")]
    )
    outer = inner.resolved("layered", [Refusal("price_list", "unpriced leaf")])

    assert [r.source for r in outer.refusals] == ["price_list", "inner_price_list"]
    assert outer.refusal == Refusal("price_list", "unpriced leaf")
    assert outer.source == "layered/inner_law"
