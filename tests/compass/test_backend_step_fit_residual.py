# SPDX-License-Identifier: MIT
"""Step fit's residual: the lookup times a factor fitted over the rows' features
in batch order, answering only inside the hull it was fitted on.

Decode steps come from ATOM's `Scheduler` and are priced through the runner's
`forward` and `_group_step_seconds`. Each holds 32 rows caching prompts of 575
and 959 tokens, or 767, so every batch's equivalent shape is B = 32 at c = 767,
and its lookup in the eager grid is f(32, 767) = 1000 + 960 + 767 + 1534 =
4261 us. All three share tokens 1/32, query_square 1/32768 and query_context
3/128; with k = N_KV, they differ in

    batch         prompts                  order   raggedness
    homogeneous   767 x 32                 0       1
    grouped       575 x 16, then 959 x 16  1/64    5/4      (384 / 24576)
    alternating   575, 959, ... x 16       31/64   5/4      (31 x 384 / 24576)
"""

import math
from fractions import Fraction

import pytest
from test_backend_step_fit import (
    MIDDLE,
    chunks,
    decode,
    eager,
    fitted,
    linear,
    point,
    view,
)
from test_dp_step_max import runner

from atom.compass.backends import RequestShape, Species
from atom.compass.backends.shape import sum_context
from atom.compass.backends.step_fit import equivalent_shape
from atom.compass.backends.step_fit_residual import V1, Recipe, Residual

HOMOGENEOUS = [767] * 32
GROUPED = [575] * 16 + [959] * 16
ALTERNATING = [575, 959] * 16
# 0.75 + 0.5 x order + 0.25 x raggedness: 1, 1.0703125 and 1.3046875 above.
ORDERED = linear(0.75, order=0.5, raggedness=0.25)


def residual(hull=None, **members):
    return Residual(fitted(members, dict.fromkeys(members, 1.0), hull))


def price(prompts, backend):
    r = runner(backend)
    return r.forward(decode(prompts)).predicted_s, r.refused_commands()


def test_one_shape_in_another_order_or_raggedness_is_priced_apart():
    batches = (HOMOGENEOUS, GROUPED, ALTERNATING)

    # One equivalent shape, so one lookup.
    assert [price(p, eager()) for p in batches] == [(4261e-6, ())] * 3
    assert [price(p, eager(residual=residual(o=ORDERED))) for p in batches] == [
        (4261e-6 * 1.0, ()),
        (4261e-6 * 1.0703125, ()),
        (4261e-6 * 1.3046875, ()),
    ]


@pytest.mark.parametrize(
    "rows,order,raggedness",
    [
        ([(1, 768, True)] * 32, 0, 1),
        ([(1, 576, True)] * 16 + [(1, 960, True)] * 16, Fraction(1, 64), 1.25),
        ([(1, 576, True), (1, 960, True)] * 16, Fraction(31, 64), 1.25),
    ],
    ids=["homogeneous", "grouped", "alternating"],
)
def test_the_recipe_reads_order_and_raggedness_off_the_rows_in_order(
    rows, order, raggedness
):
    rows = tuple(RequestShape(*r) for r in rows)

    assert equivalent_shape(rows) == (32, 1, 767)
    shared = (Fraction(1, 32), Fraction(1, 32768), Fraction(3, 128))
    assert V1(rows) == shared + (order, raggedness)


def test_a_prefill_chunk_takes_the_same_form():
    # Rows (N_Q, N_KV) of (3, 3) then (5, 15): 34 and 9 + 75 token pairs,
    # a jump of 12 over 18 context tokens, and 2 x 15 / 18.
    rows = (RequestShape(3, 3, False), RequestShape(5, 15, False))
    assert V1(rows) == (
        Fraction(8, 1024),
        Fraction(34, 2**20),
        Fraction(84, 2**20),
        Fraction(2, 3),
        Fraction(5, 3),
    )
    # 1 + 2 x tokens: 1.125 on a 64-token middle chunk, 1.015625 on the final 8.
    backend = eager(
        residual=residual(t=linear(tokens=2.0)),
        output=[point(1, 8, 192, time=3)],
        no_output=MIDDLE,
        unit="ms",
    )
    steps, refused = chunks(backend)
    assert steps == [(False, 1e-3 * 1.125)] * 3 + [(True, 3e-3 * 1.015625)]
    assert refused == ()


def test_the_term_is_fitted_and_its_detail_holds_the_lookup_and_each_member():
    exp = linear(0.0, "exp", order=1.0)
    backend = eager(residual=residual(o=ORDERED, e=exp))
    rows = [(1, 576, True), (1, 960, True)] * 16

    (term,) = backend.estimate(view(*rows)).terms
    factor = 1.3046875 + math.exp(0.484375)
    assert term.seconds == 4261e-6 * factor
    assert term.provenance.species is Species.FITTED
    assert term.provenance.detail == (
        "lookup 0.004261 s, interpolated; decode cell B=32, c 512..1024; "
        f"residual {factor!r} = 1.0 x o 1.3046875 + 1.0 x e {math.exp(0.484375)!r}"
    )
    assert backend.describe().endswith("; recipe step-fit-residual v1, members o, e")


# Both hulls bound order <= 1/4; the second also holds raggedness at 1, so its
# facet has no raggedness entry.
@pytest.mark.parametrize(
    "hull,inside,outside,named",
    [
        (
            {"facets": [[0, 0, 0, 1, 0, -0.25]]},
            GROUPED,
            ALTERNATING,
            "order=0.484375, raggedness=1.25",
        ),
        (
            {
                "equal": {"raggedness": 1.0},
                "facets": [[0, 0, 0, 1, -0.25]],
            },
            HOMOGENEOUS,
            GROUPED,
            "order=0.015625, raggedness=1.25",
        ),
    ],
    ids=["facet", "equal"],
)
def test_a_step_outside_the_residuals_hull_refuses_naming_it(
    hull, inside, outside, named
):
    backend = eager(residual=residual(hull, o=ORDERED))

    assert price(inside, backend)[1] == ()
    assert price(outside, backend) == (
        0.0,
        (
            (
                "cost:step-fit: the step at tokens=0.03125, "
                "query_square=3.05176e-05, query_context=0.0234375, "
                f"{named} is outside the residual's hull"
            ),
        ),
    )


def test_a_hull_refusal_refuses_that_step_only():
    # tokens <= 1/32: the three 64-token middle chunks are outside, the final 8 in.
    backend = eager(
        residual=residual({"facets": [[1, 0, 0, 0, 0, -1 / 32]]}, u=linear()),
        output=[point(1, 8, 192, time=3)],
        no_output=MIDDLE,
        unit="ms",
    )

    steps, refused = chunks(backend)
    assert steps == [(False, 0.0)] * 3 + [(True, 3e-3)]
    assert len(refused) == 3
    assert all("tokens=0.0625," in r and "residual's hull" in r for r in refused)


@pytest.mark.parametrize(
    "members,said",
    [
        # 1.0625 - 4 x order: 1 when grouped, -0.875 when alternating; the
        # other member answering does not rescue the step.
        (
            {"u": linear(), "falls": linear(1.0625, order=-4.0)},
            "member falls gives -0.875",
        ),
        (
            {"u": linear(), "big": linear(0.0, "exp", tokens=1e6)},
            "member big gives inf",
        ),
        ({"a": linear(1e308), "b": linear(1e308)}, "factor is inf"),
    ],
    ids=["negative", "overflow", "factor"],
)
def test_a_member_or_factor_that_is_not_positive_and_finite_refuses(members, said):
    backend = eager(residual=residual(**members))

    seconds, refused = price(ALTERNATING, backend)
    assert seconds == 0.0
    assert refused == (
        f"cost:step-fit: the residual's {said}, which is not a positive finite factor",
    )


def raggedness(rows):
    """The recipe's raggedness, normalised by half."""
    longest = max(row.context_tokens for row in rows)
    return Fraction(len(rows) * longest, 2 * sum_context(rows))


def test_the_digest_is_over_the_feature_functions_source():
    edited = Recipe(V1.name, V1.version, V1.features[:-1] + (raggedness,))

    assert edited.names == V1.names
    assert edited.digest != V1.digest
    assert Recipe(V1.name, V1.version, V1.features).digest == V1.digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("version", 2),
        ("digest", Recipe(V1.name, 1, V1.features[:-1] + (raggedness,)).digest),
    ],
    ids=["name", "version", "digest"],
)
def test_a_residual_fitted_under_another_recipe_fails_loading(field, value):
    mapping = fitted({"u": linear()}, {"u": 1.0})
    mapping["recipe"][field] = value

    with pytest.raises(ValueError, match="fitted under recipe"):
        Residual(mapping)


def test_a_member_the_residual_does_not_select_never_loads():
    tree = {"form": "tree", "nodes": []}
    loaded = Residual(fitted({"u": linear(), "tree": tree}, {"u": 1.0}))

    assert [name for name, _, _ in loaded.members] == ["u"]
    with pytest.raises(ValueError, match="member tree does not load: ValueError"):
        Residual(fitted({"u": linear(), "tree": tree}, {"u": 1.0, "tree": 1.0}))


@pytest.mark.parametrize(
    "member,weight,match",
    [
        (linear(transform="log"), 1.0, "transform 'log' is not one of"),
        ({**linear(), "coefficients": {"tokens": 1.0}}, 1.0, "the recipe's features"),
        (linear(order=math.nan), 1.0, "coefficient that is not finite"),
        ({**linear(), "intercept": None}, 1.0, "TypeError"),
        ({k: v for k, v in linear().items() if k != "form"}, 1.0, "KeyError: 'form'"),
        (linear(), 0.0, "weight 0.0 is not positive and finite"),
    ],
    ids=["transform", "features", "nan", "intercept", "form", "weight"],
)
def test_a_selected_member_that_cannot_load_fails_loading_naming_it(
    member, weight, match
):
    with pytest.raises(ValueError, match=f"member m does not load: .*{match}"):
        Residual(fitted({"m": member}, {"m": weight}))


@pytest.mark.parametrize(
    "selected,hull,match",
    [
        ({}, None, "selects no member"),
        ({"ghost": 1.0}, None, "member ghost does not load: KeyError"),
        ({"u": 1.0}, {"facets": [[1, 0]]}, "facet not 6 wide"),
        ({"u": 1.0}, {"facets": []}, "with no facets"),
        ({"u": 1.0}, {"facets": [[0, 0, 0, 0, 1, -math.inf]]}, "not finite"),
        ({"u": 1.0}, {"equal": {"rows": 1}, "facets": []}, "not all among"),
    ],
    ids=["none", "absent", "width", "no-facets", "infinite", "unknown-equal"],
)
def test_a_residual_that_cannot_be_read_fails_loading(selected, hull, match):
    with pytest.raises(ValueError, match=match):
        Residual(fitted({"u": linear()}, selected, hull))
