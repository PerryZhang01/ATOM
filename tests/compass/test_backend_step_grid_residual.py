# SPDX-License-Identifier: MIT
"""The step grid's residual: the lookup times a factor fitted over the rows' features
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

A fitted residual is loaded the same way and prices the grouped and alternating
batches at the lookup times the factor its triples were planted with.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from test_backend_step_grid import (
    B37,
    MIDDLE,
    chunks,
    decode,
    eager,
    fitted,
    graphed,
    linear,
    point,
    replay,
    view,
)
from test_dp_step_max import runner

from atom.compass.backends import CostRefused, Method, RequestShape
from atom.compass.backends.shape import sum_context
from atom.compass.backends.step_grid import equivalent_shape
from atom.compass.backends.step_grid_residual import V1, Recipe, Residual, band, fit

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

    (term,) = backend.price(view(*rows)).terms
    factor = 1.3046875 + math.exp(0.484375)
    assert term.seconds == 4261e-6 * factor
    assert term.provenance.method is Method.ESTIMATED
    assert term.provenance.detail == (
        "lookup 0.004261 s, interpolated; decode cell B=32, c 512..1024; "
        f"residual {factor!r} = 1.0 x o 1.3046875 + 1.0 x e {math.exp(0.484375)!r}"
    )
    assert backend.describe().endswith("; recipe step-grid-residual v1, members o, e")


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
                "cost:step-grid: the step at tokens=0.03125, "
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
        f"cost:step-grid: the residual's {said}, which is not a positive finite factor",
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
        ({"form": "band", "values": {}}, 1.0, "no band"),
        ({"form": "band", "values": {"a": -1.0}}, 1.0, "band factor that is not"),
    ],
    ids=["transform", "features", "nan", "intercept", "form", "weight", "none", "band"],
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


def features(v):
    return dict(zip(V1.names, map(float, V1(v.requests))))


def test_a_band_member_prices_decode_by_rung_and_prefill_by_token_band():
    values = {
        "decode rung 32": 1.25,
        "decode rung eager": 1.5,
        "prefill tokens 0-512": 2,
    }
    banded = residual(b={"form": "band", "values": values})
    r = graphed(replay(residual=banded))

    # GROUPED replays rung 32: f(32, 767) + 3200; B37 replays rung 48.
    assert r.forward(decode(GROUPED)).predicted_s == 7461e-6 * 1.25
    assert r.forward(decode(B37)).predicted_s == 0.0
    assert r.refused_commands() == (
        "cost:step-grid: the residual's member b has no band decode rung 48",
    )
    assert price(GROUPED, eager(residual=banded)) == (4261e-6 * 1.5, ())
    final = [point(1, 8, 192, time=3)]
    prefill = eager(residual=banded, output=final, no_output=MIDDLE, unit="ms")
    assert chunks(prefill) == ([(False, 2e-3)] * 3 + [(True, 6e-3)], ())
    # A batch holding a prefill row is read at its token count.
    assert band(view((1, 600, True), (511, 511, False))) == "prefill tokens 512-2048"


def triples_of(factor, n=400, b=(24, 41), prefill=True):
    """`n` triples: one in four of one to three prefill chunks (none without
    `prefill`), the rest a decode batch of B in `b` rows reading two contexts,
    grouped, shuffled or alternating. Each is measured at its lookup times
    `factor` of its view, times 1 within 0.5%."""
    rng, triples = np.random.default_rng(0), []
    for i in range(n):
        if prefill and not i % 4:
            q, c = rng.integers(8, 65, 3), rng.integers(0, 193, 3)
            rows = [(int(q[j]), int(q[j] + c[j]), False) for j in range(i % 3 + 1)]
        else:
            rows_b = rng.integers(*b)
            lo, hi = rng.integers(512, 700), rng.integers(800, 1025)
            hi_rows = rng.integers(0, rows_b + 1)
            ks = [lo] * (rows_b - hi_rows) + [hi] * hi_rows
            ks = [ks, list(rng.permutation(ks)), ([lo, hi] * rows_b)[:rows_b]][i % 3]
            rows = [(1, int(k), True) for k in ks]
        lookup = float(rng.uniform(1e-3, 5e-3))
        noise = 1.0 + rng.uniform(-0.005, 0.005)
        triples.append((view(*rows), lookup, lookup * factor(view(*rows)) * noise))
    return triples


def load_and_price(mapping, factor):
    """Load `mapping`, and price ATOM's grouped and alternating batches with it."""
    loaded = Residual(mapping)
    for prompts in (GROUPED, ALTERNATING):
        expected = 4261e-6 * factor(view(*[(1, p + 1, True) for p in prompts]))
        assert price(prompts, eager(residual=loaded)) == (
            pytest.approx(expected, rel=0.01),
            (),
        )
    return loaded


def planted(v):
    x = features(v)
    return math.exp(0.05 + 0.2 * x["order"] - 0.1 * x["raggedness"])


def test_the_fit_recovers_a_planted_factor_within_1pc_on_held_out_views():
    triples = triples_of(planted)
    mapping, report = fit(triples, seed=0)

    ((_, weight, member),) = load_and_price(mapping, planted).members
    assert weight == 1.0 and len(report["held_out"]) == 100
    for i in report["held_out"]:
        v = triples[i][0]
        assert member(v, tuple(features(v).values())) == pytest.approx(
            planted(v), rel=0.01
        )


def curved(v):
    x = features(v)
    return math.exp(0.05 + 2 * x["order"] - x["raggedness"])


def straight(v):
    x = features(v)
    return 0.2 + 3 * x["order"] + 0.5 * x["raggedness"]


def stepped(v):
    return 1.1 if band(v) == "decode rung eager" else 1.3


@pytest.mark.parametrize(
    "factor,selected",
    [(curved, "exp"), (straight, "identity"), (stepped, "band")],
    ids=["exp", "identity", "band"],
)
def test_every_member_is_kept_and_the_lowest_held_out_error_is_selected(
    factor, selected
):
    mapping, report = fit(triples_of(factor), seed=0)

    load_and_price(mapping, factor)
    assert mapping["selected"] == {selected: 1.0}
    members = mapping["members"]
    assert [(n, m["form"], m.get("transform")) for n, m in members.items()] == [
        ("identity", "linear", "identity"),
        ("exp", "linear", "exp"),
        ("band", "band", None),
    ]
    assert list(members["exp"]["coefficients"]) == list(V1.names)
    assert list(members["band"]["values"]) == [
        "decode rung eager",
        "prefill tokens 0-512",
    ]
    errors = {n: m["held_out_error"] for n, m in report["members"].items()}
    assert min(errors, key=errors.get) == selected


def test_outliers_are_dropped_until_settled_and_the_hull_is_the_kept_views():
    triples = triples_of(planted)
    held = set(fit(triples, seed=0)[1]["held_out"])
    dev = [i for i in range(len(triples)) if i not in held]
    # Five fitted triples measured 1.5x slow, one a 64-row batch no other reaches.
    wide = view(*[(1, 700, True)] * 64)
    triples[dev[0]] = (wide, 1e-3, 1.5e-3 * planted(wide))
    for i in dev[1:5]:
        v, lookup, measured = triples[i]
        triples[i] = (v, lookup, 1.5 * measured)
    # Two 600-token prefill chunks alone in their band, measured 1x and 2x:
    # every member drops both, and the band member holds no factor for them.
    for i, (rows, slow) in zip(
        dev[5:7], [((600, 600, False), 1), ((600, 700, False), 2)]
    ):
        triples[i] = (view(rows), 1e-3, slow * 1e-3 * planted(view(rows)))
    mapping, report = fit(triples, seed=0)

    counts = {n: (m["samples"], m["dropped"]) for n, m in report["members"].items()}
    assert counts == dict.fromkeys(("identity", "exp", "band"), (293, 7))
    assert report["members"]["band"]["zero"] == ["prefill tokens 512-2048"]
    assert "prefill tokens 512-2048" not in mapping["members"]["band"]["values"]
    with pytest.raises(CostRefused, match="outside the residual's hull"):
        load_and_price(mapping, planted).factor(wide)


def relative_error(member, triple):
    """`member`'s output over `triple`'s target, less 1; 0 output in a band it
    holds no factor for, an error of 1."""
    v, lookup, measured = triple
    try:
        output = member(v, tuple(features(v).values()))
    except LookupError:
        output = 0.0
    return output / (measured / lookup) - 1.0


def test_the_report_is_each_members_spread_where_fitted_and_error_where_held_out():
    triples = triples_of(planted)
    held = fit(triples, seed=0)[1]["held_out"]
    dev = [i for i in range(len(triples)) if i not in held]
    # A held-out 600-token chunk, in a band no fitted triple reaches.
    unseen = view((600, 600, False))
    triples[held[0]] = (unseen, 1e-3, 1e-3 * planted(unseen))
    mapping, report = fit(triples, seed=0)

    for name, reported in report["members"].items():
        ((_, _, member),) = Residual({**mapping, "selected": {name: 1.0}}).members
        r = np.array([relative_error(member, triples[i]) for i in dev])
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        held_out = [abs(relative_error(member, triples[i])) for i in held]
        assert (reported["centre"], reported["mad"]) == pytest.approx(
            (np.median(r), mad), rel=1e-9, abs=1e-12
        )
        assert reported["held_out_error"] == pytest.approx(
            np.median(held_out), rel=1e-9
        )


def test_the_target_is_measured_over_lookup_and_the_seed_fixes_the_split():
    triples = triples_of(planted)
    fitted_once = fit(triples, seed=0)

    assert fit(triples, seed=0) == fitted_once
    assert fit([(v, 2 * lo, 2 * m) for v, lo, m in triples], seed=0) == fitted_once
    assert fit(triples, seed=1)[1]["held_out"] != fitted_once[1]["held_out"]
    load_and_price(fitted_once[0], planted)


def test_a_feature_constant_over_the_fit_is_zeroed_and_held_at_its_value():
    # Every batch is 32 decode rows: tokens 1/32 and query_square 1/32768.
    mapping, report = fit(triples_of(planted, b=(32, 33), prefill=False), seed=0)

    load_and_price(mapping, planted)
    assert mapping["hull"]["equal"] == {"tokens": 1 / 32, "query_square": 1 / 32768}
    for name in ("identity", "exp"):
        assert report["members"][name]["zero"] == ["tokens", "query_square"]
        coefficients = mapping["members"][name]["coefficients"]
        assert coefficients["tokens"] == coefficients["query_square"] == 0.0


def negated(triples):
    v, lookup, measured = triples[3]
    return triples[:3] + [(v, -lookup, -measured)] + triples[4:]


@pytest.mark.parametrize(
    "triples,match",
    [
        (lambda: triples_of(planted)[:8], "8 triples hold out 2 and leave 6 to fit 6"),
        (
            lambda: negated(triples_of(planted)),
            r"triples \[3\] have a time that is not",
        ),
        # Decode rows only: query_square is tokens / 1024 on every batch.
        (
            lambda: triples_of(planted, prefill=False),
            "kept views' features span no hull",
        ),
    ],
    ids=["few", "time", "flat"],
)
def test_a_fit_that_cannot_be_made_is_refused_naming_why(triples, match):
    with pytest.raises(ValueError, match=match):
        fit(triples(), seed=0)
