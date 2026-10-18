# SPDX-License-Identifier: MIT
"""The step grid's residual: the lookup times a factor fitted over the rows' features
in batch order, one fit per grid kind, answering past the hull it was fitted on
by the same members, the step's source naming how far past.

Decode steps come from ATOM's `Scheduler` and are priced through the runner's
`forward` and `_group_step_seconds`. Each holds 32 rows caching prompts of 575
and 959 tokens, or 767, so every batch's equivalent shape is B = 32 at c = 767,
and its lookup in the eager grid is f(32, 767) = 1000 + 960 + 767 + 1534 =
4261 us. All three share tokens 1/32 and query_context 3/128; with k = N_KV,
they differ in

    batch         prompts                  order   raggedness
    homogeneous   767 x 32                 0       1
    grouped       575 x 16, then 959 x 16  1/64    5/4      (384 / 24576)
    alternating   575, 959, ... x 16       31/64   5/4      (31 x 384 / 24576)

A fitted residual is loaded the same way and prices the grouped and alternating
batches at the lookup times the factor its triples were planted with.
"""

import math
import re
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
from atom.compass.backends.step_grid import coordinates, equivalent_shape
from atom.compass.backends.step_grid_residual import (
    RECIPE,
    Recipe,
    Residual,
    band,
    fit,
    kind,
)

HOMOGENEOUS = [767] * 32
GROUPED = [575] * 16 + [959] * 16
ALTERNATING = [575, 959] * 16
# 0.75 + 0.5 x order + 0.25 x raggedness: 1, 1.0703125 and 1.3046875 above.
ORDERED = linear(0.75, order=0.5, raggedness=0.25)
FINAL = [point(1, 8, 192, time=3)]
MEMBERS = [("identity", "linear", "identity"), ("exp", "linear", "exp")]
MEMBERS += [("band", "band", None)]


def residual(hulls=None, **members):
    return Residual(fitted(members, dict.fromkeys(members, 1.0), hulls))


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
def test_decode_is_read_on_order_and_raggedness_in_row_order_and_no_query_square(
    rows, order, raggedness
):
    rows = tuple(RequestShape(*r) for r in rows)

    assert equivalent_shape(rows) == (32, 1, 767)
    assert RECIPE("decode", rows) == (
        Fraction(1, 32),
        Fraction(3, 128),
        order,
        raggedness,
    )


def test_a_prefill_chunk_is_read_on_query_square_too():
    # Rows (N_Q, N_KV) of (3, 3) then (5, 15): 34 and 9 + 75 token pairs,
    # a jump of 12 over 18 context tokens, and 2 x 15 / 18.
    rows = (RequestShape(3, 3, False), RequestShape(5, 15, False))
    for k in ("prefill_output", "prefill_no_output"):
        assert RECIPE(k, rows) == (
            Fraction(8, 1024),
            Fraction(34, 2**20),
            Fraction(84, 2**20),
            Fraction(2, 3),
            Fraction(5, 3),
        )
    # 1 + 2 x tokens: 1.125 on a 64-token middle chunk, 1.015625 on the final 8.
    backend = eager(
        residual=residual(t=linear(tokens=2.0)),
        output=FINAL,
        no_output=MIDDLE,
        unit="ms",
    )
    steps, refused = chunks(backend)
    assert steps == [(False, 1e-3 * 1.125)] * 3 + [(True, 3e-3 * 1.015625)]
    assert refused == ()


def test_each_kind_is_priced_by_its_own_fit_and_a_kind_it_lacks_refuses():
    mapping = fitted({"u": linear()}, {"u": 1.0})
    del mapping["kinds"]["decode"]
    mapping["kinds"]["prefill_output"]["members"]["u"]["intercept"] = 2.0
    mapping["kinds"]["prefill_no_output"]["members"]["u"]["intercept"] = 1.5
    backend = eager(
        residual=Residual(mapping), output=FINAL, no_output=MIDDLE, unit="ms"
    )

    assert chunks(backend) == ([(False, 1.5e-3)] * 3 + [(True, 6e-3)], ())
    assert price(GROUPED, backend) == (
        0.0,
        ("cost:step-grid: the residual fits no decode step",),
    )


# The coverage check asks the residual even of a step the grid refuses.
@pytest.mark.parametrize(
    "rows,produces_output,said",
    [
        ([(1, 9, True), (8, 8, False)], False, "a batch of prefill and decode rows"),
        ([(8, 8, False)], None, "a prefill batch that does not say whether it"),
    ],
    ids=["mixed", "unsaid"],
)
def test_a_step_of_no_grid_kind_is_refused_naming_why(rows, produces_output, said):
    v = view(*rows, produces_output=produces_output)

    with pytest.raises(CostRefused, match=said):
        residual(u=linear()).factor(v)


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
    assert backend.describe().endswith(
        "; recipe step-grid-residual v2, members decode: o, e; "
        "prefill_output: o, e; prefill_no_output: o, e"
    )


def source(prompts, backend):
    """The provenance of the one term `backend` prices these decode rows at."""
    rows = [(1, p + 1, True) for p in prompts]
    (term,) = backend.price(view(*rows)).terms
    return term.provenance


# Both decode hulls bound order <= 1/4; the second also holds raggedness at 1, so
# its facet has no raggedness entry. Alternating passes the first by its order,
# 31/64 - 1/4; grouped differs from the second's raggedness by 1/4.
@pytest.mark.parametrize(
    "hull,inside,outside,factor,past",
    [
        ({"facets": [[0, 0, 1, 0, -0.25]]}, GROUPED, ALTERNATING, 1.3046875, 0.234),
        (
            {"equal": {"raggedness": 1.0}, "facets": [[0, 0, 1, -0.25]]},
            HOMOGENEOUS,
            GROUPED,
            1.0703125,
            0.25,
        ),
    ],
    ids=["facet", "equal"],
)
def test_a_step_outside_the_residuals_hull_is_priced_by_its_members_naming_how_far(
    hull, inside, outside, factor, past
):
    backend = eager(residual=residual({"decode": hull}, o=ORDERED))

    assert price(inside, backend)[1] == ()
    assert source(inside, backend).source == "step-grid"
    assert price(outside, backend) == (pytest.approx(4261e-6 * factor), ())
    out = source(outside, backend)
    assert out.source == (
        f"step-grid-extrapolated: {past} past the residual's decode hull"
    )
    assert (out.method, out.validated, out.extrapolated) == (
        Method.ESTIMATED,
        False,
        True,
    )


def test_only_the_steps_outside_the_hull_are_extrapolated_each_at_its_members_price():
    # tokens <= 1/32: the three 64-token middle chunks are outside, the final 8 in.
    tokens = {"facets": [[1, 0, 0, 0, 0, -1 / 32]]}
    hulls = dict.fromkeys(("prefill_output", "prefill_no_output"), tokens)

    def backend(fitted):
        return eager(residual=fitted, output=FINAL, no_output=MIDDLE, unit="ms")

    bounded = backend(residual(hulls, u=linear()))
    steps, refused = chunks(bounded)
    assert (steps, refused) == chunks(backend(residual(u=linear())))
    assert refused == ()
    middle = [view((64, 64 * k, False), produces_output=False) for k in (1, 2, 3)]
    final = view((8, 200, False), produces_output=True)
    assert [
        bounded.price(v).terms[0].provenance.extrapolated for v in (*middle, final)
    ] == [True, True, True, False]


@pytest.mark.parametrize(
    "members,said",
    [
        # 1.0625 - 4 x order: 1 when grouped, -0.875 when alternating; the
        # other member answering does not rescue the step.
        (
            {"u": linear(), "falls": linear(1.0625, order=-4.0)},
            "decode member falls gives -0.875",
        ),
        (
            {"u": linear(), "big": linear(0.0, "exp", tokens=1e6)},
            "decode member big gives inf",
        ),
        ({"a": linear(1e308), "b": linear(1e308)}, "decode factor is inf"),
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


EDITED = {k: fs[:-1] + (raggedness,) for k, fs in RECIPE.features.items()}


def test_the_digest_is_over_the_feature_functions_source_and_each_kinds_names():
    edited = Recipe(RECIPE.name, RECIPE.version, EDITED)
    # Prefill with output reading decode's features: the same functions in the
    # same order, and another kind's names.
    moved = {**RECIPE.features, "prefill_output": RECIPE.features["decode"]}

    assert edited.names == RECIPE.names
    assert edited.digest != RECIPE.digest
    assert Recipe(RECIPE.name, RECIPE.version, moved).digest != RECIPE.digest
    assert Recipe(RECIPE.name, RECIPE.version, RECIPE.features).digest == RECIPE.digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("version", 1),
        ("digest", Recipe(RECIPE.name, RECIPE.version, EDITED).digest),
    ],
    ids=["name", "version", "digest"],
)
def test_a_residual_fitted_under_another_recipe_fails_loading(field, value):
    mapping = fitted({"u": linear()}, {"u": 1.0})
    mapping["recipe"][field] = value

    with pytest.raises(ValueError, match="fitted under recipe"):
        Residual(mapping)


@pytest.mark.parametrize("kinds", [[], ["decode", "mixed"]], ids=["none", "unknown"])
def test_a_residual_of_no_kind_or_another_fails_loading(kinds):
    mapping = fitted({"u": linear()}, {"u": 1.0})
    mapping["kinds"] = {k: mapping["kinds"]["decode"] for k in kinds}

    with pytest.raises(ValueError, match=re.escape(f"fits the kinds {kinds}, and")):
        Residual(mapping)


def test_a_member_the_residual_does_not_select_never_loads():
    tree = {"form": "tree", "nodes": []}
    loaded = Residual(fitted({"u": linear(), "tree": tree}, {"u": 1.0}))

    names = {k: [name for name, _, _ in f.members] for k, f in loaded.kinds.items()}
    assert names == {k: ["u"] for k in RECIPE.names}
    with pytest.raises(ValueError, match="decode member tree does not load: Value"):
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
    with pytest.raises(ValueError, match=f"decode member m does not load: .*{match}"):
        Residual(fitted({"m": member}, {"m": weight}))


@pytest.mark.parametrize(
    "selected,hull,match",
    [
        ({}, None, "selects no decode member"),
        ({"ghost": 1.0}, None, "decode member ghost does not load: KeyError"),
        ({"u": 1.0}, {"facets": [[1, 0]]}, "decode hull has a facet not 5 wide"),
        ({"u": 1.0}, {"facets": []}, "with no facets"),
        ({"u": 1.0}, {"facets": [[0, 0, 0, 1, -math.inf]]}, "not finite"),
        ({"u": 1.0}, {"equal": {"rows": 1}, "facets": []}, "not all among"),
    ],
    ids=["none", "absent", "width", "no-facets", "infinite", "unknown-equal"],
)
def test_a_residual_that_cannot_be_read_fails_loading(selected, hull, match):
    hulls = {"decode": hull} if hull else None
    with pytest.raises(ValueError, match=match):
        Residual(fitted({"u": linear()}, selected, hulls))


def features(v):
    return dict(zip(RECIPE.names[kind(v)], map(float, RECIPE(kind(v), v.requests))))


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
        "cost:step-grid: the residual's decode member b has no band decode rung 48",
    )
    assert price(GROUPED, eager(residual=banded)) == (4261e-6 * 1.5, ())
    prefill = eager(residual=banded, output=FINAL, no_output=MIDDLE, unit="ms")
    assert chunks(prefill) == ([(False, 2e-3)] * 3 + [(True, 6e-3)], ())
    # A batch holding a prefill row is read at its token count.
    assert band(view((1, 600, True), (511, 511, False))) == "prefill tokens 512-2048"


def test_a_top_band_held_as_lo_minus_prices_the_steps_past_the_last_edge():
    # A residual fitted when the top prefill band was spelled `8192-`.
    values = {"prefill tokens 2048-8192": 1.5, "prefill tokens 8192-": 1.25}
    fitted_before = residual(b={"form": "band", "values": values})
    top = view((8192, 8192, False), (1000, 1000, False), produces_output=False)
    below = view((4096, 4096, False), produces_output=False)

    assert band(top) == "prefill tokens 8192+"
    assert fitted_before.factor(top)[0] == 1.25
    assert fitted_before.factor(below)[0] == 1.5


def test_a_band_member_holding_the_top_band_in_both_spellings_fails_loading():
    both = {"prefill tokens 8192-": 1.25, "prefill tokens 8192+": 1.5}
    with pytest.raises(ValueError, match="member b does not load: .* both <lo>- and"):
        residual(b={"form": "band", "values": both})


def triples_of(factor, n=400, b=(24, 41), prefill=True, q=(8, 65)):
    """`n` triples: one in four of one to three prefill chunks of `q` tokens a
    row (none without `prefill`), alternately producing no output and output,
    the rest a decode batch of B in `b` rows reading two contexts, grouped,
    shuffled or alternating. Each is measured at its lookup times `factor` of
    its view, times 1 within 0.5%."""
    rng, triples = np.random.default_rng(0), []
    for i in range(n):
        output = None
        if prefill and not i % 4:
            q_rows, c = rng.integers(*q, 3), rng.integers(0, 193, 3)
            rows = [(int(t), int(t + k), False) for t, k in zip(q_rows, c)]
            rows, output = rows[: i % 3 + 1], bool(i % 8)
        else:
            rows_b = rng.integers(*b)
            lo, hi = rng.integers(512, 700), rng.integers(800, 1025)
            hi_rows = rng.integers(0, rows_b + 1)
            ks = [lo] * (rows_b - hi_rows) + [hi] * hi_rows
            ks = [ks, list(rng.permutation(ks)), ([lo, hi] * rows_b)[:rows_b]][i % 3]
            rows = [(1, int(k), True) for k in ks]
        v = view(*rows, produces_output=output)
        lookup = float(rng.uniform(1e-3, 5e-3))
        noise = 1.0 + rng.uniform(-0.005, 0.005)
        triples.append((v, lookup, lookup * factor(v) * noise))
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

    loaded = load_and_price(mapping, planted)
    held = {k: len(r["held_out"]) for k, r in report.items()}
    assert held == {"decode": 75, "prefill_output": 12, "prefill_no_output": 12}
    for k, r in report.items():
        ((_, weight, member),) = loaded.kinds[k].members
        assert weight == 1.0
        for i in r["held_out"]:
            v = triples[i][0]
            assert kind(v) == k
            assert member(v, tuple(features(v).values())) == pytest.approx(
                planted(v), rel=0.01
            )


def test_a_fit_on_decode_triples_alone_prices_atoms_decode_batches_and_no_prefill():
    # On decode rows alone query_square would be tokens / 1024, and a hull
    # over both would span nothing.
    mapping, report = fit(triples_of(planted, prefill=False), seed=0)

    assert list(mapping["kinds"]) == list(report) == ["decode"]
    loaded = load_and_price(mapping, planted)
    backend = eager(residual=loaded, output=FINAL, no_output=MIDDLE, unit="ms")
    no_output, output = (
        f"cost:step-grid: the residual fits no {k} step"
        for k in ("prefill_no_output", "prefill_output")
    )
    assert chunks(backend)[1] == (no_output,) * 3 + (output,)


def test_each_kind_is_fitted_on_its_own_triples_alone():
    triples = triples_of(planted)
    mapping, report = fit(triples, seed=0)

    for k in RECIPE.names:
        own = [t for t in triples if kind(t[0]) == k]
        alone, alone_report = fit(own, seed=0)
        assert alone["kinds"] == {k: mapping["kinds"][k]}
        assert [own[i] for i in alone_report[k]["held_out"]] == [
            triples[i] for i in report[k]["held_out"]
        ]


def curved(v):
    x = features(v)
    return math.exp(0.05 + 2 * x["order"] - x["raggedness"])


def straight(v):
    x = features(v)
    return 0.2 + 3 * x["order"] + 0.5 * x["raggedness"]


def stepped(v):
    return 1.1 if band(v) == "prefill tokens 0-512" else 1.3


def by_kind(v):
    """Curved on decode, straight on prefill with output, and stepped by token
    band on prefill without."""
    kinds = {"decode": curved, "prefill_output": straight}
    return kinds.get(kind(v), stepped)(v)


def test_each_kind_ranks_every_family_and_selects_its_lowest_held_out_error():
    # Prefill rows of 64 to 400 tokens, so chunks fall on both sides of 512.
    mapping, report = fit(triples_of(by_kind, q=(64, 401)), seed=0)

    load_and_price(mapping, by_kind)
    assert {k: fitted["selected"] for k, fitted in mapping["kinds"].items()} == {
        "decode": {"exp": 1.0},
        "prefill_output": {"identity": 1.0},
        "prefill_no_output": {"band": 1.0},
    }
    for k, fitted_kind in mapping["kinds"].items():
        members = fitted_kind["members"]
        assert [(n, m["form"], m.get("transform")) for n, m in members.items()] == (
            MEMBERS
        )
        assert list(members["exp"]["coefficients"]) == list(RECIPE.names[k])
        errors = {n: m["held_out_error"] for n, m in report[k]["members"].items()}
        assert {min(errors, key=errors.get): 1.0} == fitted_kind["selected"]
    stepped_bands = mapping["kinds"]["prefill_no_output"]["members"]["band"]
    assert list(stepped_bands["values"]) == [
        "prefill tokens 0-512",
        "prefill tokens 512-2048",
    ]


def test_outliers_are_dropped_until_settled_and_the_hull_is_the_kept_views():
    triples = triples_of(planted)
    held = {i for r in fit(triples, seed=0)[1].values() for i in r["held_out"]}
    dev = {
        k: [i for i, (v, _, _) in enumerate(triples) if kind(v) == k and i not in held]
        for k in RECIPE.names
    }
    # Five fitted decode triples measured 1.5x slow, one a 64-row batch no
    # other reaches.
    wide = view(*[(1, 700, True)] * 64)
    triples[dev["decode"][0]] = (wide, 1e-3, 1.5e-3 * planted(wide))
    for i in dev["decode"][1:5]:
        v, lookup, measured = triples[i]
        triples[i] = (v, lookup, 1.5 * measured)
    # Two 600-token chunks producing output alone in their band, measured 1x
    # and 2x: the band member drops both and holds no factor for their band.
    for i, (rows, slow) in zip(
        dev["prefill_output"], [((600, 600, False), 1), ((600, 700, False), 2)]
    ):
        v = view(rows, produces_output=True)
        triples[i] = (v, 1e-3, slow * 1e-3 * planted(v))
    mapping, report = fit(triples, seed=0)

    counts = {
        n: (m["samples"], m["dropped"]) for n, m in report["decode"]["members"].items()
    }
    assert counts == dict.fromkeys(("identity", "exp", "band"), (220, 5))
    output = report["prefill_output"]["members"]["band"]
    assert output["zero"] == ["prefill tokens 512-2048"]
    banded = mapping["kinds"]["prefill_output"]["members"]["band"]["values"]
    assert "prefill tokens 512-2048" not in banded
    *_, past = load_and_price(mapping, planted).factor(wide)
    assert past.endswith("past the residual's decode hull")


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
    held = fit(triples, seed=0)[1]["prefill_output"]["held_out"]
    # A held-out 600-token chunk producing output, in a band no fitted triple
    # of its kind reaches.
    unseen = view((600, 600, False), produces_output=True)
    triples[held[0]] = (unseen, 1e-3, 1e-3 * planted(unseen))
    mapping, report = fit(triples, seed=0)

    for k, by_member in report.items():
        held = by_member["held_out"]
        dev = [i for i, (v, _, _) in enumerate(triples) if kind(v) == k]
        dev = [i for i in dev if i not in held]
        for name, reported in by_member["members"].items():
            one = {**mapping["kinds"][k], "selected": {name: 1.0}}
            loaded = Residual({**mapping, "kinds": {k: one}})
            ((_, _, member),) = loaded.kinds[k].members
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
    held = fitted_once[1]["decode"]["held_out"]
    assert fit(triples, seed=1)[1]["decode"]["held_out"] != held
    load_and_price(fitted_once[0], planted)


def test_a_feature_constant_over_the_fit_is_zeroed_and_held_at_its_value():
    # Every batch is 32 decode rows: tokens 1/32.
    mapping, report = fit(triples_of(planted, b=(32, 33), prefill=False), seed=0)

    load_and_price(mapping, planted)
    decode_fit = mapping["kinds"]["decode"]
    assert decode_fit["hull"]["equal"] == {"tokens": 1 / 32}
    for name in ("identity", "exp"):
        assert report["decode"]["members"][name]["zero"] == ["tokens"]
        assert decode_fit["members"][name]["coefficients"]["tokens"] == 0.0


@pytest.mark.parametrize("n, spans", [(8, False), (12, True)])
def test_held_out_batches_are_validated_only_when_their_coordinates_span_a_hull(
    n, spans
):
    """8 decode triples hold out 2, which differ in B and c and span no hull; 12
    hold out 3, which do."""
    triples = triples_of(planted, n=n, prefill=False)
    mapping, report = fit(triples, seed=0)

    assert ("held_out" in mapping["kinds"]["decode"]) is spans
    loaded = Residual(mapping)
    for i in report["decode"]["held_out"]:
        v = triples[i][0]
        assert loaded.validated(v, coordinates(v.requests, prefill=False)) is spans


def replaced(triples, v):
    return triples[:3] + [(v, 1e-3, 1e-3)] + triples[4:]


def negated(triples):
    v, lookup, measured = triples[3]
    return triples[:3] + [(v, -lookup, -measured)] + triples[4:]


@pytest.mark.parametrize(
    "triples,match",
    [
        (
            lambda: triples_of(planted)[:8],
            "the decode fit: 6 triples hold out 1 and leave 5 to fit 5",
        ),
        (
            lambda: negated(triples_of(planted)),
            r"triples \[3\] have a time that is not",
        ),
        (
            lambda: replaced(triples_of(planted), view((1, 9, True), (8, 8, False))),
            "triple 3: a batch of prefill and decode rows is of no grid kind",
        ),
        (
            lambda: replaced(triples_of(planted), view((8, 8, False))),
            "triple 3: a prefill batch that does not say whether it produces",
        ),
        # Decode rows at one context: order and raggedness are constant, and
        # query_context is tokens times that context.
        (
            lambda: [
                (view(*[(1, 700, True)] * b), 1e-3, 1e-3)
                for b in list(range(24, 41)) * 2
            ],
            "the decode fit: the kept views' features span no hull",
        ),
    ],
    ids=["few", "time", "mixed", "unsaid", "collinear"],
)
def test_a_fit_that_cannot_be_made_is_refused_naming_why(triples, match):
    with pytest.raises(ValueError, match=match):
        fit(triples(), seed=0)
