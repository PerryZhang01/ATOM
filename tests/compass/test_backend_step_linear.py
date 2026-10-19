# SPDX-License-Identifier: MIT
"""The step linear law: a step priced as its features times fitted coefficients.

Pricing is checked against `StepStubBackend`, which reads the same features:
a law holding the stub's coefficients, with the stub's decode request term
zero, prices every step to the same seconds. Steps reach the law through the
runner's own `forward` and `_group_step_seconds`, and the law reaches the
runner through `run.runner` from a run file's `step_linear` key.
"""

import dataclasses
import json

import numpy as np
import pytest
import test_run_graph_pool
from test_dp_step_max import prefill, runner
from test_memory_readings import DOCUMENT, META_8B
from test_run_graph_pool import _reserved
from test_vertical_slice import _run_file

from atom.compass.artifacts import machine_spec_digest
from atom.compass.backends import (
    BatchView,
    Coefficients,
    CostRefused,
    Granularity,
    Method,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
    StepStubBackend,
)
from atom.compass.spec import MachineSpec

STAGE1 = Coefficients(decode_request=0.0)

# Prefill over (tokens, query_square) with nothing cached: the polygon
# tokens >= 1, query_square >= 1, the chord of t^2 from t = 1 to 4096, t <= 4096.
PREFILL_HULL = {
    "equal": {"query_cached": 0},
    "facets": [[-1, 0, 1], [0, -1, 1], [-4097, 1, 4096], [1, 0, -4096]],
}
# Decode at rung 2 over (context, graph_padding): the triangle with vertices
# (0, 0), (1000, 0) and (0, 1000). (600, 600) is inside its bounding box.
TRIANGLE = {"facets": [[-1, 0, 0], [0, -1, 0], [1, 1, -1000]]}
# Decode replaying no graph, whose padding is always zero.
EAGER_HULL = {"equal": {"graph_padding": 0}, "facets": [[-1, 1], [1, -100000]]}


def law(coefficients=STAGE1, prefill_hull=PREFILL_HULL, **decode_hulls):
    c = coefficients
    decode = {"step": c.decode_step, "context": c.decode_context}
    decode["graph_padding"] = c.decode_padding
    return {
        "provenance": "the stage-1 coefficients",
        "machine_spec": machine_spec_digest(MachineSpec.from_mapping(DOCUMENT)),
        "prefill": {
            "coefficients": {
                "step": c.prefill_step,
                "tokens": c.prefill_token,
                "query_square": c.prefill_query_square,
                "query_cached": c.prefill_query_cached,
            },
            "hull": prefill_hull,
        },
        "decode": {
            rung: {"coefficients": decode, "hull": hull}
            for rung, hull in (
                {"eager": EAGER_HULL, "2": TRIANGLE} | decode_hulls
            ).items()
        },
    }


def backend(**kwargs):
    return StepLinearBackend(StepLinearLaw(law(**kwargs)))


def view(*rows, rung=None):
    return BatchView(tuple(RequestShape(*r) for r in rows), capture_rung=rung)


IN_HULL = [
    view((40, 40, False)),
    view((4096, 4096, False)),
    view((7, 7, False), (300, 300, False)),
    view((1, 900, True)),
    view((1, 300, True), (1, 100, True), rung=2),
    view((1, 400, True), rung=2),
]


@pytest.mark.parametrize("batch", IN_HULL, ids=range(len(IN_HULL)))
def test_a_step_in_its_hull_prices_as_the_stub_with_one_estimated_term_per_feature(
    batch,
):
    cost = backend().price(batch)

    assert cost.seconds == StepStubBackend(STAGE1).price(batch).seconds
    kind = "prefill" if batch.prefill else "decode"
    features = (
        ["step", "tokens", "query_square", "query_cached"]
        if batch.prefill
        else ["step", "context", "graph_padding"]
    )
    assert [t.name for t in cost.terms] == [f"{kind}.{f}" for f in features]
    assert {t.provenance.method for t in cost.terms} == {Method.ESTIMATED}
    assert {t.provenance.source for t in cost.terms} == {"step-linear"}
    assert not any(t.provenance.validated for t in cost.terms)
    assert not cost.fell_back


def test_a_term_is_its_count_times_its_coefficient():
    step, context, padding = backend().price(view((1, 400, True), rung=2)).terms

    assert (step.seconds, context.seconds, padding.seconds) == (
        STAGE1.decode_step,
        400 * STAGE1.decode_context,
        400 * STAGE1.decode_padding,
    )
    assert padding.provenance.detail.startswith(f"{STAGE1.decode_padding:g} s x 400;")
    assert backend().granularity is Granularity.STEP


@pytest.mark.parametrize(
    "batch,hull",
    [
        (view((1, 600, True), (1, 600, True), rung=2), "decode rung 2"),
        (view((8, 40, False)), "prefill"),
        (view((5000, 5000, False)), "prefill"),
        (view((1, 200000, True)), "decode rung eager"),
    ],
    ids=["inside-the-box-outside-the-triangle", "cached-not-in-fit", "long", "deep"],
)
def test_a_step_outside_its_hull_is_priced_by_the_same_law_extrapolated(batch, hull):
    cost = backend().price(batch)

    assert cost.seconds == StepStubBackend(STAGE1).price(batch).seconds
    assert {t.provenance.method for t in cost.terms} == {Method.ESTIMATED}
    assert {t.provenance.source for t in cost.terms} == {"step-linear-extrapolated"}
    assert all(
        t.provenance.detail.endswith(f"outside the {hull} hull") for t in cost.terms
    )
    assert not cost.fell_back


# The square (0, 0) to (2000, 2000) over (context, graph_padding): it holds
# TRIANGLE and two rows of 600 context, (1200, 0), and not two of 1500.
SQUARE = {"facets": [[-1, 0, 0], [0, -1, 0], [1, 0, -2000], [0, 1, -2000]]}


@pytest.mark.parametrize(
    "batch,validated,source",
    [
        (view((1, 400, True), rung=2), True, "step-linear"),
        (
            view((1, 600, True), (1, 600, True), rung=2),
            True,
            "step-linear-extrapolated",
        ),
        (
            view((1, 1500, True), (1, 1500, True), rung=2),
            False,
            "step-linear-extrapolated",
        ),
        (view((1, 900, True)), False, "step-linear"),
    ],
    ids=[
        "inside-both",
        "outside-the-fit-inside-held-out",
        "outside-both",
        "no-held-out",
    ],
)
def test_a_term_is_validated_iff_its_step_is_inside_its_vectors_held_out_hull(
    batch, validated, source
):
    written = law()
    written["decode"]["2"]["held_out"] = {"hull": SQUARE, "bands": {}}
    cost = StepLinearBackend(StepLinearLaw(written)).price(batch)

    assert {t.provenance.validated for t in cost.terms} == {validated}
    assert {t.provenance.source for t in cost.terms} == {source}


def test_a_rung_with_no_vector_is_refused_by_name():
    with pytest.raises(CostRefused, match="no decode vector for rung 4"):
        backend().price(view((1, 10, True), rung=4))
    no_eager = law()
    del no_eager["decode"]["eager"]
    with pytest.raises(CostRefused, match="no decode vector for rung eager"):
        StepLinearBackend(StepLinearLaw(no_eager)).price(view((1, 10, True)))


def test_a_step_mixing_prefill_and_decode_rows_is_refused_by_name():
    with pytest.raises(CostRefused, match="1 prefill and 1 decode rows"):
        backend().price(view((8, 8, False), (1, 10, True)))


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda m: m.pop("provenance"), "provenance"),
        (lambda m: m["prefill"]["coefficients"].pop("tokens"), "coefficients for"),
        (
            lambda m: m["decode"]["2"].update(
                coefficients={"step": 1.0, "context": -1e-9, "graph_padding": 0.0}
            ),
            "context coefficient is -1e-09",
        ),
        (lambda m: m["decode"]["2"].update(hull={"facets": []}), "no facets"),
        (
            lambda m: m["decode"]["2"].update(
                hull={"equal": {"step": 1}, "facets": []}
            ),
            "not all among its features",
        ),
        # Three-wide facets over four varying features: 12 entries, which
        # reshape into three four-wide rows.
        (
            lambda m: m["prefill"].update(hull={"facets": PREFILL_HULL["facets"]}),
            "facet not 4 wide",
        ),
    ],
    ids=[
        "provenance",
        "missing-feature",
        "negative",
        "unbounded",
        "intercept",
        "facet-width",
    ],
)
def test_a_law_that_cannot_price_is_refused_when_read(edit, match):
    mapping = law()
    edit(mapping)
    with pytest.raises(ValueError, match=match):
        StepLinearLaw(mapping)


def test_the_runner_prices_its_step_through_the_law(monkeypatch):
    import torch

    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    rows = (RequestShape(40, 40, False),)
    stub = StepStubBackend(STAGE1).price(BatchView(rows)).seconds

    assert runner(backend()).forward(prefill(40)).predicted_s == stub
    # ATOM's dummy decode batch, replaying no graph on an eager runner.
    eager = runner(backend()).dummy_execution().predicted_s
    assert eager == StepStubBackend(STAGE1).price(view((1, 1, True))).seconds


def graphed(b):
    """A runner that replays graphs of widths 1 and 2, priced by `b`."""
    r = runner(b)
    r.capture_sizes_np, r.enforce_eager = np.array([1, 2], dtype=np.int32), False
    return r


def test_the_runner_prices_a_replayed_rung_from_its_own_vector():
    priced = graphed(backend(**{"1": EAGER_HULL})).dummy_execution().predicted_s

    assert priced == STAGE1.decode_step + STAGE1.decode_context
    refused = graphed(backend())
    assert refused.dummy_execution().predicted_s == 0.0
    assert refused.refused_commands() == (
        "cost:step-linear: no decode vector for rung 1",
    )


def _without_step_stub(out, **keys):
    path = _run_file(out, **keys)
    run = json.loads(path.read_text())
    if "step_stub" not in keys:
        del run["step_stub"]
    path.write_text(json.dumps(run))
    return path


def _installed(monkeypatch, tmp_path, **keys):
    """The backend `run.runner` installs from a run file with `keys`."""
    monkeypatch.setattr(test_run_graph_pool, "_run_file", _without_step_stub)
    installed = []
    from atom.compass.runner import overrides

    real = overrides.install_cost_backend

    def install(r, b):
        installed.append(b)
        real(r, b)

    monkeypatch.setattr(overrides, "install_cost_backend", install)
    _reserved(monkeypatch, tmp_path, eager=True, **keys)
    return installed[0]


def test_a_run_file_step_linear_installs_the_step_linear_backend(monkeypatch, tmp_path):
    installed = _installed(monkeypatch, tmp_path, step_linear=law())

    assert isinstance(installed, StepLinearBackend)
    assert installed.price(view((40, 40, False))).seconds == (
        StepStubBackend(STAGE1).price(view((40, 40, False))).seconds
    )


def test_a_run_file_step_stub_installs_the_stub(monkeypatch, tmp_path):
    coefficients = dataclasses.asdict(STAGE1)
    installed = _installed(monkeypatch, tmp_path, step_stub=coefficients)

    assert isinstance(installed, StepStubBackend)


@pytest.mark.parametrize(
    "keys,named",
    [
        (
            {"step_linear": law(), "step_stub": dataclasses.asdict(STAGE1)},
            "step_stub and step_linear",
        ),
        ({}, "none"),
    ],
    ids=["both", "none"],
)
def test_a_run_file_with_both_keys_or_neither_is_refused(
    monkeypatch, tmp_path, keys, named
):
    with pytest.raises(ValueError, match=f"this one has {named}"):
        _installed(monkeypatch, tmp_path, **keys)


@pytest.mark.parametrize(
    "old,new",
    [("law", "step_linear"), ("coefficients", "step_stub"), ("step_fit", "step_grid")],
)
def test_a_run_file_with_a_renamed_key_is_refused_naming_its_name_now(
    monkeypatch, tmp_path, old, new
):
    with pytest.raises(ValueError, match=rf"renamed: {old} \(now {new}\)"):
        _installed(monkeypatch, tmp_path, **{old: {}})


def test_a_run_file_step_linear_is_refused_above_tensor_parallel_width_one(monkeypatch):
    from types import SimpleNamespace

    from atom.compass import run

    monkeypatch.setattr(run, "spec", lambda: {"step_linear": law()})
    tp2 = SimpleNamespace(config=SimpleNamespace(tensor_parallel_size=2))
    with pytest.raises(ValueError, match="tensor_parallel_size 2"):
        run.runner(tp2, META_8B)


def test_a_run_file_step_linear_stating_no_engine_config_is_refused(monkeypatch):
    from types import SimpleNamespace

    from atom.compass import run

    monkeypatch.setattr(run, "spec", lambda: {"step_linear": law()})
    tp1 = SimpleNamespace(
        config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_dp_attention=False,
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )
    )
    with pytest.raises(ValueError, match="the step linear law states no engine_config"):
        run.runner(tp1, META_8B)
