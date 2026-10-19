# SPDX-License-Identifier: MIT
"""What the run summary says about the steps a run priced.

Steps are priced through the runner's own `forward`, the worker's answers are
read back as `engine_done` reads them over the worker RPC, and the frontend's
`frontend_done` writes ``summary.json``.
"""

import json
import math
import time
from types import SimpleNamespace

import pytest
import torch
from test_backend_step_linear import law
from test_dp_step_max import prefill, runner
from test_runner_control_refusals import Worker

from atom.compass import run as compass_run
from atom.compass.backends import (
    BatchView,
    Coefficients,
    CostTerm,
    Method,
    Provenance,
    Refusal,
    RequestShape,
    StepCost,
    StepLinearBackend,
    StepLinearLaw,
    StepStubBackend,
)
from atom.compass.backends.cost import GROUPS
from atom.utils import clock
from atom.utils.clock import LPRuntime

STAGE1 = Coefficients(decode_request=0.0)
CHAIN = (
    Refusal("price_list", "no entry for this shape"),
    Refusal("nearest_key", "outside the measured range"),
)
# Prefill of 1 to 32 tokens with nothing cached: the chord of t^2 from 1 to 32.
SHORT_PREFILL_HULL = {
    "equal": {"query_cached": 0},
    "facets": [[-1, 0, 1], [0, -1, 1], [-33, 1, 32], [1, 0, -32]],
}
NONE_APART = {"steps": 0, "fraction_of_predicted_seconds": 0.0}


def by_method(**groups):
    """`seconds_by_method` with `groups` and every other group at none."""
    return {g: groups.get(g, NONE_APART) for g in GROUPS}


class AnalyticalBelow(StepStubBackend):
    """The stub's seconds, answered by an analytical source after two sources above
    it declined a prefill of `tokens` or more."""

    def __init__(self, coefficients, tokens):
        super().__init__(coefficients)
        self.tokens = tokens

    def price(self, batch_view):
        cost = super().price(batch_view)
        if sum(r.query_tokens for r in batch_view.prefill) < self.tokens:
            return cost
        analytical = Provenance(Method.ANALYTICAL, "roofline", "class0", CHAIN)
        return StepCost(
            [CostTerm(t.name, t.seconds, analytical) for t in cost.terms if t.seconds]
        )


def seconds(backend, tokens):
    return backend.price(BatchView((RequestShape(tokens, tokens, False),))).seconds


def summary(
    monkeypatch, tmp_path, backend, *prompts, clock_refusals=(), jobs=(), scheduler=None
):
    """The run summary after one engine's runner priced a prefill of each
    prompt, with the engine's refused clock calls and the frontend's refused
    executor jobs given, and `scheduler` as the engine's scheduler."""
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    table = compass_run.channel_table(run)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()

    r = runner(backend)
    for tokens in prompts:
        r.forward(prefill(tokens))
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    engine.refusals.extend(clock_refusals)
    monkeypatch.setattr(clock, "_installed", engine)
    compass_run.engine_done(SimpleNamespace(runner_mgr=Worker(r), scheduler=scheduler))

    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=list(jobs)))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())


def test_a_step_a_later_source_answered_is_not_refused_and_its_answer_is_apart(
    monkeypatch, tmp_path
):
    backend = AnalyticalBelow(STAGE1, tokens=40)
    out = summary(monkeypatch, tmp_path, backend, 8, 8, 8, 40)

    short, long = seconds(backend, 8), seconds(backend, 40)
    schedule = out["schedule"]
    assert schedule["predicted_seconds"] == pytest.approx(3 * short + long)
    assert schedule["seconds_by_method"] == by_method(
        analytical={
            "steps": 1,
            "fraction_of_predicted_seconds": pytest.approx(long / (3 * short + long)),
        },
        declared={
            "steps": 3,
            "fraction_of_predicted_seconds": pytest.approx(
                3 * short / (3 * short + long)
            ),
        },
    )
    refusals = schedule["refusals"]
    assert (refusals["count"], refusals["steps"], refusals["reasons"]) == (0, 4, [])
    assert refusals["fraction_of_steps"] == 0.0
    assert out["coverage_report"] is False


@pytest.mark.parametrize(
    "where,reason",
    [
        ("clock_refusals", "clock:sleep from engine"),
        ("jobs", "executor:a job past the bound"),
    ],
)
def test_any_refusal_of_any_source_is_a_coverage_report(
    monkeypatch, tmp_path, where, reason
):
    backend = StepStubBackend(STAGE1)
    out = summary(monkeypatch, tmp_path, backend, 8, 40, **{where: [reason]})

    refusals = out["schedule"]["refusals"]
    assert (refusals["count"], refusals["reasons"]) == (1, [[reason, 1]])
    assert refusals["fraction_of_steps"] == 0.0
    assert out["coverage_report"] is True


@pytest.mark.parametrize("held_out", [False, True], ids=["no-held-out", "held-out"])
def test_steps_outside_the_held_out_hull_are_unvalidated_and_never_refused(
    monkeypatch, tmp_path, held_out
):
    written = law(prefill_hull=SHORT_PREFILL_HULL)
    if held_out:
        written["prefill"]["held_out"] = {"hull": SHORT_PREFILL_HULL, "bands": {}}
    backend = StepLinearBackend(StepLinearLaw(written))
    out = summary(monkeypatch, tmp_path, backend, 8, 8, 8, 40)

    short, long = seconds(backend, 8), seconds(backend, 40)
    schedule = out["schedule"]
    unvalidated = {
        "steps": 1,
        "fraction_of_predicted_seconds": pytest.approx(long / (3 * short + long)),
    }
    estimated = {
        "steps": 3,
        "fraction_of_predicted_seconds": pytest.approx(3 * short / (3 * short + long)),
    }
    if not held_out:
        unvalidated = {"steps": 4, "fraction_of_predicted_seconds": pytest.approx(1.0)}
        estimated = NONE_APART
    assert schedule["seconds_by_method"] == by_method(
        estimated=estimated, unvalidated=unvalidated
    )
    refusals = schedule["refusals"]
    assert (refusals["count"], refusals["steps"]) == (0, 4)
    assert out["coverage_report"] is False


def test_a_stub_run_reports_all_its_seconds_declared(monkeypatch, tmp_path):
    backend = StepStubBackend(STAGE1)
    out = summary(monkeypatch, tmp_path, backend, 8, 40)

    schedule = out["schedule"]
    assert schedule["predicted_seconds"] == pytest.approx(
        seconds(backend, 8) + seconds(backend, 40)
    )
    assert schedule["seconds_by_method"] == by_method(
        declared={"steps": 2, "fraction_of_predicted_seconds": pytest.approx(1.0)}
    )
    assert out["schema_version"] == compass_run.SUMMARY_SCHEMA_VERSION == 8
    assert schedule["refusals"]["count"] == 0
    assert schedule["refusals"]["fraction_of_steps"] == 0.0
    assert out["coverage_report"] is False


def test_each_server_lp_hands_in_one_file_with_a_section_per_kind(
    monkeypatch, tmp_path
):
    summary(monkeypatch, tmp_path, StepStubBackend(STAGE1), 8, jobs=["executor:j"])

    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "lp-engine.json",
        "lp-frontend.json",
        "run.json",
        compass_run.STEP_TABLE_FILE,
        compass_run.SUMMARY_FILE,
    ]
    engine, frontend = (
        json.loads((tmp_path / compass_run.LP_FILE.format(lp)).read_text())
        for lp in ("engine", "frontend")
    )
    assert sorted(engine) == ["memory", "priced", "refusals"]
    assert engine["priced"]["steps"] == 1
    assert frontend == {"refusals": ["executor:j"]}
