# SPDX-License-Identifier: MIT
"""The run summary names each assumed treatment its steps were priced under.

Two DP-attention ranks each price two prefills through the runner's own
`forward`, with a step linear law keyed at their `(dp, ep)`; each rank's worker
answers `engine_done` over the worker RPC, and `frontend_done` writes
``summary.json``.
"""

import json
import math
import time
from types import SimpleNamespace

from test_backend_step_linear import backend, law
from test_dp_step_max import dp2, prefill, runner  # noqa: F401  the dp2 fixture
from test_run_refusal_summary import summary
from test_runner_control_refusals import Worker
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.utils import clock
from atom.utils.clock import LPRuntime


def dp_summary(monkeypatch, tmp_path, ranks):
    """The run summary after `engine_done` on each rank, then `frontend_done`."""
    run = json.loads(_run_file(tmp_path, data_parallel_size=len(ranks)).read_text())
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    table = compass_run.channel_table(run)
    authority = compass_run._RecordingAuthority(run)
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    for rank, r in enumerate(ranks):
        monkeypatch.setattr(compass_run, "_member", f"dp{rank}")
        compass_run.engine_done(SimpleNamespace(runner_mgr=Worker(r)))
    authority.started = authority.finished = time.monotonic()
    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())


def test_a_dp_attention_run_at_dp2_names_uniform_routing_and_its_steps(
    monkeypatch, tmp_path, dp2  # noqa: F811
):
    ranks = [
        runner(StepLinearBackend(StepLinearLaw(law()), 2), dp_size=2, dp_rank=r)
        for r in range(2)
    ]
    # Two steps per rank, so a count reads 4 where a per-rank flag reads 2.
    dp2(
        lambda: [ranks[0].forward(prefill(40)) for _ in range(2)],
        lambda: [ranks[1].forward(prefill(8)) for _ in range(2)],
    )

    out = dp_summary(monkeypatch, tmp_path, ranks)

    assert out["schedule"]["treatments"] == {"uniform routing": 4}
    assert out["schedule"]["refusals"]["steps"] == 4
    assert out["schema_version"] == compass_run.SUMMARY_SCHEMA_VERSION == 8


def test_a_one_rank_run_names_no_treatment(monkeypatch, tmp_path):
    out = summary(monkeypatch, tmp_path, backend(), 8, 40)

    assert out["schedule"]["treatments"] == {}
    assert out["schedule"]["refusals"]["steps"] == 2
