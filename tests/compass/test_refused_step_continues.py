# SPDX-License-Identifier: MIT
"""A step no cost source answers costs 0 s and the run goes on, unless the run
file says ``on_refusal: abort``.

Requests are driven through ATOM's own `Scheduler` and the runner's own
`forward`, priced by a coarse law with no decode vector for the eager rung, so
every decode step is refused. The worker's answers are read back as
`engine_done` reads them, and `frontend_done` writes ``summary.json``.
"""

import json
import math
import time
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig
from test_backend_coarse import backend, law
from test_dp_slice import _deployment
from test_dp_step_max import BLOCK, dp2, runner  # noqa: F401  the dp2 fixture
from test_run_graph_pool import _reserved
from test_runner_control_refusals import Worker

from atom.compass import run as compass_run
from atom.compass.backends import CoarseBackend, CoarseLaw
from atom.compass.backends.ladder import CostRefused
from atom.compass.runner import overrides
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime

MAX_TOKENS = 4
REFUSED = "cost:coarse-law: no decode vector for rung eager"


def no_eager_rung():
    without = law()
    del without["decode"]["eager"]
    return CoarseBackend(CoarseLaw(without))


def run_file(tmp_path, monkeypatch, **keys):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": f"inproc:{tmp_path}",
        "out_dir": str(tmp_path),
        **keys,
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


def drive(r, requests=3):
    """ATOM's schedule, forward, postprocess, until every request finished;
    the requests and each step's reply."""
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    sequences = [
        Sequence(
            list(range(5, 5 + 8 * (i + 1))),
            BLOCK,
            sampling_params=SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True),
        )
        for i in range(requests)
    ]
    for s in sequences:
        scheduler.add(s)
    replies = []
    while scheduler.has_unfinished_requests():
        assert len(replies) < 100, "the driven requests do not finish"
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
            continue
        batch, seqs = scheduled
        replies.append(r.forward(batch))
        scheduler.postprocess(list(seqs.values()), replies[-1], batch=batch)
    return sequences, replies


def finish(monkeypatch, tmp_path, run, r):
    """The engine's and the frontend's hand-ins after the finish, and the summary."""
    table = compass_run.channel_table(run)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    compass_run.engine_done(SimpleNamespace(runner_mgr=Worker(r)))

    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())


def test_a_refused_step_costs_nothing_and_every_request_is_answered(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = run_file(tmp_path, monkeypatch)
    r = runner(no_eager_rung())

    sequences, replies = drive(r)

    assert [s.num_completion_tokens for s in sequences] == [MAX_TOKENS] * 3
    refused = sum(reply.predicted_s == 0.0 for reply in replies)
    assert 0 < refused < len(replies)
    out = finish(monkeypatch, tmp_path, run, r)
    refusals = out["schedule"]["refusals"]
    assert refusals["reasons"] == [[REFUSED, refused]]
    assert refusals["steps"] == len(replies)
    assert refusals["fraction_of_steps"] == refused / len(replies)
    assert out["schedule"]["predicted_seconds"] == pytest.approx(
        sum(reply.predicted_s for reply in replies)
    )
    assert out["coverage_report"] is True


def test_under_abort_the_refused_step_ends_the_run_and_the_summary_is_written(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = run_file(tmp_path, monkeypatch, on_refusal="abort")
    for module, name in [(compass_run, "_authority"), (clock, "_installed")]:
        monkeypatch.setattr(module, name, None)
    with compass_run.frontend(_deployment()):
        pass
    r = runner(no_eager_rung())
    r.compass_abort = True
    engine = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
    monkeypatch.setattr(clock, "_installed", engine)

    engine_core = SimpleNamespace(runner_mgr=Worker(r))
    with pytest.raises(CostRefused), compass_run.engine_loop(engine_core):
        drive(r)

    path = tmp_path / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "no summary.json in 10 wall seconds"
        time.sleep(0.05)
    out = json.loads(path.read_text())
    [[reason, count]] = out["schedule"]["refusals"]["reasons"]
    assert (reason.startswith("ended:engine raised CostRefused("), count) == (True, 1)
    assert out["coverage_report"] is True


def test_a_refused_rank_still_joins_the_group_max(dp2):  # noqa: F811
    ranks = [
        runner(no_eager_rung(), dp_size=2, dp_rank=0),
        runner(backend(), dp_size=2, dp_rank=1),
    ]

    refused, priced = dp2(ranks[0].dummy_execution, ranks[1].dummy_execution)

    assert priced.predicted_s > 0
    assert refused.predicted_s == priced.predicted_s
    assert ranks[0].refused_commands() == (REFUSED,)
    assert ranks[1].refused_commands() == ()


@pytest.mark.parametrize(
    "keys, abort",
    [({}, False), ({"on_refusal": "mark"}, False), ({"on_refusal": "abort"}, True)],
)
def test_the_run_file_tells_the_runner_whether_a_refusal_ends_the_run(
    monkeypatch, tmp_path, keys, abort
):
    installed = []
    monkeypatch.setattr(
        overrides, "install_cost_backend", lambda r, b: installed.append(r)
    )
    _reserved(monkeypatch, tmp_path, eager=True, **keys)
    assert [r.compass_abort for r in installed] == [abort]


@pytest.mark.parametrize("value", ["skip", "Mark", None])
def test_another_on_refusal_is_refused_when_the_run_file_is_read(
    monkeypatch, tmp_path, value
):
    run_file(tmp_path, monkeypatch, on_refusal=value)
    with pytest.raises(ValueError, match="on_refusal is"):
        compass_run.spec()


def test_a_run_that_ends_mid_run_leaves_its_seconds_and_shares_absent_not_zero(
    monkeypatch, tmp_path
):
    # The engine priced a prefill, then raised; it handed in no priced steps.
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = run_file(tmp_path, monkeypatch, on_refusal="abort")
    for module, name in [(compass_run, "_authority"), (clock, "_installed")]:
        monkeypatch.setattr(module, name, None)
    with compass_run.frontend(_deployment()):
        pass
    r = runner(no_eager_rung())
    r.compass_abort = True
    engine = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
    monkeypatch.setattr(clock, "_installed", engine)

    with (
        pytest.raises(CostRefused),
        compass_run.engine_loop(SimpleNamespace(runner_mgr=Worker(r))),
    ):
        drive(r)

    path = tmp_path / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "no summary.json in 10 wall seconds"
        time.sleep(0.05)
    schedule = json.loads(path.read_text())["schedule"]
    assert r.priced_steps()["steps"] > 0
    assert schedule["predicted_seconds"] is None
    assert schedule["analytical"] is None
    assert schedule["extrapolated"] is None
    assert schedule["refusals"]["steps"] is None
    assert schedule["refusals"]["fraction_of_steps"] is None
