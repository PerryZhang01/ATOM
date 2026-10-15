# SPDX-License-Identifier: MIT
"""A step no cost source answers costs 0 s and the run goes on, unless the run
file says ``on_refusal: abort``.

Requests are driven through ATOM's own `Scheduler` and the runner's own
`forward`, priced by a step linear law with no decode vector for the eager
rung, so every decode step is refused. The worker's answers are read back as
`engine_done` reads them, and `frontend_done` writes ``summary.json``.
"""

import json
import math
import time
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig
from test_backend_step_linear import backend, law
from test_dp_slice import _deployment
from test_dp_step_max import BLOCK, dp2, runner  # noqa: F401  the dp2 fixture
from test_run_graph_pool import _reserved
from test_runner_control_refusals import Worker

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.resolver import CostRefused
from atom.compass.runner import overrides
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime

MAX_TOKENS = 4
REFUSED = "cost:step-linear: no decode vector for rung eager"


def no_eager_rung():
    without = law()
    del without["decode"]["eager"]
    return StepLinearBackend(StepLinearLaw(without))


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


def recording(r, directory, monkeypatch):
    """`r`, keeping the parity record in `directory`, with the engine fields
    its first step states."""
    monkeypatch.setenv(parity.ENV, str(directory))
    quant = SimpleNamespace(compute_hash=lambda: "q")
    fields = {"kv_cache_dtype": "bf16", "quant_config": quant}
    vars(r.config).update(fields, enforce_eager=True, tensor_parallel_size=1)
    return r


def cost_lines(directory, dp_rank=0):
    """A rank's cost lines, below the version line its table starts with."""
    text = (directory / f"dp{dp_rank}.cost.jsonl").read_text()
    version, *lines = [json.loads(line) for line in text.splitlines()]
    assert version == {"version": overrides.COST_TABLE_VERSION}
    return lines


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


def test_a_refused_steps_cost_line_holds_its_reason_and_no_device_time(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    r = recording(runner(no_eager_rung()), tmp_path, monkeypatch)

    _, replies = drive(r)

    lines = cost_lines(tmp_path)
    steps = [s["step"] for s in parity.read(tmp_path)[0]]
    assert [line["step"] for line in lines] == steps == list(range(len(replies)))
    assert [line["group_s"] for line in lines] == [x.predicted_s for x in replies]
    refused = [line for line in lines if line["refused"]]
    assert 0 < len(refused) < len(lines)
    assert [line["refused"] for line in refused] == list(r.refused_commands())
    for line in refused:
        assert (line["refused"], line["device_s"], line["terms"]) == (REFUSED, 0.0, [])
    for line in lines:
        if line["refused"] is None:
            assert line["device_s"] > 0 and line["terms"]


def test_a_refused_ranks_cost_line_charges_nothing_and_leaves_at_the_group_max(
    dp2, tmp_path, monkeypatch  # noqa: F811
):
    ranks = [
        recording(runner(no_eager_rung(), dp_size=2, dp_rank=0), tmp_path, monkeypatch),
        recording(runner(backend(), dp_size=2, dp_rank=1), tmp_path, monkeypatch),
    ]

    _, priced = dp2(ranks[0].dummy_execution, ranks[1].dummy_execution)

    [refused], [line] = cost_lines(tmp_path, 0), cost_lines(tmp_path, 1)
    assert [refused[k] for k in ("refused", "device_s", "charge_s")] == [REFUSED, 0, 0]
    assert refused["group_s"] == line["group_s"] == priced.predicted_s > 0
    assert line["charge_s"] == line["device_s"] == priced.predicted_s


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


def test_an_aborted_step_0_rerun_leaves_its_cost_table_the_version_line_alone(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    _, replies = drive(recording(runner(backend()), tmp_path, monkeypatch))
    assert len(cost_lines(tmp_path)) == len(replies) > 1
    no_prefill = law()
    del no_prefill["prefill"]
    r = runner(StepLinearBackend(StepLinearLaw(no_prefill)))
    r.compass_abort = True

    with pytest.raises(CostRefused):
        drive(recording(r, tmp_path, monkeypatch))

    assert [s["step"] for s in parity.read(tmp_path)[0]] == [0]
    assert cost_lines(tmp_path) == []


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


@pytest.mark.parametrize(
    "error", [RuntimeError("the clock raised"), SystemExit()], ids=["live", "dead"]
)
def test_a_run_that_ends_mid_run_counts_what_live_workers_priced_and_no_zeros(
    monkeypatch, tmp_path, error
):
    # The engine raises after its steps; `SystemExit` is how a dead worker shows.
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = run_file(tmp_path, monkeypatch)
    for module, name in [(compass_run, "_authority"), (clock, "_installed")]:
        monkeypatch.setattr(module, name, None)
    with compass_run.frontend(_deployment()):
        pass
    r = runner(no_eager_rung())
    engine = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
    monkeypatch.setattr(clock, "_installed", engine)

    with (
        pytest.raises(type(error)),
        compass_run.engine_loop(SimpleNamespace(runner_mgr=Worker(r))),
    ):
        _, replies = drive(r)
        raise error

    path = tmp_path / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "no summary.json in 10 wall seconds"
        time.sleep(0.05)
    schedule = json.loads(path.read_text())["schedule"]
    refusals, ended = schedule["refusals"], f"ended:engine raised {error!r}"
    hand_in = json.loads((tmp_path / compass_run.LP_FILE.format("engine")).read_text())
    if isinstance(error, SystemExit):
        assert sorted(hand_in) == ["refusals"]
        assert schedule["predicted_seconds"] is None
        assert schedule["seconds_by_method"] is None
        assert schedule["treatments"] is None
        assert (refusals["steps"], refusals["fraction_of_steps"]) == (None, None)
        assert refusals["reasons"] == [[ended, 1]]
        return
    priced = r.priced_steps()
    refused = sum(reply.predicted_s == 0.0 for reply in replies)
    assert sorted(hand_in) == ["memory", "priced", "refusals"]
    assert priced["seconds"] > 0
    assert schedule["predicted_seconds"] == pytest.approx(priced["seconds"])
    for g, share in schedule["seconds_by_method"].items():
        assert share["steps"] == priced[f"{g}_steps"]
    assert refusals["steps"] == len(replies)
    assert refusals["fraction_of_steps"] == refused / len(replies)
    assert refusals["reasons"] == sorted([[REFUSED, refused], [ended, 1]])
