# SPDX-License-Identifier: MIT
"""A simulated run at DP-attention: one engine LP whose members are the DP ranks.

ATOM's API server at ``-tp 2 --enable-dp-attention --enable-expert-parallel``
runs two DP ranks at TP 1, each its own engine-core process. Each process
joins the one engine LP as the member of its rank and owns its rank's
channels; each rank's worker starts on the host with its groups on gloo.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_vertical_slice import (
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    UnansweredRequests,
    _free_port,
    _run_file,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.clock import NER, TAR, LpId
from atom.compass.detect.determinism import compare_step_tables
from atom.compass.parity import read
from atom.compass.runner.overrides import start_on_host
from atom.utils import CpuGpuBuffer, clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

#: One request per entry, by its max_tokens; unequal, so one rank finishes first.
MAX_TOKENS = (2, 4, 6, 8)
DONE = "data: [DONE]"


class Requests(Traffic):
    """The traffic LP: every request of `MAX_TOKENS` sent at 0, streamed."""

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for i, n in enumerate(MAX_TOKENS):
            self.sent, seq = rt.stamp_send(self.http)
            self.open.add(i)
            body = {
                "model": MODEL,
                "prompt": f"request {i} through the data-parallel slice",
                "max_tokens": n,
                "stream": True,
            }
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()
        if self.open:
            raise UnansweredRequests(f"requests {sorted(self.open)} never finished")

    def _take(self) -> None:
        super()._take()
        done = sum(line == DONE for _, line in self.events)
        self.open = set(range(done, len(MAX_TOKENS)))


def run_dp_slice(out: Path, seed: str) -> dict:
    """One run in its own process tree: its step table, summary, parity record."""
    out.mkdir()
    run_file = _run_file(out, data_parallel_size=2)
    port = _free_port()
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
    )
    args = ["--model", MODEL, "--host", "127.0.0.1", "--server-port", str(port)]
    args += ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]
    args += ["--enforce-eager", "--max-model-len", "2048"]
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "atom.entrypoints.openai.api_server", *args]
            + ["--compass-run", str(run_file)],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            traffic = Requests(json.loads(run_file.read_text()), port, server)
            traffic.run()
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    tail = (out / "server.log").read_text()[-4000:]
    assert code == 0, f"the server exited {code} after the finish:\n{tail}"
    return {
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "events": traffic.events,
        "parity": read(out / "parity"),
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("dp_slice")
    return [run_dp_slice(root / f"seed{seed}", seed) for seed in ("1", "2")]


def step_rows(run: dict) -> list[tuple]:
    """(step, rank, batch size, dummy, step seconds): each rank's forward calls
    beside the engine LP's TAR grants, one grant per joined step."""
    seconds = [
        float(f[2]) - float(f[1])
        for f in map(str.split, run["table"].splitlines()[1:])
        if f[0] == "engine" and f[3] == TAR
    ]
    steps = run["parity"]
    assert sorted(steps) == [0, 1]
    assert len(steps[0]) == len(steps[1]) == len(seconds)
    return [
        (k, rank, len(steps[rank][k]["batch"]), steps[rank][k]["is_dummy_run"], s)
        for k, s in enumerate(seconds)
        for rank in (0, 1)
    ]


@NEEDS_A_RUN
def test_four_requests_are_answered_and_the_run_ends_by_the_finish(runs):
    run = runs[0]
    rows = step_rows(run)
    print("\nstep rank batch dummy seconds")
    for row in rows:
        print(*row)
    assert sum(line == DONE for _, line in run["events"]) == len(MAX_TOKENS)
    finish = [r for r in run["table"].splitlines()[1:] if " inf " in r]
    assert sorted(r.split()[0] for r in finish) == ["engine", "frontend", "traffic"]
    assert run["summary"]["schedule"]["refusals"]["count"] == 0
    # Both ranks served requests, and the engine LP's every step took time.
    assert {rank for _, rank, size, _, _ in rows if size} == {0, 1}
    assert all(s > 0 for *_, s in rows)
    # A step where one rank ran a request and the other its dummy batch.
    by_step = {}
    for k, rank, size, dummy, _ in rows:
        by_step.setdefault(k, {})[rank] = (size, dummy)
    assert any(
        sorted(d for _, d in ranks.values()) == [False, True]
        and any(size for size, _ in ranks.values())
        for ranks in by_step.values()
    )


@NEEDS_A_RUN
def test_two_hash_seeds_give_byte_identical_step_tables(runs):
    left, right = (r["table"] for r in runs)
    code, report = compare_step_tables(left, right, "seed 1", "seed 2")
    print(f"\n{report}")
    assert code == 0, report
    assert left == right


# --- without a driver ---------------------------------------------------------


def _deployment(tp=1, dp=1, dp_attention=False, rank=0):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp, data_parallel_rank=rank),
        tensor_parallel_size=tp,
        enable_dp_attention=dp_attention,
        pipeline_parallel_size=1,
        enable_rapidserve=False,
        runner_qualname=compass_run.ATOM_RUNNER,
        kv_transfer_config={},
    )


@pytest.mark.parametrize(
    "deployment, width",
    [
        (_deployment(tp=2, dp_attention=True), 1),
        (_deployment(tp=2), 2),
        (_deployment(dp=2), 4),
    ],
)
def test_a_deployment_whose_dp_width_the_run_file_does_not_state_is_refused(
    monkeypatch, tmp_path, deployment, width
):
    run = _run_file(tmp_path, data_parallel_size=width)
    monkeypatch.setenv(compass_run.ENV, str(run))
    with pytest.raises(ValueError, match=f"data_parallel_size is {width}"):
        compass_run.frontend(deployment)
    assert clock.installed() is None


def test_each_dp_rank_joins_the_engine_lp_as_its_own_member(monkeypatch, tmp_path):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, data_parallel_size=2)))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    monkeypatch.setattr(compass_run, "_member", None)
    joined = []
    monkeypatch.setattr(
        compass_run, "connect", lambda lp, endpoint, member: joined.append((lp, member))
    )
    with compass_run.engine(_deployment(dp=2, rank=1)):
        pass
    assert joined == [(compass_run.ENGINE, "dp1")]
    assert utils.LP_OF_RANK == {0: compass_run.ENGINE, 1: compass_run.ENGINE}
    assert clock.installed().me == compass_run.ENGINE


def test_the_authority_joins_the_ranks_and_releases_each_its_own_channels(tmp_path):
    run = json.loads(_run_file(tmp_path, data_parallel_size=2).read_text())
    ca = compass_run._RecordingAuthority(run)
    engine, frontend = compass_run.ENGINE, compass_run.FRONTEND
    request = "frontend->engine:request#dp1"
    assert ca.on_request(LpId("traffic"), NER, math.inf, []) == []
    assert ca.on_request(frontend, NER, math.inf, [(request, 0, 1.0)]) == []
    assert ca.on_request(engine, NER, math.inf, [], member="dp0") == []
    replies = ca.on_request(engine, NER, math.inf, [], member="dp1")
    assert sorted(to for to, _, _ in replies) == [(engine, "dp0"), (engine, "dp1")]
    released = {to[1]: r for to, _, r in replies}
    assert released["dp1"][request] == [(0, 1.0)]
    assert all(ch.endswith("#dp0") for ch in released["dp0"])
    assert [str(r) for r in ca.steps.rows if r.event == "release"] == [
        f"engine 1.0 1.0 release {request} 0 -"
    ]


def _finish(monkeypatch, tmp_path, ranks):
    """`engine_done` on each of `ranks` of a two-rank run, then `frontend_done`."""
    run = json.loads(_run_file(tmp_path, data_parallel_size=2).read_text())
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    table = compass_run.channel_table(run)
    # The authority starts before the engines, clearing an earlier run's files.
    authority = compass_run._RecordingAuthority(run)
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    for rank in ranks:
        monkeypatch.setattr(compass_run, "_member", f"dp{rank}")
        refused = (f"command:on dp{rank}",)
        worker = SimpleNamespace(call_func=lambda name, wait_out, r=refused: r)
        compass_run.engine_done(SimpleNamespace(runner_mgr=worker))

    authority.started = authority.finished = time.monotonic()
    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    return compass_run.frontend_done(SimpleNamespace(close=lambda: None))


def test_every_ranks_refusals_reach_the_run_summary(monkeypatch, tmp_path):
    # A rank file of another run in a reused out_dir is not this run's.
    (tmp_path / compass_run.COMMANDS_FILE.format("engine.dp7")).write_text(
        '["command:an earlier run"]'
    )
    assert _finish(monkeypatch, tmp_path, (0, 1))

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    assert summary["schedule"]["refusals"]["reasons"] == [
        ["command:on dp0", 1],
        ["command:on dp1", 1],
    ]


def test_a_rank_that_wrote_no_refusal_record_fails_the_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(compass_run, "HAND_IN_WAIT_S", 0.2)
    with pytest.raises(TimeoutError, match=r"no commands-engine\.dp0\.json in"):
        _finish(monkeypatch, tmp_path, (1,))
    assert not (tmp_path / compass_run.SUMMARY_FILE).exists()


def test_fake_eplb_is_refused_at_dp_width_above_one(monkeypatch, tmp_path):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, data_parallel_size=2)))
    deployment = _deployment(tp=2, dp_attention=True)
    deployment.fake_eplb = True
    with pytest.raises(ValueError, match="--fake-eplb"):
        compass_run.frontend(deployment)
    assert clock.installed() is None


def test_a_dp_rank_starts_on_the_host_with_no_device_communicator(monkeypatch):
    calls = {}

    def record(name):
        return lambda *a, **k: calls.setdefault(name, (a, k))

    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(
            init_distributed_environment=record("world"),
            initialize_model_parallel=record("groups"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.utils",
        SimpleNamespace(get_distributed_init_method=lambda ip, p: f"tcp://{ip}:{p}"),
    )
    monkeypatch.setattr(torch.cuda, "Stream", torch.cuda.Stream)
    monkeypatch.setattr(CpuGpuBuffer, "__init__", CpuGpuBuffer.__init__)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
        ),
    )
    runner = SimpleNamespace()
    start_on_host(runner, 0, config)
    assert runner.device == torch.device("cpu")
    world = {
        "world_size": 1,
        "rank": 0,
        "distributed_init_method": "tcp://127.0.0.1:29501",
        "backend": "gloo",
        "local_rank": 0,
        "data_parallel_size": 2,
        "data_parallel_rank": 1,
    }
    assert calls["world"] == ((), world)
    groups = {"data_parallel_size": 2, "custom_group_config": {}}
    assert calls["groups"] == ((1, 1), groups)
