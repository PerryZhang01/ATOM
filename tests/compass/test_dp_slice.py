# SPDX-License-Identifier: MIT
"""A simulated run at DP-attention: one engine LP whose members are the DP ranks.

ATOM's API server at ``-tp 2 --enable-dp-attention --enable-expert-parallel``
runs two DP ranks at TP 1, each its own engine-core process. Each process
joins the one engine LP as the member of its rank and owns its rank's
channels; each rank's worker starts on the host with its groups on gloo.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import copy
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
from test_memory_readings import DOCUMENT
from test_vertical_slice import (
    CONSTANT,
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
from atom.compass.clock import NER, LpId
from atom.compass.detect.scenario import check_run
from atom.compass.parity import read
from atom.compass.runner.overrides import start_on_host
from atom.utils import CpuGpuBuffer, clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

#: One request per entry, by its max_tokens; unequal, so one rank finishes first.
MAX_TOKENS = (2, 4, 6, 8)
#: The slice's machine.
DP_MACHINE = copy.deepcopy(DOCUMENT)
#: The tokenizers probed on the checked-in MI308X machine as well, so a slice
#: model of theirs (Qwen3.8-27B) resolves its tokenizer.
DP_MACHINE["host"]["tokenizers"] += json.loads(
    (TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text()
)["host"]["tokenizers"]
#: Declared test values for a pipeline stage's send: every part is rendezvous.
DP_MACHINE["interconnect"]["intra_node"]["pp"] = {
    carrier: {
        "eager_threshold_bytes": 1,
        "latency_s": 2.0**-20,
        "overhead_s": 2.0**-22,
        "per_byte_s": 2.0**-36,
        "rendezvous_fixed_s": 2.0**-17,
    }
    for carrier in ("rccl", "gloo")
}
#: And gloo's wake, about a picosecond: a blocked stage goes on as it completes.
DP_MACHINE["interconnect"]["intra_node"]["pp"]["gloo"] |= {
    "wake_s": 2.0**-40,
    "wake_rise_s": 1.0,
}
#: The run test's machine: an HTTP intake of about a picosecond, so the four
#: requests reach the engine together and each rank batches two.
BATCHED = copy.deepcopy(DP_MACHINE)
BATCHED["host"]["http_intake"] = {"fixed_s": 2.0**-40, "per_byte_s": 2.0**-60}
DONE = "data: [DONE]"
#: The slice's parallelism: two DP ranks at TP 1, with expert parallelism.
DP_WIDTH = ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]


@pytest.fixture(autouse=True)
def device_classes_put_back(monkeypatch):
    """`start_on_host` swaps these for host stand-ins process-wide; undo it per test."""
    monkeypatch.setattr(torch.cuda, "Stream", torch.cuda.Stream)
    monkeypatch.setattr(torch.cuda, "Event", torch.cuda.Event)
    monkeypatch.setattr(CpuGpuBuffer, "__init__", CpuGpuBuffer.__init__)


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


def run_dp_slice(
    out: Path, seed: str, width=DP_WIDTH, dp=2, pp=1, machine=DP_MACHINE
) -> dict:
    """One run in its own process tree: its step table, summary, parity record.

    `width` is the server's parallelism flags, `dp` the DP ranks and `pp` the
    pipeline stages they start; a pipeline keeps no parity record."""
    out.mkdir()
    run_file = _run_file(
        out,
        data_parallel_size=dp,
        pipeline_parallel_size=pp,
        step_stub=CONSTANT,
        machine=machine,
    )
    port = _free_port()
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
    )
    if pp == 1:
        env["ATOM_COMPASS_PARITY_RECORD"] = str(out / "parity")
    args = ["--model", MODEL, "--host", "127.0.0.1", "--server-port", str(port)]
    args += width
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
        "out": out,
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "events": traffic.events,
        "parity": read(out / "parity") if pp == 1 else None,
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("dp_slice")
    return [
        run_dp_slice(root / f"seed{seed}", seed, machine=BATCHED) for seed in ("1", "2")
    ]


@NEEDS_A_RUN
def test_four_requests_meet_every_scenario_check_on_both_ranks(runs):
    run = runs[0]
    print(f"\n{run['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    steps = run["parity"]
    assert [len(steps[rank][0]["batch"]) for rank in (0, 1)] == [2, 2]
    # Both ranks served requests, and in some step one rank ran a request
    # while the other ran its dummy batch.
    assert all(any(s["batch"] for s in steps[rank]) for rank in (0, 1))
    assert any(a["is_dummy_run"] != b["is_dummy_run"] for a, b in zip(*steps.values()))


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
        f"engine 0.0 1.0 release {request} 0 -"
    ]


def test_a_standalone_authority_starts_the_run_once_every_rank_has_asked(tmp_path):
    run = json.loads(_run_file(tmp_path, data_parallel_size=2).read_text())
    ca = compass_run._RecordingAuthority(run)
    ca.on_request(LpId("traffic"), NER, math.inf, [])
    ca.on_request(compass_run.FRONTEND, NER, math.inf, [])
    # The member positionally, as the clock transport passes it.
    ca.on_request(compass_run.ENGINE, NER, math.inf, [], math.inf, "dp0")
    assert ca.started is None
    ca.on_request(compass_run.ENGINE, NER, math.inf, [], math.inf, "dp1")
    assert ca.started is not None


# Each rank's own `priced_steps` reply; no field is equal across the ranks.
PRICED = [
    {
        "steps": 2,
        "seconds": 1.0,
        "analytical_steps": 0,
        "analytical_seconds": 0.0,
        "unvalidated_steps": 1,
        "unvalidated_seconds": 0.25,
    },
    {
        "steps": 3,
        "seconds": 3.0,
        "analytical_steps": 1,
        "analytical_seconds": 0.5,
        "unvalidated_steps": 2,
        "unvalidated_seconds": 0.75,
    },
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
        replies = {
            "refused_commands": (f"command:on dp{rank}",),
            "priced_steps": PRICED[rank],
            "predicted_memory": {},
        }
        worker = SimpleNamespace(call_func=lambda name, wait_out, r=replies: r[name])
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
    (tmp_path / compass_run.LP_FILE.format("engine.dp7")).write_text(
        '{"refusals": ["command:an earlier run"]}'
    )
    assert _finish(monkeypatch, tmp_path, (0, 1))

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    assert summary["schedule"]["refusals"]["reasons"] == [
        ["command:on dp0", 1],
        ["command:on dp1", 1],
    ]


def test_the_run_summary_sums_every_ranks_priced_steps(monkeypatch, tmp_path):
    assert _finish(monkeypatch, tmp_path, (0, 1))

    schedule = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())["schedule"]
    assert schedule["refusals"]["steps"] == 5
    assert schedule["predicted_seconds"] == 4.0
    assert schedule["seconds_by_method"]["analytical"] == {
        "steps": 1,
        "fraction_of_predicted_seconds": 0.125,
    }
    assert schedule["seconds_by_method"]["unvalidated"] == {
        "steps": 3,
        "fraction_of_predicted_seconds": 0.25,
    }


def test_a_rank_that_wrote_no_refusal_record_fails_the_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(compass_run, "HAND_IN_WAIT_S", 0.2)
    with pytest.raises(TimeoutError, match=r"no lp-engine\.dp0\.json in"):
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
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        tp_world_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
            _managed_distributed_store=False,
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
    groups = {
        "decode_context_model_parallel_size": 1,
        "data_parallel_size": 2,
        "prefill_context_model_parallel_size": 1,
        "custom_group_config": {},
    }
    assert calls["groups"] == ((1, 1), groups)


def test_a_dp_rank_under_a_managed_store_joins_it_before_aiter_builds_groups(
    monkeypatch,
):
    """The engine core manager listens on the base port; the rank joins as a client."""
    calls = []

    def record(name):
        return lambda *a, **k: calls.append((name, a, k)) or name

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
    monkeypatch.setattr(torch.distributed, "TCPStore", record("store"))
    monkeypatch.setattr(torch.distributed, "init_process_group", record("group"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "Stream", torch.cuda.Stream)
    monkeypatch.setattr(CpuGpuBuffer, "__init__", CpuGpuBuffer.__init__)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        tp_world_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
            _managed_distributed_store=True,
        ),
    )
    start_on_host(SimpleNamespace(), 0, config)
    assert [name for name, _, _ in calls] == ["store", "group", "world", "groups"]
    assert calls[0][1:] == (("127.0.0.1", 29501), {"is_master": False})
    group = {"backend": "gloo", "store": "store", "rank": 1, "world_size": 2}
    assert calls[1][1:] == ((), group)
