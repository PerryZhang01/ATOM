# SPDX-License-Identifier: MIT
"""A simulated run of a multi-node data-parallel group, on a standalone authority.

Two process trees on one host stand in for two nodes: each runs ATOM's API
server at ``-tp 2 --enable-dp-attention --data-parallel-size 2
--data-parallel-size-local 1``, so each owns two of the four DP ranks. The
coordinator, global rank 0's, is the one frontend LP. The worker node runs no
frontend LP, and its engines join the engine LP as members ``dp2`` and ``dp3``
over TCP. Every process names the authority with ``--compass-clock-endpoint``.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_dp_slice import DP_MACHINE, MAX_TOKENS, Requests, _deployment
from test_vertical_slice import (
    CONSTANT,
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    _free_port,
    _run_file,
)

from atom.compass import run as compass_run
from atom.compass.clock import LpId
from atom.compass.detect.scenario import check_run
from atom.compass.parity import read
from atom.compass.runner import COMPASS_RUNNER_QUALNAME
from atom.config import ParallelConfig
from atom.model_engine.engine_core_mgr import CoreManager
from atom.utils import clock

DP = 4
#: The slice's machine.
MACHINE = copy.deepcopy(DP_MACHINE)


def _bound(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _master_port() -> int:
    """A DP master port whose engine sockets, 100 above it, are free too."""
    while True:
        port = _free_port()
        if all(map(_bound, range(port + 100, port + 100 + 3 * DP))):
            return port


def _wait_listening(endpoint: str, alive) -> None:
    """Until the authority at `endpoint` accepts, while `alive()` holds."""
    deadline = time.monotonic() + WALL_S
    while True:
        try:
            socket.create_connection(
                ("127.0.0.1", int(endpoint.rpartition(":")[2]))
            ).close()
            return
        except ConnectionRefusedError:
            assert alive(), "the authority exited before it listened"
            assert time.monotonic() < deadline, f"no authority in {WALL_S} s"
            time.sleep(0.05)


def run_two_trees(out: Path, seed: str) -> dict:
    """One run: the authority, then each node's API server, each its own tree."""
    out.mkdir()
    run_file = _run_file(
        out, data_parallel_size=DP, step_stub=CONSTANT, machine=MACHINE
    )
    endpoint = json.loads(run_file.read_text())["clock_endpoint"]
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
    )
    compass = ["--compass-run", str(run_file), "--compass-clock-endpoint", endpoint]
    common = [
        *("--model", MODEL, "--host", "127.0.0.1", "-tp", "2"),
        *("--enable-dp-attention", "--enforce-eager", "--max-model-len", "2048"),
        *("--data-parallel-size", "2", "--data-parallel-size-local", "1"),
        *("--data-parallel-master-port", str(_master_port())),
        *("--data-parallel-base-port", str(_free_port())),
        *compass,
    ]
    ports = [_free_port(), _free_port()]
    procs = []

    def start(name, *args):
        with open(out / f"{name}.log", "w") as log:
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", *args],
                    cwd=TREE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        return procs[-1]

    authority = start("authority", "atom.compass.run", *compass)
    watchdog = threading.Timer(WALL_S, lambda: [p.kill() for p in procs])
    watchdog.start()
    try:
        _wait_listening(endpoint, lambda: authority.poll() is None)
        for node, port in enumerate(ports):
            start(
                f"node{node}",
                "atom.entrypoints.openai.api_server",
                *common,
                *("--data-parallel-rank", str(node), "--server-port", str(port)),
            )
        traffic = Requests(json.loads(run_file.read_text()), ports[0], procs[1])
        traffic.run()
        codes = [p.wait(timeout=120) for p in procs]
    finally:
        watchdog.cancel()
        for p in procs:
            if p.poll() is None:
                p.kill()
    logs = {f.name: f.read_text()[-3000:] for f in sorted(out.glob("*.log"))}
    assert codes == [0, 0, 0], f"exit codes {codes}:\n{logs}"
    return {
        "out": out,
        "events": traffic.events,
        "handed_in": sorted(f.name for f in out.glob(compass_run.LP_FILE.format("*"))),
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("multinode_dp")
    return [run_two_trees(root / f"seed{seed}", seed) for seed in ("1", "2")]


@NEEDS_A_RUN
def test_two_trees_of_two_ranks_meet_every_scenario_check(runs):
    run = runs[0]
    print(f"\n{(run['out'] / compass_run.STEP_TABLE_FILE).read_text()}")
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    assert run["handed_in"] == sorted(
        compass_run.LP_FILE.format(n)
        for n in ["frontend"] + [f"engine.dp{r}" for r in range(DP)]
    )
    # The worker node's ranks served requests too.
    steps = read(run["out"] / "parity")
    assert all(any(s["batch"] for s in steps[rank]) for rank in (2, 3))


# --- without a driver ---------------------------------------------------------


def _node(rank: int):
    """One node's slice of a two-node group: one DP rank each, here at TP 1."""
    deployment = _deployment(dp=2)
    deployment.parallel_config = ParallelConfig(
        data_parallel_size=2, data_parallel_size_local=1, data_parallel_rank=rank
    )
    deployment.fake_eplb = False
    return deployment


@pytest.fixture
def standalone(monkeypatch, tmp_path):
    """A two-rank run file whose clock endpoint each node names."""
    run = _run_file(tmp_path, data_parallel_size=2)
    endpoint = json.loads(run.read_text())["clock_endpoint"]
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setenv(compass_run.CLOCK_ENV, endpoint)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_authority", None)
    monkeypatch.setattr(
        CoreManager,
        "broadcast_utility_command_sync",
        CoreManager.broadcast_utility_command_sync,
    )
    return endpoint


def test_each_node_joins_the_authority_and_only_the_coordinator_is_a_frontend(
    standalone, tmp_path
):
    serving = threading.Thread(
        target=compass_run.authority, args=(standalone,), daemon=True
    )
    serving.start()
    _wait_listening(standalone, serving.is_alive)

    worker = _node(1)
    with compass_run.frontend(worker):
        assert clock.installed() is None
    assert clock.installed() is None and compass_run._authority is None
    assert worker.runner_qualname == COMPASS_RUNNER_QUALNAME

    with compass_run.frontend(_node(0)):
        assert clock.installed().me == LpId("frontend")
    assert clock.installed().in_run
    assert compass_run._authority is None


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("named", [None, "tcp://127.0.0.1:1"])
def test_a_node_that_finds_no_authority_in_the_out_dir_is_refused(
    standalone, tmp_path, rank, named
):
    if named is not None:
        (tmp_path / compass_run.AUTHORITY_FILE).write_text(named)
    with pytest.raises(ValueError, match="names no authority at tcp://"):
        compass_run.frontend(_node(rank))
    assert clock.installed() is None
