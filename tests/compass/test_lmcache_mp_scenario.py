# SPDX-License-Identifier: MIT
"""`lmcache_mp` held to the scenario checks: one deployment, and two DP ranks
sharing one server, each run twice under a constant step stub and constant
station laws.

The workload is the parity workload's with a shared prefix
(`atom.compass.parity.traffic.bodies`): request 0 stores the prefix, its
blocks leave HBM when it ends since prefix caching is off, and a burst of
three requests per DP rank later loads it back. With
``OFFLOAD_ASYNC_LOOKUP_DEPTH=1`` two requests of each burst reach admission
unprefetched and look up blocking, and with ``OFFLOAD_LOOKUP_DEFER_S=0`` the
prefetched one is deferred once, then looks up blocking too: a lookup job is
priced above a step, so its reply is not in by the next pass.

The runs need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`, and as many GPUs in ``HIP_VISIBLE_DEVICES`` as DP ranks.
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest
from test_lmcache_mp_engine import _machine
from test_vertical_slice import (
    CONSTANT,
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    _free_port,
    _run_file,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.kv.lmcache_server import EVICTION_INTERVAL_S, reply_channel
from atom.compass.parity.traffic import bodies

PREFIX_WORDS = 600  # two whole 256-token chunks
BURST_S, BURST = 0.5, 3  # the burst's time, and its requests per DP rank
STATION_S = 2.0**-6  # every request-loop job: above a step under `CONSTANT`
COPY_S = 2.0**-7  # every copy: the host link's latency, its rate adding nothing
IPC = 2.0**-14
ENV = {
    "OFFLOAD_ASYNC_LOOKUP_DEPTH": "1",
    "OFFLOAD_LOOKUP_DEFER_S": "0",
    "OFFLOAD_MIN_LOAD_TOKENS": "256",
    "OFFLOAD_MIN_SAVE_TOKENS": "256",
}
ATOM_ARGS = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]
DP2 = ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]


class Workload(Traffic):
    """The traffic LP: request 0 at 0, the rest at `BURST_S`."""

    def __init__(self, requests, *args) -> None:
        super().__init__(*args)
        self.requests = requests

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            at = BURST_S if k else 0.0
            while rt.now < at:
                rt.next_event(at)
                self._take()
            self.sent, seq = rt.stamp_send(self.http)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()


def _measure(out: Path, machine: Path, width: list) -> None:
    from atom.compass import cli

    gpus = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    cli.main(
        ["measure", "memory", "--gpu", gpus, "--machine", str(machine)]
        + ["--store", str(out / "store"), "--out", str(out / "measure")]
        + ["--", *ATOM_ARGS, *width]
    )


def run_scenario(root: Path, seed: str, dp: int) -> dict:
    """One run in its own process tree, reading the readings in ``root/store``."""
    out = root / f"seed{seed}"
    out.mkdir()
    port, lmcache_port = _free_port(), _free_port()
    flags = ["--host", "127.0.0.1", "--port", str(lmcache_port)]
    flags += ["--chunk-size", "256", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
    machine = json.loads((root / "machine.json").read_text())
    run_file = _run_file(
        out,
        step_stub=CONSTANT,
        machine=machine,
        store=str(root / "store"),
        data_parallel_size=dp,
        bound_s=60.0,
        lmcache={"server": flags, "lookup_s": [STATION_S, 0.0], "request_s": STATION_S},
    )
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": lmcache_port}
    env = dict(
        os.environ,
        **ENV,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        ATOM_KV_OFFLOAD="lmcache_mp",
        ATOM_KV_OFFLOAD_EXTRA_CONFIG=json.dumps(extra),
    )
    requests = bodies(MODEL, 1 + BURST * dp, PREFIX_WORDS)
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "atom.entrypoints.openai.api_server", *ATOM_ARGS]
            + (DP2 if dp > 1 else [])
            + ["--host", "127.0.0.1", "--server-port", str(port)]
            + ["--no-enable_prefix_caching", "--compass-run", str(run_file)],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            traffic = Workload(requests, json.loads(run_file.read_text()), port, server)
            traffic.run()
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    text = (out / "server.log").read_text()
    assert code == 0, f"the server exited {code} after the finish:\n{text[-4000:]}"
    return {
        "out": out,
        "run": json.loads(run_file.read_text()),
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "log": text,
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


@pytest.fixture(scope="module", params=[1, 2], ids=["dp1", "dp2"])
def runs(request, tmp_path_factory):
    dp = request.param
    root = tmp_path_factory.mktemp(f"lmcache_dp{dp}")
    machine = _machine(root)
    spec = json.loads(machine.read_text())
    spec["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": 1e30,
        "latency_s": COPY_S,
    }
    machine.write_text(json.dumps(spec))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AITER_LOG_LEVEL", "WARNING")  # the readings are keyed by it
        _measure(root, machine, DP2 if dp > 1 else [])
    return dp, [run_scenario(root, seed, dp) for seed in ("1", "2")]


def _rows(table: str):
    return [line.split() for line in table.splitlines()[1:]]


# --- (a) ----------------------------------------------------------------------


@NEEDS_A_RUN
def test_each_run_meets_every_scenario_check_and_loads_the_prefix_back(runs):
    dp, (run, _) = runs
    failed = check_run(
        [r["out"] for r in runs[1]],
        run["sent"],
        run["taken"],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    transfers = re.findall(
        r"compass: lmcache_mp (\w+) of (\S+):\S+ (\w+): arrived", run["log"]
    )
    print("", *(" ".join(t) for t in transfers), sep="\n")
    stored = [e for op, e, ok in transfers if op == "store" and ok == "succeeded"]
    loaded = {e for op, e, ok in transfers if op == "retrieve" and ok == "succeeded"}
    assert loaded, "no request loaded the stored prefix back"
    if dp > 1:
        assert loaded - {stored[0]}, "no rank loaded a prefix another rank stored"
