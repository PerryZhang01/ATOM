# SPDX-License-Identifier: MIT
"""A SIGINT or SIGTERM stops a simulated server where its run stands.

The frontend's event loop ends its run at the signal, as the ``+inf`` grant
ends it at the finish, and after the finish the signal changes nothing; closing
the loop gives each signal back the handler it had. A server whose engine waits
for a grant no client run will give, since no traffic LP binds, leaves on the
signal alone, its summary written as an early end naming the signal, whether
or not a stamped request it holds, which only a grant would release, is open.
"""

import asyncio
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.clock import LpId, single_engine_table
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop

TREE = Path(__file__).resolve().parents[2]
TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)
SIGNALS = pytest.mark.parametrize(
    "sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"]
)
#: The API server's sequence on a co-hosting frontend, with an app that only
#: answers behind the server's `HttpChannel`: the run bootstrap, uvicorn on the
#: loop it names, `frontend_done`. It prints each stamp it reads.
SERVER = """\
import math, sys
from types import SimpleNamespace
import uvicorn
from atom.compass import run as compass_run
from atom.compass.clock_transport import connect
from atom.entrypoints.openai import api_server
from atom.utils.compass_loop import HttpChannel

deployment = SimpleNamespace(
    parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
    tensor_parallel_size=1,
    enable_dp_attention=False,
    pipeline_parallel_size=1,
    enable_rapidserve=False,
    runner_qualname=compass_run.ATOM_RUNNER,
    kv_transfer_config={},
)
with compass_run.frontend(deployment):
    engine = connect(compass_run.ENGINE, compass_run.spec()["clock_endpoint"])
    engine.send(("NER", math.inf, [], math.inf))

async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})

def stamp(scope):
    got = api_server._tracestate_stamp(scope)
    print("stamped", got, flush=True)
    return got

uvicorn.run(HttpChannel(app, stamp), host="127.0.0.1", port=int(sys.argv[1]),
            loop=api_server._loop_impl(), lifespan="off", log_level="warning")
print("frontend_done:", compass_run.frontend_done(SimpleNamespace(close=lambda: None)))
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _answers(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except OSError:
        return False


@SIGNALS
@pytest.mark.parametrize("finished", [False, True], ids=["in-run", "after-finish"])
def test_a_stop_signal_ends_a_run_in_progress_and_closing_puts_the_handler_back(
    monkeypatch, sig, finished
):
    had = []

    def mine(*_):
        had.append(sig)

    previous = signal.signal(sig, mine)
    rt = LPRuntime(LpId("frontend"), TABLE, conn=SimpleNamespace())
    rt.start_run()
    if finished:  # as the +inf grant leaves it
        rt.now = math.inf
        rt.end_run()
    monkeypatch.setattr(clock, "_installed", rt)
    try:
        loop = CompassEventLoop()
        signal.raise_signal(sig)
        # In the run, the stop ends it with nothing asked of the clock.
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
        stopped = None if finished else sig.name
        assert (loop.stopped, rt.in_run, had) == (stopped, False, [])
        assert signal.getsignal(sig) is mine
    finally:
        signal.signal(sig, previous)


@SIGNALS
@pytest.mark.parametrize("held", [False, True], ids=["idle", "holding-a-request"])
def test_a_stopped_server_leaves_with_its_summary_written(tmp_path, sig, held):
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps(
            {
                "clock_endpoint": f"tcp://127.0.0.1:{_free_port()}",
                "bound_s": 100.0,
                "admission_path": "serving",
                "ipc_s": 2.0**-10,
                "stream_s": 2.0**-9,
                "out_dir": str(tmp_path),
            }
        )
    )
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=str(TREE), **{compass_run.ENV: str(run_file)})
    log = tmp_path / "server.log"
    with log.open("w") as sink:
        server = subprocess.Popen(
            [sys.executable, "-c", SERVER, str(port)],
            cwd=TREE,
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        deadline = time.monotonic() + 120
        while not _answers(port):
            assert server.poll() is None, log.read_text()
            assert time.monotonic() < deadline, "no answer in 120 wall seconds"
            time.sleep(0.2)
        if held:  # stamped by a client no grant lets in; its connection stays open
            client.request(
                "POST", "/", b"{}", {"tracestate": tracestate_with("", 1.0, 0)}
            )
            while "stamped (1.0, 0)" not in log.read_text():
                assert time.monotonic() < deadline, "not held in 120 wall seconds"
                time.sleep(0.2)
        server.send_signal(sig)
        server.wait(timeout=30)
    finally:
        client.close()
        if server.poll() is None:
            server.kill()
            server.wait()
    out = log.read_text()
    assert server.returncode == 0 and "frontend_done: True" in out, out
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(summary["schedule"]["refusals"]["reasons"])
    assert reasons == {f"ended:frontend stopped by {sig.name}": 1}
    assert summary["coverage_report"] is True
