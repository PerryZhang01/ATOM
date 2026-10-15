# SPDX-License-Identifier: MIT
"""A SIGINT or SIGTERM stops a simulated server where its run stands.

The frontend's event loop ends its run at the signal, as the ``+inf`` grant
ends it at the finish, and closing the loop gives each signal back the handler
it had. A server whose engine waits for a grant no client run will give, since
no traffic LP binds, leaves on the signal alone, its summary written as an
early end naming the signal.
"""

import http.client
import json
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
#: answers: the run bootstrap, uvicorn on the loop it names, `frontend_done`.
SERVER = """\
import math, sys
from types import SimpleNamespace
import uvicorn
from atom.compass import run as compass_run
from atom.compass.clock_transport import connect
from atom.entrypoints.openai import api_server

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

uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]),
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
def test_a_stop_signal_ends_the_loops_run_and_closing_puts_its_handler_back(
    monkeypatch, sig
):
    had = []

    def mine(*_):
        had.append(sig)

    previous = signal.signal(sig, mine)
    rt = LPRuntime(LpId("frontend"), TABLE, conn=SimpleNamespace())
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    try:
        loop = CompassEventLoop()
        signal.raise_signal(sig)
        loop.run_forever()  # the stop ends the run with nothing asked of the clock
        loop.close()
        assert (loop.stopped, rt.in_run, had) == (sig.name, False, [])
        assert signal.getsignal(sig) is mine
    finally:
        signal.signal(sig, previous)


@SIGNALS
def test_a_stopped_server_leaves_with_its_summary_written(tmp_path, sig):
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
    server = subprocess.Popen(
        [sys.executable, "-c", SERVER, str(port)],
        cwd=TREE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not _answers(port):
            assert server.poll() is None, server.communicate()[0]
            assert time.monotonic() < deadline, "no answer in 120 wall seconds"
            time.sleep(0.2)
        server.send_signal(sig)
        out = server.communicate(timeout=30)[0]
    finally:
        if server.poll() is None:
            server.kill()
            server.communicate()
    assert server.returncode == 0 and "frontend_done: True" in out, out
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(summary["schedule"]["refusals"]["reasons"])
    assert reasons == {f"ended:frontend stopped by {sig.name}": 1}
    assert summary["coverage_report"] is True
