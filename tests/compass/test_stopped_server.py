# SPDX-License-Identifier: MIT
"""A SIGINT or SIGTERM stops a simulated server where its run stands.

The frontend's event loop ends its run at the signal, as the ``+inf`` grant
ends it at the finish, and after the finish the signal changes nothing; closing
the loop gives each signal back the handler it had. A server whose engine waits
for a grant no client run will give, since no traffic LP binds, leaves on the
signal alone, its summary written as an early end naming the signal, whether
or not a stamped request it holds, which only a grant would release, is open.
The stop first ends every LP's run through the authority, so an engine in a
process of its own leaves its loop and hands in what its steps cost, whether
the signal reaches the server alone or its process group, whose SIGTERM would
have ATOM's handler stop the engine's workers. An engine whose input thread
has read a request the authority has not released leaves the same way: the
stop's ``+inf`` grant releases it.
"""

import asyncio
import contextlib
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import zmq

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.clock import LpId, single_engine_table
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket
from atom.utils.compass_loop import CompassEventLoop

TREE = Path(__file__).resolve().parents[2]
TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)
SIGNALS = pytest.mark.parametrize(
    "sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"]
)
#: What a server script starts with: the modules it serves with, and the
#: deployment a co-hosting frontend reads.
BOOTSTRAP = """\
import math, subprocess, sys
from types import SimpleNamespace
import uvicorn
from atom.compass import run as compass_run
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
"""
#: The API server's sequence on a co-hosting frontend, with an app that only
#: answers behind the server's `HttpChannel`: the run bootstrap, uvicorn on the
#: loop it names, `frontend_done`. It prints each stamp it reads.
SERVER = BOOTSTRAP + """\
from atom.compass.clock_transport import connect
from atom.utils.compass_loop import HttpChannel

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


#: An engine in a process of its own: the run bootstrap's engine LP over its
#: loop, which asks for time until the ``+inf`` grant. Its workers answer what
#: their steps cost until a SIGTERM to its handler, standing in for ATOM's,
#: stops them. It says ``ready`` once bound to the authority.
ENGINE = """\
import math, signal
from types import SimpleNamespace
from atom.compass import run as compass_run
from atom.utils import clock

rank = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
with compass_run.engine(SimpleNamespace(parallel_config=rank, kv_transfer_config={})):
    pass
stopped = []
signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
said = {"refused_commands": [], "predicted_memory": {}}
said["priced_steps"] = {"steps": 3, "seconds": 0.75}

def call_func(name, wait_out):
    if stopped:
        raise RuntimeError("the workers stopped")
    return said[name]

print("ready", flush=True)
with compass_run.engine_loop(SimpleNamespace(runner_mgr=SimpleNamespace(call_func=call_func))):
    while not clock.idle(lambda: math.inf):
        pass
"""
#: A co-hosting frontend, as `SERVER`'s, over `ENGINE` in a process it starts.
#: Its app closes the engine at shutdown, before `frontend_done`, as the API
#: server's lifespan does: it waits for the engine, then kills it, as
#: `CoreManager.close` does.
ENGINE_SERVER = BOOTSTRAP + """\
with compass_run.frontend(deployment):
    engine = subprocess.Popen([sys.executable, "-c", sys.argv[2]], stdout=subprocess.PIPE)
    assert engine.stdout.readline() == b"ready\\n"

def close():
    try:
        engine.wait(timeout=10)
    except subprocess.TimeoutExpired:
        engine.kill()

async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        for kind in ("startup", "shutdown"):
            await receive()
            if kind == "shutdown":
                close()
            await send({"type": f"lifespan.{kind}.complete"})
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})

uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]),
            loop=api_server._loop_impl(), lifespan="on", log_level="warning")
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


@contextlib.contextmanager
def _serve(tmp_path, script, *argv):
    """`script` as a co-hosting simulated server on a free port, in a session
    of its own, once ``/health`` answers: its process, port and log. Leaving
    kills what is left of its process group."""
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
            [sys.executable, "-c", script, str(port), *argv],
            cwd=TREE,
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 120
        while not _answers(port):
            assert server.poll() is None, log.read_text()
            assert time.monotonic() < deadline, "no answer in 120 wall seconds"
            time.sleep(0.2)
        yield server, port, log
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait()


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
    with (
        _serve(tmp_path, SERVER) as (server, port, log),
        contextlib.closing(
            http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        ) as client,
    ):
        if held:  # stamped by a client no grant lets in; its connection stays open
            client.request(
                "POST", "/", b"{}", {"tracestate": tracestate_with("", 1.0, 0)}
            )
            deadline = time.monotonic() + 120
            while "stamped (1.0, 0)" not in log.read_text():
                assert time.monotonic() < deadline, "not held in 120 wall seconds"
                time.sleep(0.2)
        server.send_signal(sig)
        server.wait(timeout=30)
    out = log.read_text()
    assert server.returncode == 0 and "frontend_done: True" in out, out
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(summary["schedule"]["refusals"]["reasons"])
    assert reasons == {f"ended:frontend stopped by {sig.name}": 1}
    assert summary["coverage_report"] is True


@pytest.mark.parametrize("group", [False, True], ids=["server", "group"])
def test_a_stopped_servers_summary_carries_its_engines_priced_steps(tmp_path, group):
    with _serve(tmp_path, ENGINE_SERVER, ENGINE) as (server, _, log):
        if group:
            os.killpg(server.pid, signal.SIGTERM)
        else:
            server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)
    out = log.read_text()
    assert server.returncode == 0 and "frontend_done: True" in out, out
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(summary["schedule"]["refusals"]["reasons"])
    assert reasons == {"ended:frontend stopped by SIGTERM": 1}, out
    assert summary["schedule"]["predicted_seconds"] == 0.75, out


def test_an_engine_stopped_holding_a_read_request_hands_in_no_straggler(
    monkeypatch, tmp_path
):
    """The frontend sends request 0 and the engine's input thread reads it; the
    stop comes before the frontend asks for time again, so no grant before the
    stop's ``+inf`` one has released it."""
    endpoint, request = f"inproc:{uuid.uuid4().hex}", "frontend->engine:request#dp0"
    run = {"clock_endpoint": endpoint, "bound_s": 100.0, "out_dir": str(tmp_path)}
    run |= {"admission_path": "serving", "ipc_s": 0.001, "stream_s": 0.002}  # TABLE
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setattr(compass_run, "_member", None)
    server = clock_transport.serve(compass_run._RecordingAuthority(run), endpoint)
    ctx, address = zmq.Context(), f"inproc://{uuid.uuid4().hex}"
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    push.bind(address)
    pull.connect(address)
    engine = LPRuntime(
        compass_run.ENGINE, TABLE, clock_transport.connect(compass_run.ENGINE, endpoint)
    )
    engine.start_run()
    monkeypatch.setattr(clock, "_installed", engine)
    sock = WrappedSocket(engine, pull, request)

    def read():  # the engine's input thread: it takes the request, then waits again
        sock.recv()
        engine.back_at_wait_point()

    def stop():
        conn = clock_transport.connect(compass_run.FRONTEND, endpoint)
        frontend = compass_run._StoppingClock(compass_run.FRONTEND, TABLE, conn)
        frontend.start_run()
        WrappedSocket(frontend, push, request).send(b"request 0")
        while 0 not in engine.arrived[request]:
            time.sleep(0.01)
        frontend.end_run()  # the stop, as the frontend's event loop makes it

    threads = [threading.Thread(target=f, daemon=True) for f in (read, stop)]
    for t in threads:
        t.start()
    said = {"refused_commands": [], "priced_steps": {}, "predicted_memory": {}}
    workers = SimpleNamespace(call_func=lambda name, wait_out: said[name])
    loop = compass_run.engine_loop(SimpleNamespace(runner_mgr=workers))
    with contextlib.suppress(clock.Straggler), loop:  # it hands a raise in first
        while not clock.idle(lambda: math.inf):
            pass
    hand_in = tmp_path / compass_run.LP_FILE.format("engine")
    assert json.loads(hand_in.read_text())["refusals"] == []
    for t in threads:
        t.join(10)
    server.close()
    ctx.destroy(linger=0)
