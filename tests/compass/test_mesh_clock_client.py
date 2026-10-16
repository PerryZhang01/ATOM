# SPDX-License-Identifier: MIT
"""atomesh's clock client against the Python clock authority, over TCP.

The authority serves LPs ``traffic`` and ``router``. ``traffic`` runs here; the
router is the cargo test ``python_authority_releases_a_message_to_the_router``
in ``atom/mesh/tests/compass_clock_test.rs``, which cargo builds and runs. A
relay between the router and the authority records every frame. Skips where
cargo cannot resolve atomesh's dependencies: no cargo, or, as in a gate run in a
network namespace, no registry to reach and no lockfile.
"""

import contextlib
import math
import os
import pathlib
import shutil
import socket
import subprocess
import threading

import pytest

from atom.compass.clock import NER, ChannelTable, ClockAuthority, LpId, LpRegistry
from atom.compass.clock_transport import connect, decode, encode, serve
from atom.compass.clock_transport.stream import _framed, _receive

MESH = pathlib.Path(__file__).resolve().parents[2] / "atom" / "mesh"
INF = math.inf
TRAFFIC, ROUTER = LpId("traffic"), LpId("router")
REQUEST, STREAM = "traffic->router:http", "router->traffic:stream"


def _relay(endpoint: str, frames: list) -> tuple[str, threading.Thread]:
    """An endpoint that forwards one connection's frames to `endpoint` and its
    replies back, appending ``(sender, frame)`` to `frames`, and the thread
    that ends with that connection."""
    listener = socket.create_server(("127.0.0.1", 0))
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)

    def pump(src, dst, sender):
        with src.makefile("rb") as stream:
            while (frame := _receive(stream)) is not None:
                frames.append((sender, frame))
                dst.sendall(_framed(frame))
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)

    def accept():
        with (
            listener,
            listener.accept()[0] as router,
            socket.create_connection((host, int(port))) as clock,
        ):
            up = threading.Thread(target=pump, args=(router, clock, "router"))
            up.start()
            pump(clock, router, "clock")
            up.join()

    relay = threading.Thread(target=accept, daemon=True)
    relay.start()
    return f"tcp://127.0.0.1:{listener.getsockname()[1]}", relay


def test_the_rust_router_binds_is_granted_and_is_released_a_message():
    resolved = shutil.which("cargo") and subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=MESH,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if not resolved or resolved.returncode:
        pytest.skip("cargo cannot resolve atomesh's dependencies here")
    registry = LpRegistry()
    for lp in (TRAFFIC, ROUTER):
        registry.register(lp)
    table = ChannelTable(registry)
    table.declare(REQUEST, TRAFFIC, ROUTER, 0.5, "inline")
    table.declare(STREAM, ROUTER, TRAFFIC, 0.25, "inline")
    server = serve(ClockAuthority(table), "tcp://127.0.0.1:0")
    frames, replies = [], []
    endpoint, relay = _relay(server.endpoint, frames)
    traffic = connect(TRAFFIC, server.endpoint)

    def run_traffic():
        traffic.send((NER, INF, [(REQUEST, 0, 0.5)], INF))
        replies.append(traffic.recv())
        traffic.send((NER, INF, [], INF))
        replies.append(traffic.recv())

    try:
        router = subprocess.Popen(
            [
                *("cargo", "test", "--test", "compass_clock_test", "--"),
                *("--ignored", "--exact"),
                "python_authority_releases_a_message_to_the_router",
            ],
            cwd=MESH,
            env={**os.environ, "ATOM_COMPASS_CLOCK_ENDPOINT": endpoint},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=run_traffic, daemon=True).start()
        out, _ = router.communicate(timeout=1800)
        relay.join(timeout=10)
    finally:
        traffic.close()
        server.close()
    assert router.returncode == 0, out
    assert replies == [(0.75, {STREAM: [(0, 0.75)]}), (INF, {STREAM: []})]
    assert [(sender, frame.decode()) for sender, frame in frames] == [
        ("router", '{"kind":"BIND","lp":"router"}'),
        ("clock", '{"kind":"JOINED","now":0.0,"seqs":{"router->traffic:stream":0}}'),
        ("router", '{"kind":"NER","log":[],"t":"+inf","t_daemon":"+inf"}'),
        (
            "clock",
            '{"G":0.5,"kind":"GRANT","released":{"traffic->router:http":[[0,0.5]]}}',
        ),
        (
            "router",
            '{"kind":"NER","log":[["router->traffic:stream",0,0.75]],"t":"+inf","t_daemon":"+inf"}',
        ),
        ("clock", '{"G":"+inf","kind":"GRANT","released":{"traffic->router:http":[]}}'),
    ]
    assert all(encode(decode(f)) == f for sender, f in frames if sender == "router")
