# SPDX-License-Identifier: MIT
"""The ``tcp`` carrier of `atom.compass.clock_transport`, against the in-process one.

The same scripted runs go over both carriers and each LP's reply frames are
compared byte for byte. The socket server runs in a thread of this process,
except in the one test that starts it in a child process.
"""

import math
import os
import pathlib
import socket
import subprocess
import sys
import threading

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    BackdatedEvent,
    ChannelTable,
    ClockAbort,
    ClockAuthority,
    LpId,
    LpRegistry,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.clock_transport import MalformedMessage, connect, serve

from .test_clock_transport import (
    AB,
    IPC_S,
    SCRIPTS,
    STREAM_S,
    A,
    B,
    _ask,
    _drive,
    _later,
    _result,
    _table,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
INF = math.inf
LOCAL = "tcp://127.0.0.1:0"

THREE_LP = {"admission_path": "serving", "ipc_s": IPC_S, "stream_s": STREAM_S}
FIVE_LP = {**THREE_LP, "router_s": 0.003, "kv_write_req_s": 0.004}

#: One request through prefill and decode, on the five-LP table: the prefill
#: engine computes and answers, the relay hands the request to decode, decode
#: asks prefill for the KV and streams the output. Every LP ends in NER(+inf).
PREFILL_DECODE_SCRIPTS = {
    "traffic": [("send", "traffic->frontend-P:http"), (NER, INF), (NER, INF)],
    "frontend-P": [
        (NER, INF),
        ("send", "frontend-P->engine-P:request#dp0"),
        (NER, INF),
        ("send", "frontend-P->frontend-D:relay"),
        (NER, INF),
    ],
    "engine-P": [
        (NER, INF),
        (TAR, 0.05),
        ("send", "engine-P->frontend-P:output#dp0"),
        (NER, INF),
        (NER, INF),
    ],
    "frontend-D": [
        (NER, INF),
        ("send", "frontend-D->engine-D:request#dp0"),
        (NER, INF),
        ("send", "frontend-D->traffic:stream"),
        (NER, INF),
    ],
    "engine-D": [
        (NER, INF),
        ("send", "engine-D->engine-P:kv_write_req"),
        (TAR, 0.02),
        ("send", "engine-D->frontend-D:output#dp0"),
        (NER, INF),
    ],
}

RUNS = {
    "three-lp": (single_engine_table, THREE_LP, SCRIPTS),
    "five-lp": (prefill_decode_table, FIVE_LP, PREFILL_DECODE_SCRIPTS),
}

CHILD = """
import sys
from atom.compass.clock import ClockAuthority, prefill_decode_table
from atom.compass.clock_transport import serve

server = serve(ClockAuthority(prefill_decode_table(**{kwargs!r})), {endpoint!r})
print(server.endpoint, flush=True)
sys.stdin.read()
"""


class _Tap:
    """A reply slot that keeps every frame its LP reads."""

    def __init__(self, slot):
        self.slot, self.frames = slot, []

    def get(self):
        self.frames.append(self.slot.get())
        return self.frames[-1]


@pytest.fixture
def served():
    servers = []

    def start(authority, endpoint):
        servers.append(serve(authority, endpoint))
        return servers[-1].endpoint

    yield start
    for server in servers:
        server.close()


def _connect(lp, endpoint):
    """`connect`, bounded by `_result`'s wait."""
    return _result(_later(connect, lp, endpoint))


def _replies(endpoint, table, scripts):
    """Run `scripts` against `endpoint`: each LP's request count and reply frames."""
    conns = {name: _connect(LpId(name), endpoint) for name in scripts}
    taps = {}
    for name, conn in conns.items():
        conn._slot = taps[name] = _Tap(conn._slot)
    runs = {name: _later(_drive, conns[name], table, scripts[name]) for name in scripts}
    sent = {name: _result(run)[0] for name, run in runs.items()}
    for conn in conns.values():
        conn.close()
    return sent, {name: tap.frames for name, tap in taps.items()}


@pytest.mark.parametrize("run", RUNS)
def test_a_scripted_run_gives_each_lp_the_same_reply_bytes_over_both_carriers(
    served, run
):
    build, kwargs, scripts = RUNS[run]
    table = build(**kwargs)
    over = {
        endpoint: _replies(served(ClockAuthority(table), endpoint), table, scripts)
        for endpoint in ("inproc:conformance", LOCAL)
    }
    (sent, inproc), (tcp_sent, tcp) = over.values()
    assert inproc == tcp
    assert sent == tcp_sent == {name: len(inproc[name]) for name in scripts}
    # Not vacuous: every LP was granted time and read the finish.
    for frames in inproc.values():
        assert len(frames) > 1 and frames[-1].startswith(b'{"G":"+inf"')


def test_a_standalone_clock_runs_in_a_process_of_its_own():
    table = prefill_decode_table(**FIVE_LP)
    # The child serves until its stdin closes; it is killed so that a child
    # stalled before it serves cannot hold up leaving this block.
    with subprocess.Popen(
        [sys.executable, "-c", CHILD.format(kwargs=FIVE_LP, endpoint=LOCAL)],
        env=dict(os.environ, PYTHONPATH=str(REPO)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            endpoint = _result(_later(child.stdout.readline)).strip()
            assert endpoint.startswith("tcp://127.0.0.1:")
            standalone = _replies(endpoint, table, PREFILL_DECODE_SCRIPTS)
        finally:
            child.kill()
    inproc = serve(ClockAuthority(table), "inproc:standalone-control")
    try:
        assert standalone == _replies(inproc.endpoint, table, PREFILL_DECODE_SCRIPTS)
    finally:
        inproc.close()


def test_a_held_reply_over_tcp_delays_no_other_lps_reply(served):
    registry = LpRegistry()
    a, b, c = (LpId(name) for name in "abc")
    for lp in (a, b, c):
        registry.register(lp)
    table = ChannelTable(registry)
    table.declare("a->c:m", a, c, 1.0, "inline")
    table.declare("c->a:m", c, a, 1.0, "inline")
    endpoint = served(ClockAuthority(table), LOCAL)
    conns = {lp: _connect(lp, endpoint) for lp in (a, b, c)}
    _result(_later(conns[b].send, (NER, INF, [], INF)))
    held = _later(conns[b].recv)

    def steps(conn):
        replies = []
        for k in range(1, 51):
            conn.send((TAR, k * 0.5, [], INF))
            replies.append(conn.recv())
        return replies

    a_run, c_run = _later(steps, conns[a]), _later(steps, conns[c])
    replies = _result(a_run) + _result(c_run)
    assert len(replies) == 100
    assert held.empty()
    for lp in (a, c):
        conns[lp].send((NER, INF, [], INF))
    assert _result(held) == (INF, {})
    for conn in conns.values():
        conn.close()


def test_a_standalone_clock_says_which_port_it_ended_up_on(served):
    endpoint = served(ClockAuthority(prefill_decode_table(**FIVE_LP)), LOCAL)
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    assert host == "127.0.0.1" and int(port) > 0


def test_a_name_the_clock_does_not_hold_is_refused_over_tcp(served):
    endpoint = served(ClockAuthority(prefill_decode_table(**FIVE_LP)), LOCAL)
    with pytest.raises(KeyError, match="not a participant"):
        _connect(LpId("pp-stage-7"), endpoint)
    traffic = _connect(LpId("traffic"), endpoint)
    with pytest.raises(KeyError, match="already bound"):
        _connect(LpId("traffic"), endpoint)
    traffic.close()


def _fake_clock(answer: bytes):
    """A listener that reads one bind frame, writes `answer` and hangs up."""
    listener = socket.create_server(("127.0.0.1", 0))

    def once():
        sock, _ = listener.accept()
        with sock:
            size = int.from_bytes(sock.recv(4), "big")
            sock.recv(size)
            sock.sendall(answer)
        listener.close()

    threading.Thread(target=once, daemon=True).start()
    return f"tcp://127.0.0.1:{listener.getsockname()[1]}"


@pytest.mark.parametrize(
    "answer, match",
    [
        (b"\x00\x00\x00\x40{", "inside a frame"),
        (b"\x00\x00", "inside a frame"),
        (b"\x7f\xff\xff\xff", "2147483647-byte frame is over"),
        (b"", "with no reply"),
    ],
    ids=["mid-frame", "mid-length", "oversized", "no-reply"],
)
def test_a_clock_that_hangs_up_mid_frame_is_a_malformed_message(answer, match):
    with pytest.raises(MalformedMessage, match=match):
        _connect(LpId("a"), _fake_clock(answer))


def test_a_refusal_crosses_tcp_as_itself_and_the_connection_serves_on(served):
    ca = ClockAuthority(_table())
    a = _connect(A, served(ca, LOCAL))
    # A frame the loop refuses to queue is answered in place of a reply.
    a.send(("GRANT", 1.0, {}))
    with pytest.raises(ValueError, match="sends only TAR, NER"):
        _result(_later(a.recv))
    assert _ask(a, (TAR, 1.0, [], INF)) == (1.0, {})
    with pytest.raises(BackdatedEvent, match="before a's clock 1.0") as refused:
        _ask(a, (TAR, 2.0, [(AB, 0, 1.1)], INF))
    assert refused.value.table == ca.lp_table()
    a.close()
    with pytest.raises(ValueError, match="closed file"):
        a.send((TAR, 2.0, [], INF))


@pytest.mark.parametrize("endpoint", ["tcp://127.0.0.1", "tcp://:9", "tcp:9"])
def test_a_tcp_endpoint_without_host_and_port_is_refused(endpoint):
    with pytest.raises(ValueError, match="is not tcp://<host>:<port>"):
        _connect(LpId("a"), endpoint)


def test_an_engine_closed_while_it_waits_in_ner_ends_the_run_for_every_lp(served):
    """engine-P's connection ends with its NER out, as when its process exits
    waiting for a grant. Traffic is bound and busy, so no finish can come."""
    ca = ClockAuthority(prefill_decode_table(**FIVE_LP))
    endpoint = served(ca, LOCAL)
    conns = {name: _connect(LpId(name), endpoint) for name in PREFILL_DECODE_SCRIPTS}
    waiting = [name for name in conns if name != "traffic"]
    for name in waiting:
        _result(_later(conns[name].send, (NER, INF, [], INF)))
    pending = [_later(conns[name].recv) for name in waiting if name != "engine-P"]
    conns["engine-P"].close()
    named = "^engine-P closed its connection"
    for box in pending:
        with pytest.raises(ClockAbort, match=named) as abort:
            _result(box)
        assert abort.value.table == ca.lp_table()
    # The busy LP's next request gets the same refusal.
    with pytest.raises(ClockAbort, match=named):
        _ask(conns["traffic"], (NER, INF, [], INF))
    for conn in conns.values():
        conn.close()


def test_a_bind_whose_joined_reply_cannot_be_framed_is_refused(served, monkeypatch):
    """The refusal answers the bind, and `a` stays bound: its connection
    ending ends the run."""
    ca = ClockAuthority(_table())
    endpoint = served(ca, LOCAL)
    b = _connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    parked = _later(b.recv)
    monkeypatch.setattr(ca, "on_bind", lambda lp: (math.nan, {}))
    with pytest.raises(ValueError, match=r"^\('JOINED', nan, \{\}\) is not a message"):
        _connect(A, endpoint)
    with pytest.raises(ClockAbort, match="^a closed its connection"):
        _result(parked)
    b.close()
