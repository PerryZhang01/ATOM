# SPDX-License-Identifier: MIT
"""LP `lmcache`: its channel rows, its launch, and its stations on the clock.

The end-to-end test starts the stand-in the way the frontend does, in a process
of its own, against an authority served here over TCP; this test plays the
engine LP on real ZMQ sockets, and the traffic and frontend LPs idle.
"""

import copy
import json
import math
import pickle
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import msgspec
import pytest
import zmq
from conftest import atom_config_double
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.zmq_impl.wire import encode_operation
from test_memory_readings import CONFIG_JSON, DOCUMENT
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.clock.channels import ReceiveMode
from atom.compass.clock_transport import connect, serve
from atom.compass.kv.lmcache_server import LMCacheLP
from atom.compass.spec.rules import SpecRefusal
from atom.utils import clock
from atom.utils.clock import LPRuntime

IPC = 2.0**-10
LOOKUP_S = (2.0**-8, 2.0**-10)  # fixed, per prompt chunk
REQUEST_S = 2.0**-9
LATENCY_S, BANDWIDTH = 2.0**-6, 1e9
CHUNK = 4
TOKENS = tuple(range(1, 4 * CHUNK + 3))  # four whole chunks and a tail
LOOK, XFER = "engine->lmcache:lookup#dp0", "engine->lmcache:transfer#dp0.tp0"
XFER_BACK = "lmcache->engine:transfer#dp0.tp0"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(tmp_path, **entry):
    machine = copy.deepcopy(DOCUMENT)
    machine["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": BANDWIDTH,
        "latency_s": LATENCY_S,
    }
    run = {
        "clock_endpoint": "tcp://127.0.0.1:0",
        "bound_s": 60.0,
        "admission_path": "serving",
        "ipc_s": IPC,
        "stream_s": 2.0**-12,
        "machine": machine,
        "out_dir": str(tmp_path),
        "lmcache": {
            "server": ["--chunk-size", str(CHUNK), "--l1-size-gb", "1"]
            + ["--eviction-policy", "LRU", "--host", "127.0.0.1"]
            + ["--port", str(_free_port())],
            "lookup_s": list(LOOKUP_S),
            "request_s": REQUEST_S,
            **entry,
        },
    }
    return json.loads(json.dumps(run))  # as the run file holds it


OFFLOAD = {"kv_connector": "lmcache_mp", "kv_role": "offload"}
#: The recipe's form: lmcache_mp as one sub of a multi.
MULTI = {"kv_connector": "multi", "connectors": [{"kv_connector": "moriio"}, OFFLOAD]}


def _config(kv=OFFLOAD, **overrides):
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    return atom_config_double(kv_transfer_config=kv, hf_config=hf, **overrides)


def _key(rid, worker_id=None):
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=worker_id,
        token_ids=TOKENS,
        start=0,
        end=4 * CHUNK,
        request_id=rid,
        num_kv_readers=1,
    )


# --- the rows -----------------------------------------------------------------


@pytest.mark.parametrize("stated", [None, 2.0**-12])
def test_the_rows_join_each_dp_rank_and_writer_to_lmcache(tmp_path, stated):
    run = _run(tmp_path, writers=2)
    run["data_parallel_size"] = 2
    if stated is not None:
        run["lmcache_ipc_s"] = stated
    table = compass_run.channel_table(run)
    lm, engine = LpId("lmcache"), LpId("engine")
    kinds = ["lookup#dp0", "lookup#dp1"] + [
        f"transfer#dp{r}.tp{k}" for r in (0, 1) for k in (0, 1)
    ]
    assert {c.name for c in table.channels_into(lm)} == {
        f"engine->lmcache:{k}" for k in kinds
    }
    assert {c.name for c in table.channels_from(lm)} == {
        f"lmcache->engine:{k}" for k in kinds
    }
    lookahead = stated or IPC
    for c in table.channels_into(lm) + table.channels_from(lm):
        assert c.lookahead_s == lookahead
        thread = c.target == engine and "lookup" in c.name
        assert c.receive is (ReceiveMode.THREAD if thread else ReceiveMode.INLINE)
    assert table.distance(engine, lm) + table.distance(lm, engine) == 2 * lookahead
    # Every channel is owned by the member of the DP rank it names.
    members = compass_run._members(table, 2)
    ClockAuthority(table, members=members)
    assert "lmcache->engine:transfer#dp1.tp0" in members[engine]["dp1"]
    assert "engine->lmcache:transfer#dp0.tp1" in members[engine]["dp0"]
    assert "lmcache" in compass_run._hand_ins(run)


def test_a_prefill_decode_run_joins_lmcache_to_the_prefill_engine(tmp_path):
    run = _run(tmp_path)
    run.update(router_s=2.0**-12, kv_write_req_s=2.0**-10)
    table = compass_run.channel_table(run)
    assert {c.target for c in table.channels_from(LpId("lmcache"))} == {
        LpId("engine-P")
    }


def test_a_zero_lmcache_ipc_s_is_refused_as_a_cycle(tmp_path):
    run = _run(tmp_path, writers=1)
    run["lmcache_ipc_s"] = 0.0
    with pytest.raises(ValueError, match="lmcache_ipc_s is 0.0.*cycle"):
        compass_run.channel_table(run)


# --- the launch's refusals ------------------------------------------------------


@pytest.mark.parametrize("kv", [OFFLOAD, MULTI], ids=["lmcache_mp", "multi"])
def test_a_deployment_offloading_with_no_lmcache_entry_is_refused(tmp_path, kv):
    run = _run(tmp_path)
    del run["lmcache"]
    with pytest.raises(ValueError, match="has no lmcache entry"):
        compass_run._launch_lmcache(_config(kv), run, None)


def test_the_decode_side_of_an_offloading_deployment_is_refused(tmp_path):
    with pytest.raises(ValueError, match="serves the prefill side"):
        compass_run._launch_lmcache(_config(MULTI), _run(tmp_path), "D")


@pytest.mark.parametrize("tp, writers", [(2, 1), (1, 2)], ids=["fewer", "more"])
def test_writers_other_than_the_deployments_writer_ranks_are_refused(
    tmp_path, tp, writers
):
    run = _run(tmp_path, writers=writers)
    with pytest.raises(ValueError, match=f"writers is {writers}; .* from {tp} rank"):
        compass_run._launch_lmcache(_config(tensor_parallel_size=tp), run, None)


def test_an_lmcache_entry_with_no_deployment_to_start_it_is_refused(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="offloads no KV.*has an lmcache entry"):
        compass_run._launch_lmcache(_config({}), run, None)
    compass_run._launch_lmcache(_config({}), run, "D")  # prefill starts it


def test_an_entry_without_its_laws_is_refused_by_name(tmp_path):
    run = _run(tmp_path)
    del run["lmcache"]["lookup_s"]
    with pytest.raises(ValueError, match="states no lookup_s"):
        compass_run._launch_lmcache(_config(), run, None)


def test_a_machine_without_a_host_link_is_refused_by_name(tmp_path):
    run = _run(tmp_path)
    del run["machine"]["interconnect"]["host_link"]
    with pytest.raises(SpecRefusal, match="interconnect.host_link.latency_s"):
        compass_run._launch_lmcache(_config(), run, None)


# --- the stations, end to end ---------------------------------------------------


def _send(rt, sock, ch, op, *payloads):
    stamp = rt.stamp_send(ch)
    sock.send_multipart(
        [pickle.dumps((ch, *stamp)), msgspec.msgpack.encode(0), encode_operation(op)]
        + [msgspec.msgpack.encode(p) for p in payloads]
    )


def _take(rt, frames, got):
    ch, arrival, seq = pickle.loads(frames[0])
    rt.check_arrival(ch, arrival, seq)
    payload = msgspec.msgpack.decode(frames[3]) if len(frames) > 3 else None
    got.append((ch, seq, arrival, payload))


def test_server_jobs_start_and_end_against_the_station_laws(tmp_path, monkeypatch):
    run = _run(tmp_path)
    authority = ClockAuthority(compass_run.channel_table(run), bound_s=run["bound_s"])
    served = serve(authority, run["clock_endpoint"])
    run["clock_endpoint"] = served.endpoint
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setattr(clock, "_installed", None)
    started = []
    real = subprocess.Popen
    monkeypatch.setattr(
        compass_run.subprocess,
        "Popen",
        lambda *a, **k: started.append((a, real(*a, **k))),
    )
    compass_run._launch_lmcache(_config(), run, None)
    (argv, *_), proc = started[0]
    chunk_bytes = int(argv[-1])
    idle = [connect(LpId(lp), served.endpoint) for lp in ("traffic", "frontend")]
    for conn in idle:
        conn.send(("NER", math.inf, [], math.inf))

    table = compass_run.channel_table(run)
    rt = LPRuntime(LpId("engine"), table, connect(LpId("engine"), served.endpoint))
    ctx = zmq.Context()
    url = f"tcp://127.0.0.1:{run['lmcache']['server'][-1]}"
    look, xfer = ctx.socket(zmq.DEALER), ctx.socket(zmq.DEALER)
    look.connect(url)
    xfer.connect(url)
    got, stop = [], threading.Event()

    def polling_loop():  # lmcache's client thread: takes each lookup reply
        while not stop.is_set():
            if look.poll(10):
                _take(rt, look.recv_multipart(), got)
                ch, seq = got[-1][:2]
                while seq not in rt.released[ch]:
                    time.sleep(0.001)
                with rt.lock:
                    rt.count_done_locked(ch, seq)

    def take_transfer_replies():
        while rt.released[XFER_BACK] - rt.arrived[XFER_BACK]:
            _take(rt, xfer.recv_multipart(), got)

    taker = threading.Thread(target=polling_loop, daemon=True)
    taker.start()
    try:
        _send(rt, look, LOOK, "lookup", _key("r0"), 1)
        _send(rt, look, LOOK, "query_prefetch_status", "r0")
        _send(rt, xfer, XFER, "retrieve", _key("r0", 0), 1, [[0]], b"", 0)
        _send(rt, xfer, XFER, "store", _key("r0", 0), 1, [[0]], b"")
        while sum(g[0] == XFER_BACK for g in got) < 2:
            rt.next_event(math.inf)
            take_transfer_replies()
        _send(rt, xfer, XFER, "store", _key("r0", 0), 1, [[0]], b"")
        _send(rt, look, LOOK, "lookup", _key("r1"), 1)
        _send(rt, look, LOOK, "query_prefetch_status", "r1")
        while rt.next_event(math.inf) != math.inf:
            take_transfer_replies()
        assert proc.wait(120) == 0
    finally:
        stop.set()
        taker.join()
        if proc.poll() is None:
            proc.kill()
        look.close(linger=0)
        xfer.close(linger=0)
        ctx.term()
        served.close()

    # Hand-derived: both stations start at the requests' arrival; the request
    # loop serves its jobs one after the other, the copy engine a retrieve of
    # nothing resident, then the store; a copy is priced over the chunks it
    # moves, so the failed retrieve and the second store of the same range
    # cost the link's latency alone.
    a = IPC
    lookup = LOOKUP_S[0] + LOOKUP_S[1] * 4
    end1 = a + lookup
    end2 = end1 + REQUEST_S
    end3 = a + LATENCY_S
    end4 = end3 + LATENCY_S + 4 * chunk_bytes / BANDWIDTH
    a2 = end4 + IPC + IPC  # the second store and pair, sent at the store's reply
    end5 = a2 + lookup
    end6 = end5 + REQUEST_S
    end7 = a2 + LATENCY_S
    back = "lmcache->engine:"
    expected = [
        (back + "lookup#dp0", 0, end1 + IPC, None),
        (back + "lookup#dp0", 1, end2 + IPC, 0),  # nothing stored yet
        (back + "transfer#dp0.tp0", 0, end3 + IPC, [b"", False]),
        (back + "transfer#dp0.tp0", 1, end4 + IPC, [b"", True]),
        (back + "lookup#dp0", 2, end5 + IPC, None),
        (back + "lookup#dp0", 3, end6 + IPC, 4),  # the store's four chunks
        (back + "transfer#dp0.tp0", 2, end7 + IPC, [b"", True]),
    ]
    assert sorted(got, key=lambda g: g[2]) == got  # taken in timestamp order
    assert [g[:2] for g in got] == [e[:2] for e in expected]
    assert [g[2] for g in got] == pytest.approx([e[2] for e in expected])
    assert [g[3] for g in got] == [e[3] for e in expected]
    handed = json.loads((tmp_path / "lp-lmcache.json").read_text())
    assert handed == {"refusals": []}


# --- the server's ticks ---------------------------------------------------------


class _Authority:
    """An LP runtime whose authority grants each NER at its earlier deadline,
    up to `until`, then ``+inf``; its grant is always already here."""

    def __init__(self, until):
        self.now, self.until, self.asked = 0.0, until, []
        self.released = self.arrived = {}
        self.conn, self.peer = socket.socketpair()
        self.peer.send(b"!")

    def ask_next_event(self, t, t_daemon):
        self.asked.append((t, t_daemon))

    def take_grant(self):
        grant = min(self.asked[-1])
        self.now = grant if grant <= self.until else math.inf
        return self.now


def test_the_eviction_and_session_ticks_land_on_their_grids_as_daemon_deadlines():
    ticks = {"evict": [], "expire_sessions": []}
    server = SimpleNamespace(**{name: times.append for name, times in ticks.items()})
    rt, ctx = _Authority(until=130.0), zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    try:
        LMCacheLP(
            rt, sock, server, lookup_s=LOOKUP_S, request_s=REQUEST_S, copy=None
        ).serve()
    finally:
        sock.close(linger=0)
        ctx.term()
        rt.conn.close()
        rt.peer.close()
    # The session sweep runs at the server's start and every 60 s after; the
    # eviction loop every second from one second in.
    assert ticks["expire_sessions"] == [0.0, 60.0, 120.0]
    assert ticks["evict"] == [float(s) for s in range(1, 131)]
    # Neither keeps the run alive: each is asked as a daemon deadline only.
    assert rt.asked == [(math.inf, float(s)) for s in range(1, 132)]
