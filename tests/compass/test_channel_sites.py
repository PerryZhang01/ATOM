# SPDX-License-Identifier: MIT
"""ATOM's channel sockets, pollers and output queue, made through `atom.utils.zmq_shim`.

Every channel send and receive row of the sync inventory (K4, K5) sits in a
module that imports the shim as `zmq`, and the kinds its call carries are named
by the function that records its addresses. Then ATOM's own threads and
transport run over real zmq sockets with a runtime installed: the PP stages, the
engine's input and output threads and the frontend's READY wait and output
thread.
"""

import ast
import math
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from aiter_stub import stubbed_aiter

from atom.compass.audit import sync_scan
from atom.compass.clock import LpId, single_engine_table
from atom.distributed.pp_transport import PPStageTransport
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.sequence import get_exit_sequence
from atom.utils import clock, get_open_zmq_ipc_path, make_zmq_socket, zmq_shim
from atom.utils.clock import LPRuntime, WrappedPoller, WrappedSocket
from tests.compass.clock.test_lp_runtime import FakeConn, _owner

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.engine_core_mgr import CoreManager

IPC = 0.001
REQ = "frontend->engine:request#dp0"
OUT = "engine->frontend:output#dp0"
ENGINE = ("EngineCore.__init__",)
MGR = ("CoreManager.__init__", "DisaggCoreManager.__init__._connect_proc")

#: Row symbol -> (the kind of each channel its call carries, the functions that
#: record the addresses of those channels). ``relay`` is the output queue the
#: function makes with `clock.relay_queue`.
SITES = {
    "PPStageTransport.send_metadata": (("meta",), ENGINE),
    "PPStageTransport.recv_metadata": (("meta",), ENGINE),
    "PPStageTransport.send_tokens": (("tokens",), ENGINE),
    "PPStageTransport.recv_tokens": (("tokens",), ENGINE),
    "PPStageTransport.send_kv_status": (("kv_status",), ENGINE),
    "PPStageTransport.recv_kv_status": (("kv_status",), ENGINE),
    "EngineCore.process_input_sockets": (("request", "control"), ENGINE),
    "EngineCore.process_output_sockets": (("output", "relay"), ENGINE),
    "PrefillEngineCore._process_engine_step": (("prefill_done",), ENGINE),
    "DecodeEngineCore._recv_prefill_done": (("prefill_done",), ENGINE),
    "DecodeEngineCore._send_block_assignment": (("block_assignment",), ENGINE),
    "PrefillEngineCore._recv_block_assignments": (("block_assignment",), ENGINE),
    "CoreManager._send_request": (("request",), MGR),
    "CoreManager._send_control": (("control",), MGR),
    "CoreManager._create_output_thread.process_outputs_socket": (("output",), MGR),
}

#: Function -> the expression each `clock.name_endpoints` keyword must carry there;
#: ``None`` is a ``**`` mapping keyed by the parameter names.
ADDRESSES = {
    "EngineCore.__init__": {
        "input_address": "input_address",
        "control_address": "self.control_address",
        "output_address": "output_address",
        "config": "config",
    },
    "CoreManager.__init__": {None: "addresses"},
    "DisaggCoreManager.__init__._connect_proc": {
        "input_address": "in_addr",
        "control_address": "ctrl_addr",
        "output_address": "out_addr",
    },
}


def _config(meta=(), token="", kv="", p2d="", d2p=""):
    pc = SimpleNamespace(pp_meta_addrs=list(meta), pp_token_addr=token)
    pc.pp_kv_status_addr = kv
    return SimpleNamespace(parallel_config=pc, disagg_p2d_addr=p2d, disagg_d2p_addr=d2p)


class _Naming(ast.NodeVisitor):
    """Function qualname -> its `clock.name_endpoints` and `clock.relay_queue` calls."""

    def __init__(self):
        self.scope, self.calls = [], {}

    def _push(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_ClassDef = _push

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("name_endpoints", "relay_queue"):
            self.calls.setdefault(".".join(self.scope), []).append(node)
        self.generic_visit(node)


def _imports_the_shim(tree) -> bool:
    shim = plain = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            plain |= any(a.name.split(".")[0] == "zmq" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            plain |= (node.module or "").split(".")[0] == "zmq"
            shim |= node.module == "atom.utils" and any(
                (a.name, a.asname) == ("zmq_shim", "zmq") for a in node.names
            )
    return shim and not plain


def _kinds_named(monkeypatch, *config):
    rt = LPRuntime(LpId("engine"), _engine_table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    clock.name_endpoints(0, "a:in", "a:ctl", "a:out", *config)
    return {kind.split("#")[0] for kind in rt.endpoints.values()}


def test_every_channel_row_maps_to_an_address_its_wiring_names(monkeypatch):
    rows = [
        r for r in sync_scan.load_inventory()["sites"] if r["mechanism"] in ("K4", "K5")
    ]
    # `flush_pp_send` waits for `pp_ack`, which the stage loop receives before
    # the call, not on an ATOM socket.
    acks = [r for r in rows if "flush_pp_send" in r["expr"]]
    # Compass's own KV connector wraps its socket in `clock.WrappedSocket` on
    # the channel it reads from the table, and a pipeline stage's `pp_data`
    # rides the PP CPU group; no address names either channel. An lmcache_mp
    # LOOKUP leaves on lmcache's own client socket, which no ATOM address names.
    own_files = (
        "atom/compass/kv/connector.py",
        "atom/compass/runner/pp_send.py",
        "atom/kv_transfer/offload/chunked_scheduler.py",
    )
    own = [r for r in rows if r["file"] in own_files]
    rows = [r for r in rows if r not in acks + own]
    root = sync_scan.repo_root_from_here()
    trees = {f: ast.parse((root / f).read_text()) for f in {r["file"] for r in rows}}
    naming = _Naming()
    for tree in trees.values():
        naming.visit(tree)
    base = _kinds_named(monkeypatch)
    full = _kinds_named(monkeypatch, _config(["a:m"], "a:t", "a:k", "a:p", "a:d"))
    for r in rows:
        print(r["mechanism"], r["id"], "->", *SITES.get(r["symbol"], ("?",))[0])
    assert rows and acks and own
    assert sorted({r["symbol"] for r in rows}) == sorted(SITES), "unmapped or stale"
    passed = {
        fn: {k.arg: ast.unparse(k.value) for k in c.keywords}
        for fn, calls in naming.calls.items()
        for c in calls
        if c.func.attr == "name_endpoints"
    }
    assert passed == ADDRESSES

    def covers(call, kind):
        if call.func.attr == "relay_queue":
            return kind == "relay"
        return kind in (full if any(k.arg == "config" for k in call.keywords) else base)

    unnamed = [
        (symbol, kind, fn)
        for symbol, (kinds, fns) in SITES.items()
        for fn in fns
        for kind in kinds
        if not any(covers(c, kind) for c in naming.calls.get(fn, ()))
    ]
    assert not unnamed
    assert [f for f, tree in trees.items() if not _imports_the_shim(tree)] == []


@pytest.fixture
def ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)


@pytest.fixture
def shim_ctx():
    """The context the shim's `Context` makes while a runtime is installed."""
    ctx = zmq_shim._Context()
    yield ctx
    ctx.destroy(linger=0)


def _engine_table():
    return single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=IPC)


def _pp_table(stages):
    return single_engine_table(
        admission_path="serving", ipc_s=IPC, stream_s=IPC, pp=stages
    )


def _stage(monkeypatch, table, rank, stages, ctx, conn=None, owner=None):
    rt = LPRuntime(LpId(f"engine-pp{rank}"), table, conn, owner=owner)
    monkeypatch.setattr(clock, "_installed", rt)
    addrs = [""] + [f"inproc://meta{k}" for k in range(1, stages)]
    clock.name_endpoints(0, "", "", "", _config(addrs, "inproc://tok", "inproc://kv"))
    stage = PPStageTransport(
        rank, stages, addrs, "inproc://tok", "inproc://kv", ctx=ctx
    )
    stage.rt = rt
    return stage


def test_the_shim_is_pyzmq_until_a_runtime_is_installed(monkeypatch):
    real = zmq.Context.instance()
    swapped = (zmq_shim.Context, zmq_shim.Poller, zmq_shim.Socket)
    assert swapped == (zmq.Context, zmq.Poller, zmq.Socket)
    assert (zmq_shim.PUSH, zmq_shim.error, zmq_shim.asyncio) == (
        zmq.PUSH,
        zmq.error,
        zmq.asyncio,
    )
    assert type(clock.relay_queue(0)) is queue.Queue
    monkeypatch.setattr(
        clock, "_installed", LPRuntime(LpId("engine"), _engine_table(), None)
    )
    ctx = zmq_shim.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.bind("inproc://unnamed")
    made = (
        ctx is real,
        isinstance(sock, zmq_shim.Socket),
        type(sock),
        zmq_shim.Poller(),
    )
    ctx.destroy(linger=0)  # before asserting: a later `instance()` must not inherit it
    assert made[:3] == (False, True, zmq_shim._Socket)
    assert type(made[3]) is WrappedPoller


def test_an_option_set_by_attribute_on_a_shim_socket_reaches_pyzmq(
    monkeypatch, shim_ctx
):
    rt = LPRuntime(LpId("engine"), _engine_table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    clock.name_endpoints(0, "", "", "inproc://named")
    plain, named = shim_ctx.socket(zmq.PULL), shim_ctx.socket(zmq.PUSH)
    named.connect("inproc://named")
    for sock in (plain, named):
        sock.rcvtimeo = 123
    assert type(named) is zmq_shim._ChannelSocket
    assert [s.raw.getsockopt(zmq.RCVTIMEO) for s in (plain, named)] == [123, 123]


def test_an_engine_names_its_own_and_its_configured_addresses(monkeypatch):
    rt = LPRuntime(LpId("engine"), _engine_table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    clock.name_endpoints(1, "in", "ctl", "out", _config(["", "m1"], "t", "k", "p", "d"))
    assert rt.endpoints == {
        "in": "request#dp1",
        "ctl": "control#dp1",
        "out": "output#dp1",
        "m1": "meta#pp1",
        "t": "tokens",
        "k": "kv_status",
        "p": "prefill_done",
        "d": "block_assignment",
    }


def test_pp_stages_talk_through_their_wrapped_sockets(monkeypatch):
    zmq.Context.instance()  # pyzmq's own process-wide context exists already
    table = _pp_table(2)
    last = _stage(monkeypatch, table, 1, 2, None)  # takes `zmq.Context.instance()`
    head = _stage(monkeypatch, table, 0, 2, last._ctx)
    head.send_metadata("batch")
    assert last.recv_metadata(timeout_ms=5000) == "batch"
    last.send_tokens("out")
    assert head.recv_tokens(timeout_ms=5000) == "out"
    last.send_kv_status("kv")
    assert head.recv_kv_status(timeout_ms=5000) == [(1, "kv")]
    socks = (*head._meta_send, head._token_recv, head._kv_status_recv)
    socks += (last._meta_recv, last._token_send, last._kv_status_send)
    assert [s.ch.split(":")[1] for s in socks] == [
        "meta#pp1",
        "tokens",
        "kv_status",
    ] * 2
    head.close()
    last.close()


def test_an_idle_downstream_stage_polls_on_a_daemon_deadline(monkeypatch, shim_ctx):
    """Its poll waits for the head's batch and keeps no run alive."""
    stage = _stage(monkeypatch, _pp_table(2), 1, 2, shim_ctx, FakeConn((0.1, {})))
    stage.rt.start_run()
    assert stage.recv_metadata(timeout_ms=100) is None
    assert stage.rt.conn.sent == [("NER", math.inf, [], 0.1)]
    stage.close()


def test_a_head_hears_each_downstream_stage_on_its_own_channel(monkeypatch):
    """Both stages' KV status reach the head's one socket; it waits for each
    released frame, the later stage's sent after the grant."""
    table = _pp_table(3)
    late = threading.Timer(0.2, lambda: last.send_kv_status("b"))
    last = _stage(monkeypatch, table, 2, 3, None, owner=late)
    mid = _stage(monkeypatch, table, 1, 3, last._ctx)
    kv = [f"engine-pp{k}->engine-pp0:kv_status" for k in (1, 2)]
    grant = (1.0, {ch: [(0, IPC)] for ch in kv})
    head = _stage(monkeypatch, table, 0, 3, last._ctx, FakeConn(grant))
    head.send_metadata("batch")
    assert [s.recv_metadata(timeout_ms=5000) for s in (mid, last)] == ["batch"] * 2
    assert [s.ch.split(":")[1] for s in head._meta_send] == ["meta#pp1", "meta#pp2"]
    assert head._kv_status_recv.into == tuple(kv)
    for s in (head, mid, last):
        s.rt.start_run()
    mid.send_kv_status("a")
    head.rt.advance_to(1.0)
    late.start()
    assert head.recv_kv_status(timeout_ms=0) == [(1, "a"), (2, "b")]
    for s in (head, mid, last):
        s.close()


def test_the_engine_input_thread_takes_a_request_sent_by_the_frontend(
    monkeypatch, shim_ctx
):
    table = _engine_table()
    frontend = LPRuntime(LpId("frontend"), table, None)
    owner, call = _owner()
    conn = FakeConn((1.0, {REQ: [(0, IPC)]}))
    engine_rt = LPRuntime(LpId("engine"), table, conn, owner=owner)
    addrs = [get_open_zmq_ipc_path() for _ in range(2)]
    monkeypatch.setattr(clock, "_installed", frontend)
    clock.name_endpoints(0, *addrs, "")
    mgr = CoreManager.__new__(CoreManager)
    mgr.input_sockets, mgr.control_sockets = (
        [make_zmq_socket(shim_ctx, a, zmq.ROUTER, bind=True)] for a in addrs
    )
    mgr._control_send_lock = threading.Lock()
    monkeypatch.setattr(clock, "_installed", engine_rt)
    clock.name_endpoints(0, *addrs, "")
    engine = EngineCore.__new__(EngineCore)
    engine.label, engine.input_queue = "engine", queue.Queue()
    thread = threading.Thread(
        target=engine.process_input_sockets, args=addrs, daemon=True
    )
    thread.start()
    assert mgr.input_sockets[0].raw.poll(5000) and mgr.control_sockets[0].raw.poll(5000)
    mgr.engine_core_identities = [mgr.input_sockets[0].recv_multipart()[0]]
    mgr.control_identities = [mgr.control_sockets[0].recv_multipart()[0]]
    frontend.start_run()
    mgr._send_request(
        0, pickle.dumps((EngineCoreRequestType.ADD, [SimpleNamespace(id=7)]))
    )
    engine_rt.start_run()
    call(engine_rt.advance_to, 1.0)  # returns once the input thread is back at its poll
    [request] = engine.input_queue.get(timeout=5)
    assert request.id == 7
    # Receipt is stamped on the engine's clock: the arrival, one IPC after 0.
    assert request.queue_timing.received_at == IPC
    assert engine_rt.handled[REQ] == {0}
    frontend.end_run()
    engine_rt.end_run()
    mgr._send_control(0, pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
    thread.join(5)
    assert not thread.is_alive()


def test_a_stamped_frame_is_refused_by_recv_multipart(ctx):
    pull = ctx.socket(zmq.PULL)
    address = f"tcp://127.0.0.1:{pull.bind_to_random_port('tcp://127.0.0.1')}"
    push = ctx.socket(zmq.PUSH)
    push.connect(address)
    frontend = LPRuntime(LpId("frontend"), _engine_table(), None)
    frontend.start_run()
    WrappedSocket(frontend, push, REQ).send(b"")
    rx = WrappedSocket(LPRuntime(LpId("engine"), _engine_table(), None), pull, REQ)
    with pytest.raises(RuntimeError, match="seq 0 read by recv_multipart"):
        rx.recv_multipart()


def test_the_engine_output_thread_sends_each_item_with_its_put_stamp(monkeypatch, ctx):
    rt = LPRuntime(LpId("engine"), _engine_table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    pull = ctx.socket(zmq.PULL)
    address = f"tcp://127.0.0.1:{pull.bind_to_random_port('tcp://127.0.0.1')}"
    clock.name_endpoints(0, "", "", address)
    engine = EngineCore.__new__(EngineCore)
    engine.label, engine.output_queue = "engine", clock.relay_queue(0)
    thread = threading.Thread(
        target=engine.process_output_sockets, args=(address,), daemon=True
    )
    thread.start()
    frames = []
    rt.start_run()
    for item in (("METRICS", {"m": 1}), [get_exit_sequence()]):
        engine.output_queue.put_nowait(item)
        if pull.poll(5000):
            frames.append(pull.recv_multipart())
        rt.end_run()  # the run closes only once every stamped item has gone out
    thread.join(5)
    assert [pickle.loads(h) for h, _ in frames] == [(OUT, IPC, 0), (OUT, None, None)]
    assert pickle.loads(frames[0][1]) == (EngineCoreRequestType.METRICS, {"m": 1})


def test_the_frontend_waits_for_ready_and_reads_output_through_the_wrapper(
    monkeypatch, ctx, shim_ctx
):
    table = _engine_table()
    monkeypatch.setattr(clock, "_installed", LPRuntime(LpId("frontend"), table, None))
    address = get_open_zmq_ipc_path()
    clock.name_endpoints(0, "", "", address)
    mgr = CoreManager.__new__(CoreManager)
    mgr.ctx, mgr.label, mgr.max_pool_tokens, mgr.latest_metrics = (
        shim_ctx,
        "fe",
        None,
        {},
    )
    mgr.output_sockets = [make_zmq_socket(shim_ctx, address, zmq.PULL, bind=True)]
    push = ctx.socket(zmq.PUSH)
    push.connect(address)
    engine = WrappedSocket(LPRuntime(LpId("engine"), table, None), push, OUT)
    engine.send(pickle.dumps((EngineCoreRequestType.READY, {"max_pool_tokens": 7})))
    mgr._wait_for_all_ready_signals()
    assert mgr.max_pool_tokens == 7
    thread = mgr._create_output_thread(0, mgr.output_sockets[0], "inproc://stop")
    thread.start()
    engine.send(pickle.dumps((EngineCoreRequestType.METRICS, {"m": 2})))
    deadline = time.monotonic() + 5
    while 0 not in mgr.latest_metrics and time.monotonic() < deadline:
        time.sleep(0.01)
    stop = shim_ctx.socket(zmq.PAIR)
    stop.connect("inproc://stop")
    stop.send(b"")
    thread.join(5)
    assert mgr.latest_metrics == {0: {"m": 2}} and mgr.output_sockets[0].closed
