# SPDX-License-Identifier: MIT
"""The detokenization station: one width-1 station per frontend output thread.

ATOM's own output thread (`CoreManager._create_output_thread`) reads engine
output through the zmq shim on a frontend whose loop is a `CompassEventLoop`,
against the co-hosted clock authority. Stream updates go through ATOM's
`StreamBatchDispatcher`, whose tokenizer's decode `wrap_decode` charges.
"""

import json
import math
import pickle
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import zmq
from aiter_stub import stubbed_aiter

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.streaming_dispatch import StreamBatchDispatcher
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.request import RequestOutput
from atom.sampling_params import SamplingParams
from atom.utils import clock, get_open_zmq_ipc_path, make_zmq_socket, zmq_shim
from atom.utils.clock import LPRuntime, WrappedSocket
from atom.utils.compass_loop import CompassEventLoop, wrap_decode
from tests.compass.test_tokenizer_station import _entry

with stubbed_aiter():
    from atom.model_engine.engine_core_mgr import CoreManager

IPC = 0.001
TABLE = single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=0.002)
OUT = "engine->frontend:output#dp0"


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def _run(monkeypatch, callback, token_counts) -> None:
    """Send one STREAM message per count at engine time 0, all arriving at IPC,
    to an output thread whose callback is `callback(loop)`; run to +inf."""
    endpoint = f"inproc:test-detok-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    traffic = clock_transport.connect(LpId("traffic"), endpoint)
    traffic.send((NER, math.inf, [], math.inf))
    frontend = LPRuntime(
        LpId("frontend"), TABLE, clock_transport.connect(LpId("frontend"), endpoint)
    )
    monkeypatch.setattr(clock, "_installed", frontend)
    address = get_open_zmq_ipc_path()
    clock.name_endpoints(0, "", "", address)
    ctx, raw = zmq_shim._Context(), zmq.Context()
    loop = CompassEventLoop()
    mgr = CoreManager.__new__(CoreManager)
    mgr.ctx, mgr.label, mgr._lb_lock, mgr._seq_load = ctx, "fe", threading.Lock(), {}
    mgr._seq_id_to_callback = {7: callback(loop)}
    mgr._flush_stream_batch_fn = api_server.flush_stream_batch
    thread = mgr._create_output_thread(
        0, make_zmq_socket(ctx, address, zmq.PULL, bind=True), "inproc://detok-stop"
    )
    thread.start()
    push = raw.socket(zmq.PUSH)
    push.connect(address)
    engine = LPRuntime(LpId("engine"), TABLE, None)
    engine.start_run()
    for n in token_counts:
        out = RequestOutput(7, list(range(n)), False)
        WrappedSocket(engine, push, OUT).send(
            pickle.dumps((EngineCoreRequestType.STREAM, [(7, out)]))
        )
    engine_conn = clock_transport.connect(LpId("engine"), endpoint)
    engine_conn.send((NER, math.inf, engine.send_log, math.inf))
    guard = threading.Timer(20, loop.call_soon_threadsafe, (loop.stop,))
    guard.start()
    frontend.start_run()
    try:
        loop.run_forever()
    finally:
        guard.cancel()
        stop = ctx.socket(zmq.PAIR)
        stop.connect("inproc://detok-stop")
        stop.send(b"")
        thread.join(5)
        loop.close()
        ctx.destroy(linger=0)
        raw.destroy(linger=0)
        server.close()
    assert frontend.now == math.inf


def test_three_messages_on_one_thread_predicted_beside_observed(monkeypatch):
    tokenizer, entry = _Tokenizer(), _entry()
    tokenizer.decode = wrap_decode(tokenizer.decode, entry)
    dispatcher = StreamBatchDispatcher(tokenizer)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)
    state, observed = dispatcher.new_state(), []

    def callback(loop):
        # Each update's in-job clock read, and the loop time it is delivered at.
        collector = SimpleNamespace(
            put_nowait=lambda c: observed.append((c["finished_at"], loop.time()))
        )
        return lambda out: api_server._send_stream_chunk_direct(
            out, "r", collector, loop, state
        )

    _run(monkeypatch, callback, [100, 50, 25])
    # Predicted by hand: one stream; each update decodes the window before it
    # and the window with it, ``fixed + tokens / (rate x derate)`` per call.
    # The three arrive together at IPC and run back to back.
    rate = entry.decode_tokens_per_s * entry.derate
    windows = [(0, 100), (100, 150), (50, 75)]
    start, predicted = IPC, []
    for prefix, full in windows:
        end = start + 2 * entry.decode_fixed_s + (prefix + full) / rate
        predicted.append((start, end))
        start = end
    # (job start, completion): (0.001, 0.203), (0.203, 0.705), (0.705, 0.957)
    assert observed == [pytest.approx(row) for row in predicted]


def test_callbacks_of_jobs_completing_together_run_in_job_order(monkeypatch):
    def callback(loop):
        return lambda out: loop.call_soon_threadsafe(
            seen.append, len(out.output_tokens)
        )

    seen = []
    _run(monkeypatch, callback, [1, 2, 3, 4, 5])
    assert seen == [1, 2, 3, 4, 5]


def test_a_non_streaming_completion_is_delivered_after_its_loop_thread_decode(
    monkeypatch,
):
    # ATOM's `generate_async` on the frontend loop: preprocess on the executor,
    # 100 tokens in one finished output, then its final decode on the loop thread.
    entry = _entry()
    tokenizer = _Tokenizer()
    tokenizer.decode = wrap_decode(tokenizer.decode, entry)
    endpoint = f"inproc:test-loop-decode-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    for lp in ("traffic", "engine"):
        clock_transport.connect(LpId(lp), endpoint).send((NER, math.inf, [], math.inf))
    frontend = LPRuntime(
        LpId("frontend"), TABLE, clock_transport.connect(LpId("frontend"), endpoint)
    )
    monkeypatch.setattr(clock, "_installed", frontend)
    loop = CompassEventLoop()
    seq = SimpleNamespace(id=7, num_prompt_tokens=1, max_tokens=100)

    def preprocess(prompt, params, stream_callback, **kw):
        seq.callback = stream_callback
        return seq

    def add_request(seqs):
        seq.callback(RequestOutput(7, list(range(100)), True, "length"))

    monkeypatch.setattr(api_server, "tokenizer", tokenizer)
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
            core_mgr=SimpleNamespace(add_request=add_request),
        ),
    )
    delivered = []

    async def request():
        async for response in api_server.generate_async("p", SamplingParams(), "r"):
            delivered.append((response["latency"], loop.time()))

    frontend.start_run()
    task = loop.create_task(request())
    guard = threading.Timer(20, loop.call_soon_threadsafe, (loop.stop,))
    guard.start()
    try:
        loop.run_forever()
    finally:
        guard.cancel()
        loop.close()
        server.close()
    task.result()
    # Predicted by hand: everything before the decode takes no simulated time,
    # and the decode of 100 tokens costs ``fixed + 100 / (rate x derate)``.
    predicted = entry.decode_fixed_s + 100 / (entry.decode_tokens_per_s * entry.derate)
    assert predicted == pytest.approx(0.201)
    assert delivered == [pytest.approx((predicted, predicted))]


def test_a_loop_thread_decode_outside_the_run_asks_for_no_time(monkeypatch):
    frontend = LPRuntime(LpId("frontend"), TABLE, None)  # no connection to ask on
    monkeypatch.setattr(clock, "_installed", frontend)
    decode = wrap_decode(_Tokenizer().decode, _entry())
    assert decode([1, 2]) == "xx" and frontend.now == 0.0


def test_a_decode_on_another_thread_with_no_job_open_is_refused(monkeypatch):
    # In the run, a thread other than the clock owner with no job open asks the
    # authority for nothing: the charge goes through `advance_to`, which refuses it.
    frontend = LPRuntime(LpId("frontend"), TABLE, None, owner=threading.Thread())
    monkeypatch.setattr(clock, "_installed", frontend)
    frontend.start_run()
    decode = wrap_decode(_Tokenizer().decode, _entry())
    with pytest.raises(RuntimeError, match="only the clock owner"):
        decode([1, 2])
    assert frontend.now == 0.0


def test_a_frame_from_before_the_run_decoded_on_an_output_thread_is_a_summary_refusal(
    monkeypatch, tmp_path
):
    # An engine frame sent outside the run reaches ATOM's output thread with no
    # job open after the frontend's run has started; its decode is refused, and
    # ATOM logs the refusal as `flush_stream_batch failed` and drops the update.
    # A stamped frame after it keeps the run open until the first is handled.
    run = {
        "admission_path": "serving",
        "ipc_s": IPC,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": f"inproc:test-detok-refusal-{uuid.uuid4().hex}",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    (tmp_path / compass_run.COMMANDS_FILE.format("engine")).write_text("[]")
    monkeypatch.setattr(compass_run, "_authority", authority)
    server = clock_transport.serve(authority, run["clock_endpoint"])
    clock_transport.connect(LpId("traffic"), run["clock_endpoint"]).send(
        (NER, math.inf, [], math.inf)
    )
    frontend = LPRuntime(
        LpId("frontend"),
        TABLE,
        clock_transport.connect(LpId("frontend"), run["clock_endpoint"]),
    )
    monkeypatch.setattr(clock, "_installed", frontend)
    tokenizer = _Tokenizer()
    tokenizer.decode = wrap_decode(tokenizer.decode, _entry())
    dispatcher = StreamBatchDispatcher(tokenizer)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)
    state = dispatcher.new_state()
    address = get_open_zmq_ipc_path()
    clock.name_endpoints(0, "", "", address)
    ctx, raw = zmq_shim._Context(), zmq.Context()
    loop = CompassEventLoop()
    collector = SimpleNamespace(put_nowait=lambda c: None)
    mgr = CoreManager.__new__(CoreManager)
    mgr.ctx, mgr.label, mgr._lb_lock, mgr._seq_load = ctx, "fe", threading.Lock(), {}
    mgr._seq_id_to_callback = {
        7: lambda out: api_server._send_stream_chunk_direct(
            out, "r", collector, loop, state
        )
    }
    mgr._flush_stream_batch_fn = api_server.flush_stream_batch
    thread = mgr._create_output_thread(
        0, make_zmq_socket(ctx, address, zmq.PULL, bind=True), "inproc://detok-ref"
    )
    thread.start()
    push = raw.socket(zmq.PUSH)
    push.connect(address)
    engine = LPRuntime(LpId("engine"), TABLE, None)
    frontend.start_run()
    authority.started = time.monotonic()
    for n in (2, 3):
        out = RequestOutput(7, list(range(n)), False)
        frame = pickle.dumps((EngineCoreRequestType.STREAM, [(7, out)]))
        WrappedSocket(engine, push, OUT).send(frame)
        engine.start_run()  # so only the first frame is unstamped
    clock_transport.connect(LpId("engine"), run["clock_endpoint"]).send(
        (NER, math.inf, engine.send_log, math.inf)
    )
    guard = threading.Timer(20, loop.call_soon_threadsafe, (loop.stop,))
    guard.start()
    try:
        loop.run_forever()
    finally:
        guard.cancel()
        stop = ctx.socket(zmq.PAIR)
        stop.connect("inproc://detok-ref")
        stop.send(b"")
        thread.join(5)
        loop.close()
        ctx.destroy(linger=0)
        raw.destroy(linger=0)
        server.close()
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    refusals = summary["schedule"]["refusals"]
    print("refusals:", json.dumps(refusals))
    assert refusals["reasons"] == [
        ["clock:advance_to from EngineCoreOutputThread-DP0", 1]
    ]
