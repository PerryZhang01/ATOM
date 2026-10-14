# SPDX-License-Identifier: MIT
"""The tokenizer station: FIFO placement, the job registry, and in-job clock reads.

The end-to-end tests run ATOM's real `InputOutputProcessor.preprocess` through
`run_in_executor` on an event loop whose clock is an `LPRuntime`, with a clock
authority that grants every request in full. Its selector jumps the clock to the
next timer once no station job is open, which is all the loop needs to show a
result arriving at its job's completion time.
"""

import asyncio
import logging
import queue
import selectors
import threading
import time
from types import SimpleNamespace

import pytest

from atom.compass.clock import LpId, RefusalTally, single_engine_table
from atom.compass.spec.tokenizers import Backend, table
from atom.entrypoints.openai.api_server import _prepare_multimodal_inputs
from atom.model_engine import llm_engine
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import Refused, SimExecutor, Station, wrap_encode

ENCODE_FIXED_S, ENCODE_TOKENS_PER_S, DERATE = 0.001, 1000.0, 0.5


def _entry():
    raw = {
        "id": "test-bpe",
        "backend": "fast",
        "vocab_size": 1000,
        "fingerprint": "sha256:0",
        "applies_to": ["LlamaForCausalLM"],
        "encode_fixed_s": ENCODE_FIXED_S,
        "encode_tokens_per_s": ENCODE_TOKENS_PER_S,
        "decode_fixed_s": 0.001,
        "decode_tokens_per_s": 1000.0,
        "derate": DERATE,
    }
    return table([raw]).resolve("LlamaForCausalLM", Backend.FAST)


class _GrantAll:
    """A clock authority that grants every request in full and releases nothing."""

    def send(self, msg):
        self.t = msg[1]

    def recv(self):
        return self.t, {}


class _Selector(selectors.DefaultSelector):
    def select(self, timeout=None):
        if timeout and not self.executor.station.unresolved():
            self.rt.next_event(self.rt.read_clock() + timeout)
            timeout = 0
        return super().select(timeout)


class _Loop(asyncio.SelectorEventLoop):
    def __init__(self, rt):
        self.rt = rt
        super().__init__(_Selector())
        self._selector.rt = rt

    def time(self):
        return self.rt.read_clock()


@pytest.fixture
def loop():
    rt = LPRuntime(
        LpId("frontend"),
        single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002),
        _GrantAll(),
    )
    loop = _Loop(rt)
    yield loop
    loop.close()


def _executor(loop, width):
    """A `SimExecutor` frozen at `width` with one resident wait job, and its release."""
    ex = SimExecutor(loop, max_workers=width + 1)
    loop.set_default_executor(ex)
    loop._selector.executor = ex
    resident = queue.Queue()
    ex.submit(resident.get)
    ex.start_run()
    return ex, lambda: (resident.put(None), ex.shutdown(wait=True))


@pytest.mark.parametrize(
    "width, expected",
    [
        (1, [(0, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.1), (1.1, 1.5)]),
        (2, [(0, 0.5), (0, 0.2), (0.2, 0.5), (0.5, 0.6), (0.5, 0.9)]),
    ],
)
def test_fifo_start_and_completion_in_submission_order(width, expected):
    st = Station(width)
    submitted, service = [0, 0, 0.1, 0.2, 0.2], [0.5, 0.2, 0.3, 0.1, 0.4]
    for t, d in zip(submitted, service):
        st.charge(st.enqueue(t), d)
    # Real threads finish last-first; nothing is placed until job 0 is done.
    for k in (4, 3, 2, 1):
        assert st.finish(k) == range(0)
    assert st.unresolved()
    assert st.finish(0) == range(5)
    assert not st.unresolved()
    observed = [(st.start_of(k), st.completion(k)) for k in range(5)]
    assert observed == [pytest.approx(row) for row in expected]


def test_start_of_waits_for_earlier_jobs_with_one_diagnostic(caplog):
    st = Station(1, diag_s=0.05)
    first, second = st.enqueue(0.0), st.enqueue(0.0)
    st.charge(first, 0.3)
    st.finish(second)
    late = threading.Timer(0.3, st.finish, (first,))
    late.start()
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert st.start_of(second) == pytest.approx(0.3)
    late.join()
    assert len([r for r in caplog.records if "still waiting" in r.message]) == 1


def test_five_jobs_at_width_two_predicted_beside_observed(loop):
    tokens = [100, 50, 150, 25, 200]
    rate = ENCODE_TOKENS_PER_S * DERATE
    service = [ENCODE_FIXED_S + n / rate for n in tokens]
    # Predicted by hand: all five submitted at 0; each starts on the earlier free
    # of the two servers, in submission order.
    s = [0.0, 0.0, service[1], service[0], service[0] + service[3]]
    predicted = [(s[k], s[k] + service[k]) for k in range(5)]

    encode = wrap_encode(lambda text: list(range(int(text))), _entry())
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="llama"), speculative_config=None
    )
    proc = llm_engine.InputOutputProcessor(config, SimpleNamespace(encode=encode), 16)
    _, release = _executor(loop, width=2)

    async def request(n):
        def do_preprocess():
            start = loop.rt.read_clock()
            return start, proc.preprocess(str(n), SamplingParams())

        start, seq = await loop.run_in_executor(None, do_preprocess)
        return start, loop.time(), seq.arrive_time

    async def run():
        return await asyncio.gather(*map(request, tokens))

    try:
        clock.install(loop.rt)
        rows = loop.run_until_complete(run())
    finally:
        clock.install(None)
        release()
    print("\n  k  tokens  service    predicted s, c        observed s, c")
    for k, (n, d, p, o) in enumerate(zip(tokens, service, predicted, rows)):
        print(f"  {k}  {n:6d}  {d:.3f}  {p[0]:.3f}, {p[1]:.3f}  {o[0]:.3f}, {o[1]:.3f}")
    assert [row[:2] for row in rows] == [pytest.approx(p) for p in predicted]
    # The arrival stamp is read inside the job, after the encode is charged.
    assert [row[2] for row in rows] == pytest.approx([p[1] for p in predicted])


def test_jobs_at_one_instant_draw_request_ids_in_submission_order(loop):
    # Job 0's real thread is the slow one: a job run beside it would finish
    # first and draw the lower request id.
    encode = wrap_encode(lambda text: list(range(int(text))), _entry())

    def slow_first(text):
        if text == "100":
            time.sleep(0.2)
        return encode(text)

    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="llama"), speculative_config=None
    )
    proc = llm_engine.InputOutputProcessor(
        config, SimpleNamespace(encode=slow_first), 16
    )
    _, release = _executor(loop, width=2)

    async def request(n):
        def do_preprocess():
            return proc.preprocess(str(n), SamplingParams())

        seq = await loop.run_in_executor(None, do_preprocess)
        return loop.time(), seq.id

    async def run():
        return await asyncio.wait_for(asyncio.gather(*map(request, (100, 50))), 10)

    try:
        clock.install(loop.rt)
        (c0, id0), (c1, id1) = loop.run_until_complete(run())
    finally:
        clock.install(None)
        release()
    # Both start at 0 on the two servers: each completes at its own charge.
    assert [c0, c1] == pytest.approx([0.201, 0.101])
    assert id0 < id1


def test_a_wait_job_submitted_during_the_run_passes_through(loop):
    # ATOM's output handler resubmits `outputs_queue.get` after every output.
    ex, release = _executor(loop, width=1)
    outputs = queue.Queue()
    outputs.put("output")
    try:
        got = loop.run_until_complete(loop.run_in_executor(None, outputs.get))
    finally:
        release()
    assert got == "output"
    assert ex.refusals == [] and ex.station.jobs == []


def test_a_job_cancelled_before_it_starts_is_skipped_and_the_next_one_places(loop):
    encode = wrap_encode(lambda text: list(range(int(text))), _entry())
    ex, release = _executor(loop, width=1)
    go = threading.Event()

    def request(n):
        def do_preprocess():
            go.wait(10)
            return encode(str(n))

        return loop.run_in_executor(None, do_preprocess)

    async def run():
        first, cancelled, last = map(request, (100, 50, 25))
        cancelled.cancel()  # queued behind `first` on the only free thread
        await asyncio.sleep(0)
        go.set()
        await first
        await last
        return loop.time()

    # A job that is never placed leaves the station unresolved, and the clock stops.
    stop = threading.Timer(10, loop.call_soon_threadsafe, (loop.stop,))
    stop.start()
    try:
        assert loop.run_until_complete(run()) == pytest.approx(0.201 + 0.051)
    finally:
        stop.cancel()
        release()
    assert not ex.station.unresolved()


def _unregistered():
    return None


@pytest.mark.parametrize("fn", [_unregistered, _prepare_multimodal_inputs])
def test_an_unregistered_job_refuses_by_name_and_counts(loop, fn):
    ex, release = _executor(loop, width=1)
    try:
        with pytest.raises(Refused) as refused:
            loop.run_until_complete(loop.run_in_executor(None, fn))
    finally:
        release()
    assert refused.value.args[0] == ("executor", fn.__qualname__)
    record = RefusalTally.of(ex.refusals, steps=1).record()
    assert record["by_source"] == [["executor", 1]]
    assert record["reasons"] == [[f"executor:{fn.__qualname__}", 1]]
