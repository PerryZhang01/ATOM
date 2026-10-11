# SPDX-License-Identifier: MIT
"""compass_harness's transport times aiperf's records on the stamps ATOM writes.

A stub server streams SSE events stamped by ``stamp_events`` with known
arrivals; a stand-in traffic LP publishes each request's send stamp and collects
the stream reports on the adapter sockets. Requests go through aiperf's own
``InferenceClient``, which looks the ``http`` transport up in aiperf's plugin
registry, and aiperf's TTFT and request-latency metrics read the records. Skips
by name as ``test_harness_pacing.py`` does.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from aiperf.plugin import plugins

# isort: split
import aiperf
import zmq
import zmq.asyncio
from aiohttp import web
from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.messages import InferenceResultsMessage
from aiperf.common.models import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
    ParsedResponseRecord,
    RequestInfo,
)
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.workers.inference_client import InferenceClient
from compass_harness.timesource import EPOCH_NS
from compass_harness.transport import (
    ADDRESS_ENV,
    CompassTransport,
    addresses,
    credit_key,
)

from atom.compass.carriers import stamp_events, tracestate_stamp
from compass_harness import fingerprint

# Per request: its send stamp, then the arrival of each token event; [DONE]
# leaves with the last token, as ATOM coalesces them.
SENDS = {0: (2.0, 0), 1: (2.5, 1), 2: (7.25, 2)}
TOKENS = {0: [2.75, 3.0, 4.5], 1: [3.5, 3.625], 2: [9.0, 9.5, 9.75, 10.0]}


def _chunk(text: str) -> str:
    delta = {"choices": [{"index": 0, "delta": {"content": text}}]}
    return f"data: {json.dumps({'object': 'chat.completion.chunk', **delta})}\n\n"


class _Frontend:
    """Stand-in for the frontend LP's runtime: hands out the scripted stamps."""

    me = "frontend"

    def __init__(self) -> None:
        self.arrival, self.seq = 0.0, 0

    def stamp_send(self, ch: str) -> tuple[float, int]:
        assert ch == "frontend->traffic:stream"
        self.seq += 1
        return self.arrival, self.seq - 1


class Harness:
    """A stub ATOM server, a stand-in traffic LP and aiperf's inference client."""

    def __init__(self, tmp_path, monkeypatch, timeout: float = 30.0) -> None:
        self.prefix = str(tmp_path / "lp")
        monkeypatch.setenv(ADDRESS_ENV, self.prefix)
        self.frontend = _Frontend()
        self.timeout = timeout
        self.tracestate: dict[int, str] = {}
        self.unstamped: set[int] = set()
        self.undone: set[int] = set()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        num = int(request.headers["X-Request-ID"])
        self.tracestate[num] = request.headers["tracestate"]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i, arrival in enumerate(TOKENS[num]):
            self.frontend.arrival = arrival
            text = _chunk(f"t{i}")
            if num in self.unstamped and i == 1:
                await resp.write(text.encode())
                continue
            if i == len(TOKENS[num]) - 1 and num not in self.undone:
                text += "data: [DONE]\n\n"
            await resp.write(stamp_events(text, self.frontend).encode())
            await asyncio.sleep(0.01)
        await resp.write_eof()
        return resp

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]

        self.ctx = zmq.asyncio.Context()
        self.ctx.setsockopt(zmq.LINGER, 0)
        stamps, reports = addresses(self.prefix)
        self.pub = self.ctx.socket(zmq.XPUB)
        self.pub.bind(stamps)
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.bind(reports)

        endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="m")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type="chat",
                base_urls=[f"http://127.0.0.1:{port}"],
                streaming=True,
                timeout=self.timeout,
            ),
        )
        self.client = InferenceClient(model_endpoint=endpoint, service_id="worker_0")
        await self.client.initialize()
        assert await self.pub.recv() == b"\x01"  # the worker has subscribed
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.stop()
        await self.runner.cleanup()
        self.ctx.destroy(linger=0)

    async def stamp(self, num: int) -> None:
        await self.pub.send_pyobj((_key(num), *SENDS[num]))

    async def send(self, num: int):
        info = RequestInfo(
            model_endpoint=self.client.model_endpoint,
            credit_num=num,
            credit_phase=CreditPhase.PROFILING,
            phase_index=0,
            conversation_id=f"c{num}",
            turn_index=0,
            x_request_id=str(num),
            x_correlation_id=f"x{num}",
            drop_perf_ns=time.perf_counter_ns(),
            credit_issued_ns=time.time_ns(),
            payload_bytes=b'{"model": "m", "messages": [], "stream": true}',
        )
        record = await self.client.send_request(info)
        # The hop to the record processor re-validates the record.
        wire = InferenceResultsMessage(service_id="worker_0", record=record)
        return InferenceResultsMessage.model_validate_json(
            wire.model_dump_json()
        ).record

    def metric(self, record, metric_cls) -> int:
        parsed = self.client.endpoint.extract_response_data(record)
        return metric_cls().parse_record(
            ParsedResponseRecord(request=record, responses=parsed), MetricRecordDict()
        )


def _key(num: int) -> tuple:
    return credit_key(CreditPhase.PROFILING, 0, num)


def _run(coro):
    """Run `coro`, failing rather than hanging when a stamp or report never comes."""
    return asyncio.run(asyncio.wait_for(coro, 30))


def _ns(seconds: float) -> int:
    return round(seconds * 1e9) + EPOCH_NS


def test_the_http_transport_is_compass_transport():
    from aiperf.plugin.enums import PluginType

    assert plugins.get_class(PluginType.TRANSPORT, "http") is CompassTransport


def test_aiperf_ttft_and_latency_are_the_stamped_differences(tmp_path, monkeypatch):
    async def run():
        async with Harness(tmp_path, monkeypatch) as h:
            for num in SENDS:
                await h.stamp(num)
            records = [await h.send(num) for num in SENDS]
            reports = [
                await h.pull.recv_pyobj()
                for _ in range(sum(map(len, TOKENS.values())) + 3)
            ]
            return h, records, reports

    h, records, reports = _run(run())
    rows = []
    for num, record in zip(SENDS, records):
        sent, seq = SENDS[num]
        first, finish = TOKENS[num][0], TOKENS[num][-1]
        assert record.error is None
        assert tracestate_stamp(h.tracestate[num]) == (sent, seq)
        ttft = h.metric(record, TTFTMetric)
        latency = h.metric(record, RequestLatencyMetric)
        rows.append((num, sent, first, finish, ttft / 1e9, latency / 1e9))
        assert ttft == _ns(first) - _ns(sent)
        assert latency == _ns(finish) - _ns(sent)
        assert record.credit_drop_latency is None
        assert record.request_info.credit_issued_ns is None
        assert record.trace_data is None  # no wall timing of the real socket
        assert record.timestamp_ns == _ns(sent)  # the run duration's anchor
    print("\ncredit  sent  first  finish  aiperf TTFT  aiperf latency")
    for row in rows:
        print("{:6d} {:5.3f} {:6.3f} {:7.3f} {:12.3f} {:15.3f}".format(*row))

    expected, seq = [], 0
    for num, tokens in TOKENS.items():
        for arrival in tokens:
            expected.append((_key(num), seq, arrival, False))
            seq += 1
        expected.append((_key(num), seq, tokens[-1], True))
        seq += 1
    assert reports == expected


def test_the_stamp_may_reach_the_worker_before_or_after_its_credit(
    tmp_path, monkeypatch
):
    async def run():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.stamp(0)
            await asyncio.sleep(0.05)
            before = await h.send(0)
            pending = asyncio.ensure_future(h.send(1))
            await asyncio.sleep(0.05)
            assert not pending.done()
            await h.stamp(1)
            return before, await pending

    for record, num in zip(_run(run()), (0, 1)):
        assert record.error is None
        assert record.start_perf_ns == _ns(SENDS[num][0])


def test_an_unstamped_event_fails_its_request_by_name(tmp_path, monkeypatch):
    async def run():
        async with Harness(tmp_path, monkeypatch) as h:
            h.unstamped.add(2)
            await h.stamp(2)
            return await h.send(2)

    record = _run(run())
    assert (
        "request 2 (credit ('profiling', 0, 2)): SSE event 1 carries no compass stamp"
        in (record.error.message)
    )
    # Only stamped times remain: the send and the one stamped event.
    assert record.start_perf_ns == _ns(SENDS[2][0])
    assert [r.perf_ns for r in record.responses] == [_ns(TOKENS[2][0])]
    assert record.end_perf_ns == _ns(TOKENS[2][0])
    assert record.recv_start_perf_ns is None


def test_a_stream_without_a_stamped_done_fails_its_request_by_name(
    tmp_path, monkeypatch
):
    async def run():
        async with Harness(tmp_path, monkeypatch) as h:
            h.undone.add(1)
            await h.stamp(1)
            return await h.send(1)

    record = _run(run())
    assert record.error.message == (
        "request 1 (credit ('profiling', 0, 1)): the stream ended without a "
        "stamped [DONE] event"
    )
    assert [r.perf_ns for r in record.responses] == [_ns(t) for t in TOKENS[1]]


def test_a_stamp_after_its_request_timed_out_leaves_later_requests_served(
    tmp_path, monkeypatch
):
    async def run():
        async with Harness(tmp_path, monkeypatch, timeout=0.2) as h:
            late = await h.send(0)
            await h.stamp(0)
            await h.stamp(1)
            return late, await h.send(1)

    late, served = _run(run())
    assert "credit ('profiling', 0, 0): no send stamp from the traffic LP" in (
        late.error.message
    )
    assert served.error is None
    assert served.start_perf_ns == _ns(SENDS[1][0])


def test_a_changed_pinned_function_refuses_the_run_by_name(tmp_path):
    shutil.copytree(Path(aiperf.__file__).parent, tmp_path / "aiperf")
    fingerprint.check(tmp_path)
    path = tmp_path / "aiperf" / "transports" / "aiohttp_client.py"
    text = path.read_text()
    path.write_text(
        text.replace("first_token_acquired = False", "first_token_acquired = True", 1)
    )
    # A timing_manager's imports, against the changed copy.
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aiperf.plugin import plugins; import aiperf.timing.phase.runner",
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{tmp_path}:{os.environ.get('PYTHONPATH', '')}",
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert run.returncode != 0
    assert run.stderr.rstrip().endswith(
        "RuntimeError: compass_harness was built against other aiperf source; these "
        "functions changed: aiperf.transports.aiohttp_client:AioHttpClient._request"
    )
