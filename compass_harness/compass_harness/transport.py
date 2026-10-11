# SPDX-License-Identifier: MIT
"""aiperf's ``http`` transport, timed on the Compass clock instead of the wall clock.

Runs in each aiperf worker. The traffic LP in ``timing_manager`` stamps every
request it sends with an ``(arrival, seq)`` on the HTTP channel and publishes it
on the adapter's stamp socket, keyed by credit; the stamp and the credit reach
the worker in either order. The request carries its stamp in ``tracestate``.
Each SSE event ATOM streams back carries its own stamp in a comment line; the
transport reports each one to the traffic LP as it arrives, ``[DONE]`` marked
final, and rewrites the record's times from the stamps, so aiperf's TTFT, ITL
and latency read simulated time measured from the send.

The two sockets live under the path prefix in ``COMPASS_HARNESS_IPC``.
"""

import asyncio
import os

import zmq
import zmq.asyncio
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import ErrorDetails
from aiperf.transports.aiohttp_transport import AioHttpTransport

from atom.compass.carriers import sse_stamp, tracestate_with
from compass_harness import ADDRESS_ENV
from compass_harness.timesource import sim_ns as _ns


def addresses(prefix: str) -> tuple[str, str]:
    """The stamp (traffic LP to workers) and report (workers to traffic LP) addresses."""
    return f"ipc://{prefix}.stamps", f"ipc://{prefix}.reports"


def credit_key(phase, phase_index, num) -> tuple:
    """A credit's key: credit numbers restart in each phase."""
    return str(phase), phase_index, num


class CompassTransport(AioHttpTransport):
    """``AioHttpTransport`` whose record times are the stamps ATOM's Compass clock wrote."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stamps: dict[tuple, asyncio.Future] = {}
        self._sending: dict[tuple, tuple[float, int]] = {}

    @on_init
    async def _connect_traffic_lp(self) -> None:
        if not self.model_endpoint.endpoint.streaming:
            raise ValueError(
                "Compass times a request by its stream's stamps: pass --streaming."
            )
        prefix = os.environ.get(ADDRESS_ENV)
        if not prefix:
            raise RuntimeError(
                f"{ADDRESS_ENV} is not set: the worker cannot reach the traffic LP."
            )
        stamps, reports = addresses(prefix)
        ctx = zmq.asyncio.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(stamps)
        self._sub.subscribe(b"")
        self._push = ctx.socket(zmq.PUSH)
        self._push.connect(reports)
        self._reader = asyncio.create_task(self._read_stamps())

    @on_stop
    async def _close_traffic_lp(self) -> None:
        self._reader.cancel()
        self._sub.close(linger=0)
        self._push.close(linger=0)

    async def _read_stamps(self) -> None:
        # ponytail: every worker keeps every stamp, its own and the others'; drop
        # a phase's stamps at its end if memory matters.
        while True:
            key, arrival, seq = await self._sub.recv_pyobj()
            # A request that timed out waiting cancelled its future.
            if not (fut := self._stamp(key)).done():
                fut.set_result((arrival, seq))

    def _stamp(self, key: tuple) -> asyncio.Future:
        if key not in self._stamps:
            self._stamps[key] = asyncio.get_running_loop().create_future()
        return self._stamps[key]

    def build_headers(self, request_info) -> dict[str, str]:
        headers = super().build_headers(request_info)
        arrival, seq = self._sending[self._key(request_info)]
        old = [k for k in headers if k.lower() == "tracestate"]
        headers["tracestate"] = tracestate_with(
            ",".join(headers.pop(k) for k in old) or None, arrival, seq
        )
        return headers

    @staticmethod
    def _key(request_info) -> tuple:
        return credit_key(
            request_info.credit_phase, request_info.phase_index, request_info.credit_num
        )

    async def send_request(self, request_info, payload, *, first_token_callback=None):
        key = self._key(request_info)
        try:
            sent, _ = self._sending[key] = await asyncio.wait_for(
                self._stamp(key), self.model_endpoint.endpoint.timeout
            )
        except TimeoutError:
            raise RuntimeError(
                f"credit {key}: no send stamp from the traffic LP"
            ) from None
        stamped, done = [], False
        first_token = first_token_callback

        async def on_event(_wall_ttft_ns, message) -> bool:
            # aiperf calls this for each SSE event until it returns True.
            nonlocal done, first_token
            stamp = next(
                (
                    sse_stamp(f": {p.value}")
                    for p in message.packets
                    if p.name == "comment" and p.value.startswith("compass ")
                ),
                None,
            )
            if stamp is None:
                raise ValueError(
                    f"request {request_info.x_request_id} (credit {key}): SSE event "
                    f"{len(stamped)} carries no compass stamp"
                )
            arrival, seq = stamp
            done = any(
                p.name == "data" and p.value == "[DONE]" for p in message.packets
            )
            await self._push.send_pyobj((key, seq, arrival, done))
            message.perf_ns = _ns(arrival)
            stamped.append(message)
            if first_token is not None and await first_token(
                message.perf_ns - _ns(sent), message
            ):
                first_token = None
            return False

        try:
            record = await super().send_request(
                request_info, payload, first_token_callback=on_event
            )
        finally:
            del self._sending[key], self._stamps[key]
        if record.error is None and not done:
            record.error = ErrorDetails(
                type="CompassStampError",
                message=f"request {request_info.x_request_id} (credit {key}): the "
                "stream ended without a stamped [DONE] event",
            )
        # Every time aiperf derives a latency or the run's duration from is a stamp;
        # the readings with none (response headers, credit issue and drop) are dropped.
        record.responses = stamped
        record.timestamp_ns = record.start_perf_ns = _ns(sent)
        record.end_perf_ns = stamped[-1].perf_ns if stamped else record.start_perf_ns
        record.recv_start_perf_ns = request_info.drop_perf_ns = None
        request_info.credit_issued_ns = None
        # aiperf's http_req_* metrics would time the real localhost socket.
        record.trace_data = None
        return record
