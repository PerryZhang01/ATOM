# SPDX-License-Identifier: MIT
"""The tracestate and SSE comment carriers, and the stream writer in `_client_stream`.

The writer is driven through the shipped wrapper every streaming endpoint
returns, with a real `LPRuntime` installed as the frontend's clock, and the
chat response is ATOM's own `stream_chat_response`.
"""

import asyncio
import re

import pytest

from atom.compass.carriers import sse_stamp, tracestate_stamp, tracestate_with
from atom.compass.clock import LpId, prefill_decode_table, single_engine_table
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.serving_chat import stream_chat_response
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.utils import clock
from atom.utils.clock import LPRuntime

STREAM = "frontend->traffic:stream"
STREAM_S = 0.002
VENDORS = "rojo=00f067aa0ba902b7, congo=t61rcWkgMzE"
FRAMES = ["data: a\n\n", "event: x\ndata: b\n\n", "data: c\n\ndata: [DONE]\n\n"]


def _runtime(now, *, in_run=True):
    table = single_engine_table(
        admission_path="serving", ipc_s=0.001, stream_s=STREAM_S
    )
    rt = LPRuntime(LpId("frontend"), table, conn=None)
    rt.now = now
    if in_run:
        rt.start_run()
    return rt


@pytest.fixture
def installed():
    def install(rt):
        clock.install(rt)
        return rt

    yield install
    clock.install(None)


def _client_text(source):
    async def run():
        return [c async for c in api_server._client_stream(source, "req-1")]

    return asyncio.run(run())


async def _replay(frames):
    for f in frames:
        yield f


class TestTracestate:
    @pytest.mark.parametrize("arrival", [0.0, 12.345, 1e-05, 0.1 + 0.2, float("inf")])
    def test_round_trip(self, arrival):
        assert tracestate_stamp(tracestate_with(None, arrival, 17)) == (arrival, 17)

    def test_other_vendors_keep_their_order(self):
        header = tracestate_with(VENDORS, 1.5, 3)
        assert header == VENDORS + ",compass=a:1.5;s:3"
        assert tracestate_stamp(header) == (1.5, 3)

    def test_a_space_after_the_comma_is_allowed(self):
        assert tracestate_stamp("rojo=1, compass=a:1.5;s:3") == (1.5, 3)

    @pytest.mark.parametrize("header", [None, "", VENDORS, "compassion=a:1;s:2"])
    def test_no_compass_entry_reads_as_none(self, header):
        assert tracestate_stamp(header) is None

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "a:x;s:1",
            "a:1.0",
            "a:1.0;s:-1",
            "s:1;a:1.0",
            "a:nan;s:1",
            "a:-1.0;s:1",
            "a:1.0;s:1x",
            "a:1.2.3;s:1",
        ],
    )
    def test_a_malformed_entry_is_refused_by_name(self, value):
        with pytest.raises(ValueError, match="^malformed compass tracestate entry"):
            tracestate_stamp(f"{VENDORS},compass={value}")

    def test_two_entries_are_refused(self):
        with pytest.raises(ValueError, match="more than one compass entry"):
            tracestate_stamp("compass=a:1.0;s:1,rojo=1,compass=a:2.0;s:2")

    def test_writing_a_second_entry_is_refused(self):
        with pytest.raises(ValueError, match="already has a compass entry"):
            tracestate_with("rojo=1,compass=a:1.0;s:1", 2.0, 2)


class TestSseComment:
    @pytest.mark.parametrize(
        "line", ["", "data: x", ": ping", ": compassion", "event: compass"]
    )
    def test_any_other_line_reads_as_none(self, line):
        assert sse_stamp(line) is None

    @pytest.mark.parametrize(
        "line",
        [
            ": compass a=x s=1",
            ": compass a=1.0",
            ": compass a=1.0 s=1 more",
            ": compass s=1 a=1.0",
        ],
    )
    def test_a_malformed_comment_is_refused_by_name(self, line):
        with pytest.raises(ValueError, match="^malformed compass SSE comment"):
            sse_stamp(line)


class TestTheStreamWriter:
    def test_compass_off_the_stream_is_unchanged(self):
        assert _client_text(_replay(FRAMES)) == FRAMES

    def test_outside_the_run_the_stream_is_unchanged(self, installed):
        rt = installed(_runtime(1.0, in_run=False))
        assert _client_text(_replay(FRAMES)) == FRAMES
        assert rt.send_log == []

    def test_an_unterminated_frame_is_refused_by_name(self, installed):
        rt = installed(_runtime(1.0))
        with pytest.raises(
            ValueError, match=r"^unterminated SSE frame 'data: \{\"x\":'"
        ):
            _client_text(_replay(['data: a\n\ndata: {"x":', "1}\n\n"]))
        assert rt.send_log == []

    def test_a_chat_stream_carries_one_stamp_per_event(self, installed):
        rt = installed(_runtime(2.0))
        collector = StreamOutputCollector("req-1")
        collector.put_nowait({"text": "Hi", "token_ids": [1], "finished": True})
        sent = []

        async def chat():
            async for c in stream_chat_response(
                request_id="req-1",
                model="model",
                stream_collector=collector,
                seq_id=0,
                num_prompt_tokens=1,
                cleanup_stream=lambda *a, **k: None,
                cleanup_request=lambda *a, **k: None,
            ):
                sent.append(c)
                yield c

        text = "".join(_client_text(chat()))
        events = text.split("\n\n")[:-1]
        stamps = [sse_stamp(e.split("\n", 1)[0]) for e in events]
        assert stamps == [(a, s) for _, s, a in rt.send_log]
        assert rt.send_log == [(STREAM, s, 2.0 + STREAM_S) for s in range(len(events))]
        assert len(events) > len(sent) and events[-1].endswith("data: [DONE]")
        assert re.sub(r"(?m)^: compass .*\n", "", text) == "".join(sent)

    def test_behind_the_router_lp_frontend_d_stamps_its_stream_to_the_router(
        self, installed
    ):
        table = prefill_decode_table(
            admission_path="serving",
            ipc_s=0.001,
            stream_s=STREAM_S,
            router_s=0.004,
            kv_write_req_s=0.001,
            router_in_s=0.001,
        )
        rt = LPRuntime(LpId("frontend-D"), table, conn=None)
        rt.now = 2.0
        rt.start_run()
        installed(rt)
        text = "".join(_client_text(_replay(FRAMES)))
        stamps = [sse_stamp(e.split("\n", 1)[0]) for e in text.split("\n\n")[:-1]]
        assert rt.send_log == [
            ("frontend-D->router:stream", s, 2.001) for s in range(len(stamps))
        ]
        assert stamps == [(a, s) for _, s, a in rt.send_log]
