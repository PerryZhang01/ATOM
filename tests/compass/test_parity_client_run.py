# SPDX-License-Identifier: MIT
"""A client run's export, written from aiperf's raw records, splits the record
the server wrote, its warmup apart.

Requests go through ATOM's API server handlers over
`test_parity_response_ids.py`'s planted engine, and the record is written by
ATOM's scheduler and the simulated runner. Each request's raw record carries
the response body the handler returned, as a whole body or as an SSE data
chunk, in the fields aiperf's `RawRecordInfo` writes.
"""

import asyncio
import json
import queue

import pytest
from test_parity_record import _config, _key, _prompt, _Runner
from test_parity_response_ids import SAMPLING, _frontend, engine  # noqa: F401

from atom.compass.parity.client_run import NAME, RAW, response_id, write
from atom.compass.parity.served import served
from atom.compass.validate.runs import CLIENT_RUN
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import CompletionRequest
from atom.model_engine.scheduler import Scheduler

#: The payload's request parameters, beside what the export leaves out.
PARAMETERS = {"ignore_eos": True, **SAMPLING}


def raw(responses, conversation="trace-0", payload=None, phase="profiling") -> dict:
    """One raw record as aiperf's `RawRecordInfo` dumps it, its payload spliced."""
    return {
        "metadata": {
            "session_num": 0,
            "x_request_id": "x",
            "conversation_id": conversation,
            "turn_index": 0,
            "request_start_ns": 1,
            "request_end_ns": 2,
            "worker_id": "w",
            "record_processor_id": "p",
            "benchmark_phase": phase,
        },
        "start_perf_ns": 1,
        "payload": payload
        or {"messages": [], "model": "m", "stream": True, **PARAMETERS},
        "status": 200,
        "responses": responses,
    }


def sse(*packets) -> dict:
    return {"perf_ns": 1, "packets": [{"name": n, "value": v} for n, v in packets]}


def _write_raw(artifacts, records) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / RAW).write_text("".join(json.dumps(r) + "\n" for r in records))


def test_a_record_splits_by_the_exports_aiperf_s_raw_records_give(
    engine, monkeypatch, tmp_path  # noqa: F811
):
    _frontend(monkeypatch, tmp_path / "steps")
    # Client run 0 warms up on request 4 alone, then sends 0 and 1; client run
    # 1 warms up on request 5 while it sends 2 and 3.
    served_in, runs = [[4], [0, 1], [5, 2, 3]], [(4, 0, 1), (5, 2, 3)]
    bodies = {}
    for i in range(6):
        request = CompletionRequest(
            model="m",
            prompt_token_ids=_prompt(i),
            max_tokens=3,
            ignore_eos=True,
            **SAMPLING,
        )
        bodies[i] = asyncio.run(api_server.completions(request, None)).body.decode()
    sent = [seq for seqs in engine.sent for seq in seqs]  # request i's is sent[i]
    scheduler, runner = Scheduler(_config()), _Runner(_config())
    for group in served_in:  # each group served to its end before the next
        for i in group:
            scheduler.add(sent[i])
        while scheduler.has_unfinished_requests():
            scheduled = scheduler.schedule()
            if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
                continue
            batch, seqs = scheduled
            scheduler.postprocess(
                list(seqs.values()),
                runner.forward(batch),
                stream_output_queue=queue.Queue(),
                batch=batch,
            )
    exports = []
    for k, (warm, whole, chunked) in enumerate(runs):
        cell = tmp_path / f"cell{k}"
        _write_raw(
            cell / "artifacts",
            # A warmup request; a whole body for one request, an SSE data
            # chunk after a comment line for the other; and a request that
            # failed before any chunk.
            [
                raw([sse(("data", bodies[warm]))], phase="warmup"),
                raw([{"perf_ns": 1, "text": bodies[whole]}], f"trace-{whole}"),
                raw([sse(("comment", "compass"), ("data", bodies[chunked]))]),
                raw([]),
            ],
        )
        path, export, unnamed = write(cell / "artifacts")
        assert (path, unnamed) == (cell / CLIENT_RUN, 1)
        assert json.loads(path.read_text()) == export
        ids = {i: json.loads(bodies[i])["id"] for i in (warm, whole, chunked)}
        assert export["warmup"] == {
            ids[warm]: {"trace": "trace-0", "sampling": PARAMETERS}
        }
        assert set(export["requests"]) == {ids[whole], ids[chunked]}
        exports.append(path)
    split = served(tmp_path / "steps", exports)[0]

    run_of = {_key(_prompt(i)): k for k, (_, *named) in enumerate(runs) for i in named}
    held = [{run_of.get(row[0]) for row in step["batch"]} for step in split]
    assert {None} in held and {None, 1} in held  # warmup alone, and beside run 1
    for step, runs_held in zip(split, held):
        # A step belongs to the client run of the profiling requests it holds.
        assert step["client_run"] == max(runs_held - {None}, default=None)
        assert step["sampling"] == (
            []
            if step["client_run"] is None
            else [json.dumps(PARAMETERS, sort_keys=True)]
        )
    assert {t for s in split for t in s["traces"]} == {"trace-0", "trace-2"}


@pytest.mark.parametrize(
    "responses, rid",
    [
        ([sse(("comment", "compass a=1.0 s=0"), ("data", '{"id": "c-1"}'))], "c-1"),
        ([sse(("comment", "x")), sse(("data", '{"id": "c-2"}'))], "c-2"),
        ([{"perf_ns": 1, "text": '{"id": "c-3"}', "content_type": "json"}], "c-3"),
        ([sse(("data", '{"id": "c-4"}')), sse(("data", '{"id": "c-5"}'))], "c-4"),
        ([sse(("data", "[DONE]"))], None),
        ([sse(("data", '{"error": "too long"}'))], None),
        ([sse(("data", None))], None),
        ([], None),
        (None, None),
    ],
)
def test_a_request_is_named_by_its_first_data_chunk_s_id(responses, rid):
    assert response_id(responses) == rid


def test_the_trace_is_the_conversation_and_the_parameters_are_what_shape_no_prompt(
    tmp_path,
):
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function"}],
        "model": "m",
        "max_completion_tokens": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
        **PARAMETERS,
    }
    _write_raw(
        tmp_path,
        [
            raw([sse(("data", '{"id": "a"}'))], "t1::sa:agent-2:aux:0", payload),
            raw([sse(("data", '{"id": "b"}'))], "t1", {"prompt": "x", "max_tokens": 3}),
        ],
    )
    path, _, _ = write(tmp_path)

    assert path == tmp_path / NAME
    assert json.loads(path.read_text()) == {
        "requests": {
            "a": {"trace": "t1", "sampling": PARAMETERS},
            "b": {"trace": "t1", "sampling": {}},
        },
        "warmup": {},
    }


@pytest.mark.parametrize(
    "records, match",
    [
        (
            [raw([sse(("data", '{"id": "a"}'))]), raw([sse(("data", '{"id": "a"}'))])],
            "two requests got response id a",
        ),
        (
            [
                raw([sse(("data", '{"id": "a"}'))], phase="warmup"),
                raw([sse(("data", '{"id": "a"}'))]),
            ],
            "two requests got response id a",
        ),
        ([raw([sse(("data", '{"id": "a"}'))], None)], "has no conversation_id"),
        ([raw([], phase="cooldown")], "neither warmup nor profiling"),
    ],
)
def test_an_export_that_cannot_name_its_requests_refuses(tmp_path, records, match):
    _write_raw(tmp_path, records)
    with pytest.raises(ValueError, match=match):
        write(tmp_path)
    assert not (tmp_path / NAME).exists()


def test_the_planted_records_are_aiperf_s_own():
    models = pytest.importorskip("aiperf.common.models")
    planted = raw(
        [
            sse(("comment", "compass"), ("data", '{"id": "a"}')),
            {"perf_ns": 2, "text": "{}", "content_type": "application/json"},
        ]
    )
    record = models.RawRecordInfo.model_validate(
        {k: v for k, v in planted.items() if k != "payload"}
    )

    dumped = record.model_dump(exclude_none=True, mode="json")
    assert dumped["responses"] == planted["responses"]
    metadata = dumped["metadata"]
    assert {k: metadata[k] for k in planted["metadata"]} == planted["metadata"]
    assert response_id(dumped["responses"]) == "a"
