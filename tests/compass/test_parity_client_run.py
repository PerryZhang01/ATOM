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

from atom.compass.parity.client_run import AIPERF, NAME, RAW, response_id, write
from atom.compass.parity.served import served
from atom.compass.validate.runs import CLIENT_RUN
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import CompletionRequest
from atom.model_engine.scheduler import Scheduler

#: The payload's request parameters, beside what the export leaves out.
PARAMETERS = {"ignore_eos": True, **SAMPLING}
#: aiperf's input config beside its dataset, as a real cell's client wrote it.
CONFIG = {
    "models": {"items": [{"name": "/models/m"}]},
    "endpoint": {
        "urls": ["http://127.0.0.1:40165"],
        "type": "chat",
        "streaming": True,
        "use_server_token_count": True,
        "extra": {"ignore_eos": True},
    },
    "phases": [
        {
            "name": "profiling",
            "kind": "profiling",
            "type": "concurrency",
            "duration": 1800.0,
            "concurrency": 64,
            "grace_period": 30.0,
            "timing_mode": "agentic_replay",
            "trajectory_start_min_ratio": 0.0,
            "trajectory_start_max_ratio": 0.0,
            "system_idle_gap_cap_seconds": 10.0,
            "warmup_requests_per_lane": 1,
        }
    ],
    "artifacts": {"dir": "/cell/real1/artifacts", "raw": True},
    "tokenizer": {"name": "/models/m"},
    "scenario": "inferencex-agentx-mvp",
    "unsafe_override": True,
}
#: The export's `arguments` of `CONFIG` and a `weka` dataset.
ARGUMENTS = {
    "phases.profiling.kind": "profiling",
    "phases.profiling.type": "concurrency",
    "phases.profiling.duration": 1800.0,
    "phases.profiling.concurrency": 64,
    "phases.profiling.grace_period": 30.0,
    "phases.profiling.timing_mode": "agentic_replay",
    "phases.profiling.trajectory_start_min_ratio": 0.0,
    "phases.profiling.trajectory_start_max_ratio": 0.0,
    "phases.profiling.system_idle_gap_cap_seconds": 10.0,
    "phases.profiling.warmup_requests_per_lane": 1,
    "datasets.main.type": "file",
    "datasets.main.format": "weka_trace",
    "datasets.main.random_seed": 20260707,
    "endpoint.type": "chat",
    "endpoint.streaming": True,
    "endpoint.use_server_token_count": True,
    "endpoint.extra": {"ignore_eos": True},
    "endpoint.timeout": None,
    "scenario": "inferencex-agentx-mvp",
    "benchmark_id": "000000000000",
}


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


def weka(path, *ids) -> dict:
    """aiperf's input config entry for weka traces `ids` at `path`: one file
    when `path` is a `.json`, else a directory holding a file each."""
    files = [path] if path.suffix == ".json" else [path / f"{t}.json" for t in ids]
    files[0].parent.mkdir(parents=True, exist_ok=True)
    for f, t in zip(files, ids):
        f.write_text(json.dumps({"id": t, "block_size": 64, "requests": []}))
    return {
        "name": "main",
        "type": "file",
        "path": str(path),
        "format": "weka_trace",
        "random_seed": 20260707,
    }


def _write_raw(
    artifacts, records, dataset=None, config=CONFIG, benchmark_id="000000000000"
) -> None:
    """`records` as aiperf's raw records, beside its export, whose input
    config is `config` naming `dataset`, by default a directory of trace
    `trace-0`, and whose benchmark id is `benchmark_id`."""
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / RAW).write_text("".join(json.dumps(r) + "\n" for r in records))
    dataset = dataset or weka(artifacts / "traces", "trace-0")
    config = {"input_config": config | {"datasets": [dataset]}}
    config["benchmark_id"] = benchmark_id
    (artifacts / AIPERF).write_text(json.dumps(config))


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
        "traces": ["trace-0"],
        "arguments": ARGUMENTS,
    }


@pytest.mark.parametrize(
    "path, traces", [("traces", ["t0", "t1", "t2"]), ("t1.json", ["t1"])]
)
def test_the_export_names_every_trace_aiperf_was_given_not_only_those_finished(
    tmp_path, path, traces
):
    dataset = weka(tmp_path / path, "t1", "t2", "t0")
    if path == "traces":  # aiperf reads only a directory's `.json` files
        (tmp_path / path / "notes.txt").write_text("{}")
    _write_raw(tmp_path, [raw([sse(("data", '{"id": "a"}'))], "t1")], dataset)

    export = json.loads(write(tmp_path)[0].read_text())

    assert export["traces"] == traces
    assert {r["trace"] for r in export["requests"].values()} == {"t1"}


def test_the_export_names_what_decides_the_requests_aiperf_sends_and_not_where(
    tmp_path,
):
    # Another run's URL, artifact directory and model path, and no override.
    elsewhere = CONFIG | {
        "endpoint": CONFIG["endpoint"] | {"urls": ["http://10.0.0.2:8000"]},
        "artifacts": {"dir": "/cell/real2/artifacts", "raw": True},
        "models": {"items": [{"name": "/mnt/m"}]},
        "tokenizer": {"name": "/mnt/m"},
        "unsafe_override": False,
    }
    # The same traces at another concurrency, with a warmup phase before it,
    # a request timeout and another benchmark id.
    (profiling,) = CONFIG["phases"]
    warmup = {"name": "warmup", "kind": "warmup", "type": "concurrency"}
    other = CONFIG | {
        "phases": [warmup, profiling | {"concurrency": 16}],
        "endpoint": CONFIG["endpoint"] | {"timeout": 60.0},
    }
    records = [raw([sse(("data", '{"id": "a"}'))], "t0")]
    got = {}
    for name, config, benchmark_id in (
        ("here", CONFIG, "000000000000"),
        ("elsewhere", elsewhere, "000000000000"),
        ("other", other, "0f8c2b6a"),
    ):
        dataset = weka(tmp_path / name / "traces", "t0")
        _write_raw(tmp_path / name, records, dataset, config, benchmark_id)
        got[name] = write(tmp_path / name)[1]["arguments"]

    assert got["here"] == got["elsewhere"] == ARGUMENTS
    assert got["other"] == ARGUMENTS | {
        "phases.profiling.concurrency": 16,
        "phases.warmup.kind": "warmup",
        "phases.warmup.type": "concurrency",
        "endpoint.timeout": 60.0,
        "benchmark_id": "0f8c2b6a",
    }


@pytest.mark.parametrize(
    "entry", [{"format": "mooncake_trace"}, {"path": None, "records": []}]
)
def test_a_dataset_not_of_weka_traces_at_a_path_refuses(tmp_path, entry):
    dataset = weka(tmp_path / "traces", "t0") | entry
    _write_raw(tmp_path, [raw([sse(("data", '{"id": "a"}'))], "t0")], dataset)

    with pytest.raises(ValueError, match="dataset main is not weka traces at a path"):
        write(tmp_path)
    assert not (tmp_path / NAME).exists()


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


def test_the_planted_dataset_is_aiperf_s_own(tmp_path):
    config = pytest.importorskip("aiperf.config.dataset.config")
    planted = weka(tmp_path / "traces", "t0")

    dataset = config.FileDataset.model_validate(planted)
    assert (str(dataset.path), str(dataset.format)) == (planted["path"], "weka_trace")


def test_the_planted_config_is_aiperf_s_own(tmp_path):
    config = pytest.importorskip("aiperf.config.config")
    planted = CONFIG | {"datasets": [weka(tmp_path / "traces", "t0")]}

    # As aiperf's JSON exporter dumps its input config.
    dumped = config.BenchmarkConfig.model_validate(planted).model_dump(
        mode="json", exclude_unset=True, exclude_none=True
    )
    assert dumped == planted
