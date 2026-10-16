# SPDX-License-Identifier: MIT
"""Each request's response id, from the API server to the step record and its split.

Requests go through ATOM's API server handlers over a planted engine: its
`preprocess` builds the sequence and answers it at once, and its core manager
is ATOM's own, whose `add_request` sends each sequence the way it reaches the
engine core, pickled. The record is then written by ATOM's scheduler and the
simulated runner from the sequences the engine core received.
"""

import asyncio
import inspect
import json
import pickle
import queue
from types import SimpleNamespace

import pytest
from test_parity_record import BLOCK, _config, _key, _prompt, _Runner

from atom.compass import run
from atom.compass.parity import ENV, compare, read
from atom.compass.parity.served import served
from atom.compass.validate.runs import _requests
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import CompletionRequest
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence

SAMPLING = {"temperature": 0.0, "top_p": 1.0}


@pytest.fixture
def engine(monkeypatch):
    """The planted engine; `sent` holds what each `add_request` sent, unpickled."""
    # ATOM's own `add_request`, whatever an earlier test's frontend wrapped,
    # and back as it was after the test.
    for cls in (CoreManager, DisaggCoreManager):
        monkeypatch.setattr(cls, "add_request", inspect.unwrap(cls.add_request))

    def preprocess(tokens, sampling_params, stream_callback=None, **_kwargs):
        seq = Sequence(list(tokens), BLOCK, sampling_params)
        stream_callback(
            SimpleNamespace(output_tokens=[5], finished=True, finish_reason="length")
        )
        return seq

    sent = []
    planted = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=None,
        sent=sent,
    )

    def core(cls=CoreManager):
        manager = cls.__new__(cls)
        manager.label, manager.pp_size, manager.local_engine_count = "t", 1, 1
        manager._seq_id_to_callback = {}
        manager._send_request = lambda _rank, payload: sent.append(
            pickle.loads(payload)[1]
        )
        planted.core_mgr = manager

    planted.core = core
    core()
    monkeypatch.setattr(api_server, "engine", planted)
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(
        api_server, "tokenizer", SimpleNamespace(decode=lambda *_a, **_k: "x")
    )
    return planted


def _frontend(monkeypatch, record=None):
    """`run.frontend` as `LLMEngine` enters it on a real run."""
    if record is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, str(record))
    monkeypatch.delenv(run.ENV, raising=False)
    config = SimpleNamespace(runner_qualname=run.ATOM_RUNNER)
    with run.frontend(config):
        pass


def _complete(tokens) -> str:
    """Send one completion request; its response `id`."""
    request = CompletionRequest(
        model="m", prompt_token_ids=tokens, max_tokens=3, ignore_eos=True, **SAMPLING
    )
    response = asyncio.run(api_server.completions(request, None))
    return json.loads(response.body)["id"]


def test_a_server_not_recording_sends_its_sequences_unchanged(engine, monkeypatch):
    add = CoreManager.add_request
    _frontend(monkeypatch)
    _complete(_prompt(0))

    assert CoreManager.add_request is add
    assert [s.external_request_id for s in engine.sent[0]] == [None]


@pytest.mark.parametrize("cls", [CoreManager, DisaggCoreManager])
def test_a_recording_server_s_sequences_carry_their_response_ids(
    engine, monkeypatch, tmp_path, cls
):
    _frontend(monkeypatch, tmp_path)
    engine.core(cls)
    ids = [_complete(_prompt(0))]
    # The streaming and multimodal paths hand their sequences over in the same way.
    dispatcher = SimpleNamespace(new_state=lambda: None, enqueue=lambda **_: None)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)
    for name in ("_stream_loops", "_request_start_times", "_seq_id_to_request_id"):
        monkeypatch.setattr(api_server, name, {})
    ids.append("cmpl-stream")
    asyncio.run(api_server.setup_streaming_request(_prompt(1), None, ids[-1]))

    async def multimodal():
        async for _ in api_server.generate_async_multimodal(
            _prompt(2), {}, None, ids[-1]
        ):
            pass

    ids.append("chatcmpl-multimodal")
    asyncio.run(multimodal())

    assert [s.external_request_id for seqs in engine.sent for s in seqs] == [
        i for i in ids for _ in range(1 if cls is CoreManager else 2)
    ]


def test_a_record_splits_per_client_run_by_the_response_ids_its_clients_saw(
    engine, monkeypatch, tmp_path
):
    _frontend(monkeypatch, tmp_path)
    groups = [[0, 1], [2, 3]]
    clients = [{_complete(_prompt(i)): i for i in group} for group in groups]
    received = iter(seq for seqs in engine.sent for seq in seqs)
    config = _config()
    scheduler, runner = Scheduler(config), _Runner(config)
    for group in groups:  # each client run served to its end before the next
        for _ in group:
            scheduler.add(next(received))
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
    exports = [
        {rid: {"trace": f"t{i}", "sampling": SAMPLING} for rid, i in c.items()}
        for c in clients
    ]
    split = served(tmp_path, exports)[0]

    keys = {_key(_prompt(i)): k for k, group in enumerate(groups) for i in group}
    assert {s["client_run"] for s in split} == {0, 1}
    for step in split:
        assert {keys[row[0]] for row in step["batch"]} == {step["client_run"]}
    names = {k: v for s in split for k, v in s.get("response_ids", {}).items()}
    assert names == {_key(_prompt(i)): rid for c in clients for rid, i in c.items()}


def test_a_record_written_before_response_ids_reads_as_it_did(tmp_path):
    for side, named in (("old", [[1, "k1"]]), ("new", [[1, "k1", "cmpl-a"]])):
        (tmp_path / side).mkdir()
        lines = [
            {"step": 0, "batch": [[1, 8, 8]], "rows": [[8, 0, False]], "named": named},
            {"step": 1, "batch": [[1, 1, 9]], "rows": [[1, 9, True]], "named": []},
        ]
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (tmp_path / side / "dp0.jsonl").write_text(text)
    old, new = read(tmp_path / "old")[0], read(tmp_path / "new")[0]

    assert [s["batch"] for s in old] == [[["k1", 8, 8]], [["k1", 1, 9]]]
    assert "response_ids" not in old[0] and "response_ids" not in old[1]
    assert new[0].pop("response_ids") == {"k1": "cmpl-a"}
    assert new == old
    assert compare(tmp_path / "old", tmp_path / "new")["first_divergence"] == {0: None}
    steps = [json.loads(x) for x in (tmp_path / "new" / "dp0.jsonl").open()]
    assert _requests(steps) == {(8, 0): 8}
