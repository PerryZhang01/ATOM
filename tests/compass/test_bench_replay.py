# SPDX-License-Identifier: MIT
"""A batch rebuilt from a step-record line by the replay rule.

The lines come from `test_parity_record.py`'s drives of ATOM's own `Scheduler`
and are recorded again through the simulated runner's `forward`; the tokenizer
is `test_filler_tokens.py`'s byte-level BPE.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_record import (  # noqa: F401  _unpriced is an autouse fixture
    BLOCK,
    PROMPT,
    _config,
    _drive,
    _prompt,
    _Runner,
    _unpriced,
)

from atom.compass.bench import replay
from atom.compass.parity import ENV, read
from atom.compass.runner.step_output import DeferredTokenStream, filler_token_ids
from atom.model_engine.llm_engine import _load_tokenizer


@pytest.fixture
def rule(model_dir):  # noqa: F811
    """The rule for a model of `model_type`, over a pool of `blocks` blocks."""

    def build(model_type="llama", blocks=4096):
        config = _config(blocks=blocks)
        config.model, config.trust_remote_code = str(model_dir), False
        config.hf_config = SimpleNamespace(model_type=model_type)
        return replay.ReplayRule(config)

    return build


def _line(rows, produces_output=True):
    return {"rows": rows, "produces_output": produces_output}


def _short(i):
    return _prompt(i)[:20]


def _preempting(i):
    return _prompt(i)[: 16 if i < 4 else PROMPT]


@pytest.mark.parametrize(
    "ranks, prompt, drive",
    [
        # Middle chunks that produce no output, then two requests decoding.
        ({0: [0, 1]}, _prompt, {}),
        # Two prompts opening in one batch.
        ({0: [0, 1]}, _short, {}),
        ({0: [0, 1], 1: [2, 3]}, _prompt, {}),
        # Request 4 is preempted mid-prefill and restarts at token 0.
        ({0: range(5)}, _preempting, {"blocks": 19, "delayer": True}),
    ],
)
def test_every_recorded_step_rebuilt_records_its_rows_and_rung(
    tmp_path, monkeypatch, rule, ranks, prompt, drive
):
    _drive(tmp_path / "recorded", monkeypatch, ranks, prompt, **drive)
    recorded = read(tmp_path / "recorded")
    rebuild = rule().batch
    monkeypatch.setenv(ENV, str(tmp_path / "replayed"))
    for rank, steps in recorded.items():
        runner = _Runner(_config(rank))
        for step in steps:
            runner.forward(rebuild(step))

    def views(record):
        return {
            rank: [(s["rows"], s["rung"], s["produces_output"]) for s in steps]
            for rank, steps in record.items()
        }

    replayed = views(read(tmp_path / "replayed"))
    assert replayed == views(recorded)
    rungs = {rung for steps in replayed.values() for _, rung, _ in steps}
    assert None in rungs and len(rungs) > 1


def test_rows_hold_contiguous_blocks_of_a_fresh_pool_in_row_order(rule):
    # 40, 17 and 33 context tokens fill 3, 2 and 3 blocks of 16.
    line, rebuild = _line([[5, 40, False], [3, 17, False], [1, 33, True]]), rule()
    batch = rebuild.batch(line)
    assert [list(table) for table in batch.block_tables] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, 7],
    ]
    assert batch.num_scheduled_tokens.tolist() == [5, 3, 1]
    assert batch.context_lens.tolist() == [40, 17, 33]
    assert batch.num_cached_tokens == [35, 14, 32]
    # Each rebuild draws from a fresh pool, and its rows are new requests.
    again = rebuild.batch(line)
    assert [list(t) for t in again.block_tables] == [[0, 1, 2], [3, 4], [5, 6, 7]]
    assert not set(again.req_ids) & set(batch.req_ids)
    with pytest.raises(ValueError, match="hold 8 KV blocks .* the pool has 7"):
        rule(blocks=7).batch(line)


def test_tokens_are_the_fillers_the_simulated_runner_reports(
    rule, model_dir  # noqa: F811
):
    tokenizer = _load_tokenizer(str(model_dir))
    config = _config()
    fillers = filler_token_ids(tokenizer, config.eos_token_id, config.stop_token_ids)
    queries = [7, 2, 1]
    batch = rule().batch(_line([[q, 3 * BLOCK, False] for q in queries]))
    rows = np.split(batch.scheduled_tokens, np.cumsum(queries)[:-1])
    # What the simulated runner reports for requests numbered 0, 1 and 2.
    numbered = SimpleNamespace(req_ids=range(3), total_seqs_num=3)
    reported = DeferredTokenStream(fillers)._reply(numbered, deferred=True)
    assert [row.tolist() for row in rows] == [
        list(ids) * q for ids, q in zip(reported["token_ids"], queries)
    ]
    assert len(set(reported["token_ids"])) == 3


@pytest.mark.parametrize(
    "model_type, slots", [("qwen3_next", [[0], [1], [2]]), ("llama", [])]
)
def test_a_model_that_keeps_request_state_gives_each_row_a_fresh_slot(
    rule, model_type, slots
):
    rows = [[1, 9, True], [1, 30, True], [1, 2, True]]
    batch = rule(model_type).batch(_line(rows))
    assert batch.state_slots == slots
    assert batch.state_slots_committed == [slot for slot, in slots]


@pytest.mark.parametrize("produces_output", [True, False])
def test_prefill_rows_are_final_chunks_exactly_when_the_line_produces_output(
    rule, produces_output
):
    line = _line([[64, 64, False], [16, 80, False]], produces_output)
    batch = rule().batch(line)
    assert batch.is_final_chunk == [produces_output] * 2
    assert batch.produces_output() == produces_output


def test_the_batch_carries_the_rules_name_and_version(rule):
    batch = rule().batch(_line([[1, 9, True]]))
    assert batch.replay_rule == {"name": replay.NAME, "version": replay.VERSION}


@pytest.mark.parametrize(
    "line, refusal",
    [
        (_line([[1, 9, True], [8, 8, False]]), "decode row before a prefill row"),
        (_line([[2, 9, True]]), "a speculative step"),
        (_line([[1, 9, True]], False), "the line says it does not"),
    ],
)
def test_a_line_the_rule_cannot_rebuild_is_refused(rule, line, refusal):
    with pytest.raises(ValueError, match=refusal):
        rule().batch(line)


def test_a_tokenizer_with_no_filler_id_is_refused(rule, monkeypatch):
    monkeypatch.setattr(replay, "filler_token_ids", lambda *_: [])
    with pytest.raises(ValueError, match="no filler token"):
        rule()
