# SPDX-License-Identifier: MIT
"""The per-step record, taken off ATOM's own scheduler, and the first-divergence compare.

Each run drives the real `Scheduler` and the simulated runner's `forward` in the
engine's order -- schedule, forward, postprocess -- one scheduler per DP rank,
with every prompt distinct.
"""

import hashlib
import json
import queue
import runpy
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.compass import run
from atom.compass.backends.shape import ShapeStubBackend
from atom.compass.parity import ENV, RUNNER, StepRecording, compare, read
from atom.compass.runner import overrides
from atom.compass.runner.overrides import NonAllocatingRunner, install_cost_backend
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.sampling_params import SamplingParams
from atom.utils import clock

PROMPT, BUDGET, BLOCK = 200, 64, 16
#: The capture ladder every runner here replays decode steps from.
LADDER = [1, 2, 4, 8]


def _prompt(i):
    return list(range(5 + 1000 * i, 5 + 1000 * i + PROMPT))


def _key(tokens):
    data = np.asarray(tokens, dtype=np.int32).tobytes()
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def _config(dp_rank=0, pipeline_parallel_size=1, budget=BUDGET, blocks=4096):
    return MockConfig(
        max_num_seqs=8,
        num_kvcache_blocks=blocks,
        kv_cache_block_size=BLOCK,
        max_model_len=2048,
        max_num_batched_tokens=budget,
        pipeline_parallel_size=pipeline_parallel_size,
        parallel_config=SimpleNamespace(
            data_parallel_rank=dp_rank, data_parallel_size=1
        ),
    )


class _Runner(NonAllocatingRunner):
    """The simulated runner's `forward`, with the attributes it reads."""

    def __init__(self, config, rank=0):
        self.config, self.rank = config, rank
        install_cost_backend(self, ShapeStubBackend())
        self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
        self._token_stream = DeferredTokenStream([0])


@pytest.fixture(scope="module", autouse=True)
def _unpriced():
    """Reporting is under test here, not the price: `test_dp_step_max.py` prices."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        yield


def _drive(
    directory,
    monkeypatch,
    ranks,
    prompt=_prompt,
    budget=BUDGET,
    blocks=4096,
    delayer=False,
):
    """Run each DP rank's requests to completion; return its batches and sequences.

    `delayer` installs ATOM's `PrefillDelayer` at its defaults, as its DP engine does.
    """
    monkeypatch.setenv(ENV, str(directory))
    built = {}
    for dp_rank, requests in ranks.items():
        config = _config(dp_rank, budget=budget, blocks=blocks)
        scheduler, runner = Scheduler(config), _Runner(config)
        if delayer:
            scheduler.set_prefill_delayer(PrefillDelayer(1, None, budget))
        sequences = [
            Sequence(
                prompt(i),
                BLOCK,
                sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
            )
            for i in requests
        ]
        for sequence in sequences:
            scheduler.add(sequence)
        batches = []
        while scheduler.has_unfinished_requests():
            assert len(batches) < 100, "the driven sequence does not terminate"
            scheduled = scheduler.schedule()
            if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
                continue
            batch, seqs = scheduled
            batches.append(batch)
            reply = runner.forward(batch)
            if delayer and batch.total_seqs_num_prefill:
                scheduler.prefill_delayer.notify_prefill_executed()
            scheduler.postprocess(
                list(seqs.values()),
                reply,
                stream_output_queue=queue.Queue(),
                batch=batch,
            )
        built[dp_rank] = (batches, {s.id: s for s in sequences}, runner)
    return built


@pytest.mark.parametrize(
    "length, first_rows",
    # At 20 tokens both prompts open in one batch, the second at row 1, and no
    # chunk is a middle one.
    [(PROMPT, [0, 0]), (20, [0, 1])],
)
def test_the_record_is_the_batches_the_scheduler_built(
    tmp_path, monkeypatch, length, first_rows
):
    # An earlier run's file in the same directory is replaced, not appended to.
    _drive(tmp_path, monkeypatch, {0: [2]})
    built = _drive(tmp_path, monkeypatch, {0: [0, 1]}, lambda i: _prompt(i)[:length])
    batches, sequences, runner = built[0]
    keys, expected, opened_at = {}, [], []
    for step, batch in enumerate(batches):
        rows, shapes = [], []
        for i, req_id in enumerate(batch.req_ids):
            if req_id not in keys:
                opened_at.append(i)
            key = keys.setdefault(req_id, _key(sequences[req_id].prompt_token_ids))
            num, context = int(batch.num_scheduled_tokens[i]), int(
                batch.context_lens[i]
            )
            rows.append([key, num, context])
            # Every prefill chunk here is 8 tokens or more; a decode is 1.
            shapes.append([num, context, num == 1])
        decoding = all(decode for *_, decode in shapes)
        expected.append(
            {
                "step": step,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": batch.produces_output(),
                "batch": rows,
                "rows": shapes,
                # A decode step replays the narrowest graph that holds it.
                "rung": min(w for w in LADDER if w >= len(rows)) if decoding else None,
            }
        )
    assert len(set(keys.values())) == 2 and opened_at == first_rows
    assert any(not step["produces_output"] for step in expected) == (length == PROMPT)
    assert {step["rung"] for step in expected} == {None, 2}
    assert read(tmp_path) == {0: expected}

    # The dummy batch `ModelRunner.dummy_execution` builds for DP synchronisation.
    seq = Sequence([0], block_size=BLOCK, id=-1)
    seq.type = SequenceType.DECODE
    runner.forward(
        ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            total_tokens_num=1,
            total_tokens_num_decode=1,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
        )
    )
    assert read(tmp_path)[0][-1] == {
        "step": len(batches),
        "dp_rank": 0,
        "is_dummy_run": True,
        "produces_output": True,
        "batch": [],
        "rows": [[1, 1, True]],
        "rung": 1,
    }


def test_two_records_of_one_sequence_compare_equal(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1], 1: [2, 3]})
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"] == {0: None, 1: None}
    ranks = report["request_dp_rank"]
    assert ranks["real"] == ranks["simulated"]
    assert sorted(ranks["real"].values()) == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "budget, length",
    # At a 128-token budget requests 1, 2 and 3 open with a different first
    # chunk in each run, 64 tokens in one and 128 in the other.
    [(BUDGET, PROMPT), (128, 150)],
)
def test_a_request_moved_to_the_other_rank_names_its_first_step(
    tmp_path, monkeypatch, budget, length
):
    def prompt(i):
        return _prompt(i)[:length]

    real = _drive(
        tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]}, prompt, budget
    )
    _drive(tmp_path / "simulated", monkeypatch, {0: [0], 1: [1, 2, 3]}, prompt, budget)
    report = compare(tmp_path / "real", tmp_path / "simulated")
    real_record = read(tmp_path / "real")
    first = report["first_divergence"]
    # Rank 0 parts where request 1 is first scheduled on the real side.
    moved = next(
        step for step, b in enumerate(real[0][0]) if any(r == 1 for r in b.req_ids)
    )
    assert first[0]["step"] == moved
    assert first[0]["real"] == real_record[0][moved]
    assert first[1]["step"] == 0
    ranks = report["request_dp_rank"]
    assert ranks["real"].keys() == ranks["simulated"].keys()
    assert [
        (ranks["real"][key], ranks["simulated"][key])
        for key in ranks["real"]
        if ranks["real"][key] != ranks["simulated"][key]
    ] == [(0, 1)]


def test_a_shifted_chunk_boundary_names_its_first_step(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    shutil.copytree(tmp_path / "real", tmp_path / "simulated")
    path = tmp_path / "simulated" / "dp0.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    # Request 0's first two chunks, 64 and 64, become 60 and 68.
    assert [s["batch"][0][1:] for s in steps[:2]] == [[64, 64], [64, 128]]
    steps[0]["batch"][0][1:] = [60, 60]
    steps[1]["batch"][0][1:] = [68, 128]
    path.write_text("".join(json.dumps(s) + "\n" for s in steps))
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"][1] is None
    first = report["first_divergence"][0]
    assert first["step"] == 0
    assert first["real"]["batch"][0][1:] == [64, 64]
    assert first["simulated"]["batch"][0][1:] == [60, 60]


def test_a_record_that_stops_early_or_has_no_rank_parts_where_it_ends(
    tmp_path, monkeypatch
):
    _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    shutil.copytree(tmp_path / "real", tmp_path / "simulated")
    real = read(tmp_path / "real")
    (tmp_path / "simulated" / "dp0.jsonl").unlink()
    cut = tmp_path / "simulated" / "dp1.jsonl"
    lines = cut.read_text().splitlines(keepends=True)
    # Cut after the line naming rank 1's last request, so every row is keyed.
    end = max(n for n, line in enumerate(lines) if json.loads(line)["named"]) + 1
    assert end < len(lines)
    cut.write_text("".join(lines[:end]))
    first = compare(tmp_path / "real", tmp_path / "simulated")["first_divergence"]
    assert first == {
        0: {"step": 0, "real": real[0][0], "simulated": None},
        1: {"step": end, "real": real[1][end], "simulated": None},
    }


def test_requests_behind_one_system_prompt_keep_their_keys_at_every_budget(
    tmp_path, monkeypatch
):
    # A 192-token system prompt, then a distinct 36-token tail; the first
    # chunks differ with the budget.
    def prompt(i):
        return _prompt(9)[:192] + _prompt(i)[:36]

    keys = []
    for budget in (64, 128, 512):
        _drive(tmp_path / str(budget), monkeypatch, {0: [0, 1, 2, 3]}, prompt, budget)
        steps = read(tmp_path / str(budget))[0]
        keys.append({row[0] for step in steps for row in step["batch"]})
    assert keys[0] == keys[1] == keys[2] == {_key(prompt(i)) for i in range(4)}


def test_two_requests_with_one_prompt_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="with the same prompt"):
        _drive(tmp_path, monkeypatch, {0: [0, 1]}, lambda i: _prompt(0))


def test_a_window_that_does_not_continue_the_last_is_refused(tmp_path, monkeypatch):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    runner = _Runner(_config())
    runner.forward(batches[0])
    with pytest.raises(ValueError, match="starts at token 128, not at 64"):
        runner.forward(batches[2])


@pytest.mark.parametrize(
    "blocks, starts",
    # Four 16-token requests decode beside request 4's 200-token prompt, and
    # the pool is too small for all of them, so request 4 is preempted.
    [
        # Mid-prefill, held by the delayer: it restarts at token 0.
        (19, [0, 64, 128, 0, 64, 128, 192]),
        # After its final chunk: it re-prefills, already named.
        (17, [0, 64, 128, 192, 0, 64, 128, 192]),
    ],
)
def test_a_preempted_request_keeps_its_whole_prompt_key(
    tmp_path, monkeypatch, blocks, starts
):
    def prompt(i):
        return _prompt(i)[: 16 if i < 4 else PROMPT]

    built = _drive(
        tmp_path, monkeypatch, {0: range(5)}, prompt, blocks=blocks, delayer=True
    )
    batches, sequences, _ = built[0]
    last = list(sequences)[-1]
    assert [
        int(b.num_cached_tokens[i])
        for b in batches
        for i, req_id in enumerate(b.req_ids)
        if req_id == last and b.num_scheduled_tokens[i] > 1
    ] == starts
    steps = read(tmp_path)[0]
    assert {row[0] for step in steps for row in step["batch"]} == {
        _key(prompt(i)) for i in range(5)
    }


def test_one_key_on_two_ranks_is_refused(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0], 1: [0]})
    _drive(tmp_path / "simulated", monkeypatch, {0: [0]})
    with pytest.raises(ValueError, match="the real record schedules request"):
        compare(tmp_path / "real", tmp_path / "simulated")


def test_an_empty_record_is_refused(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0]})
    (tmp_path / "simulated").mkdir()
    with pytest.raises(ValueError, match="the simulated record has no steps"):
        compare(tmp_path / "real", tmp_path / "simulated")


@pytest.mark.parametrize("env, rank", [(None, 0), ("out", 1)])
def test_only_tp_rank_0_records_and_only_when_asked(tmp_path, monkeypatch, env, rank):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    if env is None:
        monkeypatch.delenv(ENV)
    else:
        monkeypatch.setenv(ENV, str(tmp_path / env))
    _Runner(_config(), rank).forward(batches[0])

    class _Base:
        def __init__(self, config, rank):
            self.config, self.rank, self.ran = config, rank, []

        def forward(self, batch):
            self.ran.append(batch)
            return "ran"

    class _Recorded(StepRecording, _Base):
        step_clock_ns = staticmethod(time.monotonic_ns)

    # A real runner that does not record still runs its forward.
    recorded = _Recorded(_config(), rank)
    assert recorded.forward(batches[0]) == "ran"
    assert recorded.ran == [batches[0]]
    assert not (tmp_path / "out").exists()


def test_pipeline_parallelism_is_refused(tmp_path, monkeypatch):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    runner = _Runner(_config(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="each stage would write it"):
        runner.forward(batches[0])


def test_the_mixin_records_from_the_first_forward_after_construction(
    tmp_path, monkeypatch
):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]

    class _Base:
        """`ModelRunner`'s shape: construction runs a warmup forward."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.forward(batches[1])

        def forward(self, batch):
            return "ran"

    class _Recorded(StepRecording, _Base):
        step_clock_ns = staticmethod(time.monotonic_ns)

    monkeypatch.setenv(ENV, str(tmp_path / "out"))
    runner = _Recorded(0, _config())
    assert not (tmp_path / "out").exists()
    assert runner.forward(batches[0]) == "ran"
    assert [s["step"] for s in read(tmp_path / "out")[0]] == [0]


def test_a_real_runner_records_the_batch_it_was_handed_and_times_its_forward(
    tmp_path, monkeypatch, capsys
):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    record = tmp_path / "real" / "dp0.jsonl"

    class _Base:
        """A forward that rewrites the batch it runs, as ATOM's does."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.inside = []

        def forward(self, batch):
            # The line is written once this returns, not before.
            assert len(record.read_text().splitlines()) == len(self.inside)
            self.inside.append(time.monotonic_ns())
            batch.num_scheduled_tokens[:] = 0
            batch.context_lens[:] = 0
            return "ran"

    class _Recorded(StepRecording, _Base):
        step_clock_ns = staticmethod(time.monotonic_ns)

    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = _Recorded(0, _config())
    assert [runner.forward(b) for b in batches] == ["ran"] * len(batches)
    real, simulated = read(tmp_path / "real")[0], read(tmp_path / "simulated")[0]
    for step, inside in zip(real, runner.inside, strict=True):
        assert step.pop("t_enter_ns") <= inside <= step.pop("t_exit_ns")
    assert real == simulated
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"] == {0: None}

    # The documented invocation prints the same report.
    argv = ["python -m atom.compass.parity", str(tmp_path / "real")]
    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "simulated")])
    runpy.run_module("atom.compass.parity", run_name="__main__")
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(report))


@pytest.mark.parametrize(
    "record, named, runs",
    [
        (None, run.ATOM_RUNNER, run.ATOM_RUNNER),
        (None, "x.RapidServeModelRunner", "x.RapidServeModelRunner"),
        ("record", run.ATOM_RUNNER, RUNNER),
    ],
)
def test_a_real_run_records_through_the_recording_runner_only_when_asked(
    tmp_path, monkeypatch, record, named, runs
):
    monkeypatch.delenv(run.ENV, raising=False)
    monkeypatch.delenv(ENV, raising=False)
    if record:
        monkeypatch.setenv(ENV, str(tmp_path / record))
    config = SimpleNamespace(runner_qualname=named)
    with run.frontend(config):
        pass
    assert config.runner_qualname == runs
    assert clock.installed() is None


def test_a_real_run_asked_to_record_refuses_a_runner_that_cannot(tmp_path, monkeypatch):
    monkeypatch.delenv(run.ENV, raising=False)
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(ValueError, match="only .* has a recording runner"):
        run.frontend(SimpleNamespace(runner_qualname="x.RapidServeModelRunner"))
