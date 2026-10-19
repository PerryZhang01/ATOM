# SPDX-License-Identifier: MIT
"""A real sending stage's record stamps each idle `flush_pp_send` that settles
a send, on the next forward's line, and the PP gate names the steps where the
flush it infers and that stamp disagree."""

import json
import time

import numpy as np
from test_parity_record import GRAPHABLE, LADDER, _config, _drive
from test_pp_gate_holds import _step
from test_pp_stage_input import MS, pp2_record

from atom.compass.parity import ENV, StepRecording, compare, read, recurrence


def test_a_flush_that_settles_a_send_is_stamped_on_the_next_forward(
    tmp_path, monkeypatch
):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]

    class _Base:
        """ATOM's runner: a forward settles the send before it and, on a
        sending stage, leaves its own pending; a flush settles it."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.attn_metadata_builder = GRAPHABLE
            self._pp_pending_send, self.sends = [], True

        def forward(self, batch):
            self._pp_pending_send = ["work"] if self.sends else []
            return "ran"

        def flush_pp_send(self):
            self._pp_pending_send.clear()
            return True

    class _Recorded(StepRecording, _Base):
        step_clock_ns = staticmethod(time.monotonic_ns)

    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = _Recorded(0, _config())
    runner.forward(batches[0])
    assert runner.flush_pp_send()
    assert runner._pp_pending_send == []  # ATOM's flush ran
    assert runner.flush_pp_send()
    runner.sends = False
    runner.forward(batches[1])
    runner.flush_pp_send()  # the forward settled its send: nothing pending
    for batch in batches[2:]:
        runner.forward(batch)
    steps = read(tmp_path / "real")[0]
    assert ["t_flush_ns" in s for s in steps] == [False, True] + [False] * (
        len(steps) - 2
    )
    enter, exit_ = steps[1]["t_flush_ns"]
    assert steps[0]["t_exit_ns"] <= enter <= exit_ <= steps[1]["t_enter_ns"]
    # A stamp times a step; it is not part of its decision.
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"] == {0: None}


def test_the_gate_reports_a_sending_stages_inferred_and_stamped_flushes():
    """The head waits for its tokens before its third forward, so it infers
    a flush there; the record stamps one before the second."""
    steps = [_step(n, 2 * n, 1) for n in range(3)]
    steps[1]["t_flush_ns"] = [MS, MS]
    sends = [[0, 0, 0, 0, t * MS] for t in (50, 80)]
    held = [False, False, (0, 60 * MS)]
    *_, report = recurrence._charged(steps, (0, 2), None, held, None, sends)
    assert report["flush"] == ([2], [1])
    *_, last = recurrence._charged(steps, (1, 2), [(0, 0)] * 3, None)
    assert last["flush"] is None


def _gate(tmp_path, capsys, head, last):
    """The script's output on a record of `head` and `last`, split per stage."""
    for name, steps in zip(("dp0", "pp1.dp0"), (head, last)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in steps)
        )
    assert recurrence.main(str(tmp_path)) == 0
    return capsys.readouterr().out.split("dp0 pp1 kind")


def test_the_script_names_each_sending_stages_disagreements(tmp_path, capsys):
    head, last = pp2_record()
    head[2]["t_flush_ns"] = [19 * MS, 19 * MS]
    stage0, stage1 = _gate(tmp_path, capsys, head, last)
    assert "flush: 1 steps disagree, 2 inferred, 1 stamped" in stage0
    assert "step 3: inferred only" in stage0
    assert "flush" not in stage1


def test_a_record_without_flush_stamps_says_so_and_counts_nothing(tmp_path, capsys):
    stage0, stage1 = _gate(tmp_path, capsys, *pp2_record())
    assert "flush: the record stamps no flush\n" in stage0
    assert "disagree" not in stage0 and "inferred only" not in stage0
    assert "flush" not in stage1
