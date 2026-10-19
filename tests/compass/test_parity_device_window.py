# SPDX-License-Identifier: MIT
"""A real runner's stamps inside each forward, and its device window, on a planted clock.

`StepRecording` is composed over a base that calls what `ModelRunner.forward`
calls, in its order: the staging gate, the launch, and, on a forward that
produces output, the wait on the previous output's token copy. Host time is a
counter, so every stamp is distinct, and device time is planted per event: each
forward's launch and exit events land at known device times, and an event
completes once the device clock has passed it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_parity_record import GRAPHABLE, LADDER, _config, _drive

from atom.compass.backends.step_labels import label
from atom.compass.parity import ENV, TIMESTAMPS, StepRecording, compare, read

GAP, BUSY = 7_000, 100_000  # ns


class _Device:
    """A device clock; `tick` moves it on each poll."""

    def __init__(self):
        self.now, self.tick, self.times = -1, 0, []


class _Event:
    def __init__(self, device, t):
        self.device, self.t = device, t

    def query(self):
        self.device.now += self.device.tick
        return self.t <= self.device.now

    def elapsed_time(self, other):
        assert self.query() and other.query(), "an event read before it completed"
        return (other.t - self.t) / 1e6  # ms, as torch reports it


class _Processor:
    def __init__(self):
        self.calls = 0

    def recv_async_output(self, handle):
        self.calls += 1
        return handle

    def recv_mtp_status_async(self):
        return None, None

    def prepare_sampled_ids(self, batch):
        return self.recv_async_output([])

    def recv_async_output_draft(self):
        return []


TARGET, SAMPLE, KV, PROPOSE = 60_000, 2_000, 9_000, 5_000  # ns
#: `eagle-dp` is EAGLE under DP lockstep; DSpark writes its draft KV on every forward.
DRAFTERS = ("eagle", "eagle-dp", "dspark")


def _recorded(warmup, drafter=None):
    """With `drafter` (`DRAFTERS`), a forward calls the drafter's
    `compute_draft_kv` after its target, except an `eagle-dp` middle chunk,
    which runs an align-only `propose_draft_token_ids` instead. An output
    forward also waits for the MTP status before its launch, then samples,
    calls `prepare_sampled_ids` right after `forward_done_event`, proposes and
    waits for the draft copy, as ATOM does. Each pass's device time lands on
    the next event recorded."""
    device = _Device()
    clock = iter(range(1, 10**9))

    class _Base:
        """`ModelRunner.forward`'s calls, with the device one forward behind the host."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.attn_metadata_builder = GRAPHABLE
            self.tokenID_processor = _Processor()
            if drafter:

                def draft_kv(positions, hidden, next_token_ids):
                    # EAGLE's returns early where a row samples (-1).
                    if drafter == "dspark" or -1 not in next_token_ids:
                        self.next = "kv"

                self.drafter = SimpleNamespace(compute_draft_kv=draft_kv)
            self.exited = False
            self.forward(warmup)

        def _gate_staging_reuse(self):
            pass

        def run_model(self, input_ids, batch):
            return "logits"

        def forward(self, batch):
            p, output = self.tokenID_processor, batch.produces_output()
            self._gate_staging_reuse()
            if drafter and output:
                p.recv_mtp_status_async()
            self.next = "launch"
            assert self.run_model([], batch) == "logits"
            self.next = "target"
            align = drafter == "eagle-dp" and not output
            if drafter and not align:
                self.drafter.compute_draft_kv([], "hidden", [-1] if output else [7])
            if align:
                self.propose_draft_token_ids(batch, align_only=True)
            if drafter and output:
                self.next = "sample"
                p.prepare_sampled_ids(batch)
                self.propose_draft_token_ids(batch)
                p.recv_async_output_draft()
            elif output:
                assert p.prepare_sampled_ids(batch) == []
            if not drafter:
                self.next = "busy"
            # The previous forward's exit event has completed, this one's has not.
            device.now = device.times[-2] if len(device.times) > 1 else -1
            return "ran"

        def propose_draft_token_ids(self, batch, align_only=False):
            self.next = "propose"

        def exit(self):
            self.exited = True

    class _Recorded(StepRecording, _Base):
        @staticmethod
        def step_clock_ns():
            return next(clock)

        def device_event(self):
            # Launch after the previous exit, exit after the launch.
            last = device.times[-1] if device.times else 0
            step = {"launch": GAP, "target": TARGET, "sample": SAMPLE}
            step |= {"kv": KV, "propose": PROPOSE, "idle": 0}
            device.times.append(last + step.get(self.next, BUSY + len(device.times)))
            self.next = "idle"
            return _Event(device, device.times[-1])

    return _Recorded, device


def test_each_line_carries_its_host_stamps_and_device_window(tmp_path, monkeypatch):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    simulated = read(tmp_path / "simulated")[0]
    assert not {key for step in simulated for key in step} & set(TIMESTAMPS)
    record = tmp_path / "real" / "dp0.jsonl"

    recorded, device = _recorded(batches[-1])
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, _config())
    assert runner.tokenID_processor.calls == 1 and device.times == []
    assert [runner.forward(b) for b in batches] == ["ran"] * len(batches)

    # Each line waits for its exit event; the last one's has not completed.
    assert len(record.read_text().splitlines()) == len(batches) - 1
    device.tick = BUSY // 3
    runner.exit()
    assert runner.exited

    real = read(tmp_path / "real")[0]
    assert len(real) == len(batches)
    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    for n, (step, output) in enumerate(zip(real, outputs)):
        stamps = [step["t_enter_ns"], *step["t_gate_ns"], *step["t_launch_ns"]]
        stamps += step["t_recv_ns"] if output else []
        assert stamps == sorted(stamps) and stamps[-1] < step["t_exit_ns"]
        assert ("t_recv_ns" in step) == output and "d_target_ns" not in step
        assert step["d_busy_ns"] == BUSY + 2 * n + 1
        assert step.get("d_gap_ns") == (GAP if n else None)
    assert runner.tokenID_processor.calls == 1 + sum(outputs)
    assert compare(tmp_path / "real", tmp_path / "simulated")["first_divergence"] == {
        0: None
    }


@pytest.mark.parametrize("drafter", DRAFTERS)
def test_with_a_drafter_each_line_splits_its_device_window_at_the_target(
    tmp_path, monkeypatch, drafter
):
    """A middle chunk's target ends where its first drafter pass starts; an
    output forward's where it samples, less a draft-KV pass before that."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1], drafter=drafter)
    device.tick = 10**12
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, _config())
    for b in batches:
        runner.forward(b)
    runner.exit()

    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    for step, output in zip(read(tmp_path / "real")[0], outputs):
        keys = {"t_mtp_ns", "t_recv_ns", "t_draft_recv_ns"}
        assert keys <= step.keys() if output else not keys & step.keys()
        target = TARGET + SAMPLE * output
        kv = KV * (drafter == "dspark" or drafter == "eagle" and not output)
        draft = kv + PROPOSE * (output or drafter == "eagle-dp")
        assert (step["d_target_ns"], step["d_busy_ns"]) == (target, target + draft)
        if output:
            stamps = [step["t_enter_ns"], *step["t_gate_ns"], *step["t_mtp_ns"]]
            stamps += [*step["t_launch_ns"], *step["t_recv_ns"]]
            stamps += [*step["t_draft_recv_ns"], step["t_exit_ns"]]
            assert stamps == sorted(stamps)
    # No step of the record is left without its target end.
    assert {s.cut for s in label(tmp_path / "real" / "dp0.jsonl")[0]} <= {None, "mixed"}
