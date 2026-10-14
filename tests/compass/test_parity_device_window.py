# SPDX-License-Identifier: MIT
"""A real runner's stamps inside each forward, and its device window, on a planted clock.

`StepRecording` is composed over a base that calls what `ModelRunner.forward`
calls, in its order: the staging gate, the launch, and, on a forward that
produces output, the wait on the previous output's token copy. Host time is a
counter, so every stamp is distinct, and device time is planted per event: each
forward's launch and exit events land at known device times, and an event
completes once the device clock has passed it.
"""

import numpy as np
from test_parity_record import GRAPHABLE, LADDER, _config, _drive

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


def _recorded(warmup):
    device = _Device()
    clock = iter(range(1, 10**9))

    class _Base:
        """`ModelRunner.forward`'s calls, with the device one forward behind the host."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.attn_metadata_builder = GRAPHABLE
            self.tokenID_processor = _Processor()
            self.exited = False
            self.forward(warmup)

        def _gate_staging_reuse(self):
            pass

        def run_model(self, input_ids, batch):
            return "logits"

        def forward(self, batch):
            self._gate_staging_reuse()
            assert self.run_model([], batch) == "logits"
            if batch.produces_output():
                assert self.tokenID_processor.recv_async_output([]) == []
            # The previous forward's exit event has completed, this one's has not.
            device.now = device.times[-2] if len(device.times) > 1 else -1
            return "ran"

        def exit(self):
            self.exited = True

    class _Recorded(StepRecording, _Base):
        @staticmethod
        def step_clock_ns():
            return next(clock)

        @staticmethod
        def device_event():
            # Launch after the previous exit, exit after the launch.
            last = device.times[-1] if device.times else 0
            gap = BUSY + len(device.times) if len(device.times) % 2 else GAP
            device.times.append(last + gap)
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
        assert ("t_recv_ns" in step) == output
        assert step["d_busy_ns"] == BUSY + 2 * n + 1
        assert step.get("d_gap_ns") == (GAP if n else None)
    assert runner.tokenID_processor.calls == 1 + sum(outputs)
    assert compare(tmp_path / "real", tmp_path / "simulated")["first_divergence"] == {
        0: None
    }
