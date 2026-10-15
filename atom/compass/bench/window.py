# SPDX-License-Identifier: MIT
"""The bench's window: a step's device time, read off the step recording's events.

`BenchWindow.bench_step` runs one batch through the runner's `forward` again
and again, with the bench standing where `StepRecording`'s record would. Before
each forward it queues a preload on the worker's stream, device work meant to
outlast the host's issue of the whole forward, and records event F at the
forward's entry; `StepRecording` records A at `run_model` entry and B at the
forward's exit. A repeat's value is device time from F to B; F to A, the device
work input preparation queues, and A to B, the record's `d_busy_ns`, are kept
beside it.

A repeat counts only if F is still pending once the forward has returned and B
is recorded: the device was still in the preload, so nothing the host did shows
between F and B. Otherwise it is discarded and the preload doubles. A forward
with a host synchronisation inside never qualifies, and after `DOUBLINGS`
doublings its point refuses.

`WARMUPS` forwards run first, with no preload, and size the first preload: the
longest the host took to issue one of them. Then `REPEATS` repeats are kept,
and the statistic is their median. Every worker runs each forward, and the
workers agree on it: a repeat counts only where it counts on every worker, and
its times are each the slowest worker's, so the reply of rank 0, the only one
the runner manager returns, carries the step charged to its slowest rank.
"""

import copy
import statistics

import torch
import torch.distributed as dist

#: The measuring protocol's version; it changes whenever a count below does.
PROTOCOL = 1
WARMUPS = 5
REPEATS = 20
DOUBLINGS = 5


class BenchWindow:
    """Put before `StepRecording` in the bases.

    The class it is composed into supplies, beside `StepRecording`'s
    `step_clock_ns` and `device_event`, `preload(ns)`, which queues about `ns`
    of device work on the current stream, and `bench_group()`, the process
    group the workers agree over, or None for one worker.
    """

    def bench_step(self, batch) -> dict:
        """`batch` measured: its kept repeats and their median, or a refusal."""
        preload = max(self._repeat(batch, 0)[4] for _ in range(WARMUPS))
        kept, discarded = [], 0
        while len(kept) < REPEATS:
            missed, *times, _host = self._repeat(batch, preload)
            if not missed:
                kept.append(times)
                continue
            discarded += 1
            if discarded > DOUBLINGS:
                return {
                    "protocol": PROTOCOL,
                    "refused": (
                        f"the forward's entry event completed before the forward "
                        f"returned behind every preload up to {preload} ns: a host "
                        "synchronisation inside the forward"
                    ),
                }
            preload *= 2
        return {
            "protocol": PROTOCOL,
            "median_ns": statistics.median(ns for ns, _, _ in kept),
            "repeats": kept,  # each [F to B, F to A, A to B] in ns
            "discarded": discarded,
            "preload_ns": preload,
        }

    def _repeat(self, batch, preload: int) -> list[int]:
        """One forward behind `preload` ns, agreed by every worker:
        [missed, F to B, F to A, A to B, host ns]."""
        forward = self._parity_record = _Forward(preload)
        try:
            # A real forward may rewrite batch fields in place.
            self.forward(copy.deepcopy(batch))
        finally:
            self._parity_record = None
        group = self.bench_group()
        if group is None:
            return forward.values
        values = torch.tensor(forward.values, dtype=torch.int64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        return values.tolist()


class _Forward:
    """What `record_step` hands a bench forward to, in place of a `StepRecord`."""

    def __init__(self, preload: int) -> None:
        self.preload = preload

    def add(self, batch, runner, forward):
        start = runner.step_clock_ns()
        if self.preload:
            runner.preload(self.preload)
        entry = runner.device_event()
        # F stands where the previous forward's exit would, so the record's
        # device gap is F to A.
        runner._last_exit = entry
        reply = forward(batch)
        _stamps, window = runner.step_stamps()  # records B
        host = runner.step_clock_ns() - start
        missed = entry.query()
        exit_event = runner._last_exit
        fields = window(True)
        self.values = [
            int(missed),
            round(entry.elapsed_time(exit_event) * 1e6),
            fields["d_gap_ns"],
            fields["d_busy_ns"],
            host,
        ]
        return reply
