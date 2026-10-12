# SPDX-License-Identifier: MIT
"""Each step's scheduling decision, recorded on a real or a simulated runner.

A simulated run reuses ATOM's scheduler, so it should build the batches a real
run builds. This records them where both runs can be read the same way, at the
model runner's `forward`: one JSON line per call, built before the call runs,
because a real forward rewrites some batch fields in place. Each line carries
the batch as the `BatchView` a simulated runner prices it from (`rows`, `rung`);
a real runner's line also carries `t_enter_ns` and `t_exit_ns`, the worker's
monotonic clock around its forward, and is written once the forward returns.
Inside the forward it carries `[enter, exit]` on that clock around the staging
gate (`t_gate_ns`), the launch (`t_launch_ns`) and the wait for the previous
output's token copy (`t_recv_ns`, only on a forward that reaches it). Its device
window is `d_busy_ns`, from an event recorded at the launch's entry to one
recorded at the forward's exit, and `d_gap_ns`, from the previous forward's
exit event to this launch's entry event, absent on the first line; summed in
order they lay every forward on one device timeline. The events are polled,
never waited on, so a line is written once its exit event has completed, and
`exit` writes the rest. `compare` reads two records and names, per DP rank,
the first step where they part; timestamps are not part of a step's decision.

Setting `ATOM_COMPASS_PARITY_RECORD` to a directory turns recording on: TP rank 0
of each DP rank writes `dp<rank>.jsonl` there, replacing any earlier file.
Unset, nothing is recorded. A runner with a `host_key` also writes it to
`dp<rank>.key.json`. The simulated runner records from
`NonAllocatingRunner.forward`; a real one records through `StepRecording`,
composed into `RUNNER`, which the frontend names on a real run with the
variable set.

A request is named by its key, because request ids are numbered per run: a
digest of its whole prompt as its prefill windows carry it, taken when its final
chunk is scheduled, so it does not move with chunking. A line records request
ids and names each request whose final chunk it holds; `read` puts the keys in.
Prefix caching is not covered: the worker never sees a cached prefix, so parity
runs need caching off or the same cache hits in both runs. Two requests of one
run with one prompt could not be told apart in either record, so the second is
refused.
"""

import collections
import functools
import hashlib
import json
import os
import pathlib
import time
from itertools import zip_longest

import numpy as np

ENV = "ATOM_COMPASS_PARITY_RECORD"
RUNNER = "atom.compass.parity.runner.RecordingModelRunner"
#: Line keys that time a step rather than describe its decision.
TIMESTAMPS = (
    "t_enter_ns",
    "t_exit_ns",
    "t_gate_ns",
    "t_launch_ns",
    "t_recv_ns",
    "d_busy_ns",
    "d_gap_ns",
)


class StepRecord:
    """One DP rank's record: a JSON line per forward call."""

    def __init__(self, directory, dp_rank: int) -> None:
        self.path = pathlib.Path(directory) / f"dp{dp_rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.dp_rank = dp_rank
        self.step = 0
        self.keys: dict[int, str] = {}  # request id -> key
        self.owners: dict[str, int] = {}  # key -> request id
        self.prompts: dict[int, tuple] = {}  # request id -> (digest, next position)
        # (line, its device window: a callable taking `wait`, or None once closed)
        self.pending: collections.deque[tuple] = collections.deque()

    def add(self, batch, runner, forward=None):
        """Append `batch`'s line; with `forward`, run it inside the line's timestamps.

        `batch` is recorded as (request id, scheduled tokens, context length)
        per request, and as the `BatchView` `runner` prices it from: `rows`
        (query tokens, context tokens, decode) and `rung`, the graph width the
        step replays or None. Each prefill window is added to its request's
        digest; the line holding a request's final chunk names it under
        `named`. A dummy batch is fabricated for DP synchronisation and names
        no request, so its `batch` is empty; its `rows` are what it runs.

        The line is built before `forward` runs, since a real forward rewrites
        batch fields in place, and written after it returns, with `t_enter_ns`
        and `t_exit_ns` read off `runner.step_clock_ns()` around it and
        `runner.step_stamps()` added; a line with a device window waits in
        `pending` until the window closes. Returns `forward`'s reply, or None
        without one.
        """
        from atom.compass.runner.projection import batch_view, forward_mode

        view = batch_view(batch, forward_mode(batch, runner), runner)
        rows, named = [], []
        if not batch.is_dummy_run:
            ends = np.cumsum(batch.num_scheduled_tokens)
            for i, req_id in enumerate(batch.req_ids):
                num = int(batch.num_scheduled_tokens[i])
                rows.append([req_id, num, int(batch.context_lens[i])])
                if batch.is_final_chunk is None or req_id in self.keys:
                    continue
                start = int(batch.num_cached_tokens[i])
                if start == 0:  # a preempted prefill restarts its prompt
                    self.prompts.pop(req_id, None)
                digest, end = self.prompts.pop(
                    req_id, (hashlib.blake2b(digest_size=8), start)
                )
                if start != end:
                    raise ValueError(
                        f"request {req_id}'s prefill window starts at token "
                        f"{start}, not at {end} where its last one ended, so its "
                        "prompt cannot be digested whole."
                    )
                window = batch.scheduled_tokens[ends[i] - num : ends[i]]
                digest.update(np.asarray(window, dtype=np.int32).tobytes())
                if not batch.is_final_chunk[i]:
                    self.prompts[req_id] = (digest, start + num)
                    continue
                key = digest.hexdigest()
                if key in self.owners:
                    raise ValueError(
                        f"requests {self.owners[key]} and {req_id} were scheduled "
                        f"with the same prompt (key {key}), so no record can tell "
                        "them apart; give each request a prompt of its own."
                    )
                self.keys[req_id], self.owners[key] = key, req_id
                named.append([req_id, key])
        line = {
            "step": self.step,
            "dp_rank": self.dp_rank,
            "is_dummy_run": bool(batch.is_dummy_run),
            "produces_output": bool(batch.produces_output()),
            "batch": rows,
            "named": named,
            "rows": [
                [r.query_tokens, r.context_tokens, r.decode] for r in view.requests
            ],
            "rung": view.capture_rung,
        }
        reply = None
        window = None
        if forward is not None:
            line["t_enter_ns"] = runner.step_clock_ns()
            reply = forward(batch)
            line["t_exit_ns"] = runner.step_clock_ns()
            stamps, window = runner.step_stamps()
            line.update(stamps)
        self.pending.append((line, window))
        self.flush()
        self.step += 1
        return reply

    def flush(self, wait: bool = False) -> None:
        """Write the lines whose device windows have closed, in step order.

        A line stops the writing while its window is open, unless `wait`.
        """
        with self.path.open("a") as f:
            while self.pending:
                line, window = self.pending[0]
                if window is not None:
                    fields = window(wait)
                    if fields is None:
                        return
                    line.update(fields)
                f.write(json.dumps(line) + "\n")
                self.pending.popleft()


def record_step(runner, batch, forward=None):
    """Record `batch` if `ENV` names a directory and `runner` is TP rank 0.

    With `forward`, it runs on `batch` whether or not the step is recorded,
    inside the line's timestamps when it is, and its reply is returned.
    """
    if not hasattr(runner, "_parity_record"):
        directory = os.environ.get(ENV)
        record = None
        if directory and runner.rank == 0:
            if runner.config.pipeline_parallel_size > 1:
                # Every stage's TP rank 0 would write the same file.
                raise ValueError(
                    f"{ENV} records one file per DP rank, and under pipeline "
                    "parallelism each stage would write it."
                )
            dp_rank = runner.config.parallel_config.data_parallel_rank
            record = StepRecord(directory, dp_rank)
            key = getattr(runner, "host_key", None)
            if key is not None:
                record.path.with_suffix(".key.json").write_text(json.dumps(key))
        runner._parity_record = record
    if runner._parity_record is not None:
        return runner._parity_record.add(batch, runner, forward)
    return None if forward is None else forward(batch)


class StepRecording:
    """Records each forward call; put it before `ModelRunner` in the bases.

    ATOM's `ModelRunner.__init__` warms up with a forward over a fabricated
    batch, which the simulated runner declines, so recording starts once
    construction returns. The class it is composed into supplies
    `step_clock_ns`, the clock its lines are timed by, and `device_event`,
    which records a timing event on the current stream and returns it.

    A recorded forward is also stamped inside: around `_gate_staging_reuse`,
    `run_model` and the instance's `tokenID_processor.recv_async_output`, with
    device events at `run_model` entry and at the forward's exit.
    """

    _recording = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stamps: dict = {}
        self._last_exit = None
        processor = getattr(self, "tokenID_processor", None)
        if processor is not None:
            recv = processor.recv_async_output
            processor.recv_async_output = functools.partial(
                self._stamped, "t_recv_ns", recv
            )
        self._recording = True

    def forward(self, batch):
        if self._recording:
            return record_step(self, batch, super().forward)
        return super().forward(batch)

    def _stamped(self, key, call, *args):
        """`call(*args)`, stamped under `key` when this forward is recorded."""
        if getattr(self, "_parity_record", None) is None:
            return call(*args)
        enter = self.step_clock_ns()
        reply = call(*args)
        self._stamps[key] = [enter, self.step_clock_ns()]
        return reply

    def _gate_staging_reuse(self):
        return self._stamped("t_gate_ns", super()._gate_staging_reuse)

    def run_model(self, *args):
        if getattr(self, "_parity_record", None) is not None:
            self._stamps["launch_event"] = self.device_event()
        return self._stamped("t_launch_ns", super().run_model, *args)

    def step_stamps(self) -> tuple:
        """At a recorded forward's exit: its stamps, and its device window callable or None."""
        stamps, self._stamps = self._stamps, {}
        launch = stamps.pop("launch_event", None)
        window = None
        if launch is not None:
            done, previous = self.device_event(), self._last_exit
            self._last_exit = done
            window = functools.partial(_window, previous, launch, done)
        return stamps, window

    def exit(self):
        if getattr(self, "_parity_record", None) is not None:
            self._parity_record.flush(wait=True)
        return super().exit()


def _window(previous, launch, done, wait: bool):
    """The device window's fields once `done` has completed, polled, else None."""
    while not done.query():
        if not wait:
            return None
        time.sleep(1e-3)
    fields = {"d_busy_ns": round(launch.elapsed_time(done) * 1e6)}
    if previous is not None:
        fields["d_gap_ns"] = round(previous.elapsed_time(launch) * 1e6)
    return fields


def read(directory) -> dict[int, list[dict]]:
    """A record directory as {DP rank: steps}, each named request under its key.

    A request the record never named keeps its request id, so a record that
    ends inside a request's prefill parts from a full one at that request's
    first step.
    """
    record = {}
    for path in sorted(pathlib.Path(directory).glob("dp*.jsonl")):
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        keys = dict(pair for step in steps for pair in step.pop("named"))
        for step in steps:
            for row in step["batch"]:
                row[0] = keys.get(row[0], row[0])
        record[int(path.stem[2:])] = steps
    return record


def _request_ranks(name: str, record: dict[int, list[dict]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, steps in record.items():
        for step in steps:
            for key, _num, _context in step["batch"]:
                if ranks.setdefault(key, rank) != rank:
                    raise ValueError(
                        f"the {name} record schedules request {key} on DP ranks "
                        f"{ranks[key]} and {rank}; a request lives on one rank, "
                        "so the key joins two requests."
                    )
    return ranks


def compare(real, simulated) -> dict:
    """Two record directories: per DP rank, the first step whose decision differs.

    A rank's entry is None where its steps all agree, else the step index and
    both steps, None for a record that has already ended. A record with no
    steps is refused: two runs that scheduled nothing agree, and that means
    nothing. Steps are compared and reported without their timestamps.
    """
    records = {"real": read(real), "simulated": read(simulated)}
    for name, record in records.items():
        if not any(record.values()):
            raise ValueError(f"the {name} record has no steps to compare.")
        for steps in record.values():
            for step in steps:
                for key in TIMESTAMPS:
                    step.pop(key, None)
    first = {}
    for rank in sorted(records["real"].keys() | records["simulated"].keys()):
        pairs = zip_longest(
            records["real"].get(rank, []), records["simulated"].get(rank, [])
        )
        first[rank] = next(
            (
                {"step": i, "real": a, "simulated": b}
                for i, (a, b) in enumerate(pairs)
                if a != b
            ),
            None,
        )
    return {
        "first_divergence": first,
        "request_dp_rank": {
            name: _request_ranks(name, record) for name, record in records.items()
        },
    }
