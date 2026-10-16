# SPDX-License-Identifier: MIT
"""Each step's scheduling decision, recorded on a real or a simulated runner.

A simulated run reuses ATOM's scheduler, so it should build the batches a real
run builds. This records them where both runs can be read the same way, at the
model runner's `forward`: one JSON line per call, built before the call runs,
because a real forward rewrites some batch fields in place. Each line carries
the batch as the `BatchView` a simulated runner prices it from (`rows`, `rung`,
and above DP width 1 `moe_rows`); a real runner settles that view from the
`ForwardMode` its own forward settled, so recording adds no DP collective. Its
line also carries `t_enter_ns` and `t_exit_ns`, the worker's
monotonic clock around its forward, and is written once the forward returns.
Inside the forward it carries `[enter, exit]` on that clock around the staging
gate (`t_gate_ns`), the launch (`t_launch_ns`) and the wait for the previous
output's token copy (`t_recv_ns`, only on a forward that reaches it); with a
drafter, also around the waits for the previous output's MTP status
(`t_mtp_ns`, before launch) and its draft copy (`t_draft_recv_ns`). Its device
window is `d_busy_ns`, from an event recorded at the launch's entry to one
recorded at the forward's exit, with a drafter `d_target_ns`, from that entry
event to one recorded where `forward_done_event` is, and `d_gap_ns`, from the previous forward's
exit event to this launch's entry event, absent on the first line; summed in
order they lay every forward on one device timeline. The events are polled,
never waited on, so a line is written once its exit event has completed, and
`exit` writes the rest. `compare` reads two records and names, per DP rank,
the first step where they part; timestamps are not part of a step's decision.

Setting `ATOM_COMPASS_PARITY_RECORD` to a directory turns recording on: TP rank 0
of each DP rank writes `dp<rank>.jsonl` there, replacing any earlier file.
Unset, nothing is recorded. On its first step it also writes the engine its
runner resolved to `dp<rank>.engine.json` (`README.md` says what it holds), and
a runner with a `host_key` writes that to `dp<rank>.key.json`. The simulated
runner records from `NonAllocatingRunner.forward`; a real one records through
`StepRecording`, composed into `RUNNER`, which the frontend names on a real run
with the variable set.

A request is named by its key, because request ids are numbered per run: a
digest of its whole prompt as its prefill windows carry it, taken when its final
chunk is scheduled, so it does not move with chunking. A line records request
ids and names each request whose final chunk it holds; `read` puts the keys in.
Prefix caching is not covered: the worker never sees a cached prefix, so parity
runs need caching off or the same cache hits in both runs. Requests of one
record with one prompt are named in the order their final chunks are scheduled:
the first by the digest, the n-th after it by `<digest>.<n>`. A run that
schedules them in the other order names them the other way round, and `compare`
may then report a step where the twins' rows differ.
"""

import collections
import copy
import functools
import hashlib
import json
import os
import pathlib
import time
from itertools import zip_longest

import numpy as np

ENV = "ATOM_COMPASS_PARITY_RECORD"
#: The file in a real run's record directory holding when its workload was sent
#: and when the server admitted each request.
ARRIVALS = "arrivals.json"
RUNNER = "atom.compass.parity.runner.RecordingModelRunner"
#: Line keys that time a step rather than describe its decision.
TIMESTAMPS = (
    "t_enter_ns",
    "t_exit_ns",
    "t_gate_ns",
    "t_launch_ns",
    "t_recv_ns",
    "t_mtp_ns",
    "t_draft_recv_ns",
    "d_busy_ns",
    "d_gap_ns",
    "d_target_ns",
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
        self.prompt_count: collections.Counter[str] = collections.Counter()
        self.prompts: dict[int, tuple] = {}  # request id -> (digest, next position)
        # (line, its device window: a callable taking `wait`, or None once closed)
        self.pending: collections.deque[tuple] = collections.deque()

    def add(self, batch, runner, forward=None):
        """Append `batch`'s line; with `forward`, run it inside the line's timestamps.

        `batch` is recorded as (request id, scheduled tokens, context length)
        per request, and as the `BatchView` `runner` prices it from: `rows`
        (query tokens, context tokens, decode), `rung`, the graph width the
        step replays or None, and `moe_rows`, the rows the DP group's MoE
        gather carries, absent at DP width 1. The view's mode is the one
        `forward` settled, stamped by `run_model` under `forward_mode`;
        without one it is `forward_mode`'s, which runs the DP collective
        itself. Each prefill window is added to its request's
        digest; the line holding a request's final chunk names it under
        `named`. A dummy batch is fabricated for DP synchronisation and names
        no request, so its `batch` is empty; its `rows` are what it runs.

        The line is built from the batch as handed in, since a real forward
        rewrites batch fields in place, and written after it returns, with `t_enter_ns`
        and `t_exit_ns` read off `runner.step_clock_ns()` around it and
        `runner.step_stamps()` added; a line with a device window waits in
        `pending` until the window closes. Returns `forward`'s reply, or
        without one the mode the view was built from, so a caller pricing the
        step need not run the collective again.
        """
        from atom.compass.runner.projection import batch_view, forward_mode

        # The view is settled after `forward`, from the fields it was handed.
        before = copy.copy(batch)
        before.num_scheduled_tokens = np.copy(batch.num_scheduled_tokens)
        before.context_lens = np.copy(batch.context_lens)
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
                n = self.prompt_count[key]
                self.prompt_count[key] += 1
                if n:
                    key = f"{key}.{n}"
                self.keys[req_id] = key
                named.append([req_id, key])
        line = {
            "step": self.step,
            "dp_rank": self.dp_rank,
            "is_dummy_run": bool(batch.is_dummy_run),
            "produces_output": bool(batch.produces_output()),
            "batch": rows,
            "named": named,
        }
        reply, window, mode = None, None, None
        if forward is not None:
            line["t_enter_ns"] = runner.step_clock_ns()
            reply = forward(batch)
            line["t_exit_ns"] = runner.step_clock_ns()
            stamps, window = runner.step_stamps()
            mode = stamps.pop("forward_mode", None)
            line.update(stamps)
        mode = mode or forward_mode(before, runner)
        view = batch_view(before, mode, runner)
        line["rows"] = [
            [r.query_tokens, r.context_tokens, r.decode] for r in view.requests
        ]
        line["rung"] = view.capture_rung
        if view.moe_rows is not None:
            line["moe_rows"] = view.moe_rows
        self.pending.append((line, window))
        self.flush()
        self.step += 1
        return mode if forward is None else reply

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
    Without `forward`, a recorded step returns the `ForwardMode` it settled
    and an unrecorded one None.
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
            from atom.compass.run.step_grid import BOUNDS, keyed

            dp_rank = runner.config.parallel_config.data_parallel_rank
            record = StepRecord(directory, dp_rank)
            # By its first forward the runner has resolved its engine.
            engine = keyed(runner.config) | {
                name: getattr(runner.config, name)
                for name in (*BOUNDS, "num_kvcache_blocks")
            }
            record.path.with_suffix(".engine.json").write_text(json.dumps(engine))
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
    device events at `run_model` entry and at the forward's exit. With a
    drafter, also around the processor's `recv_mtp_status_async` and
    `recv_async_output_draft`, with a device event at its
    `prepare_sampled_ids` entry. At
    `run_model` entry it also keeps the `ForwardMode` the forward settled,
    which the line's view is built from.
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
        if processor is not None and hasattr(self, "drafter"):
            for key, name in (
                ("t_mtp_ns", "recv_mtp_status_async"),
                ("t_draft_recv_ns", "recv_async_output_draft"),
            ):
                call = getattr(processor, name)
                setattr(processor, name, functools.partial(self._stamped, key, call))
            sampled = processor.prepare_sampled_ids
            processor.prepare_sampled_ids = functools.partial(
                self._target_done, sampled
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

    def _target_done(self, call, *args):
        """`call(*args)` after a device event where `postprocess` has just
        recorded `forward_done_event`: the target's end, before the draft."""
        if getattr(self, "_parity_record", None) is not None:
            self._stamps["target_event"] = self.device_event()
        return call(*args)

    def _gate_staging_reuse(self):
        return self._stamped("t_gate_ns", super()._gate_staging_reuse)

    def run_model(self, *args):
        if getattr(self, "_parity_record", None) is not None:
            from atom.utils.forward_context import get_forward_context

            self._stamps["launch_event"] = self.device_event()
            context = get_forward_context().context
            self._stamps["forward_mode"] = getattr(context, "forward_mode", None)
        return self._stamped("t_launch_ns", super().run_model, *args)

    def step_stamps(self) -> tuple:
        """At a recorded forward's exit: its stamps, and its device window callable or None."""
        stamps, self._stamps = self._stamps, {}
        launch = stamps.pop("launch_event", None)
        target = stamps.pop("target_event", None)
        window = None
        if launch is not None:
            done, previous = self.device_event(), self._last_exit
            self._last_exit = done
            window = functools.partial(_window, previous, launch, done, target)
        return stamps, window

    def exit(self):
        if getattr(self, "_parity_record", None) is not None:
            self._parity_record.flush(wait=True)
        return super().exit()


def _window(previous, launch, done, target, wait: bool):
    """The device window's fields once `done` has completed, polled, else None."""
    while not done.query():
        if not wait:
            return None
        time.sleep(1e-3)
    fields = {"d_busy_ns": round(launch.elapsed_time(done) * 1e6)}
    if target is not None:
        fields["d_target_ns"] = round(launch.elapsed_time(target) * 1e6)
    if previous is not None:
        fields["d_gap_ns"] = round(previous.elapsed_time(launch) * 1e6)
    return fields


def read(directory) -> dict[int, list[dict]]:
    """A record directory as {DP rank: steps}, each named request under its key.

    A request the record never named keeps its request id, so a record that
    ends inside a request's prefill parts from a full one at that request's
    first step. With `ARRIVALS`, the worker-clock times a workload was sent,
    the steps entered before the first send are the engine's start, not the
    workload's: they are left out and the rest renumbered from 0.
    """
    directory, record, start = pathlib.Path(directory), {}, None
    if (directory / ARRIVALS).exists():
        start = min(json.loads((directory / ARRIVALS).read_text())["sent_ns"])
    # Each rank's record, and not the cost lines a simulated run writes beside it.
    for path in sorted(directory.glob("dp*[0-9].jsonl")):
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        keys = dict(pair for step in steps for pair in step.pop("named"))
        if start is not None:
            steps = [s for s in steps if s["t_enter_ns"] >= start]
            for i, step in enumerate(steps):
                step["step"] = i
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
    nothing. Steps are compared and reported without their timestamps. Under
    `steps`, each rank's count of step pairs compared and of identical ones.
    """
    records = {"real": read(real), "simulated": read(simulated)}
    for name, record in records.items():
        if not any(record.values()):
            raise ValueError(f"the {name} record has no steps to compare.")
        for steps in record.values():
            for step in steps:
                for key in TIMESTAMPS:
                    step.pop(key, None)
    first, counts = {}, {}
    for rank in sorted(records["real"].keys() | records["simulated"].keys()):
        pairs = list(
            zip_longest(
                records["real"].get(rank, []), records["simulated"].get(rank, [])
            )
        )
        first[rank] = next(
            (
                {"step": i, "real": a, "simulated": b}
                for i, (a, b) in enumerate(pairs)
                if a != b
            ),
            None,
        )
        counts[rank] = {
            "compared": len(pairs),
            "identical": sum(a == b for a, b in pairs),
        }
    return {
        "first_divergence": first,
        "steps": counts,
        "request_dp_rank": {
            name: _request_ranks(name, record) for name, record in records.items()
        },
    }
