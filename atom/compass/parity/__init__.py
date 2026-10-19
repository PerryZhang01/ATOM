# SPDX-License-Identifier: MIT
"""Each step's scheduling decision, recorded on a real or a simulated runner.

A simulated run reuses ATOM's scheduler, so it should build the batches a real
run builds. This records them where both runs can be read the same way, at the
model runner's `forward`: one JSON line per call, built before the call runs,
because a real forward rewrites some batch fields in place. Each line carries
the batch as the `BatchView` a simulated runner prices it from (`rows`, `rung`,
and above DP width 1 `moe_rows`, with an MTP drafter `draft_moe_rows` too); a
real runner settles that view from the `ForwardMode` its own forward settled,
so recording adds no DP collective. Its line also carries `t_enter_ns` and `t_exit_ns`, the worker's
monotonic clock around its forward, and is written once the forward returns.
Inside the forward it carries `[enter, exit]` on that clock around the staging
gate (`t_gate_ns`), the launch (`t_launch_ns`) and the wait for the previous
output's token copy (`t_recv_ns`, only on a forward that reaches it), and with
a drafter its MTP status (`t_mtp_ns`) and draft copy (`t_draft_recv_ns`). Its device
window is `d_busy_ns`, from an event recorded at the launch's entry to one
recorded at the forward's exit (`d_target_ns` to one recorded just after
`postprocess` records `forward_done_event`, or on a forward that samples
nothing, before the drafter's `compute_draft_kv`), and `d_gap_ns`, from the previous forward's
exit event to this launch's entry event, absent on the first line; summed in
order they lay every forward on one device timeline. A later pipeline stage's
receive of its input is `t_pp_recv_ns`, its device part `d_recv_ns`; an idle
`flush_pp_send` since the last forward that settled a pending send is `t_flush_ns`. The events are polled,
never waited on, so a line is written once its exit event has completed, and
`exit` writes the rest. Under `kv` a line carries the lmcache offload loads
and saves the scheduler handed the worker since the previous line
(`StepRecord.note_kv`), where any were. `compare` reads two records and names, per DP rank,
the first step where they part; timestamps are not part of a step's decision.

Setting `ATOM_COMPASS_PARITY_RECORD` to a directory turns recording on: TP rank 0
of each DP rank writes `dp<rank>.jsonl` there, replacing any earlier file
(pipeline stage `k` above 0: `pp<k>.dp<rank>.jsonl`, its other files alike;
a RapidServe engine in its `record_dir`).
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
A named request's response id follows its key if it has one (`carry_response_ids`).
Prefix caching is not covered: the worker never sees a cached prefix, so parity
runs need caching off or the same cache hits in both runs. Requests of one
record with one prompt are named in the order their final chunks are scheduled:
the first by the digest, the n-th after it by `<digest>.<n>`. A run that
schedules them in the other order names them the other way round, and `compare`
may then report a step where the twins' rows differ.
"""

import bisect
import collections
import copy
import functools
import hashlib
import json
import os
import pathlib
import sys
import time
from itertools import zip_longest

import numpy as np

ENV = "ATOM_COMPASS_PARITY_RECORD"
#: The file in a real run's record directory holding when its workload was sent
#: and when the server admitted each request.
ARRIVALS = "arrivals.json"
RUNNER = "atom.compass.parity.runner.RecordingModelRunner"
#: The recording runner of a real RapidServe run.
RAPID = "atom.compass.parity.runner.RecordingRapidServeModelRunner"
#: Line keys that time a step rather than describe its decision.
TIMESTAMPS = (
    "t_enter_ns",
    "t_exit_ns",
    "t_gate_ns",
    "t_launch_ns",
    "t_recv_ns",
    "t_mtp_ns",
    "t_draft_recv_ns",
    "t_pp_recv_ns",
    "t_flush_ns",
    "d_busy_ns",
    "d_gap_ns",
    "d_target_ns",
    "d_recv_ns",
    "t_pull_ns",
    "t_put_ns",
)


class StepRecord:
    """One DP rank's record: a JSON line per forward call."""

    def __init__(self, directory, dp_rank: int, stage: int = 0) -> None:
        name = f"pp{stage}.dp{dp_rank}" if stage else f"dp{dp_rank}"
        self.path = pathlib.Path(directory) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.dp_rank = dp_rank
        self.step = 0
        self.keys: dict[int, str] = {}  # request id -> key
        self.prompt_count: collections.Counter[str] = collections.Counter()
        self.prompts: dict[int, tuple] = {}  # request id -> (digest, next position)
        # (line, its device window: a callable taking `wait`, or None once closed)
        self.pending: collections.deque[tuple] = collections.deque()
        # Offload decisions handed to the worker, not yet on a line.
        self.kv: list[list] = []

    def note_kv(self, meta) -> None:
        """Keep each lmcache offload load and save `meta` hands the worker:
        ``[request id, "load", HBM-cached tokens, offload-cached tokens]`` or
        ``[request id, "save", tokens already saved, tokens computed]``."""
        for req in getattr(meta, "requests", ()):
            if (ls := req.load_spec) is not None:
                self.kv.append(
                    [req.req_id, "load", ls.hbm_cached_tokens, ls.lmcache_cached_tokens]
                )
            if (ss := req.save_spec) is not None:
                self.kv.append(
                    [req.req_id, "save", ss.skip_leading_tokens, len(req.token_ids)]
                )

    def add(self, batch, runner, forward=None):
        """Append `batch`'s line; with `forward`, run it inside the line's timestamps.

        `batch` is recorded as (request id, scheduled tokens, context length)
        per request, and as the `BatchView` `runner` prices it from: `rows`
        (query tokens, context tokens, decode), `rung`, the graph width the
        step replays or None, and `moe_rows`, the rows the DP group's MoE
        gather carries, absent at DP width 1, and `draft_moe_rows` where the
        view has it. The view's mode is the one `forward` settled, stamped by
        `run_model` under `forward_mode`; without one it is `forward_mode`'s,
        which runs the DP collective itself. Each prefill window is added to its request's
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
                if batch.external_request_ids[i] is not None:
                    named[-1].append(batch.external_request_ids[i])
        line = {
            "step": self.step,
            "dp_rank": self.dp_rank,
            "is_dummy_run": bool(batch.is_dummy_run),
            "produces_output": bool(batch.produces_output()),
            "batch": rows,
            "named": named,
        }
        if self.kv:
            line["kv"], self.kv = self.kv, []
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
        if view.draft_moe_rows is not None:
            line["draft_moe_rows"] = view.draft_moe_rows
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
    if _record(runner) is not None:
        return runner._parity_record.add(batch, runner, forward)
    return None if forward is None else forward(batch)


def record_kv(runner, meta) -> None:
    """Keep the offload decisions in `meta`, dispatched to `runner`'s worker,
    for its next recorded step's line, as `record_step` would record it."""
    if _record(runner) is not None:
        runner._parity_record.note_kv(meta)


def record_dir(directory, config) -> pathlib.Path:
    """Where the engine of `config` records under `directory`: RapidServe's
    prefill and decode engines in ``prefill/`` and ``decode/``, any other there."""
    if not getattr(config, "enable_rapidserve", False):
        return pathlib.Path(directory)
    return pathlib.Path(directory) / (
        "decode" if config.disagg_is_decode else "prefill"
    )


def _record(runner):
    """`runner`'s record, made on its first call; None unless recorded."""
    if not hasattr(runner, "_parity_record"):
        directory = os.environ.get(ENV)
        record = None
        if directory and runner.rank == 0:
            directory = record_dir(directory, runner.config)
            from atom.compass.run.step_grid import BOUNDS, keyed

            pc = runner.config.parallel_config
            pp = runner.config.pipeline_parallel_size > 1
            stage = pc.pipeline_parallel_rank if pp else 0
            record = StepRecord(directory, pc.data_parallel_rank, stage)
            # By its first step the runner has resolved its engine.
            engine = keyed(runner.config) | {
                name: getattr(runner.config, name)
                for name in (*BOUNDS, "num_kvcache_blocks")
            }
            record.path.with_suffix(".engine.json").write_text(json.dumps(engine))
            key = getattr(runner, "host_key", None)
            if key is not None:
                record.path.with_suffix(".key.json").write_text(json.dumps(key))
        runner._parity_record = record
    return runner._parity_record


def carry_response_ids() -> None:
    """Set a one-sequence request's `external_request_id` to its response id,
    the `request_id` of the API server function calling `add_request`, read
    off its frame. A sequence with an id keeps it; a fan-out's get none."""
    from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

    for cls in (CoreManager, DisaggCoreManager):

        def add_request(self, seqs, add=cls.add_request):
            if len(seqs) == 1 and seqs[0].external_request_id is None:
                caller = sys._getframe(1).f_locals
                seqs[0].external_request_id = caller.get("request_id")
            return add(self, seqs)

        cls.add_request = add_request


def _memory_format(name: str):
    import torch

    return getattr(torch, name)


def key_memory_formats_by_name() -> None:
    """Pickle a `torch.memory_format` as its name in `torch`.

    Pickle otherwise saves one under the first module in `sys.modules` that
    reaches it, so inductor's FX graph cache keys a weight by the process's
    import order: an API server's engine and `compass measure memory`'s keyed
    the same graphs apart, and a server started after the measurement compiled
    them again, reading `non_torch` above it. Each real runner calls this
    before its model compiles."""
    import copyreg

    import torch

    copyreg.pickle(
        torch.memory_format,
        lambda f: (_memory_format, (str(f).removeprefix("torch."),)),
    )


class StepRecording:
    """Records each forward call; put it before `ModelRunner` in the bases.

    ATOM's `ModelRunner.__init__` warms up with a forward over a fabricated
    batch, which the simulated runner declines, so recording starts once
    construction returns. The class it is composed into supplies
    `step_clock_ns`, the clock its lines are timed by, and `device_event`,
    which records a timing event on the current stream and returns it.

    A recorded forward is also stamped inside: around `_gate_staging_reuse`,
    `run_model` and the instance's `tokenID_processor.recv_async_output`, with
    device events at `run_model` entry and at the forward's exit. At
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
        pp = getattr(getattr(self, "config", None), "pipeline_parallel_size", 1) > 1
        if pp and not getattr(processor, "is_deferred_out", True):
            # A stage reads this forward's tokens off the device: the wait is
            # the call. Off a pipeline the gate releases that wait itself.
            sampled = processor.prepare_sampled_ids
            processor.prepare_sampled_ids = functools.partial(
                self._stamped, "t_recv_ns", sampled
            )
        if pp:
            from aiter.dist.parallel_state import get_pp_group

            group = get_pp_group()
            if not group.is_first_rank:
                recv = group.recv_tensor_dict
                group.recv_tensor_dict = functools.partial(self._pp_received, recv)
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
            draft_kv = self.drafter.compute_draft_kv
            self.drafter.compute_draft_kv = functools.partial(
                self._target_done, draft_kv
            )
        self._recording = True

    def forward(self, batch):
        if self._recording:
            return record_step(self, batch, super().forward)
        return super().forward(batch)

    def process_kvconnector_output(self, meta):
        if self._recording:
            record_kv(self, meta)
        return super().process_kvconnector_output(meta)

    def flush_pp_send(self):
        """ATOM's idle flush, stamped for the next forward when a send is pending."""
        if getattr(self, "_pp_pending_send", None):
            return self._stamped("t_flush_ns", super().flush_pp_send)
        return super().flush_pp_send()

    def _stamped(self, key, call, *args):
        """`call(*args)`, stamped under `key` when this forward is recorded."""
        if getattr(self, "_parity_record", None) is None:
            return call(*args)
        enter = self.step_clock_ns()
        reply = call(*args)
        self._stamps[key] = [enter, self.step_clock_ns()]
        return reply

    def _pp_received(self, call, *args, **kwargs):
        """The stage before's output, received under `t_pp_recv_ns`, then a
        device event where the received tensors are in."""
        reply = self._stamped("t_pp_recv_ns", lambda: call(*args, **kwargs))
        if getattr(self, "_parity_record", None) is not None:
            self._stamps["recv_event"] = self.device_event()
        return reply

    def _target_done(self, call, *args):
        """`call(*args)` after a device event where `postprocess` has just
        recorded `forward_done_event`, or where `compute_draft_kv` starts: the
        target's end, before the draft. The later of the two wins."""
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
        names = {"d_target_ns": "target_event", "d_recv_ns": "recv_event"}
        marks = {k: stamps.pop(e) for k, e in names.items() if e in stamps}
        window = None
        if launch is not None:
            done, previous = self.device_event(), self._last_exit
            self._last_exit = done
            window = functools.partial(_window, previous, launch, done, marks)
        return stamps, window

    def exit(self):
        if getattr(self, "_parity_record", None) is not None:
            self._parity_record.flush(wait=True)
        return super().exit()


def _window(previous, launch, done, marks, wait: bool):
    """The device window's fields once `done` has completed, polled, else None."""
    while not done.query():
        if not wait:
            return None
        time.sleep(1e-3)
    fields = {"d_busy_ns": round(launch.elapsed_time(done) * 1e6)}
    for key, event in marks.items():
        fields[key] = round(launch.elapsed_time(event) * 1e6)
    if previous is not None:
        fields["d_gap_ns"] = round(previous.elapsed_time(launch) * 1e6)
    return fields


def read(directory, stage: int = 0) -> dict[int, list[dict]]:
    """A record directory's `stage` as {DP rank: steps}, each named request under its key.

    A request the record never named keeps its request id, so a record that
    ends inside a request's prefill parts from a full one at that request's
    first step. With `ARRIVALS`, the worker-clock times a workload was sent,
    the steps entered before the first send are the engine's start, not the
    workload's: they are left out and the rest renumbered from 0. A step
    naming requests with response ids holds them by key in `response_ids`.
    A real rank's ``dp<rank>.pulls.jsonl`` (`clock.PullStamps`) gives each
    step the first ``t_pull_ns`` and ``t_put_ns`` at or after its
    ``t_exit_ns`` and before the next step's ``t_enter_ns``, where there is one.
    """
    directory, record, start = pathlib.Path(directory), {}, None
    if (directory / ARRIVALS).exists():
        start = min(json.loads((directory / ARRIVALS).read_text())["sent_ns"])
    # Each rank's record, and not the cost lines a simulated run writes beside it.
    prefix = f"pp{stage}." if stage else ""
    for path in sorted(directory.glob(f"{prefix}dp*[0-9].jsonl")):
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        if (pulls := path.with_suffix(".pulls.jsonl")).exists():
            _join_stamps(steps, pulls)
        keys = {}
        for step in steps:
            for req_id, key, *response_id in step.pop("named"):
                keys[req_id] = key
                if response_id:
                    step.setdefault("response_ids", {})[key] = response_id[0]
        if start is not None:
            steps = [s for s in steps if s["t_enter_ns"] >= start]
            for i, step in enumerate(steps):
                step["step"] = i
        for step in steps:
            for row in (*step["batch"], *step.get("kv", ())):
                row[0] = keys.get(row[0], row[0])
        record[int(path.stem[len(prefix) + 2 :])] = steps
    return record


def _join_stamps(steps: list[dict], path) -> None:
    """Put into each step the first stamp of each key in `path` at or after
    its exit and before the next step's entry."""
    times = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        for key, ns in json.loads(line).items():
            times[key].append(ns)
    ends = [s["t_enter_ns"] for s in steps[1:]] + [float("inf")]
    for step, end in zip(steps, ends):
        for key, ts in times.items():
            i = bisect.bisect_left(ts, step["t_exit_ns"])
            if i < len(ts) and ts[i] < end:
                step[key] = ts[i]


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
    nothing. Steps are compared and reported without timestamps or response ids. Under
    `steps`, each rank's count of step pairs compared and of identical ones.
    """
    records = {"real": read(real), "simulated": read(simulated)}
    for name, record in records.items():
        if not any(record.values()):
            raise ValueError(f"the {name} record has no steps to compare.")
        for steps in record.values():
            for step in steps:
                for key in (*TIMESTAMPS, "response_ids"):
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
