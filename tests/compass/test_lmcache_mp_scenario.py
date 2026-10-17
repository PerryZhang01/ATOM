# SPDX-License-Identifier: MIT
"""`lmcache_mp` held to the scenario checks: one deployment, and two DP ranks
sharing one server, each run twice under a constant step stub and constant
station laws.

The workload is the parity workload's with a shared prefix
(`atom.compass.parity.traffic.bodies`): request 0 stores the prefix, its
blocks leave HBM when it ends since prefix caching is off, and a burst of
three requests per DP rank later loads it back. With
``OFFLOAD_ASYNC_LOOKUP_DEPTH=1`` two requests of each burst reach admission
unprefetched and look up blocking, and with ``OFFLOAD_LOOKUP_DEFER_S=0`` the
prefetched one is deferred once, then looks up blocking too: a lookup job is
priced above a step, so its reply is not in by the next pass.

The runs need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`, and as many GPUs in ``HIP_VISIBLE_DEVICES`` as DP ranks.
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest
from test_lmcache_mp_engine import _machine
from test_parity_record import _config as _parity_config
from test_parity_record import _key
from test_parity_record import _Runner as _SimRunner
from test_vertical_slice import (
    CONSTANT,
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    _free_port,
    _run_file,
)

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.carriers import tracestate_with
from atom.compass.clock import TAR
from atom.compass.detect.scenario import check_run, expected_timeline
from atom.compass.kv.lmcache_server import reply_channel
from atom.compass.parity import ENV as RECORD
from atom.compass.parity import StepRecording, read
from atom.compass.parity.traffic import bodies
from atom.compass.runner import overrides
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence

PREFIX_WORDS = 600  # two whole 256-token chunks
BURST_S, BURST = 0.5, 3  # the burst's time, and its requests per DP rank
STATION_S = 2.0**-6  # every request-loop job: above a step under `CONSTANT`
COPY_S = 2.0**-7  # every copy: the host link's latency, its rate adding nothing
OFFLOAD_ENV = {
    "OFFLOAD_ASYNC_LOOKUP_DEPTH": "1",
    "OFFLOAD_LOOKUP_DEFER_S": "0",
    "OFFLOAD_MIN_LOAD_TOKENS": "256",
    "OFFLOAD_MIN_SAVE_TOKENS": "256",
}
ATOM_ARGS = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]
DP2 = ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]


class Workload(Traffic):
    """The traffic LP: request 0 at 0, the rest at `BURST_S`."""

    def __init__(self, requests, *args) -> None:
        super().__init__(*args)
        self.requests = requests

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            at = BURST_S if k else 0.0
            while rt.now < at:
                rt.next_event(at)
                self._take()
            self.sent, seq = rt.stamp_send(self.http)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()


def _measure(out: Path, machine: Path, width: list) -> None:
    from atom.compass import cli

    gpus = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    cli.main(
        ["measure", "memory", "--gpu", gpus, "--machine", str(machine)]
        + ["--store", str(out / "store"), "--out", str(out / "measure")]
        + ["--", *ATOM_ARGS, *width]
    )


def run_scenario(root: Path, seed: str, dp: int, lmcache_port: int) -> dict:
    """One run in its own process tree, reading the readings in ``root/store``."""
    out = root / f"seed{seed}"
    out.mkdir()
    port = _free_port()
    flags = ["--host", "127.0.0.1", "--port", str(lmcache_port)]
    flags += ["--chunk-size", "256", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
    machine = json.loads((root / "machine.json").read_text())
    run_file = _run_file(
        out,
        step_stub=CONSTANT,
        machine=machine,
        store=str(root / "store"),
        data_parallel_size=dp,
        bound_s=60.0,
        lmcache={"server": flags, "lookup_s": [STATION_S, 0.0], "request_s": STATION_S},
    )
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": lmcache_port}
    env = dict(
        os.environ,
        **OFFLOAD_ENV,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        ATOM_KV_OFFLOAD="lmcache_mp",
        ATOM_KV_OFFLOAD_EXTRA_CONFIG=json.dumps(extra),
    )
    requests = bodies(MODEL, 1 + BURST * dp, PREFIX_WORDS)
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "atom.entrypoints.openai.api_server", *ATOM_ARGS]
            + (DP2 if dp > 1 else [])
            + ["--host", "127.0.0.1", "--server-port", str(port)]
            + ["--no-enable_prefix_caching", "--compass-run", str(run_file)],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            traffic = Workload(requests, json.loads(run_file.read_text()), port, server)
            traffic.run()
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    text = (out / "server.log").read_text()
    assert code == 0, f"the server exited {code} after the finish:\n{text[-4000:]}"
    return {
        "out": out,
        "run": json.loads(run_file.read_text()),
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "log": text,
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


@pytest.fixture(scope="module", params=[1, 2], ids=["dp1", "dp2"])
def runs(request, tmp_path_factory):
    dp = request.param
    root = tmp_path_factory.mktemp(f"lmcache_dp{dp}")
    machine = _machine(root)
    spec = json.loads(machine.read_text())
    spec["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": 1e30,
        "latency_s": COPY_S,
    }
    machine.write_text(json.dumps(spec))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AITER_LOG_LEVEL", "WARNING")  # the readings are keyed by it
        _measure(root, machine, DP2 if dp > 1 else [])
    lmcache_port = _free_port()  # the run file names it, so both runs share it
    return dp, [run_scenario(root, seed, dp, lmcache_port) for seed in ("1", "2")]


def _releases(table: str, lp=None) -> list[tuple[float, str, int]]:
    """``(arrival, channel, seq)`` of each message released, to `lp` alone if named."""
    return sorted(
        (float(f[2]), f[4], int(f[5]))
        for f in map(str.split, table.splitlines()[1:])
        if f[3] == "release" and lp in (None, f[0])
    )


def station_timeline(table: str, ipc: float) -> dict[str, list[float]]:
    """Each reply channel's arrivals, worked out from the requests LP lmcache
    was released alone: each station starts a job at the request's arrival or
    once its previous job ends, whichever is later, in ``(arrival, channel,
    seq)`` order; a request-loop job takes `STATION_S` and a copy `COPY_S`;
    the reply arrives `ipc` after the job ends."""
    free, want = {}, defaultdict(list)
    for arrival, ch, _ in _releases(table, "lmcache"):
        station = ":transfer#" in ch
        free[station] = max(arrival, free.get(station, 0.0))
        free[station] += COPY_S if station else STATION_S
        want[reply_channel(ch)].append(free[station] + ipc)
    return want


def _sends(run, rank) -> dict[str, list[float]]:
    """Each of DP rank `rank`'s channels to LP lmcache: its sends' times, in seq
    order, each its arrival less the channel's lookahead."""
    ipc = run["run"].get("lmcache_ipc_s", run["run"]["ipc_s"])
    sent = defaultdict(list)
    for arrival, ch, _ in sorted(
        _releases(run["table"], "lmcache"), key=lambda r: r[1:]
    ):
        if re.search(rf"#dp{rank}(\.|$)", ch):
            sent[ch].append(arrival - ipc)
    return sent


def step_starts(table: str, record) -> list[float]:
    """Each step's start, its first TAR row's clock, as `expected_timeline`
    lays the record's steps over the engine's TAR rows."""
    tars = [
        (float(f[1]), float(f[2]))
        for f in map(str.split, table.splitlines()[1:])
        if f[0] == "engine" and f[3] == TAR
    ]
    stub = Coefficients(**CONSTANT)
    rows = (stub.prefill_step, stub.decode_step, tars)
    n = len(record[0])
    return [
        tars[len(expected_timeline({r: s[:k] for r, s in record.items()}, *rows))][0]
        for k in range(n)
    ]


# --- (a) ----------------------------------------------------------------------


@NEEDS_A_RUN
def test_each_run_meets_every_scenario_check_and_loads_the_prefix_back(runs):
    dp, (run, _) = runs
    failed = check_run(
        [r["out"] for r in runs[1]],
        run["sent"],
        run["taken"],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    transfers = re.findall(
        r"compass: lmcache_mp (\w+) of (\S+):\S+ (\w+): arrived", run["log"]
    )
    print("", *(" ".join(t) for t in transfers), sep="\n")
    stored = [e for op, e, ok in transfers if op == "store" and ok == "succeeded"]
    loaded = {e for op, e, ok in transfers if op == "retrieve" and ok == "succeeded"}
    assert loaded, "no request loaded the stored prefix back"
    if dp > 1:
        assert loaded - {stored[0]}, "no rank loaded a prefix another rank stored"


# --- (b) ----------------------------------------------------------------------


@NEEDS_A_RUN
def test_each_server_job_and_reply_follows_the_constant_station_laws(runs):
    _, (run, _) = runs
    got = defaultdict(list)
    for arrival, ch, _ in sorted(_releases(run["table"]), key=lambda r: r[1:]):
        if ch.startswith("lmcache->"):
            got[ch].append(arrival)
    want = station_timeline(run["table"], run["run"]["ipc_s"])
    assert sorted(got) == sorted(want)
    for ch in want:
        assert got[ch] == pytest.approx(want[ch], rel=0, abs=1e-12), ch


@NEEDS_A_RUN
def test_each_offload_decision_is_sent_once_from_each_writer_before_its_forward(runs):
    """A step line's load or save goes out once on each writer rank's transfer
    channel, after the previous step starts and by its own step's start."""
    _, (run, _) = runs
    record = read(run["out"] / "parity")
    starts = step_starts(run["table"], record)
    writers = run["run"]["lmcache"].get("writers", 1)
    for rank, steps in record.items():
        decided = [k for k, step in enumerate(steps) for _ in step.get("kv", ())]
        sent = _sends(run, rank)
        for ch in (f"engine->lmcache:transfer#dp{rank}.tp{w}" for w in range(writers)):
            assert len(sent[ch]) == len(decided), ch
            for k, at in zip(decided, sent[ch]):
                assert (starts[k - 1] if k else 0.0) <= at <= starts[k], (ch, k, at)


TAKEN = re.compile(
    r"lmcache_mp (store|retrieve) of atom-offload-dp(\d+):(\d+) \w+: "
    r"arrived at (\S+), taken at (\S+)"
)
REPORTED = re.compile(r"lmcache_mp reports at (\S+): (.*)")


@NEEDS_A_RUN
def test_a_load_parked_request_is_released_no_earlier_than_its_load_is_reported(runs):
    """The shell takes each reply no earlier than it arrives and reports it
    once, no earlier than it took it; the request a load parked prefills on a
    step that starts no earlier than that report."""
    _, (run, _) = runs
    takes, reports, retrieved = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in run["log"].splitlines():
        if m := TAKEN.search(line):
            op, rank, rid, arrived, at = m.groups()
            assert float(at) >= float(arrived), line
            takes[op == "retrieve", rid].append(float(at))
            if op == "retrieve":
                retrieved[int(rank)].append(rid)
        elif m := REPORTED.search(line):
            for kind, rid in re.findall(r"(Save|Load)OperationId\(req_id=(\d+)", m[2]):
                reports[kind == "Load", rid].append(float(m[1]))
    assert takes.keys() == reports.keys()
    for op, at in takes.items():
        assert len(reports[op]) == len(at), op
        assert all(r >= t for t, r in zip(at, reports[op])), op
    record = read(run["out"] / "parity")
    starts = step_starts(run["table"], record)
    for rank, steps in record.items():
        loads = [
            (k, key)
            for k, step in enumerate(steps)
            for key, kind, *_ in step.get("kv", ())
            if kind == "load"
        ]
        for (k, key), rid in zip(loads, retrieved[rank], strict=True):
            p = next(
                j
                for j in range(k, len(steps))
                if key in {r[0] for r in steps[j]["batch"]}
            )
            assert starts[p] >= reports[True, rid][0], (rank, key, rid, p)


@NEEDS_A_RUN
def test_each_dp_member_parks_once_a_step_and_leaves_at_the_latest_park(runs):
    """Each member parks once before each forward and sends LP lmcache nothing
    while parked; every member leaves when the last one parks, as the step
    starts. Each member's park CPU is reported."""
    dp, (run, _) = runs
    if dp == 1:
        pytest.skip("a member parks at DP width above one")
    record = read(run["out"] / "parity")
    starts, parks = step_starts(run["table"], record), {}
    for rank in record:
        lp = compass_run.LP_FILE.format(f"engine.dp{rank}")
        parks[rank] = json.loads((run["out"] / lp).read_text())["parks"]
        cpu_us = 1e6 * parks[rank]["cpu_s"] / parks[rank]["count"]
        print(f"\ndp{rank}: {parks[rank]['count']} parks, {cpu_us:.1f} us CPU each")
        assert parks[rank]["count"] == len(parks[rank]["at"]) == len(starts)
        sent = sorted(t for times in _sends(run, rank).values() for t in times)
        for parked, left in parks[rank]["at"]:
            assert not [t for t in sent if parked < t < left], (rank, parked, left)
    for k, start in enumerate(starts):
        at = [parks[rank]["at"][k] for rank in record]
        assert {left for _, left in at} == {max(p for p, _ in at)} == {start}, k


@NEEDS_A_RUN
def test_each_rank_looks_up_blocking_by_both_triggers(runs):
    """Each rank looks a request up with no prefetch to consume, and stops
    waiting for a deferred prefetch at least once."""
    dp, (run, _) = runs
    for rank in read(run["out"] / "parity"):
        lp = compass_run.LP_FILE.format("engine" if dp == 1 else f"engine.dp{rank}")
        n = json.loads((run["out"] / lp).read_text())["lookups"]
        print(f"\ndp{rank}: {n}")
        assert n["lookup_n"] > n["lookup_prefetched"], (rank, n)
        assert n["lookup_defer_expired"] > 0, (rank, n)


# --- (c): the parity record's offload decisions, without a driver -------------


class _Dispatched:
    """The base the runners below dispatch offload metadata to."""

    def process_kvconnector_output(self, meta):
        self.dispatched = meta


def test_a_step_line_carries_the_offload_decisions_dispatched_before_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(RECORD, str(tmp_path))
    monkeypatch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
    config = _parity_config()
    scheduler = Scheduler(config)
    runner = type("Sim", (_SimRunner, _Dispatched), {})(config)
    seq = Sequence(list(range(5, 25)), 16)  # one 20-token prompt, one step
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    meta = LMCacheOffloadMetadata()
    meta.add_request(
        LMCacheReqMeta(seq.id, list(range(20)), [0, 1], LoadSpec(0, 16), SaveSpec(16))
    )
    runner.process_kvconnector_output(meta)
    assert runner.dispatched is meta
    runner.forward(batch)
    runner.forward(batch)  # nothing dispatched before it
    key = _key(seq.prompt_token_ids)
    steps = read(tmp_path)[0]
    assert steps[0]["kv"] == [[key, "load", 0, 16], [key, "save", 16, 20]]
    assert "kv" not in steps[1]


def test_a_real_runner_keeps_what_it_dispatches_once_recording(monkeypatch):
    kept = []
    monkeypatch.setattr(parity, "record_kv", lambda runner, meta: kept.append(meta))
    runner = object.__new__(type("Real", (StepRecording, _Dispatched), {}))
    runner.process_kvconnector_output("warmup")
    runner._recording = True
    runner.process_kvconnector_output("meta")
    assert kept == ["meta"] and runner.dispatched == "meta"
