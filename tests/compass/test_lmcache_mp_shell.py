# SPDX-License-Identifier: MIT
"""The engine side of `lmcache_mp` on a simulated run, against LP `lmcache`.

The CPU tests build ATOM's scheduler half the way an engine process does,
inside `compass_run.engine`, and drive it on the step loop against the
stand-in server, started in a process of its own as the frontend starts it,
and an authority served here. Each reply is checked against the step that
reports it, from times derived by hand from the station laws.

The last test starts a whole simulated run, as `test_vertical_slice` does, and
needs a driver and `ATOM_COMPASS_SLICE_MODEL`.
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lmcache_lp import (
    BANDWIDTH,
    CHUNK,
    IPC,
    LATENCY_S,
    LOOKUP_S,
    OFFLOAD,
    REQUEST_S,
    TOKENS,
    _config,
    _run,
)
from test_memory_readings import DOCUMENT
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

from atom.compass import run as compass_run
from atom.compass.backends import KvGeometry, Parallelism
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.clock_transport import connect, serve
from atom.compass.kv import SimulatedKVConnector
from atom.compass.kv.lmcache_mp import LMCacheMPShell
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.mp import native_state_worker, transfer
from atom.utils import clock, forward_context
from atom.utils.distributed import utils as dist_utils

STEP = 0.01  # the step loop's clock calls, one per step
CHUNKS = len(TOKENS) // CHUNK


@pytest.fixture
def factory(monkeypatch):
    """The connector factory's `lmcache_mp` entry, as it was, after the test."""
    registry = KVConnectorFactory._registry
    monkeypatch.setitem(registry, "lmcache_mp", registry["lmcache_mp"])


def _kv(port) -> dict:
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": int(port)}
    return {**OFFLOAD, "kv_connector_extra_config": extra}


class Engine:
    """An engine process's scheduler half on the step loop, against LP lmcache
    and an authority served here; the traffic and frontend LPs idle."""

    def __init__(self, tmp_path, monkeypatch, tp=1, **entry) -> None:
        monkeypatch.setattr(clock, "_installed", None)
        monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
        self.run = run = _run(tmp_path, writers=tp, **entry)
        table = compass_run.channel_table(run)
        self.served = serve(
            ClockAuthority(table, bound_s=run["bound_s"]), "tcp://127.0.0.1:0"
        )
        run["clock_endpoint"] = self.served.endpoint
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        started, real = [], subprocess.Popen
        monkeypatch.setattr(
            compass_run.subprocess,
            "Popen",
            lambda *a, **k: started.append((a, real(*a, **k))) or started[-1][1],
        )
        port = run["lmcache"]["server"][-1]
        self.config = _config(
            _kv(port), tensor_parallel_size=tp, kv_cache_block_size=CHUNK
        )
        compass_run._launch_lmcache(self.config, run, None)
        (argv, *_), self.proc = started[0]
        self.chunk_bytes = int(argv[-1])
        self.idle = [
            connect(LpId(lp), self.served.endpoint) for lp in ("traffic", "frontend")
        ]
        for conn in self.idle:
            conn.send(("NER", math.inf, [], math.inf))
        with compass_run.engine(self.config):
            self.shell = KVConnectorFactory.create_connector(self.config, "scheduler")
            self.shell.bind_block_manager(SimpleNamespace(paged_state_checkpoints=None))
        self.rt = clock.installed()
        # What the shell hands ATOM's scheduler half at each step, when anything.
        self.reported = []
        impl = self.shell._inner._impl
        real = impl.process_completions

        def spy(output):
            done = [*output.finished_saving, *output.finished_loading]
            if done or output.failed_loading:
                self.reported.append((self.rt.now, done, list(output.failed_loading)))
            return real(output)

        monkeypatch.setattr(impl, "process_completions", spy)

    def build(self, *requests) -> None:
        meta = LMCacheOffloadMetadata()
        for request in requests:
            meta.add_request(request)
        self.shell._inner._impl.build_connector_meta = lambda: meta
        self.shell.build_connector_meta()

    def step(self) -> None:
        self.rt.advance_to(self.rt.now + STEP)
        self.shell.process_completions(KVConnectorOutput())

    def finish(self) -> list:
        """Idle to the run's end; LP lmcache's refusals."""
        try:
            while self.rt.next_event(math.inf) != math.inf:
                pass
            assert self.proc.wait(120) == 0
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.served.close()
        return json.loads(Path(self.run["out_dir"], "lp-lmcache.json").read_text())


def _save(req, generation=0):
    return LMCacheReqMeta(
        req,
        list(TOKENS),
        list(range(len(TOKENS) // CHUNK + 1)),
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(req, generation),
    )


def _at_step(t: float) -> float:
    """The first step at or after `t`."""
    return math.ceil(t / STEP - 1e-9) * STEP


def test_each_reply_is_reported_one_step_after_the_step_that_takes_it(
    tmp_path, monkeypatch, factory
):
    engine = Engine(tmp_path, monkeypatch)
    shell, rt = engine.shell, engine.rt
    assert isinstance(shell, LMCacheMPShell)
    # lmcache's chunk-size request went unstamped while the engine was built.
    assert shell._inner._impl.chunk_size == CHUNK
    copy = LATENCY_S + CHUNKS * engine.chunk_bytes / BANDWIDTH

    engine.build(_save("r0"))  # at 0
    stored = IPC + copy + IPC
    while rt.now < stored + 3 * STEP:
        engine.step()
    # A blocking lookup: LOOKUP, then QUERY_PREFETCH_STATUS, each a round trip
    # the step loop idles through.
    asked = rt.now
    assert shell._inner._impl._lookup_client.lookup(list(TOKENS), "r1") == 4 * CHUNK
    looked = asked + 2 * IPC + LOOKUP_S[0] + LOOKUP_S[1] * CHUNKS
    assert rt.now == pytest.approx(looked + 2 * IPC + REQUEST_S)
    # The same lookup sent ahead and consumed at once: its poll loop idles.
    client, asked = shell._inner._impl._lookup_client, rt.now
    assert client.submit(list(TOKENS), "r2")
    assert client.lookup(list(TOKENS), "r2") == 4 * CHUNK
    looked = asked + 2 * IPC + LOOKUP_S[0] + LOOKUP_S[1] * CHUNKS
    assert rt.now == pytest.approx(looked + 2 * IPC + REQUEST_S)
    load = LMCacheReqMeta(
        "r1",
        list(TOKENS),
        list(range(CHUNKS + 1)),
        load_spec=LoadSpec(0, CHUNKS * CHUNK, can_load=True),
        load_operation=LoadOperationId("r1", 0),
    )
    sent = rt.now
    engine.build(load)
    loaded = sent + IPC + copy + IPC
    while rt.now < loaded + 3 * STEP:
        engine.step()
    assert engine.finish() == {"refusals": []}

    print(f"\nstore reply at {stored!r}, retrieve reply at {loaded!r}")
    for row in engine.reported:
        print("reported", row)
    # Each is taken at the first step at or after its arrival, and reported at
    # the step after that one, as the engine's aggregation answers.
    assert engine.reported == [
        (pytest.approx(_at_step(stored) + STEP), [SaveOperationId("r0", 0)], []),
        (
            pytest.approx(sent + _at_step(loaded - sent) + STEP),
            [LoadOperationId("r1", 0)],
            [],
        ),
    ]


@pytest.mark.parametrize("width", [1, 2])
def test_two_writer_ranks_report_a_save_once_both_copies_end(
    tmp_path, monkeypatch, factory, width
):
    engine = Engine(tmp_path, monkeypatch, tp=2, copy_width=width)
    rt = engine.rt
    # Each TP rank of two stores its own shard of a chunk.
    whole = KvGeometry.from_hf_config(
        engine.config.hf_config,
        block_size=CHUNK,
        parallelism=Parallelism(tp_size=1),
        kv_dtype=engine.config.kv_cache_dtype,
    ).bytes_per_block
    assert engine.chunk_bytes * 2 == whole
    copy = LATENCY_S + CHUNKS * engine.chunk_bytes / BANDWIDTH

    engine.build(_save("r0"))
    # One copy server takes the second rank's store after the first's.
    last = IPC + copy * (3 - width) + IPC
    while rt.now < last + 3 * STEP:
        engine.step()
    assert engine.finish() == {"refusals": []}
    assert engine.reported == [
        (pytest.approx(_at_step(last) + STEP), [SaveOperationId("r0", 0)], [])
    ]


def test_a_rank_dates_its_transfer_on_the_lp_clock(monkeypatch):
    monkeypatch.setattr(clock, "_installed", SimpleNamespace(read_clock=lambda: 7.0))
    assert transfer._PendingSave(None, None, 0, CHUNK).started_at == 7.0
    assert transfer._PendingLoad(None, None).started_at == 7.0
    assert native_state_worker._NativePending(None, None).started_at == 7.0


@pytest.mark.parametrize("url", ["grpc://h:5555", "grpc+unix:///tmp/lmcache.sock"])
def test_a_server_url_lmcache_reaches_over_grpc_is_refused_by_its_scheme(url):
    kv = {**OFFLOAD, "kv_connector_extra_config": {"lmcache.mp.server_urls": url}}
    with pytest.raises(ValueError, match=f"{re.escape(url)}.* over gRPC"):
        LMCacheMPShell(_config(kv))


COMPASS = {"kv_connector": "compass", "kv_role": "kv_producer"}


@pytest.mark.parametrize(
    "kv",
    [OFFLOAD, {"kv_connector": "multi", "connectors": [COMPASS, OFFLOAD]}],
    ids=["lmcache_mp", "multi"],
)
def test_the_compass_runner_builds_lmcache_mp_as_a_worker_that_moves_nothing(
    kv, monkeypatch, factory
):
    built = []
    monkeypatch.setattr(
        forward_context,
        "set_kv_cache_data",
        lambda _, config, **k: built.append(
            KVConnectorFactory.create_connector(config, "worker")
        ),
    )
    runner = object.__new__(NonAllocatingRunner)
    runner.config = _config(kv)
    assert runner.allocate_kv_cache(100) is True
    [worker] = built
    halves = getattr(worker, "_connectors", [worker])
    assert [type(h) for h in halves] == [SimulatedKVConnector] * len(halves)
    assert worker.get_finished() in ((set(), set()), KVConnectorOutput())


def test_the_compass_runner_refuses_a_real_backend_beside_lmcache_mp(monkeypatch):
    monkeypatch.setattr(forward_context, "set_kv_cache_data", lambda *a, **k: None)
    runner = object.__new__(NonAllocatingRunner)
    multi = [{"kv_connector": "moriio", "kv_role": "kv_producer"}, OFFLOAD]
    runner.config = _config({"kv_connector": "multi", "connectors": multi})
    with pytest.raises(RunnerRefusal, match="kv_connector 'moriio' is a real"):
        runner.allocate_kv_cache(100)


# --- a whole run --------------------------------------------------------------

PROMPT = " ".join(f"word{k}" for k in range(600))  # several lmcache chunks
#: Each client run's request: sent at its start, or that many simulated seconds
#: after it. The second loads the first's prefix back from the server; the
#: third comes after the eviction loop has emptied it and the session sweep
#: has run past the session TTL.
LATE_S = 700.0


class Delayed(Traffic):
    """The slice's traffic LP, sending `PROMPT` `after` seconds into its run."""

    def __init__(self, after, *args) -> None:
        super().__init__(*args)
        self.after = after

    def run(self) -> None:
        self.rt.start_run()
        until = self.rt.now + self.after
        while self.rt.now < until:
            self.rt.next_event(until)
        super().run()

    def _carry(self, method, path, body, tracestate, port) -> None:
        body = body and {**body, "prompt": PROMPT}
        super()._carry(method, path, body, tracestate, port)


@NEEDS_A_RUN
def test_a_one_deployment_run_loads_a_stored_prefix_back(tmp_path):
    port, server_port = _free_port(), _free_port()
    machine = json.loads(json.dumps(DOCUMENT))
    machine["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": BANDWIDTH,
        "latency_s": LATENCY_S,
    }
    flags = ["--host", "127.0.0.1", "--port", str(server_port), "--chunk-size", "256"]
    flags += ["--l1-size-gb", "1", "--eviction-policy", "LRU"]
    flags += ["--eviction-trigger-watermark", "0"]  # every tick evicts
    run_file = _run_file(
        tmp_path,
        step_stub=CONSTANT,
        client_runs=3,
        bound_s=LATE_S + 60.0,
        machine=machine,
        lmcache={"server": flags, "lookup_s": list(LOOKUP_S), "request_s": REQUEST_S},
    )
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": server_port}
    env = dict(
        os.environ,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_KV_OFFLOAD="lmcache_mp",
        ATOM_KV_OFFLOAD_EXTRA_CONFIG=json.dumps(extra),
        OFFLOAD_MIN_LOAD_TOKENS="256",
        OFFLOAD_MIN_SAVE_TOKENS="256",
    )
    with open(tmp_path / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "atom.entrypoints.openai.api_server"]
            + ["--model", MODEL, "--host", "127.0.0.1", "--server-port", str(port)]
            + ["--enforce-eager", "--max-model-len", "2048"]
            + ["--no-enable_prefix_caching", "--compass-run", str(run_file)],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            for after in (0.0, 0.0, LATE_S):
                run = json.loads(run_file.read_text())
                traffic = Delayed(after, run, port, server)
                traffic.run()
                assert traffic.events[-1][1] == "data: [DONE]"
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    text = (tmp_path / "server.log").read_text()
    assert code == 0, f"the server exited {code}:\n{text[-4000:]}"
    replies = re.findall(r"compass: lmcache_mp (\w+) of \S+:(\S+) (\w+): (.*)", text)
    reports = re.findall(r"compass: lmcache_mp reports at (.*)", text)
    print("", *(" ".join(r) for r in replies), *reports, sep="\n")
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    assert summary["coverage_report"] is False
    stores = {r for op, r, ok, _ in replies if op == "store" and ok == "succeeded"}
    loads = [(r, ok) for op, r, ok, _ in replies if op == "retrieve"]
    # The second request loads the first's stored prefix; the third finds the
    # server emptied by its eviction loop and loads nothing.
    assert len(stores) >= 2 and len(loads) == 1 and loads[0][1] == "succeeded"
    assert any(f"loaded ['LoadOperationId(req_id={loads[0][0]!r}" in r for r in reports)
    table = (tmp_path / compass_run.STEP_TABLE_FILE).read_text()
    ticks = [
        float(r.split()[2]) for r in table.splitlines()[1:] if r.startswith("lmcache ")
    ]
    assert {1.0, 60.0, 660.0} <= set(ticks), "the eviction and session ticks ran"
