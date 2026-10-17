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
import test_vertical_slice
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
from atom.compass.kv import WRITE_REQ_KEY, SimulatedKVConnector
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
COMPASS = {"kv_connector": "compass", "kv_role": "kv_producer"}


@pytest.fixture
def factory(monkeypatch):
    """The connector factory's `lmcache_mp` and `multi` entries, as they were,
    after the test."""
    registry = KVConnectorFactory._registry
    for name in ("lmcache_mp", "multi"):
        monkeypatch.setitem(registry, name, registry[name])


def _kv(port) -> dict:
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": int(port)}
    return {**OFFLOAD, "kv_connector_extra_config": extra}


class Engine:
    """An engine process's scheduler half on the step loop, against LP lmcache
    and an authority served here; every other LP idles. With `pd`, the engine
    is the prefill side of a prefill-decode run, its connector a `multi` of
    `compass` and `lmcache_mp`."""

    def __init__(self, tmp_path, monkeypatch, tp=1, pd=False, **entry) -> None:
        monkeypatch.setattr(clock, "_installed", None)
        monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
        self.run = run = _run(tmp_path, writers=tp, **entry)
        if pd:
            run |= {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10}
            run["kv_link"] = "inter_node"
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
        kv = _kv(port)
        if pd:
            write = {WRITE_REQ_KEY: f"inproc://kv-write-req-{os.getpid()}"}
            kv = {"kv_connector": "multi", "connectors": [COMPASS | write, kv]}
        self.config = _config(kv, tensor_parallel_size=tp, kv_cache_block_size=CHUNK)
        compass_run._launch_lmcache(self.config, run, compass_run._pd_side(self.config))
        (argv, *_), self.proc = started[0]
        self.chunk_bytes = int(argv[-1])
        me = "engine-P" if pd else "engine"
        self.idle = [
            connect(lp, self.served.endpoint)
            for lp in table.registry.ids()
            if lp.name not in (me, "lmcache")
        ]
        for conn in self.idle:
            conn.send(("NER", math.inf, [], math.inf))
        with compass_run.engine(self.config):
            self.shell = KVConnectorFactory.create_connector(self.config, "scheduler")
            self.shell.bind_block_manager(SimpleNamespace(paged_state_checkpoints=None))
        self.rt = clock.installed()
        subs = getattr(self.shell, "_connectors", [self.shell])
        [self.lm] = [s for s in subs if isinstance(s, LMCacheMPShell)]
        # What the shell hands ATOM's scheduler half at each step, when anything.
        self.reported = []
        impl = self.lm._inner._impl
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
        self.lm._inner._impl.build_connector_meta = lambda: meta
        self.lm.build_connector_meta()

    def step(self) -> KVConnectorOutput:
        self.rt.advance_to(self.rt.now + STEP)
        return self.shell.process_completions(KVConnectorOutput())

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
    busy = []
    while rt.now < stored + 3 * STEP:
        engine.step()
        busy.append((rt.now, shell.has_pending_work()))
    # The engine keeps stepping from the send to the step that reports the
    # reply, though ATOM's scheduler half was handed no save to wait for.
    reported = _at_step(stored) + STEP
    assert [t for t, b in busy if b] == [t for t, _ in busy if t < reported - 1e-9]
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


def test_a_multi_beside_the_prefill_connector_reports_its_writes_and_saves(
    tmp_path, monkeypatch, factory
):
    engine = Engine(tmp_path, monkeypatch, pd=True)
    rt = engine.rt
    assert rt.me == LpId("engine-P")
    [pd, _] = engine.shell._connectors
    copy = LATENCY_S + CHUNKS * engine.chunk_bytes / BANDWIDTH
    pd._due["w0"] = 0.0  # a write to decode, done at 0
    engine.build(_save("r0"))
    stored = IPC + copy + IPC
    sent = []
    while rt.now < stored + 3 * STEP:
        if (out := engine.step()).finished_sending:
            sent.append((rt.now, out.finished_sending))
    try:
        assert engine.finish() == {"refusals": []}
    finally:
        for sock in pd._socks.values():
            sock.raw.close(linger=0)
    # Both one call late: the write's end seen at the first step, the store's
    # reply taken at the first step at or after its arrival.
    assert sent == [(pytest.approx(2 * STEP), {"w0"})]
    assert engine.reported == [
        (pytest.approx(_at_step(stored) + STEP), [SaveOperationId("r0", 0)], [])
    ]


def test_a_released_reply_still_on_the_wire_is_waited_for(
    tmp_path, monkeypatch, factory
):
    import zmq

    engine = Engine(tmp_path, monkeypatch)
    rt = engine.rt
    copy = LATENCY_S + CHUNKS * engine.chunk_bytes / BANDWIDTH
    engine.build(_save("r0"))
    [(back, sock)] = engine.lm._socks.items()

    class OnTheWire:
        """The writer's socket, its released reply unread until it is polled."""

        polled = False

        def recv_multipart(self, flags=0):
            if not self.polled and rt.released[back] - rt.arrived[back]:
                raise zmq.Again
            return sock.recv_multipart(flags)

        def poll(self, *args):
            self.polled = True
            return sock.poll(*args)

    engine.lm._socks[back] = OnTheWire()
    stored = IPC + copy + IPC
    while rt.now < stored + 3 * STEP:
        engine.step()
    assert engine.finish() == {"refusals": []}
    assert engine.reported == [
        (pytest.approx(_at_step(stored) + STEP), [SaveOperationId("r0", 0)], [])
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

#: The machine the run is priced on and its memory readings measured on: the
#: spec of the device `ATOM_COMPASS_SLICE_MACHINE` names, with a host link for
#: the server's copies and the suite's tokenizer entry for Qwen3.
MACHINE = Path(
    os.environ.get(
        "ATOM_COMPASS_SLICE_MACHINE",
        TREE / "scripts/compass/machines/mi308x-hjbog39.json",
    )
)
PROMPT = " ".join(f"word{k}" for k in range(200))  # a few lmcache chunks
#: The third client run's request comes this long after its start: the
#: eviction loop has emptied the server, and the session sweep has run.
LATE_S = 700.0
ATOM_ARGS = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]


class Delayed(Traffic):
    """The slice's traffic LP, sending `PROMPT` `after` seconds into its run,
    and scraping nothing."""

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

    def _scrape(self) -> None:
        pass


def _machine(out: Path) -> Path:
    machine = json.loads(MACHINE.read_text())
    machine["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": BANDWIDTH,
        "latency_s": LATENCY_S,
    }
    machine["host"]["tokenizers"].append(DOCUMENT["host"]["tokenizers"][0])
    origins = machine["provenance"]["origins"]
    for field in ("bandwidth_bytes_per_s", "latency_s"):
        origins[f"interconnect.host_link.{field}"] = {"origin": "assumed"}
    origins["host.tokenizers[qwen3-151k-bpe fast]"] = {"origin": "assumed"}
    (out / "machine.json").write_text(json.dumps(machine))
    return out / "machine.json"


@NEEDS_A_RUN
def test_a_one_deployment_run_loads_a_stored_prefix_back(tmp_path, monkeypatch):
    """Three client runs of one prompt on a deployment that offloads to LP
    lmcache, its memory readings measured first on the GPU
    ``HIP_VISIBLE_DEVICES`` names: the first stores the prompt's chunks, the
    second loads them back, prefix caching being off, and the third, past the
    eviction loop and a session sweep, finds the server empty."""
    from atom.compass import cli

    monkeypatch.setattr(test_vertical_slice, "SCRAPE_S", math.inf)
    # The readings are keyed by the performance environment the server runs in.
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    machine = _machine(tmp_path)
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    cli.main(
        ["measure", "memory", "--gpu", gpu, "--machine", str(machine)]
        + ["--store", str(tmp_path / "store"), "--out", str(tmp_path / "measure")]
        + ["--", *ATOM_ARGS]
    )
    port, server_port = _free_port(), _free_port()
    flags = ["--host", "127.0.0.1", "--port", str(server_port), "--chunk-size", "256"]
    flags += ["--l1-size-gb", "1", "--eviction-policy", "LRU"]
    flags += ["--eviction-trigger-watermark", "0"]  # every tick evicts
    run_file = _run_file(
        tmp_path,
        step_stub=CONSTANT,
        client_runs=3,
        bound_s=LATE_S + 60.0,
        machine=json.loads(machine.read_text()),
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
            [sys.executable, "-m", "atom.entrypoints.openai.api_server", *ATOM_ARGS]
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
    stores = [r for op, r, ok, _ in replies if op == "store" and ok == "succeeded"]
    loads = [(r, ok) for op, r, ok, _ in replies if op == "retrieve"]
    # The second request loads the first's stored prefix; the third finds the
    # server emptied by its eviction loop and loads nothing.
    assert len(set(stores)) >= 2 and len(loads) == 1 and loads[0][1] == "succeeded"
    [loaded] = [r for r in reports if "loaded []" not in r]
    assert "LoadOperationId" in loaded and loads[0][0] in loaded
    table = (tmp_path / compass_run.STEP_TABLE_FILE).read_text()
    ticks = {float(r.split()[2]) for r in table.splitlines()[1:] if r[:8] == "lmcache "}
    assert {1.0, 60.0, 660.0} <= ticks, "the eviction and session ticks ran"
