# SPDX-License-Identifier: MIT
"""One request end to end through ATOM's real serving stack on the simulated clock.

A run is ATOM's API server started with ``--compass-run``: its engine core and
TP1 worker processes, the Compass runner priced by `ShapeStubBackend`, and
the Clock Authority co-hosted in the API server. The traffic LP is `Traffic`
below, in this process. The run is made twice, in two process trees with
different `PYTHONHASHSEED`, and the two step tables the authority writes are
compared byte for byte.

The engine core and the worker import aiter, and the worker's construction
still touches the device, so those tests skip where there is no driver. They
also skip without `ATOM_COMPASS_SLICE_MODEL`, a model directory holding a
config and a tokenizer; no weight is read.
"""

import dataclasses
import http.client
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_memory_readings import DOCUMENT, TOKENIZER

from atom.compass import run as compass_run
from atom.compass.backends.shape import Coefficients
from atom.compass.carriers import sse_stamp, tracestate_with
from atom.compass.clock import LpId
from atom.compass.clock_transport import connect
from atom.compass.detect.determinism import CONFIGURATION_PREFIX, compare_step_tables
from atom.utils import clock
from atom.utils.clock import LPRuntime

MODEL = os.environ.get("ATOM_COMPASS_SLICE_MODEL")
NEEDS_A_RUN = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="a run needs a driver, since the engine core and the TP1 worker import "
    "aiter, and ATOM_COMPASS_SLICE_MODEL naming a model directory",
)

TREE = Path(__file__).resolve().parents[2]
TRAFFIC = LpId("traffic")
HTTP, STREAM = "traffic->frontend:http", "frontend->traffic:stream"
SCRAPE_S = 16.0
WALL_S = 600.0
MAX_TOKENS = 4


class UnansweredRequests(AssertionError):
    """The run finished with requests that never had their final response."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _retry(what, server, attempt):
    deadline = time.monotonic() + WALL_S
    while True:
        try:
            return attempt()
        except ConnectionRefusedError:
            assert server.poll() is None, f"the server exited before {what}"
            assert time.monotonic() < deadline, f"no {what} in {WALL_S} wall seconds"
            time.sleep(0.05)


class Traffic:
    """The traffic LP: one streamed completion at 0, then one scrape once it is
    answered, and a periodic scrape on a daemon deadline.

    The owner stamps every send and takes each stream event the authority
    releases; threads only carry bytes. An event is checked against the clock
    as it is read, and taken once released. Requests go to `port` and scrapes
    to `scrape_port`, by default the same.
    """

    http, stream = HTTP, STREAM

    def __init__(self, run: dict, port: int, server, scrape_port=None) -> None:
        bound = run["bound_s"]
        if not math.isfinite(bound):
            raise ValueError(
                f"the run file's bound_s is {bound}; a run needs a finite one"
            )
        self.bound = (bound, "run file bound_s")
        self.port, self.server = port, server
        self.scrape_port = scrape_port or port
        endpoint = run["clock_endpoint"]
        conn = _retry("clock", server, lambda: connect(TRAFFIC, endpoint))
        self.rt = LPRuntime(TRAFFIC, compass_run.channel_table(run), conn)
        self.cv = threading.Condition()
        self.frames: dict[int, tuple[float, str]] = {}  # read, not yet taken
        self.events: list[tuple[float, str]] = []  # taken, in order
        self.errors: list[BaseException] = []
        self.open: set[int] = set()

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        self.sent, seq = rt.stamp_send(self.http)
        self.open.add(0)
        body = {
            "model": MODEL,
            "prompt": "one request through the slice",
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        self._carry(
            "POST",
            "/v1/completions",
            body,
            tracestate_with(None, self.sent, seq),
            self.port,
        )
        next_scrape, scraped = SCRAPE_S, False
        while rt.next_event(math.inf, next_scrape) != math.inf:
            self._take()
            if rt.now >= next_scrape:
                next_scrape += SCRAPE_S
                self._scrape()
            if not self.open and not scraped:
                scraped = True
                self._scrape()
        rt.close()
        if self.open:
            raise UnansweredRequests(
                f"requests {sorted(self.open)} were sent and never given their final "
                "response"
            )

    def _scrape(self) -> None:
        arrival, seq = self.rt.stamp_send(self.http)
        self._carry(
            "GET",
            "/metrics",
            None,
            tracestate_with(None, arrival, seq),
            self.scrape_port,
        )

    def _carry(self, method, path, body, tracestate, port) -> None:
        threading.Thread(
            target=self._read,
            args=(method, path, body, tracestate, port),
            daemon=True,
        ).start()

    def _read(self, method, path, body, tracestate, port) -> None:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WALL_S)
            _retry("HTTP listener", self.server, conn.connect)
            headers = {"tracestate": tracestate, "content-type": "application/json"}
            conn.request(method, path, body and json.dumps(body), headers)
            stamp = None
            for raw in conn.getresponse():
                line = raw.decode().rstrip("\r\n")
                stamp = sse_stamp(line) or stamp
                if line.startswith("data:") and stamp is not None:
                    arrival, seq = stamp
                    self.rt.check_arrival(self.stream, arrival, seq)
                    with self.cv:
                        self.frames[seq] = (arrival, line)
                        self.cv.notify_all()
                    stamp = None
        except BaseException as e:  # noqa: BLE001 - the owner raises it
            with self.cv:
                self.errors.append(e)
                self.cv.notify_all()

    def _take(self) -> None:
        """Take every released stream event, waiting for any not read yet."""
        with self.rt.lock:
            due = sorted(self.rt.released[self.stream] - self.rt.handled[self.stream])
        for seq in due:
            with self.cv:
                arrived = self.cv.wait_for(
                    lambda seq=seq: seq in self.frames or self.errors, WALL_S
                )
                if self.errors:
                    raise self.errors[0]
                assert arrived, f"stream seq {seq} released and not read"
                self.events.append(self.frames.pop(seq))
            if self.events[-1][1] == "data: [DONE]":
                self.open.discard(0)
            with self.rt.lock:
                self.rt.count_done_locked(self.stream, seq)


def _run_file(out: Path, **overrides) -> Path:
    run = {
        "clock_endpoint": f"tcp://127.0.0.1:{_free_port()}",
        "bound_s": 600.0,
        "admission_path": "serving",
        "ipc_s": 2.0**-14,
        "stream_s": 2.0**-12,
        "coefficients": dataclasses.asdict(Coefficients()),
        "machine": DOCUMENT,
        "parameter_count": 8_000_000_000,
        "out_dir": str(out),
    } | overrides
    (out / "run.json").write_text(json.dumps(run))
    return out / "run.json"


def run_slice(out: Path, seed: str) -> dict:
    """One run in its own process tree; its step table, summary and request times."""
    out.mkdir()
    run_file = _run_file(out)
    port = _free_port()
    env = dict(
        os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(TREE), AITER_LOG_LEVEL="WARNING"
    )
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "atom.entrypoints.openai.api_server",
                "--model",
                MODEL,
                "--host",
                "127.0.0.1",
                "--server-port",
                str(port),
                "--enforce-eager",
                "--max-model-len",
                "2048",
                "--compass-run",
                str(run_file),
            ],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        # A run that hangs ends here: the clock's socket closes with the server.
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            traffic = Traffic(json.loads(run_file.read_text()), port, server)
            traffic.run()
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    tail = (out / "server.log").read_text()[-4000:]
    assert code == 0, f"the server exited {code} after the finish:\n{tail}"
    return {
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "request": (traffic.sent, traffic.events[0][0], traffic.events[-1][0]),
        "events": traffic.events,
        "bound": traffic.bound,
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("slice")
    return [run_slice(root / f"seed{seed}", seed) for seed in ("1", "2")]


@NEEDS_A_RUN
def test_one_request_completes_and_the_run_ends_by_the_finish(runs):
    run = runs[0]
    arrive, first_token, leave = run["request"]
    print(f"\nrequest 0: arrive {arrive!r} first token {first_token!r} leave {leave!r}")
    print(run["table"])
    assert run["events"][-1][1] == "data: [DONE]"
    assert len(run["events"]) > 1
    assert 0 < arrive < first_token <= leave
    finish = [r for r in run["table"].splitlines()[1:] if " inf " in r]
    assert sorted(r.split()[0] for r in finish) == ["engine", "frontend", "traffic"]
    assert run["table"].splitlines()[-3:] == finish
    summary = run["summary"]
    assert summary["schedule"]["grants_total"] > 0
    assert summary["schedule"]["refusals"]["count"] == 0
    assert summary["coverage_report"] is False
    assert 0 < summary["cost"]["wall_seconds"]
    assert run["bound"] == (600.0, "run file bound_s")


@NEEDS_A_RUN
def test_two_hash_seeds_give_byte_identical_step_tables(runs):
    left, right = (r["table"] for r in runs)
    code, report = compare_step_tables(left, right, "seed 1", "seed 2")
    print(f"\n{report}")
    assert code == 0, report
    assert left == right


def _release(table: str, lp: str, channel: str, seq: int) -> float:
    """The time `lp` released message `seq` on `channel`, from a step table."""
    for row in table.splitlines()[1:]:
        f = row.split()
        if f[0] == lp and f[3] == "release" and f[4] == channel and f[5] == str(seq):
            return float(f[2])
    raise AssertionError(f"{lp} released no {channel} {seq}")


@NEEDS_A_RUN
def test_the_served_tokenizer_charges_the_request_its_encode_and_decode(runs):
    table = runs[0]["table"]
    run = json.loads(table.splitlines()[0].removeprefix(CONFIGURATION_PREFIX))
    # The request leaves for the engine once its prompt is encoded.
    received = _release(table, "frontend", HTTP, 0)
    sent = _release(table, "engine", "frontend->engine:request#dp0", 0)
    encode = sent - received - run["ipc_s"]
    # The first token leaves for the traffic LP once the frame carrying it is decoded.
    output = _release(table, "frontend", "engine->frontend:output#dp0", 1)
    decode = _release(table, "traffic", STREAM, 0) - output - run["stream_s"]
    print(f"\nrequest 0: encode {encode!r} s, first frame's decode {decode!r} s")
    assert encode >= TOKENIZER["encode_fixed_s"]
    assert decode >= TOKENIZER["decode_fixed_s"]


# --- without a driver: the bootstrap off, and its refusals -------------------


def _config(dp=1, pp=1, runner=compass_run.ATOM_RUNNER):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        pipeline_parallel_size=pp,
        enable_rapidserve=False,
        runner_qualname=runner,
    )


def test_with_no_run_file_nothing_is_installed(monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    config, model_runner = _config(), SimpleNamespace()
    with compass_run.frontend(config), compass_run.engine(config):
        compass_run.runner(model_runner)
    assert clock.installed() is None
    assert config.runner_qualname == compass_run.ATOM_RUNNER
    assert vars(model_runner) == {}


def test_a_run_with_no_finite_bound_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, bound_s=None)))
    with pytest.raises(ValueError, match="bound_s is None; a run needs a finite"):
        compass_run.frontend(_config())


@pytest.mark.parametrize(
    "config, refusal",
    [
        (_config(pp=2), r"got \(dp, pp\)=\(1, 2\)"),
        (_config(runner="my.Runner"), "runner 'my.Runner' is named"),
    ],
)
def test_a_deployment_the_slice_does_not_run_is_refused(
    monkeypatch, tmp_path, config, refusal
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    with pytest.raises(ValueError, match=refusal):
        compass_run.frontend(config)
    assert clock.installed() is None
