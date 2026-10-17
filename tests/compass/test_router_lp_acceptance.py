# SPDX-License-Identifier: MIT
"""The router LP at 1P1D DP2, with no driver: atomesh, built from this tree, run
as `pd_sim.sh` runs the agentic cell's router (``--dp-aware``, dp_sticky), under
the standalone clock authority, between scripted frontends.

Each frontend is its LP and answers the router at a constant cost: prefill
sends its relay `PREFILL_S` after the request reaches it, decode one stream
event per token, `DECODE_S` apart. The engines are idle LPs. The run is made
twice, the second under another `PYTHONHASHSEED` and with a random wall delay
on every hop into and out of the router: before each send by the traffic LP and
the frontends, and before a frontend reads what the router posted.
`check_run` holds the two to the scenario checks, and each request's (prefill,
decode) rank must be the least-load choice worked out by hand in `RANKS`. The
same pair of runs under ``--policy random`` (as `pd_sim.sh` runs a cell that
maps ranks outside aiperf_agentic) must pick the same ranks, and
``--policy power_of_two`` must be refused by name.

Skips where cargo cannot resolve atomesh's dependencies, as
``test_mesh_clock_client.py`` does.
"""

import heapq
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_pd_slice import PD
from test_vertical_slice import CONSTANT, TREE, _free_port, _retry, _run_file

from atom.compass import run as compass_run
from atom.compass.carriers import (
    relay_stamp,
    stamp_events,
    tracestate_stamp,
    with_relay_stamp,
)
from atom.compass.clock import LpId
from atom.compass.clock_transport import connect
from atom.compass.detect.scenario import check_run
from atom.compass.parity.traffic import Traffic
from atom.utils.clock import LPRuntime, channel_of

MESH = TREE / "atom" / "mesh"
WALL_S = 120.0
U = 2.0**-6
PREFILL_S, DECODE_S = 4 * U, U
# Each request's send time on the traffic clock, and its max_tokens.
WORKLOAD = [(0.0, 6), (5.5 * U, 8), (6 * U, 1), (11.5 * U, 2)]
# Least load, ties to rank 0. Prefill load is held from placement until the
# relay is handled, 4.12U after the send; decode load until the stream's last
# event is, 0.02U plus U per token later.
#   0: nothing held                                         -> (0, 0)
#   1: p0 freed at 4.12U; d0 held by 0, streaming since 5.14U -> (0, 1)
#   2: p0 held by 1; d0 by 0, d1 by 1                       -> (1, 0)
#   3: p0 freed at 9.62U, p1 at 10.12U; d0 freed at 10.14U
#      and 11.14U, d1 held by 1                             -> (0, 0)
# A missed prefill drop sends 1 to p1, a missed decode drop 3 to d1, and a
# decode drop at the stream's first event 1 to d0.
RANKS = [(0, 0), (0, 1), (1, 0), (0, 0)]


@pytest.fixture(scope="module")
def atomesh():
    resolved = shutil.which("cargo") and subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=MESH,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if not resolved or resolved.returncode:
        pytest.skip("cargo cannot resolve atomesh's dependencies here")
    subprocess.run(
        ["cargo", "build", "--bin", "atomesh"], cwd=MESH, check=True, timeout=1800
    )
    return Path(os.environ.get("CARGO_TARGET_DIR", MESH / "target")) / "debug/atomesh"


class Frontend:
    """Frontend LP ``frontend-<side>``: an HTTP server the router posts to, and
    the LP's clock owner, which replies to each request once it is released."""

    def __init__(self, run: dict, side: str, delay) -> None:
        self.run, self.side, self.delay = run, side, delay
        self.lp = LpId(f"frontend-{side}")
        self.cv = threading.Condition()
        self.bodies: dict[int, dict] = {}  # inbound seq -> its request body
        self.replies: dict[int, list] = {}  # inbound seq -> its reply's parts
        self.ranks: dict[int, int] = {}  # request -> the DP rank it was sent to
        self.errors: list[BaseException] = []
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.owner = threading.Thread(target=self._own, daemon=True)
        self.owner.start()

    def _handler(self):
        fe = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # the router's worker discovery
                role = "kv_producer" if fe.side == "P" else "kv_consumer"
                info = {"dp_size": 2, "tp_size": 1, "kv_role": role}
                info |= {"model_id": "m", "data": [{"id": "m"}]}
                self._send(json.dumps(info).encode(), "application/json")

            def do_POST(self) -> None:
                time.sleep(fe.delay())
                raw = self.rfile.read(int(self.headers["content-length"]))
                body = json.loads(raw)
                stamp = (
                    tracestate_stamp(self.headers["tracestate"])
                    if fe.side == "P"
                    else relay_stamp(raw)
                )
                seq = stamp[1]
                with fe.cv:
                    request = int(re.match(r"request (\d+)", body["prompt"])[1])
                    fe.ranks[request] = body["data_parallel_rank"]
                    fe.bodies[seq] = body
                    fe.cv.notify_all()
                if fe.side == "P":
                    self._send(fe.part(seq, 0).encode(), "application/json")
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for k in range(body["max_tokens"]):
                    self.wfile.write(fe.part(seq, k).encode())
                    self.wfile.flush()

            def _send(self, data: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def part(self, seq: int, k: int) -> str:
        """Part `k` of the reply to `seq`, once the owner has made it, after
        the injected delay."""
        with self.cv:
            self.cv.wait_for(lambda: len(self.replies.get(seq, ())) > k, WALL_S)
            text = self.replies[seq][k]
        time.sleep(self.delay())
        return text

    def _own(self) -> None:
        try:
            run = self.run
            conn = connect(self.lp, run["clock_endpoint"])
            rt = LPRuntime(self.lp, compass_run.channel_table(run), conn)
            inbound = channel_of(rt, "http" if self.side == "P" else "relay")
            out = channel_of(rt, "relay" if self.side == "P" else "stream")
            rt.start_run()
            due = []  # (time, inbound seq, part) of each send not made yet
            while (G := rt.next_event(due[0][0] if due else math.inf)) != math.inf:
                for seq in sorted(rt.released[inbound] - rt.handled[inbound]):
                    with self.cv:
                        assert self.cv.wait_for(lambda s=seq: s in self.bodies, WALL_S)
                        self.replies[seq] = []
                    with rt.lock:
                        rt.count_done_locked(inbound, seq)
                    n = 1 if self.side == "P" else self.bodies[seq]["max_tokens"]
                    for k in range(n):
                        cost = PREFILL_S if self.side == "P" else (k + 1) * DECODE_S
                        heapq.heappush(due, (G + cost, seq, k))
                while due and due[0][0] <= rt.now:
                    _, seq, k = heapq.heappop(due)
                    text = self._reply(rt, out, self.bodies[seq], k)
                    with self.cv:
                        self.replies[seq].append(text)
                        self.cv.notify_all()
            rt.close()
            compass_run._hand_in(run, self.lp.name, [])
        except BaseException as e:  # noqa: BLE001 - the test raises it
            self.errors.append(e)

    def _reply(self, rt, out: str, body: dict, k: int) -> str:
        if self.side == "P":
            params = {"kv_transfer_params": {"dp_rank": body["data_parallel_rank"]}}
            raw = json.dumps(params).encode()
            return with_relay_stamp(raw, lambda: rt.stamp_send(out)).decode()
        n = body["max_tokens"]
        events = [{"choices": [{"index": 0, "text": "x"}]}]
        if k == n - 1:
            events.append({"choices": [], "usage": {"completion_tokens": n}})
        text = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
        return stamp_events(text + "data: [DONE]\n\n" * (k == n - 1), rt)


def _idle(run: dict, lp: str, member: str, errors: list) -> None:
    """A member of an engine LP that is never sent anything."""
    try:
        conn = connect(LpId(lp), run["clock_endpoint"], member)
        rt = LPRuntime(LpId(lp), compass_run.channel_table(run), conn)
        rt.start_run()
        while rt.next_event(math.inf) != math.inf:
            pass
        compass_run._hand_in(run, f"{lp}.{member}", [])
    except BaseException as e:  # noqa: BLE001 - the test raises it
        errors.append(e)


class DelayedTraffic(Traffic):
    """`Traffic`, sleeping `delay()` wall seconds before each send."""

    def __init__(self, *args, delay) -> None:
        super().__init__(*args)
        self.delay = delay

    def _read(self, body: dict, tracestate: str) -> None:
        time.sleep(self.delay())
        super()._read(body, tracestate)


def _launch(out: Path, name: str, args: list, **env) -> subprocess.Popen:
    with open(out / f"{name}.log", "w") as log:
        return subprocess.Popen(
            args, env=dict(os.environ, **env), stdout=log, stderr=subprocess.STDOUT
        )


def run_router_lp(
    atomesh: Path, out: Path, hash_seed: str, delay, policy: str = "dp_sticky"
) -> list:
    """One run in `out`; each request's (prefill, decode) rank, in order."""
    out.mkdir()
    run_file = _run_file(
        out, data_parallel_size=2, step_stub=CONSTANT, router_in_s=2.0**-14, **PD
    )
    run = json.loads(run_file.read_text())
    table = compass_run.channel_table(run)
    lookaheads = {
        c.name: table.lookahead(c.name) for c in table.channels_from(compass_run.ROUTER)
    }
    endpoint = run["clock_endpoint"]
    errors, procs, frontends = [], [], {}
    try:
        authority = _launch(
            out,
            "authority",
            [sys.executable, "-m", "atom.compass.run", "--compass-run", str(run_file)]
            + ["--compass-clock-endpoint", endpoint],
            PYTHONHASHSEED=hash_seed,
            PYTHONPATH=str(TREE),
        )
        procs.append(authority)
        host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
        _retry(
            "clock", authority, lambda: socket.create_connection((host, port)).close()
        )
        frontends = {s: Frontend(run, s, delay) for s in "PD"}
        idle = [
            threading.Thread(
                target=_idle, args=(run, f"engine-{s}", f"dp{r}", errors), daemon=True
            )
            for s in "PD"
            for r in range(2)
        ]
        for t in idle:
            t.start()
        router_port = _free_port()
        router = _launch(
            out,
            "router",
            [str(atomesh), "launch", "--host", "127.0.0.1", "--port", str(router_port)]
            + ["--pd-disaggregation", "--prefill", frontends["P"].url]
            + ["--decode", frontends["D"].url, "--policy", policy, "--dp-aware"]
            + ["--backend", "atom", "--disable-health-check"]
            + ["--disable-circuit-breaker", "--disable-retries"]
            + ["--prometheus-port", str(_free_port()), "--log-level", "info"],
            ATOM_COMPASS_CLOCK_ENDPOINT=endpoint,
            ATOM_COMPASS_SEED="20260707",
            ATOM_COMPASS_ROUTER_LOOKAHEADS=json.dumps(lookaheads),
        )
        procs.append(router)
        deadline = time.monotonic() + WALL_S
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{router_port}/v1/models")
                break
            except OSError:
                assert router.poll() is None, (out / "router.log").read_text()
                assert time.monotonic() < deadline, f"no router in {WALL_S} s"
                time.sleep(0.5)
        bodies = [
            {"model": "m", "prompt": f"request {i}", "max_tokens": n, "stream": True}
            for i, (_, n) in enumerate(WORKLOAD)
        ]
        offsets = [t for t, _ in WORKLOAD]
        DelayedTraffic(run, router_port, bodies, offsets, delay=delay).run()
        for t in [*idle, *(f.owner for f in frontends.values())]:
            t.join(WALL_S)
        assert authority.wait(WALL_S) == 0, (out / "authority.log").read_text()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for f in frontends.values():
            f.http.shutdown()
    errors += [e for f in frontends.values() for e in f.errors]
    if errors:
        raise errors[0]
    p, d = (frontends[s].ranks for s in "PD")
    return [(p[i], d[i]) for i in range(len(WORKLOAD))]


@pytest.fixture(scope="module")
def runs(atomesh, tmp_path_factory):
    root = tmp_path_factory.mktemp("router_lp")
    wall = random.Random(1)
    delays = {"1": lambda: 0.0, "2": lambda: wall.uniform(0.0, 0.02)}
    return {
        root / f"seed{s}": run_router_lp(atomesh, root / f"seed{s}", s, delay)
        for s, delay in delays.items()
    }


def test_both_runs_meet_every_scenario_check(runs):
    failed = check_run(list(runs))
    assert not failed, "\n".join(failed)


def test_each_request_takes_the_least_load_ranks_under_any_host_delay(runs):
    print("\n(prefill, decode) rank per request, per run:", list(runs.values()))
    assert list(runs.values()) == [RANKS, RANKS]


def test_random_picks_the_same_ranks_from_one_seed_under_any_host_delay(
    atomesh, tmp_path
):
    wall = random.Random(2)
    first = run_router_lp(atomesh, tmp_path / "a", "1", lambda: 0.0, "random")
    second = run_router_lp(
        atomesh, tmp_path / "b", "2", lambda: wall.uniform(0.0, 0.02), "random"
    )
    print("\nrandom: (prefill, decode) rank per request, per run:", [first, second])
    assert first == second


def test_power_of_two_is_refused_by_name(atomesh, tmp_path):
    # A clock that accepts the bind and never answers: the refusal comes first.
    clock = socket.create_server(("127.0.0.1", 0))
    router = _launch(
        tmp_path,
        "router",
        [str(atomesh), "launch", "--host", "127.0.0.1", "--port", str(_free_port())]
        + ["--pd-disaggregation", "--prefill", "http://127.0.0.1:9"]
        + ["--decode", "http://127.0.0.1:9", "--policy", "power_of_two", "--dp-aware"]
        + ["--backend", "atom", "--prometheus-port", str(_free_port())],
        ATOM_COMPASS_CLOCK_ENDPOINT=f"tcp://127.0.0.1:{clock.getsockname()[1]}",
        ATOM_COMPASS_SEED="1",
        ATOM_COMPASS_ROUTER_LOOKAHEADS="{}",
    )
    try:
        code = router.wait(30)
    except subprocess.TimeoutExpired:
        router.kill()
        code = "still running after 30 s"
    finally:
        clock.close()
    log = (tmp_path / "router.log").read_text()
    assert code == 1, log
    assert "policy power_of_two is refused" in log
