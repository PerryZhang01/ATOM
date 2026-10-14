# SPDX-License-Identifier: MIT
"""A simulated run that replays a real record's step times, and the compare that reads both.

ATOM's `Scheduler` drives the simulated runner's `forward` on one DP rank, as in
`test_parity_record.py`; a "real" record is that run with entry and exit
times written in, and the replay is the same run priced by `ReplayBackend`.
"""

import http.server
import json
import queue
import threading
import time

import pytest
from test_parity_record import BLOCK, _config, _prompt, _Runner
from test_run_step_grid import B37, decode, deployment
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.parity import ARRIVALS, ENV, compare, traffic
from atom.compass.parity.replay import ReplayBackend
from atom.compass.runner.overrides import install_cost_backend
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

#: Worker-clock nanoseconds: the first send, and step i's entry and exit.
SENT = [5_000_000, 6_000_000]
ENTER, EXIT = 9_000_000, 7_000_000


def _run(directory, monkeypatch, late=(None, 0), replay=None):
    """Requests 0-3 to completion, request `late[0]` added after step
    `late[1]`; with `replay`, each step priced by it on a clock its replies
    advance. Returns that clock after each step, the runner, and the requests
    in each step."""
    monkeypatch.setenv(ENV, str(directory))
    config = _config()
    scheduler, runner = Scheduler(config), _Runner(config)
    now = [0.0]
    if replay is not None:
        install_cost_backend(runner, replay)
        runner.compass_host, replay.now = replay, lambda: now[0]
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    sequences = [Sequence(_prompt(i), BLOCK, sampling_params=params) for i in range(4)]
    for i, sequence in enumerate(sequences):
        if i != late[0]:
            scheduler.add(sequence)
    clock, members = [], []
    index = {s.id: i for i, s in enumerate(sequences)}
    while scheduler.has_unfinished_requests():
        assert len(clock) < 100, "the driven sequence does not terminate"
        batch, seqs = scheduler.schedule()
        members.append([index[r] for r in batch.req_ids])
        reply = runner.forward(batch)
        now[0] += reply.predicted_s
        clock.append(now[0])
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )
        if late[0] is not None and len(clock) == late[1] + 1:
            scheduler.add(sequences[late[0]])
    return clock, runner, members


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A record of the run, with step i entered at ENTER * (i + 1) and left
    EXIT later, two requests sent before it, and a dummy step at start before
    those."""
    directory = tmp_path / "real"
    _, _, members = _run(directory, monkeypatch)
    path = directory / "dp0.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    for i, step in enumerate(steps):
        step.update(t_enter_ns=ENTER * (i + 1), t_exit_ns=ENTER * (i + 1) + EXIT)
        step["step"] += 1
    start = dict(steps[0], step=0, is_dummy_run=True, batch=[], named=[])
    start.update(t_enter_ns=0, t_exit_ns=EXIT)
    path.write_text("".join(json.dumps(s) + "\n" for s in [start, *steps]))
    (directory / ARRIVALS).write_text(json.dumps({"sent_ns": SENT[::-1]}))
    return directory, members


def test_each_replayed_step_ends_where_the_real_forward_returned(
    real, tmp_path, monkeypatch
):
    directory, members = real
    n = len(members)
    clock, runner, _ = _run(
        tmp_path / "sim", monkeypatch, replay=ReplayBackend(directory, 0)
    )
    exits = [ENTER * (i + 1) + EXIT for i in range(n)]
    assert clock == pytest.approx([(t - SENT[0]) / 1e9 for t in exits])
    assert runner._priced.steps == n
    assert compare(directory, tmp_path / "sim")["steps"] == {
        0: {"compared": n, "identical": n}
    }
    assert compare(directory, tmp_path / "sim")["first_divergence"] == {0: None}


def test_a_request_released_one_step_late_is_named_at_its_first_step(
    real, tmp_path, monkeypatch
):
    directory, members = real
    due = next(i for i, step in enumerate(members) if 1 in step)
    assert due > 0, "request 1 must wait for a step to be released late"
    replay = ReplayBackend(directory, 0)
    _, _, late = _run(tmp_path / "sim", monkeypatch, (1, due), replay)
    assert 1 not in late[due]
    report = compare(directory, tmp_path / "sim")
    assert report["first_divergence"][0]["step"] == due
    assert report["steps"][0]["identical"] < report["steps"][0]["compared"]


def test_a_step_past_the_end_of_the_real_record_is_refused_by_name(
    real, tmp_path, monkeypatch
):
    directory, members = real
    n = len(members)
    path = directory / "dp0.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    _, runner, _ = _run(
        tmp_path / "sim", monkeypatch, replay=ReplayBackend(directory, 0)
    )
    refused = runner._refused_commands
    assert len(refused) == n - 2
    assert refused[0] == f"cost:replay: {path} has 2 steps, and this is step 2"


def test_a_run_file_naming_a_record_replays_it_on_the_engine_clock(
    real, tmp_path, monkeypatch
):
    directory, _ = real
    monkeypatch.delenv(ENV)
    r = deployment(tmp_path)
    run = json.loads(_run_file(tmp_path).read_text())
    del run["step_stub"]
    (tmp_path / "run.json").write_text(json.dumps(run | {"replay": str(directory)}))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    engine = compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r)
    engine.now = 0.002
    charged = [r.forward(decode(B37)).predicted_s for _ in "ab"]
    ends = [(ENTER * n + EXIT - SENT[0]) / 1e9 for n in (1, 2)]
    assert charged == pytest.approx([t - engine.now for t in ends])


def test_a_real_workload_is_sent_in_order_and_its_send_times_written(tmp_path):
    seen, admitted = [], []  # (prompt's first word, requests admitted before it)

    class Server(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((body["prompt"].split()[0], len(admitted)))
            time.sleep(0.02)  # the admission
            admitted.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"data: {}\n\ndata: [DONE]\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    traffic.real(server.server_port, traffic.bodies("m", 6), 0.0, tmp_path)
    server.shutdown()
    sent = json.loads((tmp_path / ARRIVALS).read_text())["sent_ns"]
    assert seen == [(str(k), k) for k in range(6)]
    assert sent == sorted(sent) and len(set(sent)) == 6


def test_a_real_workload_writes_when_each_request_was_admitted(tmp_path):
    class Server(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            time.sleep(0.02)  # the admission
            self.send_response(200)
            self.end_headers()
            time.sleep(0.2)  # the stream, after the headers
            self.wfile.write(b"data: [DONE]\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    traffic.real(server.server_port, traffic.bodies("m", 3), 0.0, tmp_path)
    server.shutdown()
    arrivals = json.loads((tmp_path / ARRIVALS).read_text())
    waits = [a - s for s, a in zip(arrivals["sent_ns"], arrivals["admitted_ns"])]
    assert len(waits) == 3 and all(0.02e9 <= w < 0.2e9 for w in waits), waits
