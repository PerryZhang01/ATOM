# SPDX-License-Identifier: MIT
"""A simulated run that replays a real record's step times, and the compare that reads both.

ATOM's `Scheduler` drives the simulated runner's `forward` on one DP rank, as in
`test_parity_record.py`; a "real" record is that run with entry and exit
times written in, and the replay is the same run with `ReplayHost` as its host.
"""

import http.server
import json
import queue
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from test_memory_readings import META_8B
from test_parity_record import BLOCK, _config, _prompt, _Runner
from test_pd_slice import PD
from test_run_step_grid import B37, decode, deployment
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.parity import ARRIVALS, ENV, compare, read, traffic
from atom.compass.parity.replay import ReplayHost
from atom.compass.runner.overrides import RunnerRefusal, _says_why
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime, charge

#: Worker-clock nanoseconds: the first send, and step i's entry and exit.
SENT = [5_000_000, 6_000_000]
ENTER, EXIT = 9_000_000, 7_000_000


class _Arrival:
    """A clock connection granting each TAR its target; `deliver` runs once,
    when a grant first reaches `at`."""

    def __init__(self, at, deliver):
        self.at, self.deliver = at, deliver

    def send(self, request):
        self.t = request[1]

    def recv(self):
        if self.at is not None and self.at <= self.t:
            self.at = None
            self.deliver()
        return self.t, {}


def _run(directory, monkeypatch, late=(None, 0), replay=None, at=None):
    """Requests 0-3 to completion, request `late[0]` added after step
    `late[1]`, or with `at`, once the engine LP's clock reaches `at`; with
    `replay`, a record directory, each step ended by its `ReplayHost` on a
    clock its replies advance, through `clock.charge` with `at`, and priced by
    `_Runner`'s step stub. Returns that clock after each step, the runner, and
    the requests in each step."""
    monkeypatch.setenv(ENV, str(directory))
    config = _config()
    scheduler, runner = Scheduler(config), _Runner(config)
    now = [0.0]
    if replay is not None:
        runner.compass_host = host = ReplayHost(replay, 0)
        host.now = lambda: now[0]
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    sequences = [Sequence(_prompt(i), BLOCK, sampling_params=params) for i in range(4)]
    for i, sequence in enumerate(sequences):
        if i != late[0]:
            scheduler.add(sequence)
    if at is not None:
        lp = LPRuntime(
            LpId("engine"),
            single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002),
            _Arrival(at, lambda: scheduler.add(sequences[late[0]])),
        )
        monkeypatch.setattr("atom.utils.clock._installed", lp)
    clock, members = [], []
    index = {s.id: i for i, s in enumerate(sequences)}
    while scheduler.has_unfinished_requests():
        assert len(clock) < 100, "the driven sequence does not terminate"
        batch, seqs = scheduler.schedule()
        members.append([index[r] for r in batch.req_ids])
        reply = runner.forward(batch)
        if at is None:
            now[0] += reply.predicted_s
        else:
            charge(reply)
            now[0] = lp.now
        clock.append(now[0])
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )
        if late[0] is not None and at is None and len(clock) == late[1] + 1:
            scheduler.add(sequences[late[0]])
    return clock, runner, members


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A record of the run, with step i entered at ENTER * (i + 1) and left
    EXIT later, two requests sent before it, and a dummy step at start before
    those."""
    directory = tmp_path / "real"
    _, _, members = _run(directory, monkeypatch)
    _time_record(directory)
    return directory, members


def _time_record(directory):
    """Write `real`'s times into the record `_run` wrote in `directory`."""
    path = directory / "dp0.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    for i, step in enumerate(steps):
        step.update(t_enter_ns=ENTER * (i + 1), t_exit_ns=ENTER * (i + 1) + EXIT)
        step["step"] += 1
    start = dict(steps[0], step=0, is_dummy_run=True, batch=[], named=[])
    start.update(t_enter_ns=0, t_exit_ns=EXIT)
    path.write_text("".join(json.dumps(s) + "\n" for s in [start, *steps]))
    (directory / ARRIVALS).write_text(json.dumps({"sent_ns": SENT[::-1]}))


#: Worker-clock nanoseconds from a step's exit to the engine's next input pull.
PULL = 1_000_000


def test_an_arrival_between_a_return_and_the_pull_joins_the_next_step_as_it_did(
    tmp_path, monkeypatch
):
    # Request 3, arriving once request 2's prefill is done, is the next step's
    # prefill; a step later, that step decodes the other three instead.
    _, _, plain = _run(tmp_path / "plain", monkeypatch)
    due = next(i for i, step in enumerate(plain) if 3 in step)
    real = tmp_path / "real"
    _, _, members = _run(real, monkeypatch, late=(3, due - 1))
    assert members == plain
    _time_record(real)
    lines = (real / "dp0.jsonl").read_text().splitlines()
    pulls = [{"t_pull_ns": json.loads(line)["t_exit_ns"] + PULL} for line in lines]
    (real / "dp0.pulls.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pulls))
    # It reaches the engine halfway from step due - 1's return to its pull.
    exit_s = (ENTER * due + EXIT - SENT[0]) / 1e9
    at = exit_s + PULL / 2e9

    clock, _, took = _run(tmp_path / "sim", monkeypatch, (3, None), real, at)

    assert took == members
    assert clock[due - 1] == pytest.approx(exit_s + PULL / 1e9)
    assert compare(real, tmp_path / "sim")["first_divergence"] == {0: None}
    (real / "dp0.pulls.jsonl").rename(tmp_path / "dropped")
    with pytest.raises(RunnerRefusal, match="this is step"):  # a step longer
        _run(tmp_path / "sim2", monkeypatch, (3, None), real, at)
    assert compare(real, tmp_path / "sim2")["first_divergence"][0]["step"] == due


def test_a_step_takes_the_first_pull_and_put_after_its_exit_and_compare_skips_them(
    tmp_path,
):
    for side in ("a", "b"):
        (tmp_path / side).mkdir()
        lines = [
            {"step": i, "batch": [], "named": [], "t_enter_ns": t, "t_exit_ns": t + 10}
            for i, t in enumerate((10, 40, 70))
        ]
        (tmp_path / side / "dp0.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines)
        )
    stamps = [("t_pull_ns", 15), ("t_put_ns", 22), ("t_pull_ns", 25)]
    stamps += [("t_pull_ns", 30), ("t_pull_ns", 75), ("t_pull_ns", 90)]
    (tmp_path / "a" / "dp0.pulls.jsonl").write_text(
        "".join(json.dumps({k: t}) + "\n" for k, t in stamps)
    )

    steps = read(tmp_path / "a")[0]

    assert [(s.get("t_pull_ns"), s.get("t_put_ns")) for s in steps] == [
        (25, 22),
        (None, None),
        (90, None),
    ]
    assert compare(tmp_path / "a", tmp_path / "b")["first_divergence"] == {0: None}


def test_each_replayed_step_ends_where_the_real_forward_returned(
    real, tmp_path, monkeypatch
):
    directory, members = real
    n = len(members)
    clock, runner, _ = _run(tmp_path / "sim", monkeypatch, replay=directory)
    exits = [ENTER * (i + 1) + EXIT for i in range(n)]
    assert clock == pytest.approx([(t - SENT[0]) / 1e9 for t in exits])
    assert runner._priced.steps == n
    assert compare(directory, tmp_path / "sim")["steps"] == {
        0: {"compared": n, "identical": n}
    }
    assert compare(directory, tmp_path / "sim")["first_divergence"] == {0: None}
    # The step stub's price reaches the cost record and no clock.
    lines = (tmp_path / "sim" / "dp0.cost.jsonl").read_text().splitlines()[1:]
    cost = [json.loads(line) for line in lines]
    starts = [0.0, *clock[:-1]]
    assert [c["charge_s"] for c in cost] == pytest.approx(
        [t - s for s, t in zip(starts, clock)]
    )
    assert {t["source"] for c in cost for t in c["terms"]} == {"step-stub"}
    assert all(c["device_s"] > 0 for c in cost)


def test_a_request_released_one_step_late_is_named_at_its_first_step(
    real, tmp_path, monkeypatch
):
    directory, members = real
    due = next(i for i, step in enumerate(members) if 1 in step)
    assert due > 0, "request 1 must wait for a step to be released late"
    _, _, late = _run(tmp_path / "sim", monkeypatch, (1, due), directory)
    assert 1 not in late[due]
    report = compare(directory, tmp_path / "sim")
    assert report["first_divergence"][0]["step"] == due
    assert report["steps"][0]["identical"] < report["steps"][0]["compared"]


def test_a_step_past_the_end_of_the_real_record_ends_the_run_by_name(
    real, tmp_path, monkeypatch
):
    directory, _ = real
    path = directory / "dp0.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    why = f"replay: {path} has 2 steps, and this is step 2"
    with pytest.raises(RunnerRefusal, match=f"^{re.escape(why)}$"):
        _run(tmp_path / "sim", monkeypatch, replay=directory)


def test_a_run_file_naming_a_record_replays_it_on_the_engine_clock(
    real, tmp_path, monkeypatch
):
    directory, _ = real
    monkeypatch.delenv(ENV)
    r = deployment(tmp_path)
    run = json.loads(_run_file(tmp_path).read_text()) | {"replay": str(directory)}
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    engine = compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    # A replay ends steps and prices nothing, and replaces the host laws.
    refused = {
        "has none": {k: v for k, v in run.items() if k != "step_stub"},
        "in place of its host laws": run | {"host": {}},
    }
    for why, bad in refused.items():
        (tmp_path / "run.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=why):
            compass_run.runner(r, META_8B)
    (tmp_path / "run.json").write_text(json.dumps(run))
    compass_run.runner(r, META_8B)
    engine.now = 0.002
    charged = [r.forward(decode(B37)).predicted_s for _ in "ab"]
    ends = [(ENTER * n + EXIT - SENT[0]) / 1e9 for n in (1, 2)]
    assert charged == pytest.approx([t - engine.now for t in ends])


def test_a_worker_that_replays_past_its_record_names_why_in_the_engine_hand_in(
    real, tmp_path, monkeypatch
):
    # ATOM's process monitor hands the engine a bare SystemExit for a dead worker.
    directory, _ = real
    path = directory / "dp0.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    monkeypatch.delenv(ENV)
    r = deployment(tmp_path)
    run = json.loads(_run_file(tmp_path).read_text())
    (tmp_path / "run.json").write_text(json.dumps(run | {"replay": str(directory)}))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    now = tmp_path / compass_run.NOW_FILE.format(compass_run.ENGINE.name)
    monkeypatch.setenv(compass_run.NOW_ENV, str(now))
    engine = compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    monkeypatch.setattr(clock, "_installed", engine)
    compass_run.runner(r, META_8B)

    with pytest.raises(SystemExit), compass_run.engine_loop(SimpleNamespace()):
        with pytest.raises(RunnerRefusal):
            for _ in "abc":
                r.forward(decode(B37))
        raise SystemExit()

    hand_in = tmp_path / compass_run.LP_FILE.format(compass_run.ENGINE.name)
    why = RunnerRefusal(f"replay: {path} has 2 steps, and this is step 2")
    assert json.loads(hand_in.read_text())["refusals"] == [
        f"ended:engine raised SystemExit(); rank 0 raised {why!r}"
    ]


def test_an_earlier_runs_raised_file_is_not_read_into_this_runs_end(
    tmp_path, monkeypatch
):
    path = _run_file(tmp_path)
    run = json.loads(path.read_text())
    (tmp_path / "now-engine.0.raised").write_text("rank 0 raised RunnerRefusal()")
    compass_run._RecordingAuthority(run)  # a rerun into the same out_dir
    monkeypatch.setenv(compass_run.ENV, str(path))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now-engine.bin"))
    engine = compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    monkeypatch.setattr(clock, "_installed", engine)
    with pytest.raises(SystemExit), compass_run.engine_loop(SimpleNamespace()):
        raise SystemExit()
    hand_in = tmp_path / compass_run.LP_FILE.format(compass_run.ENGINE.name)
    assert json.loads(hand_in.read_text())["refusals"] == [
        "ended:engine raised SystemExit()"
    ]


def test_a_raised_file_that_cannot_be_written_leaves_the_refusal(tmp_path):
    def forward(self, batch):
        raise RunnerRefusal("why")

    runner = SimpleNamespace(rank=0, compass_raised_file=tmp_path / "no" / "f")
    with pytest.raises(RunnerRefusal, match="^why$"):
        _says_why(forward)(runner, None)


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
    arrived = [threading.Event() for _ in range(3)]
    ended = [0] * 3  # just before each stream's end was written

    class Server(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            k = int(body["prompt"].split()[0])
            arrived[k].set()
            time.sleep(0.02)  # the admission
            self.send_response(200)
            self.end_headers()
            if k < 2:  # the stream, open until request k + 1 arrives, or 10 s
                arrived[k + 1].wait(10)
            ended[k] = time.monotonic_ns()
            self.wfile.write(b"data: [DONE]\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    traffic.real(server.server_port, traffic.bodies("m", 3), 0.0, tmp_path)
    server.shutdown()
    arrivals = json.loads((tmp_path / ARRIVALS).read_text())
    sent, admitted = arrivals["sent_ns"], arrivals["admitted_ns"]
    assert len(admitted) == 3, admitted
    assert all(s + 0.02e9 <= a for s, a in zip(sent, admitted)), (sent, admitted)
    # request k + 1 is sent only once k is admitted, and k's stream outlives that
    assert all(a < e for a, e in zip(admitted, ended[:2])), (admitted, ended)


@pytest.mark.parametrize("alone", [[], ["0", "0", "1"]], ids=["default", "alone"])
def test_the_real_workload_runs_request_0_alone_only_when_asked(
    tmp_path, monkeypatch, alone
):
    done = []  # just before each stream's end was written
    second = threading.Event()  # request 1 reached the server

    class Server(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            k = int(body["prompt"].split()[0])
            if k == 1:
                second.set()
            self.send_response(200)
            self.end_headers()
            if k == 0 and not alone:  # open until request 1 arrives, or 10 s
                second.wait(10)
            else:
                time.sleep(0.3)  # the stream, longer than a gap
            done.append(time.monotonic_ns())
            self.wfile.write(b"data: [DONE]\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv(compass_run.ENV, raising=False)
    argv = [str(server.server_port), "m", str(tmp_path), "3", "0.1", *alone]
    subprocess.run([sys.executable, "-m", traffic.__name__, *argv], check=True)
    server.shutdown()
    sent = json.loads((tmp_path / ARRIVALS).read_text())["sent_ns"]
    if not alone:  # the default sends request 1 during request 0's stream
        assert sent[1] < min(done), (sent, done)
    else:  # request 1 after request 0's end, request 2 a gap after that end
        assert min(done) <= sent[1] and min(done) + 0.1e9 <= sent[2], (sent, done)


@pytest.mark.parametrize("role, side", [("kv_producer", "P"), ("kv_consumer", "D")])
def test_each_side_of_a_prefill_decode_run_replays_its_own_record(
    real, tmp_path, monkeypatch, role, side
):
    directory, _ = real
    record = directory / {"P": "prefill", "D": "decode"}[side]
    record.mkdir()
    for name in ("dp0.jsonl", ARRIVALS):
        (directory / name).rename(record / name)
    monkeypatch.delenv(ENV)
    r = deployment(
        tmp_path, kv_transfer_config={"kv_connector": "compass", "kv_role": role}
    )
    run = json.loads(_run_file(tmp_path, **PD).read_text())
    (tmp_path / "run.json").write_text(json.dumps(run | {"replay": str(directory)}))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    engine = compass_run._PublishedClock(
        LpId(f"engine-{side}"), compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    engine.now = 0.002
    charged = [r.forward(decode(B37)).predicted_s for _ in "ab"]
    ends = [(ENTER * n + EXIT - SENT[0]) / 1e9 for n in (1, 2)]
    assert charged == pytest.approx([t - engine.now for t in ends])
