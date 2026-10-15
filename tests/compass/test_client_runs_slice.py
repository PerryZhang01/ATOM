# SPDX-License-Identifier: MIT
"""Two client runs on one simulated server, each with its own summary.

ATOM's API server runs with a run file declaring two client runs. A traffic
LP sends one streamed completion and leaves at the end of its client run; a
second binds where the first left and sends another prompt. Each client
run's stream, less its start, and its summary are those of a fresh server
serving that client alone. The prompts are shorter than one KV block, so no
prefix block carries over. Every server runs with the stream detokenizer's
audits off: their count is per process, so a fresh server audits its first
stream update and a later client run's first update need not be.

The tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`.
"""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from test_vertical_slice import (
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    _free_port,
    _run_file,
)

from atom.compass import run as compass_run

PROMPTS = ("one request through the slice", "a second client with its own prompt")


class Prompted(Traffic):
    """The slice's traffic LP, sending `prompt`."""

    def __init__(self, prompt, *args) -> None:
        super().__init__(*args)
        self.prompt = prompt

    def _carry(self, method, path, body, tracestate, port) -> None:
        body = body and {**body, "prompt": self.prompt}
        super()._carry(method, path, body, tracestate, port)


def serve(out: Path, prompts) -> list[dict]:
    """One server serving one client run per prompt, in sequence: each client
    run's summary, and its stream events and request times less its start."""
    out.mkdir()
    overrides = {"client_runs": len(prompts)} if len(prompts) > 1 else {}
    run_file = _run_file(out, **overrides)
    port = _free_port()
    env = dict(
        os.environ,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_DETOKENIZER_DELTA_REUSE="off",
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
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        clients = []
        try:
            for prompt in prompts:
                run = json.loads(run_file.read_text())
                traffic = Prompted(prompt, run, port, server)
                start = traffic.rt.now
                traffic.run()
                clients.append((start, traffic))
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    tail = (out / "server.log").read_text()[-4000:]
    assert code == 0, f"the server exited {code} after the finish:\n{tail}"
    return [
        {
            "start": start,
            "summary": json.loads(
                (out / f"client-{k}" / compass_run.SUMMARY_FILE).read_text()
            ),
            "events": [(a - start, line) for a, line in traffic.events],
            "sent": traffic.sent - start,
        }
        for k, (start, traffic) in enumerate(clients)
    ]


def _texts(events) -> list:
    """Each stream event's finish reason. A completion's id is drawn at random,
    and the filler token a simulated step reports follows the request's id,
    which runs on across client runs (`atom.compass.runner.step_output`)."""
    return [
        (
            line
            if line == "data: [DONE]"
            else [c["finish_reason"] for c in json.loads(line[6:])["choices"]]
        )
        for _, line in events
    ]


def _refusals(schedule) -> dict:
    """The refusals, with the object addresses an executor's job name holds
    taken out (the metrics render's is refused at the base: #996)."""
    refusals = dict(schedule["refusals"])
    refusals["reasons"] = [
        [re.sub(r" at 0x[0-9a-f]+", "", r), n] for r, n in refusals["reasons"]
    ]
    return refusals


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("client_runs")
    one_server = serve(root / "one-server", PROMPTS)
    fresh = [serve(root / f"fresh-{k}", (p,))[0] for k, p in enumerate(PROMPTS)]
    return one_server, fresh


@NEEDS_A_RUN
def test_two_client_runs_on_one_server_each_with_its_own_summary(runs):
    one_server, fresh = runs
    first, second = (c["summary"]["client_run"] for c in one_server)
    print(f"\nclient runs: {json.dumps(first)} {json.dumps(second)}")
    assert (first["index"], first["start"], second["index"]) == (0, 0.0, 1)
    assert second["start"] == first["end"] == one_server[1]["start"] > 0
    for k, (mine, theirs) in enumerate(zip(one_server, fresh)):
        print(f"client run {k}: {json.dumps(mine['summary']['schedule'])}")
        print(f"fresh server {k}: {json.dumps(theirs['summary']['schedule'])}")
        assert mine["events"][-1][1] == "data: [DONE]"
        assert _texts(mine["events"]) == _texts(theirs["events"])
        times = [mine["sent"], *(t for t, _ in mine["events"])]
        assert times == pytest.approx(
            [theirs["sent"], *(t for t, _ in theirs["events"])], rel=0, abs=1e-9
        )
        ours, fresh_one = mine["summary"]["schedule"], theirs["summary"]["schedule"]
        # A later client run's seconds are a difference of two sums.
        assert ours["predicted_seconds"] == pytest.approx(
            fresh_one["predicted_seconds"], rel=1e-12
        )
        assert ours["seconds_by_method"] == fresh_one["seconds_by_method"]
        assert _refusals(ours) == _refusals(fresh_one)
        assert (
            mine["summary"]["coverage_report"] == theirs["summary"]["coverage_report"]
        )
        assert mine["summary"]["configuration"] == theirs["summary"]["configuration"]
        span = (
            mine["summary"]["client_run"]["end"]
            - mine["summary"]["client_run"]["start"]
        )
        assert span == pytest.approx(
            theirs["summary"]["client_run"]["end"], rel=0, abs=1e-9
        )
