# SPDX-License-Identifier: MIT
"""Chat requests with images held to the scenario checks, each run twice under
a constant step stub.

The workload is the parity workload's with one inline image per request
(`atom.compass.parity.traffic.bodies`), sent at once: the frontend prepares
each request's media on its tokenizer station at ``image_preprocess_s`` an
image, then tokenizes it, and the engine prefills the image's tokens with the
prompt.

The runs need a driver, as in `test_vertical_slice`, and
`ATOM_COMPASS_IMAGE_MODEL`, a multimodal model directory ATOM serves; its
memory readings are measured first on the GPU ``HIP_VISIBLE_DEVICES`` names,
under the machine spec `test_lmcache_mp_engine` reads.
"""

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from test_lmcache_mp_engine import MACHINE
from test_vertical_slice import (
    CONSTANT,
    HTTP,
    TREE,
    WALL_S,
    Traffic,
    _free_port,
    _release,
    _run_file,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.parity.traffic import bodies

MODEL = os.environ.get("ATOM_COMPASS_IMAGE_MODEL")
NEEDS_AN_IMAGE_RUN = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="a run needs a driver and ATOM_COMPASS_IMAGE_MODEL naming a "
    "multimodal model directory",
)
REQUESTS = 3
IMAGE_S = 2.0**-5
ARGS = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]
ARGS += ["--no-enable_prefix_caching"]


class Workload(Traffic):
    """The traffic LP: every request at 0."""

    def __init__(self, requests, *args) -> None:
        super().__init__(*args)
        self.requests = requests

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for body in self.requests:
            stamp = tracestate_with(None, *rt.stamp_send(self.http))
            self._carry("POST", "/v1/chat/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()


def run_images(out: Path, seed: str) -> dict:
    """One run in its own process tree, reading the readings in its parent's store."""
    out.mkdir()
    run_file = _run_file(
        out,
        step_stub=CONSTANT,
        image_preprocess_s=IMAGE_S,
        machine=json.loads(MACHINE.read_text()),
        store=str(out.parent / "store"),
    )
    port = _free_port()
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
    )
    requests = bodies(MODEL, REQUESTS, images=1)
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "atom.entrypoints.openai.api_server"]
            + [*ARGS, "--host", "127.0.0.1", "--server-port", str(port)]
            + ["--compass-run", str(run_file)],
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
    tail = (out / "server.log").read_text()[-4000:]
    assert code == 0, f"the server exited {code} after the finish:\n{tail}"
    return {
        "out": out,
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from atom.compass import cli

    root = tmp_path_factory.mktemp("images")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AITER_LOG_LEVEL", "WARNING")  # the readings are keyed by it
        cli.main(
            ["measure", "memory", "--machine", str(MACHINE)]
            + ["--gpu", os.environ.get("HIP_VISIBLE_DEVICES", "0")]
            + ["--store", str(root / "store"), "--out", str(root / "measure")]
            + ["--", *ARGS]
        )
    return [run_images(root / f"seed{seed}", seed) for seed in ("1", "2")]


@NEEDS_AN_IMAGE_RUN
def test_image_requests_meet_every_scenario_check(runs):
    print(f"\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        runs[0]["sent"],
        runs[0]["taken"],
        {"engine": runs[0]["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)


@NEEDS_AN_IMAGE_RUN
def test_a_request_leaves_for_the_engine_once_its_image_is_prepared(runs):
    table = runs[0]["table"]
    received = sorted(_release(table, "frontend", HTTP, k) for k in range(REQUESTS))
    sent = sorted(
        _release(table, "engine", "frontend->engine:request#dp0", k)
        for k in range(REQUESTS)
    )
    print(f"\nreceived {received}\nsent {sent}")
    assert all(s - r >= IMAGE_S for r, s in zip(received, sent))
