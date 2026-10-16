# SPDX-License-Identifier: MIT
"""`compass trace steps`: the steps a simulated server served, published as one
`shape_population` and read back through `ArtifactStore`.

Records are driven through ATOM's `Scheduler` and `NonAllocatingRunner` as
`test_parity_record.py` drives them, and run files are built by `runfile` from
step stub files. The command itself runs against a planted server that writes
such a record; ``test_a_simulated_server_serves_a_client_and_publishes_its_steps``
starts ATOM's API server, skipping without a driver and without
``ATOM_COMPASS_SLICE_MODEL``, as `test_vertical_slice.py` does.
"""

import copy
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
import torch
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_drive_runfile import build
from test_memory_readings import DOCUMENT
from test_parity_record import LADDER, _drive

from atom.compass import parity
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Key,
    Kind,
    Topology,
)
from atom.compass.backends.step_stub import Coefficients
from atom.compass.bench import measure
from atom.compass.drive import serve, trace_steps
from atom.compass.parity.served import served

TREE = Path(__file__).resolve().parents[2]
#: run.json's identity, as `serve` writes it once its server stops.
IDENTITY = {
    "host_key": {"model": "sha256:c0f1", "host_cpu": "a host CPU"},
    "source": {"revision_kind": "tree", "revision": "5ca1ab1e"},
}
#: The default step stub and a constant one, each its own backend.
STUBS = (Coefficients(), Coefficients.constant(2.0**-9, 2.0**-11))
#: A planted server: it writes the record `PLANTED_RECORD` names where the
#: driver's step record goes, answers ``/health`` once and exits.
PLANTED = """\
import http.server, os, shutil, sys
shutil.copytree(os.environ["PLANTED_RECORD"], os.environ["ATOM_COMPASS_PARITY_RECORD"])

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

port = int(sys.argv[sys.argv.index("--server-port") + 1])
http.server.HTTPServer(("127.0.0.1", port), Health).handle_request()
"""


@pytest.fixture(autouse=True)
def _roots(monkeypatch):
    """The trees a published entry names, planted: a test's tree is no snapshot."""
    monkeypatch.setattr(
        trace_steps, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT)
    )


def stub(tmp_path, k: int) -> Path:
    path = tmp_path / f"stub{k}.json"
    path.write_text(json.dumps(dataclasses.asdict(STUBS[k])))
    return path


def stopped(tmp_path, monkeypatch, ranks, **drive) -> tuple[Path, dict]:
    """A stopped server's `--out`: the record `ranks` drives, and run.json."""
    out = tmp_path / "out"
    built = _drive(out / serve.STEPS, monkeypatch, ranks, **drive)
    # Each request is named with the response id its client saw, as the API
    # server's requests are.
    for path in (out / serve.STEPS).glob("dp*[0-9].jsonl"):
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for line in lines:
            for named in line["named"]:
                named.append(f"cmpl-{named[0]}")
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    (out / serve.RUN_JSON).write_text(json.dumps(IDENTITY))
    return out, built


def entry(store, key):
    return ArtifactStore(store).read(key)


def member(read, stem, rank, extension):
    return json.loads(read.read_member(stem, rank, extension))


def test_the_rung_histogram_is_the_decode_steps_per_rung_and_two_backends_two_entries(
    tmp_path, monkeypatch
):
    # Three prompts' blocks fit at once, so three decode on rung 4, then the
    # fourth alone on rung 1.
    out, built = stopped(tmp_path, monkeypatch, {0: [0, 1, 2, 3]}, blocks=45)
    runs = [build(tmp_path, "--step-stub", str(stub(tmp_path, k))) for k in (0, 1)]
    keys = [trace_steps.publish(out, run, "compass trace steps") for run in runs]

    # A decode step replays the narrowest rung that holds its batch.
    widths = [
        len(b.req_ids) for b in built[0][0] if (b.num_scheduled_tokens == 1).all()
    ]
    rungs = Counter(min(rung for rung in LADDER if rung >= w) for w in widths)
    assert len(rungs) > 1 and rungs != Counter(widths)
    rank = Topology().ranks()[0]
    for key in keys:
        summary = member(entry(runs[0]["store"], key), "summary", rank, "json")
        assert summary["rungs"] == [[rung, n] for rung, n in sorted(rungs.items())]
    assert [k.values["backend"] for k in keys] == [r["backend"]["digest"] for r in runs]
    assert keys[0].values["backend"] != keys[1].values["backend"]
    same = [{f: v for f, v in k.values.items() if f != "backend"} for k in keys]
    assert same[0] == same[1]
    assert len(list((Path(runs[0]["store"]) / "shape_population").iterdir())) == 2


def test_a_summary_reads_structures_widths_contexts_raggedness_and_rungs():
    def line(rows, rung=None, output=True, dummy=False):
        return {"rows": rows, "rung": rung, "produces_output": output} | {
            "is_dummy_run": dummy
        }

    decode = [[1, 50, True], [1, 150, True]]
    lines = [
        line([[64, 64, False]], output=False),
        line([[8, 72, False], [20, 20, False]]),
        line([[1, 100, True], [30, 30, False]]),
        line(decode, rung=2),
        line(decode, rung=2),
        line([[1, 7, True]], rung=1),
        line([[1, 1, False]], dummy=True),
    ]
    assert trace_steps.summary(lines) == {
        "structures": [
            ["decode graph", 3],
            ["mixed eager", 1],
            ["prefill eager", 1],
            ["prefill eager no output", 1],
        ],
        "batch_widths": [[1, 2], [2, 4]],
        "context_multisets": [
            [(7,), 1],
            [(20, 72), 1],
            [(30, 100), 1],
            [(50, 150), 2],
            [(64,), 1],
        ],
        # B times the longest context over the contexts' sum.
        "raggedness": [[1.0, 2], [2 * 150 / 200, 2], [2 * 100 / 130, 1]]
        + [[2 * 72 / 92, 1]],
        "rungs": [[1, 1], [2, 2]],
        "one_width": {
            "mixed eager": 2,
            "prefill eager": 2,
            "prefill eager no output": 1,
        },
    }


def test_each_dp_rank_publishes_its_steps_engine_and_requests(tmp_path, monkeypatch):
    out, _ = stopped(tmp_path, monkeypatch, {0: [0, 2], 1: [1]})
    run = build(tmp_path, "--step-stub", str(stub(tmp_path, 0)), atom=("-dp", "2"))
    key = trace_steps.publish(out, run, "compass trace steps")

    read = entry(run["store"], key)
    assert read.topology == Topology(dp=2)
    # Unpacked as `measure steps` unpacks an entry, it reads as the record.
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    for rank in read.topology.ranks():
        for stem, extension, name in (
            ("steps", "jsonl", f"dp{rank.dp}.jsonl"),
            ("engine", "json", f"dp{rank.dp}.engine.json"),
        ):
            body = read.read_member(stem, rank, extension)
            assert body == (out / serve.STEPS / name).read_bytes()
            (unpacked / name).write_bytes(body)
    record = parity.read(out / serve.STEPS)
    assert parity.read(unpacked) == record
    named = {
        r: sorted({row[0] for step in steps for row in step["batch"]})
        for r, steps in record.items()
    }
    ranks = read.topology.ranks()
    assert [member(read, "summary", r, "json")["requests"] for r in ranks] == [
        named[0],
        named[1],
    ]
    assert len(named[0]) == 2 and len(named[1]) == 1
    every = json.dumps(sorted(named[0] + named[1])).encode()
    assert key.values["workload"] == "sha256:" + hashlib.sha256(every).hexdigest()
    engine = json.loads((out / serve.STEPS / "dp0.engine.json").read_text())
    stated = json.loads(key.values["engine_config"])
    assert stated == {k: engine[k] for k in stated}
    assert {"max_num_seqs", "num_kvcache_blocks"}.isdisjoint(stated)
    assert key.values["model"] == IDENTITY["host_key"]["model"]


def test_measure_steps_reads_a_published_population_by_its_token(tmp_path, monkeypatch):
    out, _ = stopped(tmp_path, monkeypatch, {0: [0, 1]})
    run = build(tmp_path, "--step-stub", str(stub(tmp_path, 0)))
    key = trace_steps.publish(out, run, "compass trace steps")

    store, engine = ArtifactStore(run["store"]), json.loads(key.values["engine_config"])
    source, lines, stated = measure._read(key.token(), store, [], None, engine, {})
    assert source == str(key)
    assert stated == json.loads((out / serve.STEPS / "dp0.engine.json").read_text())
    assert lines == served(out / serve.STEPS)[0]


def test_a_shape_population_key_without_its_backend_refuses_naming_four_fields():
    with pytest.raises(ArtifactRefusal) as refused:
        Key.of(Kind.SHAPE_POPULATION, model="m", workload="w", engine_config="e")
    assert "states no backend" in str(refused.value)
    assert "(model, workload, engine_config, backend)" in str(refused.value)


@pytest.mark.parametrize(
    "role, side", [("kv_producer", "prefill"), ("kv_consumer", "decode")]
)
def test_a_prefill_or_a_decode_instance_refuses_before_its_server_starts(
    tmp_path, role, side
):
    kv = json.dumps({"kv_connector": "compass", "kv_role": role})
    flags = ["--step-stub", str(stub(tmp_path, 0)), "--machine", "m", "--store", "s"]
    with pytest.raises(ValueError, match=f"makes this a {side} instance"):
        trace_steps.main(
            [*flags, "--out", str(tmp_path / "out"), "--", "--model", "m"]
            + ["--kv-transfer-config", kv]
        )
    assert not (tmp_path / "out").exists()


def test_trace_steps_serves_simulated_and_publishes_what_its_server_recorded(
    tmp_path, monkeypatch, capsys
):
    record = tmp_path / "record"
    _drive(record, monkeypatch, {0: [0, 1]})
    (tmp_path / "planted.py").write_text(PLANTED)
    monkeypatch.setattr(serve, "SERVER", (sys.executable, str(tmp_path / "planted.py")))
    monkeypatch.setattr(serve, "identity", lambda *args: IDENTITY)
    monkeypatch.setenv("PLANTED_RECORD", str(record))
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    flags = ["--step-stub", str(stub(tmp_path, 1)), "--out", str(tmp_path / "out")]
    flags += ["--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store")]
    trace_steps.main([*flags, "--", "--model", str(tmp_path)])

    out = tmp_path / "out"
    run = json.loads((out / serve.RUN_FILE).read_text())
    assert run["client_runs"] is None and run["out_dir"] == str(out)
    printed = capsys.readouterr().out
    token = printed.rsplit(" as ", 1)[1].strip()
    (published,) = (tmp_path / "store" / "shape_population").iterdir()
    assert published.name.endswith(token)
    read = json.loads((published / "entry.json").read_text())
    key = Key.of(Kind.SHAPE_POPULATION, **read["key"])
    assert key.values["backend"] == run["backend"]["digest"]
    rank = Topology().ranks()[0]
    steps = entry(tmp_path / "store", key).read_member("steps", rank, "jsonl")
    assert steps == (record / "dp0.jsonl").read_bytes()


MODEL = os.environ.get("ATOM_COMPASS_SLICE_MODEL")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="needs a driver and ATOM_COMPASS_SLICE_MODEL",
)
def test_a_simulated_server_serves_a_client_and_publishes_its_steps(tmp_path):
    from test_vertical_slice import Traffic

    machine = copy.deepcopy(DOCUMENT)
    applies = machine["host"]["tokenizers"][0]["applies_to"]
    arch = json.loads((Path(MODEL) / "config.json").read_text())["architectures"][0]
    applies += [] if arch in applies else [arch]
    (tmp_path / "machine.json").write_text(json.dumps(machine))
    out, store = tmp_path / "out", tmp_path / "store"
    argv = ["trace", "steps", "--step-stub", str(stub(tmp_path, 1))]
    argv += ["--machine", str(tmp_path / "machine.json"), "--store", str(store)]
    argv += ["--out", str(out), "--", "--model", MODEL, "--enforce-eager"]
    driver = subprocess.Popen(
        [sys.executable, "-c", "from atom.compass.cli import main; main()", *argv],
        env=dict(os.environ, PYTHONPATH=str(TREE)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(1800):
            if (out / serve.READY_FILE).exists():
                break
            assert driver.poll() is None, driver.communicate()[1][-4000:]
            time.sleep(0.5)
        ready = json.loads((out / serve.READY_FILE).read_text())
        run = json.loads(Path(ready["run_file"]).read_text())
        port = int(ready["url"].rsplit(":", 1)[1])
        Traffic(run, port, driver).run()
    finally:
        driver.send_signal(signal.SIGTERM)
        stdout, stderr = driver.communicate(timeout=300)
    assert driver.returncode == 0, stderr[-4000:]
    assert "published shape_population(" in stdout
    (published,) = (store / "shape_population").iterdir()
    read = json.loads((published / "entry.json").read_text())
    key = Key.of(Kind.SHAPE_POPULATION, **read["key"])
    rank = Topology().ranks()[0]
    summary = member(entry(store, key), "summary", rank, "json")
    assert len(summary["requests"]) == 1 and summary["structures"]
