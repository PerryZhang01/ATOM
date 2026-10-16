# SPDX-License-Identifier: MIT
"""`compass record`, on records `StepRecord` wrote on
`test_parity_device_window.py`'s planted device, with steps from two planted
client runs.

A planted server stands in for ATOM's: it copies such a record to where the
step record's variable points, answers ``/health`` once and exits. `rocm-smi`,
`rocminfo` and the checkouts run.json's identity reads are planted as
`test_drive_gpu.py` and `test_drive_identity.py` plant them, so run.json's
identity is the one `identity` builds, and `fit.samples` reads it.
"""

import json
import os
import re
import shutil
import sys

import pytest
from test_drive_gpu import FREE
from test_drive_identity import CARD, GIT, _checkout, _git
from test_fit_samples import HOST, RUNS, edit
from test_fit_samples import record as planted_out
from test_parity_record import BLOCK, BUDGET, LADDER
from test_parity_served import recorded

from atom.compass import cli
from atom.compass.backends.step_labels import TABLE_VERSION, label
from atom.compass.drive import identity, record, serve
from atom.compass.fit.samples import load

PLANTED = """\
import http.server, os, shutil, sys
shutil.copytree(os.environ["PLANTED_RECORD"], os.environ["ATOM_COMPASS_PARITY_RECORD"])
port = int(sys.argv[sys.argv.index("--server-port") + 1])

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

http.server.HTTPServer(("127.0.0.1", port), Health).handle_request()
"""
#: The engine `test_parity_record.py`'s config resolves.
ENGINE = {
    "max_num_seqs": 8,
    "max_num_batched_tokens": BUDGET,
    "max_model_len": 2048,
    "kv_cache_block_size": BLOCK,
    "num_kvcache_blocks": 4096,
    "ladder": LADDER,
}


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_stopped_server_s_directory_gains_its_engine_entry_and_cost_lines(
    tmp_path, monkeypatch
):
    """The named result."""
    out = planted_out(tmp_path, monkeypatch)
    before = json.loads((out / "run.json").read_text())
    record.write_engine_and_costs(out)

    assert json.loads((out / "run.json").read_text()) == {**before, "engine": ENGINE}
    steps = [s for s in _lines(out / "steps/dp0.jsonl") if not s["is_dummy_run"]]
    table = _lines(out / "steps/dp0.cost.jsonl")
    assert table[0] == {"version": TABLE_VERSION}
    assert [(r["step"], r["d_busy_s"]) for r in table[1:]] == [
        (s["step"], s["d_busy_ns"] * 1e-9) for s in steps
    ]
    assert table[1:] == json.loads(json.dumps(label(out / "steps/dp0.jsonl")[1]))


def _two_ranks(tmp_path, monkeypatch, blocks):
    """A planted out directory whose DP rank 1 copies rank 0's record and
    states `blocks` KV blocks."""
    out = planted_out(tmp_path, monkeypatch)
    for suffix in (".jsonl", ".engine.json"):
        shutil.copy(out / f"steps/dp0{suffix}", out / f"steps/dp1{suffix}")
    edit(out / "steps/dp1.engine.json", lambda e: e.update(num_kvcache_blocks=blocks))
    return out


def test_each_dp_rank_gets_its_cost_lines_under_one_engine_entry(tmp_path, monkeypatch):
    out = _two_ranks(tmp_path, monkeypatch, ENGINE["num_kvcache_blocks"])
    record.write_engine_and_costs(out)

    assert json.loads((out / "run.json").read_text())["engine"] == ENGINE
    dp0, dp1 = ((out / f"steps/dp{r}.cost.jsonl").read_text() for r in (0, 1))
    assert dp0 == dp1


def test_a_dp_rank_no_request_reached_gets_the_version_line_alone(
    tmp_path, monkeypatch
):
    out = _two_ranks(tmp_path, monkeypatch, ENGINE["num_kvcache_blocks"])
    dummies = [
        {**s, "is_dummy_run": True, "batch": [], "named": []}
        for s in _lines(out / "steps/dp1.jsonl")
    ]
    (out / "steps/dp1.jsonl").write_text("".join(json.dumps(s) + "\n" for s in dummies))
    record.write_engine_and_costs(out)

    assert _lines(out / "steps/dp1.cost.jsonl") == [{"version": TABLE_VERSION}]
    assert len(_lines(out / "steps/dp0.cost.jsonl")) > 1


def test_dp_ranks_resolving_different_engines_refuse_the_entry(tmp_path, monkeypatch):
    out = _two_ranks(tmp_path, monkeypatch, 4000)

    refusal = "num_kvcache_blocks: 4096 in dp0; 4000 in dp1$"
    with pytest.raises(ValueError, match=refusal):
        record.write_engine_and_costs(out)
    assert "engine" not in json.loads((out / "run.json").read_text())


@pytest.mark.parametrize(
    "ours, atoms, refusal",
    [
        (["--profile-ops"], [], "--profile-ops and --profile-steps profile ops, and"),
        (["--profile-steps", "2:5"], [], "--profile-ops and --profile-steps profile"),
        ([], ["--load_dummy"], "--load_dummy is among ATOM's arguments; a record"),
        ([], ["--load-dummy=zero"], "--load-dummy=zero is among"),
        ([], ["--load_d"], "--load_d is among"),
        ([], ["--host", "0.0.0.0"], "--host is among ATOM's arguments; the driver"),
    ],
)
def test_ops_profiling_dummy_weights_and_owned_flags_refuse_before_a_server_starts(
    tmp_path, monkeypatch, ours, atoms, refusal
):
    """`serve` makes the out directory before its GPU checks and its server,
    which exits at once here, so a refusal that regresses cannot hang."""
    monkeypatch.setattr(serve, "SERVER", (sys.executable, "-c", "pass"))
    out = tmp_path / "out"
    argv = ["record", "--gpu", "7", "--out", str(out), *ours, "--", "--model", "m"]

    with pytest.raises(SystemExit, match=f"^compass record: {re.escape(refusal)}"):
        cli.main([*argv, *atoms])
    assert not out.exists()


@pytest.fixture
def planted_gpu(tmp_path, monkeypatch):
    """The planted server, `rocm-smi` listing GPU 7 free, `rocminfo`, the
    checkouts and the model whose config.json run.json digests; returns the model."""
    steps = recorded(tmp_path / "planted", monkeypatch, key=HOST)
    monkeypatch.setenv("PLANTED_RECORD", str(steps))
    (tmp_path / "server.py").write_text(PLANTED)
    monkeypatch.setattr(serve, "SERVER", (sys.executable, str(tmp_path / "server.py")))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "listing").write_text(FREE)
    (bindir / "cards.json").write_text(json.dumps({"card7": CARD}))
    (bindir / "rocm-smi").write_text(
        f'#!/bin/sh\ncase "$*" in *--showproductname*) cat {bindir}/cards.json;;\n'
        f"*--showpids*) cat {bindir}/listing;;\nesac\n"
    )
    (bindir / "rocminfo").write_text("#!/bin/sh\nexit 0\n")
    for tool in ("rocm-smi", "rocminfo"):
        (bindir / tool).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    trees = {
        m: _checkout(tmp_path / m, tag) for m, tag in (("atom", None), ("aiter", "v0"))
    }
    monkeypatch.setattr(identity, "module_root", lambda module: str(trees[module]))
    (tmp_path / "rocm" / ".info").mkdir(parents=True)
    (tmp_path / "rocm" / ".info" / "version").write_text("7.2.4\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text('{"hidden_size": 8}')
    return tmp_path / "model"


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_record_s_out_directory_loads_through_the_fit_s_loader(tmp_path, planted_gpu):
    out = tmp_path / "out"
    cli.main(f"record --gpu 7 --out {out} -- --model {planted_gpu}".split())

    assert sorted(p.name for p in out.iterdir()) == [
        *("gpu_after.txt", "gpu_before.txt", "preflight_after.txt"),
        *("preflight_before.txt", "run.json", "server.json", "server.log", "steps"),
    ]
    assert sorted(p.name for p in (out / "steps").iterdir()) == [
        *("dp0.cost.jsonl", "dp0.engine.json", "dp0.jsonl", "dp0.key.json")
    ]
    run = json.loads((out / "run.json").read_text())
    assert run.keys() == {
        *("format_version", "argv", "source", "host_key", "device", "stack"),
        *("profiled", "engine"),
    }
    assert (run["profiled"], run["engine"]) == (False, ENGINE)
    samples, config = load({out: RUNS})
    steps = _lines(out / "steps/dp0.jsonl")
    assert len(samples) == sum(not s["is_dummy_run"] for s in steps)
    aiter = _git(tmp_path / "aiter", "rev-parse", "--short", "HEAD")
    assert (config["device"], config["parallel"], config["stack"]["aiter"]) == (
        "MI308X gfx942",
        "tp1-dp1",
        aiter,
    )
