# SPDX-License-Identifier: MIT
"""The run summary names its format and the source tree that ran."""

import json
import math
import time
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal, provenance
from atom.utils import clock
from atom.utils.clock import LPRuntime


def _run(tmp_path, monkeypatch) -> dict:
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


def test_the_summary_names_its_schema_version_and_the_tree_that_ran(
    monkeypatch, tmp_path
):
    run = _run(tmp_path, monkeypatch)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    monkeypatch.setattr(compass_run, "_authority", authority)
    table = compass_run.channel_table(run)
    compass_run._hand_in(run, compass_run.ENGINE.name, [])
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    print("identity:", summary["schema_version"], summary["source_root"])
    assert summary["schema_version"] == compass_run.SUMMARY_SCHEMA_VERSION
    root = provenance.SourceRoot.from_json(summary["source_root"])
    assert (root.name, root.root) == ("atom", provenance.module_root("atom"))
    assert root.revision_kind in ("tree", "commit")


def test_a_tree_with_neither_git_nor_a_stamp_is_refused_by_name(monkeypatch, tmp_path):
    run = _run(tmp_path, monkeypatch)
    bare = tmp_path / "bare" / "atom"
    bare.mkdir(parents=True)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setattr(provenance, "module_root", lambda module: str(bare))
    with pytest.raises(ArtifactRefusal, match=f"`{bare}` is neither a git checkout"):
        compass_run._RecordingAuthority(run)


def test_the_summary_names_the_router_binary_and_keeps_it_out_of_the_configuration(
    monkeypatch, tmp_path
):
    run = _run(tmp_path, monkeypatch)
    router = {
        "path": "/usr/local/bin/atomesh",
        "commit": "45009a939",
        "status": "clean",
    }
    run["router"] = router
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    compass_run._hand_in(run, compass_run.ENGINE.name, [])
    compass_run._hand_in(run, compass_run.FRONTEND.name, [])
    compass_run._write_summary(run, authority)

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    print("router:", summary["router"])
    assert summary["router"] == router
    assert "router" not in summary["configuration"]
    assert "atomesh" not in (tmp_path / compass_run.STEP_TABLE_FILE).read_text()
