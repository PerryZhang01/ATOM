# SPDX-License-Identifier: MIT
"""run.json's identity. The source digest and aiter's commit are read off
planted git checkouts, the host key off the engine file a parity record wrote,
and the device off a planted `rocm-smi`."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch
from test_parity_record import _drive

from atom.compass.drive import identity as drive
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.run import host_key

GIT = shutil.which("git")
CARD = {
    "Card Series": "AMD Instinct MI308X",
    "Card Model": "0x74a2",
    "Card SKU": "M3080202",
    "Node ID": "9",
    "GFX Version": "gfx942",
}
ARGV = ["compass", "record", "--gpu", "7", "--out", "d", "--", "--model", "m"]
MACHINES = Path(__file__).parents[2] / "scripts/compass/machines"

pytestmark = pytest.mark.skipif(GIT is None, reason="needs git")


def _git(tree, *argv):
    return subprocess.run(
        [GIT, "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *argv],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(tree, tag=None):
    tree.mkdir()
    (tree / "a.py").write_text("a = 1\n")
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "s"]):
        _git(tree, *argv)
    if tag:
        _git(tree, "tag", tag)
    return tree


@pytest.fixture
def planted(tmp_path, monkeypatch):
    trees = {
        "atom": _checkout(tmp_path / "atom"),
        "aiter": _checkout(tmp_path / "aiter", "v0"),
    }
    monkeypatch.setattr(drive, "module_root", lambda module: str(trees[module]))
    (tmp_path / "rocm" / ".info").mkdir(parents=True)
    (tmp_path / "rocm" / ".info" / "version").write_text("7.2.4\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 27, 7))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cards.json").write_text(json.dumps({"card6": CARD, "card7": CARD}))
    (bindir / "rocm-smi").write_text(f"#!/bin/sh\ncat {bindir / 'cards.json'}\n")
    (bindir / "rocm-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text('{"hidden_size": 8}')
    config = _drive(tmp_path / "steps", monkeypatch, {0: [0]})[0][2].config
    config.model = str(tmp_path / "model")
    return trees, config, (tmp_path / "steps" / "dp0.engine.json", config.model)


def test_the_host_key_is_the_one_the_config_that_wrote_the_engine_file_gives(
    planted,
):
    _, config, served = planted
    key = drive.identity(*served, ARGV)["host_key"]
    assert key == host_key(config)
    assert key["model"].startswith("sha256:") and key["engine"]["ladder"] == [
        1,
        2,
        4,
        8,
    ]


def test_the_identity_holds_its_entries_and_no_workload(planted):
    trees, config, served = planted
    run = drive.identity(*served, ARGV)
    assert run == {
        "format_version": drive.FORMAT_VERSION,
        "argv": ARGV,
        "source": run["source"],
        "host_key": host_key(config),
        "device": [],
        "stack": {
            "rocm": "7.2.4",
            "aiter": _git(trees["aiter"], "rev-parse", "--short=9", "HEAD"),
            "rccl": "2.27.7",
            "torch": torch.__version__,
        },
        "profiled": False,
    }
    json.dumps(run)


def test_the_source_digest_is_the_tree_git_archive_ships(planted):
    trees, _, served = planted
    source = drive.identity(*served, ARGV)["source"]
    assert source["revision"] == _git(trees["atom"], "rev-parse", "HEAD^{tree}")
    assert (source["revision_kind"], source["dirty"]) == ("tree", 0)

    (trees["atom"] / "a.py").write_text("a = 2\n")
    (trees["aiter"] / "a.py").write_text("a = 2\n")
    run = drive.identity(*served, ARGV)
    assert (run["source"]["revision"], run["source"]["dirty"]) == (
        source["revision"],
        1,
    )
    commit = _git(trees["aiter"], "rev-parse", "--short=9", "HEAD")
    assert run["stack"]["aiter"] == f"{commit}-dirty"

    _git(trees["atom"], "commit", "-qam", "edit")
    assert drive.identity(*served, ARGV)["source"]["revision"] != source["revision"]


def test_aiter_s_commit_is_read_at_the_length_every_machine_spec_pins(planted):
    trees, _, served = planted
    pins = {
        json.loads(spec.read_text())["device"]["software_pinned_to"]["aiter"]
        for spec in MACHINES.glob("*.json")
    }
    assert {len(pin) for pin in pins} == {drive.AITER_ABBREV}
    # This checkout's own abbreviation is shorter, as a shallow clone's is.
    assert len(_git(trees["aiter"], "rev-parse", "--short", "HEAD")) < len(pins.pop())

    aiter = drive.identity(*served, ARGV)["stack"]["aiter"]
    assert aiter == _git(trees["aiter"], "rev-parse", "--short=9", "HEAD")


def test_an_aiter_root_git_cannot_name_refuses_naming_it(
    planted, tmp_path, monkeypatch
):
    trees, _, served = planted
    roots = {**trees, "aiter": tmp_path / "plain"}
    roots["aiter"].mkdir()
    monkeypatch.setattr(drive, "module_root", lambda module: str(roots[module]))

    named = f"git names no commit at {roots['aiter']}: "
    with pytest.raises(ValueError, match="^" + re.escape(named)):
        drive.identity(*served, ARGV)


def test_each_named_gpu_is_recorded_with_its_model(planted):
    *_, served = planted
    model = {"name": "MI308X", "arch": "gfx942"}
    assert drive.identity(*served, ARGV, (7, 6))["device"] == [
        {"index": 7, **model},
        {"index": 6, **model},
    ]
    with pytest.raises(GpuRefusal, match=r"^rocm-smi lists no GPU \[8\]"):
        drive.identity(*served, ARGV, (7, 8))


def test_a_phase_is_recorded_in_wall_and_monotonic_time_and_ends_when_it_does():
    phases = {}
    with drive.phase(phases, "warm-up"):
        assert "end" not in phases["warm-up"]
        time.sleep(0.01)
    with drive.phase(phases, "measuring"):
        time.sleep(0.01)
    for edges in phases.values():
        for clock in ("wall", "monotonic"):
            assert edges["start"][clock] < edges["end"][clock]
    assert (
        phases["warm-up"]["end"]["monotonic"]
        <= phases["measuring"]["start"]["monotonic"]
    )
