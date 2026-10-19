# SPDX-License-Identifier: MIT
"""The `compass` console script: its command table, its refusals and its entry.

`fit steps` and `fit host` run through it in `test_fit_steps.py` and
`test_fit_host.py`, and here on a machine spec that refuses; elsewhere a
command's module is planted where one is needed.
"""

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import tomllib
from test_fit_steps import LINEAR, clients, record
from test_memory_readings import DOCUMENT

from atom.compass import cli
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.spec import MachineSpec, SpecRefusal

VERBS = [
    "record",
    "trace steps",
    "trace ops",
    "measure steps",
    "measure ops",
    "measure memory",
    "fit steps",
    "fit ops",
    "fit host",
    "run",
]


def test_the_console_script_is_cli_main():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]

    assert scripts["compass"] == "atom.compass.cli:main"


def test_help_lists_every_verb(capsys):
    cli.main(["--help"])

    out = capsys.readouterr().out
    assert [v for v in VERBS if re.search(rf"^  {v} ", out, re.MULTILINE)] == VERBS


@pytest.mark.parametrize("verb", ["fit ops"])
def test_the_ops_verbs_refuse_as_the_ops_half(verb):
    with pytest.raises(SystemExit, match=f"^compass {verb}: the op-level half"):
        cli.main([*verb.split(), "--out", "d"])


def test_a_command_dispatches_its_arguments_and_its_refusals(monkeypatch):
    got = []

    def main(argv):
        got.append(argv)
        raise ValueError("planted refusal")

    monkeypatch.setitem(sys.modules, "planted_fit", types.SimpleNamespace(main=main))
    monkeypatch.setitem(cli.COMMANDS, "fit steps", ("planted_fit", ""))

    with pytest.raises(SystemExit, match="^compass fit steps: planted refusal$"):
        cli.main(["fit", "steps", "--backend", "x", "a"])
    assert got == [["--backend", "x", "a"]]


def test_a_gpu_refusal_exits_with_its_message(monkeypatch):
    def main(argv):
        raise GpuRefusal("GPU 7 has PID 9 on it before the run, so it was not started")

    monkeypatch.setitem(
        sys.modules, "planted_measure", types.SimpleNamespace(main=main)
    )
    monkeypatch.setitem(cli.COMMANDS, "measure steps", ("planted_measure", ""))

    with pytest.raises(SystemExit, match="^compass measure steps: GPU 7 has PID 9 "):
        cli.main(["measure", "steps"])


def test_a_module_not_built_refuses_and_a_missing_dependency_does_not_read_as_one(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(cli.COMMANDS, "run", ("atom.compass.no_such.run", ""))
    with pytest.raises(SystemExit, match="^compass run: not built yet$"):
        cli.main(["run", "--machine", "m"])

    (tmp_path / "planted_run.py").write_text("import planted_dependency\n")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(cli.COMMANDS, "run", ("planted_run", ""))
    with pytest.raises(ModuleNotFoundError, match="planted_dependency"):
        cli.main(["run"])


@pytest.mark.parametrize(
    "words, named",
    [
        (
            "plan",
            "compass measure steps --grid plans the step grid's points, and compass run --coverage",
        ),
        ("validate", "grading is python -m atom.compass.validate"),
        ("explain", "pass one with --machine"),
        ("probe", "pass one with --machine"),
        ("spec", "pass one with --machine"),
        ("measure collectives", "priced by compass measure ops"),
        ("discover", "renamed compass trace steps"),
        ("trace", "use compass trace steps or compass trace ops"),
    ],
)
def test_a_deprecated_command_refuses_naming_where_its_work_went(words, named):
    with pytest.raises(SystemExit, match=f"^compass {words}: .*{re.escape(named)}"):
        cli.main([*words.split(), "--out", "d"])


@pytest.mark.parametrize("argv", [[], ["fit"], ["measure", "everything"]])
def test_no_command_refuses_with_the_usage(argv):
    with pytest.raises(SystemExit, match="usage: compass <command>"):
        cli.main(argv)


@pytest.mark.parametrize("verb, args", [("fit steps", LINEAR), ("fit host", ())])
def test_a_machine_spec_that_refuses_exits_fit_with_its_refusal(tmp_path, verb, args):
    a, machine = record(tmp_path / "a", 19), tmp_path / "spec.json"
    spec = {**DOCUMENT, "schema_version": 2}
    machine.write_text(json.dumps(spec))
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(spec)

    tail = ["--machine", machine, "--store", tmp_path / "store", "--out", tmp_path]
    with pytest.raises(SystemExit) as exited:
        cli.main([*verb.split(), *args, *map(str, [a, *clients(a), *tail])])
    assert exited.value.code == f"compass {verb}: {refusal.value}"


def test_the_launcher_runs_its_own_tree_from_anywhere_it_is_linked(tmp_path):
    shadow = tmp_path / "atom"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise ImportError('the shadow atom')\n")
    launcher = Path(__file__).parents[2] / "scripts/compass/compass"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/compass").symlink_to(launcher)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}

    done = subprocess.run(
        [tmp_path / "bin/compass", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: compass <command>")
