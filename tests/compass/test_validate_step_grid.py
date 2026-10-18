# SPDX-License-Identifier: MIT
"""Grading a step-grid cell: `python -m atom.compass.validate` on a cell whose
simulated run is `step_grid/`, as `compass run --step-grid` writes it.

The cell is `test_validate_main`'s, laid out as `record` and `run` write it, its
simulated run moved to `step_grid/`. Its run file names a bundle published as
`test_run_step_grid` publishes one, under that file's deployment, which the
run's ATOM arguments resolve to here. The bundle's grid measures each real step
at the planted law's price on that step's own node, its residual's factor is
one, and its held-out hull holds every step, so the grid prices each real step
as the law does.
"""

import json
import re

import pytest
from test_backend_step_grid import fitted, linear, point
from test_memory_readings import DOCUMENT
from test_run_step_grid import BOUNDS, KEY, deployment, publish
from test_validate_cost import LAW, _steps
from test_validate_main import _as_recorded, _graded, _rows

from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid import coordinates
from atom.compass.backends.step_grid_fit import VALUE
from atom.compass.drive import run as drive_run
from atom.compass.validate.__main__ import main

ATOM_ARGS = ["--model", "planted", "--max-model-len", "2048"]


def _box(n):
    """Facets bounding each of `n` coordinates to [0, 1e6]."""
    return [
        [s * (i == j) for j in range(n)] + [-1e6 * (s > 0)]
        for i in range(n)
        for s in (1, -1)
    ]


def _bundle(steps, missing=()):
    """A replay bundle measuring each of `steps` at the law's price on nodes at
    its coordinates, those numbered in `missing` on nodes with no point."""
    law = StepLinearBackend(StepLinearLaw(LAW))
    found = {True: [], False: []}  # by whether a step is prefill
    for i, (rows, rung) in enumerate(steps):
        view = BatchView(
            tuple(RequestShape(*r) for r in rows),
            capture_rung=rung,
            produces_output=True,
        )
        at = [int(x) for x in coordinates(view.requests, rung is None)]
        found[rung is None].append((i, at, law.price(view).seconds))

    def nodes(prefill, axis):
        return sorted({at[axis] for _, at, _ in found[prefill]})

    def points(prefill, *rung):
        return [
            point(*rung, *at, time=t) for i, at, t in found[prefill] if i not in missing
        ]

    residual = fitted({"unit": linear()}, {"unit": 1.0})
    for kind, fit in residual["kinds"].items():
        fit["held_out"] = {"facets": _box(2 if kind == "decode" else 3)}
    grid = {
        "graph_mode": "replay",
        "scope": "forward",
        "unit": "s",
        "value": VALUE,
        "prefill": {
            "nodes": {a: nodes(True, i) for i, a in enumerate("BTc")},
            "output": points(True),
            "no_output": [],
        },
        "decode": {
            "nodes": {"B": {"1": nodes(False, 0)}, "c": {"1": nodes(False, 1)}},
            "points": points(False, 1),
        },
    }
    return {"grid": grid, "residual": residual, "bounds": BOUNDS}


def _as_grid(tmp_path, monkeypatch, missing=(), **config):
    """`tmp_path`'s cell with its simulated run moved to `step_grid/`, a
    `compass run --step-grid` directory over `_bundle`, its ATOM arguments
    resolving to the deployment `config` edits."""
    r = deployment(tmp_path)
    publish(tmp_path, r, member=_bundle(_steps(), missing))
    vars(r.config).update(config)
    resolved = []
    monkeypatch.setattr(
        drive_run, "_config", lambda args: resolved.append(args) or r.config
    )
    sim = (tmp_path / "c4" / "step_linear").rename(tmp_path / "c4" / "step_grid")
    run = json.loads((sim / "run.json").read_text())
    run["argv"] = ["-c", "run", "--step-grid", "planted", "--", *ATOM_ARGS]
    run["run_file"] = {
        "step_grid": {"store": str(tmp_path / "store"), **KEY},
        "host": {"laws": {}},
        "machine": DOCUMENT,
    }
    run["coverage"] = {"inputs": ["planted"], "uncovered": []}
    (sim / "run.json").write_text(json.dumps(run))
    return resolved


def _renamed(rows):
    named = json.dumps(list(rows.items())).replace("step_linear", "step_grid")
    return {tuple(k): v for k, v in json.loads(named)}


def test_a_step_grid_cell_grades_every_row_the_step_linear_cell_does(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    linear_rows = _graded(capsys, tmp_path / "c4")
    resolved = _as_grid(tmp_path, monkeypatch)

    rows = _graded(capsys, tmp_path / "c4")
    assert rows == _renamed(linear_rows)
    # The bundle loaded under the config the simulated run's ATOM arguments resolve.
    assert resolved == [ATOM_ARGS]
    assert rows["cost real0", "decode validated"][0].startswith("10 steps")
    assert rows["cost real0", "prefill validated"][0].startswith("9 steps")
    assert rows["summary step_grid", "refusals"][1] == "none"


def test_a_real_step_the_grid_refuses_is_counted_apart_and_priced_nowhere(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    # Decode step 15's node holds no point: the grid refuses that step alone.
    _as_grid(tmp_path, monkeypatch, missing={15})

    rows = _graded(capsys, tmp_path / "c4")
    for real in ("real0", "real1", "real2"):
        steps, value = rows[f"cost {real}", "refused decode rung 1"][0].split(", ")
        assert steps == "1 steps" and value.startswith("real ")
        assert rows[f"cost {real}", "decode validated"][0].startswith("9 steps")


def test_a_bundle_the_grid_refuses_to_load_refuses_the_cell_naming_why(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    _as_grid(tmp_path, monkeypatch, max_num_seqs=64)

    with pytest.raises(ValueError, match="max_num_seqs 64 is above the 48"):
        main([str(tmp_path / "c4")])


@pytest.mark.parametrize("both", [True, False])
def test_a_cell_of_two_simulated_runs_or_none_is_refused(capsys, tmp_path, both):
    _rows(capsys, tmp_path)
    cell = tmp_path / "c4"
    if both:
        (cell / "step_grid").mkdir()
    else:
        (cell / "step_linear").rename(cell / "real3")
    held = "['step_linear', 'step_grid']" if both else "[]"
    with pytest.raises(ValueError, match=re.escape(f"holds {held} of step_linear/")):
        main([str(cell)])
